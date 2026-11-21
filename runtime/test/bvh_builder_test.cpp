// Optional Windows test: map the package's XboxBvhBuilder.dll through rt::Loader, bring it up (Loader::initialise:
// TLS and DllMain), and construct an XboxBvhBuilder on a PC D3D12 device behind the Xbox-ABI device proxy.
// Needs the proxy's slot-62 shape (desc, w2, w3, riid, out). Results: docs/d3d12_vulkan.md, Appendix A.2.
// It records what the constructor does: how much of the object it writes, which device slots it calls, what it
// prints through OutputDebugString, and whether it gives up through TerminateProcess.
//
//   LEONIDA_GAME_DIR=<package root>   (contains XboxBvhBuilder.dll)
//   LEONIDA_BVH_FLAG=1                (optional: pass true as the constructor's bool argument)
//   LEONIDA_BVH_WARP=1                (optional: use the WARP adapter)
//   LEONIDA_BVH_DEVICE=vulkan         (optional: the Vulkan-backed ID3D12Device from #119 instead of a PC device)
//
// Without LEONIDA_GAME_DIR it prints "SKIP (no package)" and exits 0. Nothing from the DLL is written to disk.
//
// The exports are C++ members (demangled): XboxBvhBuilder::XboxBvhBuilder(ID3D12Device11*, bool) and ~XboxBvhBuilder().
// sizeof(XboxBvhBuilder) is not exported, so the object lives in a large buffer filled with a sentinel, and the
// highest byte the constructor changed is a lower bound for the size. OutputDebugStringA/W and TerminateProcess are
// redirected in this DLL's IAT only. Calls into the DLL run under SEH.
#include <windows.h>

#include <d3d12.h>
#include <dxgi1_6.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

#include "../shims/d3d12_xs_device.h"
#include "../shims/shims.h"
#include "../src/pe_loader.h"
#ifdef LEONIDA_VULKAN_DEVICE
#include "../shims/d3d12_com.h"
#endif

namespace {

int g_failures = 0;
uint64_t g_debug_strings = 0;
uint64_t g_terminate_calls = 0;
constexpr DWORD kTerminateCalled = 0xE0B71E00;  // raised instead of terminating, caught by the SEH guard

void check(bool condition, const std::string& what) {
    std::printf("%s %s\n", condition ? "PASS" : "FAIL", what.c_str());
    if (!condition) ++g_failures;
}

void note(const std::string& what) { std::printf("INFO %s\n", what.c_str()); }

std::string env(const char* name) {
    const char* v = std::getenv(name);
    return v ? v : "";
}

std::string hex(uint64_t v) {
    char b[32];
    std::snprintf(b, sizeof(b), "0x%llx", static_cast<unsigned long long>(v));
    return b;
}

void WINAPI debug_a(LPCSTR s) {
    ++g_debug_strings;
    std::printf("  dbg: %s%s", s ? s : "(null)", s && *s && s[std::strlen(s) - 1] == '\n' ? "" : "\n");
}
void WINAPI debug_w(LPCWSTR s) {
    ++g_debug_strings;
    std::printf("  dbg: %ls%s", s ? s : L"(null)", s && *s && s[wcslen(s) - 1] == L'\n' ? "" : "\n");
}
BOOL WINAPI terminate_stub(HANDLE, UINT exit_code) {
    ++g_terminate_calls;
    ULONG_PTR arg = exit_code;
    RaiseException(kTerminateCalled, EXCEPTION_NONCONTINUABLE, 1, &arg);
    return FALSE;
}

void** iat_slot(rt::Module& m, const char* want_dll, const char* want_name) {
    auto* dos = reinterpret_cast<IMAGE_DOS_HEADER*>(m.base);
    auto* nt = reinterpret_cast<IMAGE_NT_HEADERS64*>(m.base + dos->e_lfanew);
    auto& dir = nt->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_IMPORT];
    for (auto* d = reinterpret_cast<IMAGE_IMPORT_DESCRIPTOR*>(m.base + dir.VirtualAddress); d->Name; ++d) {
        if (rt::lower(reinterpret_cast<const char*>(m.base + d->Name)) != want_dll) continue;
        auto* lookup = reinterpret_cast<IMAGE_THUNK_DATA64*>(m.base + (d->OriginalFirstThunk ? d->OriginalFirstThunk : d->FirstThunk));
        auto* iat = reinterpret_cast<IMAGE_THUNK_DATA64*>(m.base + d->FirstThunk);
        for (; lookup->u1.AddressOfData; ++lookup, ++iat) {
            if (IMAGE_SNAP_BY_ORDINAL64(lookup->u1.Ordinal)) continue;
            auto* n = reinterpret_cast<IMAGE_IMPORT_BY_NAME*>(m.base + static_cast<uint32_t>(lookup->u1.AddressOfData));
            if (std::string(want_name) == n->Name) return reinterpret_cast<void**>(&iat->u1.Function);
        }
    }
    return nullptr;
}

bool patch(void** slot, void* fn) {
    DWORD prot = 0;
    if (!slot || !VirtualProtect(slot, sizeof(void*), PAGE_READWRITE, &prot)) return false;
    *slot = fn;
    VirtualProtect(slot, sizeof(void*), prot, &prot);
    return true;
}

struct Call {
    bool faulted = false;
    DWORD code = 0;
    ULONG_PTR arg = 0;
    char cxx_type[128] = {};  // first catchable type of a C++ exception, e.g. ".?AVruntime_error@std@@"
    char cxx_what[256] = {};  // what() when that type is a std::exception
};

// MSVC x64 C++ exceptions (0xE06D7363): ExceptionInformation = {magic, object, ThrowInfo*, image base}; the
// ThrowInfo / CatchableTypeArray / CatchableType / TypeDescriptor chain uses 32-bit RVAs from that base.
void decode_cxx(const EXCEPTION_RECORD* r, Call* c) {
    if (r->ExceptionCode != 0xE06D7363 || r->NumberParameters < 4) return;
    auto* object = reinterpret_cast<void*>(r->ExceptionInformation[1]);
    auto* info = reinterpret_cast<const uint8_t*>(r->ExceptionInformation[2]);
    auto* base = reinterpret_cast<const uint8_t*>(r->ExceptionInformation[3]);
    if (!info || !base) return;
    const int32_t types_rva = *reinterpret_cast<const int32_t*>(info + 12);  // pCatchableTypeArray
    const uint8_t* types = base + types_rva;
    const int32_t count = *reinterpret_cast<const int32_t*>(types);
    bool is_std_exception = false;
    for (int32_t i = 0; i < count; ++i) {
        const uint8_t* ct = base + *reinterpret_cast<const int32_t*>(types + 4 + 4 * i);
        const char* name = reinterpret_cast<const char*>(base + *reinterpret_cast<const int32_t*>(ct + 4) + 16);
        if (i == 0) strncpy_s(c->cxx_type, name, _TRUNCATE);
        if (std::strcmp(name, ".?AVexception@std@@") == 0) is_std_exception = true;
    }
    if (std::strcmp(c->cxx_type, ".J") == 0 && object)  // `long`: an HRESULT thrown by value
        std::snprintf(c->cxx_what, sizeof(c->cxx_what), "long (HRESULT) 0x%08lx",
                      static_cast<unsigned long>(*static_cast<const long*>(object)));
    if (is_std_exception && object) {
        const char* what = static_cast<std::exception*>(object)->what();
        if (what) strncpy_s(c->cxx_what, what, _TRUNCATE);
    }
}

int filter(EXCEPTION_POINTERS* e, Call* c) {
    c->code = e->ExceptionRecord->ExceptionCode;
    if (e->ExceptionRecord->NumberParameters) c->arg = e->ExceptionRecord->ExceptionInformation[0];
    __try {
        decode_cxx(e->ExceptionRecord, c);
    } __except (EXCEPTION_EXECUTE_HANDLER) {
    }
    return EXCEPTION_EXECUTE_HANDLER;
}

using Ctor = void* (*)(void* self, void* device, bool flag);
using Dtor = void (*)(void* self);

Call guarded_ctor(Ctor fn, void* self, void* device, bool flag) {
    Call c;
    __try {
        fn(self, device, flag);
    } __except (filter(GetExceptionInformation(), &c)) {
        c.faulted = true;
    }
    return c;
}

Call guarded_dtor(Dtor fn, void* self) {
    Call c;
    __try {
        fn(self);
    } __except (filter(GetExceptionInformation(), &c)) {
        c.faulted = true;
    }
    return c;
}

// main's loader (#95) maps, sets up TLS and runs DllMain, but doesn't register the image's x64 unwind tables yet
// (#93 adds that). Both DLLs throw and catch C++ exceptions inside, which ends the process without them. Register
// .pdata here only when the loader hasn't, so the test keeps working once #93 is in.
bool ensure_unwind_registered(rt::Module& m) {
    DWORD64 base = 0;
    if (RtlLookupFunctionEntry(reinterpret_cast<DWORD64>(m.base + m.entry_rva), &base, nullptr)) return true;
    auto* nt = reinterpret_cast<IMAGE_NT_HEADERS64*>(m.base + reinterpret_cast<IMAGE_DOS_HEADER*>(m.base)->e_lfanew);
    const auto& ex = nt->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_EXCEPTION];
    return ex.Size && RtlAddFunctionTable(reinterpret_cast<PRUNTIME_FUNCTION>(m.base + ex.VirtualAddress),
                                          ex.Size / sizeof(RUNTIME_FUNCTION), reinterpret_cast<DWORD64>(m.base));
}

// Loader::initialise() runs the module's TLS setup and DllMain (#95). SEH faults inside the DLL are reported here;
// C++ exceptions (a DllMain returning FALSE, a corrupt TLS directory) go on to the caller's catch.
DWORD guarded_initialise(rt::Loader* loader) {
    DWORD code = 0;
    __try {
        loader->initialise();
    } __except ((code = GetExceptionCode()) == 0xE06D7363 ? EXCEPTION_CONTINUE_SEARCH : EXCEPTION_EXECUTE_HANDLER) {
    }
    return code == 0xE06D7363 ? 0 : code;
}

std::string describe(const Call& c) {
    if (!c.faulted) return "returned";
    if (c.code == kTerminateCalled) return "called TerminateProcess(exit code " + hex(c.arg) + ")";
    if (c.code == 0xE06D7363)
        return std::string("threw a C++ exception ") + (c.cxx_type[0] ? c.cxx_type : "(type unknown)") +
               (c.cxx_what[0] ? std::string(": \"") + c.cxx_what + "\"" : "");
    return "faulted with exception " + hex(c.code);
}

void* symbol(rt::Module& m, const char* name) {
    auto it = m.exports.find(name);
    return it == m.exports.end() ? nullptr : it->second;
}

// The D3D12 debug layer, when the Graphics Tools feature is installed, says why a PSO was rejected.
bool enable_debug_layer() {
    ID3D12Debug* debug = nullptr;
    if (FAILED(D3D12GetDebugInterface(IID_PPV_ARGS(&debug)))) return false;
    debug->EnableDebugLayer();
    debug->Release();
    return true;
}

void print_debug_messages(ID3D12Device* device) {
    ID3D12InfoQueue* q = nullptr;
    if (FAILED(device->QueryInterface(IID_PPV_ARGS(&q)))) return;
    const UINT64 n = q->GetNumStoredMessages();
    for (UINT64 i = 0; i < n && i < 32; ++i) {
        SIZE_T len = 0;
        q->GetMessage(i, nullptr, &len);
        std::vector<uint8_t> buf(len);
        auto* m = reinterpret_cast<D3D12_MESSAGE*>(buf.data());
        if (SUCCEEDED(q->GetMessage(i, m, &len))) std::printf("  d3d12 debug [%d]: %s\n", m->ID, m->pDescription);
    }
    q->ClearStoredMessages();
    q->Release();
}

ID3D12Device* create_pc_device(bool warp) {
    IDXGIFactory4* factory = nullptr;
    if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory)))) return nullptr;
    IDXGIAdapter1* adapter = nullptr;
    if (warp) {
        factory->EnumWarpAdapter(IID_PPV_ARGS(&adapter));
    } else {
        for (UINT i = 0; factory->EnumAdapters1(i, &adapter) != DXGI_ERROR_NOT_FOUND; ++i) {
            DXGI_ADAPTER_DESC1 d{};
            adapter->GetDesc1(&d);
            if (!(d.Flags & DXGI_ADAPTER_FLAG_SOFTWARE)) break;
            adapter->Release();
            adapter = nullptr;
        }
    }
    ID3D12Device* device = nullptr;
    if (adapter) {
        DXGI_ADAPTER_DESC1 d{};
        adapter->GetDesc1(&d);
        std::printf("adapter: %ls\n", d.Description);
        D3D12CreateDevice(adapter, D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&device));
        adapter->Release();
    }
    factory->Release();
    return device;
}

void print_slot_hits(const void* proxy) {
    const unsigned n = rt::d3d12_xs::device_proxy_slot_count(proxy) + 32;
    std::string dev, heap;
    for (unsigned s = 0; s < n; ++s)
        if (auto h = rt::d3d12_xs::device_slot_hits(s)) dev += " " + std::to_string(s) + "x" + std::to_string(h);
    for (unsigned s = 0; s < 16; ++s)
        if (auto h = rt::d3d12_xs::heap_slot_hits(s)) heap += " " + std::to_string(s) + "x" + std::to_string(h);
    note("device slots hit (slot x count):" + (dev.empty() ? std::string(" none") : dev));
    note("descriptor-heap slots hit:" + (heap.empty() ? std::string(" none") : heap));
}

// Slot-62 recorder: the proxy's vtable is copied and slot 62 points here, so the raw argument words the DLL passes
// are printed before the proxy's own handler runs. Test-only; the proxy itself is unchanged.
using Slot62 = uint64_t(WINAPI*)(void*, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t);
Slot62 g_slot62 = nullptr;
int g_slot62_logged = 0;

bool looks_like_iid(uint64_t p, const IID& iid) {
    __try {
        return p && std::memcmp(reinterpret_cast<const void*>(p), &iid, sizeof(IID)) == 0;
    } __except (EXCEPTION_EXECUTE_HANDLER) {
        return false;
    }
}

uint64_t g_slot62_ok = 0, g_slot62_failed = 0;

uint64_t WINAPI record_slot62(void* self, uint64_t a1, uint64_t a2, uint64_t a3, uint64_t a4, uint64_t a5, uint64_t a6) {
    const int n = g_slot62_logged++;
    if (n < 1) {
        std::printf("  slot62 args: a1=%#llx a2=%#llx a3=%#llx a4=%#llx a5=%#llx a6=%#llx\n",
                    (unsigned long long)a1, (unsigned long long)a2, (unsigned long long)a3, (unsigned long long)a4,
                    (unsigned long long)a5, (unsigned long long)a6);
        const uint64_t args[] = {a1, a2, a3, a4, a5, a6};
        for (int i = 0; i < 6; ++i)
            if (looks_like_iid(args[i], __uuidof(ID3D12PipelineState)))
                std::printf("  slot62: a%d points at IID_ID3D12PipelineState\n", i + 1);
    }
    const auto* desc = reinterpret_cast<const D3D12_COMPUTE_PIPELINE_STATE_DESC*>(a1);
    const uint64_t r = g_slot62(self, a1, a2, a3, a4, a5, a6);
    const auto hr = static_cast<HRESULT>(static_cast<uint32_t>(r));
    (SUCCEEDED(hr) ? g_slot62_ok : g_slot62_failed)++;
    std::printf("  slot62 #%d: CS bytecode %llu bytes, root signature %s, node mask %u, flags %#x -> hr %#010lx\n", n + 1,
                desc ? (unsigned long long)desc->CS.BytecodeLength : 0ull, desc && desc->pRootSignature ? "set" : "null",
                desc ? desc->NodeMask : 0u, desc ? (unsigned)desc->Flags : 0u, static_cast<unsigned long>(hr));
    return r;
}

void** g_proxy_vtable_copy = nullptr;
bool hook_slot62(void* proxy, unsigned slots) {
    void** vtable = *static_cast<void***>(proxy);
    g_proxy_vtable_copy = new void*[slots];
    std::memcpy(g_proxy_vtable_copy, vtable, slots * sizeof(void*));
    g_slot62 = reinterpret_cast<Slot62>(vtable[62]);
    g_proxy_vtable_copy[62] = reinterpret_cast<void*>(&record_slot62);
    *static_cast<void***>(proxy) = g_proxy_vtable_copy;
    return true;
}

constexpr const char* kCtor = "??0XboxBvhBuilder@@QEAA@PEAUID3D12Device11@@_N@Z";
constexpr const char* kDtor = "??1XboxBvhBuilder@@QEAA@XZ";
constexpr size_t kObjectBuffer = 1u << 20;
constexpr uint8_t kSentinel = 0xCD;

}  // namespace

int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);  // keep every line if the DLL takes the process down
    const std::string game_dir_arg = env("LEONIDA_GAME_DIR");
    if (game_dir_arg.empty()) {
        std::printf("SKIP (no package)\n");
        return 0;
    }
    const std::filesystem::path game_dir = game_dir_arg;
    const std::filesystem::path dll = game_dir / "XboxBvhBuilder.dll";
    std::error_code ec;
    if (!std::filesystem::is_regular_file(dll, ec)) {
        std::printf("FAIL LEONIDA_GAME_DIR does not contain XboxBvhBuilder.dll\n");
        return 1;
    }

    try {
        rt::Loader loader(game_dir);
        rt::Module& bvh = loader.load(dll);
        std::printf("mapped %s at %p, size %u, entry +%#x%s\n", bvh.name.c_str(), bvh.base, bvh.size,
                    bvh.entry_rva, bvh.relocated ? " (relocated)" : "");
        int shim = 0, host = 0, unimplemented = 0;
        for (auto& r : bvh.imports) {
            if (r.binding == rt::Binding::Shim) ++shim;
            if (r.binding == rt::Binding::Host) ++host;
            if (r.binding == rt::Binding::Unimplemented) {
                ++unimplemented;
                std::printf("  UNIMPLEMENTED %s!%s\n", r.dll.c_str(), r.name.c_str());
            }
            const std::string d = rt::lower(r.dll);
            if (d == "xmem.dll" || d == "d3d12_xs.dll")
                check(r.binding == rt::Binding::Shim, r.dll + "!" + r.name + " binds to the shim");
        }
        std::printf("imports: shim=%d host=%d unimplemented=%d\n", shim, host, unimplemented);
        check(unimplemented == 0, "no unimplemented imports");

        auto ctor = reinterpret_cast<Ctor>(symbol(bvh, kCtor));
        auto dtor = reinterpret_cast<Dtor>(symbol(bvh, kDtor));
        check(ctor && dtor, "constructor and destructor are exported");
        check(patch(iat_slot(bvh, "api-ms-win-core-debug-l1-1-0.dll", "OutputDebugStringA"), reinterpret_cast<void*>(&debug_a)) &&
                  patch(iat_slot(bvh, "api-ms-win-core-debug-l1-1-0.dll", "OutputDebugStringW"), reinterpret_cast<void*>(&debug_w)) &&
                  patch(iat_slot(bvh, "api-ms-win-core-processthreads-l1-1-0.dll", "TerminateProcess"),
                        reinterpret_cast<void*>(&terminate_stub)),
              "OutputDebugStringA/W and TerminateProcess redirected in this DLL's IAT");
        if (g_failures) return 1;

        check(ensure_unwind_registered(bvh), "x64 unwind tables registered (by the loader or here)");
        const DWORD init = guarded_initialise(&loader);
        check(init == 0, "Loader::initialise (TLS, DllMain) succeeds" + (init ? " (exception " + hex(init) + ")" : ""));
        if (init) return 1;
        std::printf("TLS: %s slot %u\n", bvh.has_tls_slot ? "loader" : "no", bvh.tls_slot);

        ID3D12Device* pc = nullptr;
        const bool on_vulkan = env("LEONIDA_BVH_DEVICE") == "vulkan";
        if (on_vulkan) {
#ifdef LEONIDA_VULKAN_DEVICE
            // The Vulkan-backed ID3D12Device (#119) instead of the host's D3D12 device, behind the same proxy.
            const HRESULT vhr = rt::d3d12::create_vulkan_device_standalone(__uuidof(ID3D12Device), reinterpret_cast<void**>(&pc));
            check(SUCCEEDED(vhr) && pc, "Vulkan-backed D3D12 device created (" + hex(static_cast<uint32_t>(vhr)) + ")");
#else
            check(false, "LEONIDA_BVH_DEVICE=vulkan needs a build with the Vulkan SDK (gta6rt_d3d12_com)");
#endif
        } else {
            const bool debug_layer = enable_debug_layer();
            note(std::string("D3D12 debug layer ") + (debug_layer ? "enabled" : "not available (Graphics Tools not installed)"));
            pc = create_pc_device(env("LEONIDA_BVH_WARP") == "1");
            check(pc != nullptr, "PC D3D12 device created");
        }
        if (!pc) return 1;
        pc->AddRef();  // kept for the debug messages; the proxy takes its own reference
        void* proxy = nullptr;
        const HRESULT hr = rt::d3d12_xs::create_device_proxy(pc, __uuidof(ID3D12Device11), &proxy);
        check(SUCCEEDED(hr) && proxy, "Xbox-ABI device proxy created as ID3D12Device11 (" + hex(static_cast<uint32_t>(hr)) + ")");
        if (!proxy) return 1;
#ifdef LEONIDA_VULKAN_DEVICE
        if (on_vulkan) rt::d3d12::set_device_outer(pc, static_cast<IUnknown*>(proxy));
#endif
        rt::d3d12_xs::reset_device_proxy_stats();
        if (env("LEONIDA_BVH_RECORD62") == "1")
            check(hook_slot62(proxy, rt::d3d12_xs::device_proxy_slot_count(proxy)), "slot 62 recorder installed");

        const bool flag = env("LEONIDA_BVH_FLAG") == "1";
        std::vector<uint8_t> object(kObjectBuffer, kSentinel);
        const ULONGLONG t0 = GetTickCount64();
        Call c = guarded_ctor(ctor, object.data(), proxy, flag);
        const ULONGLONG ms = GetTickCount64() - t0;
        size_t last = 0, changed = 0;
        for (size_t i = 0; i < object.size(); ++i)
            if (object[i] != kSentinel) {
                last = i + 1;
                ++changed;
            }
        note(std::string("XboxBvhBuilder(device, ") + (flag ? "true" : "false") + ") " + describe(c) + " after " +
             std::to_string(ms) + " ms");
        note("object bytes written: up to offset " + hex(last) + " (" + std::to_string(changed) +
             " bytes differ from the sentinel): sizeof(XboxBvhBuilder) >= " + std::to_string(last));
        print_slot_hits(proxy);
        print_debug_messages(pc);
        note("slot 62 (Xbox CreateComputePipelineState) calls: " + std::to_string(rt::d3d12_xs::device_slot_hits(62)) +
             (g_slot62_logged ? ", succeeded " + std::to_string(g_slot62_ok) + ", failed " + std::to_string(g_slot62_failed)
                              : std::string()));
        note("OutputDebugString calls: " + std::to_string(g_debug_strings) + ", TerminateProcess calls: " +
             std::to_string(g_terminate_calls));
        check(!c.faulted || c.code == kTerminateCalled, "constructor returns or gives up through TerminateProcess, no fault");

        if (!c.faulted) {
            rt::d3d12_xs::reset_device_proxy_stats();
            c = guarded_dtor(dtor, object.data());
            note("~XboxBvhBuilder " + describe(c));
            print_slot_hits(proxy);
            check(!c.faulted, "destructor returns");
        }
        reinterpret_cast<IUnknown*>(proxy)->Release();
        pc->Release();
    } catch (const std::exception& e) {
        std::printf("FAIL %s\n", e.what());
        return 1;
    }

    std::printf("bvh_builder_test: %s (%d failure%s)\n", g_failures ? "FAILED" : "OK", g_failures,
                g_failures == 1 ? "" : "s");
    return g_failures ? 1 : 0;
}
