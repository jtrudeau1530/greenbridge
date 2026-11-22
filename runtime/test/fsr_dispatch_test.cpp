// Optional Windows test: map the package's amd_fidelityfx_upscaler_xbox.dll (FSR) through rt::Loader, bring it up
// (Loader::initialise: TLS and DllMain), and drive it through the public FidelityFX API on a PC D3D12
// device wrapped in the Xbox-ABI device proxy. It records what the DLL answers and which device slots it calls.
//
//   LEONIDA_GAME_DIR=<package root>   (contains amd_fidelityfx_upscaler_xbox.dll)
//   LEONIDA_FSR_WARP=1                (optional: use the WARP adapter instead of the default hardware adapter)
//
// Without LEONIDA_GAME_DIR it prints "SKIP (no package)" and exits 0. Nothing from the DLL is written to disk.
//
// Steps, each one only if the previous worked:
//   1. map: the console imports bind to the shims and nothing is unimplemented
//   2. Loader::initialise(): TLS and DllMain(DLL_PROCESS_ATTACH)
//   3. ffxConfigure(global debug) with a message callback, so the DLL's own messages are printed
//   4. CPU-only queries (upscale ratio, render resolution): do the PC struct type ids work on the Xbox build?
//   5. ffxQuery(GET_VERSIONS) for the upscaler on the proxied device
//   6. ffxCreateContext(upscale + version + backend) on the proxied device; the device/heap slots it hit are listed
// Calls into the DLL run under SEH, so an access violation inside it is reported as a result, not a crash.
//
// On 1.0.2.1 steps 1-5 work, and step 6 stops at FFX_API_RETURN_ERROR: the build does not recognise the PC DX12
// backend descriptor, and the Xbox one is not public (docs/d3d12_vulkan.md, Appendix A.3). The INFO lines are the
// measurement; the PASS/FAIL checks only cover what is known to work.
#include <windows.h>

#include <d3d12.h>
#include <dxgi1_6.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>

#include "../shims/d3d12_xs_device.h"
#include "../shims/shims.h"
#include "../src/pe_loader.h"
#include "ffx_api_min.h"

namespace {

int g_failures = 0;
int g_messages = 0;

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

void on_message(uint32_t type, const wchar_t* message) {
    ++g_messages;
    std::printf("  ffx %s: %ls\n", type == 0 ? "error" : "warning", message ? message : L"(null)");
}

// SEH wrappers: no C++ objects with destructors in these frames.
struct Call {
    bool faulted = false;
    DWORD code = 0;
    ffx::ReturnCode rc = 0;
};

Call guarded_configure(ffx::PfnConfigure fn, ffx::Context* ctx, const ffx::Header* desc) {
    Call c;
    __try {
        c.rc = fn(ctx, desc);
    } __except (c.code = GetExceptionCode(), EXCEPTION_EXECUTE_HANDLER) {
        c.faulted = true;
    }
    return c;
}

Call guarded_query(ffx::PfnQuery fn, ffx::Context* ctx, ffx::Header* desc) {
    Call c;
    __try {
        c.rc = fn(ctx, desc);
    } __except (c.code = GetExceptionCode(), EXCEPTION_EXECUTE_HANDLER) {
        c.faulted = true;
    }
    return c;
}

Call guarded_create(ffx::PfnCreateContext fn, ffx::Context* ctx, ffx::Header* desc) {
    Call c;
    __try {
        c.rc = fn(ctx, desc, nullptr);
    } __except (c.code = GetExceptionCode(), EXCEPTION_EXECUTE_HANDLER) {
        c.faulted = true;
    }
    return c;
}

Call guarded_destroy(ffx::PfnDestroyContext fn, ffx::Context* ctx) {
    Call c;
    __try {
        c.rc = fn(ctx, nullptr);
    } __except (c.code = GetExceptionCode(), EXCEPTION_EXECUTE_HANDLER) {
        c.faulted = true;
    }
    return c;
}

// Counting wrappers around the xmem shims, patched into this DLL's IAT only, so the test sees how far a call gets.
using XMemAllocFn = void*(__stdcall*)(size_t, uint64_t);
using XMemFreeFn = void(__stdcall*)(void*, uint64_t);
XMemAllocFn g_xmem_alloc = nullptr;
XMemFreeFn g_xmem_free = nullptr;
uint64_t g_alloc_calls = 0, g_alloc_bytes = 0, g_free_calls = 0;

void* __stdcall counting_alloc(size_t size, uint64_t attributes) {
    ++g_alloc_calls;
    g_alloc_bytes += size;
    return g_xmem_alloc(size, attributes);
}
void __stdcall counting_free(void* p, uint64_t attributes) {
    ++g_free_calls;
    g_xmem_free(p, attributes);
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

bool patch(void** slot, void* fn, void** old) {
    DWORD prot = 0;
    if (!slot || !VirtualProtect(slot, sizeof(void*), PAGE_READWRITE, &prot)) return false;
    *old = *slot;
    *slot = fn;
    VirtualProtect(slot, sizeof(void*), prot, &prot);
    return true;
}

// main's loader (#95) maps, sets up TLS and runs DllMain, but doesn't register the image's x64 unwind tables yet
// (#176 adds that). Both DLLs throw and catch C++ exceptions inside, which ends the process without them. Register
// .pdata here only when the loader hasn't, so the test keeps working once the loader does.
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
    if (c.faulted) return "faulted with exception " + hex(c.code);
    return std::string(ffx::return_code_name(c.rc)) + " (" + std::to_string(c.rc) + ")";
}

void* symbol(rt::Module& m, const char* name) {
    auto it = m.exports.find(name);
    return it == m.exports.end() ? nullptr : it->second;
}

// The PC device to wrap: the first hardware adapter, or WARP.
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
    // The proxy forwards the PC device's own slots and logs-and-fails beyond them; a margin covers those too.
    const unsigned n = rt::d3d12_xs::device_proxy_slot_count(proxy) + 32;
    std::string dev, heap;
    for (unsigned s = 0; s < n; ++s)
        if (auto h = rt::d3d12_xs::device_slot_hits(s)) dev += " " + std::to_string(s) + "x" + std::to_string(h);
    for (unsigned s = 0; s < 16; ++s)
        if (auto h = rt::d3d12_xs::heap_slot_hits(s)) heap += " " + std::to_string(s) + "x" + std::to_string(h);
    note("device slots hit (slot x count):" + (dev.empty() ? std::string(" none") : dev));
    note("descriptor-heap slots hit:" + (heap.empty() ? std::string(" none") : heap));
}

// The module's TLS index, whichever loader this builds against: #95 records a slot (has_tls_slot/tls_slot), #93
// a module TLS index (tls_index, kNoTlsIndex when there is none). ~0u means no TLS.
template <class M>
auto module_tls_index(const M& m, int) -> decltype(m.tls_index, uint32_t()) { return m.tls_index; }
template <class M>
auto module_tls_index(const M& m, long) -> decltype(m.tls_slot, uint32_t()) { return m.has_tls_slot ? m.tls_slot : ~0u; }

}  // namespace

int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);  // keep every line if the DLL takes the process down
    const std::string game_dir_arg = env("LEONIDA_GAME_DIR");
    if (game_dir_arg.empty()) {
        std::printf("SKIP (no package)\n");
        return 0;
    }
    const std::filesystem::path game_dir = game_dir_arg;
    const std::filesystem::path dll = game_dir / "amd_fidelityfx_upscaler_xbox.dll";
    std::error_code ec;
    if (!std::filesystem::is_regular_file(dll, ec)) {
        std::printf("FAIL LEONIDA_GAME_DIR does not contain amd_fidelityfx_upscaler_xbox.dll\n");
        return 1;
    }

    try {
        // 1. Map.
        rt::Loader loader(game_dir);
        rt::Module& fsr = loader.load(dll);
        std::printf("mapped %s at %p, size %u, entry +%#x%s\n", fsr.name.c_str(), fsr.base, fsr.size,
                    fsr.entry_rva, fsr.relocated ? " (relocated)" : "");
        int shim = 0, host = 0, unimplemented = 0;
        for (auto& r : fsr.imports) {
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

        auto configure = reinterpret_cast<ffx::PfnConfigure>(symbol(fsr, "ffxConfigure"));
        auto query = reinterpret_cast<ffx::PfnQuery>(symbol(fsr, "ffxQuery"));
        auto create = reinterpret_cast<ffx::PfnCreateContext>(symbol(fsr, "ffxCreateContext"));
        auto destroy = reinterpret_cast<ffx::PfnDestroyContext>(symbol(fsr, "ffxDestroyContext"));
        check(configure && query && create && destroy && symbol(fsr, "ffxDispatch"), "the five ffx* entry points are exported");
        if (g_failures) return 1;
        check(patch(iat_slot(fsr, "xmem.dll", "XMemAlloc"), reinterpret_cast<void*>(&counting_alloc),
                    reinterpret_cast<void**>(&g_xmem_alloc)) &&
                  patch(iat_slot(fsr, "xmem.dll", "XMemFree"), reinterpret_cast<void*>(&counting_free),
                        reinterpret_cast<void**>(&g_xmem_free)),
              "XMemAlloc/XMemFree IAT slots wrapped with counters");

        // 2. TLS and DllMain.
        check(ensure_unwind_registered(fsr), "x64 unwind tables registered (by the loader or here)");
        const DWORD init = guarded_initialise(&loader);
        check(init == 0, "Loader::initialise (TLS, DllMain) succeeds" + (init ? " (exception " + hex(init) + ")" : ""));
        if (init) return 1;
        const uint32_t tls = module_tls_index(fsr, 0);
        std::printf("TLS: %s index %u\n", tls != ~0u ? "loader" : "no", tls);
        if (tls != ~0u) {
            // __declspec(thread) code reads TEB->ThreadLocalStoragePointer[_tls_index] (gs:[58h]), not TlsGetValue.
            auto** tls_array = reinterpret_cast<void**>(__readgsqword(0x58));
            void* seen = tls_array ? tls_array[tls] : nullptr;
            note("TLS check: ThreadLocalStoragePointer[" + std::to_string(tls) + "] = " +
                 hex(reinterpret_cast<uint64_t>(seen)) + ", loader block = " + hex(reinterpret_cast<uint64_t>(fsr.tls_block)));
            // A TlsAlloc index names TEB->TlsSlots, a different array: the image would read another module's TLS
            // block, and its thread-safe statics then fault or not depending on what is there (#93 fixes this).
            check(seen == fsr.tls_block, "the image's __declspec(thread) block is the one the loader set up");
            if (seen != fsr.tls_block) return 1;
        }

        // 3. Debug messages.
        ffx::ConfigureDescGlobalDebug debug{};
        debug.header.type = ffx::CONFIGURE_DESC_TYPE_GLOBALDEBUG;
        debug.fpMessage = on_message;
        debug.debugLevel = ffx::GLOBALDEBUG_LEVEL_VERBOSE;
        Call c = guarded_configure(configure, nullptr, &debug.header);
        note("ffxConfigure(null, GLOBALDEBUG) -> " + describe(c));
        check(!c.faulted, "ffxConfigure(GLOBALDEBUG) returns");

        // 4. CPU-only queries with the PC struct type ids.
        float ratio = 0.0f;
        ffx::QueryDescUpscaleGetUpscaleRatioFromQualityMode qr{};
        qr.header.type = ffx::QUERY_DESC_TYPE_UPSCALE_GETUPSCALERATIOFROMQUALITYMODE;
        qr.qualityMode = ffx::QUALITY_MODE_PERFORMANCE;
        qr.pOutUpscaleRatio = &ratio;
        c = guarded_query(query, nullptr, &qr.header);
        note("ffxQuery(null, GETUPSCALERATIOFROMQUALITYMODE, PERFORMANCE) -> " + describe(c) + ", ratio " +
             std::to_string(ratio));
        check(!c.faulted && c.rc == ffx::RETURN_OK && std::fabs(ratio - 2.0f) < 1e-6f,
              "upscale ratio for PERFORMANCE is 2.0 (PC struct type id accepted)");

        uint32_t rw = 0, rh = 0;
        ffx::QueryDescUpscaleGetRenderResolutionFromQualityMode qres{};
        qres.header.type = ffx::QUERY_DESC_TYPE_UPSCALE_GETRENDERRESOLUTIONFROMQUALITYMODE;
        qres.displayWidth = 3840;
        qres.displayHeight = 2160;
        qres.qualityMode = ffx::QUALITY_MODE_QUALITY;
        qres.pOutRenderWidth = &rw;
        qres.pOutRenderHeight = &rh;
        c = guarded_query(query, nullptr, &qres.header);
        note("ffxQuery(null, GETRENDERRESOLUTIONFROMQUALITYMODE, 3840x2160 QUALITY) -> " + describe(c) + ", " +
             std::to_string(rw) + "x" + std::to_string(rh));
        check(!c.faulted && c.rc == ffx::RETURN_OK && rw == 2560 && rh == 1440, "QUALITY at 3840x2160 renders at 2560x1440");

        // 5/6 need a device.
        ID3D12Device* pc = create_pc_device(env("LEONIDA_FSR_WARP") == "1");
        check(pc != nullptr, "PC D3D12 device created");
        if (!pc) return 1;
        void* proxy = nullptr;
        const HRESULT hr = rt::d3d12_xs::create_device_proxy(pc, __uuidof(ID3D12Device), &proxy);
        check(SUCCEEDED(hr) && proxy, "Xbox-ABI device proxy created");
        if (!proxy) return 1;
        rt::d3d12_xs::reset_device_proxy_stats();

        uint64_t count = 0;
        ffx::QueryDescGetVersions qv{};
        qv.header.type = ffx::QUERY_DESC_TYPE_GET_VERSIONS;
        qv.createDescType = ffx::CREATE_CONTEXT_DESC_TYPE_UPSCALE;
        qv.device = proxy;
        qv.outputCount = &count;
        c = guarded_query(query, nullptr, &qv.header);
        note("ffxQuery(null, GET_VERSIONS, upscale) -> " + describe(c) + ", " + std::to_string(count) + " version(s)");
        if (!c.faulted && c.rc == ffx::RETURN_OK && count > 0 && count < 64) {
            uint64_t ids[64]{};
            const char* names[64]{};
            qv.versionIds = ids;
            qv.versionNames = names;
            c = guarded_query(query, nullptr, &qv.header);
            for (uint64_t i = 0; i < count && !c.faulted; ++i)
                note("  version " + hex(ids[i]) + " " + (names[i] ? names[i] : "(no name)"));
        }

        // 6. Create a context. The shipping build accepts one configuration only, and says so through the message
        //    callback when asked for another (measured on 1.0.2.1): 1280x720 -> 1920x1080 Quality, HDR input,
        //    inverted depth, auto exposure.
        ffx::CreateBackendDX12Desc backend{};
        backend.header.type = ffx::CREATE_CONTEXT_DESC_TYPE_BACKEND_DX12;
        backend.device = proxy;
        // The DLL insists on a version descriptor; it reports provider version 4.1.1 (GET_VERSIONS above).
        ffx::CreateContextDescUpscaleVersion version{};
        version.header.type = ffx::CREATE_CONTEXT_DESC_TYPE_UPSCALE_VERSION;
        version.header.pNext = &backend.header;
        version.version = ffx::upscaler_version(4, 1, 1);
        ffx::CreateContextDescUpscale up{};
        up.header.type = ffx::CREATE_CONTEXT_DESC_TYPE_UPSCALE;
        up.header.pNext = &version.header;
        up.flags = ffx::UPSCALE_ENABLE_HIGH_DYNAMIC_RANGE | ffx::UPSCALE_ENABLE_DEPTH_INVERTED |
                   ffx::UPSCALE_ENABLE_AUTO_EXPOSURE | ffx::UPSCALE_ENABLE_DEBUG_CHECKING;
        up.maxRenderSize = {1280, 720};
        up.maxUpscaleSize = {1920, 1080};
        up.fpMessage = on_message;
        ffx::Context ctx = nullptr;
        // Which backend descriptor the Xbox build takes is not public. Try the PC DX12 id first, then the same
        // {header, device} layout under the Xbox backend id (a hypothesis), and stop at the first that works.
        const ffx::StructType backend_types[] = {
            ffx::CREATE_CONTEXT_DESC_TYPE_BACKEND_DX12,
            ffx::backend_sub_id(ffx::BACKEND_ID_XBOX, 0x02),
            ffx::backend_sub_id(ffx::BACKEND_ID_XBOX, 0x01),
            ffx::backend_sub_id(ffx::BACKEND_ID_XBOX, 0x00),
        };
        const uint32_t versions[] = {ffx::upscaler_version(4, 1, 1), ffx::upscaler_version(4, 1, 2),
                                     ffx::upscaler_version(4, 0, 0)};
        const uint32_t base_flags = ffx::UPSCALE_ENABLE_HIGH_DYNAMIC_RANGE | ffx::UPSCALE_ENABLE_DEPTH_INVERTED |
                                    ffx::UPSCALE_ENABLE_AUTO_EXPOSURE;
        bool created = false;
        for (ffx::StructType type : backend_types) {
            for (uint32_t v : versions) {
                for (uint32_t extra : {0u, ffx::UPSCALE_ENABLE_DEBUG_CHECKING}) {
                    backend.header.type = type;
                    version.version = v;
                    up.flags = base_flags | extra;
                    rt::d3d12_xs::reset_device_proxy_stats();
                    g_alloc_calls = g_alloc_bytes = g_free_calls = 0;
                    const int messages_before = g_messages;
                    c = guarded_create(create, &ctx, &up.header);
                    char line[256];
                    std::snprintf(line, sizeof(line),
                                  "ffxCreateContext(backend %#llx, version %u.%u.%u, flags %#x) -> %s; XMemAlloc %llu "
                                  "(%llu bytes), XMemFree %llu, messages %d",
                                  static_cast<unsigned long long>(type), v >> 22, (v >> 12) & 0x3ff, v & 0xfff, up.flags,
                                  describe(c).c_str(), static_cast<unsigned long long>(g_alloc_calls),
                                  static_cast<unsigned long long>(g_alloc_bytes),
                                  static_cast<unsigned long long>(g_free_calls), g_messages - messages_before);
                    note(line);
                    created = c.faulted || (c.rc == ffx::RETURN_OK && ctx);
                    if (created) break;
                }
                if (created) break;
            }
            if (created) break;
        }
        if (!created) {
            // Control: the same chain with no backend descriptor at all. If this fails the same way, the DLL does
            // not recognise any of the backend descriptors above.
            version.header.pNext = nullptr;
            version.version = ffx::upscaler_version(4, 1, 1);
            up.flags = base_flags | ffx::UPSCALE_ENABLE_DEBUG_CHECKING;
            c = guarded_create(create, &ctx, &up.header);
            note("ffxCreateContext(no backend descriptor, 4.1.1) -> " + describe(c));
            version.header.pNext = &backend.header;
        }
        print_slot_hits(proxy);
        check(!c.faulted, "ffxCreateContext returns without a fault");
        if (!c.faulted && c.rc == ffx::RETURN_OK && ctx) {
            c = guarded_destroy(destroy, &ctx);
            note("ffxDestroyContext -> " + describe(c));
            check(!c.faulted && c.rc == ffx::RETURN_OK, "ffxDestroyContext succeeds");
        }
        note("messages from the DLL: " + std::to_string(g_messages));
        reinterpret_cast<IUnknown*>(proxy)->Release();
    } catch (const std::exception& e) {
        std::printf("FAIL %s\n", e.what());
        return 1;
    }

    std::printf("fsr_dispatch_test: %s (%d failure%s)\n", g_failures ? "FAILED" : "OK", g_failures,
                g_failures == 1 ? "" : "s");
    return g_failures ? 1 : 0;
}
