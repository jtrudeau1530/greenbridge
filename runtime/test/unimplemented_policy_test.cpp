// Tests for LEONIDA_UNIMPLEMENTED (log | fail | stop), the return policy of imports that have no binding.
//
// Everything is synthetic: the test writes a small x86-64 DLL image whose two imports exist nowhere (one named like a
// D3D12 factory, so it is classified as returning an HRESULT; one in an unknown DLL, so it is not), plus a caller
// function per import in its .text. The loader maps it, binds both imports to unimplemented thunks, and the test
// calls the image's callers the way the game would. No game data and no GPU are involved.
//
//   in-process   log and fail: return value (0 / E_NOTIMPL), an out-parameter that must stay untouched, and xmm0 = 0
//                for a caller that expects a float result.
//   subprocess   the test starts itself, because `stop` ends the process and the variable is read once per process:
//                  --stop-child   stop mode: exit code, the message (import, caller as module+RVA, rcx..r9)
//                  --env-child    prints the results for whatever LEONIDA_UNIMPLEMENTED the parent set, which pins
//                                 the default (unset = log), case-insensitivity and the response to a bad value.
// LEONIDA_TEST_DIR names the folder for the image, as for loader_test; the default is the temp directory.
#include <windows.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "../shims/shims.h"
#include "../src/pe_loader.h"

namespace {

namespace fs = std::filesystem;
using rt::shims::ReturnClass;
using rt::shims::UnimplementedPolicy;

int g_checks = 0;
int g_failed = 0;

void check(bool ok, const std::string& what) {
    ++g_checks;
    std::printf("%s %s\n", ok ? "ok   " : "FAIL ", what.c_str());
    if (!ok) ++g_failed;
}

struct Import {
    const char* dll;
    const char* name;
};
// Index 0 is classified as HRESULT-returning (a D3D12 factory name no host d3d12 exports), index 1 is not.
const Import kImports[2] = {{"d3d12_xs.dll", "D3D12CreateLeonidaPolicyProbe"},
                            {"leonida_policy_probe.dll", "PolicyProbeOther"}};
constexpr uint32_t kTextRva = 0x1000;
constexpr uint32_t kCallerBase = kTextRva + 0x200;  // caller i is at kCallerBase + 0x10 * i
constexpr uint32_t kCallRetOffset = 10;             // sub rsp,0x28 (4) + call [rip+disp] (6): the return address
constexpr const char* kImageName = "policy_image.bin";

uint32_t caller_rva(int i) { return kCallerBase + 0x10u * static_cast<uint32_t>(i); }
uint32_t return_rva(int i) { return caller_rva(i) + kCallRetOffset; }

// A valid PE32+ DLL image with the two imports above. Layout, as offsets into .text (RVA 0x1000, 0x400 bytes):
//   0x000 import descriptors   0x040/0x070 DLL names   0x0A0/0x0C0 lookup tables   0x0E0/0x100 address tables
//   0x120/0x160 hint/name      0x200.. one caller per import:  sub rsp,0x28; call [iat]; add rsp,0x28; ret
std::vector<uint8_t> make_image() {
    std::vector<uint8_t> b(0x600, 0);
    auto* dos = reinterpret_cast<IMAGE_DOS_HEADER*>(b.data());
    dos->e_magic = IMAGE_DOS_SIGNATURE;
    dos->e_lfanew = 0x80;
    auto* nt = reinterpret_cast<IMAGE_NT_HEADERS64*>(b.data() + dos->e_lfanew);
    nt->Signature = IMAGE_NT_SIGNATURE;
    nt->FileHeader.Machine = IMAGE_FILE_MACHINE_AMD64;
    nt->FileHeader.NumberOfSections = 1;
    nt->FileHeader.SizeOfOptionalHeader = sizeof(IMAGE_OPTIONAL_HEADER64);
    nt->FileHeader.Characteristics = IMAGE_FILE_EXECUTABLE_IMAGE | IMAGE_FILE_DLL | IMAGE_FILE_LARGE_ADDRESS_AWARE;
    nt->OptionalHeader.Magic = IMAGE_NT_OPTIONAL_HDR64_MAGIC;
    nt->OptionalHeader.ImageBase = 0x180000000;
    nt->OptionalHeader.AddressOfEntryPoint = 0;
    nt->OptionalHeader.SizeOfHeaders = 0x200;
    nt->OptionalHeader.SizeOfImage = 0x2000;
    nt->OptionalHeader.Subsystem = IMAGE_SUBSYSTEM_WINDOWS_CUI;
    auto* sec = IMAGE_FIRST_SECTION(nt);
    std::memcpy(sec->Name, ".text", 6);
    sec->VirtualAddress = kTextRva;
    sec->Misc.VirtualSize = 0x400;
    sec->PointerToRawData = 0x200;
    sec->SizeOfRawData = 0x400;
    sec->Characteristics = IMAGE_SCN_CNT_CODE | IMAGE_SCN_MEM_EXECUTE | IMAGE_SCN_MEM_READ;

    uint8_t* text = b.data() + 0x200;
    for (int i = 0; i < 2; ++i) {
        const uint32_t name_off = 0x040 + 0x30u * i, lookup_off = 0x0A0 + 0x20u * i, iat_off = 0x0E0 + 0x20u * i,
                       hint_off = 0x120 + 0x40u * i;
        auto* d = reinterpret_cast<IMAGE_IMPORT_DESCRIPTOR*>(text) + i;
        d->Name = kTextRva + name_off;
        d->OriginalFirstThunk = kTextRva + lookup_off;
        d->FirstThunk = kTextRva + iat_off;
        std::strcpy(reinterpret_cast<char*>(text + name_off), kImports[i].dll);
        *reinterpret_cast<uint64_t*>(text + lookup_off) = kTextRva + hint_off;
        std::strcpy(reinterpret_cast<char*>(text + hint_off + 2), kImports[i].name);

        const uint8_t code[] = {0x48, 0x83, 0xEC, 0x28,        // sub rsp, 0x28
                                0xFF, 0x15, 0, 0, 0, 0,        // call [rip + disp32]
                                0x48, 0x83, 0xC4, 0x28,        // add rsp, 0x28
                                0xC3};                         // ret
        uint8_t* fn = text + (caller_rva(i) - kTextRva);
        std::memcpy(fn, code, sizeof code);
        const int32_t disp = static_cast<int32_t>((kTextRva + iat_off) - return_rva(i));
        std::memcpy(fn + 6, &disp, 4);
    }
    auto& dir = nt->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_IMPORT];
    dir.VirtualAddress = kTextRva;
    dir.Size = 3 * sizeof(IMAGE_IMPORT_DESCRIPTOR);
    return b;
}

fs::path test_dir() {
    const char* env = std::getenv("LEONIDA_TEST_DIR");
    fs::path dir = (env && *env ? fs::path(env) : fs::temp_directory_path()) /
                   ("leonida_unimplemented_policy_" + std::to_string(GetCurrentProcessId()));
    fs::create_directories(dir);
    return dir;
}

fs::path write_image(const fs::path& dir) {
    const auto bytes = make_image();
    const fs::path p = dir / kImageName;
    std::ofstream out(p, std::ios::binary);
    out.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    if (!out) throw std::runtime_error("cannot write " + p.string());
    return p;
}

using CallFn = uint64_t(__cdecl*)(uint64_t*, uint64_t, uint64_t, uint64_t);
using FloatFn = double(__cdecl*)(double);

CallFn caller_of(const rt::Module& m, int i) { return reinterpret_cast<CallFn>(m.base + caller_rva(i)); }

// ---- subprocess helpers ---------------------------------------------------------------------------------------

struct ChildResult {
    DWORD exit_code = 0xFFFFFFFF;
    std::string output;  // stdout and stderr, interleaved
};

// Runs this executable with `args`; `env` is LEONIDA_UNIMPLEMENTED for the child (nullptr = unset).
ChildResult run_child(const std::wstring& args, const wchar_t* env) {
    wchar_t exe[MAX_PATH];
    GetModuleFileNameW(nullptr, exe, MAX_PATH);
    SetEnvironmentVariableW(L"LEONIDA_UNIMPLEMENTED", env);

    SECURITY_ATTRIBUTES sa{sizeof sa, nullptr, TRUE};
    HANDLE read = nullptr, write = nullptr;
    CreatePipe(&read, &write, &sa, 0);
    SetHandleInformation(read, HANDLE_FLAG_INHERIT, 0);
    STARTUPINFOW si{};
    si.cb = sizeof si;
    si.dwFlags = STARTF_USESTDHANDLES;
    si.hStdOutput = write;
    si.hStdError = write;
    si.hStdInput = GetStdHandle(STD_INPUT_HANDLE);
    PROCESS_INFORMATION pi{};
    std::wstring cmd = L"\"" + std::wstring(exe) + L"\" " + args;
    ChildResult r;
    if (CreateProcessW(nullptr, cmd.data(), nullptr, nullptr, TRUE, 0, nullptr, nullptr, &si, &pi)) {
        CloseHandle(write);
        char buf[4096];
        DWORD n;
        while (ReadFile(read, buf, sizeof buf, &n, nullptr) && n) r.output.append(buf, n);
        WaitForSingleObject(pi.hProcess, 30000);
        GetExitCodeProcess(pi.hProcess, &r.exit_code);
        CloseHandle(pi.hProcess);
        CloseHandle(pi.hThread);
    } else {
        CloseHandle(write);
    }
    CloseHandle(read);
    SetEnvironmentVariableW(L"LEONIDA_UNIMPLEMENTED", nullptr);
    return r;
}

bool contains(const std::string& s, const std::string& what) { return s.find(what) != std::string::npos; }

std::string hex(uint64_t v) {
    char b[32];
    std::snprintf(b, sizeof b, "%llx", static_cast<unsigned long long>(v));
    return b;
}

// ---- child modes ----------------------------------------------------------------------------------------------

int stop_child(const fs::path& dir, int which) {
    rt::Loader loader(dir);
    rt::Module& m = loader.load(write_image(dir));
    uint64_t out = 0x5A5A5A5A5A5A5A5Aull;
    std::printf("out_address=%llx\n", static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(&out)));
    std::fflush(stdout);
    caller_of(m, which)(&out, 0x2222, 0x3333, 0x4444);
    std::printf("RETURNED: stop did not stop\n");
    return 0;
}

int env_child(const fs::path& dir) {
    rt::Loader loader(dir);
    rt::Module& m = loader.load(write_image(dir));
    uint64_t out = 0x5A5A5A5A5A5A5A5Aull;
    const uint64_t a = caller_of(m, 0)(&out, 1, 2, 3);
    const uint64_t b = caller_of(m, 1)(&out, 1, 2, 3);
    std::printf("hresult=%08llx other=%08llx out=%llx\n", static_cast<unsigned long long>(a),
                static_cast<unsigned long long>(b), static_cast<unsigned long long>(out));
    return 0;
}

// ---- tests ----------------------------------------------------------------------------------------------------

void test_classification() {
    auto cls = [](const char* dll, const char* name) { return rt::shims::classify_import(dll, name); };
    check(cls("d3d12_xs.dll", "D3D12CreateLeonidaPolicyProbe") == ReturnClass::HResult, "D3D12Create* is hresult");
    check(cls("D3D12.dll", "D3D12CreateDevice") == ReturnClass::HResult, "the DLL name is case-insensitive");
    check(cls("d3d12.dll", "D3D12SerializeRootSignature") == ReturnClass::HResult, "D3D12Serialize* is hresult");
    check(cls("dxgi.dll", "CreateDXGIFactory2") == ReturnClass::HResult, "CreateDXGIFactory2 is hresult");
    check(cls("dxgi.dll", "DXGIGetDebugInterface1") == ReturnClass::HResult, "DXGIGetDebugInterface1 is hresult");
    check(cls("xgameruntime.dll", "XGameRuntimeInitialize") == ReturnClass::HResult, "XGameRuntimeInitialize");
    check(cls("xgameruntime.dll", "XUserAddAsync") == ReturnClass::HResult, "XUser* is hresult");
    check(cls("xgameruntime.dll", "XTaskQueueCreate") == ReturnClass::HResult, "XTaskQueueCreate is hresult");
    check(cls("xgameruntime.dll", "XAsyncGetStatus") == ReturnClass::HResult, "XAsyncGetStatus is hresult");
    check(cls("xgameruntime.dll", "XTaskQueueDispatch") == ReturnClass::Other, "XTaskQueueDispatch returns bool");
    check(cls("xgameruntime.dll", "XUserCloseHandle") == ReturnClass::Other, "XUserCloseHandle returns void");
    check(cls("xgameruntime.dll", "XStoreIsLicenseValid") == ReturnClass::Other, "XStoreIsLicenseValid returns bool");
    check(cls("xgameruntime.dll", "XSystemGetDeviceType") == ReturnClass::Other, "XSystem* is not listed");
    check(cls("xgameruntime.dll", "XGameRuntimeUninitialize") == ReturnClass::Other, "XGameRuntimeUninitialize");
    check(cls("kernel32.dll", "XUserAddAsync") == ReturnClass::Other, "the DLL matters");
    check(cls("leonida_policy_probe.dll", "PolicyProbeOther") == ReturnClass::Other, "unknown import is other");

    bool valid = true;
    check(rt::shims::parse_unimplemented_policy("fail", &valid) == UnimplementedPolicy::Fail && valid, "parse fail");
    check(rt::shims::parse_unimplemented_policy("STOP", &valid) == UnimplementedPolicy::Stop && valid, "parse STOP");
    check(rt::shims::parse_unimplemented_policy(nullptr, &valid) == UnimplementedPolicy::Log && valid, "null is log");
    check(rt::shims::parse_unimplemented_policy("", &valid) == UnimplementedPolicy::Log && valid, "empty is log");
    check(rt::shims::parse_unimplemented_policy("bogus", &valid) == UnimplementedPolicy::Log && !valid,
          "a bad value is log and invalid");
}

void test_in_process(const fs::path& dir) {
    rt::Loader loader(dir);
    rt::Module& m = loader.load(write_image(dir));

    int unimplemented = 0, hresult = 0;
    for (auto& r : m.imports) {
        if (r.binding != rt::Binding::Unimplemented) continue;
        ++unimplemented;
        hresult += rt::shims::classify_import(r.dll, r.name) == ReturnClass::HResult;
    }
    check(unimplemented == 2 && hresult == 1, "the image binds 2 unimplemented imports, 1 of them hresult (" +
                                                   std::to_string(unimplemented) + ", " + std::to_string(hresult) + ")");

    struct Case {
        UnimplementedPolicy policy;
        const char* name;
        uint64_t want_hresult;
    } cases[] = {{UnimplementedPolicy::Log, "log", 0}, {UnimplementedPolicy::Fail, "fail", rt::shims::kENotImpl}};
    for (const auto& c : cases) {
        rt::shims::set_unimplemented_policy(c.policy);
        for (int i = 0; i < 2; ++i) {
            const uint64_t want = i == 0 ? c.want_hresult : 0;
            const std::string tag = std::string(c.name) + ", " + (i == 0 ? "hresult" : "other") + " import: ";
            uint64_t out = 0x5A5A5A5A5A5A5A5Aull;
            const uint64_t got = caller_of(m, i)(&out, 0x2222, 0x3333, 0x4444);
            check(got == want, tag + "returns 0x" + hex(want) + " (got 0x" + hex(got) + ")");
            check(out == 0x5A5A5A5A5A5A5A5Aull, tag + "out-parameter is untouched");

            // A caller that expects a float or double sees 0.0, whatever rax holds.
            const double f = reinterpret_cast<FloatFn>(m.base + caller_rva(i))(3.5);
            check(f == 0.0, tag + "xmm0 is 0.0 for a floating-point caller");
        }
    }
    // The same call again, to see that the stub leaves the stack and callee-saved state as it found them.
    rt::shims::set_unimplemented_policy(UnimplementedPolicy::Fail);
    uint64_t out = 0;
    bool stable = true;
    for (int n = 0; n < 1000; ++n) stable &= caller_of(m, 0)(&out, 1, 2, 3) == rt::shims::kENotImpl;
    check(stable, "1000 repeated calls return the same value");
    rt::shims::set_unimplemented_policy(UnimplementedPolicy::Log);
}

void test_env_and_default(const fs::path& dir) {
    const std::wstring args = L"--env-child \"" + dir.wstring() + L"\"";
    struct Case {
        const wchar_t* env;
        const char* label;
        const char* expect;
        bool warns;
    } cases[] = {
        {nullptr, "unset (the default) behaves as log", "hresult=00000000 other=00000000", false},
        {L"log", "log", "hresult=00000000 other=00000000", false},
        {L"fail", "fail", "hresult=80004001 other=00000000", false},
        {L"FAIL", "FAIL is case-insensitive", "hresult=80004001 other=00000000", false},
        {L"bogus", "a bad value falls back to log with a warning", "hresult=00000000 other=00000000", true},
    };
    for (const auto& c : cases) {
        const ChildResult r = run_child(args, c.env);
        const std::string tag = std::string("LEONIDA_UNIMPLEMENTED ") + c.label + ": ";
        check(r.exit_code == 0, tag + "child exits 0 (" + std::to_string(r.exit_code) + ")");
        check(contains(r.output, c.expect), tag + "results '" + c.expect + "'");
        check(contains(r.output, "out=5a5a5a5a5a5a5a5a"), tag + "out-parameter untouched");
        check(contains(r.output, "expected log, fail or stop") == c.warns, tag + (c.warns ? "warns" : "does not warn"));
    }
}

void test_stop(const fs::path& dir) {
    for (int i = 0; i < 2; ++i) {
        const std::string tag = std::string("stop, ") + (i == 0 ? "hresult" : "other") + " import: ";
        const ChildResult r = run_child(L"--stop-child \"" + dir.wstring() + L"\" " + std::to_wstring(i), L"stop");
        const std::string import = std::string(kImports[i].dll) + "!" + kImports[i].name;
        check(r.exit_code == 3, tag + "the process ends with exit code 3 (" + std::to_string(r.exit_code) + ")");
        check(!contains(r.output, "RETURNED"), tag + "the call does not return");
        check(contains(r.output, "unimplemented import " + import + " called"), tag + "names the import");
        check(contains(r.output, std::string(" = ") + kImageName + "+0x" + hex(return_rva(i))),
              tag + "caller resolves to " + kImageName + "+0x" + hex(return_rva(i)));
        const auto at = r.output.find("out_address=");
        const std::string out_address = at == std::string::npos ? "?" : r.output.substr(at + 12, r.output.find_first_of("\r\n", at) - at - 12);
        check(contains(r.output, "rcx=0x" + out_address + " rdx="), tag + "prints rcx as passed (the out pointer " + out_address + ")");
        check(contains(r.output, "rdx=0x2222 r8=0x3333 r9=0x4444"), tag + "prints rdx, r8 and r9 as passed");
        check(contains(r.output, i == 0 ? "return class: hresult" : "return class: other"), tag + "prints the class");
        check(contains(r.output, "no debugger attached"), tag + "says why it ended");
    }
}

}  // namespace

int wmain(int argc, wchar_t** argv) {
    try {
        if (argc >= 4 && std::wstring(argv[1]) == L"--stop-child") return stop_child(argv[2], _wtoi(argv[3]));
        if (argc >= 3 && std::wstring(argv[1]) == L"--env-child") return env_child(argv[2]);

        const fs::path dir = test_dir();
        test_classification();
        test_in_process(dir);
        test_env_and_default(dir);
        test_stop(dir);
        std::error_code ec;
        fs::remove_all(dir, ec);
    } catch (const std::exception& e) {
        std::printf("FAIL  exception: %s\n", e.what());
        return 1;
    }
    std::printf("\n%d checks, %d failed\n", g_checks, g_failed);
    return g_failed ? 1 : 0;
}
