// Checks the registered xgameruntime shim the way a GDK title uses it at boot.
//
// Every function is resolved through rt::shims::find("xgameruntime.dll", name), the same table the loader binds
// imports to. The sequence follows a typical title: initialize the runtime, query the device, create a queue whose
// work runs on the thread pool and whose completions the game loop dispatches by hand, run async work on it and on
// the process queue, drive a custom XAsyncBegin provider to a result, then uninitialize. No game data; nothing
// online.
#include <windows.h>

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <XAsync.h>
#include <XAsyncProvider.h>
#include <XTaskQueue.h>

#include "shims.h"

namespace {

int g_checks = 0;
int g_failures = 0;

void check(bool condition, const char* what) {
    ++g_checks;
    std::printf("%s %s\n", condition ? "PASS" : "FAIL", what);
    if (!condition) ++g_failures;
}

template <class Fn>
Fn get(const char* name) {
    return reinterpret_cast<Fn>(rt::shims::find("xgameruntime.dll", name));
}

using InitFn = HRESULT(__stdcall*)();
using UninitFn = void(__stdcall*)();
using DeviceTypeFn = uint32_t(__stdcall*)();
using ConsoleIdFn = HRESULT(__stdcall*)(size_t, char*, size_t*);
using QueueCreateFn = HRESULT(__stdcall*)(XTaskQueueDispatchMode, XTaskQueueDispatchMode, XTaskQueueHandle*);
using QueueCloseFn = void(__stdcall*)(XTaskQueueHandle);
using QueueDispatchFn = bool(__stdcall*)(XTaskQueueHandle, XTaskQueuePort, uint32_t);
using QueueProcessFn = bool(__stdcall*)(XTaskQueueHandle*);
using AsyncRunFn = HRESULT(__stdcall*)(XAsyncBlock*, XAsyncWork*);
using AsyncStatusFn = HRESULT(__stdcall*)(XAsyncBlock*, bool);
using AsyncBeginFn = HRESULT(__stdcall*)(XAsyncBlock*, void*, const void*, const char*, XAsyncProvider*);
using AsyncScheduleFn = HRESULT(__stdcall*)(XAsyncBlock*, uint32_t);
using AsyncCompleteFn = void(__stdcall*)(XAsyncBlock*, HRESULT, size_t);
using AsyncResultSizeFn = HRESULT(__stdcall*)(XAsyncBlock*, size_t*);
using AsyncResultFn = HRESULT(__stdcall*)(XAsyncBlock*, const void*, size_t, void*, size_t*);

struct XVersion {
    union {
        struct {
            uint16_t major;
            uint16_t minor;
            uint16_t build;
            uint16_t revision;
        };
        uint64_t Value;
    };
};
static_assert(sizeof(XVersion) == 8, "XVersion must be 8 bytes per GDK spec");

struct XSystemAnalyticsInfo {
    XVersion osVersion;
    XVersion hostingOsVersion;
    char family[64];
    char form[64];
};
static_assert(sizeof(XSystemAnalyticsInfo) == 144, "XSystemAnalyticsInfo must be 144 bytes per GDK spec");

using AnalyticsInfoFn = XSystemAnalyticsInfo(__stdcall*)();
using PackageIdentifierFn = HRESULT(__stdcall*)(size_t, char*);
using PackageMountPathFn = HRESULT(__stdcall*)(void*, size_t, char*);
using PackagedProcessFn = bool(__stdcall*)();

const char* const kNames[] = {
    "XGameRuntimeInitialize", "XGameRuntimeUninitialize", "XSystemGetDeviceType", "XSystemGetConsoleId",
    "XSystemGetAnalyticsInfo", "XPackageGetCurrentProcessPackageIdentifier", "XPackageGetMountPath",
    "XPackageIsPackagedProcess",
    "XTaskQueueCreate", "XTaskQueueCreateComposite", "XTaskQueueGetPort", "XTaskQueueDuplicateHandle",
    "XTaskQueueCloseHandle", "XTaskQueueTerminate", "XTaskQueueDispatch", "XTaskQueueSubmitCallback",
    "XTaskQueueSubmitDelayedCallback", "XTaskQueueRegisterWaiter", "XTaskQueueUnregisterWaiter",
    "XTaskQueueRegisterMonitor", "XTaskQueueUnregisterMonitor", "XTaskQueueGetCurrentProcessTaskQueue",
    "XTaskQueueSetCurrentProcessTaskQueue", "XAsyncGetStatus", "XAsyncGetResultSize", "XAsyncCancel", "XAsyncRun",
    "XAsyncBegin", "XAsyncSchedule", "XAsyncComplete", "XAsyncGetResult",
};

std::atomic<int> g_work_runs{0};
std::atomic<DWORD> g_work_thread{0};
std::atomic<int> g_completions{0};
std::atomic<DWORD> g_completion_thread{0};

HRESULT CALLBACK work(XAsyncBlock* b) {
    g_work_thread = GetCurrentThreadId();
    ++g_work_runs;
    *static_cast<int*>(b->context) = 42;
    return S_OK;
}

void CALLBACK completed(XAsyncBlock*) {
    g_completion_thread = GetCurrentThreadId();
    ++g_completions;
}

// A provider that produces a 4-byte result on the queue's work port: the shape GDK async APIs use internally.
struct ProviderState {
    AsyncCompleteFn complete;
    std::atomic<int> ops{0};
};

HRESULT CALLBACK provider(XAsyncOp op, const XAsyncProviderData* data) {
    auto* st = static_cast<ProviderState*>(data->context);
    st->ops |= 1 << static_cast<int>(op);
    switch (op) {
        case XAsyncOp::DoWork:
            st->complete(data->async, S_OK, sizeof(uint32_t));
            return S_OK;
        case XAsyncOp::GetResult:
            *static_cast<uint32_t*>(data->buffer) = 0x1EA0;
            return S_OK;
        default:
            return S_OK;
    }
}

HMODULE g_teardown_rt_mod = nullptr;
FARPROC(WINAPI* g_teardown_get_proc)(HMODULE, LPCSTR) = nullptr;
BOOL(WINAPI* g_teardown_free_lib)(HMODULE) = nullptr;

// Registered with std::atexit before check_run_time_load() constructs any function-local statics in library_loader.
// LIFO atexit order means this runs after those statics would have been destroyed, simulating mapped-DLL
// DLL_PROCESS_DETACH or earlier atexit handlers calling GetProcAddress/FreeLibrary on a pseudo-handle during exit.
void check_late_teardown() {
    if (!g_teardown_rt_mod || !g_teardown_get_proc || !g_teardown_free_lib) return;
    if (g_teardown_get_proc(g_teardown_rt_mod, "UninitializeApiImpl") == nullptr) {
        std::fprintf(stderr, "FAIL late teardown GetProcAddress on pseudo-handle\n");
        std::_Exit(1);
    }
    SetLastError(0);
    if (g_teardown_get_proc(g_teardown_rt_mod, "LateTeardownMissingSymbol") != nullptr ||
        GetLastError() != ERROR_PROC_NOT_FOUND) {
        std::fprintf(stderr, "FAIL late teardown unserved GetProcAddress on pseudo-handle\n");
        std::_Exit(1);
    }
    if (g_teardown_free_lib(g_teardown_rt_mod) != TRUE) {
        std::fprintf(stderr, "FAIL late teardown FreeLibrary on pseudo-handle\n");
        std::_Exit(1);
    }
}

// xgameruntime.lib's thunks don't import the runtime: they LoadLibraryW(L"xgameruntime.dll") and GetProcAddress five
// names (measured in the package's libHttpClient.GDK.dll and XCurl.dll). Drive that path through the loader shims,
// bound by the api set XCurl.dll imports them from, as a mapped image would be.
void check_run_time_load() {
    using LoadW = HMODULE(WINAPI*)(LPCWSTR);
    using LoadExA = HMODULE(WINAPI*)(LPCSTR, HANDLE, DWORD);
    using GetModW = HMODULE(WINAPI*)(LPCWSTR);
    using GetProc = FARPROC(WINAPI*)(HMODULE, LPCSTR);
    using Free = BOOL(WINAPI*)(HMODULE);
    const char* set = "api-ms-win-core-libraryloader-l1-2-0.dll";
    auto load_w = reinterpret_cast<LoadW>(rt::shims::find(set, "LoadLibraryW"));
    auto load_ex_a = reinterpret_cast<LoadExA>(rt::shims::find("kernel32.dll", "LoadLibraryExA"));
    auto get_mod_w = reinterpret_cast<GetModW>(rt::shims::find(set, "GetModuleHandleW"));
    auto get_proc = reinterpret_cast<GetProc>(rt::shims::find(set, "GetProcAddress"));
    auto free_lib = reinterpret_cast<Free>(rt::shims::find(set, "FreeLibrary"));
    check(load_w && load_ex_a && get_mod_w && get_proc && free_lib, "the loader functions are registered");
    if (!(load_w && load_ex_a && get_mod_w && get_proc && free_lib)) return;

    check(rt::shims::owns_dll("xgameruntime.dll") && !rt::shims::owns_dll("kernel32.dll") &&
              !rt::shims::owns_dll(set) && !rt::shims::owns_dll("nosuch.dll"),
          "owns_dll: xgameruntime.dll yes; kernel32, api sets and unknown modules no");

    HMODULE rt_mod = load_w(L"xgameruntime.dll");
    check(rt_mod != nullptr, "LoadLibraryW(xgameruntime.dll) is served without the host runtime");
    check(load_w(L"XGameRuntime") == rt_mod && load_ex_a("C:\\Some\\Dir\\XGAMERUNTIME.DLL", nullptr, 0) == rt_mod &&
              get_mod_w(L"xgameruntime.dll") == rt_mod,
          "the same module for a bare name, a full path in any case, LoadLibraryExA and GetModuleHandleW");

    const char* entry_points[] = {"InitializeApiImpl", "InitializeApiImplEx", "InitializeApiImplEx2",
                                  "UninitializeApiImpl", "QueryApiImpl"};
    bool all = true;
    for (const char* n : entry_points)
        all = all && get_proc(rt_mod, n) &&
              reinterpret_cast<void*>(get_proc(rt_mod, n)) == rt::shims::find("xgameruntime.dll", n);
    check(all, "GetProcAddress serves the five entry points xgameruntime.lib looks up");
    SetLastError(0);
    check(get_proc(rt_mod, "NoSuchRuntimeFunction") == nullptr && GetLastError() == ERROR_PROC_NOT_FOUND,
          "an unserved name is NULL with ERROR_PROC_NOT_FOUND, as from a real module");
    SetLastError(0);
    check(get_proc(rt_mod, nullptr) == nullptr && GetLastError() == ERROR_PROC_NOT_FOUND,
          "a null proc name is NULL with ERROR_PROC_NOT_FOUND");

    struct Options {
        uint32_t unknown;
        BOOL is_inline;
        const char* game_config;
    } options{1, FALSE, nullptr};
    using InitEx2 = HRESULT(WINAPI*)(ULONG, ULONG, char, const Options*);
    using Query = HRESULT(WINAPI*)(const GUID*, REFIID, void**);
    using Uninit = HRESULT(WINAPI*)();
    auto init_ex2 = reinterpret_cast<InitEx2>(get_proc(rt_mod, "InitializeApiImplEx2"));
    auto query = reinterpret_cast<Query>(get_proc(rt_mod, "QueryApiImpl"));
    auto uninit_api = reinterpret_cast<Uninit>(get_proc(rt_mod, "UninitializeApiImpl"));
    check(init_ex2(0x2410, 0x0, 0x02, &options) == S_OK, "InitializeApiImplEx2 succeeds offline");
    const GUID some_class = {0x973a344e, 0x24bf, 0x4d0f, {0x84, 0x57, 0x56, 0xc5, 0x34, 0x89, 0x2b, 0x29}};
    void* out = reinterpret_cast<void*>(1);
    check(query(&some_class, some_class, &out) == static_cast<HRESULT>(0x80070032) && out == nullptr,
          "QueryApiImpl refuses a class it doesn't provide yet (HRESULT 0x80070032) and clears the out pointer");

    // XThreadingImpl and XErrorImpl, driven the way xgameruntime.lib's thunks in the package's DLLs do: QueryApiImpl
    // with the class id as both ids, one vtable slot with `this` first, then Release (slot 2).
    const GUID threading = {0x073b7dcb, 0x1fcf, 0x4030, {0x94, 0xbe, 0xe3, 0xc9, 0xeb, 0x62, 0x34, 0x28}};
    const GUID error_cls = {0x8ca467f7, 0x22e8, 0x4096, {0x84, 0x56, 0xbb, 0x8a, 0xa1, 0x3f, 0x79, 0xd8}};
    using Fn = void*;
    auto slot = [](void* obj, int index) { return (*static_cast<Fn* const*>(obj))[index]; };
    auto release = [&](void* obj) { reinterpret_cast<ULONG(STDMETHODCALLTYPE*)(void*)>(slot(obj, 2))(obj); };
    auto get = [&](const GUID& cls) {
        void* p = nullptr;
        return query(&cls, cls, &p) == S_OK ? p : nullptr;
    };
    void* t = get(threading);
    check(t != nullptr, "QueryApiImpl hands out XThreadingImpl for its class id");
    if (t) {
        void* again = get(threading);
        check(again == t, "XThreadingImpl is one process-wide object");
        release(again);
        XTaskQueueHandle queue = nullptr;
        using Create = HRESULT(STDMETHODCALLTYPE*)(void*, XTaskQueueDispatchMode, XTaskQueueDispatchMode,
                                                  XTaskQueueHandle*);
        check(reinterpret_cast<Create>(slot(t, 12))(t, XTaskQueueDispatchMode::Manual, XTaskQueueDispatchMode::Manual,
                                                    &queue) == S_OK && queue,
              "slot 12 is XTaskQueueCreate");
        static std::atomic<int> s_ran{0};
        using Submit = HRESULT(STDMETHODCALLTYPE*)(void*, XTaskQueueHandle, XTaskQueuePort, void*,
                                                  XTaskQueueCallback*);
        XTaskQueueCallback* cb = [](void* ctx, bool canceled) {
            if (!canceled) static_cast<std::atomic<int>*>(ctx)->fetch_add(1);
        };
        check(reinterpret_cast<Submit>(slot(t, 18))(t, queue, XTaskQueuePort::Work, &s_ran, cb) == S_OK,
              "slot 18 is XTaskQueueSubmitCallback");
        using Dispatch = bool(STDMETHODCALLTYPE*)(void*, XTaskQueueHandle, XTaskQueuePort, uint32_t);
        check(reinterpret_cast<Dispatch>(slot(t, 16))(t, queue, XTaskQueuePort::Work, 0) && s_ran.load() == 1,
              "slot 16 is XTaskQueueDispatch: the callback ran on the work port");
        XTaskQueueHandle dup = nullptr;
        using Dup = HRESULT(STDMETHODCALLTYPE*)(void*, XTaskQueueHandle, XTaskQueueHandle*);
        check(reinterpret_cast<Dup>(slot(t, 15))(t, queue, &dup) == S_OK && dup,
              "slot 15 is XTaskQueueDuplicateHandle");
        using Close = void(STDMETHODCALLTYPE*)(void*, XTaskQueueHandle);
        reinterpret_cast<Close>(slot(t, 17))(t, dup);
        reinterpret_cast<Close>(slot(t, 17))(t, queue);
        using SetSensitive = HRESULT(STDMETHODCALLTYPE*)(void*, bool);
        using IsSensitive = bool(STDMETHODCALLTYPE*)(void*);
        check(reinterpret_cast<SetSensitive>(slot(t, 27))(t, true) == S_OK &&
                  reinterpret_cast<IsSensitive>(slot(t, 30))(t) &&
                  reinterpret_cast<SetSensitive>(slot(t, 27))(t, false) == S_OK &&
                  !reinterpret_cast<IsSensitive>(slot(t, 30))(t),
              "slots 27 and 30 set and read this thread's time-sensitive flag");
        void* unk = nullptr;
        const GUID iunknown = {0, 0, 0, {0xc0, 0, 0, 0, 0, 0, 0, 0x46}};
        using QI = HRESULT(STDMETHODCALLTYPE*)(void*, REFIID, void**);
        check(reinterpret_cast<QI>(slot(t, 0))(t, iunknown, &unk) == S_OK && unk == t &&
                  reinterpret_cast<QI>(slot(t, 0))(t, error_cls, &unk) == E_NOINTERFACE && !unk,
              "QueryInterface answers IUnknown and the class id, not another class");
        release(t);
        release(t);
    }
    void* e = get(error_cls);
    check(e != nullptr, "QueryApiImpl hands out XErrorImpl for its class id");
    if (e) {
        struct Seen {
            HRESULT hr;
            char msg[32];
        } seen{};
        using Callback = bool(CALLBACK*)(HRESULT, const char*, void*);
        Callback on_error = [](HRESULT hr, const char* msg, void* ctx) {
            auto* s = static_cast<Seen*>(ctx);
            s->hr = hr;
            strncpy_s(s->msg, msg ? msg : "", _TRUNCATE);
            return true;
        };
        using SetCallback = void(STDMETHODCALLTYPE*)(void*, Callback, void*);
        using Report = void(STDMETHODCALLTYPE*)(void*, HRESULT, const char*);
        reinterpret_cast<SetCallback>(slot(e, 4))(e, on_error, &seen);
        reinterpret_cast<Report>(slot(e, 3))(e, E_FAIL, "offline test");
        check(seen.hr == E_FAIL && std::strcmp(seen.msg, "offline test") == 0,
              "slot 3 is XErrorReport and reaches the callback slot 4 (XErrorSetCallback) set");
        reinterpret_cast<SetCallback>(slot(e, 4))(e, nullptr, nullptr);
        using SetOptions = HRESULT(STDMETHODCALLTYPE*)(void*, uint32_t, uint32_t);
        check(reinterpret_cast<SetOptions>(slot(e, 5))(e, 0, 0) == S_OK, "slot 5 is XErrorSetOptions");
        release(e);
    }

    const GUID networking_cls = {0x37e56907, 0x2f10, 0x41e8, {0xb7, 0x2f, 0x36, 0xed, 0xb1, 0x85, 0x33, 0x1a}};
    void* n = get(networking_cls);
    check(n != nullptr, "QueryApiImpl hands out XNetworkingImpl for its class id");
    if (n) {
        void* again = get(networking_cls);
        check(again == n, "XNetworkingImpl is one process-wide object");
        release(again);

        // Slot 15: XNetworkingGetConnectivityHint
        struct ConnectivityHint {
            uint32_t level;
            uint32_t cost;
            uint32_t iana_type;
            bool network_initialized;
            bool approaching_limit;
            bool over_limit;
            bool roaming;
        } hint{};
        using GetHint = HRESULT(STDMETHODCALLTYPE*)(void*, ConnectivityHint*);
        check(reinterpret_cast<GetHint>(slot(n, 15))(n, nullptr) == E_POINTER,
              "slot 15 rejects null pointer with E_POINTER");
        check(reinterpret_cast<GetHint>(slot(n, 15))(n, &hint) == S_OK && hint.network_initialized &&
                  hint.level == 3 && hint.cost == 1,
              "slot 15 is XNetworkingGetConnectivityHint reporting initialized network and internet access");

        // Slot 14: XNetworkingVerifyServerCertificate
        using VerifyCert = HRESULT(STDMETHODCALLTYPE*)(void*, void*, const void*);
        check(reinterpret_cast<VerifyCert>(slot(n, 14))(n, nullptr, nullptr) == S_OK,
              "slot 14 is XNetworkingVerifyServerCertificate returning S_OK offline");

        // Slot 16 & 17: XNetworkingRegisterConnectivityHintChanged and Unregister
        using RegisterHint = HRESULT(STDMETHODCALLTYPE*)(void*, XTaskQueueHandle, void*, void*, XTaskQueueRegistrationToken*);
        using UnregisterHint = bool(STDMETHODCALLTYPE*)(void*, XTaskQueueRegistrationToken, bool);
        XTaskQueueRegistrationToken tok1{0}, tok2{0};
        check(reinterpret_cast<RegisterHint>(slot(n, 16))(n, nullptr, nullptr, nullptr, &tok1) == S_OK &&
                  reinterpret_cast<RegisterHint>(slot(n, 16))(n, nullptr, nullptr, nullptr, &tok2) == S_OK &&
                  tok1.token != tok2.token && tok1.token > 0,
              "slot 16 registers connectivity hint callbacks and hands out tokens");
        check(reinterpret_cast<UnregisterHint>(slot(n, 17))(n, tok1, false) == true &&
                  reinterpret_cast<UnregisterHint>(slot(n, 17))(n, tok2, true) == true,
              "slot 17 unregisters connectivity hint callbacks");

        // Slots 11-13: Async security info query
        using QuerySecAsync = HRESULT(STDMETHODCALLTYPE*)(void*, const wchar_t*, XAsyncBlock*);
        using QuerySecSize = HRESULT(STDMETHODCALLTYPE*)(void*, XAsyncBlock*, size_t*);
        using QuerySecResult = HRESULT(STDMETHODCALLTYPE*)(void*, XAsyncBlock*, size_t, size_t*, uint8_t*, void**);
        check(reinterpret_cast<QuerySecAsync>(slot(n, 11))(n, L"https://example.com", nullptr) == S_OK,
              "slot 11 is XNetworkingQuerySecurityInformationForUrlUtf16Async");
        size_t sec_size = 42;
        check(reinterpret_cast<QuerySecSize>(slot(n, 12))(n, nullptr, &sec_size) == S_OK && sec_size == 0,
              "slot 12 is XNetworkingQuerySecurityInformationForUrlUtf16AsyncResultSize returning 0 bytes");
        size_t used = 42;
        void* sec_info = reinterpret_cast<void*>(1);
        check(reinterpret_cast<QuerySecResult>(slot(n, 13))(n, nullptr, 0, &used, nullptr, &sec_info) == S_OK &&
                  used == 0 && sec_info == nullptr,
              "slot 13 is XNetworkingQuerySecurityInformationForUrlUtf16AsyncResult returning S_OK with null result");

        release(n);
    }
    check(uninit_api() == S_OK, "UninitializeApiImpl succeeds");

    HMODULE k32 = load_w(L"kernel32.dll");
    check(k32 == GetModuleHandleW(L"kernel32.dll"), "a host module still loads from the host");
    check(reinterpret_cast<void*>(get_proc(k32, "GetTickCount")) ==
              reinterpret_cast<void*>(GetProcAddress(k32, "GetTickCount")),
          "GetProcAddress on a host module is the host's");
    check(free_lib(rt_mod) == TRUE && load_w(L"xgameruntime.dll") == rt_mod,
          "FreeLibrary on the runtime succeeds and keeps it loaded");

    g_teardown_rt_mod = rt_mod;
    g_teardown_get_proc = get_proc;
    g_teardown_free_lib = free_lib;
}

}  // namespace

int main() {
    std::printf("xgameruntime_test: the offline GDK runtime shim, through the shim registry\n");

    int missing = 0;
    for (const char* n : kNames)
        if (!rt::shims::find("xgameruntime.dll", n)) {
            std::printf("     not registered: %s\n", n);
            ++missing;
        }
    check(missing == 0, "every xgameruntime.dll function in the table resolves through rt::shims::find");
    if (missing) {
        std::printf("xgameruntime_test: %d of %d checks failed\n", g_failures, g_checks);
        return 1;
    }

    auto init = get<InitFn>("XGameRuntimeInitialize");
    auto uninit = get<UninitFn>("XGameRuntimeUninitialize");
    check(init() == S_OK, "XGameRuntimeInitialize returns S_OK");
    check(init() == S_OK, "a second XGameRuntimeInitialize also returns S_OK (reference counted)");

    check(get<DeviceTypeFn>("XSystemGetDeviceType")() == 7, "XSystemGetDeviceType reports XboxScarlettAnaconda (7)");

    auto console_id = get<ConsoleIdFn>("XSystemGetConsoleId");
    char id[64] = {};
    size_t used = 0;
    check(console_id(8, id, &used) == HRESULT_FROM_WIN32(ERROR_INSUFFICIENT_BUFFER),
          "XSystemGetConsoleId rejects a buffer smaller than XSystemConsoleIdBytes");
    check(console_id(sizeof(id), id, &used) == S_OK && used == 39 && std::strlen(id) == 38,
          "XSystemGetConsoleId fills 39 bytes (38 characters and the terminator)");

    auto analytics_info = get<AnalyticsInfoFn>("XSystemGetAnalyticsInfo");
    XSystemAnalyticsInfo info = analytics_info();
    check(info.osVersion.major == 10 && info.osVersion.minor == 0 && info.osVersion.build == 22000 &&
              info.osVersion.revision == 0,
          "XSystemGetAnalyticsInfo returns osVersion 10.0.22000.0");
    check(info.hostingOsVersion.major == 10 && info.hostingOsVersion.minor == 0 &&
              info.hostingOsVersion.build == 22000 && info.hostingOsVersion.revision == 0,
          "XSystemGetAnalyticsInfo returns hostingOsVersion 10.0.22000.0");
    check(std::strcmp(info.family, "Xbox") == 0, "XSystemGetAnalyticsInfo returns family 'Xbox'");
    check(std::strcmp(info.form, "Xbox Series X") == 0, "XSystemGetAnalyticsInfo returns form 'Xbox Series X'");

    auto package_id = get<PackageIdentifierFn>("XPackageGetCurrentProcessPackageIdentifier");
    check(package_id(0, nullptr) == E_INVALIDARG, "XPackageGetCurrentProcessPackageIdentifier rejects null buffer and size 0");
    char pkg_id[64] = {};
    check(package_id(0, pkg_id) == E_INVALIDARG, "XPackageGetCurrentProcessPackageIdentifier rejects size 0 with buffer");
    check(package_id(10, nullptr) == E_INVALIDARG, "XPackageGetCurrentProcessPackageIdentifier rejects null pointer with size");
    check(package_id(16, pkg_id) == HRESULT_FROM_WIN32(ERROR_INSUFFICIENT_BUFFER),
          "XPackageGetCurrentProcessPackageIdentifier rejects insufficient buffer");
    check(package_id(sizeof(pkg_id), pkg_id) == S_OK && std::strlen(pkg_id) == 32,
          "XPackageGetCurrentProcessPackageIdentifier fills 33-byte package identifier");

    SetEnvironmentVariableA("LEONIDA_PACKAGE_ID", "CustomTestPackageId_123456789012");
    check(package_id(sizeof(pkg_id), pkg_id) == S_OK &&
              std::strcmp(pkg_id, "CustomTestPackageId_123456789012") == 0,
          "XPackageGetCurrentProcessPackageIdentifier respects LEONIDA_PACKAGE_ID override");
    SetEnvironmentVariableA("LEONIDA_PACKAGE_ID", nullptr);

    auto mount_path = get<PackageMountPathFn>("XPackageGetMountPath");
    check(mount_path(nullptr, 0, nullptr) == E_INVALIDARG, "XPackageGetMountPath rejects null buffer and size 0");
    char mnt_path[MAX_PATH] = {};
    check(mount_path(nullptr, 0, mnt_path) == E_INVALIDARG, "XPackageGetMountPath rejects size 0 with buffer");
    check(mount_path(nullptr, 10, nullptr) == E_INVALIDARG, "XPackageGetMountPath rejects null pointer with size");

    SetEnvironmentVariableA("LEONIDA_GAME_DIR", "C:\\LeonidaGameDir");
    check(mount_path(nullptr, 5, mnt_path) == HRESULT_FROM_WIN32(ERROR_INSUFFICIENT_BUFFER),
          "XPackageGetMountPath rejects insufficient buffer");
    check(mount_path(nullptr, sizeof(mnt_path), mnt_path) == S_OK &&
              std::strcmp(mnt_path, "C:\\LeonidaGameDir") == 0,
          "XPackageGetMountPath returns configured game directory");
    SetEnvironmentVariableA("LEONIDA_GAME_DIR", nullptr);

    auto is_packaged = get<PackagedProcessFn>("XPackageIsPackagedProcess");
    check(is_packaged() == true, "XPackageIsPackagedProcess returns true by default");
    SetEnvironmentVariableA("LEONIDA_PACKAGED", "0");
    check(is_packaged() == false, "XPackageIsPackagedProcess respects LEONIDA_PACKAGED=0 override");
    SetEnvironmentVariableA("LEONIDA_PACKAGED", nullptr);

    // Work on the thread pool, completions dispatched by the "game loop" on this thread.
    auto create = get<QueueCreateFn>("XTaskQueueCreate");
    auto close = get<QueueCloseFn>("XTaskQueueCloseHandle");
    auto dispatch = get<QueueDispatchFn>("XTaskQueueDispatch");
    auto run = get<AsyncRunFn>("XAsyncRun");
    auto status = get<AsyncStatusFn>("XAsyncGetStatus");
    XTaskQueueHandle q = nullptr;
    check(create(XTaskQueueDispatchMode::ThreadPool, XTaskQueueDispatchMode::Manual, &q) == S_OK && q,
          "XTaskQueueCreate(ThreadPool work, Manual completion)");
    int result = 0;
    XAsyncBlock block{};
    block.queue = q;
    block.context = &result;
    block.callback = completed;
    check(run(&block, work) == S_OK, "XAsyncRun accepts work on that queue");
    for (int i = 0; i < 500 && g_completions == 0; ++i) {
        while (dispatch(q, XTaskQueuePort::Completion, 0)) {}
        if (!g_completions) Sleep(2);
    }
    check(g_work_runs == 1 && result == 42, "the work callback ran once and wrote its result");
    check(g_work_thread != GetCurrentThreadId(), "the work callback ran on a thread-pool thread");
    check(g_completions == 1 && g_completion_thread == GetCurrentThreadId(),
          "the completion callback ran once, on the thread that dispatched the completion port");
    check(status(&block, false) == S_OK, "XAsyncGetStatus reports S_OK after completion");
    close(q);

    // The process task queue, used when XAsyncBlock::queue is null.
    XTaskQueueHandle pq = nullptr;
    check(get<QueueProcessFn>("XTaskQueueGetCurrentProcessTaskQueue")(&pq) && pq,
          "XTaskQueueGetCurrentProcessTaskQueue returns a queue");
    if (pq) close(pq);
    int result2 = 0;
    XAsyncBlock block2{};
    block2.context = &result2;
    check(run(&block2, work) == S_OK && status(&block2, true) == S_OK && result2 == 42,
          "XAsyncRun with a null queue completes on the process queue");

    // A custom provider: XAsyncBegin, XAsyncSchedule, XAsyncComplete, XAsyncGetResultSize, XAsyncGetResult.
    ProviderState st{get<AsyncCompleteFn>("XAsyncComplete")};
    XAsyncBlock block3{};
    static const int kIdentity = 0;
    HRESULT hr = get<AsyncBeginFn>("XAsyncBegin")(&block3, &st, &kIdentity, "xgameruntime_test", provider);
    check(hr == S_OK, "XAsyncBegin starts a provider");
    check(get<AsyncScheduleFn>("XAsyncSchedule")(&block3, 0) == S_OK, "XAsyncSchedule queues its work");
    check(status(&block3, true) == S_OK, "the provider completes with S_OK");
    size_t size = 0;
    check(get<AsyncResultSizeFn>("XAsyncGetResultSize")(&block3, &size) == S_OK && size == sizeof(uint32_t),
          "XAsyncGetResultSize reports the 4-byte result");
    uint32_t value = 0;
    size_t got = 0;
    check(get<AsyncResultFn>("XAsyncGetResult")(&block3, &kIdentity, sizeof(value), &value, &got) == S_OK &&
              value == 0x1EA0 && got == sizeof(value),
          "XAsyncGetResult returns the provider's result");
    // Cleanup comes when the async state's last reference goes, which can be on the work thread just after the
    // result is read, so allow it a moment.
    const int cleanup_bit = 1 << static_cast<int>(XAsyncOp::Cleanup);
    for (int i = 0; i < 500 && !(st.ops & cleanup_bit); ++i) Sleep(2);
    check((st.ops & (1 << static_cast<int>(XAsyncOp::Begin))) && (st.ops & (1 << static_cast<int>(XAsyncOp::DoWork))) &&
              (st.ops & (1 << static_cast<int>(XAsyncOp::GetResult))) && (st.ops & (1 << static_cast<int>(XAsyncOp::Cleanup))),
          "the provider saw Begin, DoWork, GetResult and Cleanup");

    uninit();
    uninit();
    check(init() == S_OK, "XGameRuntimeInitialize works again after balanced uninitialize calls");
    uninit();

    std::atexit(check_late_teardown);
    check_run_time_load();

    if (g_failures) {
        std::printf("xgameruntime_test: %d of %d checks failed\n", g_failures, g_checks);
        return 1;
    }
    std::printf("xgameruntime_test: OK (%d checks)\n", g_checks);
    return 0;
}
