// xgameruntime: the GDK runtime surface a title calls before anything else, offline.
//
// Every GDK title starts with XGameRuntimeInitialize and builds its async work on XTaskQueue/XAsync. This table
// serves those from third_party/xtaskqueue (Microsoft's MIT-licensed implementation from libHttpClient), plus the
// XGameRuntime and XSystem calls a title makes at boot. Signatures and enum values follow Microsoft's public GDK
// documentation (learn.microsoft.com/gaming/gdk); no GDK header or library code is used.
//
// Nothing here goes online.
//
// ASSUMPTION: the console executable is not available, so the DLL name it imports these through is unknown.
// "xgameruntime.dll" is the name the PC GDK uses. If the executable's import table names another module, register
// the same functions under that name too; nothing else changes.
#include <windows.h>

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iterator>

#include <XAsync.h>
#include <XAsyncProvider.h>
#include <XTaskQueue.h>

#include "shims.h"

namespace {

// ---- XGameRuntime ----

std::atomic<int> g_runtime_refs{0};

// ---- The run-time entry points xgameruntime.lib looks up ----
//
// A title linked against xgameruntime.lib doesn't import XGameRuntimeInitialize: the static library's thunks load
// "xgameruntime.dll" and GetProcAddress these five names (measured in the package's libHttpClient.GDK.dll and
// XCurl.dll, docs/runtime_imports.md). Signatures, the options struct and the error for an unknown class follow
// xodus-gaming/xgameruntime-docs (GPL-3.0; facts only, nothing copied): InitializeApiImplEx2 is the one
// XGameRuntimeInitialize[WithOptions] calls, and QueryApiImpl hands out the runtime's COM classes. XThreadingImpl and
// XErrorImpl are served (their layouts are measured, below); for any other class QueryApiImpl logs the class and
// interface id and refuses it, so the first boot with the executable lists the interfaces to implement next.

struct ApiImplOptions {    // InitializeApiImplEx2's fourth argument
    uint32_t unknown;      // documented as always 1
    BOOL is_inline;        // 1 when XGameRuntimeOptions::gameConfigSource is Inline
    const char* game_config;
};

constexpr HRESULT kClassNotProvided = static_cast<HRESULT>(0x80070032);  // HRESULT_FROM_WIN32(ERROR_NOT_SUPPORTED)

std::atomic<int> g_api_refs{0};

HRESULT api_initialize(const char* which, ULONG gdk_ver, ULONG gs_ver, int mode, const ApiImplOptions* options) {
    if (g_api_refs.fetch_add(1) == 0)
        std::fprintf(stderr, "[rt] xgameruntime: %s(gdk=%lu, gs=%lu, mode=%#x, options=%s) (offline)\n", which,
                     gdk_ver, gs_ver, mode & 0xFF,
                     options ? (options->is_inline ? "inline game config" : "default game config") : "none");
    return S_OK;
}

HRESULT WINAPI InitializeApiImpl(ULONG gdk_ver, ULONG gs_ver) {
    return api_initialize("InitializeApiImpl", gdk_ver, gs_ver, -1, nullptr);
}

HRESULT WINAPI InitializeApiImplEx(ULONG gdk_ver, ULONG gs_ver, char mode) {
    return api_initialize("InitializeApiImplEx", gdk_ver, gs_ver, static_cast<unsigned char>(mode), nullptr);
}

HRESULT WINAPI InitializeApiImplEx2(ULONG gdk_ver, ULONG gs_ver, char mode, const ApiImplOptions* options) {
    return api_initialize("InitializeApiImplEx2", gdk_ver, gs_ver, static_cast<unsigned char>(mode), options);
}

HRESULT WINAPI UninitializeApiImpl() {
    int prev = g_api_refs.load();
    while (prev > 0 && !g_api_refs.compare_exchange_weak(prev, prev - 1)) {}
    return S_OK;
}

void print_guid(char (&out)[40], const GUID* g) {
    if (!g) {
        std::snprintf(out, sizeof out, "(null)");
        return;
    }
    std::snprintf(out, sizeof out, "%08lx-%04x-%04x-%02x%02x-%02x%02x%02x%02x%02x%02x", g->Data1, g->Data2,
                  g->Data3, g->Data4[0], g->Data4[1], g->Data4[2], g->Data4[3], g->Data4[4], g->Data4[5],
                  g->Data4[6], g->Data4[7]);
}

// ---- The runtime classes QueryApiImpl hands out ----
//
// xgameruntime.lib's thunk for each function asks QueryApiImpl for the class (passing the class id as the interface
// id too), calls one slot of the object's vtable with the function's own arguments after `this`, and Releases it.
// The slot order is xgameruntime-docs' "Layout" list after IUnknown's three slots, and it is measured: the thunks
// statically linked into the package's libHttpClient.GDK.dll and XCurl.dll (disassembled, their code not run)
// call IXThreadingImpl slots 3-7 and 9-30, each with the argument shape of the documented function in that
// position (XAsyncBegin's five arguments at slot 7, XTaskQueueCreate's two modes and a handle pointer at 12,
// XTaskQueueTerminate's bool at 22, XThreadSetTimeSensitive's bool at 27, ...), and IXErrorImpl slot 3 with an
// HRESULT and a pointer. docs/runtime_imports.md has the table. Each function is served by the implementation the
// named export already uses (third_party/xtaskqueue), so the two ways in behave the same.

constexpr GUID kThreadingClass = {0x073b7dcb, 0x1fcf, 0x4030, {0x94, 0xbe, 0xe3, 0xc9, 0xeb, 0x62, 0x34, 0x28}};
constexpr GUID kErrorClass = {0x8ca467f7, 0x22e8, 0x4096, {0x84, 0x56, 0xbb, 0x8a, 0xa1, 0x3f, 0x79, 0xd8}};
constexpr GUID kNetworkingClass = {0x37e56907, 0x2f10, 0x41e8, {0xb7, 0x2f, 0x36, 0xed, 0xb1, 0x85, 0x33, 0x1a}};

// A function's arguments after `this`, forwarded unchanged (x64 has one calling convention).
// Specialised on the function's exact type, so a noexcept function matches only one form.
template <auto F, class T = decltype(F)> struct Method;
template <auto F, class R, class... A> struct Method<F, R (*)(A...)> {
    static R STDMETHODCALLTYPE call(void*, A... a) { return F(a...); }
};
template <auto F, class R, class... A> struct Method<F, R (*)(A...) noexcept> {
    static R STDMETHODCALLTYPE call(void*, A... a) noexcept { return F(a...); }
};

constexpr GUID kIUnknown = {0x00000000, 0x0000, 0x0000, {0xc0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x46}};

// The runtime's objects are process-wide singletons: AddRef/Release count but never free.
struct RuntimeObject {
    const void* const* vtable;
    const GUID* class_id;
    std::atomic<long> refs{1};
};

HRESULT STDMETHODCALLTYPE object_query_interface(void* self, REFIID iid, void** out) {
    if (!out) return E_POINTER;
    auto* o = static_cast<RuntimeObject*>(self);
    if (iid == kIUnknown || iid == *o->class_id) {
        o->refs.fetch_add(1);
        *out = self;
        return S_OK;
    }
    *out = nullptr;
    return E_NOINTERFACE;
}
ULONG STDMETHODCALLTYPE object_add_ref(void* self) { return static_cast<ULONG>(static_cast<RuntimeObject*>(self)->refs.fetch_add(1) + 1); }
ULONG STDMETHODCALLTYPE object_release(void* self) {
    long prev = static_cast<RuntimeObject*>(self)->refs.load();
    while (prev > 1 && !static_cast<RuntimeObject*>(self)->refs.compare_exchange_weak(prev, prev - 1)) {}
    return static_cast<ULONG>(prev > 1 ? prev - 1 : 1);  // the singleton's own reference stays
}

// A private slot whose signature isn't public: logged once, answers 0 (S_OK / false / nothing to do).
template <int Slot> uint64_t STDMETHODCALLTYPE private_slot(void*) {
    static std::atomic<bool> s_logged{false};
    if (!s_logged.exchange(true))
        std::fprintf(stderr, "[rt] xgameruntime: XThreadingImpl private slot %d called; answering 0\n", Slot);
    return 0;
}

// XThreadSetTimeSensitive / XThreadIsTimeSensitive / XThreadAssertNotTimeSensitive. The flag is per thread, as in
// the GDK; nothing is enforced offline (a time-sensitive thread may still block).
thread_local bool t_time_sensitive = false;
HRESULT STDMETHODCALLTYPE thread_set_time_sensitive(void*, bool value) {
    t_time_sensitive = value;
    return S_OK;
}
void STDMETHODCALLTYPE thread_assert_not_time_sensitive(void*) {}
bool STDMETHODCALLTYPE thread_is_time_sensitive(void*) { return t_time_sensitive; }

#define XM(fn) reinterpret_cast<const void*>(&Method<&fn>::call)
const void* const kThreadingVtable[] = {
    reinterpret_cast<const void*>(&object_query_interface), reinterpret_cast<const void*>(&object_add_ref),
    reinterpret_cast<const void*>(&object_release),
    XM(XAsyncGetStatus),                    // 3
    XM(XAsyncGetResultSize),                // 4
    XM(XAsyncCancel),                       // 5
    XM(XAsyncRun),                          // 6
    XM(XAsyncBegin),                        // 7
    reinterpret_cast<const void*>(&private_slot<8>),  // 8  XAsyncBeginAlloc (private, not called by the package)
    XM(XAsyncSchedule),                     // 9
    XM(XAsyncComplete),                     // 10
    XM(XAsyncGetResult),                    // 11
    XM(XTaskQueueCreate),                   // 12
    XM(XTaskQueueCreateComposite),          // 13
    XM(XTaskQueueGetPort),                  // 14
    XM(XTaskQueueDuplicateHandle),          // 15
    XM(XTaskQueueDispatch),                 // 16
    XM(XTaskQueueCloseHandle),              // 17
    XM(XTaskQueueSubmitCallback),           // 18
    XM(XTaskQueueSubmitDelayedCallback),    // 19
    XM(XTaskQueueRegisterWaiter),           // 20
    XM(XTaskQueueUnregisterWaiter),         // 21
    XM(XTaskQueueTerminate),                // 22
    XM(XTaskQueueRegisterMonitor),          // 23
    XM(XTaskQueueUnregisterMonitor),        // 24
    XM(XTaskQueueGetCurrentProcessTaskQueue),  // 25
    XM(XTaskQueueSetCurrentProcessTaskQueue),  // 26
    reinterpret_cast<const void*>(&thread_set_time_sensitive),         // 27
    reinterpret_cast<const void*>(&private_slot<28>),                  // 28 XTaskQueueIsEmpty (private; no arguments)
    reinterpret_cast<const void*>(&thread_assert_not_time_sensitive),  // 29
    reinterpret_cast<const void*>(&thread_is_time_sensitive),          // 30
};
#undef XM
static_assert(std::size(kThreadingVtable) == 31, "IUnknown + IXThreadingImpl's 28 methods");

// IXErrorImpl: XErrorReport(HRESULT, const char*) at slot 3 (the docs' "_unknown_"; XCurl calls it with an HRESULT
// and a string), XErrorSetCallback(XErrorCallback*, void*) and XErrorSetOptions(options, options). Offline, a report
// is logged and handed to the callback the title set, if any.
using XErrorCallback = bool(CALLBACK*)(HRESULT hr, const char* message, void* context);
std::atomic<XErrorCallback> g_error_callback{nullptr};
std::atomic<void*> g_error_context{nullptr};

void STDMETHODCALLTYPE error_report(void*, HRESULT hr, const char* message) {
    static std::atomic<int> s_logged{0};
    if (s_logged.fetch_add(1) < 64)
        std::fprintf(stderr, "[rt] xgameruntime: XErrorReport(0x%08lx, %s)\n", static_cast<unsigned long>(hr),
                     message ? message : "(null)");
    if (XErrorCallback cb = g_error_callback.load()) cb(hr, message, g_error_context.load());
}
void STDMETHODCALLTYPE error_set_callback(void*, XErrorCallback callback, void* context) {
    g_error_context = context;
    g_error_callback = callback;
}
HRESULT STDMETHODCALLTYPE error_set_options(void*, uint32_t, uint32_t) { return S_OK; }

const void* const kErrorVtable[] = {
    reinterpret_cast<const void*>(&object_query_interface), reinterpret_cast<const void*>(&object_add_ref),
    reinterpret_cast<const void*>(&object_release),
    reinterpret_cast<const void*>(&error_report),        // 3
    reinterpret_cast<const void*>(&error_set_callback),  // 4
    reinterpret_cast<const void*>(&error_set_options),   // 5
};

// IXNetworkingImpl2: 18 methods (slots 3-20). XCurl.dll calls slots 11-17 (docs/runtime_imports.md):
// the UTF-16 XNetworkingQuerySecurityInformationForUrl* trio, XNetworkingVerifyServerCertificate,
// XNetworkingGetConnectivityHint, XNetworkingRegisterConnectivityHintChanged and
// XNetworkingUnregisterConnectivityHintChanged. Offline, connectivity hints report an initialized
// network with unrestricted internet access, certificate verification succeeds, and registered callbacks
// are tracked safely.
enum class XNetworkingConnectivityLevelHint : uint32_t {
    Unknown = 0,
    None = 1,
    LocalAccess = 2,
    InternetAccess = 3,
    ConstrainedInternetAccess = 4,
};

enum class XNetworkingConnectivityCostHint : uint32_t {
    Unknown = 0,
    Unrestricted = 1,
    Fixed = 2,
    Variable = 3,
};

struct XNetworkingConnectivityHint {
    XNetworkingConnectivityLevelHint connectivityLevelHint;
    XNetworkingConnectivityCostHint connectivityCostHint;
    uint32_t ianaInterfaceType;
    bool networkInitialized;
    bool approachingDataLimit;
    bool overDataLimit;
    bool roaming;
};

using XNetworkingConnectivityHintChangedCallback = void(CALLBACK*)(void* context, const XNetworkingConnectivityHint* hint);

template <int Slot> uint64_t STDMETHODCALLTYPE networking_slot(void*) {
    static std::atomic<bool> s_logged{false};
    if (!s_logged.exchange(true))
        std::fprintf(stderr, "[rt] xgameruntime: XNetworkingImpl slot %d called; answering 0\n", Slot);
    return 0;
}

// Slot 11: XNetworkingQuerySecurityInformationForUrlUtf16Async(const wchar_t* url, XAsyncBlock* async)
HRESULT STDMETHODCALLTYPE networking_query_security_info_async(void*, const wchar_t*, XAsyncBlock* async) {
    if (async) XAsyncComplete(async, S_OK, 0);
    return S_OK;
}

// Slot 12: XNetworkingQuerySecurityInformationForUrlUtf16AsyncResultSize(XAsyncBlock* async, size_t* size)
HRESULT STDMETHODCALLTYPE networking_query_security_info_size(void*, XAsyncBlock*, size_t* size) {
    if (size) *size = 0;
    return S_OK;
}

// Slot 13: XNetworkingQuerySecurityInformationForUrlUtf16AsyncResult(XAsyncBlock*, size_t, size_t*, uint8_t*, void**)
HRESULT STDMETHODCALLTYPE networking_query_security_info_result(void*, XAsyncBlock*, size_t, size_t* used, uint8_t*, void** info) {
    if (used) *used = 0;
    if (info) *info = nullptr;
    return S_OK;
}

// Slot 14: XNetworkingVerifyServerCertificate(void* requestHandle, const void* securityInformation)
HRESULT STDMETHODCALLTYPE networking_verify_server_cert(void*, void*, const void*) {
    return S_OK;
}

// Slot 15: XNetworkingGetConnectivityHint(XNetworkingConnectivityHint* hint)
HRESULT STDMETHODCALLTYPE networking_get_connectivity_hint(void*, XNetworkingConnectivityHint* hint) {
    if (!hint) return E_POINTER;
    hint->connectivityLevelHint = XNetworkingConnectivityLevelHint::InternetAccess;
    hint->connectivityCostHint = XNetworkingConnectivityCostHint::Unrestricted;
    hint->ianaInterfaceType = 6;  // MIB_IF_TYPE_ETHERNET
    hint->networkInitialized = true;
    hint->approachingDataLimit = false;
    hint->overDataLimit = false;
    hint->roaming = false;
    return S_OK;
}

// Slot 16: XNetworkingRegisterConnectivityHintChanged(XTaskQueueHandle, void*, Callback, XTaskQueueRegistrationToken*)
std::atomic<uint64_t> g_next_network_token{1};
HRESULT STDMETHODCALLTYPE networking_register_connectivity_hint_changed(void*, XTaskQueueHandle, void*,
                                                                       XNetworkingConnectivityHintChangedCallback,
                                                                       XTaskQueueRegistrationToken* token) {
    if (token) token->token = g_next_network_token.fetch_add(1);
    return S_OK;
}

// Slot 17: XNetworkingUnregisterConnectivityHintChanged(XTaskQueueRegistrationToken token, bool wait)
bool STDMETHODCALLTYPE networking_unregister_connectivity_hint_changed(void*, XTaskQueueRegistrationToken, bool) {
    return true;
}

const void* const kNetworkingVtable[] = {
    reinterpret_cast<const void*>(&object_query_interface),
    reinterpret_cast<const void*>(&object_add_ref),
    reinterpret_cast<const void*>(&object_release),
    reinterpret_cast<const void*>(&networking_slot<3>),   // 3
    reinterpret_cast<const void*>(&networking_slot<4>),   // 4
    reinterpret_cast<const void*>(&networking_slot<5>),   // 5
    reinterpret_cast<const void*>(&networking_slot<6>),   // 6
    reinterpret_cast<const void*>(&networking_slot<7>),   // 7
    reinterpret_cast<const void*>(&networking_slot<8>),   // 8
    reinterpret_cast<const void*>(&networking_slot<9>),   // 9
    reinterpret_cast<const void*>(&networking_slot<10>),  // 10
    reinterpret_cast<const void*>(&networking_query_security_info_async),    // 11
    reinterpret_cast<const void*>(&networking_query_security_info_size),     // 12
    reinterpret_cast<const void*>(&networking_query_security_info_result),   // 13
    reinterpret_cast<const void*>(&networking_verify_server_cert),           // 14
    reinterpret_cast<const void*>(&networking_get_connectivity_hint),        // 15
    reinterpret_cast<const void*>(&networking_register_connectivity_hint_changed),    // 16
    reinterpret_cast<const void*>(&networking_unregister_connectivity_hint_changed),  // 17
    reinterpret_cast<const void*>(&networking_slot<18>),  // 18
    reinterpret_cast<const void*>(&networking_slot<19>),  // 19
    reinterpret_cast<const void*>(&networking_slot<20>),  // 20
};
static_assert(std::size(kNetworkingVtable) == 21, "IUnknown + IXNetworkingImpl2's 18 methods");

RuntimeObject g_threading{kThreadingVtable, &kThreadingClass};
RuntimeObject g_error{kErrorVtable, &kErrorClass};
RuntimeObject g_networking{kNetworkingVtable, &kNetworkingClass};

HRESULT WINAPI QueryApiImpl(const GUID* class_id, REFIID interface_id, void** out) {
    if (out) *out = nullptr;
    if (class_id && out) {
        for (RuntimeObject* o : {&g_threading, &g_error, &g_networking})
            if (*class_id == *o->class_id) return object_query_interface(o, interface_id, out);
    }
    char cls[40], iid[40];
    print_guid(cls, class_id);
    print_guid(iid, &interface_id);
    static std::atomic<int> s_logged{0};
    if (s_logged.fetch_add(1) < 256)   // one line per request, bounded
        std::fprintf(stderr, "[rt] xgameruntime: QueryApiImpl(class %s, interface %s): not provided yet\n", cls, iid);
    return kClassNotProvided;
}

// HRESULT XGameRuntimeInitialize(). Reference counted, as in the GDK: each call needs an XGameRuntimeUninitialize.
HRESULT __stdcall XGameRuntimeInitialize() {
    if (g_runtime_refs.fetch_add(1) == 0) std::fprintf(stderr, "[rt] xgameruntime: initialized (offline)\n");
    return S_OK;
}

// void XGameRuntimeUninitialize()
void __stdcall XGameRuntimeUninitialize() {
    int prev = g_runtime_refs.load();
    while (prev > 0 && !g_runtime_refs.compare_exchange_weak(prev, prev - 1)) {}
}

// ---- XSystem ----

// XSystemDeviceType, in the order of the public documentation: Unknown, Pc, XboxOne, XboxOneS, XboxOneX,
// XboxOneXDevkit, XboxScarlettLockhart, XboxScarlettAnaconda, XboxScarlettDevkit.
enum class XSystemDeviceType : uint32_t {
    Unknown = 0, Pc = 1, XboxOne = 2, XboxOneS = 3, XboxOneX = 4, XboxOneXDevkit = 5,
    XboxScarlettLockhart = 6, XboxScarlettAnaconda = 7, XboxScarlettDevkit = 8,
};

// XSystemDeviceType XSystemGetDeviceType(). The package is the Xbox Series build, so report Series X (Anaconda).
// LEONIDA_DEVICE_TYPE=<0..8> overrides it, for trying other code paths on day one.
uint32_t __stdcall XSystemGetDeviceType() {
    char s[16];
    const DWORD n = GetEnvironmentVariableA("LEONIDA_DEVICE_TYPE", s, sizeof(s));
    if (n > 0 && n < sizeof(s)) {
        int v = std::atoi(s);
        if (v >= 0 && v <= 8) return static_cast<uint32_t>(v);
    }
    return static_cast<uint32_t>(XSystemDeviceType::XboxScarlettAnaconda);
}

// HRESULT XSystemGetConsoleId(size_t consoleIdSize, char* consoleId, size_t* consoleIdUsed).
// XSystemConsoleIdBytes is 39 (including the terminator). An offline PC has no console id, so this returns a
// fixed placeholder of that length.
constexpr size_t kConsoleIdBytes = 39;
constexpr char kConsoleId[kConsoleIdBytes] = "00000000.00000000.00000000.00000000.00";
static_assert(sizeof(kConsoleId) == kConsoleIdBytes, "placeholder console id must fill XSystemConsoleIdBytes");

HRESULT __stdcall XSystemGetConsoleId(size_t size, char* out, size_t* used) {
    if (!out) return E_INVALIDARG;
    if (size < kConsoleIdBytes) return HRESULT_FROM_WIN32(ERROR_INSUFFICIENT_BUFFER);
    std::memcpy(out, kConsoleId, kConsoleIdBytes);
    if (used) *used = kConsoleIdBytes;
    return S_OK;
}

// uint64_t XSystemGetAppMemoryUsageLimit(). Reports total physical RAM, or 12 GB fallback for Series X.
uint64_t __stdcall XSystemGetAppMemoryUsageLimit() {
    MEMORYSTATUSEX ms{sizeof(ms)};
    if (GlobalMemoryStatusEx(&ms)) return ms.ullTotalPhys;
    return 12ULL * 1024 * 1024 * 1024;
}

// uint64_t XSystemGetAppMemoryUsage()
uint64_t __stdcall XSystemGetAppMemoryUsage() {
    return 0;
}

// XVersion: 4-part version structure from GDK XGameRuntimeTypes.h (8 bytes)
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
static_assert(sizeof(XVersion) == 8, "XVersion must be 8 bytes");

// XSystemAnalyticsInfo: hardware and OS version telemetry (144 bytes)
struct XSystemAnalyticsInfo {
    XVersion osVersion;
    XVersion hostingOsVersion;
    char family[64];
    char form[64];
};
static_assert(sizeof(XSystemAnalyticsInfo) == 144, "XSystemAnalyticsInfo must be 144 bytes");

// XSystemAnalyticsInfo XSystemGetAnalyticsInfo()
// Returned by value per Microsoft GDK specification.
XSystemAnalyticsInfo __stdcall XSystemGetAnalyticsInfo() {
    XSystemAnalyticsInfo info{};
    info.osVersion = {10, 0, 22000, 0};
    info.hostingOsVersion = {10, 0, 22000, 0};
    std::strncpy(info.family, "Xbox", sizeof(info.family) - 1);

    const uint32_t dt = XSystemGetDeviceType();
    if (dt == static_cast<uint32_t>(XSystemDeviceType::XboxScarlettLockhart)) {
        std::strncpy(info.form, "Xbox Series S", sizeof(info.form) - 1);
    } else {
        std::strncpy(info.form, "Xbox Series X", sizeof(info.form) - 1);
    }

    char form_override[64];
    const DWORD n = GetEnvironmentVariableA("LEONIDA_ANALYTICS_FORM", form_override, sizeof(form_override));
    if (n > 0 && n < sizeof(form_override)) {
        std::strncpy(info.form, form_override, sizeof(info.form) - 1);
    }
    return info;
}

// ---- XPackage ----

constexpr size_t XPACKAGE_IDENTIFIER_MAX_LENGTH = 33;
constexpr char kDefaultPackageIdentifier[XPACKAGE_IDENTIFIER_MAX_LENGTH] = "LeonidaPackageIdentifier00000000";
static_assert(sizeof(kDefaultPackageIdentifier) == XPACKAGE_IDENTIFIER_MAX_LENGTH,
              "default package identifier must match XPACKAGE_IDENTIFIER_MAX_LENGTH");

// HRESULT XPackageGetCurrentProcessPackageIdentifier(size_t bufferSize, char* buffer).
// Retrieves the package identifier string for the current process.
HRESULT __stdcall XPackageGetCurrentProcessPackageIdentifier(size_t bufferSize, char* buffer) {
    if (!buffer || bufferSize == 0) return E_INVALIDARG;

    char id[XPACKAGE_IDENTIFIER_MAX_LENGTH] = {};
    const DWORD n = GetEnvironmentVariableA("LEONIDA_PACKAGE_ID", id, sizeof(id));
    const char* selected = (n > 0 && n < sizeof(id)) ? id : kDefaultPackageIdentifier;
    const size_t len = std::strlen(selected);

    if (bufferSize < len + 1) return HRESULT_FROM_WIN32(ERROR_INSUFFICIENT_BUFFER);
    std::memcpy(buffer, selected, len + 1);
    return S_OK;
}

// HRESULT XPackageGetMountPath(XPackageMountHandle mount, size_t pathSize, char* path).
// Retrieves the installed filesystem path for the mounted package.
// Reads LEONIDA_GAME_DIR when set, otherwise falls back to the current module's directory.
HRESULT __stdcall XPackageGetMountPath(void* /*mount*/, size_t pathSize, char* path) {
    if (!path || pathSize == 0) return E_INVALIDARG;

    char buf[MAX_PATH] = {};
    const DWORD n = GetEnvironmentVariableA("LEONIDA_GAME_DIR", buf, sizeof(buf));
    if (n == 0 || n >= sizeof(buf)) {
        const DWORD mod_len = GetModuleFileNameA(nullptr, buf, sizeof(buf));
        if (mod_len > 0 && mod_len < sizeof(buf)) {
            char* slash = std::strrchr(buf, '\\');
            char* fslash = std::strrchr(buf, '/');
            char* last = slash > fslash ? slash : fslash;
            if (last) {
                *last = '\0';
            } else {
                return HRESULT_FROM_WIN32(ERROR_FILE_NOT_FOUND);
            }
        } else {
            return HRESULT_FROM_WIN32(ERROR_FILE_NOT_FOUND);
        }
    }

    const size_t len = std::strlen(buf);
    if (pathSize < len + 1) return HRESULT_FROM_WIN32(ERROR_INSUFFICIENT_BUFFER);
    std::memcpy(path, buf, len + 1);
    return S_OK;
}

// bool XPackageIsPackagedProcess().
// Returns true on Xbox Series console. LEONIDA_PACKAGED=0 overrides it.
bool __stdcall XPackageIsPackagedProcess() {
    char s[16];
    const DWORD n = GetEnvironmentVariableA("LEONIDA_PACKAGED", s, sizeof(s));
    if (n > 0 && n < sizeof(s)) {
        if (std::atoi(s) == 0) return false;
    }
    return true;
}

#define XGR(name) {"xgameruntime.dll", #name, reinterpret_cast<void*>(&name)}

}  // namespace

SHIM_TABLE(xgameruntime,
    // what xgameruntime.lib's thunks GetProcAddress (library_loader.cpp serves the LoadLibrary before it)
    XGR(InitializeApiImpl),
    XGR(InitializeApiImplEx),
    XGR(InitializeApiImplEx2),
    XGR(UninitializeApiImpl),
    XGR(QueryApiImpl),
    XGR(XGameRuntimeInitialize),
    XGR(XGameRuntimeUninitialize),
    XGR(XSystemGetDeviceType),
    XGR(XSystemGetConsoleId),
    XGR(XSystemGetAppMemoryUsageLimit),
    XGR(XSystemGetAppMemoryUsage),
    XGR(XSystemGetAnalyticsInfo),
    XGR(XPackageGetCurrentProcessPackageIdentifier),
    XGR(XPackageGetMountPath),
    XGR(XPackageIsPackagedProcess),
    // XTaskQueue (third_party/xtaskqueue)
    XGR(XTaskQueueCreate),
    XGR(XTaskQueueCreateComposite),
    XGR(XTaskQueueGetPort),
    XGR(XTaskQueueDuplicateHandle),
    XGR(XTaskQueueCloseHandle),
    XGR(XTaskQueueTerminate),
    XGR(XTaskQueueDispatch),
    XGR(XTaskQueueSubmitCallback),
    XGR(XTaskQueueSubmitDelayedCallback),
    XGR(XTaskQueueRegisterWaiter),
    XGR(XTaskQueueUnregisterWaiter),
    XGR(XTaskQueueRegisterMonitor),
    XGR(XTaskQueueUnregisterMonitor),
    XGR(XTaskQueueGetCurrentProcessTaskQueue),
    XGR(XTaskQueueSetCurrentProcessTaskQueue),
    // XAsync (third_party/xtaskqueue)
    XGR(XAsyncGetStatus),
    XGR(XAsyncGetResultSize),
    XGR(XAsyncCancel),
    XGR(XAsyncRun),
    XGR(XAsyncBegin),
    XGR(XAsyncSchedule),
    XGR(XAsyncComplete),
    XGR(XAsyncGetResult),
);
