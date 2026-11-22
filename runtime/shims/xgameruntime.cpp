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
// XGameRuntimeInitialize[WithOptions] calls, and QueryApiImpl hands out the runtime's COM classes. The classes'
// method layouts are not known yet, so QueryApiImpl logs each class and interface id it is asked for and refuses
// them; the first boot with the executable then lists the interfaces to implement.

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

HRESULT WINAPI QueryApiImpl(const GUID* class_id, REFIID interface_id, void** out) {
    if (out) *out = nullptr;
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
