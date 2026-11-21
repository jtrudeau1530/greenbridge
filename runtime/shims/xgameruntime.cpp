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
    if (const char* s = std::getenv("LEONIDA_DEVICE_TYPE")) {
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

#define XGR(name) {"xgameruntime.dll", #name, reinterpret_cast<void*>(&name)}

}  // namespace

SHIM_TABLE(xgameruntime,
    XGR(XGameRuntimeInitialize),
    XGR(XGameRuntimeUninitialize),
    XGR(XSystemGetDeviceType),
    XGR(XSystemGetConsoleId),
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
