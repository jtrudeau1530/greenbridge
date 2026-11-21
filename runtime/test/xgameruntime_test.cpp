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

const char* const kNames[] = {
    "XGameRuntimeInitialize", "XGameRuntimeUninitialize", "XSystemGetDeviceType", "XSystemGetConsoleId",
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

    if (g_failures) {
        std::printf("xgameruntime_test: %d of %d checks failed\n", g_failures, g_checks);
        return 1;
    }
    std::printf("xgameruntime_test: OK (%d checks)\n", g_checks);
    return 0;
}
