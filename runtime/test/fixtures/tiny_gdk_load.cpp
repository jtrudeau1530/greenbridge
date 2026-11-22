// A mapped program that reaches the GDK runtime the way xgameruntime.lib's thunks do: it has no import of
// xgameruntime.dll, loads it with LoadLibraryW and looks the five run-time entry points up with GetProcAddress
// (measured in the package's libHttpClient.GDK.dll and XCurl.dll, docs/runtime_imports.md "Run-time loaded
// modules"). Run under `gta6rt --run`, its own kernel32 imports are bound by the loader, so LoadLibraryW and
// GetProcAddress must reach runtime/shims/library_loader.cpp and the offline xgameruntime shim. Run on its own,
// LoadLibraryW goes to Windows and either fails or loads the PC's Gaming Services runtime, and this never prints
// PASS. Ordinary test code; no game data.
#include <windows.h>

#include <cstdio>

namespace {

struct Options {        // InitializeApiImplEx2's fourth argument (xodus-gaming/xgameruntime-docs)
    UINT32 unknown;     // documented as always 1
    BOOL is_inline;
    const char* game_config;
};

using InitEx2 = HRESULT(WINAPI*)(ULONG, ULONG, char, const Options*);
using Query = HRESULT(WINAPI*)(const GUID*, REFIID, void**);
using Uninit = HRESULT(WINAPI*)();

int fail(const char* stage, unsigned long value) {
    std::printf("tiny_gdk_load: FAIL %s (0x%08lx)\n", stage, value);
    return 1;
}

}  // namespace

int main() {
    HMODULE runtime = LoadLibraryW(L"xgameruntime.dll");
    if (!runtime) return fail("LoadLibraryW(xgameruntime.dll)", GetLastError());
    // The loader shim hands out a pseudo-handle for a module it owns: not an image the host loaded.
    HMODULE host_view = nullptr;
    if (GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT, L"xgameruntime.dll", &host_view) && host_view == runtime)
        return fail("xgameruntime.dll came from the host loader, not the shims", 0);

    const char* names[] = {"InitializeApiImpl", "InitializeApiImplEx", "InitializeApiImplEx2", "UninitializeApiImpl",
                           "QueryApiImpl"};
    for (const char* name : names)
        if (!GetProcAddress(runtime, name)) return fail(name, GetLastError());
    SetLastError(0);
    if (GetProcAddress(runtime, "NoSuchRuntimeFunction") || GetLastError() != ERROR_PROC_NOT_FOUND)
        return fail("an unserved name is NULL with ERROR_PROC_NOT_FOUND", GetLastError());

    auto init = reinterpret_cast<InitEx2>(GetProcAddress(runtime, "InitializeApiImplEx2"));
    auto query = reinterpret_cast<Query>(GetProcAddress(runtime, "QueryApiImpl"));
    auto uninit = reinterpret_cast<Uninit>(GetProcAddress(runtime, "UninitializeApiImpl"));
    const Options options{1, FALSE, nullptr};
    HRESULT hr = init(0, 0, 0x02, &options);
    if (hr != S_OK) return fail("InitializeApiImplEx2", static_cast<unsigned long>(hr));

    // XThreadingImpl, the first class XCurl.dll references (docs/runtime_imports.md), used as its thunks do: the
    // class id as both ids, a slot with `this` first, then Release (slot 2). Slot 12 is XTaskQueueCreate, 17
    // XTaskQueueCloseHandle.
    const GUID threading = {0x073b7dcb, 0x1fcf, 0x4030, {0x94, 0xbe, 0xe3, 0xc9, 0xeb, 0x62, 0x34, 0x28}};
    void* obj = nullptr;
    hr = query(&threading, threading, &obj);
    if (hr != S_OK || !obj) return fail("QueryApiImpl hands out XThreadingImpl", static_cast<unsigned long>(hr));
    void* const* vtable = *static_cast<void* const* const*>(obj);
    using Create = HRESULT(STDMETHODCALLTYPE*)(void*, UINT32, UINT32, void**);
    using Close = void(STDMETHODCALLTYPE*)(void*, void*);
    using Release = ULONG(STDMETHODCALLTYPE*)(void*);
    void* queue = nullptr;
    hr = reinterpret_cast<Create>(vtable[12])(obj, 0, 0, &queue);  // XTaskQueueDispatchMode::Manual twice
    if (hr != S_OK || !queue) return fail("XThreadingImpl slot 12 (XTaskQueueCreate)", static_cast<unsigned long>(hr));
    reinterpret_cast<Close>(vtable[17])(obj, queue);
    reinterpret_cast<Release>(vtable[2])(obj);

    // XNetworkingImpl is served through QueryApiImpl (slot 15: XNetworkingGetConnectivityHint).
    const GUID networking = {0x37e56907, 0x2f10, 0x41e8, {0xb7, 0x2f, 0x36, 0xed, 0xb1, 0x85, 0x33, 0x1a}};
    void* net_obj = nullptr;
    hr = query(&networking, networking, &net_obj);
    if (hr != S_OK || !net_obj)
        return fail("QueryApiImpl hands out XNetworkingImpl", static_cast<unsigned long>(hr));
    void* const* net_vtable = *static_cast<void* const* const*>(net_obj);
    struct Hint {
        UINT32 level;
        UINT32 cost;
        UINT32 iana;
        bool init;
        bool appr;
        bool over;
        bool roam;
    } hint{};
    using GetHint = HRESULT(STDMETHODCALLTYPE*)(void*, Hint*);
    hr = reinterpret_cast<GetHint>(net_vtable[15])(net_obj, &hint);
    if (hr != S_OK || !hint.init)
        return fail("XNetworkingImpl slot 15 (XNetworkingGetConnectivityHint)", static_cast<unsigned long>(hr));
    reinterpret_cast<Release>(net_vtable[2])(net_obj);

    // A class not served yet (an unprovided class id) is refused, with the out pointer cleared.
    const GUID unprovided = {0x973a344e, 0x24bf, 0x4d0f, {0x84, 0x57, 0x56, 0xc5, 0x34, 0x89, 0x2b, 0x29}};
    void* out = reinterpret_cast<void*>(1);
    hr = query(&unprovided, unprovided, &out);
    if (hr != static_cast<HRESULT>(0x80070032) || out != nullptr)
        return fail("QueryApiImpl refuses an unprovided class and clears its out pointer", static_cast<unsigned long>(hr));
    hr = uninit();
    if (hr != S_OK) return fail("UninitializeApiImpl", static_cast<unsigned long>(hr));

    // Host modules still come from the host.
    HMODULE k32 = LoadLibraryW(L"kernel32.dll");
    if (!k32 || GetProcAddress(k32, "GetTickCount") != GetProcAddress(GetModuleHandleW(L"kernel32.dll"), "GetTickCount"))
        return fail("kernel32.dll from the host", GetLastError());
    if (!FreeLibrary(runtime)) return fail("FreeLibrary(xgameruntime.dll)", GetLastError());

    std::puts("tiny_gdk_load: PASS");
    return 0;
}
