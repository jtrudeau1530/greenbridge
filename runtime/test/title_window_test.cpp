// find_title_window (shims/title_window.h): the window PresentX shows a GDKX title's frames in.
//
// The windows belong to a thread that never pumps its messages, as the title's window thread may be busy at the first
// PresentX: the lookup must still answer at once. The negative control shows that thread really does not answer a
// sent message (WM_GETTEXT, what GetWindowTextW sends across threads, times out), so a lookup that sent one would block.

#include "title_window.h"

#include <chrono>
#include <cstdio>
#include <thread>

#define EXPECT(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); return 1; } \
} while (0)

namespace {

struct Windows {
    HWND title = nullptr;    // visible, unowned, the largest client area: the title's window
    HWND small = nullptr;    // visible, unowned, smaller
    HWND hidden = nullptr;   // never shown (like the IME windows and a driver's dummy window)
    HWND owned = nullptr;    // visible but owned by `title` (a dialog, not a top-level frame)
    HWND tool = nullptr;     // visible tool window
    HWND offscreen = nullptr;  // visible, small, far off any monitor's work area
};

HWND make(const wchar_t* name, DWORD ex_style, DWORD style, int width, int height, HWND owner) {
    return CreateWindowExW(ex_style | WS_EX_NOACTIVATE, L"LeonidaTitleWindowTest", name, style, 0, 0, width, height,
                           owner, nullptr, GetModuleHandleW(nullptr), nullptr);
}

}  // namespace

// fitted_window_rect: the frame a window too large for its monitor's work area is moved to.
int check_fitted_rect() {
    RECT out{};
    // A 3840 x 2160 client area with its frame, on a 3440 x 1440 monitor with a 48-pixel taskbar: limited by the
    // height, scaled by 1392 / 2199, centred horizontally.
    const RECT console_frame{-8, -31, 3848, 2168};
    const RECT ultrawide{0, 0, 3440, 1392};
    EXPECT(rt::d3d12::fitted_window_rect(console_frame, ultrawide, out));
    EXPECT(out.top == 0 && out.bottom == 1392);
    EXPECT(out.right - out.left == 3856LL * 1392 / 2199);
    EXPECT(out.left == (3440 - (out.right - out.left)) / 2);
    // Limited by the width: a wide frame on a 1920 x 1032 work area keeps its aspect ratio.
    EXPECT(rt::d3d12::fitted_window_rect(RECT{0, 0, 3840, 1000}, RECT{-1920, 0, 0, 1032}, out));
    EXPECT(out.left == -1920 && out.right == 0 && out.bottom - out.top == 500);
    EXPECT(out.top == (1032 - 500) / 2);
    // A frame inside the work area is left alone, even when it touches the edges.
    EXPECT(!rt::d3d12::fitted_window_rect(RECT{0, 0, 3440, 1392}, ultrawide, out));
    EXPECT(!rt::d3d12::fitted_window_rect(RECT{100, 100, 900, 700}, ultrawide, out));
    // A frame that fits in size but lies partly off the work area is moved back whole, at its own size.
    EXPECT(rt::d3d12::fitted_window_rect(RECT{3000, 0, 3800, 600}, ultrawide, out));
    EXPECT(out.right - out.left == 800 && out.bottom - out.top == 600);
    EXPECT(out.left >= 0 && out.right <= 3440 && out.top >= 0 && out.bottom <= 1392);
    // Empty rectangles: nothing to do.
    EXPECT(!rt::d3d12::fitted_window_rect(RECT{0, 0, 0, 0}, ultrawide, out));
    EXPECT(!rt::d3d12::fitted_window_rect(console_frame, RECT{0, 0, 0, 0}, out));
    return 0;
}

int main() {
    if (check_fitted_rect()) return 1;

    WNDCLASSW window_class{};
    window_class.lpfnWndProc = DefWindowProcW;
    window_class.hInstance = GetModuleHandleW(nullptr);
    window_class.lpszClassName = L"LeonidaTitleWindowTest";
    EXPECT(RegisterClassW(&window_class));

    // A process with no window of its own has no title window.
    EXPECT(rt::d3d12::find_title_window(GetCurrentProcessId()) == nullptr);

    Windows windows;
    HANDLE made = CreateEventW(nullptr, TRUE, FALSE, nullptr);
    HANDLE release = CreateEventW(nullptr, TRUE, FALSE, nullptr);
    EXPECT(made && release);
    std::thread window_thread([&] {
        windows.hidden = make(L"hidden", 0, WS_OVERLAPPEDWINDOW, 900, 700, nullptr);
        windows.small = make(L"small", 0, WS_OVERLAPPEDWINDOW | WS_VISIBLE, 160, 120, nullptr);
        windows.title = make(L"title", 0, WS_OVERLAPPEDWINDOW | WS_VISIBLE, 320, 240, nullptr);
        windows.owned = make(L"owned", 0, WS_OVERLAPPEDWINDOW | WS_VISIBLE, 800, 600, windows.title);
        windows.tool = make(L"tool", WS_EX_TOOLWINDOW, WS_OVERLAPPEDWINDOW | WS_VISIBLE, 800, 600, nullptr);
        windows.offscreen = CreateWindowExW(WS_EX_NOACTIVATE, L"LeonidaTitleWindowTest", L"offscreen",
                                            WS_OVERLAPPEDWINDOW | WS_VISIBLE, -30000, 0, 100, 80, nullptr, nullptr,
                                            GetModuleHandleW(nullptr), nullptr);
        SetEvent(made);
        WaitForSingleObject(release, INFINITE);  // no message loop: like a title window thread that is busy
        DestroyWindow(windows.offscreen);
        DestroyWindow(windows.tool);
        DestroyWindow(windows.owned);
        DestroyWindow(windows.title);
        DestroyWindow(windows.small);
        DestroyWindow(windows.hidden);
    });
    WaitForSingleObject(made, INFINITE);
    int result = 0;
    [&] {
        auto check = [&](bool ok, int line, const char* what) {
            if (!ok && !result) {
                std::fprintf(stderr, "line %d: %s\n", line, what);
                result = 1;
            }
        };
#define CHECK(condition) check((condition), __LINE__, #condition)
        CHECK(windows.title && windows.small && windows.hidden && windows.owned && windows.tool);

        // Negative control: the window thread does not answer a sent message.
        wchar_t text[16]{};
        DWORD_PTR copied = 0;
        CHECK(!SendMessageTimeoutW(windows.title, WM_GETTEXT, 16, reinterpret_cast<LPARAM>(text), SMTO_BLOCK, 200,
                                   &copied));

        // The lookup answers at once anyway, and picks the visible unowned non-tool window with the largest client
        // area: not the larger hidden, owned or tool windows.
        const auto start = std::chrono::steady_clock::now();
        const HWND found = rt::d3d12::find_title_window(GetCurrentProcessId());
        const auto took = std::chrono::steady_clock::now() - start;
        CHECK(found == windows.title);
        CHECK(took < std::chrono::milliseconds(100));

        // The runtime's own window, excluded, is never the answer.
        CHECK(rt::d3d12::find_title_window(GetCurrentProcessId(), windows.title) == windows.small);
        // Another process's windows are not this title's.
        CHECK(rt::d3d12::find_title_window(GetCurrentProcessId() ^ 0x40000000u) == nullptr);

        // fit_title_window: a window that lies off its monitor's work area gets a move, and the call does not wait for
        // the window's thread, which never pumps here; a window that fits gets none.
        CHECK(windows.offscreen != nullptr);
        const auto fit_start = std::chrono::steady_clock::now();
        CHECK(rt::d3d12::fit_title_window(windows.offscreen));
        CHECK(std::chrono::steady_clock::now() - fit_start < std::chrono::milliseconds(100));
        CHECK(!rt::d3d12::fit_title_window(windows.title));
        CHECK(!rt::d3d12::fit_title_window(nullptr));
#undef CHECK
    }();
    SetEvent(release);
    window_thread.join();
    CloseHandle(made);
    CloseHandle(release);
    if (result) return result;

    // Gone with its windows.
    EXPECT(rt::d3d12::find_title_window(GetCurrentProcessId()) == nullptr);
    std::puts("title_window: ok");
    return 0;
}
