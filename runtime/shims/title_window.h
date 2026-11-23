#pragma once

// The window a GDKX title shows its frames in, for PresentX.
//
// On the console PresentX puts a frame on the display: it names no window. A GDKX title still makes its own top-level
// Win32 window (CreateWindowExW with its own class, a message loop on its own thread) before it presents, so on the PC
// that window is where the frames belong. Measured on a retail executable: by its first PresentX the title has one
// visible, unowned top-level window on a thread of its own; the other windows of the process at that point are hidden
// (the IME windows Windows adds, a driver's dummy window).
//
// The lookup sends no message to any window. The present thread is not the title's window thread, and that thread can
// be busy (or waiting on the present thread) when the first PresentX comes: GetWindowTextW, or any call that sends a
// message across threads, would block the present thread there. Measured: a lookup that read the window titles with
// GetWindowTextW stopped the retail title at its first PresentX. EnumWindows, GetWindowThreadProcessId, IsWindowVisible,
// GetWindow(GW_OWNER), GetWindowLongPtrW and GetClientRect read the window manager's own state and send nothing.

#include <windows.h>

namespace rt::d3d12 {

// The visible, unowned, non-tool top-level window of `process_id` with the largest client area, `exclude` aside (the
// runtime's own fallback window); null when the process has none. Ties keep the first one in Z order (the topmost).
inline HWND find_title_window(DWORD process_id, HWND exclude = nullptr) {
    struct Search {
        DWORD process_id;
        HWND exclude;
        HWND best;
        LONGLONG best_area;
    } search{process_id, exclude, nullptr, -1};
    EnumWindows(
        [](HWND window, LPARAM context) -> BOOL {
            auto& s = *reinterpret_cast<Search*>(context);
            DWORD owner_process = 0;
            if (window == s.exclude || !GetWindowThreadProcessId(window, &owner_process) ||
                owner_process != s.process_id)
                return TRUE;
            if (!IsWindowVisible(window) || GetWindow(window, GW_OWNER)) return TRUE;
            if (GetWindowLongPtrW(window, GWL_EXSTYLE) & WS_EX_TOOLWINDOW) return TRUE;
            RECT client{};
            const LONGLONG area =
                GetClientRect(window, &client) ? LONGLONG(client.right - client.left) * (client.bottom - client.top) : 0;
            if (area > s.best_area) {
                s.best = window;
                s.best_area = area;
            }
            return TRUE;
        },
        reinterpret_cast<LPARAM>(&search));
    return search.best;
}


// The title sizes its window for the console's output: 3840 x 2160 of client area plus the frame. On a monitor whose
// work area is smaller (a 3440 x 1440 ultrawide, any 2560 x 1440 or 1920 x 1080 one) the caption and its buttons end
// up off screen, and the window cannot be moved back by hand. A frame that does not fit `work` is scaled down by one
// factor, keeping its aspect ratio, and centred in `work`; true with the new frame in `out`. A frame that already lies
// inside `work`, or an empty rectangle, gives false: nothing to do.
inline bool fitted_window_rect(const RECT& frame, const RECT& work, RECT& out) {
    const LONGLONG width = LONGLONG(frame.right) - frame.left, height = LONGLONG(frame.bottom) - frame.top;
    const LONGLONG work_width = LONGLONG(work.right) - work.left, work_height = LONGLONG(work.bottom) - work.top;
    if (width <= 0 || height <= 0 || work_width <= 0 || work_height <= 0) return false;
    if (frame.left >= work.left && frame.top >= work.top && frame.right <= work.right && frame.bottom <= work.bottom)
        return false;
    LONGLONG fitted_width = width, fitted_height = height;
    if (width > work_width || height > work_height) {
        // the smaller of the two ratios, in integers: width * work_height against height * work_width
        if (width * work_height >= height * work_width) {
            fitted_width = work_width;
            fitted_height = height * work_width / width;
        } else {
            fitted_height = work_height;
            fitted_width = width * work_height / height;
        }
    }
    out.left = static_cast<LONG>(work.left + (work_width - fitted_width) / 2);
    out.top = static_cast<LONG>(work.top + (work_height - fitted_height) / 2);
    out.right = static_cast<LONG>(out.left + fitted_width);
    out.bottom = static_cast<LONG>(out.top + fitted_height);
    return true;
}

// Moves `window` to fitted_window_rect of its monitor's work area, when it does not fit. Like find_title_window it
// sends no message to the window: GetWindowRect, MonitorFromWindow and GetMonitorInfoW read the window manager's state,
// and SWP_ASYNCWINDOWPOS posts the move to the window's own thread instead of waiting for it. True when a move was
// requested.
inline bool fit_title_window(HWND window) {
    RECT frame{}, fitted{};
    if (!window || !GetWindowRect(window, &frame)) return false;
    MONITORINFO monitor{};
    monitor.cbSize = sizeof(monitor);
    if (!GetMonitorInfoW(MonitorFromWindow(window, MONITOR_DEFAULTTONEAREST), &monitor)) return false;
    if (!fitted_window_rect(frame, monitor.rcWork, fitted)) return false;
    return SetWindowPos(window, nullptr, fitted.left, fitted.top, fitted.right - fitted.left, fitted.bottom - fitted.top,
                        SWP_NOZORDER | SWP_NOACTIVATE | SWP_ASYNCWINDOWPOS) != FALSE;
}

}  // namespace rt::d3d12
