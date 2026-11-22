#pragma once

// The Xbox presentation core (docs/d3d12_vulkan.md section 4.5): the frame events and PresentX of a GDKX title, which
// presents without a DXGI swapchain. FramePacer (src/frame_pacer.h) does the timing and the token bookkeeping;
// WindowPresenter (d3d12_com.h, over leonida::VulkanPresenter) puts the pixels in a window. This class is the
// glue, and the surface the Xbox device and queue proxy will forward to once the vtable slots are known.
//
// WHAT IS KNOWN: the public GDKX samples give the call order, the argument counts and the field names (section 4.5).
// WHAT IS OURS: every struct layout, constant value and HRESULT below is a provisional definition of this layer; the real
// slots, layouts, enum values and the token width are unmeasured. They are named like the console's so the proxy
// can swap them, but nothing here may be read as the console's ABI.
//
// TODO, the slots to measure before the proxy can forward (index in the vtable of the Xbox device / queue):
//   ID3D12Device (Xbox):        SetFrameIntervalX   (output, interval, latency, flags)           slot ?
//                               ScheduleFrameEventX (event, offset, reserved pointer, flags)     slot ?
//                               WaitFrameEventX     (event, timeout ms, reserved pointer, flags, &token)  slot ?
//                               GetFrameStatisticsX (token, type mask, &count, array)            slot ?
//   ID3D12CommandQueue (Xbox):  PresentX (plane count, planes, params)                           slot ?
//                               SuspendX (flags)                                                 slot ?
//                               ResumeX ()                                                       slot ?
// The queue-side calls do not look at the queue: the Vulkan device has one VkQueue behind every D3D12 queue, and
// the present is submitted after the work already executed on it.

#include "d3d12_com.h"
#include "../src/frame_pacer.h"

#include <atomic>
#include <memory>
#include <mutex>

namespace rt::d3d12 {

// Provisional (ours): D3D12XBOX_FRAME_INTERVAL_* as periods in microseconds.
inline constexpr std::uint32_t xbox_frame_interval_30hz_us = 33333;
inline constexpr std::uint32_t xbox_frame_interval_60hz_us = 16667;
inline constexpr std::uint32_t xbox_frame_interval_120hz_us = 8333;

// Provisional (ours): D3D12XBOX_PRESENT_PLANE_PARAMETERS. The names are those the samples use; the layout, the
// token width (64 bits here) and the fields the console has beyond these are unmeasured.
struct XboxPresentPlaneParameters {
    std::uint64_t Token;                  // from WaitFrameEventX
    UINT ResourceCount;                   // 1 in the samples
    ID3D12Resource* const* ppResources;   // ResourceCount committed resources with ALLOW_DISPLAY, in PRESENT
    const D3D12_RECT* pSrcViewRects;      // null, or one rectangle per resource (the HDR sample sets it)
    int ColorSpace;                       // a DXGI_COLOR_SPACE_TYPE; not used yet
};

// Provisional (ours): D3D12XBOX_PRESENT_PARAMETERS. The samples show at least this field.
struct XboxPresentParameters {
    // 0 (the default): a frame that missed its vblank waits for the next one. Above 0: a frame that missed it is shown
    // at once, mid-scan, when the scan is not further than this percentage down the screen (100: wherever it is).
    // Needs VK_KHR/EXT_swapchain_maintenance1 and an IMMEDIATE or FIFO_RELAXED mode the swapchain can switch to;
    // without them it is ignored (logged once). What late means: see FramePacer::present_timing.
    float ImmediateThresholdPercent;
};

class XboxPresentation {
public:
    // `clock` null: real time (a steady high-resolution timer). Tests pass a leonida::ManualFrameClock, which must
    // outlive this object. `presenter` is what create_window_presenter returned.
    explicit XboxPresentation(std::unique_ptr<WindowPresenter> presenter, leonida::FrameClock* clock = nullptr);
    ~XboxPresentation();
    XboxPresentation(const XboxPresentation&) = delete;
    XboxPresentation& operator=(const XboxPresentation&) = delete;

    // Device side. `output` (an IDXGIOutput*, which the samples fetch from the device) is not used: there is one
    // window. `interval_period_us` is the period of a frame interval, e.g. xbox_frame_interval_60hz_us; `latency`
    // is the frames allowed in flight (the samples pass the buffer count minus 1, or 1).
    HRESULT SetFrameIntervalX(IUnknown* output, UINT interval_period_us, UINT latency, UINT flags);
    // Only the origin event (xbox_frame_event_origin) exists. `offset_us` delays it inside the frame. The pointer
    // argument the samples pass as null is not modelled.
    HRESULT ScheduleFrameEventX(UINT event, UINT offset_us, UINT flags);
    // Blocks for the next origin and while the latency is used up. HRESULT_FROM_WIN32(WAIT_TIMEOUT) after `timeout_ms`
    // (INFINITE waits for ever); `*token` is untouched then. E_INVALIDARG for a null token pointer or another event;
    // HRESULT_FROM_WIN32(ERROR_INVALID_STATE) when nothing is registered (not yet, or since SuspendX).
    HRESULT WaitFrameEventX(UINT event, DWORD timeout_ms, UINT flags, std::uint64_t* token);
    // See FramePacer::get_frame_statistics: S_FALSE and count 0 when nothing is available.
    HRESULT GetFrameStatisticsX(std::uint64_t token, UINT type_mask, UINT* count, leonida::FrameStatistic* out);

    // Queue side.
    //   0 planes: a black frame (what a title does before it releases its resources); no token is looked at.
    //   1 plane:  plane 0's first resource, which must be a single-sample 2D colour texture in the PRESENT state,
    //             is blitted to the window (format conversion by the blit; an sRGB format gets an sRGB swapchain),
    //             limited to pSrcViewRects[0] when that is set.
    //   2 planes: the two share one token; plane 0 is presented and plane 1 (the GameDVR/SDR copy) is dropped; the
    //             missing composition (HDR, GameDVR) is logged once.
    //   More than 2: E_INVALIDARG (the UMD limit, section 4.5).
    // Token: the one from WaitFrameEventX. Real behaviour for a null, repeated or out-of-order token is unmeasured and a
    // title must not freeze on it, so those are presented anyway and logged once per kind; only a token that was never
    // handed out gives E_INVALIDARG (nothing presented). Skipping a token is fine. Sources the GPU cannot blit
    // (multisampled, integer, block-compressed) give DXGI_ERROR_UNSUPPORTED and present nothing; resolving a
    // multisampled plane into a temporary is not done. While suspended: HRESULT_FROM_WIN32(ERROR_INVALID_STATE), also
    // for a present that was waiting for a swapchain image when SuspendX came.
    // `params` may be null (threshold 0). ImmediateThresholdPercent: see XboxPresentParameters; a frame that was shown
    // at once has its display statistic's percent_scanned set and is counted in immediate_presents().
    HRESULT PresentX(UINT plane_count, const XboxPresentPlaneParameters* planes, const XboxPresentParameters* params);
    // SuspendX drops the interval, the origin event and the frames in flight, and refuses presents. ResumeX allows
    // presents again; SetFrameIntervalX and ScheduleFrameEventX must be called again before the next wait.
    HRESULT SuspendX(UINT flags);
    HRESULT ResumeX();

    leonida::FramePacer& pacer() { return pacer_; }
    WindowPresenter& presenter() { return *presenter_; }
    std::uint64_t presents() const { return presents_; }
    // Frames shown mid-scan because of ImmediateThresholdPercent.
    std::uint64_t immediate_presents() const { return pacer_.immediate_presents(); }

private:
    std::unique_ptr<WindowPresenter> presenter_;
    std::unique_ptr<leonida::FrameClock> own_clock_;
    leonida::FramePacer pacer_;
    std::mutex present_mutex_;  // PresentX, SuspendX and ResumeX one at a time
    std::atomic<std::uint64_t> presents_{0};
    bool logged_planes_ = false, logged_threshold_ = false, logged_display_ = false;
    bool logged_null_token_ = false, logged_stale_token_ = false, logged_repeated_token_ = false;
};

inline constexpr UINT xbox_frame_event_origin = leonida::frame_event_origin;

}  // namespace rt::d3d12
