#include "xbox_present.h"

#include <cstdio>

namespace rt::d3d12 {

namespace {

using leonida::FramePacer;
namespace hr = leonida::pacer_hr;

HRESULT from_pacer(std::int32_t value) { return HRESULT(value); }

// The fixed text of a log line is printed once per object.
void log_once(bool& done, const char* text) {
    if (done) return;
    done = true;
    std::fprintf(stderr, "[rt] present: %s\n", text);
}

}  // namespace

XboxPresentation::XboxPresentation(std::unique_ptr<WindowPresenter> presenter, leonida::FrameClock* clock)
    : presenter_(std::move(presenter)),
      own_clock_(clock ? nullptr : std::make_unique<leonida::SteadyFrameClock>()),
      pacer_(clock ? *clock : *own_clock_) {
    // A present waiting for a swapchain image gives up once SuspendX has been called.
    presenter_->set_abort([this] { return pacer_.suspended(); });
}

XboxPresentation::~XboxPresentation() { presenter_->set_abort(nullptr); }

HRESULT XboxPresentation::SetFrameIntervalX(IUnknown*, UINT interval_period_us, UINT latency, UINT flags) {
    return from_pacer(pacer_.set_frame_interval(interval_period_us, latency, flags));
}

HRESULT XboxPresentation::ScheduleFrameEventX(UINT event, UINT offset_us, UINT flags) {
    return from_pacer(pacer_.schedule_frame_event(event, offset_us, flags));
}

HRESULT XboxPresentation::WaitFrameEventX(UINT event, DWORD timeout_ms, UINT flags, std::uint64_t* token) {
    return from_pacer(pacer_.wait_frame_event(event, timeout_ms, flags, token));
}

HRESULT XboxPresentation::GetFrameStatisticsX(std::uint64_t token, UINT type_mask, UINT* count,
                                              leonida::FrameStatistic* out) {
    return from_pacer(pacer_.get_frame_statistics(token, type_mask, count, out));
}

HRESULT XboxPresentation::PresentX(UINT plane_count, const XboxPresentPlaneParameters* planes,
                                   const XboxPresentParameters* params) {
    std::lock_guard serialise(present_mutex_);
    if (plane_count > 2 || (plane_count && !planes)) return E_INVALIDARG;
    if (pacer_.suspended()) return from_pacer(hr::invalid_state);
    // A zero, negative or NaN threshold never tears.
    const float threshold = params ? params->ImmediateThresholdPercent : 0.0f;
    const bool wants_tear = threshold > 0.0f;
    if (wants_tear && !presenter_->tear_supported())
        log_once(logged_threshold_,
                 "ImmediateThresholdPercent is not implemented here (the swapchain cannot switch to IMMEDIATE or "
                 "FIFO_RELAXED: no VK_KHR/EXT_swapchain_maintenance1 or no such mode): a late frame waits for the next vblank");

    // Unpaced (no interval registered): do not hold the queue for vblanks.
    const UINT sync_interval = pacer_.registered() ? 1 : 0;

    if (plane_count == 0) {
        const HRESULT result = presenter_->present_blank(sync_interval);
        if (SUCCEEDED(result)) {
            pacer_.note_untokened_present();
            ++presents_;
        }
        return result;
    }

    // All planes carry the token of one frame.
    const std::uint64_t token = planes[0].Token;
    for (UINT i = 0; i < plane_count; ++i) {
        const XboxPresentPlaneParameters& plane = planes[i];
        if (plane.Token != token || plane.ResourceCount == 0 || !plane.ppResources || !plane.ppResources[0])
            return E_INVALIDARG;
    }
    // Only a token that was never handed out is refused. Null, stale and repeated tokens are presented anyway (what the
    // console does with them is unmeasured, and a title that does it must not freeze); each kind is logged once.
    FramePacer::TokenIssue issue{};
    if (const std::int32_t checked = pacer_.check_present_token(token, &issue); checked != hr::ok)
        return from_pacer(checked);
    switch (issue) {
    case FramePacer::TokenIssue::null_token:
        log_once(logged_null_token_, "PresentX with the null token: presented anyway (real behaviour unmeasured)");
        break;
    case FramePacer::TokenIssue::stale:
        log_once(logged_stale_token_, "PresentX with a token older than the last presented: presented anyway");
        break;
    case FramePacer::TokenIssue::repeated:
        log_once(logged_repeated_token_, "PresentX with the token of the last present again: presented anyway");
        break;
    case FramePacer::TokenIssue::none:
        break;
    }

    const XboxPresentPlaneParameters& plane = planes[0];
    ID3D12Resource* resource = plane.ppResources[0];
    const D3D12_RESOURCE_DESC desc = resource->GetDesc();
    if (desc.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D) return E_INVALIDARG;
    D3D12_HEAP_PROPERTIES heap{};
    D3D12_HEAP_FLAGS heap_flags = D3D12_HEAP_FLAG_NONE;
    if (SUCCEEDED(resource->GetHeapProperties(&heap, &heap_flags)) && !(heap_flags & D3D12_HEAP_FLAG_ALLOW_DISPLAY))
        log_once(logged_display_, "a present plane's resource was created without D3D12_HEAP_FLAG_ALLOW_DISPLAY; presenting it anyway");

    const D3D12_RECT* rect = plane.pSrcViewRects;
    if (rect && (rect->left < 0 || rect->top < 0 || rect->left >= rect->right || rect->top >= rect->bottom ||
                 UINT64(rect->right) > desc.Width || UINT(rect->bottom) > desc.Height))
        return E_INVALIDARG;

    if (plane_count == 2)
        log_once(logged_planes_, "PresentX with 2 planes: only plane 0 is shown; HDR/GameDVR plane composition is not implemented");

    // How late the frame is, now that everything that can refuse it has: past the vblank it was due at, by what part of
    // the interval the scan has moved on. The presenter says whether it really queued it as a tear.
    FramePacer::PresentTiming timing;
    bool tear = false;
    if (wants_tear && sync_interval > 0 && presenter_->tear_supported()) {
        timing = pacer_.present_timing(token);
        tear = leonida::decide_present_sync(timing.late_us, timing.period_us, threshold) == leonida::PresentSync::immediate;
    }
    const HRESULT result = presenter_->present(resource, sync_interval, rect, tear);
    if (FAILED(result)) return pacer_.suspended() ? from_pacer(hr::invalid_state) : result;
    pacer_.commit_present_token(token, plane_count,
                                tear && presenter_->last_present_tore() ? leonida::PresentSync::immediate
                                                                        : leonida::PresentSync::vblank,
                                timing.late_us);
    ++presents_;
    return S_OK;
}

// SuspendX and ResumeX do not take present_mutex_: a PresentX stuck waiting for a swapchain image must not hold them
// (it polls the suspended state and gives up). A present that was already past that point finishes normally.
HRESULT XboxPresentation::SuspendX(UINT) {
    pacer_.suspend();
    return S_OK;
}

HRESULT XboxPresentation::ResumeX() {
    pacer_.resume();
    return S_OK;
}

}  // namespace rt::d3d12
