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
      pacer_(clock ? *clock : *own_clock_) {}

XboxPresentation::~XboxPresentation() = default;

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
    if (params && params->ImmediateThresholdPercent != 0.0f)
        log_once(logged_threshold_, "ImmediateThresholdPercent is not implemented: a late frame waits for the next vblank");

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
    if (const std::int32_t checked = pacer_.check_present_token(token); checked != hr::ok) return from_pacer(checked);

    const XboxPresentPlaneParameters& plane = planes[0];
    ID3D12Resource* resource = plane.ppResources[0];
    const D3D12_RESOURCE_DESC desc = resource->GetDesc();
    if (desc.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D || desc.SampleDesc.Count != 1) return E_INVALIDARG;
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

    const HRESULT result = presenter_->present(resource, sync_interval, rect);
    if (FAILED(result)) return result;
    pacer_.commit_present_token(token, plane_count);
    ++presents_;
    return S_OK;
}

HRESULT XboxPresentation::SuspendX(UINT) {
    std::lock_guard serialise(present_mutex_);
    pacer_.suspend();
    return S_OK;
}

HRESULT XboxPresentation::ResumeX() {
    std::lock_guard serialise(present_mutex_);
    pacer_.resume();
    return S_OK;
}

}  // namespace rt::d3d12
