// PresentX and the frame events (shims/xbox_present.h) on the Vulkan-backed device, checked on pixels: every case
// presents to a window and reads the swapchain image back (VulkanPresenter capture). The frame events run on a
// fake clock, so the test is fast and the order of tokens and presents is exact.
//
// Cases: 0 planes (blank), 1 plane (solid colour, two colours side by side, a source rectangle, an RGBA and a BGRA
// source, an sRGB source), 2 planes (plane 0 shown), token errors (null, unknown, stale, repeated), suspend/resume
// with re-registration, statistics, and a real-time loop at 60 Hz. Needs a window and a Vulkan GPU with presentation.
// Set LEONIDA_VK_VALIDATION=1 for VK_LAYER_KHRONOS_validation; any validation error fails the test.
#include "vk_test_device.h"
#include "../shims/d3d12_com.h"
#include "../shims/d3d12_xs_device.h"
#include "../shims/xbox_present.h"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>

#define EXPECT(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); return 1; } \
} while (0)

namespace {

using XboxHeapHandleFn = std::uint64_t(WINAPI*)(void* heap);  // heap slot 9: 8-byte handle in RAX

template<class T>
struct Ref {
    T* p = nullptr;
    ~Ref() { reset(); }
    void reset() {
        if (p) p->Release();
        p = nullptr;
    }
    T* operator->() const { return p; }
};

LRESULT CALLBACK window_proc(HWND window, UINT message, WPARAM w, LPARAM l) {
    return DefWindowProcW(window, message, w, l);
}

void pump() {
    MSG message;
    while (PeekMessageW(&message, nullptr, 0, 0, PM_REMOVE)) {
        TranslateMessage(&message);
        DispatchMessageW(&message);
    }
}

struct Px {
    int r, g, b;
};

constexpr HRESULT invalid_state = HRESULT_FROM_WIN32(ERROR_INVALID_STATE);
constexpr UINT width = 320, height = 240;

// The swapchain frame as RGB, whatever its channel order.
struct Frame {
    std::vector<std::uint8_t> bytes;
    VkExtent2D extent{};
    VkFormat format = VK_FORMAT_UNDEFINED;
    bool bgr() const { return format == VK_FORMAT_B8G8R8A8_UNORM || format == VK_FORMAT_B8G8R8A8_SRGB; }
    bool srgb() const { return format == VK_FORMAT_B8G8R8A8_SRGB || format == VK_FORMAT_R8G8B8A8_SRGB; }
    Px at(UINT x, UINT y) const {
        const std::uint8_t* p = &bytes[(std::size_t(y) * extent.width + x) * 4];
        return bgr() ? Px{p[2], p[1], p[0]} : Px{p[0], p[1], p[2]};
    }
};

bool close_to(Px a, Px b, int tolerance) {
    return std::abs(a.r - b.r) <= tolerance && std::abs(a.g - b.g) <= tolerance && std::abs(a.b - b.b) <= tolerance;
}

// True when every pixel of the rectangle is within the tolerance of `want`.
bool region_is(const Frame& frame, UINT x0, UINT y0, UINT x1, UINT y1, Px want, int tolerance = 1) {
    for (UINT y = y0; y < y1; ++y)
        for (UINT x = x0; x < x1; ++x)
            if (!close_to(frame.at(x, y), want, tolerance)) {
                const Px got = frame.at(x, y);
                std::fprintf(stderr, "pixel %u,%u is %d,%d,%d, expected %d,%d,%d\n", x, y, got.r, got.g, got.b, want.r,
                             want.g, want.b);
                return false;
            }
    return true;
}

}  // namespace

int main() {
    vktest::Device d;
    if (!d.create(true)) {  // no Vulkan loader, device or presentation extensions: nothing to test (ctest: skipped)
        std::printf("no Vulkan device with presentation: skipped\n");
        return 77;
    }

    WNDCLASSW window_class{};
    window_class.lpfnWndProc = window_proc;
    window_class.hInstance = GetModuleHandleW(nullptr);
    window_class.lpszClassName = L"LeonidaXboxPresentTest";
    window_class.hCursor = LoadCursorW(nullptr, reinterpret_cast<LPCWSTR>(IDC_ARROW));
    EXPECT(RegisterClassW(&window_class));
    RECT frame_rect{0, 0, int(width), int(height)};
    AdjustWindowRect(&frame_rect, WS_OVERLAPPEDWINDOW, FALSE);
    HWND window = CreateWindowExW(0, window_class.lpszClassName, L"Leonida: PresentX on Vulkan", WS_OVERLAPPEDWINDOW,
                                  100, 100, frame_rect.right - frame_rect.left, frame_rect.bottom - frame_rect.top,
                                  nullptr, nullptr, window_class.hInstance, nullptr);
    EXPECT(window);
    ShowWindow(window, SW_SHOWNORMAL);
    UpdateWindow(window);
    pump();

    {
        rt::d3d12::VulkanDeviceInfo info{};
        info.get_instance_proc = d.instance_proc;
        info.instance = d.instance;
        info.physical = d.physical;
        info.device = d.device;
        info.queue = d.queue;
        info.queue_family = d.family;
        info.presentation = true;
        info.swapchain_maintenance1 = d.swapchain_maintenance1;
        ID3D12Device* inner = nullptr;
        EXPECT(SUCCEEDED(rt::d3d12::create_vulkan_device(info, IID_PPV_ARGS(&inner))));
        Ref<ID3D12Device> device;
        EXPECT(SUCCEEDED(rt::d3d12_xs::create_device_proxy(inner, __uuidof(ID3D12Device),
                                                           reinterpret_cast<void**>(&device.p))));
        rt::d3d12::set_device_outer(inner, device.p);
        auto presenter = rt::d3d12::create_window_presenter(inner, window);
        EXPECT(presenter);
        EXPECT(presenter->enable_capture());

        leonida::ManualFrameClock clock;
        auto owner = std::make_unique<rt::d3d12::XboxPresentation>(std::move(presenter), &clock);
        rt::d3d12::XboxPresentation& xbox = *owner;

        D3D12_COMMAND_QUEUE_DESC queue_desc{};
        Ref<ID3D12CommandQueue> queue;
        Ref<ID3D12CommandAllocator> allocator;
        Ref<ID3D12GraphicsCommandList> list;
        Ref<ID3D12Fence> fence;
        Ref<ID3D12DescriptorHeap> rtv_heap;
        EXPECT(SUCCEEDED(device->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&queue.p))));
        EXPECT(SUCCEEDED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator.p))));
        EXPECT(SUCCEEDED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.p, nullptr,
                                                   IID_PPV_ARGS(&list.p))));
        EXPECT(SUCCEEDED(list->Close()));
        EXPECT(SUCCEEDED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence.p))));
        constexpr UINT rtv_slots = 8;
        const D3D12_DESCRIPTOR_HEAP_DESC heap_desc{D3D12_DESCRIPTOR_HEAP_TYPE_RTV, rtv_slots,
                                                   D3D12_DESCRIPTOR_HEAP_FLAG_NONE, 0};
        EXPECT(SUCCEEDED(device->CreateDescriptorHeap(&heap_desc, IID_PPV_ARGS(&rtv_heap.p))));
        const SIZE_T rtv_start = SIZE_T(reinterpret_cast<XboxHeapHandleFn>(
            (*reinterpret_cast<void***>(rtv_heap.p))[9])(rtv_heap.p));
        const UINT rtv_step = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);

        // A display-capable back buffer: committed, ALLOW_DISPLAY, in the PRESENT state, with a view in `slot`.
        UINT slot_count = 0;
        struct Source {
            Ref<ID3D12Resource> resource;
            D3D12_CPU_DESCRIPTOR_HANDLE rtv{};
        };
        const auto make_source = [&](DXGI_FORMAT format, Source& out) -> bool {
            const D3D12_HEAP_PROPERTIES default_heap{D3D12_HEAP_TYPE_DEFAULT};
            D3D12_RESOURCE_DESC desc{};
            desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
            desc.Width = width;
            desc.Height = height;
            desc.DepthOrArraySize = 1;
            desc.MipLevels = 1;
            desc.Format = format;
            desc.SampleDesc.Count = 1;
            desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
            if (slot_count >= rtv_slots) return false;
            if (FAILED(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_ALLOW_DISPLAY, &desc,
                                                       D3D12_RESOURCE_STATE_PRESENT, nullptr,
                                                       IID_PPV_ARGS(&out.resource.p))))
                return false;
            out.rtv.ptr = rtv_start + SIZE_T(slot_count++) * rtv_step;
            device->CreateRenderTargetView(out.resource.p, nullptr, out.rtv);
            return true;
        };
        UINT64 signalled = 0;
        // Clears the whole resource to `colour`, or the given rectangles; leaves it in PRESENT.
        const auto fill = [&](Source& s, const FLOAT colour[4], UINT rect_count = 0, const D3D12_RECT* rects = nullptr) -> bool {
            if (FAILED(allocator->Reset()) || FAILED(list->Reset(allocator.p, nullptr))) return false;
            D3D12_RESOURCE_BARRIER barrier{};
            barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
            barrier.Transition.pResource = s.resource.p;
            barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
            barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_PRESENT;
            barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_RENDER_TARGET;
            list->ResourceBarrier(1, &barrier);
            list->ClearRenderTargetView(s.rtv, colour, rect_count, rects);
            barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_RENDER_TARGET;
            barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_PRESENT;
            list->ResourceBarrier(1, &barrier);
            if (FAILED(list->Close())) return false;
            ID3D12CommandList* lists[] = {list.p};
            queue->ExecuteCommandLists(1, lists);
            if (rt::d3d12::last_execute_result(queue.p) != leonida::ReplayResult::success) return false;
            return SUCCEEDED(queue->Signal(fence.p, ++signalled)) &&
                   SUCCEEDED(fence->SetEventOnCompletion(signalled, nullptr));
        };

        // The title's frame loop: wait for the origin, render (the caller did), PresentX with that token.
        const UINT period = rt::d3d12::xbox_frame_interval_60hz_us;
        EXPECT(SUCCEEDED(xbox.SetFrameIntervalX(nullptr, period, 1, 0)));
        EXPECT(SUCCEEDED(xbox.ScheduleFrameEventX(rt::d3d12::xbox_frame_event_origin, 0, 0)));
        const auto next_token = [&](std::uint64_t& token) -> bool {
            clock.advance_us(period);
            return xbox.WaitFrameEventX(rt::d3d12::xbox_frame_event_origin, 0, 0, &token) == S_OK;
        };
        const auto grab = [&](Frame& frame) -> bool {
            return xbox.presenter().read_back(frame.bytes, &frame.extent, &frame.format) &&
                   frame.extent.width == width && frame.extent.height == height;
        };

        const FLOAT blue_ish[4] = {0.2f, 0.6f, 0.9f, 1.0f};  // 51, 153, 230
        const FLOAT red[4] = {1, 0, 0, 1}, green[4] = {0, 1, 0, 1}, blue[4] = {0, 0, 1, 1};
        std::uint64_t token = 0;
        Frame frame;
        const auto plane_of = [&](Source& s, std::uint64_t t, const D3D12_RECT* rect = nullptr) {
            return rt::d3d12::XboxPresentPlaneParameters{t, 1, &s.resource.p, rect, 0};
        };

        // 1 plane, solid colour, RGBA8.
        Source rgba;
        EXPECT(make_source(DXGI_FORMAT_R8G8B8A8_UNORM, rgba));
        EXPECT(fill(rgba, blue_ish));
        EXPECT(next_token(token) && token == 1);
        auto plane = plane_of(rgba, token);
        EXPECT(xbox.PresentX(1, &plane, nullptr) == S_OK);
        EXPECT(grab(frame));
        EXPECT(region_is(frame, 0, 0, width, height, {51, 153, 230}));
        std::printf("1 plane, RGBA8 solid: ok (swapchain format %d)\n", int(frame.format));

        // 0 planes: a black frame, and no token is needed.
        EXPECT(xbox.PresentX(0, nullptr, nullptr) == S_OK);
        EXPECT(grab(frame));
        EXPECT(region_is(frame, 0, 0, width, height, {0, 0, 0}, 0));
        // (The blank frame is a frame in flight like any other: with latency 1 the next wait needs it shown first.)
        EXPECT(xbox.pacer().frames_in_flight() == 2);  // the first frame and the blank one
        clock.advance_us(period);
        std::printf("0 planes (blank): ok\n");

        // Two colours side by side: left half red, right half blue; the swapchain shows them where they were.
        Source split;
        EXPECT(make_source(DXGI_FORMAT_R8G8B8A8_UNORM, split));
        const D3D12_RECT left_half{0, 0, LONG(width / 2), LONG(height)}, right_half{LONG(width / 2), 0, LONG(width), LONG(height)};
        EXPECT(fill(split, red, 1, &left_half));
        EXPECT(fill(split, blue, 1, &right_half));
        EXPECT(next_token(token));
        plane = plane_of(split, token);
        EXPECT(xbox.PresentX(1, &plane, nullptr) == S_OK);
        EXPECT(grab(frame));
        EXPECT(region_is(frame, 0, 0, width / 2 - 2, height, {255, 0, 0}, 0));
        EXPECT(region_is(frame, width / 2 + 2, 0, width, height, {0, 0, 255}, 0));
        std::printf("1 plane, two colours (orientation): ok\n");

        // A source rectangle: only the right half (blue) is shown, stretched over the window.
        EXPECT(next_token(token));
        plane = plane_of(split, token, &right_half);
        EXPECT(xbox.PresentX(1, &plane, nullptr) == S_OK);
        EXPECT(grab(frame));
        // (The blit filters bilinearly, so the outermost texel of the rectangle blends with its neighbour outside it: the
        // border pixels are left out.)
        EXPECT(region_is(frame, 2, 2, width - 2, height - 2, {0, 0, 255}, 0));
        // And a quarter in the middle that straddles the edge: the left part shows red, the right part blue.
        const D3D12_RECT middle{LONG(width / 4), LONG(height / 4), LONG(width * 3 / 4), LONG(height * 3 / 4)};
        EXPECT(next_token(token));
        plane = plane_of(split, token, &middle);
        EXPECT(xbox.PresentX(1, &plane, nullptr) == S_OK);
        EXPECT(grab(frame));
        EXPECT(region_is(frame, 4, 4, width / 2 - 4, height - 4, {255, 0, 0}, 0));
        EXPECT(region_is(frame, width / 2 + 4, 4, width - 4, height - 4, {0, 0, 255}, 0));
        // Rectangles that are empty, inverted or outside the texture are refused and present nothing.
        const std::uint64_t before = xbox.presents();
        const D3D12_RECT bad_rects[] = {{10, 10, 10, 20}, {20, 10, 10, 20}, {0, 0, LONG(width + 1), 10}, {-1, 0, 10, 10}};
        for (const D3D12_RECT& bad : bad_rects) {
            clock.advance_us(4 * period);  // let the queued frames show: with latency 1 the wait holds for them
            EXPECT(next_token(token));
            plane = plane_of(split, token, &bad);
            EXPECT(xbox.PresentX(1, &plane, nullptr) == E_INVALIDARG);
        }
        EXPECT(xbox.presents() == before);
        std::printf("source rectangles (valid and invalid): ok\n");

        // A BGRA source with a colour whose channels differ: red must stay red.
        Source bgra;
        EXPECT(make_source(DXGI_FORMAT_B8G8R8A8_UNORM, bgra));
        const FLOAT orange[4] = {1.0f, 0.5f, 0.0f, 1.0f};
        EXPECT(fill(bgra, orange));
        EXPECT(next_token(token));  // (the token from the refused rect presents is still unused, which is fine)
        plane = plane_of(bgra, token);
        EXPECT(xbox.PresentX(1, &plane, nullptr) == S_OK);
        EXPECT(grab(frame));
        EXPECT(region_is(frame, 0, 0, width, height, {255, 128, 0}, 1));
        std::printf("1 plane, BGRA8 source: ok\n");

        // An sRGB source: the clear value is linear, the stored (and shown) bytes are encoded: 0.5 -> 188. The
        // swapchain follows the source's format, so those bytes go through unchanged.
        Source srgb;
        EXPECT(make_source(DXGI_FORMAT_R8G8B8A8_UNORM_SRGB, srgb));
        const FLOAT grey[4] = {0.5f, 0.5f, 0.5f, 1.0f};
        EXPECT(fill(srgb, grey));
        EXPECT(next_token(token));
        plane = plane_of(srgb, token);
        EXPECT(xbox.PresentX(1, &plane, nullptr) == S_OK);
        EXPECT(grab(frame));
        EXPECT(frame.srgb());
        EXPECT(region_is(frame, 0, 0, width, height, {188, 188, 188}, 1));
        // Back to UNORM: the swapchain goes back to a UNORM format.
        EXPECT(next_token(token));
        plane = plane_of(rgba, token);
        EXPECT(xbox.PresentX(1, &plane, nullptr) == S_OK);
        EXPECT(grab(frame));
        EXPECT(!frame.srgb());
        EXPECT(region_is(frame, 0, 0, width, height, {51, 153, 230}));
        std::printf("1 plane, sRGB source (swapchain follows): ok\n");

        // 2 planes sharing a token: plane 0 is shown.
        Source second;
        EXPECT(make_source(DXGI_FORMAT_R8G8B8A8_UNORM, second));
        EXPECT(fill(second, green));
        EXPECT(next_token(token));
        rt::d3d12::XboxPresentPlaneParameters two[2] = {plane_of(rgba, token), plane_of(second, token)};
        EXPECT(xbox.PresentX(2, two, nullptr) == S_OK);
        EXPECT(grab(frame));
        EXPECT(region_is(frame, 0, 0, width, height, {51, 153, 230}));
        // Planes with different tokens, or more than two planes, are refused.
        EXPECT(next_token(token));
        two[0] = plane_of(rgba, token);
        two[1] = plane_of(second, token + 1);
        EXPECT(xbox.PresentX(2, two, nullptr) == E_INVALIDARG);
        rt::d3d12::XboxPresentPlaneParameters three[3] = {plane_of(rgba, token), plane_of(second, token), plane_of(second, token)};
        EXPECT(xbox.PresentX(3, three, nullptr) == E_INVALIDARG);
        EXPECT(xbox.PresentX(1, nullptr, nullptr) == E_INVALIDARG);
        std::printf("2 planes (plane 0 shown) and refused plane sets: ok\n");

        // The parameters struct is accepted (ImmediateThresholdPercent is ignored).
        const rt::d3d12::XboxPresentParameters params{50.0f};
        plane = plane_of(second, token);
        EXPECT(xbox.PresentX(1, &plane, &params) == S_OK);
        EXPECT(grab(frame));
        EXPECT(region_is(frame, 0, 0, width, height, {0, 255, 0}, 0));

        // Tokens: a token that was never handed out is refused and nothing is presented. A null, repeated or stale one
        // is presented anyway (and logged once per kind, see the log lines above): alternate the sources so that
        // each present shows on screen.
        const std::uint64_t presented_before = xbox.presents();
        plane = plane_of(rgba, xbox.pacer().tokens_issued() + 1);
        EXPECT(xbox.PresentX(1, &plane, nullptr) == E_INVALIDARG);
        EXPECT(xbox.presents() == presented_before);
        EXPECT(grab(frame));
        EXPECT(region_is(frame, 0, 0, width, height, {0, 255, 0}, 0));  // still the last frame
        plane = plane_of(rgba, 0);  // null token, as a loading screen might
        EXPECT(xbox.PresentX(1, &plane, nullptr) == S_OK);
        EXPECT(grab(frame));
        EXPECT(region_is(frame, 0, 0, width, height, {51, 153, 230}));
        plane = plane_of(second, token);  // repeated: the token of the last tokened present
        EXPECT(xbox.PresentX(1, &plane, nullptr) == S_OK);
        EXPECT(grab(frame));
        EXPECT(region_is(frame, 0, 0, width, height, {0, 255, 0}, 0));
        plane = plane_of(rgba, 1);  // stale: long gone
        EXPECT(xbox.PresentX(1, &plane, nullptr) == S_OK);
        EXPECT(grab(frame));
        EXPECT(region_is(frame, 0, 0, width, height, {51, 153, 230}));
        EXPECT(xbox.presents() == presented_before + 3);

        // Sources the GPU cannot blit are refused with DXGI_ERROR_UNSUPPORTED and present nothing: multisampled,
        // integer and block-compressed. The last frame stays on screen and the token is still usable afterwards.
        {
            const D3D12_HEAP_PROPERTIES default_heap{D3D12_HEAP_TYPE_DEFAULT};
            const auto make_special = [&](DXGI_FORMAT format, UINT samples, D3D12_RESOURCE_FLAGS flags,
                                          Ref<ID3D12Resource>& out) {
                D3D12_RESOURCE_DESC desc{};
                desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
                desc.Width = width;
                desc.Height = height;
                desc.DepthOrArraySize = 1;
                desc.MipLevels = 1;
                desc.Format = format;
                desc.SampleDesc.Count = samples;
                desc.Flags = flags;
                return SUCCEEDED(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_ALLOW_DISPLAY, &desc,
                                                                 D3D12_RESOURCE_STATE_COMMON, nullptr,
                                                                 IID_PPV_ARGS(&out.p)));
            };
            Ref<ID3D12Resource> msaa, integer, bc;
            EXPECT(make_special(DXGI_FORMAT_R8G8B8A8_UNORM, 4, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET, msaa));
            EXPECT(make_special(DXGI_FORMAT_R8G8B8A8_UINT, 1, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET, integer));
            EXPECT(make_special(DXGI_FORMAT_BC1_UNORM, 1, D3D12_RESOURCE_FLAG_NONE, bc));
            ID3D12Resource* unsupported[3] = {msaa.p, integer.p, bc.p};
            clock.advance_us(4 * period);  // let the queued frames show: with latency 1 the wait holds for them
            EXPECT(next_token(token));
            const std::uint64_t before_unsupported = xbox.presents();
            for (ID3D12Resource* resource : unsupported) {
                const rt::d3d12::XboxPresentPlaneParameters p{token, 1, &resource, nullptr, 0};
                EXPECT(xbox.PresentX(1, &p, nullptr) == DXGI_ERROR_UNSUPPORTED);
            }
            EXPECT(xbox.presents() == before_unsupported);
            EXPECT(grab(frame));
            EXPECT(region_is(frame, 0, 0, width, height, {51, 153, 230}));
            plane = plane_of(second, token);
            EXPECT(xbox.PresentX(1, &plane, nullptr) == S_OK);  // the same token, now with a supported source
            EXPECT(grab(frame));
            EXPECT(region_is(frame, 0, 0, width, height, {0, 255, 0}, 0));
            std::printf("unsupported sources (MSAA, integer, BC) refused: ok\n");
        }

        // Skipping a frame (waiting twice, presenting the second token) is fine.
        std::uint64_t skipped = 0;
        clock.advance_us(4 * period);
        EXPECT(next_token(skipped) && next_token(token) && token == skipped + 1);
        plane = plane_of(rgba, token);
        EXPECT(xbox.PresentX(1, &plane, nullptr) == S_OK);
        std::printf("token validation: ok\n");

        // Statistics for the frame just presented: event and present now, display after the next vblank.
        UINT count = 3;
        leonida::FrameStatistic stats[3];
        EXPECT(xbox.GetFrameStatisticsX(token, leonida::stat_frame_event | leonida::stat_present | leonida::stat_display,
                                        &count, stats) == S_FALSE);
        EXPECT(count == 2 && stats[0].type == leonida::stat_frame_event && stats[1].type == leonida::stat_present);
        clock.advance_us(period);
        count = 3;
        EXPECT(xbox.GetFrameStatisticsX(token, leonida::stat_frame_event | leonida::stat_present | leonida::stat_display,
                                        &count, stats) == S_OK);
        EXPECT(count == 3 && stats[2].type == leonida::stat_display && stats[2].token == token);
        count = 3;
        EXPECT(xbox.GetFrameStatisticsX(99999, leonida::stat_display, &count, stats) == S_FALSE && count == 0);
        std::printf("frame statistics: ok\n");

        // Suspend: presents are refused and the registration is gone; after resume it has to be redone.
        EXPECT(next_token(token));
        EXPECT(xbox.SuspendX(0) == S_OK);
        plane = plane_of(rgba, token);
        EXPECT(xbox.PresentX(1, &plane, nullptr) == invalid_state);
        EXPECT(xbox.PresentX(0, nullptr, nullptr) == invalid_state);
        EXPECT(xbox.WaitFrameEventX(rt::d3d12::xbox_frame_event_origin, 0, 0, &token) == invalid_state);
        EXPECT(xbox.ResumeX() == S_OK);
        EXPECT(xbox.WaitFrameEventX(rt::d3d12::xbox_frame_event_origin, 0, 0, &token) == invalid_state);  // not re-registered
        EXPECT(xbox.PresentX(0, nullptr, nullptr) == S_OK);  // a blank present needs no registration
        EXPECT(SUCCEEDED(xbox.SetFrameIntervalX(nullptr, period, 1, 0)));
        EXPECT(SUCCEEDED(xbox.ScheduleFrameEventX(rt::d3d12::xbox_frame_event_origin, 0, 0)));
        const std::uint64_t before_resume = xbox.pacer().tokens_issued();
        EXPECT(next_token(token) && token == before_resume + 1);
        plane = plane_of(rgba, token);
        EXPECT(xbox.PresentX(1, &plane, nullptr) == S_OK);
        EXPECT(grab(frame));
        EXPECT(region_is(frame, 0, 0, width, height, {51, 153, 230}));
        std::printf("suspend/resume with re-registration: ok\n");

        // ImmediateThresholdPercent: a frame that missed its vblank by less than the threshold is shown at once (mid-scan,
        // IMMEDIATE or FIFO_RELAXED), else it waits; the decision is the pacer's lateness on the fake clock, and the
        // display statistic and immediate_presents() tell which way it went. Without VK_*_swapchain_maintenance1 (or
        // an IMMEDIATE/FIFO_RELAXED mode the swapchain can switch to) the frame always waits.
        {
            const bool can_tear = xbox.presenter().tear_supported();
            std::printf("tearing present mode available: %s (device swapchain_maintenance1: %s)\n",
                        can_tear ? "yes" : "no", d.swapchain_maintenance1 ? "yes" : "no");
            EXPECT(d.swapchain_maintenance1 || !can_tear);
            clock.advance_us(4 * period);  // everything in flight is shown
            EXPECT(SUCCEEDED(xbox.SetFrameIntervalX(nullptr, period, 1, 0)));
            EXPECT(SUCCEEDED(xbox.ScheduleFrameEventX(rt::d3d12::xbox_frame_event_origin, 0, 0)));
            EXPECT(xbox.pacer().frames_in_flight() == 0);
            const std::uint64_t grid_epoch = clock.now_us();  // vblanks are at grid_epoch + n * period
            const std::uint64_t tears_before = xbox.immediate_presents();
            std::uint64_t expected_tears = tears_before;
            // One frame: wait for its origin, spend `late_us` after the vblank it was due at, present with `threshold`.
            // Checks the pixels, whether it was shown at once and the display statistic.
            const auto frame_with = [&](std::uint64_t late_us, float threshold, bool pass_params, bool expect_tear) -> bool {
                std::uint64_t t = 0;
                const std::uint64_t now = clock.now_us();                 // to the next vblank: origin n
                clock.advance_us(grid_epoch + ((now - grid_epoch) / period + 1) * period - now);
                if (xbox.WaitFrameEventX(rt::d3d12::xbox_frame_event_origin, 0, 0, &t) != S_OK) return false;
                clock.advance_us(period + late_us);                       // due at vblank n + 1; now = due + late_us
                rt::d3d12::XboxPresentParameters params{threshold};
                auto p = plane_of(rgba, t);
                if (xbox.PresentX(1, &p, pass_params ? &params : nullptr) != S_OK) return false;
                if (!grab(frame) || !region_is(frame, 0, 0, width, height, {51, 153, 230})) return false;
                if (expect_tear) ++expected_tears;
                if (xbox.immediate_presents() != expected_tears) {
                    std::fprintf(stderr, "late %llu us, threshold %g: immediate_presents %llu, expected %llu\n",
                                 static_cast<unsigned long long>(late_us), double(threshold),
                                 static_cast<unsigned long long>(xbox.immediate_presents()),
                                 static_cast<unsigned long long>(expected_tears));
                    return false;
                }
                // The display statistic: at once (display time = present time, scan position = lateness) or at a vblank.
                clock.advance_us(period);  // past the next vblank: shown either way
                leonida::FrameStatistic stats[2];
                UINT count = 2;
                if (xbox.GetFrameStatisticsX(t, leonida::stat_present | leonida::stat_display, &count, stats) != S_OK ||
                    count != 2)
                    return false;
                const bool at_once = stats[1].time_us == stats[0].time_us;
                if (at_once != expect_tear) {
                    std::fprintf(stderr, "late %llu us, threshold %g: shown at once = %d, expected %d\n",
                                 static_cast<unsigned long long>(late_us), double(threshold), int(at_once), int(expect_tear));
                    return false;
                }
                const float want_percent = expect_tear ? 100.0f * float(late_us) / float(period) : 0.0f;
                return std::fabs(stats[1].percent_scanned - want_percent) < 0.01f;
            };
            const bool t = can_tear;
            // The period is 16667 us: 8000 us is 48% of it, 2000 us 12%, 1000 us 6%.
            EXPECT(frame_with(8000, 60.0f, true, t));      // 48% scanned, threshold 60%: shown at once
            EXPECT(frame_with(8000, 49.0f, true, t));      // just inside the threshold
            EXPECT(frame_with(8000, 47.0f, true, false));  // the scan is past 47%: waits for the vblank
            EXPECT(frame_with(8000, 0.0f, true, false));   // threshold 0 disables it
            EXPECT(frame_with(8000, 0.0f, false, false));  // no parameters: the same
            EXPECT(frame_with(8000, -10.0f, true, false));
            EXPECT(frame_with(0, 100.0f, true, false));    // on time: nothing to tear
            EXPECT(frame_with(1000, 100.0f, true, t));     // 100: a late flip anywhere in the scan
            EXPECT(frame_with(15000, 100.0f, true, t));
            EXPECT(frame_with(2000, 10.0f, true, false));  // 12% scanned, threshold 10
            EXPECT(frame_with(2000, 13.0f, true, t));
            EXPECT(xbox.immediate_presents() == expected_tears);
            std::printf("ImmediateThresholdPercent (%s): ok, %llu of 11 frames shown at once\n",
                        can_tear ? "tearing present mode in use" : "unsupported here, frames wait",
                        static_cast<unsigned long long>(xbox.immediate_presents() - tears_before));
            if (can_tear) EXPECT(xbox.presenter().tear_presents() == xbox.immediate_presents() - tears_before);
        }

        owner.reset();  // one swapchain per window: the first presenter goes
        // Kill switch: LEONIDA_PRESENT_NO_TEAR=1 keeps the swapchain FIFO-only, so a late frame with a threshold waits.
        {
            _putenv_s("LEONIDA_PRESENT_NO_TEAR", "1");
            auto no_tear = rt::d3d12::create_window_presenter(inner, window);
            _putenv_s("LEONIDA_PRESENT_NO_TEAR", "");
            EXPECT(no_tear && no_tear->enable_capture());
            EXPECT(!no_tear->tear_supported());
            leonida::ManualFrameClock fake;
            rt::d3d12::XboxPresentation fifo_only(std::move(no_tear), &fake);
            EXPECT(SUCCEEDED(fifo_only.SetFrameIntervalX(nullptr, period, 1, 0)));
            EXPECT(SUCCEEDED(fifo_only.ScheduleFrameEventX(rt::d3d12::xbox_frame_event_origin, 0, 0)));
            fake.advance_us(period);
            std::uint64_t t = 0;
            EXPECT(fifo_only.WaitFrameEventX(rt::d3d12::xbox_frame_event_origin, 0, 0, &t) == S_OK);
            fake.advance_us(period + 8000);  // half an interval late
            rt::d3d12::XboxPresentParameters fifo_params{100.0f};
            auto p = plane_of(rgba, t);
            EXPECT(fifo_only.PresentX(1, &p, &fifo_params) == S_OK);
            EXPECT(fifo_only.immediate_presents() == 0 && fifo_only.pacer().frames_in_flight() == 1);
            EXPECT(fifo_only.presenter().read_back(frame.bytes, &frame.extent, &frame.format));
            EXPECT(region_is(frame, 0, 0, width, height, {51, 153, 230}));
            std::printf("no tearing mode (LEONIDA_PRESENT_NO_TEAR): late frame waits for the vblank: ok\n");
        }
        // Real time, 60 Hz: 30 frames cycling through three colours; tokens count up one per wait, each frame shows
        // the colour that was presented, and the loop takes about 30 vblanks.
        auto presenter2 = rt::d3d12::create_window_presenter(inner, window);
        EXPECT(presenter2 && presenter2->enable_capture());
        rt::d3d12::XboxPresentation real(std::move(presenter2));  // the real clock
        Source colours[3];
        const FLOAT* wanted[3] = {red, green, blue};
        const Px expect_rgb[3] = {{255, 0, 0}, {0, 255, 0}, {0, 0, 255}};
        for (int i = 0; i < 3; ++i) EXPECT(make_source(DXGI_FORMAT_R8G8B8A8_UNORM, colours[i]) && fill(colours[i], wanted[i]));
        EXPECT(SUCCEEDED(real.SetFrameIntervalX(nullptr, period, 1, 0)));
        EXPECT(SUCCEEDED(real.ScheduleFrameEventX(rt::d3d12::xbox_frame_event_origin, 0, 0)));
        constexpr int frames = 30;
        const auto start = std::chrono::steady_clock::now();
        std::uint64_t last = 0;
        for (int i = 0; i < frames; ++i) {
            pump();
            std::uint64_t t = 0;
            EXPECT(real.WaitFrameEventX(rt::d3d12::xbox_frame_event_origin, 1000, 0, &t) == S_OK);
            EXPECT(t == last + 1);
            last = t;
            auto p = plane_of(colours[i % 3], t);
            EXPECT(real.PresentX(1, &p, nullptr) == S_OK);
            EXPECT(real.presenter().read_back(frame.bytes, &frame.extent, &frame.format));
            EXPECT(close_to(frame.at(width / 2, height / 2), expect_rgb[i % 3], 0));
        }
        const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count();
        std::printf("real time: %d frames at 60 Hz in %lld ms (ideal %d)\n", frames, static_cast<long long>(ms),
                    frames * int(period) / 1000);
        EXPECT(ms >= (frames - 1) * int(period) / 1000 * 9 / 10 && ms < 5000);
        EXPECT(real.PresentX(0, nullptr, nullptr) == S_OK);  // the shutdown blank
    }
    DestroyWindow(window);
    UnregisterClassW(window_class.lpszClassName, window_class.hInstance);

    if (d.validation_errors()) {
        std::fprintf(stderr, "%u validation errors\n", d.validation_errors());
        return 1;
    }
    std::printf("xbox_present_test: all cases pass\n");
    return 0;
}
