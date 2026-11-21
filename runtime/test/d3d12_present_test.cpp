// Milestone M0 (docs/d3d12_vulkan.md section 6): a window shows a stable clear colour for 60 frames, through the
// Vulkan-backed device behind the Xbox-ABI proxy, with no Vulkan validation errors. Each frame resets the list,
// moves the back buffer PRESENT -> RENDER_TARGET, clears it, moves it back, executes, presents and signals a fence.
// The window is then resized, which recreates the swapchain and scales the blit. Last, the colour on screen is read
// back with GDI when the window is in front (skipped otherwise, e.g. on a locked desktop). No shaders or game data.
// Set LEONIDA_VK_VALIDATION=1 to run under VK_LAYER_KHRONOS_validation; any validation error fails the test.
#include "vk_test_device.h"
#include "../shims/d3d12_com.h"
#include "../shims/d3d12_xs_device.h"

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

void set_client_size(HWND window, int width, int height) {
    RECT rect{0, 0, width, height};
    AdjustWindowRect(&rect, WS_OVERLAPPEDWINDOW, FALSE);
    SetWindowPos(window, HWND_TOP, 0, 0, rect.right - rect.left, rect.bottom - rect.top,
                 SWP_NOMOVE | SWP_NOZORDER | SWP_SHOWWINDOW);
}

}  // namespace

int main() {
    vktest::Device d;
    EXPECT(d.create(true));

    WNDCLASSW window_class{};
    window_class.lpfnWndProc = window_proc;
    window_class.hInstance = GetModuleHandleW(nullptr);
    window_class.lpszClassName = L"LeonidaPresentTest";
    window_class.hCursor = LoadCursorW(nullptr, reinterpret_cast<LPCWSTR>(IDC_ARROW));
    EXPECT(RegisterClassW(&window_class));
    constexpr int width = 320, height = 240;
    RECT frame_rect{0, 0, width, height};
    AdjustWindowRect(&frame_rect, WS_OVERLAPPEDWINDOW, FALSE);
    HWND window = CreateWindowExW(0, window_class.lpszClassName, L"Leonida M0: D3D12 clear on Vulkan",
                                  WS_OVERLAPPEDWINDOW, 100, 100, frame_rect.right - frame_rect.left,
                                  frame_rect.bottom - frame_rect.top, nullptr, nullptr, window_class.hInstance,
                                  nullptr);
    EXPECT(window);
    ShowWindow(window, SW_SHOWNORMAL);
    SetForegroundWindow(window);
    UpdateWindow(window);
    pump();

    const FLOAT colour[4] = {0.2f, 0.6f, 0.9f, 1.0f};
    bool screen_checked = false;
    {
        rt::d3d12::VulkanDeviceInfo info{};
        info.get_instance_proc = d.instance_proc;
        info.instance = d.instance;
        info.physical = d.physical;
        info.device = d.device;
        info.queue = d.queue;
        info.queue_family = d.family;
        info.presentation = true;
        ID3D12Device* inner = nullptr;
        EXPECT(SUCCEEDED(rt::d3d12::create_vulkan_device(info, IID_PPV_ARGS(&inner))));
        Ref<ID3D12Device> device;
        EXPECT(SUCCEEDED(rt::d3d12_xs::create_device_proxy(inner, __uuidof(ID3D12Device),
                                                           reinterpret_cast<void**>(&device.p))));
        rt::d3d12::set_device_outer(inner, device.p);
        auto presenter = rt::d3d12::create_window_presenter(inner, window);
        EXPECT(presenter);

        D3D12_COMMAND_QUEUE_DESC queue_desc{};
        Ref<ID3D12CommandQueue> queue;
        Ref<ID3D12CommandAllocator> allocator;
        Ref<ID3D12GraphicsCommandList> list;
        Ref<ID3D12Fence> fence;
        Ref<ID3D12DescriptorHeap> rtv_heap;
        Ref<ID3D12Resource> back_buffer;
        EXPECT(SUCCEEDED(device->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&queue.p))));
        EXPECT(SUCCEEDED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator.p))));
        EXPECT(SUCCEEDED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.p, nullptr,
                                                   IID_PPV_ARGS(&list.p))));
        EXPECT(SUCCEEDED(list->Close()));
        EXPECT(SUCCEEDED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence.p))));
        const D3D12_DESCRIPTOR_HEAP_DESC heap_desc{D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 1, D3D12_DESCRIPTOR_HEAP_FLAG_NONE, 0};
        EXPECT(SUCCEEDED(device->CreateDescriptorHeap(&heap_desc, IID_PPV_ARGS(&rtv_heap.p))));
        const D3D12_CPU_DESCRIPTOR_HANDLE rtv{SIZE_T(reinterpret_cast<XboxHeapHandleFn>(
            (*reinterpret_cast<void***>(rtv_heap.p))[9])(rtv_heap.p))};
        const D3D12_HEAP_PROPERTIES default_heap{D3D12_HEAP_TYPE_DEFAULT};
        D3D12_RESOURCE_DESC desc{};
        desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        desc.Width = width;
        desc.Height = height;
        desc.DepthOrArraySize = 1;
        desc.MipLevels = 1;
        desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        desc.SampleDesc.Count = 1;
        desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
        EXPECT(SUCCEEDED(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &desc,
                                                         D3D12_RESOURCE_STATE_PRESENT, nullptr,
                                                         IID_PPV_ARGS(&back_buffer.p))));
        device->CreateRenderTargetView(back_buffer.p, nullptr, rtv);

        D3D12_RESOURCE_BARRIER barrier{};
        barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        barrier.Transition.pResource = back_buffer.p;
        barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        ID3D12CommandList* lists[] = {list.p};
        UINT64 signalled = 0;
        const auto frame = [&](UINT sync_interval) -> bool {
            pump();
            if (FAILED(fence->SetEventOnCompletion(signalled, nullptr))) return false;  // the list's last run is done
            if (FAILED(allocator->Reset()) || FAILED(list->Reset(allocator.p, nullptr))) return false;
            barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_PRESENT;
            barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_RENDER_TARGET;
            list->ResourceBarrier(1, &barrier);
            list->ClearRenderTargetView(rtv, colour, 0, nullptr);
            barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_RENDER_TARGET;
            barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_PRESENT;
            list->ResourceBarrier(1, &barrier);
            if (FAILED(list->Close())) return false;
            queue->ExecuteCommandLists(1, lists);
            if (rt::d3d12::last_execute_result(queue.p) != leonida::ReplayResult::success) return false;
            if (FAILED(presenter->present(back_buffer.p, sync_interval))) return false;
            return SUCCEEDED(queue->Signal(fence.p, ++signalled));
        };

        for (int i = 0; i < 60; ++i) EXPECT(frame(1));
        EXPECT(presenter->presented() == 60);
        const VkExtent2D first = presenter->extent();
        EXPECT(first.width == width && first.height == height);

        // A resize recreates the swapchain; the 320x240 back buffer is scaled to the new size. Unsynchronised
        // presents (sync interval 0) switch the present mode on the way.
        set_client_size(window, 400, 300);
        for (int i = 0; i < 10; ++i) EXPECT(frame(i < 5 ? 1 : 0));
        EXPECT(presenter->extent().width == 400 && presenter->extent().height == 300);
        EXPECT(presenter->presented() == 70);
        EXPECT(SUCCEEDED(fence->SetEventOnCompletion(signalled, nullptr)));

        // What reached the screen: the centre of the client area, if the window is in front.
        Sleep(200);
        pump();
        if (GetForegroundWindow() == window && !IsIconic(window)) {
            RECT client{};
            GetClientRect(window, &client);
            POINT centre{(client.right - client.left) / 2, (client.bottom - client.top) / 2};
            ClientToScreen(window, &centre);
            HDC screen = GetDC(nullptr);
            const COLORREF pixel = GetPixel(screen, centre.x, centre.y);
            ReleaseDC(nullptr, screen);
            const int r = GetRValue(pixel), g = GetGValue(pixel), b = GetBValue(pixel);
            std::printf("screen pixel at the window centre: %d,%d,%d (cleared to 51,153,230)\n", r, g, b);
            EXPECT(pixel != CLR_INVALID);
            EXPECT(std::abs(r - 51) <= 3 && std::abs(g - 153) <= 3 && std::abs(b - 230) <= 3);
            screen_checked = true;
        } else {
            std::printf("the window is not in front: on-screen colour check skipped\n");
        }

        presenter.reset();  // before the device's queue and resources go
    }
    DestroyWindow(window);
    UnregisterClassW(window_class.lpszClassName, window_class.hInstance);

    if (d.validation_errors()) {
        std::fprintf(stderr, "%u validation errors\n", d.validation_errors());
        return 1;
    }
    std::printf("M0: 70 frames cleared and presented through the Xbox-ABI proxy on Vulkan, swapchain recreated on "
                "resize%s\n", screen_checked ? ", colour confirmed on screen" : "");
    return 0;
}
