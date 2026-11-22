// Real Win32 HWND GPU test for the Vulkan-backed IDXGIFactory4 / IDXGISwapChain3 COM shim
// (runtime/shims/dxgi_swapchain.h, runtime/shims/dxgi_com.{h,cpp}).
// No shaders or game data. Set LEONIDA_VK_VALIDATION=1 to run under VK_LAYER_KHRONOS_validation
// (Core + Synchronization Validation); any validation error fails the test.
#ifndef VK_USE_PLATFORM_WIN32_KHR
#define VK_USE_PLATFORM_WIN32_KHR
#endif

#include "../shims/d3d12_com.h"
#include "../shims/dxgi_com.h"
#include "vulkan_loader.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

int fail(const char* what) {
    std::fprintf(stderr, "FAIL: %s\n", what);
    return 1;
}

unsigned& validation_errors() {
    static unsigned count = 0;
    return count;
}

VKAPI_ATTR VkBool32 VKAPI_CALL on_validation(VkDebugUtilsMessageSeverityFlagBitsEXT severity,
                                             VkDebugUtilsMessageTypeFlagsEXT types,
                                             const VkDebugUtilsMessengerCallbackDataEXT* data, void*) {
    if ((severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) &&
        (types & VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT))
        ++validation_errors();
    std::fprintf(stderr, "validation: %s\n", data && data->pMessage ? data->pMessage : "(no message)");
    return VK_FALSE;
}

D3D12_RESOURCE_BARRIER make_transition(ID3D12Resource* res, D3D12_RESOURCE_STATES before, D3D12_RESOURCE_STATES after) {
    D3D12_RESOURCE_BARRIER b{};
    b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    b.Transition.pResource = res;
    b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    b.Transition.StateBefore = before;
    b.Transition.StateAfter = after;
    return b;
}

bool wait_fence(ID3D12Fence* fence, UINT64 value) {
    HANDLE ev = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    if (!ev) return false;
    const HRESULT hr = fence->SetEventOnCompletion(value, ev);
    if (FAILED(hr)) {
        CloseHandle(ev);
        return false;
    }
    const DWORD w = WaitForSingleObject(ev, 5000);
    CloseHandle(ev);
    return w == WAIT_OBJECT_0;
}

LRESULT CALLBACK wnd_proc(HWND hwnd, UINT msg, WPARAM wp, LPARAM lp) { return DefWindowProcW(hwnd, msg, wp, lp); }

HWND create_test_window(int width, int height) {
    WNDCLASSW wc{};
    wc.lpfnWndProc = wnd_proc;
    wc.hInstance = GetModuleHandleW(nullptr);
    wc.lpszClassName = L"LeonidaDxgiSwapchainTestWnd";
    RegisterClassW(&wc);
    RECT rc{0, 0, width, height};
    AdjustWindowRect(&rc, WS_OVERLAPPEDWINDOW, FALSE);
    return CreateWindowExW(0, wc.lpszClassName, L"Leonida DXGI Test", WS_OVERLAPPEDWINDOW, CW_USEDEFAULT,
                           CW_USEDEFAULT, rc.right - rc.left, rc.bottom - rc.top, nullptr, nullptr, wc.hInstance,
                           nullptr);
}

}  // namespace

int main() {
    const PFN_vkGetInstanceProcAddr gipa = load_vulkan();
    if (!gipa) {
        std::printf("SKIP: vulkan-1.dll not available\n");
        return 77;  // ctest skips
    }
    const auto create_inst = reinterpret_cast<PFN_vkCreateInstance>(gipa(nullptr, "vkCreateInstance"));
    const auto enum_layers =
        reinterpret_cast<PFN_vkEnumerateInstanceLayerProperties>(gipa(nullptr, "vkEnumerateInstanceLayerProperties"));
    if (!create_inst) return fail("vkCreateInstance missing");

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4996)
#endif
    const char* env_val = std::getenv("LEONIDA_VK_VALIDATION");
#ifdef _MSC_VER
#pragma warning(pop)
#endif
    const bool want_val = env_val && std::strcmp(env_val, "0") != 0;
    const char* val_layer = "VK_LAYER_KHRONOS_validation";
    bool val_active = false;
    if (want_val && enum_layers) {
        std::uint32_t cnt = 0;
        enum_layers(&cnt, nullptr);
        std::vector<VkLayerProperties> layers(cnt);
        enum_layers(&cnt, layers.data());
        for (const auto& l : layers) {
            if (!std::strcmp(l.layerName, val_layer)) val_active = true;
        }
        if (!val_active) return fail("LEONIDA_VK_VALIDATION is set but VK_LAYER_KHRONOS_validation is not installed");
    }

    std::vector<const char*> inst_exts = {VK_KHR_SURFACE_EXTENSION_NAME, VK_KHR_WIN32_SURFACE_EXTENSION_NAME};
    if (val_active) inst_exts.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);

    VkDebugUtilsMessengerCreateInfoEXT dbg{VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT};
    dbg.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    dbg.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT;
    dbg.pfnUserCallback = on_validation;

    const VkValidationFeatureEnableEXT sync_val = VK_VALIDATION_FEATURE_ENABLE_SYNCHRONIZATION_VALIDATION_EXT;
    VkValidationFeaturesEXT val_feats{VK_STRUCTURE_TYPE_VALIDATION_FEATURES_EXT, &dbg};
    val_feats.enabledValidationFeatureCount = 1;
    val_feats.pEnabledValidationFeatures = &sync_val;

    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.apiVersion = VK_API_VERSION_1_3;
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ici.pApplicationInfo = &app;
    ici.enabledExtensionCount = static_cast<std::uint32_t>(inst_exts.size());
    ici.ppEnabledExtensionNames = inst_exts.data();
    if (val_active) {
        ici.enabledLayerCount = 1;
        ici.ppEnabledLayerNames = &val_layer;
        ici.pNext = &val_feats;
    }

    VkInstance instance = VK_NULL_HANDLE;
    const VkResult created = create_inst(&ici, nullptr, &instance);
    if (created == VK_ERROR_INCOMPATIBLE_DRIVER) {  // no Vulkan driver at all, as on a GPU-less runner
        std::printf("SKIP: no Vulkan driver\n");
        return 77;
    }
    if (created != VK_SUCCESS) return fail("vkCreateInstance");

    const auto destroy_inst = reinterpret_cast<PFN_vkDestroyInstance>(gipa(instance, "vkDestroyInstance"));
    const auto gdpa = reinterpret_cast<PFN_vkGetDeviceProcAddr>(gipa(instance, "vkGetDeviceProcAddr"));
    VkDebugUtilsMessengerEXT messenger = VK_NULL_HANDLE;
    PFN_vkDestroyDebugUtilsMessengerEXT destroy_dbg = nullptr;
    if (val_active) {
        const auto create_dbg =
            reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(gipa(instance, "vkCreateDebugUtilsMessengerEXT"));
        destroy_dbg =
            reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(gipa(instance, "vkDestroyDebugUtilsMessengerEXT"));
        if (create_dbg) create_dbg(instance, &dbg, nullptr, &messenger);
    }

    const auto enum_devs = reinterpret_cast<PFN_vkEnumeratePhysicalDevices>(gipa(instance, "vkEnumeratePhysicalDevices"));
    const auto get_props = reinterpret_cast<PFN_vkGetPhysicalDeviceProperties>(gipa(instance, "vkGetPhysicalDeviceProperties"));
    const auto get_qf = reinterpret_cast<PFN_vkGetPhysicalDeviceQueueFamilyProperties>(
        gipa(instance, "vkGetPhysicalDeviceQueueFamilyProperties"));
    const auto create_dev = reinterpret_cast<PFN_vkCreateDevice>(gipa(instance, "vkCreateDevice"));

    std::uint32_t pcnt = 0;
    enum_devs(instance, &pcnt, nullptr);
    std::vector<VkPhysicalDevice> physicals(pcnt);
    enum_devs(instance, &pcnt, physicals.data());

    VkPhysicalDevice physical = VK_NULL_HANDLE;
    std::uint32_t family = 0;
    VkPhysicalDeviceProperties chosen{};
    for (auto cand : physicals) {
        VkPhysicalDeviceProperties props{};
        get_props(cand, &props);
        if (props.apiVersion < VK_API_VERSION_1_3) continue;
        std::uint32_t fcnt = 0;
        get_qf(cand, &fcnt, nullptr);
        std::vector<VkQueueFamilyProperties> fams(fcnt);
        get_qf(cand, &fcnt, fams.data());
        for (std::uint32_t i = 0; i < fcnt; ++i) {
            if (!(fams[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)) continue;
            if (!physical || props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
                physical = cand;
                family = i;
                chosen = props;
            }
            break;
        }
    }
    if (!physical) {
        std::printf("SKIP: no Vulkan 1.3 GPU found\n");
        if (messenger && destroy_dbg) destroy_dbg(instance, messenger, nullptr);
        destroy_inst(instance, nullptr);
        return 77;  // ctest skips
    }

    VkPhysicalDeviceVulkan13Features f13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
    f13.dynamicRendering = VK_TRUE;
    f13.synchronization2 = VK_TRUE;
    VkPhysicalDeviceVulkan12Features f12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES, &f13};
    f12.timelineSemaphore = VK_TRUE;
    f12.separateDepthStencilLayouts = VK_TRUE;

    const float prio = 1.0f;
    VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qci.queueFamilyIndex = family;
    qci.queueCount = 1;
    qci.pQueuePriorities = &prio;

    const char* dev_ext = VK_KHR_SWAPCHAIN_EXTENSION_NAME;
    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, &f12};
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci;
    dci.enabledExtensionCount = 1;
    dci.ppEnabledExtensionNames = &dev_ext;

    VkDevice vk_dev = VK_NULL_HANDLE;
    if (create_dev(physical, &dci, nullptr, &vk_dev) != VK_SUCCESS) return fail("vkCreateDevice");
    const auto destroy_dev = reinterpret_cast<PFN_vkDestroyDevice>(gdpa(vk_dev, "vkDestroyDevice"));
    const auto wait_idle = reinterpret_cast<PFN_vkDeviceWaitIdle>(gdpa(vk_dev, "vkDeviceWaitIdle"));
    VkQueue vk_queue = VK_NULL_HANDLE;
    reinterpret_cast<PFN_vkGetDeviceQueue>(gdpa(vk_dev, "vkGetDeviceQueue"))(vk_dev, family, 0, &vk_queue);

    rt::d3d12::VulkanDeviceInfo info{};
    info.get_instance_proc = gipa;
    info.instance = instance;
    info.physical = physical;
    info.device = vk_dev;
    info.queue = vk_queue;
    info.queue_family = family;
    info.separate_depth_stencil_layouts = true;

    ID3D12Device* dev = nullptr;
    if (FAILED(rt::d3d12::create_vulkan_device(info, IID_PPV_ARGS(&dev))) || !dev) return fail("create_vulkan_device");

    HWND hwnd = create_test_window(128, 128);
    if (!hwnd) return fail("CreateWindowExW");

    IDXGIFactory4* factory = nullptr;
    if (FAILED(rt::d3d12::create_dxgi_factory(dev, IID_PPV_ARGS(&factory))) || !factory)
        return fail("create_dxgi_factory");

    D3D12_COMMAND_QUEUE_DESC qd{D3D12_COMMAND_LIST_TYPE_DIRECT, 0, D3D12_COMMAND_QUEUE_FLAG_NONE, 0};
    ID3D12CommandQueue* queue = nullptr;
    ID3D12CommandAllocator* alloc = nullptr;
    ID3D12GraphicsCommandList* cl = nullptr;
    ID3D12Fence* fence = nullptr;
    if (FAILED(dev->CreateCommandQueue(&qd, IID_PPV_ARGS(&queue))) ||
        FAILED(dev->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&alloc))) ||
        FAILED(dev->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, alloc, nullptr, IID_PPV_ARGS(&cl))) ||
        FAILED(dev->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence))))
        return fail("queue/alloc/cl/fence");
    cl->Close();

    DXGI_SWAP_CHAIN_DESC1 scd{};
    scd.Width = 128;
    scd.Height = 128;
    scd.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
    scd.SampleDesc.Count = 1;
    scd.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    scd.BufferCount = 2;
    scd.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;

    IDXGISwapChain1* sc1 = nullptr;
    if (FAILED(factory->CreateSwapChainForHwnd(queue, hwnd, &scd, nullptr, nullptr, &sc1)) || !sc1)
        return fail("CreateSwapChainForHwnd");
    IDXGISwapChain3* sc = nullptr;
    if (FAILED(sc1->QueryInterface(IID_PPV_ARGS(&sc))) || !sc) return fail("QI IDXGISwapChain3");
    sc1->Release();

    D3D12_DESCRIPTOR_HEAP_DESC rtv_hd{D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 1, D3D12_DESCRIPTOR_HEAP_FLAG_NONE, 0};
    ID3D12DescriptorHeap* rtv_heap = nullptr;
    if (FAILED(dev->CreateDescriptorHeap(&rtv_hd, IID_PPV_ARGS(&rtv_heap)))) return fail("CreateDescriptorHeap");
    const D3D12_CPU_DESCRIPTOR_HANDLE h_rtv = rtv_heap->GetCPUDescriptorHandleForHeapStart();

    UINT64 fence_val = 0;
    auto run_clear_readback_present = [&](UINT exp_w, UINT exp_h, const float rgba[4], std::uint8_t exp_r,
                                          std::uint8_t exp_g, std::uint8_t exp_b) -> int {
        DXGI_SWAP_CHAIN_DESC1 cur_desc{};
        if (FAILED(sc->GetDesc1(&cur_desc)) || cur_desc.Width != exp_w || cur_desc.Height != exp_h) {
            std::fprintf(stderr, "FAIL: GetDesc1 (%ux%u) != expected (%ux%u)\n", cur_desc.Width, cur_desc.Height,
                         exp_w, exp_h);
            return 1;
        }
        const UINT idx = sc->GetCurrentBackBufferIndex();
        ID3D12Resource* bb = nullptr;
        if (FAILED(sc->GetBuffer(idx, IID_PPV_ARGS(&bb))) || !bb) return fail("GetBuffer");

        const D3D12_RESOURCE_DESC bb_desc = bb->GetDesc();
        if (bb_desc.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D || bb_desc.Width != exp_w ||
            bb_desc.Height != exp_h) {
            std::fprintf(stderr, "FAIL: BackBuffer GetDesc (%llux%u) != expected (%ux%u)\n",
                         static_cast<unsigned long long>(bb_desc.Width), bb_desc.Height, exp_w, exp_h);
            bb->Release();
            return 1;
        }
        D3D12_HEAP_PROPERTIES hp_out{};
        D3D12_HEAP_FLAGS hf_out{};
        if (FAILED(bb->GetHeapProperties(&hp_out, &hf_out)) || hp_out.Type != D3D12_HEAP_TYPE_DEFAULT) {
            bb->Release();
            return fail("BackBuffer GetHeapProperties invalid");
        }

        dev->CreateRenderTargetView(bb, nullptr, h_rtv);

        D3D12_PLACED_SUBRESOURCE_FOOTPRINT fp{};
        UINT64 total_bytes = 0;
        dev->GetCopyableFootprints(&bb_desc, 0, 1, 0, &fp, nullptr, nullptr, &total_bytes);

        D3D12_HEAP_PROPERTIES rb_hp{D3D12_HEAP_TYPE_READBACK};
        D3D12_RESOURCE_DESC rb_desc{};
        rb_desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        rb_desc.Width = total_bytes;
        rb_desc.Height = 1;
        rb_desc.DepthOrArraySize = 1;
        rb_desc.MipLevels = 1;
        rb_desc.SampleDesc.Count = 1;
        rb_desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        ID3D12Resource* rb = nullptr;
        if (FAILED(dev->CreateCommittedResource(&rb_hp, D3D12_HEAP_FLAG_NONE, &rb_desc, D3D12_RESOURCE_STATE_COPY_DEST,
                                                nullptr, IID_PPV_ARGS(&rb)))) {
            bb->Release();
            return fail("CreateCommittedResource(readback)");
        }

        alloc->Reset();
        cl->Reset(alloc, nullptr);
        auto b_rt = make_transition(bb, D3D12_RESOURCE_STATE_PRESENT, D3D12_RESOURCE_STATE_RENDER_TARGET);
        cl->ResourceBarrier(1, &b_rt);
        cl->ClearRenderTargetView(h_rtv, rgba, 0, nullptr);
        auto b_src = make_transition(bb, D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_COPY_SOURCE);
        cl->ResourceBarrier(1, &b_src);
        D3D12_TEXTURE_COPY_LOCATION src_loc{bb, D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, {0}};
        D3D12_TEXTURE_COPY_LOCATION dst_loc{};
        dst_loc.pResource = rb;
        dst_loc.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
        dst_loc.PlacedFootprint = fp;
        cl->CopyTextureRegion(&dst_loc, 0, 0, 0, &src_loc, nullptr);
        auto b_pres = make_transition(bb, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_PRESENT);
        cl->ResourceBarrier(1, &b_pres);
        cl->Close();

        ID3D12CommandList* lists[1] = {cl};
        queue->ExecuteCommandLists(1, lists);
        queue->Signal(fence, ++fence_val);
        if (!wait_fence(fence, fence_val)) {
            rb->Release();
            bb->Release();
            return fail("wait_fence");
        }

        void* mapped = nullptr;
        if (FAILED(rb->Map(0, nullptr, &mapped)) || !mapped) {
            rb->Release();
            bb->Release();
            return fail("readback Map");
        }
        const auto* base = static_cast<const std::uint8_t*>(mapped);
        const bool is_bgra = (bb_desc.Format == DXGI_FORMAT_B8G8R8A8_UNORM ||
                              bb_desc.Format == DXGI_FORMAT_B8G8R8A8_UNORM_SRGB);
        bool mismatch = false;
        std::uint8_t bad_r = 0, bad_g = 0, bad_b = 0, bad_a = 0;
        for (UINT y = 0; y < exp_h && !mismatch; ++y) {
            const std::uint8_t* row = base + static_cast<std::size_t>(y) * fp.Footprint.RowPitch;
            for (UINT x = 0; x < exp_w; ++x) {
                const std::uint8_t* px = row + static_cast<std::size_t>(x) * 4;
                const std::uint8_t r = is_bgra ? px[2] : px[0];
                const std::uint8_t g = px[1];
                const std::uint8_t b = is_bgra ? px[0] : px[2];
                const std::uint8_t a = px[3];
                if (std::abs(int(r) - int(exp_r)) > 1 || std::abs(int(g) - int(exp_g)) > 1 ||
                    std::abs(int(b) - int(exp_b)) > 1 || a != 255) {
                    mismatch = true;
                    bad_r = r;
                    bad_g = g;
                    bad_b = b;
                    bad_a = a;
                    break;
                }
            }
        }
        rb->Unmap(0, nullptr);
        rb->Release();
        bb->Release();

        if (mismatch) {
            std::fprintf(stderr, "FAIL: backbuffer pixel (%u,%u,%u,%u) != expected (%u,%u,%u,255)\n", bad_r, bad_g,
                         bad_b, bad_a, exp_r, exp_g, exp_b);
            return 1;
        }
        if (FAILED(sc->Present(0, 0))) return fail("Present");
        return 0;
    };

    // Frame 1: 128x128 clear to (64, 128, 255, 255), read back all 16,384 pixels, Present.
    const float blue_ish[4] = {64.0f / 255.0f, 128.0f / 255.0f, 1.0f, 1.0f};
    if (run_clear_readback_present(128, 128, blue_ish, 64, 128, 255) != 0) return 1;

    // Frame 2: ResizeBuffers to explicit 37x19 on the unchanged 128x128 HWND, clear to green, read back all 703 pixels, Present.
    if (FAILED(sc->ResizeBuffers(2, 37, 19, DXGI_FORMAT_UNKNOWN, 0))) return fail("ResizeBuffers");
    const float green[4] = {0.0f, 1.0f, 0.0f, 1.0f};
    if (run_clear_readback_present(37, 19, green, 0, 255, 0) != 0) return 1;

    rtv_heap->Release();
    sc->Release();
    fence->Release();
    cl->Release();
    alloc->Release();
    queue->Release();
    factory->Release();
    dev->Release();
    DestroyWindow(hwnd);
    wait_idle(vk_dev);
    destroy_dev(vk_dev, nullptr);
    if (messenger && destroy_dbg) destroy_dbg(instance, messenger, nullptr);
    destroy_inst(instance, nullptr);

    if (validation_errors() != 0) {
        std::fprintf(stderr, "FAIL: %u Vulkan validation errors\n", validation_errors());
        return 1;
    }
    std::printf("device: %s (validation %s)\n", chosen.deviceName, val_active ? "on" : "off");
    std::printf("dxgi_swapchain_test: PASS (CreateSwapChainForHwnd, GetBuffer+GetDesc, clear+readback, Present, "
                "ResizeBuffers; 0 validation errors)\n");
    return 0;
}
