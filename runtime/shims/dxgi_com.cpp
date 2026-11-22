#include "dxgi_com.h"
#include "d3d12_stubs.h"

#include <atomic>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <set>
#include <string>
#include <vector>

namespace rt::dxgi {
namespace {

void stub(const char* method) {
    static std::mutex mutex;
    static std::set<std::string> seen;
    std::lock_guard lock(mutex);
    if (seen.insert(method).second) std::fprintf(stderr, "[rt] dxgi: unimplemented %s\n", method);
}

VkFormat dxgi_to_vk_format(DXGI_FORMAT fmt) {
    switch (fmt) {
    case DXGI_FORMAT_R8G8B8A8_UNORM: return VK_FORMAT_R8G8B8A8_UNORM;
    case DXGI_FORMAT_R8G8B8A8_UNORM_SRGB: return VK_FORMAT_R8G8B8A8_SRGB;
    case DXGI_FORMAT_B8G8R8A8_UNORM_SRGB: return VK_FORMAT_B8G8R8A8_SRGB;
    case DXGI_FORMAT_R16G16B16A16_FLOAT: return VK_FORMAT_R16G16B16A16_SFLOAT;
    case DXGI_FORMAT_B8G8R8A8_UNORM:
    default: return VK_FORMAT_B8G8R8A8_UNORM;
    }
}

DXGI_FORMAT vk_to_dxgi_format(VkFormat fmt) {
    switch (fmt) {
    case VK_FORMAT_R8G8B8A8_UNORM: return DXGI_FORMAT_R8G8B8A8_UNORM;
    case VK_FORMAT_R8G8B8A8_SRGB: return DXGI_FORMAT_R8G8B8A8_UNORM_SRGB;
    case VK_FORMAT_B8G8R8A8_SRGB: return DXGI_FORMAT_B8G8R8A8_UNORM_SRGB;
    case VK_FORMAT_R16G16B16A16_SFLOAT: return DXGI_FORMAT_R16G16B16A16_FLOAT;
    case VK_FORMAT_B8G8R8A8_UNORM:
    default: return DXGI_FORMAT_B8G8R8A8_UNORM;
    }
}

// Real ID3D12Resource COM object representing a swapchain backbuffer. Its pointer value is the
// leonida::ResourceId registered in VulkanContext, matching d3d12_com.h's resource_id(ID3D12Resource*) contract.
class BackBufferResource final : public ID3D12ResourceStubs {
public:
    explicit BackBufferResource(const D3D12_RESOURCE_DESC& desc) : desc_(desc) {}

    ULONG STDMETHODCALLTYPE AddRef() override { return ++refs_; }
    ULONG STDMETHODCALLTYPE Release() override {
        const ULONG left = --refs_;
        if (!left) delete this;
        return left;
    }
    HRESULT STDMETHODCALLTYPE QueryInterface(REFIID riid, void** out) override {
        if (!out) return E_POINTER;
        *out = nullptr;
        if (riid == __uuidof(IUnknown) || riid == __uuidof(ID3D12Object) || riid == __uuidof(ID3D12DeviceChild) ||
            riid == __uuidof(ID3D12Pageable) || riid == __uuidof(ID3D12Resource)) {
            *out = static_cast<ID3D12Resource*>(this);
            AddRef();
            return S_OK;
        }
        return E_NOINTERFACE;
    }

    HRESULT STDMETHODCALLTYPE SetName(LPCWSTR) override { return S_OK; }

#ifdef _MSC_VER
    D3D12_RESOURCE_DESC STDMETHODCALLTYPE GetDesc(void) override { return desc_; }
#else
    D3D12_RESOURCE_DESC* STDMETHODCALLTYPE GetDesc(D3D12_RESOURCE_DESC* ret) override {
        if (ret) *ret = desc_;
        return ret;
    }
#endif

    D3D12_GPU_VIRTUAL_ADDRESS STDMETHODCALLTYPE GetGPUVirtualAddress(void) override { return 0; }

    HRESULT STDMETHODCALLTYPE GetHeapProperties(D3D12_HEAP_PROPERTIES* props, D3D12_HEAP_FLAGS* flags) override {
        if (props) {
            *props = {};
            props->Type = D3D12_HEAP_TYPE_DEFAULT;
            props->CreationNodeMask = 1;
            props->VisibleNodeMask = 1;
        }
        if (flags) *flags = D3D12_HEAP_FLAG_ALLOW_ONLY_RT_DS_TEXTURES;
        return S_OK;
    }

    leonida::ResourceId resource_id() const { return reinterpret_cast<leonida::ResourceId>(this); }

private:
    std::atomic<ULONG> refs_{1};
    D3D12_RESOURCE_DESC desc_{};
};

class SwapChain final : public IDXGISwapChain3 {
public:
    SwapChain(leonida::VulkanContext& context, PFN_vkGetInstanceProcAddr gipa, VkInstance instance,
              VkPhysicalDevice physical, PFN_vkGetDeviceProcAddr gdpa, VkQueue queue, std::uint32_t queue_family)
        : context_(context),
          swapchain_(gipa, instance, physical, gdpa, context.device(), queue, queue_family, context.submit_mutex()) {}

    ~SwapChain() { unregister_buffers(); }

    bool init(HWND hwnd, const DXGI_SWAP_CHAIN_DESC1& desc) {
        hwnd_ = hwnd;
        const std::uint32_t w = desc.Width ? desc.Width : 256u;
        const std::uint32_t h = desc.Height ? desc.Height : 256u;
        const std::uint32_t count = desc.BufferCount ? desc.BufferCount : 2u;
        if (!swapchain_.create(hwnd, w, h, dxgi_to_vk_format(desc.Format), count)) return false;
        register_buffers();
        std::uint32_t idx = 0;
        return swapchain_.acquire_next_image(idx);
    }

    // IUnknown
    ULONG STDMETHODCALLTYPE AddRef() override { return ++refs_; }
    ULONG STDMETHODCALLTYPE Release() override {
        const ULONG left = --refs_;
        if (!left) delete this;
        return left;
    }
    HRESULT STDMETHODCALLTYPE QueryInterface(REFIID riid, void** out) override {
        if (!out) return E_POINTER;
        *out = nullptr;
        if (riid == __uuidof(IUnknown) || riid == __uuidof(IDXGIObject) || riid == __uuidof(IDXGIDeviceSubObject) ||
            riid == __uuidof(IDXGISwapChain) || riid == __uuidof(IDXGISwapChain1) ||
            riid == __uuidof(IDXGISwapChain2) || riid == __uuidof(IDXGISwapChain3)) {
            *out = static_cast<IDXGISwapChain3*>(this);
            AddRef();
            return S_OK;
        }
        return E_NOINTERFACE;
    }

    // IDXGIObject
    HRESULT STDMETHODCALLTYPE SetPrivateData(REFGUID, UINT, const void*) override { return S_OK; }
    HRESULT STDMETHODCALLTYPE SetPrivateDataInterface(REFGUID, const IUnknown*) override { return S_OK; }
    HRESULT STDMETHODCALLTYPE GetPrivateData(REFGUID, UINT*, void*) override { return DXGI_ERROR_NOT_FOUND; }
    HRESULT STDMETHODCALLTYPE GetParent(REFIID, void**) override { stub("IDXGISwapChain::GetParent"); return E_NOTIMPL; }

    // IDXGIDeviceSubObject
    HRESULT STDMETHODCALLTYPE GetDevice(REFIID, void**) override { stub("IDXGISwapChain::GetDevice"); return E_NOTIMPL; }

    // IDXGISwapChain
    HRESULT STDMETHODCALLTYPE Present(UINT, UINT) override {
        const std::uint32_t cur = swapchain_.current_back_buffer_index();
        if (!swapchain_.present(cur)) return E_FAIL;
        std::uint32_t next = 0;
        return swapchain_.acquire_next_image(next) ? S_OK : E_FAIL;
    }

    HRESULT STDMETHODCALLTYPE GetBuffer(UINT buffer, REFIID riid, void** surface) override {
        if (!surface) return E_POINTER;
        *surface = nullptr;
        if (buffer >= buffers_.size()) return E_INVALIDARG;
        return buffers_[buffer]->QueryInterface(riid, surface);
    }

    HRESULT STDMETHODCALLTYPE SetFullscreenState(BOOL, IDXGIOutput*) override { return S_OK; }
    HRESULT STDMETHODCALLTYPE GetFullscreenState(BOOL* fs, IDXGIOutput** out) override {
        if (fs) *fs = FALSE;
        if (out) *out = nullptr;
        return S_OK;
    }

    HRESULT STDMETHODCALLTYPE GetDesc(DXGI_SWAP_CHAIN_DESC* desc) override {
        if (!desc) return E_INVALIDARG;
        std::memset(desc, 0, sizeof(*desc));
        desc->BufferDesc.Width = swapchain_.extent().width;
        desc->BufferDesc.Height = swapchain_.extent().height;
        desc->BufferDesc.Format = vk_to_dxgi_format(swapchain_.format());
        desc->SampleDesc.Count = 1;
        desc->BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
        desc->BufferCount = swapchain_.image_count();
        desc->OutputWindow = hwnd_;
        desc->Windowed = TRUE;
        desc->SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
        return S_OK;
    }

    HRESULT STDMETHODCALLTYPE ResizeBuffers(UINT count, UINT width, UINT height, DXGI_FORMAT fmt, UINT) override {
        unregister_buffers();
        const std::uint32_t w = width ? width : swapchain_.extent().width;
        const std::uint32_t h = height ? height : swapchain_.extent().height;
        const VkFormat vk_fmt = (fmt != DXGI_FORMAT_UNKNOWN) ? dxgi_to_vk_format(fmt) : VK_FORMAT_UNDEFINED;
        if (!swapchain_.resize(w, h, vk_fmt, count)) return E_FAIL;
        register_buffers();
        std::uint32_t next = 0;
        return swapchain_.acquire_next_image(next) ? S_OK : E_FAIL;
    }

    HRESULT STDMETHODCALLTYPE ResizeTarget(const DXGI_MODE_DESC*) override { return S_OK; }
    HRESULT STDMETHODCALLTYPE GetContainingOutput(IDXGIOutput**) override { return DXGI_ERROR_NOT_CURRENTLY_AVAILABLE; }
    HRESULT STDMETHODCALLTYPE GetFrameStatistics(DXGI_FRAME_STATISTICS*) override { return E_NOTIMPL; }
    HRESULT STDMETHODCALLTYPE GetLastPresentCount(UINT* p) override { if (p) *p = 0; return S_OK; }

    // IDXGISwapChain1
    HRESULT STDMETHODCALLTYPE GetDesc1(DXGI_SWAP_CHAIN_DESC1* desc) override {
        if (!desc) return E_INVALIDARG;
        std::memset(desc, 0, sizeof(*desc));
        desc->Width = swapchain_.extent().width;
        desc->Height = swapchain_.extent().height;
        desc->Format = vk_to_dxgi_format(swapchain_.format());
        desc->SampleDesc.Count = 1;
        desc->BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
        desc->BufferCount = swapchain_.image_count();
        desc->Scaling = DXGI_SCALING_STRETCH;
        desc->SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
        desc->AlphaMode = DXGI_ALPHA_MODE_IGNORE;
        return S_OK;
    }
    HRESULT STDMETHODCALLTYPE GetFullscreenDesc(DXGI_SWAP_CHAIN_FULLSCREEN_DESC*) override { return E_NOTIMPL; }
    HRESULT STDMETHODCALLTYPE GetHwnd(HWND* out) override { if (!out) return E_INVALIDARG; *out = hwnd_; return S_OK; }
    HRESULT STDMETHODCALLTYPE GetCoreWindow(REFIID, void**) override { return E_NOTIMPL; }
    HRESULT STDMETHODCALLTYPE Present1(UINT sync, UINT flags, const DXGI_PRESENT_PARAMETERS*) override {
        return Present(sync, flags);
    }
    BOOL STDMETHODCALLTYPE IsTemporaryMonoSupported() override { return FALSE; }
    HRESULT STDMETHODCALLTYPE GetRestrictToOutput(IDXGIOutput**) override { return E_NOTIMPL; }
    HRESULT STDMETHODCALLTYPE SetBackgroundColor(const DXGI_RGBA*) override { return S_OK; }
    HRESULT STDMETHODCALLTYPE GetBackgroundColor(DXGI_RGBA*) override { return E_NOTIMPL; }
    HRESULT STDMETHODCALLTYPE SetRotation(DXGI_MODE_ROTATION) override { return S_OK; }
    HRESULT STDMETHODCALLTYPE GetRotation(DXGI_MODE_ROTATION* r) override {
        if (r) *r = DXGI_MODE_ROTATION_IDENTITY;
        return S_OK;
    }

    // IDXGISwapChain2
    HRESULT STDMETHODCALLTYPE SetSourceSize(UINT, UINT) override { return S_OK; }
    HRESULT STDMETHODCALLTYPE GetSourceSize(UINT* w, UINT* h) override {
        if (w) *w = swapchain_.extent().width;
        if (h) *h = swapchain_.extent().height;
        return S_OK;
    }
    HRESULT STDMETHODCALLTYPE SetMaximumFrameLatency(UINT) override { return S_OK; }
    HRESULT STDMETHODCALLTYPE GetMaximumFrameLatency(UINT* n) override { if (n) *n = 2; return S_OK; }
    HANDLE STDMETHODCALLTYPE GetFrameLatencyWaitableObject() override { return nullptr; }
    HRESULT STDMETHODCALLTYPE SetMatrixTransform(const DXGI_MATRIX_3X2_F*) override { return S_OK; }
    HRESULT STDMETHODCALLTYPE GetMatrixTransform(DXGI_MATRIX_3X2_F*) override { return E_NOTIMPL; }

    // IDXGISwapChain3
    UINT STDMETHODCALLTYPE GetCurrentBackBufferIndex() override { return swapchain_.current_back_buffer_index(); }
    HRESULT STDMETHODCALLTYPE CheckColorSpaceSupport(DXGI_COLOR_SPACE_TYPE, UINT* s) override {
        if (s) *s = DXGI_SWAP_CHAIN_COLOR_SPACE_SUPPORT_FLAG_PRESENT;
        return S_OK;
    }
    HRESULT STDMETHODCALLTYPE SetColorSpace1(DXGI_COLOR_SPACE_TYPE) override { return S_OK; }
    HRESULT STDMETHODCALLTYPE ResizeBuffers1(UINT count, UINT w, UINT h, DXGI_FORMAT fmt, UINT flags, const UINT*,
                                             IUnknown* const*) override {
        return ResizeBuffers(count, w, h, fmt, flags);
    }

private:
    void register_buffers() {
        const std::uint32_t count = swapchain_.image_count();
        D3D12_RESOURCE_DESC desc{};
        desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        desc.Width = swapchain_.extent().width;
        desc.Height = swapchain_.extent().height;
        desc.DepthOrArraySize = 1;
        desc.MipLevels = 1;
        desc.Format = vk_to_dxgi_format(swapchain_.format());
        desc.SampleDesc.Count = 1;
        desc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
        desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
        buffers_.reserve(count);
        for (std::uint32_t i = 0; i < count; ++i) {
            auto* bb = new BackBufferResource(desc);
            context_.add_resource(bb->resource_id(), swapchain_.resource_binding(i), VK_IMAGE_LAYOUT_UNDEFINED);
            buffers_.push_back(bb);
        }
    }

    void unregister_buffers() {
        for (auto* bb : buffers_) {
            context_.remove_resource(bb->resource_id());
            bb->Release();
        }
        buffers_.clear();
    }

    std::atomic<ULONG> refs_{1};
    leonida::VulkanContext& context_;
    leonida::VulkanSwapchain swapchain_;
    HWND hwnd_ = nullptr;
    std::vector<BackBufferResource*> buffers_;
};

class Factory final : public IDXGIFactory4 {
public:
    Factory(leonida::VulkanContext& context, PFN_vkGetInstanceProcAddr gipa, VkInstance instance,
            VkPhysicalDevice physical, PFN_vkGetDeviceProcAddr gdpa, VkQueue queue, std::uint32_t queue_family)
        : context_(context), gipa_(gipa), instance_(instance), physical_(physical), gdpa_(gdpa),
          queue_(queue), queue_family_(queue_family) {}

    ULONG STDMETHODCALLTYPE AddRef() override { return ++refs_; }
    ULONG STDMETHODCALLTYPE Release() override {
        const ULONG left = --refs_;
        if (!left) delete this;
        return left;
    }
    HRESULT STDMETHODCALLTYPE QueryInterface(REFIID riid, void** out) override {
        if (!out) return E_POINTER;
        *out = nullptr;
        if (riid == __uuidof(IUnknown) || riid == __uuidof(IDXGIObject) || riid == __uuidof(IDXGIFactory) ||
            riid == __uuidof(IDXGIFactory1) || riid == __uuidof(IDXGIFactory2) ||
            riid == __uuidof(IDXGIFactory3) || riid == __uuidof(IDXGIFactory4)) {
            *out = static_cast<IDXGIFactory4*>(this);
            AddRef();
            return S_OK;
        }
        return E_NOINTERFACE;
    }

    // IDXGIObject
    HRESULT STDMETHODCALLTYPE SetPrivateData(REFGUID, UINT, const void*) override { return S_OK; }
    HRESULT STDMETHODCALLTYPE SetPrivateDataInterface(REFGUID, const IUnknown*) override { return S_OK; }
    HRESULT STDMETHODCALLTYPE GetPrivateData(REFGUID, UINT*, void*) override { return DXGI_ERROR_NOT_FOUND; }
    HRESULT STDMETHODCALLTYPE GetParent(REFIID, void**) override { return E_NOTIMPL; }

    // IDXGIFactory
    HRESULT STDMETHODCALLTYPE EnumAdapters(UINT, IDXGIAdapter**) override { return DXGI_ERROR_NOT_FOUND; }
    HRESULT STDMETHODCALLTYPE MakeWindowAssociation(HWND hwnd, UINT) override { assoc_hwnd_ = hwnd; return S_OK; }
    HRESULT STDMETHODCALLTYPE GetWindowAssociation(HWND* hwnd) override {
        if (!hwnd) return E_INVALIDARG;
        *hwnd = assoc_hwnd_;
        return S_OK;
    }
    HRESULT STDMETHODCALLTYPE CreateSwapChain(IUnknown*, DXGI_SWAP_CHAIN_DESC*, IDXGISwapChain**) override {
        stub("IDXGIFactory::CreateSwapChain");
        return E_NOTIMPL;
    }
    HRESULT STDMETHODCALLTYPE CreateSoftwareAdapter(HMODULE, IDXGIAdapter**) override { return E_NOTIMPL; }

    // IDXGIFactory1
    HRESULT STDMETHODCALLTYPE EnumAdapters1(UINT, IDXGIAdapter1**) override { return DXGI_ERROR_NOT_FOUND; }
    BOOL STDMETHODCALLTYPE IsCurrent() override { return TRUE; }

    // IDXGIFactory2
    BOOL STDMETHODCALLTYPE IsWindowedStereoEnabled() override { return FALSE; }
    HRESULT STDMETHODCALLTYPE CreateSwapChainForHwnd(IUnknown*, HWND hwnd, const DXGI_SWAP_CHAIN_DESC1* desc,
                                                     const DXGI_SWAP_CHAIN_FULLSCREEN_DESC*, IDXGIOutput*,
                                                     IDXGISwapChain1** out) override {
        if (!out) return E_POINTER;
        *out = nullptr;
        if (!hwnd || !desc) return E_INVALIDARG;
        auto* sc = new SwapChain(context_, gipa_, instance_, physical_, gdpa_, queue_, queue_family_);
        if (!sc->init(hwnd, *desc)) {
            sc->Release();
            return E_FAIL;
        }
        *out = static_cast<IDXGISwapChain1*>(sc);
        return S_OK;
    }
    HRESULT STDMETHODCALLTYPE CreateSwapChainForCoreWindow(IUnknown*, IUnknown*, const DXGI_SWAP_CHAIN_DESC1*,
                                                           IDXGIOutput*, IDXGISwapChain1**) override {
        return E_NOTIMPL;
    }
    HRESULT STDMETHODCALLTYPE GetSharedResourceAdapterLuid(HANDLE, LUID*) override { return E_NOTIMPL; }
    HRESULT STDMETHODCALLTYPE RegisterStereoStatusWindow(HWND, UINT, DWORD*) override { return E_NOTIMPL; }
    HRESULT STDMETHODCALLTYPE RegisterStereoStatusEvent(HANDLE, DWORD*) override { return E_NOTIMPL; }
    void STDMETHODCALLTYPE UnregisterStereoStatus(DWORD) override {}
    HRESULT STDMETHODCALLTYPE RegisterOcclusionStatusWindow(HWND, UINT, DWORD*) override { return E_NOTIMPL; }
    HRESULT STDMETHODCALLTYPE RegisterOcclusionStatusEvent(HANDLE, DWORD*) override { return E_NOTIMPL; }
    void STDMETHODCALLTYPE UnregisterOcclusionStatus(DWORD) override {}
    HRESULT STDMETHODCALLTYPE CreateSwapChainForComposition(IUnknown*, const DXGI_SWAP_CHAIN_DESC1*, IDXGIOutput*,
                                                            IDXGISwapChain1**) override {
        return E_NOTIMPL;
    }

    // IDXGIFactory3
    UINT STDMETHODCALLTYPE GetCreationFlags() override { return 0; }

    // IDXGIFactory4
    HRESULT STDMETHODCALLTYPE EnumAdapterByLuid(LUID, REFIID, void**) override { return DXGI_ERROR_NOT_FOUND; }
    HRESULT STDMETHODCALLTYPE EnumWarpAdapter(REFIID, void**) override { return DXGI_ERROR_NOT_FOUND; }

private:
    std::atomic<ULONG> refs_{1};
    leonida::VulkanContext& context_;
    PFN_vkGetInstanceProcAddr gipa_;
    VkInstance instance_;
    VkPhysicalDevice physical_;
    PFN_vkGetDeviceProcAddr gdpa_;
    VkQueue queue_;
    std::uint32_t queue_family_;
    HWND assoc_hwnd_ = nullptr;
};

}  // namespace

HRESULT create_factory(leonida::VulkanContext& context, PFN_vkGetInstanceProcAddr gipa, VkInstance instance,
                       VkPhysicalDevice physical, PFN_vkGetDeviceProcAddr gdpa, VkQueue queue,
                       std::uint32_t queue_family, REFIID riid, void** out) {
    if (!out) return E_POINTER;
    *out = nullptr;
    auto* factory = new Factory(context, gipa, instance, physical, gdpa, queue, queue_family);
    const HRESULT hr = factory->QueryInterface(riid, out);
    factory->Release();
    return hr;
}

}  // namespace rt::dxgi
