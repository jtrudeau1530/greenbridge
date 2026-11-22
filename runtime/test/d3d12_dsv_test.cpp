// Depth-stencil view validation: CreateDepthStencilView must refuse a view that names a mip or layer outside the
// resource, a view dimension that mismatches the texture's dimension or sample count, a resource that was not
// created for depth, or a view format outside the resource's depth family. On the PC runtime these writes put a
// null TargetView in the heap slot; an accepted view writes a non-null one.

#include "../shims/d3d12_com.h"

#include <cstdio>
#include <cstring>

namespace {

int fail(const char* msg, HRESULT hr = S_OK) {
    std::fprintf(stderr, "FAIL: %s (hr=0x%08X)\n", msg, static_cast<unsigned>(hr));
    return 1;
}

D3D12_HEAP_PROPERTIES default_heap() {
    D3D12_HEAP_PROPERTIES hp{};
    hp.Type = D3D12_HEAP_TYPE_DEFAULT;
    return hp;
}

}  // namespace

int main() {
    ID3D12Device* dev = nullptr;
    HRESULT hr = rt::d3d12::create_vulkan_device_standalone(IID_PPV_ARGS(&dev));
    if (FAILED(hr) || !dev) {
        std::printf("SKIP: no Vulkan device (hr=0x%08X)\n", static_cast<unsigned>(hr));
        return 77;
    }

    D3D12_DESCRIPTOR_HEAP_DESC dsv_heap_desc{};
    dsv_heap_desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_DSV;
    dsv_heap_desc.NumDescriptors = 16;
    ID3D12DescriptorHeap* dsv_heap = nullptr;
    hr = dev->CreateDescriptorHeap(&dsv_heap_desc, IID_PPV_ARGS(&dsv_heap));
    if (FAILED(hr) || !dsv_heap) return fail("CreateDescriptorHeap(DSV)", hr);
    const UINT inc = dev->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_DSV);
    const D3D12_CPU_DESCRIPTOR_HANDLE dsv_cpu = dsv_heap->GetCPUDescriptorHandleForHeapStart();
    const auto slot = [&](UINT i) { return D3D12_CPU_DESCRIPTOR_HANDLE{dsv_cpu.ptr + inc * i}; };
    const auto is_null = [&](UINT i) {
        return *reinterpret_cast<const std::uint64_t*>(slot(i).ptr) == 0;
    };

    const D3D12_HEAP_PROPERTIES hp = default_heap();
    D3D12_RESOURCE_DESC depth_desc{};
    depth_desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    depth_desc.Width = 64;
    depth_desc.Height = 64;
    depth_desc.DepthOrArraySize = 4;
    depth_desc.MipLevels = 3;
    depth_desc.Format = DXGI_FORMAT_D32_FLOAT;
    depth_desc.SampleDesc = {1, 0};
    depth_desc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
    depth_desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;

    D3D12_RESOURCE_DESC msaa_desc = depth_desc;
    msaa_desc.DepthOrArraySize = 2;
    msaa_desc.MipLevels = 1;
    msaa_desc.SampleDesc = {4, 0};

    D3D12_RESOURCE_DESC color_desc = depth_desc;
    color_desc.MipLevels = 1;
    color_desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    color_desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;

    D3D12_RESOURCE_DESC typeless_desc = depth_desc;
    typeless_desc.Format = DXGI_FORMAT_R32_TYPELESS;

    ID3D12Resource* depth = nullptr;
    ID3D12Resource* msaa = nullptr;
    ID3D12Resource* color = nullptr;
    ID3D12Resource* typeless = nullptr;
    hr = dev->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &depth_desc, D3D12_RESOURCE_STATE_COMMON,
                                      nullptr, IID_PPV_ARGS(&depth));
    if (FAILED(hr) || !depth) return fail("CreateCommittedResource(depth)", hr);
    hr = dev->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &msaa_desc, D3D12_RESOURCE_STATE_COMMON,
                                      nullptr, IID_PPV_ARGS(&msaa));
    if (FAILED(hr) || !msaa) return fail("CreateCommittedResource(msaa)", hr);
    hr = dev->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &color_desc, D3D12_RESOURCE_STATE_COMMON,
                                      nullptr, IID_PPV_ARGS(&color));
    if (FAILED(hr) || !color) return fail("CreateCommittedResource(color)", hr);
    hr = dev->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &typeless_desc, D3D12_RESOURCE_STATE_COMMON,
                                      nullptr, IID_PPV_ARGS(&typeless));
    if (FAILED(hr) || !typeless) return fail("CreateCommittedResource(typeless)", hr);

    const auto dsv = [&](DXGI_FORMAT format, D3D12_DSV_DIMENSION dim) {
        D3D12_DEPTH_STENCIL_VIEW_DESC d{};
        d.Format = format;
        d.ViewDimension = dim;
        return d;
    };

    // Valid views must write a non-null TargetView.
    D3D12_DEPTH_STENCIL_VIEW_DESC d0 = dsv(DXGI_FORMAT_D32_FLOAT, D3D12_DSV_DIMENSION_TEXTURE2D);
    d0.Texture2D.MipSlice = 1;
    dev->CreateDepthStencilView(depth, &d0, slot(0));
    if (is_null(0)) return fail("TEXTURE2D DSV mip 1 should be written");

    D3D12_DEPTH_STENCIL_VIEW_DESC d1 = dsv(DXGI_FORMAT_D32_FLOAT, D3D12_DSV_DIMENSION_TEXTURE2DARRAY);
    d1.Texture2DArray.MipSlice = 0;
    d1.Texture2DArray.FirstArraySlice = 1;
    d1.Texture2DArray.ArraySize = 2;
    dev->CreateDepthStencilView(depth, &d1, slot(1));
    if (is_null(1)) return fail("TEXTURE2DARRAY DSV slices 1-2 should be written");

    dev->CreateDepthStencilView(depth, nullptr, slot(2));
    if (is_null(2)) return fail("default DSV should be written");

    D3D12_DEPTH_STENCIL_VIEW_DESC d3 = dsv(DXGI_FORMAT_D32_FLOAT, D3D12_DSV_DIMENSION_TEXTURE2DMS);
    dev->CreateDepthStencilView(msaa, &d3, slot(3));
    if (is_null(3)) return fail("TEXTURE2DMS DSV should be written");

    D3D12_DEPTH_STENCIL_VIEW_DESC d4 = dsv(DXGI_FORMAT_D32_FLOAT, D3D12_DSV_DIMENSION_TEXTURE2D);
    d4.Texture2D.MipSlice = 0;
    dev->CreateDepthStencilView(typeless, &d4, slot(4));
    if (is_null(4)) return fail("D32_FLOAT view of R32_TYPELESS should be written");

    // Refused views must leave a null TargetView.
    D3D12_DEPTH_STENCIL_VIEW_DESC d5 = dsv(DXGI_FORMAT_D32_FLOAT, D3D12_DSV_DIMENSION_TEXTURE2D);
    d5.Texture2D.MipSlice = 3;  // MipLevels is 3
    dev->CreateDepthStencilView(depth, &d5, slot(5));
    if (!is_null(5)) return fail("out-of-bounds mip DSV should be null");

    D3D12_DEPTH_STENCIL_VIEW_DESC d6 = dsv(DXGI_FORMAT_D32_FLOAT, D3D12_DSV_DIMENSION_TEXTURE2DARRAY);
    d6.Texture2DArray.FirstArraySlice = 4;  // DepthOrArraySize is 4
    d6.Texture2DArray.ArraySize = 1;
    dev->CreateDepthStencilView(depth, &d6, slot(6));
    if (!is_null(6)) return fail("out-of-bounds slice DSV should be null");

    D3D12_DEPTH_STENCIL_VIEW_DESC d7 = dsv(DXGI_FORMAT_D32_FLOAT, D3D12_DSV_DIMENSION_TEXTURE2DARRAY);
    d7.Texture2DArray.ArraySize = 5;
    dev->CreateDepthStencilView(depth, &d7, slot(7));
    if (!is_null(7)) return fail("overrunning slice count DSV should be null");

    D3D12_DEPTH_STENCIL_VIEW_DESC d8 = dsv(DXGI_FORMAT_D32_FLOAT, D3D12_DSV_DIMENSION_TEXTURE2DMS);
    dev->CreateDepthStencilView(depth, &d8, slot(8));
    if (!is_null(8)) return fail("TEXTURE2DMS on non-MSAA resource should be null");

    D3D12_DEPTH_STENCIL_VIEW_DESC d9 = dsv(DXGI_FORMAT_D32_FLOAT, D3D12_DSV_DIMENSION_TEXTURE2D);
    dev->CreateDepthStencilView(msaa, &d9, slot(9));
    if (!is_null(9)) return fail("non-MS view on MSAA resource should be null");

    D3D12_DEPTH_STENCIL_VIEW_DESC d10 = dsv(DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_DSV_DIMENSION_TEXTURE2D);
    dev->CreateDepthStencilView(color, &d10, slot(10));
    if (!is_null(10)) return fail("DSV on a render-target texture should be null");

    D3D12_DEPTH_STENCIL_VIEW_DESC d11 = dsv(DXGI_FORMAT_D24_UNORM_S8_UINT, D3D12_DSV_DIMENSION_TEXTURE2D);
    dev->CreateDepthStencilView(depth, &d11, slot(11));
    if (!is_null(11)) return fail("D24S8 view of a D32 resource should be null");

    D3D12_DEPTH_STENCIL_VIEW_DESC d12 = dsv(DXGI_FORMAT_D24_UNORM_S8_UINT, D3D12_DSV_DIMENSION_TEXTURE2D);
    dev->CreateDepthStencilView(typeless, &d12, slot(12));
    if (!is_null(12)) return fail("D24S8 view of R32_TYPELESS should be null");

    D3D12_DEPTH_STENCIL_VIEW_DESC d13 = dsv(DXGI_FORMAT_D32_FLOAT, D3D12_DSV_DIMENSION_TEXTURE2DMSARRAY);
    d13.Texture2DMSArray.FirstArraySlice = 2;  // DepthOrArraySize is 2
    d13.Texture2DMSArray.ArraySize = 1;
    dev->CreateDepthStencilView(msaa, &d13, slot(13));
    if (!is_null(13)) return fail("out-of-bounds MSAA slice DSV should be null");

    D3D12_DEPTH_STENCIL_VIEW_DESC d14 = dsv(DXGI_FORMAT_D32_FLOAT, D3D12_DSV_DIMENSION_TEXTURE1D);
    dev->CreateDepthStencilView(depth, &d14, slot(14));
    if (!is_null(14)) return fail("TEXTURE1D view of a 2D resource should be null");

    typeless->Release();
    color->Release();
    msaa->Release();
    depth->Release();
    dsv_heap->Release();
    dev->Release();

    std::printf("PASS: d3d12_dsv_test (CreateDepthStencilView validation)\n");
    return 0;
}
