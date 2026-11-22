#include <d3d12.h>
#include <windows.h>
#include <cstdint>
#include <cstdio>

#include "../shims/d3d12_com.h"

#define CHECK(cond)                                                                                                    \
    do {                                                                                                               \
        if (!(cond)) {                                                                                                 \
            std::fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #cond);                                               \
            return 1;                                                                                                  \
        }                                                                                                              \
    } while (0)

int main() {
    ID3D12Device* dev = nullptr;
    if (FAILED(rt::d3d12::create_vulkan_device_standalone(IID_PPV_ARGS(&dev))) || !dev) {
        std::fprintf(stderr, "SKIP: no Vulkan device available\n");
        return 0;
    }

    D3D12_DESCRIPTOR_HEAP_DESC heap_desc{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 3,
                                         D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, 0};
    ID3D12DescriptorHeap* heap = nullptr;
    CHECK(SUCCEEDED(dev->CreateDescriptorHeap(&heap_desc, IID_PPV_ARGS(&heap))) && heap);

    const D3D12_CPU_DESCRIPTOR_HANDLE start = heap->GetCPUDescriptorHandleForHeapStart();
    const UINT inc = dev->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    CHECK(start.ptr != 0 && inc >= 32);

    const D3D12_CPU_DESCRIPTOR_HANDLE cbv0{start.ptr};
    const D3D12_CPU_DESCRIPTOR_HANDLE srv0{start.ptr + inc};
    const D3D12_CPU_DESCRIPTOR_HANDLE uav0{start.ptr + 2 * inc};

    D3D12_HEAP_PROPERTIES default_heap{D3D12_HEAP_TYPE_DEFAULT};
    D3D12_RESOURCE_DESC buf_desc{D3D12_RESOURCE_DIMENSION_BUFFER, 0, 1024, 1, 1, 1, DXGI_FORMAT_UNKNOWN, {1, 0},
                                 D3D12_TEXTURE_LAYOUT_ROW_MAJOR, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS};
    ID3D12Resource* buf = nullptr;
    CHECK(SUCCEEDED(dev->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &buf_desc,
                                                 D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&buf))));

    D3D12_RESOURCE_DESC tex_desc{D3D12_RESOURCE_DIMENSION_TEXTURE2D, 0, 64, 64, 1, 1, DXGI_FORMAT_R8G8B8A8_UNORM,
                                 {1, 0}, D3D12_TEXTURE_LAYOUT_UNKNOWN, D3D12_RESOURCE_FLAG_NONE};
    ID3D12Resource* tex = nullptr;
    CHECK(SUCCEEDED(dev->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &tex_desc,
                                                 D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&tex))));

    D3D12_CONSTANT_BUFFER_VIEW_DESC cbv_desc{0x12340000ULL, 256};
    dev->CreateConstantBufferView(&cbv_desc, cbv0);

    D3D12_SHADER_RESOURCE_VIEW_DESC srv_desc{};
    srv_desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    srv_desc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
    srv_desc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    srv_desc.Texture2D.MipLevels = 1;
    dev->CreateShaderResourceView(tex, &srv_desc, srv0);

    D3D12_UNORDERED_ACCESS_VIEW_DESC uav_desc{};
    uav_desc.Format = DXGI_FORMAT_R32_UINT;
    uav_desc.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
    uav_desc.Buffer.NumElements = 64;
    dev->CreateUnorderedAccessView(buf, nullptr, &uav_desc, uav0);

    CHECK(*reinterpret_cast<const std::uint32_t*>(cbv0.ptr) == 1);  // CBV
    CHECK(*reinterpret_cast<const std::uint32_t*>(srv0.ptr) != 0);  // SRV (GFX10 T# written for texture)
    CHECK(*reinterpret_cast<const std::uint32_t*>(uav0.ptr) == 3);  // UAV

    tex->Release();
    buf->Release();
    heap->Release();
    dev->Release();
    std::printf("PASS: d3d12_shader_views_test (CBV, SRV, UAV)\n");
    return 0;
}
