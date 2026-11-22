// The Vulkan-backed ID3D12Device behind the Xbox-ABI device proxy, end to end, on a GPU. Everything goes through
// the proxy the way the game will call it: device methods (forwarded slots), descriptor heaps from slot 14 with
// start handles read in the Xbox shape (RAX), a texture from slot 76 with the Xbox layout word, committed
// resources, views, command lists, a queue and a fence. Results are read back through D3D12 itself
// (CopyTextureRegion into a readback heap, then Map). GetCopyableFootprints, GetResourceAllocationInfo and
// GetResourceTiling are compared with the host PC runtime's, and so are copies (d3d12_copy_compare.h).
// WriteToSubresource/ReadFromSubresource run on devices of their own, once per copy path. No shaders or game data.
// Set LEONIDA_VK_VALIDATION=1 to run under VK_LAYER_KHRONOS_validation; any validation error fails the test.
#include "vk_test_device.h"
#include "../shims/d3d12_com.h"
#include "../shims/d3d12_xs_device.h"
#include "d3d12_copy_compare.h"  // D3D-46: copies compared with the host runtime

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#define EXPECT(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); return 1; } \
} while (0)

namespace {

// The Xbox call shapes measured in the prior art (docs/d3d12_vulkan.md section 7).
using XboxHeapHandleFn = std::uint64_t(WINAPI*)(void* heap);  // heap slots 9/10: 8-byte handle in RAX
using XboxCreateResourceFn = HRESULT(WINAPI*)(void* device, const D3D12_HEAP_PROPERTIES*, D3D12_HEAP_FLAGS,
                                              const D3D12_RESOURCE_DESC*, D3D12_RESOURCE_STATES,
                                              const D3D12_CLEAR_VALUE*, const IID*, void**);  // device slot 76

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

void* slot(void* object, unsigned index) { return (*reinterpret_cast<void***>(object))[index]; }

D3D12_RESOURCE_DESC texture_desc(UINT width, UINT height, DXGI_FORMAT format, D3D12_RESOURCE_FLAGS flags,
                                 UINT16 mips = 1, UINT16 layers = 1) {
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    desc.Width = width;
    desc.Height = height;
    desc.DepthOrArraySize = layers;
    desc.MipLevels = mips;
    desc.Format = format;
    desc.SampleDesc.Count = 1;
    desc.Flags = flags;
    return desc;
}

D3D12_RESOURCE_DESC buffer_desc(UINT64 bytes) {
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Width = bytes;
    desc.Height = 1;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    desc.SampleDesc.Count = 1;
    desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    return desc;
}

D3D12_RESOURCE_DESC volume_desc(UINT16 mips) {
    auto desc = texture_desc(8, 6, DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_RESOURCE_FLAG_NONE, mips, 16);
    desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE3D;
    return desc;
}

D3D12_TEXTURE_COPY_LOCATION subresource(ID3D12Resource* resource, UINT index) {
    D3D12_TEXTURE_COPY_LOCATION location{};
    location.pResource = resource;
    location.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    location.SubresourceIndex = index;
    return location;
}

D3D12_TEXTURE_COPY_LOCATION footprint(ID3D12Resource* buffer, const D3D12_PLACED_SUBRESOURCE_FOOTPRINT& placed) {
    D3D12_TEXTURE_COPY_LOCATION location{};
    location.pResource = buffer;
    location.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    location.PlacedFootprint = placed;
    return location;
}

// Places one subresource of `resource` in a buffer at the next aligned offset after `cursor`.
D3D12_PLACED_SUBRESOURCE_FOOTPRINT place(ID3D12Device* device, ID3D12Resource* resource, UINT index,
                                         UINT64& cursor) {
    const D3D12_RESOURCE_DESC desc = resource->GetDesc();
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT layout{};
    UINT64 total = 0;
    const UINT64 base = (cursor + D3D12_TEXTURE_DATA_PLACEMENT_ALIGNMENT - 1) / D3D12_TEXTURE_DATA_PLACEMENT_ALIGNMENT *
                        D3D12_TEXTURE_DATA_PLACEMENT_ALIGNMENT;
    device->GetCopyableFootprints(&desc, index, 1, base, &layout, nullptr, nullptr, &total);
    cursor = base + total;
    return layout;
}

std::array<std::uint8_t, 4> pattern(UINT x, UINT y) {
    return {std::uint8_t(x * 16 + y), std::uint8_t(255 - y * 16), std::uint8_t(x * y), 255};
}

D3D12_RESOURCE_BARRIER transition(ID3D12Resource* resource, D3D12_RESOURCE_STATES before,
                                  D3D12_RESOURCE_STATES after) {
    D3D12_RESOURCE_BARRIER barrier{};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Transition.pResource = resource;
    barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    barrier.Transition.StateBefore = before;
    barrier.Transition.StateAfter = after;
    return barrier;
}

std::uint8_t texel_byte(UINT x, UINT y, UINT z, UINT i) { return std::uint8_t(x * 31 + y * 7 + z * 61 + i * 3 + 1); }

// WriteToSubresource/ReadFromSubresource on `device`. Textures in a CPU-visible custom heap copy on the host when the
// device has VK_EXT_host_image_copy, through a staging submit when it doesn't; main runs this once per path. Covers
// zero-filled first reads, boxes with pitches unlike the texture's, a mip, a volume, BC1 blocks, a default-heap
// texture (always staged), the rejected calls, and both directions between the CPU copies and GPU copies.
int cpu_copies(ID3D12Device* device, bool bc, const char* path) {
    const D3D12_HEAP_PROPERTIES cpu_heap{D3D12_HEAP_TYPE_CUSTOM, D3D12_CPU_PAGE_PROPERTY_WRITE_BACK, D3D12_MEMORY_POOL_L0};
    const D3D12_HEAP_PROPERTIES default_heap{D3D12_HEAP_TYPE_DEFAULT};
    const D3D12_HEAP_PROPERTIES upload_heap{D3D12_HEAP_TYPE_UPLOAD};
    const D3D12_HEAP_PROPERTIES readback_heap{D3D12_HEAP_TYPE_READBACK};
    const auto create = [&](const D3D12_HEAP_PROPERTIES& heap, const D3D12_RESOURCE_DESC& desc,
                            D3D12_RESOURCE_STATES state, Ref<ID3D12Resource>& out) {
        return device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, state, nullptr, IID_PPV_ARGS(&out.p));
    };
    unsigned bad = 0;
    const auto expect_bytes = [&](const char* what, UINT x, UINT y, UINT z, const std::uint8_t* got,
                                  const std::array<std::uint8_t, 4>& want) {
        if (std::memcmp(got, want.data(), 4) == 0) return;
        if (bad++ < 8)
            std::fprintf(stderr, "%s, %s (%u,%u,%u): got %02x %02x %02x %02x, want %02x %02x %02x %02x\n", path, what,
                         x, y, z, got[0], got[1], got[2], got[3], want[0], want[1], want[2], want[3]);
    };
    const auto texel = [](UINT x, UINT y, UINT z) {
        return std::array<std::uint8_t, 4>{texel_byte(x, y, z, 0), texel_byte(x, y, z, 1), texel_byte(x, y, z, 2),
                                           texel_byte(x, y, z, 3)};
    };
    constexpr std::array<std::uint8_t, 4> zero{};

    Ref<ID3D12Resource> tex, vol, bc1, gpu_only, buffer, upload, readback;
    EXPECT(SUCCEEDED(create(cpu_heap, texture_desc(8, 8, DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_RESOURCE_FLAG_NONE, 2),
                            D3D12_RESOURCE_STATE_COMMON, tex)));
    D3D12_RESOURCE_DESC volume = texture_desc(4, 4, DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_RESOURCE_FLAG_NONE, 1, 4);
    volume.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE3D;
    EXPECT(SUCCEEDED(create(cpu_heap, volume, D3D12_RESOURCE_STATE_COMMON, vol)));
    if (bc)
        EXPECT(SUCCEEDED(create(cpu_heap, texture_desc(8, 8, DXGI_FORMAT_BC1_UNORM, D3D12_RESOURCE_FLAG_NONE),
                                D3D12_RESOURCE_STATE_COMMON, bc1)));
    EXPECT(SUCCEEDED(create(default_heap, texture_desc(4, 4, DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_RESOURCE_FLAG_NONE),
                            D3D12_RESOURCE_STATE_COPY_SOURCE, gpu_only)));
    EXPECT(SUCCEEDED(create(cpu_heap, buffer_desc(4096), D3D12_RESOURCE_STATE_COMMON, buffer)));
    EXPECT(SUCCEEDED(create(upload_heap, buffer_desc(4096), D3D12_RESOURCE_STATE_GENERIC_READ, upload)));
    EXPECT(SUCCEEDED(create(readback_heap, buffer_desc(4096), D3D12_RESOURCE_STATE_COPY_DEST, readback)));

    // A texture in a CPU-visible heap maps only without a pointer; a default-heap texture not at all.
    void* pointer = nullptr;
    EXPECT(tex->Map(0, nullptr, nullptr) == S_OK);
    EXPECT(tex->Map(1, nullptr, nullptr) == S_OK);
    EXPECT(tex->Map(0, nullptr, &pointer) == E_INVALIDARG);
    EXPECT(tex->Map(2, nullptr, nullptr) == E_INVALIDARG);
    EXPECT(gpu_only->Map(0, nullptr, nullptr) == E_INVALIDARG);
    tex->Unmap(1, nullptr);

    // A new texture reads back zero-filled.
    std::vector<std::uint8_t> got(1024, 0xcd);
    EXPECT(SUCCEEDED(tex->ReadFromSubresource(got.data(), 32, 0, 0, nullptr)));
    for (UINT y = 0; y < 8; ++y)
        for (UINT x = 0; x < 8; ++x) expect_bytes("new texture", x, y, 0, &got[y * 32 + x * 4], zero);

    // A 5x4 box from rows 100 bytes apart, read back whole at a 40-byte pitch.
    std::vector<std::uint8_t> src(1024, 0);
    const D3D12_BOX box{2, 1, 0, 7, 5, 1};
    for (UINT y = 0; y < 4; ++y)
        for (UINT x = 0; x < 5; ++x) std::memcpy(&src[y * 100 + x * 4], texel(x + 2, y + 1, 0).data(), 4);
    EXPECT(SUCCEEDED(tex->WriteToSubresource(0, &box, src.data(), 100, 0)));
    std::fill(got.begin(), got.end(), std::uint8_t(0xcd));
    EXPECT(SUCCEEDED(tex->ReadFromSubresource(got.data(), 40, 0, 0, nullptr)));
    for (UINT y = 0; y < 8; ++y)
        for (UINT x = 0; x < 8; ++x) {
            const bool inside = x >= 2 && x < 7 && y >= 1 && y < 5;
            expect_bytes("boxed write", x, y, 0, &got[y * 40 + x * 4], inside ? texel(x, y, 0) : zero);
        }

    // Mip 1 (4x4) written whole, read back as a box.
    for (UINT y = 0; y < 4; ++y)
        for (UINT x = 0; x < 4; ++x) std::memcpy(&src[y * 16 + x * 4], texel(x, y, 1).data(), 4);
    EXPECT(SUCCEEDED(tex->WriteToSubresource(1, nullptr, src.data(), 16, 0)));
    const D3D12_BOX mip_box{1, 1, 0, 3, 4, 1};
    std::fill(got.begin(), got.end(), std::uint8_t(0xcd));
    EXPECT(SUCCEEDED(tex->ReadFromSubresource(got.data(), 8, 0, 1, &mip_box)));
    for (UINT y = 0; y < 3; ++y)
        for (UINT x = 0; x < 2; ++x) expect_bytes("mip 1", x + 1, y + 1, 1, &got[y * 8 + x * 4], texel(x + 1, y + 1, 1));

    // A volume: three slices of a 3x4 box, rows 12 and slices 64 bytes apart; read back whole at other pitches.
    std::fill(src.begin(), src.end(), std::uint8_t(0));
    const D3D12_BOX volume_box{1, 0, 1, 4, 4, 4};
    for (UINT z = 0; z < 3; ++z)
        for (UINT y = 0; y < 4; ++y)
            for (UINT x = 0; x < 3; ++x) std::memcpy(&src[z * 64 + y * 12 + x * 4], texel(x + 1, y, z + 1).data(), 4);
    EXPECT(SUCCEEDED(vol->WriteToSubresource(0, &volume_box, src.data(), 12, 64)));
    std::fill(got.begin(), got.end(), std::uint8_t(0xcd));
    EXPECT(SUCCEEDED(vol->ReadFromSubresource(got.data(), 20, 96, 0, nullptr)));
    for (UINT z = 0; z < 4; ++z)
        for (UINT y = 0; y < 4; ++y)
            for (UINT x = 0; x < 4; ++x)
                expect_bytes("volume", x, y, z, &got[z * 96 + y * 20 + x * 4], x >= 1 && z >= 1 ? texel(x, y, z) : zero);

    // BC1: 2x2 blocks of 8 bytes written whole, then one block replaced through a box.
    if (bc1.p) {
        std::uint8_t blocks[32];
        for (UINT i = 0; i < 32; ++i) blocks[i] = std::uint8_t(0x40 + i);
        EXPECT(SUCCEEDED(bc1->WriteToSubresource(0, nullptr, blocks, 16, 0)));
        const D3D12_BOX corner{4, 4, 0, 8, 8, 1};
        const std::uint8_t replaced[8] = {0xe0, 0xe1, 0xe2, 0xe3, 0xe4, 0xe5, 0xe6, 0xe7};
        EXPECT(SUCCEEDED(bc1->WriteToSubresource(0, &corner, replaced, 8, 0)));
        std::memcpy(blocks + 24, replaced, 8);
        std::uint8_t back[64] = {};
        EXPECT(SUCCEEDED(bc1->ReadFromSubresource(back, 32, 0, 0, nullptr)));
        for (UINT row = 0; row < 2; ++row)
            if (std::memcmp(back + row * 32, blocks + row * 16, 16) != 0 && bad++ < 8)
                std::fprintf(stderr, "%s, BC1 block row %u differs\n", path, row);
        const D3D12_BOX cut{2, 0, 0, 8, 4, 1};  // starts inside a block
        EXPECT(bc1->WriteToSubresource(0, &cut, blocks, 16, 0) == E_INVALIDARG);
    }

    // Calls D3D12 rejects.
    const D3D12_BOX outside{0, 0, 0, 9, 1, 1}, empty{2, 2, 0, 2, 3, 1};
    EXPECT(tex->WriteToSubresource(0, &outside, src.data(), 100, 0) == E_INVALIDARG);
    EXPECT(tex->WriteToSubresource(0, &empty, src.data(), 100, 0) == E_INVALIDARG);
    EXPECT(tex->WriteToSubresource(2, nullptr, src.data(), 100, 0) == E_INVALIDARG);  // 2 mips, 1 layer
    EXPECT(tex->WriteToSubresource(0, nullptr, nullptr, 32, 0) == E_INVALIDARG);
    EXPECT(tex->ReadFromSubresource(nullptr, 32, 0, 0, nullptr) == E_INVALIDARG);
    EXPECT(tex->WriteToSubresource(0, nullptr, src.data(), 16, 0) == E_INVALIDARG);  // 32-byte rows 16 apart
    EXPECT(vol->WriteToSubresource(0, nullptr, src.data(), 16, 32) == E_INVALIDARG);  // 64-byte slices 32 apart
    EXPECT(buffer->WriteToSubresource(0, nullptr, src.data(), 16, 0) == E_INVALIDARG);

    // A default-heap texture: D3D12 wants a CPU-visible heap, but the copy still works (and is logged).
    for (UINT y = 0; y < 4; ++y)
        for (UINT x = 0; x < 4; ++x) std::memcpy(&src[y * 16 + x * 4], texel(x, y, 7).data(), 4);
    EXPECT(SUCCEEDED(gpu_only->WriteToSubresource(0, nullptr, src.data(), 16, 0)));

    // GPU copies see the CPU's writes (mip 1, the default-heap texture), and the CPU sees a GPU copy's (a 4x4
    // upload into the bottom left of mip 0, which is in COMMON and promoted).
    D3D12_COMMAND_QUEUE_DESC queue_desc{};
    Ref<ID3D12CommandQueue> queue;
    Ref<ID3D12CommandAllocator> allocator;
    Ref<ID3D12GraphicsCommandList> list;
    Ref<ID3D12Fence> fence;
    EXPECT(SUCCEEDED(device->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&queue.p))));
    EXPECT(SUCCEEDED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator.p))));
    EXPECT(SUCCEEDED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.p, nullptr,
                                               IID_PPV_ARGS(&list.p))));
    EXPECT(SUCCEEDED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence.p))));
    std::uint8_t* up = nullptr;
    std::uint8_t* down = nullptr;
    EXPECT(SUCCEEDED(upload->Map(0, nullptr, reinterpret_cast<void**>(&up))));
    EXPECT(SUCCEEDED(readback->Map(0, nullptr, reinterpret_cast<void**>(&down))));
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT square{0, {DXGI_FORMAT_R8G8B8A8_UNORM, 4, 4, 1, 256}};
    for (UINT y = 0; y < 4; ++y)
        for (UINT x = 0; x < 4; ++x) std::memcpy(up + y * 256 + x * 4, texel(x, y + 4, 0).data(), 4);
    std::memset(down, 0xcd, 4096);
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT second = square;
    second.Offset = 2048;
    {
        const auto dst = subresource(tex.p, 0);
        const auto in = footprint(upload.p, square);
        list->CopyTextureRegion(&dst, 0, 4, 0, &in, nullptr);
        const auto out = footprint(readback.p, square);
        const auto mip1 = subresource(tex.p, 1);
        list->CopyTextureRegion(&out, 0, 0, 0, &mip1, nullptr);
        const auto out2 = footprint(readback.p, second);
        const auto staged = subresource(gpu_only.p, 0);
        list->CopyTextureRegion(&out2, 0, 0, 0, &staged, nullptr);
    }
    EXPECT(list->Close() == S_OK);
    ID3D12CommandList* lists[] = {list.p};
    queue->ExecuteCommandLists(1, lists);
    EXPECT(rt::d3d12::last_execute_result(queue.p) == leonida::ReplayResult::success);
    EXPECT(SUCCEEDED(queue->Signal(fence.p, 1)) && SUCCEEDED(fence->SetEventOnCompletion(1, nullptr)));
    for (UINT y = 0; y < 4; ++y)
        for (UINT x = 0; x < 4; ++x) {
            expect_bytes("GPU copy of mip 1", x, y, 1, down + y * 256 + x * 4, texel(x, y, 1));
            expect_bytes("GPU copy of the default-heap texture", x, y, 7, down + 2048 + y * 256 + x * 4, texel(x, y, 7));
        }
    const D3D12_BOX bottom_left{0, 4, 0, 4, 8, 1};
    std::fill(got.begin(), got.end(), std::uint8_t(0xcd));
    EXPECT(SUCCEEDED(tex->ReadFromSubresource(got.data(), 16, 0, 0, &bottom_left)));
    for (UINT y = 0; y < 4; ++y)
        for (UINT x = 0; x < 4; ++x) expect_bytes("CPU read of a GPU upload", x, y + 4, 0, &got[y * 16 + x * 4], texel(x, y + 4, 0));
    upload->Unmap(0, nullptr);
    readback->Unmap(0, nullptr);
    EXPECT(!bad);
    std::printf("WriteToSubresource/ReadFromSubresource (%s): boxes, pitches, a mip, a volume%s, rejected calls and "
                "GPU copies both ways matched\n", path, bc1.p ? ", BC1" : "");
    return 0;
}

// The host PC runtime's D3D12 device to compare answers with: the default adapter, or the one LEONIDA_HOST_ADAPTER
// names ("warp", or part of an adapter's name); false if there is none.
bool create_host_device(Ref<ID3D12Device>& host) {
    copycmp::Ptr<ID3D12Device> device;
    std::string name;
    if (!copycmp::create_host(device, name)) return false;
    host.p = device.p;
    device.p = nullptr;
    return true;
}

// GetCopyableFootprints must match the PC runtime's exactly, since applications size and fill upload buffers from
// it. Returns the number of mismatches, or -1 when the host has no D3D12 device to compare with.
int compare_footprints(ID3D12Device* ours) {
    Ref<ID3D12Device> host;
    if (!create_host_device(host)) return -1;
    struct Case {
        D3D12_RESOURCE_DESC desc;
        UINT subresources;
    };
    const Case cases[] = {
        {volume_desc(5), 5},
        {buffer_desc(300), 1},
        {texture_desc(8, 8, DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_RESOURCE_FLAG_NONE, 4, 2), 8},
        {texture_desc(13, 7, DXGI_FORMAT_R16G16B16A16_FLOAT, D3D12_RESOURCE_FLAG_NONE, 4), 4},
        {texture_desc(10, 6, DXGI_FORMAT_BC1_UNORM, D3D12_RESOURCE_FLAG_NONE, 4), 4},
        {texture_desc(4, 4, DXGI_FORMAT_BC7_UNORM_SRGB, D3D12_RESOURCE_FLAG_NONE, 1, 3), 3},
        {texture_desc(16, 16, DXGI_FORMAT_BC5_UNORM, D3D12_RESOURCE_FLAG_NONE, 5), 5},
        {texture_desc(8, 8, DXGI_FORMAT_D32_FLOAT_S8X24_UINT, D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL, 2), 4},
        {texture_desc(8, 8, DXGI_FORMAT_D24_UNORM_S8_UINT, D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL, 2, 2), 8},
        {texture_desc(300, 5, DXGI_FORMAT_D32_FLOAT_S8X24_UINT, D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL), 2},
        {texture_desc(8, 8, DXGI_FORMAT_D32_FLOAT, D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL), 1},
        {texture_desc(8, 8, DXGI_FORMAT_D16_UNORM, D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL, 4), 4},
        {texture_desc(6, 6, DXGI_FORMAT_R32_TYPELESS, D3D12_RESOURCE_FLAG_NONE), 1},
        {texture_desc(16, 16, DXGI_FORMAT_R32G32_UINT, D3D12_RESOURCE_FLAG_NONE, 3), 3},
        {texture_desc(16, 16, DXGI_FORMAT_R32G32_SINT, D3D12_RESOURCE_FLAG_NONE, 3), 3},
        {texture_desc(16, 16, DXGI_FORMAT_R32G32_TYPELESS, D3D12_RESOURCE_FLAG_NONE, 3), 3},
        {texture_desc(16, 16, DXGI_FORMAT_R16G16B16A16_UINT, D3D12_RESOURCE_FLAG_NONE, 3), 3},
        {texture_desc(16, 16, DXGI_FORMAT_R16G16B16A16_SINT, D3D12_RESOURCE_FLAG_NONE, 3), 3},
        {texture_desc(16, 16, DXGI_FORMAT_R10G10B10A2_UINT, D3D12_RESOURCE_FLAG_NONE, 3), 3},
        {texture_desc(16, 16, DXGI_FORMAT_R16G16_TYPELESS, D3D12_RESOURCE_FLAG_NONE, 3), 3},
        {texture_desc(16, 16, DXGI_FORMAT_R16G16_UINT, D3D12_RESOURCE_FLAG_NONE, 3), 3},
        {texture_desc(16, 16, DXGI_FORMAT_R16G16_SINT, D3D12_RESOURCE_FLAG_NONE, 3), 3},
        {texture_desc(16, 16, DXGI_FORMAT_R8G8_TYPELESS, D3D12_RESOURCE_FLAG_NONE, 3), 3},
        {texture_desc(16, 16, DXGI_FORMAT_R8G8_UINT, D3D12_RESOURCE_FLAG_NONE, 3), 3},
        {texture_desc(16, 16, DXGI_FORMAT_R8G8_SINT, D3D12_RESOURCE_FLAG_NONE, 3), 3},
    };
    int mismatches = 0;
    for (const Case& c : cases) {
        for (const UINT64 base : {UINT64(0), UINT64(1024)}) {
            std::vector<D3D12_PLACED_SUBRESOURCE_FOOTPRINT> a(c.subresources), b(c.subresources);
            std::vector<UINT> rows_a(c.subresources), rows_b(c.subresources);
            std::vector<UINT64> size_a(c.subresources), size_b(c.subresources);
            UINT64 total_a = 0, total_b = 0;
            ours->GetCopyableFootprints(&c.desc, 0, c.subresources, base, a.data(), rows_a.data(), size_a.data(),
                                        &total_a);
            host->GetCopyableFootprints(&c.desc, 0, c.subresources, base, b.data(), rows_b.data(), size_b.data(),
                                        &total_b);
            for (UINT i = 0; i < c.subresources; ++i) {
                const auto &x = a[i], &y = b[i];
                if (x.Offset == y.Offset && x.Footprint.Format == y.Footprint.Format &&
                    x.Footprint.Width == y.Footprint.Width && x.Footprint.Height == y.Footprint.Height &&
                    x.Footprint.Depth == y.Footprint.Depth && x.Footprint.RowPitch == y.Footprint.RowPitch &&
                    rows_a[i] == rows_b[i] && size_a[i] == size_b[i])
                    continue;
                if (mismatches++ < 24)
                    std::fprintf(stderr,
                                 "footprint format %d base %llu subresource %u: ours offset %llu fmt %d %ux%ux%u pitch %u "
                                 "rows %u row %llu; host offset %llu fmt %d %ux%ux%u pitch %u rows %u row %llu\n",
                                 int(c.desc.Format), (unsigned long long)base, i, (unsigned long long)x.Offset,
                                 int(x.Footprint.Format), x.Footprint.Width, x.Footprint.Height, x.Footprint.Depth,
                                 x.Footprint.RowPitch, rows_a[i], (unsigned long long)size_a[i],
                                 (unsigned long long)y.Offset, int(y.Footprint.Format), y.Footprint.Width,
                                 y.Footprint.Height, y.Footprint.Depth, y.Footprint.RowPitch, rows_b[i],
                                 (unsigned long long)size_b[i]);
            }
            if (total_a != total_b && mismatches++ < 8)
                std::fprintf(stderr, "footprint format %d base %llu: total ours %llu host %llu\n", int(c.desc.Format),
                             (unsigned long long)base, (unsigned long long)total_a, (unsigned long long)total_b);
        }
    }
    return mismatches;
}

// GetResourceAllocationInfo against the PC runtime. Buffer sizes, every alignment and which descriptions are invalid
// must match it exactly. A texture's size is the driver's choice on both sides, so ours is only checked to be a
// whole number of alignments and at least the texels of its top mip; both sizes are printed. Also checks the
// Device4/Device8 variants' per-resource layout. Returns the number of mismatches, or -1 without a host device.
int compare_allocation_info(ID3D12Device* ours) {
    Ref<ID3D12Device> host;
    if (!create_host_device(host)) return -1;
    const auto with = [](D3D12_RESOURCE_DESC desc, UINT64 alignment, UINT samples = 1) {
        desc.Alignment = alignment;
        desc.SampleDesc.Count = samples;
        return desc;
    };
    constexpr UINT64 k4 = D3D12_SMALL_RESOURCE_PLACEMENT_ALIGNMENT, k64 = D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT;
    constexpr D3D12_RESOURCE_FLAGS none = D3D12_RESOURCE_FLAG_NONE, rt = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
    constexpr DXGI_FORMAT rgba8 = DXGI_FORMAT_R8G8B8A8_UNORM;
    struct Case {
        const char* name;
        D3D12_RESOURCE_DESC desc;
        UINT64 top_mip_bytes;  // 0: the size must match the host's exactly (buffers, invalid descriptions)
    };
    D3D12_RESOURCE_DESC bad_buffer = buffer_desc(256);
    bad_buffer.Height = 2;
    D3D12_RESOURCE_DESC typed_buffer = buffer_desc(256);
    typed_buffer.Format = DXGI_FORMAT_R8_UNORM;
    D3D12_RESOURCE_DESC big_volume = volume_desc(1);
    big_volume.Width = big_volume.Height = 32;
    big_volume.DepthOrArraySize = 8;
    const Case cases[] = {
        {"buffer 300", buffer_desc(300), 0},
        {"buffer 64 KiB", buffer_desc(65536), 0},
        {"buffer 64 KiB + 1", buffer_desc(65537), 0},
        {"buffer aligned 64 KiB", with(buffer_desc(1000), k64), 0},
        {"buffer width 0", buffer_desc(0), 0},
        {"buffer height 2", bad_buffer, 0},
        {"buffer with a format", typed_buffer, 0},
        {"256x256 full chain", texture_desc(256, 256, rgba8, none, 0), 256 * 256 * 4},
        {"16x16 small", with(texture_desc(16, 16, rgba8, none), k4), 16 * 16 * 4},
        {"16x16 small render target", with(texture_desc(16, 16, rgba8, rt), k4), 16 * 16 * 4},
        {"1024x1024 small (too big)", with(texture_desc(1024, 1024, rgba8, none), k4), 1024 * 1024 * 4},
        {"64x64 BC1 small", with(texture_desc(64, 64, DXGI_FORMAT_BC1_UNORM, none, 3), k4), 16 * 16 * 8},
        {"64x64 depth", texture_desc(64, 64, DXGI_FORMAT_D32_FLOAT, D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL),
         64 * 64 * 4},
        {"8x8 array of 6", texture_desc(8, 8, rgba8, none, 4, 6), 8 * 8 * 4 * 6},
        {"32x32x8 volume", big_volume, 32 * 32 * 8 * 4},
        {"64x64 4x MSAA", with(texture_desc(64, 64, rgba8, rt), 0, 4), 64 * 64 * 4 * 4},
        {"64x64 32x MSAA", with(texture_desc(64, 64, rgba8, rt), 0, 32), 64 * 64 * 4 * 32},
        {"64x64 64x MSAA (invalid)", with(texture_desc(64, 64, rgba8, rt), 0, 64), 0},
        {"64x64 zero sample count (invalid)", with(texture_desc(64, 64, rgba8, rt), 0, 0), 0},
        {"64x64 4x MSAA small", with(texture_desc(64, 64, rgba8, rt), k64, 4), 64 * 64 * 4 * 4},
        {"1024x1024 4x MSAA small", with(texture_desc(1024, 1024, rgba8, rt), k64, 4), 1024 * 1024 * 4 * 4},
        {"zero width", texture_desc(0, 16, rgba8, none), 0},
        {"alignment 12345", with(texture_desc(16, 16, rgba8, none), 12345), 0},
        {"4x MSAA with 2 mips", with(texture_desc(64, 64, rgba8, rt, 2), 0, 4), 0},
        {"16x16 R32G32_UINT", texture_desc(16, 16, DXGI_FORMAT_R32G32_UINT, none, 1), 16 * 16 * 8},
        {"16x16 R32G32_SINT", texture_desc(16, 16, DXGI_FORMAT_R32G32_SINT, none, 1), 16 * 16 * 8},
        {"16x16 R32G32_TYPELESS", texture_desc(16, 16, DXGI_FORMAT_R32G32_TYPELESS, none, 1), 16 * 16 * 8},
        {"16x16 R16G16B16A16_UINT", texture_desc(16, 16, DXGI_FORMAT_R16G16B16A16_UINT, none, 1), 16 * 16 * 8},
        {"16x16 R16G16B16A16_SINT", texture_desc(16, 16, DXGI_FORMAT_R16G16B16A16_SINT, none, 1), 16 * 16 * 8},
        {"16x16 R10G10B10A2_UINT", texture_desc(16, 16, DXGI_FORMAT_R10G10B10A2_UINT, none, 1), 16 * 16 * 4},
        {"16x16 R16G16_TYPELESS", texture_desc(16, 16, DXGI_FORMAT_R16G16_TYPELESS, none, 1), 16 * 16 * 4},
        {"16x16 R16G16_UINT", texture_desc(16, 16, DXGI_FORMAT_R16G16_UINT, none, 1), 16 * 16 * 4},
        {"16x16 R16G16_SINT", texture_desc(16, 16, DXGI_FORMAT_R16G16_SINT, none, 1), 16 * 16 * 4},
        {"16x16 R8G8_TYPELESS", texture_desc(16, 16, DXGI_FORMAT_R8G8_TYPELESS, none, 1), 16 * 16 * 2},
        {"16x16 R8G8_UINT", texture_desc(16, 16, DXGI_FORMAT_R8G8_UINT, none, 1), 16 * 16 * 2},
        {"16x16 R8G8_SINT", texture_desc(16, 16, DXGI_FORMAT_R8G8_SINT, none, 1), 16 * 16 * 2},
    };
    int mismatches = 0;
    const auto mismatch = [&](const char* name, const D3D12_RESOURCE_ALLOCATION_INFO& a,
                              const D3D12_RESOURCE_ALLOCATION_INFO& b) {
        if (mismatches++ < 24)
            std::fprintf(stderr, "allocation info %s: ours size %llu alignment %llu, host size %llu alignment %llu\n",
                         name, (unsigned long long)a.SizeInBytes, (unsigned long long)a.Alignment,
                         (unsigned long long)b.SizeInBytes, (unsigned long long)b.Alignment);
    };
    for (const Case& c : cases) {
        const D3D12_RESOURCE_ALLOCATION_INFO a = ours->GetResourceAllocationInfo(0, 1, &c.desc);
        const D3D12_RESOURCE_ALLOCATION_INFO b = host->GetResourceAllocationInfo(0, 1, &c.desc);
        if (b.SizeInBytes == UINT64_MAX || !c.top_mip_bytes) {
            if (a.SizeInBytes != b.SizeInBytes || (b.SizeInBytes != UINT64_MAX && a.Alignment != b.Alignment))
                mismatch(c.name, a, b);
            continue;
        }
        if (a.Alignment != b.Alignment || a.SizeInBytes == UINT64_MAX || a.SizeInBytes % a.Alignment ||
            a.SizeInBytes < c.top_mip_bytes)
            mismatch(c.name, a, b);
        std::printf("allocation info %s: alignment %llu, size ours %llu host %llu\n", c.name,
                    (unsigned long long)a.Alignment, (unsigned long long)a.SizeInBytes,
                    (unsigned long long)b.SizeInBytes);
    }

    // Several resources at once: each at its own alignment after the one before, the largest alignment overall.
    // Buffers only for the host comparison (their sizes are fixed); the texture checks our own layout.
    const D3D12_RESOURCE_DESC buffers[] = {buffer_desc(300), buffer_desc(70000), buffer_desc(1)};
    const D3D12_RESOURCE_ALLOCATION_INFO a = ours->GetResourceAllocationInfo(0, 3, buffers);
    const D3D12_RESOURCE_ALLOCATION_INFO b = host->GetResourceAllocationInfo(0, 3, buffers);
    if (a.SizeInBytes != b.SizeInBytes || a.Alignment != b.Alignment) mismatch("three buffers", a, b);

    Ref<ID3D12Device8> ours8;
    if (FAILED(ours->QueryInterface(IID_PPV_ARGS(&ours8.p)))) return mismatches + 1;
    const D3D12_RESOURCE_DESC mixed[] = {with(texture_desc(16, 16, rgba8, none), k4), buffer_desc(300),
                                         with(texture_desc(64, 64, rgba8, rt), 0, 4)};
    D3D12_RESOURCE_DESC1 mixed1[3]{};
    for (int i = 0; i < 3; ++i) std::memcpy(&mixed1[i], &mixed[i], sizeof(mixed[i]));
    D3D12_RESOURCE_ALLOCATION_INFO1 each[3]{}, each2[3]{};
    const D3D12_RESOURCE_ALLOCATION_INFO total = ours8->GetResourceAllocationInfo1(0, 3, mixed, each);
    const D3D12_RESOURCE_ALLOCATION_INFO total2 = ours8->GetResourceAllocationInfo2(0, 3, mixed1, each2);
    const auto aligned = [](UINT64 v, UINT64 to) { return (v + to - 1) / to * to; };
    bool layout_ok = total.SizeInBytes == total2.SizeInBytes && total.Alignment == total2.Alignment &&
                     each[0].Offset == 0 && each[0].Alignment == k4 && each[1].Alignment == k64 &&
                     each[2].Alignment == D3D12_DEFAULT_MSAA_RESOURCE_PLACEMENT_ALIGNMENT &&
                     total.Alignment == D3D12_DEFAULT_MSAA_RESOURCE_PLACEMENT_ALIGNMENT &&
                     total.SizeInBytes == each[2].Offset + each[2].SizeInBytes;
    for (int i = 0; i < 3; ++i) {
        const D3D12_RESOURCE_ALLOCATION_INFO single = ours->GetResourceAllocationInfo(0, 1, &mixed[i]);
        layout_ok = layout_ok && each[i].SizeInBytes == single.SizeInBytes &&
                    each[i].Alignment == single.Alignment && each[i].Offset == each2[i].Offset &&
                    each[i].SizeInBytes == each2[i].SizeInBytes && each[i].Alignment == each2[i].Alignment &&
                    (i == 0 || each[i].Offset == aligned(each[i - 1].Offset + each[i - 1].SizeInBytes,
                                                         each[i].Alignment));
    }
    // An invalid description in the list makes the whole answer invalid.
    const D3D12_RESOURCE_DESC with_bad[] = {buffer_desc(300), buffer_desc(0)};
    layout_ok = layout_ok && ours8->GetResourceAllocationInfo1(0, 2, with_bad, each).SizeInBytes == UINT64_MAX &&
                host->GetResourceAllocationInfo(0, 2, with_bad).SizeInBytes == UINT64_MAX;
    if (!layout_ok) {
        ++mismatches;
        std::fprintf(stderr, "allocation info: Device4/Device8 per-resource layout wrong (total %llu/%llu)\n",
                     (unsigned long long)total.SizeInBytes, (unsigned long long)total.Alignment);
    }
    return mismatches;
}

D3D12_RESOURCE_DESC reserved_desc(D3D12_RESOURCE_DIMENSION dimension, UINT64 width, UINT height, UINT16 depth,
                                  UINT16 mips, DXGI_FORMAT format) {
    D3D12_RESOURCE_DESC desc = dimension == D3D12_RESOURCE_DIMENSION_BUFFER
                                   ? buffer_desc(width)
                                   : texture_desc(UINT(width), height, format, D3D12_RESOURCE_FLAG_NONE, mips, depth);
    desc.Dimension = dimension;
    if (dimension != D3D12_RESOURCE_DIMENSION_BUFFER) desc.Layout = D3D12_TEXTURE_LAYOUT_64KB_UNDEFINED_SWIZZLE;
    return desc;
}

// GetResourceTiling against the PC runtime, for the same reserved resources. What D3D12 defines must match: the
// tile shape, every standard mip's tile counts and its start within its array slice, and that no mip filling a
// whole tile is packed. Which smaller mips are packed (D3D11.3 spec 5.9.2.8.5: "may or may not be packed,
// depending on the IHV") and how many tiles they take is the driver's choice (src/resource_tiling.h), and with it
// the slice size and the total: compared only against a host that splits at the same mip. A layout the host
// refuses is skipped (drivers differ in what they accept); one the host accepts and ours refuses is a mismatch.
// Returns the number of mismatches, or -1 without a host D3D12 device with tiled resources or with every layout
// skipped.
int compare_tiling(ID3D12Device* ours) {
    Ref<ID3D12Device> host;
    if (!create_host_device(host)) return -1;
    D3D12_FEATURE_DATA_D3D12_OPTIONS options{};
    if (FAILED(host->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS, &options, sizeof(options))) ||
        options.TiledResourcesTier == D3D12_TILED_RESOURCES_TIER_NOT_SUPPORTED)
        return -1;
    const bool volumes = options.TiledResourcesTier >= D3D12_TILED_RESOURCES_TIER_3;
    const auto B = D3D12_RESOURCE_DIMENSION_BUFFER, T2 = D3D12_RESOURCE_DIMENSION_TEXTURE2D,
               T3 = D3D12_RESOURCE_DIMENSION_TEXTURE3D;
    const D3D12_RESOURCE_DESC cases[] = {
        reserved_desc(B, 200000, 1, 1, 1, DXGI_FORMAT_UNKNOWN),
        reserved_desc(T2, 1024, 1024, 1, 0, DXGI_FORMAT_R8G8B8A8_UNORM),
        reserved_desc(T2, 1000, 300, 3, 0, DXGI_FORMAT_R8G8B8A8_UNORM_SRGB),
        reserved_desc(T2, 512, 512, 1, 0, DXGI_FORMAT_R8_UNORM),
        reserved_desc(T2, 512, 512, 1, 0, DXGI_FORMAT_R16_FLOAT),
        reserved_desc(T2, 512, 512, 1, 0, DXGI_FORMAT_R16G16B16A16_FLOAT),
        reserved_desc(T2, 512, 512, 2, 0, DXGI_FORMAT_R32G32B32A32_FLOAT),
        reserved_desc(T2, 2048, 2048, 1, 0, DXGI_FORMAT_BC1_UNORM),
        reserved_desc(T2, 2048, 1024, 2, 0, DXGI_FORMAT_BC7_UNORM),
        reserved_desc(T2, 1000, 600, 1, 0, DXGI_FORMAT_BC3_UNORM),
        reserved_desc(T2, 128, 128, 1, 1, DXGI_FORMAT_R8G8B8A8_UNORM),
        reserved_desc(T2, 100, 100, 1, 0, DXGI_FORMAT_R8G8B8A8_UNORM),
        reserved_desc(T2, 4096, 64, 1, 0, DXGI_FORMAT_B8G8R8A8_UNORM),
        reserved_desc(T2, 1024, 1024, 4, 3, DXGI_FORMAT_R32_FLOAT),
        reserved_desc(T3, 256, 256, 64, 0, DXGI_FORMAT_R8G8B8A8_UNORM),
        reserved_desc(T3, 128, 128, 128, 0, DXGI_FORMAT_R8_UNORM),
        reserved_desc(T3, 64, 64, 64, 0, DXGI_FORMAT_R16G16B16A16_FLOAT),
        reserved_desc(T3, 64, 64, 64, 0, DXGI_FORMAT_R32G32B32A32_FLOAT),
    };
    struct Tiling {
        UINT total = 0, count = 64;
        D3D12_PACKED_MIP_INFO packed{};
        D3D12_TILE_SHAPE shape{};
        D3D12_SUBRESOURCE_TILING subresources[64]{};
    };
    int mismatches = 0, compared = 0;
    const auto report = [&](int index, const char* what, UINT a, UINT b) {
        if (mismatches++ < 24) std::fprintf(stderr, "tiling case %d %s: ours %u host %u\n", index, what, a, b);
    };
    for (int index = 0; index < int(std::size(cases)); ++index) {
        const D3D12_RESOURCE_DESC& desc = cases[index];
        if (desc.Dimension == T3 && !volumes) continue;
        Ref<ID3D12Resource> a, b;
        const HRESULT ours_hr =
            ours->CreateReservedResource(&desc, D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&a.p));
        const HRESULT host_hr =
            host->CreateReservedResource(&desc, D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&b.p));
        if (FAILED(host_hr)) {
            // Which reserved resources a PC driver accepts varies: an RTX 3090 on driver 616.56 refuses texture
            // arrays whose mips reach the packed tail (cases 2, 6 and 8, at tier 3), an RTX 5070 on 621.01 accepts
            // them. The console is the reference for what the layer accepts, not the host, so there is nothing to
            // compare here.
            std::printf("GetResourceTiling case %d: the host runtime refuses it (0x%08lx), skipped\n", index,
                        static_cast<unsigned long>(host_hr));
            continue;
        }
        if (FAILED(ours_hr)) {
            report(index, "creation refused by ours, accepted by the host (HRESULT low word)", ours_hr & 0xffff, 0);
            continue;
        }
        ++compared;
        Tiling x, y;
        ours->GetResourceTiling(a.p, &x.total, &x.packed, &x.shape, &x.count, 0, x.subresources);
        host->GetResourceTiling(b.p, &y.total, &y.packed, &y.shape, &y.count, 0, y.subresources);
        if (x.shape.WidthInTexels != y.shape.WidthInTexels) report(index, "tile width", x.shape.WidthInTexels, y.shape.WidthInTexels);
        if (x.shape.HeightInTexels != y.shape.HeightInTexels) report(index, "tile height", x.shape.HeightInTexels, y.shape.HeightInTexels);
        if (x.shape.DepthInTexels != y.shape.DepthInTexels) report(index, "tile depth", x.shape.DepthInTexels, y.shape.DepthInTexels);
        if (x.count != y.count) {
            report(index, "subresource count", x.count, y.count);
            continue;
        }
        // Which small mips are packed is the driver's choice (D3D11.3 functional spec 5.9.2.8.5, D3D12_PACKED_MIP_INFO):
        // only a mip that fills at least one standard tile in every dimension must stay standard; a coarser mip with
        // any dimension below the tile "may or may not be packed, depending on the IHV", and once a mip is packed all
        // coarser ones are. Hosts differ: an RTX 5070 packs from the first mip below the tile (like ours), a Radeon
        // 880M keeps one or two more standard, WARP packs nothing. So the layer must keep every mip that fills a
        // tile standard and be self-consistent; the host's boundary is informational, and tile counts are compared
        // only for mips both treat as standard (and the packed totals only when both split at the same mip).
        const UINT layers = desc.Dimension == T2 ? desc.DepthOrArraySize : 1;
        const UINT mips = x.count / layers;
        const UINT slice_x = x.total / layers, slice_y = y.total / layers;
        UINT shared = 0;
        if (desc.Dimension != B) {
            UINT required = 0;  // leading mips that fill a whole tile in every dimension
            for (UINT m = 0; m < mips; ++m) {
                const UINT64 w = std::max<UINT64>(1, desc.Width >> m);
                const UINT h = std::max<UINT>(1, desc.Height >> m);
                const UINT d = desc.Dimension == T3 ? std::max<UINT>(1, desc.DepthOrArraySize >> m) : 1;
                if (w < x.shape.WidthInTexels || h < x.shape.HeightInTexels || d < x.shape.DepthInTexels) break;
                ++required;
            }
            if (x.packed.NumStandardMips < required || x.packed.NumStandardMips + x.packed.NumPackedMips != int(mips))
                report(index, "standard mips (a mip filling a tile must not be packed)", x.packed.NumStandardMips, required);
            if (y.packed.NumStandardMips < required)
                report(index, "host standard mips below the D3D12 minimum", y.packed.NumStandardMips, required);
            if (bool(x.packed.NumPackedMips) != bool(x.packed.NumTilesForPackedMips))
                report(index, "packed tiles", x.packed.NumPackedMips, x.packed.NumTilesForPackedMips);
            if (x.packed.NumStandardMips != y.packed.NumStandardMips)
                std::printf("GetResourceTiling case %d: standard mips ours %u, host %u (driver's choice, not compared)\n",
                            index, x.packed.NumStandardMips, y.packed.NumStandardMips);
            shared = std::min(x.packed.NumStandardMips, y.packed.NumStandardMips);
            if (x.packed.NumStandardMips == y.packed.NumStandardMips) {
                if (x.packed.NumPackedMips != y.packed.NumPackedMips) report(index, "packed mips", x.packed.NumPackedMips, y.packed.NumPackedMips);
                if (x.packed.StartTileIndexInOverallResource != y.packed.StartTileIndexInOverallResource)
                    report(index, "packed start", x.packed.StartTileIndexInOverallResource, y.packed.StartTileIndexInOverallResource);
                if (slice_x - x.packed.NumTilesForPackedMips != slice_y - y.packed.NumTilesForPackedMips)
                    report(index, "standard tiles per slice", slice_x - x.packed.NumTilesForPackedMips,
                           slice_y - y.packed.NumTilesForPackedMips);
            }
        } else if (x.total != y.total) {
            report(index, "buffer tiles", x.total, y.total);
        }
        for (UINT i = 0; i < x.count; ++i) {
            const auto &s = x.subresources[i], &t = y.subresources[i];
            const UINT layer = i / mips;
            if (desc.Dimension != B && i % mips >= x.packed.NumStandardMips) {
                // Packed: no tiles of its own. (On slices after the first the PC runtime writes a tile index here,
                // not the D3D12_PACKED_TILE its documentation gives; ours follows the documentation.)
                if (s.WidthInTiles || s.HeightInTiles || s.DepthInTiles || s.StartTileIndexInOverallResource != D3D12_PACKED_TILE)
                    report(index, "packed subresource", i, 0);
                continue;
            }
            if (desc.Dimension != B && i % mips >= shared) continue;  // standard in ours only: the host packs it
            if (s.WidthInTiles != t.WidthInTiles || s.HeightInTiles != t.HeightInTiles || s.DepthInTiles != t.DepthInTiles ||
                s.StartTileIndexInOverallResource - layer * slice_x != t.StartTileIndexInOverallResource - layer * slice_y)
                report(index, "standard subresource", i, 0);
        }
    }
    return compared ? mismatches : -1;
}


// whole_rectangle is a rectangle over the whole source: by the documentation the same as a null rectangle.
enum class ResolveScenario { full, rectangle, null_rectangle, whole_rectangle };

const char* resolve_scenario_name(ResolveScenario scenario) {
    switch (scenario) {
    case ResolveScenario::full: return "ResolveSubresource";
    case ResolveScenario::rectangle: return "ResolveSubresourceRegion(rect)";
    case ResolveScenario::null_rectangle: return "ResolveSubresourceRegion(null rect)";
    case ResolveScenario::whole_rectangle: return "ResolveSubresourceRegion(whole-source rect)";
    }
    return "unknown";
}

UINT resolve_pixel_size(DXGI_FORMAT format) {
    switch (format) {
    case DXGI_FORMAT_R8G8B8A8_UNORM:
    case DXGI_FORMAT_B8G8R8A8_UNORM:
    case DXGI_FORMAT_R10G10B10A2_UNORM: return 4;
    case DXGI_FORMAT_R16G16B16A16_FLOAT: return 8;
    case DXGI_FORMAT_R32G32B32A32_FLOAT: return 16;
    default: return 0;
    }
}

// Runs one deterministic resolve on either the host D3D12 device or the Vulkan-backed proxy and returns only
// packed texel bytes (row-pitch padding is discarded). The source/destination are cleared to solid colours so
// every format and sample count exercises the resolve path without requiring a shader or game data.
bool capture_resolve(ID3D12Device* device, DXGI_FORMAT format, UINT samples, ResolveScenario scenario,
                     std::vector<std::uint8_t>& pixels) {
#define RESOLVE_CHECK(condition) do { \
    if (!(condition)) { \
        std::fprintf(stderr, "resolve capture failed at line %d: %s (format %d, samples %u, %s)\n", \
                     __LINE__, #condition, int(format), samples, resolve_scenario_name(scenario)); \
        return false; \
    } \
} while (false)

    const bool proxy = rt::d3d12_xs::is_device_proxy(device);
    const UINT src_width = 8, src_height = 8;
    const UINT dst_width = scenario == ResolveScenario::full ? src_width : 12;
    const UINT dst_height = scenario == ResolveScenario::full ? src_height : 10;
    const UINT pixel_size = resolve_pixel_size(format);
    RESOLVE_CHECK(pixel_size != 0);

    const D3D12_HEAP_PROPERTIES default_heap{D3D12_HEAP_TYPE_DEFAULT};
    const D3D12_HEAP_PROPERTIES readback_heap{D3D12_HEAP_TYPE_READBACK};
    D3D12_RESOURCE_DESC src_desc =
        texture_desc(src_width, src_height, format, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET);
    src_desc.SampleDesc.Count = samples;
    D3D12_RESOURCE_DESC dst_desc =
        texture_desc(dst_width, dst_height, format, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET);

    Ref<ID3D12Resource> src, dst, readback;
    RESOLVE_CHECK(SUCCEEDED(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &src_desc,
                                                             D3D12_RESOURCE_STATE_RENDER_TARGET, nullptr,
                                                             IID_PPV_ARGS(&src.p))));
    RESOLVE_CHECK(SUCCEEDED(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &dst_desc,
                                                             D3D12_RESOURCE_STATE_RENDER_TARGET, nullptr,
                                                             IID_PPV_ARGS(&dst.p))));

    D3D12_DESCRIPTOR_HEAP_DESC heap_desc{D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 2, D3D12_DESCRIPTOR_HEAP_FLAG_NONE, 0};
    Ref<ID3D12DescriptorHeap> rtv_heap;
    RESOLVE_CHECK(SUCCEEDED(device->CreateDescriptorHeap(&heap_desc, IID_PPV_ARGS(&rtv_heap.p))));
    D3D12_CPU_DESCRIPTOR_HANDLE src_rtv{}, dst_rtv{};
    if (proxy) {
        const std::uint64_t start =
            reinterpret_cast<XboxHeapHandleFn>(slot(rtv_heap.p, 9))(rtv_heap.p);
        const UINT increment = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
        RESOLVE_CHECK(start != 0 && increment >= sizeof(leonida::TargetView));
        src_rtv = {SIZE_T(start)};
        dst_rtv = {SIZE_T(start + increment)};
    } else {
        src_rtv = rtv_heap->GetCPUDescriptorHandleForHeapStart();
        dst_rtv = {src_rtv.ptr + device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV)};
    }
    D3D12_RENDER_TARGET_VIEW_DESC msaa_view{};
    msaa_view.Format = format;
    msaa_view.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE2DMS;
    device->CreateRenderTargetView(src.p, &msaa_view, src_rtv);
    device->CreateRenderTargetView(dst.p, nullptr, dst_rtv);

    D3D12_COMMAND_QUEUE_DESC queue_desc{};
    queue_desc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    Ref<ID3D12CommandQueue> queue;
    Ref<ID3D12CommandAllocator> allocator;
    Ref<ID3D12GraphicsCommandList> list;
    Ref<ID3D12GraphicsCommandList7> list7;
    Ref<ID3D12Fence> fence;
    RESOLVE_CHECK(SUCCEEDED(device->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&queue.p))));
    RESOLVE_CHECK(SUCCEEDED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator.p))));
    RESOLVE_CHECK(SUCCEEDED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.p, nullptr,
                                                       IID_PPV_ARGS(&list.p))));
    RESOLVE_CHECK(SUCCEEDED(list->QueryInterface(IID_PPV_ARGS(&list7.p))));
    RESOLVE_CHECK(SUCCEEDED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence.p))));

    UINT64 cursor = 0;
    const auto readback_layout = place(device, dst.p, 0, cursor);
    RESOLVE_CHECK(cursor != 0 && readback_layout.Footprint.RowPitch >= dst_width * pixel_size);
    const D3D12_RESOURCE_DESC readback_desc = buffer_desc(cursor);
    RESOLVE_CHECK(SUCCEEDED(device->CreateCommittedResource(&readback_heap, D3D12_HEAP_FLAG_NONE,
                                                             &readback_desc, D3D12_RESOURCE_STATE_COPY_DEST,
                                                             nullptr, IID_PPV_ARGS(&readback.p))));

    const FLOAT green[4] = {0, 1, 0, 1};
    const FLOAT red[4] = {1, 0, 0, 1};
    // Use a rectangle even for the full source: multisampled images are cleared as rendering attachments,
    // not with vkCmdClearColorImage (which is for single-sample transfer clears).
    const D3D12_RECT full_source{0, 0, LONG(src_width), LONG(src_height)};
    list->ClearRenderTargetView(src_rtv, green, 1, &full_source);
    list->ClearRenderTargetView(dst_rtv, red, 0, nullptr);
    const D3D12_RESOURCE_BARRIER before_resolve[] = {
        transition(src.p, D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_RESOLVE_SOURCE),
        transition(dst.p, D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_RESOLVE_DEST)};
    list->ResourceBarrier(2, before_resolve);
    if (scenario == ResolveScenario::full) {
        list->ResolveSubresource(dst.p, 0, src.p, 0, format);
    } else {
        D3D12_RECT rectangle{2, 1, 6, 5};
        if (scenario == ResolveScenario::whole_rectangle) rectangle = full_source;
        list7->ResolveSubresourceRegion(dst.p, 0, 3, 2, src.p, 0,
                                        scenario == ResolveScenario::null_rectangle ? nullptr : &rectangle,
                                        format, D3D12_RESOLVE_MODE_AVERAGE);
    }
    const auto to_copy = transition(dst.p, D3D12_RESOURCE_STATE_RESOLVE_DEST, D3D12_RESOURCE_STATE_COPY_SOURCE);
    list->ResourceBarrier(1, &to_copy);
    const auto readback_location = footprint(readback.p, readback_layout);
    const auto resolved_location = subresource(dst.p, 0);
    list->CopyTextureRegion(&readback_location, 0, 0, 0, &resolved_location, nullptr);
    RESOLVE_CHECK(list->Close() == S_OK);

    ID3D12CommandList* lists[] = {list.p};
    queue->ExecuteCommandLists(1, lists);
    if (proxy)
        RESOLVE_CHECK(rt::d3d12::last_execute_result(queue.p) == leonida::ReplayResult::success);
    RESOLVE_CHECK(SUCCEEDED(queue->Signal(fence.p, 1)) && SUCCEEDED(fence->SetEventOnCompletion(1, nullptr)));

    D3D12_RANGE read_range{0, SIZE_T(cursor)};
    std::uint8_t* mapped = nullptr;
    RESOLVE_CHECK(SUCCEEDED(readback->Map(0, &read_range, reinterpret_cast<void**>(&mapped))) && mapped);
    pixels.resize(size_t(dst_width) * dst_height * pixel_size);
    for (UINT row = 0; row < dst_height; ++row)
        std::memcpy(pixels.data() + size_t(row) * dst_width * pixel_size,
                    mapped + readback_layout.Offset + size_t(row) * readback_layout.Footprint.RowPitch,
                    size_t(dst_width) * pixel_size);
    const D3D12_RANGE written{0, 0};
    readback->Unmap(0, &written);
#undef RESOLVE_CHECK
    return true;
}

// Compare D3D12 multisample quality queries for depth/stencil formats with the host runtime. The Vulkan
// backend has one standard sample pattern, so every supported count reports one quality level (index 0).
// A count the Vulkan GPU cannot do for depth and stencil attachments (depth_sample_counts, from its limits) is not
// something the layer can offer: a software host such as WARP advertises 16x, an RTX or Radeon does not.
int compare_depth_multisample_quality_levels(ID3D12Device* ours, VkSampleCountFlags depth_sample_counts) {
    Ref<ID3D12Device> host;
    if (!create_host_device(host)) return -1;

    constexpr DXGI_FORMAT formats[] = {
        DXGI_FORMAT_D32_FLOAT,
        DXGI_FORMAT_D16_UNORM,
        DXGI_FORMAT_D24_UNORM_S8_UINT,
        DXGI_FORMAT_D32_FLOAT_S8X24_UINT,
    };
    constexpr UINT sample_counts[] = {2, 4, 8, 16, 32};
    unsigned compared = 0, skipped = 0;
    int mismatches = 0;
    for (DXGI_FORMAT format : formats) {
        for (UINT samples : sample_counts) {
            D3D12_FEATURE_DATA_MULTISAMPLE_QUALITY_LEVELS host_levels{};
            host_levels.Format = format;
            host_levels.SampleCount = samples;
            D3D12_FEATURE_DATA_MULTISAMPLE_QUALITY_LEVELS our_levels = host_levels;
            const HRESULT host_hr = host->CheckFeatureSupport(D3D12_FEATURE_MULTISAMPLE_QUALITY_LEVELS,
                                                               &host_levels, sizeof(host_levels));
            if (FAILED(host_hr) || host_levels.NumQualityLevels == 0 || !(depth_sample_counts & samples)) {
                ++skipped;
                continue;
            }
            ++compared;
            const HRESULT our_hr = ours->CheckFeatureSupport(D3D12_FEATURE_MULTISAMPLE_QUALITY_LEVELS,
                                                             &our_levels, sizeof(our_levels));
            if (FAILED(our_hr) || our_levels.NumQualityLevels != 1) {
                std::fprintf(stderr,
                             "depth MSAA quality mismatch: format %d samples %u (host %u quality levels, Vulkan "
                             "HRESULT 0x%08lx / %u levels)\n",
                             int(format), samples, host_levels.NumQualityLevels,
                             static_cast<unsigned long>(our_hr), our_levels.NumQualityLevels);
                ++mismatches;
            }
        }
    }
    std::printf("Depth/stencil MSAA quality queries: %u compared, %u host-unsupported skipped, %d mismatches\n",
                compared, skipped, mismatches);
    return mismatches;
}

// Compare the Vulkan-backed resolve output byte-for-byte with the host runtime for every common supported format,
// sample count and region form. A host-unsupported combination is a deliberate skip; a mismatch in advertised
// support or in the resulting pixels is a failure.
int compare_resolves(ID3D12Device* ours, VkSampleCountFlags color_sample_counts) {
    Ref<ID3D12Device> host;
    if (!create_host_device(host)) return -1;

    constexpr DXGI_FORMAT formats[] = {
        DXGI_FORMAT_R8G8B8A8_UNORM,
        DXGI_FORMAT_B8G8R8A8_UNORM,
        DXGI_FORMAT_R10G10B10A2_UNORM,
        DXGI_FORMAT_R16G16B16A16_FLOAT,
        DXGI_FORMAT_R32G32B32A32_FLOAT,
    };
    constexpr UINT sample_counts[] = {2, 4, 8, 16, 32};
    constexpr UINT required =
        UINT(D3D12_FORMAT_SUPPORT1_RENDER_TARGET) | UINT(D3D12_FORMAT_SUPPORT1_MULTISAMPLE_RENDERTARGET) |
        UINT(D3D12_FORMAT_SUPPORT1_MULTISAMPLE_RESOLVE);
    constexpr ResolveScenario scenarios[] = {
        ResolveScenario::full, ResolveScenario::rectangle, ResolveScenario::null_rectangle,
        ResolveScenario::whole_rectangle};
    int mismatches = 0;
    unsigned compared = 0, matched = 0, skipped = 0;

    for (DXGI_FORMAT format : formats) {
        D3D12_FEATURE_DATA_FORMAT_SUPPORT host_format{format}, our_format{format};
        const HRESULT host_format_hr =
            host->CheckFeatureSupport(D3D12_FEATURE_FORMAT_SUPPORT, &host_format, sizeof(host_format));
        const HRESULT our_format_hr =
            ours->CheckFeatureSupport(D3D12_FEATURE_FORMAT_SUPPORT, &our_format, sizeof(our_format));
        if (FAILED(host_format_hr) || (UINT(host_format.Support1) & required) != required) {
            std::printf("resolve format %d: host does not advertise multisample colour resolve, skipped\n", int(format));
            ++skipped;
            continue;
        }
        if (FAILED(our_format_hr) || (UINT(our_format.Support1) & required) != required) {
            std::fprintf(stderr, "resolve format %d: Vulkan device does not match host multisample format support\n",
                         int(format));
            ++mismatches;
            continue;
        }

        for (UINT samples : sample_counts) {
            D3D12_FEATURE_DATA_MULTISAMPLE_QUALITY_LEVELS host_levels{};
            host_levels.Format = format;
            host_levels.SampleCount = samples;
            D3D12_FEATURE_DATA_MULTISAMPLE_QUALITY_LEVELS our_levels = host_levels;
            const HRESULT host_hr = host->CheckFeatureSupport(D3D12_FEATURE_MULTISAMPLE_QUALITY_LEVELS,
                                                               &host_levels, sizeof(host_levels));
            const HRESULT our_hr = ours->CheckFeatureSupport(D3D12_FEATURE_MULTISAMPLE_QUALITY_LEVELS,
                                                              &our_levels, sizeof(our_levels));
            // Not a count the Vulkan GPU can do for colour attachments (a software host like WARP offers 16x).
            if (FAILED(host_hr) || host_levels.NumQualityLevels == 0 || !(color_sample_counts & samples)) {
                ++skipped;
                continue;
            }
            if (FAILED(our_hr) || our_levels.NumQualityLevels == 0) {
                std::fprintf(stderr, "resolve format %d samples %u: Vulkan reports unsupported, host supports it\n",
                             int(format), samples);
                ++mismatches;
                continue;
            }

            for (ResolveScenario scenario : scenarios) {
                std::vector<std::uint8_t> host_pixels, our_pixels;
                if (!capture_resolve(host.p, format, samples, scenario, host_pixels) ||
                    !capture_resolve(ours, format, samples, scenario, our_pixels)) {
                    ++mismatches;
                    continue;
                }
                ++compared;
                if (host_pixels != our_pixels && scenario == ResolveScenario::null_rectangle) {
                    // A null rectangle is the whole source (ID3D12GraphicsCommandList1::ResolveSubresourceRegion).
                    // Some hosts do not honour that (the Radeon 880M on driver 26.11.2 and WARP resolve to the
                    // destination's origin and leave the rest unwritten; an RTX 5070 follows the documentation), so
                    // where the host disagrees with its own whole-source rectangle, the layer is held to the
                    // documented result: that rectangle's.
                    std::vector<std::uint8_t> host_whole;
                    if (capture_resolve(host.p, format, samples, ResolveScenario::whole_rectangle, host_whole) &&
                        host_whole != host_pixels && host_whole == our_pixels) {
                        std::printf("resolve format %d samples %u: the host's null-rectangle resolve differs from "
                                    "its whole-source rectangle (host quirk); ours follows the documentation\n",
                                    int(format), samples);
                        ++matched;
                        continue;
                    }
                }
                if (host_pixels != our_pixels) {
                    std::fprintf(stderr, "resolve mismatch: format %d samples %u %s (host %zu bytes, Vulkan %zu)\n",
                                 int(format), samples, resolve_scenario_name(scenario),
                                 host_pixels.size(), our_pixels.size());
                    ++mismatches;
                } else {
                    ++matched;
                }
            }
        }
    }
    std::printf("Resolve comparison: %u compared, %u matched, %u unsupported format/sample combinations skipped, "
                "%d mismatches\n", compared, matched, skipped, mismatches);
    if (compared == 0) {
        std::fprintf(stderr, "Resolve comparison: no common supported format/sample combination was exercised\n");
        ++mismatches;
    }
    return mismatches;
}


}  // namespace

int main() {
    vktest::Device d;
    EXPECT(d.create());
    EXPECT(d.depth_format_usable(VK_FORMAT_D32_SFLOAT_S8_UINT));
    VkSampleCountFlags depth_sample_counts = 0, color_sample_counts = 0;
    {
        VkPhysicalDeviceProperties properties{};
        d.instance_fn<PFN_vkGetPhysicalDeviceProperties>("vkGetPhysicalDeviceProperties")(d.physical, &properties);
        depth_sample_counts =
            properties.limits.framebufferDepthSampleCounts & properties.limits.framebufferStencilSampleCounts;
        color_sample_counts = properties.limits.framebufferColorSampleCounts;
    }
    constexpr UINT size = 8, small_size = 4;
    unsigned mismatches = 0;
    const auto report = [&](const char* what, UINT index, const std::uint8_t* got, const std::uint8_t* want, size_t n) {
        if (std::memcmp(got, want, n) == 0) return;
        if (mismatches++ < 8) {
            std::fprintf(stderr, "%s %u: got", what, index);
            for (size_t i = 0; i < n; ++i) std::fprintf(stderr, " %02x", got[i]);
            std::fprintf(stderr, ", want");
            for (size_t i = 0; i < n; ++i) std::fprintf(stderr, " %02x", want[i]);
            std::fprintf(stderr, "\n");
        }
    };

    {
        rt::d3d12::VulkanDeviceInfo info{};
        info.get_instance_proc = d.instance_proc;
        info.instance = d.instance;
        info.physical = d.physical;
        info.device = d.device;
        info.queue = d.queue;
        info.queue_family = d.family;
        ID3D12Device* inner = nullptr;
        EXPECT(SUCCEEDED(rt::d3d12::create_vulkan_device(info, IID_PPV_ARGS(&inner))));
        Ref<ID3D12Device> device;
        EXPECT(SUCCEEDED(rt::d3d12_xs::create_device_proxy(inner, __uuidof(ID3D12Device),
                                                           reinterpret_cast<void**>(&device.p))));
        rt::d3d12::set_device_outer(inner, device.p);  // the proxy now owns `inner`
        EXPECT(rt::d3d12_xs::is_device_proxy(device.p));
        EXPECT(rt::d3d12_xs::device_proxy_slot_count(device.p) == 80);  // ID3D12Device11

        {
            Ref<ID3D12Device11> device11;
            Ref<IUnknown> device12;
            EXPECT(SUCCEEDED(device->QueryInterface(IID_PPV_ARGS(&device11.p))) && device11.p == device.p);
            EXPECT(device->QueryInterface(__uuidof(ID3D12Device12), reinterpret_cast<void**>(&device12.p)) ==
                   E_NOINTERFACE);
        }
        EXPECT(device->GetNodeCount() == 1);
        const D3D_FEATURE_LEVEL levels[] = {D3D_FEATURE_LEVEL_11_0, D3D_FEATURE_LEVEL_12_1, D3D_FEATURE_LEVEL_12_2};
        D3D12_FEATURE_DATA_FEATURE_LEVELS feature_levels{3, levels, D3D_FEATURE_LEVEL(0)};
        EXPECT(SUCCEEDED(device->CheckFeatureSupport(D3D12_FEATURE_FEATURE_LEVELS, &feature_levels,
                                                     sizeof(feature_levels))));
        EXPECT(feature_levels.MaxSupportedFeatureLevel == D3D_FEATURE_LEVEL_12_1);
        D3D12_FEATURE_DATA_ARCHITECTURE architecture{};
        EXPECT(SUCCEEDED(device->CheckFeatureSupport(D3D12_FEATURE_ARCHITECTURE, &architecture, sizeof(architecture))));
        D3D12_FEATURE_DATA_D3D12_OPTIONS options{};
        EXPECT(SUCCEEDED(device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS, &options, sizeof(options))));
        EXPECT(options.ResourceBindingTier == D3D12_RESOURCE_BINDING_TIER_3);
        EXPECT(options.MaxGPUVirtualAddressBitsPerResource == 40);

        D3D12_FEATURE_DATA_GPU_VIRTUAL_ADDRESS_SUPPORT va_support{};
        EXPECT(SUCCEEDED(device->CheckFeatureSupport(D3D12_FEATURE_GPU_VIRTUAL_ADDRESS_SUPPORT, &va_support, sizeof(va_support))));
        EXPECT(va_support.MaxGPUVirtualAddressBitsPerResource == 40);
        EXPECT(va_support.MaxGPUVirtualAddressBitsPerProcess == 48);

        D3D12_FEATURE_DATA_FORMAT_SUPPORT format_support{DXGI_FORMAT_R8G8B8A8_UNORM};
        EXPECT(SUCCEEDED(device->CheckFeatureSupport(D3D12_FEATURE_FORMAT_SUPPORT, &format_support, sizeof(format_support))));
        EXPECT(format_support.Support1 & D3D12_FORMAT_SUPPORT1_TEXTURE2D);
        EXPECT(format_support.Support1 & D3D12_FORMAT_SUPPORT1_RENDER_TARGET);
        EXPECT(format_support.Support1 & D3D12_FORMAT_SUPPORT1_SHADER_SAMPLE);

        D3D12_FEATURE_DATA_MULTISAMPLE_QUALITY_LEVELS invalid_sample_count{};
        invalid_sample_count.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        invalid_sample_count.SampleCount = D3D12_MAX_MULTISAMPLE_SAMPLE_COUNT * 2;
        EXPECT(device->CheckFeatureSupport(D3D12_FEATURE_MULTISAMPLE_QUALITY_LEVELS, &invalid_sample_count,
                                           sizeof(invalid_sample_count)) == E_INVALIDARG);
        EXPECT(invalid_sample_count.NumQualityLevels == 0);
        invalid_sample_count.SampleCount = 0;
        EXPECT(device->CheckFeatureSupport(D3D12_FEATURE_MULTISAMPLE_QUALITY_LEVELS, &invalid_sample_count,
                                           sizeof(invalid_sample_count)) == E_INVALIDARG);

        D3D12_RESOURCE_DESC unsupported_quality =
            texture_desc(size, size, DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET);
        unsupported_quality.SampleDesc.Count = 4;
        unsupported_quality.SampleDesc.Quality = 1;
        EXPECT(device->GetResourceAllocationInfo(0, 1, &unsupported_quality).SizeInBytes == UINT64_MAX);

        D3D12_FEATURE_DATA_FORMAT_SUPPORT invalid_format{DXGI_FORMAT_UNKNOWN};
        EXPECT(FAILED(device->CheckFeatureSupport(D3D12_FEATURE_FORMAT_SUPPORT, &invalid_format, sizeof(invalid_format))));

        // D3D-60: CheckFeatureSupport and committed resource creation for integer and typeless formats
        {
            const D3D12_HEAP_PROPERTIES default_heap{D3D12_HEAP_TYPE_DEFAULT};
            for (DXGI_FORMAT f : {
                     DXGI_FORMAT_R32G32_TYPELESS, DXGI_FORMAT_R32G32_UINT, DXGI_FORMAT_R32G32_SINT,
                     DXGI_FORMAT_R16G16B16A16_UINT, DXGI_FORMAT_R16G16B16A16_SINT,
                     DXGI_FORMAT_R10G10B10A2_UINT, DXGI_FORMAT_R16G16_TYPELESS, DXGI_FORMAT_R16G16_UINT,
                     DXGI_FORMAT_R16G16_SINT, DXGI_FORMAT_R8G8_TYPELESS, DXGI_FORMAT_R8G8_UINT,
                     DXGI_FORMAT_R8G8_SINT}) {
                D3D12_FEATURE_DATA_FORMAT_SUPPORT fs{f};
                EXPECT(SUCCEEDED(device->CheckFeatureSupport(D3D12_FEATURE_FORMAT_SUPPORT, &fs, sizeof(fs))));
                EXPECT(fs.Support1 & D3D12_FORMAT_SUPPORT1_TEXTURE2D);
                Ref<ID3D12Resource> tex;
                const D3D12_RESOURCE_DESC desc = texture_desc(16, 16, f, D3D12_RESOURCE_FLAG_NONE, 1);
                EXPECT(SUCCEEDED(device->CreateCommittedResource(
                    &default_heap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COMMON, nullptr,
                    IID_PPV_ARGS(&tex.p))));
                EXPECT(tex.p != nullptr);
            }
        }

        // Push constants limit check (D3D-14): Vulkan spec guarantees >= 128 bytes.
        EXPECT(rt::d3d12::device_max_push_constants_size(device.p) >= 128);
        EXPECT(rt::d3d12::device_max_push_constants_size(inner) == rt::d3d12::device_max_push_constants_size(device.p));
        // The test GPU may have either answer (Intel Arc and a Radeon 680M on driver 22.40.84.06 report 128 bytes);
        // the device must flag the fallback exactly when the 144-byte block does not fit.
        EXPECT(rt::d3d12::device_push_constants_need_fallback(device.p) ==
               (rt::d3d12::device_max_push_constants_size(device.p) < 144));

        // Simulated fallback path when maxPushConstantsSize < 144 bytes:
        {
            rt::d3d12::VulkanDeviceInfo fallback_info = info;
            fallback_info.max_push_constants_size = 128;
            fallback_info.push_constants_need_fallback = true;
            ID3D12Device* fallback_inner = nullptr;
            EXPECT(SUCCEEDED(rt::d3d12::create_vulkan_device(fallback_info, IID_PPV_ARGS(&fallback_inner))));
            Ref<ID3D12Device> fallback_device;
            EXPECT(SUCCEEDED(rt::d3d12_xs::create_device_proxy(fallback_inner, __uuidof(ID3D12Device),
                                                               reinterpret_cast<void**>(&fallback_device.p))));
            rt::d3d12::set_device_outer(fallback_inner, fallback_device.p);
            EXPECT(rt::d3d12::device_max_push_constants_size(fallback_device.p) == 128);
            EXPECT(rt::d3d12::device_push_constants_need_fallback(fallback_device.p));
        }

        const int footprint_mismatches = compare_footprints(device.p);
        if (footprint_mismatches < 0)
            std::printf("GetCopyableFootprints: no host D3D12 device to compare with, skipped\n");
        EXPECT(footprint_mismatches <= 0);
        const int allocation_mismatches = compare_allocation_info(device.p);
        if (allocation_mismatches < 0)
            std::printf("GetResourceAllocationInfo: no host D3D12 device to compare with, skipped\n");
        EXPECT(allocation_mismatches <= 0);

        // D3D-46: copies against the host runtime (d3d12_copy_compare.h).
        const int copy_mismatches = copycmp::compare_copies(device.p, d.bc, d.depth_format_usable(VK_FORMAT_D24_UNORM_S8_UINT));
        if (copy_mismatches < 0) std::printf("Copies: no host D3D12 device to compare with, skipped\n");
        EXPECT(copy_mismatches <= 0);
        const int tiling_mismatches = compare_tiling(device.p);
        if (tiling_mismatches < 0)
            std::printf("GetResourceTiling: no host D3D12 device with tiled resources, or no layout it accepts; "
                        "skipped\n");
        EXPECT(tiling_mismatches <= 0);

        const int depth_quality_mismatches = compare_depth_multisample_quality_levels(device.p, depth_sample_counts);
        if (depth_quality_mismatches < 0)
            std::printf("Depth/stencil MSAA quality queries: no host D3D12 device to compare with, skipped\n");
        EXPECT(depth_quality_mismatches <= 0);

        const int resolve_mismatches = compare_resolves(device.p, color_sample_counts);
        if (resolve_mismatches < 0)
            std::printf("Resolve comparison: no host D3D12 device to compare with, skipped\n");
        EXPECT(resolve_mismatches <= 0);

        // Command objects; their GetDevice is the proxy, never the inner device.
        D3D12_COMMAND_QUEUE_DESC queue_desc{};
        queue_desc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
        Ref<ID3D12CommandQueue> queue;
        Ref<ID3D12CommandAllocator> allocator;
        Ref<ID3D12GraphicsCommandList> list;
        Ref<ID3D12Fence> fence;
        EXPECT(SUCCEEDED(device->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&queue.p))));
        EXPECT(SUCCEEDED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator.p))));
        EXPECT(SUCCEEDED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.p, nullptr,
                                                   IID_PPV_ARGS(&list.p))));
        EXPECT(SUCCEEDED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence.p))));
        {
            Ref<ID3D12Device> parent;
            EXPECT(SUCCEEDED(queue->GetDevice(IID_PPV_ARGS(&parent.p))) && parent.p == device.p);
        }
        // Residency and power (D3D-33): no-ops on Vulkan that still check their arguments, through the proxy's
        // forwarded slots. EnqueueMakeResident must signal its fence, or a queue waiting on it would hang.
        {
            Ref<ID3D12Fence> residency;
            EXPECT(SUCCEEDED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&residency.p))));
            ID3D12Pageable* const pageables[] = {fence.p, allocator.p};
            EXPECT(device->MakeResident(2, pageables) == S_OK);
            EXPECT(device->MakeResident(0, nullptr) == S_OK);
            EXPECT(device->MakeResident(1, nullptr) == E_INVALIDARG);
            EXPECT(device->Evict(2, pageables) == S_OK);
            EXPECT(device->Evict(0, nullptr) == S_OK);
            EXPECT(device->Evict(1, nullptr) == E_INVALIDARG);
            EXPECT(device->SetStablePowerState(TRUE) == S_OK);
            EXPECT(device->SetStablePowerState(FALSE) == S_OK);

            Ref<ID3D12Device1> device1;
            EXPECT(SUCCEEDED(device->QueryInterface(IID_PPV_ARGS(&device1.p))));
            const D3D12_RESIDENCY_PRIORITY priorities[] = {D3D12_RESIDENCY_PRIORITY_MAXIMUM,
                                                           D3D12_RESIDENCY_PRIORITY(D3D12_RESIDENCY_PRIORITY_LOW + 5)};
            EXPECT(device1->SetResidencyPriority(2, pageables, priorities) == S_OK);
            EXPECT(device1->SetResidencyPriority(0, nullptr, nullptr) == S_OK);
            EXPECT(device1->SetResidencyPriority(1, pageables, nullptr) == E_INVALIDARG);
            EXPECT(device1->SetResidencyPriority(1, nullptr, priorities) == E_INVALIDARG);

            Ref<ID3D12Device3> device3;
            EXPECT(SUCCEEDED(device->QueryInterface(IID_PPV_ARGS(&device3.p))));
            EXPECT(device3->EnqueueMakeResident(D3D12_RESIDENCY_FLAG_NONE, 2, pageables, residency.p, 7) == S_OK);
            EXPECT(residency->GetCompletedValue() == 7);
            EXPECT(device3->EnqueueMakeResident(D3D12_RESIDENCY_FLAG_DENY_OVERBUDGET, 0, nullptr, residency.p, 9) ==
                   S_OK);
            EXPECT(residency->GetCompletedValue() == 9);
            // Refused calls leave the fence alone.
            EXPECT(device3->EnqueueMakeResident(D3D12_RESIDENCY_FLAG_NONE, 1, pageables, nullptr, 11) == E_INVALIDARG);
            EXPECT(device3->EnqueueMakeResident(D3D12_RESIDENCY_FLAG_NONE, 1, nullptr, residency.p, 11) ==
                   E_INVALIDARG);
            EXPECT(device3->EnqueueMakeResident(D3D12_RESIDENCY_FLAGS(0x2), 1, pageables, residency.p, 11) ==
                   E_INVALIDARG);
            EXPECT(residency->GetCompletedValue() == 9);
            // The GPU sees the signal: a queue waits on it, then signals past it.
            EXPECT(SUCCEEDED(queue->Wait(residency.p, 9)));
            EXPECT(SUCCEEDED(queue->Signal(residency.p, 10)));
            EXPECT(SUCCEEDED(residency->SetEventOnCompletion(10, nullptr)) && residency->GetCompletedValue() == 10);

            Ref<ID3D12Device6> device6;
            EXPECT(SUCCEEDED(device->QueryInterface(IID_PPV_ARGS(&device6.p))));
            const HANDLE done = CreateEventW(nullptr, TRUE, FALSE, nullptr);
            EXPECT(done);
            BOOL further = TRUE;
            const HRESULT background = device6->SetBackgroundProcessingMode(
                D3D12_BACKGROUND_PROCESSING_MODE_DISABLE_BACKGROUND_WORK, D3D12_MEASUREMENTS_ACTION_COMMIT_RESULTS,
                done, &further);
            const DWORD signalled = WaitForSingleObject(done, 0);
            CloseHandle(done);
            EXPECT(background == S_OK && signalled == WAIT_OBJECT_0 && further == FALSE);
            EXPECT(device6->SetBackgroundProcessingMode(D3D12_BACKGROUND_PROCESSING_MODE_ALLOWED,
                                                        D3D12_MEASUREMENTS_ACTION_KEEP_ALL, nullptr, nullptr) == S_OK);
            EXPECT(device6->SetBackgroundProcessingMode(D3D12_BACKGROUND_PROCESSING_MODE(4),
                                                        D3D12_MEASUREMENTS_ACTION_KEEP_ALL, nullptr, nullptr) ==
                   E_INVALIDARG);
            EXPECT(device6->SetBackgroundProcessingMode(D3D12_BACKGROUND_PROCESSING_MODE_ALLOWED,
                                                        D3D12_MEASUREMENTS_ACTION(4), nullptr, nullptr) ==
                   E_INVALIDARG);
        }
        {
            Ref<ID3D12Device4> device4;
            Ref<ID3D12GraphicsCommandList> closed;
            EXPECT(SUCCEEDED(device->QueryInterface(IID_PPV_ARGS(&device4.p))));
            EXPECT(SUCCEEDED(device4->CreateCommandList1(0, D3D12_COMMAND_LIST_TYPE_DIRECT,
                                                         D3D12_COMMAND_LIST_FLAG_NONE, IID_PPV_ARGS(&closed.p))));
            EXPECT(closed->Close() == E_FAIL);  // CreateCommandList1 returns a closed list
        }

        // Descriptor heaps from slot 14, start handles in the Xbox shape.
        D3D12_DESCRIPTOR_HEAP_DESC heap_desc{D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 2, D3D12_DESCRIPTOR_HEAP_FLAG_NONE, 0};
        Ref<ID3D12DescriptorHeap> rtv_heap, dsv_heap;
        EXPECT(SUCCEEDED(device->CreateDescriptorHeap(&heap_desc, IID_PPV_ARGS(&rtv_heap.p))));
        heap_desc = {D3D12_DESCRIPTOR_HEAP_TYPE_DSV, 1, D3D12_DESCRIPTOR_HEAP_FLAG_NONE, 0};
        EXPECT(SUCCEEDED(device->CreateDescriptorHeap(&heap_desc, IID_PPV_ARGS(&dsv_heap.p))));
        EXPECT(rt::d3d12_xs::is_heap_proxy(rtv_heap.p) && rt::d3d12_xs::is_heap_proxy(dsv_heap.p));
        const std::uint64_t rtv_start = reinterpret_cast<XboxHeapHandleFn>(slot(rtv_heap.p, 9))(rtv_heap.p);
        const std::uint64_t dsv_start = reinterpret_cast<XboxHeapHandleFn>(slot(dsv_heap.p, 9))(dsv_heap.p);
        const UINT increment = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
        EXPECT(rtv_start && dsv_start && increment >= sizeof(leonida::TargetView));
        const D3D12_CPU_DESCRIPTOR_HANDLE rtv0{SIZE_T(rtv_start)}, rtv1{SIZE_T(rtv_start + increment)},
            dsv0{SIZE_T(dsv_start)};

        // Resources: the render target through slot 76 with the Xbox tiled layout word, the others through the
        // PC CreateCommittedResource.
        const D3D12_HEAP_PROPERTIES default_heap{D3D12_HEAP_TYPE_DEFAULT};
        const D3D12_HEAP_PROPERTIES upload_heap{D3D12_HEAP_TYPE_UPLOAD};
        const D3D12_HEAP_PROPERTIES readback_heap{D3D12_HEAP_TYPE_READBACK};
        const auto create = [&](const D3D12_HEAP_PROPERTIES& heap, const D3D12_RESOURCE_DESC& desc,
                                D3D12_RESOURCE_STATES state, Ref<ID3D12Resource>& out) {
            return device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, state, nullptr,
                                                   IID_PPV_ARGS(&out.p));
        };
        Ref<ID3D12Resource> refused_buffer;
        D3D12_RESOURCE_DESC invalid_buffer_samples = buffer_desc(256);
        invalid_buffer_samples.SampleDesc.Count = 0;
        EXPECT(create(default_heap, invalid_buffer_samples, D3D12_RESOURCE_STATE_COPY_DEST, refused_buffer) ==
               E_INVALIDARG && !refused_buffer.p);
        invalid_buffer_samples.SampleDesc.Count = 1;
        invalid_buffer_samples.SampleDesc.Quality = 1;
        EXPECT(create(default_heap, invalid_buffer_samples, D3D12_RESOURCE_STATE_COPY_DEST, refused_buffer) ==
               E_INVALIDARG && !refused_buffer.p);

        Ref<ID3D12Resource> rt_resource, ds, untouched, upload, readback, volume;
        D3D12_RESOURCE_DESC xbox_rt =
            texture_desc(size, size, DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET);
        xbox_rt.Layout = static_cast<D3D12_TEXTURE_LAYOUT>(0x101);
        const IID resource_iid = __uuidof(ID3D12Resource);
        EXPECT(SUCCEEDED(reinterpret_cast<XboxCreateResourceFn>(slot(device.p, 76))(
            device.p, &default_heap, D3D12_HEAP_FLAG_NONE, &xbox_rt, D3D12_RESOURCE_STATE_COMMON, nullptr,
            &resource_iid, reinterpret_cast<void**>(&rt_resource.p))));
        EXPECT(SUCCEEDED(create(default_heap,
                                texture_desc(size, size, DXGI_FORMAT_D32_FLOAT_S8X24_UINT,
                                             D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL),
                                D3D12_RESOURCE_STATE_DEPTH_WRITE, ds)));
        EXPECT(SUCCEEDED(create(default_heap,
                                texture_desc(small_size, small_size, DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_RESOURCE_FLAG_NONE),
                                D3D12_RESOURCE_STATE_COPY_SOURCE, untouched)));
        EXPECT(SUCCEEDED(create(upload_heap, buffer_desc(65536), D3D12_RESOURCE_STATE_GENERIC_READ, upload)));
        EXPECT(SUCCEEDED(create(readback_heap, buffer_desc(65536), D3D12_RESOURCE_STATE_COPY_DEST, readback)));
        EXPECT(SUCCEEDED(create(default_heap, volume_desc(0), D3D12_RESOURCE_STATE_COPY_DEST, volume)));
        EXPECT(volume->GetDesc().MipLevels == 5);
        EXPECT(rt_resource->GetDesc().Width == size && rt_resource->GetDesc().Layout == D3D12_TEXTURE_LAYOUT_UNKNOWN);
        EXPECT(rt::d3d12::vulkan_image(rt_resource.p) && !rt::d3d12::vulkan_image(upload.p));

        // Upload buffers are mapped and zeroed.
        std::uint8_t* up = nullptr;
        std::uint8_t* down = nullptr;
        EXPECT(SUCCEEDED(upload->Map(0, nullptr, reinterpret_cast<void**>(&up))) && up);
        EXPECT(SUCCEEDED(readback->Map(0, nullptr, reinterpret_cast<void**>(&down))) && down);
        bool zeroed = true;
        for (int i = 0; i < 65536; ++i) zeroed = zeroed && up[i] == 0;
        EXPECT(zeroed);
        D3D12_HEAP_PROPERTIES heap_properties{};
        EXPECT(SUCCEEDED(upload->GetHeapProperties(&heap_properties, nullptr)) &&
               heap_properties.Type == D3D12_HEAP_TYPE_UPLOAD);

        // Views into the heaps.
        device->CreateRenderTargetView(rt_resource.p, nullptr, rtv0);
        D3D12_RENDER_TARGET_VIEW_DESC rtv_desc{};
        rtv_desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        rtv_desc.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE2D;
        device->CreateRenderTargetView(rt_resource.p, &rtv_desc, rtv1);
        device->CreateDepthStencilView(ds.p, nullptr, dsv0);

        // List 1: clears and barriers on never-written resources (their first use zero-fills them), then copies of
        // the colour, depth plane, stencil plane and the untouched texture into the readback buffer.
        UINT64 cursor = 0;
        const auto rt_place = place(device.p, rt_resource.p, 0, cursor);
        const auto depth_place = place(device.p, ds.p, 0, cursor);
        const auto stencil_place = place(device.p, ds.p, 1, cursor);
        const auto untouched_place = place(device.p, untouched.p, 0, cursor);
        EXPECT(rt_place.Footprint.RowPitch == 256 && depth_place.Footprint.Format == DXGI_FORMAT_R32_TYPELESS &&
               stencil_place.Footprint.Format == DXGI_FORMAT_R8_TYPELESS);
        std::memset(down, 0xcd, 65536);

        D3D12_RESOURCE_BARRIER barrier =
            transition(rt_resource.p, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_RENDER_TARGET);
        list->ResourceBarrier(1, &barrier);
        const FLOAT red[4] = {1, 0, 0, 1}, blue[4] = {0, 0, 1, 1};
        const D3D12_RECT square{2, 2, 6, 6}, top_half{0, 0, 8, 4};
        list->ClearRenderTargetView(rtv0, red, 0, nullptr);
        list->ClearRenderTargetView(rtv1, blue, 1, &square);
        list->ClearDepthStencilView(dsv0, D3D12_CLEAR_FLAG_DEPTH | D3D12_CLEAR_FLAG_STENCIL, 0.25f, 0x42, 1, &top_half);
        barrier = transition(rt_resource.p, D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_COPY_SOURCE);
        list->ResourceBarrier(1, &barrier);
        barrier = transition(ds.p, D3D12_RESOURCE_STATE_DEPTH_WRITE, D3D12_RESOURCE_STATE_COPY_SOURCE);
        list->ResourceBarrier(1, &barrier);
        Ref<ID3D12GraphicsCommandList7> list7;
        EXPECT(SUCCEEDED(list->QueryInterface(IID_PPV_ARGS(&list7.p))));
        D3D12_TEXTURE_BARRIER readable{};  // an enhanced barrier that only makes the untouched texture readable
        readable.SyncBefore = D3D12_BARRIER_SYNC_NONE;
        readable.SyncAfter = D3D12_BARRIER_SYNC_COPY;
        readable.AccessBefore = D3D12_BARRIER_ACCESS_NO_ACCESS;
        readable.AccessAfter = D3D12_BARRIER_ACCESS_COPY_SOURCE;
        readable.LayoutBefore = D3D12_BARRIER_LAYOUT_COPY_SOURCE;
        readable.LayoutAfter = D3D12_BARRIER_LAYOUT_COPY_SOURCE;
        readable.pResource = untouched.p;
        readable.Subresources = {0xffffffff, 0, 0, 0, 0, 0};
        D3D12_BARRIER_GROUP group{};
        group.Type = D3D12_BARRIER_TYPE_TEXTURE;
        group.NumBarriers = 1;
        group.pTextureBarriers = &readable;
        list7->Barrier(1, &group);
        for (const auto& [resource, index, placed] :
             {std::tuple{rt_resource.p, 0u, rt_place}, std::tuple{ds.p, 0u, depth_place},
              std::tuple{ds.p, 1u, stencil_place}, std::tuple{untouched.p, 0u, untouched_place}}) {
            const auto dst = footprint(readback.p, placed);
            const auto src = subresource(resource, index);
            list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
        }
        EXPECT(list->Close() == S_OK);

        ID3D12CommandList* lists[] = {list.p};
        queue->ExecuteCommandLists(1, lists);
        EXPECT(rt::d3d12::last_execute_result(queue.p) == leonida::ReplayResult::success);
        EXPECT(SUCCEEDED(queue->Signal(fence.p, 1)));
        EXPECT(SUCCEEDED(fence->SetEventOnCompletion(1, nullptr)));  // null event: blocks until done
        EXPECT(fence->GetCompletedValue() == 1);

        for (UINT row = 0; row < size; ++row) {
            for (UINT col = 0; col < size; ++col) {
                const bool inside = col >= 2 && col < 6 && row >= 2 && row < 6;
                const std::uint8_t want[4] = {std::uint8_t(inside ? 0 : 255), 0, std::uint8_t(inside ? 255 : 0), 255};
                report("colour texel", row * size + col,
                       down + rt_place.Offset + row * rt_place.Footprint.RowPitch + col * 4, want, 4);
                const float depth = row < 4 ? 0.25f : 0.0f;  // the bottom half was zero-filled, never cleared
                report("depth texel", row * size + col,
                       down + depth_place.Offset + row * depth_place.Footprint.RowPitch + col * 4,
                       reinterpret_cast<const std::uint8_t*>(&depth), 4);
                const std::uint8_t stencil = row < 4 ? 0x42 : 0;
                report("stencil texel", row * size + col,
                       down + stencil_place.Offset + row * stencil_place.Footprint.RowPitch + col, &stencil, 1);
            }
        }
        const std::uint8_t zero[4] = {};
        for (UINT row = 0; row < small_size; ++row)
            for (UINT col = 0; col < small_size; ++col)
                report("untouched texel", row * small_size + col,
                       down + untouched_place.Offset + row * untouched_place.Footprint.RowPitch + col * 4, zero, 4);

        // List 2: uploads. A pattern goes from the upload buffer into a texture, which is copied whole into a
        // texture created in COMMON (implicit promotion to COPY_DEST), and in part (a boxed region) into a smaller
        // one; a BC1 texture makes a round trip; a buffer region is copied.
        Ref<ID3D12Resource> source, whole, part, bc;
        const D3D12_RESOURCE_DESC rgba = texture_desc(size, size, DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_RESOURCE_FLAG_NONE);
        EXPECT(SUCCEEDED(create(default_heap, rgba, D3D12_RESOURCE_STATE_COPY_DEST, source)));
        EXPECT(SUCCEEDED(create(default_heap, rgba, D3D12_RESOURCE_STATE_COMMON, whole)));
        EXPECT(SUCCEEDED(create(default_heap,
                                texture_desc(small_size, small_size, DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_RESOURCE_FLAG_NONE),
                                D3D12_RESOURCE_STATE_COPY_DEST, part)));
        if (d.bc)
            EXPECT(SUCCEEDED(create(default_heap,
                                    texture_desc(size, size, DXGI_FORMAT_BC1_UNORM, D3D12_RESOURCE_FLAG_NONE),
                                    D3D12_RESOURCE_STATE_COPY_DEST, bc)));

        UINT64 up_cursor = 0;
        const auto pattern_place = place(device.p, source.p, 0, up_cursor);
        for (UINT row = 0; row < size; ++row)
            for (UINT col = 0; col < size; ++col)
                std::memcpy(up + pattern_place.Offset + row * pattern_place.Footprint.RowPitch + col * 4,
                            pattern(col, row).data(), 4);
        D3D12_PLACED_SUBRESOURCE_FOOTPRINT bc_up{}, bc_down{};
        if (bc.p) {
            bc_up = place(device.p, bc.p, 0, up_cursor);
            for (UINT block_row = 0; block_row < 2; ++block_row)
                for (UINT i = 0; i < 16; ++i)  // two 8-byte blocks per row
                    up[bc_up.Offset + block_row * bc_up.Footprint.RowPitch + i] = std::uint8_t(0x30 + block_row * 16 + i);
        }
        const UINT64 bytes_at = (up_cursor + 255) / 256 * 256;
        for (UINT i = 0; i < 64; ++i) up[bytes_at + i] = std::uint8_t(0xa0 + i);

        cursor = 0;
        const auto whole_place = place(device.p, whole.p, 0, cursor);
        const auto part_place = place(device.p, part.p, 0, cursor);
        if (bc.p) bc_down = place(device.p, bc.p, 0, cursor);
        const UINT64 bytes_down = (cursor + 255) / 256 * 256;
        std::memset(down, 0xcd, 65536);

        EXPECT(SUCCEEDED(allocator->Reset()));
        EXPECT(SUCCEEDED(list->Reset(allocator.p, nullptr)));
        {
            const auto dst = subresource(source.p, 0);
            const auto src = footprint(upload.p, pattern_place);
            list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
        }
        barrier = transition(source.p, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COPY_SOURCE);
        list->ResourceBarrier(1, &barrier);
        list->CopyResource(whole.p, source.p);
        {
            const auto dst = subresource(part.p, 0);
            const auto src = subresource(source.p, 0);
            const D3D12_BOX box{2, 2, 0, 6, 6, 1};
            list->CopyTextureRegion(&dst, 0, 0, 0, &src, &box);
        }
        barrier = transition(whole.p, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COPY_SOURCE);  // promoted
        list->ResourceBarrier(1, &barrier);
        barrier = transition(part.p, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COPY_SOURCE);
        list->ResourceBarrier(1, &barrier);
        if (bc.p) {
            const auto dst = subresource(bc.p, 0);
            const auto src = footprint(upload.p, bc_up);
            list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
            barrier = transition(bc.p, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COPY_SOURCE);
            list->ResourceBarrier(1, &barrier);
        }
        for (const auto& [resource, placed] : {std::pair{whole.p, whole_place}, std::pair{part.p, part_place},
                                               std::pair{bc.p, bc_down}}) {
            if (!resource) continue;
            const auto dst = footprint(readback.p, placed);
            const auto src = subresource(resource, 0);
            list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
        }
        list->CopyBufferRegion(readback.p, bytes_down, upload.p, bytes_at, 64);
        EXPECT(list->Close() == S_OK);
        queue->ExecuteCommandLists(1, lists);
        EXPECT(rt::d3d12::last_execute_result(queue.p) == leonida::ReplayResult::success);
        EXPECT(SUCCEEDED(queue->Signal(fence.p, 2)));
        EXPECT(SUCCEEDED(fence->SetEventOnCompletion(2, nullptr)));

        for (UINT row = 0; row < size; ++row)
            for (UINT col = 0; col < size; ++col)
                report("whole-copy texel", row * size + col,
                       down + whole_place.Offset + row * whole_place.Footprint.RowPitch + col * 4,
                       pattern(col, row).data(), 4);
        for (UINT row = 0; row < small_size; ++row)
            for (UINT col = 0; col < small_size; ++col)
                report("boxed-copy texel", row * small_size + col,
                       down + part_place.Offset + row * part_place.Footprint.RowPitch + col * 4,
                       pattern(col + 2, row + 2).data(), 4);
        if (bc.p)
            for (UINT block_row = 0; block_row < 2; ++block_row)
                report("BC1 block row", block_row, down + bc_down.Offset + block_row * bc_down.Footprint.RowPitch,
                       up + bc_up.Offset + block_row * bc_up.Footprint.RowPitch, 16);
        else
            std::printf("BC1 round trip: no textureCompressionBC, skipped\n");
        report("buffer copy", 0, down + bytes_down, up + bytes_at, 64);

        // Mip 1 is 4x3x8. Move a two-row box across slices through each of the three copy paths.
        // The footprint has a third row, so using the box height as the slice stride corrupts every later slice.
        {
            Ref<ID3D12Resource> clone, cropped;
            EXPECT(SUCCEEDED(create(default_heap, volume_desc(0), D3D12_RESOURCE_STATE_COPY_DEST, clone)));
            EXPECT(SUCCEEDED(create(default_heap, volume_desc(0), D3D12_RESOURCE_STATE_COPY_DEST, cropped)));
            cursor = 0;
            const auto layout = place(device.p, volume.p, 1, cursor);
            EXPECT(layout.Footprint.Depth == 8 && layout.Footprint.Height == 3 && layout.Footprint.Width == 4);
            const auto offset = [&](UINT x, UINT y, UINT z) {
                return layout.Offset + (z * 3 + y) * layout.Footprint.RowPitch + x * 4;
            };
            for (UINT z = 0; z < 8; ++z)
                for (UINT y = 0; y < 3; ++y)
                    for (UINT x = 0; x < 4; ++x)
                        std::memcpy(up + offset(x, y, z), pattern(x, z + 8 * y).data(), 4);
            std::memset(down, 0xcd, 65536);
            const auto in = footprint(upload.p, layout), out = footprint(readback.p, layout);
            const auto v = subresource(volume.p, 1), c = subresource(clone.p, 1), target = subresource(cropped.p, 1);
            EXPECT(SUCCEEDED(allocator->Reset()) && SUCCEEDED(list->Reset(allocator.p, nullptr)));
            const D3D12_BOX upload_box{1, 1, 1, 4, 3, 5};
            list->CopyTextureRegion(&v, 0, 0, 2, &in, &upload_box);
            auto readable_volume = transition(volume.p, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COPY_SOURCE);
            list->ResourceBarrier(1, &readable_volume);
            list->CopyResource(clone.p, volume.p);
            readable_volume = transition(clone.p, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COPY_SOURCE);
            list->ResourceBarrier(1, &readable_volume);
            const D3D12_BOX clone_box{0, 0, 2, 3, 2, 6};
            list->CopyTextureRegion(&target, 1, 1, 3, &c, &clone_box);
            readable_volume = transition(cropped.p, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COPY_SOURCE);
            list->ResourceBarrier(1, &readable_volume);
            const D3D12_BOX output_box{1, 1, 3, 4, 3, 7};
            list->CopyTextureRegion(&out, 0, 0, 1, &target, &output_box);
            EXPECT(SUCCEEDED(list->Close()));
            queue->ExecuteCommandLists(1, lists);
            EXPECT(rt::d3d12::last_execute_result(queue.p) == leonida::ReplayResult::success);
            EXPECT(SUCCEEDED(queue->Signal(fence.p, 3)) && SUCCEEDED(fence->SetEventOnCompletion(3, nullptr)));
            const std::array<std::uint8_t, 4> untouched_bytes{0xcd, 0xcd, 0xcd, 0xcd};
            for (UINT z = 0; z < 8; ++z)
                for (UINT y = 0; y < 3; ++y)
                    for (UINT x = 0; x < 4; ++x) {
                        const auto want = (x < 3 && y < 2 && z >= 1 && z < 5)
                            ? pattern(x + 1, z + 8 * (y + 1)) : untouched_bytes;
                        report("volume texel", (z * 3 + y) * 4 + x, down + offset(x, y, z), want.data(), 4);
                    }
        }
        volume.reset();

        // A resolve: a 4x multisampled target is cleared and resolved into a single-sampled texture, which is read
        // back. First the shapes D3D12 refuses: a multisampled texture that is not a single-mip render target or
        // depth-stencil, a sample count that is not a power of two, and a multisampled volume.
        {
            Ref<ID3D12Resource> msaa, resolved, refused;
            D3D12_RESOURCE_DESC shape =
                texture_desc(size, size, DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET, 2);
            shape.SampleDesc.Count = 4;
            EXPECT(create(default_heap, shape, D3D12_RESOURCE_STATE_COMMON, refused) == E_INVALIDARG && !refused.p);
            shape.MipLevels = 1;
            shape.Flags = D3D12_RESOURCE_FLAG_NONE;
            EXPECT(create(default_heap, shape, D3D12_RESOURCE_STATE_COMMON, refused) == E_INVALIDARG && !refused.p);
            shape.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
            shape.SampleDesc.Count = 3;
            EXPECT(create(default_heap, shape, D3D12_RESOURCE_STATE_COMMON, refused) == E_INVALIDARG && !refused.p);
            shape.SampleDesc.Count = D3D12_MAX_MULTISAMPLE_SAMPLE_COUNT * 2;
            EXPECT(create(default_heap, shape, D3D12_RESOURCE_STATE_COMMON, refused) == E_INVALIDARG && !refused.p);
            shape.SampleDesc.Count = 0;
            EXPECT(create(default_heap, shape, D3D12_RESOURCE_STATE_COMMON, refused) == E_INVALIDARG && !refused.p);
            shape.SampleDesc.Count = 4;
            shape.SampleDesc.Quality = 1;  // The Vulkan backend exposes only quality level 0.
            EXPECT(create(default_heap, shape, D3D12_RESOURCE_STATE_COMMON, refused) == E_INVALIDARG && !refused.p);
            shape.SampleDesc.Quality = 0;
            shape = volume_desc(1);
            shape.SampleDesc.Count = 4;
            EXPECT(create(default_heap, shape, D3D12_RESOURCE_STATE_COMMON, refused) == E_INVALIDARG && !refused.p);

            D3D12_RESOURCE_DESC samples4 =
                texture_desc(size, size, DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET);
            samples4.SampleDesc.Count = 4;
            EXPECT(SUCCEEDED(create(default_heap, samples4, D3D12_RESOURCE_STATE_RENDER_TARGET, msaa)));
            EXPECT(SUCCEEDED(create(default_heap, rgba, D3D12_RESOURCE_STATE_RESOLVE_DEST, resolved)));
            EXPECT(msaa->GetDesc().SampleDesc.Count == 4);
            D3D12_RENDER_TARGET_VIEW_DESC msaa_view{};
            msaa_view.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
            msaa_view.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE2DMS;
            device->CreateRenderTargetView(msaa.p, &msaa_view, rtv1);  // the lists that used this slot have run
            cursor = 0;
            const auto resolved_place = place(device.p, resolved.p, 0, cursor);
            std::memset(down, 0xcd, 65536);

            EXPECT(SUCCEEDED(allocator->Reset()) && SUCCEEDED(list->Reset(allocator.p, nullptr)));
            const FLOAT green[4] = {0, 1, 0, 1};
            list->ClearRenderTargetView(rtv1, green, 0, nullptr);
            barrier = transition(msaa.p, D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_RESOLVE_SOURCE);
            list->ResourceBarrier(1, &barrier);
            list->ResolveSubresource(resolved.p, 0, msaa.p, 0, DXGI_FORMAT_R8G8B8A8_UNORM);
            barrier = transition(resolved.p, D3D12_RESOURCE_STATE_RESOLVE_DEST, D3D12_RESOURCE_STATE_COPY_SOURCE);
            list->ResourceBarrier(1, &barrier);
            {
                const auto dst = footprint(readback.p, resolved_place);
                const auto src = subresource(resolved.p, 0);
                list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
            }
            EXPECT(list->Close() == S_OK);
            queue->ExecuteCommandLists(1, lists);
            EXPECT(rt::d3d12::last_execute_result(queue.p) == leonida::ReplayResult::success);
            EXPECT(SUCCEEDED(queue->Signal(fence.p, 4)) && SUCCEEDED(fence->SetEventOnCompletion(4, nullptr)));
            const std::uint8_t resolved_green[4] = {0, 255, 0, 255};
            for (UINT row = 0; row < size; ++row)
                for (UINT col = 0; col < size; ++col)
                    report("resolved texel", row * size + col,
                           down + resolved_place.Offset + row * resolved_place.Footprint.RowPitch + col * 4,
                           resolved_green, 4);

            // Move a cropped two-colour region into a larger red texture.
            constexpr UINT region_width = 10, region_height = 9;
            constexpr auto region_format = DXGI_FORMAT_R8G8B8A8_UNORM;
            const auto region_desc =
                texture_desc(region_width, region_height, region_format, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET);
            Ref<ID3D12Resource> region_target;
            EXPECT(SUCCEEDED(create(default_heap, region_desc, D3D12_RESOURCE_STATE_RENDER_TARGET, region_target)));
            device->CreateRenderTargetView(region_target.p, nullptr, rtv0);
            cursor = 0;
            const auto region_place = place(device.p, region_target.p, 0, cursor);
            EXPECT(SUCCEEDED(allocator->Reset()) && SUCCEEDED(list->Reset(allocator.p, nullptr)));
            barrier = transition(msaa.p, D3D12_RESOURCE_STATE_RESOLVE_SOURCE, D3D12_RESOURCE_STATE_RENDER_TARGET);
            list->ResourceBarrier(1, &barrier);
            const D3D12_RECT blue_patch{2, 3, 4, 5};
            list->ClearRenderTargetView(rtv1, blue, 1, &blue_patch);
            list->ClearRenderTargetView(rtv0, red, 0, nullptr);
            const D3D12_RESOURCE_BARRIER resolve_barriers[] = {
                transition(msaa.p, D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_RESOLVE_SOURCE),
                transition(region_target.p, D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_RESOLVE_DEST)};
            list->ResourceBarrier(2, resolve_barriers);
            D3D12_RECT region{1, 2, 5, 5};
            const auto region_copy = subresource(region_target.p, 0), region_out = footprint(readback.p, region_place);
            const auto resolve_and_readback = [&](UINT x, D3D12_RECT* rect, UINT64 fence_value) {
                list7->ResolveSubresourceRegion(region_target.p, 0, x, 1, msaa.p, 0, rect, region_format,
                                                D3D12_RESOLVE_MODE_AVERAGE);
                const auto to_copy = transition(region_target.p, D3D12_RESOURCE_STATE_RESOLVE_DEST,
                                                D3D12_RESOURCE_STATE_COPY_SOURCE);
                list->ResourceBarrier(1, &to_copy);
                list->CopyTextureRegion(&region_out, 0, 0, 0, &region_copy, nullptr);
                if (list->Close() != S_OK) return false;
                queue->ExecuteCommandLists(1, lists);
                return rt::d3d12::last_execute_result(queue.p) == leonida::ReplayResult::success &&
                       SUCCEEDED(queue->Signal(fence.p, fence_value)) &&
                       SUCCEEDED(fence->SetEventOnCompletion(fence_value, nullptr));
            };
            EXPECT(resolve_and_readback(4, &region, 5));
            const std::uint8_t resolved_red[4] = {255, 0, 0, 255}, resolved_blue[4] = {0, 0, 255, 255};
            for (UINT row = 0; row < region_height; ++row)
                for (UINT col = 0; col < region_width; ++col) {
                    const bool in_region = row >= 1 && row < 4 && col >= 4 && col < 8;
                    const bool in_blue = row >= 2 && row < 4 && col >= 5 && col < 7;
                    report("partial resolve texel", row * region_width + col,
                           down + region_place.Offset + row * region_place.Footprint.RowPitch + col * 4,
                           in_blue ? resolved_blue : in_region ? resolved_green : resolved_red, 4);
                }

            // A null rectangle resolves the entire source at the destination offset.
            EXPECT(SUCCEEDED(allocator->Reset()) && SUCCEEDED(list->Reset(allocator.p, nullptr)));
            const auto to_resolve = transition(region_target.p, D3D12_RESOURCE_STATE_COPY_SOURCE,
                                               D3D12_RESOURCE_STATE_RESOLVE_DEST);
            list->ResourceBarrier(1, &to_resolve);
            EXPECT(resolve_and_readback(1, nullptr, 6));
            for (UINT row = 0; row < region_height; ++row)
                for (UINT col = 0; col < region_width; ++col) {
                    const bool in_source = row >= 1 && col >= 1 && col < 9;
                    const bool in_blue = row >= 4 && row < 6 && col >= 3 && col < 5;
                    report("whole source at offset", row * region_width + col,
                           down + region_place.Offset + row * region_place.Footprint.RowPitch + col * 4,
                           in_blue ? resolved_blue : in_source ? resolved_green : resolved_red, 4);
                }

            D3D12_RECT out_of_source{0, 0, LONG(size + 1), 1};
            for (auto [x, rect] : {std::pair{0u, &out_of_source}, std::pair{7u, &region}}) {
                EXPECT(SUCCEEDED(list->Reset(allocator.p, nullptr)));
                list->ResourceBarrier(1, &to_resolve);
                list7->ResolveSubresourceRegion(region_target.p, 0, x, 1, msaa.p, 0, rect, region_format,
                                                D3D12_RESOLVE_MODE_AVERAGE);
                EXPECT(list->Close() == S_OK);
                queue->ExecuteCommandLists(1, lists);
                EXPECT(rt::d3d12::last_execute_result(queue.p) == leonida::ReplayResult::invalid_copy);
            }
            EXPECT(SUCCEEDED(list->Reset(allocator.p, nullptr)));
            list7->ResolveSubresourceRegion(region_target.p, 0, 4, 1, msaa.p, 0, &region, region_format,
                                            D3D12_RESOLVE_MODE_MAX);
            EXPECT(list->Close() == E_INVALIDARG);
            EXPECT(SUCCEEDED(list->Reset(allocator.p, nullptr)));
            list7->ResolveSubresourceRegion(region_target.p, 0, 4, 1, msaa.p, 0, &region,
                                            DXGI_FORMAT_R8G8B8A8_UNORM_SRGB, D3D12_RESOLVE_MODE_AVERAGE);
            EXPECT(list->Close() == E_INVALIDARG);

            // A resolve of a resource into itself is refused when it is recorded, one from a single-sampled
            // texture when it is executed; nothing is submitted.
            EXPECT(SUCCEEDED(list->Reset(allocator.p, nullptr)));
            list->ResolveSubresource(resolved.p, 0, resolved.p, 0, DXGI_FORMAT_R8G8B8A8_UNORM);
            EXPECT(list->Close() == E_INVALIDARG);
            EXPECT(SUCCEEDED(list->Reset(allocator.p, nullptr)));
            list->ResolveSubresource(resolved.p, 0, source.p, 0, DXGI_FORMAT_R8G8B8A8_UNORM);
            EXPECT(list->Close() == S_OK);
            queue->ExecuteCommandLists(1, lists);
            EXPECT(rt::d3d12::last_execute_result(queue.p) == leonida::ReplayResult::invalid_copy);
        }

        // Reserved resources (docs/d3d12_vulkan.md section 4.2): fully backed at creation, so they copy like
        // committed ones and their first use reads zero; D3D12's description rules hold; tile-mapping calls are
        // accepted and change nothing.
        {
            const auto B = D3D12_RESOURCE_DIMENSION_BUFFER, T2 = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
            Ref<ID3D12Resource> tiled, tiled_buffer, xbox_tiled, refused;
            const D3D12_RESOURCE_DESC tiled_desc = reserved_desc(T2, size, size, 1, 1, DXGI_FORMAT_R8G8B8A8_UNORM);
            EXPECT(SUCCEEDED(device->CreateReservedResource(&tiled_desc, D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                                            IID_PPV_ARGS(&tiled.p))));
            EXPECT(tiled->GetDesc().Layout == D3D12_TEXTURE_LAYOUT_64KB_UNDEFINED_SWIZZLE);
            D3D12_HEAP_PROPERTIES reserved_heap{};
            EXPECT(tiled->GetHeapProperties(&reserved_heap, nullptr) == E_INVALIDARG);  // no heap of its own
            EXPECT(device->CreateReservedResource(&tiled_desc, D3D12_RESOURCE_STATE_COMMON, nullptr,
                                                  __uuidof(ID3D12Resource), nullptr) == S_FALSE);
            const auto refuses = [&](D3D12_RESOURCE_DESC desc) {
                const HRESULT hr = device->CreateReservedResource(&desc, D3D12_RESOURCE_STATE_COMMON, nullptr,
                                                                  IID_PPV_ARGS(&refused.p));
                return hr == E_INVALIDARG && !refused.p;
            };
            auto bad = tiled_desc;
            bad.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;  // textures need a 64 KiB swizzle layout
            EXPECT(refuses(bad));
            bad.Layout = D3D12_TEXTURE_LAYOUT_64KB_STANDARD_SWIZZLE;  // StandardSwizzle64KBSupported is FALSE
            EXPECT(refuses(bad));
            EXPECT(refuses(reserved_desc(T2, 64, 64, 1, 1, DXGI_FORMAT_R32G32B32_FLOAT)));  // 12-byte texels: no tiles
            bad = reserved_desc(B, 65536, 1, 1, 1, DXGI_FORMAT_UNKNOWN);
            bad.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;  // buffers are ROW_MAJOR
            EXPECT(refuses(bad));
            const D3D12_RESOURCE_DESC tiled_buffer_desc = reserved_desc(B, 2 * 65536, 1, 1, 1, DXGI_FORMAT_UNKNOWN);
            EXPECT(SUCCEEDED(device->CreateReservedResource(&tiled_buffer_desc, D3D12_RESOURCE_STATE_COMMON, nullptr,
                                                            IID_PPV_ARGS(&tiled_buffer.p))));
            auto xbox_desc = tiled_desc;
            xbox_desc.Layout = static_cast<D3D12_TEXTURE_LAYOUT>(0x101);  // the Xbox tiled layout word
            EXPECT(SUCCEEDED(device->CreateReservedResource(&xbox_desc, D3D12_RESOURCE_STATE_COMMON, nullptr,
                                                            IID_PPV_ARGS(&xbox_tiled.p))));
            {
                Ref<ID3D12Device4> device4;
                Ref<ID3D12Device10> device10;
                Ref<ID3D12Resource> one, two;
                EXPECT(SUCCEEDED(device->QueryInterface(IID_PPV_ARGS(&device4.p))));
                EXPECT(SUCCEEDED(device->QueryInterface(IID_PPV_ARGS(&device10.p))));
                EXPECT(SUCCEEDED(device4->CreateReservedResource1(&tiled_desc, D3D12_RESOURCE_STATE_COMMON, nullptr,
                                                                  nullptr, IID_PPV_ARGS(&one.p))));
                EXPECT(SUCCEEDED(device10->CreateReservedResource2(&tiled_desc, D3D12_BARRIER_LAYOUT_COMMON, nullptr,
                                                                   nullptr, 0, nullptr, IID_PPV_ARGS(&two.p))));
                UINT tiles = 0;
                device->GetResourceTiling(two.p, &tiles, nullptr, nullptr, nullptr, 0, nullptr);
                EXPECT(tiles == 1);  // 8x8 RGBA8: one packed tile
            }

            // Tiling: a resource that is not reserved has none, as on the PC runtime; a first index past the end
            // returns no subresources.
            UINT total = 7, count = 4;
            D3D12_PACKED_MIP_INFO packed{1, 1, 1, 1};
            D3D12_TILE_SHAPE shape{1, 1, 1};
            D3D12_SUBRESOURCE_TILING tilings[4]{};
            device->GetResourceTiling(upload.p, &total, &packed, &shape, &count, 0, tilings);
            EXPECT(total == 0 && count == 0 && shape.WidthInTexels == 0 && packed.NumStandardMips == 0 &&
                   packed.NumTilesForPackedMips == 0);
            count = 4;
            device->GetResourceTiling(tiled_buffer.p, &total, &packed, &shape, &count, 0, tilings);
            EXPECT(total == 2 && count == 1 && shape.WidthInTexels == 65536 && tilings[0].WidthInTiles == 2);
            count = 4;
            device->GetResourceTiling(tiled_buffer.p, nullptr, nullptr, nullptr, &count, 1, tilings);
            EXPECT(count == 0);

            // Mapping calls: accepted, logged once, and the resources keep their memory.
            D3D12_TILED_RESOURCE_COORDINATE origin{};
            D3D12_TILE_REGION_SIZE one_tile{1, FALSE, 0, 0, 0};
            const D3D12_TILE_RANGE_FLAGS unmap = D3D12_TILE_RANGE_FLAG_NULL;
            queue->UpdateTileMappings(tiled.p, 1, &origin, &one_tile, nullptr, 1, &unmap, nullptr, nullptr,
                                      D3D12_TILE_MAPPING_FLAG_NONE);
            queue->CopyTileMappings(tiled_buffer.p, &origin, tiled_buffer.p, &origin, &one_tile,
                                    D3D12_TILE_MAPPING_FLAG_NONE);

            // Upload a pattern into the reserved texture and bytes into the reserved buffer, read both back, and
            // read the never-written Xbox-layout texture, which is zero.
            UINT64 tiled_up_cursor = 0;
            const auto tiled_up = place(device.p, tiled.p, 0, tiled_up_cursor);
            for (UINT row = 0; row < size; ++row)
                for (UINT col = 0; col < size; ++col)
                    std::memcpy(up + tiled_up.Offset + row * tiled_up.Footprint.RowPitch + col * 4,
                                pattern(col, row).data(), 4);
            const UINT64 tiled_bytes_up = (tiled_up_cursor + 255) / 256 * 256;
            for (UINT i = 0; i < 256; ++i) up[tiled_bytes_up + i] = std::uint8_t(0x5a ^ i);
            UINT64 tiled_down_cursor = 0;
            const auto tiled_down = place(device.p, tiled.p, 0, tiled_down_cursor);
            const auto xbox_down = place(device.p, xbox_tiled.p, 0, tiled_down_cursor);
            const UINT64 tiled_bytes_down = (tiled_down_cursor + 255) / 256 * 256;
            std::memset(down, 0xcd, 65536);

            EXPECT(SUCCEEDED(allocator->Reset()) && SUCCEEDED(list->Reset(allocator.p, nullptr)));
            {
                const auto dst = subresource(tiled.p, 0);
                const auto src = footprint(upload.p, tiled_up);
                list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
            }
            barrier = transition(tiled.p, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COPY_SOURCE);
            list->ResourceBarrier(1, &barrier);
            barrier = transition(xbox_tiled.p, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_SOURCE);
            list->ResourceBarrier(1, &barrier);
            for (const auto& [resource, placed] : {std::pair{tiled.p, tiled_down}, std::pair{xbox_tiled.p, xbox_down}}) {
                const auto dst = footprint(readback.p, placed);
                const auto src = subresource(resource, 0);
                list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
            }
            list->CopyBufferRegion(tiled_buffer.p, 65536 + 4096, upload.p, tiled_bytes_up, 256);  // into the second tile
            barrier = transition(tiled_buffer.p, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COPY_SOURCE);
            list->ResourceBarrier(1, &barrier);
            list->CopyBufferRegion(readback.p, tiled_bytes_down, tiled_buffer.p, 65536 + 4096, 256);
            EXPECT(list->Close() == S_OK);
            queue->ExecuteCommandLists(1, lists);
            EXPECT(rt::d3d12::last_execute_result(queue.p) == leonida::ReplayResult::success);
            EXPECT(SUCCEEDED(queue->Signal(fence.p, 9)) && SUCCEEDED(fence->SetEventOnCompletion(9, nullptr)));
            for (UINT row = 0; row < size; ++row)
                for (UINT col = 0; col < size; ++col) {
                    report("reserved texel", row * size + col,
                           down + tiled_down.Offset + row * tiled_down.Footprint.RowPitch + col * 4,
                           pattern(col, row).data(), 4);
                    report("unwritten reserved texel", row * size + col,
                           down + xbox_down.Offset + row * xbox_down.Footprint.RowPitch + col * 4, zero, 4);
                }
            report("reserved buffer", 0, down + tiled_bytes_down, up + tiled_bytes_up, 256);
        }

        // Every child holds the proxy; once they are gone the proxy holds the only reference.
        upload->Unmap(0, nullptr);
        readback->Unmap(0, nullptr);
        for (auto* r : {&rt_resource, &ds, &untouched, &upload, &readback, &source, &whole, &part, &bc}) r->reset();
        list7.reset();
        list.reset();
        allocator.reset();
        queue.reset();
        fence.reset();
        rtv_heap.reset();
        dsv_heap.reset();
        EXPECT(rt::d3d12_xs::device_proxy_ref_count(device.p) == 1);
    }  // the device is released here, before the Vulkan device it runs on
    EXPECT(!mismatches);

    // CPU texture copies, once per path. A second D3D12 device on the same Vulkan device, told the extension is
    // off, takes the staging path even on a GPU that has host image copy.
    for (const bool host : {true, false}) {
        if (host && !d.host_image_copy) {
            std::printf("WriteToSubresource/ReadFromSubresource (host image copy): no VK_EXT_host_image_copy, "
                        "skipped\n");
            continue;
        }
        rt::d3d12::VulkanDeviceInfo info{};
        info.get_instance_proc = d.instance_proc;
        info.instance = d.instance;
        info.physical = d.physical;
        info.device = d.device;
        info.queue = d.queue;
        info.queue_family = d.family;
        info.host_image_copy = host;
        Ref<ID3D12Device> device;
        EXPECT(SUCCEEDED(rt::d3d12::create_vulkan_device(info, IID_PPV_ARGS(&device.p))));
        EXPECT(cpu_copies(device.p, d.bc, host ? "host image copy" : "staging copy") == 0);
    }

    // The path D3D12XboxCreateDevice takes with LEONIDA_D3D12=vulkan: a device on its own Vulkan instance.
    {
        Ref<ID3D12Device> standalone;
        EXPECT(SUCCEEDED(rt::d3d12::create_vulkan_device_standalone(IID_PPV_ARGS(&standalone.p))));
        D3D12_COMMAND_QUEUE_DESC queue_desc{};
        Ref<ID3D12CommandQueue> queue;
        Ref<ID3D12Fence> fence;
        EXPECT(SUCCEEDED(standalone->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&queue.p))));
        EXPECT(SUCCEEDED(standalone->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence.p))));
        EXPECT(SUCCEEDED(queue->Signal(fence.p, 5)));
        EXPECT(SUCCEEDED(fence->SetEventOnCompletion(5, nullptr)) && fence->GetCompletedValue() == 5);
    }

    if (d.validation_errors()) {
        std::fprintf(stderr, "%u validation errors\n", d.validation_errors());
        return 1;
    }
    std::printf("Vulkan device behind the Xbox-ABI proxy: clears, zero-filled first use, uploads, texture, buffer and "
                "BC copies, a resolve, reserved resources, and GetCopyableFootprints and GetResourceTiling all matched\n");
    return 0;
}
