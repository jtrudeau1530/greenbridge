// The Vulkan-backed ID3D12Device behind the Xbox-ABI device proxy, end to end, on a GPU. Everything goes through
// the proxy the way the game will call it: device methods (forwarded slots), descriptor heaps from slot 14 with
// start handles read in the Xbox shape (RAX), a texture from slot 76 with the Xbox layout word, committed
// resources, views, command lists, a queue and a fence. Results are read back through D3D12 itself
// (CopyTextureRegion into a readback heap, then Map). GetCopyableFootprints and GetResourceAllocationInfo are
// compared with the host PC runtime's.
// No shaders or game data. Set LEONIDA_VK_VALIDATION=1 to run under VK_LAYER_KHRONOS_validation; any validation
// error fails the test.
#include "vk_test_device.h"
#include "../shims/d3d12_com.h"
#include "../shims/d3d12_xs_device.h"

#include <array>
#include <cstdint>
#include <cstring>
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

// The host PC runtime's D3D12 device on the default adapter, to compare answers with; false if there is none.
bool create_host_device(Ref<ID3D12Device>& host) {
    const HMODULE d3d12 = LoadLibraryExA("d3d12.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
    const auto create = d3d12 ? reinterpret_cast<PFN_D3D12_CREATE_DEVICE>(
                                    reinterpret_cast<void*>(GetProcAddress(d3d12, "D3D12CreateDevice")))
                              : nullptr;
    return create && SUCCEEDED(create(nullptr, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&host.p)));
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
        {"64x64 4x MSAA small", with(texture_desc(64, 64, rgba8, rt), k64, 4), 64 * 64 * 4 * 4},
        {"1024x1024 4x MSAA small", with(texture_desc(1024, 1024, rgba8, rt), k64, 4), 1024 * 1024 * 4 * 4},
        {"zero width", texture_desc(0, 16, rgba8, none), 0},
        {"alignment 12345", with(texture_desc(16, 16, rgba8, none), 12345), 0},
        {"4x MSAA with 2 mips", with(texture_desc(64, 64, rgba8, rt, 2), 0, 4), 0},
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

}  // namespace

int main() {
    vktest::Device d;
    EXPECT(d.create());
    EXPECT(d.depth_format_usable(VK_FORMAT_D32_SFLOAT_S8_UINT));
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

        D3D12_FEATURE_DATA_FORMAT_SUPPORT invalid_format{DXGI_FORMAT_UNKNOWN};
        EXPECT(FAILED(device->CheckFeatureSupport(D3D12_FEATURE_FORMAT_SUPPORT, &invalid_format, sizeof(invalid_format))));

        // Push constants limit check (D3D-14): Vulkan spec guarantees >= 128 bytes.
        EXPECT(rt::d3d12::device_max_push_constants_size(device.p) >= 128);
        EXPECT(rt::d3d12::device_max_push_constants_size(inner) == rt::d3d12::device_max_push_constants_size(device.p));
        EXPECT(!rt::d3d12::device_push_constants_need_fallback(device.p));

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
                "BC copies, a resolve, and GetCopyableFootprints all matched\n");
    return 0;
}
