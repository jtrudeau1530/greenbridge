// The 1/2/3 method variants of the Vulkan-backed ID3D12Device (D3D-28), each compared with the method it wraps:
// CreateCommittedResource1/2/3 with CreateCommittedResource, CreateCommandQueue1 with CreateCommandQueue and
// GetCopyableFootprints1 with GetCopyableFootprints. The device is called directly, not through the Xbox-ABI proxy,
// because the proxy's slot 76 is the Xbox texture method, not CreateCommittedResource3 (docs/d3d12_vulkan.md
// section 7). The newer arguments' edge cases are checked against the PC runtime's answers, live when the host has
// a D3D12 device. A texture from CreateCommittedResource3 goes through an upload, a copy and a readback on a queue
// from CreateCommandQueue1. No shaders or game data. Set LEONIDA_VK_VALIDATION=1 to run under VK_LAYER_KHRONOS_validation; any validation error fails.
#include "vk_test_device.h"
#include "../shims/d3d12_com.h"

#include <cstdint>
#include <cstring>
#include <vector>

#define EXPECT(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); return 1; } \
} while (0)

namespace {

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

D3D12_RESOURCE_DESC1 desc1(const D3D12_RESOURCE_DESC& desc) {
    D3D12_RESOURCE_DESC1 out{};
    out.Dimension = desc.Dimension;
    out.Alignment = desc.Alignment;
    out.Width = desc.Width;
    out.Height = desc.Height;
    out.DepthOrArraySize = desc.DepthOrArraySize;
    out.MipLevels = desc.MipLevels;
    out.Format = desc.Format;
    out.SampleDesc = desc.SampleDesc;
    out.Layout = desc.Layout;
    out.Flags = desc.Flags;
    return out;
}

bool same_desc(const D3D12_RESOURCE_DESC& a, const D3D12_RESOURCE_DESC& b) {
    return a.Dimension == b.Dimension && a.Alignment == b.Alignment && a.Width == b.Width && a.Height == b.Height &&
           a.DepthOrArraySize == b.DepthOrArraySize && a.MipLevels == b.MipLevels && a.Format == b.Format &&
           a.SampleDesc.Count == b.SampleDesc.Count && a.SampleDesc.Quality == b.SampleDesc.Quality &&
           a.Layout == b.Layout && a.Flags == b.Flags;
}

// Never dereferenced: the device refuses any session before looking at it.
ID3D12ProtectedResourceSession* fake_session() {
    static int dummy;
    return reinterpret_cast<ID3D12ProtectedResourceSession*>(&dummy);
}

struct Case {
    const char* name;
    D3D12_HEAP_TYPE heap;
    D3D12_RESOURCE_DESC desc;
};

// A spread of what CreateCommittedResource accepts and refuses, so each variant is compared on both.
std::vector<Case> cases() {
    auto ms = texture_desc(8, 8, DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET);
    ms.SampleDesc.Count = 4;
    auto ms_mips = ms;
    ms_mips.MipLevels = 2;
    return {
        {"upload buffer", D3D12_HEAP_TYPE_UPLOAD, buffer_desc(4096)},
        {"readback buffer", D3D12_HEAP_TYPE_READBACK, buffer_desc(300)},
        {"default buffer", D3D12_HEAP_TYPE_DEFAULT, buffer_desc(65536)},
        {"render target", D3D12_HEAP_TYPE_DEFAULT,
         texture_desc(16, 8, DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET)},
        {"depth-stencil", D3D12_HEAP_TYPE_DEFAULT,
         texture_desc(8, 8, DXGI_FORMAT_D32_FLOAT_S8X24_UINT, D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL)},
        {"full mip chain", D3D12_HEAP_TYPE_DEFAULT,
         texture_desc(16, 16, DXGI_FORMAT_R16G16B16A16_FLOAT, D3D12_RESOURCE_FLAG_NONE, 0, 2)},
        {"BC7 array", D3D12_HEAP_TYPE_DEFAULT, texture_desc(8, 8, DXGI_FORMAT_BC7_UNORM, D3D12_RESOURCE_FLAG_NONE, 1, 3)},
        {"XR_BIAS texture", D3D12_HEAP_TYPE_DEFAULT,
         texture_desc(8, 8, DXGI_FORMAT_R10G10B10_XR_BIAS_A2_UNORM, D3D12_RESOURCE_FLAG_NONE)},
        {"volume", D3D12_HEAP_TYPE_DEFAULT, volume_desc(0)},
        {"multisampled", D3D12_HEAP_TYPE_DEFAULT, ms},
        // Refused by the base method; each variant must refuse them the same way.
        {"texture on upload heap", D3D12_HEAP_TYPE_UPLOAD,
         texture_desc(4, 4, DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_RESOURCE_FLAG_NONE)},
        {"multisampled with mips", D3D12_HEAP_TYPE_DEFAULT, ms_mips},
        {"unknown format", D3D12_HEAP_TYPE_DEFAULT,
         texture_desc(4, 4, DXGI_FORMAT_R1_UNORM, D3D12_RESOURCE_FLAG_NONE)},
    };
}

enum class Via { base, v1, v2, v3 };

HRESULT create(ID3D12Device11* device, Via via, const D3D12_HEAP_PROPERTIES& heap, const D3D12_RESOURCE_DESC& desc,
               void** out, ID3D12ProtectedResourceSession* session = nullptr) {
    const D3D12_RESOURCE_DESC1 d1 = desc1(desc);
    const IID& iid = __uuidof(ID3D12Resource);
    switch (via) {
    case Via::base:
        return device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COMMON,
                                               nullptr, iid, out);
    case Via::v1:
        return device->CreateCommittedResource1(&heap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COMMON,
                                                nullptr, session, iid, out);
    case Via::v2:
        return device->CreateCommittedResource2(&heap, D3D12_HEAP_FLAG_NONE, &d1, D3D12_RESOURCE_STATE_COMMON,
                                                nullptr, session, iid, out);
    case Via::v3:
        return device->CreateCommittedResource3(&heap, D3D12_HEAP_FLAG_NONE, &d1,
                                                desc.Dimension == D3D12_RESOURCE_DIMENSION_BUFFER
                                                    ? D3D12_BARRIER_LAYOUT_UNDEFINED
                                                    : D3D12_BARRIER_LAYOUT_COMMON,
                                                nullptr, session, 0, nullptr, iid, out);
    }
    return E_FAIL;
}

// Each variant against CreateCommittedResource on every case: the same result, and on success the same
// description, heap properties, image-or-buffer and Map answer. Returns the number of differences.
int compare_creation(ID3D12Device11* device) {
    int differences = 0;
    for (const Case& c : cases()) {
        const D3D12_HEAP_PROPERTIES heap{c.heap};
        Ref<ID3D12Resource> base;
        const HRESULT base_hr = create(device, Via::base, heap, c.desc, reinterpret_cast<void**>(&base.p));
        const HRESULT base_check = create(device, Via::base, heap, c.desc, nullptr);  // null output: a check
        for (const Via via : {Via::v1, Via::v2, Via::v3}) {
            Ref<ID3D12Resource> variant;
            const HRESULT hr = create(device, via, heap, c.desc, reinterpret_cast<void**>(&variant.p));
            const HRESULT check = create(device, via, heap, c.desc, nullptr);
            const auto differ = [&](const char* what) {
                if (differences++ < 24)
                    std::fprintf(stderr, "%s, CreateCommittedResource%d: %s differs\n", c.name, int(via), what);
            };
            if (hr != base_hr) differ("result");
            if (check != base_check) differ("null-output result");
            if (!base.p || !variant.p) {
                if (base.p || variant.p) differ("output");
                continue;
            }
            if (!same_desc(base->GetDesc(), variant->GetDesc())) differ("description");
            D3D12_HEAP_PROPERTIES a{}, b{};
            D3D12_HEAP_FLAGS fa{}, fb{};
            base->GetHeapProperties(&a, &fa);
            variant->GetHeapProperties(&b, &fb);
            if (a.Type != b.Type || fa != fb) differ("heap properties");
            if (!rt::d3d12::vulkan_image(base.p) != !rt::d3d12::vulkan_image(variant.p)) differ("image");
            void* ma = nullptr;
            void* mb = nullptr;
            if ((base->Map(0, nullptr, &ma) == S_OK) != (variant->Map(0, nullptr, &mb) == S_OK)) differ("Map");
        }
    }
    return differences;
}

bool same_footprints(ID3D12Device11* device, const D3D12_RESOURCE_DESC* desc, UINT first, UINT count, UINT64 base) {
    std::vector<D3D12_PLACED_SUBRESOURCE_FOOTPRINT> a(count), b(count);
    std::vector<UINT> rows_a(count), rows_b(count);
    std::vector<UINT64> size_a(count), size_b(count);
    UINT64 total_a = 0, total_b = 1;
    device->GetCopyableFootprints(desc, first, count, base, a.data(), rows_a.data(), size_a.data(), &total_a);
    const D3D12_RESOURCE_DESC1 d1 = desc ? desc1(*desc) : D3D12_RESOURCE_DESC1{};
    device->GetCopyableFootprints1(desc ? &d1 : nullptr, first, count, base, b.data(), rows_b.data(), size_b.data(),
                                   &total_b);
    for (UINT i = 0; i < count; ++i) {  // field by field: the struct has padding
        const D3D12_SUBRESOURCE_FOOTPRINT &x = a[i].Footprint, &y = b[i].Footprint;
        if (a[i].Offset != b[i].Offset || x.Format != y.Format || x.Width != y.Width || x.Height != y.Height ||
            x.Depth != y.Depth || x.RowPitch != y.RowPitch)
            return false;
    }
    return total_a == total_b && rows_a == rows_b && size_a == size_b;
}

struct Edge {
    const char* name;
    HRESULT got, want;  // `want` is the PC runtime's answer, measured on an RTX 4090 (driver 621.14)
};

// The edge cases of the newer arguments, run on any ID3D12Device10: ours, or the host's PC runtime to compare
// with. A GetCopyableFootprints1 answer is S_FALSE when every output is all ones (the failure answer), S_OK when
// the footprint is the ordinary one, E_FAIL otherwise.
std::vector<Edge> edge_answers(ID3D12Device10* device) {
    const D3D12_HEAP_PROPERTIES default_heap{D3D12_HEAP_TYPE_DEFAULT}, upload_heap{D3D12_HEAP_TYPE_UPLOAD};
    D3D12_RESOURCE_DESC1 texture = desc1(texture_desc(8, 8, DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_RESOURCE_FLAG_NONE));
    D3D12_RESOURCE_DESC1 buffer = desc1(buffer_desc(300));
    D3D12_RESOURCE_DESC1 texture_region = texture, buffer_region = buffer;
    texture_region.SamplerFeedbackMipRegion = buffer_region.SamplerFeedbackMipRegion = {4, 4, 1};
    const auto v2 = [&](const D3D12_HEAP_PROPERTIES& heap, const D3D12_RESOURCE_DESC1& desc) {
        Ref<ID3D12Resource> r;
        return device->CreateCommittedResource2(&heap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COMMON,
                                                nullptr, nullptr, IID_PPV_ARGS(&r.p));
    };
    const auto v3 = [&](const D3D12_HEAP_PROPERTIES& heap, const D3D12_RESOURCE_DESC1& desc, D3D12_BARRIER_LAYOUT layout,
                        UINT32 castable_count = 0, bool output = true) {
        Ref<ID3D12Resource> r;
        return device->CreateCommittedResource3(&heap, D3D12_HEAP_FLAG_NONE, &desc, layout, nullptr, nullptr,
                                                castable_count, nullptr, __uuidof(ID3D12Resource),
                                                output ? reinterpret_cast<void**>(&r.p) : nullptr);
    };
    const auto footprint = [&](const D3D12_RESOURCE_DESC1& desc) {
        D3D12_PLACED_SUBRESOURCE_FOOTPRINT layout{};
        UINT rows = 0;
        UINT64 row_size = 0, total = 0;
        device->GetCopyableFootprints1(&desc, 0, 1, 0, &layout, &rows, &row_size, &total);
        if (layout.Offset == ~0ull && rows == ~0u && row_size == ~0ull && total == ~0ull) return S_FALSE;
        return layout.Offset == 0 && rows != ~0u && total != ~0ull ? S_OK : E_FAIL;
    };
    const GUID creator{0x1d2c3b4a, 0x5f6e, 0x7a8b, {0x9c, 0x0d, 0x1e, 0x2f, 0x3a, 0x4b, 0x5c, 0x6d}};
    void* queue = nullptr;
    const HRESULT null_queue = device->CreateCommandQueue1(nullptr, creator, __uuidof(ID3D12CommandQueue), &queue);
    if (queue) static_cast<IUnknown*>(queue)->Release();
    return {
        {"CreateCommittedResource2, texture with a mip region", v2(default_heap, texture_region), E_INVALIDARG},
        {"CreateCommittedResource2, buffer with a mip region", v2(upload_heap, buffer_region), S_OK},
        {"CreateCommittedResource3, texture with a mip region",
         v3(default_heap, texture_region, D3D12_BARRIER_LAYOUT_COMMON), E_INVALIDARG},
        {"CreateCommittedResource3, buffer with a mip region",
         v3(upload_heap, buffer_region, D3D12_BARRIER_LAYOUT_UNDEFINED), S_OK},
        {"CreateCommittedResource3, buffer in layout COMMON", v3(upload_heap, buffer, D3D12_BARRIER_LAYOUT_COMMON),
         E_INVALIDARG},
        {"CreateCommittedResource3, texture in layout UNDEFINED",
         v3(default_heap, texture, D3D12_BARRIER_LAYOUT_UNDEFINED), S_OK},
        {"CreateCommittedResource3, castable count with a null list",
         v3(default_heap, texture, D3D12_BARRIER_LAYOUT_COMMON, 2), S_OK},
        {"CreateCommittedResource3, null output",
         v3(default_heap, texture, D3D12_BARRIER_LAYOUT_COMMON, 0, false), S_FALSE},
        {"CreateCommandQueue1, null description", null_queue, E_INVALIDARG},
        {"GetCopyableFootprints1, texture with a mip region", footprint(texture_region), S_FALSE},
        {"GetCopyableFootprints1, buffer with a mip region", footprint(buffer_region), S_OK},
    };
}

}  // namespace

int main() {
    vktest::Device d;
    const auto device_status = d.initialize();
    if (device_status == vktest::DeviceStatus::unavailable) return 77;   // no Vulkan GPU: ctest skips
    if (device_status != vktest::DeviceStatus::ready) return 1;
    {
        rt::d3d12::VulkanDeviceInfo info{};
        info.get_instance_proc = d.instance_proc;
        info.instance = d.instance;
        info.physical = d.physical;
        info.device = d.device;
        info.queue = d.queue;
        info.queue_family = d.family;
        Ref<ID3D12Device11> device;
        EXPECT(SUCCEEDED(rt::d3d12::create_vulkan_device(info, IID_PPV_ARGS(&device.p))));

        // CreateCommittedResource1/2/3 against CreateCommittedResource.
        const int differences = compare_creation(device.p);
        EXPECT(differences == 0);

        // A protected session is refused by every variant (the layer reports no session support); nothing is
        // handed out.
        const D3D12_HEAP_PROPERTIES default_heap{D3D12_HEAP_TYPE_DEFAULT}, upload_heap{D3D12_HEAP_TYPE_UPLOAD},
            readback_heap{D3D12_HEAP_TYPE_READBACK};
        for (const Via via : {Via::v1, Via::v2, Via::v3}) {
            void* out = reinterpret_cast<void*>(1);
            EXPECT(create(device.p, via, upload_heap, buffer_desc(256), &out, fake_session()) == E_INVALIDARG);
            EXPECT(out == nullptr);
        }

        // The newer arguments' edge cases, each with the answer the PC runtime gives (and, when the host has a
        // D3D12 device, compared with it live).
        const std::vector<Edge> ours = edge_answers(device.p);
        for (const Edge& e : ours) {
            if (e.got == e.want) continue;
            std::fprintf(stderr, "%s: got %08lx, want %08lx\n", e.name, e.got, e.want);
            return 1;
        }
        {
            const HMODULE d3d12 = LoadLibraryExA("d3d12.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
            const auto create_host = d3d12 ? reinterpret_cast<PFN_D3D12_CREATE_DEVICE>(
                                                 reinterpret_cast<void*>(GetProcAddress(d3d12, "D3D12CreateDevice")))
                                           : nullptr;
            Ref<ID3D12Device10> host;
            if (create_host && SUCCEEDED(create_host(nullptr, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&host.p)))) {
                const std::vector<Edge> theirs = edge_answers(host.p);
                for (std::size_t i = 0; i < ours.size(); ++i) {
                    if (ours[i].got == theirs[i].got) continue;
                    std::fprintf(stderr, "%s: ours %08lx, host runtime %08lx\n", ours[i].name, ours[i].got,
                                 theirs[i].got);
                    return 1;
                }
            } else {
                std::printf("no host D3D12 device: edge cases checked against their recorded answers only\n");
            }
        }

        // Castable formats: the layer reports relaxed format casting and passes a real list to
        // VkImageFormatListCreateInfo. A count with a null list is still no list; a buffer cannot have one.
        {
            D3D12_FEATURE_DATA_D3D12_OPTIONS12 options12{};
            EXPECT(SUCCEEDED(device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS12, &options12,
                                                         sizeof(options12))));
            EXPECT(options12.RelaxedFormatCastingSupported == TRUE);
            EXPECT(device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS12, &options12,
                                               sizeof(options12) - 1) == E_INVALIDARG);
            const D3D12_RESOURCE_DESC1 texture =
                desc1(texture_desc(4, 4, DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_RESOURCE_FLAG_NONE));
            const DXGI_FORMAT own[] = {DXGI_FORMAT_R8G8B8A8_UNORM, DXGI_FORMAT_R8G8B8A8_UNORM_SRGB};
            Ref<ID3D12Resource> r;
            EXPECT(SUCCEEDED(device->CreateCommittedResource3(&default_heap, D3D12_HEAP_FLAG_NONE, &texture,
                                                              D3D12_BARRIER_LAYOUT_COMMON, nullptr, nullptr, 2, own,
                                                              IID_PPV_ARGS(&r.p))));
            EXPECT(rt::d3d12::vulkan_image(r.p));
            r.reset();
            const D3D12_RESOURCE_DESC1 buffer = desc1(buffer_desc(256));
            EXPECT(device->CreateCommittedResource3(&upload_heap, D3D12_HEAP_FLAG_NONE, &buffer,
                                                    D3D12_BARRIER_LAYOUT_UNDEFINED, nullptr, nullptr, 2, own,
                                                    IID_PPV_ARGS(&r.p)) == E_INVALIDARG && !r.p);
            const DXGI_FORMAT unknown[] = {DXGI_FORMAT_UNKNOWN};
            EXPECT(device->CreateCommittedResource3(&default_heap, D3D12_HEAP_FLAG_NONE, &texture,
                                                    D3D12_BARRIER_LAYOUT_COMMON, nullptr, nullptr, 1, unknown,
                                                    IID_PPV_ARGS(&r.p)) == E_INVALIDARG && !r.p);
        }

        // The creation-time castable list is also the view allow-list. An unlisted SRGB view leaves the slot null;
        // a listed same-size cross-family cast writes a descriptor and creates its Vulkan view.
        {
            Ref<ID3D12DescriptorHeap> heap;
            D3D12_DESCRIPTOR_HEAP_DESC heap_desc{};
            heap_desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
            heap_desc.NumDescriptors = 3;
            heap_desc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
            EXPECT(SUCCEEDED(device->CreateDescriptorHeap(&heap_desc, IID_PPV_ARGS(&heap.p))));
            D3D12_CPU_DESCRIPTOR_HANDLE slot = heap->GetCPUDescriptorHandleForHeapStart();
            const UINT stride = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

            const D3D12_RESOURCE_DESC1 unorm_desc =
                desc1(texture_desc(4, 4, DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_RESOURCE_FLAG_NONE));
            const DXGI_FORMAT unorm_only[] = {DXGI_FORMAT_R8G8B8A8_UNORM};
            Ref<ID3D12Resource> unorm_texture;
            EXPECT(SUCCEEDED(device->CreateCommittedResource3(&default_heap, D3D12_HEAP_FLAG_NONE, &unorm_desc,
                                                              D3D12_BARRIER_LAYOUT_COMMON, nullptr, nullptr, 1, unorm_only,
                                                              IID_PPV_ARGS(&unorm_texture.p))));
            D3D12_SHADER_RESOURCE_VIEW_DESC srv{};
            srv.Format = DXGI_FORMAT_R8G8B8A8_UNORM_SRGB;
            srv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
            srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
            srv.Texture2D.MipLevels = 1;
            device->CreateShaderResourceView(unorm_texture.p, &srv, slot);
            std::uint32_t words[8]{};
            std::memcpy(words, reinterpret_cast<void*>(slot.ptr), sizeof(words));
            bool zero = true;
            for (std::uint32_t word : words) zero = zero && word == 0;
            EXPECT(zero);

            // The resource's own format is viewable even when the castable list omits it. Vulkan still needs it
            // in VkImageFormatListCreateInfo, so creation adds it; the default-format SRV must write a descriptor.
            const DXGI_FORMAT srgb_only[] = {DXGI_FORMAT_R8G8B8A8_UNORM_SRGB};
            Ref<ID3D12Resource> self_format_texture;
            EXPECT(SUCCEEDED(device->CreateCommittedResource3(&default_heap, D3D12_HEAP_FLAG_NONE, &unorm_desc,
                                                              D3D12_BARRIER_LAYOUT_COMMON, nullptr, nullptr, 1,
                                                              srgb_only, IID_PPV_ARGS(&self_format_texture.p))));
            srv.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
            slot.ptr += stride;
            device->CreateShaderResourceView(self_format_texture.p, &srv, slot);
            std::memcpy(words, reinterpret_cast<void*>(slot.ptr), sizeof(words));
            bool nonzero = false;
            for (std::uint32_t word : words) nonzero = nonzero || word != 0;
            EXPECT(nonzero);

            const D3D12_RESOURCE_DESC1 uint_desc =
                desc1(texture_desc(4, 4, DXGI_FORMAT_R8G8B8A8_UINT, D3D12_RESOURCE_FLAG_NONE));
            const DXGI_FORMAT uint_views[] = {DXGI_FORMAT_R8G8B8A8_UINT, DXGI_FORMAT_R32_UINT};
            Ref<ID3D12Resource> uint_texture;
            EXPECT(SUCCEEDED(device->CreateCommittedResource3(&default_heap, D3D12_HEAP_FLAG_NONE, &uint_desc,
                                                              D3D12_BARRIER_LAYOUT_COMMON, nullptr, nullptr, 2, uint_views,
                                                              IID_PPV_ARGS(&uint_texture.p))));
            srv.Format = DXGI_FORMAT_R32_UINT;
            slot.ptr += stride;
            device->CreateShaderResourceView(uint_texture.p, &srv, slot);
            std::memcpy(words, reinterpret_cast<void*>(slot.ptr), sizeof(words));
            nonzero = false;
            for (std::uint32_t word : words) nonzero = nonzero || word != 0;
            EXPECT(nonzero);
        }

        // A texture from CreateCommittedResource3 (a castable count with a null list, which is no list) is
        // uploaded, copied back and compared on a queue from CreateCommandQueue1.
        Ref<ID3D12CommandQueue> queue, queue1;
        D3D12_COMMAND_QUEUE_DESC queue_desc{};
        queue_desc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
        queue_desc.Priority = D3D12_COMMAND_QUEUE_PRIORITY_HIGH;
        EXPECT(SUCCEEDED(device->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&queue.p))));
        // CreateCommandQueue1 against CreateCommandQueue: the same description, a working queue, the same refusal
        // of a missing description.
        const GUID creator{0x1d2c3b4a, 0x5f6e, 0x7a8b, {0x9c, 0x0d, 0x1e, 0x2f, 0x3a, 0x4b, 0x5c, 0x6d}};
        EXPECT(SUCCEEDED(device->CreateCommandQueue1(&queue_desc, creator, IID_PPV_ARGS(&queue1.p))));
        {
            const D3D12_COMMAND_QUEUE_DESC a = queue->GetDesc(), b = queue1->GetDesc();
            EXPECT(a.Type == b.Type && a.Priority == b.Priority && a.Flags == b.Flags && a.NodeMask == b.NodeMask);
            void* out = nullptr;
            EXPECT(device->CreateCommandQueue(nullptr, __uuidof(ID3D12CommandQueue), &out) ==
                   device->CreateCommandQueue1(nullptr, creator, __uuidof(ID3D12CommandQueue), &out));
        }

        constexpr UINT size = 8;
        const D3D12_RESOURCE_DESC1 castable_desc =
            desc1(texture_desc(size, size, DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET));
        Ref<ID3D12Resource> texture, upload, readback;
        EXPECT(SUCCEEDED(device->CreateCommittedResource3(&default_heap, D3D12_HEAP_FLAG_NONE, &castable_desc,
                                                          D3D12_BARRIER_LAYOUT_COPY_DEST, nullptr, nullptr, 3, nullptr,
                                                          IID_PPV_ARGS(&texture.p))));
        EXPECT(rt::d3d12::vulkan_image(texture.p));
        const D3D12_RESOURCE_DESC1 bytes = desc1(buffer_desc(4096));
        EXPECT(SUCCEEDED(device->CreateCommittedResource3(&upload_heap, D3D12_HEAP_FLAG_NONE, &bytes,
                                                          D3D12_BARRIER_LAYOUT_UNDEFINED, nullptr, nullptr, 0, nullptr,
                                                          IID_PPV_ARGS(&upload.p))));
        EXPECT(SUCCEEDED(device->CreateCommittedResource2(&readback_heap, D3D12_HEAP_FLAG_NONE, &bytes,
                                                          D3D12_RESOURCE_STATE_COPY_DEST, nullptr, nullptr,
                                                          IID_PPV_ARGS(&readback.p))));
        D3D12_PLACED_SUBRESOURCE_FOOTPRINT placed{};
        device->GetCopyableFootprints1(&castable_desc, 0, 1, 0, &placed, nullptr, nullptr, nullptr);
        std::uint8_t* up = nullptr;
        std::uint8_t* down = nullptr;
        EXPECT(SUCCEEDED(upload->Map(0, nullptr, reinterpret_cast<void**>(&up))));
        EXPECT(SUCCEEDED(readback->Map(0, nullptr, reinterpret_cast<void**>(&down))));
        for (UINT y = 0; y < size; ++y)
            for (UINT x = 0; x < size * 4; ++x) up[placed.Offset + y * placed.Footprint.RowPitch + x] = std::uint8_t(y * 32 + x);
        std::memset(down, 0xcd, 4096);

        Ref<ID3D12CommandAllocator> allocator;
        Ref<ID3D12GraphicsCommandList> list;
        Ref<ID3D12Fence> fence;
        EXPECT(SUCCEEDED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator.p))));
        EXPECT(SUCCEEDED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.p, nullptr,
                                                   IID_PPV_ARGS(&list.p))));
        EXPECT(SUCCEEDED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence.p))));
        D3D12_TEXTURE_COPY_LOCATION from_buffer{}, to_texture{}, from_texture{}, to_buffer{};
        from_buffer.pResource = upload.p;
        from_buffer.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
        from_buffer.PlacedFootprint = placed;
        to_texture.pResource = texture.p;
        to_texture.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        from_texture = to_texture;
        to_buffer = from_buffer;
        to_buffer.pResource = readback.p;
        list->CopyTextureRegion(&to_texture, 0, 0, 0, &from_buffer, nullptr);
        D3D12_RESOURCE_BARRIER barrier{};
        barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        barrier.Transition.pResource = texture.p;
        barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
        barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
        list->ResourceBarrier(1, &barrier);
        list->CopyTextureRegion(&to_buffer, 0, 0, 0, &from_texture, nullptr);
        EXPECT(list->Close() == S_OK);
        ID3D12CommandList* lists[] = {list.p};
        queue1->ExecuteCommandLists(1, lists);  // on the queue CreateCommandQueue1 made
        EXPECT(rt::d3d12::last_execute_result(queue1.p) == leonida::ReplayResult::success);
        EXPECT(SUCCEEDED(queue1->Signal(fence.p, 1)));
        EXPECT(SUCCEEDED(fence->SetEventOnCompletion(1, nullptr)) && fence->GetCompletedValue() == 1);
        unsigned wrong = 0;
        for (UINT y = 0; y < size; ++y) {
            const std::size_t row = std::size_t(placed.Offset + y * placed.Footprint.RowPitch);
            wrong += std::memcmp(down + row, up + row, size * 4) != 0;
        }
        EXPECT(wrong == 0);
        upload->Unmap(0, nullptr);
        readback->Unmap(0, nullptr);

        // GetCopyableFootprints1 against GetCopyableFootprints, byte for byte, including the failure answer.
        struct Footprints {
            D3D12_RESOURCE_DESC desc;
            UINT first, count;
        };
        const Footprints footprints[] = {
            {buffer_desc(300), 0, 1},
            {buffer_desc(300), 0, 2},  // refused: a buffer has one subresource
            {volume_desc(5), 0, 5},
            {texture_desc(8, 8, DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_RESOURCE_FLAG_NONE, 4, 2), 0, 8},
            {texture_desc(10, 6, DXGI_FORMAT_BC1_UNORM, D3D12_RESOURCE_FLAG_NONE, 4), 1, 3},
            {texture_desc(8, 8, DXGI_FORMAT_D24_UNORM_S8_UINT, D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL, 2, 2), 0, 8},
            {texture_desc(300, 5, DXGI_FORMAT_D32_FLOAT_S8X24_UINT, D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL), 0, 2},
            {texture_desc(4, 4, DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_RESOURCE_FLAG_NONE), 0, 2},  // refused: too many
        };
        for (const Footprints& f : footprints)
            for (const UINT64 base : {UINT64(0), UINT64(1024)})
                EXPECT(same_footprints(device.p, &f.desc, f.first, f.count, base));
        EXPECT(same_footprints(device.p, nullptr, 0, 1, 0));
    }
    if (d.validation_errors()) {
        std::fprintf(stderr, "%u validation errors\n", d.validation_errors());
        return 1;
    }
    std::printf("D3D12 method variants: CreateCommittedResource1/2/3, CreateCommandQueue1 and GetCopyableFootprints1 "
                "matched their base methods and the PC runtime's edge-case answers; a texture round-tripped\n");
    return 0;
}
