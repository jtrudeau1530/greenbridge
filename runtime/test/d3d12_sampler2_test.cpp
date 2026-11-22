// ID3D12Device11::CreateSampler2 on the Vulkan-backed device (D3D-28, part 2), compared with CreateSampler: the S#
// bytes it writes for descriptions CreateSampler can express, for UINT border colours (the S#'s border types are
// the same for integers) and for non-normalized coordinates (CreateSampler's S# with FORCE_UNNORMALIZED). Which
// descriptions the PC runtime refuses, and the D3D12_OPTIONS17 answer, are compared with the host's runtime live
// when there is one: a refused description leaves the slot as it was. No shaders or game data. Set
// LEONIDA_VK_VALIDATION=1 to run under VK_LAYER_KHRONOS_validation; any validation error fails.
#include "vk_test_device.h"
#include "../shims/d3d12_com.h"
#include "../src/gfx10_descriptors.h"

#include <array>
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
    ~Ref() {
        if (p) p->Release();
    }
    T* operator->() const { return p; }
};

constexpr unsigned char kUnwritten = 0xcd;

// A CPU sampler heap on any device, its slots filled with kUnwritten before each write.
struct Slots {
    Ref<ID3D12DescriptorHeap> heap;
    SIZE_T start = 0;
    UINT size = 0;
    UINT next = 0;
    static constexpr UINT kCount = 256;

    bool create(ID3D12Device* device) {
        const D3D12_DESCRIPTOR_HEAP_DESC desc{D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER, kCount,
                                              D3D12_DESCRIPTOR_HEAP_FLAG_NONE, 0};
        if (FAILED(device->CreateDescriptorHeap(&desc, IID_PPV_ARGS(&heap.p)))) return false;
        start = heap->GetCPUDescriptorHandleForHeapStart().ptr;
        size = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER);
        return start && size;
    }
    D3D12_CPU_DESCRIPTOR_HANDLE take() {
        const D3D12_CPU_DESCRIPTOR_HANDLE h{start + SIZE_T(next++ % kCount) * size};
        std::memset(reinterpret_cast<void*>(h.ptr), kUnwritten, size);
        return h;
    }
    std::vector<unsigned char> bytes(D3D12_CPU_DESCRIPTOR_HANDLE h) const {
        const auto* p = reinterpret_cast<const unsigned char*>(h.ptr);
        return {p, p + size};
    }
    bool unwritten(D3D12_CPU_DESCRIPTOR_HANDLE h) const {
        for (const unsigned char b : bytes(h))
            if (b != kUnwritten) return false;
        return true;
    }
};

D3D12_SAMPLER_DESC sampler(D3D12_FILTER filter, D3D12_TEXTURE_ADDRESS_MODE address, FLOAT max_lod = 1000.0f) {
    D3D12_SAMPLER_DESC d{};
    d.Filter = filter;
    d.AddressU = d.AddressV = d.AddressW = address;
    d.MaxAnisotropy = 1;
    d.ComparisonFunc = D3D12_COMPARISON_FUNC_NEVER;
    d.MaxLOD = max_lod;
    return d;
}

D3D12_SAMPLER_DESC2 desc2(const D3D12_SAMPLER_DESC& d, D3D12_SAMPLER_FLAGS flags = D3D12_SAMPLER_FLAG_NONE) {
    D3D12_SAMPLER_DESC2 out{};
    out.Filter = d.Filter;
    out.AddressU = d.AddressU;
    out.AddressV = d.AddressV;
    out.AddressW = d.AddressW;
    out.MipLODBias = d.MipLODBias;
    out.MaxAnisotropy = d.MaxAnisotropy;
    out.ComparisonFunc = d.ComparisonFunc;
    std::memcpy(out.FloatBorderColor, d.BorderColor, sizeof(out.FloatBorderColor));
    out.MinLOD = d.MinLOD;
    out.MaxLOD = d.MaxLOD;
    out.Flags = flags;
    return out;
}

D3D12_SAMPLER_DESC with_border(D3D12_SAMPLER_DESC d, FLOAT r, FLOAT g, FLOAT b, FLOAT a) {
    d.BorderColor[0] = r;
    d.BorderColor[1] = g;
    d.BorderColor[2] = b;
    d.BorderColor[3] = a;
    return d;
}

D3D12_SAMPLER_DESC2 with_uint_border(D3D12_SAMPLER_DESC2 d, UINT r, UINT g, UINT b, UINT a) {
    d.Flags = D3D12_SAMPLER_FLAGS(d.Flags | D3D12_SAMPLER_FLAG_UINT_BORDER_COLOR);
    d.UintBorderColor[0] = r;
    d.UintBorderColor[1] = g;
    d.UintBorderColor[2] = b;
    d.UintBorderColor[3] = a;
    return d;
}

// Descriptions CreateSampler expresses: CreateSampler2 without flags must write the same bytes.
std::vector<D3D12_SAMPLER_DESC> plain_cases() {
    const D3D12_SAMPLER_DESC border = sampler(D3D12_FILTER_MIN_MAG_MIP_LINEAR, D3D12_TEXTURE_ADDRESS_MODE_BORDER);
    D3D12_SAMPLER_DESC aniso = sampler(D3D12_FILTER_ANISOTROPIC, D3D12_TEXTURE_ADDRESS_MODE_WRAP);
    aniso.MaxAnisotropy = 8;
    aniso.MipLODBias = -1.5f;
    D3D12_SAMPLER_DESC shadow =
        sampler(D3D12_FILTER_COMPARISON_MIN_MAG_LINEAR_MIP_POINT, D3D12_TEXTURE_ADDRESS_MODE_CLAMP, 4.0f);
    shadow.ComparisonFunc = D3D12_COMPARISON_FUNC_LESS_EQUAL;
    D3D12_SAMPLER_DESC lods = sampler(D3D12_FILTER_MIN_POINT_MAG_LINEAR_MIP_POINT, D3D12_TEXTURE_ADDRESS_MODE_MIRROR);
    lods.MinLOD = 1.25f;
    lods.MaxLOD = 6.5f;
    D3D12_SAMPLER_DESC mixed = sampler(D3D12_FILTER_MAXIMUM_MIN_MAG_MIP_LINEAR, D3D12_TEXTURE_ADDRESS_MODE_WRAP);
    mixed.AddressV = D3D12_TEXTURE_ADDRESS_MODE_MIRROR_ONCE;
    mixed.AddressW = D3D12_TEXTURE_ADDRESS_MODE_BORDER;
    return {
        border,
        with_border(border, 0, 0, 0, 1),
        with_border(border, 1, 1, 1, 1),
        with_border(border, 0.25f, 0.5f, 0, 1),
        aniso,
        shadow,
        lods,
        mixed,
        sampler(D3D12_FILTER_MIN_MAG_MIP_POINT, D3D12_TEXTURE_ADDRESS_MODE_CLAMP, 0.0f),
        sampler(D3D12_FILTER_MINIMUM_MIN_MAG_MIP_POINT, D3D12_TEXTURE_ADDRESS_MODE_CLAMP),
    };
}

struct Edge {
    const char* name;
    bool written, want;  // `want`: whether the PC runtime writes the slot (RTX 4090, driver 621.14, and WARP)
};

// Which descriptions are written and which leave the slot as it was, on any ID3D12Device11: ours, or the host's PC
// runtime to compare with.
std::vector<Edge> edge_answers(ID3D12Device11* device, Slots& slots) {
    const auto written = [&](const D3D12_SAMPLER_DESC2* d) {
        const D3D12_CPU_DESCRIPTOR_HANDLE h = slots.take();
        device->CreateSampler2(d, h);
        return !slots.unwritten(h);
    };
    const D3D12_SAMPLER_DESC2 point = desc2(sampler(D3D12_FILTER_MIN_MAG_MIP_POINT, D3D12_TEXTURE_ADDRESS_MODE_CLAMP, 0),
                                            D3D12_SAMPLER_FLAG_NON_NORMALIZED_COORDINATES);
    const auto nn = [&](auto change) {
        D3D12_SAMPLER_DESC2 d = point;
        change(d);
        return written(&d);
    };
    using D = D3D12_SAMPLER_DESC2;
    D unknown = point;
    unknown.Flags = D3D12_SAMPLER_FLAGS(0x4);
    D both = with_uint_border(point, 0, 0, 0, 1);
    both.AddressU = both.AddressV = both.AddressW = D3D12_TEXTURE_ADDRESS_MODE_BORDER;
    D uint_custom = with_uint_border(desc2(sampler(D3D12_FILTER_MIN_MAG_MIP_LINEAR, D3D12_TEXTURE_ADDRESS_MODE_BORDER)),
                                     7, 0x40000000, 1, 1);
    return {
        {"null description", written(nullptr), false},
        {"unknown flag 0x4", written(&unknown), false},
        {"UINT border, custom colour", written(&uint_custom), true},
        {"UINT border and non-normalized", written(&both), true},
        {"non-normalized, point, clamp, LOD 0", written(&point), true},
        {"non-normalized, linear, mip point",
         nn([](D& d) { d.Filter = D3D12_FILTER_MIN_MAG_LINEAR_MIP_POINT; }), true},
        {"non-normalized, mip linear", nn([](D& d) { d.Filter = D3D12_FILTER_MIN_MAG_MIP_LINEAR; }), false},
        {"non-normalized, min point mag linear",
         nn([](D& d) { d.Filter = D3D12_FILTER_MIN_POINT_MAG_LINEAR_MIP_POINT; }), false},
        {"non-normalized, anisotropic", nn([](D& d) { d.Filter = D3D12_FILTER_ANISOTROPIC; d.MaxAnisotropy = 8; }),
         false},
        {"non-normalized, comparison",
         nn([](D& d) { d.Filter = D3D12_FILTER_COMPARISON_MIN_MAG_MIP_POINT; d.ComparisonFunc = D3D12_COMPARISON_FUNC_LESS; }),
         false},
        {"non-normalized, minimum", nn([](D& d) { d.Filter = D3D12_FILTER_MINIMUM_MIN_MAG_MIP_POINT; }), true},
        {"non-normalized, maximum linear",
         nn([](D& d) { d.Filter = D3D12_FILTER_MAXIMUM_MIN_MAG_LINEAR_MIP_POINT; }), true},
        {"non-normalized, MaxAnisotropy 16 on a point filter", nn([](D& d) { d.MaxAnisotropy = 16; }), true},
        {"non-normalized, ComparisonFunc on a point filter",
         nn([](D& d) { d.ComparisonFunc = D3D12_COMPARISON_FUNC_LESS; }), true},
        {"non-normalized, AddressU wrap", nn([](D& d) { d.AddressU = D3D12_TEXTURE_ADDRESS_MODE_WRAP; }), false},
        {"non-normalized, AddressV mirror", nn([](D& d) { d.AddressV = D3D12_TEXTURE_ADDRESS_MODE_MIRROR; }), false},
        {"non-normalized, AddressW wrap", nn([](D& d) { d.AddressW = D3D12_TEXTURE_ADDRESS_MODE_WRAP; }), true},
        {"non-normalized, border", nn([](D& d) { d.AddressU = D3D12_TEXTURE_ADDRESS_MODE_BORDER; }), true},
        {"non-normalized, MaxLOD 5", nn([](D& d) { d.MaxLOD = 5; }), false},
        {"non-normalized, MaxLOD -0", nn([](D& d) { d.MaxLOD = -0.0f; }), true},
        {"non-normalized, MinLOD and MaxLOD 1", nn([](D& d) { d.MinLOD = d.MaxLOD = 1; }), false},
        {"non-normalized, MipLODBias 1", nn([](D& d) { d.MipLODBias = 1; }), false},
    };
}

leonida::gfx10::SamplerWords words_of(const std::vector<unsigned char>& bytes) {
    leonida::gfx10::SamplerWords w{};
    std::memcpy(w.data(), bytes.data(), sizeof(w));
    return w;
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
        Slots slots;
        EXPECT(slots.create(device.p));

        const auto via1 = [&](const D3D12_SAMPLER_DESC& desc) {
            const D3D12_CPU_DESCRIPTOR_HANDLE h = slots.take();
            device->CreateSampler(&desc, h);
            return slots.bytes(h);
        };
        const auto via2 = [&](const D3D12_SAMPLER_DESC2& desc) {
            const D3D12_CPU_DESCRIPTOR_HANDLE h = slots.take();
            device->CreateSampler2(&desc, h);
            return slots.bytes(h);
        };

        // Without flags, the same bytes as CreateSampler.
        int plain = 0;
        for (const D3D12_SAMPLER_DESC& desc : plain_cases()) {
            EXPECT(via2(desc2(desc)) == via1(desc));
            ++plain;
        }

        // A comparison that always returns zero still needs a comparison-enabled Vulkan sampler,
        // alongside an ordinary sampler with otherwise identical settings in the same set.
        const auto regular = sampler(D3D12_FILTER_MIN_MAG_MIP_POINT, D3D12_TEXTURE_ADDRESS_MODE_CLAMP);
        auto comparison = regular;
        comparison.Filter = D3D12_FILTER_COMPARISON_MIN_MAG_MIP_POINT;
        const auto regular_words = words_of(via1(regular));
        const auto never_words = words_of(via1(comparison));
        EXPECT(never_words != regular_words);
        EXPECT(words_of(via2(desc2(comparison))) == never_words);
        const auto regular_info = leonida::gfx10::vulkan_sampler(leonida::gfx10::decode(regular_words));
        const auto never_info = leonida::gfx10::vulkan_sampler(leonida::gfx10::decode(never_words));
        EXPECT(!regular_info.compareEnable);
        EXPECT(never_info.compareEnable && never_info.compareOp == VK_COMPARE_OP_NEVER);

        // UINT border colours take the S#'s border types as the float ones do, marked as integers in
        // BORDER_COLOR_PTR (kIntegerBorderPtr), so that they are their own key and VK_BORDER_COLOR_INT_* in Vulkan.
        // Anything else is the application's colour, including the bits of 1.0f, which as integers are not white.
        const D3D12_SAMPLER_DESC border = sampler(D3D12_FILTER_MIN_MAG_MIP_LINEAR, D3D12_TEXTURE_ADDRESS_MODE_BORDER);
        const D3D12_SAMPLER_DESC2 border2 = desc2(border);
        const auto marked = [](std::vector<unsigned char> bytes) {
            leonida::gfx10::SamplerWords w = words_of(bytes);
            w[3] |= leonida::gfx10::kIntegerBorderPtr;
            std::memcpy(bytes.data(), w.data(), sizeof(w));
            return bytes;
        };
        for (const std::array<UINT, 4> c : {std::array<UINT, 4>{0, 0, 0, 0}, {0, 0, 0, 1}, {1, 1, 1, 1}}) {
            const std::vector<unsigned char> got = via2(with_uint_border(border2, c[0], c[1], c[2], c[3]));
            EXPECT(got == marked(via1(with_border(border, FLOAT(c[0]), FLOAT(c[1]), FLOAT(c[2]), FLOAT(c[3])))));
            const leonida::gfx10::Sampler s = leonida::gfx10::decode(words_of(got));
            EXPECT(s.integer_border && s.border_color_index == 0);
            EXPECT(leonida::gfx10::vulkan_sampler(s).borderColor ==
                   (c[3] == 0   ? VK_BORDER_COLOR_INT_TRANSPARENT_BLACK
                    : c[0] == 0 ? VK_BORDER_COLOR_INT_OPAQUE_BLACK
                                : VK_BORDER_COLOR_INT_OPAQUE_WHITE));
        }
        EXPECT(via2(with_uint_border(border2, 7, 9, 0, 1)) == via1(with_border(border, 0.25f, 0, 0, 1)));
        const UINT one = 0x3f800000;
        EXPECT(via2(with_uint_border(border2, one, one, one, one)) == via1(with_border(border, 0.25f, 0, 0, 1)));
        // ...and without the flag, the union's bits are floats.
        D3D12_SAMPLER_DESC2 float_bits = border2;
        for (UINT& c : float_bits.UintBorderColor) c = one;
        EXPECT(via2(float_bits) == via1(with_border(border, 1, 1, 1, 1)));

        // Non-normalized coordinates: CreateSampler's S# with FORCE_UNNORMALIZED set, nothing else changed.
        for (const D3D12_FILTER filter : {D3D12_FILTER_MIN_MAG_MIP_POINT, D3D12_FILTER_MIN_MAG_LINEAR_MIP_POINT,
                                          D3D12_FILTER_MINIMUM_MIN_MAG_MIP_POINT}) {
            D3D12_SAMPLER_DESC desc = sampler(filter, D3D12_TEXTURE_ADDRESS_MODE_CLAMP, 0.0f);
            desc.AddressV = D3D12_TEXTURE_ADDRESS_MODE_BORDER;
            desc.AddressW = D3D12_TEXTURE_ADDRESS_MODE_WRAP;
            const leonida::gfx10::SamplerWords normalized = words_of(via1(desc));
            const leonida::gfx10::SamplerWords got =
                words_of(via2(desc2(desc, D3D12_SAMPLER_FLAG_NON_NORMALIZED_COORDINATES)));
            leonida::gfx10::SamplerWords want = normalized;
            want[0] |= 1u << 15;
            EXPECT(got == want);
            EXPECT(leonida::gfx10::decode(got).unnormalized);
            EXPECT(!leonida::gfx10::decode(normalized).unnormalized);
        }

        // A null handle is ignored.
        const D3D12_SAMPLER_DESC2 any = desc2(border);
        device->CreateSampler2(&any, D3D12_CPU_DESCRIPTOR_HANDLE{0});

        // D3D12_OPTIONS17: non-normalized samplers, no manual write tracking.
        D3D12_FEATURE_DATA_D3D12_OPTIONS17 ours17{};
        EXPECT(SUCCEEDED(device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS17, &ours17, sizeof(ours17))));
        EXPECT(ours17.NonNormalizedCoordinateSamplersSupported && !ours17.ManualWriteTrackingResourceSupported);

        // The edge cases, each with the PC runtime's answer, and compared with it live when the host has a device.
        const std::vector<Edge> ours = edge_answers(device.p, slots);
        for (const Edge& e : ours) {
            if (e.written == e.want) continue;
            std::fprintf(stderr, "%s: written %d, want %d\n", e.name, e.written, e.want);
            return 1;
        }
        {
            const HMODULE d3d12 = LoadLibraryExA("d3d12.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
            const auto create_host = d3d12 ? reinterpret_cast<PFN_D3D12_CREATE_DEVICE>(
                                                 reinterpret_cast<void*>(GetProcAddress(d3d12, "D3D12CreateDevice")))
                                           : nullptr;
            Ref<ID3D12Device11> host;
            Slots host_slots;
            if (create_host && SUCCEEDED(create_host(nullptr, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&host.p))) &&
                host_slots.create(host.p)) {
                D3D12_FEATURE_DATA_D3D12_OPTIONS17 theirs17{};
                EXPECT(SUCCEEDED(host->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS17, &theirs17,
                                                           sizeof(theirs17))));
                if (!theirs17.NonNormalizedCoordinateSamplersSupported)
                    std::printf("host reports no non-normalized samplers: its answers may differ from the layer's\n");
                EXPECT(!theirs17.NonNormalizedCoordinateSamplersSupported || ours17.NonNormalizedCoordinateSamplersSupported);
                const std::vector<Edge> theirs = edge_answers(host.p, host_slots);
                for (std::size_t i = 0; i < ours.size(); ++i) {
                    if (ours[i].written == theirs[i].written) continue;
                    std::fprintf(stderr, "%s: ours written %d, host runtime %d\n", ours[i].name, ours[i].written,
                                 theirs[i].written);
                    return 1;
                }
                // The host also tells the UINT and float border colours apart, and normalized from not.
                const auto host2 = [&](const D3D12_SAMPLER_DESC2& desc) {
                    const D3D12_CPU_DESCRIPTOR_HANDLE h = host_slots.take();
                    host->CreateSampler2(&desc, h);
                    return host_slots.bytes(h);
                };
                const D3D12_SAMPLER_DESC point = sampler(D3D12_FILTER_MIN_MAG_MIP_POINT, D3D12_TEXTURE_ADDRESS_MODE_CLAMP, 0);
                EXPECT(host2(desc2(point)) != host2(desc2(point, D3D12_SAMPLER_FLAG_NON_NORMALIZED_COORDINATES)));
            } else {
                std::printf("no host D3D12 device: edge cases checked against their recorded answers only\n");
            }
        }

        if (d.validation_errors()) {
            std::fprintf(stderr, "%u validation errors\n", d.validation_errors());
            return 1;
        }
        std::printf("D3D12 CreateSampler2: %d plain descriptions, 6 border colours and 3 non-normalized samplers "
                    "match CreateSampler; %zu edge cases match the PC runtime\n",
                    plain, ours.size());
    }
    return 0;
}
