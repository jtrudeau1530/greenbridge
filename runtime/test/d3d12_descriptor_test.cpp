// Descriptors through the Xbox-ABI proxy over the Vulkan device.
//   - CopyDescriptors and CopyDescriptorsSimple, checked on the bytes of the heap slots.
//   - Texture views and samplers (milestone M2, the device's part): CreateShaderResourceView,
//     CreateUnorderedAccessView and CreateSampler write a T# or an S# into the heap slot and put the view or the
//     sampler into the device's bindless set 0. The slots are decoded and checked. Then a compute shader samples
//     an uploaded 8x8 texture by those bytes: it looks the T# and the S# up in the descriptor hash table, as a
//     translated shader does. It checks the view's own slot, a copied slot, a slot that must give the fallback
//     image, and a packed RGB9E5 view with distinct channels and varying shared exponents. The sampler wraps
//     in U and the shader samples one tile to the right, so a missed S# would show as well (the fallback
//     sampler clamps).
//   - Default and mip-1 volume SRVs: sample a 4x4x4 grid through each view of one uploaded 8x8x4 texture.
//   - UINT border colours (#538, Vonsek's case): an R8G8B8A8_UINT texture sampled one tile to the right through
//     CreateSampler2 samplers that clamp to a UINT (0,0,0,0) and (1,1,1,1) border, and a float white border on
//     the UNORM texture. Every texel is the border: 0, 1 and 255 in each channel. Both white borders are built in
//     at the same S# type, so this also checks that they are two samplers, not one.
// The device has no pipelines yet, so the test makes that one pipeline in Vulkan itself and binds the device's set
// (rt::d3d12::bindless_set). The shader is test/glsl/cs_sample.comp; without glslangValidator at build time the
// sampling is skipped. No game data.
// Set LEONIDA_VK_VALIDATION=1 to run under VK_LAYER_KHRONOS_validation.
#include "vk_test_device.h"
#include "../shims/d3d12_com.h"
#include "../shims/d3d12_xs_device.h"
#include "../src/gfx10_descriptors.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <span>
#include <vector>

#define EXPECT(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); return 1; } \
} while (0)

namespace {

using XboxHeapHandleFn = std::uint64_t(WINAPI*)(void* heap);  // heap slots 9/10: 8-byte handle in RAX

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

D3D12_RESOURCE_DESC buffer_desc(UINT64 bytes, D3D12_RESOURCE_FLAGS flags) {
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Width = bytes;
    desc.Height = 1;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    desc.SampleDesc.Count = 1;
    desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    desc.Flags = flags;
    return desc;
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

// The uploaded texture: a gradient no two texels of which are equal.
std::array<std::uint8_t, 4> texel(UINT x, UINT y) {
    return {std::uint8_t(x * 32), std::uint8_t(y * 32), std::uint8_t(128 + x + 8 * y), 255};
}

#ifdef LEONIDA_TEST_CS_SAMPLE_SPV
// What cs_sample.comp is pushed: a T#, an S#, and where in the output buffer its 64 texels go.
struct SampleKeys {
    std::uint32_t tsharp[8];
    std::uint32_t ssharp[4];
    std::uint32_t first;
    std::uint32_t integer;  // the image is an integer one: cs_sample.comp reads it as a utexture2D
};

// Runs test/glsl/cs_sample.comp once for each of `keys` on `set0`, the device's bindless set, and copies the
// texels they wrote to `results` (64 RGBA8 texels each). Set 1 is the test's output buffer.
bool sample_through_set(vktest::Device& d, VkDescriptorSet set0, VkDescriptorSetLayout set0_layout,
                        std::span<const SampleKeys> keys, std::uint8_t* results) {
    const VkDevice dev = d.device;
    std::ifstream file(LEONIDA_TEST_CS_SAMPLE_SPV, std::ios::binary | std::ios::ate);
    if (!file) return false;
    std::vector<std::uint32_t> code(std::size_t(file.tellg()) / 4);
    file.seekg(0);
    file.read(reinterpret_cast<char*>(code.data()), std::streamsize(code.size() * 4));
    if (code.empty()) return false;

    constexpr VkDeviceSize texels = 64;
    void* mapped = nullptr;
    const VkBuffer output = d.buffer(keys.size() * texels * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &mapped);
    if (!output) return false;

    VkShaderModule module = VK_NULL_HANDLE;
    VkDescriptorSetLayout set1_layout = VK_NULL_HANDLE;
    VkDescriptorPool pool = VK_NULL_HANDLE;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkDescriptorSet set1 = VK_NULL_HANDLE;

    VkShaderModuleCreateInfo module_info{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    module_info.codeSize = code.size() * 4;
    module_info.pCode = code.data();
    const VkDescriptorSetLayoutBinding binding{0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT,
                                               nullptr};
    VkDescriptorSetLayoutCreateInfo set1_info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    set1_info.bindingCount = 1;
    set1_info.pBindings = &binding;
    const VkDescriptorPoolSize pool_size{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1};
    VkDescriptorPoolCreateInfo pool_info{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pool_info.maxSets = 1;
    pool_info.poolSizeCount = 1;
    pool_info.pPoolSizes = &pool_size;
    bool ok = d.fn<PFN_vkCreateShaderModule>("vkCreateShaderModule")(dev, &module_info, nullptr, &module) == VK_SUCCESS &&
              d.fn<PFN_vkCreateDescriptorSetLayout>("vkCreateDescriptorSetLayout")(dev, &set1_info, nullptr,
                                                                                   &set1_layout) == VK_SUCCESS &&
              d.fn<PFN_vkCreateDescriptorPool>("vkCreateDescriptorPool")(dev, &pool_info, nullptr, &pool) == VK_SUCCESS;
    if (ok) {
        VkDescriptorSetAllocateInfo allocate{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        allocate.descriptorPool = pool;
        allocate.descriptorSetCount = 1;
        allocate.pSetLayouts = &set1_layout;
        ok = d.fn<PFN_vkAllocateDescriptorSets>("vkAllocateDescriptorSets")(dev, &allocate, &set1) == VK_SUCCESS;
    }
    if (ok) {
        const VkDescriptorBufferInfo buffer_info{output, 0, VK_WHOLE_SIZE};
        VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        write.dstSet = set1;
        write.descriptorCount = 1;
        write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        write.pBufferInfo = &buffer_info;
        d.fn<PFN_vkUpdateDescriptorSets>("vkUpdateDescriptorSets")(dev, 1, &write, 0, nullptr);

        const VkDescriptorSetLayout layouts[2] = {set0_layout, set1_layout};
        const VkPushConstantRange push{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(SampleKeys)};
        VkPipelineLayoutCreateInfo layout_info{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        layout_info.setLayoutCount = 2;
        layout_info.pSetLayouts = layouts;
        layout_info.pushConstantRangeCount = 1;
        layout_info.pPushConstantRanges = &push;
        VkComputePipelineCreateInfo pipeline_info{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        pipeline_info.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
        pipeline_info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        pipeline_info.stage.module = module;
        pipeline_info.stage.pName = "main";
        ok = d.fn<PFN_vkCreatePipelineLayout>("vkCreatePipelineLayout")(dev, &layout_info, nullptr, &layout) == VK_SUCCESS;
        pipeline_info.layout = layout;
        ok = ok && d.fn<PFN_vkCreateComputePipelines>("vkCreateComputePipelines")(dev, VK_NULL_HANDLE, 1, &pipeline_info,
                                                                                  nullptr, &pipeline) == VK_SUCCESS;
    }
    if (ok) {
        const auto bind_sets = d.fn<PFN_vkCmdBindDescriptorSets>("vkCmdBindDescriptorSets");
        const auto push_constants = d.fn<PFN_vkCmdPushConstants>("vkCmdPushConstants");
        const auto dispatch = d.fn<PFN_vkCmdDispatch>("vkCmdDispatch");
        ok = d.run([&](VkCommandBuffer cb) {
            d.fn<PFN_vkCmdBindPipeline>("vkCmdBindPipeline")(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
            const VkDescriptorSet sets[2] = {set0, set1};
            bind_sets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 2, sets, 0, nullptr);
            for (const SampleKeys& k : keys) {
                push_constants(cb, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(k), &k);
                dispatch(cb, 1, 1, 1);
            }
            return true;
        });
    }
    if (ok) std::memcpy(results, mapped, keys.size() * texels * 4);
    if (pipeline) d.fn<PFN_vkDestroyPipeline>("vkDestroyPipeline")(dev, pipeline, nullptr);
    if (layout) d.fn<PFN_vkDestroyPipelineLayout>("vkDestroyPipelineLayout")(dev, layout, nullptr);
    if (pool) d.fn<PFN_vkDestroyDescriptorPool>("vkDestroyDescriptorPool")(dev, pool, nullptr);
    if (set1_layout) d.fn<PFN_vkDestroyDescriptorSetLayout>("vkDestroyDescriptorSetLayout")(dev, set1_layout, nullptr);
    if (module) d.fn<PFN_vkDestroyShaderModule>("vkDestroyShaderModule")(dev, module, nullptr);
    return ok;
}
#endif

}  // namespace

int main() {
    vktest::Device d;
    EXPECT(d.create());

    constexpr UINT size = 8, texels = size * size, bytes = texels * 4;
    std::vector<std::uint8_t> results(4 * bytes), after_destruction(bytes), borders(3 * bytes);
    bool sampled_through_set = false;
    {
        rt::d3d12::VulkanDeviceInfo info{};
        info.get_instance_proc = d.instance_proc;
        info.instance = d.instance;
        info.physical = d.physical;
        info.device = d.device;
        info.queue = d.queue;
        info.queue_family = d.family;
        info.texture_compression_bc = d.bc;
        info.bindless = d.bindless;
        info.bindless_partially_bound = d.partially_bound;
        ID3D12Device* inner = nullptr;
        EXPECT(SUCCEEDED(rt::d3d12::create_vulkan_device(info, IID_PPV_ARGS(&inner))));
        Ref<ID3D12Device> device;
        EXPECT(SUCCEEDED(rt::d3d12_xs::create_device_proxy(inner, __uuidof(ID3D12Device),
                                                           reinterpret_cast<void**>(&device.p))));
        rt::d3d12::set_device_outer(inner, device.p);  // the proxy now owns `inner`

        D3D12_FEATURE_DATA_D3D12_OPTIONS8 options8{};
        EXPECT(device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS8, &options8, sizeof(options8)) == S_OK);
        EXPECT(options8.UnalignedBlockTexturesSupported == (d.bc ? TRUE : FALSE));
        for (UINT query_size : {0u, UINT(sizeof(options8) - 1), UINT(sizeof(options8) + 1)}) {
            D3D12_FEATURE_DATA_D3D12_OPTIONS8 output[] = {{83}, {42}};
            EXPECT(device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS8, output, query_size) == E_INVALIDARG);
            EXPECT(output[0].UnalignedBlockTexturesSupported == 83 && output[1].UnalignedBlockTexturesSupported == 42);
        }

        // --- Copies. Two host-side heaps; slot i of the source holds 32 bytes of (0x10 * (i + 1) + byte index).
        {
            const UINT step = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
            constexpr UINT source_slots = 4, destination_slots = 8;
            Ref<ID3D12DescriptorHeap> source_heap, destination_heap;
            D3D12_DESCRIPTOR_HEAP_DESC heap_desc{D3D12_DESCRIPTOR_HEAP_TYPE_RTV, source_slots,
                                                 D3D12_DESCRIPTOR_HEAP_FLAG_NONE, 0};
            EXPECT(SUCCEEDED(device->CreateDescriptorHeap(&heap_desc, IID_PPV_ARGS(&source_heap.p))));
            heap_desc.NumDescriptors = destination_slots;
            EXPECT(SUCCEEDED(device->CreateDescriptorHeap(&heap_desc, IID_PPV_ARGS(&destination_heap.p))));
            auto* const source = reinterpret_cast<std::uint8_t*>(
                reinterpret_cast<XboxHeapHandleFn>(slot(source_heap.p, 9))(source_heap.p));
            auto* const destination = reinterpret_cast<std::uint8_t*>(
                reinterpret_cast<XboxHeapHandleFn>(slot(destination_heap.p, 9))(destination_heap.p));
            EXPECT(source && destination && step >= 16);
            for (UINT i = 0; i < source_slots; ++i)
                for (UINT b = 0; b < step; ++b) source[i * step + b] = std::uint8_t(0x10 * (i + 1) + b);
            const auto from = [&](UINT i) { return D3D12_CPU_DESCRIPTOR_HANDLE{SIZE_T(source + i * step)}; };
            const auto to = [&](UINT i) { return D3D12_CPU_DESCRIPTOR_HANDLE{SIZE_T(destination + i * step)}; };
            // Destination slot `at` holds source slot `want`, or is still zero (want < 0).
            const auto holds = [&](UINT at, int want) {
                const std::uint8_t* got = destination + at * step;
                if (want < 0) return std::all_of(got, got + step, [](std::uint8_t byte) { return byte == 0; });
                return std::memcmp(got, source + UINT(want) * step, step) == 0;
            };
            const auto clear = [&] { std::memset(destination, 0, std::size_t(destination_slots) * step); };
            const auto rtv = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;

            // CopyDescriptorsSimple: one run of descriptors.
            device->CopyDescriptorsSimple(2, to(1), from(2), rtv);
            EXPECT(holds(0, -1) && holds(1, 2) && holds(2, 3) && holds(3, -1));
            device->CopyDescriptorsSimple(0, to(5), from(0), rtv);
            EXPECT(holds(5, -1));

            // CopyDescriptors walks both range lists together: destination ranges {3, 4} and {0}, source ranges
            // {0} and {1, 2}, so 3 <- 0, 4 <- 1, 0 <- 2.
            clear();
            {
                const D3D12_CPU_DESCRIPTOR_HANDLE destinations[] = {to(3), to(0)}, sources[] = {from(0), from(1)};
                const UINT destination_sizes[] = {2, 1}, source_sizes[] = {1, 2};
                device->CopyDescriptors(2, destinations, destination_sizes, 2, sources, source_sizes, rtv);
                EXPECT(holds(3, 0) && holds(4, 1) && holds(0, 2));
                EXPECT(holds(1, -1) && holds(2, -1) && holds(5, -1) && holds(6, -1) && holds(7, -1));
            }
            // Null size arrays: every range is one descriptor.
            clear();
            {
                const D3D12_CPU_DESCRIPTOR_HANDLE destinations[] = {to(6), to(2)}, sources[] = {from(3), from(1)};
                device->CopyDescriptors(2, destinations, nullptr, 2, sources, nullptr, rtv);
                EXPECT(holds(6, 3) && holds(2, 1) && holds(7, -1) && holds(3, -1));
            }
            // One side sized, the other not, and a destination longer than the source: the copy stops with the
            // source.
            clear();
            {
                const D3D12_CPU_DESCRIPTOR_HANDLE destinations[] = {to(1)}, sources[] = {from(0), from(3)};
                const UINT destination_sizes[] = {5};
                device->CopyDescriptors(1, destinations, destination_sizes, 2, sources, nullptr, rtv);
                EXPECT(holds(1, 0) && holds(2, 3) && holds(3, -1) && holds(4, -1) && holds(5, -1) && holds(0, -1));
            }
            // Nothing to copy from or to.
            clear();
            {
                const D3D12_CPU_DESCRIPTOR_HANDLE destinations[] = {to(0)}, sources[] = {from(0)};
                device->CopyDescriptors(1, destinations, nullptr, 0, sources, nullptr, rtv);
                device->CopyDescriptors(1, nullptr, nullptr, 1, sources, nullptr, rtv);
                device->CopyDescriptors(1, destinations, nullptr, 1, nullptr, nullptr, rtv);
                EXPECT(holds(0, -1));
            }
        }

        // --- Sampling through descriptors.
        if (!d.bindless) {
            std::printf("descriptors: copies matched; sampling skipped, the device lacks a feature the bindless set "
                        "needs\n");
            return d.validation_errors() ? 1 : 0;
        }
        const D3D12_HEAP_PROPERTIES default_heap{D3D12_HEAP_TYPE_DEFAULT};
        const D3D12_HEAP_PROPERTIES upload_heap{D3D12_HEAP_TYPE_UPLOAD};
        D3D12_RESOURCE_DESC texture_desc{};
        texture_desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        texture_desc.Width = size;
        texture_desc.Height = size;
        texture_desc.DepthOrArraySize = 1;
        texture_desc.MipLevels = 1;
        texture_desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        texture_desc.SampleDesc.Count = 1;
        Ref<ID3D12Resource> texture, upload;
        EXPECT(SUCCEEDED(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &texture_desc,
                                                         D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                                         IID_PPV_ARGS(&texture.p))));
        const auto upload_desc = buffer_desc(65536, D3D12_RESOURCE_FLAG_NONE);
        EXPECT(SUCCEEDED(device->CreateCommittedResource(&upload_heap, D3D12_HEAP_FLAG_NONE, &upload_desc,
                                                         D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
                                                         IID_PPV_ARGS(&upload.p))));
        D3D12_PLACED_SUBRESOURCE_FOOTPRINT placed{};
        device->GetCopyableFootprints(&texture_desc, 0, 1, 0, &placed, nullptr, nullptr, nullptr);
        void* mapped = nullptr;
        EXPECT(SUCCEEDED(upload->Map(0, nullptr, &mapped)));
        for (UINT y = 0; y < size; ++y)
            for (UINT x = 0; x < size; ++x)
                std::memcpy(static_cast<std::uint8_t*>(mapped) + placed.Offset + y * placed.Footprint.RowPitch + x * 4,
                            texel(x, y).data(), 4);
        // The same bytes in an R8G8B8A8_UINT texture, from further into the upload buffer.
        D3D12_RESOURCE_DESC uint_desc = texture_desc;
        uint_desc.Format = DXGI_FORMAT_R8G8B8A8_UINT;
        Ref<ID3D12Resource> uint_texture;
        EXPECT(SUCCEEDED(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &uint_desc,
                                                         D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                                         IID_PPV_ARGS(&uint_texture.p))));
        D3D12_PLACED_SUBRESOURCE_FOOTPRINT uint_placed{};
        device->GetCopyableFootprints(&uint_desc, 0, 1, 32768, &uint_placed, nullptr, nullptr, nullptr);
        for (UINT y = 0; y < size; ++y)
            for (UINT x = 0; x < size; ++x)
                std::memcpy(static_cast<std::uint8_t*>(mapped) + uint_placed.Offset +
                                y * uint_placed.Footprint.RowPitch + x * 4,
                            texel(x, y).data(), 4);

        // Shader-visible heaps, start handles in the Xbox shape.
        const UINT step = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        const UINT sampler_step = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER);
        Ref<ID3D12DescriptorHeap> view_heap, sampler_heap;
        D3D12_DESCRIPTOR_HEAP_DESC heap_desc{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 10,
                                             D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, 0};
        EXPECT(SUCCEEDED(device->CreateDescriptorHeap(&heap_desc, IID_PPV_ARGS(&view_heap.p))));
        heap_desc = {D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER, 4, D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, 0};
        EXPECT(SUCCEEDED(device->CreateDescriptorHeap(&heap_desc, IID_PPV_ARGS(&sampler_heap.p))));
        const std::uint64_t view_cpu = reinterpret_cast<XboxHeapHandleFn>(slot(view_heap.p, 9))(view_heap.p);
        const std::uint64_t sampler_cpu = reinterpret_cast<XboxHeapHandleFn>(slot(sampler_heap.p, 9))(sampler_heap.p);
        EXPECT(view_cpu && sampler_cpu);
        const auto view_slot = [&](UINT i) { return D3D12_CPU_DESCRIPTOR_HANDLE{SIZE_T(view_cpu + i * step)}; };
        const auto view_bytes = [&](UINT i) { return reinterpret_cast<std::uint8_t*>(view_cpu + i * step); };

        // The view in slot 1, a copy of it in slot 5, and in slot 3 bytes no view was created for.
        constexpr UINT own = 1, copied = 5, unknown = 3;
        device->CreateShaderResourceView(texture.p, nullptr, view_slot(own));
        EXPECT(!std::all_of(view_bytes(own), view_bytes(own) + step, [](std::uint8_t byte) { return byte == 0; }));
        device->CopyDescriptorsSimple(1, view_slot(copied), view_slot(own), D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        EXPECT(std::memcmp(view_bytes(copied), view_bytes(own), step) == 0);
        std::memset(view_bytes(unknown), 0x5a, step);
        D3D12_SAMPLER_DESC sampler_desc{};
        // Wrapping in U, so that it differs from the fallback sampler (nearest, clamping): the shader samples one
        // tile to the right, which gives the texture with this sampler and its right edge with the fallback.
        sampler_desc.Filter = D3D12_FILTER_MIN_MAG_MIP_POINT;
        sampler_desc.AddressU = D3D12_TEXTURE_ADDRESS_MODE_WRAP;
        sampler_desc.AddressV = sampler_desc.AddressW = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
        sampler_desc.MaxAnisotropy = 1;
        sampler_desc.ComparisonFunc = D3D12_COMPARISON_FUNC_NEVER;
        sampler_desc.MaxLOD = D3D12_FLOAT32_MAX;
        device->CreateSampler(&sampler_desc, D3D12_CPU_DESCRIPTOR_HANDLE{SIZE_T(sampler_cpu)});
        const auto* sampler_bytes = reinterpret_cast<const std::uint8_t*>(sampler_cpu);
        EXPECT(!std::all_of(sampler_bytes, sampler_bytes + 16, [](std::uint8_t byte) { return byte == 0; }));
        EXPECT(sampler_step >= 16);

        // What the slots hold says what the view and the sampler are, in the structures the console's shaders read.
        namespace g = leonida::gfx10;
        const auto image_at = [&](UINT i, g::Image& image) {
            g::ImageWords words{};
            std::memcpy(words.data(), view_bytes(i), sizeof(words));
            return g::is_image(words) && g::decode(words, image);
        };
        const auto null_at = [&](UINT i) {
            return std::all_of(view_bytes(i), view_bytes(i) + step, [](std::uint8_t byte) { return byte == 0; });
        };
        g::Image image{};
        EXPECT(image_at(own, image));
        EXPECT(image.type == g::ImageType::image_2d && image.width == size && image.height == size);
        EXPECT(image.format == g::format_from_vulkan(VK_FORMAT_R8G8B8A8_UNORM) && image.format != 0);
        EXPECT(image.first_mip == 0 && image.mip_count == 1 && image.layer_count == 1 && image.resource_mips == 1);
        EXPECT(image.swizzle == g::identity_swizzle && image.address >= (1ull << 32) && image.address % 256 == 0);
        g::SamplerWords sampler_words{};
        std::memcpy(sampler_words.data(), sampler_bytes, sizeof(sampler_words));
        const g::Sampler sampler = g::decode(sampler_words);
        EXPECT(sampler.mag_filter == g::Filter::point && sampler.min_filter == g::Filter::point);
        EXPECT(sampler.mip_filter == g::MipFilter::point && sampler.max_anisotropy == 1);
        EXPECT(sampler.address_u == g::AddressMode::wrap && sampler.address_v == g::AddressMode::clamp_last_texel &&
               sampler.address_w == g::AddressMode::clamp_last_texel);
        EXPECT(sampler.compare == g::CompareFunction::never && sampler.min_lod == 0.0f && sampler.max_lod > 14.9f);

        // The UINT texture's view in slot 0, and three border samplers after the first: UINT (0,0,0,0) and
        // (1,1,1,1) from CreateSampler2, which the proxy does not have (the inner device does), and float white.
        constexpr UINT uint_slot = 0, uint_black = 1, uint_white = 2, float_white = 3;
        device->CreateShaderResourceView(uint_texture.p, nullptr, view_slot(uint_slot));
        {
            Ref<ID3D12Device11> inner11;
            EXPECT(SUCCEEDED(inner->QueryInterface(IID_PPV_ARGS(&inner11.p))));
            D3D12_SAMPLER_DESC2 border_desc{};
            border_desc.Filter = D3D12_FILTER_MIN_MAG_MIP_POINT;
            border_desc.AddressU = D3D12_TEXTURE_ADDRESS_MODE_BORDER;
            border_desc.AddressV = border_desc.AddressW = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
            border_desc.MaxAnisotropy = 1;
            border_desc.ComparisonFunc = D3D12_COMPARISON_FUNC_NEVER;
            const auto at = [&](UINT i) { return D3D12_CPU_DESCRIPTOR_HANDLE{SIZE_T(sampler_cpu + i * sampler_step)}; };
            border_desc.Flags = D3D12_SAMPLER_FLAG_UINT_BORDER_COLOR;
            inner11->CreateSampler2(&border_desc, at(uint_black));
            for (UINT& c : border_desc.UintBorderColor) c = 1;
            inner11->CreateSampler2(&border_desc, at(uint_white));
            border_desc.Flags = D3D12_SAMPLER_FLAG_NONE;
            for (FLOAT& c : border_desc.FloatBorderColor) c = 1.0f;
            inner11->CreateSampler2(&border_desc, at(float_white));
        }
        const auto sampler_at = [&](UINT i) {
            return reinterpret_cast<const std::uint8_t*>(sampler_cpu + i * sampler_step);
        };
        // The two white borders differ only in the integer mark.
        g::SamplerWords white_words{}, float_white_words{};
        std::memcpy(white_words.data(), sampler_at(uint_white), sizeof(white_words));
        std::memcpy(float_white_words.data(), sampler_at(float_white), sizeof(float_white_words));
        EXPECT(white_words != float_white_words && g::decode(white_words).integer_border);
        EXPECT(white_words[3] == (float_white_words[3] | g::kIntegerBorderPtr));

        // A view with every field away from its default: mips 1 and 2 of a BGRA cube, with the view's mapping
        // swapping red and blue and forcing alpha to one. The texture is another one, so its address differs.
        Ref<ID3D12Resource> cube, writable;
        D3D12_RESOURCE_DESC cube_desc = texture_desc;
        cube_desc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
        cube_desc.MipLevels = 4;
        cube_desc.DepthOrArraySize = 6;
        EXPECT(SUCCEEDED(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &cube_desc,
                                                         D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&cube.p))));
        D3D12_SHADER_RESOURCE_VIEW_DESC cube_view{};
        cube_view.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
        cube_view.ViewDimension = D3D12_SRV_DIMENSION_TEXTURECUBE;
        cube_view.Shader4ComponentMapping =
            D3D12_ENCODE_SHADER_4_COMPONENT_MAPPING(2, 1, 0, D3D12_SHADER_COMPONENT_MAPPING_FORCE_VALUE_1);
        cube_view.TextureCube.MostDetailedMip = 1;
        cube_view.TextureCube.MipLevels = 2;
        constexpr UINT cube_slot = 6, read_slot = 4, write_slot = 7, null_slot = 2;
        device->CreateShaderResourceView(cube.p, &cube_view, view_slot(cube_slot));
        g::Image cube_image{};
        EXPECT(image_at(cube_slot, cube_image));
        EXPECT(cube_image.type == g::ImageType::cube && cube_image.layer_count == 6 && cube_image.first_layer == 0);
        EXPECT(cube_image.first_mip == 1 && cube_image.mip_count == 2 && cube_image.resource_mips == 4);
        EXPECT(cube_image.format == g::format_from_vulkan(VK_FORMAT_B8G8R8A8_UNORM));
        // BGRA keeps blue in the hardware's first channel, so reading blue as red is X, and so on.
        EXPECT((cube_image.swizzle == g::Swizzle{g::select_x, g::select_y, g::select_z, g::select_one}));
        EXPECT(cube_image.address != image.address);

        // An unordered-access view is the same T# as a shader-resource view of the same mip: one key, one entry.
        D3D12_RESOURCE_DESC writable_desc = texture_desc;
        writable_desc.Format = DXGI_FORMAT_R32_UINT;
        writable_desc.MipLevels = 2;
        writable_desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
        EXPECT(SUCCEEDED(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &writable_desc,
                                                         D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr,
                                                         IID_PPV_ARGS(&writable.p))));
        D3D12_SHADER_RESOURCE_VIEW_DESC read_view{};
        read_view.Format = DXGI_FORMAT_R32_UINT;
        read_view.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
        read_view.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        read_view.Texture2D.MostDetailedMip = 1;
        read_view.Texture2D.MipLevels = 1;
        D3D12_UNORDERED_ACCESS_VIEW_DESC write_view{};
        write_view.Format = DXGI_FORMAT_R32_UINT;
        write_view.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
        write_view.Texture2D.MipSlice = 1;
        device->CreateShaderResourceView(writable.p, &read_view, view_slot(read_slot));
        device->CreateUnorderedAccessView(writable.p, nullptr, &write_view, view_slot(write_slot));
        g::Image write_image{};
        EXPECT(image_at(write_slot, write_image));
        EXPECT(write_image.first_mip == 1 && write_image.mip_count == 1 && write_image.resource_mips == 2);
        EXPECT(write_image.format == g::format_from_vulkan(VK_FORMAT_R32_UINT));
        EXPECT(std::memcmp(view_bytes(read_slot), view_bytes(write_slot), step) == 0);

        // Null descriptors: no resource, a view the texture cannot have, a view past its mips.
        std::memset(view_bytes(null_slot), 0x77, step);
        device->CreateShaderResourceView(nullptr, &read_view, view_slot(null_slot));
        EXPECT(null_at(null_slot));
        std::memset(view_bytes(null_slot), 0x77, step);
        device->CreateUnorderedAccessView(texture.p, nullptr, nullptr, view_slot(null_slot));  // not ALLOW_UNORDERED_ACCESS
        EXPECT(null_at(null_slot));
        std::memset(view_bytes(null_slot), 0x77, step);
        read_view.Texture2D.MostDetailedMip = 2;
        device->CreateShaderResourceView(writable.p, &read_view, view_slot(null_slot));
        EXPECT(null_at(null_slot));
        // A volume has a depth, one Vulkan array layer and a view-relative mip range.
        D3D12_RESOURCE_DESC volume_desc = texture_desc;
        volume_desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE3D;
        volume_desc.DepthOrArraySize = 4;
        volume_desc.MipLevels = 2;
        Ref<ID3D12Resource> volume;
        EXPECT(SUCCEEDED(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &volume_desc,
                                                         D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                                         IID_PPV_ARGS(&volume.p))));
        constexpr UINT volume_all = 8, volume_mip = 9;
        device->CreateShaderResourceView(volume.p, nullptr, view_slot(volume_all));
        g::Image vol{};
        EXPECT(image_at(volume_all, vol) && vol.type == g::ImageType::image_3d);
        EXPECT(vol.width == size && vol.height == size && vol.depth == 4 && vol.layer_count == 1);
        EXPECT(vol.first_mip == 0 && vol.mip_count == 2);
        D3D12_SHADER_RESOURCE_VIEW_DESC volume_view{};
        volume_view.Format = volume_desc.Format;
        volume_view.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE3D;
        volume_view.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        volume_view.Texture3D = {1, 1, 0};
        device->CreateShaderResourceView(volume.p, &volume_view, view_slot(volume_mip));
        EXPECT(image_at(volume_mip, vol) && vol.type == g::ImageType::image_3d);
        EXPECT(vol.first_mip == 1 && vol.mip_count == 1 && vol.depth == 4 && vol.resource_mips == 2);
        // D3D12 requires DENY_SHADER_RESOURCE to accompany ALLOW_DEPTH_STENCIL. If the layer accepts
        // this transfer-only volume at creation, requesting an SRV must still clear the destination.
        D3D12_RESOURCE_DESC denied_desc = volume_desc;
        denied_desc.Flags = D3D12_RESOURCE_FLAG_DENY_SHADER_RESOURCE;
        Ref<ID3D12Resource> denied_volume;
        const HRESULT denied_hr = device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &denied_desc,
                D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&denied_volume.p));
        EXPECT(denied_hr == S_OK || denied_hr == E_INVALIDARG);
        if (denied_hr == S_OK) {
            device->CopyDescriptorsSimple(1, view_slot(null_slot), view_slot(volume_mip),
                                          D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
            EXPECT(!null_at(null_slot));
            device->CreateShaderResourceView(denied_volume.p, &volume_view, view_slot(null_slot));
            EXPECT(null_at(null_slot));
            denied_volume.reset();
        }
        volume_view.Texture3D.MostDetailedMip = 2;
        device->CreateShaderResourceView(volume.p, &volume_view, view_slot(null_slot));
        EXPECT(null_at(null_slot));
        volume_view.Texture3D.MostDetailedMip = 0;
        device->CreateShaderResourceView(texture.p, &volume_view, view_slot(null_slot));
        EXPECT(null_at(null_slot));
        volume_view.Texture3D.ResourceMinLODClamp = 0.5f;
        device->CreateShaderResourceView(volume.p, &volume_view, view_slot(null_slot));
        EXPECT(null_at(null_slot));
        volume_view.Texture3D.ResourceMinLODClamp = 0;
        volume_view.Format = DXGI_FORMAT_R8G8B8A8_UINT;
        device->CreateShaderResourceView(volume.p, &volume_view, view_slot(null_slot));
        EXPECT(null_at(null_slot));
        D3D12_SHADER_RESOURCE_VIEW_DESC flat_view{};
        flat_view.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        flat_view.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
        flat_view.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        flat_view.Texture2D.MipLevels = 1;
        std::memset(view_bytes(null_slot), 0x77, step);
        device->CreateShaderResourceView(volume.p, &flat_view, view_slot(null_slot));
        EXPECT(null_at(null_slot));
        D3D12_PLACED_SUBRESOURCE_FOOTPRINT volume_placed[2]{};
        // Place both volume mips after the UINT texture, leaving the lower offsets for the RGBA8 and RGB9E5 uploads.
        const UINT64 volume_start = uint_placed.Offset + UINT64(uint_placed.Footprint.RowPitch) * uint_placed.Footprint.Height;
        device->GetCopyableFootprints(&volume_desc, 0, 2, volume_start, volume_placed, nullptr, nullptr, nullptr);
        const auto& last = volume_placed[1];
        EXPECT(last.Offset + UINT64(last.Footprint.RowPitch) * last.Footprint.Height * last.Footprint.Depth <= upload_desc.Width);
        for (UINT mip = 0; mip < 2; ++mip) {
            const auto& fp = volume_placed[mip].Footprint;
            for (UINT z = 0; z < fp.Depth; ++z)
                for (UINT y = 0; y < fp.Height; ++y)
                    for (UINT x = 0; x < fp.Width; ++x) {
                        const std::uint8_t rgba[]{std::uint8_t(x * 19), std::uint8_t(y * 23),
                                                  std::uint8_t(z * 53), std::uint8_t(mip ? 101 : 237)};
                        const auto offset = volume_placed[mip].Offset + (z * fp.Height + y) * fp.RowPitch + 4 * x;
                        std::memcpy(static_cast<std::uint8_t*>(mapped) + offset, rgba, 4);
                    }
        }

        // D3D-75: plane 1 of a depth-stencil texture is its stencil. The T# is 8_UINT at the texture's own address,
        // with the eight bits read as G and R, B, A as 0, 0, 1 (D3D12's X24_TYPELESS_G8_UINT / X32_..._G8X24_UINT).
        // A stencil view of a colour texture, or one in another format, is a null descriptor.
        // Plane 0, as R24_UNORM_X8_TYPELESS or R32_FLOAT_X8X24_TYPELESS, is a depth T# of the format backing the
        // texture at the same address, the depth read as R. A GPU without a usable D24 backs R24G8 with
        // D32_SFLOAT_S8_UINT (the layer's depth_usable rule), so the expected code follows the backing. Each depth
        // view format belongs to its own family only: the other one's is a null descriptor, as D3D12 refuses it.
        VkFormatProperties d24_properties{};
        d.instance_fn<PFN_vkGetPhysicalDeviceFormatProperties>("vkGetPhysicalDeviceFormatProperties")(
            d.physical, VK_FORMAT_D24_UNORM_S8_UINT, &d24_properties);
        const VkFormat d24_backing =
            (d24_properties.optimalTilingFeatures & VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT)
                ? VK_FORMAT_D24_UNORM_S8_UINT
                : VK_FORMAT_D32_SFLOAT_S8_UINT;
        for (const auto& [resource_format, stencil_view_format, depth_view_format, other_depth_view_format, depth_vk] :
             {std::tuple{DXGI_FORMAT_R24G8_TYPELESS, DXGI_FORMAT_X24_TYPELESS_G8_UINT,
                         DXGI_FORMAT_R24_UNORM_X8_TYPELESS, DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS, d24_backing},
              std::tuple{DXGI_FORMAT_R32G8X24_TYPELESS, DXGI_FORMAT_X32_TYPELESS_G8X24_UINT,
                         DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS, DXGI_FORMAT_R24_UNORM_X8_TYPELESS,
                         VK_FORMAT_D32_SFLOAT_S8_UINT}}) {
            D3D12_RESOURCE_DESC depth_desc = texture_desc;
            depth_desc.Format = resource_format;
            depth_desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;
            Ref<ID3D12Resource> depth_stencil;
            EXPECT(SUCCEEDED(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &depth_desc,
                                                             D3D12_RESOURCE_STATE_COMMON, nullptr,
                                                             IID_PPV_ARGS(&depth_stencil.p))));
            D3D12_SHADER_RESOURCE_VIEW_DESC plane_view{};
            plane_view.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
            plane_view.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
            plane_view.Texture2D.MipLevels = 1;
            plane_view.Format = stencil_view_format;
            plane_view.Texture2D.PlaneSlice = 1;
            std::memset(view_bytes(null_slot), 0x77, step);
            device->CreateShaderResourceView(depth_stencil.p, &plane_view, view_slot(null_slot));
            g::Image stencil_image{};
            EXPECT(image_at(null_slot, stencil_image));
            EXPECT(stencil_image.format == g::stencil_format && stencil_image.type == g::ImageType::image_2d);
            EXPECT((stencil_image.swizzle == g::Swizzle{g::select_zero, g::select_x, g::select_zero, g::select_one}));
            EXPECT(stencil_image.width == size && stencil_image.mip_count == 1);

            plane_view.Format = depth_view_format;
            plane_view.Texture2D.PlaneSlice = 0;
            std::memset(view_bytes(null_slot), 0x77, step);
            device->CreateShaderResourceView(depth_stencil.p, &plane_view, view_slot(null_slot));
            g::Image depth_image{};
            EXPECT(image_at(null_slot, depth_image));
            EXPECT(depth_image.format == g::format_from_vulkan(depth_vk) && depth_image.format != 0);
            EXPECT(depth_image.address == stencil_image.address && depth_image.swizzle[0] == g::select_x);

            plane_view.Format = other_depth_view_format;  // the other depth family's view format
            std::memset(view_bytes(null_slot), 0x77, step);
            device->CreateShaderResourceView(depth_stencil.p, &plane_view, view_slot(null_slot));
            EXPECT(null_at(null_slot));

            plane_view.Format = DXGI_FORMAT_R8_UINT;  // not a stencil view format
            plane_view.Texture2D.PlaneSlice = 1;
            std::memset(view_bytes(null_slot), 0x77, step);
            device->CreateShaderResourceView(depth_stencil.p, &plane_view, view_slot(null_slot));
            EXPECT(null_at(null_slot));

            // The other family's stencil view format is refused too, as D3D12 refuses it: X24_TYPELESS_G8_UINT
            // belongs to R24G8 only, X32_TYPELESS_G8X24_UINT to R32G8X24 only.
            plane_view.Format = stencil_view_format == DXGI_FORMAT_X24_TYPELESS_G8_UINT
                                    ? DXGI_FORMAT_X32_TYPELESS_G8X24_UINT
                                    : DXGI_FORMAT_X24_TYPELESS_G8_UINT;
            std::memset(view_bytes(null_slot), 0x77, step);
            device->CreateShaderResourceView(depth_stencil.p, &plane_view, view_slot(null_slot));
            EXPECT(null_at(null_slot));
        }
        D3D12_SHADER_RESOURCE_VIEW_DESC colour_plane{};
        colour_plane.Format = DXGI_FORMAT_X24_TYPELESS_G8_UINT;
        colour_plane.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
        colour_plane.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        colour_plane.Texture2D.MipLevels = 1;
        colour_plane.Texture2D.PlaneSlice = 1;
        std::memset(view_bytes(null_slot), 0x77, step);
        device->CreateShaderResourceView(texture.p, &colour_plane, view_slot(null_slot));
        EXPECT(null_at(null_slot));
        cube.reset();
        writable.reset();

        // A packed RGB9E5 SRV uses the same four-byte copy footprint as RGBA8. Its shared exponent
        // changes on each diagonal. The last pattern permutes the mantissas to detect channel swaps.
        Ref<ID3D12Resource> packed;
        D3D12_RESOURCE_DESC packed_desc = texture_desc;
        packed_desc.Format = DXGI_FORMAT_R9G9B9E5_SHAREDEXP;
        EXPECT(SUCCEEDED(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &packed_desc,
                                                         D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                                         IID_PPV_ARGS(&packed.p))));
        constexpr UINT packed_slot = cube_slot;  // The cube was released above.
        device->CreateShaderResourceView(packed.p, nullptr, view_slot(packed_slot));
        D3D12_PLACED_SUBRESOURCE_FOOTPRINT packed_placed{};
        device->GetCopyableFootprints(&packed_desc, 0, 1, 4096, &packed_placed, nullptr, nullptr, nullptr);
        EXPECT(packed_placed.Offset == 4096 && packed_placed.Footprint.RowPitch == D3D12_TEXTURE_DATA_PITCH_ALIGNMENT);
        constexpr auto rgb9e5 = [](UINT r, UINT g, UINT b, UINT e) { return r | (g << 9) | (b << 18) | (e << 27); };
        constexpr UINT packed_values[] = {rgb9e5(64, 128, 256, 14), rgb9e5(64, 128, 256, 15),
                                          rgb9e5(64, 128, 256, 16), rgb9e5(256, 64, 128, 14)};
        for (UINT y = 0; y < size; ++y) {
            auto* row = static_cast<std::uint8_t*>(mapped) + packed_placed.Offset + y * packed_placed.Footprint.RowPitch;
            for (UINT x = 0; x < size; ++x)
                std::memcpy(row + x * 4, &packed_values[(x + y) % 4], 4);
        }

        // OPTIONS8 applies to the top-level mip. The four BC1 blocks cover 4x4, 3x4, 4x1 and 3x1 texels.
        constexpr UINT bc_width = 7, bc_height = 5;
        Ref<ID3D12Resource> bc;
        D3D12_PLACED_SUBRESOURCE_FOOTPRINT bc_placed{};
        if (options8.UnalignedBlockTexturesSupported) {
            auto bc_desc = texture_desc;
            bc_desc.Width = bc_width;
            bc_desc.Height = bc_height;
            bc_desc.Format = DXGI_FORMAT_BC1_UNORM;
            EXPECT(SUCCEEDED(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &bc_desc,
                                                             D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                                             IID_PPV_ARGS(&bc.p))));
            device->CreateShaderResourceView(bc.p, nullptr, view_slot(null_slot));
            device->GetCopyableFootprints(&bc_desc, 0, 1, 8192, &bc_placed, nullptr, nullptr, nullptr);
            EXPECT(bc_placed.Offset == 8192 && bc_placed.Footprint.RowPitch == D3D12_TEXTURE_DATA_PITCH_ALIGNMENT);
            // Endpoint 0 is RGB565 red, green, blue or white. Every two-bit selector picks that endpoint.
            constexpr std::uint64_t blocks[2][2] = {{0xf800, 0x07e0}, {0x001f, 0xffff}};
            for (UINT row = 0; row < 2; ++row)
                std::memcpy(static_cast<std::uint8_t*>(mapped) + bc_placed.Offset + row * bc_placed.Footprint.RowPitch,
                            blocks[row], sizeof(blocks[row]));
        }

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

        // Upload, then the texture into a shader-resource state. Executing the list also brings set 0 up to date.
        D3D12_TEXTURE_COPY_LOCATION to_texture{}, from_upload{};
        to_texture.pResource = texture.p;
        to_texture.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        from_upload.pResource = upload.p;
        from_upload.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
        from_upload.PlacedFootprint = placed;
        list->CopyTextureRegion(&to_texture, 0, 0, 0, &from_upload, nullptr);
        to_texture.pResource = uint_texture.p;
        from_upload.PlacedFootprint = uint_placed;
        list->CopyTextureRegion(&to_texture, 0, 0, 0, &from_upload, nullptr);
        to_texture.pResource = packed.p;
        from_upload.PlacedFootprint = packed_placed;
        list->CopyTextureRegion(&to_texture, 0, 0, 0, &from_upload, nullptr);
        if (bc.p) {
            to_texture.pResource = bc.p;
            from_upload.PlacedFootprint = bc_placed;
            list->CopyTextureRegion(&to_texture, 0, 0, 0, &from_upload, nullptr);
            const auto barrier =
                transition(bc.p, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            list->ResourceBarrier(1, &barrier);
        }
        to_texture.pResource = volume.p;
        for (UINT mip = 0; mip < 2; ++mip) {
            to_texture.SubresourceIndex = mip;
            from_upload.PlacedFootprint = volume_placed[mip];
            list->CopyTextureRegion(&to_texture, 0, 0, 0, &from_upload, nullptr);
        }
        const D3D12_RESOURCE_BARRIER sampled[] = {
            transition(texture.p, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE),
            transition(packed.p, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE),
            transition(uint_texture.p, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE),
            transition(volume.p, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE)};
        list->ResourceBarrier(UINT(std::size(sampled)), sampled);
        EXPECT(list->Close() == S_OK);
        ID3D12CommandList* lists[] = {list.p};
        queue->ExecuteCommandLists(1, lists);
        EXPECT(rt::d3d12::last_execute_result(queue.p) == leonida::ReplayResult::success);
        EXPECT(SUCCEEDED(queue->Signal(fence.p, 1)));
        EXPECT(SUCCEEDED(fence->SetEventOnCompletion(1, nullptr)));

        // Set 0 now holds what the views and the sampler named. Four runs of the shader: the view's own slot, its
        // copy, the slot of bytes no view was made for, and the packed RGB view; each with the sampler's S#.
        const VkDescriptorSet set0 = rt::d3d12::bindless_set(device.p);
        const VkDescriptorSetLayout set0_layout = rt::d3d12::bindless_set_layout(device.p);
        EXPECT(set0 && set0_layout);
#ifdef LEONIDA_TEST_CS_SAMPLE_SPV
        const UINT tables[] = {own, copied, unknown, packed_slot};
        SampleKeys keys[std::size(tables)] = {};
        for (UINT k = 0; k < std::size(tables); ++k) {
            std::memcpy(keys[k].tsharp, view_bytes(tables[k]), sizeof(keys[k].tsharp));
            std::memcpy(keys[k].ssharp, sampler_bytes, sizeof(keys[k].ssharp));
            keys[k].first = k * texels;
        }
        EXPECT(sample_through_set(d, set0, set0_layout, keys, results.data()));
        sampled_through_set = true;
        if (bc.p) {
            SampleKeys bc_key = keys[0];
            std::memcpy(bc_key.tsharp, view_bytes(null_slot), sizeof(bc_key.tsharp));
            std::array<std::uint8_t, bytes> samples{};
            EXPECT(sample_through_set(d, set0, set0_layout, {&bc_key, 1}, samples.data()));
            constexpr std::uint8_t colors[2][2][4] = {{{255, 0, 0, 255}, {0, 255, 0, 255}},
                                                      {{0, 0, 255, 255}, {255, 255, 255, 255}}};
            for (UINT y = 0; y < size; ++y) {
                for (UINT x = 0; x < size; ++x) {
                    // cs_sample.comp samples at the centres of an 8x8 grid with nearest filtering.
                    const UINT tex_x = (2 * x + 1) * bc_width / (2 * size);
                    const UINT tex_y = (2 * y + 1) * bc_height / (2 * size);
                    EXPECT(std::memcmp(samples.data() + 4 * (y * size + x), colors[tex_y / 4][tex_x / 4], 4) == 0);
                }
            }
            std::printf("OPTIONS8: 64 point samples of a 7x5 BC1 top mip matched\n");
        }

        SampleKeys volumes[2]{};
        std::array<std::uint8_t, 2 * bytes> volume_pixels{};
        for (UINT mip = 0; mip < 2; ++mip) {
            std::memcpy(volumes[mip].tsharp, view_bytes(mip ? volume_mip : volume_all), sizeof(volumes[mip].tsharp));
            std::memcpy(volumes[mip].ssharp, sampler_bytes, sizeof(volumes[mip].ssharp));
            volumes[mip].first = mip * texels;
        }
        EXPECT(sample_through_set(d, set0, set0_layout, volumes, volume_pixels.data()));
        unsigned volume_errors = 0;
        for (UINT mip = 0; mip < 2; ++mip)
            for (UINT i = 0; i < texels; ++i) {
                const UINT x = i % 4, y = (i / 4) % 4, z = i / 16;
                const std::uint8_t expected[]{
                    std::uint8_t((mip ? x : 2 * x + 1) * 19), std::uint8_t((mip ? y : 2 * y + 1) * 23),
                    std::uint8_t((mip ? z / 2 : z) * 53), std::uint8_t(mip ? 101 : 237)};
                if (std::memcmp(&volume_pixels[(mip * texels + i) * 4], expected, 4)) ++volume_errors;
            }
        std::printf("volume SRVs: %u of %u sampled RGBA texels differ\n", volume_errors, 2 * texels);
        EXPECT(volume_errors == 0);

        // The border samplers, one tile to the right: every texel is the border.
        SampleKeys border_keys[3] = {};
        const UINT border_views[3] = {uint_slot, uint_slot, own};
        const UINT border_samplers[3] = {uint_black, uint_white, float_white};
        for (UINT k = 0; k < 3; ++k) {
            std::memcpy(border_keys[k].tsharp, view_bytes(border_views[k]), sizeof(border_keys[k].tsharp));
            std::memcpy(border_keys[k].ssharp, sampler_at(border_samplers[k]), sizeof(border_keys[k].ssharp));
            border_keys[k].first = k * texels;
            border_keys[k].integer = border_views[k] == uint_slot;
        }
        EXPECT(sample_through_set(d, set0, set0_layout, border_keys, borders.data()));
#endif

        // A destroyed texture takes its views out of the set: the next ExecuteCommandLists writes that, and nothing
        // may still name the destroyed image views (the validation layer reports a set that does).
        bc.reset();
        packed.reset();
        texture.reset();
        EXPECT(SUCCEEDED(allocator->Reset()) && SUCCEEDED(list->Reset(allocator.p, nullptr)));
        EXPECT(list->Close() == S_OK);
        queue->ExecuteCommandLists(1, lists);
        EXPECT(rt::d3d12::last_execute_result(queue.p) == leonida::ReplayResult::success);
        EXPECT(SUCCEEDED(queue->Signal(fence.p, 2)));
        EXPECT(SUCCEEDED(fence->SetEventOnCompletion(2, nullptr)));
#ifdef LEONIDA_TEST_CS_SAMPLE_SPV
        // The view's bytes are still in the slot, but they find nothing any more: the fallback.
        SampleKeys gone = keys[0];
        gone.first = 0;
        EXPECT(sample_through_set(d, set0, set0_layout, {&gone, 1}, after_destruction.data()));
#endif

        struct PackedFormat {
            DXGI_FORMAT dxgi;
            VkFormat vk;
            const char* name;
        };
        constexpr PackedFormat packed_formats[] = {
            {DXGI_FORMAT_B4G4R4A4_UNORM, VK_FORMAT_A4R4G4B4_UNORM_PACK16, "B4G4R4A4"},
            {DXGI_FORMAT_B5G5R5A1_UNORM, VK_FORMAT_A1R5G5B5_UNORM_PACK16, "B5G5R5A1"}
        };
        UINT64 packed_signal = 2;
        for (const auto& format : packed_formats) {
            D3D12_RESOURCE_DESC format_desc = texture_desc;
            format_desc.Format = format.dxgi;
            Ref<ID3D12Resource> format_texture;
            EXPECT(SUCCEEDED(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &format_desc,
                                                             D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                                             IID_PPV_ARGS(&format_texture.p))));
            D3D12_PLACED_SUBRESOURCE_FOOTPRINT format_footprint{};
            UINT64 row_size = 0;
            device->GetCopyableFootprints(&format_desc, 0, 1, 49152, &format_footprint, nullptr, &row_size, nullptr);
            EXPECT(format_footprint.Footprint.Format == format.dxgi && row_size == size * 2 &&
                   format_footprint.Footprint.RowPitch == 256);
            constexpr std::uint16_t b5_words[] = {0xfc00, 0x7c00, 0x83e1, 0xc117};
            constexpr std::array<std::uint8_t, 4> b5_rgba[] = {
                {255, 0, 0, 255}, {255, 0, 0, 0}, {0, 255, 8, 255}, {132, 66, 189, 255}
            };
            std::array<std::array<std::uint8_t, 4>, texels> expected{};
            for (UINT y = 0; y < size; ++y)
                for (UINT x = 0; x < size; ++x) {
                    auto& rgba = expected[y * size + x];
                    std::uint16_t word;
                    if (format.dxgi == DXGI_FORMAT_B5G5R5A1_UNORM) {
                        const UINT i = (x + y) % std::size(b5_words);
                        word = b5_words[i];
                        rgba = b5_rgba[i];
                    } else {
                        const UINT r = x, g = y + 8, b = (x + y) & 15, a = 15 - x;
                        word = std::uint16_t(b | (g << 4) | (r << 8) | (a << 12));
                        rgba = {std::uint8_t(r * 17), std::uint8_t(g * 17), std::uint8_t(b * 17), std::uint8_t(a * 17)};
                    }
                    std::memcpy(static_cast<std::uint8_t*>(mapped) + format_footprint.Offset +
                                    y * format_footprint.Footprint.RowPitch + x * 2,
                                &word, 2);
                }
            // Set 0 takes in a view at the next ExecuteCommandLists, so it is made before the upload runs.
            device->CreateShaderResourceView(format_texture.p, nullptr, view_slot(null_slot));
            EXPECT(SUCCEEDED(allocator->Reset()) && SUCCEEDED(list->Reset(allocator.p, nullptr)));
            D3D12_TEXTURE_COPY_LOCATION to_packed{format_texture.p, D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, {}};
            to_packed.SubresourceIndex = 0;
            D3D12_TEXTURE_COPY_LOCATION from_packed{upload.p, D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT, {}};
            from_packed.PlacedFootprint = format_footprint;
            list->CopyTextureRegion(&to_packed, 0, 0, 0, &from_packed, nullptr);
            const D3D12_RESOURCE_BARRIER readable =
                transition(format_texture.p, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            list->ResourceBarrier(1, &readable);
            EXPECT(list->Close() == S_OK);
            queue->ExecuteCommandLists(1, lists);
            EXPECT(rt::d3d12::last_execute_result(queue.p) == leonida::ReplayResult::success);
            ++packed_signal;
            EXPECT(SUCCEEDED(queue->Signal(fence.p, packed_signal)) &&
                   SUCCEEDED(fence->SetEventOnCompletion(packed_signal, nullptr)));

            g::Image packed_image{};
            EXPECT(image_at(null_slot, packed_image));
            EXPECT(packed_image.format == g::format_from_vulkan(format.vk) && packed_image.format != 0);
            if (format.dxgi == DXGI_FORMAT_B5G5R5A1_UNORM) EXPECT(packed_image.swizzle == g::channels_bgra);
#ifdef LEONIDA_TEST_CS_SAMPLE_SPV
            SampleKeys packed_keys{};
            std::memcpy(packed_keys.tsharp, view_bytes(null_slot), sizeof(packed_keys.tsharp));
            std::memcpy(packed_keys.ssharp, sampler_bytes, sizeof(packed_keys.ssharp));
            std::vector<std::uint8_t> packed_results(bytes);
            EXPECT(sample_through_set(d, set0, set0_layout, {&packed_keys, 1}, packed_results.data()));
            for (UINT i = 0; i < texels; ++i) {
                const auto& want = expected[i];
                if (std::memcmp(&packed_results[i * 4], want.data(), 4) != 0) {
                    std::fprintf(stderr, "%s texel %u: got %02x %02x %02x %02x, want %02x %02x %02x %02x\n", format.name, i,
                                 packed_results[i * 4], packed_results[i * 4 + 1], packed_results[i * 4 + 2],
                                 packed_results[i * 4 + 3], want[0], want[1], want[2], want[3]);
                    EXPECT(false);
                }
            }
            std::printf("%s sampled: %u texels matched\n", format.name, texels);
#endif
        }

        upload.reset();
        volume.reset();
        uint_texture.reset();
        view_heap.reset();
        sampler_heap.reset();
        list.reset();
        allocator.reset();
        queue.reset();
        fence.reset();
        EXPECT(rt::d3d12_xs::device_proxy_ref_count(device.p) == 1);
    }

    if (!sampled_through_set) {
        if (d.validation_errors()) {
            std::fprintf(stderr, "%u validation errors\n", d.validation_errors());
            return 1;
        }
        std::printf("descriptors: copies matched; views and samplers wrote the T#s and S#s expected; the sampling "
                    "shader was not built (no glslangValidator)\n");
        return 0;
    }

    // The view's slot and its copy give the texture, texel for texel; the unknown descriptor gives the fallback
    // image, one colour everywhere.
    unsigned mismatches = 0;
    for (UINT i = 0; i < texels; ++i) {
        // Each RGB channel is mantissa * 2^(exponent - 24). The shader packs the result to RGBA8.
        constexpr std::uint8_t packed_expected[][4] = {{16, 32, 64, 255}, {32, 64, 128, 255},
                                                       {64, 128, 255, 255}, {64, 16, 32, 255}};
        const auto* packed_want = packed_expected[(i % size + i / size) % 4];
        const auto* packed_got = results.data() + 3 * bytes + i * 4;
        if (std::memcmp(packed_got, packed_want, 4) != 0 && mismatches++ < 4)
            std::fprintf(stderr, "RGB9E5 texel %u: got %u,%u,%u,%u, want %u,%u,%u,%u\n", i,
                         packed_got[0], packed_got[1], packed_got[2], packed_got[3],
                         packed_want[0], packed_want[1], packed_want[2], packed_want[3]);
        const auto want = texel(i % size, i / size);
        for (UINT k = 0; k < 2; ++k) {
            const std::uint8_t* got = results.data() + k * bytes + i * 4;
            if (std::memcmp(got, want.data(), 4) != 0 && mismatches++ < 4)
                std::fprintf(stderr, "table %u texel %u: got %u,%u,%u,%u, want %u,%u,%u,%u\n", k, i, unsigned(got[0]),
                             unsigned(got[1]), unsigned(got[2]), unsigned(got[3]), unsigned(want[0]), unsigned(want[1]),
                             unsigned(want[2]), unsigned(want[3]));
        }
        if (std::memcmp(results.data() + 2 * bytes + i * 4, results.data() + 2 * bytes, 4) != 0 && mismatches++ < 4)
            std::fprintf(stderr, "fallback texel %u differs from texel 0\n", i);
    }
    EXPECT(!mismatches);
    EXPECT(std::memcmp(results.data() + 2 * bytes, texel(0, 0).data(), 4) != 0);  // not the texture
    // After the texture was destroyed its T# finds the fallback too: the same colour as the unknown descriptor.
    for (UINT i = 0; i < texels; ++i) {
        if (std::memcmp(after_destruction.data() + i * 4, results.data() + 2 * bytes, 4) != 0 && mismatches++ < 4)
            std::fprintf(stderr, "texel %u after the texture was destroyed is not the fallback\n", i);
    }
    EXPECT(!mismatches);
    // UINT (0,0,0,0), UINT (1,1,1,1) and float white borders: 0, 1 and 255 in every channel of every texel.
    const std::uint8_t border_values[3] = {0, 1, 255};
    const char* border_names[3] = {"UINT transparent black", "UINT opaque white", "float opaque white"};
    for (UINT k = 0; k < 3; ++k)
        for (UINT i = 0; i < texels * 4; ++i) {
            const std::uint8_t got = borders[k * bytes + i];
            if (got != border_values[k] && mismatches++ < 4)
                std::fprintf(stderr, "%s border, texel %u channel %u: got %u, want %u\n", border_names[k], i / 4,
                             i % 4, unsigned(got), unsigned(border_values[k]));
        }
    EXPECT(!mismatches);
    if (d.validation_errors()) {
        std::fprintf(stderr, "%u validation errors\n", d.validation_errors());
        return 1;
    }
    std::printf("descriptors: copies matched; a T# and an S# from CreateShaderResourceView and CreateSampler were "
                "found in the device's bindless set by a shader, %u RGBA8 and RGB9E5 texels matched, and an unknown descriptor "
                "and a destroyed texture's gave the fallback; UINT black, UINT white and float white borders read "
                "0, 1 and 255\n", texels);
    return 0;
}
