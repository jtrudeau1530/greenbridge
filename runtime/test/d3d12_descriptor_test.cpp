// Descriptors through the Xbox-ABI proxy over the Vulkan device.
//   - CopyDescriptors and CopyDescriptorsSimple, checked on the bytes of the heap slots.
//   - Texture views and samplers (milestone M2, the device's part): CreateShaderResourceView,
//     CreateUnorderedAccessView and CreateSampler write a T# or an S# into the heap slot and put the view or the
//     sampler into the device's bindless set 0. The slots are decoded and checked. Then a compute shader samples
//     an 8x8 RGBA8 texture through its own slot and a copy, a fallback image, and two packed textures.
//     RGB9E5 varies the shared exponent and B5G5R5A1 varies color and alpha. Sampling one tile to the right
//     distinguishes the wrapping sampler from the clamping fallback.
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
    std::vector<std::uint8_t> results(5 * bytes), after_destruction(bytes);
    bool sampled_through_set = false;
    {
        rt::d3d12::VulkanDeviceInfo info{};
        info.get_instance_proc = d.instance_proc;
        info.instance = d.instance;
        info.physical = d.physical;
        info.device = d.device;
        info.queue = d.queue;
        info.queue_family = d.family;
        info.bindless = d.bindless;
        info.bindless_partially_bound = d.partially_bound;
        ID3D12Device* inner = nullptr;
        EXPECT(SUCCEEDED(rt::d3d12::create_vulkan_device(info, IID_PPV_ARGS(&inner))));
        Ref<ID3D12Device> device;
        EXPECT(SUCCEEDED(rt::d3d12_xs::create_device_proxy(inner, __uuidof(ID3D12Device),
                                                           reinterpret_cast<void**>(&device.p))));
        rt::d3d12::set_device_outer(inner, device.p);  // the proxy now owns `inner`

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

        // Shader-visible heaps, start handles in the Xbox shape.
        const UINT step = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        const UINT sampler_step = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER);
        Ref<ID3D12DescriptorHeap> view_heap, sampler_heap;
        D3D12_DESCRIPTOR_HEAP_DESC heap_desc{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 9,
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
        // A volume has no 2D view: neither the default view nor an explicit one may become a 2D-array T#.
        D3D12_RESOURCE_DESC volume_desc = texture_desc;
        volume_desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE3D;
        volume_desc.DepthOrArraySize = 4;
        Ref<ID3D12Resource> volume;
        EXPECT(SUCCEEDED(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &volume_desc,
                                                         D3D12_RESOURCE_STATE_COMMON, nullptr,
                                                         IID_PPV_ARGS(&volume.p))));
        std::memset(view_bytes(null_slot), 0x77, step);
        device->CreateShaderResourceView(volume.p, nullptr, view_slot(null_slot));
        EXPECT(null_at(null_slot));
        D3D12_SHADER_RESOURCE_VIEW_DESC flat_view{};
        flat_view.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        flat_view.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
        flat_view.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        flat_view.Texture2D.MipLevels = 1;
        std::memset(view_bytes(null_slot), 0x77, step);
        device->CreateShaderResourceView(volume.p, &flat_view, view_slot(null_slot));
        EXPECT(null_at(null_slot));
        volume.reset();
        cube.reset();
        writable.reset();

        struct PackedUpload {
            Ref<ID3D12Resource> texture;
            D3D12_PLACED_SUBRESOURCE_FOOTPRINT placed{};
        };
        const auto stage_packed = [&](PackedUpload& target, DXGI_FORMAT format, UINT slot_index, UINT64 offset,
                                      const auto& values) {
            D3D12_RESOURCE_DESC desc = texture_desc;
            desc.Format = format;
            if (FAILED(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &desc,
                                                       D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                                       IID_PPV_ARGS(&target.texture.p)))) return false;
            device->CreateShaderResourceView(target.texture.p, nullptr, view_slot(slot_index));
            UINT64 row_bytes = 0;
            device->GetCopyableFootprints(&desc, 0, 1, offset, &target.placed, nullptr, &row_bytes, nullptr);
            constexpr auto texel_bytes = sizeof(values[0]);
            if (target.placed.Offset != offset || row_bytes != size * texel_bytes ||
                target.placed.Footprint.RowPitch != D3D12_TEXTURE_DATA_PITCH_ALIGNMENT) return false;
            for (UINT y = 0; y < size; ++y) {
                auto* row = static_cast<std::uint8_t*>(mapped) + offset + y * target.placed.Footprint.RowPitch;
                for (UINT x = 0; x < size; ++x)
                    std::memcpy(row + x * texel_bytes, &values[(x + y) % std::size(values)], texel_bytes);
            }
            return true;
        };
        constexpr auto rgb9e5 = [](UINT r, UINT g, UINT b, UINT e) { return r | (g << 9) | (b << 18) | (e << 27); };
        constexpr UINT rgb9e5_values[] = {rgb9e5(64, 128, 256, 14), rgb9e5(64, 128, 256, 15),
                                          rgb9e5(64, 128, 256, 16), rgb9e5(256, 64, 128, 14)};
        constexpr std::uint16_t bgr5a1_values[] = {0xfc00, 0x7c00, 0x83e1, 0xc117};
        constexpr UINT rgb9e5_slot = 0, bgr5a1_slot = 8;
        PackedUpload rgb9e5_upload, bgr5a1_upload;
        EXPECT(stage_packed(rgb9e5_upload, DXGI_FORMAT_R9G9B9E5_SHAREDEXP, rgb9e5_slot, 4096, rgb9e5_values));
        EXPECT(stage_packed(bgr5a1_upload, DXGI_FORMAT_B5G5R5A1_UNORM, bgr5a1_slot, 8192, bgr5a1_values));
        g::Image bgr5a1_image{};
        EXPECT(image_at(bgr5a1_slot, bgr5a1_image));
        EXPECT(bgr5a1_image.format == g::format_from_vulkan(VK_FORMAT_A1R5G5B5_UNORM_PACK16));
        EXPECT(bgr5a1_image.swizzle == g::channels_bgra);

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

        const auto copy_for_sampling = [&](ID3D12Resource* dst, const D3D12_PLACED_SUBRESOURCE_FOOTPRINT& footprint) {
            D3D12_TEXTURE_COPY_LOCATION to{}, from{};
            to.pResource = dst;
            to.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
            from.pResource = upload.p;
            from.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
            from.PlacedFootprint = footprint;
            list->CopyTextureRegion(&to, 0, 0, 0, &from, nullptr);
            const auto barrier = transition(dst, D3D12_RESOURCE_STATE_COPY_DEST,
                                            D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            list->ResourceBarrier(1, &barrier);
        };
        copy_for_sampling(texture.p, placed);
        copy_for_sampling(rgb9e5_upload.texture.p, rgb9e5_upload.placed);
        copy_for_sampling(bgr5a1_upload.texture.p, bgr5a1_upload.placed);
        EXPECT(list->Close() == S_OK);
        ID3D12CommandList* lists[] = {list.p};
        queue->ExecuteCommandLists(1, lists);
        EXPECT(rt::d3d12::last_execute_result(queue.p) == leonida::ReplayResult::success);
        EXPECT(SUCCEEDED(queue->Signal(fence.p, 1)));
        EXPECT(SUCCEEDED(fence->SetEventOnCompletion(1, nullptr)));


        const VkDescriptorSet set0 = rt::d3d12::bindless_set(device.p);
        const VkDescriptorSetLayout set0_layout = rt::d3d12::bindless_set_layout(device.p);
        EXPECT(set0 && set0_layout);
#ifdef LEONIDA_TEST_CS_SAMPLE_SPV
        const UINT tables[] = {own, copied, unknown, rgb9e5_slot, bgr5a1_slot};
        SampleKeys keys[std::size(tables)] = {};
        for (UINT k = 0; k < std::size(tables); ++k) {
            std::memcpy(keys[k].tsharp, view_bytes(tables[k]), sizeof(keys[k].tsharp));
            std::memcpy(keys[k].ssharp, sampler_bytes, sizeof(keys[k].ssharp));
            keys[k].first = k * texels;
        }
        EXPECT(sample_through_set(d, set0, set0_layout, keys, results.data()));
        sampled_through_set = true;
#endif

        // A destroyed texture takes its views out of the set: the next ExecuteCommandLists writes that, and nothing
        // may still name the destroyed image views (the validation layer reports a set that does).
        rgb9e5_upload.texture.reset();
        bgr5a1_upload.texture.reset();
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

        upload.reset();
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
    const auto check_sample = [&](UINT table, const char* name, UINT i, const std::uint8_t* want) {
        const auto* got = results.data() + table * bytes + i * 4;
        if (std::memcmp(got, want, 4) != 0 && mismatches++ < 4)
            std::fprintf(stderr, "%s texel %u: got %u,%u,%u,%u, want %u,%u,%u,%u\n", name, i,
                         unsigned(got[0]), unsigned(got[1]), unsigned(got[2]), unsigned(got[3]),
                         unsigned(want[0]), unsigned(want[1]), unsigned(want[2]), unsigned(want[3]));
    };
    constexpr std::uint8_t rgb9e5_expected[][4] = {{16, 32, 64, 255}, {32, 64, 128, 255},
                                                   {64, 128, 255, 255}, {64, 16, 32, 255}};
    constexpr std::uint8_t bgr5a1_expected[][4] = {{255, 0, 0, 255}, {255, 0, 0, 0},
                                                   {0, 255, 8, 255}, {132, 66, 189, 255}};
    for (UINT i = 0; i < texels; ++i) {
        const UINT pattern = (i % size + i / size) % 4;
        check_sample(3, "RGB9E5", i, rgb9e5_expected[pattern]);
        check_sample(4, "B5G5R5A1", i, bgr5a1_expected[pattern]);
        const auto want = texel(i % size, i / size);
        check_sample(0, "RGBA8", i, want.data());
        check_sample(1, "copied RGBA8", i, want.data());
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
    if (d.validation_errors()) {
        std::fprintf(stderr, "%u validation errors\n", d.validation_errors());
        return 1;
    }
    std::printf("descriptors: copies matched; a T# and an S# from CreateShaderResourceView and CreateSampler were "
                "found in the device's bindless set by a shader, %u texels matched for each of RGBA8, RGB9E5 and "
                "B5G5R5A1, and an unknown descriptor and a destroyed texture's gave the fallback\n", texels);
    return 0;
}
