#include "predicated_copy_controls.h"
// Record synthetic draws between the physical query boundaries emitted by the Leonida replay queue.
#include "vk_test_device.h"
#include "../src/vulkan_backend.h"

#include <cstring>
#include <fstream>

#define EXPECT(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); return 1; } \
} while (0)

namespace {

using namespace leonida;
constexpr std::uint32_t side = 8;
constexpr std::uint32_t query_count = 4;

VkShaderModule load_shader(vktest::Device& d, const char* path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    const std::streamoff bytes = file.tellg();
    if (bytes <= 0 || bytes % 4 != 0) return VK_NULL_HANDLE;
    std::vector<std::uint32_t> code(static_cast<std::size_t>(bytes) / 4);
    file.seekg(0);
    if (!file.read(reinterpret_cast<char*>(code.data()), bytes)) return VK_NULL_HANDLE;
    VkShaderModuleCreateInfo info{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    info.codeSize = static_cast<std::size_t>(bytes);
    info.pCode = code.data();
    VkShaderModule module = VK_NULL_HANDLE;
    if (d.fn<PFN_vkCreateShaderModule>("vkCreateShaderModule")(d.device, &info, nullptr, &module) != VK_SUCCESS)
        return VK_NULL_HANDLE;
    return module;
}

struct Fixture {
    vktest::Device& d;
    VkShaderModule vs = VK_NULL_HANDLE, fs = VK_NULL_HANDLE;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkQueryPool occlusion = VK_NULL_HANDLE, second = VK_NULL_HANDLE, statistics = VK_NULL_HANDLE;

    ~Fixture() {
        const auto destroy_pool = d.fn<PFN_vkDestroyQueryPool>("vkDestroyQueryPool");
        const auto destroy_shader = d.fn<PFN_vkDestroyShaderModule>("vkDestroyShaderModule");
        if (statistics) destroy_pool(d.device, statistics, nullptr);
        if (second) destroy_pool(d.device, second, nullptr);
        if (occlusion) destroy_pool(d.device, occlusion, nullptr);
        if (pipeline) d.fn<PFN_vkDestroyPipeline>("vkDestroyPipeline")(d.device, pipeline, nullptr);
        if (layout) d.fn<PFN_vkDestroyPipelineLayout>("vkDestroyPipelineLayout")(d.device, layout, nullptr);
        if (fs) destroy_shader(d.device, fs, nullptr);
        if (vs) destroy_shader(d.device, vs, nullptr);
    }

    bool create(const char* vertex, const char* fragment) {
        vs = load_shader(d, vertex);
        fs = load_shader(d, fragment);
        if (!vs || !fs) return false;
        VkPipelineLayoutCreateInfo layout_info{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        if (d.fn<PFN_vkCreatePipelineLayout>("vkCreatePipelineLayout")(d.device, &layout_info, nullptr, &layout) !=
            VK_SUCCESS)
            return false;
        const VkPipelineShaderStageCreateInfo stages[] = {
            {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_VERTEX_BIT, vs, "main"},
            {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_FRAGMENT_BIT, fs, "main"},
        };
        const VkPipelineVertexInputStateCreateInfo vi{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
        VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
        ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        const VkViewport vp{0, 0, float(side), float(side), 0, 1};
        const VkRect2D scissor{{}, {side, side}};
        VkPipelineViewportStateCreateInfo viewport{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
        viewport.viewportCount = viewport.scissorCount = 1;
        viewport.pViewports = &vp;
        viewport.pScissors = &scissor;
        VkPipelineRasterizationStateCreateInfo raster{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
        raster.lineWidth = 1;
        VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
        ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
        const VkPipelineColorBlendStateCreateInfo blend{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
        const VkPipelineRenderingCreateInfo rendering{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
        const auto scissor_state = VK_DYNAMIC_STATE_SCISSOR;
        VkPipelineDynamicStateCreateInfo dynamic{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
        dynamic.dynamicStateCount = 1;
        dynamic.pDynamicStates = &scissor_state;
        VkGraphicsPipelineCreateInfo info{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, &rendering};
        info.stageCount = 2;
        info.pStages = stages;
        info.pVertexInputState = &vi;
        info.pInputAssemblyState = &ia;
        info.pViewportState = &viewport;
        info.pRasterizationState = &raster;
        info.pMultisampleState = &ms;
        info.pColorBlendState = &blend;
        info.pDynamicState = &dynamic;
        info.layout = layout;
        if (d.fn<PFN_vkCreateGraphicsPipelines>("vkCreateGraphicsPipelines")(
                d.device, VK_NULL_HANDLE, 1, &info, nullptr, &pipeline) != VK_SUCCESS)
            return false;
        const auto pool = d.fn<PFN_vkCreateQueryPool>("vkCreateQueryPool");
        VkQueryPoolCreateInfo query{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
        query.queryType = VK_QUERY_TYPE_OCCLUSION;
        query.queryCount = query_count;
        if (pool(d.device, &query, nullptr, &occlusion) != VK_SUCCESS ||
            pool(d.device, &query, nullptr, &second) != VK_SUCCESS) return false;
        if (d.pipeline_statistics_query) {
            query.queryType = VK_QUERY_TYPE_PIPELINE_STATISTICS;
            query.queryCount = 1;
            query.pipelineStatistics = 0x7ff;  // the same counters as the D3D12 heap
            if (pool(d.device, &query, nullptr, &statistics) != VK_SUCCESS) return false;
        }
        return true;
    }

    void draw(VkCommandBuffer cb, std::uint32_t extent) const {
        VkRenderingInfo rendering{VK_STRUCTURE_TYPE_RENDERING_INFO};
        rendering.renderArea.extent = {side, side};
        rendering.layerCount = 1;
        d.fn<PFN_vkCmdBeginRendering>("vkCmdBeginRendering")(cb, &rendering);
        d.fn<PFN_vkCmdBindPipeline>("vkCmdBindPipeline")(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
        const VkRect2D scissor{{}, {extent, extent}};
        d.fn<PFN_vkCmdSetScissor>("vkCmdSetScissor")(cb, 0, 1, &scissor);
        d.fn<PFN_vkCmdDraw>("vkCmdDraw")(cb, 3, 1, 0, 0);
        d.fn<PFN_vkCmdEndRendering>("vkCmdEndRendering")(cb);
    }
};

// Main's replay has no D3D12 draws yet. Insert one native Vulkan draw while each actual query recorded
// by the queue is active, immediately before its end. Everything else goes to the real Vulkan driver.
struct Draws {
    Fixture& fixture;
    std::vector<std::uint32_t> extents;
    std::size_t next = 0;
    bool overflow = false;
};
Draws* draws = nullptr;

VKAPI_ATTR void VKAPI_CALL end_query_with_draw(VkCommandBuffer cb, VkQueryPool pool, std::uint32_t index) {
    if (draws->next == draws->extents.size()) {
        draws->overflow = true;
    } else if (const auto extent = draws->extents[draws->next++]) {
        draws->fixture.draw(cb, extent);
    }
    draws->fixture.d.fn<PFN_vkCmdEndQuery>("vkCmdEndQuery")(cb, pool, index);
}

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL get_device_proc(VkDevice device, const char* name) {
    if (std::strcmp(name, "vkCmdEndQuery") == 0) return reinterpret_cast<PFN_vkVoidFunction>(end_query_with_draw);
    return draws->fixture.d.device_proc(device, name);
}

int run(vktest::Device& d, const char* vertex, const char* fragment) {
    std::uint32_t family_count = 0;
    const auto families = d.instance_fn<PFN_vkGetPhysicalDeviceQueueFamilyProperties>(
        "vkGetPhysicalDeviceQueueFamilyProperties");
    families(d.physical, &family_count, nullptr);
    std::vector<VkQueueFamilyProperties> queues(family_count);
    families(d.physical, &family_count, queues.data());
    if (!d.occlusion_query_precise || !(queues[d.family].queueFlags & VK_QUEUE_COMPUTE_BIT)) return 77;

    Fixture fixture{d};
    EXPECT(fixture.create(vertex, fragment));
    constexpr std::uint32_t bursts = 130;
    Draws schedule{fixture, {8, 4, 2, 8, 4, 2, 3}};
    schedule.extents.insert(schedule.extents.end(), 2 * bursts + 1, 1);
    schedule.extents.insert(schedule.extents.end(), {2, 0, 0, 0});
    draws = &schedule;
    VulkanContext context(get_device_proc, d.device, true, d.conditional_rendering);
    EXPECT(context.loaded());
    VkPhysicalDeviceMemoryProperties memory{};
    const auto get_memory = d.instance_fn<PFN_vkGetPhysicalDeviceMemoryProperties>("vkGetPhysicalDeviceMemoryProperties");
    get_memory(d.physical, &memory);
    context.set_occlusion_resolve_memory(memory);
    VkPhysicalDeviceProperties properties{};
    d.instance_fn<PFN_vkGetPhysicalDeviceProperties>("vkGetPhysicalDeviceProperties")(d.physical, &properties);
    context.configure_predication(get_device_proc, memory, properties.limits.maxStorageBufferRange, d.family);
    VulkanFence gate(context, 0);
    PredicatedCopyControl copy_control;
    EXPECT(copy_control.create(d, context, properties.limits));
    VulkanQueue queue(context, d.queue, d.family);
    EXPECT(queue.ok() && gate.semaphore());
    struct OpenGate { VulkanFence& fence; ~OpenGate() { fence.signal(1); } } open_gate{gate};
    constexpr QueryHeapId a = 1, b = 2, stats = 3;
    constexpr auto occ = d3d12_query::occlusion;
    constexpr auto binary = d3d12_query::binary_occlusion;
    for (const auto& [id, pool] : {std::pair{a, fixture.occlusion}, std::pair{b, fixture.second}})
        context.add_query_pool(id, {pool, d3d12_query_heap::occlusion, query_count, true,
                                    std::vector<bool>(query_count, false)});

    constexpr ResourceId dst_id = 4, predicate_id = 5;
    constexpr VkDeviceSize binding_offset = 24, size = 160;
    void* mapping = nullptr;
    ResourceBinding dst{};
    dst.buffer = d.buffer(size, VK_BUFFER_USAGE_TRANSFER_DST_BIT, &mapping);
    dst.buffer_offset = binding_offset;
    dst.size = size - binding_offset;
    EXPECT(dst.buffer && mapping);
    std::memset(mapping, 0xcd, size);
    context.add_resource(dst_id, dst);
    if (d.conditional_rendering) {
        void* words = nullptr;
        ResourceBinding predicate{};
        predicate.size = 8;
        predicate.buffer = d.buffer(8, VK_BUFFER_USAGE_CONDITIONAL_RENDERING_BIT_EXT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, &words);
        EXPECT(predicate.buffer && words);
        std::memset(words, 0, 8);
        context.add_resource(predicate_id, predicate);
    }

    CommandList crossing;
    EXPECT(crossing.begin_query({a, occ, 0}));
    EXPECT(crossing.begin_query({a, occ, 1}));
    EXPECT(crossing.end_query({a, occ, 0}));
    EXPECT(crossing.resolve_queries({a, occ, 0, 1, dst_id, 8}));   // B is still active
    EXPECT(crossing.end_query({a, occ, 1}));
    EXPECT(crossing.close());
    CommandList nested;
    EXPECT(nested.begin_query({a, occ, 2}));
    EXPECT(nested.begin_query({b, occ, 2}));                       // same index, different logical heaps
    EXPECT(nested.end_query({b, occ, 2}));
    EXPECT(nested.end_query({a, occ, 2}));
    EXPECT(nested.close());
    const CommandList* first[] = {&crossing, &nested};
    queue.wait(gate.semaphore(), 1);
    EXPECT(queue.execute(first) == ReplayResult::success);

    CommandList later;
    EXPECT(later.resolve_queries({a, occ, 0, 3, dst_id, 24}));
    EXPECT(later.resolve_queries({b, occ, 2, 1, dst_id, 48}));
    EXPECT(later.resolve_queries({b, occ, 0, 1, dst_id, 56}));    // never begun
    EXPECT(later.begin_query({a, binary, 3}));
    EXPECT(later.end_query({a, binary, 3}));
    if (d.conditional_rendering) EXPECT(later.set_predication({predicate_id, 0, d3d12_predication::equal_zero}));
    // Alternate the two compute pipelines on the shared scratch buffer, all under the false predicate.
    EXPECT(later.resolve_queries({a, binary, 3, 1, dst_id, 64}));
    EXPECT(later.resolve_queries({a, occ, 0, 1, dst_id, 72}));
    EXPECT(later.resolve_queries({a, binary, 3, 1, dst_id, 80}));
    EXPECT(copy_control.record(later));
    EXPECT(later.close());
    const CommandList* second[] = {&later};
    EXPECT(queue.execute(second) == ReplayResult::success);

    CommandList chunked;
    EXPECT(chunked.begin_query({a, occ, 3}));
    for (std::uint32_t i = 0; i < bursts; ++i) {
        EXPECT(chunked.begin_query({b, occ, 0}));
        EXPECT(chunked.end_query({b, occ, 0}));
    }
    EXPECT(chunked.end_query({a, occ, 3}));
    EXPECT(chunked.resolve_queries({a, occ, 3, 1, dst_id, 88}));
    EXPECT(chunked.resolve_queries({b, occ, 0, 1, dst_id, 96}));
    EXPECT(chunked.close());
    const CommandList* third[] = {&chunked};
    EXPECT(queue.execute(third) == ReplayResult::success);

    CommandList bundle;
    EXPECT(bundle.begin_query({a, occ, 0}));
    EXPECT(bundle.end_query({a, occ, 0}));
    EXPECT(bundle.close());
    CommandList duplicate;
    EXPECT(duplicate.begin_query({a, occ, 0}));
    EXPECT(duplicate.execute_bundle(bundle));
    EXPECT(duplicate.end_query({a, occ, 0}));
    EXPECT(duplicate.close());
    const CommandList* invalid[] = {&duplicate};
    EXPECT(queue.execute(invalid) == ReplayResult::invalid_query);

    if (fixture.statistics) {
        context.add_query_pool(stats, {fixture.statistics, d3d12_query_heap::pipeline_statistics, 1, false, {false}});
        CommandList incompatible;
        EXPECT(incompatible.begin_query({stats, d3d12_query::pipeline_statistics, 0}));
        EXPECT(incompatible.resolve_queries({a, occ, 3, 1, dst_id, 0}));
        EXPECT(incompatible.end_query({stats, d3d12_query::pipeline_statistics, 0}));
        EXPECT(incompatible.close());
        const CommandList* rejected[] = {&incompatible};
        EXPECT(queue.execute(rejected) == ReplayResult::unsupported);  // the dispatch would change CSInvocations
    }

    // Overwrite every result from the first physical pool while the GPU is still gated. The first two
    // submissions still refer to that pool and must be able to execute once the gate opens.
    CommandList replace;
    for (const auto& [heap, index] : {std::pair{a, 0u}, std::pair{a, 1u}, std::pair{a, 2u}, std::pair{b, 2u}}) {
        EXPECT(replace.begin_query({heap, occ, index}));
        EXPECT(replace.end_query({heap, occ, index}));
    }
    EXPECT(replace.resolve_queries({a, occ, 0, 2, dst_id, 112}));
    EXPECT(replace.close());
    const CommandList* fourth[] = {&replace};
    EXPECT(queue.execute(fourth) == ReplayResult::success);
    EXPECT(!schedule.overflow && schedule.next == schedule.extents.size());
    EXPECT(gate.signal(1));
    EXPECT(queue.idle());

    const std::map<VkDeviceSize, std::uint64_t> expected = {
        {8, 80}, {24, 80}, {32, 20}, {40, 84}, {48, 16}, {56, 0}, {64, 1}, {72, 80}, {80, 1},
        {88, 2 * bursts + 1}, {96, 1}, {112, 4}, {120, 0},
    };
    for (VkDeviceSize offset = 0; offset < size; offset += 8) {
        const auto it = expected.find(offset - binding_offset);
        const auto want = it == expected.end() ? 0xcdcdcdcdcdcdcdcdULL : it->second;
        std::uint64_t actual = 0;
        std::memcpy(&actual, static_cast<const char*>(mapping) + offset, sizeof(actual));
        if (actual != want) {
            std::fprintf(stderr, "at byte %llu: got %llu, wanted %llu\n", static_cast<unsigned long long>(offset),
                         static_cast<unsigned long long>(actual), static_cast<unsigned long long>(want));
            return 1;
        }
    }
    std::printf("crossing 80/20; nested 84/16; %u segments summed; alternating binary results 1/1; "
                "offsets and gated overwrites exact\n",
                2 * bursts + 1);
    EXPECT(copy_control.check());
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    EXPECT(argc == 3);
    vktest::Device device;
    const auto status = device.initialize();
    if (status == vktest::DeviceStatus::unavailable) return 77;
    EXPECT(status == vktest::DeviceStatus::ready);
    const int result = run(device, argv[1], argv[2]);
    EXPECT(device.validation_errors() == 0);
    return result;
}
