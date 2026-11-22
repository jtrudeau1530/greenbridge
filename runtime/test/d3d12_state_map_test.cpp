// D3D12 resource state -> Vulkan layout/access/stage mapping and subresource decoding. Needs the Vulkan headers,
// not a GPU. On Windows the IR's D3D12 constants are also checked against the SDK's d3d12.h.
#ifndef VK_NO_PROTOTYPES
#define VK_NO_PROTOTYPES
#endif
#include "../src/vulkan_replay.h"
#include "../src/vulkan_memory.h"  // resizable_bar, memory_type_order

#ifdef _WIN32
#include <windows.h>
#include <d3d12.h>

namespace s = leonida::d3d12_state;
static_assert(s::common == D3D12_RESOURCE_STATE_COMMON);
static_assert(s::vertex_and_constant_buffer == D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER);
static_assert(s::index_buffer == D3D12_RESOURCE_STATE_INDEX_BUFFER);
static_assert(s::render_target == D3D12_RESOURCE_STATE_RENDER_TARGET);
static_assert(s::unordered_access == D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
static_assert(s::depth_write == D3D12_RESOURCE_STATE_DEPTH_WRITE);
static_assert(s::depth_read == D3D12_RESOURCE_STATE_DEPTH_READ);
static_assert(s::non_pixel_shader_resource == D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
static_assert(s::pixel_shader_resource == D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
static_assert(s::stream_out == D3D12_RESOURCE_STATE_STREAM_OUT);
static_assert(s::indirect_argument == D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT);
static_assert(s::predication == D3D12_RESOURCE_STATE_PREDICATION);
static_assert(s::copy_dest == D3D12_RESOURCE_STATE_COPY_DEST);
static_assert(s::copy_source == D3D12_RESOURCE_STATE_COPY_SOURCE);
static_assert(s::resolve_dest == D3D12_RESOURCE_STATE_RESOLVE_DEST);
static_assert(s::resolve_source == D3D12_RESOURCE_STATE_RESOLVE_SOURCE);
static_assert(s::raytracing_acceleration_structure == D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE);
static_assert(s::shading_rate_source == D3D12_RESOURCE_STATE_SHADING_RATE_SOURCE);
static_assert(s::generic_read == D3D12_RESOURCE_STATE_GENERIC_READ);
static_assert(s::present == D3D12_RESOURCE_STATE_PRESENT);
static_assert(leonida::all_subresources == D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES);
static_assert(leonida::barrier_begin_only == D3D12_RESOURCE_BARRIER_FLAG_BEGIN_ONLY);
static_assert(leonida::barrier_end_only == D3D12_RESOURCE_BARRIER_FLAG_END_ONLY);
static_assert(leonida::clear_depth == D3D12_CLEAR_FLAG_DEPTH);
static_assert(leonida::clear_stencil == D3D12_CLEAR_FLAG_STENCIL);
static_assert(leonida::texture_barrier_discard == D3D12_TEXTURE_BARRIER_FLAG_DISCARD);
static_assert(leonida::d3d12_sync::none == D3D12_BARRIER_SYNC_NONE);
static_assert(leonida::d3d12_sync::all == D3D12_BARRIER_SYNC_ALL);
static_assert(leonida::d3d12_sync::draw == D3D12_BARRIER_SYNC_DRAW);
static_assert(leonida::d3d12_sync::index_input == D3D12_BARRIER_SYNC_INDEX_INPUT);
static_assert(leonida::d3d12_sync::vertex_shading == D3D12_BARRIER_SYNC_VERTEX_SHADING);
static_assert(leonida::d3d12_sync::pixel_shading == D3D12_BARRIER_SYNC_PIXEL_SHADING);
static_assert(leonida::d3d12_sync::depth_stencil == D3D12_BARRIER_SYNC_DEPTH_STENCIL);
static_assert(leonida::d3d12_sync::render_target == D3D12_BARRIER_SYNC_RENDER_TARGET);
static_assert(leonida::d3d12_sync::compute_shading == D3D12_BARRIER_SYNC_COMPUTE_SHADING);
static_assert(leonida::d3d12_sync::raytracing == D3D12_BARRIER_SYNC_RAYTRACING);
static_assert(leonida::d3d12_sync::copy == D3D12_BARRIER_SYNC_COPY);
static_assert(leonida::d3d12_sync::resolve == D3D12_BARRIER_SYNC_RESOLVE);
static_assert(leonida::d3d12_sync::execute_indirect == D3D12_BARRIER_SYNC_EXECUTE_INDIRECT);
static_assert(leonida::d3d12_sync::predication == D3D12_BARRIER_SYNC_PREDICATION);
static_assert(leonida::d3d12_sync::all_shading == D3D12_BARRIER_SYNC_ALL_SHADING);
static_assert(leonida::d3d12_sync::non_pixel_shading == D3D12_BARRIER_SYNC_NON_PIXEL_SHADING);
static_assert(leonida::d3d12_sync::emit_raytracing_acceleration_structure_postbuild_info == D3D12_BARRIER_SYNC_EMIT_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO);
static_assert(leonida::d3d12_sync::clear_unordered_access_view == D3D12_BARRIER_SYNC_CLEAR_UNORDERED_ACCESS_VIEW);
static_assert(leonida::d3d12_sync::video_decode == D3D12_BARRIER_SYNC_VIDEO_DECODE);
static_assert(leonida::d3d12_sync::video_process == D3D12_BARRIER_SYNC_VIDEO_PROCESS);
static_assert(leonida::d3d12_sync::video_encode == D3D12_BARRIER_SYNC_VIDEO_ENCODE);
static_assert(leonida::d3d12_sync::build_raytracing_acceleration_structure == D3D12_BARRIER_SYNC_BUILD_RAYTRACING_ACCELERATION_STRUCTURE);
static_assert(leonida::d3d12_sync::copy_raytracing_acceleration_structure == D3D12_BARRIER_SYNC_COPY_RAYTRACING_ACCELERATION_STRUCTURE);
static_assert(leonida::d3d12_sync::split == D3D12_BARRIER_SYNC_SPLIT);
static_assert(leonida::d3d12_access::common == D3D12_BARRIER_ACCESS_COMMON);
static_assert(leonida::d3d12_access::vertex_buffer == D3D12_BARRIER_ACCESS_VERTEX_BUFFER);
static_assert(leonida::d3d12_access::constant_buffer == D3D12_BARRIER_ACCESS_CONSTANT_BUFFER);
static_assert(leonida::d3d12_access::index_buffer == D3D12_BARRIER_ACCESS_INDEX_BUFFER);
static_assert(leonida::d3d12_access::render_target == D3D12_BARRIER_ACCESS_RENDER_TARGET);
static_assert(leonida::d3d12_access::unordered_access == D3D12_BARRIER_ACCESS_UNORDERED_ACCESS);
static_assert(leonida::d3d12_access::depth_stencil_write == D3D12_BARRIER_ACCESS_DEPTH_STENCIL_WRITE);
static_assert(leonida::d3d12_access::depth_stencil_read == D3D12_BARRIER_ACCESS_DEPTH_STENCIL_READ);
static_assert(leonida::d3d12_access::shader_resource == D3D12_BARRIER_ACCESS_SHADER_RESOURCE);
static_assert(leonida::d3d12_access::stream_output == D3D12_BARRIER_ACCESS_STREAM_OUTPUT);
static_assert(leonida::d3d12_access::indirect_argument == D3D12_BARRIER_ACCESS_INDIRECT_ARGUMENT);
static_assert(leonida::d3d12_access::predication == D3D12_BARRIER_ACCESS_PREDICATION);
static_assert(leonida::d3d12_access::copy_dest == D3D12_BARRIER_ACCESS_COPY_DEST);
static_assert(leonida::d3d12_access::copy_source == D3D12_BARRIER_ACCESS_COPY_SOURCE);
static_assert(leonida::d3d12_access::resolve_dest == D3D12_BARRIER_ACCESS_RESOLVE_DEST);
static_assert(leonida::d3d12_access::resolve_source == D3D12_BARRIER_ACCESS_RESOLVE_SOURCE);
static_assert(leonida::d3d12_access::raytracing_acceleration_structure_read == D3D12_BARRIER_ACCESS_RAYTRACING_ACCELERATION_STRUCTURE_READ);
static_assert(leonida::d3d12_access::raytracing_acceleration_structure_write == D3D12_BARRIER_ACCESS_RAYTRACING_ACCELERATION_STRUCTURE_WRITE);
static_assert(leonida::d3d12_access::shading_rate_source == D3D12_BARRIER_ACCESS_SHADING_RATE_SOURCE);
static_assert(leonida::d3d12_access::video_decode_read == D3D12_BARRIER_ACCESS_VIDEO_DECODE_READ);
static_assert(leonida::d3d12_access::video_decode_write == D3D12_BARRIER_ACCESS_VIDEO_DECODE_WRITE);
static_assert(leonida::d3d12_access::video_process_read == D3D12_BARRIER_ACCESS_VIDEO_PROCESS_READ);
static_assert(leonida::d3d12_access::video_process_write == D3D12_BARRIER_ACCESS_VIDEO_PROCESS_WRITE);
static_assert(leonida::d3d12_access::video_encode_read == D3D12_BARRIER_ACCESS_VIDEO_ENCODE_READ);
static_assert(leonida::d3d12_access::video_encode_write == D3D12_BARRIER_ACCESS_VIDEO_ENCODE_WRITE);
static_assert(leonida::d3d12_access::no_access == D3D12_BARRIER_ACCESS_NO_ACCESS);
static_assert(leonida::d3d12_layout::undefined == std::uint32_t(D3D12_BARRIER_LAYOUT_UNDEFINED));
static_assert(leonida::d3d12_layout::common == std::uint32_t(D3D12_BARRIER_LAYOUT_COMMON));
static_assert(leonida::d3d12_layout::present == std::uint32_t(D3D12_BARRIER_LAYOUT_PRESENT));
static_assert(leonida::d3d12_layout::generic_read == std::uint32_t(D3D12_BARRIER_LAYOUT_GENERIC_READ));
static_assert(leonida::d3d12_layout::render_target == std::uint32_t(D3D12_BARRIER_LAYOUT_RENDER_TARGET));
static_assert(leonida::d3d12_layout::unordered_access == std::uint32_t(D3D12_BARRIER_LAYOUT_UNORDERED_ACCESS));
static_assert(leonida::d3d12_layout::depth_stencil_write == std::uint32_t(D3D12_BARRIER_LAYOUT_DEPTH_STENCIL_WRITE));
static_assert(leonida::d3d12_layout::depth_stencil_read == std::uint32_t(D3D12_BARRIER_LAYOUT_DEPTH_STENCIL_READ));
static_assert(leonida::d3d12_layout::shader_resource == std::uint32_t(D3D12_BARRIER_LAYOUT_SHADER_RESOURCE));
static_assert(leonida::d3d12_layout::copy_source == std::uint32_t(D3D12_BARRIER_LAYOUT_COPY_SOURCE));
static_assert(leonida::d3d12_layout::copy_dest == std::uint32_t(D3D12_BARRIER_LAYOUT_COPY_DEST));
static_assert(leonida::d3d12_layout::resolve_source == std::uint32_t(D3D12_BARRIER_LAYOUT_RESOLVE_SOURCE));
static_assert(leonida::d3d12_layout::resolve_dest == std::uint32_t(D3D12_BARRIER_LAYOUT_RESOLVE_DEST));
static_assert(leonida::d3d12_layout::shading_rate_source == std::uint32_t(D3D12_BARRIER_LAYOUT_SHADING_RATE_SOURCE));
static_assert(leonida::d3d12_layout::video_decode_read == std::uint32_t(D3D12_BARRIER_LAYOUT_VIDEO_DECODE_READ));
static_assert(leonida::d3d12_layout::video_decode_write == std::uint32_t(D3D12_BARRIER_LAYOUT_VIDEO_DECODE_WRITE));
static_assert(leonida::d3d12_layout::video_process_read == std::uint32_t(D3D12_BARRIER_LAYOUT_VIDEO_PROCESS_READ));
static_assert(leonida::d3d12_layout::video_process_write == std::uint32_t(D3D12_BARRIER_LAYOUT_VIDEO_PROCESS_WRITE));
static_assert(leonida::d3d12_layout::video_encode_read == std::uint32_t(D3D12_BARRIER_LAYOUT_VIDEO_ENCODE_READ));
static_assert(leonida::d3d12_layout::video_encode_write == std::uint32_t(D3D12_BARRIER_LAYOUT_VIDEO_ENCODE_WRITE));
static_assert(leonida::d3d12_layout::direct_queue_common == std::uint32_t(D3D12_BARRIER_LAYOUT_DIRECT_QUEUE_COMMON));
static_assert(leonida::d3d12_layout::direct_queue_generic_read == std::uint32_t(D3D12_BARRIER_LAYOUT_DIRECT_QUEUE_GENERIC_READ));
static_assert(leonida::d3d12_layout::direct_queue_unordered_access == std::uint32_t(D3D12_BARRIER_LAYOUT_DIRECT_QUEUE_UNORDERED_ACCESS));
static_assert(leonida::d3d12_layout::direct_queue_shader_resource == std::uint32_t(D3D12_BARRIER_LAYOUT_DIRECT_QUEUE_SHADER_RESOURCE));
static_assert(leonida::d3d12_layout::direct_queue_copy_source == std::uint32_t(D3D12_BARRIER_LAYOUT_DIRECT_QUEUE_COPY_SOURCE));
static_assert(leonida::d3d12_layout::direct_queue_copy_dest == std::uint32_t(D3D12_BARRIER_LAYOUT_DIRECT_QUEUE_COPY_DEST));
static_assert(leonida::d3d12_layout::compute_queue_common == std::uint32_t(D3D12_BARRIER_LAYOUT_COMPUTE_QUEUE_COMMON));
static_assert(leonida::d3d12_layout::compute_queue_generic_read == std::uint32_t(D3D12_BARRIER_LAYOUT_COMPUTE_QUEUE_GENERIC_READ));
static_assert(leonida::d3d12_layout::compute_queue_unordered_access == std::uint32_t(D3D12_BARRIER_LAYOUT_COMPUTE_QUEUE_UNORDERED_ACCESS));
static_assert(leonida::d3d12_layout::compute_queue_shader_resource == std::uint32_t(D3D12_BARRIER_LAYOUT_COMPUTE_QUEUE_SHADER_RESOURCE));
static_assert(leonida::d3d12_layout::compute_queue_copy_source == std::uint32_t(D3D12_BARRIER_LAYOUT_COMPUTE_QUEUE_COPY_SOURCE));
static_assert(leonida::d3d12_layout::compute_queue_copy_dest == std::uint32_t(D3D12_BARRIER_LAYOUT_COMPUTE_QUEUE_COPY_DEST));

template <typename SdkLayout>
constexpr bool video_queue_common_matches_sdk() {
    if constexpr (requires { SdkLayout::D3D12_BARRIER_LAYOUT_VIDEO_QUEUE_COMMON; }) {
        return leonida::d3d12_layout::video_queue_common ==
               std::uint32_t(SdkLayout::D3D12_BARRIER_LAYOUT_VIDEO_QUEUE_COMMON);
    }
    // SDK 10.0.28000.0 omits this legacy enumerator. Its value is 30 in older headers.
    return leonida::d3d12_layout::video_queue_common == 30;
}
static_assert(video_queue_common_matches_sdk<D3D12_BARRIER_LAYOUT>());
static_assert(leonida::d3d12_query_heap::occlusion == std::uint32_t(D3D12_QUERY_HEAP_TYPE_OCCLUSION));
static_assert(leonida::d3d12_query_heap::timestamp == std::uint32_t(D3D12_QUERY_HEAP_TYPE_TIMESTAMP));
static_assert(leonida::d3d12_query_heap::pipeline_statistics == std::uint32_t(D3D12_QUERY_HEAP_TYPE_PIPELINE_STATISTICS));
static_assert(leonida::d3d12_query_heap::so_statistics == std::uint32_t(D3D12_QUERY_HEAP_TYPE_SO_STATISTICS));
static_assert(leonida::d3d12_query_heap::video_decode_statistics ==
              std::uint32_t(D3D12_QUERY_HEAP_TYPE_VIDEO_DECODE_STATISTICS));
static_assert(leonida::d3d12_query_heap::copy_queue_timestamp == std::uint32_t(D3D12_QUERY_HEAP_TYPE_COPY_QUEUE_TIMESTAMP));
static_assert(leonida::d3d12_query_heap::pipeline_statistics1 == std::uint32_t(D3D12_QUERY_HEAP_TYPE_PIPELINE_STATISTICS1));
static_assert(leonida::d3d12_query::occlusion == std::uint32_t(D3D12_QUERY_TYPE_OCCLUSION));
static_assert(leonida::d3d12_query::binary_occlusion == std::uint32_t(D3D12_QUERY_TYPE_BINARY_OCCLUSION));
static_assert(leonida::d3d12_query::timestamp == std::uint32_t(D3D12_QUERY_TYPE_TIMESTAMP));
static_assert(leonida::d3d12_query::pipeline_statistics == std::uint32_t(D3D12_QUERY_TYPE_PIPELINE_STATISTICS));
static_assert(leonida::d3d12_query::so_statistics_stream0 == std::uint32_t(D3D12_QUERY_TYPE_SO_STATISTICS_STREAM0));
static_assert(leonida::d3d12_query::so_statistics_stream3 == std::uint32_t(D3D12_QUERY_TYPE_SO_STATISTICS_STREAM3));
static_assert(leonida::d3d12_query::video_decode_statistics == std::uint32_t(D3D12_QUERY_TYPE_VIDEO_DECODE_STATISTICS));
static_assert(leonida::d3d12_query::pipeline_statistics1 == std::uint32_t(D3D12_QUERY_TYPE_PIPELINE_STATISTICS1));
static_assert(leonida::query_result_size(leonida::d3d12_query::pipeline_statistics) ==
              sizeof(D3D12_QUERY_DATA_PIPELINE_STATISTICS));
#endif

#include <cstdio>

#define CHECK(condition) do {                                    \
    if (!(condition)) {                                          \
        std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); \
        return 1;                                                \
    }                                                            \
} while (0)

int main() {
    using namespace leonida;
    namespace s = d3d12_state;
    const auto layout = [](std::uint32_t states, bool presentable = false) {
        return vulkan_access(states, presentable).layout;
    };

    // Single states.
    CHECK(layout(s::common) == VK_IMAGE_LAYOUT_GENERAL);
    CHECK(layout(s::present, true) == VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);
    CHECK(layout(s::render_target) == VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    CHECK(layout(s::render_target, true) == VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    CHECK(layout(s::unordered_access) == VK_IMAGE_LAYOUT_GENERAL);
    CHECK(layout(s::depth_write) == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
    CHECK(layout(s::depth_read) == VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL);
    CHECK(layout(s::pixel_shader_resource) == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    CHECK(layout(s::non_pixel_shader_resource) == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    CHECK(layout(s::copy_dest) == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    CHECK(layout(s::copy_source) == VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
    CHECK(layout(s::resolve_dest) == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    CHECK(layout(s::resolve_source) == VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);

    // Read-only combinations.
    CHECK(layout(s::pixel_shader_resource | s::non_pixel_shader_resource) == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    CHECK(layout(s::depth_read | s::pixel_shader_resource) == VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL);
    CHECK(layout(s::depth_read | s::pixel_shader_resource | s::non_pixel_shader_resource) ==
          VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL);
    CHECK(layout(s::pixel_shader_resource | s::copy_source) == VK_IMAGE_LAYOUT_GENERAL);
    CHECK(layout(s::depth_read | s::pixel_shader_resource | s::copy_source) == VK_IMAGE_LAYOUT_GENERAL);
    CHECK(layout(s::generic_read) == VK_IMAGE_LAYOUT_GENERAL);

    // Accesses and stages.
    const auto cb = vulkan_access(s::vertex_and_constant_buffer, false);
    CHECK(cb.access & VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT);
    CHECK(cb.access & VK_ACCESS_UNIFORM_READ_BIT);
    CHECK(cb.access & VK_ACCESS_SHADER_READ_BIT);  // root CBV loaded through a physical storage buffer
    CHECK(cb.stages & VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT);
    const auto rt = vulkan_access(s::render_target, false);
    CHECK(rt.access == (VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT));
    CHECK(rt.stages == VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT);
    const auto srv = vulkan_access(s::pixel_shader_resource | s::non_pixel_shader_resource, false);
    CHECK(srv.access == VK_ACCESS_SHADER_READ_BIT);
    CHECK((srv.stages & VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT) && (srv.stages & VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT));
    const auto generic = vulkan_access(s::generic_read, false);
    CHECK(generic.access & VK_ACCESS_INDEX_READ_BIT);
    CHECK(generic.access & VK_ACCESS_INDIRECT_COMMAND_READ_BIT);
    CHECK(generic.access & VK_ACCESS_TRANSFER_READ_BIT);
    CHECK(!(generic.access & (VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT)));
    const auto present = vulkan_access(s::present, true);
    CHECK(present.access == 0 && present.stages == VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
    const auto common = vulkan_access(s::common, false);
    CHECK(common.access == (VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT));
    // An unknown bit falls back to GENERAL with every access.
    const auto unknown = vulkan_access(s::pixel_shader_resource | 0x80000000u, false);
    CHECK(unknown.layout == VK_IMAGE_LAYOUT_GENERAL);
    CHECK(unknown.access & VK_ACCESS_MEMORY_WRITE_BIT);
    CHECK(unknown.stages & VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
    // PREDICATION is INDIRECT_ARGUMENT's bit. With VK_EXT_conditional_rendering it also covers the predicate read,
    // as does GENERIC_READ, which contains it; without the extension those flags would be invalid and stay out.
    static_assert(s::predication == s::indirect_argument);
    const auto predicate = vulkan_access(s::predication, false, true);
    CHECK(predicate.access == (VK_ACCESS_INDIRECT_COMMAND_READ_BIT | VK_ACCESS_CONDITIONAL_RENDERING_READ_BIT_EXT));
    CHECK(predicate.stages ==
          (VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT | VK_PIPELINE_STAGE_CONDITIONAL_RENDERING_BIT_EXT));
    CHECK(vulkan_access(s::generic_read, false, true).access & VK_ACCESS_CONDITIONAL_RENDERING_READ_BIT_EXT);
    CHECK(!(vulkan_access(s::predication, false).access & VK_ACCESS_CONDITIONAL_RENDERING_READ_BIT_EXT));
    CHECK(!(vulkan_access(s::copy_dest, false, true).stages & VK_PIPELINE_STAGE_CONDITIONAL_RENDERING_BIT_EXT));

    // Subresource indices: mip + layer * mips + plane * mips * layers.
    VkImageSubresourceRange range{};
    ResourceBinding color{};
    color.image = reinterpret_cast<VkImage>(std::uintptr_t{1});
    color.mip_levels = 3;
    color.array_layers = 4;
    CHECK(image_range(color, all_subresources, range));
    CHECK(range.aspectMask == VK_IMAGE_ASPECT_COLOR_BIT && range.baseMipLevel == 0 && range.levelCount == 3);
    CHECK(range.baseArrayLayer == 0 && range.layerCount == 4);
    CHECK(image_range(color, 7, range));  // mip 1, layer 2
    CHECK(range.baseMipLevel == 1 && range.levelCount == 1 && range.baseArrayLayer == 2 && range.layerCount == 1);
    CHECK(image_range(color, 11, range));  // last subresource
    CHECK(range.baseMipLevel == 2 && range.baseArrayLayer == 3);
    CHECK(!image_range(color, 12, range));  // no plane 1 on a colour image

    // A volume attachment can select slices inside a single mip. Its tracked layout covers the whole mip.
    ResourceBinding volume = color;
    volume.volume = true;
    volume.array_layers = 1;
    volume.depth = 8;
    CHECK(detail::view_range(volume, {1, 1, 2, 2}, VK_IMAGE_ASPECT_COLOR_BIT, range));
    CHECK(range.baseMipLevel == 1 && range.baseArrayLayer == 0 && range.layerCount == 1);
    CHECK(!detail::view_range(volume, {1, 1, 2, 3}, VK_IMAGE_ASPECT_COLOR_BIT, range));
    CHECK(!detail::view_range(volume, {1, 1, 4, 1}, VK_IMAGE_ASPECT_COLOR_BIT, range));
    CHECK(!detail::view_range(volume, {1, 1, 0, 0}, VK_IMAGE_ASPECT_COLOR_BIT, range));
    CHECK(!detail::view_range(volume, {1, 3, 0, 1}, VK_IMAGE_ASPECT_COLOR_BIT, range));

    // Validate the last byte of the last volume slice before recording an upload.
    detail::Prepared upload{};
    upload.binding.image = color.image;
    upload.binding.format = VK_FORMAT_R8_UNORM;
    upload.binding.width = 4;
    upload.binding.height = 3;
    upload.binding.depth = 2;
    upload.binding.volume = true;
    upload.source.buffer = reinterpret_cast<VkBuffer>(std::uintptr_t{2});
    upload.source.size = 5 * 256 + 4;
    CopyTexture copy{};
    copy.src = {2, true, 0, 0, 4, 3, 2, 256};
    CHECK(detail::prepare_texture_copy(copy, upload) == ReplayResult::success);
    --upload.source.size;
    CHECK(detail::prepare_texture_copy(copy, upload) == ReplayResult::invalid_copy);
    ++upload.source.size;
    copy.z = 1;
    CHECK(detail::prepare_texture_copy(copy, upload) == ReplayResult::invalid_copy);
    CHECK(upload.buffer_image_copies.size() == 1);

    ResourceBinding depth_stencil = color;
    depth_stencil.aspects = VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT;
    depth_stencil.mip_levels = 2;
    depth_stencil.array_layers = 1;
    CHECK(image_range(depth_stencil, 1, range));
    CHECK(range.aspectMask == VK_IMAGE_ASPECT_DEPTH_BIT && range.baseMipLevel == 1);
    CHECK(image_range(depth_stencil, 2, range));
    CHECK(range.aspectMask == VK_IMAGE_ASPECT_STENCIL_BIT && range.baseMipLevel == 0);
    CHECK(image_range(depth_stencil, 3, range));
    CHECK(range.aspectMask == VK_IMAGE_ASPECT_STENCIL_BIT && range.baseMipLevel == 1);
    CHECK(!image_range(depth_stencil, 4, range));
    CHECK(image_range(depth_stencil, all_subresources, range));
    CHECK(range.aspectMask == depth_stencil.aspects && range.levelCount == 2);

    ResourceBinding depth_only = depth_stencil;
    depth_only.aspects = VK_IMAGE_ASPECT_DEPTH_BIT;
    CHECK(image_range(depth_only, 1, range) && range.aspectMask == VK_IMAGE_ASPECT_DEPTH_BIT);
    CHECK(!image_range(depth_only, 2, range));

    ResourceBinding empty = color;
    empty.mip_levels = 0;
    CHECK(!image_range(empty, all_subresources, range));

    // Enhanced barriers.
    namespace y = d3d12_sync;
    namespace a = d3d12_access;
    namespace l = d3d12_layout;
    CHECK(vulkan_sync(y::none) == VK_PIPELINE_STAGE_2_NONE);
    CHECK(vulkan_sync(y::all) == VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT);
    CHECK(vulkan_sync(y::split) == VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT);
    CHECK(vulkan_sync(y::render_target) == VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT);
    CHECK(vulkan_sync(y::copy | y::pixel_shading) ==
          (VK_PIPELINE_STAGE_2_COPY_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT));
    CHECK(vulkan_sync(y::raytracing) == VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT);
    CHECK(vulkan_barrier_access(a::no_access) == VK_ACCESS_2_NONE);
    const auto cb_scope = vulkan_barrier_scope(y::pixel_shading, a::constant_buffer);
    CHECK(cb_scope.access == (VK_ACCESS_2_UNIFORM_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT));
    CHECK(cb_scope.stages == VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT);
    CHECK(vulkan_barrier_access(a::common) == (VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT));
    CHECK(vulkan_barrier_access(a::copy_source | a::shader_resource) ==
          (VK_ACCESS_2_TRANSFER_READ_BIT | VK_ACCESS_2_SHADER_READ_BIT));

    // vulkan_barrier_scope: the stages are widened until every access has one that can perform it; the access is
    // never dropped. (vk_command_replay_test part C checks every pair against the validation layer.)
    {
        const auto scope = [](std::uint32_t sync, std::uint32_t access) { return vulkan_barrier_scope(sync, access); };
        // Valid D3D12, invalid as mapped before: vertex-buffer reads under SYNC_ALL_SHADING.
        BarrierScope widened = scope(y::all_shading, a::vertex_buffer);
        CHECK(widened.access == VK_ACCESS_2_VERTEX_ATTRIBUTE_READ_BIT);
        CHECK(widened.stages == (vulkan_sync(y::all_shading) | VK_PIPELINE_STAGE_2_VERTEX_ATTRIBUTE_INPUT_BIT));
        widened = scope(y::non_pixel_shading, a::index_buffer);
        CHECK(widened.stages & VK_PIPELINE_STAGE_2_INDEX_INPUT_BIT);
        widened = scope(y::clear_unordered_access_view, a::copy_source);  // CLEAR only writes
        CHECK(widened.stages & VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT);
        widened = scope(y::copy, a::unordered_access);
        CHECK(widened.stages == (VK_PIPELINE_STAGE_2_COPY_BIT | VK_PIPELINE_STAGE_2_PRE_RASTERIZATION_SHADERS_BIT |
                                 VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT));
        // Pairs Vulkan already accepts are left exactly as vulkan_sync / vulkan_barrier_access map them.
        const std::pair<std::uint32_t, std::uint32_t> unchanged[] = {
            {y::copy, a::copy_dest},           {y::pixel_shading, a::shader_resource},
            {y::render_target, a::render_target}, {y::depth_stencil, a::depth_stencil_write},
            {y::draw, a::vertex_buffer},       {y::vertex_shading, a::vertex_buffer},
            {y::index_input, a::index_buffer}, {y::execute_indirect, a::indirect_argument},
            {y::all, a::common},               {y::compute_shading, a::unordered_access},
            {y::clear_unordered_access_view, a::unordered_access}};
        for (const auto& [sync, access] : unchanged) {
            const BarrierScope same = scope(sync, access);
            CHECK(same.stages == vulkan_sync(sync) && same.access == vulkan_barrier_access(access));
        }
        // SYNC_PREDICATION / ACCESS_PREDICATION share EXECUTE_INDIRECT's and INDIRECT_ARGUMENT's bits; with the
        // extension they add the conditional-rendering stage and read. An access named under a scope without that
        // stage (SYNC_COPY) widens the scope by it; without the extension nothing changes.
        static_assert(y::predication == y::execute_indirect && a::predication == a::indirect_argument);
        const BarrierScope predicated = vulkan_barrier_scope(y::predication, a::predication, true);
        CHECK(predicated.stages ==
              (VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT | VK_PIPELINE_STAGE_2_CONDITIONAL_RENDERING_BIT_EXT));
        CHECK(predicated.access ==
              (VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT | VK_ACCESS_2_CONDITIONAL_RENDERING_READ_BIT_EXT));
        CHECK(vulkan_barrier_scope(y::copy, a::predication, true).stages &
              VK_PIPELINE_STAGE_2_CONDITIONAL_RENDERING_BIT_EXT);
        const BarrierScope plain = vulkan_barrier_scope(y::predication, a::predication);
        CHECK(plain.stages == vulkan_sync(y::predication) && plain.access == vulkan_barrier_access(a::predication));
        CHECK(vulkan_barrier_scope(y::copy, a::copy_dest, true).access == VK_ACCESS_2_TRANSFER_WRITE_BIT);
        // No stages, no access (SYNC_NONE pairs with ACCESS_NO_ACCESS in D3D12).
        CHECK(scope(y::none, a::no_access).stages == 0 && scope(y::none, a::no_access).access == 0);
        CHECK(scope(y::none, a::shader_resource).access == 0);
        // Every single sync bit against every single access bit: the stages only grow, the access never changes.
        for (std::uint32_t s_bit = 1; s_bit; s_bit <<= 1) {
            for (std::uint32_t a_bit = 1; a_bit; a_bit <<= 1) {
                const BarrierScope any = scope(s_bit, a_bit);
                CHECK((any.stages & vulkan_sync(s_bit)) == vulkan_sync(s_bit));
                CHECK(any.access == (vulkan_sync(s_bit) ? vulkan_barrier_access(a_bit) : 0));
            }
        }
    }
    CHECK(vulkan_layout(l::undefined, false) == VK_IMAGE_LAYOUT_UNDEFINED);
    CHECK(vulkan_layout(l::common, false) == VK_IMAGE_LAYOUT_GENERAL);
    CHECK(vulkan_layout(l::present, true) == VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);
    CHECK(vulkan_layout(l::render_target, false) == VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    CHECK(vulkan_layout(l::direct_queue_shader_resource, false) == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    CHECK(vulkan_layout(l::compute_queue_copy_dest, false) == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    CHECK(vulkan_layout(l::generic_read, false) == VK_IMAGE_LAYOUT_GENERAL);

    ResourceBinding simultaneous = color;
    simultaneous.simultaneous_access = true;
    CHECK(image_layout(simultaneous, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL) == VK_IMAGE_LAYOUT_GENERAL);
    CHECK(image_layout(simultaneous, VK_IMAGE_LAYOUT_UNDEFINED) == VK_IMAGE_LAYOUT_UNDEFINED);
    CHECK(image_layout(color, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL) == VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);

    // D3D12_BARRIER_SUBRESOURCE_RANGE.
    CHECK(barrier_range(color, {7, 0, 0, 0, 0, 0}, range));  // a subresource index
    CHECK(range.baseMipLevel == 1 && range.baseArrayLayer == 2 && range.levelCount == 1);
    CHECK(barrier_range(color, {1, 2, 1, 3, 0, 1}, range));
    CHECK(range.baseMipLevel == 1 && range.levelCount == 2 && range.baseArrayLayer == 1 && range.layerCount == 3);
    CHECK(!barrier_range(color, {2, 2, 0, 1, 0, 1}, range));  // mips 2..3 of 3
    CHECK(!barrier_range(color, {0, 1, 0, 1, 1, 1}, range));  // no plane 1
    CHECK(barrier_range(depth_stencil, {0, 1, 0, 1, 1, 1}, range) && range.aspectMask == VK_IMAGE_ASPECT_STENCIL_BIT);
    CHECK(barrier_range(depth_stencil, {0, 2, 0, 1, 0, 2}, range) && range.aspectMask == depth_stencil.aspects);

    // Layout tracking: per plane with separate depth/stencil layouts, merged without.
    LayoutTracker separate(true), merged(false);
    separate.track(1, depth_stencil, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
    merged.track(1, depth_stencil, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
    const VkImageSubresourceRange stencil_mip1{VK_IMAGE_ASPECT_STENCIL_BIT, 1, 1, 0, 1};
    separate.set(1, stencil_mip1, VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL);
    CHECK(separate.layout(1, 3) == VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL);  // stencil, mip 1
    CHECK(separate.layout(1, 1) == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL);  // depth, mip 1
    CHECK(separate.layout(1, 4) == VK_IMAGE_LAYOUT_UNDEFINED);                         // no plane 2
    std::vector<LayoutPiece> pieces;
    CHECK(separate.pieces(1, {depth_stencil.aspects, 0, 2, 0, 1}, pieces) && pieces.size() == 4);
    CHECK(separate.pieces(1, {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 2, 0, 1}, pieces) && pieces.size() == 1);
    CHECK(pieces[0].layout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL && pieces[0].range.levelCount == 2);
    merged.set(1, {depth_stencil.aspects, 1, 1, 0, 1}, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
    CHECK(merged.layout(1, 1) == VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL && merged.layout(1, 3) == merged.layout(1, 1));
    CHECK(!merged.pieces(2, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}, pieces));  // untracked
    merged.forget(1);
    CHECK(!merged.tracked(1));

    // DiscardResource by subresource range (D3D-26), through prepare(): each listed subresource becomes one range,
    // and the command loop goes on to the next command after a discard's own loop.
    {
        ResourceBinding target{};
        target.image = reinterpret_cast<VkImage>(std::uintptr_t{3});
        target.mip_levels = 3;
        target.array_layers = 2;
        const auto resolve = [&](ResourceId id) -> const ResourceBinding* { return id == 3 ? &target : nullptr; };
        CommandList list;
        CHECK(list.discard({3, 1, 3, false}));  // mips 1 and 2 of layer 0, mip 0 of layer 1
        CHECK(list.discard({3, 5, 1, false}));  // mip 2 of layer 1, the last subresource
        CHECK(list.discard({3, 0, 2, true}));   // a partial discard keeps the contents: nothing to record
        CHECK(list.close());
        PreparedList prepared;
        CHECK(prepare(list, resolve, ReplayFunctions{}, prepared) == ReplayResult::success);
        CHECK(prepared.commands.size() == 3);
        const auto& first = prepared.commands[0].discards;
        CHECK(first.size() == 3);
        CHECK(first[0].baseMipLevel == 1 && first[0].baseArrayLayer == 0 && first[0].levelCount == 1);
        CHECK(first[1].baseMipLevel == 2 && first[1].baseArrayLayer == 0);
        CHECK(first[2].baseMipLevel == 0 && first[2].baseArrayLayer == 1 && first[2].layerCount == 1);
        const auto& second = prepared.commands[1].discards;
        CHECK(second.size() == 1 && second[0].baseMipLevel == 2 && second[0].baseArrayLayer == 1);
        CHECK(prepared.commands[2].discards.empty());

        // One subresource past the end is refused.
        CommandList past;
        CHECK(past.discard({3, 4, 3, false}));
        CHECK(past.close());
        CHECK(prepare(past, resolve, ReplayFunctions{}, prepared) == ReplayResult::invalid_subresource);
    }

    // Indirect argument and count resources can occupy different slices of one Vulkan buffer.
    {
        ResourceBinding arguments{}, counts{};
        arguments.buffer = counts.buffer = reinterpret_cast<VkBuffer>(std::uintptr_t{1});
        arguments.buffer_offset = 256;
        arguments.size = 128;
        counts.buffer_offset = 512;
        counts.size = 16;
        const auto resolve = [&](ResourceId id) -> const ResourceBinding* {
            if (id == 1) return &arguments;
            if (id == 2) return &counts;
            return nullptr;
        };

        static std::vector<VkDeviceSize> offsets;
        ReplayFunctions vk{};
        vk.draw_indirect = [](VkCommandBuffer, VkBuffer, VkDeviceSize offset, std::uint32_t, std::uint32_t) {
            offsets.push_back(offset);
        };
        vk.draw_indexed_indirect = vk.draw_indirect;
        vk.draw_indirect_count = [](VkCommandBuffer, VkBuffer, VkDeviceSize offset, VkBuffer,
                                    VkDeviceSize count_offset, std::uint32_t, std::uint32_t) {
            offsets.push_back(offset);
            offsets.push_back(count_offset);
        };
        vk.draw_indexed_indirect_count = vk.draw_indirect_count;
        vk.dispatch_indirect = [](VkCommandBuffer, VkBuffer, VkDeviceSize offset) {
            offsets.push_back(offset);
        };

        CommandList list;
        CHECK(list.execute_indirect({IndirectKind::draw, 16, 2, 1, 4, 0, 0}));
        CHECK(list.execute_indirect({IndirectKind::draw_indexed, 20, 2, 1, 8, 0, 0}));
        CHECK(list.execute_indirect({IndirectKind::draw, 16, 2, 1, 12, 2, 4}));
        CHECK(list.execute_indirect({IndirectKind::draw_indexed, 20, 2, 1, 16, 2, 8}));
        CHECK(list.execute_indirect({IndirectKind::dispatch, 20, 2, 1, 20, 0, 0}));
        CHECK(list.close());
        CHECK(replay(list, VK_NULL_HANDLE, resolve, vk) == ReplayResult::success);
        const std::vector<VkDeviceSize> expected{260, 264, 268, 516, 272, 520, 276, 296};
        CHECK(offsets == expected);

        // Reject invalid spans before invoking Vulkan, including after adding the backing-buffer base.
        const auto rejected = [&](const ExecuteIndirect& command) {
            CommandList invalid;
            return invalid.execute_indirect(command) && invalid.close() &&
                   replay(invalid, VK_NULL_HANDLE, resolve, vk) == ReplayResult::invalid_indirect_arguments;
        };
        CHECK(rejected({IndirectKind::draw, 12, 1, 1, 0, 0, 0}));
        CHECK(rejected({IndirectKind::draw, 16, 1, 1, 2, 0, 0}));
        CHECK(rejected({IndirectKind::draw, 16, 2, 1, 112, 0, 0}));
        CHECK(rejected({IndirectKind::draw, 16, 1, 1, 4, 2, 16}));
        arguments.buffer_offset = UINT64_MAX - 3;
        CHECK(rejected({IndirectKind::dispatch, 12, 1, 1, 0, 0, 0}));
        CHECK(offsets == expected);
    }

    // Resizable BAR detection (D3D-96) on the heap layouts real cards report.
    {
        constexpr VkDeviceSize mib = VkDeviceSize(1) << 20;
        constexpr VkMemoryPropertyFlags local = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
        constexpr VkMemoryPropertyFlags host = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        constexpr VkMemoryPropertyFlags cached = host | VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
        constexpr VkMemoryHeapFlags vram = VK_MEMORY_HEAP_DEVICE_LOCAL_BIT;
        struct Type {
            VkMemoryPropertyFlags flags;
            std::uint32_t heap;
        };
        const auto memory = [](std::initializer_list<VkMemoryHeap> heaps, std::initializer_list<Type> types) {
            VkPhysicalDeviceMemoryProperties out{};
            for (const VkMemoryHeap& heap : heaps) out.memoryHeaps[out.memoryHeapCount++] = heap;
            for (const Type& type : types) out.memoryTypes[out.memoryTypeCount++] = {type.flags, type.heap};
            return out;
        };
        // RTX 5060 Ti, driver 621.14, Resizable BAR on: the host-visible VRAM type is in the 15.67 GiB VRAM heap.
        CHECK(resizable_bar(memory({{16829644800ull, vram}, {8551161856ull, 0}},
                                   {{0, 1}, {local, 0}, {host, 1}, {cached, 1}, {local | host, 0}})));
        // RTX 3090, driver 616.56, Resizable BAR off: that type is in a heap of its own, the 214 MiB BAR window.
        CHECK(!resizable_bar(memory({{24ull << 30, vram}, {32ull << 30, 0}, {224395264ull, vram}},
                                    {{local, 0}, {host, 1}, {cached, 1}, {local | host, 2}})));
        // A 256 MiB window (AMD without Smart Access Memory, RADV without Resizable BAR) is not enough; one byte more is.
        VkPhysicalDeviceMemoryProperties window =
            memory({{8ull << 30, vram}, {16ull << 30, 0}, {256 * mib, vram}}, {{local, 0}, {host, 1}, {local | host, 2}});
        CHECK(!resizable_bar(window));
        window.memoryHeaps[2].size = 256 * mib + 1;
        CHECK(resizable_bar(window));
        // A GPU whose memory is all device-local and host-visible (an integrated GPU).
        CHECK(resizable_bar(memory({{8ull << 30, vram}}, {{local, 0}, {local | host, 0}})));
        // Not without HOST_COHERENT, not in a heap without DEVICE_LOCAL, not with a heap index past the heaps.
        CHECK(!resizable_bar(memory({{8ull << 30, vram}}, {{local | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT, 0}})));
        CHECK(!resizable_bar(memory({{8ull << 30, 0}}, {{local | host, 0}})));
        CHECK(!resizable_bar(memory({{8ull << 30, vram}}, {{local | host, 1}})));
        CHECK(!resizable_bar(VkPhysicalDeviceMemoryProperties{}));
    }

    // Memory type choice. Heaps as an RTX 3090 without Resizable BAR reports them: VRAM, system memory, and the
    // 214 MiB BAR heap that holds the only DEVICE_LOCAL | HOST_VISIBLE type.
    {
        constexpr VkMemoryPropertyFlags local = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
        constexpr VkMemoryPropertyFlags host = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        VkPhysicalDeviceMemoryProperties memory{};
        memory.memoryHeapCount = 3;
        memory.memoryHeaps[0] = {24ull << 30, VK_MEMORY_HEAP_DEVICE_LOCAL_BIT};
        memory.memoryHeaps[1] = {32ull << 30, 0};
        memory.memoryHeaps[2] = {224395264, VK_MEMORY_HEAP_DEVICE_LOCAL_BIT};
        memory.memoryTypeCount = 4;
        memory.memoryTypes[0] = {local, 0};
        memory.memoryTypes[1] = {host, 1};
        memory.memoryTypes[2] = {host | VK_MEMORY_PROPERTY_HOST_CACHED_BIT, 1};
        memory.memoryTypes[3] = {local | host, 2};
        using order = std::vector<std::uint32_t>;
        constexpr VkDeviceSize mib = 1 << 20;
        CHECK(memory_type_order(memory, 0xF, host, local, 256 * mib) == (order{1, 2}));     // the 256 MiB arena
        CHECK(memory_type_order(memory, 0xF, host, local, 128 * mib) == (order{3, 1, 2}));  // fits the BAR heap
        CHECK(memory_type_order(memory, 0xF, host, local, 224395264) == (order{3, 1, 2}));  // exactly fits
        CHECK(memory_type_order(memory, 0x7, host, local, 128 * mib) == (order{1, 2}));     // type 3 not allowed
        CHECK(memory_type_order(memory, 0xF, 0, local, 1 * mib) == (order{0, 3, 1, 2}));
        CHECK(memory_type_order(memory, 0xF, host, local, 64ull << 30).empty());           // nothing that large
        memory.memoryHeaps[2].size = 24ull << 30;                                         // Resizable BAR on
        CHECK(memory_type_order(memory, 0xF, host, local, 256 * mib) == (order{3, 1, 2}));
    }

    std::printf("state map, enhanced barriers, subresource decoding, layout tracking, discard ranges, BAR detection "
                "and memory types OK\n");
    return 0;
}
