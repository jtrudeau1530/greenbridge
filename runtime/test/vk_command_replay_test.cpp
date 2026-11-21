// Replays recorded command lists on offscreen images and reads them back. No shaders or game data.
// Part A uses only Vulkan 1.0 functions and no layout tracker. Part B uses the tracker with separate depth/stencil
// layouts: rectangle clears, a depth clear while the stencil plane is in another state, a texture used after
// implicit promotion from common, and enhanced barriers.
// Set LEONIDA_VK_VALIDATION=1 to run under VK_LAYER_KHRONOS_validation; any validation error fails the test.
#include "vk_test_device.h"
#include "../src/vulkan_replay.h"

#include <array>
#include <cstdint>
#include <cstring>
#include <map>
#include <tuple>

#define EXPECT(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); return 1; } \
} while (0)

namespace {

using namespace leonida;
namespace s = d3d12_state;

using Rgba = std::array<std::uint8_t, 4>;
constexpr Rgba red{255, 0, 0, 255}, green{0, 255, 0, 255}, blue{0, 0, 255, 255};

unsigned g_mismatches = 0;

void expect_rgba(const char* what, const std::uint8_t* data, std::uint32_t index, const Rgba& expected) {
    if (std::equal(expected.begin(), expected.end(), data + index * 4)) return;
    if (g_mismatches++ < 8)
        std::fprintf(stderr, "%s texel %u: got %u,%u,%u,%u\n", what, index, unsigned(data[index * 4]),
                     unsigned(data[index * 4 + 1]), unsigned(data[index * 4 + 2]), unsigned(data[index * 4 + 3]));
}

void expect_depth(const char* what, const std::uint8_t* data, std::uint32_t index, float expected) {
    float depth;
    std::memcpy(&depth, data + index * 4, 4);
    if (depth != expected && g_mismatches++ < 8)
        std::fprintf(stderr, "%s texel %u: got %f, want %f\n", what, index, depth, expected);
}

void expect_byte(const char* what, const std::uint8_t* data, std::uint32_t index, std::uint8_t expected) {
    if (data[index] != expected && g_mismatches++ < 8)
        std::fprintf(stderr, "%s texel %u: got 0x%02x, want 0x%02x\n", what, index, unsigned(data[index]),
                     unsigned(expected));
}

struct Views {
    vktest::Device& device;
    std::map<std::tuple<VkImage, VkImageAspectFlags, std::uint32_t, std::uint32_t, std::uint32_t>, VkImageView> views;
    ~Views() {
        const auto destroy = device.fn<PFN_vkDestroyImageView>("vkDestroyImageView");
        for (const auto& [key, view] : views) destroy(device.device, view, nullptr);
    }
    VkImageView operator()(const ResourceBinding& r, const VkImageSubresourceRange& range) {
        auto& view = views[{r.image, range.aspectMask, range.baseMipLevel, range.baseArrayLayer, range.layerCount}];
        if (view) return view;
        VkImageViewCreateInfo info{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        info.image = r.image;
        info.viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
        info.format = r.format;
        info.subresourceRange = range;
        device.fn<PFN_vkCreateImageView>("vkCreateImageView")(device.device, &info, nullptr, &view);
        return view;
    }
};

int part_a(vktest::Device& d) {
    const bool has_stencil = d.depth_format_usable(VK_FORMAT_D32_SFLOAT_S8_UINT);
    const VkFormat depth_format = has_stencil ? VK_FORMAT_D32_SFLOAT_S8_UINT : VK_FORMAT_D32_SFLOAT;
    const VkImageAspectFlags depth_aspects = VK_IMAGE_ASPECT_DEPTH_BIT | (has_stencil ? VK_IMAGE_ASPECT_STENCIL_BIT : 0);
    constexpr std::uint32_t width = 4, height = 4, layers = 2, texels = width * height;
    const VkImage color = d.image(VK_FORMAT_R8G8B8A8_UNORM, width, height, layers, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT);
    const VkImage depth = d.image(depth_format, width, height, 1, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT);
    constexpr VkDeviceSize depth_offset = texels * layers * 4, stencil_offset = depth_offset + texels * 4;
    void* mapped = nullptr;
    const VkBuffer buffer = d.buffer(stencil_offset + texels, VK_BUFFER_USAGE_TRANSFER_DST_BIT, &mapped);
    EXPECT(color && depth && buffer);

    ResourceBinding color_binding{};
    color_binding.image = color;
    color_binding.array_layers = layers;
    ResourceBinding depth_binding{};
    depth_binding.image = depth;
    depth_binding.aspects = depth_aspects;
    constexpr ResourceId color_id = 1, depth_id = 2, missing_id = 3;
    const auto resolve = [&](ResourceId id) -> const ResourceBinding* {
        if (id == color_id) return &color_binding;
        if (id == depth_id) return &depth_binding;
        return nullptr;
    };
    ReplayFunctions vk{};
    vk.pipeline_barrier = d.fn<PFN_vkCmdPipelineBarrier>("vkCmdPipelineBarrier");
    vk.clear_color_image = d.fn<PFN_vkCmdClearColorImage>("vkCmdClearColorImage");
    vk.clear_depth_stencil_image = d.fn<PFN_vkCmdClearDepthStencilImage>("vkCmdClearDepthStencilImage");

    CommandList list;
    TargetView descriptor{color_id, 0, 0, 1};
    EXPECT(list.clear_render_target(descriptor, {0, 0, 1, 1}));
    descriptor = {color_id, 0, 1, 1};  // the CPU descriptor slot is reused; the first clear keeps its own view
    EXPECT(list.clear_render_target(descriptor, {1, 0, 0, 1}));
    EXPECT(list.clear_render_target(descriptor, {0, 1, 0, 1}));
    const TargetView depth_view{depth_id, 0, 0, 1};
    EXPECT(list.clear_depth_stencil(depth_view, clear_depth | (has_stencil ? clear_stencil : 0), 0.25f, 0x5a));
    EXPECT(list.clear_depth_stencil(depth_view, clear_depth, 0.75f, 0));  // stencil keeps 0x5a
    EXPECT(list.uav_barrier({0}));
    EXPECT(list.aliasing_barrier({0, 0}));
    EXPECT(list.transition({color_id, all_subresources, s::render_target, s::copy_source}, barrier_begin_only));
    EXPECT(list.transition({color_id, all_subresources, s::render_target, s::copy_source}, barrier_end_only));
    EXPECT(list.transition({depth_id, all_subresources, s::depth_write, s::copy_source}));
    EXPECT(list.close());

    bool ok = d.run([&](VkCommandBuffer cb) {
        d.initial_layout(cb, color, VK_IMAGE_ASPECT_COLOR_BIT, layers, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
        d.initial_layout(cb, depth, depth_aspects, 1, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL);

        // Rejected lists write nothing to the command buffer.
        CommandList bad;
        if (!bad.clear_render_target({color_id, 0, 0, 1}, {1, 1, 1, 1})) return false;
        if (replay(bad, cb, resolve, vk) != ReplayResult::list_open) return false;
        const auto rejected = [&](auto&& record, ReplayResult expected) {
            bad.close();
            bad.reset();
            record();
            bad.close();
            const ReplayResult got = replay(bad, cb, resolve, vk);
            if (got != expected) std::fprintf(stderr, "rejected list: got %d, want %d\n", int(got), int(expected));
            return got == expected;
        };
        const Rect rect{0, 0, 1, 1};
        if (!rejected([&] { bad.transition({missing_id, 0, s::common, s::copy_dest}); }, ReplayResult::unresolved_resource) ||
            !rejected([&] { bad.transition({color_id, layers, s::render_target, s::copy_source}); },
                      ReplayResult::invalid_subresource) ||
            !rejected([&] { bad.clear_render_target({color_id, 0, 1, 2}, {1, 1, 1, 1}); },
                      ReplayResult::invalid_subresource) ||
            !rejected([&] { bad.clear_render_target({depth_id, 0, 0, 1}, {1, 1, 1, 1}); }, ReplayResult::invalid_clear) ||
            !rejected([&] { bad.clear_depth_stencil({color_id, 0, 0, 1}, clear_depth, 1.0f, 0); },
                      ReplayResult::invalid_clear) ||
            // Rectangle clears and enhanced barriers need the Vulkan 1.3 functions this part does not pass.
            !rejected([&] { bad.clear_render_target({color_id, 0, 0, 1}, {1, 1, 1, 1}, {&rect, 1}); },
                      ReplayResult::unsupported) ||
            !rejected([&] { bad.barrier(GlobalBarrier{d3d12_sync::all, d3d12_sync::all, 0, 0}); },
                      ReplayResult::unsupported))
            return false;
        LayoutTracker separate(true);
        if (replay(list, cb, resolve, vk, &separate) != ReplayResult::unsupported) return false;  // flag mismatch

        if (replay(list, cb, resolve, vk) != ReplayResult::success) return false;
        d.read_back(cb, color, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_ASPECT_COLOR_BIT, width, height, layers,
                    buffer, 0);
        d.read_back(cb, depth, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_ASPECT_DEPTH_BIT, width, height, 1,
                    buffer, depth_offset);
        if (has_stencil)
            d.read_back(cb, depth, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_ASPECT_STENCIL_BIT, width, height, 1,
                        buffer, stencil_offset);
        return true;
    });
    EXPECT(ok);
    const auto* data = static_cast<const std::uint8_t*>(mapped);
    for (std::uint32_t i = 0; i < texels * layers; ++i) expect_rgba("A colour", data, i, i < texels ? blue : green);
    for (std::uint32_t i = 0; i < texels; ++i) {
        expect_depth("A depth", data + depth_offset, i, 0.75f);
        if (has_stencil) expect_byte("A stencil", data + stencil_offset, i, 0x5a);
    }
    std::printf("A: %u colour texels across two layers, %u depth%s texels\n", texels * layers, texels,
                has_stencil ? "/stencil" : "");
    return 0;
}

int part_b(vktest::Device& d) {
    if (!d.depth_format_usable(VK_FORMAT_D32_SFLOAT_S8_UINT)) {
        std::printf("B: skipped, no D32_SFLOAT_S8_UINT\n");
        return 0;
    }
    constexpr std::uint32_t size = 8, texels = size * size;
    const VkImage rt = d.image(VK_FORMAT_R8G8B8A8_UNORM, size, size, 1, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT);
    const VkImage ds = d.image(VK_FORMAT_D32_SFLOAT_S8_UINT, size, size, 1, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT);
    const VkImage promoted = d.image(VK_FORMAT_R8G8B8A8_UNORM, size, size, 1, VK_IMAGE_USAGE_SAMPLED_BIT);
    const VkImage discarded = d.image(VK_FORMAT_R8G8B8A8_UNORM, size, size, 1, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT);
    constexpr VkDeviceSize depth_offset = texels * 4, stencil_offset = depth_offset + texels * 4,
                           discard_offset = stencil_offset + texels;
    void* mapped = nullptr;
    const VkBuffer buffer = d.buffer(discard_offset + texels * 4, VK_BUFFER_USAGE_TRANSFER_DST_BIT, &mapped);
    EXPECT(rt && ds && promoted && discarded && buffer);

    const auto binding = [&](VkImage image, VkFormat format, VkImageAspectFlags aspects) {
        ResourceBinding b{};
        b.image = image;
        b.format = format;
        b.aspects = aspects;
        b.width = b.height = size;
        return b;
    };
    constexpr ResourceId rt_id = 10, ds_id = 11, promoted_id = 12, buffer_id = 13, discarded_id = 14;
    const std::map<ResourceId, ResourceBinding> bindings{
        {rt_id, binding(rt, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_ASPECT_COLOR_BIT)},
        {ds_id, binding(ds, VK_FORMAT_D32_SFLOAT_S8_UINT, depth_stencil_aspects)},
        {promoted_id, binding(promoted, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_ASPECT_COLOR_BIT)},
        {buffer_id, ResourceBinding{}},  // a buffer: no image
        {discarded_id, binding(discarded, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_ASPECT_COLOR_BIT)},
    };
    const auto resolve = [&](ResourceId id) -> const ResourceBinding* {
        const auto it = bindings.find(id);
        return it == bindings.end() ? nullptr : &it->second;
    };

    Views views{d, {}};
    ReplayFunctions vk{};
    vk.pipeline_barrier = d.fn<PFN_vkCmdPipelineBarrier>("vkCmdPipelineBarrier");
    vk.clear_color_image = d.fn<PFN_vkCmdClearColorImage>("vkCmdClearColorImage");
    vk.clear_depth_stencil_image = d.fn<PFN_vkCmdClearDepthStencilImage>("vkCmdClearDepthStencilImage");
    vk.pipeline_barrier2 = d.fn<PFN_vkCmdPipelineBarrier2>("vkCmdPipelineBarrier2");
    vk.begin_rendering = d.fn<PFN_vkCmdBeginRendering>("vkCmdBeginRendering");
    vk.end_rendering = d.fn<PFN_vkCmdEndRendering>("vkCmdEndRendering");
    vk.clear_attachments = d.fn<PFN_vkCmdClearAttachments>("vkCmdClearAttachments");
    vk.image_view = [&](const ResourceBinding& r, const VkImageSubresourceRange& range) { return views(r, range); };
    vk.separate_depth_stencil_layouts = true;

    LayoutTracker tracker(true);
    tracker.track(rt_id, bindings.at(rt_id), VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);         // render_target
    tracker.track(ds_id, bindings.at(ds_id), VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL);  // depth_write
    tracker.track(promoted_id, bindings.at(promoted_id), VK_IMAGE_LAYOUT_GENERAL);               // common
    tracker.track(discarded_id, bindings.at(discarded_id), VK_IMAGE_LAYOUT_GENERAL);             // common

    CommandList list;
    // Rectangle clears: the inner square, a rectangle clipped at the edge, and an empty one that clears nothing.
    const Rect inner{2, 2, 6, 6}, clipped{6, 6, 20, 20}, empty{3, 3, 3, 9};
    const Rect rects[] = {inner, empty};
    EXPECT(list.clear_render_target({rt_id, 0, 0, 1}, {1, 0, 0, 1}));
    EXPECT(list.clear_render_target({rt_id, 0, 0, 1}, {0, 0, 1, 1}, rects));
    EXPECT(list.clear_render_target({rt_id, 0, 0, 1}, {0, 1, 0, 1}, {&clipped, 1}));
    EXPECT(list.clear_render_target({rt_id, 0, 0, 1}, {1, 1, 1, 1}, {&empty, 1}));

    // Depth and stencil: clear both, move only the stencil plane (subresource 1) to depth_read, then clear depth
    // twice (full view through a transfer, left half through rendering) while the stencil plane stays read-only.
    const Rect left_half{0, 0, 4, 8};
    EXPECT(list.clear_depth_stencil({ds_id, 0, 0, 1}, clear_depth | clear_stencil, 1.0f, 0x11));
    EXPECT(list.transition({ds_id, 1, s::depth_write, s::depth_read | s::pixel_shader_resource}));
    EXPECT(list.clear_depth_stencil({ds_id, 0, 0, 1}, clear_depth, 0.25f, 0));
    EXPECT(list.clear_depth_stencil({ds_id, 0, 0, 1}, clear_depth, 0.5f, 0, {&left_half, 1}));
    EXPECT(list.transition({ds_id, 0, s::depth_write, s::copy_source}));
    EXPECT(list.transition({ds_id, 1, s::depth_read | s::pixel_shader_resource, s::copy_source}));

    // Implicit promotion: the texture was created in common and a copy promoted it to copy_dest, so the application
    // names copy_dest as `before`. The image is still in GENERAL; the tracker knows it.
    EXPECT(list.transition({promoted_id, all_subresources, s::copy_dest, s::copy_source}));

    // Enhanced barriers.
    namespace y = d3d12_sync;
    namespace a = d3d12_access;
    namespace l = d3d12_layout;
    const BarrierSubresourceRange all{all_subresources, 0, 0, 0, 0, 0};
    const BarrierSubresourceRange first{0, 1, 0, 1, 0, 1};
    EXPECT(list.barrier(TextureBarrier{y::render_target, y::split, a::render_target, a::copy_source, l::render_target,
                                       l::copy_source, rt_id, all, 0}));  // split begin: dropped
    EXPECT(list.barrier(TextureBarrier{y::split, y::copy, a::render_target, a::copy_source, l::render_target,
                                       l::copy_source, rt_id, all, 0}));
    EXPECT(list.barrier(TextureBarrier{y::none, y::render_target, a::no_access, a::render_target, l::undefined,
                                       l::render_target, discarded_id, all, texture_barrier_discard}));
    EXPECT(list.clear_render_target({discarded_id, 0, 0, 1}, {0, 1, 0, 1}));
    EXPECT(list.barrier(TextureBarrier{y::render_target, y::copy, a::render_target, a::copy_source, l::render_target,
                                       l::copy_source, discarded_id, first, 0}));
    EXPECT(list.barrier(BufferBarrier{y::copy, y::pixel_shading, a::copy_dest, a::shader_resource, buffer_id}));
    EXPECT(list.barrier(GlobalBarrier{y::all, y::all, a::common, a::common}));
    EXPECT(list.close());

    bool ok = d.run([&](VkCommandBuffer cb) {
        d.initial_layout(cb, rt, VK_IMAGE_ASPECT_COLOR_BIT, 1, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
        d.initial_layout(cb, ds, depth_stencil_aspects, 1, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
        d.initial_layout(cb, promoted, VK_IMAGE_ASPECT_COLOR_BIT, 1, VK_IMAGE_LAYOUT_GENERAL);
        d.initial_layout(cb, discarded, VK_IMAGE_ASPECT_COLOR_BIT, 1, VK_IMAGE_LAYOUT_GENERAL);
        if (replay(list, cb, resolve, vk, &tracker) != ReplayResult::success) return false;
        d.read_back(cb, rt, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_ASPECT_COLOR_BIT, size, size, 1, buffer, 0);
        d.read_back(cb, ds, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_ASPECT_DEPTH_BIT, size, size, 1, buffer,
                    depth_offset);
        d.read_back(cb, ds, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_ASPECT_STENCIL_BIT, size, size, 1, buffer,
                    stencil_offset);
        d.read_back(cb, discarded, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_ASPECT_COLOR_BIT, size, size, 1,
                    buffer, discard_offset);
        return true;
    });
    EXPECT(ok);

    EXPECT(tracker.layout(rt_id, 0) == VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
    EXPECT(tracker.layout(ds_id, 0) == VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
    EXPECT(tracker.layout(ds_id, 1) == VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
    EXPECT(tracker.layout(promoted_id, 0) == VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
    EXPECT(tracker.layout(discarded_id, 0) == VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);

    const auto* data = static_cast<const std::uint8_t*>(mapped);
    for (std::uint32_t row = 0; row < size; ++row) {
        for (std::uint32_t col = 0; col < size; ++col) {
            const std::uint32_t i = row * size + col;
            const bool in_inner = col >= 2 && col < 6 && row >= 2 && row < 6;
            const bool in_clipped = col >= 6 && row >= 6;
            expect_rgba("B colour", data, i, in_clipped ? green : in_inner ? blue : red);
            expect_depth("B depth", data + depth_offset, i, col < 4 ? 0.5f : 0.25f);
            expect_byte("B stencil", data + stencil_offset, i, 0x11);
            expect_rgba("B discard", data + discard_offset, i, green);
        }
    }
    std::printf("B: rectangle clears, per-plane depth clears, promotion and enhanced barriers on %u texels\n", texels);
    return 0;
}

} // namespace

int main() {
    vktest::Device device;
    EXPECT(device.create());
    if (part_a(device) || part_b(device)) return 1;
    if (g_mismatches) return 1;
    if (device.validation_errors()) {
        std::fprintf(stderr, "%u validation errors\n", device.validation_errors());
        return 1;
    }
    return 0;
}
