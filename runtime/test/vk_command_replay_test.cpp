// Replays recorded command lists on offscreen images and reads them back. No shaders or game data.
// Part A uses only Vulkan 1.0 functions and no layout tracker. Part B uses the tracker with separate depth/stencil
// layouts: rectangle clears, a depth clear while the stencil plane is in another state, a texture used after
// implicit promotion from common, and enhanced barriers. Part C copies a volume (footprint, box and whole) and
// resolves a multisampled target. Part D records every D3D12 sync/access pair as enhanced barriers, which the
// validation layer must accept. Part E (D3D-26) runs immediate buffer writes, atomic copies and discards.
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

// Part C: volumes and resolves. A 4x4x4 volume is uploaded from a footprint whose slices lie `height` rows apart,
// copied in part (a 2x2x2 block) and whole, and read back; a 4x multisampled target is cleared and resolved.
// Nothing is transitioned before its first use: the images start in UNDEFINED and are zero-filled then.
int part_c(vktest::Device& d) {
    constexpr std::uint32_t size = 4, wide = 8, pitch = 256, slice = size * pitch;
    constexpr VkDeviceSize bytes = 16384, up_at = 512, back_at = 0, part_at = 4096, resolved_at = 8192;
    constexpr VkFormat format = VK_FORMAT_R8G8B8A8_UNORM;
    void *up_mapped = nullptr, *down_mapped = nullptr;
    const VkBuffer up = d.buffer(bytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, &up_mapped);
    const VkBuffer down = d.buffer(bytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT, &down_mapped);
    const VkImage volume = d.image(format, size, size, 1, VK_IMAGE_USAGE_SAMPLED_BIT, size);
    const VkImage part = d.image(format, size, size, 1, VK_IMAGE_USAGE_SAMPLED_BIT, size);
    const VkImage whole = d.image(format, size, size, 1, VK_IMAGE_USAGE_SAMPLED_BIT, size);
    const VkImage msaa = d.image(format, wide, wide, 1, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, 1, 4);
    const VkImage resolved = d.image(format, wide, wide, 1, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT);
    EXPECT(up && down && volume && part && whole && msaa && resolved);
    auto* u = static_cast<std::uint8_t*>(up_mapped);
    const auto* r = static_cast<const std::uint8_t*>(down_mapped);
    const auto texel = [&](std::uint32_t x, std::uint32_t y, std::uint32_t z) {
        return u + up_at + z * slice + y * pitch + x * 4;
    };
    for (std::uint32_t z = 0; z < size; ++z)
        for (std::uint32_t y = 0; y < size; ++y)
            for (std::uint32_t x = 0; x < size; ++x) {
                const Rgba value{std::uint8_t(x * 16 + y), std::uint8_t(z * 40 + 1), std::uint8_t(x ^ y ^ (z << 4)), 255};
                std::copy(value.begin(), value.end(), texel(x, y, z));
            }

    const auto image_binding = [&](VkImage image, std::uint32_t extent, std::uint32_t depth, std::uint32_t samples) {
        ResourceBinding b{};
        b.image = image;
        b.format = format;
        b.width = b.height = extent;
        b.depth = depth;
        b.volume = depth > 1;
        b.samples = samples;
        return b;
    };
    const auto buffer_binding = [&](VkBuffer buffer) {
        ResourceBinding b{};
        b.buffer = buffer;
        b.size = bytes;
        return b;
    };
    constexpr ResourceId up_id = 20, down_id = 21, volume_id = 22, part_id = 23, whole_id = 24, msaa_id = 25,
                         resolved_id = 26;
    const std::map<ResourceId, ResourceBinding> bindings{
        {up_id, buffer_binding(up)},
        {down_id, buffer_binding(down)},
        {volume_id, image_binding(volume, size, size, 1)},
        {part_id, image_binding(part, size, size, 1)},
        {whole_id, image_binding(whole, size, size, 1)},
        {msaa_id, image_binding(msaa, wide, 1, 4)},
        {resolved_id, image_binding(resolved, wide, 1, 1)},
    };
    const auto resolve = [&](ResourceId id) -> const ResourceBinding* {
        const auto it = bindings.find(id);
        return it == bindings.end() ? nullptr : &it->second;
    };
    LayoutTracker tracker(true);
    for (const ResourceId id : {volume_id, part_id, whole_id, msaa_id, resolved_id})
        tracker.track(id, bindings.at(id), VK_IMAGE_LAYOUT_UNDEFINED);

    ReplayFunctions vk{};
    vk.pipeline_barrier = d.fn<PFN_vkCmdPipelineBarrier>("vkCmdPipelineBarrier");
    vk.clear_color_image = d.fn<PFN_vkCmdClearColorImage>("vkCmdClearColorImage");
    vk.clear_depth_stencil_image = d.fn<PFN_vkCmdClearDepthStencilImage>("vkCmdClearDepthStencilImage");
    vk.copy_buffer = d.fn<PFN_vkCmdCopyBuffer>("vkCmdCopyBuffer");
    vk.copy_buffer_to_image = d.fn<PFN_vkCmdCopyBufferToImage>("vkCmdCopyBufferToImage");
    vk.copy_image_to_buffer = d.fn<PFN_vkCmdCopyImageToBuffer>("vkCmdCopyImageToBuffer");
    vk.copy_image = d.fn<PFN_vkCmdCopyImage>("vkCmdCopyImage");
    vk.separate_depth_stencil_layouts = true;
    const ReplayFunctions without_resolve = vk;
    vk.resolve_image = d.fn<PFN_vkCmdResolveImage>("vkCmdResolveImage");

    const auto sub = [](ResourceId id) { return CopyLocation{id, false, 0, 0, 0, 0, 0, 0}; };
    const auto slices = [&](ResourceId id, VkDeviceSize offset, std::uint32_t depth) {
        return CopyLocation{id, true, 0, offset, size, size, depth, pitch};
    };
    CommandList list;
    EXPECT(list.copy_texture({sub(volume_id), 0, 0, 0, slices(up_id, up_at, size), false, {}}));  // four slices up
    EXPECT(list.transition({volume_id, 0, s::copy_dest, s::copy_source}));
    EXPECT(list.copy_texture({sub(part_id), 2, 0, 1, sub(volume_id), true, {1, 1, 1, 3, 3, 3}}));  // 2x2x2 to (2, 0, 1)
    EXPECT(list.copy_resource({whole_id, volume_id}));
    EXPECT(list.transition({part_id, 0, s::copy_dest, s::copy_source}));
    EXPECT(list.transition({whole_id, 0, s::copy_dest, s::copy_source}));
    // Slices 1 and 2 of the whole copy, to slice 1 of a footprint; then all of the partly written volume.
    EXPECT(list.copy_texture({slices(down_id, back_at, size), 0, 0, 1, sub(whole_id), true, {0, 0, 1, size, size, 3}}));
    EXPECT(list.copy_texture({slices(down_id, part_at, size), 0, 0, 0, sub(part_id), false, {}}));
    EXPECT(list.clear_render_target({msaa_id, 0, 0, 1}, {0, 1, 0, 1}));
    EXPECT(list.transition({msaa_id, 0, s::render_target, s::resolve_source}));
    EXPECT(list.resolve_subresource({resolved_id, 0, msaa_id, 0}));
    EXPECT(list.transition({resolved_id, 0, s::resolve_dest, s::copy_source}));
    EXPECT(list.copy_texture({{down_id, true, 0, resolved_at, wide, wide, 1, pitch}, 0, 0, 0, sub(resolved_id), false, {}}));
    EXPECT(list.close());

    bool ok = d.run([&](VkCommandBuffer cb) {
        // Rejected lists write nothing and leave the tracker alone.
        CommandList bad;
        const auto rejected = [&](auto&& record, ReplayResult expected, const ReplayFunctions& functions) {
            if (!bad.closed()) bad.close();
            bad.reset();
            if (!record()) return false;
            bad.close();
            const ReplayResult got = replay(bad, cb, resolve, functions, &tracker);
            if (got != expected) std::fprintf(stderr, "rejected list: got %d, want %d\n", int(got), int(expected));
            return got == expected;
        };
        const auto copy = [&](const CopyTexture& c) { return [&bad, c] { return bad.copy_texture(c); }; };
        const auto resolving = [&](const ResolveSubresource& c) { return [&bad, c] { return bad.resolve_subresource(c); }; };
        constexpr auto invalid = ReplayResult::invalid_copy;
        if (!rejected(resolving({msaa_id, 0, resolved_id, 0}), invalid, vk) ||          // single-sampled source
            !rejected(resolving({volume_id, 0, msaa_id, 0}), invalid, vk) ||            // another size
            !rejected(resolving({resolved_id, 1, msaa_id, 0}), ReplayResult::invalid_subresource, vk) ||
            !rejected(resolving({resolved_id, 0, msaa_id, 0}), ReplayResult::unsupported, without_resolve) ||
            // Slices 3 and 4 of 4; a box deeper than its footprint; a footprint past the end of the buffer.
            !rejected(copy({sub(part_id), 0, 0, 3, sub(volume_id), true, {0, 0, 0, size, size, 2}}), invalid, vk) ||
            !rejected(copy({sub(volume_id), 0, 0, 0, slices(up_id, up_at, 2), true, {0, 0, 0, size, size, 3}}), invalid, vk) ||
            !rejected(copy({sub(volume_id), 0, 0, 0, slices(up_id, 13312, size), false, {}}), invalid, vk) ||
            // A multisampled texture has no footprint and copies only to its own sample count.
            !rejected(copy({{down_id, true, 0, 0, wide, wide, 1, pitch}, 0, 0, 0, sub(msaa_id), false, {}}), invalid, vk) ||
            !rejected(copy({sub(resolved_id), 0, 0, 0, sub(msaa_id), false, {}}), invalid, vk))
            return false;
        if (tracker.layout(volume_id, 0) != VK_IMAGE_LAYOUT_UNDEFINED) return false;
        return replay(list, cb, resolve, vk, &tracker) == ReplayResult::success;
    });
    EXPECT(ok);
    for (const ResourceId id : {volume_id, part_id, whole_id, msaa_id, resolved_id})
        EXPECT(tracker.layout(id, 0) == VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);

    const auto expect_texel = [&](const char* what, const std::uint8_t* got, const std::uint8_t* want, std::uint32_t index) {
        Rgba expected{};
        std::copy(want, want + 4, expected.begin());
        expect_rgba(what, got, index, expected);
    };
    const std::uint8_t zero[4] = {};
    for (std::uint32_t z = 0; z < size; ++z) {
        for (std::uint32_t y = 0; y < size; ++y) {
            for (std::uint32_t x = 0; x < size; ++x) {
                const std::uint32_t row = z * slice + y * pitch;
                if (z == 1 || z == 2) expect_texel("C slices", r + back_at + row, texel(x, y, z), x);
                // The block came from (1..2, 1..2, 1..2); the rest of the volume was zero-filled on first use.
                const bool inside = x >= 2 && y < 2 && z >= 1 && z < 3;
                expect_texel("C block", r + part_at + row, inside ? texel(x - 1, y + 1, z) : zero, x);
            }
        }
    }
    EXPECT(r[back_at] == 0xcd && r[back_at + 3 * slice] == 0xcd && r[back_at + slice + 16] == 0xcd);  // not written
    for (std::uint32_t row = 0; row < wide; ++row)
        for (std::uint32_t col = 0; col < wide; ++col) expect_rgba("C resolve", r + resolved_at + row * pitch, col, green);
    std::printf("C: a %ux%ux%u volume uploaded, copied in part and whole and read back; a 4x resolve of %u texels\n",
                size, size, size, wide * wide);
    return 0;
}

// Part D: every D3D12 sync bit against every access bit, as global, buffer and texture barriers. Whatever pair the
// application names, the Vulkan barrier must be valid, so every access has a stage in its scope that can perform it
// (vulkan_barrier_scope). Under LEONIDA_VK_VALIDATION=1 a pair that breaks that rule is a validation error. The
// image is simultaneous-access, so it stays in GENERAL, which every access may use: only the stage/access rule is
// under test.
int part_d(vktest::Device& d) {
    namespace y = d3d12_sync;
    namespace a = d3d12_access;
    namespace l = d3d12_layout;
    const VkImage image = d.image(VK_FORMAT_R8G8B8A8_UNORM, 4, 4, 1,
                                  VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT |
                                      VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT);
    EXPECT(image);
    constexpr ResourceId image_id = 30, buffer_id = 31;
    ResourceBinding image_binding{};
    image_binding.image = image;
    image_binding.format = VK_FORMAT_R8G8B8A8_UNORM;
    image_binding.width = image_binding.height = 4;
    image_binding.simultaneous_access = true;
    const ResourceBinding buffer_binding{};  // a buffer: no image
    const auto resolve = [&](ResourceId id) -> const ResourceBinding* {
        if (id == image_id) return &image_binding;
        return id == buffer_id ? &buffer_binding : nullptr;
    };
    ReplayFunctions vk{};
    vk.pipeline_barrier = d.fn<PFN_vkCmdPipelineBarrier>("vkCmdPipelineBarrier");
    vk.clear_color_image = d.fn<PFN_vkCmdClearColorImage>("vkCmdClearColorImage");
    vk.clear_depth_stencil_image = d.fn<PFN_vkCmdClearDepthStencilImage>("vkCmdClearDepthStencilImage");
    vk.pipeline_barrier2 = d.fn<PFN_vkCmdPipelineBarrier2>("vkCmdPipelineBarrier2");
    vk.separate_depth_stencil_layouts = true;
    LayoutTracker tracker(true);
    tracker.track(image_id, image_binding, VK_IMAGE_LAYOUT_GENERAL);

    // Ray-tracing, video and split syncs map to ALL_COMMANDS, which every access accepts; they are kept anyway.
    const std::uint32_t syncs[] = {y::all, y::draw, y::index_input, y::vertex_shading, y::pixel_shading,
                                   y::depth_stencil, y::render_target, y::compute_shading, y::raytracing, y::copy,
                                   y::resolve, y::execute_indirect, y::all_shading, y::non_pixel_shading,
                                   y::emit_raytracing_acceleration_structure_postbuild_info,
                                   y::clear_unordered_access_view, y::video_decode,
                                   y::build_raytracing_acceleration_structure};
    const std::uint32_t accesses[] = {a::common, a::vertex_buffer, a::constant_buffer, a::index_buffer,
                                      a::render_target, a::unordered_access, a::depth_stencil_write,
                                      a::depth_stencil_read, a::shader_resource, a::stream_output,
                                      a::indirect_argument, a::copy_dest, a::copy_source, a::resolve_dest,
                                      a::resolve_source, a::raytracing_acceleration_structure_read,
                                      a::raytracing_acceleration_structure_write, a::shading_rate_source,
                                      a::video_decode_read, a::no_access};
    const BarrierSubresourceRange all{all_subresources, 0, 0, 0, 0, 0};
    CommandList list;
    unsigned pairs = 0;
    for (const std::uint32_t sync : syncs) {
        for (const std::uint32_t access : accesses) {
            EXPECT(list.barrier(GlobalBarrier{sync, sync, access, access}));
            EXPECT(list.barrier(BufferBarrier{sync, sync, access, access, buffer_id}));
            EXPECT(list.barrier(TextureBarrier{sync, sync, access, access, l::common, l::common, image_id, all, 0}));
            ++pairs;
        }
    }
    EXPECT(list.close());
    const unsigned errors_before = d.validation_errors();
    const bool ok = d.run([&](VkCommandBuffer cb) {
        d.initial_layout(cb, image, VK_IMAGE_ASPECT_COLOR_BIT, 1, VK_IMAGE_LAYOUT_GENERAL);
        return replay(list, cb, resolve, vk, &tracker) == ReplayResult::success;
    });
    EXPECT(ok);
    const unsigned errors = d.validation_errors() - errors_before;
    std::printf("D: %u sync/access pairs as global, buffer and texture barriers%s\n", pairs,
                d.validation ? (errors ? ": validation errors" : ", no validation errors") : " (validation off)");
    return 0;
}

// Part E (D3D-26): WriteBufferImmediate in its three modes, AtomicCopyBufferUINT/UINT64 and DiscardResource.
// Buffers carry console VAs; one binding is a range inside another buffer, as resources in the GPU address arena
// are. The writes and copies are ordered so that a missing barrier would let a copy read a stale or a later value.
// A colour target is cleared, discarded and cleared again; a second one is only discarded in part.
int part_e(vktest::Device& d) {
    constexpr VkDeviceSize bytes = 256, c_offset = 128, c_size = 64;
    constexpr std::uint64_t va_a = 0x10000, va_b = 0x20000, va_c = 0x30000;
    constexpr std::uint32_t size = 8, texels = size * size;
    constexpr VkBufferUsageFlags usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    void *a_mapped = nullptr, *b_mapped = nullptr, *image_mapped = nullptr;
    const VkBuffer a = d.buffer(bytes, usage, &a_mapped);
    const VkBuffer b = d.buffer(bytes, usage, &b_mapped);
    const VkBuffer pixels = d.buffer(texels * 4 * 2, VK_BUFFER_USAGE_TRANSFER_DST_BIT, &image_mapped);
    const VkImage rt = d.image(VK_FORMAT_R8G8B8A8_UNORM, size, size, 1, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT);
    const VkImage kept = d.image(VK_FORMAT_R8G8B8A8_UNORM, size, size, 1, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT);
    EXPECT(a && b && pixels && rt && kept);

    const auto buffer_binding = [](VkBuffer buffer, VkDeviceSize offset, VkDeviceSize length, std::uint64_t va) {
        ResourceBinding r{};
        r.buffer = buffer;
        r.buffer_offset = offset;
        r.size = length;
        r.va = va;
        return r;
    };
    const auto image_binding = [&](VkImage image) {
        ResourceBinding r{};
        r.image = image;
        r.format = VK_FORMAT_R8G8B8A8_UNORM;
        r.width = r.height = size;
        return r;
    };
    constexpr ResourceId a_id = 30, b_id = 31, c_id = 32, rt_id = 33, kept_id = 34;
    const std::map<ResourceId, ResourceBinding> bindings{
        {a_id, buffer_binding(a, 0, bytes, va_a)},
        {b_id, buffer_binding(b, 0, c_offset, va_b)},         // the first half of b
        {c_id, buffer_binding(b, c_offset, c_size, va_c)},    // a range at 128 in b
        {rt_id, image_binding(rt)},
        {kept_id, image_binding(kept)},
    };
    const auto resolve = [&](ResourceId id) -> const ResourceBinding* {
        const auto it = bindings.find(id);
        return it == bindings.end() ? nullptr : &it->second;
    };
    LayoutTracker tracker(true);
    tracker.track(rt_id, bindings.at(rt_id), VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    tracker.track(kept_id, bindings.at(kept_id), VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);

    ReplayFunctions vk{};
    vk.pipeline_barrier = d.fn<PFN_vkCmdPipelineBarrier>("vkCmdPipelineBarrier");
    vk.clear_color_image = d.fn<PFN_vkCmdClearColorImage>("vkCmdClearColorImage");
    vk.clear_depth_stencil_image = d.fn<PFN_vkCmdClearDepthStencilImage>("vkCmdClearDepthStencilImage");
    vk.copy_buffer = d.fn<PFN_vkCmdCopyBuffer>("vkCmdCopyBuffer");
    vk.copy_buffer_to_image = d.fn<PFN_vkCmdCopyBufferToImage>("vkCmdCopyBufferToImage");
    vk.copy_image_to_buffer = d.fn<PFN_vkCmdCopyImageToBuffer>("vkCmdCopyImageToBuffer");
    vk.copy_image = d.fn<PFN_vkCmdCopyImage>("vkCmdCopyImage");
    vk.separate_depth_stencil_layouts = true;
    const ReplayFunctions without_update = vk;
    vk.update_buffer = d.fn<PFN_vkCmdUpdateBuffer>("vkCmdUpdateBuffer");
    vk.buffer_at = [&](std::uint64_t va, std::uint64_t length, VkBuffer& buffer, VkDeviceSize& offset) {
        for (const auto& [id, r] : bindings) {
            if (r.image || !r.va || va < r.va || va - r.va >= r.size || length > r.size - (va - r.va)) continue;
            buffer = r.buffer;
            offset = r.buffer_offset + (va - r.va);
            return true;
        }
        return false;
    };

    namespace m = write_immediate_mode;
    CommandList list;
    EXPECT(list.write_immediate({va_a, 0x11111111, m::normal}));
    EXPECT(list.write_immediate({va_a + 4, 0x22222222, m::marker_in}));
    EXPECT(list.write_immediate({va_c, 0x33333333, m::marker_out}));       // lands at b + 128
    EXPECT(list.write_immediate({va_c + c_size - 4, 0x55555555, m::normal}));  // the last word of c
    // The copies wait for the writes before them (RAW on a); the MARKER_OUT write after them waits for them (WAR).
    EXPECT(list.atomic_copy({b_id, 0, a_id, 4, 4, {{a_id, 0, 0, 8}}}));
    EXPECT(list.atomic_copy({b_id, 8, a_id, 0, 8, {{a_id, 0, 0, 8}, {c_id, 0, 0, 4}}}));
    EXPECT(list.write_immediate({va_a, 0x44444444, m::marker_out}));
    EXPECT(list.atomic_copy({b_id, 16, a_id, 0, 4, {{a_id, 0, 0, 4}}}));
    EXPECT(list.atomic_copy({c_id, 8, b_id, 0, 4, {{b_id, 0, 0, 4}}}));    // into the range inside b
    EXPECT(list.discard({a_id, 0, all_subresources, false}));             // a buffer: kept
    // rt: red, discarded whole, then green over the whole view. kept: red, then a discard with rectangles.
    EXPECT(list.clear_render_target({rt_id, 0, 0, 1}, {1, 0, 0, 1}));
    EXPECT(list.discard({rt_id, 0, all_subresources, false}));
    EXPECT(list.clear_render_target({rt_id, 0, 0, 1}, {0, 1, 0, 1}));
    EXPECT(list.transition({rt_id, 0, s::render_target, s::copy_source}));
    EXPECT(list.clear_render_target({kept_id, 0, 0, 1}, {1, 0, 0, 1}));
    EXPECT(list.discard({kept_id, 0, 1, true}));
    EXPECT(list.transition({kept_id, 0, s::render_target, s::copy_source}));
    EXPECT(list.close());

    bool ok = d.run([&](VkCommandBuffer cb) {
        d.initial_layout(cb, rt, VK_IMAGE_ASPECT_COLOR_BIT, 1, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
        d.initial_layout(cb, kept, VK_IMAGE_ASPECT_COLOR_BIT, 1, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
        CommandList bad;
        const auto rejected = [&](auto&& record, ReplayResult expected, const ReplayFunctions& functions) {
            if (!bad.closed()) bad.close();
            bad.reset();
            if (!record()) return false;
            bad.close();
            const ReplayResult got = replay(bad, cb, resolve, functions, &tracker);
            if (got != expected) std::fprintf(stderr, "E rejected list: got %d, want %d\n", int(got), int(expected));
            return got == expected;
        };
        const auto write = [&](std::uint64_t va) { return [&bad, va] { return bad.write_immediate({va, 1, 0}); }; };
        const auto atomic = [&](AtomicCopyBuffer c) { return [&bad, c] { return bad.atomic_copy(c); }; };
        const auto discard = [&](DiscardResource c) { return [&bad, c] { return bad.discard(c); }; };
        constexpr auto unresolved = ReplayResult::unresolved_resource;
        if (!rejected(write(va_c + c_size), unresolved, vk) ||          // one past c
            !rejected(write(0x40000), unresolved, vk) ||                // no resource there
            !rejected(write(va_a), ReplayResult::unsupported, without_update) ||
            !rejected(atomic({b_id, c_offset, a_id, 0, 8, {{a_id, 0, 0, 8}}}), ReplayResult::invalid_copy, vk) ||  // past b
            !rejected(atomic({b_id, 0, a_id, 0, 4, {{99, 0, 0, 4}}}), unresolved, vk) ||
            !rejected(atomic({b_id, 0, a_id, 0, 4, {{a_id, 0, 0, bytes + 4}}}), ReplayResult::invalid_copy, vk) ||
            !rejected(discard({rt_id, 1, 1, false}), ReplayResult::invalid_subresource, vk) ||
            !rejected(discard({a_id, 1, 1, false}), ReplayResult::invalid_subresource, vk))
            return false;
        if (replay(list, cb, resolve, vk, &tracker) != ReplayResult::success) return false;
        d.read_back(cb, rt, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_ASPECT_COLOR_BIT, size, size, 1, pixels, 0);
        d.read_back(cb, kept, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_ASPECT_COLOR_BIT, size, size, 1, pixels,
                    texels * 4);
        VkMemoryBarrier host{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        host.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        host.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
        vk.pipeline_barrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &host, 0, nullptr, 0,
                            nullptr);
        return true;
    });
    EXPECT(ok);
    EXPECT(tracker.layout(rt_id, 0) == VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
    EXPECT(tracker.layout(kept_id, 0) == VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);

    const auto word = [](const void* base, VkDeviceSize offset) {
        std::uint32_t value;
        std::memcpy(&value, static_cast<const std::uint8_t*>(base) + offset, 4);
        return value;
    };
    constexpr std::uint32_t untouched = 0xcdcdcdcd;
    std::map<VkDeviceSize, std::uint32_t> want_a{{0, 0x44444444}, {4, 0x22222222}};
    std::map<VkDeviceSize, std::uint32_t> want_b{{0, 0x22222222},      {8, 0x11111111},
                                                 {12, 0x22222222},     {16, 0x44444444},
                                                 {c_offset, 0x33333333}, {c_offset + 8, 0x22222222},
                                                 {c_offset + c_size - 4, 0x55555555}};
    for (VkDeviceSize offset = 0; offset < bytes; offset += 4) {
        for (const auto& [name, base, want] : {std::tuple{"E buffer a", a_mapped, &want_a},
                                               std::tuple{"E buffer b", b_mapped, &want_b}}) {
            const auto it = want->find(offset);
            const std::uint32_t expected = it == want->end() ? untouched : it->second;
            const std::uint32_t got = word(base, offset);
            if (got != expected && g_mismatches++ < 8)
                std::fprintf(stderr, "%s +%llu: got 0x%08x, want 0x%08x\n", name, (unsigned long long)offset, got,
                             expected);
        }
    }
    const auto* image_data = static_cast<const std::uint8_t*>(image_mapped);
    for (std::uint32_t i = 0; i < texels; ++i) {
        expect_rgba("E discarded then cleared", image_data, i, green);
        expect_rgba("E partly discarded", image_data + texels * 4, i, red);
    }
    std::printf("E: immediate writes in three modes, UINT and UINT64 atomic copies, buffer and texture discards\n");
    return 0;
}

} // namespace

int main() {
    vktest::Device device;
    EXPECT(device.create());
    if (part_a(device) || part_b(device) || part_c(device) || part_d(device) || part_e(device)) return 1;
    if (g_mismatches) return 1;
    if (device.validation_errors()) {
        std::fprintf(stderr, "%u validation errors\n", device.validation_errors());
        return 1;
    }
    return 0;
}
