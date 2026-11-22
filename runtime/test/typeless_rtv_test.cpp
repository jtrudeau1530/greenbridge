// CPU-only replay tests for typed RGBA8 views. No Vulkan loader, GPU or game data.
#include "../src/vulkan_replay.h"
#include <array>
#include <cstdio>
#include <cstring>

#define EXPECT(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); return 1; } \
} while (0)

namespace {
unsigned image_clears = 0, attachment_clears = 0;
VkFormat requested_format = VK_FORMAT_UNDEFINED;
VkClearColorValue captured{};
VkClearRect captured_rect{};
VkRect2D render_area{};
VKAPI_ATTR void VKAPI_CALL barrier(VkCommandBuffer, VkPipelineStageFlags, VkPipelineStageFlags, VkDependencyFlags,
    std::uint32_t, const VkMemoryBarrier*, std::uint32_t, const VkBufferMemoryBarrier*,
    std::uint32_t, const VkImageMemoryBarrier*) {}
VKAPI_ATTR void VKAPI_CALL clear(VkCommandBuffer, VkImage, VkImageLayout, const VkClearColorValue* value,
    std::uint32_t, const VkImageSubresourceRange*) { captured = *value; ++image_clears; }
VKAPI_ATTR void VKAPI_CALL begin(VkCommandBuffer, const VkRenderingInfo* info) { render_area = info->renderArea; }
VKAPI_ATTR void VKAPI_CALL end(VkCommandBuffer) {}
VKAPI_ATTR void VKAPI_CALL attachments(VkCommandBuffer, std::uint32_t n, const VkClearAttachment* values,
    std::uint32_t count, const VkClearRect* rects) {
    if (n == 1 && count == 1) { captured = values[0].clearValue.color; captured_rect = rects[0]; ++attachment_clears; }
}
leonida::ReplayFunctions functions() {
    leonida::ReplayFunctions vk{};
    vk.pipeline_barrier = barrier;
    vk.clear_color_image = clear;
    vk.begin_rendering = begin;
    vk.end_rendering = end;
    vk.clear_attachments = attachments;
    vk.image_view = [](const auto& binding, const auto&) {
        requested_format = binding.format;
        return reinterpret_cast<VkImageView>(std::uintptr_t{1});
    };
    return vk;
}
leonida::ResourceBinding resource() {
    leonida::ResourceBinding r{};
    r.image = reinterpret_cast<VkImage>(std::uintptr_t{1});
    r.format = VK_FORMAT_R8G8B8A8_UNORM;
    r.width = 5; r.height = 3;
    r.rgba8_typeless = true;
    return r;
}
void reset_counters() {
    image_clears = attachment_clears = 0;
    requested_format = VK_FORMAT_UNDEFINED;
    captured = {}; captured_rect = {}; render_area = {};
}
leonida::ReplayResult run(const leonida::CommandList& list, const leonida::ResourceBinding& resource,
                          const leonida::ReplayFunctions& vk) {
    reset_counters();
    return leonida::replay(list, VK_NULL_HANDLE, [&](auto) { return &resource; }, vk);
}
} // namespace

int main() {
    using namespace leonida;
    const auto vk = functions();
    const auto binding = resource();
    const ClearColor color{1.5f, 2.5f, 256, -1};
    const Rect rectangle{1, 1, 4, 3};
    const std::pair<ColorViewFormat, VkFormat> formats[] = {
        {ColorViewFormat::rgba8_unorm, VK_FORMAT_R8G8B8A8_UNORM},
        {ColorViewFormat::rgba8_srgb, VK_FORMAT_R8G8B8A8_SRGB},
        {ColorViewFormat::rgba8_uint, VK_FORMAT_R8G8B8A8_UINT}
    };
    unsigned checks = 0;
    for (const auto& [format, expected] : formats) {
        for (const bool rect : {false, true}) {
            CommandList list;
            TargetView target{1, 0, 0, 1, format};
            EXPECT(list.clear_render_target(target, color, rect ? std::span<const Rect>(&rectangle, 1) : std::span<const Rect>{}));
            target.format = ColorViewFormat::resource; // Mutating the descriptor must not change the recorded format.
            EXPECT(list.close());
            EXPECT(run(list, binding, vk) == ReplayResult::success);
            const bool attachment = rect || format != ColorViewFormat::rgba8_unorm;
            EXPECT(image_clears == unsigned(!attachment) && attachment_clears == unsigned(attachment));
            if (attachment) {
                EXPECT(requested_format == expected);
                EXPECT(render_area.extent.width == 5 && render_area.extent.height == 3);
                EXPECT(captured_rect.rect.offset.x == (rect ? 1 : 0));
                EXPECT(captured_rect.rect.offset.y == (rect ? 1 : 0));
                EXPECT(captured_rect.rect.extent.width == (rect ? 3u : 5u));
                EXPECT(captured_rect.rect.extent.height == (rect ? 2u : 3u));
            }
            if (format == ColorViewFormat::rgba8_uint) {
                EXPECT(captured.uint32[0] == 1 && captured.uint32[1] == 2 && captured.uint32[2] == 255 && captured.uint32[3] == 0);
            } else {
                EXPECT(std::memcmp(captured.float32, color.data(), sizeof(captured)) == 0);
            }
            ++checks;
        }
    }
    // Whole-view clears must keep main's predication attachment path, including same-format UNORM.
    auto conditional = vk;
    conditional.begin_conditional_rendering = +[](VkCommandBuffer, const VkConditionalRenderingBeginInfoEXT*) {};
    conditional.end_conditional_rendering = end;
    ResourceBinding predicate{};
    predicate.buffer = reinterpret_cast<VkBuffer>(std::uintptr_t{2});
    predicate.size = 8;
    for (const auto& [format, expected] : formats) {
        CommandList predicated;
        EXPECT(predicated.set_predication({2, 0, d3d12_predication::equal_zero}));
        EXPECT(predicated.clear_render_target({1, 0, 0, 1, format}, color) && predicated.close());
        PreparedList prepared;
        reset_counters();
        const auto resolve = [&](ResourceId id) { return id == 2 ? &predicate : &binding; };
        EXPECT(prepare(predicated, resolve, conditional, prepared) == ReplayResult::success);
        EXPECT(prepared.commands[1].rects.size() == 1);
        EXPECT(prepared.commands[1].rects[0].rect.extent.width == binding.width);
        EXPECT(prepared.commands[1].rects[0].rect.extent.height == binding.height);
        EXPECT(requested_format == expected);
        ++checks;
    }
    const auto rejects = [&](ResourceBinding r, TargetView view) {
        CommandList list;
        if (!list.clear_render_target(view, color) || !list.close()) return false;
        return run(list, r, vk) == ReplayResult::invalid_clear && !image_clears && !attachment_clears;
    };
    EXPECT(rejects(binding, {1, 0, 0, 1})); // A typeless resource has no default typed RTV format.
    EXPECT(rejects(binding, {1, 0, 0, 1, static_cast<ColorViewFormat>(99)}));
    auto wrong = binding; wrong.rgba8_typeless = false;
    EXPECT(rejects(wrong, {1, 0, 0, 1, ColorViewFormat::rgba8_srgb}));
    wrong = binding; wrong.format = VK_FORMAT_R32_SFLOAT;
    EXPECT(rejects(wrong, {1, 0, 0, 1, ColorViewFormat::rgba8_uint}));
    wrong = binding; wrong.mip_levels = 2;
    EXPECT(rejects(wrong, {1, 0, 0, 1, ColorViewFormat::rgba8_uint}));
    wrong = binding; wrong.array_layers = 2;
    EXPECT(rejects(wrong, {1, 0, 0, 1, ColorViewFormat::rgba8_srgb}));
    wrong = binding; wrong.volume = true;
    EXPECT(rejects(wrong, {1, 0, 0, 1, ColorViewFormat::rgba8_srgb}));
    wrong = binding; wrong.samples = 4;
    EXPECT(rejects(wrong, {1, 0, 0, 1, ColorViewFormat::rgba8_uint}));
    checks += 8;
    CommandList list;
    EXPECT(list.clear_render_target({1, 0, 0, 1, ColorViewFormat::rgba8_srgb}, color) && list.close());
    auto unavailable = vk; unavailable.begin_rendering = nullptr;
    EXPECT(run(list, binding, unavailable) == ReplayResult::unsupported && !image_clears && !attachment_clears);
    CommandList same_format;
    EXPECT(same_format.clear_render_target({1, 0, 0, 1, ColorViewFormat::rgba8_unorm}, color) && same_format.close());
    EXPECT(run(same_format, binding, unavailable) == ReplayResult::success && image_clears == 1 && !attachment_clears);
    ++checks;
    auto legacy = binding;
    legacy.rgba8_typeless = false;
    legacy.format = VK_FORMAT_UNDEFINED; // Existing ownerless float-only replay clients omit this metadata.
    CommandList legacy_list;
    EXPECT(legacy_list.clear_render_target({1, 0, 0, 1}, color) && legacy_list.close());
    EXPECT(run(legacy_list, legacy, unavailable) == ReplayResult::success && image_clears == 1 && !attachment_clears);
    EXPECT(std::memcmp(captured.float32, color.data(), sizeof(captured)) == 0);
    ++checks;
    CommandList clipped;
    const Rect outside{20, 20, 30, 30};
    EXPECT(clipped.clear_render_target({1, 0, 0, 1, ColorViewFormat::rgba8_srgb}, color, {&outside, 1}) && clipped.close());
    EXPECT(run(clipped, binding, vk) == ReplayResult::success && !image_clears && !attachment_clears);
    EXPECT(requested_format == VK_FORMAT_UNDEFINED);
    CommandList invalid_batch;
    EXPECT(invalid_batch.clear_render_target({1, 0, 0, 1, ColorViewFormat::rgba8_srgb}, color));
    EXPECT(invalid_batch.clear_render_target({1, 0, 0, 1, static_cast<ColorViewFormat>(99)}, color) && invalid_batch.close());
    EXPECT(run(invalid_batch, binding, vk) == ReplayResult::invalid_clear && !image_clears && !attachment_clears);
    checks += 3;
    std::printf("typeless_rtv_test: %u format, snapshot, whole/rectangle and rejection scenarios passed\n", checks);
    return 0;
}
