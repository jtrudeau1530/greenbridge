// Value-level replay regression: full and rectangular UINT clears must carry integer values, not float bits.
// No Vulkan loader, GPU, shaders or game data. Expected conversions also match the WARP differential GPU test.
#include "../src/vulkan_replay.h"
#include <array>
#include <cstdio>
#include <cstring>
#include <limits>

namespace {
VkClearColorValue captured{};
unsigned calls = 0;
VKAPI_ATTR void VKAPI_CALL barrier(VkCommandBuffer, VkPipelineStageFlags, VkPipelineStageFlags, VkDependencyFlags,
    std::uint32_t, const VkMemoryBarrier*, std::uint32_t, const VkBufferMemoryBarrier*,
    std::uint32_t, const VkImageMemoryBarrier*) {}
VKAPI_ATTR void VKAPI_CALL clear(VkCommandBuffer, VkImage, VkImageLayout, const VkClearColorValue* color,
    std::uint32_t, const VkImageSubresourceRange*) { captured = *color; ++calls; }
VKAPI_ATTR void VKAPI_CALL begin(VkCommandBuffer, const VkRenderingInfo*) {}
VKAPI_ATTR void VKAPI_CALL end(VkCommandBuffer) {}
VKAPI_ATTR void VKAPI_CALL attachments(VkCommandBuffer, std::uint32_t n, const VkClearAttachment* a,
    std::uint32_t, const VkClearRect*) { if (n == 1) { captured = a[0].clearValue.color; ++calls; } }

bool replay_color(VkFormat format, const leonida::ClearColor& color, bool rectangle) {
    leonida::CommandList list;
    const leonida::Rect rect{1, 1, 3, 3};
    if (!list.clear_render_target({1, 0, 0, 1}, color,
        rectangle ? std::span<const leonida::Rect>(&rect, 1) : std::span<const leonida::Rect>{}) || !list.close()) return false;
    leonida::ResourceBinding binding{};
    binding.image = reinterpret_cast<VkImage>(std::uintptr_t{1});
    binding.format = format;
    binding.width = binding.height = 4;
    leonida::ReplayFunctions vk{};
    vk.pipeline_barrier = barrier;
    vk.clear_color_image = clear;
    vk.begin_rendering = begin;
    vk.end_rendering = end;
    vk.clear_attachments = attachments;
    vk.image_view = [](const auto&, const auto&) { return reinterpret_cast<VkImageView>(std::uintptr_t{1}); };
    calls = 0;
    return leonida::replay(list, VK_NULL_HANDLE, [&](auto) { return &binding; }, vk) == leonida::ReplayResult::success && calls == 1;
}
} // namespace

int main() {
    const float nan = std::numeric_limits<float>::quiet_NaN(), inf = std::numeric_limits<float>::infinity();
    struct Case { leonida::ClearColor color; std::array<std::uint32_t, 4> u8, u32; };
    const Case cases[] = {
        {{1, 2, 3, 4}, {1, 2, 3, 4}, {1, 2, 3, 4}},
        {{1.5f, 2.5f, 3.5f, 4.5f}, {1, 2, 3, 4}, {1, 2, 3, 4}},
        {{-1, -0.0f, 254.5f, 255.5f}, {0, 0, 254, 255}, {0, 0, 254, 255}},
        {{256, 65536, 16777216, 4294967296.0f}, {255, 255, 255, 255}, {256, 65536, 16777216, 0xffffffffu}},
        {{nan, inf, -inf, 0.5f}, {0, 255, 0, 0}, {0, 0xffffffffu, 0, 0}},
        {{0.499f, 0.501f, 1.499f, 1.501f}, {0, 0, 1, 1}, {0, 0, 1, 1}},
        {{4294967040.0f, std::numeric_limits<float>::max(), 255, 0}, {255, 255, 255, 0}, {0xffffff00u, 0xffffffffu, 255, 0}},
    };
    unsigned checks = 0;
    for (const auto& c : cases) {
        for (const VkFormat format : {VK_FORMAT_R8G8B8A8_UINT, VK_FORMAT_R32_UINT, VK_FORMAT_R32G32B32A32_UINT}) {
            for (const bool rect : {false, true}) {
                const auto& expected = format == VK_FORMAT_R8G8B8A8_UINT ? c.u8 : c.u32;
                if (!replay_color(format, c.color, rect)) return 1;
                for (unsigned i = 0; i < 4; ++i) {
                    if (captured.uint32[i] != expected[i]) {
                        std::fprintf(stderr, "UINT clear: format %d rect %d channel %u: got %u, expected %u\n",
                            int(format), int(rect), i, captured.uint32[i], expected[i]);
                        return 1;
                    }
                }
                ++checks;
            }
        }
        for (const VkFormat format : {VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_R32_SFLOAT}) {
            for (const bool rect : {false, true}) {
                if (!replay_color(format, c.color, rect) || std::memcmp(captured.float32, c.color.data(), sizeof(captured)) != 0)
                    return 1;
                ++checks;
            }
        }
    }
    std::printf("clear_color_test: %u full/rectangle UINT and float replay checks passed\n", checks);
    return 0;
}
