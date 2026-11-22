// Value-level replay regression: full and rectangular integer clears must carry integer values, not float bits.
// No Vulkan loader, GPU, shaders or game data. Expected conversions also match the WARP differential GPU test.
#include "../src/vulkan_replay.h"
#include <array>
#include <cmath>
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

// Integer RTV format with its component range; A2B10G10R10 has a 2-bit alpha.
struct IntegerFormat { VkFormat format; const char* name; double low, high, alpha_high; bool is_signed; };

// Reference conversion measured against WARP: NaN -> 0, clamp to the range, truncate fractions toward zero.
std::uint32_t expected_bits(float input, double low, double high, bool is_signed) {
    const double v = double(input);
    const double r = v != v ? 0.0 : v <= low ? low : v >= high ? high : std::trunc(v);
    return is_signed ? static_cast<std::uint32_t>(static_cast<std::int32_t>(r)) : static_cast<std::uint32_t>(r);
}
} // namespace

int main() {
    const float nan = std::numeric_limits<float>::quiet_NaN(), inf = std::numeric_limits<float>::infinity();
    const leonida::ClearColor cases[] = {
        {1, 2, 3, 4},
        {1.5f, 2.5f, 3.5f, 4.5f},
        {-1, -0.0f, 254.5f, 255.5f},
        {256, 65536, 16777216, 4294967296.0f},
        {nan, inf, -inf, 0.5f},
        {0.499f, 0.501f, 1.499f, 1.501f},
        {4294967040.0f, std::numeric_limits<float>::max(), 255, 0},
        // Component-range boundaries for the 8-, 16-, 10- and 2-bit formats and the signed 32-bit limits.
        {127.5f, 128, -128.5f, -129},
        {32767.5f, 32768, -32768.5f, -32769},
        {65535.5f, 65536, 1023.5f, 1024},
        {3.5f, 4, 2147483520.0f, -2147483648.0f},
    };
    constexpr double u32max = 4294967295.0, s32min = -2147483648.0, s32max = 2147483647.0;
    const IntegerFormat integer_formats[] = {
        {VK_FORMAT_R8_UINT, "R8_UINT", 0, 255, 255, false},
        {VK_FORMAT_R8G8_UINT, "R8G8_UINT", 0, 255, 255, false},
        {VK_FORMAT_R8G8B8A8_UINT, "R8G8B8A8_UINT", 0, 255, 255, false},
        {VK_FORMAT_R16_UINT, "R16_UINT", 0, 65535, 65535, false},
        {VK_FORMAT_R16G16_UINT, "R16G16_UINT", 0, 65535, 65535, false},
        {VK_FORMAT_R16G16B16A16_UINT, "R16G16B16A16_UINT", 0, 65535, 65535, false},
        {VK_FORMAT_R32_UINT, "R32_UINT", 0, u32max, u32max, false},
        {VK_FORMAT_R32G32_UINT, "R32G32_UINT", 0, u32max, u32max, false},
        {VK_FORMAT_R32G32B32A32_UINT, "R32G32B32A32_UINT", 0, u32max, u32max, false},
        {VK_FORMAT_A2B10G10R10_UINT_PACK32, "A2B10G10R10_UINT_PACK32", 0, 1023, 3, false},
        {VK_FORMAT_R8_SINT, "R8_SINT", -128, 127, 127, true},
        {VK_FORMAT_R8G8_SINT, "R8G8_SINT", -128, 127, 127, true},
        {VK_FORMAT_R8G8B8A8_SINT, "R8G8B8A8_SINT", -128, 127, 127, true},
        {VK_FORMAT_R16_SINT, "R16_SINT", -32768, 32767, 32767, true},
        {VK_FORMAT_R16G16_SINT, "R16G16_SINT", -32768, 32767, 32767, true},
        {VK_FORMAT_R16G16B16A16_SINT, "R16G16B16A16_SINT", -32768, 32767, 32767, true},
        {VK_FORMAT_R32_SINT, "R32_SINT", s32min, s32max, s32max, true},
        {VK_FORMAT_R32G32_SINT, "R32G32_SINT", s32min, s32max, s32max, true},
        {VK_FORMAT_R32G32B32A32_SINT, "R32G32B32A32_SINT", s32min, s32max, s32max, true},
    };
    unsigned checks = 0, failed_formats = 0;
    for (const auto& f : integer_formats) {
        bool format_ok = true;
        unsigned case_index = 0;
        for (const auto& c : cases) {
            for (const bool rect : {false, true}) {
                if (!replay_color(f.format, c, rect)) {
                    std::fprintf(stderr, "%s: replay failed (case %u rect %d)\n", f.name, case_index, int(rect));
                    return 1;
                }
                for (unsigned i = 0; i < 4 && format_ok; ++i) {
                    const std::uint32_t expected = expected_bits(c[i], f.low, i == 3 ? f.alpha_high : f.high, f.is_signed);
                    if (captured.uint32[i] != expected) {
                        std::fprintf(stderr, "integer clear %s case %u rect %d channel %u: got 0x%08x, expected 0x%08x\n",
                            f.name, case_index, int(rect), i, captured.uint32[i], expected);
                        format_ok = false;
                    }
                }
                if (!format_ok) break;
                ++checks;
            }
            if (!format_ok) break;
            ++case_index;
        }
        if (!format_ok) ++failed_formats;
    }
    for (const auto& c : cases) {
        for (const VkFormat format : {VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_R32_SFLOAT}) {
            for (const bool rect : {false, true}) {
                auto expected = c;
                // D3D12 rounds a UNORM tie up (0.5 is code 128 of 255); NVIDIA rounds it down, so the layer clears with the code above.
                if (format == VK_FORMAT_R8G8B8A8_UNORM)
                    for (auto& v : expected)
                        if (v == 0.5f) v = 128.0f / 255.0f;
                if (!replay_color(format, c, rect) || std::memcmp(captured.float32, expected.data(), sizeof(captured)) != 0) {
                    std::fprintf(stderr, "float clear: format %d rect %d changed the value\n", int(format), int(rect));
                    return 1;
                }
                ++checks;
            }
        }
    }
    if (failed_formats) {
        std::fprintf(stderr, "clear_color_test: %u integer format(s) failed\n", failed_formats);
        return 1;
    }
    std::printf("clear_color_test: %u full/rectangle integer and float replay checks passed\n", checks);
    return 0;
}
