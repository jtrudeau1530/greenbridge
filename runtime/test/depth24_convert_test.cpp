// The compute pass behind D24_UNORM_S8_UINT on a GPU without D24 (src/depth24_convert.comp, vulkan_depth24.h), checked
// on the GPU against the C++ reference of depth24.h over the whole 24-bit range:
//   1. every one of the 16777216 words, upload direction: the float must be the one nearest to d / 16777215
//      (the top byte of the word is junk and must be ignored);
//   2. the same floats back through the readback direction: every word must return unchanged, with a zero top byte;
//   3. 16777216 float bit patterns (every 256th pattern of the 32-bit range, offset by a rotating low byte, plus the
//      specials: zeros, denormals, NaNs, infinities, negatives, 1.0 and its neighbours), readback direction, against
//      depth24_quantize;
//   4. a pitched copy (row and slice strides larger than the extent, an offset into the buffer): only the texels of
//      the region change, to the right values.
// Set LEONIDA_VK_VALIDATION=1 to run under VK_LAYER_KHRONOS_validation; any validation error fails the test.
#include "vk_test_device.h"
#include "../src/vulkan_depth24.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

#define EXPECT(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); return 1; } \
} while (0)

namespace {

using namespace leonida;

constexpr std::uint32_t words = 1u << 24;

float as_float(std::uint32_t bits) {
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}
std::uint32_t as_bits(float f) {
    std::uint32_t bits;
    std::memcpy(&bits, &f, 4);
    return bits;
}

}  // namespace

int main() {
    vktest::Device gpu;
    if (!gpu.create()) {
        std::printf("depth24_convert_test: no Vulkan device, skipped\n");
        return 0;
    }

    // The converter allocates nothing here (every job brings its scratch buffer), but the callback must exist.
    Depth24Converter converter(gpu.device_proc, gpu.device, [](const VkMemoryRequirements&) { return VK_NULL_HANDLE; });
    EXPECT(converter.loaded());

    constexpr VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                         VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    std::uint32_t *app_words = nullptr, *scratch_words = nullptr;
    const VkBuffer app = gpu.buffer(std::uint64_t(words) * 4 + 4096, usage, reinterpret_cast<void**>(&app_words));
    const VkBuffer scratch = gpu.buffer(std::uint64_t(words) * 4, usage, reinterpret_cast<void**>(&scratch_words));
    EXPECT(app && scratch && app_words && scratch_words);

    const auto convert = [&](bool to_float, std::uint32_t width, std::uint32_t height, std::uint32_t slices,
                             std::uint32_t row_words, std::uint32_t slice_words, std::uint64_t offset) {
        Depth24Job job;
        job.to_float = to_float;
        job.app = app;
        job.app_offset = offset;
        job.app_size = std::uint64_t(words) * 4 + 4096;
        job.width = width;
        job.height = height;
        job.slices = slices;
        job.app_row_words = row_words;
        job.app_slice_words = slice_words;
        job.scratch = scratch;
        if (!converter.prepare(job)) return false;
        return gpu.run([&](VkCommandBuffer command_buffer) {
            converter.run(command_buffer, job);
            VkMemoryBarrier host{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
            host.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            host.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
            gpu.fn<PFN_vkCmdPipelineBarrier>("vkCmdPipelineBarrier")(command_buffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                                                     VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &host, 0, nullptr,
                                                                     0, nullptr);
            return true;
        });
    };

    // 1. d -> float, for every d, with junk in the top byte.
    for (std::uint32_t d = 0; d < words; ++d) app_words[d] = d | (0xa5u << 24);
    std::memset(scratch_words, 0xcd, std::size_t(words) * 4);
    EXPECT(convert(true, 4096, 4096, 1, 4096, 4096u * 4096u, 0));
    std::uint64_t wrong = 0;
    for (std::uint32_t d = 0; d < words; ++d) {
        const float want = depth24_to_float(d);
        if (as_bits(want) != scratch_words[d] && wrong++ < 8)
            std::fprintf(stderr, "to float: d=%u got %08x want %08x\n", d, scratch_words[d], as_bits(want));
    }
    EXPECT(wrong == 0);
    std::printf("depth24: all %u words to float match the nearest float of d / 16777215\n", words);

    // 2. the same floats back: every word returns, the top byte becomes 0.
    std::memset(app_words, 0xcd, std::size_t(words) * 4);
    EXPECT(convert(false, 4096, 4096, 1, 4096, 4096u * 4096u, 0));
    wrong = 0;
    for (std::uint32_t d = 0; d < words; ++d)
        if (app_words[d] != d && wrong++ < 8) std::fprintf(stderr, "round trip: d=%u got %08x\n", d, app_words[d]);
    EXPECT(wrong == 0);
    std::printf("depth24: all %u words survive float and back\n", words);

    // 3. float patterns to words, against the reference.
    {
        std::uint32_t x = 7;
        for (std::uint32_t i = 0; i < words; ++i) {
            x = x * 1664525u + 1013904223u;
            scratch_words[i] = i * 256u + (x >> 24);  // every 256th pattern of the 32-bit range, jittered inside it
        }
        const std::uint32_t specials[] = {0x00000000, 0x80000000, 0x00000001, 0x007fffff, 0x00800000, 0x33000000,
                                          0x33800000, 0x7f800000, 0xff800000, 0x7fc00000, 0xffc00000, 0x7fa00000,
                                          0x3f800000, 0x3f7fffff, 0x3f800001, 0x3effffff, 0x3f000000, 0x3f000001,
                                          0xbf800000, 0x40000000, 0x7f7fffff, 0xff7fffff, 0x3f7ffffe, 0x34000000,
                                          0x34000001, 0x33ffffff, 0x33c00000, 0x33c00001, 0x33bfffff};
        for (std::size_t i = 0; i < sizeof(specials) / 4; ++i) scratch_words[i] = specials[i];
        // Exactly the floats around the half steps, where the tie rule shows.
        std::uint32_t y = 99;
        for (std::uint32_t i = 64; i < 64 + 400000; i += 3) {
            y = y * 1664525u + 1013904223u;
            const float half = float((double(y >> 8) + 0.5) / 16777215.0);
            scratch_words[i] = as_bits(half);
            scratch_words[i + 1] = as_bits(std::nextafter(half, 0.0f));
            scratch_words[i + 2] = as_bits(std::nextafter(half, 1.0f));
        }
        std::memset(app_words, 0xcd, std::size_t(words) * 4);
        std::vector<std::uint32_t> input(scratch_words, scratch_words + words);
        EXPECT(convert(false, 4096, 4096, 1, 4096, 4096u * 4096u, 0));
        wrong = 0;
        for (std::uint32_t i = 0; i < words; ++i) {
            const std::uint32_t want = depth24_quantize(as_float(input[i]));
            if (app_words[i] != want && wrong++ < 8)
                std::fprintf(stderr, "to words: f=%08x got %u want %u\n", input[i], app_words[i], want);
        }
        EXPECT(wrong == 0);
        std::printf("depth24: %u float patterns (specials, ties and their neighbours included) quantise like the reference\n",
                    words);
    }

    // 4. a pitched copy into the middle of a buffer: words outside the region stay, inside they are converted.
    {
        constexpr std::uint32_t w = 37, h = 5, s = 3, row = 64, slice = 400;
        constexpr std::uint64_t offset = 1100;  // bytes: not a multiple of the 256-byte descriptor window
        std::memset(app_words, 0xcd, 4096 * 4);
        for (std::uint32_t z = 0; z < s; ++z)
            for (std::uint32_t y = 0; y < h; ++y)
                for (std::uint32_t x = 0; x < w; ++x)
                    scratch_words[(z * h + y) * w + x] = as_bits(depth24_to_float((z * 1000 + y * 100 + x) * 5000u + 1));
        EXPECT(convert(false, w, h, s, row, slice, offset));
        for (std::uint32_t i = 0; i < 4096; ++i) {
            const std::uint32_t rel = i - std::uint32_t(offset / 4);
            const bool inside = i >= offset / 4 && rel / slice < s && (rel % slice) / row < h && (rel % slice) % row < w;
            const std::uint32_t want = inside ? ((rel / slice * 1000 + (rel % slice) / row * 100 + (rel % slice) % row) * 5000u + 1)
                                              : 0xcdcdcdcdu;
            if (app_words[i] != want) {
                std::fprintf(stderr, "pitched: word %u got %08x want %08x\n", i, app_words[i], want);
                return 1;
            }
        }
        // And upload from the same layout: the words come out in the scratch, tightly packed.
        std::memset(scratch_words, 0xcd, 4096 * 4);
        EXPECT(convert(true, w, h, s, row, slice, offset));
        for (std::uint32_t z = 0; z < s; ++z)
            for (std::uint32_t y = 0; y < h; ++y)
                for (std::uint32_t x = 0; x < w; ++x) {
                    const std::uint32_t d = (z * 1000 + y * 100 + x) * 5000u + 1;
                    EXPECT(scratch_words[(z * h + y) * w + x] == as_bits(depth24_to_float(d)));
                }
        std::printf("depth24: a pitched copy at an offset touches only its texels\n");
    }

    EXPECT(gpu.validation_errors() == 0);
    std::printf("depth24_convert_test: ok\n");
    return 0;
}
