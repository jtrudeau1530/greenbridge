// A buffer UAV clear on a resource that is a range of a larger VkBuffer (ResourceBinding::buffer_offset, as the GPU
// address arena places every buffer) writes that range, not the start of the VkBuffer. UavView::offset is relative to
// the resource, as copies and atomic copies already treat it. No Vulkan loader, GPU or game data: the transfer
// functions are fakes that record the ranges they are given.
#include "../src/vulkan_replay.h"
#include <cstdint>
#include <cstdio>
#include <vector>

namespace {
struct Write {
    char kind;  // 'f' fill, 'u' update, 'c' copy (dst; src in from)
    VkDeviceSize offset, size, from;
};
std::vector<Write> writes;

VKAPI_ATTR void VKAPI_CALL barrier(VkCommandBuffer, VkPipelineStageFlags, VkPipelineStageFlags, VkDependencyFlags,
                                   std::uint32_t, const VkMemoryBarrier*, std::uint32_t, const VkBufferMemoryBarrier*,
                                   std::uint32_t, const VkImageMemoryBarrier*) {}
VKAPI_ATTR void VKAPI_CALL fill(VkCommandBuffer, VkBuffer, VkDeviceSize offset, VkDeviceSize size, std::uint32_t) {
    writes.push_back({'f', offset, size, 0});
}
VKAPI_ATTR void VKAPI_CALL update(VkCommandBuffer, VkBuffer, VkDeviceSize offset, VkDeviceSize size, const void*) {
    writes.push_back({'u', offset, size, 0});
}
VKAPI_ATTR void VKAPI_CALL copy(VkCommandBuffer, VkBuffer, VkBuffer, std::uint32_t count, const VkBufferCopy* r) {
    for (std::uint32_t i = 0; i < count; ++i) writes.push_back({'c', r[i].dstOffset, r[i].size, r[i].srcOffset});
}

int failures = 0;
void expect(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++failures;
    }
}

// Every byte written (and read by a copy) lies in [begin, end) of the VkBuffer.
bool inside(VkDeviceSize begin, VkDeviceSize end) {
    if (writes.empty()) return false;
    for (const Write& w : writes) {
        if (w.offset < begin || w.offset + w.size > end) return false;
        if (w.kind == 'c' && (w.from < begin || w.from + w.size > end)) return false;
    }
    return true;
}
}  // namespace

int main() {
    using namespace leonida;
    constexpr VkDeviceSize at = 65536, bytes = 256;
    ResourceBinding range{};
    range.buffer = reinterpret_cast<VkBuffer>(std::uintptr_t{0x1000});
    range.buffer_offset = at;  // a range at 64 KiB of a shared buffer
    range.size = bytes;
    const auto resolve = [&](ResourceId id) -> const ResourceBinding* { return id == 1 ? &range : nullptr; };
    ReplayFunctions vk{};
    vk.pipeline_barrier = barrier;
    vk.fill_buffer = fill;
    vk.update_buffer = update;
    vk.copy_buffer = copy;

    const auto run = [&](const UavView& view, std::span<const Rect> rects = {}) {
        writes.clear();
        CommandList list;
        list.clear_unordered_access(view, false, {7, 7, 7, 7}, rects);
        list.close();
        return replay(list, VK_NULL_HANDLE, resolve, vk);
    };

    // A raw view of the whole resource: one fill of [at, at + 256).
    expect(run({1, true, 0, 0, bytes, 4, 0, 0, 1}) == ReplayResult::success, "raw whole-view clear replays");
    expect(writes.size() == 1 && writes[0].kind == 'f' && writes[0].offset == at && writes[0].size == bytes,
           "raw whole-view clear fills the resource's range of the buffer");
    // A view starting 16 bytes into the resource.
    expect(run({1, true, 0, 16, 64, 4, 0, 0, 1}) == ReplayResult::success, "raw view at 16 replays");
    expect(inside(at + 16, at + 80), "raw view at 16 stays in [at + 16, at + 80)");
    // R8_UINT elements from byte 1: unaligned ends are copied from the aligned middle, all inside the view.
    expect(run({1, true, VK_FORMAT_R8_UINT, 1, 37, 1, 0, 0, 1}) == ReplayResult::success, "R8 view at 1 replays");
    expect(inside(at + 1, at + 38), "R8 view at 1 stays in [at + 1, at + 38)");
    // R32G32B32A32_UINT (16-byte elements) is written with vkCmdUpdateBuffer and doubled with copies.
    expect(run({1, true, VK_FORMAT_R32G32B32A32_UINT, 32, 128, 16, 0, 0, 1}) == ReplayResult::success,
           "RGBA32 view at 32 replays");
    expect(inside(at + 32, at + 160), "RGBA32 view at 32 stays in [at + 32, at + 160)");
    // Rectangles count elements from the view's first byte.
    const Rect rect{2, 0, 5, 1};
    expect(run({1, true, 0, 8, 64, 4, 0, 0, 1}, std::span<const Rect>(&rect, 1)) == ReplayResult::success,
           "rectangle clear replays");
    expect(inside(at + 16, at + 28), "elements 2..5 of a view at 8 are [at + 16, at + 28)");

    if (failures) return 1;
    std::printf("replay_uav_clear_offset: buffer UAV clears write their resource's range of a shared buffer\n");
    return 0;
}
