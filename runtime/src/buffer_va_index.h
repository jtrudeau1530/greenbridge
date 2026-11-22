#pragma once

// Which registered buffer a console VA falls in (VulkanContext::buffer_at, for vertex and index views and
// WriteBufferImmediate). Ranges are kept ordered by their first VA, so a lookup is a binary search instead of a scan of
// every registered resource. No Vulkan or Windows dependency; not thread-safe (VulkanContext guards it with
// resource_mutex_, like the bindings themselves).
//
// D3D12 lets placed buffers alias, so ranges may overlap. A lookup then returns the containing range that starts
// closest below the VA (the larger ID on a tie): the walk goes down from the nearest start and stops once no range can
// still reach the VA, which `longest_` bounds. Without overlaps the first range it looks at is the answer, or none is.

#include <cstddef>
#include <cstdint>
#include <map>
#include <utility>

namespace leonida {

template<typename Id>
class BufferVaIndex {
public:
    // Registers [va, va + size). A null VA or an empty range is not indexed (no lookup can land in it).
    void insert(std::uint64_t va, std::uint64_t size, Id id) {
        if (!va || !size) return;
        ranges_[{va, id}] = size;
        if (size > longest_) longest_ = size;
    }

    // Forgets what insert(va, size, id) registered; nothing when it was not indexed.
    void erase(std::uint64_t va, Id id) {
        ranges_.erase({va, id});
        if (ranges_.empty()) longest_ = 0;  // otherwise kept: a bound that is too large only lengthens a miss
    }

    struct Hit {
        Id id;
        std::uint64_t start;  // the range's first VA
        std::uint64_t size;
    };

    // The range that holds `va`; false when none does.
    bool find(std::uint64_t va, Hit& out) const {
        auto it = ranges_.upper_bound({va, max_id()});
        while (it != ranges_.begin()) {
            --it;
            const std::uint64_t start = it->first.first;
            const std::uint64_t offset = va - start;  // start <= va here
            if (offset >= longest_) return false;    // no range that starts this low reaches va
            if (offset < it->second) {
                out = {it->first.second, start, it->second};
                return true;
            }
        }
        return false;
    }

    std::size_t size() const { return ranges_.size(); }

private:
    static constexpr Id max_id() { return ~Id(0); }

    std::map<std::pair<std::uint64_t, Id>, std::uint64_t> ranges_;  // (first VA, id) -> bytes
    std::uint64_t longest_ = 0;  // no indexed range is longer
};

} // namespace leonida
