#pragma once

// The descriptor hash table for set 0, binding 2 (docs/d3d12_vulkan.md section 5.1 and docs/design.md):
// Shaders look up bindless resource/sampler indices by FNV-1a hashing the 8-dword descriptor key (T# or padded S#)
// and probing up to 8 slots in this table. Miss or fallback yields index 0.

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <span>
#include <vector>

namespace leonida {

// FNV-1a constants for 32-bit dwords
inline constexpr std::uint32_t kFnvOffsetBasis = 2166136261u;
inline constexpr std::uint32_t kFnvPrime = 16777619u;
inline constexpr std::uint32_t kDescriptorProbes = 8;
inline constexpr std::uint32_t kSlotOccupiedMarker = 0xA11CA7EDu;

// Key types:
// T# key is 8 dwords.
// S# key is 4 dwords followed by 4x 0xFFFFFFFF.
using DescriptorKey = std::array<std::uint32_t, 8>;

inline std::uint32_t hash_descriptor_key(const std::uint32_t key[8]) {
    std::uint32_t h = kFnvOffsetBasis;
    for (int i = 0; i < 8; ++i) {
        h = (h ^ key[i]) * kFnvPrime;
    }
    return h;
}

inline std::uint32_t hash_descriptor_key(const DescriptorKey& key) {
    return hash_descriptor_key(key.data());
}

inline DescriptorKey make_tsharp_key(const std::uint32_t tsharp[8]) {
    DescriptorKey k{};
    std::memcpy(k.data(), tsharp, sizeof(DescriptorKey));
    return k;
}

inline DescriptorKey make_ssharp_key(const std::uint32_t ssharp[4]) {
    DescriptorKey k{};
    k[0] = ssharp[0];
    k[1] = ssharp[1];
    k[2] = ssharp[2];
    k[3] = ssharp[3];
    k[4] = 0xFFFFFFFFu;
    k[5] = 0xFFFFFFFFu;
    k[6] = 0xFFFFFFFFu;
    k[7] = 0xFFFFFFFFu;
    return k;
}

// Memory layout of each entry in the table: 10 uint32s = 40 bytes.
struct DescriptorEntry {
    std::uint32_t key[8] = {};
    std::uint32_t index = 0;
    std::uint32_t pad = 0; // 0 when empty, kSlotOccupiedMarker when occupied
};
static_assert(sizeof(DescriptorEntry) == 40);

class DescriptorTable {
public:
    explicit DescriptorTable(std::uint32_t initial_capacity = 16) {
        std::uint32_t cap = 16;
        while (cap < initial_capacity) cap <<= 1;
        mask_ = cap - 1;
        entries_.resize(cap);
    }

    // Inserts or updates a key -> index mapping.
    // If the key already exists in one of the 8 probe slots, its index is updated.
    // If insertion would exceed 8 probes or 50% load, the table grows and re-hashes.
    bool insert(const std::uint32_t key[8], std::uint32_t index) {
        std::lock_guard lock(mutex_);
        return insert_locked(key, index);
    }

    bool insert(const DescriptorKey& key, std::uint32_t index) {
        return insert(key.data(), index);
    }

    // Queries the index associated with key. Returns 0 if not found.
    std::uint32_t lookup(const std::uint32_t key[8]) const {
        std::lock_guard lock(mutex_);
        const std::uint32_t h = hash_descriptor_key(key);
        for (std::uint32_t k = 0; k < kDescriptorProbes; ++k) {
            const std::uint32_t slot = (h + k) & mask_;
            const auto& entry = entries_[slot];
            if (entry.pad == kSlotOccupiedMarker && std::memcmp(entry.key, key, sizeof(entry.key)) == 0) {
                return entry.index;
            }
            if (entry.pad != kSlotOccupiedMarker) {
                return 0; // Empty slot encountered: key is definitely absent
            }
        }
        return 0; // Exceeded 8 probes without a match
    }

    std::uint32_t lookup(const DescriptorKey& key) const {
        return lookup(key.data());
    }

    bool contains(const std::uint32_t key[8]) const {
        std::lock_guard lock(mutex_);
        const std::uint32_t h = hash_descriptor_key(key);
        for (std::uint32_t k = 0; k < kDescriptorProbes; ++k) {
            const std::uint32_t slot = (h + k) & mask_;
            const auto& entry = entries_[slot];
            if (entry.pad == kSlotOccupiedMarker && std::memcmp(entry.key, key, sizeof(entry.key)) == 0) {
                return true;
            }
            if (entry.pad != kSlotOccupiedMarker) return false;
        }
        return false;
    }

    bool contains(const DescriptorKey& key) const {
        return contains(key.data());
    }

    void clear() {
        std::lock_guard lock(mutex_);
        std::fill(entries_.begin(), entries_.end(), DescriptorEntry{});
        occupied_count_ = 0;
    }

    std::size_t size() const {
        std::lock_guard lock(mutex_);
        return occupied_count_;
    }

    std::size_t capacity() const {
        std::lock_guard lock(mutex_);
        return entries_.size();
    }

    std::uint32_t mask() const {
        std::lock_guard lock(mutex_);
        return mask_;
    }

    std::size_t size_dwords() const {
        std::lock_guard lock(mutex_);
        return 1 + entries_.size() * 10;
    }

    std::size_t size_bytes() const {
        return size_dwords() * sizeof(std::uint32_t);
    }

    // Exports the table buffer format consumed by SPIR-V set 0, binding 2.
    // Word 0 is mask; followed by entries[mask + 1] of 10 dwords: key[8], index, pad.
    std::vector<std::uint32_t> build_gpu_buffer(bool clear_pad = false) const {
        std::lock_guard lock(mutex_);
        std::vector<std::uint32_t> out(1 + entries_.size() * 10, 0);
        out[0] = mask_;
        for (std::size_t s = 0; s < entries_.size(); ++s) {
            const std::size_t base = 1 + s * 10;
            std::memcpy(&out[base], entries_[s].key, sizeof(entries_[s].key));
            out[base + 8] = entries_[s].index;
            out[base + 9] = clear_pad ? 0 : entries_[s].pad;
        }
        return out;
    }

    void write_to_buffer(void* dest, bool clear_pad = false) const {
        if (!dest) return;
        const auto buf = build_gpu_buffer(clear_pad);
        std::memcpy(dest, buf.data(), buf.size() * sizeof(std::uint32_t));
    }

private:
    bool insert_locked(const std::uint32_t key[8], std::uint32_t index) {
        while (true) {
            const std::uint32_t h = hash_descriptor_key(key);
            int empty_slot = -1;

            for (std::uint32_t k = 0; k < kDescriptorProbes; ++k) {
                const std::uint32_t slot = (h + k) & mask_;
                auto& entry = entries_[slot];
                if (entry.pad == kSlotOccupiedMarker) {
                    if (std::memcmp(entry.key, key, sizeof(entry.key)) == 0) {
                        entry.index = index;
                        return true;
                    }
                } else if (empty_slot < 0) {
                    empty_slot = static_cast<int>(slot);
                }
            }

            // Check if we can place in an empty slot without exceeding 50% load
            if (empty_slot >= 0 && (occupied_count_ + 1) * 2 <= entries_.size()) {
                auto& entry = entries_[empty_slot];
                std::memcpy(entry.key, key, sizeof(entry.key));
                entry.index = index;
                entry.pad = kSlotOccupiedMarker;
                ++occupied_count_;
                return true;
            }

            // Table is full (>50% load) or no slot found in 8 probes: grow and retry
            grow_locked();
        }
    }

    void grow_locked() {
        std::uint32_t new_cap = static_cast<std::uint32_t>(entries_.size() * 2);
        while (true) {
            std::vector<DescriptorEntry> new_entries(new_cap);
            const std::uint32_t new_mask = new_cap - 1;
            bool rehash_ok = true;

            for (const auto& old : entries_) {
                if (old.pad != kSlotOccupiedMarker) continue;
                const std::uint32_t h = hash_descriptor_key(old.key);
                bool placed = false;
                for (std::uint32_t k = 0; k < kDescriptorProbes; ++k) {
                    const std::uint32_t s = (h + k) & new_mask;
                    if (new_entries[s].pad != kSlotOccupiedMarker) {
                        new_entries[s] = old;
                        placed = true;
                        break;
                    }
                }
                if (!placed) {
                    rehash_ok = false;
                    break;
                }
            }

            if (rehash_ok) {
                entries_ = std::move(new_entries);
                mask_ = new_mask;
                break;
            }
            new_cap *= 2;
        }
    }

    mutable std::mutex mutex_;
    std::uint32_t mask_ = 15;
    std::vector<DescriptorEntry> entries_;
    std::size_t occupied_count_ = 0;
};

} // namespace leonida
