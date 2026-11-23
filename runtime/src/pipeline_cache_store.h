// What the runtime's VkPipelineCache keeps between sessions: the files it starts from and saves to, the checks that a
// file is whole and belongs to this GPU and driver, and the pipeline creation feedback
// (VK_EXT_pipeline_creation_feedback, core in Vulkan 1.3) that says whether the cache paid off.
//
// The build's pipelines stage writes pipeline_cache.bin beside gta6rt.exe: a seed, never written here. A session saves
// what it learned to a second file in the same folder, or to LEONIDA_PIPELINE_CACHE, and the next one starts from
// that, so it has every pipeline the game has already made.
#pragma once

#include "cached_pso.h"

#include <vulkan/vulkan_core.h>

#include <array>
#include <atomic>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <string>
#include <system_error>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/stat.h>
#endif

namespace leonida::pipeline_cache_store {

// A session cache file bigger than this is neither loaded nor saved (LEONIDA_PIPELINE_CACHE_MAX_MB overrides): saving
// reads the whole cache into memory next to the driver's copy and writes all of it, and the full-corpus seed is 3.8 GB
// (docs/design.md); a session that made a few pipelines should not write that at exit.
inline constexpr std::uint64_t kDefaultMaxSessionBytes = std::uint64_t{1} << 30;

// LEONIDA_PIPELINE_CACHE=off: no session file.
inline bool is_off(const std::wstring& value) {
    return value.size() == 3 && (value[0] | 0x20) == L'o' && (value[1] | 0x20) == L'f' && (value[2] | 0x20) == L'f';
}

inline constexpr const wchar_t* kSessionFileName = L"pipeline_cache.user.bin";

// A cache blob's VkPipelineCacheHeaderVersionOne names the GPU and the driver build it came from; a blob from another
// one would be ignored by the driver at best, so it is not even handed over.
inline bool header_matches(const void* data, std::size_t size, const cached_pso::Identity& identity) {
    static_assert(sizeof(VkPipelineCacheHeaderVersionOne) == 32);
    if (!data || size < sizeof(VkPipelineCacheHeaderVersionOne)) return false;
    VkPipelineCacheHeaderVersionOne header{};
    // The header is little-endian, like the host. The driver interprets the opaque payload.
    std::memcpy(&header, data, sizeof(header));
    return header.headerSize == sizeof(header) && header.headerVersion == VK_PIPELINE_CACHE_HEADER_VERSION_ONE &&
           header.vendorID == identity.vendor && header.deviceID == identity.device &&
           std::memcmp(header.pipelineCacheUUID, identity.cache_uuid, VK_UUID_SIZE) == 0;
}

// The session file is the driver's cache data behind a 32-byte frame: the driver wants back exactly the bytes it gave
// (VUID-VkPipelineCacheCreateInfo-initialDataSize-00768), and its own header says nothing about their count, so a
// file cut short, or with a flipped byte, would otherwise reach it with a matching header.
inline constexpr std::size_t kFrameBytes = 32;

inline std::uint64_t payload_hash(const void* data, std::size_t size) {
    const auto* bytes = static_cast<const unsigned char*>(data);
    std::uint64_t h = 0xcbf29ce484222325ull;
    std::size_t i = 0;
    for (; i + 8 <= size; i += 8) {
        std::uint64_t word;
        std::memcpy(&word, bytes + i, 8);
        h = (h ^ word) * 0x100000001b3ull;
        h ^= h >> 29;
    }
    for (; i < size; ++i) h = (h ^ bytes[i]) * 0x100000001b3ull;
    return h;
}

inline std::array<char, kFrameBytes> make_frame(const void* payload, std::size_t size) {
    std::array<char, kFrameBytes> frame{};
    const std::uint32_t version = 1;
    const std::uint64_t bytes = size, hash = payload_hash(payload, size);
    std::memcpy(frame.data(), "LPCS", 4);
    std::memcpy(frame.data() + 4, &version, 4);
    std::memcpy(frame.data() + 8, &bytes, 8);
    std::memcpy(frame.data() + 16, &hash, 8);
    return frame;
}

// The driver's bytes inside a whole session file, or null: the frame, the length (the header of the driver's data is
// 32 bytes at least) and the hash all have to agree.
inline const char* unframe(const char* file, std::size_t file_size, std::size_t& payload_size) {
    if (!file || file_size < kFrameBytes + sizeof(VkPipelineCacheHeaderVersionOne)) return nullptr;
    std::uint32_t version;
    std::uint64_t bytes, hash;
    std::memcpy(&version, file + 4, 4);
    std::memcpy(&bytes, file + 8, 8);
    std::memcpy(&hash, file + 16, 8);
    if (std::memcmp(file, "LPCS", 4) != 0 || version != 1 || bytes != file_size - kFrameBytes ||
        hash != payload_hash(file + kFrameBytes, std::size_t(bytes)))
        return nullptr;
    payload_size = std::size_t(bytes);
    return file + kFrameBytes;
}

// Which file a path names right now (volume and index on Windows, device and inode elsewhere), whatever it is called.
struct FileId {
    std::uint64_t volume = 0, index = 0;
    bool known = false;
    friend bool operator==(const FileId& a, const FileId& b) {
        return a.known && b.known && a.volume == b.volume && a.index == b.index;
    }
};

inline FileId file_id(const std::filesystem::path& path) {
    FileId id;
#ifdef _WIN32
    // Opened for attributes only, sharing everything, so it works on a file another process has open.
    const HANDLE file = CreateFileW(path.c_str(), FILE_READ_ATTRIBUTES, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                                    nullptr, OPEN_EXISTING, FILE_FLAG_BACKUP_SEMANTICS, nullptr);
    if (file == INVALID_HANDLE_VALUE) return id;
    BY_HANDLE_FILE_INFORMATION info{};
    if (GetFileInformationByHandle(file, &info)) {
        id.volume = info.dwVolumeSerialNumber;
        id.index = (std::uint64_t{info.nFileIndexHigh} << 32) | info.nFileIndexLow;
        id.known = true;
    }
    CloseHandle(file);
#else
    struct stat st{};
    if (stat(path.c_str(), &st) == 0) {
        id.volume = std::uint64_t(st.st_dev);
        id.index = std::uint64_t(st.st_ino);
        id.known = true;
    }
#endif
    return id;
}

// An absolute path with its directories resolved (links and junctions followed, `..` removed) as of now; the file
// itself need not exist. Taken once, when the device is made, so a later change of working directory or a retargeted
// link cannot move where the session saves. Unchanged if the filesystem will not say.
inline std::filesystem::path resolved(const std::filesystem::path& path) {
    namespace fs = std::filesystem;
    std::error_code ec;
    fs::path head = fs::absolute(path, ec);
    if (ec) return path;
    head = head.lexically_normal();
    fs::path tail;  // the part that does not exist yet
    while (head.has_relative_path() && !fs::exists(head, ec) && head != head.parent_path()) {
        tail = tail.empty() ? head.filename() : head.filename() / tail;
        head = head.parent_path();
    }
#ifdef _WIN32
    const HANDLE handle = CreateFileW(head.c_str(), FILE_READ_ATTRIBUTES,
                                      FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr, OPEN_EXISTING,
                                      FILE_FLAG_BACKUP_SEMANTICS, nullptr);
    if (handle != INVALID_HANDLE_VALUE) {
        std::wstring final_name(32768, L'\0');
        const DWORD length = GetFinalPathNameByHandleW(handle, final_name.data(), DWORD(final_name.size()),
                                                       FILE_NAME_NORMALIZED | VOLUME_NAME_DOS);
        CloseHandle(handle);
        if (length && length < final_name.size()) {
            final_name.resize(length);
            if (final_name.rfind(L"\\\\?\\UNC\\", 0) == 0) final_name = L"\\\\" + final_name.substr(8);
            else if (final_name.rfind(L"\\\\?\\", 0) == 0) final_name = final_name.substr(4);
            head = final_name;
        }
    }
#else
    if (const fs::path real = fs::canonical(head, ec); !ec) head = real;
#endif
    return tail.empty() ? head : head / tail;
}

// True when both paths reach the same file, now: the same file on disk, or (while one does not exist yet) the same
// resolved name.
inline bool same_file(const std::filesystem::path& a, const std::filesystem::path& b) {
    if (file_id(a) == file_id(b)) return true;
    std::wstring left = resolved(a).wstring(), right = resolved(b).wstring();
    for (std::wstring* name : {&left, &right})
        for (wchar_t& c : *name) c = wchar_t(c < 128 ? std::tolower(int(c)) : c);
    return left == right;
}

// A file as it was when it was read: whether there was one, and its time and size.
struct Stamp {
    bool exists = false;
    std::filesystem::file_time_type time{};
    std::uintmax_t size = 0;
    friend bool operator==(const Stamp& a, const Stamp& b) {
        return a.exists == b.exists && (!a.exists || (a.time == b.time && a.size == b.size));
    }
};

inline Stamp stamp_of(const std::filesystem::path& path) {
    Stamp stamp;
    std::error_code ec;
    stamp.time = std::filesystem::last_write_time(path, ec);
    if (ec) return {};
    stamp.size = std::filesystem::file_size(path, ec);
    if (ec) return {};
    stamp.exists = true;
    return stamp;
}

// The session file starts a session only when it is at least as new as the seed: a rebuilt seed holds pipelines the
// older session file lacks, and the next save then replaces the session file with a superset.
inline bool session_file_is_current(const std::filesystem::path& session, const std::filesystem::path& seed) {
    const Stamp session_stamp = stamp_of(session), seed_stamp = stamp_of(seed);
    if (!session_stamp.exists) return false;
    return !seed_stamp.exists || session_stamp.time >= seed_stamp.time;  // no seed: nothing to be older than
}

// What the pipelines of one device took to create, from VK_EXT_pipeline_creation_feedback.
struct Stats {
    std::atomic<std::uint32_t> created{0};     // every pipeline made, whatever its feedback says
    std::atomic<std::uint32_t> pipelines{0};   // of those, with valid feedback
    std::atomic<std::uint32_t> cache_hits{0};  // of those, found in the application's VkPipelineCache
    std::atomic<std::uint64_t> nanoseconds{0}; // their total creation time
};

// Whether a session has anything to save. With feedback, that is a pipeline that was not found in the cache; one whose
// feedback is missing or not valid counts as not found, because the cache cannot be shown to have had it. Without
// feedback (before Vulkan 1.3) it is a cache that grew since it was loaded.
inline bool learned_something(bool feedback, std::uint32_t created, std::uint32_t cache_hits, std::size_t cache_bytes,
                              std::size_t loaded_bytes) {
    return feedback ? created > cache_hits : cache_bytes > loaded_bytes;
}

// Chained into one vkCreate*Pipelines call; a no-op when the device has no feedback (before Vulkan 1.3 and without the
// extension enabled, the struct must not be chained).
class CreationFeedback {
public:
    CreationFeedback(bool enabled, std::uint32_t stage_count)
        : enabled_(enabled), stages_(enabled ? stage_count : 0) {
        info_.pPipelineCreationFeedback = &pipeline_;
        info_.pipelineStageCreationFeedbackCount = std::uint32_t(stages_.size());
        info_.pPipelineStageCreationFeedbacks = stages_.data();
    }

    // The pNext for the create info: this struct, then `next`.
    const void* link(const void* next) {
        if (!enabled_) return next;
        info_.pNext = next;
        return &info_;
    }

    // Call after the pipeline was created.
    void record(Stats& stats) const {
        stats.created.fetch_add(1, std::memory_order_relaxed);
        if (!enabled_ || !(pipeline_.flags & VK_PIPELINE_CREATION_FEEDBACK_VALID_BIT)) return;
        stats.pipelines.fetch_add(1, std::memory_order_relaxed);
        stats.nanoseconds.fetch_add(pipeline_.duration, std::memory_order_relaxed);
        if (pipeline_.flags & VK_PIPELINE_CREATION_FEEDBACK_APPLICATION_PIPELINE_CACHE_HIT_BIT)
            stats.cache_hits.fetch_add(1, std::memory_order_relaxed);
    }

private:
    bool enabled_;
    VkPipelineCreationFeedback pipeline_{};
    std::vector<VkPipelineCreationFeedback> stages_;
    VkPipelineCreationFeedbackCreateInfo info_{VK_STRUCTURE_TYPE_PIPELINE_CREATION_FEEDBACK_CREATE_INFO};
};

}  // namespace leonida::pipeline_cache_store
