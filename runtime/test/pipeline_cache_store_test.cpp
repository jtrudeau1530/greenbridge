// The parts of the session cache that need no GPU: the frame that tells a whole file from a cut one, whether a session
// has anything to save, and which names reach the same file.
#include "../src/pipeline_cache_store.h"

#include <chrono>
#include <cstdio>
#include <fstream>
#include <string>

namespace fs = std::filesystem;
namespace store = leonida::pipeline_cache_store;

#define CHECK(condition) do {                                        \
    if (!(condition)) {                                              \
        std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); \
        return 1;                                                    \
    }                                                                \
} while (0)

static std::string driver_data() {
    VkPipelineCacheHeaderVersionOne header{};
    header.headerSize = sizeof(header);
    header.headerVersion = VK_PIPELINE_CACHE_HEADER_VERSION_ONE;
    header.vendorID = 0x10de;
    header.deviceID = 0x2684;
    std::string data(reinterpret_cast<const char*>(&header), sizeof(header));
    return data + "the driver's own pipelines, however many";
}

static std::string framed(const std::string& payload) {
    const auto frame = store::make_frame(payload.data(), payload.size());
    return std::string(frame.data(), frame.size()) + payload;
}

static int test_frame() {
    const std::string payload = driver_data();
    const std::string file = framed(payload);
    std::size_t size = 0;
    const char* got = store::unframe(file.data(), file.size(), size);
    CHECK(got && size == payload.size() && std::string(got, size) == payload);

    // Whatever is missing, added or changed, the driver is not handed the bytes.
    CHECK(!store::unframe(file.data(), file.size() - 1, size));
    CHECK(!store::unframe(file.data(), store::kFrameBytes + sizeof(VkPipelineCacheHeaderVersionOne) + 1, size));
    CHECK(!store::unframe(file.data(), store::kFrameBytes, size));
    CHECK(!store::unframe(payload.data(), payload.size(), size));  // a raw cache is not a session file
    std::string longer = file + "x";
    CHECK(!store::unframe(longer.data(), longer.size(), size));
    for (const std::size_t at : {std::size_t(0), store::kFrameBytes + 3, file.size() / 2, file.size() - 1}) {
        std::string changed = file;
        changed[at] ^= 1;
        CHECK(!store::unframe(changed.data(), changed.size(), size));
    }
    leonida::cached_pso::Identity identity;
    identity.vendor = 0x10de;
    identity.device = 0x2684;
    CHECK(store::header_matches(payload.data(), payload.size(), identity));
    CHECK(!store::header_matches(payload.data(), 31, identity));
    identity.device = 1;
    CHECK(!store::header_matches(payload.data(), payload.size(), identity));
    return 0;
}

static int test_learned() {
    // With feedback: a pipeline the cache did not have, or whose feedback says nothing, is something to save.
    CHECK(!store::learned_something(true, 5, 5, 100, 100));
    CHECK(store::learned_something(true, 5, 4, 100, 100));
    CHECK(store::learned_something(true, 1, 0, 100, 100));
    CHECK(!store::learned_something(true, 0, 0, 200, 100));
    // A driver that clears the valid bit on a successful creation still counts the creation.
    store::Stats stats;
    store::CreationFeedback feedback(true, 1);
    feedback.link(nullptr);
    feedback.record(stats);
    CHECK(stats.created == 1 && stats.pipelines == 0 && stats.cache_hits == 0);
    CHECK(store::learned_something(true, stats.created, stats.cache_hits, 100, 100));
    // Without feedback only growth tells.
    CHECK(store::learned_something(false, 3, 0, 101, 100));
    CHECK(!store::learned_something(false, 3, 0, 100, 100));
    return 0;
}

static int test_names(const fs::path& dir) {
    const auto a = dir / "a.bin";
    std::ofstream(a, std::ios::binary) << "a";
    std::ofstream(dir / "b.bin", std::ios::binary) << "b";
    fs::create_directory(dir / "sub");
    CHECK(store::same_file(a, a));
    CHECK(store::same_file(a, dir / "sub" / ".." / "a.bin"));
    CHECK(!store::same_file(a, dir / "b.bin"));
    CHECK(store::same_file(dir / "later.bin", dir / "sub" / ".." / "later.bin"));  // neither exists yet
    CHECK(!store::same_file(dir / "later.bin", dir / "other.bin"));
    std::error_code ec;
    fs::create_hard_link(a, dir / "alias.bin", ec);
    if (!ec) CHECK(store::same_file(a, dir / "alias.bin"));
    // A relative name is resolved against the working directory as it is now.
    const fs::path before = fs::current_path();
    fs::current_path(dir);
    const fs::path relative = store::resolved("a.bin");
    fs::current_path(dir / "sub");
    CHECK(relative.is_absolute() && store::same_file(relative, a));
    fs::current_path(before);
    const store::Stamp stamp = store::stamp_of(a);
    CHECK(stamp.exists && stamp == store::stamp_of(a));
    std::ofstream(a, std::ios::binary | std::ios::app) << "more";
    CHECK(!(stamp == store::stamp_of(a)));
    CHECK(!store::stamp_of(dir / "none").exists);
    return 0;
}

int main() {
    const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
    const auto dir = fs::temp_directory_path() / ("leonida_pipeline_cache_store_" + std::to_string(stamp));
    if (!fs::create_directory(dir)) return 1;
    const int result = test_frame() || test_learned() || test_names(dir);
    std::error_code ec;
    fs::remove_all(dir, ec);
    if (!result) std::printf("pipeline cache store: ok\n");
    return result;
}
