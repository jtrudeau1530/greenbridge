// Script the Vulkan size and data calls without loading a driver.
#include "pipeline_cache_snapshot.h"

#include <cstdio>
#include <cstring>

#define CHECK(condition) do {                                            \
    if (!(condition)) {                                                  \
        std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition);      \
        return 1;                                                        \
    }                                                                    \
} while (0)

int main() {
    int queries = 0, reads = 0;
    const auto growing = [&](size_t* size, void* data) {
        if (!data) {
            *size = (++queries == 1) ? 32 : 64;
            return VK_SUCCESS;
        }
        if (++reads == 1) {
            std::memset(data, 'p', *size);
            return VK_INCOMPLETE;
        }
        if (*size != 64) return VK_ERROR_UNKNOWN;
        *size = 48;  // A successful read can return less than the size query.
        std::memset(data, 's', *size);
        return VK_SUCCESS;
    };
    CHECK(pipeline_cache_snapshot(growing) == std::vector<char>(48, 's'));
    CHECK(queries == 2 && reads == 2);

    auto rejected = [](VkResult query_result, size_t available, VkResult read_result, size_t written) {
        int calls = 0;
        auto get = [&](size_t* size, void* data) {
            ++calls;
            *size = data ? written : available;
            return data ? read_result : query_result;
        };
        try {
            pipeline_cache_snapshot(get);
        } catch (const std::exception&) {
            return calls;
        }
        return 0;
    };
    CHECK(rejected(VK_ERROR_DEVICE_LOST, 32, VK_SUCCESS, 32) == 1);
    CHECK(rejected(VK_SUCCESS, 0, VK_SUCCESS, 0) == 1);
    CHECK(rejected(VK_SUCCESS, 16, VK_SUCCESS, 16) == 1);
    CHECK(rejected(VK_SUCCESS, 32, VK_ERROR_DEVICE_LOST, 32) == 2);
    CHECK(rejected(VK_SUCCESS, 32, VK_SUCCESS, 0) == 2);
    CHECK(rejected(VK_SUCCESS, 32, VK_SUCCESS, 16) == 2);
    CHECK(rejected(VK_SUCCESS, 32, VK_SUCCESS, 33) == 2);
    CHECK(rejected(VK_SUCCESS, 32, VK_INCOMPLETE, 32) == 8);
    return 0;
}
