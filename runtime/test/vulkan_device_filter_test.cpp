// Unit tests for LEONIDA_VK_DEVICE name matching (runtime/src/vulkan_device_filter.h). No Vulkan instance and no GPU:
// the filter is a pure function of the environment and a device name.
#include "vulkan_device_filter.h"

#include <cstdio>
#include <cstdlib>
#include <string>

namespace {

int failures = 0;

void expect(bool ok, const char* what) {
    if (ok) return;
    std::fprintf(stderr, "FAIL %s\n", what);
    ++failures;
}

void set_filter(const char* value) {
#ifdef _WIN32
    _putenv_s("LEONIDA_VK_DEVICE", value);
#else
    if (value[0] == '\0') {
        unsetenv("LEONIDA_VK_DEVICE");
    } else {
        setenv("LEONIDA_VK_DEVICE", value, 1);
    }
#endif
}

}  // namespace

int main() {
    using leonida::vulkan_device_filter;
    using leonida::vulkan_device_wanted;

    const char* saved = std::getenv("LEONIDA_VK_DEVICE");
    const std::string saved_value = saved ? saved : "";

    // Unset (or empty) takes every device, including a null name.
    set_filter("");
    expect(vulkan_device_filter().empty(), "empty filter");
    expect(vulkan_device_wanted("AMD Radeon RX 7900 XTX"), "unset matches a real name");
    expect(vulkan_device_wanted(""), "unset matches an empty name");
    expect(vulkan_device_wanted(nullptr), "unset matches a null name");

    // A filter is a case-insensitive substring of the device name.
    set_filter("RADEON");
    expect(vulkan_device_filter() == "radeon", "filter is lower-cased");
    expect(vulkan_device_wanted("AMD Radeon RX 7900 XTX"), "case-insensitive match");
    expect(vulkan_device_wanted("amd radeon graphics"), "lower-case name matches");
    expect(!vulkan_device_wanted("NVIDIA GeForce RTX 5070 Ti"), "different vendor does not match");

    set_filter("rtx 5070");
    expect(vulkan_device_wanted("NVIDIA GeForce RTX 5070 Ti"), "substring with a space matches");
    expect(!vulkan_device_wanted("NVIDIA GeForce RTX 4070 Ti"), "other model does not match");

    set_filter("7900");
    expect(vulkan_device_wanted("AMD Radeon RX 7900 XTX"), "model number matches");
    expect(!vulkan_device_wanted(nullptr), "a filter never matches a null name");
    expect(!vulkan_device_wanted(""), "a filter never matches an empty name");

    // Do not leak the test's value into anything that runs later in this process.
#ifdef _WIN32
    _putenv_s("LEONIDA_VK_DEVICE", saved_value.c_str());
#else
    if (saved_value.empty()) {
        unsetenv("LEONIDA_VK_DEVICE");
    } else {
        setenv("LEONIDA_VK_DEVICE", saved_value.c_str(), 1);
    }
#endif

    if (failures != 0) {
        std::fprintf(stderr, "%d failure(s)\n", failures);
        return 1;
    }
    std::printf("vulkan_device_filter: all checks passed\n");
    return 0;
}
