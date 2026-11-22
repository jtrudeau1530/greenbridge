#pragma once

// LEONIDA_VK_DEVICE: which Vulkan GPU the runtime and its tests take on a machine with more than one. The value is
// part of the device name, compared without case ("5070", "radeon"); unset or empty takes every device, and the
// caller then picks as before (the last discrete GPU, else the first that fits). Several GPUs of one vendor share
// one driver, so the loader's own driver filters cannot tell them apart. A value that matches no device must fail
// the caller, not fall back, so a typo cannot silently test another GPU (as LEONIDA_HOST_ADAPTER does).

#include <cctype>
#include <cstdlib>
#include <string>

namespace leonida {

// The filter, lower-cased; empty when unset.
inline std::string vulkan_device_filter() {
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4996)  // getenv: read at device selection only
#endif
    const char* value = std::getenv("LEONIDA_VK_DEVICE");
#ifdef _MSC_VER
#pragma warning(pop)
#endif
    std::string out = value ? value : "";
    for (char& c : out) c = char(std::tolower(static_cast<unsigned char>(c)));
    return out;
}

// True when `device_name` (VkPhysicalDeviceProperties::deviceName) is wanted.
inline bool vulkan_device_wanted(const char* device_name) {
    const std::string filter = vulkan_device_filter();
    if (filter.empty()) return true;
    std::string name = device_name ? device_name : "";
    for (char& c : name) c = char(std::tolower(static_cast<unsigned char>(c)));
    return name.find(filter) != std::string::npos;
}

} // namespace leonida
