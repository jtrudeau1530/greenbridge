// VK_KHR_shader_fma for the test programs: translated shaders built with `rdna2spirv.py --fma khr` use OpFmaKHR
// (capability FMAKHR, SPV_KHR_fma), so the device must enable the extension and the shaderFmaFloat32 feature before
// the module can be created. A test program that creates a device calls query(), link_after() and extension() when
// supported(), and refuses a module that needs the capability on a device without it (exit 77, like every other
// missing device feature).
#pragma once

#include <vulkan/vulkan.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace leonida::vk_test {

#ifdef VK_KHR_SHADER_FMA_EXTENSION_NAME
inline constexpr bool shader_fma_in_headers = true;
#else
inline constexpr bool shader_fma_in_headers = false;
#endif

// OpCapability (17) FMAKHR (6030; spirv/unified1/spirv.h).
inline bool spirv_needs_fma(const void* data, std::size_t bytes) {
    if (bytes < 20 || bytes % 4) return false;
    std::vector<std::uint32_t> words(bytes / 4);
    std::memcpy(words.data(), data, bytes);
    for (std::size_t i = 5; i < words.size() && (words[i] >> 16);) {
        const std::uint32_t op = words[i] & 0xffff, count = words[i] >> 16;
        if (op == 17 && count == 2 && i + 1 < words.size() && words[i + 1] == 6030) return true;
        i += count;
    }
    return false;
}

struct ShaderFma {
#ifdef VK_KHR_SHADER_FMA_EXTENSION_NAME
    VkPhysicalDeviceShaderFmaFeaturesKHR features{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FMA_FEATURES_KHR};
#endif
    bool float32 = false;  // the device lists the extension and reports shaderFmaFloat32

    bool supported() const { return float32; }

    // Asks the physical device. Enables exactly what it reports (shaderFmaFloat16/32/64), so features never
    // fail device creation.
    void query(PFN_vkGetInstanceProcAddr gipa, VkInstance inst, VkPhysicalDevice pd) {
#ifdef VK_KHR_SHADER_FMA_EXTENSION_NAME
        const auto enumerate = reinterpret_cast<PFN_vkEnumerateDeviceExtensionProperties>(
            gipa(inst, "vkEnumerateDeviceExtensionProperties"));
        const auto features2 = reinterpret_cast<PFN_vkGetPhysicalDeviceFeatures2>(
            gipa(inst, "vkGetPhysicalDeviceFeatures2"));
        if (!enumerate || !features2) return;
        std::uint32_t count = 0;
        enumerate(pd, nullptr, &count, nullptr);
        std::vector<VkExtensionProperties> extensions(count);
        enumerate(pd, nullptr, &count, extensions.data());
        bool listed = false;
        for (const auto& e : extensions)
            if (!std::strcmp(e.extensionName, VK_KHR_SHADER_FMA_EXTENSION_NAME)) listed = true;
        if (!listed) return;
        features = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FMA_FEATURES_KHR};
        VkPhysicalDeviceFeatures2 query{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &features};
        features2(pd, &query);
        features.pNext = nullptr;
        float32 = features.shaderFmaFloat32 == VK_TRUE;
#else
        (void)gipa; (void)inst; (void)pd;
#endif
    }

    // Puts the features structure into the device pNext chain right after `base` (any structure in the chain).
    void link_after(void* base) {
#ifdef VK_KHR_SHADER_FMA_EXTENSION_NAME
        if (!float32) return;
        auto* b = static_cast<VkBaseOutStructure*>(base);
        features.pNext = b->pNext;
        b->pNext = reinterpret_cast<VkBaseOutStructure*>(&features);
#else
        (void)base;
#endif
    }

    // The extension name to add to ppEnabledExtensionNames when supported().
    static const char* extension() {
#ifdef VK_KHR_SHADER_FMA_EXTENSION_NAME
        return VK_KHR_SHADER_FMA_EXTENSION_NAME;
#else
        return "VK_KHR_shader_fma";
#endif
    }

    // True when the module can be used: it does not need FMAKHR, or the device enables shaderFmaFloat32.
    // Otherwise says so and the caller returns 77.
    bool accepts(const void* code, std::size_t bytes, const char* name) const {
        if (float32 || !spirv_needs_fma(code, bytes)) return true;
        std::fprintf(stderr, "unsupported: %s needs OpFmaKHR (VK_KHR_shader_fma, shaderFmaFloat32) and the device "
                             "lacks it; translate with --fma glsl\n", name);
        return false;
    }
};

}  // namespace leonida::vk_test
