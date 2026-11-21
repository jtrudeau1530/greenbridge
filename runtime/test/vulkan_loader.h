// Loads the Vulkan loader at run time: vulkan-1.dll on Windows, libvulkan.so.1 elsewhere. No import library.
#pragma once

#ifndef VK_NO_PROTOTYPES
#define VK_NO_PROTOTYPES
#endif
#include <vulkan/vulkan.h>

#include <cstdio>

#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

inline PFN_vkGetInstanceProcAddr load_vulkan() {
    // Keep the Vulkan library loaded for the lifetime of the tool.
#ifdef _WIN32
    constexpr auto library_name = "vulkan-1.dll";
    static const auto library = LoadLibraryA(library_name);
    const auto entry = library ? GetProcAddress(library, "vkGetInstanceProcAddr") : nullptr;
#else
    constexpr auto library_name = "libvulkan.so.1";
    static const auto library = dlopen(library_name, RTLD_NOW | RTLD_LOCAL);
    const auto entry = library ? dlsym(library, "vkGetInstanceProcAddr") : nullptr;
#endif
    if (!entry) std::fprintf(stderr, "could not load vkGetInstanceProcAddr from %s\n", library_name);
    return reinterpret_cast<PFN_vkGetInstanceProcAddr>(entry);
}
