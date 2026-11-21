// Loads Vulkan at run time. The macOS test tools default to MoltenVK.
#pragma once

#ifndef VK_NO_PROTOTYPES
#define VK_NO_PROTOTYPES
#endif
#include <vulkan/vulkan.h>

#include <cstdio>
#include <cstdlib>

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
#elif defined(__APPLE__)
    static const auto entry = []() -> void* {
        // The default is direct MoltenVK: a portability driver would otherwise require instance opt-in.
        const char* requested = std::getenv("LEONIDA_VULKAN_LOADER");
        const bool overridden = requested && *requested;
        const char* name = overridden ? requested : "libMoltenVK.dylib";
        auto library = dlopen(name, RTLD_NOW | RTLD_LOCAL);
        if (!library && !overridden) {
            name = "/opt/homebrew/lib/libMoltenVK.dylib";
            library = dlopen(name, RTLD_NOW | RTLD_LOCAL);
        }
        if (!library) {
            std::fprintf(stderr, "could not load %s: %s\n", name, dlerror());
            return nullptr;
        }
        const auto symbol = dlsym(library, "vkGetInstanceProcAddr");
        if (!symbol) std::fprintf(stderr, "%s has no vkGetInstanceProcAddr\n", name);
        return symbol;
    }();
#else
    constexpr auto library_name = "libvulkan.so.1";
    static const auto library = dlopen(library_name, RTLD_NOW | RTLD_LOCAL);
    const auto entry = library ? dlsym(library, "vkGetInstanceProcAddr") : nullptr;
#endif
#ifndef __APPLE__
    if (!entry) std::fprintf(stderr, "could not load vkGetInstanceProcAddr from %s\n", library_name);
#endif
    return reinterpret_cast<PFN_vkGetInstanceProcAddr>(entry);
}
