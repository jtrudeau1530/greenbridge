// Loads the Vulkan loader at run time: vulkan-1.dll on Windows, MoltenVK on macOS, libvulkan.so.1 elsewhere. No import library.
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
#elif defined(__APPLE__)
    // MoltenVK directly: through the loader it is a portability driver, which every vkCreateInstance here would
    // have to opt into. The Vulkan SDK installs it in /usr/local/lib (a default search path), Homebrew does not.
    constexpr auto library_name = "libMoltenVK.dylib";
    static const auto library = [] {
        auto handle = dlopen(library_name, RTLD_NOW | RTLD_LOCAL);
        return handle ? handle : dlopen("/opt/homebrew/lib/libMoltenVK.dylib", RTLD_NOW | RTLD_LOCAL);
    }();
    const auto entry = library ? dlsym(library, "vkGetInstanceProcAddr") : nullptr;
#else
    constexpr auto library_name = "libvulkan.so.1";
    static const auto library = dlopen(library_name, RTLD_NOW | RTLD_LOCAL);
    const auto entry = library ? dlsym(library, "vkGetInstanceProcAddr") : nullptr;
#endif
    if (!entry) std::fprintf(stderr, "could not load vkGetInstanceProcAddr from %s\n", library_name);
    return reinterpret_cast<PFN_vkGetInstanceProcAddr>(entry);
}
