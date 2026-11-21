// Loads the Vulkan loader at run time: vulkan-1.dll on Windows, libvulkan.so.1 elsewhere. No import library.
#pragma once
#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>

#ifdef _WIN32
#include <windows.h>
inline PFN_vkGetInstanceProcAddr load_vulkan() {
    HMODULE lib = LoadLibraryA("vulkan-1.dll");
    return lib ? reinterpret_cast<PFN_vkGetInstanceProcAddr>(GetProcAddress(lib, "vkGetInstanceProcAddr")) : nullptr;
}
#else
#include <dlfcn.h>
inline PFN_vkGetInstanceProcAddr load_vulkan() {
    void* lib = dlopen("libvulkan.so.1", RTLD_NOW);
    return lib ? reinterpret_cast<PFN_vkGetInstanceProcAddr>(dlsym(lib, "vkGetInstanceProcAddr")) : nullptr;
}
#endif
