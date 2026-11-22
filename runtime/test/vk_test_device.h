// Shared setup for the GPU tests of the D3D12 layer: a Vulkan 1.3 device with dynamicRendering, synchronization2,
// timelineSemaphore and separateDepthStencilLayouts, images, buffers and one-shot submissions.
// Set LEONIDA_VK_VALIDATION=1 to run under VK_LAYER_KHRONOS_validation; Device::validation_errors() counts errors.
#pragma once

#include "vulkan_loader.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <utility>
#include <vector>

namespace vktest {

inline unsigned& validation_error_count() {
    static unsigned count = 0;
    return count;
}

inline VKAPI_ATTR VkBool32 VKAPI_CALL on_validation(VkDebugUtilsMessageSeverityFlagBitsEXT severity,
                                                    VkDebugUtilsMessageTypeFlagsEXT,
                                                    const VkDebugUtilsMessengerCallbackDataEXT* data, void*) {
    if (severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) ++validation_error_count();
    std::fprintf(stderr, "validation: %s\n", data && data->pMessage ? data->pMessage : "(no message)");
    return VK_FALSE;
}

class Device {
public:
    PFN_vkGetInstanceProcAddr instance_proc = nullptr;
    PFN_vkGetDeviceProcAddr device_proc = nullptr;
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice physical = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue queue = VK_NULL_HANDLE;
    std::uint32_t family = 0;
    bool validation = false;
    bool bc = false;  // textureCompressionBC enabled
    bool bindless = false;         // the features the bindless set 0 needs are enabled
    bool partially_bound = false;  // and its slots can stay empty
    bool conditional_rendering = false;  // VK_EXT_conditional_rendering enabled, where available

    Device() = default;
    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;

    ~Device() {
        if (device) {
            fn<PFN_vkDeviceWaitIdle>("vkDeviceWaitIdle")(device);
            const auto destroy_image = fn<PFN_vkDestroyImage>("vkDestroyImage");
            const auto destroy_buffer = fn<PFN_vkDestroyBuffer>("vkDestroyBuffer");
            const auto free_memory = fn<PFN_vkFreeMemory>("vkFreeMemory");
            for (auto [image, memory] : images_) {
                destroy_image(device, image, nullptr);
                free_memory(device, memory, nullptr);
            }
            for (auto [buffer, memory] : buffers_) {
                destroy_buffer(device, buffer, nullptr);
                free_memory(device, memory, nullptr);
            }
            if (pool_) fn<PFN_vkDestroyCommandPool>("vkDestroyCommandPool")(device, pool_, nullptr);
            fn<PFN_vkDestroyDevice>("vkDestroyDevice")(device, nullptr);
        }
        if (messenger_) {
            reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(
                instance_proc(instance, "vkDestroyDebugUtilsMessengerEXT"))(instance, messenger_, nullptr);
        }
        if (instance)
            reinterpret_cast<PFN_vkDestroyInstance>(instance_proc(instance, "vkDestroyInstance"))(instance, nullptr);
    }

    template<typename F>
    F fn(const char* name) const {
        return reinterpret_cast<F>(device_proc(device, name));
    }

    template<typename F>
    F instance_fn(const char* name) const {
        return reinterpret_cast<F>(instance_proc(instance, name));
    }

    unsigned validation_errors() const { return validation_error_count(); }

    bool create() {
        instance_proc = load_vulkan();
        if (!instance_proc) return false;
        const auto create_instance = reinterpret_cast<PFN_vkCreateInstance>(instance_proc(nullptr, "vkCreateInstance"));
        const auto enumerate_layers = reinterpret_cast<PFN_vkEnumerateInstanceLayerProperties>(
            instance_proc(nullptr, "vkEnumerateInstanceLayerProperties"));

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4996)  // getenv: read once, at startup
#endif
        const char* env = std::getenv("LEONIDA_VK_VALIDATION");
#ifdef _MSC_VER
#pragma warning(pop)
#endif
        validation = env && std::strcmp(env, "0") != 0;
        const char* layer = "VK_LAYER_KHRONOS_validation";
        const char* debug_utils = VK_EXT_DEBUG_UTILS_EXTENSION_NAME;
        VkDebugUtilsMessengerCreateInfoEXT messenger_info{VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT};
        messenger_info.messageSeverity =
            VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
        messenger_info.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                                     VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                                     VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
        messenger_info.pfnUserCallback = on_validation;

        VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
        app.apiVersion = VK_API_VERSION_1_3;
        VkInstanceCreateInfo instance_info{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
        instance_info.pApplicationInfo = &app;
        if (validation) {
            std::uint32_t count = 0;
            enumerate_layers(&count, nullptr);
            std::vector<VkLayerProperties> layers(count);
            enumerate_layers(&count, layers.data());
            bool found = false;
            for (const auto& properties : layers) found = found || std::strcmp(properties.layerName, layer) == 0;
            if (!found) {
                std::fprintf(stderr, "LEONIDA_VK_VALIDATION is set but %s is not installed\n", layer);
                return false;
            }
            instance_info.pNext = &messenger_info;
            instance_info.enabledLayerCount = 1;
            instance_info.ppEnabledLayerNames = &layer;
            instance_info.enabledExtensionCount = 1;
            instance_info.ppEnabledExtensionNames = &debug_utils;
        }
        if (create_instance(&instance_info, nullptr, &instance) != VK_SUCCESS) return false;
        if (validation &&
            instance_fn<PFN_vkCreateDebugUtilsMessengerEXT>("vkCreateDebugUtilsMessengerEXT")(
                instance, &messenger_info, nullptr, &messenger_) != VK_SUCCESS)
            return false;

        const auto enumerate = instance_fn<PFN_vkEnumeratePhysicalDevices>("vkEnumeratePhysicalDevices");
        const auto properties_of = instance_fn<PFN_vkGetPhysicalDeviceProperties>("vkGetPhysicalDeviceProperties");
        const auto families_of =
            instance_fn<PFN_vkGetPhysicalDeviceQueueFamilyProperties>("vkGetPhysicalDeviceQueueFamilyProperties");
        std::uint32_t count = 0;
        enumerate(instance, &count, nullptr);
        std::vector<VkPhysicalDevice> devices(count);
        enumerate(instance, &count, devices.data());
        VkPhysicalDeviceProperties chosen{};
        for (auto candidate : devices) {
            VkPhysicalDeviceProperties props{};
            properties_of(candidate, &props);
            if (props.apiVersion < VK_API_VERSION_1_3) continue;
            std::uint32_t family_count = 0;
            families_of(candidate, &family_count, nullptr);
            std::vector<VkQueueFamilyProperties> families(family_count);
            families_of(candidate, &family_count, families.data());
            for (std::uint32_t i = 0; i < family_count; ++i) {
                if (!(families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)) continue;
                if (!physical || props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
                    physical = candidate;
                    family = i;
                    chosen = props;
                }
                break;
            }
        }
        if (!physical) {
            std::fprintf(stderr, "no Vulkan 1.3 device with a graphics queue\n");
            return false;
        }
        std::printf("device: %s%s\n", chosen.deviceName, validation ? " (validation on)" : "");

        VkPhysicalDeviceVulkan13Features features13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
        VkPhysicalDeviceVulkan12Features features12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES, &features13};
        VkPhysicalDeviceFeatures2 features{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &features12};
        instance_fn<PFN_vkGetPhysicalDeviceFeatures2>("vkGetPhysicalDeviceFeatures2")(physical, &features);
        if (!features13.dynamicRendering || !features13.synchronization2 || !features12.timelineSemaphore ||
            !features12.separateDepthStencilLayouts) {
            std::fprintf(stderr, "the device lacks dynamicRendering, synchronization2, timelineSemaphore or "
                                 "separateDepthStencilLayouts\n");
            return false;
        }
        // VK_EXT_conditional_rendering (SetPredication), where available.
        {
            const auto extensions_of =
                instance_fn<PFN_vkEnumerateDeviceExtensionProperties>("vkEnumerateDeviceExtensionProperties");
            std::uint32_t extension_count = 0;
            extensions_of(physical, nullptr, &extension_count, nullptr);
            std::vector<VkExtensionProperties> extensions(extension_count);
            extensions_of(physical, nullptr, &extension_count, extensions.data());
            for (const auto& e : extensions)
                if (!std::strcmp(e.extensionName, VK_EXT_CONDITIONAL_RENDERING_EXTENSION_NAME)) {
                    VkPhysicalDeviceConditionalRenderingFeaturesEXT supported{
                        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CONDITIONAL_RENDERING_FEATURES_EXT};
                    VkPhysicalDeviceFeatures2 query{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &supported};
                    instance_fn<PFN_vkGetPhysicalDeviceFeatures2>("vkGetPhysicalDeviceFeatures2")(physical, &query);
                    conditional_rendering = supported.conditionalRendering == VK_TRUE;
                }
        }
        VkPhysicalDeviceConditionalRenderingFeaturesEXT enable_conditional{
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CONDITIONAL_RENDERING_FEATURES_EXT};
        enable_conditional.conditionalRendering = VK_TRUE;

        VkPhysicalDeviceVulkan13Features enable13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
                                                  conditional_rendering ? &enable_conditional : nullptr};
        enable13.dynamicRendering = VK_TRUE;
        enable13.synchronization2 = VK_TRUE;
        VkPhysicalDeviceVulkan12Features enable12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES, &enable13};
        enable12.timelineSemaphore = VK_TRUE;
        enable12.separateDepthStencilLayouts = VK_TRUE;
        // What the bindless set 0 of the translated shaders needs, where the device has it: runtime arrays of
        // images indexed dynamically, updated while the set is bound; and, to leave slots empty, partially bound
        // bindings.
        bindless = features12.runtimeDescriptorArray && features.features.shaderSampledImageArrayDynamicIndexing &&
                   features.features.shaderStorageImageArrayDynamicIndexing &&
                   features12.descriptorBindingSampledImageUpdateAfterBind &&
                   features12.descriptorBindingStorageImageUpdateAfterBind &&
                   features12.descriptorBindingStorageBufferUpdateAfterBind;
        partially_bound = bindless && features12.descriptorBindingPartiallyBound;
        enable12.runtimeDescriptorArray = bindless;
        enable12.descriptorBindingSampledImageUpdateAfterBind = bindless;
        enable12.descriptorBindingStorageImageUpdateAfterBind = bindless;
        enable12.descriptorBindingStorageBufferUpdateAfterBind = bindless;
        enable12.descriptorBindingPartiallyBound = partially_bound;

        const float priority = 1.0f;
        VkDeviceQueueCreateInfo queue_info{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
        queue_info.queueFamilyIndex = family;
        queue_info.queueCount = 1;
        queue_info.pQueuePriorities = &priority;
        VkPhysicalDeviceFeatures enable{};
        enable.textureCompressionBC = features.features.textureCompressionBC;  // BC uploads, where available
        bc = features.features.textureCompressionBC == VK_TRUE;
        enable.shaderSampledImageArrayDynamicIndexing = bindless;
        enable.shaderStorageImageArrayDynamicIndexing = bindless;
        const char* conditional_extension = VK_EXT_CONDITIONAL_RENDERING_EXTENSION_NAME;
        VkDeviceCreateInfo device_info{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, &enable12};
        device_info.queueCreateInfoCount = 1;
        device_info.pQueueCreateInfos = &queue_info;
        device_info.pEnabledFeatures = &enable;
        if (conditional_rendering) {
            device_info.enabledExtensionCount = 1;
            device_info.ppEnabledExtensionNames = &conditional_extension;
        }
        if (instance_fn<PFN_vkCreateDevice>("vkCreateDevice")(physical, &device_info, nullptr, &device) != VK_SUCCESS)
            return false;
        device_proc = instance_fn<PFN_vkGetDeviceProcAddr>("vkGetDeviceProcAddr");
        fn<PFN_vkGetDeviceQueue>("vkGetDeviceQueue")(device, family, 0, &queue);
        instance_fn<PFN_vkGetPhysicalDeviceMemoryProperties>("vkGetPhysicalDeviceMemoryProperties")(physical,
                                                                                                    &memory_);
        VkCommandPoolCreateInfo pool{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        pool.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        pool.queueFamilyIndex = family;
        return fn<PFN_vkCreateCommandPool>("vkCreateCommandPool")(device, &pool, nullptr, &pool_) == VK_SUCCESS;
    }

    bool depth_format_usable(VkFormat format) const {
        VkFormatProperties properties{};
        instance_fn<PFN_vkGetPhysicalDeviceFormatProperties>("vkGetPhysicalDeviceFormatProperties")(physical, format,
                                                                                                    &properties);
        constexpr VkFormatFeatureFlags needed = VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT |
                                                VK_FORMAT_FEATURE_TRANSFER_SRC_BIT | VK_FORMAT_FEATURE_TRANSFER_DST_BIT;
        return (properties.optimalTilingFeatures & needed) == needed;
    }

    // depth > 1: a 3D image. samples > 1: a multisampled one.
    VkImage image(VkFormat format, std::uint32_t width, std::uint32_t height, std::uint32_t layers,
                  VkImageUsageFlags usage, std::uint32_t depth = 1, std::uint32_t samples = 1) {
        VkImageCreateInfo info{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        info.imageType = depth > 1 ? VK_IMAGE_TYPE_3D : VK_IMAGE_TYPE_2D;
        info.format = format;
        info.extent = {width, height, depth};
        info.mipLevels = 1;
        info.arrayLayers = layers;
        info.samples = VkSampleCountFlagBits(samples);
        info.usage = usage | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        VkImage image = VK_NULL_HANDLE;
        if (fn<PFN_vkCreateImage>("vkCreateImage")(device, &info, nullptr, &image) != VK_SUCCESS) return VK_NULL_HANDLE;
        VkMemoryRequirements requirements{};
        fn<PFN_vkGetImageMemoryRequirements>("vkGetImageMemoryRequirements")(device, image, &requirements);
        const VkDeviceMemory memory = allocate(requirements, 0);
        images_.push_back({image, memory});
        if (!memory || fn<PFN_vkBindImageMemory>("vkBindImageMemory")(device, image, memory, 0) != VK_SUCCESS)
            return VK_NULL_HANDLE;
        return image;
    }

    // A host-visible, coherent buffer, mapped.
    VkBuffer buffer(VkDeviceSize size, VkBufferUsageFlags usage, void** mapped) {
        VkBufferCreateInfo info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        info.size = size;
        info.usage = usage;
        VkBuffer buffer = VK_NULL_HANDLE;
        if (fn<PFN_vkCreateBuffer>("vkCreateBuffer")(device, &info, nullptr, &buffer) != VK_SUCCESS)
            return VK_NULL_HANDLE;
        VkMemoryRequirements requirements{};
        fn<PFN_vkGetBufferMemoryRequirements>("vkGetBufferMemoryRequirements")(device, buffer, &requirements);
        const VkDeviceMemory memory =
            allocate(requirements, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        buffers_.push_back({buffer, memory});
        if (!memory || fn<PFN_vkBindBufferMemory>("vkBindBufferMemory")(device, buffer, memory, 0) != VK_SUCCESS)
            return VK_NULL_HANDLE;
        if (fn<PFN_vkMapMemory>("vkMapMemory")(device, memory, 0, size, 0, mapped) != VK_SUCCESS) return VK_NULL_HANDLE;
        std::memset(*mapped, 0xcd, size_t(size));
        return buffer;
    }

    // Records `body` into a one-shot command buffer, submits it and waits for the queue.
    template<typename Body>
    bool run(Body&& body) {
        VkCommandBufferAllocateInfo info{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        info.commandPool = pool_;
        info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        info.commandBufferCount = 1;
        VkCommandBuffer command_buffer = VK_NULL_HANDLE;
        if (fn<PFN_vkAllocateCommandBuffers>("vkAllocateCommandBuffers")(device, &info, &command_buffer) != VK_SUCCESS)
            return false;
        VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        fn<PFN_vkBeginCommandBuffer>("vkBeginCommandBuffer")(command_buffer, &begin);
        const bool recorded = body(command_buffer);
        fn<PFN_vkEndCommandBuffer>("vkEndCommandBuffer")(command_buffer);
        VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        submit.commandBufferCount = 1;
        submit.pCommandBuffers = &command_buffer;
        const bool ok = recorded && fn<PFN_vkQueueSubmit>("vkQueueSubmit")(queue, 1, &submit, VK_NULL_HANDLE) == VK_SUCCESS &&
                        fn<PFN_vkQueueWaitIdle>("vkQueueWaitIdle")(queue) == VK_SUCCESS;
        fn<PFN_vkFreeCommandBuffers>("vkFreeCommandBuffers")(device, pool_, 1, &command_buffer);
        return ok;
    }

    // UNDEFINED -> layout for a whole image: what resource creation does for a D3D12 initial state.
    void initial_layout(VkCommandBuffer command_buffer, VkImage image, VkImageAspectFlags aspects,
                        std::uint32_t layers, VkImageLayout layout) const {
        VkImageMemoryBarrier barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        barrier.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
        barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        barrier.newLayout = layout;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image = image;
        barrier.subresourceRange = {aspects, 0, 1, 0, layers};
        fn<PFN_vkCmdPipelineBarrier>("vkCmdPipelineBarrier")(command_buffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                                                             VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0,
                                                             nullptr, 1, &barrier);
    }

    // Copies one aspect of layers [0, layers) from an image in `layout` to buffer + offset, then makes it host-visible.
    void read_back(VkCommandBuffer command_buffer, VkImage image, VkImageLayout layout, VkImageAspectFlags aspect,
                   std::uint32_t width, std::uint32_t height, std::uint32_t layers, VkBuffer buffer,
                   VkDeviceSize offset) const {
        VkBufferImageCopy copy{};
        copy.bufferOffset = offset;
        copy.imageSubresource = {aspect, 0, 0, layers};
        copy.imageExtent = {width, height, 1};
        fn<PFN_vkCmdCopyImageToBuffer>("vkCmdCopyImageToBuffer")(command_buffer, image, layout, buffer, 1, &copy);
        VkMemoryBarrier host{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        host.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        host.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
        fn<PFN_vkCmdPipelineBarrier>("vkCmdPipelineBarrier")(command_buffer, VK_PIPELINE_STAGE_TRANSFER_BIT,
                                                             VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &host, 0, nullptr, 0,
                                                             nullptr);
    }

private:
    VkDeviceMemory allocate(const VkMemoryRequirements& requirements, VkMemoryPropertyFlags flags) {
        for (std::uint32_t i = 0; i < memory_.memoryTypeCount; ++i) {
            if (!(requirements.memoryTypeBits & (1u << i))) continue;
            if ((memory_.memoryTypes[i].propertyFlags & flags) != flags) continue;
            VkMemoryAllocateInfo info{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
            info.allocationSize = requirements.size;
            info.memoryTypeIndex = i;
            VkDeviceMemory memory = VK_NULL_HANDLE;
            if (fn<PFN_vkAllocateMemory>("vkAllocateMemory")(device, &info, nullptr, &memory) == VK_SUCCESS)
                return memory;
            return VK_NULL_HANDLE;
        }
        return VK_NULL_HANDLE;
    }

    VkDebugUtilsMessengerEXT messenger_ = VK_NULL_HANDLE;
    VkCommandPool pool_ = VK_NULL_HANDLE;
    VkPhysicalDeviceMemoryProperties memory_{};
    std::vector<std::pair<VkImage, VkDeviceMemory>> images_;
    std::vector<std::pair<VkBuffer, VkDeviceMemory>> buffers_;
};

} // namespace vktest
