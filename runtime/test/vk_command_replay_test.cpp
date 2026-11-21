// Replays a recorded command list (clears and D3D12 barriers) on one offscreen 4x4 colour image with two layers
// and one 4x4 depth-stencil image, then reads both back. No shaders or game data.
// Set LEONIDA_VK_VALIDATION=1 to run under VK_LAYER_KHRONOS_validation; any validation error fails the test.
#include "vulkan_loader.h"
#include "../src/vulkan_replay.h"

#include <array>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

struct Resources {
    VkInstance instance = VK_NULL_HANDLE;
    VkDebugUtilsMessengerEXT messenger = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkCommandPool pool = VK_NULL_HANDLE;
    VkImage color = VK_NULL_HANDLE;
    VkImage depth = VK_NULL_HANDLE;
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory color_memory = VK_NULL_HANDLE;
    VkDeviceMemory depth_memory = VK_NULL_HANDLE;
    VkDeviceMemory buffer_memory = VK_NULL_HANDLE;
    PFN_vkGetInstanceProcAddr instance_proc = nullptr;
    PFN_vkGetDeviceProcAddr device_proc = nullptr;

    ~Resources() {
        if (device) {
#define CLEAN(name, ...) reinterpret_cast<PFN_##name>(device_proc(device, #name))(device, __VA_ARGS__, nullptr)
            if (pool) CLEAN(vkDestroyCommandPool, pool);
            if (color) CLEAN(vkDestroyImage, color);
            if (depth) CLEAN(vkDestroyImage, depth);
            if (buffer) CLEAN(vkDestroyBuffer, buffer);
            if (color_memory) CLEAN(vkFreeMemory, color_memory);
            if (depth_memory) CLEAN(vkFreeMemory, depth_memory);
            if (buffer_memory) CLEAN(vkFreeMemory, buffer_memory);
#undef CLEAN
            reinterpret_cast<PFN_vkDestroyDevice>(device_proc(device, "vkDestroyDevice"))(device, nullptr);
        }
        if (messenger) {
            reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(
                instance_proc(instance, "vkDestroyDebugUtilsMessengerEXT"))(instance, messenger, nullptr);
        }
        if (instance)
            reinterpret_cast<PFN_vkDestroyInstance>(instance_proc(instance, "vkDestroyInstance"))(instance, nullptr);
    }
};

unsigned g_validation_errors = 0;

VKAPI_ATTR VkBool32 VKAPI_CALL on_validation(VkDebugUtilsMessageSeverityFlagBitsEXT severity,
                                             VkDebugUtilsMessageTypeFlagsEXT,
                                             const VkDebugUtilsMessengerCallbackDataEXT* data, void*) {
    if (severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) ++g_validation_errors;
    std::fprintf(stderr, "validation: %s\n", data && data->pMessage ? data->pMessage : "(no message)");
    return VK_FALSE;
}

#define EXPECT(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); return 1; } \
} while (0)
#define VK_CHECK(call) do { \
    const VkResult result = (call); \
    if (result != VK_SUCCESS) { std::fprintf(stderr, "%s failed (%d)\n", #call, int(result)); return 1; } \
} while (0)

} // namespace

int main() {
    using namespace leonida;
    namespace s = d3d12_state;
    Resources r;
    r.instance_proc = load_vulkan();
    EXPECT(r.instance_proc);
#define INSTANCE(name) const auto name = reinterpret_cast<PFN_##name>(r.instance_proc(r.instance, #name))
#define DEVICE(name) const auto name = reinterpret_cast<PFN_##name>(r.device_proc(r.device, #name))
    INSTANCE(vkCreateInstance);
    INSTANCE(vkEnumerateInstanceLayerProperties);

    const char* validation_env = std::getenv("LEONIDA_VK_VALIDATION");
    const bool validation = validation_env && std::strcmp(validation_env, "0") != 0;
    const char* layer = "VK_LAYER_KHRONOS_validation";
    const char* debug_utils = VK_EXT_DEBUG_UTILS_EXTENSION_NAME;
    VkDebugUtilsMessengerCreateInfoEXT messenger_info{};
    messenger_info.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
    messenger_info.messageSeverity =
        VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    messenger_info.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                                 VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                                 VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
    messenger_info.pfnUserCallback = on_validation;

    VkApplicationInfo app{};
    app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app.apiVersion = VK_API_VERSION_1_1;
    VkInstanceCreateInfo instance_info{};
    instance_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    instance_info.pApplicationInfo = &app;
    if (validation) {
        uint32_t layer_count = 0;
        VK_CHECK(vkEnumerateInstanceLayerProperties(&layer_count, nullptr));
        std::vector<VkLayerProperties> layers(layer_count);
        VK_CHECK(vkEnumerateInstanceLayerProperties(&layer_count, layers.data()));
        bool found = false;
        for (const auto& properties : layers) found = found || std::strcmp(properties.layerName, layer) == 0;
        if (!found) {
            std::fprintf(stderr, "LEONIDA_VK_VALIDATION is set but %s is not installed\n", layer);
            return 1;
        }
        instance_info.pNext = &messenger_info;
        instance_info.enabledLayerCount = 1;
        instance_info.ppEnabledLayerNames = &layer;
        instance_info.enabledExtensionCount = 1;
        instance_info.ppEnabledExtensionNames = &debug_utils;
    }
    VK_CHECK(vkCreateInstance(&instance_info, nullptr, &r.instance));
    if (validation) {
        INSTANCE(vkCreateDebugUtilsMessengerEXT);
        VK_CHECK(vkCreateDebugUtilsMessengerEXT(r.instance, &messenger_info, nullptr, &r.messenger));
    }
    INSTANCE(vkEnumeratePhysicalDevices);
    INSTANCE(vkGetPhysicalDeviceProperties);
    INSTANCE(vkGetPhysicalDeviceQueueFamilyProperties);
    INSTANCE(vkGetPhysicalDeviceMemoryProperties);
    INSTANCE(vkGetPhysicalDeviceFormatProperties);
    INSTANCE(vkCreateDevice);
    r.device_proc = reinterpret_cast<PFN_vkGetDeviceProcAddr>(r.instance_proc(r.instance, "vkGetDeviceProcAddr"));

    uint32_t device_count = 0;
    VK_CHECK(vkEnumeratePhysicalDevices(r.instance, &device_count, nullptr));
    EXPECT(device_count);
    std::vector<VkPhysicalDevice> devices(device_count);
    VK_CHECK(vkEnumeratePhysicalDevices(r.instance, &device_count, devices.data()));
    VkPhysicalDevice physical = VK_NULL_HANDLE;
    uint32_t graphics_family = 0;
    VkPhysicalDeviceProperties properties{};
    for (auto candidate : devices) {
        uint32_t family_count = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(candidate, &family_count, nullptr);
        std::vector<VkQueueFamilyProperties> families(family_count);
        vkGetPhysicalDeviceQueueFamilyProperties(candidate, &family_count, families.data());
        for (uint32_t i = 0; i < family_count; ++i) {
            if (!(families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)) continue;
            VkPhysicalDeviceProperties props{};
            vkGetPhysicalDeviceProperties(candidate, &props);
            if (!physical || props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
                physical = candidate;
                graphics_family = i;
                properties = props;
            }
            break;
        }
    }
    EXPECT(physical);
    std::printf("device: %s%s\n", properties.deviceName, validation ? " (validation on)" : "");

    // D32_SFLOAT_S8_UINT when the device has it, so stencil is exercised; D32_SFLOAT (always available as a depth
    // attachment when the S8 variant is not) otherwise.
    const auto depth_usable = [&](VkFormat format) {
        VkFormatProperties format_properties{};
        vkGetPhysicalDeviceFormatProperties(physical, format, &format_properties);
        constexpr VkFormatFeatureFlags needed = VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT |
                                                VK_FORMAT_FEATURE_TRANSFER_SRC_BIT |
                                                VK_FORMAT_FEATURE_TRANSFER_DST_BIT;
        return (format_properties.optimalTilingFeatures & needed) == needed;
    };
    const bool has_stencil = depth_usable(VK_FORMAT_D32_SFLOAT_S8_UINT);
    const VkFormat depth_format = has_stencil ? VK_FORMAT_D32_SFLOAT_S8_UINT : VK_FORMAT_D32_SFLOAT;
    EXPECT(has_stencil || depth_usable(depth_format));
    const VkImageAspectFlags depth_aspects =
        VK_IMAGE_ASPECT_DEPTH_BIT | (has_stencil ? VK_IMAGE_ASPECT_STENCIL_BIT : 0);

    const float priority = 1.0f;
    VkDeviceQueueCreateInfo queue_info{};
    queue_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queue_info.queueFamilyIndex = graphics_family;
    queue_info.queueCount = 1;
    queue_info.pQueuePriorities = &priority;
    VkDeviceCreateInfo device_info{};
    device_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    device_info.queueCreateInfoCount = 1;
    device_info.pQueueCreateInfos = &queue_info;
    VK_CHECK(vkCreateDevice(physical, &device_info, nullptr, &r.device));

    DEVICE(vkGetDeviceQueue);
    DEVICE(vkCreateImage);
    DEVICE(vkCreateBuffer);
    DEVICE(vkGetImageMemoryRequirements);
    DEVICE(vkGetBufferMemoryRequirements);
    DEVICE(vkAllocateMemory);
    DEVICE(vkBindImageMemory);
    DEVICE(vkBindBufferMemory);
    DEVICE(vkCreateCommandPool);
    DEVICE(vkAllocateCommandBuffers);
    DEVICE(vkBeginCommandBuffer);
    DEVICE(vkCmdPipelineBarrier);
    DEVICE(vkCmdClearColorImage);
    DEVICE(vkCmdClearDepthStencilImage);
    DEVICE(vkCmdCopyImageToBuffer);
    DEVICE(vkEndCommandBuffer);
    DEVICE(vkQueueSubmit);
    DEVICE(vkQueueWaitIdle);
    DEVICE(vkMapMemory);
    DEVICE(vkUnmapMemory);

    constexpr uint32_t width = 4, height = 4, layers = 2, texels = width * height;
    constexpr VkDeviceSize color_bytes = texels * layers * 4;
    constexpr VkDeviceSize depth_offset = color_bytes, stencil_offset = depth_offset + texels * 4;
    constexpr VkDeviceSize bytes = stencil_offset + texels;
    VkImageCreateInfo image_info{};
    image_info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    image_info.imageType = VK_IMAGE_TYPE_2D;
    image_info.format = VK_FORMAT_R8G8B8A8_UNORM;
    image_info.extent = {width, height, 1};
    image_info.mipLevels = 1;
    image_info.arrayLayers = layers;
    image_info.samples = VK_SAMPLE_COUNT_1_BIT;
    image_info.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                       VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    VK_CHECK(vkCreateImage(r.device, &image_info, nullptr, &r.color));
    image_info.format = depth_format;
    image_info.arrayLayers = 1;
    image_info.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                       VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    VK_CHECK(vkCreateImage(r.device, &image_info, nullptr, &r.depth));
    VkBufferCreateInfo buffer_info{};
    buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buffer_info.size = bytes;
    buffer_info.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    VK_CHECK(vkCreateBuffer(r.device, &buffer_info, nullptr, &r.buffer));

    VkPhysicalDeviceMemoryProperties memory_properties{};
    vkGetPhysicalDeviceMemoryProperties(physical, &memory_properties);
    const auto allocate = [&](const VkMemoryRequirements& requirements, VkMemoryPropertyFlags flags,
                              VkDeviceMemory* memory) {
        for (uint32_t i = 0; i < memory_properties.memoryTypeCount; ++i) {
            if (!(requirements.memoryTypeBits & (1u << i))) continue;
            if ((memory_properties.memoryTypes[i].propertyFlags & flags) != flags) continue;
            VkMemoryAllocateInfo info{};
            info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
            info.allocationSize = requirements.size;
            info.memoryTypeIndex = i;
            return vkAllocateMemory(r.device, &info, nullptr, memory);
        }
        return VK_ERROR_FEATURE_NOT_PRESENT;
    };
    VkMemoryRequirements requirements{};
    vkGetImageMemoryRequirements(r.device, r.color, &requirements);
    VK_CHECK(allocate(requirements, 0, &r.color_memory));
    VK_CHECK(vkBindImageMemory(r.device, r.color, r.color_memory, 0));
    vkGetImageMemoryRequirements(r.device, r.depth, &requirements);
    VK_CHECK(allocate(requirements, 0, &r.depth_memory));
    VK_CHECK(vkBindImageMemory(r.device, r.depth, r.depth_memory, 0));
    vkGetBufferMemoryRequirements(r.device, r.buffer, &requirements);
    VK_CHECK(allocate(requirements, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                      &r.buffer_memory));
    VK_CHECK(vkBindBufferMemory(r.device, r.buffer, r.buffer_memory, 0));

    VkCommandPoolCreateInfo pool_info{};
    pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pool_info.queueFamilyIndex = graphics_family;
    VK_CHECK(vkCreateCommandPool(r.device, &pool_info, nullptr, &r.pool));
    VkCommandBufferAllocateInfo command_info{};
    command_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    command_info.commandPool = r.pool;
    command_info.commandBufferCount = 1;
    VkCommandBuffer command_buffer;
    VK_CHECK(vkAllocateCommandBuffers(r.device, &command_info, &command_buffer));
    VkCommandBufferBeginInfo begin_info{};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK(vkBeginCommandBuffer(command_buffer, &begin_info));

    // Resource creation's job, done by hand here: put each image in the layout of its D3D12 initial state
    // (render_target and depth_write).
    const auto initial = [&](VkImage image, VkImageAspectFlags aspects, uint32_t layer_count, uint32_t state) {
        const VulkanAccess to = vulkan_access(state, false);
        VkImageMemoryBarrier barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barrier.dstAccessMask = to.access;
        barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        barrier.newLayout = to.layout;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image = image;
        barrier.subresourceRange = {aspects, 0, 1, 0, layer_count};
        vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, to.stages, 0, 0, nullptr, 0,
                             nullptr, 1, &barrier);
    };
    initial(r.color, VK_IMAGE_ASPECT_COLOR_BIT, layers, s::render_target);
    initial(r.depth, depth_aspects, 1, s::depth_write);

    ResourceBinding color_binding{};
    color_binding.image = r.color;
    color_binding.array_layers = layers;
    ResourceBinding depth_binding{};
    depth_binding.image = r.depth;
    depth_binding.aspects = depth_aspects;
    constexpr ResourceId color_id = 1, depth_id = 2, missing_id = 3;
    const auto resolve = [&](ResourceId id) -> const ResourceBinding* {
        if (id == color_id) return &color_binding;
        if (id == depth_id) return &depth_binding;
        return nullptr;
    };
    const ReplayFunctions vk{vkCmdPipelineBarrier, vkCmdClearColorImage, vkCmdClearDepthStencilImage};

    // Rejected lists write nothing to the command buffer.
    {
        CommandList bad;
        EXPECT(bad.clear_render_target({color_id, 0, 0, 1}, {1, 1, 1, 1}));
        EXPECT(replay(bad, command_buffer, resolve, vk) == ReplayResult::list_open);
        EXPECT(bad.transition({missing_id, 0, s::common, s::copy_dest}));
        EXPECT(bad.close());
        EXPECT(replay(bad, command_buffer, resolve, vk) == ReplayResult::unresolved_resource);
        EXPECT(bad.reset());
        EXPECT(bad.transition({color_id, layers, s::render_target, s::copy_source}));  // no plane 1
        EXPECT(bad.close());
        EXPECT(replay(bad, command_buffer, resolve, vk) == ReplayResult::invalid_subresource);
        EXPECT(bad.reset());
        EXPECT(bad.clear_render_target({color_id, 0, 1, 2}, {1, 1, 1, 1}));  // layers 1..2 of 2
        EXPECT(bad.close());
        EXPECT(replay(bad, command_buffer, resolve, vk) == ReplayResult::invalid_subresource);
        EXPECT(bad.reset());
        EXPECT(bad.clear_render_target({depth_id, 0, 0, 1}, {1, 1, 1, 1}));
        EXPECT(bad.close());
        EXPECT(replay(bad, command_buffer, resolve, vk) == ReplayResult::invalid_clear);
        EXPECT(bad.reset());
        EXPECT(bad.clear_depth_stencil({color_id, 0, 0, 1}, clear_depth, 1.0f, 0));
        EXPECT(bad.close());
        EXPECT(replay(bad, command_buffer, resolve, vk) == ReplayResult::invalid_clear);
    }

    // Reuse the CPU descriptor slot for another array layer; the recorded clears keep their own view.
    CommandList list;
    TargetView descriptor{color_id, 0, 0, 1};
    EXPECT(list.clear_render_target(descriptor, {0, 0, 1, 1}));
    descriptor = {color_id, 0, 1, 1};
    EXPECT(list.clear_render_target(descriptor, {1, 0, 0, 1}));
    EXPECT(list.clear_render_target(descriptor, {0, 1, 0, 1}));
    descriptor = {};
    const TargetView depth_view{depth_id, 0, 0, 1};
    EXPECT(list.clear_depth_stencil(depth_view, clear_depth | (has_stencil ? clear_stencil : 0), 0.25f, 0x5a));
    EXPECT(list.clear_depth_stencil(depth_view, clear_depth, 0.75f, 0));  // stencil keeps 0x5a
    EXPECT(list.uav_barrier({0}));
    EXPECT(list.aliasing_barrier({0, 0}));
    EXPECT(list.transition({color_id, all_subresources, s::render_target, s::copy_source}, barrier_begin_only));
    EXPECT(list.transition({color_id, all_subresources, s::render_target, s::copy_source}, barrier_end_only));
    EXPECT(list.transition({depth_id, all_subresources, s::depth_write, s::copy_source}));
    EXPECT(list.close());
    EXPECT(replay(list, command_buffer, resolve, vk) == ReplayResult::success);

    // Both images are now in copy_source (TRANSFER_SRC_OPTIMAL), as the recorded transitions left them.
    VkBufferImageCopy copy{};
    copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, layers};
    copy.imageExtent = {width, height, 1};
    vkCmdCopyImageToBuffer(command_buffer, r.color, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, r.buffer, 1, &copy);
    copy.bufferOffset = depth_offset;
    copy.imageSubresource = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 0, 1};
    vkCmdCopyImageToBuffer(command_buffer, r.depth, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, r.buffer, 1, &copy);
    if (has_stencil) {
        copy.bufferOffset = stencil_offset;
        copy.imageSubresource = {VK_IMAGE_ASPECT_STENCIL_BIT, 0, 0, 1};
        vkCmdCopyImageToBuffer(command_buffer, r.depth, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, r.buffer, 1, &copy);
    }
    VkMemoryBarrier host{};
    host.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    host.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    host.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT,
                         0, 1, &host, 0, nullptr, 0, nullptr);
    VK_CHECK(vkEndCommandBuffer(command_buffer));

    VkQueue queue;
    vkGetDeviceQueue(r.device, graphics_family, 0, &queue);
    VkSubmitInfo submit{};
    submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &command_buffer;
    VK_CHECK(vkQueueSubmit(queue, 1, &submit, VK_NULL_HANDLE));
    VK_CHECK(vkQueueWaitIdle(queue));
    void* mapped = nullptr;
    VK_CHECK(vkMapMemory(r.device, r.buffer_memory, 0, bytes, 0, &mapped));
    const auto* data = static_cast<const uint8_t*>(mapped);
    constexpr std::array<uint8_t, 4> blue{0, 0, 255, 255}, green{0, 255, 0, 255};
    unsigned mismatches = 0;
    for (uint32_t i = 0; i < texels * layers; ++i) {
        const auto& expected = i < texels ? blue : green;
        if (std::equal(expected.begin(), expected.end(), data + i * 4)) continue;
        if (mismatches++ < 4)
            std::fprintf(stderr, "colour texel %u: got %u,%u,%u,%u\n", i, unsigned(data[i * 4]),
                         unsigned(data[i * 4 + 1]), unsigned(data[i * 4 + 2]), unsigned(data[i * 4 + 3]));
    }
    for (uint32_t i = 0; i < texels; ++i) {
        float depth;
        std::memcpy(&depth, data + depth_offset + i * 4, 4);
        if (depth != 0.75f && mismatches++ < 8) std::fprintf(stderr, "depth texel %u: got %f\n", i, depth);
        if (has_stencil && data[stencil_offset + i] != 0x5a && mismatches++ < 8)
            std::fprintf(stderr, "stencil texel %u: got 0x%02x\n", i, unsigned(data[stencil_offset + i]));
    }
    vkUnmapMemory(r.device, r.buffer_memory);
    if (mismatches) return 1;
    if (g_validation_errors) {
        std::fprintf(stderr, "%u validation errors\n", g_validation_errors);
        return 1;
    }
    std::printf("%u colour texels across two layers, %u depth%s texels matched\n", texels * layers, texels,
                has_stencil ? "/stencil" : "");
    return 0;
}
