// One offscreen 4x4 image, two layers, one submission. No shaders or game data.
#include "vulkan_loader.h"
#include "../src/vulkan_clear.h"

#include <array>
#include <cstdint>
#include <vector>

namespace {

struct Resources {
    VkInstance instance = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkCommandPool pool = VK_NULL_HANDLE;
    VkImage image = VK_NULL_HANDLE;
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory image_memory = VK_NULL_HANDLE;
    VkDeviceMemory buffer_memory = VK_NULL_HANDLE;
    PFN_vkGetInstanceProcAddr instance_proc = nullptr;
    PFN_vkGetDeviceProcAddr device_proc = nullptr;

    ~Resources() {
        if (device) {
#define CLEAN(name, ...) reinterpret_cast<PFN_##name>(device_proc(device, #name))(device, __VA_ARGS__, nullptr)
            if (pool) CLEAN(vkDestroyCommandPool, pool);
            if (image) CLEAN(vkDestroyImage, image);
            if (buffer) CLEAN(vkDestroyBuffer, buffer);
            if (image_memory) CLEAN(vkFreeMemory, image_memory);
            if (buffer_memory) CLEAN(vkFreeMemory, buffer_memory);
#undef CLEAN
            reinterpret_cast<PFN_vkDestroyDevice>(device_proc(device, "vkDestroyDevice"))(device, nullptr);
        }
        if (instance)
            reinterpret_cast<PFN_vkDestroyInstance>(instance_proc(instance, "vkDestroyInstance"))(instance, nullptr);
    }
};

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
    Resources r;
    r.instance_proc = load_vulkan();
    EXPECT(r.instance_proc);
#define INSTANCE(name) const auto name = reinterpret_cast<PFN_##name>(r.instance_proc(r.instance, #name))
#define DEVICE(name) const auto name = reinterpret_cast<PFN_##name>(r.device_proc(r.device, #name))
    INSTANCE(vkCreateInstance);
    VkInstanceCreateInfo instance_info{};
    instance_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    VK_CHECK(vkCreateInstance(&instance_info, nullptr, &r.instance));
    INSTANCE(vkEnumeratePhysicalDevices);
    INSTANCE(vkGetPhysicalDeviceProperties);
    INSTANCE(vkGetPhysicalDeviceQueueFamilyProperties);
    INSTANCE(vkGetPhysicalDeviceMemoryProperties);
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
    std::printf("device: %s\n", properties.deviceName);
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
    DEVICE(vkCmdCopyImageToBuffer);
    DEVICE(vkEndCommandBuffer);
    DEVICE(vkQueueSubmit);
    DEVICE(vkQueueWaitIdle);
    DEVICE(vkMapMemory);
    DEVICE(vkUnmapMemory);

    constexpr uint32_t width = 4, height = 4, layers = 2;
    constexpr VkDeviceSize bytes = width * height * layers * 4;
    VkImageCreateInfo image_info{};
    image_info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    image_info.imageType = VK_IMAGE_TYPE_2D;
    image_info.format = VK_FORMAT_R8G8B8A8_UNORM;
    image_info.extent = {width, height, 1};
    image_info.mipLevels = 1;
    image_info.arrayLayers = layers;
    image_info.samples = VK_SAMPLE_COUNT_1_BIT;
    image_info.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    VK_CHECK(vkCreateImage(r.device, &image_info, nullptr, &r.image));
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
    vkGetImageMemoryRequirements(r.device, r.image, &requirements);
    VK_CHECK(allocate(requirements, 0, &r.image_memory));
    VK_CHECK(vkBindImageMemory(r.device, r.image, r.image_memory, 0));
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

    VkImageMemoryBarrier transition{};
    transition.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    transition.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    transition.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    transition.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    transition.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    transition.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    transition.image = r.image;
    transition.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, layers};
    vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                         0, 0, nullptr, 0, nullptr, 1, &transition);

    // Reuse the CPU descriptor slot for another array layer and discard it before replay.
    ClearTargetView descriptor{1, 0, 0, 1};
    ClearCommandList list;
    EXPECT(list.clear_render_target(descriptor, {0, 0, 1, 1}));
    descriptor = {2, 0, 1, 1};
    EXPECT(list.clear_render_target(descriptor, {1, 0, 0, 1}));
    EXPECT(list.clear_render_target(descriptor, {0, 1, 0, 1}));
    bool second_resource_available = false;
    const auto resolve = [&](ClearResourceId id) {
        return id == 1 || (id == 2 && second_resource_available) ? r.image : VK_NULL_HANDLE;
    };
    const auto replay = [&] {
        return replay_clears(list, command_buffer, resolve, vkCmdPipelineBarrier, vkCmdClearColorImage);
    };
    EXPECT(replay() == ClearReplayResult::list_open);
    EXPECT(list.close());
    descriptor = {};
    EXPECT(replay() == ClearReplayResult::unresolved_target);
    second_resource_available = true;
    EXPECT(replay() == ClearReplayResult::success);

    transition.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    transition.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    transition.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    transition.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                         0, 0, nullptr, 0, nullptr, 1, &transition);
    VkBufferImageCopy copy{};
    copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, layers};
    copy.imageExtent = {width, height, 1};
    vkCmdCopyImageToBuffer(command_buffer, r.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, r.buffer, 1, &copy);
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
    const auto* pixels = static_cast<const uint8_t*>(mapped);
    constexpr std::array<uint8_t, 4> blue{0, 0, 255, 255}, green{0, 255, 0, 255};
    const bool matches = [&] {
        for (uint32_t i = 0; i < width * height * layers; ++i) {
            const auto& expected = i < width * height ? blue : green;
            if (!std::equal(expected.begin(), expected.end(), pixels + i * 4)) {
                std::fprintf(stderr, "pixel %u: got %u,%u,%u,%u\n", i, unsigned(pixels[i * 4]),
                             unsigned(pixels[i * 4 + 1]), unsigned(pixels[i * 4 + 2]), unsigned(pixels[i * 4 + 3]));
                return false;
            }
        }
        return true;
    }();
    vkUnmapMemory(r.device, r.buffer_memory);
    if (!matches) return 1;
    std::printf("%u pixels matched across two layers\n", width * height * layers);
    return 0;
}
