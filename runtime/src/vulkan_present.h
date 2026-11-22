#pragma once

// Puts a D3D12 back buffer on screen through a Vulkan swapchain, for whichever front end the game turns out to use
// (DXGI or the console's PresentX; docs/d3d12_vulkan.md question 6). The back buffer is an ordinary texture in the
// D3D12 PRESENT (common) state; present() blits it to the acquired swapchain image, scaling if the window size
// differs, and queues the present. The application's own work reaches the same VkQueue first, so every draw it
// submitted before present() is in the frame.
//
// Needs VK_KHR_swapchain on the device and a surface from the caller (which this object then owns). The swapchain
// is recreated when the window changes size or the surface reports it out of date; a minimised window skips
// presents. Two frames are in flight; present() waits for the older one before reusing its command buffer.
//
// Beyond the plain blit (the Xbox PresentX path, docs/d3d12_vulkan.md section 4.5): a source rectangle (only that part
// of the back buffer is scaled to the window), a blank present (the window cleared to black), a swapchain format
// that follows the source (an sRGB source gets an sRGB swapchain, so the blit's decode and encode cancel; every
// other source gets a UNORM one), and an optional capture that copies each presented frame to host memory.

#include "vulkan_backend.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace leonida {

// A sub-rectangle of a back buffer in texels; left/top inclusive, right/bottom exclusive (D3D12_RECT's convention).
struct PresentRect {
    std::int32_t left, top, right, bottom;
};

namespace detail {
inline bool is_srgb_format(VkFormat format) {
    switch (format) {
    case VK_FORMAT_R8_SRGB: case VK_FORMAT_R8G8_SRGB: case VK_FORMAT_R8G8B8_SRGB: case VK_FORMAT_B8G8R8_SRGB:
    case VK_FORMAT_R8G8B8A8_SRGB: case VK_FORMAT_B8G8R8A8_SRGB: case VK_FORMAT_A8B8G8R8_SRGB_PACK32:
        return true;
    default:
        return false;
    }
}
// What vkCmdBlitImage cannot read: integer and block-compressed colour formats.
inline bool blit_unsupported_format(VkFormat format) {
    switch (format) {
    case VK_FORMAT_R8_UINT: case VK_FORMAT_R8_SINT: case VK_FORMAT_R8G8_UINT: case VK_FORMAT_R8G8_SINT:
    case VK_FORMAT_R8G8B8A8_UINT: case VK_FORMAT_R8G8B8A8_SINT: case VK_FORMAT_R16_UINT: case VK_FORMAT_R16_SINT:
    case VK_FORMAT_R16G16_UINT: case VK_FORMAT_R16G16_SINT: case VK_FORMAT_R16G16B16A16_UINT:
    case VK_FORMAT_R16G16B16A16_SINT: case VK_FORMAT_R32_UINT: case VK_FORMAT_R32_SINT: case VK_FORMAT_R32G32_UINT:
    case VK_FORMAT_R32G32_SINT: case VK_FORMAT_R32G32B32A32_UINT: case VK_FORMAT_R32G32B32A32_SINT:
    case VK_FORMAT_A2B10G10R10_UINT_PACK32: case VK_FORMAT_UNDEFINED:
        return true;
    default:
        return format >= VK_FORMAT_BC1_RGB_UNORM_BLOCK && format <= VK_FORMAT_BC7_SRGB_BLOCK;
    }
}
}  // namespace detail

class VulkanPresenter {
public:
    VulkanPresenter(VulkanContext& context, PFN_vkGetInstanceProcAddr get_instance, VkInstance instance,
                    VkPhysicalDevice physical, VkQueue queue, std::uint32_t family, VkSurfaceKHR surface)
        : context_(context), instance_(instance), physical_(physical), queue_(queue), surface_(surface) {
#define LEONIDA_INSTANCE_FN(name) name##_ = reinterpret_cast<PFN_##name>(get_instance(instance, #name));
        LEONIDA_INSTANCE_FN(vkGetPhysicalDeviceSurfaceSupportKHR)
        LEONIDA_INSTANCE_FN(vkGetPhysicalDeviceSurfaceCapabilitiesKHR)
        LEONIDA_INSTANCE_FN(vkGetPhysicalDeviceSurfaceFormatsKHR)
        LEONIDA_INSTANCE_FN(vkGetPhysicalDeviceSurfacePresentModesKHR)
        LEONIDA_INSTANCE_FN(vkDestroySurfaceKHR)
        LEONIDA_INSTANCE_FN(vkGetDeviceProcAddr)
        LEONIDA_INSTANCE_FN(vkGetPhysicalDeviceMemoryProperties)
#undef LEONIDA_INSTANCE_FN
        if (!vkGetDeviceProcAddr_ || !vkDestroySurfaceKHR_) return;
        const VkDevice device = context.device();
#define LEONIDA_DEVICE_FN(name) name##_ = reinterpret_cast<PFN_##name>(vkGetDeviceProcAddr_(device, #name));
        LEONIDA_DEVICE_FN(vkCreateSwapchainKHR)
        LEONIDA_DEVICE_FN(vkDestroySwapchainKHR)
        LEONIDA_DEVICE_FN(vkGetSwapchainImagesKHR)
        LEONIDA_DEVICE_FN(vkAcquireNextImageKHR)
        LEONIDA_DEVICE_FN(vkQueuePresentKHR)
        LEONIDA_DEVICE_FN(vkCmdBlitImage)
        LEONIDA_DEVICE_FN(vkCmdClearColorImage)
        LEONIDA_DEVICE_FN(vkCreateFence)
        LEONIDA_DEVICE_FN(vkDestroyFence)
        LEONIDA_DEVICE_FN(vkWaitForFences)
        LEONIDA_DEVICE_FN(vkResetFences)
#undef LEONIDA_DEVICE_FN
        if (!vkCreateSwapchainKHR_ || !vkQueuePresentKHR_ || !vkCmdBlitImage_ || !vkCreateFence_) return;

        VkBool32 supported = VK_FALSE;
        vkGetPhysicalDeviceSurfaceSupportKHR_(physical, family, surface, &supported);
        if (!supported) return;

        auto& vk = context.vk;
        VkCommandPoolCreateInfo pool{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        pool.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        pool.queueFamilyIndex = family;
        if (vk.vkCreateCommandPool(device, &pool, nullptr, &pool_) != VK_SUCCESS) return;
        for (Frame& frame : frames_) {
            VkCommandBufferAllocateInfo allocate{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
            allocate.commandPool = pool_;
            allocate.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            allocate.commandBufferCount = 1;
            VkFenceCreateInfo fence{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
            fence.flags = VK_FENCE_CREATE_SIGNALED_BIT;
            VkSemaphoreCreateInfo semaphore{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
            if (vk.vkAllocateCommandBuffers(device, &allocate, &frame.commands) != VK_SUCCESS ||
                vkCreateFence_(device, &fence, nullptr, &frame.done) != VK_SUCCESS ||
                vk.vkCreateSemaphore(device, &semaphore, nullptr, &frame.acquired) != VK_SUCCESS)
                return;
        }
        ok_ = recreate(true);
    }

    ~VulkanPresenter() {
        auto& vk = context_.vk;
        const VkDevice device = context_.device();
        {
            std::lock_guard lock(context_.submit_mutex());
            vk.vkQueueWaitIdle(queue_);
        }
        release(swapchain_);
        swapchain_ = VK_NULL_HANDLE;
        release_capture();
        for (Frame& frame : frames_) {
            if (frame.done) vkDestroyFence_(device, frame.done, nullptr);
            if (frame.acquired) vk.vkDestroySemaphore(device, frame.acquired, nullptr);
        }
        if (pool_) vk.vkDestroyCommandPool(device, pool_, nullptr);
        if (surface_ && vkDestroySurfaceKHR_) vkDestroySurfaceKHR_(instance_, surface_, nullptr);
    }

    VulkanPresenter(const VulkanPresenter&) = delete;
    VulkanPresenter& operator=(const VulkanPresenter&) = delete;

    bool ok() const { return ok_; }
    VkExtent2D extent() const { return extent_; }
    VkFormat format() const { return format_; }
    std::uint64_t presented() const { return presented_; }

    // Blits the colour texture `source` (mip 0, layer 0, in D3D12's PRESENT state) to the window and presents it.
    // `source_rect`, when given, limits the blit to that part of the texture (stretched to the whole window); it
    // must be non-empty and inside the texture (VK_ERROR_UNKNOWN otherwise). Integer, block-compressed and
    // multisampled sources are refused (VK_ERROR_FORMAT_NOT_SUPPORTED); the other formats are converted by the blit.
    // The blit filters bilinearly, so with a rectangle the outermost texels blend with their neighbours outside it.
    // vsync: FIFO; otherwise MAILBOX or IMMEDIATE when the surface has them. VK_SUCCESS also covers a skipped
    // present (minimised window) and one the surface reported out of date (the swapchain is recreated next time);
    // VK_ERROR_OUT_OF_DATE_KHR means recreating the swapchain failed, and the next present tries again.
    //
    // A failure after an image was acquired (the blit's submission) is recovered here, so the next present does not
    // wait on a fence nothing will signal: an empty batch takes the acquire semaphore and signals the frame's fence,
    // and the swapchain is recreated next time, which gives back the image that was never presented. If even that
    // batch fails, the presenter stops (ok() is false) and its owner has to make a new one.
    VkResult present(ResourceId source, bool vsync, const PresentRect* source_rect = nullptr) {
        ResourceBinding binding;
        {
            std::lock_guard lock(context_.mutex_);
            const ResourceBinding* found = context_.resolve(source);
            if (!found || !found->image || found->aspects != VK_IMAGE_ASPECT_COLOR_BIT) return VK_ERROR_UNKNOWN;
            binding = *found;
        }
        if (binding.samples > 1 || detail::blit_unsupported_format(binding.format))
            return VK_ERROR_FORMAT_NOT_SUPPORTED;
        if (source_rect && (source_rect->left < 0 || source_rect->top < 0 || source_rect->left >= source_rect->right ||
                            source_rect->top >= source_rect->bottom ||
                            std::uint32_t(source_rect->right) > binding.width ||
                            std::uint32_t(source_rect->bottom) > binding.height))
            return VK_ERROR_UNKNOWN;
        return submit(&binding, source, vsync, source_rect, detail::is_srgb_format(binding.format));
    }

    // Presents a black frame: the swapchain image cleared, nothing read. PresentX with no planes.
    VkResult present_blank(bool vsync) { return submit(nullptr, ResourceId{}, vsync, nullptr, want_srgb_); }

    // From now on every presented frame is also copied to host memory (the swapchain gets TRANSFER_SRC usage), for
    // read_back(). False when the surface cannot do it or its format is not 8 bits per channel with 4 channels.
    // Call before the first present.
    bool enable_capture() {
        if (!ok_) return false;
        capture_ = true;
        stale_ = true;
        if (!recreate(vsync_) || (swapchain_ && !capture_buffer_)) {
            capture_ = false;
            return false;
        }
        return true;
    }

    // The last presented frame as the swapchain stores it (4 bytes a texel, rows tightly packed, format() order).
    // Waits for the frame to finish. False when nothing was presented since enable_capture().
    bool read_back(std::vector<std::uint8_t>& pixels, VkExtent2D* extent, VkFormat* format) {
        if (!capture_ || !capture_ptr_ || !captured_) return false;
        const Frame& frame = frames_[(frame_index_ + frames_.size() - 1) % frames_.size()];
        vkWaitForFences_(context_.device(), 1, &frame.done, VK_TRUE, UINT64_MAX);
        pixels.assign(capture_ptr_, capture_ptr_ + std::size_t(captured_extent_.width) * captured_extent_.height * 4);
        if (extent) *extent = captured_extent_;
        if (format) *format = captured_format_;
        return true;
    }

    // Shared by present() and present_blank(). `binding` null: clear instead of blit.
    VkResult submit(const ResourceBinding* binding, ResourceId source, bool vsync, const PresentRect* source_rect,
                    bool srgb) {
        if (!ok_) return VK_ERROR_INITIALIZATION_FAILED;
        if (vsync != vsync_ || stale_ || srgb != want_srgb_) {
            want_srgb_ = srgb;
            if (!recreate(vsync)) return VK_ERROR_OUT_OF_DATE_KHR;
        }
        if (!swapchain_) {  // minimised: try again next frame
            stale_ = true;
            return VK_SUCCESS;
        }
        if (binding && !presentable(source)) return VK_ERROR_UNKNOWN;  // checked before an image is acquired for it
        auto& vk = context_.vk;
        const VkDevice device = context_.device();
        Frame& frame = frames_[frame_index_ % frames_.size()];
        if (const VkResult waited = vkWaitForFences_(device, 1, &frame.done, VK_TRUE, UINT64_MAX); waited != VK_SUCCESS)
            return stop(waited);

        std::uint32_t image = 0;
        VkResult acquired = vkAcquireNextImageKHR_(device, swapchain_, UINT64_MAX, frame.acquired, VK_NULL_HANDLE, &image);
        if (acquired == VK_ERROR_OUT_OF_DATE_KHR) {
            if (!recreate(vsync)) return acquired;
            if (!swapchain_) return VK_SUCCESS;
            acquired = vkAcquireNextImageKHR_(device, swapchain_, UINT64_MAX, frame.acquired, VK_NULL_HANDLE, &image);
        }
        if (acquired != VK_SUCCESS && acquired != VK_SUBOPTIMAL_KHR) return acquired;  // nothing was acquired
        if (acquired == VK_SUBOPTIMAL_KHR) stale_ = true;

        bool recorded = false;
        if (binding) {
            // The back buffer's layout is tracked like any other texture; this holds the context while it is read.
            std::lock_guard lock(context_.mutex_);
            const ResourceBinding* current = context_.resolve(source);
            if (current && current->image && current->aspects == VK_IMAGE_ASPECT_COLOR_BIT) {
                record(frame.commands, *current, source, image, source_rect);
                recorded = true;
            }
        } else {
            record_blank(frame.commands, image);
            recorded = true;
        }
        if (!recorded) return give_back(frame, VK_ERROR_UNKNOWN);  // released by another thread since presentable()
        if (const VkResult reset = vkResetFences_(device, 1, &frame.done); reset != VK_SUCCESS) return stop(reset);
        const VkPipelineStageFlags wait_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
        VkSubmitInfo submit_info{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        submit_info.waitSemaphoreCount = 1;
        submit_info.pWaitSemaphores = &frame.acquired;
        submit_info.pWaitDstStageMask = &wait_stage;
        submit_info.commandBufferCount = 1;
        submit_info.pCommandBuffers = &frame.commands;
        submit_info.signalSemaphoreCount = 1;
        submit_info.pSignalSemaphores = &ready_[image];  // one per image: reused only once that image comes back
        VkPresentInfoKHR present{VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
        present.waitSemaphoreCount = 1;
        present.pWaitSemaphores = &ready_[image];
        present.swapchainCount = 1;
        present.pSwapchains = &swapchain_;
        present.pImageIndices = &image;
        VkResult submitted, result = VK_SUCCESS;
        {
            std::lock_guard lock(context_.submit_mutex());
            submitted = vk.vkQueueSubmit(queue_, 1, &submit_info, frame.done);
            if (submitted == VK_SUCCESS) result = vkQueuePresentKHR_(queue_, &present);
        }
        if (submitted != VK_SUCCESS) return give_back(frame, submitted);
        ++frame_index_;
        // Only a present the engine took counts (SUBOPTIMAL is shown too). OUT_OF_DATE was not shown: the swapchain
        // is recreated next time, and the caller sees VK_SUCCESS, since its work was submitted.
        if (result == VK_SUCCESS || result == VK_SUBOPTIMAL_KHR) ++presented_;
        if (capture_ && capture_ptr_) {
            captured_ = true;
            captured_extent_ = extent_;
            captured_format_ = format_;
        }
        if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR) {
            stale_ = true;
            result = VK_SUCCESS;
        }
        return result;
    }

private:
    struct Frame {
        VkCommandBuffer commands = VK_NULL_HANDLE;
        VkFence done = VK_NULL_HANDLE;
        VkSemaphore acquired = VK_NULL_HANDLE;
    };

    // Whether `source` is a colour texture the device knows, so present() can refuse it before acquiring an image.
    bool presentable(ResourceId source) {
        std::lock_guard lock(context_.mutex_);
        const ResourceBinding* binding = context_.resolve(source);
        return binding && binding->image && binding->aspects == VK_IMAGE_ASPECT_COLOR_BIT;
    }

    // An image was acquired into `frame` but its blit was not submitted. An empty batch waits on the acquire
    // semaphore and signals the frame's fence, so both are usable again; the swapchain is recreated on the next
    // present, which gives back the image that was never presented. Returns `failure`, the error to report.
    VkResult give_back(Frame& frame, VkResult failure) {
        const VkDevice device = context_.device();
        if (vkResetFences_(device, 1, &frame.done) != VK_SUCCESS) return stop(failure);
        const VkPipelineStageFlags wait_stage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
        VkSubmitInfo submit_info{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        submit_info.waitSemaphoreCount = 1;
        submit_info.pWaitSemaphores = &frame.acquired;
        submit_info.pWaitDstStageMask = &wait_stage;
        VkResult submitted;
        {
            std::lock_guard lock(context_.submit_mutex());
            submitted = context_.vk.vkQueueSubmit(queue_, 1, &submit_info, frame.done);
        }
        if (submitted != VK_SUCCESS) return stop(failure);
        stale_ = true;
        return failure;
    }

    // The swapchain image as a binding, for the barrier helpers.
    ResourceBinding target_of(std::uint32_t image) const {
        ResourceBinding target{};
        target.image = images_[image];
        target.format = format_;
        target.aspects = VK_IMAGE_ASPECT_COLOR_BIT;
        return target;
    }

    // The frame's fence or acquire semaphore can no longer be trusted: no more presents (ok() is false), and the
    // owner makes a new presenter. Returns `failure`.
    VkResult stop(VkResult failure) {
        ok_ = false;
        return failure;
    }

    void record(VkCommandBuffer commands, const ResourceBinding& binding, ResourceId source, std::uint32_t image,
                const PresentRect* source_rect) {
        auto& vk = context_.vk;
        const ReplayFunctions& fn = context_.replay_;
        vk.vkResetCommandBuffer(commands, 0);
        VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vk.vkBeginCommandBuffer(commands, &begin);

        constexpr VkAccessFlags any = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
        const VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        // PRESENT is D3D12's common state: GENERAL for an ordinary texture. A never-written back buffer is
        // zero-filled first, as any committed resource is.
        const VkImageLayout common = image_layout(binding, VK_IMAGE_LAYOUT_GENERAL);
        auto pieces = detail::current(&context_.tracker_, source, range, common);
        detail::initialize(fn, commands, binding, source, pieces, common, &context_.tracker_);
        for (const auto& piece : pieces)
            detail::image_barrier(fn, commands, binding, piece.range, piece.layout,
                                  VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_ACCESS_MEMORY_WRITE_BIT,
                                  VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_ACCESS_TRANSFER_READ_BIT,
                                  VK_PIPELINE_STAGE_TRANSFER_BIT);
        ResourceBinding target{};  // the swapchain image, for the barrier helpers
        target.image = images_[image];
        target.format = format_;
        target.aspects = VK_IMAGE_ASPECT_COLOR_BIT;
        // The acquire semaphore is waited at the transfer stage, so the swapchain image's barrier starts there.
        detail::image_barrier(fn, commands, target, range, VK_IMAGE_LAYOUT_UNDEFINED,
                              VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 0, VK_PIPELINE_STAGE_TRANSFER_BIT,
                              VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);

        const PresentRect whole{0, 0, std::int32_t(binding.width), std::int32_t(binding.height)};
        const PresentRect& src = source_rect ? *source_rect : whole;
        const std::int32_t sw = src.right - src.left, sh = src.bottom - src.top;
        const std::int32_t dw = std::int32_t(extent_.width), dh = std::int32_t(extent_.height);
        VkImageBlit blit{};
        blit.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        blit.srcOffsets[0] = {src.left, src.top, 0};
        blit.srcOffsets[1] = {src.right, src.bottom, 1};
        blit.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        blit.dstOffsets[1] = {dw, dh, 1};
        vkCmdBlitImage_(commands, binding.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, images_[image],
                        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit,
                        sw == dw && sh == dh ? VK_FILTER_NEAREST : VK_FILTER_LINEAR);

        finish_image(commands, image);
        for (const auto& piece : pieces)
            detail::image_barrier(fn, commands, binding, piece.range, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                  piece.layout, VK_ACCESS_TRANSFER_READ_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, any,
                                  VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
        vk.vkEndCommandBuffer(commands);
    }

    // Clears the swapchain image to opaque black.
    void record_blank(VkCommandBuffer commands, std::uint32_t image) {
        auto& vk = context_.vk;
        const ReplayFunctions& fn = context_.replay_;
        vk.vkResetCommandBuffer(commands, 0);
        VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vk.vkBeginCommandBuffer(commands, &begin);
        const VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        detail::image_barrier(fn, commands, target_of(image), range, VK_IMAGE_LAYOUT_UNDEFINED,
                              VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 0, VK_PIPELINE_STAGE_TRANSFER_BIT,
                              VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
        const VkClearColorValue black{{0.0f, 0.0f, 0.0f, 1.0f}};
        vkCmdClearColorImage_(commands, images_[image], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &black, 1, &range);
        finish_image(commands, image);
        vk.vkEndCommandBuffer(commands);
    }

    // The swapchain image is written (TRANSFER_DST): optionally copy it to the capture buffer, then hand it to
    // the presentation engine.
    void finish_image(VkCommandBuffer commands, std::uint32_t image) {
        const ReplayFunctions& fn = context_.replay_;
        const VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        VkImageLayout layout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        VkAccessFlags access = VK_ACCESS_TRANSFER_WRITE_BIT;
        if (capture_ && capture_buffer_) {
            detail::image_barrier(fn, commands, target_of(image), range, layout, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                  access, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_READ_BIT,
                                  VK_PIPELINE_STAGE_TRANSFER_BIT);
            VkBufferImageCopy copy{};
            copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
            copy.imageExtent = {extent_.width, extent_.height, 1};
            fn.copy_image_to_buffer(commands, images_[image], VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, capture_buffer_, 1,
                                    &copy);
            layout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
            access = VK_ACCESS_TRANSFER_READ_BIT;
        }
        detail::image_barrier(fn, commands, target_of(image), range, layout, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, access,
                              VK_PIPELINE_STAGE_TRANSFER_BIT, 0, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT);
    }

    // (Re)creates the capture buffer for the current swapchain size: host visible, coherent, persistently mapped.
    bool make_capture_buffer() {
        release_capture();
        auto& vk = context_.vk;
        const VkDevice device = context_.device();
        capture_size_ = VkDeviceSize(extent_.width) * extent_.height * 4;
        VkBufferCreateInfo info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        info.size = capture_size_;
        info.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        if (vk.vkCreateBuffer(device, &info, nullptr, &capture_buffer_) != VK_SUCCESS) return false;
        VkMemoryRequirements requirements{};
        vk.vkGetBufferMemoryRequirements(device, capture_buffer_, &requirements);
        VkPhysicalDeviceMemoryProperties memory{};
        vkGetPhysicalDeviceMemoryProperties_(physical_, &memory);
        constexpr VkMemoryPropertyFlags wanted = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        std::uint32_t type = UINT32_MAX;
        for (std::uint32_t i = 0; i < memory.memoryTypeCount; ++i)
            if ((requirements.memoryTypeBits & (1u << i)) && (memory.memoryTypes[i].propertyFlags & wanted) == wanted) {
                type = i;
                break;
            }
        VkMemoryAllocateInfo allocate{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        allocate.allocationSize = requirements.size;
        allocate.memoryTypeIndex = type;
        void* mapped = nullptr;
        if (type == UINT32_MAX || vk.vkAllocateMemory(device, &allocate, nullptr, &capture_memory_) != VK_SUCCESS ||
            vk.vkBindBufferMemory(device, capture_buffer_, capture_memory_, 0) != VK_SUCCESS ||
            vk.vkMapMemory(device, capture_memory_, 0, VK_WHOLE_SIZE, 0, &mapped) != VK_SUCCESS) {
            release_capture();
            return false;
        }
        capture_ptr_ = static_cast<std::uint8_t*>(mapped);
        captured_ = false;
        return true;
    }

    void release_capture() {
        auto& vk = context_.vk;
        const VkDevice device = context_.device();
        if (capture_memory_) vk.vkFreeMemory(device, capture_memory_, nullptr);  // unmaps
        if (capture_buffer_) vk.vkDestroyBuffer(device, capture_buffer_, nullptr);
        capture_memory_ = VK_NULL_HANDLE;
        capture_buffer_ = VK_NULL_HANDLE;
        capture_ptr_ = nullptr;
        captured_ = false;
    }

    // (Re)creates the swapchain for the surface's current size. A zero-sized surface (minimised) leaves no
    // swapchain and is retried on the next present. Nothing is published until every piece exists: on any failure
    // the presenter is left with no swapchain and stale_ set, so the next present tries again, and no image is ever
    // acquired without its per-image semaphore.
    bool recreate(bool vsync) {
        auto& vk = context_.vk;
        const VkDevice device = context_.device();
        {
            std::lock_guard lock(context_.submit_mutex());
            vk.vkQueueWaitIdle(queue_);  // the old swapchain's images and semaphores may still be in use
        }
        stale_ = true;  // until this succeeds
        VkSurfaceCapabilitiesKHR caps{};
        if (vkGetPhysicalDeviceSurfaceCapabilitiesKHR_(physical_, surface_, &caps) != VK_SUCCESS) return false;
        VkSwapchainKHR old = swapchain_;
        swapchain_ = VK_NULL_HANDLE;
        if (caps.currentExtent.width == 0 || caps.currentExtent.height == 0) {
            release(old);
            vsync_ = vsync;
            stale_ = false;
            return true;
        }
        if (!(caps.supportedUsageFlags & VK_IMAGE_USAGE_TRANSFER_DST_BIT)) {
            release(old);
            return false;
        }

        std::uint32_t count = 0;
        vkGetPhysicalDeviceSurfaceFormatsKHR_(physical_, surface_, &count, nullptr);
        std::vector<VkSurfaceFormatKHR> formats(count);
        vkGetPhysicalDeviceSurfaceFormatsKHR_(physical_, surface_, &count, formats.data());
        if (formats.empty()) return false;
        // An sRGB source wants an sRGB swapchain (the blit decodes and encodes, which cancel); any other source
        // wants UNORM (the bytes go through as they are). BGRA before RGBA, as the surface lists them.
        VkSurfaceFormatKHR chosen = formats[0];
        bool found = false;
        for (const VkFormat wanted : want_srgb_ ? std::array<VkFormat, 2>{VK_FORMAT_B8G8R8A8_SRGB, VK_FORMAT_R8G8B8A8_SRGB}
                                                : std::array<VkFormat, 2>{VK_FORMAT_B8G8R8A8_UNORM, VK_FORMAT_R8G8B8A8_UNORM}) {
            for (const auto& f : formats)
                if (f.format == wanted && f.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
                    chosen = f;
                    found = true;
                    break;
                }
            if (found) break;
        }
        if (!found && want_srgb_) {
            if (!warned_srgb_) std::fprintf(stderr, "[rt] present: the surface has no sRGB 8-bit format; an sRGB back buffer is shown darker\n");
            warned_srgb_ = true;
            for (const auto& f : formats)
                if ((f.format == VK_FORMAT_B8G8R8A8_UNORM || f.format == VK_FORMAT_R8G8B8A8_UNORM) &&
                    f.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
                    chosen = f;
                    break;
                }
        }
        vkGetPhysicalDeviceSurfacePresentModesKHR_(physical_, surface_, &count, nullptr);
        std::vector<VkPresentModeKHR> modes(count);
        vkGetPhysicalDeviceSurfacePresentModesKHR_(physical_, surface_, &count, modes.data());
        VkPresentModeKHR mode = VK_PRESENT_MODE_FIFO_KHR;  // always available
        if (!vsync) {
            for (const VkPresentModeKHR wanted : {VK_PRESENT_MODE_MAILBOX_KHR, VK_PRESENT_MODE_IMMEDIATE_KHR})
                if (std::find(modes.begin(), modes.end(), wanted) != modes.end()) {
                    mode = wanted;
                    break;
                }
        }

        VkSwapchainCreateInfoKHR info{VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR};
        info.surface = surface_;
        info.minImageCount = std::max(caps.minImageCount + 1, 2u);
        if (caps.maxImageCount) info.minImageCount = std::min(info.minImageCount, caps.maxImageCount);
        info.imageFormat = chosen.format;
        info.imageColorSpace = chosen.colorSpace;
        info.imageExtent = caps.currentExtent;
        info.imageArrayLayers = 1;
        info.imageUsage = VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        if (capture_) {
            if (!(caps.supportedUsageFlags & VK_IMAGE_USAGE_TRANSFER_SRC_BIT)) return false;
            info.imageUsage |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
        }
        info.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
        info.preTransform = caps.currentTransform;
        info.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
        if (!(caps.supportedCompositeAlpha & info.compositeAlpha)) info.compositeAlpha = VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR;
        info.presentMode = mode;
        info.clipped = VK_TRUE;
        info.oldSwapchain = old;
        VkSwapchainKHR fresh = VK_NULL_HANDLE;
        const VkResult created = vkCreateSwapchainKHR_(device, &info, nullptr, &fresh);
        release(old);  // retired by the creation attempt, whether or not it succeeded
        if (created != VK_SUCCESS) return false;

        std::vector<VkImage> images;
        std::vector<VkSemaphore> ready;
        const auto discard = [&] {
            for (VkSemaphore semaphore : ready) vk.vkDestroySemaphore(device, semaphore, nullptr);
            vkDestroySwapchainKHR_(device, fresh, nullptr);
            return false;
        };
        if (vkGetSwapchainImagesKHR_(device, fresh, &count, nullptr) != VK_SUCCESS || !count) return discard();
        images.resize(count);
        if (vkGetSwapchainImagesKHR_(device, fresh, &count, images.data()) != VK_SUCCESS) return discard();
        images.resize(count);
        for (std::uint32_t i = 0; i < count; ++i) {
            VkSemaphoreCreateInfo semaphore_info{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
            VkSemaphore semaphore = VK_NULL_HANDLE;
            if (vk.vkCreateSemaphore(device, &semaphore_info, nullptr, &semaphore) != VK_SUCCESS) return discard();
            ready.push_back(semaphore);
        }
        swapchain_ = fresh;
        images_ = std::move(images);
        ready_ = std::move(ready);
        format_ = chosen.format;
        extent_ = caps.currentExtent;
        if (capture_) {
            // A capture that cannot be made fails the recreation with the swapchain in place and stale_ still set.
            const bool four_bytes = format_ == VK_FORMAT_B8G8R8A8_UNORM || format_ == VK_FORMAT_R8G8B8A8_UNORM ||
                                    format_ == VK_FORMAT_B8G8R8A8_SRGB || format_ == VK_FORMAT_R8G8B8A8_SRGB;
            if (!four_bytes || !make_capture_buffer()) return false;
        }
        vsync_ = vsync;
        stale_ = false;
        return true;
    }

    // Destroys a swapchain that is being replaced (or the last one), with the per-image semaphores that went with it.
    void release(VkSwapchainKHR swapchain) {
        const VkDevice device = context_.device();
        for (VkSemaphore semaphore : ready_)
            if (semaphore) context_.vk.vkDestroySemaphore(device, semaphore, nullptr);
        ready_.clear();
        images_.clear();
        if (swapchain) vkDestroySwapchainKHR_(device, swapchain, nullptr);
    }

    VulkanContext& context_;
    VkInstance instance_;
    VkPhysicalDevice physical_;
    VkQueue queue_;
    VkSurfaceKHR surface_;
    VkSwapchainKHR swapchain_ = VK_NULL_HANDLE;
    VkFormat format_ = VK_FORMAT_UNDEFINED;
    VkExtent2D extent_{};
    std::vector<VkImage> images_;
    std::vector<VkSemaphore> ready_;  // per swapchain image: blit done, present may start
    std::array<Frame, 2> frames_{};
    VkCommandPool pool_ = VK_NULL_HANDLE;
    std::uint64_t frame_index_ = 0;
    std::uint64_t presented_ = 0;
    bool vsync_ = true;
    bool stale_ = false;
    bool ok_ = false;
    bool want_srgb_ = false;   // the swapchain format follows the source being presented
    bool warned_srgb_ = false;
    bool capture_ = false;
    VkBuffer capture_buffer_ = VK_NULL_HANDLE;
    VkDeviceMemory capture_memory_ = VK_NULL_HANDLE;
    VkDeviceSize capture_size_ = 0;
    std::uint8_t* capture_ptr_ = nullptr;
    bool captured_ = false;
    VkExtent2D captured_extent_{};
    VkFormat captured_format_ = VK_FORMAT_UNDEFINED;

    PFN_vkGetPhysicalDeviceSurfaceSupportKHR vkGetPhysicalDeviceSurfaceSupportKHR_ = nullptr;
    PFN_vkGetPhysicalDeviceSurfaceCapabilitiesKHR vkGetPhysicalDeviceSurfaceCapabilitiesKHR_ = nullptr;
    PFN_vkGetPhysicalDeviceSurfaceFormatsKHR vkGetPhysicalDeviceSurfaceFormatsKHR_ = nullptr;
    PFN_vkGetPhysicalDeviceSurfacePresentModesKHR vkGetPhysicalDeviceSurfacePresentModesKHR_ = nullptr;
    PFN_vkDestroySurfaceKHR vkDestroySurfaceKHR_ = nullptr;
    PFN_vkGetDeviceProcAddr vkGetDeviceProcAddr_ = nullptr;
    PFN_vkCreateSwapchainKHR vkCreateSwapchainKHR_ = nullptr;
    PFN_vkDestroySwapchainKHR vkDestroySwapchainKHR_ = nullptr;
    PFN_vkGetSwapchainImagesKHR vkGetSwapchainImagesKHR_ = nullptr;
    PFN_vkAcquireNextImageKHR vkAcquireNextImageKHR_ = nullptr;
    PFN_vkQueuePresentKHR vkQueuePresentKHR_ = nullptr;
    PFN_vkCmdBlitImage vkCmdBlitImage_ = nullptr;
    PFN_vkCmdClearColorImage vkCmdClearColorImage_ = nullptr;
    PFN_vkGetPhysicalDeviceMemoryProperties vkGetPhysicalDeviceMemoryProperties_ = nullptr;
    PFN_vkCreateFence vkCreateFence_ = nullptr;
    PFN_vkDestroyFence vkDestroyFence_ = nullptr;
    PFN_vkWaitForFences vkWaitForFences_ = nullptr;
    PFN_vkResetFences vkResetFences_ = nullptr;
};

} // namespace leonida
