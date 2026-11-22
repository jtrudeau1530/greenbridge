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

#include "vulkan_backend.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <vector>

namespace leonida {

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
    // vsync: FIFO; otherwise MAILBOX or IMMEDIATE when the surface has them. VK_SUCCESS also covers a skipped
    // present (minimised window) and one the surface reported out of date (the swapchain is recreated next time);
    // VK_ERROR_OUT_OF_DATE_KHR means recreating the swapchain failed, and the next present tries again.
    //
    // A failure after an image was acquired (the blit's submission) is recovered here, so the next present does not
    // wait on a fence nothing will signal: an empty batch takes the acquire semaphore and signals the frame's fence,
    // and the swapchain is recreated next time, which gives back the image that was never presented. If even that
    // batch fails, the presenter stops (ok() is false) and its owner has to make a new one.
    VkResult present(ResourceId source, bool vsync) {
        if (!ok_) return VK_ERROR_INITIALIZATION_FAILED;
        if (vsync != vsync_ || stale_) {
            if (!recreate(vsync)) return VK_ERROR_OUT_OF_DATE_KHR;
        }
        if (!swapchain_) {  // minimised: try again next frame
            stale_ = true;
            return VK_SUCCESS;
        }
        if (!presentable(source)) return VK_ERROR_UNKNOWN;  // checked before an image is acquired for it
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
        {
            // The back buffer's layout is tracked like any other texture: the resource lock while it is resolved,
            // then the tracker's, in the context's order (resource_mutex_ -> tracker_mutex_ -> submit_mutex_).
            std::shared_lock resource_lock(context_.resource_mutex_);
            std::lock_guard tracker_lock(context_.tracker_mutex_);
            const ResourceBinding* binding = context_.resolve(source);
            if (binding && binding->image && binding->aspects == VK_IMAGE_ASPECT_COLOR_BIT) {
                record(frame.commands, *binding, source, image);
                recorded = true;
            }
        }
        if (!recorded) return give_back(frame, VK_ERROR_UNKNOWN);  // released by another thread since presentable()
        if (const VkResult reset = vkResetFences_(device, 1, &frame.done); reset != VK_SUCCESS) return stop(reset);
        const VkPipelineStageFlags wait_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
        VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        submit.waitSemaphoreCount = 1;
        submit.pWaitSemaphores = &frame.acquired;
        submit.pWaitDstStageMask = &wait_stage;
        submit.commandBufferCount = 1;
        submit.pCommandBuffers = &frame.commands;
        submit.signalSemaphoreCount = 1;
        submit.pSignalSemaphores = &ready_[image];  // one per image: reused only once that image comes back
        VkPresentInfoKHR present{VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
        present.waitSemaphoreCount = 1;
        present.pWaitSemaphores = &ready_[image];
        present.swapchainCount = 1;
        present.pSwapchains = &swapchain_;
        present.pImageIndices = &image;
        VkResult submitted, result = VK_SUCCESS;
        {
            std::lock_guard lock(context_.submit_mutex());
            submitted = vk.vkQueueSubmit(queue_, 1, &submit, frame.done);
            if (submitted == VK_SUCCESS) result = vkQueuePresentKHR_(queue_, &present);
        }
        if (submitted != VK_SUCCESS) return give_back(frame, submitted);
        ++frame_index_;
        // Only a present the engine took counts (SUBOPTIMAL is shown too). OUT_OF_DATE was not shown: the swapchain
        // is recreated next time, and the caller sees VK_SUCCESS, since its work was submitted.
        if (result == VK_SUCCESS || result == VK_SUBOPTIMAL_KHR) ++presented_;
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
        std::shared_lock lock(context_.resource_mutex_);
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
        VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        submit.waitSemaphoreCount = 1;
        submit.pWaitSemaphores = &frame.acquired;
        submit.pWaitDstStageMask = &wait_stage;
        VkResult submitted;
        {
            std::lock_guard lock(context_.submit_mutex());
            submitted = context_.vk.vkQueueSubmit(queue_, 1, &submit, frame.done);
        }
        if (submitted != VK_SUCCESS) return stop(failure);
        stale_ = true;
        return failure;
    }

    // The frame's fence or acquire semaphore can no longer be trusted: no more presents (ok() is false), and the
    // owner makes a new presenter. Returns `failure`.
    VkResult stop(VkResult failure) {
        ok_ = false;
        return failure;
    }

    void record(VkCommandBuffer commands, const ResourceBinding& binding, ResourceId source, std::uint32_t image) {
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

        const std::int32_t sw = std::int32_t(binding.width), sh = std::int32_t(binding.height);
        const std::int32_t dw = std::int32_t(extent_.width), dh = std::int32_t(extent_.height);
        VkImageBlit blit{};
        blit.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        blit.srcOffsets[1] = {sw, sh, 1};
        blit.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        blit.dstOffsets[1] = {dw, dh, 1};
        vkCmdBlitImage_(commands, binding.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, images_[image],
                        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit,
                        sw == dw && sh == dh ? VK_FILTER_NEAREST : VK_FILTER_LINEAR);

        detail::image_barrier(fn, commands, target, range, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                              VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, VK_ACCESS_TRANSFER_WRITE_BIT,
                              VK_PIPELINE_STAGE_TRANSFER_BIT, 0, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT);
        for (const auto& piece : pieces)
            detail::image_barrier(fn, commands, binding, piece.range, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                  piece.layout, VK_ACCESS_TRANSFER_READ_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, any,
                                  VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
        vk.vkEndCommandBuffer(commands);
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
        VkSurfaceFormatKHR chosen = formats[0];
        for (const auto& f : formats)
            if ((f.format == VK_FORMAT_B8G8R8A8_UNORM || f.format == VK_FORMAT_R8G8B8A8_UNORM) &&
                f.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
                chosen = f;
                break;
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
    PFN_vkCreateFence vkCreateFence_ = nullptr;
    PFN_vkDestroyFence vkDestroyFence_ = nullptr;
    PFN_vkWaitForFences vkWaitForFences_ = nullptr;
    PFN_vkResetFences vkResetFences_ = nullptr;
};

} // namespace leonida
