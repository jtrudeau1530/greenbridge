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
//
// Tearing on request (PresentX's ImmediateThresholdPercent): with VK_KHR_swapchain_maintenance1 (or the EXT original)
// the vsync swapchain is created with FIFO plus IMMEDIATE (FIFO_RELAXED when the surface cannot switch to IMMEDIATE),
// and a present that asks for it is queued in that mode alone. Without the extension, or without either mode, a
// request is ignored and the frame waits for the vblank.

#include "vulkan_backend.h"

// Use the original EXT spellings. They are aliases of the promoted KHR types and are available in older headers.
#if defined(VK_EXT_SWAPCHAIN_MAINTENANCE_1_EXTENSION_NAME) && defined(VK_EXT_SURFACE_MAINTENANCE_1_EXTENSION_NAME)
#define LEONIDA_SWAPCHAIN_MAINTENANCE1 1
#endif

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <map>
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
                    VkPhysicalDevice physical, VkQueue queue, std::uint32_t family, VkSurfaceKHR surface,
                    bool swapchain_maintenance1 = false,
                    std::function<VkExtent2D()> extent_fn = nullptr)
        : context_(context), instance_(instance), physical_(physical), queue_(queue), surface_(surface),
          maintenance1_(swapchain_maintenance1), extent_fn_(std::move(extent_fn)) {
#define LEONIDA_INSTANCE_FN(name) name##_ = reinterpret_cast<PFN_##name>(get_instance(instance, #name));
        LEONIDA_INSTANCE_FN(vkGetPhysicalDeviceSurfaceSupportKHR)
        LEONIDA_INSTANCE_FN(vkGetPhysicalDeviceSurfaceCapabilitiesKHR)
        LEONIDA_INSTANCE_FN(vkGetPhysicalDeviceSurfaceFormatsKHR)
        LEONIDA_INSTANCE_FN(vkGetPhysicalDeviceSurfacePresentModesKHR)
        LEONIDA_INSTANCE_FN(vkGetPhysicalDeviceSurfaceCapabilities2KHR)
        LEONIDA_INSTANCE_FN(vkDestroySurfaceKHR)
        LEONIDA_INSTANCE_FN(vkGetDeviceProcAddr)
        LEONIDA_INSTANCE_FN(vkGetPhysicalDeviceMemoryProperties)
        LEONIDA_INSTANCE_FN(vkGetPhysicalDeviceFormatProperties)
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
        ok_ = recreate(true, false);
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
    // The present mode a tearing present uses (IMMEDIATE, else FIFO_RELAXED) while the current swapchain was created
    // for it, else VK_PRESENT_MODE_MAX_ENUM_KHR: ImmediateThresholdPercent cannot be honoured then.
    VkPresentModeKHR tear_mode() const { return tear_mode_; }
    bool tear_supported() const { return tear_mode_ != VK_PRESENT_MODE_MAX_ENUM_KHR; }
    // Whether the last present was queued in the tearing mode, and how many were.
    bool last_present_tore() const { return last_tore_; }
    std::uint64_t tear_presents() const { return tear_presents_; }

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
    //
    // `tear` (only with vsync): queue the present in tear_mode() instead of FIFO when the swapchain can, so a frame
    // that missed its vblank is shown at once; last_present_tore() says whether it was.
    VkResult present(ResourceId source, bool vsync, const PresentRect* source_rect = nullptr, bool tear = false) {
        ResourceBinding binding;
        {
            std::shared_lock lock(context_.resource_mutex_);  // the resource table (vulkan_backend.h)
            const ResourceBinding* found = context_.resolve(source);
            if (!found || !found->image || found->aspects != VK_IMAGE_ASPECT_COLOR_BIT) return VK_ERROR_UNKNOWN;
            binding = *found;
        }
        // Multisampled sources (no resolve into a temporary yet), integer formats (a UNORM swapchain cannot take them)
        // and anything the GPU cannot blit from are refused.
        if (binding.samples > 1 || detail::blit_unsupported_format(binding.format) ||
            !(format_features(binding.format) & VK_FORMAT_FEATURE_BLIT_SRC_BIT))
            return VK_ERROR_FORMAT_NOT_SUPPORTED;
        if (source_rect && (source_rect->left < 0 || source_rect->top < 0 || source_rect->left >= source_rect->right ||
                            source_rect->top >= source_rect->bottom ||
                            std::uint32_t(source_rect->right) > binding.width ||
                            std::uint32_t(source_rect->bottom) > binding.height))
            return VK_ERROR_UNKNOWN;
        return submit(&binding, source, vsync, source_rect, detail::is_srgb_format(binding.format), tear);
    }

    // Polled while a present waits for a swapchain image (every 50 ms): when it returns true the present gives up with
    // VK_NOT_READY, so a caller that is being suspended is never held by a window that stopped taking frames.
    void set_abort(std::function<bool()> abort) { abort_ = std::move(abort); }

    // Presents a black frame: the swapchain image cleared, nothing read. PresentX with no planes.
    VkResult present_blank(bool vsync, bool tear = false) {
        return submit(nullptr, ResourceId{}, vsync, nullptr, swapchain_srgb_, tear);
    }

    // From now on every presented frame is also copied to host memory (the swapchain gets TRANSFER_SRC usage), for
    // read_back(). False when the surface cannot do it or its format is not 8 bits per channel with 4 channels.
    // Call before the first present.
    bool enable_capture() {
        if (!ok_) return false;
        capture_ = true;
        stale_ = true;
        if (!recreate(vsync_, swapchain_srgb_) || (swapchain_ && !capture_buffer_)) {
            capture_ = false;
            return false;
        }
        return true;
    }

    // The last presented frame as the swapchain stores it (4 bytes a texel, rows tightly packed, format() order).
    // Waits for the frame to finish. False if no captured frame is available or the wait fails.
    bool read_back(std::vector<std::uint8_t>& pixels, VkExtent2D* extent, VkFormat* format) {
        if (!capture_ || !capture_ptr_ || !captured_) return false;
        const Frame& frame = frames_[(frame_index_ + frames_.size() - 1) % frames_.size()];
        if (vkWaitForFences_(context_.device(), 1, &frame.done, VK_TRUE, UINT64_MAX) != VK_SUCCESS)
            return false;
        pixels.assign(capture_ptr_, capture_ptr_ + std::size_t(captured_extent_.width) * captured_extent_.height * 4);
        if (extent) *extent = captured_extent_;
        if (format) *format = captured_format_;
        return true;
    }

    // Shared by present() and present_blank(). `binding` null: clear instead of blit.
    VkResult submit(const ResourceBinding* binding, ResourceId source, bool vsync, const PresentRect* source_rect,
                    bool srgb, bool tear = false) {
        last_tore_ = false;
        if (!ok_) return VK_ERROR_INITIALIZATION_FAILED;
        if (vsync != vsync_ || stale_ || srgb != swapchain_srgb_) {
            // The swapchain's colour encoding is only recorded once the new swapchain exists, so a failed
            // recreate is tried again at the next present.
            if (!recreate(vsync, srgb)) return VK_ERROR_OUT_OF_DATE_KHR;
        }
        if (!swapchain_) {  // minimised: try again next frame
            stale_ = true;
            return VK_SUCCESS;
        }
        if (binding && !presentable(source)) return VK_ERROR_UNKNOWN;  // checked before an image is acquired for it
        if (binding && !(format_features(format_) & VK_FORMAT_FEATURE_BLIT_DST_BIT)) return VK_ERROR_FORMAT_NOT_SUPPORTED;
        auto& vk = context_.vk;
        const VkDevice device = context_.device();
        Frame& frame = frames_[frame_index_ % frames_.size()];
        if (const VkResult waited = vkWaitForFences_(device, 1, &frame.done, VK_TRUE, UINT64_MAX); waited != VK_SUCCESS)
            return stop(waited);

        std::uint32_t image = 0;
        VkResult acquired = acquire(frame.acquired, &image);
        if (acquired == VK_ERROR_OUT_OF_DATE_KHR) {
            if (!recreate(vsync, swapchain_srgb_)) return acquired;
            if (!swapchain_) return VK_SUCCESS;
            acquired = acquire(frame.acquired, &image);
        }
        if (acquired != VK_SUCCESS && acquired != VK_SUBOPTIMAL_KHR) return acquired;  // nothing was acquired
        if (acquired == VK_SUBOPTIMAL_KHR) stale_ = true;

        bool recorded = false;
        std::unique_lock tracker_lock(context_.tracker_mutex_, std::defer_lock);
        bool rollback_source = false;
        VkImageLayout prev_layout = VK_IMAGE_LAYOUT_UNDEFINED;
        if (binding) {
            // Lock hierarchy: resource_mutex_ -> tracker_mutex_ -> submit_mutex_ (vulkan_backend.h).
            // record() only touches subresource 0 (mip 0, layer 0, color) of `source`, so saving that one layout
            // rolls back a failed submit in O(1) without copying the entire LayoutTracker map.
            std::shared_lock resource_lock(context_.resource_mutex_);
            const ResourceBinding* current = context_.resolve(source);
            if (current && current->image && current->aspects == VK_IMAGE_ASPECT_COLOR_BIT) {
                tracker_lock.lock();
                rollback_source = context_.tracker_.tracked(source);
                if (rollback_source) prev_layout = context_.tracker_.layout(source, 0);
                record(frame.commands, *current, source, image, source_rect);
                recorded = true;
            }
        } else {
            record_blank(frame.commands, image);
            recorded = true;
        }
        const auto rollback = [&] {
            if (rollback_source)
                context_.tracker_.set(source, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}, prev_layout);
        };
        if (!recorded) return give_back(frame, VK_ERROR_UNKNOWN);  // released by another thread since presentable()
        if (const VkResult reset = vkResetFences_(device, 1, &frame.done); reset != VK_SUCCESS) {
            rollback();
            return stop(reset);
        }
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
#ifdef LEONIDA_SWAPCHAIN_MAINTENANCE1
        // The swapchain was created with FIFO and tear_mode_ (both come from the same recreate).
        VkSwapchainPresentModeInfoEXT present_mode{VK_STRUCTURE_TYPE_SWAPCHAIN_PRESENT_MODE_INFO_EXT};
        const bool tearing = tear && vsync_ && tear_supported();
        if (tearing) {
            present_mode.swapchainCount = 1;
            present_mode.pPresentModes = &tear_mode_;
            present.pNext = &present_mode;
        }
#else
        const bool tearing = false;
        (void)tear;
#endif
        VkResult submitted, result = VK_SUCCESS;
        {
            std::lock_guard lock(context_.submit_mutex());
            submitted = vk.vkQueueSubmit(queue_, 1, &submit_info, frame.done);
            if (submitted != VK_SUCCESS) rollback();
            // Commit/rollback happened before releasing tracker_mutex_ inside submit_mutex(), preserving submission
            // order without holding tracker_mutex_ across vkQueuePresentKHR.
            if (tracker_lock.owns_lock()) tracker_lock.unlock();
            if (submitted == VK_SUCCESS) result = vkQueuePresentKHR_(queue_, &present);
        }
        if (submitted != VK_SUCCESS) return give_back(frame, submitted);
        ++frame_index_;
        // Only a present the engine took counts (SUBOPTIMAL is shown too). OUT_OF_DATE was not shown: the swapchain
        // is recreated next time, and the caller sees VK_SUCCESS, since its work was submitted.
        if (result == VK_SUCCESS || result == VK_SUBOPTIMAL_KHR) {
            ++presented_;
            if (tearing) {
                last_tore_ = true;
                ++tear_presents_;
            }
        }
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
        std::shared_lock lock(context_.resource_mutex_);  // the resource table (vulkan_backend.h)
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
                        sw == dw && sh == dh ? VK_FILTER_NEAREST : blit_filter(binding.format));

        finish_image(commands, image);
        for (const auto& piece : pieces)
            detail::image_barrier(fn, commands, binding, piece.range, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                  piece.layout, VK_ACCESS_TRANSFER_READ_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, any,
                                  VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
        vk.vkEndCommandBuffer(commands);
    }

    // vkGetPhysicalDeviceFormatProperties, optimal tiling, cached per format.
    VkFormatFeatureFlags format_features(VkFormat format) {
        const auto it = features_.find(format);
        if (it != features_.end()) return it->second;
        VkFormatProperties properties{};
        if (vkGetPhysicalDeviceFormatProperties_) vkGetPhysicalDeviceFormatProperties_(physical_, format, &properties);
        return features_[format] = properties.optimalTilingFeatures;
    }
    // Scaling blits filter bilinearly when the source format allows it, else take the nearest texel.
    VkFilter blit_filter(VkFormat source) {
        return (format_features(source) & VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT) ? VK_FILTER_LINEAR
                                                                                              : VK_FILTER_NEAREST;
    }

    VkResult acquire(VkSemaphore semaphore, std::uint32_t* image) {
        for (;;) {
            const VkResult result = vkAcquireNextImageKHR_(context_.device(), swapchain_, 50'000'000, semaphore,
                                                           VK_NULL_HANDLE, image);
            if (result != VK_TIMEOUT && result != VK_NOT_READY) return result;
            if (abort_ && abort_()) return VK_NOT_READY;
        }
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
            // Two frames may be in flight and share `capture_buffer_`: serialize the prior frame's capture copy
            // before overwriting it, while moving this frame's swapchain image to TRANSFER_SRC.
            VkBufferMemoryBarrier capture_waw{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
            capture_waw.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            capture_waw.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            capture_waw.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            capture_waw.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            capture_waw.buffer = capture_buffer_;
            capture_waw.offset = 0;
            capture_waw.size = VK_WHOLE_SIZE;
            VkImageMemoryBarrier to_src{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
            to_src.srcAccessMask = access;
            to_src.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            to_src.oldLayout = layout;
            to_src.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
            to_src.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            to_src.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            to_src.image = images_[image];
            to_src.subresourceRange = range;
            fn.pipeline_barrier(commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0,
                                nullptr, 1, &capture_waw, 1, &to_src);
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
    // swapchain and is retried on the next present. Keep the old swapchain until the surface queries succeed and
    // the replacement can be attempted. An incomplete replacement stays stale so the next present tries again.
    bool recreate(bool vsync, bool srgb) {
        auto& vk = context_.vk;
        const VkDevice device = context_.device();
        {
            std::lock_guard lock(context_.submit_mutex());
            vk.vkQueueWaitIdle(queue_);  // the old swapchain's images and semaphores may still be in use
        }
        stale_ = true;  // until this succeeds
        tear_mode_ = VK_PRESENT_MODE_MAX_ENUM_KHR;
        VkSurfaceCapabilitiesKHR caps{};
        if (vkGetPhysicalDeviceSurfaceCapabilitiesKHR_(physical_, surface_, &caps) != VK_SUCCESS) return false;
        VkExtent2D extent = caps.currentExtent;
        // When the platform leaves the swapchain extent to the application (0xFFFFFFFF, as on Intel UHD / Win32 WSI),
        // query the window client area or fall back to the last known extent, and clamp to the surface bounds (D3D-165).
        if (extent.width == 0xFFFFFFFF || extent.height == 0xFFFFFFFF) {
            if (extent_fn_) extent = extent_fn_();
            // ponytail: if no window extent function was supplied, fall back to last extent_ or minImageExtent
            if (extent.width == 0xFFFFFFFF || extent.height == 0xFFFFFFFF || (extent.width == 0 && extent.height == 0)) {
                extent = extent_.width && extent_.height ? extent_ : caps.minImageExtent;
            }
            if (extent.width && extent.height) {
                extent.width = std::clamp(extent.width, caps.minImageExtent.width, caps.maxImageExtent.width);
                extent.height = std::clamp(extent.height, caps.minImageExtent.height, caps.maxImageExtent.height);
            }
        }
        if (extent.width == 0 || extent.height == 0) {
            release(swapchain_);
            swapchain_ = VK_NULL_HANDLE;
            vsync_ = vsync;
            swapchain_srgb_ = srgb;
            stale_ = false;
            return true;
        }
        VkImageUsageFlags usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        if (capture_) usage |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
        if ((caps.supportedUsageFlags & usage) != usage) return false;

        std::uint32_t count = 0;
        if (vkGetPhysicalDeviceSurfaceFormatsKHR_(physical_, surface_, &count, nullptr) != VK_SUCCESS || !count)
            return false;
        std::vector<VkSurfaceFormatKHR> formats(count);
        if (vkGetPhysicalDeviceSurfaceFormatsKHR_(physical_, surface_, &count, formats.data()) != VK_SUCCESS || !count)
            return false;
        formats.resize(count);
        // An sRGB source wants an sRGB swapchain (the blit decodes and encodes, which cancel); any other source
        // wants UNORM (the bytes go through as they are). BGRA before RGBA, as the surface lists them.
        VkSurfaceFormatKHR chosen = formats[0];
        bool found = false;
        for (const VkFormat wanted : srgb ? std::array<VkFormat, 2>{VK_FORMAT_B8G8R8A8_SRGB, VK_FORMAT_R8G8B8A8_SRGB}
                                                : std::array<VkFormat, 2>{VK_FORMAT_B8G8R8A8_UNORM, VK_FORMAT_R8G8B8A8_UNORM}) {
            for (const auto& f : formats)
                if (f.format == wanted && f.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
                    chosen = f;
                    found = true;
                    break;
                }
            if (found) break;
        }
        if (!found && srgb) {
            if (!warned_srgb_) std::fprintf(stderr, "[rt] present: the surface has no sRGB 8-bit format; an sRGB back buffer is shown darker\n");
            warned_srgb_ = true;
            for (const auto& f : formats)
                if ((f.format == VK_FORMAT_B8G8R8A8_UNORM || f.format == VK_FORMAT_R8G8B8A8_UNORM) &&
                    f.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
                    chosen = f;
                    break;
                }
        }
        if (vkGetPhysicalDeviceSurfacePresentModesKHR_(physical_, surface_, &count, nullptr) != VK_SUCCESS || !count)
            return false;
        std::vector<VkPresentModeKHR> modes(count);
        if (vkGetPhysicalDeviceSurfacePresentModesKHR_(physical_, surface_, &count, modes.data()) != VK_SUCCESS || !count)
            return false;
        modes.resize(count);
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
        info.imageExtent = extent;
        info.imageArrayLayers = 1;
        info.imageUsage = usage;
        info.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
        info.preTransform = caps.currentTransform;
        info.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
        if (!(caps.supportedCompositeAlpha & info.compositeAlpha)) info.compositeAlpha = VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR;
        info.presentMode = mode;
#ifdef LEONIDA_SWAPCHAIN_MAINTENANCE1
        // A vsync swapchain that can also be presented in IMMEDIATE (or FIFO_RELAXED), switched per present.
        VkSwapchainPresentModesCreateInfoEXT switchable{VK_STRUCTURE_TYPE_SWAPCHAIN_PRESENT_MODES_CREATE_INFO_EXT};
        std::array<VkPresentModeKHR, 2> switchable_modes{VK_PRESENT_MODE_FIFO_KHR, VK_PRESENT_MODE_MAX_ENUM_KHR};
        VkPresentModeKHR tear = VK_PRESENT_MODE_MAX_ENUM_KHR;
        if (vsync && mode == VK_PRESENT_MODE_FIFO_KHR) tear = find_tear_mode(modes, info.minImageCount, info.imageUsage, info.imageExtent);
        if (tear != VK_PRESENT_MODE_MAX_ENUM_KHR) {
            switchable_modes[1] = tear;
            switchable.presentModeCount = 2;
            switchable.pPresentModes = switchable_modes.data();
            info.pNext = &switchable;
        }
#endif
        info.clipped = VK_TRUE;
        const VkSwapchainKHR old = swapchain_;
        swapchain_ = VK_NULL_HANDLE;
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
#ifdef LEONIDA_SWAPCHAIN_MAINTENANCE1
        tear_mode_ = tear;
#endif
        images_ = std::move(images);
        ready_ = std::move(ready);
        format_ = chosen.format;
        extent_ = extent;
        if (capture_) {
            // A capture that cannot be made fails the recreation with the swapchain in place and stale_ still set.
            const bool four_bytes = format_ == VK_FORMAT_B8G8R8A8_UNORM || format_ == VK_FORMAT_R8G8B8A8_UNORM ||
                                    format_ == VK_FORMAT_B8G8R8A8_SRGB || format_ == VK_FORMAT_R8G8B8A8_SRGB;
            if (!four_bytes || !make_capture_buffer()) return false;
        }
        vsync_ = vsync;
        swapchain_srgb_ = srgb;
        stale_ = false;
        return true;
    }

#ifdef LEONIDA_SWAPCHAIN_MAINTENANCE1
    // The mode a FIFO swapchain can also present in, so that a late frame can be shown at once: IMMEDIATE, else
    // FIFO_RELAXED, each only if the surface lists it and reports it compatible with FIFO (VK_KHR_surface_maintenance1).
    // `min_images` is raised to what that mode needs too. The mode must also accept the swapchain's image usage and
    // extent (its capabilities can differ from FIFO's). MAX_ENUM when there is none or the extension is not in use.
    VkPresentModeKHR find_tear_mode(const std::vector<VkPresentModeKHR>& modes, std::uint32_t& min_images,
                                    VkImageUsageFlags usage, VkExtent2D extent) {
        if (!maintenance1_ || !vkGetPhysicalDeviceSurfaceCapabilities2KHR_) return VK_PRESENT_MODE_MAX_ENUM_KHR;
        // The capabilities of the surface for one present mode, and the modes it can switch to from there.
        const auto ask = [&](VkPresentModeKHR asked, VkSurfaceCapabilitiesKHR& caps,
                             std::array<VkPresentModeKHR, 16>* compatible, std::uint32_t* compatible_count) {
            VkSurfacePresentModeEXT for_mode{VK_STRUCTURE_TYPE_SURFACE_PRESENT_MODE_EXT};
            for_mode.presentMode = asked;
            VkPhysicalDeviceSurfaceInfo2KHR surface_info{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SURFACE_INFO_2_KHR, &for_mode};
            surface_info.surface = surface_;
            VkSurfacePresentModeCompatibilityEXT compat{VK_STRUCTURE_TYPE_SURFACE_PRESENT_MODE_COMPATIBILITY_EXT};
            if (compatible) {
                compat.presentModeCount = std::uint32_t(compatible->size());
                compat.pPresentModes = compatible->data();
            }
            VkSurfaceCapabilities2KHR out{VK_STRUCTURE_TYPE_SURFACE_CAPABILITIES_2_KHR, compatible ? &compat : nullptr};
            if (vkGetPhysicalDeviceSurfaceCapabilities2KHR_(physical_, &surface_info, &out) != VK_SUCCESS) return false;
            caps = out.surfaceCapabilities;
            if (compatible_count) *compatible_count = std::min<std::uint32_t>(compat.presentModeCount, 16);
            return true;
        };
        VkSurfaceCapabilitiesKHR fifo_caps{};
        std::array<VkPresentModeKHR, 16> compatible{};
        std::uint32_t compatible_count = 0;
        if (!ask(VK_PRESENT_MODE_FIFO_KHR, fifo_caps, &compatible, &compatible_count)) return VK_PRESENT_MODE_MAX_ENUM_KHR;
        for (const VkPresentModeKHR wanted : {VK_PRESENT_MODE_IMMEDIATE_KHR, VK_PRESENT_MODE_FIFO_RELAXED_KHR}) {
            if (std::find(modes.begin(), modes.end(), wanted) == modes.end() ||
                std::find(compatible.begin(), compatible.begin() + compatible_count, wanted) ==
                    compatible.begin() + compatible_count)
                continue;
            VkSurfaceCapabilitiesKHR mode_caps{};
            if (!ask(wanted, mode_caps, nullptr, nullptr)) continue;
            if ((mode_caps.supportedUsageFlags & usage) != usage) continue;
            if (extent.width < mode_caps.minImageExtent.width || extent.height < mode_caps.minImageExtent.height ||
                extent.width > mode_caps.maxImageExtent.width || extent.height > mode_caps.maxImageExtent.height)
                continue;
            // One image count that every mode of the swapchain accepts.
            std::uint32_t count = std::max({min_images, fifo_caps.minImageCount, mode_caps.minImageCount});
            const std::uint32_t most = std::min(fifo_caps.maxImageCount ? fifo_caps.maxImageCount : UINT32_MAX,
                                                mode_caps.maxImageCount ? mode_caps.maxImageCount : UINT32_MAX);
            if (count > most) continue;
            min_images = count;
            return wanted;
        }
        return VK_PRESENT_MODE_MAX_ENUM_KHR;
    }
#endif

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
    bool maintenance1_ = false;  // VK_KHR/EXT_swapchain_maintenance1 (and the surface extension) are enabled
    VkPresentModeKHR tear_mode_ = VK_PRESENT_MODE_MAX_ENUM_KHR;  // set with the swapchain, see tear_mode()
    bool last_tore_ = false;
    std::uint64_t tear_presents_ = 0;
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
    bool swapchain_srgb_ = false;  // the swapchain format follows the source being presented (set once it exists)
    std::function<bool()> abort_;
    std::function<VkExtent2D()> extent_fn_;
    std::map<VkFormat, VkFormatFeatureFlags> features_;
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
    PFN_vkGetPhysicalDeviceSurfaceCapabilities2KHR vkGetPhysicalDeviceSurfaceCapabilities2KHR_ = nullptr;
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
    PFN_vkGetPhysicalDeviceFormatProperties vkGetPhysicalDeviceFormatProperties_ = nullptr;
    PFN_vkCreateFence vkCreateFence_ = nullptr;
    PFN_vkDestroyFence vkDestroyFence_ = nullptr;
    PFN_vkWaitForFences vkWaitForFences_ = nullptr;
    PFN_vkResetFences vkResetFences_ = nullptr;
};

} // namespace leonida
