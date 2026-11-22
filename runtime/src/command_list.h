#pragma once

// The command-list IR (docs/d3d12_vulkan.md section 3.2): what a D3D12 command list records, kept until
// ExecuteCommandLists replays it on Vulkan (vulkan_replay.h). This slice covers legacy and enhanced barriers and
// render-target, depth-stencil and unordered-access clears (whole views or rectangles), copies, resolves and
// predication. It has no Windows or Vulkan dependency.

#include <array>
#include <cstdint>
#include <optional>
#include <span>
#include <variant>
#include <vector>

namespace leonida {

using ResourceId = std::uintptr_t;
using ClearColor = std::array<float, 4>;

// D3D12 enum values from the public d3d12.h, so the IR builds without the Windows SDK.
// d3d12_state_map_test checks every one against d3d12.h on Windows.
namespace d3d12_state {
inline constexpr std::uint32_t common = 0;
inline constexpr std::uint32_t vertex_and_constant_buffer = 0x1;
inline constexpr std::uint32_t index_buffer = 0x2;
inline constexpr std::uint32_t render_target = 0x4;
inline constexpr std::uint32_t unordered_access = 0x8;
inline constexpr std::uint32_t depth_write = 0x10;
inline constexpr std::uint32_t depth_read = 0x20;
inline constexpr std::uint32_t non_pixel_shader_resource = 0x40;
inline constexpr std::uint32_t pixel_shader_resource = 0x80;
inline constexpr std::uint32_t stream_out = 0x100;
inline constexpr std::uint32_t indirect_argument = 0x200;
inline constexpr std::uint32_t copy_dest = 0x400;
inline constexpr std::uint32_t copy_source = 0x800;
inline constexpr std::uint32_t resolve_dest = 0x1000;
inline constexpr std::uint32_t resolve_source = 0x2000;
inline constexpr std::uint32_t raytracing_acceleration_structure = 0x400000;
inline constexpr std::uint32_t shading_rate_source = 0x1000000;
inline constexpr std::uint32_t generic_read = 0xac3;
inline constexpr std::uint32_t present = 0;
} // namespace d3d12_state

// D3D12_BARRIER_SYNC (enhanced barriers).
namespace d3d12_sync {
inline constexpr std::uint32_t none = 0;
inline constexpr std::uint32_t all = 0x1;
inline constexpr std::uint32_t draw = 0x2;
inline constexpr std::uint32_t index_input = 0x4;
inline constexpr std::uint32_t vertex_shading = 0x8;
inline constexpr std::uint32_t pixel_shading = 0x10;
inline constexpr std::uint32_t depth_stencil = 0x20;
inline constexpr std::uint32_t render_target = 0x40;
inline constexpr std::uint32_t compute_shading = 0x80;
inline constexpr std::uint32_t raytracing = 0x100;
inline constexpr std::uint32_t copy = 0x200;
inline constexpr std::uint32_t resolve = 0x400;
inline constexpr std::uint32_t execute_indirect = 0x800;
inline constexpr std::uint32_t predication = 0x800;
inline constexpr std::uint32_t all_shading = 0x1000;
inline constexpr std::uint32_t non_pixel_shading = 0x2000;
inline constexpr std::uint32_t emit_raytracing_acceleration_structure_postbuild_info = 0x4000;
inline constexpr std::uint32_t clear_unordered_access_view = 0x8000;
inline constexpr std::uint32_t video_decode = 0x100000;
inline constexpr std::uint32_t video_process = 0x200000;
inline constexpr std::uint32_t video_encode = 0x400000;
inline constexpr std::uint32_t build_raytracing_acceleration_structure = 0x800000;
inline constexpr std::uint32_t copy_raytracing_acceleration_structure = 0x1000000;
inline constexpr std::uint32_t split = 0x80000000;
} // namespace d3d12_sync

// D3D12_BARRIER_ACCESS.
namespace d3d12_access {
inline constexpr std::uint32_t common = 0;
inline constexpr std::uint32_t vertex_buffer = 0x1;
inline constexpr std::uint32_t constant_buffer = 0x2;
inline constexpr std::uint32_t index_buffer = 0x4;
inline constexpr std::uint32_t render_target = 0x8;
inline constexpr std::uint32_t unordered_access = 0x10;
inline constexpr std::uint32_t depth_stencil_write = 0x20;
inline constexpr std::uint32_t depth_stencil_read = 0x40;
inline constexpr std::uint32_t shader_resource = 0x80;
inline constexpr std::uint32_t stream_output = 0x100;
inline constexpr std::uint32_t indirect_argument = 0x200;
inline constexpr std::uint32_t predication = 0x200;
inline constexpr std::uint32_t copy_dest = 0x400;
inline constexpr std::uint32_t copy_source = 0x800;
inline constexpr std::uint32_t resolve_dest = 0x1000;
inline constexpr std::uint32_t resolve_source = 0x2000;
inline constexpr std::uint32_t raytracing_acceleration_structure_read = 0x4000;
inline constexpr std::uint32_t raytracing_acceleration_structure_write = 0x8000;
inline constexpr std::uint32_t shading_rate_source = 0x10000;
inline constexpr std::uint32_t video_decode_read = 0x20000;
inline constexpr std::uint32_t video_decode_write = 0x40000;
inline constexpr std::uint32_t video_process_read = 0x80000;
inline constexpr std::uint32_t video_process_write = 0x100000;
inline constexpr std::uint32_t video_encode_read = 0x200000;
inline constexpr std::uint32_t video_encode_write = 0x400000;
inline constexpr std::uint32_t no_access = 0x80000000;
} // namespace d3d12_access

// D3D12_BARRIER_LAYOUT.
namespace d3d12_layout {
inline constexpr std::uint32_t undefined = 0xffffffff;
inline constexpr std::uint32_t common = 0;
inline constexpr std::uint32_t present = 0;
inline constexpr std::uint32_t generic_read = 1;
inline constexpr std::uint32_t render_target = 2;
inline constexpr std::uint32_t unordered_access = 3;
inline constexpr std::uint32_t depth_stencil_write = 4;
inline constexpr std::uint32_t depth_stencil_read = 5;
inline constexpr std::uint32_t shader_resource = 6;
inline constexpr std::uint32_t copy_source = 7;
inline constexpr std::uint32_t copy_dest = 8;
inline constexpr std::uint32_t resolve_source = 9;
inline constexpr std::uint32_t resolve_dest = 10;
inline constexpr std::uint32_t shading_rate_source = 11;
inline constexpr std::uint32_t video_decode_read = 12;
inline constexpr std::uint32_t video_decode_write = 13;
inline constexpr std::uint32_t video_process_read = 14;
inline constexpr std::uint32_t video_process_write = 15;
inline constexpr std::uint32_t video_encode_read = 16;
inline constexpr std::uint32_t video_encode_write = 17;
inline constexpr std::uint32_t direct_queue_common = 18;
inline constexpr std::uint32_t direct_queue_generic_read = 19;
inline constexpr std::uint32_t direct_queue_unordered_access = 20;
inline constexpr std::uint32_t direct_queue_shader_resource = 21;
inline constexpr std::uint32_t direct_queue_copy_source = 22;
inline constexpr std::uint32_t direct_queue_copy_dest = 23;
inline constexpr std::uint32_t compute_queue_common = 24;
inline constexpr std::uint32_t compute_queue_generic_read = 25;
inline constexpr std::uint32_t compute_queue_unordered_access = 26;
inline constexpr std::uint32_t compute_queue_shader_resource = 27;
inline constexpr std::uint32_t compute_queue_copy_source = 28;
inline constexpr std::uint32_t compute_queue_copy_dest = 29;
inline constexpr std::uint32_t video_queue_common = 30;
} // namespace d3d12_layout

// D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, D3D12_RESOURCE_BARRIER_FLAGS, D3D12_CLEAR_FLAGS and
// D3D12_TEXTURE_BARRIER_FLAGS.
inline constexpr std::uint32_t all_subresources = 0xffffffff;
inline constexpr std::uint32_t barrier_begin_only = 0x1;
inline constexpr std::uint32_t barrier_end_only = 0x2;
inline constexpr std::uint32_t clear_depth = 0x1;
inline constexpr std::uint32_t clear_stencil = 0x2;
inline constexpr std::uint32_t texture_barrier_discard = 0x1;

// Portable color-view interpretation. Other views inherit their resource format as before.
enum class ColorViewFormat : std::uint32_t { resource, rgba8_unorm, rgba8_srgb, rgba8_uint };

// A view captured by value while recording: the CPU descriptor slot may be rewritten after the call.
struct TargetView {
    ResourceId resource;
    std::uint32_t mip;
    std::uint32_t first_layer;
    std::uint32_t layer_count;
    ColorViewFormat format = ColorViewFormat::resource;
};

// D3D12_RECT: right and bottom are exclusive.
struct Rect {
    std::int32_t left;
    std::int32_t top;
    std::int32_t right;
    std::int32_t bottom;
};

// D3D12 subresource index: mip + layer * mips + plane * mips * layers, or all_subresources.
struct TransitionBarrier {
    ResourceId resource;
    std::uint32_t subresource;
    std::uint32_t before;
    std::uint32_t after;
};

// resource 0 means every UAV access, as a null pResource does in D3D12.
struct UavBarrier {
    ResourceId resource;
};

// Either resource may be 0 ("any placed or reserved resource").
struct AliasingBarrier {
    ResourceId before;
    ResourceId after;
};

// Enhanced barriers (ID3D12GraphicsCommandList7::Barrier).
struct GlobalBarrier {
    std::uint32_t sync_before;
    std::uint32_t sync_after;
    std::uint32_t access_before;
    std::uint32_t access_after;
};

// Offset and size are not kept: D3D12 requires 0 and UINT64_MAX (the whole buffer).
struct BufferBarrier {
    std::uint32_t sync_before;
    std::uint32_t sync_after;
    std::uint32_t access_before;
    std::uint32_t access_after;
    ResourceId resource;
};

// D3D12_BARRIER_SUBRESOURCE_RANGE: mip_count == 0 means index_or_first_mip is a subresource index
// (all_subresources for every one).
struct BarrierSubresourceRange {
    std::uint32_t index_or_first_mip;
    std::uint32_t mip_count;
    std::uint32_t first_layer;
    std::uint32_t layer_count;
    std::uint32_t first_plane;
    std::uint32_t plane_count;
};

struct TextureBarrier {
    std::uint32_t sync_before;
    std::uint32_t sync_after;
    std::uint32_t access_before;
    std::uint32_t access_after;
    std::uint32_t layout_before;
    std::uint32_t layout_after;
    ResourceId resource;
    BarrierSubresourceRange range;
    std::uint32_t flags;  // texture_barrier_discard
};

// rects empty: the whole view.
// CopyBufferRegion.
struct CopyBuffer {
    ResourceId dst;
    std::uint64_t dst_offset;
    ResourceId src;
    std::uint64_t src_offset;
    std::uint64_t size;
};

// D3D12_TEXTURE_COPY_LOCATION: a texture subresource, or a placed footprint in a buffer. The footprint's format is
// not kept: the copy uses the texture's own format for the plane, which D3D12 requires to be compatible.
struct CopyLocation {
    ResourceId resource;
    bool footprint;
    std::uint32_t subresource;  // texture
    std::uint64_t offset;       // footprint: byte offset in the buffer
    std::uint32_t width;
    std::uint32_t height;
    std::uint32_t depth;
    std::uint32_t row_pitch;
};

// D3D12_BOX: right, bottom and back are exclusive.
struct CopyBox {
    std::uint32_t left;
    std::uint32_t top;
    std::uint32_t front;
    std::uint32_t right;
    std::uint32_t bottom;
    std::uint32_t back;
};

// CopyTextureRegion. Without a box, the whole source subresource or footprint is copied.
struct CopyTexture {
    CopyLocation dst;
    std::uint32_t x;
    std::uint32_t y;
    std::uint32_t z;
    CopyLocation src;
    bool has_box;
    CopyBox box;
};

// CopyResource: every subresource, or the whole buffer.
struct CopyResource {
    ResourceId dst;
    ResourceId src;
};

struct ResolveRegion {
    std::uint32_t dst_x;
    std::uint32_t dst_y;
    std::optional<Rect> src_rect;
};

// Without a region, both subresources have the same extent. A region with no source rectangle takes the
// entire source to the given destination offset. The resolve uses the images' own format.
struct ResolveSubresource {
    ResourceId dst;
    std::uint32_t dst_subresource;
    ResourceId src;
    std::uint32_t src_subresource;
    std::optional<ResolveRegion> region = std::nullopt;
};

// D3D12_TEXTURE_DATA_PITCH_ALIGNMENT and D3D12_TEXTURE_DATA_PLACEMENT_ALIGNMENT.
inline constexpr std::uint32_t texture_data_pitch_alignment = 256;
inline constexpr std::uint32_t texture_data_placement_alignment = 512;

struct ClearRenderTarget {
    TargetView target;
    ClearColor color;
    std::vector<Rect> rects;
};

struct ClearDepthStencil {
    TargetView target;
    std::uint32_t flags;  // clear_depth | clear_stencil
    float depth;
    std::uint8_t stencil;
    std::vector<Rect> rects;
};

// What a UAV descriptor said when ClearUnorderedAccessView* was recorded (the CPU slot may be rewritten after).
// A buffer view is a byte range of the buffer and an element format; a texture view is one mip and a layer range.
struct UavView {
    ResourceId resource;
    bool buffer;
    std::uint32_t format;        // the view's VkFormat value; 0 (VK_FORMAT_UNDEFINED): a raw or structured buffer
    std::uint64_t offset;        // buffer: first byte of the view
    std::uint64_t size;          // buffer: bytes in the view
    std::uint32_t element_size;  // buffer: bytes an element (rects count elements), 4 for a raw view
    std::uint32_t mip;           // texture
    std::uint32_t first_layer;
    std::uint32_t layer_count;
};

// ClearUnorderedAccessViewUint / ...Float; the resource must be in unordered_access state at execution. values are
// the four UINTs, or the four FLOATs' bits. rects empty: the whole view. A buffer view's rects are element ranges
// [left, right) (top and bottom must cover row 0).
struct ClearUnorderedAccess {
    UavView view;
    bool is_float;
    std::array<std::uint32_t, 4> values;
    std::vector<Rect> rects;
};

// D3D12_PREDICATION_OP.
namespace d3d12_predication {
inline constexpr std::uint32_t equal_zero = 0;      // skip the predicated commands when the value is 0
inline constexpr std::uint32_t not_equal_zero = 1;  // skip them when it isn't
} // namespace d3d12_predication

// D3D12 requires the predicate's offset to be a multiple of 8 (the value is 64 bits).
inline constexpr std::uint64_t predication_alignment = 8;

// SetPredication. buffer 0 turns predication off. A command list starts without predication, and it does not carry
// over to the next list.
struct Predication {
    ResourceId buffer;
    std::uint64_t offset;
    std::uint32_t op;  // d3d12_predication
};

using Command = std::variant<TransitionBarrier, UavBarrier, AliasingBarrier, GlobalBarrier, BufferBarrier,
                             TextureBarrier, ClearRenderTarget, ClearDepthStencil, CopyBuffer, CopyTexture,
                             CopyResource, ResolveSubresource, Predication, ClearUnorderedAccess>;

class CommandList {
public:
    // D3D12 split barriers: the begin half only marks the resource unusable until the end half, so the transition
    // is recorded once, at the end. A full barrier (flags 0) is recorded where it is issued.
    bool transition(const TransitionBarrier& barrier, std::uint32_t flags = 0) {
        if (closed_ || !barrier.resource) return false;
        if (flags & ~(barrier_begin_only | barrier_end_only)) return false;
        if (flags == (barrier_begin_only | barrier_end_only)) return false;
        if (flags & barrier_begin_only) return true;
        commands_.push_back(barrier);
        return true;
    }

    bool uav_barrier(const UavBarrier& barrier) { return append(barrier); }
    bool aliasing_barrier(const AliasingBarrier& barrier) { return append(barrier); }

    // Enhanced split barriers follow the same rule: the half with sync_after == split is dropped, and the half with
    // sync_before == split is replayed as a full barrier.
    bool barrier(const GlobalBarrier& barrier) {
        if (!valid_sync(barrier.sync_before, barrier.sync_after)) return false;
        if (barrier.sync_after == d3d12_sync::split) return !closed_;
        return append(barrier);
    }

    bool barrier(const BufferBarrier& barrier) {
        if (!barrier.resource || !valid_sync(barrier.sync_before, barrier.sync_after)) return false;
        if (barrier.sync_after == d3d12_sync::split) return !closed_;
        return append(barrier);
    }

    bool barrier(const TextureBarrier& barrier) {
        if (!barrier.resource || !valid_sync(barrier.sync_before, barrier.sync_after)) return false;
        if (barrier.layout_after == d3d12_layout::undefined && barrier.layout_before != d3d12_layout::undefined)
            return false;
        if (barrier.flags & ~texture_barrier_discard) return false;
        if (barrier.range.mip_count && (!barrier.range.layer_count || !barrier.range.plane_count)) return false;
        if (barrier.sync_after == d3d12_sync::split) return !closed_;
        return append(barrier);
    }

    // Copies. The rules D3D12 states for the call itself are checked here; ranges against the resources are
    // checked at replay, when the resources are known.
    bool copy_buffer(const CopyBuffer& copy) {
        if (!copy.dst || !copy.src || !copy.size) return false;
        return append(copy);
    }

    bool copy_texture(const CopyTexture& copy) {
        if (!valid(copy.dst) || !valid(copy.src) || (copy.dst.footprint && copy.src.footprint)) return false;
        if (copy.has_box && (copy.box.right <= copy.box.left || copy.box.bottom <= copy.box.top ||
                             copy.box.back <= copy.box.front))
            return false;
        return append(copy);
    }

    bool copy_resource(const CopyResource& copy) {
        if (!copy.dst || !copy.src || copy.dst == copy.src) return false;
        return append(copy);
    }

    // A resource has one sample count, so it cannot be resolved into itself.
    bool resolve_subresource(const ResolveSubresource& resolve) {
        if (!resolve.dst || !resolve.src || resolve.dst == resolve.src) return false;
        if (const auto& region = resolve.region) {
            if (region->dst_x > INT32_MAX || region->dst_y > INT32_MAX) return false;
            if (const auto& r = region->src_rect;
                r && (r->left < 0 || r->top < 0 || r->right <= r->left || r->bottom <= r->top))
                return false;
        }
        return append(resolve);
    }

    // ClearRenderTargetView; the resource must be in render_target state at execution. No rects: the whole view.
    bool clear_render_target(const TargetView& target, const ClearColor& color, std::span<const Rect> rects = {}) {
        if (!valid(target)) return false;
        return append(ClearRenderTarget{target, color, {rects.begin(), rects.end()}});
    }

    // ClearDepthStencilView; the cleared planes must be in depth_write state at execution.
    // D3D12 requires depth in [0, 1] and at least one clear flag.
    bool clear_depth_stencil(const TargetView& target, std::uint32_t flags, float depth, std::uint8_t stencil,
                             std::span<const Rect> rects = {}) {
        if (!valid(target)) return false;
        if (!flags || (flags & ~(clear_depth | clear_stencil))) return false;
        if ((flags & clear_depth) && !(depth >= 0.0f && depth <= 1.0f)) return false;
        return append(ClearDepthStencil{target, flags, depth, stencil, {rects.begin(), rects.end()}});
    }

    // A view of nothing (no resource, an empty buffer range or no layers) is invalid, as is a rectangle that is
    // inside out. Ranges against the resource are checked at replay.
    bool clear_unordered_access(const UavView& view, bool is_float, const std::array<std::uint32_t, 4>& values,
                                std::span<const Rect> rects = {}) {
        if (!view.resource) return false;
        if (view.buffer ? (!view.size || !view.element_size) : !view.layer_count) return false;
        for (const Rect& r : rects)
            if (r.right < r.left || r.bottom < r.top) return false;
        return append(ClearUnorderedAccess{view, is_float, values, {rects.begin(), rects.end()}});
    }

    // ExecuteBundle copies the closed bundle's commands at the call site. The COM layer validates D3D12 list
    // types and bundle-only restrictions before calling this method.
    bool execute_bundle(const CommandList& bundle) {
        if (this == &bundle || closed_ || !bundle.closed_) return false;
        commands_.insert(commands_.end(), bundle.commands_.begin(), bundle.commands_.end());
        return true;
    }

    // SetPredication. Turning it off ignores the offset and op, as D3D12 does.
    bool set_predication(const Predication& predication) {
        if (!predication.buffer) return append(Predication{0, 0, 0});
        if (predication.op != d3d12_predication::equal_zero && predication.op != d3d12_predication::not_equal_zero)
            return false;
        if (predication.offset % predication_alignment) return false;
        return append(predication);
    }

    bool close() {
        if (closed_) return false;
        closed_ = true;
        return true;
    }

    bool reset() {
        if (!closed_) return false;
        commands_.clear();
        closed_ = false;
        return true;
    }

    bool closed() const { return closed_; }
    std::span<const Command> commands() const { return commands_; }

private:
    static bool valid(const TargetView& target) { return target.resource && target.layer_count; }

    // A placed footprint needs aligned offsets and pitches and a non-empty size.
    static bool valid(const CopyLocation& location) {
        if (!location.resource) return false;
        if (!location.footprint) return true;
        return location.width && location.height && location.depth && location.row_pitch &&
               location.offset % texture_data_placement_alignment == 0 &&
               location.row_pitch % texture_data_pitch_alignment == 0;
    }

    // A split begin and end cannot be the same barrier.
    static bool valid_sync(std::uint32_t before, std::uint32_t after) {
        return !(before == d3d12_sync::split && after == d3d12_sync::split);
    }

    bool append(const Command& command) {
        if (closed_) return false;
        commands_.push_back(command);
        return true;
    }

    std::vector<Command> commands_;
    bool closed_ = false;
};

} // namespace leonida
