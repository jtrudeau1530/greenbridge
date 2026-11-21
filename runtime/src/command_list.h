#pragma once

// The command-list IR (docs/d3d12_vulkan.md section 3.2): what a D3D12 command list records, kept until
// ExecuteCommandLists replays it on Vulkan (vulkan_replay.h). This slice covers resource barriers and full-view
// render-target and depth-stencil clears. It has no Windows or Vulkan dependency.

#include <array>
#include <cstdint>
#include <span>
#include <variant>
#include <vector>

namespace leonida {

using ResourceId = std::uintptr_t;
using ClearColor = std::array<float, 4>;

// D3D12_RESOURCE_STATES values from the public d3d12.h, so the IR builds without the Windows SDK.
// d3d12_state_map_test checks them against d3d12.h on Windows.
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

// D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, D3D12_RESOURCE_BARRIER_FLAGS and D3D12_CLEAR_FLAGS.
inline constexpr std::uint32_t all_subresources = 0xffffffff;
inline constexpr std::uint32_t barrier_begin_only = 0x1;
inline constexpr std::uint32_t barrier_end_only = 0x2;
inline constexpr std::uint32_t clear_depth = 0x1;
inline constexpr std::uint32_t clear_stencil = 0x2;

// A view captured by value while recording: the CPU descriptor slot may be rewritten after the call.
struct TargetView {
    ResourceId resource;
    std::uint32_t mip;
    std::uint32_t first_layer;
    std::uint32_t layer_count;
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

struct ClearRenderTarget {
    TargetView target;
    ClearColor color;
};

struct ClearDepthStencil {
    TargetView target;
    std::uint32_t flags;  // clear_depth | clear_stencil
    float depth;
    std::uint8_t stencil;
};

using Command = std::variant<TransitionBarrier, UavBarrier, AliasingBarrier, ClearRenderTarget, ClearDepthStencil>;

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

    // ClearRenderTargetView with NumRects == 0; the resource must be in render_target state at execution.
    bool clear_render_target(const TargetView& target, const ClearColor& color) {
        if (!valid(target)) return false;
        return append(ClearRenderTarget{target, color});
    }

    // ClearDepthStencilView with NumRects == 0; the resource must be in depth_write state at execution.
    // D3D12 requires depth in [0, 1] and at least one clear flag.
    bool clear_depth_stencil(const TargetView& target, std::uint32_t flags, float depth, std::uint8_t stencil) {
        if (!valid(target)) return false;
        if (!flags || (flags & ~(clear_depth | clear_stencil))) return false;
        if ((flags & clear_depth) && !(depth >= 0.0f && depth <= 1.0f)) return false;
        return append(ClearDepthStencil{target, flags, depth, stencil});
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

    bool append(const Command& command) {
        if (closed_) return false;
        commands_.push_back(command);
        return true;
    }

    std::vector<Command> commands_;
    bool closed_ = false;
};

} // namespace leonida
