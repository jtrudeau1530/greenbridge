#pragma once

#include <array>
#include <cstdint>
#include <span>
#include <vector>

namespace leonida {

using ClearResourceId = std::uintptr_t;
using ClearColor = std::array<float, 4>;

struct ClearTargetView {
    ClearResourceId resource;
    std::uint32_t mip;
    std::uint32_t first_layer;
    std::uint32_t layer_count;
};

struct ClearRenderTarget {
    ClearTargetView target;
    ClearColor color;
};

// The full-view ClearRenderTargetView subset of the command-list IR.
class ClearCommandList {
public:
    // The caller reads the CPU descriptor during recording. Store its view by value.
    bool clear_render_target(const ClearTargetView& target, const ClearColor& color) {
        if (closed_ || !target.resource || !target.layer_count) return false;
        commands_.push_back({target, color});
        return true;
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
    std::span<const ClearRenderTarget> commands() const { return commands_; }

private:
    std::vector<ClearRenderTarget> commands_;
    bool closed_ = false;
};

} // namespace leonida
