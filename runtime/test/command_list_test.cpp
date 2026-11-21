#include "../src/command_list.h"

#include <cstdio>

#define CHECK(condition) do {                                    \
    if (!(condition)) {                                          \
        std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); \
        return 1;                                                \
    }                                                            \
} while (0)

int main() {
    using namespace leonida;
    namespace s = d3d12_state;
    CommandList list;
    TargetView descriptor{12, 1, 2, 3};
    ClearColor color{1, 0, 0, 1};
    CHECK(!list.closed());
    CHECK(!list.clear_render_target({}, color));
    CHECK(list.clear_render_target(descriptor, color));
    // The CPU descriptor slot is reused; the recorded view keeps the old contents.
    descriptor = {24, 0, 0, 1};
    color[0] = 0;
    color[1] = 1;
    CHECK(list.clear_render_target(descriptor, color));

    // Barriers: a split barrier records only its end half.
    CHECK(!list.transition({0, 0, s::common, s::render_target}));
    CHECK(list.transition({24, all_subresources, s::render_target, s::copy_source}, barrier_begin_only));
    CHECK(list.transition({24, all_subresources, s::render_target, s::copy_source}, barrier_end_only));
    CHECK(!list.transition({24, 0, s::common, s::copy_dest}, barrier_begin_only | barrier_end_only));
    CHECK(!list.transition({24, 0, s::common, s::copy_dest}, 0x4));
    CHECK(list.uav_barrier({0}));
    CHECK(list.aliasing_barrier({0, 24}));

    // Depth-stencil clears: at least one flag, no unknown flags, depth in [0, 1].
    const TargetView depth{48, 0, 0, 1};
    CHECK(!list.clear_depth_stencil(depth, 0, 1.0f, 0));
    CHECK(!list.clear_depth_stencil(depth, 0x4, 1.0f, 0));
    CHECK(!list.clear_depth_stencil(depth, clear_depth, 1.5f, 0));
    CHECK(!list.clear_depth_stencil(depth, clear_depth, -0.0001f, 0));
    CHECK(!list.clear_depth_stencil({}, clear_depth, 1.0f, 0));
    CHECK(list.clear_depth_stencil(depth, clear_stencil, 7.0f, 0x5a));  // depth is ignored without clear_depth
    CHECK(list.clear_depth_stencil(depth, clear_depth | clear_stencil, 0.25f, 3));

    CHECK(list.close());
    CHECK(list.closed());
    CHECK(!list.close());
    CHECK(!list.clear_render_target(descriptor, color));
    CHECK(!list.transition({24, 0, s::copy_source, s::render_target}));
    CHECK(!list.uav_barrier({0}));

    const auto commands = list.commands();
    CHECK(commands.size() == 7);
    const auto* first = std::get_if<ClearRenderTarget>(&commands[0]);
    const auto* second = std::get_if<ClearRenderTarget>(&commands[1]);
    CHECK(first && second);
    CHECK(first->target.resource == 12 && second->target.resource == 24);
    CHECK(first->target.mip == 1 && first->target.first_layer == 2 && first->target.layer_count == 3);
    const ClearColor red{1, 0, 0, 1};
    CHECK(first->color == red && second->color == color);
    const auto* transition = std::get_if<TransitionBarrier>(&commands[2]);
    CHECK(transition && transition->resource == 24 && transition->subresource == all_subresources);
    CHECK(transition->before == s::render_target && transition->after == s::copy_source);
    CHECK(std::get_if<UavBarrier>(&commands[3]) && std::get<UavBarrier>(commands[3]).resource == 0);
    CHECK(std::get_if<AliasingBarrier>(&commands[4]) && std::get<AliasingBarrier>(commands[4]).after == 24);
    const auto* stencil = std::get_if<ClearDepthStencil>(&commands[5]);
    CHECK(stencil && stencil->flags == clear_stencil && stencil->stencil == 0x5a);
    const auto* both = std::get_if<ClearDepthStencil>(&commands[6]);
    CHECK(both && both->flags == (clear_depth | clear_stencil) && both->depth == 0.25f && both->stencil == 3);

    CHECK(list.reset());
    CHECK(!list.closed() && list.commands().empty());
    CHECK(!list.reset());
    CHECK(list.clear_render_target({36, 0, 0, 1}, color));
    CHECK(list.close());
    CHECK(list.commands().size() == 1);
    CHECK(std::get<ClearRenderTarget>(list.commands()[0]).target.resource == 36);
    return 0;
}
