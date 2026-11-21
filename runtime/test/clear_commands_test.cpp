#include "../src/clear_commands.h"

#include <cstdio>

#define CHECK(condition) do {                                    \
    if (!(condition)) {                                          \
        std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); \
        return 1;                                                \
    }                                                            \
} while (0)

int main() {
    using namespace leonida;
    ClearCommandList list;
    ClearTargetView descriptor{12, 1, 2, 3};
    ClearColor color{1, 0, 0, 1};
    CHECK(!list.closed());
    CHECK(!list.clear_render_target({}, color));
    CHECK(list.clear_render_target(descriptor, color));
    descriptor = {24, 0, 0, 1};
    color[0] = 0;
    color[1] = 1;
    CHECK(list.clear_render_target(descriptor, color));
    CHECK(list.close());
    CHECK(list.closed());
    CHECK(!list.close());
    CHECK(!list.clear_render_target(descriptor, color));

    const auto commands = list.commands();
    CHECK(commands.size() == 2);
    CHECK(commands[0].target.resource == 12 && commands[1].target.resource == 24);
    CHECK(commands[0].target.mip == 1 && commands[0].target.first_layer == 2 && commands[0].target.layer_count == 3);
    const ClearColor red{1, 0, 0, 1};
    CHECK(commands[0].color == red && commands[1].color == color);

    CHECK(list.reset());
    CHECK(!list.closed() && list.commands().empty());
    CHECK(!list.reset());
    CHECK(list.clear_render_target({36, 0, 0, 1}, color));
    CHECK(list.close());
    CHECK(list.commands().size() == 1 && list.commands()[0].target.resource == 36);
    return 0;
}
