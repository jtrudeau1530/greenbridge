#include "../src/conservative_raster.h"
#include <cstdio>
#include <limits>

int main() {
    using leonida::conservative_raster::tier1;
    unsigned failures = 0;
    const auto check = [&](bool value) { if (!value) ++failures; };
    check(tier1(true, 8, 0.0f));
    check(tier1(true, 8, 1.0f / 256.0f));
    check(tier1(true, 8, 0.5f - 1.0f / 512.0f));
    check(!tier1(false, 8, 0.0f));
    check(!tier1(true, 7, 0.0f));
    check(!tier1(true, 8, 0.5f));
    check(!tier1(true, 8, -0.001f));
    check(!tier1(true, 8, std::numeric_limits<float>::infinity()));
    check(!tier1(true, 8, std::numeric_limits<float>::quiet_NaN()));
    using leonida::conservative_raster::tier;
    check(tier(false, 8, 0, true) == 0);
    check(tier(true, 7, 0, true) == 0);
    check(tier(true, 8, 0, true) == 2);
    check(tier(true, 8, 0, false) == 1);
    check(tier(true, 8, 1.0f / 512.0f, true) == 2);
    check(tier(true, 8, std::nextafter(1.0f / 512.0f, 1.0f), true) == 1);
    check(tier(true, 8, 1.0f / 256.0f, true) == 1);
    check(tier(true, 16, 0.003f, true) == 2);
    check(tier(true, 16, 0.003f, false) == 1);
    check(tier(true, 8, std::numeric_limits<float>::quiet_NaN(), true) == 0);
    if (failures) std::fprintf(stderr, "conservative raster capability checks: %u failures\n", failures);
    else std::puts("conservative raster: enabled extension, precision and uncertainty bounds checked");
    return failures ? 1 : 0;
}
