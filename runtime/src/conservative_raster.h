#pragma once
#include <cmath>
#include <cstdint>
#include <algorithm>

namespace leonida::conservative_raster {
// Tier 1 bounds the outer uncertainty to half a pixel and requires a 16.8 rasterization grid. Leave room for
// half a grid step at that minimum precision in addition to Vulkan's fixed primitive overestimation. Do not
// infer Tier 3 from degenerate-triangle support: InnerCoverage needs separate work.
inline bool tier1(bool extension_enabled, std::uint32_t subpixel_bits, float overestimate) {
    constexpr float maximum = 0.5f - 1.0f / 512.0f;
    return extension_enabled && subpixel_bits >= 8 && std::isfinite(overestimate) &&
           overestimate >= 0.0f && overestimate <= maximum;
}
inline std::uint32_t tier(bool extension_enabled, std::uint32_t subpixel_bits, float overestimate,
                          bool degenerate_triangles) {
    if (!tier1(extension_enabled, subpixel_bits, overestimate)) return 0;
    // Tier 2 allows 1/256 pixel total uncertainty and requires post-snap degenerates. Reserve half of the
    // device's actual raster grid step, conservatively capped at float's useful precision, for snapping.
    const float snap = std::ldexp(0.5f, -int(std::min(subpixel_bits, 24u)));
    return degenerate_triangles && overestimate <= 1.0f / 256.0f - snap ? 2u : 1u;
}
} // namespace leonida::conservative_raster
