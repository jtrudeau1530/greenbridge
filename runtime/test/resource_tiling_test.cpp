// GetResourceTiling's arithmetic (src/resource_tiling.h), without a GPU. The expected tile shapes, standard mip
// counts, per-mip tile counts and start indices are what the PC D3D12 runtime reported for the same reserved
// resources on an RTX 5070; packed-tile counts are this layer's own rule (tight linear size in whole tiles), and
// where the PC runtime chose differently the comment says so. d3d12_vulkan_device_test makes the same comparison
// against the host runtime on the machine it runs on.
#include "../src/resource_tiling.h"

#include <cstdio>

#define CHECK(condition) do {                                        \
    if (!(condition)) {                                              \
        std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); \
        return 1;                                                    \
    }                                                                \
} while (0)

namespace {

using leonida::packed_tile;
using leonida::ResourceTiling;
using leonida::TiledKind;
using leonida::TilingDesc;

bool same(const leonida::SubresourceTiling& s, std::uint32_t w, std::uint32_t h, std::uint32_t d, std::uint32_t start) {
    return s.width == w && s.height == h && s.depth == d && s.start == start;
}

}  // namespace

int main() {
    ResourceTiling t;

    // A buffer: 64 KiB tiles in a row, no mips.
    CHECK(resource_tiling({TiledKind::buffer, 200000}, t));
    CHECK(t.total_tiles == 4 && t.shape.width == 65536 && t.shape.height == 1 && t.shape.depth == 1);
    CHECK(t.standard_mips == 0 && t.packed_mips == 0 && t.subresources.size() == 1 && same(t.subresources[0], 4, 1, 1, 0));
    CHECK(!resource_tiling({TiledKind::buffer, 0}, t));

    // 1024x1024 RGBA8, full chain: four standard mips (64 + 16 + 4 + 1 tiles), seven packed in one tile.
    CHECK(resource_tiling({TiledKind::texture2d, 1024, 1024, 1, 11, 4, 1}, t));
    CHECK(t.shape.width == 128 && t.shape.height == 128 && t.shape.depth == 1);
    CHECK(t.standard_mips == 4 && t.packed_mips == 7 && t.packed_tiles == 1 && t.packed_start == 85);
    CHECK(t.total_tiles == 86 && t.subresources.size() == 11);
    CHECK(same(t.subresources[0], 8, 8, 1, 0) && same(t.subresources[1], 4, 4, 1, 64));
    CHECK(same(t.subresources[2], 2, 2, 1, 80) && same(t.subresources[3], 1, 1, 1, 84));
    CHECK(same(t.subresources[4], 0, 0, 0, packed_tile) && same(t.subresources[10], 0, 0, 0, packed_tile));

    // A 1000x300 array of 3: mip 2 (250x75) is the first below a tile in height. Each slice holds its standard
    // tiles, then its packed ones. The packed tail is 99,392 bytes, two tiles (the PC runtime reports three).
    CHECK(resource_tiling({TiledKind::texture2d, 1000, 300, 3, 10, 4, 1}, t));
    CHECK(t.standard_mips == 2 && t.packed_mips == 8 && t.packed_start == 32 && t.packed_tiles == 2);
    CHECK(t.total_tiles == 3 * 34 && t.subresources.size() == 30);
    CHECK(same(t.subresources[0], 8, 3, 1, 0) && same(t.subresources[1], 4, 2, 1, 24));
    CHECK(same(t.subresources[10], 8, 3, 1, 34) && same(t.subresources[11], 4, 2, 1, 58));
    CHECK(same(t.subresources[12], 0, 0, 0, packed_tile) && same(t.subresources[20], 8, 3, 1, 68));

    // Element sizes: 1, 2, 8 and 16 bytes.
    CHECK(resource_tiling({TiledKind::texture2d, 512, 512, 1, 10, 1, 1}, t));
    CHECK(t.shape.width == 256 && t.shape.height == 256 && t.standard_mips == 2 && t.packed_start == 5);
    CHECK(resource_tiling({TiledKind::texture2d, 512, 512, 1, 10, 2, 1}, t));
    CHECK(t.shape.width == 256 && t.shape.height == 128 && t.standard_mips == 2 && t.packed_start == 10);
    CHECK(same(t.subresources[0], 2, 4, 1, 0) && same(t.subresources[1], 1, 2, 1, 8));
    CHECK(resource_tiling({TiledKind::texture2d, 512, 512, 1, 10, 8, 1}, t));
    CHECK(t.shape.width == 128 && t.shape.height == 64 && t.standard_mips == 3 && t.packed_start == 42);
    CHECK(resource_tiling({TiledKind::texture2d, 512, 512, 1, 10, 16, 1}, t));
    CHECK(t.shape.width == 64 && t.shape.height == 64 && t.standard_mips == 4 && t.total_tiles == 86);

    // Block-compressed: the shape is in texels, four per block side. BC1 (8-byte blocks) tiles 512x256 texels,
    // BC7 (16-byte blocks) 256x256; a two-slice BC7 array has one packed tile per slice.
    CHECK(resource_tiling({TiledKind::texture2d, 2048, 2048, 1, 12, 8, 4}, t));
    CHECK(t.shape.width == 512 && t.shape.height == 256 && t.standard_mips == 3 && t.packed_mips == 9);
    CHECK(same(t.subresources[0], 4, 8, 1, 0) && same(t.subresources[2], 1, 2, 1, 40) && t.packed_start == 42);
    CHECK(resource_tiling({TiledKind::texture2d, 2048, 1024, 2, 12, 16, 4}, t));
    CHECK(t.shape.width == 256 && t.shape.height == 256 && t.packed_tiles == 1 && t.total_tiles == 86);
    CHECK(same(t.subresources[0], 8, 4, 1, 0) && same(t.subresources[12], 8, 4, 1, 43));
    CHECK(same(t.subresources[14], 2, 1, 1, 83) && same(t.subresources[15], 0, 0, 0, packed_tile));

    // One whole tile with one mip: nothing packed, and the packed start is 0.
    CHECK(resource_tiling({TiledKind::texture2d, 128, 128, 1, 1, 4, 1}, t));
    CHECK(t.total_tiles == 1 && t.standard_mips == 1 && t.packed_mips == 0 && t.packed_tiles == 0 && t.packed_start == 0);
    // Smaller than a tile: everything is packed (53,260 bytes, one tile; the PC runtime reports two).
    CHECK(resource_tiling({TiledKind::texture2d, 100, 100, 1, 7, 4, 1}, t));
    CHECK(t.standard_mips == 0 && t.packed_mips == 7 && t.packed_start == 0 && t.packed_tiles == 1 && t.total_tiles == 1);
    // Too thin for a tile: 4096x64 packs from mip 0 (1.4 MB, 22 tiles, as the PC runtime reports).
    CHECK(resource_tiling({TiledKind::texture2d, 4096, 64, 1, 13, 4, 1}, t));
    CHECK(t.standard_mips == 0 && t.packed_mips == 13 && t.packed_tiles == 22);
    // A short chain that stops before any mip is packed.
    CHECK(resource_tiling({TiledKind::texture2d, 1024, 1024, 1, 3, 4, 1}, t));
    CHECK(t.total_tiles == 84 && t.packed_mips == 0 && t.packed_start == 0);

    // Volumes: three-dimensional tiles, and depth also decides when mips pack.
    CHECK(resource_tiling({TiledKind::texture3d, 256, 256, 64, 9, 4, 1}, t));
    CHECK(t.shape.width == 32 && t.shape.height == 32 && t.shape.depth == 16);
    CHECK(t.standard_mips == 3 && t.packed_mips == 6 && t.packed_start == 292 && t.total_tiles == 293);
    CHECK(same(t.subresources[0], 8, 8, 4, 0) && same(t.subresources[1], 4, 4, 2, 256) && same(t.subresources[2], 2, 2, 1, 288));
    CHECK(resource_tiling({TiledKind::texture3d, 128, 128, 128, 8, 1, 1}, t));
    CHECK(t.shape.width == 64 && t.shape.height == 32 && t.shape.depth == 32 && t.standard_mips == 2);
    CHECK(t.packed_start == 36 && t.packed_tiles == 1);  // 37,449 bytes (the PC runtime reports two tiles)
    CHECK(resource_tiling({TiledKind::texture3d, 64, 64, 64, 7, 8, 1}, t));
    CHECK(t.shape.width == 32 && t.shape.height == 16 && t.shape.depth == 16 && t.standard_mips == 2);
    CHECK(same(t.subresources[0], 2, 4, 4, 0) && same(t.subresources[1], 1, 2, 2, 32) && t.packed_start == 36);

    // Element sizes that have no tile shape, and empty descriptions.
    CHECK(!resource_tiling({TiledKind::texture2d, 64, 64, 1, 1, 12, 1}, t));
    CHECK(!resource_tiling({TiledKind::texture2d, 64, 64, 1, 0, 4, 1}, t));
    CHECK(!resource_tiling({TiledKind::texture2d, 0, 64, 1, 1, 4, 1}, t));
    std::printf("resource tiling: buffer, 2D, array, block-compressed and volume layouts match\n");
    return 0;
}
