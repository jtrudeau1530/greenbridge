#pragma once

// How a D3D12 reserved (tiled) resource divides into 64 KiB tiles: ID3D12Device::GetResourceTiling
// (docs/d3d12_vulkan.md section 4.2). No Windows or Vulkan dependency.
//
// What D3D12 defines, and d3d12_vulkan_device_test compares with the PC runtime: the standard tile shape for each
// element size, which mips are standard (at least one tile shape in every dimension) and which are packed (the
// rest, from the first mip smaller than a tile on any axis), the tile counts of every standard mip, and the order
// of the tiles: each array slice holds its standard mips in order, then its packed tiles.
//
// What a driver chooses: how many tiles the packed mips take. Here it is their tight linear size rounded up to
// whole tiles (at least one). The PC runtime on an NVIDIA GPU sometimes reports one more; applications read the
// number rather than assume it, and this layer backs every reserved resource fully, so any count is consistent.

#include <algorithm>
#include <cstdint>
#include <vector>

namespace leonida {

inline constexpr std::uint32_t tile_size_in_bytes = 65536;  // D3D12_TILED_RESOURCE_TILE_SIZE_IN_BYTES
inline constexpr std::uint32_t packed_tile = 0xffffffff;    // D3D12_PACKED_TILE

// In texels, or in bytes for a buffer (65536 x 1 x 1).
struct TileShape {
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::uint32_t depth = 0;
};

enum class TiledKind { buffer, texture2d, texture3d };

struct TilingDesc {
    TiledKind kind = TiledKind::texture2d;
    std::uint64_t width = 0;            // bytes for a buffer
    std::uint32_t height = 1;
    std::uint32_t depth_or_layers = 1;  // depth for a volume, array size otherwise
    std::uint32_t mips = 1;
    std::uint32_t element_bytes = 0;    // bytes per texel, or per block of a block-compressed format
    std::uint32_t block = 1;            // texels per block side: 4 for BC1-7
};

// D3D12_SUBRESOURCE_TILING: a packed mip has no tiles of its own and starts at packed_tile.
struct SubresourceTiling {
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::uint32_t depth = 0;
    std::uint32_t start = 0;
};

struct ResourceTiling {
    std::uint32_t total_tiles = 0;
    TileShape shape;
    std::uint32_t standard_mips = 0;  // D3D12_PACKED_MIP_INFO, per array slice
    std::uint32_t packed_mips = 0;
    std::uint32_t packed_tiles = 0;
    std::uint32_t packed_start = 0;   // the first packed tile of slice 0; 0 without packed mips
    std::vector<SubresourceTiling> subresources;  // mip + layer * mips
};

// The standard tile shape in elements (D3D12 "standard swizzle" tile shapes for 64 KiB tiles), or {} for an
// element size that cannot be tiled (3- and 12-byte formats).
constexpr TileShape element_tile_shape(bool volume, std::uint32_t element_bytes) {
    switch (element_bytes) {
    case 1: return volume ? TileShape{64, 32, 32} : TileShape{256, 256, 1};
    case 2: return volume ? TileShape{32, 32, 32} : TileShape{256, 128, 1};
    case 4: return volume ? TileShape{32, 32, 16} : TileShape{128, 128, 1};
    case 8: return volume ? TileShape{32, 16, 16} : TileShape{128, 64, 1};
    case 16: return volume ? TileShape{16, 16, 16} : TileShape{64, 64, 1};
    default: return {};
    }
}

// False when the resource cannot be tiled: an untileable element size, no mips, or more than 2^32 tiles.
inline bool resource_tiling(const TilingDesc& desc, ResourceTiling& out) {
    out = {};
    const auto up = [](std::uint64_t value, std::uint64_t by) { return (value + by - 1) / by; };
    if (desc.kind == TiledKind::buffer) {
        const std::uint64_t tiles = up(desc.width, tile_size_in_bytes);
        if (!tiles || tiles > 0xffffffffu) return false;
        out.total_tiles = std::uint32_t(tiles);
        out.shape = {tile_size_in_bytes, 1, 1};
        out.subresources = {{out.total_tiles, 1, 1, 0}};
        return true;
    }
    const bool volume = desc.kind == TiledKind::texture3d;
    const TileShape elements = element_tile_shape(volume, desc.element_bytes);
    if (!elements.width || !desc.mips || !desc.block || !desc.width || !desc.height || !desc.depth_or_layers)
        return false;
    out.shape = {elements.width * desc.block, elements.height * desc.block, elements.depth};
    const std::uint32_t layers = volume ? 1 : desc.depth_or_layers;
    const std::uint32_t depth = volume ? desc.depth_or_layers : 1;

    // One array slice: standard mips while every axis holds a whole tile, then the packed tail.
    std::vector<SubresourceTiling> slice(desc.mips);
    std::uint64_t standard_tiles = 0, packed_bytes = 0;
    for (std::uint32_t mip = 0; mip < desc.mips; ++mip) {
        const std::uint64_t w = std::max<std::uint64_t>(1, desc.width >> mip);
        const std::uint32_t h = std::max(1u, desc.height >> mip), d = std::max(1u, depth >> mip);
        const bool standard = out.packed_mips == 0 && w >= out.shape.width && h >= out.shape.height &&
                              d >= out.shape.depth;
        if (standard) {
            const std::uint64_t tw = up(w, out.shape.width), th = up(h, out.shape.height), td = up(d, out.shape.depth);
            slice[mip] = {std::uint32_t(tw), std::uint32_t(th), std::uint32_t(td), std::uint32_t(standard_tiles)};
            standard_tiles += tw * th * td;
            ++out.standard_mips;
        } else {
            slice[mip] = {0, 0, 0, packed_tile};
            packed_bytes += up(w, desc.block) * up(h, desc.block) * d * desc.element_bytes;
            ++out.packed_mips;
        }
    }
    const std::uint64_t packed_tiles = out.packed_mips ? std::max<std::uint64_t>(1, up(packed_bytes, tile_size_in_bytes)) : 0;
    const std::uint64_t per_slice = standard_tiles + packed_tiles;
    if (per_slice * layers > 0xffffffffu) return false;
    out.packed_tiles = std::uint32_t(packed_tiles);
    out.packed_start = out.packed_mips ? std::uint32_t(standard_tiles) : 0;
    out.total_tiles = std::uint32_t(per_slice * layers);
    out.subresources.reserve(std::size_t(desc.mips) * layers);
    for (std::uint32_t layer = 0; layer < layers; ++layer)
        for (SubresourceTiling s : slice) {
            if (s.start != packed_tile) s.start += std::uint32_t(per_slice * layer);
            out.subresources.push_back(s);
        }
    return true;
}

} // namespace leonida
