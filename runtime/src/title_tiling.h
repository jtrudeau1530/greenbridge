#pragma once

// The console layout of a title's textures in memory (D3D12_RESOURCE_DESC::Layout = 0x100 | GFX10 swizzle mode), as the
// repository measured it (docs/textures.md, tools/tex_headers.py, tools/ttpf_mips.py): the GFX10 mip chain, standalone
// mips smallest-first after one shared tail tile, SW_4KB_S (mode 5) and SW_64KB_D (mode 10) with the RbPlus patterns.
// Mode 0 is linear. arrays and cubes are slice-major in memory, volumes single-mip SW_4KB_S3.

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <vector>

namespace leonida::title_tiling {

struct Mip {
    std::uint32_t blocks_w = 0, blocks_h = 0, padded_w = 0, padded_h = 0, tile_x = 0, tile_y = 0;
    std::uint64_t offset = 0, region = 0;  // in one slice (slice-major) or of the mip's first slice (mip-major)
    bool in_tail = false;
};
struct Layout {
    bool ok = false;
    std::uint64_t slice_size = 0, size = 0;
    std::uint32_t tile = 0, block_w = 0, block_h = 0, first_tail = 0, bytes = 0, slices = 1;
    bool mip_major = false, linear = false;
    std::vector<Mip> mips;
};

inline int element_log(std::uint32_t bytes) {
    switch (bytes) {
        case 1: return 0;
        case 2: return 1;
        case 4: return 2;
        case 8: return 3;
        case 16: return 4;
        default: return -1;
    }
}

// Bit orders: bit i of the in-tile offset comes from x bit (s - 1) for s > 0, y bit (-s - 1) for s < 0.
inline const signed char* pattern(std::uint32_t tile, int log) {
    static constexpr signed char p4[5][16] = {
        {1, 2, 3, 4, -1, -2, -3, -4, -5, 5, -6, 6},   {0, 1, 2, 3, -1, -2, -3, 4, -4, 5, -5, 6},
        {0, 0, 1, 2, -1, -2, -3, 3, -4, 4, -5, 5},    {0, 0, 0, 1, -1, -2, 2, 3, -3, 4, -4, 5},
        {0, 0, 0, 0, -1, -2, 1, 2, -3, 3, -4, 4}};
    static constexpr signed char p64[5][16] = {
        {1, 2, 3, -2, -1, -3, 4, -4, -5, 5, -6, 6, -7, 7, -8, 8}, {0, 1, 2, 3, -1, -2, -3, 4, -4, 5, -5, 6, -6, 7, -7, 8},
        {0, 0, 1, 2, -1, -2, 3, -3, -4, 4, -5, 5, -6, 6, -7, 7},  {0, 0, 0, 1, -1, 2, 3, -2, -3, 4, -4, 5, -5, 6, -6, 7},
        {0, 0, 0, 0, 1, -1, 2, -2, -3, 3, -4, 4, -5, 5, -6, 6}};
    return tile == 4096 ? p4[log] : p64[log];
}

inline Layout layout_2d(std::uint32_t mode, std::uint32_t width, std::uint32_t height, std::uint32_t slices, std::uint32_t mips,
                        std::uint32_t bytes, std::uint32_t block) {
    Layout l;
    const int log = element_log(bytes);
    if (log < 0 || !width || !height || !mips) return l;
    l.bytes = bytes;
    l.slices = std::max(1u, slices);
    const std::uint32_t ew = (width + block - 1) / block, eh = (height + block - 1) / block;
    const auto at_mip = [](std::uint32_t v, std::uint32_t i) { return std::max(1u, (v + (1u << i) - 1) >> i); };
    if (mode == 0) {  // linear: rows of 256 bytes, mips smallest first
        l.linear = true;
        l.first_tail = mips;
        l.mips.resize(mips);
        std::uint64_t off = 0;
        for (int i = int(mips) - 1; i >= 0; --i) {
            Mip& m = l.mips[i];
            m.blocks_w = at_mip(ew, i);
            m.blocks_h = at_mip(eh, i);
            m.padded_w = ((m.blocks_w * bytes + 255) & ~255u) / bytes;
            m.padded_h = m.blocks_h;
            m.offset = off;
            m.region = std::uint64_t(m.padded_w) * bytes * m.padded_h;
            off += m.region;
        }
        l.slice_size = off;
        l.size = off * l.slices;
        l.ok = true;
        return l;
    }
    if (mode != 5 && mode != 10) return l;
    l.tile = mode == 5 ? 4096 : 65536;
    l.mip_major = false;  // in memory every slice holds its whole chain (the pool files are mip-major, the title rearranges)
    const int half = mode == 5 ? 6 : 8;
    const std::uint32_t max_tail = mode == 5 ? 8 : 12;
    l.block_w = 1u << (half - log / 2);
    l.block_h = 1u << (half - (log + 1) / 2);
    static constexpr std::uint32_t origin[5][2] = {{16, 16}, {16, 8}, {8, 8}, {8, 4}, {4, 4}};
    l.first_tail = mips;
    for (std::uint32_t i = 0; i < mips; ++i) {
        Mip m;
        m.blocks_w = at_mip(ew, i);
        m.blocks_h = at_mip(eh, i);
        if (mips > 1 && m.blocks_w <= l.block_w / 2 && m.blocks_h <= l.block_h && mips - i <= max_tail) {
            l.first_tail = i;
            break;
        }
        m.padded_w = (m.blocks_w + l.block_w - 1) / l.block_w * l.block_w;
        m.padded_h = (m.blocks_h + l.block_h - 1) / l.block_h * l.block_h;
        m.region = std::uint64_t(m.padded_w) * m.padded_h * bytes;
        l.mips.push_back(m);
    }
    const std::uint32_t copies = l.mip_major ? l.slices : 1;
    std::uint64_t off = l.first_tail < mips ? std::uint64_t(l.tile) * copies : 0;
    for (int i = int(l.first_tail) - 1; i >= 0; --i) {
        l.mips[i].offset = off;
        off += l.mips[i].region * copies;
    }
    for (std::uint32_t i = l.first_tail; i < mips; ++i) {
        const std::uint32_t entry = max_tail - 1 - (i - l.first_tail);
        const std::uint32_t moff = entry > 6 ? 16u << entry : entry << 8;
        std::uint32_t x = 0, y = 0;
        for (int bit = 0; bit < 6; ++bit) {
            x |= ((moff >> (9 + 2 * bit)) & 1) << bit;
            y |= ((moff >> (8 + 2 * bit)) & 1) << bit;
        }
        Mip m;
        m.blocks_w = at_mip(ew, i);
        m.blocks_h = at_mip(eh, i);
        m.padded_w = l.block_w;
        m.padded_h = l.block_h;
        m.region = l.tile;
        m.tile_x = x * origin[log][0];
        m.tile_y = y * origin[log][1];
        m.in_tail = true;
        l.mips.push_back(m);
    }
    l.slice_size = off;
    l.size = l.mip_major ? off : off * l.slices;
    l.ok = true;
    return l;
}

// Where slice `s` of `mip` starts.
inline std::uint64_t slice_offset(const Layout& l, const Mip& m, std::uint32_t s) {
    return l.mip_major ? m.offset + s * m.region : s * l.slice_size + m.offset;
}

// One mip of one slice out of `tiled` (the whole allocation) into tight rows.
inline bool detile(const Layout& l, std::uint32_t mip, std::uint32_t slice, const std::uint8_t* tiled, std::uint64_t tiled_size,
                   std::vector<std::uint8_t>& out) {
    const Mip& m = l.mips[mip];
    const std::uint64_t base = slice_offset(l, m, slice);
    out.assign(std::size_t(m.blocks_w) * m.blocks_h * l.bytes, 0);
    if (l.linear) {
        const std::uint64_t row = std::uint64_t(m.padded_w) * l.bytes;
        for (std::uint32_t y = 0; y < m.blocks_h; ++y) {
            if (base + y * row + std::uint64_t(m.blocks_w) * l.bytes > tiled_size) return false;
            std::memcpy(out.data() + std::size_t(y) * m.blocks_w * l.bytes, tiled + base + y * row, std::size_t(m.blocks_w) * l.bytes);
        }
        return true;
    }
    const signed char* pat = pattern(l.tile, element_log(l.bytes));
    const int bits = l.tile == 4096 ? 12 : 16;
    std::vector<std::uint32_t> xo(m.blocks_w);
    for (std::uint32_t x = 0; x < m.blocks_w; ++x) {
        std::uint32_t o = (x / l.block_w) * l.tile;
        const std::uint32_t xx = x + m.tile_x;
        for (int b = 0; b < bits; ++b)
            if (pat[b] > 0) o |= ((xx >> (pat[b] - 1)) & 1u) << b;
        xo[x] = o;
    }
    const std::uint64_t row = std::uint64_t(m.padded_w / l.block_w) * l.tile;
    for (std::uint32_t y = 0; y < m.blocks_h; ++y) {
        std::uint64_t o = base + (y / l.block_h) * row;
        const std::uint32_t yy = y + m.tile_y;
        std::uint32_t low = 0;
        for (int b = 0; b < bits; ++b)
            if (pat[b] < 0) low |= ((yy >> (-pat[b] - 1)) & 1u) << b;
        o += low;
        std::uint8_t* d = out.data() + std::size_t(y) * m.blocks_w * l.bytes;
        for (std::uint32_t x = 0; x < m.blocks_w; ++x, d += l.bytes) {
            const std::uint64_t at = o + xo[x];
            if (at + l.bytes > tiled_size) return false;
            std::memcpy(d, tiled + at, l.bytes);
        }
    }
    return true;
}

// Single-mip SW_4KB_S volumes: the whole volume in tight rows and slices.
inline bool detile_volume(std::uint32_t width, std::uint32_t height, std::uint32_t depth, std::uint32_t bytes, const std::uint8_t* tiled,
                          std::uint64_t tiled_size, std::vector<std::uint8_t>& out) {
    static constexpr signed char p[5][12] = {{1, 2, 101, -1, 102, -2, 3, 103, -3, 4, 104, -4},
                                             {0, 1, 101, -1, 102, -2, 2, 103, -3, 3, 104, -4},
                                             {0, 0, 1, -1, 101, -2, 2, 102, -3, 3, 103, -4},
                                             {0, 0, 0, 1, 101, -1, 2, 102, -2, 3, 103, -3},
                                             {0, 0, 0, 0, 101, -1, 1, 102, -2, 2, 103, -3}};
    static constexpr std::uint32_t t[5][3] = {{16, 16, 16}, {8, 16, 16}, {8, 16, 8}, {8, 8, 8}, {4, 8, 8}};
    const int log = element_log(bytes);
    if (log < 0) return false;
    const std::uint32_t bw = t[log][0], bh = t[log][1], bd = t[log][2];
    const std::uint32_t pw = (width + bw - 1) / bw, ph = (height + bh - 1) / bh;
    const std::uint64_t z_stride = std::uint64_t(pw) * ph * 4096;
    out.assign(std::size_t(width) * height * depth * bytes, 0);
    std::uint8_t* d = out.data();
    for (std::uint32_t z = 0; z < depth; ++z)
        for (std::uint32_t y = 0; y < height; ++y)
            for (std::uint32_t x = 0; x < width; ++x, d += bytes) {
                std::uint64_t o = (z / bd) * z_stride + (std::uint64_t(y / bh) * pw + x / bw) * 4096;
                for (int b = 0; b < 12; ++b) {
                    const int s = p[log][b];
                    if (s >= 101) o |= std::uint64_t((z >> (s - 101)) & 1) << b;
                    else if (s > 0) o |= std::uint64_t((x >> (s - 1)) & 1) << b;
                    else if (s < 0) o |= std::uint64_t((y >> (-s - 1)) & 1) << b;
                }
                if (o + bytes > tiled_size) return false;
                std::memcpy(d, tiled + o, bytes);
            }
    return true;
}
inline std::uint64_t volume_size(std::uint32_t width, std::uint32_t height, std::uint32_t depth, std::uint32_t bytes) {
    static constexpr std::uint32_t t[5][3] = {{16, 16, 16}, {8, 16, 16}, {8, 16, 8}, {8, 8, 8}, {4, 8, 8}};
    const int log = element_log(bytes);
    if (log < 0) return 0;
    const std::uint32_t pw = (width + t[log][0] - 1) / t[log][0], ph = (height + t[log][1] - 1) / t[log][1],
                        pd = (depth + t[log][2] - 1) / t[log][2];
    return std::uint64_t(pw) * ph * pd * 4096;
}

}  // namespace leonida::title_tiling
