// Boot stand-in: textures and buffers the title places at its own CPU addresses (no unified GPU memory on the PC).
#pragma once
#include <cstdint>
struct ID3D12Device;
struct ID3D12Resource;
#include <algorithm>
#include <vector>
namespace rt::title_textures {
// The one layout title-side texture memory has here: every subresource linear, rows 256-byte aligned, mips of a slice in
// order, slices one after another. The XG computer reports it and Map/Unmap reads it back.
struct MipLayout {
    std::uint64_t offset, size, slice2d;  // offset inside the array slice
    std::uint32_t pitch, rows, width, height, depth, elements_wide, bytes_per_element, block;
};
struct LinearLayout {
    std::uint64_t slice_size = 0, total = 0;
    std::uint32_t layers = 1;
    std::vector<MipLayout> mips;
};
inline void format_block(std::uint32_t f, std::uint32_t& bytes, std::uint32_t& edge) {
    edge = 1;
    bytes = 4;
    if ((f >= 70 && f <= 72) || (f >= 79 && f <= 81)) { bytes = 8; edge = 4; }
    else if ((f >= 73 && f <= 78) || (f >= 82 && f <= 84) || (f >= 94 && f <= 99)) { bytes = 16; edge = 4; }
    else if (f >= 1 && f <= 4) bytes = 16;
    else if (f >= 5 && f <= 8) bytes = 12;
    else if (f >= 9 && f <= 22) bytes = 8;
    else if ((f >= 48 && f <= 59) || f == 85 || f == 86 || f == 115) bytes = 2;
    else if (f >= 60 && f <= 66) bytes = 1;
}
inline LinearLayout linear_layout(std::uint32_t dimension, std::uint64_t width, std::uint32_t height, std::uint32_t depth_or_layers,
                                  std::uint32_t mips, std::uint32_t format) {
    LinearLayout l;
    std::uint32_t bytes, edge;
    format_block(format, bytes, edge);
    const bool volume = dimension == 4;
    l.layers = volume ? 1 : std::max(1u, depth_or_layers);
    std::uint64_t at = 0;
    for (std::uint32_t m = 0; m < std::max(1u, mips); ++m) {
        MipLayout k{};
        k.width = static_cast<std::uint32_t>(std::max<std::uint64_t>(1, width >> m));
        k.height = std::max(1u, height >> m);
        k.depth = volume ? std::max(1u, depth_or_layers >> m) : 1;
        k.block = edge;
        k.bytes_per_element = bytes;
        k.elements_wide = (k.width + edge - 1) / edge;
        k.rows = (k.height + edge - 1) / edge;
        k.pitch = (k.elements_wide * bytes + 255) & ~255u;
        k.slice2d = std::uint64_t(k.pitch) * k.rows;
        k.size = k.slice2d * k.depth;
        k.offset = at;
        at += (k.size + 255) & ~std::uint64_t{255};
        l.mips.push_back(k);
    }
    l.slice_size = at;
    l.total = (at * l.layers + 0xFFFF) & ~std::uint64_t{0xFFFF};
    return l;
}
// XG texture computer CopyIntoSubresource: the linear data for one subresource of the resource at `base`.
void note_linear_upload(std::uint64_t base, std::uint32_t subresource, const void* data, std::uint32_t row_pitch,
                        std::uint32_t slice_pitch, const std::uint32_t* resource_desc);
// Device slot 70 (resource description, SRV description with the address at +0x28, destination descriptor).
bool create_view(ID3D12Device* device, std::uint64_t resource_desc, std::uint64_t view_desc, std::uint64_t destination);
// A layer texture whose title address is `base` (Resource::title_shadow): uploads to it go to the image.
void register_shadow(std::uint64_t base, ID3D12Resource* resource);
// Before each ExecuteCommandLists: late uploads and buffer contents.
void refresh(ID3D12Device* device);
}  // namespace rt::title_textures
