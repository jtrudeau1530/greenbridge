// Boot stand-in: textures and buffers the title places at its own CPU addresses (no unified GPU memory on the PC).
#pragma once
#include <cstdint>
struct ID3D12Device;
namespace rt::title_textures {
// XG texture computer CopyIntoSubresource: the linear data for one subresource of the resource at `base`.
void note_linear_upload(std::uint64_t base, std::uint32_t subresource, const void* data, std::uint32_t row_pitch,
                        std::uint32_t slice_pitch, const std::uint32_t* resource_desc);
// Device slot 70 (resource description, SRV description with the address at +0x28, destination descriptor).
bool create_view(ID3D12Device* device, std::uint64_t resource_desc, std::uint64_t view_desc, std::uint64_t destination);
// Before each ExecuteCommandLists: late uploads and buffer contents.
void refresh(ID3D12Device* device);
}  // namespace rt::title_textures
