// D3D12 COM objects over the Vulkan backend (src/vulkan_backend.h): command allocator, graphics command list,
// command queue and fence. Their vtables are the Windows SDK's (ID3D12GraphicsCommandList7, ID3D12CommandQueue,
// ID3D12CommandAllocator, ID3D12Fence1); every method not implemented here is a generated stub that reports its
// name once (tools/gen_d3d12_stubs.py). The Xbox-specific slots measured for the device do not apply to these
// interfaces as far as is known; docs/d3d12_vulkan.md section 3.1 lists what is still unverified.
//
// Contracts with the rest of the layer (the device proxy that will hand these objects out):
// - an ID3D12Resource's leonida::ResourceId is its pointer value; the device registers it with the VulkanContext;
// - the CPU handle of an RTV or DSV points at a leonida::TargetView in the device's descriptor heap.
#pragma once

#include <windows.h>
#include <d3d12.h>

#include "../src/vulkan_backend.h"

namespace rt::d3d12 {

HRESULT create_command_queue(leonida::VulkanContext& context, VkQueue queue, std::uint32_t queue_family,
                             const D3D12_COMMAND_QUEUE_DESC& desc, REFIID riid, void** out);
HRESULT create_command_allocator(D3D12_COMMAND_LIST_TYPE type, REFIID riid, void** out);
HRESULT create_command_list(D3D12_COMMAND_LIST_TYPE type, ID3D12CommandAllocator* allocator, REFIID riid,
                            void** out);
HRESULT create_fence(leonida::VulkanContext& context, UINT64 initial_value, D3D12_FENCE_FLAGS flags, REFIID riid,
                     void** out);

// ExecuteCommandLists returns void; this is what the last call on the queue did.
leonida::ReplayResult last_execute_result(ID3D12CommandQueue* queue);

inline leonida::ResourceId resource_id(const ID3D12Resource* resource) {
    return reinterpret_cast<leonida::ResourceId>(resource);
}

}  // namespace rt::d3d12
