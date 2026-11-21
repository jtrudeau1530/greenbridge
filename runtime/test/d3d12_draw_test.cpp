// Milestone M1 (docs/d3d12_vulkan.md section 6) through the Xbox-ABI proxy over the Vulkan device:
//   - a graphics PSO whose VS and PS come from containers shaped like the game's, found in a translator-style
//     index by the SHA-256 of their FSSM programs; the input layout (POSITION, COLOR) is matched to the VS input
//     signature by semantic, and the VS declares them in the opposite register order;
//   - a vertex buffer copied into the GPU address arena, an index buffer read straight from an upload heap, root
//     constants (a clip-space offset) and a root CBV (a tint) through the 32 user-data dwords;
//   - a render target cleared blue, two indexed draws with different root arguments (top-left red, bottom-right
//     green), and one back-facing triangle (bottom-left) that culling must remove;
//   - the image read back through CopyTextureRegion and compared pixel by pixel. Where the quadrants land also
//     checks the viewport flip (D3D clip space is y-up) and the front-face winding.
// The shaders are GLSL written to the translator's contract (test/glsl/vs_draw.vert, ps_draw.frag). No game data.
// Set LEONIDA_VK_VALIDATION=1 to run under VK_LAYER_KHRONOS_validation.
#include "vk_test_device.h"
#include "d3d12_test_shaders.h"
#include "../shims/d3d12_com.h"
#include "../shims/d3d12_xs_device.h"
#include "../src/shader_library.h"

#include <array>
#include <cstring>

#if !defined(LEONIDA_TEST_VS_SPV) || !defined(LEONIDA_TEST_PS_SPV)
#error "LEONIDA_TEST_VS_SPV and LEONIDA_TEST_PS_SPV must name the compiled test shaders"
#endif

#define EXPECT(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); return 1; } \
} while (0)

using namespace d3d12test;
using XboxHeapHandleFn = std::uint64_t(WINAPI*)(void* heap);  // heap slot 9: handle in RAX

namespace {

struct Vertex {
    float x, y;
    std::uint8_t r, g, b, a;
};

}  // namespace

int main() {
    vktest::Device d;
    EXPECT(d.create());

    // Shader programs and the index that maps their hashes to the compiled test modules.
    const auto vs_program = fssm_program({0x56530001, 0xbf810000});
    const auto ps_program = fssm_program({0x50530001, 0xbf810000});
    const auto vs_container = shader_container(vs_program), ps_container = shader_container(ps_program);
    const auto library = std::filesystem::temp_directory_path() / "leonida_draw_test";
    std::filesystem::create_directories(library);
    std::filesystem::copy_file(LEONIDA_TEST_VS_SPV, library / "vs_draw.spv", std::filesystem::copy_options::overwrite_existing);
    std::filesystem::copy_file(LEONIDA_TEST_PS_SPV, library / "ps_draw.spv", std::filesystem::copy_options::overwrite_existing);
    {
        std::ofstream index(library / "index.json", std::ios::binary);
        index << "{\"vs_draw\": {\"stage\": \"vs\", \"fssm_sha256\": \"" << leonida::sha256_hex(vs_program)
              << "\", \"bindings\": [], \"push_constant_bytes\": 144, \"input_signature\": ["
                 "{\"semantic\": \"COLOR\", \"semantic_index\": 0, \"register\": 0, \"mask\": 15, \"rw_mask\": 15, "
                 "\"system_value\": 0, \"component_type\": \"float\"}, "
                 "{\"semantic\": \"POSITION\", \"semantic_index\": 0, \"register\": 1, \"mask\": 3, \"rw_mask\": 3, "
                 "\"system_value\": 0, \"component_type\": \"float\"}]},\n"
                 " \"ps_draw\": {\"stage\": \"ps\", \"fssm_sha256\": \"" << leonida::sha256_hex(ps_program)
              << "\", \"bindings\": [], \"push_constant_bytes\": 144, \"outputs\": [0]}}";
    }

    constexpr UINT size = 64;
    std::vector<std::uint8_t> pixels(size * 256);
    {
        rt::d3d12::VulkanDeviceInfo info{};
        info.get_instance_proc = d.instance_proc;
        info.instance = d.instance;
        info.physical = d.physical;
        info.device = d.device;
        info.queue = d.queue;
        info.queue_family = d.family;
        info.buffer_device_address = true;
        info.shader_dir = library.string();
        ID3D12Device* inner = nullptr;
        EXPECT(SUCCEEDED(rt::d3d12::create_vulkan_device(info, IID_PPV_ARGS(&inner))));
        Ref<ID3D12Device> device;
        EXPECT(SUCCEEDED(rt::d3d12_xs::create_device_proxy(inner, __uuidof(ID3D12Device),
                                                           reinterpret_cast<void**>(&device.p))));
        rt::d3d12::set_device_outer(inner, device.p);

        // Root signature: [0] two constants (s0, s1: offset), [1] a root CBV (s2:s3: tint).
        D3D12_ROOT_PARAMETER params[2] = {};
        params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
        params[0].Constants = {0, 0, 2};
        params[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
        params[1].Descriptor = {1, 0};
        D3D12_VERSIONED_ROOT_SIGNATURE_DESC root_desc{};
        root_desc.Version = D3D_ROOT_SIGNATURE_VERSION_1_0;
        root_desc.Desc_1_0 = {2, params, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT};
        const auto root_blob = serialize_root_signature(root_desc);
        EXPECT(!root_blob.empty());
        Ref<ID3D12RootSignature> root;
        EXPECT(SUCCEEDED(device->CreateRootSignature(0, root_blob.data(), root_blob.size(), IID_PPV_ARGS(&root.p))));

        // The PSO. Input layout order (POSITION, COLOR) differs from the VS's registers (COLOR 0, POSITION 1).
        const D3D12_INPUT_ELEMENT_DESC elements[2] = {
            {"POSITION", 0, DXGI_FORMAT_R32G32_FLOAT, 0, 0, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
            {"COLOR", 0, DXGI_FORMAT_R8G8B8A8_UNORM, 0, D3D12_APPEND_ALIGNED_ELEMENT,
             D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0}};
        D3D12_GRAPHICS_PIPELINE_STATE_DESC pso_desc{};
        pso_desc.pRootSignature = root.p;
        pso_desc.VS = {vs_container.data(), vs_container.size()};
        pso_desc.PS = {ps_container.data(), ps_container.size()};
        pso_desc.BlendState.RenderTarget[0].RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;
        pso_desc.SampleMask = 0xffffffff;
        pso_desc.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
        pso_desc.RasterizerState.CullMode = D3D12_CULL_MODE_BACK;
        pso_desc.RasterizerState.DepthClipEnable = TRUE;
        pso_desc.InputLayout = {elements, 2};
        pso_desc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
        pso_desc.NumRenderTargets = 1;
        pso_desc.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
        pso_desc.SampleDesc.Count = 1;
        Ref<ID3D12PipelineState> pso, unfed;
        EXPECT(SUCCEEDED(device->CreateGraphicsPipelineState(&pso_desc, IID_PPV_ARGS(&pso.p))));
        pso_desc.InputLayout = {elements, 1};  // COLOR missing: D3D12 rejects a layout that leaves an input unfed
        EXPECT(device->CreateGraphicsPipelineState(&pso_desc, IID_PPV_ARGS(&unfed.p)) == E_INVALIDARG && !unfed.p);

        // Geometry: a quad in the top-left quadrant (front-facing, clockwise in D3D screen space) and a triangle in
        // the bottom-left one wound the other way (back-facing).
        const Vertex vertices[7] = {
            {-1, 1, 255, 0, 0, 255}, {0, 1, 255, 0, 0, 255}, {-1, 0, 255, 0, 0, 255}, {0, 0, 255, 0, 0, 255},
            {-1, 0, 255, 0, 0, 255}, {-1, -1, 255, 0, 0, 255}, {0, -1, 255, 0, 0, 255}};
        const std::uint16_t indices[6] = {0, 1, 2, 2, 1, 3};
        const float tints[2][3] = {{1, 0, 1}, {0, 1, 0}};  // red stays red; then green

        const D3D12_HEAP_PROPERTIES default_heap{D3D12_HEAP_TYPE_DEFAULT};
        const D3D12_HEAP_PROPERTIES upload_heap{D3D12_HEAP_TYPE_UPLOAD};
        const D3D12_HEAP_PROPERTIES readback_heap{D3D12_HEAP_TYPE_READBACK};
        Ref<ID3D12Resource> vertex_buffer, upload, readback, target;
        const auto vb_desc = buffer_desc(sizeof(vertices));
        EXPECT(SUCCEEDED(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &vb_desc,
                                                         D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                                         IID_PPV_ARGS(&vertex_buffer.p))));
        const auto upload_desc = buffer_desc(4096);
        EXPECT(SUCCEEDED(device->CreateCommittedResource(&upload_heap, D3D12_HEAP_FLAG_NONE, &upload_desc,
                                                         D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
                                                         IID_PPV_ARGS(&upload.p))));
        const auto readback_desc = buffer_desc(size * 256);
        EXPECT(SUCCEEDED(device->CreateCommittedResource(&readback_heap, D3D12_HEAP_FLAG_NONE, &readback_desc,
                                                         D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                                         IID_PPV_ARGS(&readback.p))));
        D3D12_RESOURCE_DESC rt_desc{};
        rt_desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        rt_desc.Width = rt_desc.Height = size;
        rt_desc.DepthOrArraySize = 1;
        rt_desc.MipLevels = 1;
        rt_desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        rt_desc.SampleDesc.Count = 1;
        rt_desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
        EXPECT(SUCCEEDED(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &rt_desc,
                                                         D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&target.p))));

        // Upload layout: vertices at 0, indices at 256, the tints at 512 and 768 (CBVs are 256-byte aligned).
        std::uint8_t* up = nullptr;
        EXPECT(SUCCEEDED(upload->Map(0, nullptr, reinterpret_cast<void**>(&up))));
        std::memcpy(up, vertices, sizeof(vertices));
        std::memcpy(up + 256, indices, sizeof(indices));
        std::memcpy(up + 512, tints[0], sizeof(tints[0]));
        std::memcpy(up + 768, tints[1], sizeof(tints[1]));
        const D3D12_GPU_VIRTUAL_ADDRESS upload_va = upload->GetGPUVirtualAddress();

        const D3D12_DESCRIPTOR_HEAP_DESC heap_desc{D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 1, D3D12_DESCRIPTOR_HEAP_FLAG_NONE, 0};
        Ref<ID3D12DescriptorHeap> rtv_heap;
        EXPECT(SUCCEEDED(device->CreateDescriptorHeap(&heap_desc, IID_PPV_ARGS(&rtv_heap.p))));
        const D3D12_CPU_DESCRIPTOR_HANDLE rtv{
            SIZE_T(reinterpret_cast<XboxHeapHandleFn>(slot(rtv_heap.p, 9))(rtv_heap.p))};
        device->CreateRenderTargetView(target.p, nullptr, rtv);

        D3D12_COMMAND_QUEUE_DESC queue_desc{};
        Ref<ID3D12CommandQueue> queue;
        Ref<ID3D12CommandAllocator> allocator;
        Ref<ID3D12GraphicsCommandList> list;
        Ref<ID3D12Fence> fence;
        EXPECT(SUCCEEDED(device->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&queue.p))));
        EXPECT(SUCCEEDED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator.p))));
        EXPECT(SUCCEEDED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.p, pso.p,
                                                   IID_PPV_ARGS(&list.p))));
        EXPECT(SUCCEEDED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence.p))));

        // Upload the vertices, prepare the target.
        list->CopyBufferRegion(vertex_buffer.p, 0, upload.p, 0, sizeof(vertices));
        D3D12_RESOURCE_BARRIER barriers[2] = {
            transition(vertex_buffer.p, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER),
            transition(target.p, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_RENDER_TARGET)};
        list->ResourceBarrier(2, barriers);
        const FLOAT blue[4] = {0, 0, 1, 1};
        list->ClearRenderTargetView(rtv, blue, 0, nullptr);

        // State for the draws; the pipeline came with CreateCommandList.
        list->SetGraphicsRootSignature(root.p);
        list->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        const D3D12_VERTEX_BUFFER_VIEW vbv{vertex_buffer->GetGPUVirtualAddress(), sizeof(vertices), sizeof(Vertex)};
        list->IASetVertexBuffers(0, 1, &vbv);
        const D3D12_INDEX_BUFFER_VIEW ibv{upload_va + 256, sizeof(indices), DXGI_FORMAT_R16_UINT};
        list->IASetIndexBuffer(&ibv);
        const D3D12_VIEWPORT viewport{0, 0, float(size), float(size), 0, 1};
        const D3D12_RECT scissor{0, 0, LONG(size), LONG(size)};
        list->RSSetViewports(1, &viewport);
        list->RSSetScissorRects(1, &scissor);
        list->OMSetRenderTargets(1, &rtv, FALSE, nullptr);

        // Draw 1: no offset, red tint -> top-left quadrant red.
        const float origin[2] = {0, 0}, down_right[2] = {1, -1};
        list->SetGraphicsRoot32BitConstants(0, 2, origin, 0);
        list->SetGraphicsRootConstantBufferView(1, upload_va + 512);
        list->DrawIndexedInstanced(6, 1, 0, 0, 0);
        // Draw 2: offset (1, -1), green tint -> bottom-right quadrant green.
        list->SetGraphicsRoot32BitConstants(0, 2, down_right, 0);
        list->SetGraphicsRootConstantBufferView(1, upload_va + 768);
        list->DrawIndexedInstanced(6, 1, 0, 0, 0);
        // Draw 3: the back-facing triangle in the bottom-left quadrant; culled, so that quadrant stays blue.
        list->SetGraphicsRoot32BitConstants(0, 2, origin, 0);
        list->SetGraphicsRootConstantBufferView(1, upload_va + 512);
        list->DrawInstanced(3, 1, 4, 0);

        D3D12_RESOURCE_BARRIER to_copy = transition(target.p, D3D12_RESOURCE_STATE_RENDER_TARGET,
                                                    D3D12_RESOURCE_STATE_COPY_SOURCE);
        list->ResourceBarrier(1, &to_copy);
        D3D12_TEXTURE_COPY_LOCATION dst{};
        dst.pResource = readback.p;
        dst.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
        dst.PlacedFootprint = {0, {DXGI_FORMAT_R8G8B8A8_UNORM, size, size, 1, 256}};
        D3D12_TEXTURE_COPY_LOCATION src{};
        src.pResource = target.p;
        src.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
        EXPECT(list->Close() == S_OK);
        ID3D12CommandList* lists[] = {list.p};
        queue->ExecuteCommandLists(1, lists);
        EXPECT(rt::d3d12::last_execute_result(queue.p) == leonida::ReplayResult::success);
        EXPECT(SUCCEEDED(queue->Signal(fence.p, 1)));
        EXPECT(SUCCEEDED(fence->SetEventOnCompletion(1, nullptr)));
        void* mapped = nullptr;
        EXPECT(SUCCEEDED(readback->Map(0, nullptr, &mapped)));
        std::memcpy(pixels.data(), mapped, pixels.size());

        // A draw without a topology is invalid; Close reports it.
        EXPECT(SUCCEEDED(allocator->Reset()) && SUCCEEDED(list->Reset(allocator.p, pso.p)));
        list->SetGraphicsRootSignature(root.p);
        list->DrawInstanced(3, 1, 0, 0);
        EXPECT(list->Close() == E_INVALIDARG);

        for (auto* r : {&vertex_buffer, &upload, &readback, &target}) r->reset();
        pso.reset();
        root.reset();
        rtv_heap.reset();
        list.reset();
        allocator.reset();
        queue.reset();
        fence.reset();
        EXPECT(rt::d3d12_xs::device_proxy_ref_count(device.p) == 1);
    }
    std::filesystem::remove_all(library);

    unsigned mismatches = 0;
    for (UINT row = 0; row < size; ++row) {
        for (UINT col = 0; col < size; ++col) {
            const bool top = row < size / 2, left = col < size / 2;
            const std::array<std::uint8_t, 4> want = top && left     ? std::array<std::uint8_t, 4>{255, 0, 0, 255}
                                                     : !top && !left ? std::array<std::uint8_t, 4>{0, 255, 0, 255}
                                                                     : std::array<std::uint8_t, 4>{0, 0, 255, 255};
            const std::uint8_t* got = pixels.data() + row * 256 + col * 4;
            if (std::equal(want.begin(), want.end(), got)) continue;
            if (mismatches++ < 6)
                std::fprintf(stderr, "pixel (%u, %u): got %u,%u,%u,%u want %u,%u,%u,%u\n", col, row, got[0], got[1],
                             got[2], got[3], want[0], want[1], want[2], want[3]);
        }
    }
    EXPECT(!mismatches);
    if (d.validation_errors()) {
        std::fprintf(stderr, "%u validation errors\n", d.validation_errors());
        return 1;
    }
    std::printf("M1: two indexed draws and a culled triangle matched all %u pixels: input layout by semantic, root "
                "constants and a root CBV per draw, vertex buffer in the arena, index buffer in an upload heap\n",
                size * size);
    return 0;
}
