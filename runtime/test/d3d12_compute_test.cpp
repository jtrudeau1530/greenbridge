// The compute path end to end, through the Xbox-ABI proxy over the Vulkan device:
//   - a root signature serialized by the PC runtime, created with CreateRootSignature;
//   - a compute PSO created through device slot 62 (the Xbox shape) from a shader container shaped like the
//     game's (DXBC, DXIL, XDAT 0x0D, XBCH, zlib), found by the SHA-256 of its FSSM program in a translator index;
//   - buffers and a shader-visible descriptor heap in the GPU address arena, an upload buffer reached through its
//     biased console VA;
//   - root constants, a root UAV and a descriptor table set on the command list, two dispatches that each keep the
//     arguments they were recorded with, and the results copied to a readback buffer.
// The shader (test/glsl/cs_user_data.comp) is written to the translator's push-constant and address contract.
// No game data. Set LEONIDA_VK_VALIDATION=1 to run under VK_LAYER_KHRONOS_validation.
#include "vk_test_device.h"
#include "../shims/d3d12_com.h"
#include "../src/shader_library.h"
#include "../shims/d3d12_xs_device.h"

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <vector>

#ifndef LEONIDA_TEST_CS_SPV
#error "LEONIDA_TEST_CS_SPV must name the compiled test shader"
#endif

#define EXPECT(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); return 1; } \
} while (0)

namespace {

using XboxHeapHandleFn = std::uint64_t(WINAPI*)(void* heap);  // heap slots 9/10: handle in RAX
// Device slot 62 as XboxBvhBuilder.dll calls it on 1.0.2.1: (desc, 0, 0, riid, out) (main's 28cf235).
using XboxComputePipelineFn = HRESULT(WINAPI*)(void* device, const D3D12_COMPUTE_PIPELINE_STATE_DESC*, std::uint64_t,
                                               std::uint64_t, const IID*, void**);
// Device slot 67 as amd_fidelityfx_upscaler_xbox.dll calls it: (desc, riid, out).
using XboxNativeComputePipelineFn = HRESULT(WINAPI*)(void* device, const void*, const IID*, void**);
using SerializeFn = HRESULT(WINAPI*)(const D3D12_VERSIONED_ROOT_SIGNATURE_DESC*, ID3DBlob**, ID3DBlob**);

template<class T>
struct Ref {
    T* p = nullptr;
    ~Ref() { reset(); }
    void reset() {
        if (p) p->Release();
        p = nullptr;
    }
    T* operator->() const { return p; }
};

void* slot(void* object, unsigned index) { return (*reinterpret_cast<void***>(object))[index]; }

void put32(std::vector<std::uint8_t>& out, std::uint32_t v) {
    for (int i = 0; i < 4; ++i) out.push_back(std::uint8_t(v >> (8 * i)));
}

// zlib with one stored block (RFC 1950/1951), enough to wrap a test program.
std::vector<std::uint8_t> zlib_stored(const std::vector<std::uint8_t>& data) {
    std::vector<std::uint8_t> out = {0x78, 0x01, 0x01};
    const auto length = std::uint16_t(data.size());
    out.insert(out.end(), {std::uint8_t(length), std::uint8_t(length >> 8), std::uint8_t(~length),
                           std::uint8_t(~length >> 8)});
    out.insert(out.end(), data.begin(), data.end());
    std::uint32_t a = 1, b = 0;
    for (const std::uint8_t x : data) {
        a = (a + x) % 65521;
        b = (b + a) % 65521;
    }
    const std::uint32_t adler = (b << 16) | a;
    out.insert(out.end(), {std::uint8_t(adler >> 24), std::uint8_t(adler >> 16), std::uint8_t(adler >> 8),
                           std::uint8_t(adler)});
    return out;
}

// DXBC { DXIL { stub, XDAT { 0x0D: XBCH + zlib(fssm) } } }, the layout tools/fssm_unpack.py reads.
std::vector<std::uint8_t> shader_container(const std::vector<std::uint8_t>& fssm) {
    const auto stream = zlib_stored(fssm);
    std::vector<std::uint8_t> xbch = {'X', 'B', 'C', 'H'};
    for (const std::uint32_t v : {1u, std::uint32_t(stream.size()), 0u, std::uint32_t(fssm.size()), 0u}) put32(xbch, v);
    xbch.insert(xbch.end(), stream.begin(), stream.end());
    std::vector<std::uint8_t> dxil(16, 0);
    dxil.insert(dxil.end(), {'X', 'D', 'A', 'T'});
    put32(dxil, 1);
    put32(dxil, 12);
    put32(dxil, 0x0d);
    put32(dxil, 16 + 12 + 12);
    put32(dxil, std::uint32_t(xbch.size()));
    dxil.insert(dxil.end(), xbch.begin(), xbch.end());
    std::vector<std::uint8_t> out = {'D', 'X', 'B', 'C'};
    out.resize(28, 0);
    put32(out, 1);
    put32(out, 36);
    out.insert(out.end(), {'D', 'X', 'I', 'L'});
    put32(out, std::uint32_t(dxil.size()));
    out.insert(out.end(), dxil.begin(), dxil.end());
    return out;
}

std::vector<std::uint8_t> serialize(const D3D12_VERSIONED_ROOT_SIGNATURE_DESC& desc) {
    const HMODULE d3d12 = LoadLibraryExA("d3d12.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
    const auto fn = d3d12 ? reinterpret_cast<SerializeFn>(
                                reinterpret_cast<void*>(GetProcAddress(d3d12, "D3D12SerializeVersionedRootSignature")))
                          : nullptr;
    std::vector<std::uint8_t> out;
    ID3DBlob* blob = nullptr;
    if (fn && SUCCEEDED(fn(&desc, &blob, nullptr)) && blob) {
        const auto* p = static_cast<const std::uint8_t*>(blob->GetBufferPointer());
        out.assign(p, p + blob->GetBufferSize());
        blob->Release();
    }
    return out;
}

D3D12_RESOURCE_DESC buffer_desc(UINT64 bytes, D3D12_RESOURCE_FLAGS flags) {
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Width = bytes;
    desc.Height = 1;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    desc.SampleDesc.Count = 1;
    desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    desc.Flags = flags;
    return desc;
}

D3D12_RESOURCE_BARRIER transition(ID3D12Resource* resource, D3D12_RESOURCE_STATES before,
                                  D3D12_RESOURCE_STATES after) {
    D3D12_RESOURCE_BARRIER barrier{};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Transition.pResource = resource;
    barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    barrier.Transition.StateBefore = before;
    barrier.Transition.StateAfter = after;
    return barrier;
}

}  // namespace

int main() {
    vktest::Device d;
    const auto device_status = d.initialize();
    if (device_status == vktest::DeviceStatus::unavailable) {
        std::fprintf(stderr, "SKIP: no usable Vulkan 1.3 device\n");
        return 77;
    }
    EXPECT(device_status == vktest::DeviceStatus::ready);

    // The shader program the PSO carries, and a translator index that maps its hash to the test module.
    constexpr std::uint32_t code_word = 5000;
    std::vector<std::uint8_t> fssm = {'F', 'S', 'S', 'M'};
    put32(fssm, code_word);
    for (std::uint32_t i = 0; i < 14; ++i) put32(fssm, 0xbf810000u + i);  // filler, s_endpgm-like words
    const auto container = shader_container(fssm);
    EXPECT(leonida::fssm_program(container) == fssm);
    const auto library = std::filesystem::temp_directory_path() / "leonida_compute_test";
    std::filesystem::create_directories(library);
    std::filesystem::copy_file(LEONIDA_TEST_CS_SPV, library / "cs_user_data.spv",
                               std::filesystem::copy_options::overwrite_existing);
    {
        std::ofstream index(library / "index.json", std::ios::binary);
        // An entry for the same program with set-0 bindings comes first, as in one group of the corpus: the PSO
        // must take the binding-free entry, which needs no set 0 (the bound one has no module either).
        index << "{\"cs_user_data_bound\": {\"stage\": \"cs\", \"fssm_sha256\": \"" << leonida::sha256_hex(fssm)
              << "\", \"bindings\": [{\"binding\": 0}], \"push_constant_bytes\": 144, \"local_size\": [64, 1, 1]},\n"
                 " \"cs_user_data\": {\"stage\": \"cs\", \"fssm_sha256\": \"" << leonida::sha256_hex(fssm)
              << "\", \"bindings\": [], \"push_constant_bytes\": 144, \"local_size\": [64, 1, 1], "
                 "\"uses_code_base\": true, \"wave64\": false}}";
    }

    constexpr UINT elements = 64, bytes = elements * 4;
    const std::uint32_t table_values[4] = {1000, 2000, 3000, 4000};
    std::vector<std::uint32_t> first(elements), second(elements);
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

        // Root signature: [0] two constants (s0, s1), [1] a root UAV (s2:s3), [2] a one-CBV table (s4:s5).
        D3D12_DESCRIPTOR_RANGE range{D3D12_DESCRIPTOR_RANGE_TYPE_CBV, 1, 1, 0, 0};
        D3D12_ROOT_PARAMETER params[3] = {};
        params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
        params[0].Constants = {0, 0, 2};
        params[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
        params[1].Descriptor = {0, 0};
        params[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        params[2].DescriptorTable = {1, &range};
        D3D12_VERSIONED_ROOT_SIGNATURE_DESC root_desc{};
        root_desc.Version = D3D_ROOT_SIGNATURE_VERSION_1_0;
        root_desc.Desc_1_0 = {3, params, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_NONE};
        const auto root_blob = serialize(root_desc);
        EXPECT(!root_blob.empty());
        Ref<ID3D12RootSignature> root;
        EXPECT(SUCCEEDED(device->CreateRootSignature(0, root_blob.data(), root_blob.size(), IID_PPV_ARGS(&root.p))));
        Ref<ID3D12RootSignature> junk;
        EXPECT(device->CreateRootSignature(0, "junk", 4, IID_PPV_ARGS(&junk.p)) == E_INVALIDARG && !junk.p);

        // The PSO through slot 62, the Xbox call shape; an unknown program is refused.
        D3D12_COMPUTE_PIPELINE_STATE_DESC pso_desc{};
        pso_desc.pRootSignature = root.p;
        pso_desc.CS = {container.data(), container.size()};
        const IID pso_iid = __uuidof(ID3D12PipelineState);
        Ref<ID3D12PipelineState> pso, unknown;
        EXPECT(SUCCEEDED(reinterpret_cast<XboxComputePipelineFn>(slot(device.p, 62))(
            device.p, &pso_desc, 0, 0, &pso_iid, reinterpret_cast<void**>(&pso.p))));
        auto other = fssm;
        other[8] ^= 1;
        const auto other_container = shader_container(other);
        pso_desc.CS = {other_container.data(), other_container.size()};
        EXPECT(device->CreateComputePipelineState(&pso_desc, IID_PPV_ARGS(&unknown.p)) == E_INVALIDARG && !unknown.p);
        pso_desc.CS = {container.data(), container.size()};

        // The PSO through slot 67, compute pipeline from native code; maps to CreateComputePipelineState.
        struct XboxNativeComputeDesc {
            ID3D12RootSignature* root_signature;
            const void* ptr1;
            std::uint64_t size1;
            const void* ptr2;
            std::uint64_t size2;
            const void* code;
            std::uint64_t code_size;
        } native_desc{};
        native_desc.root_signature = root.p;
        native_desc.code = fssm.data();
        native_desc.code_size = fssm.size();
        Ref<ID3D12PipelineState> pso67, unknown67;
        EXPECT(SUCCEEDED(reinterpret_cast<XboxNativeComputePipelineFn>(slot(device.p, 67))(
            device.p, &native_desc, &pso_iid, reinterpret_cast<void**>(&pso67.p))));
        EXPECT(pso67.p != nullptr);
        native_desc.code = other.data();
        native_desc.code_size = other.size();
        EXPECT(reinterpret_cast<XboxNativeComputePipelineFn>(slot(device.p, 67))(
            device.p, &native_desc, &pso_iid, reinterpret_cast<void**>(&unknown67.p)) == E_INVALIDARG && !unknown67.p);
        void* out_null = reinterpret_cast<void*>(0x1111);
        EXPECT(reinterpret_cast<XboxNativeComputePipelineFn>(slot(device.p, 67))(
            device.p, nullptr, &pso_iid, &out_null) == E_INVALIDARG && !out_null);
        out_null = reinterpret_cast<void*>(0x1111);
        native_desc.code = fssm.data();
        native_desc.code_size = fssm.size();
        EXPECT(reinterpret_cast<XboxNativeComputePipelineFn>(slot(device.p, 67))(
            device.p, &native_desc, nullptr, &out_null) == E_POINTER && !out_null);
        EXPECT(reinterpret_cast<XboxNativeComputePipelineFn>(slot(device.p, 67))(
            device.p, &native_desc, &pso_iid, nullptr) == E_POINTER);

        // A device whose maxPushConstantsSize is the Vulkan minimum (128 bytes) cannot hold the 144-byte block, so
        // its PSOs are refused instead of creating a pipeline layout the spec forbids (#224's fallback flag).
        {
            rt::d3d12::VulkanDeviceInfo limited_info = info;
            limited_info.max_push_constants_size = 128;
            limited_info.push_constants_need_fallback = true;
            Ref<ID3D12Device> limited;
            EXPECT(SUCCEEDED(rt::d3d12::create_vulkan_device(limited_info, IID_PPV_ARGS(&limited.p))));
            EXPECT(rt::d3d12::device_push_constants_need_fallback(limited.p));
            Ref<ID3D12RootSignature> limited_root;
            EXPECT(SUCCEEDED(limited->CreateRootSignature(0, root_blob.data(), root_blob.size(),
                                                        IID_PPV_ARGS(&limited_root.p))));
            D3D12_COMPUTE_PIPELINE_STATE_DESC limited_desc = pso_desc;
            limited_desc.pRootSignature = limited_root.p;
            Ref<ID3D12PipelineState> refused;
            EXPECT(limited->CreateComputePipelineState(&limited_desc, IID_PPV_ARGS(&refused.p)) == E_NOTIMPL && !refused.p);
        }

        // Buffers in the arena, an upload buffer outside it, a readback buffer for the results.
        const D3D12_HEAP_PROPERTIES default_heap{D3D12_HEAP_TYPE_DEFAULT};
        const D3D12_HEAP_PROPERTIES upload_heap{D3D12_HEAP_TYPE_UPLOAD};
        const D3D12_HEAP_PROPERTIES readback_heap{D3D12_HEAP_TYPE_READBACK};
        Ref<ID3D12Resource> out_a, out_b, constants, readback;
        const auto uav = buffer_desc(bytes, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
        EXPECT(SUCCEEDED(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &uav,
                                                         D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr,
                                                         IID_PPV_ARGS(&out_a.p))));
        EXPECT(SUCCEEDED(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &uav,
                                                         D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr,
                                                         IID_PPV_ARGS(&out_b.p))));
        const auto constants_desc = buffer_desc(256, D3D12_RESOURCE_FLAG_NONE);
        EXPECT(SUCCEEDED(device->CreateCommittedResource(&upload_heap, D3D12_HEAP_FLAG_NONE, &constants_desc,
                                                         D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
                                                         IID_PPV_ARGS(&constants.p))));
        const auto readback_desc = buffer_desc(2 * bytes, D3D12_RESOURCE_FLAG_NONE);
        EXPECT(SUCCEEDED(device->CreateCommittedResource(&readback_heap, D3D12_HEAP_FLAG_NONE, &readback_desc,
                                                         D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                                         IID_PPV_ARGS(&readback.p))));
        const D3D12_GPU_VIRTUAL_ADDRESS va_a = out_a->GetGPUVirtualAddress(), va_b = out_b->GetGPUVirtualAddress();
        EXPECT(va_a >= 0x10000 && va_b >= 0x10000 && va_a != va_b && va_a < (1ull << 32) && va_b < (1ull << 32));
        EXPECT(va_a % D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT == 0);
        EXPECT(constants->GetGPUVirtualAddress() != 0);  // device address - heap base: wraps, still valid
        void* mapped = nullptr;
        EXPECT(SUCCEEDED(constants->Map(0, nullptr, &mapped)));
        std::memcpy(mapped, table_values, sizeof(table_values));

        // A shader-visible heap in the arena; its first slot gets the upload buffer's VA, written as raw bytes.
        const D3D12_DESCRIPTOR_HEAP_DESC heap_desc{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 4,
                                                   D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, 0};
        Ref<ID3D12DescriptorHeap> heap;
        EXPECT(SUCCEEDED(device->CreateDescriptorHeap(&heap_desc, IID_PPV_ARGS(&heap.p))));
        const std::uint64_t cpu = reinterpret_cast<XboxHeapHandleFn>(slot(heap.p, 9))(heap.p);
        const std::uint64_t gpu = reinterpret_cast<XboxHeapHandleFn>(slot(heap.p, 10))(heap.p);
        EXPECT(cpu && gpu >= 0x10000 && gpu < (1ull << 32));
        const std::uint64_t target = constants->GetGPUVirtualAddress();
        std::memcpy(reinterpret_cast<void*>(cpu), &target, sizeof(target));

        D3D12_COMMAND_QUEUE_DESC queue_desc{};
        Ref<ID3D12CommandQueue> queue;
        Ref<ID3D12CommandAllocator> allocator;
        Ref<ID3D12GraphicsCommandList> list;
        Ref<ID3D12Fence> fence;
        EXPECT(SUCCEEDED(device->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&queue.p))));
        EXPECT(SUCCEEDED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator.p))));
        EXPECT(SUCCEEDED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.p, nullptr,
                                                   IID_PPV_ARGS(&list.p))));
        EXPECT(SUCCEEDED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence.p))));

        // Misuse is reported by Close: a dispatch without a pipeline, a root view set on a constants parameter.
        list->Dispatch(1, 1, 1);
        EXPECT(list->Close() == E_INVALIDARG);
        EXPECT(SUCCEEDED(list->Reset(allocator.p, pso.p)));
        list->SetComputeRootSignature(root.p);
        list->SetComputeRootUnorderedAccessView(0, va_a);
        EXPECT(list->Close() == E_INVALIDARG);

        // Two dispatches: the second changes s0 and the UAV; each must keep what it was recorded with.
        EXPECT(SUCCEEDED(list->Reset(allocator.p, nullptr)));
        ID3D12DescriptorHeap* heaps[] = {heap.p};
        list->SetDescriptorHeaps(1, heaps);
        // This root signature and PSO are released right after being set, as D3D12 allows: the list keeps them,
        // so the arguments below still have a layout and ExecuteCommandLists still finds the pipeline.
        ID3D12RootSignature* held_root = nullptr;
        {
            Ref<ID3D12RootSignature> own_root;
            EXPECT(SUCCEEDED(device->CreateRootSignature(0, root_blob.data(), root_blob.size(),
                                                         IID_PPV_ARGS(&own_root.p))));
            D3D12_COMPUTE_PIPELINE_STATE_DESC own_desc = pso_desc;
            own_desc.pRootSignature = own_root.p;
            Ref<ID3D12PipelineState> own_pso;
            EXPECT(SUCCEEDED(device->CreateComputePipelineState(&own_desc, IID_PPV_ARGS(&own_pso.p))));
            list->SetComputeRootSignature(own_root.p);
            list->SetPipelineState(own_pso.p);
            held_root = own_root.p;  // still alive: the list holds it
        }
        const std::uint32_t base_and_step[2] = {7, 3};
        list->SetComputeRoot32BitConstants(0, 2, base_and_step, 0);
        list->SetComputeRootUnorderedAccessView(1, va_a);
        list->SetComputeRootDescriptorTable(2, D3D12_GPU_DESCRIPTOR_HANDLE{gpu});
        // Setting the bound root signature again keeps the arguments, as D3D12 does.
        list->SetComputeRootSignature(held_root);
        list->Dispatch(1, 1, 1);
        list->SetComputeRoot32BitConstant(0, 100, 0);
        list->SetComputeRootUnorderedAccessView(1, va_b);
        list->Dispatch(1, 1, 1);
        D3D12_RESOURCE_BARRIER barriers[2] = {
            transition(out_a.p, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE),
            transition(out_b.p, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE)};
        list->ResourceBarrier(2, barriers);
        list->CopyBufferRegion(readback.p, 0, out_a.p, 0, bytes);
        list->CopyBufferRegion(readback.p, bytes, out_b.p, 0, bytes);
        EXPECT(list->Close() == S_OK);
        ID3D12CommandList* lists[] = {list.p};
        queue->ExecuteCommandLists(1, lists);
        EXPECT(rt::d3d12::last_execute_result(queue.p) == leonida::ReplayResult::success);
        EXPECT(SUCCEEDED(queue->Signal(fence.p, 1)));
        EXPECT(SUCCEEDED(fence->SetEventOnCompletion(1, nullptr)));

        void* results = nullptr;
        EXPECT(SUCCEEDED(readback->Map(0, nullptr, &results)));
        std::memcpy(first.data(), results, bytes);
        std::memcpy(second.data(), static_cast<const std::uint8_t*>(results) + bytes, bytes);

        for (auto* r : {&out_a, &out_b, &constants, &readback}) r->reset();
        pso.reset();
        root.reset();
        heap.reset();
        list.reset();
        allocator.reset();
        queue.reset();
        fence.reset();
        EXPECT(rt::d3d12_xs::device_proxy_ref_count(device.p) == 1);
    }
    std::filesystem::remove_all(library);

    unsigned mismatches = 0;
    for (std::uint32_t i = 0; i < elements; ++i) {
        const std::uint32_t a = 7 + 3 * i + table_values[i % 4] + code_word;
        const std::uint32_t b = 100 + 3 * i + table_values[i % 4] + code_word;
        if ((first[i] != a || second[i] != b) && mismatches++ < 4)
            std::fprintf(stderr, "element %u: got %u and %u, want %u and %u\n", i, first[i], second[i], a, b);
    }
    EXPECT(!mismatches);
    if (d.validation_errors()) {
        std::fprintf(stderr, "%u validation errors\n", d.validation_errors());
        return 1;
    }
    std::printf("compute: root constants, a root UAV and a descriptor table reached the shader through the GPU "
                "address arena; %u elements matched in each of two dispatches\n", elements);
    return 0;
}
