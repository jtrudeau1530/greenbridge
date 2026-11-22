// Subgroup size control in PSOs, end to end through the Xbox-ABI proxy over the Vulkan device (src/subgroup_size.h):
// three translated compute shaders from runtime/test/shaders, each dispatched through a D3D12 PSO found by the
// SHA-256 of its FSSM program, writing eight words through the V# in s[0:3] (four root constants), read back:
//   w64 native    cs_w64x_readlane translated natively: v_readlane across both halves of a wave64. Exact only in a
//                 64-wide subgroup, so the PSO must ask for 64 (and full subgroups) on a device that can give it.
//                 On a device that cannot (NVIDIA: 32 only) the PSO reports it, and the result is not judged.
//   w64 emulated  the same shader translated with --wave64-emulate: exact on any size, the PSO asks for nothing.
//   w32 native    cs_wqm_w32, a wave32 shader with whole-quad mode on lane masks; the PSO asks for 32 where the
//                 device's default or range would otherwise change it.
// Each case also prints a control run with the layer's size request switched off, to show what the request changes
// on this GPU; the control is not judged. No game data. Set LEONIDA_VK_VALIDATION=1 to run under
// VK_LAYER_KHRONOS_validation; validation errors fail the test.
#include "vk_test_device.h"
#include "../shims/d3d12_com.h"
#include "../src/shader_library.h"
#include "../shims/d3d12_xs_device.h"

#include <array>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

#ifndef LEONIDA_TEST_SUBGROUP_SHADERS
#error "LEONIDA_TEST_SUBGROUP_SHADERS must name the folder with the translated test shaders"
#endif

#define EXPECT(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); return 1; } \
} while (0)

namespace {

using XboxComputePipelineFn = HRESULT(WINAPI*)(void* device, const D3D12_COMPUTE_PIPELINE_STATE_DESC*, std::uint64_t,
                                               std::uint64_t, const IID*, void**);
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

// runtime/test/shaders/compute_fixture.py's header for a compute shader without "; group:" or "; rsrc2:" lines:
// 0x148 bytes, COMPUTE_PGM_RSRC2 0x8088 (four user SGPRs, TGID_X, 512 bytes of LDS) at 0x138 and a 64x1x1 group.
std::vector<std::uint8_t> fixture_fssm() {
    std::vector<std::uint8_t> fssm(0x138, 0);
    for (const std::uint32_t v : {0x8088u, 64u, 1u, 1u}) put32(fssm, v);
    return fssm;
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

using Words = std::array<std::uint32_t, 8>;

struct Shader {
    const char* label;
    const char* folder;  // under LEONIDA_TEST_SUBGROUP_SHADERS; its spirv/ holds index.json and the module
    Words expected;
};

constexpr Shader w64_native{"w64 native (cs_w64x_readlane)", "w64_native", {121, 121, 16, 97, 16, 100, 100, 121}};
constexpr Shader w64_emulated{"w64 emulated (cs_w64x_readlane, --wave64-emulate)", "w64_emulated",
                              {121, 121, 16, 97, 16, 100, 100, 121}};
constexpr Shader w32_native{"w32 native (cs_wqm_w32)", "w32_native", {12, 1, 4, 0, 8, 15728880, 4, 32}};

struct Result {
    int failure = 0;      // a line number, 0 when the case ran
    Words words{};
    std::vector<leonida::SubgroupRequest> requests;
};

// One dispatch of the shader's PSO on a device created with `caps` as what the layer knows of the GPU.
Result run(vktest::Device& d, const Shader& shader, const leonida::SubgroupSizeCaps& caps) {
    Result result;
    const auto fail = [&](int line) {
        result.failure = line;
        return result;
    };
#define CASE_EXPECT(condition) do { if (!(condition)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); return fail(__LINE__); } } while (0)

    const std::filesystem::path library = std::filesystem::path(LEONIDA_TEST_SUBGROUP_SHADERS) / shader.folder / "spirv";
    const auto fssm = fixture_fssm();
    const auto container = shader_container(fssm);
    CASE_EXPECT(leonida::fssm_program(container) == fssm);
    {
        leonida::ShaderLibrary index;
        CASE_EXPECT(index.load(library.string()));
        CASE_EXPECT(index.find("cs", leonida::sha256_hex(fssm)));  // the translator's header is the fixture's
    }

    rt::d3d12::VulkanDeviceInfo info{};
    info.get_instance_proc = d.instance_proc;
    info.instance = d.instance;
    info.physical = d.physical;
    info.device = d.device;
    info.queue = d.queue;
    info.queue_family = d.family;
    info.buffer_device_address = true;
    info.shader_dir = library.string();
    info.subgroup = caps;
    ID3D12Device* inner = nullptr;
    CASE_EXPECT(SUCCEEDED(rt::d3d12::create_vulkan_device(info, IID_PPV_ARGS(&inner))));
    Ref<ID3D12Device> device;
    CASE_EXPECT(SUCCEEDED(rt::d3d12_xs::create_device_proxy(inner, __uuidof(ID3D12Device),
                                                            reinterpret_cast<void**>(&device.p))));
    rt::d3d12::set_device_outer(inner, device.p);

    // Root signature: [0] four constants (s0..s3), the V# the shaders store through.
    D3D12_ROOT_PARAMETER param{};
    param.ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    param.Constants = {0, 0, 4};
    D3D12_VERSIONED_ROOT_SIGNATURE_DESC root_desc{};
    root_desc.Version = D3D_ROOT_SIGNATURE_VERSION_1_0;
    root_desc.Desc_1_0 = {1, &param, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_NONE};
    const auto root_blob = serialize(root_desc);
    CASE_EXPECT(!root_blob.empty());
    Ref<ID3D12RootSignature> root;
    CASE_EXPECT(SUCCEEDED(device->CreateRootSignature(0, root_blob.data(), root_blob.size(), IID_PPV_ARGS(&root.p))));

    D3D12_COMPUTE_PIPELINE_STATE_DESC pso_desc{};
    pso_desc.pRootSignature = root.p;
    pso_desc.CS = {container.data(), container.size()};
    const IID pso_iid = __uuidof(ID3D12PipelineState);
    Ref<ID3D12PipelineState> pso;
    CASE_EXPECT(SUCCEEDED(reinterpret_cast<XboxComputePipelineFn>(slot(device.p, 62))(
        device.p, &pso_desc, 0, 0, &pso_iid, reinterpret_cast<void**>(&pso.p))));
    const auto* requests = rt::d3d12::pipeline_subgroup_requests(pso.p);
    CASE_EXPECT(requests && requests->size() == 1);
    result.requests = *requests;

    constexpr UINT bytes = 256;
    const D3D12_HEAP_PROPERTIES default_heap{D3D12_HEAP_TYPE_DEFAULT};
    const D3D12_HEAP_PROPERTIES readback_heap{D3D12_HEAP_TYPE_READBACK};
    Ref<ID3D12Resource> out, readback;
    const auto uav = buffer_desc(bytes, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
    CASE_EXPECT(SUCCEEDED(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &uav,
                                                          D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr,
                                                          IID_PPV_ARGS(&out.p))));
    const auto readback_desc = buffer_desc(bytes, D3D12_RESOURCE_FLAG_NONE);
    CASE_EXPECT(SUCCEEDED(device->CreateCommittedResource(&readback_heap, D3D12_HEAP_FLAG_NONE, &readback_desc,
                                                          D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                                          IID_PPV_ARGS(&readback.p))));
    const D3D12_GPU_VIRTUAL_ADDRESS va = out->GetGPUVirtualAddress();
    CASE_EXPECT(va >= 0x10000 && va < (1ull << 32));

    D3D12_COMMAND_QUEUE_DESC queue_desc{};
    Ref<ID3D12CommandQueue> queue;
    Ref<ID3D12CommandAllocator> allocator;
    Ref<ID3D12GraphicsCommandList> list;
    Ref<ID3D12Fence> fence;
    CASE_EXPECT(SUCCEEDED(device->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&queue.p))));
    CASE_EXPECT(SUCCEEDED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator.p))));
    CASE_EXPECT(SUCCEEDED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.p, pso.p,
                                                    IID_PPV_ARGS(&list.p))));
    CASE_EXPECT(SUCCEEDED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence.p))));

    // The V# the shaders read from s[0:3]: base = the buffer's console VA (the shader adds the heap base), stride 0,
    // num_records unlimited, no format bits.
    list->SetComputeRootSignature(root.p);
    const std::uint32_t vsharp[4] = {std::uint32_t(va), std::uint32_t(va >> 32) & 0xffff, 0xffffffffu, 0};
    list->SetComputeRoot32BitConstants(0, 4, vsharp, 0);
    list->Dispatch(1, 1, 1);
    D3D12_RESOURCE_BARRIER barrier{};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Transition = {out.p, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                          D3D12_RESOURCE_STATE_COPY_SOURCE};
    list->ResourceBarrier(1, &barrier);
    list->CopyBufferRegion(readback.p, 0, out.p, 0, bytes);
    CASE_EXPECT(list->Close() == S_OK);
    ID3D12CommandList* lists[] = {list.p};
    queue->ExecuteCommandLists(1, lists);
    CASE_EXPECT(rt::d3d12::last_execute_result(queue.p) == leonida::ReplayResult::success);
    CASE_EXPECT(SUCCEEDED(queue->Signal(fence.p, 1)));
    CASE_EXPECT(SUCCEEDED(fence->SetEventOnCompletion(1, nullptr)));
    void* mapped = nullptr;
    CASE_EXPECT(SUCCEEDED(readback->Map(0, nullptr, &mapped)));
    std::memcpy(result.words.data(), mapped, sizeof(result.words));

    for (auto* r : {&out, &readback}) r->reset();
    pso.reset();
    root.reset();
    list.reset();
    allocator.reset();
    queue.reset();
    fence.reset();
    CASE_EXPECT(rt::d3d12_xs::device_proxy_ref_count(device.p) == 1);
#undef CASE_EXPECT
    return result;
}

std::string words_text(const Words& w) {
    std::string s;
    for (const std::uint32_t v : w) s += (s.empty() ? "" : " ") + std::to_string(v);
    return s;
}

// What the decision function says about the shader's own index entry on this device.
leonida::SubgroupRequest expected_request(const Shader& shader, const leonida::SubgroupSizeCaps& caps) {
    leonida::ShaderLibrary index;
    const auto dir = std::filesystem::path(LEONIDA_TEST_SUBGROUP_SHADERS) / shader.folder / "spirv";
    if (!index.load(dir.string())) return {};
    const auto* entry = index.find("cs", leonida::sha256_hex(fixture_fssm()));
    return entry ? leonida::decide_subgroup_size(*entry, caps) : leonida::SubgroupRequest{};
}

}  // namespace

int main() {
    vktest::Device d;
    EXPECT(d.create());
    const leonida::SubgroupSizeCaps& caps = d.subgroup;
    leonida::SubgroupSizeCaps off = caps;  // the control: the layer knows nothing of the device's sizes
    off.min_size = off.max_size = off.default_size = 0;
    off.subgroup_size_control = off.compute_full_subgroups = false;
    off.required_stages = 0;

    unsigned judged = 0;
    for (const Shader* shader : {&w64_native, &w64_emulated, &w32_native}) {
        const leonida::SubgroupRequest want = expected_request(*shader, caps);
        const Result result = run(d, *shader, caps);
        EXPECT(!result.failure);
        EXPECT(result.requests.size() == 1);
        const leonida::SubgroupRequest& got = result.requests[0];
        // The PSO asked for what the decision function says.
        EXPECT(got.outcome == want.outcome && got.size == want.size && got.flags == want.flags);
        EXPECT(got.stage == VK_SHADER_STAGE_COMPUTE_BIT);

        const bool exact = result.words == shader->expected;
        std::printf("%s: outcome %d, required size %u, full subgroups %s; result %s (%s)\n", shader->label,
                    int(got.outcome), got.size, got.flags ? "yes" : "no", exact ? "exact" : "DIFFERS",
                    words_text(result.words).c_str());
        if (shader == &w64_native && got.outcome == leonida::SubgroupOutcome::unavailable) {
            // This GPU cannot give a wave64 compute shader 64 lanes. The layer said so (a log line, above); the
            // build for this GPU should have used --wave64-emulate, which the next case checks.
            std::printf("  this device has no 64-wide compute subgroup (sizes %u..%u): native wave64 not judged\n",
                        caps.min_size, caps.max_size);
        } else {
            if (!exact) std::fprintf(stderr, "%s: wanted %s\n", shader->label, words_text(shader->expected).c_str());
            EXPECT(exact);
            ++judged;
        }
        if (shader == &w64_native && got.outcome != leonida::SubgroupOutcome::unavailable) {
            EXPECT(got.outcome == leonida::SubgroupOutcome::required || got.outcome == leonida::SubgroupOutcome::device_fixed);
            if (got.outcome == leonida::SubgroupOutcome::required) {
                EXPECT(got.size == 64);
                EXPECT(!caps.compute_full_subgroups ||
                       got.flags == VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT);
            }
        }
        if (shader == &w64_emulated) EXPECT(got.outcome == leonida::SubgroupOutcome::emulated && !got.size && !got.flags);

        // Control, not judged: the same PSO with no size asked for.
        if (shader != &w64_emulated) {
            const Result control = run(d, *shader, off);
            EXPECT(!control.failure);
            std::printf("  control, nothing requested: %s (%s)\n", control.words == shader->expected ? "exact" : "differs",
                        words_text(control.words).c_str());
        }
    }
    if (d.validation_errors()) {
        std::fprintf(stderr, "%u validation errors\n", d.validation_errors());
        return 1;
    }
    std::printf("subgroup size: %u of 3 shaders judged exact through the layer's PSOs (sizes %u..%u, default %u)\n",
                judged, caps.min_size, caps.max_size, caps.default_size);
    return 0;
}
