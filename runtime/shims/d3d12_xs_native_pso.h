// Xbox device slot 57: native graphics pipelines (design and sources in runtime/src/native_pso.h and HANDOFF.md).
//
// The handler reads the 0x108-byte description, finds each stage's source container through the HASH digest in its
// records (<shaders>/native_hashes.txt, written by tools/native_pso_hashes.py), and creates the pipeline on the
// device behind the proxy from the game's own containers, as a PC title would: CreateGraphicsPipelineState for
// VS/PS/GS, ID3D12Device2::CreatePipelineState for MS/PS. Blob A (blend, raster, depth) is not decoded: blending
// and culling are off and a depth target is written with ALWAYS.
//
// Header-only, so every target that compiles d3d12_xs_device.cpp gets it without a build change.
//
// A stage that cannot be resolved refuses the call with E_NOTIMPL (the title skips a failed pipeline, issue #7496),
// or, with LEONIDA_STUB_PIPELINES set, asks the device for a pipeline whose vertex stage has no translation, which
// #7417's device answers with a no-op stub. LEONIDA_NATIVE_PSO=0 refuses every call.
#pragma once

#include "../src/native_pso.h"

#include <windows.h>

#include <d3d12.h>

#include <atomic>
#include <climits>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <cstdio>
#include <exception>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

namespace rt::d3d12_xs {

// Where the translator's output is (<work>/shaders/spirv); asked once, on the first slot-57 call. The Vulkan device
// sets it; without a provider, LEONIDA_SHADERS is read.
using ShaderDirectoryProvider = std::filesystem::path (*)();

struct NativePsoStats {
    std::uint64_t calls, created, stubbed, refused;
};

namespace native_pso_detail {

namespace np = leonida::native_pso;

// No function-local statics: their guards fault on the first shim call under Wine with the static CRT (#7175).
inline std::atomic<ShaderDirectoryProvider> g_provider{nullptr};
inline SRWLOCK g_lock = SRWLOCK_INIT;
inline bool g_loaded = false;  // under g_lock
inline std::filesystem::path* g_dir = nullptr;
inline np::HashMap* g_map = nullptr;  // null when there is no usable map
inline std::atomic<std::uint64_t> g_calls{0}, g_created{0}, g_stubbed{0}, g_refused{0};
inline std::atomic<bool> g_first_logged{false}, g_odd_logged{false};

inline bool env_flag(const wchar_t* name, bool fallback) {
    wchar_t value[8] = {};
    const DWORD n = GetEnvironmentVariableW(name, value, 8);
    if (n == 0 || n >= 8) return fallback;
    return value[0] != L'0';
}

inline std::filesystem::path env_shader_directory() {
    const DWORD size = GetEnvironmentVariableW(L"LEONIDA_SHADERS", nullptr, 0);
    if (!size) return {};
    std::wstring value(size, L'\0');
    const DWORD length = GetEnvironmentVariableW(L"LEONIDA_SHADERS", value.data(), size);
    if (!length || length >= size) return {};
    value.resize(length);
    return std::filesystem::path(value);
}

// The map and its folder, loaded on the first call. Read-only afterwards, so callers use them without the lock.
inline const np::HashMap* hash_map(const std::filesystem::path** dir) {
    AcquireSRWLockExclusive(&g_lock);
    if (!g_loaded) {
        g_loaded = true;
        try {
            const ShaderDirectoryProvider provider = g_provider.load();
            auto path = provider ? provider() : env_shader_directory();
            const auto file = path / L"native_hashes.txt";
            std::ifstream in(file, std::ios::binary);
            const std::string text{std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
            auto map = std::make_unique<np::HashMap>();
            std::string error;
            if (path.empty()) {
                std::fprintf(stderr, "[rt] d3d12_xs: slot 57: no translated shader folder (LEONIDA_SHADERS)\n");
            } else if (!in && text.empty()) {
                std::fprintf(stderr, "[rt] d3d12_xs: slot 57: no native_hashes.txt in %s; rerun make_game.py's "
                             "shaders stage\n", reinterpret_cast<const char*>(path.u8string().c_str()));
            } else if (!map->parse(text, &error)) {
                std::fprintf(stderr, "[rt] d3d12_xs: slot 57: %s\n", error.c_str());
            } else {
                std::fprintf(stderr, "[rt] d3d12_xs: slot 57: %zu native shader digests\n", map->size());
                g_map = map.release();
                g_dir = new std::filesystem::path(std::move(path));
            }
        } catch (const std::exception& e) {
            std::fprintf(stderr, "[rt] d3d12_xs: slot 57: native_hashes.txt not loaded: %s\n", e.what());
        }
    }
    ReleaseSRWLockExclusive(&g_lock);
    *dir = g_dir;
    return g_map;
}

inline bool read_container(const std::filesystem::path& dir, const std::string& relative,
                           std::vector<std::uint8_t>& out) {
    try {
        const auto path = dir / std::filesystem::path(std::u8string(relative.begin(), relative.end()));
        std::ifstream in(path, std::ios::binary | std::ios::ate);
        if (!in) return false;
        const std::streamoff size = in.tellg();
        if (size <= 0 || size > (64 << 20)) return false;
        out.resize(static_cast<std::size_t>(size));
        in.seekg(0);
        return static_cast<bool>(in.read(reinterpret_cast<char*>(out.data()), size));
    } catch (const std::exception&) {
        return false;
    }
}

inline bool log_this(std::uint64_t n) { return (n & (n - 1)) == 0; }

// Blend and culling off; a depth target is written with ALWAYS (blob A is not decoded).
inline void fill_fixed_state(const np::FixedState& s, D3D12_GRAPHICS_PIPELINE_STATE_DESC& pc) {
    for (auto& rt : pc.BlendState.RenderTarget) rt.RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;
    pc.SampleMask = s.sample_mask ? s.sample_mask : UINT_MAX;
    pc.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
    pc.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
    pc.RasterizerState.DepthClipEnable = TRUE;
    pc.DSVFormat = static_cast<DXGI_FORMAT>(s.dsv_format);
    pc.DepthStencilState.DepthEnable = pc.DSVFormat != DXGI_FORMAT_UNKNOWN;
    pc.DepthStencilState.DepthWriteMask = D3D12_DEPTH_WRITE_MASK_ALL;
    pc.DepthStencilState.DepthFunc = D3D12_COMPARISON_FUNC_ALWAYS;
    pc.PrimitiveTopologyType = s.topology_type ? static_cast<D3D12_PRIMITIVE_TOPOLOGY_TYPE>(s.topology_type)
                                               : D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
    pc.NumRenderTargets = s.rt_count;
    for (unsigned i = 0; i < s.rt_count; ++i) pc.RTVFormats[i] = static_cast<DXGI_FORMAT>(s.rtv_formats[i]);
    pc.SampleDesc.Count = s.sample_count;
}

template <D3D12_PIPELINE_STATE_SUBOBJECT_TYPE T, typename P>
struct alignas(void*) Subobject {
    D3D12_PIPELINE_STATE_SUBOBJECT_TYPE type = T;
    P value{};
};

struct MeshStream {
    Subobject<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_ROOT_SIGNATURE, ID3D12RootSignature*> root;
    Subobject<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_MS, D3D12_SHADER_BYTECODE> ms;
    Subobject<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PS, D3D12_SHADER_BYTECODE> ps;
    Subobject<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_BLEND, D3D12_BLEND_DESC> blend;
    Subobject<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_SAMPLE_MASK, UINT> sample_mask;
    Subobject<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RASTERIZER, D3D12_RASTERIZER_DESC> raster;
    Subobject<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL, D3D12_DEPTH_STENCIL_DESC> depth;
    Subobject<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RENDER_TARGET_FORMATS, D3D12_RT_FORMAT_ARRAY> rtvs;
    Subobject<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL_FORMAT, DXGI_FORMAT> dsv;
    Subobject<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_SAMPLE_DESC, DXGI_SAMPLE_DESC> sample;
};

inline HRESULT create_mesh(ID3D12Device* real, const D3D12_GRAPHICS_PIPELINE_STATE_DESC& pc,
                           D3D12_SHADER_BYTECODE ms, const IID& riid, void** out) {
    ID3D12Device2* device2 = nullptr;
    if (FAILED(real->QueryInterface(__uuidof(ID3D12Device2), reinterpret_cast<void**>(&device2))) || !device2)
        return E_NOTIMPL;
    MeshStream s;
    s.root.value = pc.pRootSignature;
    s.ms.value = ms;
    s.ps.value = pc.PS;
    s.blend.value = pc.BlendState;
    s.sample_mask.value = pc.SampleMask;
    s.raster.value = pc.RasterizerState;
    s.depth.value = pc.DepthStencilState;
    s.rtvs.value.NumRenderTargets = pc.NumRenderTargets;
    for (unsigned i = 0; i < 8; ++i) s.rtvs.value.RTFormats[i] = pc.RTVFormats[i];
    s.dsv.value = pc.DSVFormat;
    s.sample.value = pc.SampleDesc;
    const D3D12_PIPELINE_STATE_STREAM_DESC stream{sizeof(s), &s};
    const HRESULT hr = device2->CreatePipelineState(&stream, riid, out);
    device2->Release();
    return hr;
}

inline const char* status_name(np::PlanStatus s) {
    switch (s) {
        case np::PlanStatus::ok: return "ok";
        case np::PlanStatus::malformed: return "malformed description";
        case np::PlanStatus::missing: return "no translated container for the stage's digest";
        case np::PlanStatus::ambiguous: return "the stage's records name two containers";
        case np::PlanStatus::unsupported: return "unsupported stage set";
    }
    return "?";
}

// A bare FSSM blob that no index names: the device's own "no translated shader" path answers it.
inline constexpr std::uint8_t kStubProgram[16] = {'F', 'S', 'S', 'M'};

inline HRESULT unresolved(ID3D12Device* real, const np::GraphicsDesc& desc, const np::FixedState* state,
                          const IID& riid, void** out, const char* why, std::size_t block) {
    const bool stub = env_flag(L"LEONIDA_STUB_PIPELINES", false);
    const std::uint64_t n = (stub ? g_stubbed : g_refused).fetch_add(1) + 1;
    if (log_this(n))
        std::fprintf(stderr, "[rt] d3d12_xs: slot 57: %s (block %zu); %s (%llu so far)\n", why, block,
                     stub ? "asking the device for a stub" : "refused with E_NOTIMPL",
                     static_cast<unsigned long long>(n));
    if (!stub) return E_NOTIMPL;
    D3D12_GRAPHICS_PIPELINE_STATE_DESC pc{};
    pc.pRootSignature = static_cast<ID3D12RootSignature*>(const_cast<void*>(desc.root_signature));
    fill_fixed_state(state ? *state : np::FixedState{}, pc);
    pc.VS = {kStubProgram, sizeof(kStubProgram)};
    return real->CreateGraphicsPipelineState(&pc, riid, out);
}

}  // namespace native_pso_detail

inline void set_native_shader_directory_provider(ShaderDirectoryProvider provider) {
    using namespace native_pso_detail;
    g_provider.store(provider);
}

inline NativePsoStats native_pso_stats() {
    using namespace native_pso_detail;
    return {g_calls.load(), g_created.load(), g_stubbed.load(), g_refused.load()};
}

// Test helper: forgets the loaded map and the counters, so the next call reads the provider again.
inline void reset_native_pso_state() {
    using namespace native_pso_detail;
    AcquireSRWLockExclusive(&g_lock);
    delete g_map;
    delete g_dir;
    g_map = nullptr;
    g_dir = nullptr;
    g_loaded = false;
    ReleaseSRWLockExclusive(&g_lock);
    g_calls = g_created = g_stubbed = g_refused = 0;
}

// slot 57 (desc, riid, out) on the device behind the proxy.
inline HRESULT create_native_graphics_pipeline(ID3D12Device* real, const void* desc_words, const IID* riid,
                                               void** out) {
    using namespace native_pso_detail;
    const std::uint64_t call = g_calls.fetch_add(1) + 1;
    if (!out) return E_POINTER;
    *out = nullptr;
    if (!riid) return E_POINTER;
    if (!desc_words) return E_INVALIDARG;
    const auto& desc = *static_cast<const np::GraphicsDesc*>(desc_words);
    if (!g_first_logged.exchange(true))
        std::fprintf(stderr, "[rt] d3d12_xs: slot 57 first call: desc %p, blob A %llu B, blob B %llu B, riid %p\n",
                     desc_words, static_cast<unsigned long long>(desc.state_a.size),
                     static_cast<unsigned long long>(desc.state_b.size), static_cast<const void*>(riid));
    if (!env_flag(L"LEONIDA_NATIVE_PSO", true)) {
        g_refused.fetch_add(1);
        return E_NOTIMPL;
    }

    const std::filesystem::path* dir = nullptr;
    const np::HashMap* map = hash_map(&dir);
    const np::HashMap empty;
    const np::Plan plan = np::plan(desc, map ? *map : empty);
    if (plan.status == np::PlanStatus::malformed) {
        const std::uint64_t n = g_refused.fetch_add(1) + 1;
        if (log_this(n))
            std::fprintf(stderr, "[rt] d3d12_xs: slot 57 call %llu: malformed description (blob B %llu B); "
                         "E_INVALIDARG (%llu so far)\n", static_cast<unsigned long long>(call),
                         static_cast<unsigned long long>(desc.state_b.size), static_cast<unsigned long long>(n));
        return E_INVALIDARG;
    }
    if (plan.status != np::PlanStatus::ok || !dir)
        return unresolved(real, desc, &plan.state, *riid, out, map ? status_name(plan.status) : "no native_hashes.txt",
                          plan.failed_block);

    std::vector<std::uint8_t> code[np::kStageBlocks];
    D3D12_GRAPHICS_PIPELINE_STATE_DESC pc{};
    D3D12_SHADER_BYTECODE ms{};
    for (const np::PlannedStage& s : plan.stages) {
        if (!read_container(*dir, s.shader->path, code[s.block]))
            return unresolved(real, desc, &plan.state, *riid, out, "container not readable", s.block);
        const D3D12_SHADER_BYTECODE bc{code[s.block].data(), code[s.block].size()};
        const std::string& stage = s.shader->stage;
        (stage == "vs" ? pc.VS : stage == "ps" ? pc.PS : stage == "gs" ? pc.GS : ms) = bc;
    }
    pc.pRootSignature = static_cast<ID3D12RootSignature*>(const_cast<void*>(desc.root_signature));
    fill_fixed_state(plan.state, pc);
    std::vector<D3D12_INPUT_ELEMENT_DESC> elements;
    for (const np::InputElement& e : plan.state.elements)
        elements.push_back({e.name.c_str(), e.semantic_index, static_cast<DXGI_FORMAT>(e.format), e.input_slot,
                            e.aligned_byte_offset, static_cast<D3D12_INPUT_CLASSIFICATION>(e.classification),
                            e.step_rate});
    pc.InputLayout = {elements.data(), static_cast<UINT>(elements.size())};

    const HRESULT hr = plan.mesh ? create_mesh(real, pc, ms, *riid, out)
                                 : real->CreateGraphicsPipelineState(&pc, *riid, out);
    if (FAILED(hr)) {
        const std::uint64_t n = g_refused.fetch_add(1) + 1;
        if (log_this(n))
            std::fprintf(stderr, "[rt] d3d12_xs: slot 57 call %llu: the device refused 0x%08lx (%s) (%llu so far)\n",
                         static_cast<unsigned long long>(call), static_cast<unsigned long>(hr),
                         plan.stages.front().shader->path.c_str(), static_cast<unsigned long long>(n));
        return hr;
    }
    // The title keeps the pipeline shifted right by one (issue #7496): an odd pointer would lose its low bit.
    if (*out && (reinterpret_cast<std::uintptr_t>(*out) & 1)) {
        if (!g_odd_logged.exchange(true)) std::fprintf(stderr, "[rt] d3d12_xs: slot 57: odd pipeline pointer\n");
        static_cast<IUnknown*>(*out)->Release();
        *out = nullptr;
        g_refused.fetch_add(1);
        return E_FAIL;
    }
    g_created.fetch_add(1);
    return hr;
}

}  // namespace rt::d3d12_xs
