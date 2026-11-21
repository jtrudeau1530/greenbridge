// Portable D3D12 root-signature model + packer to user SGPRs (s0..s31).
//
// No d3d12.h, no windows.h: this compiles on Linux CI and in the Windows
// runtime alike. It models exactly what docs/d3d12_vulkan.md section 3.5
// needs: root parameters in order, static samplers (no dwords), the 64-dword
// D3D12 cost limit, and the 32-dword console user-SGPR budget. Anything that
// does not fit is refused loudly, never guessed (see pack()).
//
// Numbering follows D3D12 docs: descriptor tables cost 1 dword of the 64-dword
// limit, root views (CBV/SRV/UAV) cost 2, 32-bit constants cost their count.
#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace rt::rootsig {

// D3D12_ROOT_SIGNATURE_VERSION subset the layer handles.
enum class Version : uint32_t { V1_0 = 1, V1_1 = 2 };

// D3D12_ROOT_PARAMETER_TYPE subset (order matches D3D12 docs).
enum class ParamType : uint32_t {
    Constants = 0,  // 32BIT_CONSTANTS: NumValues dwords
    Descriptor = 1,  // CBV/SRV/UAV root descriptor: 2 dwords (64-bit VA)
    Table = 2,  // DESCRIPTOR_TABLE: 2 dwords (table GPU VA)
};

// D3D12_SHADER_VISIBILITY (D3D12 order: ALL 0, VERTEX 1, HULL 2, DOMAIN 3,
// GEOMETRY 4, PIXEL 5).
enum class Visibility : uint32_t {
    All = 0,
    Vertex = 1,
    Geometry = 4,
    Pixel = 5,
    Compute = 6,  // no D3D12 value; Xbox extension bucket, opaque for packing
    Mesh = 7,  // Xbox extension bucket; treated as opaque for packing
};

// One descriptor range inside a table (SRV/CBV/UAV/sampler + count).
struct DescriptorRange {
    uint32_t range_type = 0;  // 0 SRV, 1 UAV, 2 CBV, 3 sampler (D3D12 order)
    uint32_t num_descriptors = 0;
};

// Static sampler: no user-SGPR dwords; registered in the hash table instead.
struct StaticSampler {
    uint32_t filter = 0;
    uint32_t address_u = 0;
    uint32_t address_v = 0;
    uint32_t address_w = 0;
    float mip_lod_bias = 0.0f;
    uint32_t max_anisotropy = 0;
    uint32_t comparison_func = 0;
    uint32_t border_color = 0;  // 0 transparent-black .. 4 opaque-white
    float min_lod = 0.0f;
    float max_lod = 0.0f;
    uint32_t shader_register = 0;
    uint32_t register_space = 0;
    Visibility visibility = Visibility::All;
};

struct RootParameter {
    ParamType type = ParamType::Constants;
    Visibility visibility = Visibility::All;
    // Constants:
    uint32_t num_values = 0;  // shader register + space describe the slot
    uint32_t shader_register = 0;
    uint32_t register_space = 0;
    // Root descriptor:
    uint32_t descriptor_register = 0;
    uint32_t descriptor_space = 0;
    // Descriptor table:
    std::vector<DescriptorRange> ranges;
};

struct RootSignature {
    Version version = Version::V1_1;
    std::vector<RootParameter> params;  // at most 64 root parameters (D3D12)
    std::vector<StaticSampler> static_samplers;
    uint32_t flags = 0;  // low 16 bits: PC meaning; high byte: Xbox type bits
};

// Where one packed dword came from (provenance for diagnostics).
struct DwordProvenance {
    uint32_t param_index = 0;  // index into RootSignature::params
    uint32_t sub_index = 0;  // constant index, or 0 = VA lo, 1 = VA hi
};

struct PackResult {
    bool ok = false;
    std::string error;  // set when !ok; names the refusing parameter
    std::array<uint32_t, 32> dwords{};  // valid dwords [0, dword_count)
    uint32_t dword_count = 0;
    std::vector<DwordProvenance> provenance;  // same length as dword_count
    uint32_t d3d12_cost = 0;  // cost against the 64-dword D3D12 limit
};

constexpr uint32_t kMaxUserSgprs = 32;
constexpr uint32_t kMaxD3d12Cost = 64;

// Packs root parameters in order. Static samplers consume no dwords.
// Fails (ok == false) when the signature needs more than 32 user SGPRs
// (spill layout unknown: refuse, do not guess) or exceeds the 64-dword
// D3D12 cost limit, or holds more than 64 root parameters.
[[nodiscard]] PackResult pack(const RootSignature& sig);

// D3D12 cost of one parameter against the 64-dword limit.
[[nodiscard]] constexpr uint32_t param_cost(ParamType type, uint32_t num_values) noexcept {
    switch (type) {
        case ParamType::Constants:
            return num_values;
        case ParamType::Descriptor:
            return 2;
        case ParamType::Table:
            return 1;
    }
    return 0;
}

// Canonical binary codec for caching root signatures (magic "LRS1",
// little-endian). This is Leonida's own layout, NOT the Microsoft RTS0
// format: it round-trips pack() inputs byte-exactly and is fully tested on
// every CI leg. Never write game blobs in this format and claim otherwise.
[[nodiscard]] std::vector<uint8_t> encode_canonical(const RootSignature& sig);

struct DecodeResult {
    bool ok = false;
    std::string error;
    RootSignature sig;
};

[[nodiscard]] DecodeResult decode_canonical(const uint8_t* data, size_t size);

// RTS0 chunk parser (HYPOTHESIS until the Windows oracle test passes).
//
// The DXBC "RTS0" chunk is the console serializer's output for one root
// signature. Its exact byte layout is undocumented, so this decodes it as
// the straight-line flattening of the public descriptor structs (version,
// counts, then per-parameter fields in d3d12.h order). Every deviation the
// oracle test finds is a hard failure to investigate, not a quirk to absorb.
// Returns ok == false with a reason for unknown versions or truncation.
[[nodiscard]] DecodeResult parse_rts0(const uint8_t* data, size_t size);

}  // namespace rt::rootsig
