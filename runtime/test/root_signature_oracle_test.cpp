// Windows oracle for the RTS0 parser hypothesis (needs a real PC d3d12.lib).
//
// Builds synthetic versioned root-signature descriptors by hand (no game
// data), serializes each with Microsoft's D3D12SerializeVersionedRootSignature,
// and requires rt::rootsig::parse_rts0 to recover the same model. If the
// hypothesis layout in root_signature.cpp ever disagrees with the real
// serializer, this fails loudly here instead of corrupting draws silently.
// Versions 1.0, 1.1 and 1.2 are each checked with static samplers (1.2 only where the host serializes it).
// NOT portable: runs on Windows CI only, like root_sig_test.cpp.
#include <windows.h>
#include <d3d12.h>

#include <cstdint>
#include <cstdio>
#include <vector>

#include "root_signature.h"

namespace {

int g_failures = 0;

void check(bool condition, const char* what) {
    std::printf("%s %s\n", condition ? "PASS" : "FAIL", what);
    if (!condition) ++g_failures;
}

struct Blobs {
    ID3DBlob* blob = nullptr;
    ID3DBlob* error = nullptr;
    ~Blobs() {
        if (blob) blob->Release();
        if (error) error->Release();
    }
};

D3D12_VERSIONED_ROOT_SIGNATURE_DESC make_desc(
    const D3D12_ROOT_PARAMETER1* params, uint32_t nparams) {
    D3D12_VERSIONED_ROOT_SIGNATURE_DESC desc{};
    desc.Version = D3D_ROOT_SIGNATURE_VERSION_1_1;
    desc.Desc_1_1.NumParameters = nparams;
    desc.Desc_1_1.pParameters = params;
    desc.Desc_1_1.NumStaticSamplers = 0;
    desc.Desc_1_1.pStaticSamplers = nullptr;
    desc.Desc_1_1.Flags = D3D12_ROOT_SIGNATURE_FLAG_NONE;
    return desc;
}

bool parse_blob(ID3DBlob* blob, rt::rootsig::RootSignature& model) {
    const auto* bytes = static_cast<const uint8_t*>(blob->GetBufferPointer());
    const size_t size = blob->GetBufferSize();
    // Diagnostics first: if the layout hypothesis is wrong, the CI log shows
    // the real bytes instead of just a crash or a bare failure.
    std::printf("oracle blob: %zu bytes: ", size);
    for (size_t i = 0; i < 32 && i < size; ++i) {
        std::printf("%02x", bytes[i]);
    }
    std::printf("\n");
    if (GetEnvironmentVariableA("LEONIDA_ORACLE_DUMP", nullptr, 0)) {  // the whole blob as u32 words, for layouts
        for (size_t i = 0; i + 4 <= size; i += 4)
            std::printf("  +%3zu: %08x\n", i, unsigned(bytes[i] | bytes[i + 1] << 8 | bytes[i + 2] << 16 | bytes[i + 3] << 24));
    }
    std::fflush(stdout);
    const rt::rootsig::DecodeResult d = rt::rootsig::parse_rts0(bytes, size);
    if (!d.ok) {
        std::printf("oracle parse failed: %s\n", d.error.c_str());
        return false;
    }
    model = d.sig;
    return true;
}

}  // namespace

int main() {
    using namespace rt::rootsig;

    // 4 constants (VS-visible) + 1 CBV: parses back to the same model and
    // packs to 6 dwords, exactly like the portable packer says for 1.1.
    D3D12_ROOT_PARAMETER1 params[2]{};
    params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    params[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_VERTEX;
    params[0].Constants.ShaderRegister = 0;
    params[0].Constants.RegisterSpace = 0;
    params[0].Constants.Num32BitValues = 4;
    params[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
    params[1].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    params[1].Descriptor.ShaderRegister = 2;
    params[1].Descriptor.RegisterSpace = 0;
    params[1].Descriptor.Flags = D3D12_ROOT_DESCRIPTOR_FLAG_NONE;

    const D3D12_VERSIONED_ROOT_SIGNATURE_DESC desc = make_desc(params, 2);
    Blobs blobs;
    const HRESULT hr =
        D3D12SerializeVersionedRootSignature(&desc, &blobs.blob, &blobs.error);
    check(hr == S_OK, "Microsoft serializer accepts the synthetic desc");
    if (hr != S_OK) {
        if (blobs.error != nullptr) {
            std::printf("serializer error: %s\n",
                        static_cast<const char*>(blobs.error->GetBufferPointer()));
        }
        std::printf("%d FAILURES\n", g_failures + 1);
        return 1;
    }

    RootSignature model;
    check(parse_blob(blobs.blob, model), "oracle blob parses");
    if (g_failures != 0) {
        // Never index into a model that failed to parse (that segfaulted CI).
        std::printf("%d FAILURES\n", g_failures);
        return 1;
    }
    check(model.params.size() == 2, "oracle model has 2 params");
    check(model.params[0].type == ParamType::Constants &&
              model.params[0].num_values == 4 &&
              model.params[0].visibility == Visibility::Vertex,
          "oracle constants match");
    check(model.params[1].type == ParamType::Descriptor,
          "oracle CBV matches");
    const PackResult packed = pack(model);
    check(packed.ok && packed.dword_count == 6, "oracle model packs to 6 dwords");

    // The descriptor count and range offset are independent. The final SRV range is unbounded
    // and begins immediately after the fixed range, but the whole table still occupies one GPU address.
    {
        D3D12_DESCRIPTOR_RANGE1 ranges[] = {
            {D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 2, 3, 0, D3D12_DESCRIPTOR_RANGE_FLAG_DATA_STATIC, 0},
            {D3D12_DESCRIPTOR_RANGE_TYPE_SRV, UINT32_MAX, 7, 1,
             D3D12_DESCRIPTOR_RANGE_FLAG_DESCRIPTORS_VOLATILE, D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND},
        };
        D3D12_ROOT_PARAMETER1 table_and_constants[2] = {{}, params[0]};
        auto& table = table_and_constants[0];
        table.ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        table.ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
        table.DescriptorTable = {2, ranges};
        const auto unbounded_desc = make_desc(table_and_constants, 2);
        Blobs unbounded;
        const HRESULT result =
            D3D12SerializeVersionedRootSignature(&unbounded_desc, &unbounded.blob, &unbounded.error);
        check(result == S_OK, "Microsoft serializer accepts an unbounded SRV range");
        RootSignature table_model;
        if (SUCCEEDED(result) && unbounded.blob &&
            parse_blob(unbounded.blob, table_model) && table_model.params.size() == 2 &&
            table_model.params[0].ranges.size() == 2) {
            const auto& fixed = table_model.params[0].ranges[0];
            const auto& tail = table_model.params[0].ranges[1];
            check(table_model.params[0].type == ParamType::Table &&
                      table_model.params[0].visibility == Visibility::Pixel &&
                      fixed.num_descriptors == 2 && fixed.offset_in_table == 0 &&
                      tail.range_type == D3D12_DESCRIPTOR_RANGE_TYPE_SRV &&
                      tail.num_descriptors == UINT32_MAX && tail.base_shader_register == 7 &&
                      tail.register_space == 1 && tail.flags == D3D12_DESCRIPTOR_RANGE_FLAG_DESCRIPTORS_VOLATILE &&
                      tail.offset_in_table == kRangeOffsetAppend,
                  "fixed and unbounded SRV ranges survive native serialization");
            const auto table_pack = pack(table_model);
            check(table_pack.ok && table_pack.d3d12_cost == 5 && table_pack.dword_count == 6 &&
                      table_pack.provenance.size() == 6 &&
                      table_pack.provenance[1].param_index == 0 && table_pack.provenance[2].param_index == 1,
                  "unbounded SRV table occupies two SGPRs before the following constants");
        } else {
            check(false, "native unbounded table parses with two ranges and two parameters");
        }
    }

    // Static samplers, one signature per version: every field round-trips through the real serializer, and 1.2
    // adds D3D12_STATIC_SAMPLER_DESC1::Flags (a 14th dword) to each sampler record. Each also carries a table and a
    // root descriptor, so the 1.1/1.2 descriptor and range flags sit between the samplers and the header.
    D3D12_STATIC_SAMPLER_DESC1 samplers[2]{};
    samplers[0].Filter = D3D12_FILTER_ANISOTROPIC;
    samplers[0].AddressU = D3D12_TEXTURE_ADDRESS_MODE_WRAP;
    samplers[0].AddressV = D3D12_TEXTURE_ADDRESS_MODE_MIRROR;
    samplers[0].AddressW = D3D12_TEXTURE_ADDRESS_MODE_BORDER;
    samplers[0].MipLODBias = -1.5f;
    samplers[0].MaxAnisotropy = 8;
    samplers[0].ComparisonFunc = D3D12_COMPARISON_FUNC_LESS_EQUAL;
    samplers[0].BorderColor = D3D12_STATIC_BORDER_COLOR_OPAQUE_WHITE;
    samplers[0].MinLOD = 1.0f;
    samplers[0].MaxLOD = 9.0f;
    samplers[0].ShaderRegister = 4;
    samplers[0].RegisterSpace = 2;
    samplers[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
    samplers[1].Filter = D3D12_FILTER_MIN_MAG_MIP_POINT;
    samplers[1].AddressU = samplers[1].AddressV = samplers[1].AddressW = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
    samplers[1].MaxLOD = D3D12_FLOAT32_MAX;
    samplers[1].ShaderRegister = 7;
    samplers[1].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    samplers[1].BorderColor = D3D12_STATIC_BORDER_COLOR_OPAQUE_BLACK;
    D3D12_STATIC_SAMPLER_DESC samplers0[2]{};
    for (int i = 0; i < 2; ++i) {
        const D3D12_STATIC_SAMPLER_DESC1& s = samplers[i];
        samplers0[i] = {s.Filter, s.AddressU, s.AddressV, s.AddressW, s.MipLODBias, s.MaxAnisotropy,
                        s.ComparisonFunc, s.BorderColor, s.MinLOD, s.MaxLOD, s.ShaderRegister, s.RegisterSpace,
                        s.ShaderVisibility};
    }
    // Two ranges, so the range stride (20 bytes in 1.0, 24 with flags) is under test too. Every field differs from
    // its neighbours (an explicit table offset, nonzero registers and spaces, distinct flags), so a swapped or
    // shifted field read shows up instead of matching by accident.
    D3D12_DESCRIPTOR_RANGE range0[2] = {
        {D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 3, 1, 0, D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND},
        {D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 2, 4, 2, 7}};
    D3D12_DESCRIPTOR_RANGE1 range1[2] = {
        {D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 3, 1, 0, D3D12_DESCRIPTOR_RANGE_FLAG_DATA_STATIC,
         D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND},
        {D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 2, 4, 2, D3D12_DESCRIPTOR_RANGE_FLAG_DATA_VOLATILE, 7}};
    D3D12_ROOT_PARAMETER params0[2]{};
    params0[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    params0[0].DescriptorTable = {2, range0};
    params0[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;
    params0[1].Descriptor = {5, 1};
    D3D12_ROOT_PARAMETER1 params1[2]{};
    params1[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    params1[0].DescriptorTable = {2, range1};
    params1[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;
    params1[1].Descriptor = {5, 1, D3D12_ROOT_DESCRIPTOR_FLAG_DATA_VOLATILE};

    const auto same_sampler = [](const StaticSampler& got, const D3D12_STATIC_SAMPLER_DESC1& want, bool flags) {
        return got.filter == uint32_t(want.Filter) && got.address_u == uint32_t(want.AddressU) &&
               got.address_v == uint32_t(want.AddressV) && got.address_w == uint32_t(want.AddressW) &&
               got.mip_lod_bias == want.MipLODBias && got.max_anisotropy == want.MaxAnisotropy &&
               got.comparison_func == uint32_t(want.ComparisonFunc) &&
               got.border_color == uint32_t(want.BorderColor) && got.min_lod == want.MinLOD &&
               got.max_lod == want.MaxLOD && got.shader_register == want.ShaderRegister &&
               got.register_space == want.RegisterSpace &&
               got.visibility == static_cast<Visibility>(want.ShaderVisibility) &&
               got.flags == (flags ? uint32_t(want.Flags) : 0u);
    };
    // Ranges as the model should hold them: 1.0 has no range flags (0 in the model); 1.1 and 1.2 use range1.
    const auto same_ranges = [&](const std::vector<DescriptorRange>& got, bool range_flags) {
        if (got.size() != 2) return false;
        for (int i = 0; i < 2; ++i) {
            const D3D12_DESCRIPTOR_RANGE1& want = range1[i];
            if (got[i].range_type != uint32_t(want.RangeType) || got[i].num_descriptors != want.NumDescriptors ||
                got[i].base_shader_register != want.BaseShaderRegister ||
                got[i].register_space != want.RegisterSpace ||
                got[i].flags != (range_flags ? uint32_t(want.Flags) : 0u) ||
                got[i].offset_in_table != want.OffsetInDescriptorsFromTableStart)
                return false;
        }
        return true;
    };
    const auto serialize = [&](const D3D12_VERSIONED_ROOT_SIGNATURE_DESC& d, Version version, const char* name,
                               bool sampler_flags) {
        const bool range_flags = version != Version::V1_0;
        Blobs out;
        char what[160];
        const HRESULT result = D3D12SerializeVersionedRootSignature(&d, &out.blob, &out.error);
        std::snprintf(what, sizeof(what), "%s: Microsoft serializer accepts it", name);
        check(result == S_OK, what);
        if (result != S_OK) {
            if (out.error) std::printf("serializer error: %s\n", static_cast<const char*>(out.error->GetBufferPointer()));
            return;
        }
        RootSignature got;
        std::snprintf(what, sizeof(what), "%s: parses", name);
        const bool parsed = parse_blob(out.blob, got);
        check(parsed, what);
        if (!parsed) return;
        std::snprintf(what, sizeof(what), "%s: version, a two-range table, a root SRV and two samplers", name);
        check(got.version == version && got.params.size() == 2 && got.params[0].type == ParamType::Table &&
                  got.params[0].ranges.size() == 2 && got.params[0].ranges[0].range_type == 0 &&
                  got.params[0].ranges[0].num_descriptors == 3 && got.params[0].ranges[1].range_type == 1 &&
                  got.params[0].ranges[1].num_descriptors == 2 &&
                  got.params[1].type == ParamType::Descriptor && got.params[1].descriptor_register == 5 &&
                  got.params[1].descriptor_space == 1 && got.static_samplers.size() == 2,
              what);
        if (got.static_samplers.size() != 2) return;
        std::snprintf(what, sizeof(what), "%s: every range field round-trips (flags before the table offset)", name);
        check(same_ranges(got.params[0].ranges, range_flags), what);
        std::snprintf(what, sizeof(what), "%s: every sampler field round-trips", name);
        check(same_sampler(got.static_samplers[0], samplers[0], sampler_flags) &&
                  same_sampler(got.static_samplers[1], samplers[1], sampler_flags),
              what);
        // The canonical cache codec keeps all of it, flags included.
        const std::vector<uint8_t> bytes = encode_canonical(got);
        const DecodeResult again = decode_canonical(bytes.data(), bytes.size());
        std::snprintf(what, sizeof(what), "%s: canonical codec round-trips the parsed model", name);
        check(again.ok && again.sig.version == version && again.sig.static_samplers.size() == 2 &&
                  again.sig.params.size() == 2 && same_ranges(again.sig.params[0].ranges, range_flags) &&
                  same_sampler(again.sig.static_samplers[0], samplers[0], sampler_flags) &&
                  same_sampler(again.sig.static_samplers[1], samplers[1], sampler_flags),
              what);
    };

    D3D12_VERSIONED_ROOT_SIGNATURE_DESC v10{};
    v10.Version = D3D_ROOT_SIGNATURE_VERSION_1_0;
    v10.Desc_1_0 = {2, params0, 2, samplers0, D3D12_ROOT_SIGNATURE_FLAG_NONE};
    serialize(v10, Version::V1_0, "1.0 with static samplers", false);

    D3D12_VERSIONED_ROOT_SIGNATURE_DESC v11{};
    v11.Version = D3D_ROOT_SIGNATURE_VERSION_1_1;
    v11.Desc_1_1 = {2, params1, 2, samplers0, D3D12_ROOT_SIGNATURE_FLAG_NONE};
    serialize(v11, Version::V1_1, "1.1 with static samplers", false);

    // The 1.2 cases below need a host that serializes 1.2. Windows 10's built-in d3d12.dll stops at 1.1 and rejects
    // even an empty 1.2 description; their sampler flags have no 1.1 form, so they are skipped there.
    D3D12_VERSIONED_ROOT_SIGNATURE_DESC empty12{};
    empty12.Version = D3D_ROOT_SIGNATURE_VERSION_1_2;
    Blobs probe;
    if (FAILED(D3D12SerializeVersionedRootSignature(&empty12, &probe.blob, &probe.error))) {
        std::printf("SKIP 1.2 with sampler flags, 1.2 with non-normalized coordinates (the host d3d12.dll can't "
                    "serialize root signature 1.2, e.g. Windows 10)\n");
        if (g_failures != 0) std::printf("%d FAILURES\n", g_failures);
        return g_failures == 0 ? 0 : 1;
    }

    // 1.2: a UINT border colour on one sampler, no flags on the other.
    samplers[0].BorderColor = D3D12_STATIC_BORDER_COLOR_OPAQUE_WHITE_UINT;
    samplers[0].Flags = D3D12_SAMPLER_FLAG_UINT_BORDER_COLOR;
    D3D12_VERSIONED_ROOT_SIGNATURE_DESC v12{};
    v12.Version = D3D_ROOT_SIGNATURE_VERSION_1_2;
    v12.Desc_1_2 = {2, params1, 2, samplers, D3D12_ROOT_SIGNATURE_FLAG_NONE};
    serialize(v12, Version::V1_2, "1.2 with sampler flags", true);

    // 1.2: non-normalized coordinates, which D3D12 allows only with point/linear filtering, no mips, no anisotropy,
    // no comparison and clamp or border addressing.
    samplers[0] = {};
    samplers[0].Filter = D3D12_FILTER_MIN_MAG_LINEAR_MIP_POINT;
    samplers[0].AddressU = samplers[0].AddressV = samplers[0].AddressW = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
    samplers[0].ShaderRegister = 4;
    samplers[0].RegisterSpace = 2;
    samplers[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
    samplers[0].BorderColor = D3D12_STATIC_BORDER_COLOR_OPAQUE_BLACK;
    samplers[0].Flags = D3D12_SAMPLER_FLAG_NON_NORMALIZED_COORDINATES;
    serialize(v12, Version::V1_2, "1.2 with non-normalized coordinates", true);

    if (g_failures != 0) std::printf("%d FAILURES\n", g_failures);
    return g_failures == 0 ? 0 : 1;
}
