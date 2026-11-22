#include "d3d12_xs_rootsig.h"

#include <cstring>

namespace rt::d3d12_xs {
namespace {

D3D12_ROOT_SIGNATURE_FLAGS clear_high_byte(D3D12_ROOT_SIGNATURE_FLAGS flags) {
    return static_cast<D3D12_ROOT_SIGNATURE_FLAGS>(static_cast<uint32_t>(flags) & ~kXboxRootSignatureFlagHighByte);
}

}  // namespace

bool copy_versioned_root_signature_desc_clear_xbox_flags(const D3D12_VERSIONED_ROOT_SIGNATURE_DESC& in,
                                                         D3D12_VERSIONED_ROOT_SIGNATURE_DESC& out) {
    out = in;
    switch (in.Version) {
        case D3D_ROOT_SIGNATURE_VERSION_1_0:
            out.Desc_1_0.Flags = clear_high_byte(in.Desc_1_0.Flags);
            return true;
        case D3D_ROOT_SIGNATURE_VERSION_1_1:
            out.Desc_1_1.Flags = clear_high_byte(in.Desc_1_1.Flags);
            return true;
        case D3D_ROOT_SIGNATURE_VERSION_1_2:
            out.Desc_1_2.Flags = clear_high_byte(in.Desc_1_2.Flags);
            return true;
        default:
            // Unknown version: forward unchanged.
            return false;
    }
}

bool downgrade_root_signature_1_2_to_1_1(const D3D12_VERSIONED_ROOT_SIGNATURE_DESC& in,
                                         std::vector<D3D12_STATIC_SAMPLER_DESC>& samplers,
                                         D3D12_VERSIONED_ROOT_SIGNATURE_DESC& out) {
    if (in.Version != D3D_ROOT_SIGNATURE_VERSION_1_2) return false;
    const D3D12_ROOT_SIGNATURE_DESC2& desc = in.Desc_1_2;
    if (desc.NumStaticSamplers != 0 && !desc.pStaticSamplers) return false;
    for (UINT i = 0; i < desc.NumStaticSamplers; ++i)
        if (desc.pStaticSamplers[i].Flags != D3D12_SAMPLER_FLAG_NONE) return false;

    samplers.resize(desc.NumStaticSamplers);
    for (UINT i = 0; i < desc.NumStaticSamplers; ++i) {
        const D3D12_STATIC_SAMPLER_DESC1& s = desc.pStaticSamplers[i];
        // D3D12_STATIC_SAMPLER_DESC is D3D12_STATIC_SAMPLER_DESC1 without the trailing Flags.
        samplers[i] = {s.Filter, s.AddressU, s.AddressV, s.AddressW, s.MipLODBias, s.MaxAnisotropy,
                       s.ComparisonFunc, s.BorderColor, s.MinLOD, s.MaxLOD, s.ShaderRegister, s.RegisterSpace,
                       s.ShaderVisibility};
    }
    out = {};
    out.Version = D3D_ROOT_SIGNATURE_VERSION_1_1;
    out.Desc_1_1.NumParameters = desc.NumParameters;
    out.Desc_1_1.pParameters = desc.pParameters;  // D3D12_ROOT_PARAMETER1 in both versions
    out.Desc_1_1.NumStaticSamplers = desc.NumStaticSamplers;
    out.Desc_1_1.pStaticSamplers = samplers.empty() ? nullptr : samplers.data();
    out.Desc_1_1.Flags = desc.Flags;
    return true;
}

void copy_root_signature_desc_clear_xbox_flags(const D3D12_ROOT_SIGNATURE_DESC& in, D3D12_ROOT_SIGNATURE_DESC& out) {
    out = in;
    out.Flags = clear_high_byte(in.Flags);
}

bool copy_rts0_blob_clear_xbox_flags(const void* data, size_t size, std::vector<uint8_t>& out) {
    out.clear();
    if (!data || size < 36) return false;

    const uint8_t* bytes = static_cast<const uint8_t*>(data);
    // The host deserializer only consumes the DXBC container the serializer
    // produces, so a non-DXBC blob is forwarded unchanged (no rewrite).
    if (std::memcmp(bytes, "DXBC", 4) != 0) return false;

    uint32_t chunk_count = 0;
    std::memcpy(&chunk_count, bytes + 28, sizeof(chunk_count));
    if (chunk_count == 0 || chunk_count > 16 || 32 + chunk_count * 4 > size) return false;

    for (uint32_t i = 0; i < chunk_count; ++i) {
        uint32_t off = 0;
        std::memcpy(&off, bytes + 32 + i * 4, sizeof(off));
        // Chunk header (FourCC + size) plus the fixed RTS0 header up to flags.
        if (off > size || size - off < 8 + 24) continue;
        if (std::memcmp(bytes + off, "RTS0", 4) != 0) continue;

        const uint8_t* payload = bytes + off + 8;
        uint32_t flags = 0;
        std::memcpy(&flags, payload + 20, sizeof(flags));
        const uint32_t cleared = static_cast<uint32_t>(clear_high_byte(
            static_cast<D3D12_ROOT_SIGNATURE_FLAGS>(flags)));
        if (cleared == flags) return false;  // no Xbox high byte

        out.assign(bytes, bytes + size);
        std::memcpy(out.data() + off + 8 + 20, &cleared, sizeof(cleared));
        return true;
    }
    return false;
}

}  // namespace rt::d3d12_xs
