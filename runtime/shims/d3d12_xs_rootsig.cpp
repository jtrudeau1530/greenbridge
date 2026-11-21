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
