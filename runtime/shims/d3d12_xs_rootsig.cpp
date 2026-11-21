#include "d3d12_xs_rootsig.h"

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

}  // namespace rt::d3d12_xs
