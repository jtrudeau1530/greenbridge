// Helpers for forwarding Xbox root-signature descriptions to the PC d3d12.dll.
//
// The measured Xbox descriptions carry type bits in the high byte of D3D12_ROOT_SIGNATURE_FLAGS (observed value
// 0x03000000). The PC serializer rejects those bits, so the shims pass a shallow copy with bits 24-31 cleared and
// every other flag bit preserved. The pointed-to parameter/sampler arrays are not modified.
#pragma once

#include <d3d12.h>
#include <cstdint>

namespace rt::d3d12_xs {

constexpr uint32_t kXboxRootSignatureFlagHighByte = 0xFF000000u;

// Always copies `in` to `out`. Returns true for the known 1.0/1.1/1.2 descriptor versions and clears the high flag
// byte in the copy. Returns false for an unknown version, leaving the copy byte-for-byte identical to `in` so the
// caller can forward it unchanged and log the unsupported version.
bool copy_versioned_root_signature_desc_clear_xbox_flags(const D3D12_VERSIONED_ROOT_SIGNATURE_DESC& in,
                                                         D3D12_VERSIONED_ROOT_SIGNATURE_DESC& out);

// Legacy D3D12SerializeRootSignature descriptor (version 1.0): copies and clears the high flag byte.
void copy_root_signature_desc_clear_xbox_flags(const D3D12_ROOT_SIGNATURE_DESC& in, D3D12_ROOT_SIGNATURE_DESC& out);

}  // namespace rt::d3d12_xs
