#pragma once

#include <d3d12.h>

// D3D12_FEATURE values introduced after Windows SDK 10.0.26100.0. Keep their ABI values and payload available to
// the runtime and its tests without requiring a newer d3d12.h.
namespace rt::d3d12::compat {

inline constexpr D3D12_FEATURE kFeatureApplicationSpecificDriverState = static_cast<D3D12_FEATURE>(56);
inline constexpr D3D12_FEATURE kFeatureBytecodeBypassHashSupported = static_cast<D3D12_FEATURE>(57);

struct FeatureSupportFlag {
    BOOL Supported;
};

static_assert(sizeof(FeatureSupportFlag) == sizeof(BOOL));

}  // namespace rt::d3d12::compat
