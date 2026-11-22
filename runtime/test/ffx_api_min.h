// The subset of the public FidelityFX API that fsr_dispatch_test.cpp calls, written out from the FidelityFX SDK
// v2.4.0 headers Kits/FidelityFX/api/include/ffx_api.h, api/include/dx12/ffx_api_dx12.h and
// upscalers/include/ffx_upscale.h (https://github.com/GPUOpen-LibrariesAndSDKs/FidelityFX-SDK). Those headers carry
// the MIT license below. Only declarations are here: no AMD code, and nothing from the game's DLL.
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
// documentation files (the "Software"), to deal in the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and
// to permit persons to whom the Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all copies or substantial portions of
// the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
// THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
// CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.
#pragma once

#include <cstdint>

namespace ffx {

using ReturnCode = uint32_t;
using StructType = uint64_t;
using Context = void*;

enum : ReturnCode {
    RETURN_OK = 0,
    RETURN_ERROR = 1,
    RETURN_ERROR_UNKNOWN_DESCTYPE = 2,
    RETURN_ERROR_RUNTIME_ERROR = 3,
    RETURN_NO_PROVIDER = 4,
    RETURN_ERROR_MEMORY = 5,
    RETURN_ERROR_PARAMETER = 6,
    RETURN_PROVIDER_NO_SUPPORT_NEW_DESCTYPE = 7,
};

constexpr uint32_t EFFECT_MASK = 0x00ff0000u;
constexpr uint32_t BACKEND_MASK = 0xff000000u;
constexpr uint32_t EFFECT_ID_UPSCALE = 0x00010000u;
constexpr uint32_t BACKEND_ID_DX12 = 0x00000000u;
constexpr uint32_t BACKEND_ID_XBOX = 0x02000000u;  // declared in ffx_api.h; its descriptors are not public

constexpr StructType effect_sub_id(uint32_t effect, uint32_t sub) { return (effect & EFFECT_MASK) | (sub & ~EFFECT_MASK); }
constexpr StructType backend_sub_id(uint32_t backend, uint32_t sub) {
    return (backend & BACKEND_MASK) | (sub & ~BACKEND_MASK);
}

struct Header {
    StructType type;
    Header* pNext;
};

using Message = void (*)(uint32_t type, const wchar_t* message);

constexpr StructType CONFIGURE_DESC_TYPE_GLOBALDEBUG = 7u;
constexpr uint32_t GLOBALDEBUG_LEVEL_VERBOSE = 0xfffffffu;
struct ConfigureDescGlobalDebug {
    Header header;
    uint64_t effectId;
    Message fpMessage;
    uint32_t debugLevel;
};

constexpr StructType QUERY_DESC_TYPE_GET_VERSIONS = 4u;
struct QueryDescGetVersions {
    Header header;
    uint64_t createDescType;
    void* device;
    uint64_t* outputCount;
    uint64_t* versionIds;
    const char** versionNames;
};

struct Dimensions2D {
    uint32_t width;
    uint32_t height;
};

constexpr StructType CREATE_CONTEXT_DESC_TYPE_UPSCALE = effect_sub_id(EFFECT_ID_UPSCALE, 0x00);
constexpr uint32_t UPSCALE_ENABLE_HIGH_DYNAMIC_RANGE = 1u << 0;
constexpr uint32_t UPSCALE_ENABLE_DEPTH_INVERTED = 1u << 3;
constexpr uint32_t UPSCALE_ENABLE_AUTO_EXPOSURE = 1u << 5;
constexpr uint32_t UPSCALE_ENABLE_DEBUG_CHECKING = 1u << 7;
struct CreateContextDescUpscale {
    Header header;
    uint32_t flags;
    Dimensions2D maxRenderSize;
    Dimensions2D maxUpscaleSize;
    Message fpMessage;
};

constexpr uint32_t upscaler_version(uint32_t major, uint32_t minor, uint32_t patch) {
    return (major << 22) | (minor << 12) | patch;
}
constexpr StructType CREATE_CONTEXT_DESC_TYPE_UPSCALE_VERSION = effect_sub_id(EFFECT_ID_UPSCALE, 0x0b);
struct CreateContextDescUpscaleVersion {
    Header header;
    uint32_t version;  // FFX_UPSCALER_VERSION the application was built against
};

constexpr StructType QUERY_DESC_TYPE_UPSCALE_GETUPSCALERATIOFROMQUALITYMODE = effect_sub_id(EFFECT_ID_UPSCALE, 0x02);
struct QueryDescUpscaleGetUpscaleRatioFromQualityMode {
    Header header;
    uint32_t qualityMode;
    float* pOutUpscaleRatio;
};

constexpr StructType QUERY_DESC_TYPE_UPSCALE_GETRENDERRESOLUTIONFROMQUALITYMODE = effect_sub_id(EFFECT_ID_UPSCALE, 0x03);
struct QueryDescUpscaleGetRenderResolutionFromQualityMode {
    Header header;
    uint32_t displayWidth;
    uint32_t displayHeight;
    uint32_t qualityMode;
    uint32_t* pOutRenderWidth;
    uint32_t* pOutRenderHeight;
};

enum : uint32_t { QUALITY_MODE_NATIVEAA = 0, QUALITY_MODE_QUALITY = 1, QUALITY_MODE_PERFORMANCE = 3 };

// ffx_api_dx12.h. The DLL's own backend descriptor would be a BACKEND_ID_XBOX sub-id; its layout is not public.
constexpr StructType CREATE_CONTEXT_DESC_TYPE_BACKEND_DX12 = backend_sub_id(BACKEND_ID_DX12, 0x02);
struct CreateBackendDX12Desc {
    Header header;
    void* device;  // ID3D12Device*
};

using PfnCreateContext = ReturnCode (*)(Context* context, Header* desc, const void* memCb);
using PfnDestroyContext = ReturnCode (*)(Context* context, const void* memCb);
using PfnConfigure = ReturnCode (*)(Context* context, const Header* desc);
using PfnQuery = ReturnCode (*)(Context* context, Header* desc);
using PfnDispatch = ReturnCode (*)(Context* context, const Header* desc);

inline const char* return_code_name(ReturnCode rc) {
    switch (rc) {
        case RETURN_OK: return "OK";
        case RETURN_ERROR: return "ERROR";
        case RETURN_ERROR_UNKNOWN_DESCTYPE: return "ERROR_UNKNOWN_DESCTYPE";
        case RETURN_ERROR_RUNTIME_ERROR: return "ERROR_RUNTIME_ERROR";
        case RETURN_NO_PROVIDER: return "NO_PROVIDER";
        case RETURN_ERROR_MEMORY: return "ERROR_MEMORY";
        case RETURN_ERROR_PARAMETER: return "ERROR_PARAMETER";
        case RETURN_PROVIDER_NO_SUPPORT_NEW_DESCTYPE: return "PROVIDER_NO_SUPPORT_NEW_DESCTYPE";
        default: return "?";
    }
}

}  // namespace ffx
