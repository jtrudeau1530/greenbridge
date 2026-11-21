// Xbox Series D3D12 imports forwarded to the Windows D3D12 exports.
// The SDK types describe the Windows side. The measured guest ABI differs in the root-signature flag word: Xbox
// sets bits in the high byte (observed 0x03000000) and the PC serializer rejects them, so descriptors are copied
// with that byte cleared before forwarding. D3D12XboxCreateDevice returns an Xbox-ABI proxy (d3d12_xs_device.h)
// over the PC device; only the measured slots are overridden, the rest follow the PC vtable.
#include <windows.h>
#include <d3d12.h>

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <type_traits>

#include "d3d12_xs_device.h"
#include "d3d12_xs_rootsig.h"
#include "shims.h"

namespace {

FARPROC host(const char* name) {
    static HMODULE d3d12 = LoadLibraryExA("d3d12.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
    FARPROC p = d3d12 ? GetProcAddress(d3d12, name) : nullptr;
    if (!p) std::fprintf(stderr, "[rt] d3d12_xs: host d3d12.dll has no %s\n", name);
    return p;
}

using SerializeFn = decltype(&::D3D12SerializeVersionedRootSignature);
using DeserializerFn = decltype(&::D3D12CreateRootSignatureDeserializer);
using LegacySerializeFn = decltype(&::D3D12SerializeRootSignature);
using CreateDeviceFn = decltype(&::D3D12CreateDevice);

HRESULT WINAPI serialize_root_signature(const D3D12_VERSIONED_ROOT_SIGNATURE_DESC* desc,
                                        ID3DBlob** blob, ID3DBlob** error) {
    static auto fn = reinterpret_cast<SerializeFn>(host("D3D12SerializeVersionedRootSignature"));
    if (!fn) return E_NOTIMPL;
    if (!desc) return fn(desc, blob, error);

    D3D12_VERSIONED_ROOT_SIGNATURE_DESC cleared{};
    if (!rt::d3d12_xs::copy_versioned_root_signature_desc_clear_xbox_flags(*desc, cleared)) {
        std::fprintf(stderr,
                     "[rt] d3d12_xs: D3D12SerializeVersionedRootSignature: unknown descriptor version %d; "
                     "forwarding unchanged (Xbox flags not cleared)\n",
                     static_cast<int>(desc->Version));
        return fn(desc, blob, error);
    }
    return fn(&cleared, blob, error);
}

// The package DLLs import the versioned serializer; the legacy entry point is kept because it is trivial to forward
// with the same flag fix and the test uses it.
HRESULT WINAPI serialize_root_signature_legacy(const D3D12_ROOT_SIGNATURE_DESC* desc,
                                               D3D_ROOT_SIGNATURE_VERSION version, ID3DBlob** blob,
                                               ID3DBlob** error) {
    static auto fn = reinterpret_cast<LegacySerializeFn>(host("D3D12SerializeRootSignature"));
    if (!fn) return E_NOTIMPL;
    if (!desc) return fn(desc, version, blob, error);

    D3D12_ROOT_SIGNATURE_DESC cleared{};
    rt::d3d12_xs::copy_root_signature_desc_clear_xbox_flags(*desc, cleared);
    return fn(&cleared, version, blob, error);
}

HRESULT WINAPI create_root_signature_deserializer(const void* data, SIZE_T size, REFIID iid, void** out) {
    static auto fn = reinterpret_cast<DeserializerFn>(host("D3D12CreateRootSignatureDeserializer"));
    return fn ? fn(data, size, iid, out) : E_NOTIMPL;
}

// Measured shape: D3D12XboxCreateDevice(adapter, params, riid, out), where params is a 24-word block that is only
// logged and not used to create the device. The struct's total size is unknown, so the log reads 24 x 32-bit words
// (96 bytes) and no more. The PC D3D12CreateDevice needs a minimum feature level instead, so 12_0 is used as the
// least restrictive floor; higher feature levels are still exposed by the created device.
HRESULT WINAPI D3D12XboxCreateDevice(IUnknown* adapter, const void* params, const IID* riid, void** out) {
    if (!out) return E_POINTER;
    *out = nullptr;
    if (!riid) return E_POINTER;

    std::fprintf(stderr, "[rt] d3d12_xs: D3D12XboxCreateDevice adapter=%p params=%p riid=%08lx-%04x-%04x\n",
                 adapter, params, static_cast<unsigned long>(riid->Data1), riid->Data2, riid->Data3);
    if (params) {
        const uint32_t* words = static_cast<const uint32_t*>(params);
        char line[24 * 11 + 1];
        line[0] = '\0';
        size_t used = 0;
        for (int i = 0; i < 24 && used < sizeof(line); ++i) {
            int n = std::snprintf(line + used, sizeof(line) - used, "%s%08x", i ? " " : "",
                                  static_cast<unsigned>(words[i]));
            if (n < 0) break;
            used += static_cast<size_t>(n);
        }
        std::fprintf(stderr, "[rt] d3d12_xs: D3D12XboxCreateDevice params[24] =%s\n", line);
    } else {
        std::fprintf(stderr, "[rt] d3d12_xs: D3D12XboxCreateDevice params = null\n");
    }

    static std::atomic<bool> proxy_logged{false};
    if (!proxy_logged.exchange(true, std::memory_order_relaxed)) {
        std::fprintf(stderr,
                     "[rt] d3d12_xs: D3D12XboxCreateDevice: wrapping the PC ID3D12Device in the Xbox-ABI proxy "
                     "(slot 62 -> CreateComputePipelineState, slot 76 -> CreateCommittedResource).\n");
    }

    static auto create = reinterpret_cast<CreateDeviceFn>(host("D3D12CreateDevice"));
    if (!create) return E_NOTIMPL;

    ID3D12Device* real = nullptr;
    HRESULT hr = create(adapter, D3D_FEATURE_LEVEL_12_0, __uuidof(ID3D12Device),
                        reinterpret_cast<void**>(&real));
    if (FAILED(hr) || !real) {
        if (real) real->Release();
        std::fprintf(stderr, "[rt] d3d12_xs: D3D12XboxCreateDevice: PC D3D12CreateDevice failed (hr=0x%08lx)\n",
                     static_cast<unsigned long>(hr));
        return FAILED(hr) ? hr : E_FAIL;
    }

    hr = rt::d3d12_xs::create_device_proxy(real, *riid, out);
    if (FAILED(hr)) {
        // create_device_proxy keeps the caller's reference on failure.
        real->Release();
        std::fprintf(stderr, "[rt] d3d12_xs: D3D12XboxCreateDevice: proxy creation failed (hr=0x%08lx)\n",
                     static_cast<unsigned long>(hr));
        return hr;
    }
    std::fprintf(stderr, "[rt] d3d12_xs: D3D12XboxCreateDevice -> proxy %p (PC device %p)\n", *out,
                 static_cast<void*>(real));
    return S_OK;
}

static_assert(std::is_same_v<decltype(&serialize_root_signature), SerializeFn>);
static_assert(std::is_same_v<decltype(&create_root_signature_deserializer), DeserializerFn>);
static_assert(std::is_same_v<decltype(&serialize_root_signature_legacy), LegacySerializeFn>);

}  // namespace

SHIM_TABLE(d3d12_xs,
    {"d3d12_xs.dll", "D3D12SerializeVersionedRootSignature", reinterpret_cast<void*>(&serialize_root_signature)},
    {"d3d12_xs.dll", "D3D12SerializeRootSignature", reinterpret_cast<void*>(&serialize_root_signature_legacy)},
    {"d3d12_xs.dll", "D3D12CreateRootSignatureDeserializer",
     reinterpret_cast<void*>(&create_root_signature_deserializer)},
    {"d3d12_xs.dll", "D3D12XboxCreateDevice", reinterpret_cast<void*>(&D3D12XboxCreateDevice)},
);
