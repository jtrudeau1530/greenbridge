// Xbox Series root-signature imports forwarded to the Windows D3D12 exports.
// The SDK types describe the Windows side. Xbox ABI and binary compatibility,
// including the layout of guest descriptions and serialized blobs, remain unverified.
#include <windows.h>
#include <d3d12.h>

#include <cstdio>
#include <type_traits>

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

HRESULT WINAPI serialize_root_signature(const D3D12_VERSIONED_ROOT_SIGNATURE_DESC* desc,
                                        ID3DBlob** blob, ID3DBlob** error) {
    static auto fn = reinterpret_cast<SerializeFn>(host("D3D12SerializeVersionedRootSignature"));
    return fn ? fn(desc, blob, error) : E_NOTIMPL;
}

HRESULT WINAPI create_root_signature_deserializer(const void* data, SIZE_T size, REFIID iid, void** out) {
    static auto fn = reinterpret_cast<DeserializerFn>(host("D3D12CreateRootSignatureDeserializer"));
    return fn ? fn(data, size, iid, out) : E_NOTIMPL;
}

static_assert(std::is_same_v<decltype(&serialize_root_signature), SerializeFn>);
static_assert(std::is_same_v<decltype(&create_root_signature_deserializer), DeserializerFn>);

}  // namespace

SHIM_TABLE(d3d12_xs,
    {"d3d12_xs.dll", "D3D12SerializeVersionedRootSignature", reinterpret_cast<void*>(&serialize_root_signature)},
    {"d3d12_xs.dll", "D3D12CreateRootSignatureDeserializer", reinterpret_cast<void*>(&create_root_signature_deserializer)},
);
