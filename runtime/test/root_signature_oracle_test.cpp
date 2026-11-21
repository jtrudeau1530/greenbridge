// Windows oracle for the RTS0 parser hypothesis (needs a real PC d3d12.lib).
//
// Builds synthetic versioned root-signature descriptors by hand (no game
// data), serializes each with Microsoft's D3D12SerializeVersionedRootSignature,
// and requires rt::rootsig::parse_rts0 to recover the same model. If the
// hypothesis layout in root_signature.cpp ever disagrees with the real
// serializer, this fails loudly here instead of corrupting draws silently.
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
    check(model.params.size() == 2, "oracle model has 2 params");
    check(model.params[0].type == ParamType::Constants &&
              model.params[0].num_values == 4 &&
              model.params[0].visibility == Visibility::Vertex,
          "oracle constants match");
    check(model.params[1].type == ParamType::Descriptor,
          "oracle CBV matches");
    const PackResult packed = pack(model);
    check(packed.ok && packed.dword_count == 6, "oracle model packs to 6 dwords");

    if (g_failures != 0) std::printf("%d FAILURES\n", g_failures);
    return g_failures == 0 ? 0 : 1;
}
