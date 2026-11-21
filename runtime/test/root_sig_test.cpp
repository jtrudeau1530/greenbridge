// Checks the clean-room root-signature flag fix against the real PC d3d12.dll.
//
// A synthetic empty root signature is serialized twice: untouched (Xbox flags 0x03000000) must be rejected by the
// PC serializer, and the copy produced by the shim helper (high byte cleared) must be accepted. The same is done
// for the legacy D3D12SerializeRootSignature path, plus a direct check that only bits 24-31 are cleared.
#include <windows.h>
#include <d3d12.h>

#include <cstdio>

#include "d3d12_xs_rootsig.h"
#include "shims.h"

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

constexpr uint32_t kXboxFlags = 0x03000000u;

D3D12_ROOT_SIGNATURE_DESC make_legacy_desc(uint32_t flags) {
    D3D12_ROOT_SIGNATURE_DESC desc{};
    desc.NumParameters = 0;
    desc.pParameters = nullptr;
    desc.NumStaticSamplers = 0;
    desc.pStaticSamplers = nullptr;
    desc.Flags = static_cast<D3D12_ROOT_SIGNATURE_FLAGS>(flags);
    return desc;
}

D3D12_ROOT_SIGNATURE_DESC1 make_legacy_desc1(uint32_t flags) {
    D3D12_ROOT_SIGNATURE_DESC1 desc{};
    desc.NumParameters = 0;
    desc.pParameters = nullptr;
    desc.NumStaticSamplers = 0;
    desc.pStaticSamplers = nullptr;
    desc.Flags = static_cast<D3D12_ROOT_SIGNATURE_FLAGS>(flags);
    return desc;
}

D3D12_ROOT_SIGNATURE_DESC2 make_legacy_desc2(uint32_t flags) {
    D3D12_ROOT_SIGNATURE_DESC2 desc{};
    desc.NumParameters = 0;
    desc.pParameters = nullptr;
    desc.NumStaticSamplers = 0;
    desc.pStaticSamplers = nullptr;
    desc.Flags = static_cast<D3D12_ROOT_SIGNATURE_FLAGS>(flags);
    return desc;
}

void test_versioned_1_0() {
    D3D12_VERSIONED_ROOT_SIGNATURE_DESC xbox{};
    xbox.Version = D3D_ROOT_SIGNATURE_VERSION_1_0;
    xbox.Desc_1_0 = make_legacy_desc(kXboxFlags);

    {
        Blobs blobs;
        HRESULT hr = D3D12SerializeVersionedRootSignature(&xbox, &blobs.blob, &blobs.error);
        check(FAILED(hr), "versioned 1.0 with Xbox flags 0x03000000 is rejected by the PC serializer");
    }

    D3D12_VERSIONED_ROOT_SIGNATURE_DESC cleared{};
    bool known = rt::d3d12_xs::copy_versioned_root_signature_desc_clear_xbox_flags(xbox, cleared);
    check(known, "versioned 1.0 descriptor version is recognised");
    check(cleared.Version == xbox.Version, "versioned 1.0 copy keeps the version");
    check(cleared.Desc_1_0.Flags == 0, "versioned 1.0 copy has the high flag byte cleared");
    check(xbox.Desc_1_0.Flags == static_cast<D3D12_ROOT_SIGNATURE_FLAGS>(kXboxFlags),
          "versioned 1.0 original descriptor is unchanged");

    Blobs blobs;
    HRESULT hr = D3D12SerializeVersionedRootSignature(&cleared, &blobs.blob, &blobs.error);
    check(SUCCEEDED(hr), "versioned 1.0 copy with cleared flags is accepted by the PC serializer");
}

void test_versioned_1_1() {
    D3D12_VERSIONED_ROOT_SIGNATURE_DESC xbox{};
    xbox.Version = D3D_ROOT_SIGNATURE_VERSION_1_1;
    xbox.Desc_1_1 = make_legacy_desc1(kXboxFlags);

    {
        Blobs blobs;
        HRESULT hr = D3D12SerializeVersionedRootSignature(&xbox, &blobs.blob, &blobs.error);
        check(FAILED(hr), "versioned 1.1 with Xbox flags 0x03000000 is rejected by the PC serializer");
    }

    D3D12_VERSIONED_ROOT_SIGNATURE_DESC cleared{};
    bool known = rt::d3d12_xs::copy_versioned_root_signature_desc_clear_xbox_flags(xbox, cleared);
    check(known, "versioned 1.1 descriptor version is recognised");
    check(cleared.Desc_1_1.Flags == 0, "versioned 1.1 copy has the high flag byte cleared");
    check(xbox.Desc_1_1.Flags == static_cast<D3D12_ROOT_SIGNATURE_FLAGS>(kXboxFlags),
          "versioned 1.1 original descriptor is unchanged");

    Blobs blobs;
    HRESULT hr = D3D12SerializeVersionedRootSignature(&cleared, &blobs.blob, &blobs.error);
    check(SUCCEEDED(hr), "versioned 1.1 copy with cleared flags is accepted by the PC serializer");
}

void test_versioned_1_2() {
    D3D12_VERSIONED_ROOT_SIGNATURE_DESC xbox{};
    xbox.Version = D3D_ROOT_SIGNATURE_VERSION_1_2;
    xbox.Desc_1_2 = make_legacy_desc2(kXboxFlags);

    {
        Blobs blobs;
        HRESULT hr = D3D12SerializeVersionedRootSignature(&xbox, &blobs.blob, &blobs.error);
        check(FAILED(hr), "versioned 1.2 with Xbox flags 0x03000000 is rejected by the PC serializer");
    }

    D3D12_VERSIONED_ROOT_SIGNATURE_DESC cleared{};
    bool known = rt::d3d12_xs::copy_versioned_root_signature_desc_clear_xbox_flags(xbox, cleared);
    check(known, "versioned 1.2 descriptor version is recognised");
    check(cleared.Desc_1_2.Flags == 0, "versioned 1.2 copy has the high flag byte cleared");
    check(xbox.Desc_1_2.Flags == static_cast<D3D12_ROOT_SIGNATURE_FLAGS>(kXboxFlags),
          "versioned 1.2 original descriptor is unchanged");

    Blobs blobs;
    HRESULT hr = D3D12SerializeVersionedRootSignature(&cleared, &blobs.blob, &blobs.error);
    check(SUCCEEDED(hr), "versioned 1.2 copy with cleared flags is accepted by the PC serializer");
}

void test_legacy_entry_point() {
    D3D12_ROOT_SIGNATURE_DESC xbox = make_legacy_desc(kXboxFlags);

    {
        Blobs blobs;
        HRESULT hr = D3D12SerializeRootSignature(&xbox, D3D_ROOT_SIGNATURE_VERSION_1_0, &blobs.blob, &blobs.error);
        check(FAILED(hr), "legacy serializer rejects flags 0x03000000");
    }

    D3D12_ROOT_SIGNATURE_DESC cleared{};
    rt::d3d12_xs::copy_root_signature_desc_clear_xbox_flags(xbox, cleared);
    check(cleared.Flags == 0, "legacy copy has the high flag byte cleared");
    check(xbox.Flags == static_cast<D3D12_ROOT_SIGNATURE_FLAGS>(kXboxFlags), "legacy original is unchanged");

    Blobs blobs;
    HRESULT hr = D3D12SerializeRootSignature(&cleared, D3D_ROOT_SIGNATURE_VERSION_1_0, &blobs.blob, &blobs.error);
    check(SUCCEEDED(hr), "legacy copy with cleared flags is accepted by the PC serializer");
}

void test_preserves_low_bits_and_unknown_versions() {
    // Flags 0x03000045: clear the high byte but keep the valid low bits (input assembler + stream output).
    D3D12_VERSIONED_ROOT_SIGNATURE_DESC xbox{};
    xbox.Version = D3D_ROOT_SIGNATURE_VERSION_1_0;
    xbox.Desc_1_0 = make_legacy_desc(0x03000045u);

    D3D12_VERSIONED_ROOT_SIGNATURE_DESC cleared{};
    (void)rt::d3d12_xs::copy_versioned_root_signature_desc_clear_xbox_flags(xbox, cleared);
    check(cleared.Desc_1_0.Flags == static_cast<D3D12_ROOT_SIGNATURE_FLAGS>(0x45u),
          "only the high flag byte is cleared; low flag bits are preserved");

    D3D12_VERSIONED_ROOT_SIGNATURE_DESC unknown = xbox;
    unknown.Version = static_cast<D3D_ROOT_SIGNATURE_VERSION>(0x7f);
    D3D12_VERSIONED_ROOT_SIGNATURE_DESC forwarded{};
    bool known = rt::d3d12_xs::copy_versioned_root_signature_desc_clear_xbox_flags(unknown, forwarded);
    check(!known, "unknown descriptor version reports that it was not rewritten");
    const bool unchanged = forwarded.Version == unknown.Version &&
                           forwarded.Desc_1_0.NumParameters == unknown.Desc_1_0.NumParameters &&
                           forwarded.Desc_1_0.pParameters == unknown.Desc_1_0.pParameters &&
                           forwarded.Desc_1_0.NumStaticSamplers == unknown.Desc_1_0.NumStaticSamplers &&
                           forwarded.Desc_1_0.pStaticSamplers == unknown.Desc_1_0.pStaticSamplers &&
                           forwarded.Desc_1_0.Flags == unknown.Desc_1_0.Flags;
    check(unchanged, "unknown descriptor version is copied unchanged for forwarding");
}

void test_registered_shim() {
    using SerializeFn =
        HRESULT(WINAPI*)(const D3D12_VERSIONED_ROOT_SIGNATURE_DESC*, ID3DBlob**, ID3DBlob**);
    auto fn = reinterpret_cast<SerializeFn>(
        rt::shims::find("d3d12_xs.dll", "D3D12SerializeVersionedRootSignature"));
    check(fn != nullptr, "shim registry exposes d3d12_xs!D3D12SerializeVersionedRootSignature");
    check(rt::shims::find("d3d12_xs.dll", "D3D12SerializeRootSignature") != nullptr,
          "shim registry exposes d3d12_xs!D3D12SerializeRootSignature");
    check(rt::shims::find("d3d12_xs.dll", "D3D12CreateRootSignatureDeserializer") != nullptr,
          "shim registry exposes d3d12_xs!D3D12CreateRootSignatureDeserializer");
    check(rt::shims::find("d3d12_xs.dll", "D3D12XboxCreateDevice") != nullptr,
          "shim registry exposes d3d12_xs!D3D12XboxCreateDevice");
    if (!fn) return;

    D3D12_VERSIONED_ROOT_SIGNATURE_DESC xbox{};
    xbox.Version = D3D_ROOT_SIGNATURE_VERSION_1_0;
    xbox.Desc_1_0 = make_legacy_desc(kXboxFlags);

    Blobs blobs;
    HRESULT hr = fn(&xbox, &blobs.blob, &blobs.error);
    check(SUCCEEDED(hr), "registered shim clears the Xbox flag byte and the PC serializer accepts the result");
    check(xbox.Desc_1_0.Flags == static_cast<D3D12_ROOT_SIGNATURE_FLAGS>(kXboxFlags),
          "registered shim leaves the caller's descriptor unchanged");

    D3D12_VERSIONED_ROOT_SIGNATURE_DESC xbox12{};
    xbox12.Version = D3D_ROOT_SIGNATURE_VERSION_1_2;
    xbox12.Desc_1_2 = make_legacy_desc2(kXboxFlags);

    Blobs blobs12;
    HRESULT hr12 = fn(&xbox12, &blobs12.blob, &blobs12.error);
    check(SUCCEEDED(hr12), "registered shim clears the Xbox flag byte for 1.2 and the PC serializer accepts");
    check(xbox12.Desc_1_2.Flags == static_cast<D3D12_ROOT_SIGNATURE_FLAGS>(kXboxFlags),
          "registered shim leaves the caller's 1.2 descriptor unchanged");
}

}  // namespace

int main() {
    std::printf("root_sig_test: PC d3d12.dll root-signature flag handling\n");
    test_versioned_1_0();
    test_versioned_1_1();
    test_versioned_1_2();
    test_legacy_entry_point();
    test_preserves_low_bits_and_unknown_versions();
    test_registered_shim();
    std::printf("root_sig_test: %s (%d failure%s)\n", g_failures ? "FAILED" : "OK", g_failures,
                g_failures == 1 ? "" : "s");
    return g_failures ? 1 : 0;
}
