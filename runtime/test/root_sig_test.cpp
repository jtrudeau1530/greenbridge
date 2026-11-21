// Checks the clean-room root-signature flag fix against the real PC d3d12.dll.
//
// A synthetic empty root signature is serialized twice: untouched (Xbox flags 0x03000000) must be rejected by the
// PC serializer, and the copy produced by the shim helper (high byte cleared) must be accepted. The same is done
// for the legacy D3D12SerializeRootSignature path, plus a direct check that only bits 24-31 are cleared.
#include <windows.h>
#include <d3d12.h>

#include <cstdio>
#include <cstring>
#include <vector>

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

// Older in-box D3D12 runtimes (e.g. Windows 11 23H2) refuse root signature
// 1.2 entirely, so the serializer checks that expect 1.2 to be accepted are
// skipped there. The byte-level helper checks do not touch the serializer and
// always run.
bool supports_root_signature_1_2() {
    ID3D12Device* device = nullptr;
    HRESULT hr = D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&device));
    if (FAILED(hr) || !device) return false;
    D3D12_FEATURE_DATA_ROOT_SIGNATURE feature{D3D_ROOT_SIGNATURE_VERSION_1_2};
    hr = device->CheckFeatureSupport(D3D12_FEATURE_ROOT_SIGNATURE, &feature, sizeof(feature));
    device->Release();
    return SUCCEEDED(hr) && feature.HighestVersion >= D3D_ROOT_SIGNATURE_VERSION_1_2;
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

void test_versioned_1_2(bool supports_1_2) {
    D3D12_VERSIONED_ROOT_SIGNATURE_DESC xbox{};
    xbox.Version = D3D_ROOT_SIGNATURE_VERSION_1_2;
    xbox.Desc_1_2 = make_legacy_desc2(kXboxFlags);

    // Byte-level: the helper recognises 1.2 and clears only the high byte.
    // These never touch the PC serializer, so they run regardless of runtime.
    D3D12_VERSIONED_ROOT_SIGNATURE_DESC cleared{};
    bool known = rt::d3d12_xs::copy_versioned_root_signature_desc_clear_xbox_flags(xbox, cleared);
    check(known, "versioned 1.2 descriptor version is recognised");
    check(cleared.Desc_1_2.Flags == 0, "versioned 1.2 copy has the high flag byte cleared");
    check(xbox.Desc_1_2.Flags == static_cast<D3D12_ROOT_SIGNATURE_FLAGS>(kXboxFlags),
          "versioned 1.2 original descriptor is unchanged");

    if (!supports_1_2) {
        std::printf("SKIP versioned 1.2 serializer checks (runtime HighestVersion < 1.2)\n");
        return;
    }

    {
        Blobs blobs;
        HRESULT hr = D3D12SerializeVersionedRootSignature(&xbox, &blobs.blob, &blobs.error);
        check(FAILED(hr), "versioned 1.2 with Xbox flags 0x03000000 is rejected by the PC serializer");
    }

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

void test_registered_shim(bool supports_1_2) {
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

    if (supports_1_2) {
        D3D12_VERSIONED_ROOT_SIGNATURE_DESC xbox12{};
        xbox12.Version = D3D_ROOT_SIGNATURE_VERSION_1_2;
        xbox12.Desc_1_2 = make_legacy_desc2(kXboxFlags);

        Blobs blobs12;
        HRESULT hr12 = fn(&xbox12, &blobs12.blob, &blobs12.error);
        check(SUCCEEDED(hr12), "registered shim clears the Xbox flag byte for 1.2 and the PC serializer accepts");
        check(xbox12.Desc_1_2.Flags == static_cast<D3D12_ROOT_SIGNATURE_FLAGS>(kXboxFlags),
              "registered shim leaves the caller's 1.2 descriptor unchanged");
    } else {
        std::printf("SKIP registered shim 1.2 (runtime HighestVersion < 1.2)\n");
    }
}

void test_deserializer() {
    // A blob whose RTS0 flags carry the Xbox high byte is rejected by the host
    // deserializer; the shim clears the byte on a copy and succeeds.
    D3D12_ROOT_SIGNATURE_DESC desc = make_legacy_desc(0);
    Blobs blobs;
    HRESULT hr = D3D12SerializeRootSignature(&desc, D3D_ROOT_SIGNATURE_VERSION_1_0, &blobs.blob, &blobs.error);
    check(SUCCEEDED(hr), "serialize a clean legacy root signature for the deserializer test");
    if (FAILED(hr) || !blobs.blob) return;

    const uint8_t* bytes = static_cast<const uint8_t*>(blobs.blob->GetBufferPointer());
    const SIZE_T size = blobs.blob->GetBufferSize();

    std::vector<uint8_t> tampered(bytes, bytes + size);
    uint32_t chunk_count = 0;
    std::memcpy(&chunk_count, bytes + 28, sizeof(chunk_count));
    bool found_rts0 = false;
    for (uint32_t i = 0; i < chunk_count; ++i) {
        uint32_t off = 0;
        std::memcpy(&off, bytes + 32 + i * 4, sizeof(off));
        if (off > size || size - off < 8 + 24) continue;
        if (std::memcmp(bytes + off, "RTS0", 4) != 0) continue;
        uint32_t flags = 0;
        std::memcpy(&flags, bytes + off + 8 + 20, sizeof(flags));
        check(flags == 0, "clean RTS0 flags are zero before tampering");
        std::memcpy(tampered.data() + off + 8 + 20, &kXboxFlags, sizeof(kXboxFlags));
        found_rts0 = true;
    }
    check(found_rts0, "RTS0 chunk found in the serialized blob");
    if (!found_rts0) return;

    std::vector<uint8_t> cleared;
    const bool rewritten = rt::d3d12_xs::copy_rts0_blob_clear_xbox_flags(tampered.data(), tampered.size(), cleared);
    check(rewritten, "blob helper reports a rewrite for the Xbox high byte");
    if (rewritten) {
        uint32_t cleared_flags = 0;
        for (uint32_t i = 0; i < chunk_count; ++i) {
            uint32_t off = 0;
            std::memcpy(&off, cleared.data() + 32 + i * 4, sizeof(off));
            if (off > cleared.size() || cleared.size() - off < 8 + 24) continue;
            if (std::memcmp(cleared.data() + off, "RTS0", 4) == 0) {
                std::memcpy(&cleared_flags, cleared.data() + off + 8 + 20, sizeof(cleared_flags));
            }
        }
        check(cleared_flags == 0, "blob helper cleared the Xbox high byte");
    }

    {
        ID3D12RootSignatureDeserializer* direct = nullptr;
        hr = D3D12CreateRootSignatureDeserializer(tampered.data(), tampered.size(), IID_PPV_ARGS(&direct));
        check(FAILED(hr), "host deserializer rejects the Xbox high byte");
    }

    using DeserializeFn = HRESULT(WINAPI*)(const void*, SIZE_T, REFIID, void**);
    auto deserialize = reinterpret_cast<DeserializeFn>(
        rt::shims::find("d3d12_xs.dll", "D3D12CreateRootSignatureDeserializer"));
    check(deserialize != nullptr, "shim registry exposes d3d12_xs!D3D12CreateRootSignatureDeserializer");
    if (!deserialize) return;

    ID3D12RootSignatureDeserializer* shim_de = nullptr;
    hr = deserialize(tampered.data(), tampered.size(), IID_PPV_ARGS(&shim_de));
    check(SUCCEEDED(hr), "shim deserializer clears the Xbox high byte and succeeds");
    if (SUCCEEDED(hr) && shim_de) {
        const D3D12_ROOT_SIGNATURE_DESC* d = shim_de->GetRootSignatureDesc();
        check(d->Flags == 0, "shim deserializer returns flags with the high byte cleared");
        shim_de->Release();
    }
}

}  // namespace

int main() {
    std::printf("root_sig_test: PC d3d12.dll root-signature flag handling\n");
    const bool supports_1_2 = supports_root_signature_1_2();
    test_versioned_1_0();
    test_versioned_1_1();
    test_versioned_1_2(supports_1_2);
    test_legacy_entry_point();
    test_preserves_low_bits_and_unknown_versions();
    test_registered_shim(supports_1_2);
    test_deserializer();
    std::printf("root_sig_test: %s (%d failure%s)\n", g_failures ? "FAILED" : "OK", g_failures,
                g_failures == 1 ? "" : "s");
    return g_failures ? 1 : 0;
}
