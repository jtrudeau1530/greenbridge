// Portable tests for runtime/src/root_signature.{h,cpp} (no windows.h).
//
// Own main + PASS/FAIL lines like the other runtime tests; returns nonzero
// on any failure so CTest gates on it. Everything here is hand-built: no
// game data, no DXBC, no serializer needed.
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

rt::rootsig::RootParameter consts(uint32_t n) {
    rt::rootsig::RootParameter p;
    p.type = rt::rootsig::ParamType::Constants;
    p.num_values = n;
    return p;
}

rt::rootsig::RootParameter view() {
    rt::rootsig::RootParameter p;
    p.type = rt::rootsig::ParamType::Descriptor;
    return p;
}

rt::rootsig::RootParameter table() {
    rt::rootsig::RootParameter p;
    p.type = rt::rootsig::ParamType::Table;
    p.ranges.push_back({0, 8});
    return p;
}

void put_u32(std::vector<uint8_t>& out, uint32_t v) {
    out.push_back(static_cast<uint8_t>(v));
    out.push_back(static_cast<uint8_t>(v >> 8));
    out.push_back(static_cast<uint8_t>(v >> 16));
    out.push_back(static_cast<uint8_t>(v >> 24));
}

}  // namespace

int main() {
    using namespace rt::rootsig;

    // Mixed signature: 4 constants + CBV + table + 1 static sampler.
    {
        RootSignature sig;
        sig.params = {consts(4), view(), table()};
        StaticSampler s;
        s.shader_register = 3;
        sig.static_samplers = {s};
        const PackResult r = pack(sig);
        check(r.ok, "mixed signature packs");
        check(r.dword_count == 8, "mixed signature uses 8 dwords");
        check(r.d3d12_cost == 4 + 2 + 1, "mixed signature costs 7");
        check(r.provenance.size() == 8 && r.provenance[0].param_index == 0 &&
                  r.provenance[0].sub_index == 0 && r.provenance[4].param_index == 1 &&
                  r.provenance[4].sub_index == 0 && r.provenance[5].sub_index == 1 &&
                  r.provenance[6].param_index == 2,
              "provenance tracks param and sub-dword");
    }

    // Exactly 32 dwords still fits.
    {
        RootSignature sig;
        for (int i = 0; i < 16; ++i) sig.params.push_back(view());
        const PackResult r = pack(sig);
        check(r.ok && r.dword_count == 32, "16 views fill 32 dwords");
    }

    // 34 dwords: spill is refused loudly, never truncated.
    {
        RootSignature sig;
        for (int i = 0; i < 17; ++i) sig.params.push_back(view());
        const PackResult r = pack(sig);
        check(!r.ok && r.error.find("spill") != std::string::npos,
              "17 views refused as spill");
    }

    // 65 constants exceed the 64-dword D3D12 cost limit.
    {
        RootSignature sig;
        sig.params = {consts(65)};
        const PackResult r = pack(sig);
        check(!r.ok && r.error.find("64-dword") != std::string::npos,
              "65 constants refused by cost limit");
    }

    // Canonical codec round-trips byte-exactly, sampler included.
    {
        RootSignature sig;
        sig.version = Version::V1_0;
        sig.params = {consts(2), table()};
        sig.flags = 0x03000000u;  // Xbox high byte survives the model
        StaticSampler s;
        s.filter = 21;
        s.mip_lod_bias = 0.5f;
        sig.static_samplers = {s};
        const std::vector<uint8_t> bytes = encode_canonical(sig);
        const DecodeResult d = decode_canonical(bytes.data(), bytes.size());
        check(d.ok, "canonical round-trip decodes");
        check(d.sig.params.size() == 2 && d.sig.params[0].num_values == 2 &&
                  d.sig.params[1].ranges.size() == 1 &&
                  d.sig.static_samplers.size() == 1 && d.sig.flags == 0x03000000u &&
                  d.sig.static_samplers[0].filter == 21,
              "canonical round-trip preserves model");
    }

    // Canonical decoder controls: bad magic, truncation, trailing bytes.
    {
        const std::vector<uint8_t> bad = {0, 1, 2, 3};
        check(!decode_canonical(bad.data(), bad.size()).ok, "bad magic rejected");
        RootSignature sig;
        sig.params = {consts(1)};
        const std::vector<uint8_t> bytes = encode_canonical(sig);
        check(!decode_canonical(bytes.data(), bytes.size() - 1).ok,
              "truncated canonical rejected");
        std::vector<uint8_t> trailed = bytes;
        trailed.push_back(0);
        check(!decode_canonical(trailed.data(), trailed.size()).ok,
              "trailing bytes rejected");
    }

    // RTS0 hypothesis: a hand-built blob per the documented layout parses.
    {
        std::vector<uint8_t> blob;
        put_u32(blob, 2);  // version 1.1
        put_u32(blob, 2);  // 2 params
        put_u32(blob, 0);  // 0 samplers
        put_u32(blob, 0x03000000u);  // Xbox flags
        put_u32(blob, 0);  // Constants
        put_u32(blob, 0);  // visibility All
        put_u32(blob, 7);  // register (ignored by packer)
        put_u32(blob, 0);  // space
        put_u32(blob, 3);  // 3 values
        put_u32(blob, 1);  // Descriptor
        put_u32(blob, 4);  // visibility Pixel
        put_u32(blob, 0);  // register
        put_u32(blob, 0);  // space
        const DecodeResult d = parse_rts0(blob.data(), blob.size());
        check(d.ok, "hand-built RTS0 parses");
        check(d.sig.params.size() == 2 && d.sig.params[0].num_values == 3 &&
                  d.sig.flags == 0x03000000u,
              "RTS0 model matches hand-built blob");
        const PackResult r = pack(d.sig);
        check(r.ok && r.dword_count == 3 + 2, "parsed RTS0 packs to 5 dwords");
    }

    // RTS0 controls: unknown version and truncation fail loudly.
    {
        std::vector<uint8_t> blob;
        put_u32(blob, 9);
        put_u32(blob, 0);
        put_u32(blob, 0);
        put_u32(blob, 0);
        const DecodeResult d = parse_rts0(blob.data(), blob.size());
        check(!d.ok, "unknown RTS0 version rejected");
        std::vector<uint8_t> cut = {1, 0};
        check(!parse_rts0(cut.data(), cut.size()).ok, "truncated RTS0 rejected");
    }

    if (g_failures != 0) std::printf("%d FAILURES\n", g_failures);
    return g_failures == 0 ? 0 : 1;
}
