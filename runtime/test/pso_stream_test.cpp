// Pipeline state stream copies and the derived-state patch (device slot 55) with synthetic streams only: no D3D12
// device, no game data. The layout rule is the one measured on the console's slot 86 stream (see the offsets check).
#include <d3d12.h>

#include <atomic>
#include <barrier>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <memory>
#include <thread>
#include <vector>

#include "d3d12_xs_pso_stream.h"

namespace {

int g_failures = 0;

void check(bool ok, const char* what) {
    std::printf("%s %s\n", ok ? "PASS" : "FAIL", what);
    if (!ok) ++g_failures;
}

std::size_t align_up(std::size_t v, std::size_t a) { return (v + a - 1) / a * a; }

// Builds a stream the way the console lays it out: u32 type, payload at its natural alignment, next entry 8-aligned.
struct StreamBuilder {
    std::vector<unsigned char> bytes;
    void entry(unsigned type, const void* payload, std::size_t size, std::size_t payload_align) {
        const std::size_t at = bytes.size();
        bytes.resize(at + 4);
        std::memcpy(bytes.data() + at, &type, 4);
        const std::size_t pay = align_up(at + 4, payload_align);
        bytes.resize(pay, 0);
        const auto* p = static_cast<const unsigned char*>(payload);
        bytes.insert(bytes.end(), p, p + size);
        bytes.resize(align_up(bytes.size(), 8), 0);
    }
};

// The test's own copy of the layout rule: payload offset of the first subobject of `wanted`, or npos.
std::size_t payload_of_type(const std::vector<unsigned char>& s, unsigned wanted) {
    std::size_t off = 0;
    while (off + 4 <= s.size()) {
        unsigned type;
        std::memcpy(&type, s.data() + off, 4);
        std::size_t size = sizeof(D3D12_SHADER_BYTECODE), align = 8;
        if (type == D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_BLEND) { size = sizeof(D3D12_BLEND_DESC); align = 4; }
        else if (type == D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_SAMPLE_MASK) { size = 4; align = 4; }
        else if (type == D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RASTERIZER) { size = sizeof(D3D12_RASTERIZER_DESC); align = 4; }
        else if (type == D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL1) { size = sizeof(D3D12_DEPTH_STENCIL_DESC1); align = 4; }
        else if (type == D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_ROOT_SIGNATURE) { size = 8; align = 8; }
        else if (type == D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_INPUT_LAYOUT) { size = sizeof(D3D12_INPUT_LAYOUT_DESC); align = 8; }
        const std::size_t pay = align_up(off + 4, align);
        if (type == wanted) return pay;
        off = align_up(pay + size, 8);
    }
    return static_cast<std::size_t>(-1);
}

const unsigned char kVsCode[32] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
                                   17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32};
const char kSemantic[] = "POSITION";

D3D12_INPUT_ELEMENT_DESC make_element(const char* name) {
    D3D12_INPUT_ELEMENT_DESC e{};
    e.SemanticName = name;
    e.Format = DXGI_FORMAT_R32G32B32_FLOAT;
    return e;
}

// Root signature (null), VS, BLEND, SAMPLE_MASK, RASTERIZER, DEPTH_STENCIL1, INPUT_LAYOUT.
std::vector<unsigned char> base_stream(const D3D12_INPUT_ELEMENT_DESC* elements, unsigned element_count) {
    StreamBuilder s;
    ID3D12RootSignature* root = nullptr;
    s.entry(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_ROOT_SIGNATURE, &root, 8, 8);
    D3D12_SHADER_BYTECODE vs{kVsCode, sizeof(kVsCode)};
    s.entry(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_VS, &vs, sizeof(vs), 8);
    D3D12_BLEND_DESC blend{};
    s.entry(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_BLEND, &blend, sizeof(blend), 4);
    UINT mask = 0xffffffffu;
    s.entry(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_SAMPLE_MASK, &mask, 4, 4);
    D3D12_RASTERIZER_DESC raster{};
    raster.FillMode = D3D12_FILL_MODE_SOLID;
    s.entry(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RASTERIZER, &raster, sizeof(raster), 4);
    D3D12_DEPTH_STENCIL_DESC1 ds1{};
    ds1.DepthEnable = TRUE;
    s.entry(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL1, &ds1, sizeof(ds1), 4);
    D3D12_INPUT_LAYOUT_DESC layout{elements, element_count};
    s.entry(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_INPUT_LAYOUT, &layout, sizeof(layout), 8);
    return s.bytes;
}

// One derived record: tag at +0, payload at +4.
std::vector<unsigned char> record(unsigned tag, const void* payload, std::size_t size) {
    std::vector<unsigned char> r(rt::d3d12_xs::kDerivedRecordSize, 0);
    std::memcpy(r.data(), &tag, 4);
    if (size) std::memcpy(r.data() + 4, payload, size);
    return r;
}

void test_layout_matches_console_offsets() {
    // Console slot 86 stream: types 0..5 then BLEND at +0x88 (payload +0x8c), SAMPLE_MASK at +0x1d8, RASTERIZER at
    // +0x1e0, DEPTH_STENCIL1 at +0x210 (measured in the notes, mesure sur l'exe).
    StreamBuilder s;
    ID3D12RootSignature* root = nullptr;
    s.entry(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_ROOT_SIGNATURE, &root, 8, 8);
    D3D12_SHADER_BYTECODE sb{};
    for (unsigned i = 0; i < 5; ++i) s.entry(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_VS + i, &sb, sizeof(sb), 8);
    check(s.bytes.size() == 0x88, "six entries (types 0..5) end at +0x88");
    D3D12_BLEND_DESC blend{};
    s.entry(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_BLEND, &blend, sizeof(blend), 4);
    check(payload_of_type(s.bytes, D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_BLEND) == 0x8c,
          "BLEND payload at +0x8c");
    check(s.bytes.size() == 0x1d8, "BLEND entry ends at +0x1d8");
    UINT mask = 1;
    s.entry(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_SAMPLE_MASK, &mask, 4, 4);
    check(s.bytes.size() == 0x1e0, "SAMPLE_MASK entry is 8 bytes, next entry at +0x1e0");
}

void test_capture_and_deep_copy() {
    D3D12_INPUT_ELEMENT_DESC elements[1] = {make_element(kSemantic)};
    std::vector<unsigned char> base = base_stream(elements, 1);
    D3D12_PIPELINE_STATE_STREAM_DESC desc{base.size(), base.data()};
    auto rec = rt::d3d12_xs::PsoStreamRecord::capture(desc);
    check(rec != nullptr, "a synthetic base stream is captured");
    if (!rec) return;
    check(rec->bytes().size() == base.size(), "the copy has the stream's size");

    // Scribble over the source and its pointed-to data: the copy must still hold the original content.
    std::memset(base.data(), 0xee, base.size());
    elements[0].SemanticName = "XXXXXXXX";
    D3D12_SHADER_BYTECODE vs_copy{};
    std::memcpy(&vs_copy, rec->bytes().data() + payload_of_type(rec->bytes(), D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_VS),
                sizeof(vs_copy));
    check(vs_copy.pShaderBytecode != kVsCode && vs_copy.BytecodeLength == sizeof(kVsCode) &&
              std::memcmp(vs_copy.pShaderBytecode, kVsCode, sizeof(kVsCode)) == 0,
          "shader bytecode is copied, not referenced");
    D3D12_INPUT_LAYOUT_DESC il{};
    std::memcpy(&il, rec->bytes().data() + payload_of_type(rec->bytes(), D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_INPUT_LAYOUT),
                sizeof(il));
    check(il.NumElements == 1 && std::strcmp(il.pInputElementDescs[0].SemanticName, kSemantic) == 0,
          "input layout semantic names are copied");
}

void test_derive_single_and_all_tags() {
    D3D12_INPUT_ELEMENT_DESC elements[1] = {make_element(kSemantic)};
    std::vector<unsigned char> base = base_stream(elements, 1);
    auto rec = rt::d3d12_xs::PsoStreamRecord::capture({base.size(), base.data()});
    if (!rec) { check(false, "base captured for derive tests"); return; }

    D3D12_BLEND_DESC blend{};
    blend.RenderTarget[0].RenderTargetWriteMask = 0x5;
    auto r2 = record(2, &blend, sizeof(blend));
    std::vector<unsigned char> out;
    check(rt::d3d12_xs::derive_stream(*rec, r2.data(), 1, out) == rt::d3d12_xs::DeriveStatus::ok,
          "tag 2 (blend) derives");
    const std::size_t bp = payload_of_type(out, D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_BLEND);
    check(std::memcmp(out.data() + bp, &blend, sizeof(blend)) == 0, "tag 2 payload lands in the BLEND subobject");
    check(out.size() == rec->bytes().size(), "derived stream keeps the base size");

    D3D12_RASTERIZER_DESC raster{};
    raster.FillMode = D3D12_FILL_MODE_WIREFRAME;
    D3D12_DEPTH_STENCIL_DESC ds{};
    ds.DepthEnable = FALSE;
    UINT mask = 0x3;
    std::vector<unsigned char> three;
    auto r0 = record(0, &ds, sizeof(ds));
    auto r1 = record(1, &raster, sizeof(raster));
    auto r3 = record(3, &mask, 4);
    three.insert(three.end(), r0.begin(), r0.end());
    three.insert(three.end(), r1.begin(), r1.end());
    three.insert(three.end(), r3.begin(), r3.end());
    check(rt::d3d12_xs::derive_stream(*rec, three.data(), 3, out) == rt::d3d12_xs::DeriveStatus::ok,
          "three records (tags 0, 1, 3) derive");
    const std::size_t rp = payload_of_type(out, D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RASTERIZER);
    const std::size_t mp = payload_of_type(out, D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_SAMPLE_MASK);
    const std::size_t dp = payload_of_type(out, D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL1);
    check(std::memcmp(out.data() + rp, &raster, sizeof(raster)) == 0, "tag 1 payload lands in RASTERIZER");
    check(std::memcmp(out.data() + mp, &mask, 4) == 0, "tag 3 payload lands in SAMPLE_MASK");
    check(std::memcmp(out.data() + dp, &ds, sizeof(ds)) == 0, "tag 0 patches the DS1 prefix");
    const std::size_t tail = dp + sizeof(ds);
    check(out[tail] == 0 && out[tail + 1] == 0 && out[tail + 2] == 0 && out[tail + 3] == 0,
          "tag 0 keeps the DS1 depth-bounds field of the base");
}

void test_stencil_mask_record() {
    // Tag 5 carries the two stencil mask bytes at the DEPTH_STENCIL offsets 0x10/0x11 and zeros elsewhere (read from the
    // title's writes). It overlays only those two bytes of the base's DEPTH_STENCIL (or DEPTH_STENCIL1) payload.
    D3D12_INPUT_ELEMENT_DESC elements[1] = {make_element(kSemantic)};
    std::vector<unsigned char> base = base_stream(elements, 1);
    auto rec = rt::d3d12_xs::PsoStreamRecord::capture({base.size(), base.data()});
    if (!rec) { check(false, "base captured for the tag 5 tests"); return; }
    const std::size_t dp = payload_of_type(rec->bytes(), D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL1);

    unsigned char mask_payload[0x16] = {};
    mask_payload[0x10] = 0x0f;  // read mask
    mask_payload[0x11] = 0x3c;  // write mask
    auto r5 = record(5, mask_payload, sizeof(mask_payload));
    std::vector<unsigned char> out;
    check(rt::d3d12_xs::derive_stream(*rec, r5.data(), 1, out) == rt::d3d12_xs::DeriveStatus::ok,
          "tag 5 alone derives on a base with DEPTH_STENCIL1");
    check(out[dp + 0x10] == 0x0f && out[dp + 0x11] == 0x3c, "tag 5 overlays the two stencil mask bytes");
    bool rest_same = true;
    for (std::size_t i = 0; i < 0x38; ++i)
        if (i != 0x10 && i != 0x11 && out[dp + i] != rec->bytes()[dp + i]) rest_same = false;
    check(rest_same, "tag 5 leaves the other DEPTH_STENCIL1 bytes unchanged");

    // Order: records apply in the order they are given. Tag 0 writes its own masks, tag 5 overrides them.
    D3D12_DEPTH_STENCIL_DESC ds{};
    ds.StencilReadMask = 0x01;
    ds.StencilWriteMask = 0x02;
    auto r0 = record(0, &ds, sizeof(ds));
    std::vector<unsigned char> seq;
    seq.insert(seq.end(), r0.begin(), r0.end());
    seq.insert(seq.end(), r5.begin(), r5.end());
    check(rt::d3d12_xs::derive_stream(*rec, seq.data(), 2, out) == rt::d3d12_xs::DeriveStatus::ok,
          "tag 0 then tag 5 derive together");
    check(out[dp + 0x10] == 0x0f && out[dp + 0x11] == 0x3c, "tag 5 after tag 0 wins (record order)");

    seq.clear();
    seq.insert(seq.end(), r5.begin(), r5.end());
    seq.insert(seq.end(), r0.begin(), r0.end());
    check(rt::d3d12_xs::derive_stream(*rec, seq.data(), 2, out) == rt::d3d12_xs::DeriveStatus::ok,
          "tag 5 then tag 0 derive together");
    check(out[dp + 0x10] == 0x01 && out[dp + 0x11] == 0x02, "tag 0 after tag 5 wins (record order)");
}

void test_refusals() {
    D3D12_INPUT_ELEMENT_DESC elements[1] = {make_element(kSemantic)};
    std::vector<unsigned char> base = base_stream(elements, 1);
    auto rec = rt::d3d12_xs::PsoStreamRecord::capture({base.size(), base.data()});
    if (!rec) { check(false, "base captured for refusal tests"); return; }
    std::vector<unsigned char> out;

    auto r6 = record(6, nullptr, 0);
    check(rt::d3d12_xs::derive_stream(*rec, r6.data(), 1, out) == rt::d3d12_xs::DeriveStatus::unknown_tag,
          "tag 6 is refused (meaning not known)");

    // A base without a rasterizer subobject cannot take a tag-1 record.
    StreamBuilder s;
    ID3D12RootSignature* root = nullptr;
    s.entry(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_ROOT_SIGNATURE, &root, 8, 8);
    D3D12_BLEND_DESC blend{};
    s.entry(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_BLEND, &blend, sizeof(blend), 4);
    auto norast = rt::d3d12_xs::PsoStreamRecord::capture({s.bytes.size(), s.bytes.data()});
    if (!norast) { check(false, "stream without rasterizer captured"); return; }
    D3D12_RASTERIZER_DESC raster{};
    auto r1 = record(1, &raster, sizeof(raster));
    check(rt::d3d12_xs::derive_stream(*norast, r1.data(), 1, out) == rt::d3d12_xs::DeriveStatus::missing_state,
          "tag 1 on a base without RASTERIZER is refused");

    // A subobject type this module does not walk: VIEW_INSTANCING (payload with a pointer, not modelled).
    StreamBuilder v;
    unsigned char junk[32] = {};
    v.entry(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_VIEW_INSTANCING, junk, sizeof(junk), 8);
    check(rt::d3d12_xs::PsoStreamRecord::capture({v.bytes.size(), v.bytes.data()}) == nullptr,
          "a stream with an unmodelled subobject type is not captured");
}

void test_blob_dedup() {
    D3D12_INPUT_ELEMENT_DESC elements[1] = {make_element(kSemantic)};
    std::vector<unsigned char> base = base_stream(elements, 1);

    // Two records captured from identical streams share the same interned shader bytecode and input layout.
    auto r1 = rt::d3d12_xs::PsoStreamRecord::capture({base.size(), base.data()});
    auto r2 = rt::d3d12_xs::PsoStreamRecord::capture({base.size(), base.data()});
    check(r1 && r2, "two identical streams are captured");
    if (!r1 || !r2) return;

    D3D12_SHADER_BYTECODE vs1{}, vs2{};
    std::memcpy(&vs1, r1->bytes().data() + payload_of_type(r1->bytes(), D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_VS),
                sizeof(vs1));
    std::memcpy(&vs2, r2->bytes().data() + payload_of_type(r2->bytes(), D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_VS),
                sizeof(vs2));
    check(vs1.pShaderBytecode == vs2.pShaderBytecode && vs1.pShaderBytecode != kVsCode,
          "identical shader bytecode is shared across records");

    D3D12_INPUT_LAYOUT_DESC il1{}, il2{};
    std::memcpy(&il1,
                r1->bytes().data() + payload_of_type(r1->bytes(), D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_INPUT_LAYOUT),
                sizeof(il1));
    std::memcpy(&il2,
                r2->bytes().data() + payload_of_type(r2->bytes(), D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_INPUT_LAYOUT),
                sizeof(il2));
    check(il1.pInputElementDescs == il2.pInputElementDescs &&
              il1.pInputElementDescs[0].SemanticName == il2.pInputElementDescs[0].SemanticName,
          "identical input layouts and semantic names are shared across records");

    // Releasing one record keeps the shared blob alive for the remaining record.
    r1.reset();
    check(std::memcmp(vs2.pShaderBytecode, kVsCode, sizeof(kVsCode)) == 0 &&
              std::strcmp(il2.pInputElementDescs[0].SemanticName, kSemantic) == 0,
          "shared blobs stay valid after one owning record is destroyed");
}

D3D12_SHADER_BYTECODE vs_of(const rt::d3d12_xs::PsoStreamRecord& rec) {
    D3D12_SHADER_BYTECODE vs{};
    std::memcpy(&vs, rec.bytes().data() + payload_of_type(rec.bytes(), D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_VS),
                sizeof(vs));
    return vs;
}

std::vector<unsigned char> vs_stream(const std::vector<unsigned char>& code) {
    StreamBuilder s;
    D3D12_SHADER_BYTECODE vs{code.data(), code.size()};
    s.entry(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_VS, &vs, sizeof(vs), 8);
    return s.bytes;
}

std::unique_ptr<rt::d3d12_xs::PsoStreamRecord> capture_bytes(std::vector<unsigned char>& s) {
    return rt::d3d12_xs::PsoStreamRecord::capture({s.size(), s.data()});
}

void test_distinct_blobs_do_not_alias() {
    // Same length, one byte apart (first, middle, last): each must get its own copy.
    std::vector<unsigned char> a(256, 0x11), b = a, c = a, d = a;
    b[0] ^= 1;
    c[128] ^= 1;
    d[255] ^= 1;
    std::vector<unsigned char> sa = vs_stream(a), sb = vs_stream(b), sc = vs_stream(c), sd = vs_stream(d);
    auto ra = capture_bytes(sa), rb = capture_bytes(sb), rc = capture_bytes(sc), rd = capture_bytes(sd);
    if (!ra || !rb || !rc || !rd) { check(false, "same-length variants captured"); return; }
    const void* p[4] = {vs_of(*ra).pShaderBytecode, vs_of(*rb).pShaderBytecode, vs_of(*rc).pShaderBytecode,
                        vs_of(*rd).pShaderBytecode};
    bool distinct = true;
    for (int i = 0; i < 4; ++i)
        for (int j = i + 1; j < 4; ++j) distinct = distinct && p[i] != p[j];
    check(distinct, "different bytecodes of the same length do not alias");
    check(std::memcmp(p[0], a.data(), a.size()) == 0 && std::memcmp(p[1], b.data(), b.size()) == 0 &&
              std::memcmp(p[2], c.data(), c.size()) == 0 && std::memcmp(p[3], d.data(), d.size()) == 0,
          "each same-length variant keeps its own bytes");
}

void test_stream_output_names_copied() {
    char name[] = "SV_Position";
    D3D12_SO_DECLARATION_ENTRY entry{};
    entry.SemanticName = name;
    entry.ComponentCount = 4;
    D3D12_SO_DECLARATION_ENTRY other_entry = entry;
    auto fill_padding = [](D3D12_SO_DECLARATION_ENTRY& decl, unsigned char value) {
        auto* bytes = reinterpret_cast<unsigned char*>(&decl);
        constexpr std::size_t kNameOffset = offsetof(D3D12_SO_DECLARATION_ENTRY, SemanticName);
        constexpr std::size_t kTailOffset = offsetof(D3D12_SO_DECLARATION_ENTRY, OutputSlot) + sizeof(decl.OutputSlot);
        std::memset(bytes + sizeof(decl.Stream), value, kNameOffset - sizeof(decl.Stream));
        std::memset(bytes + kTailOffset, value, sizeof(decl) - kTailOffset);
    };
    fill_padding(entry, 0xaa);
    fill_padding(other_entry, 0x55);
    UINT stride = 16;
    D3D12_STREAM_OUTPUT_DESC so{&entry, 1, &stride, 1, 0};
    StreamBuilder s, other;
    s.entry(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_STREAM_OUTPUT, &so, sizeof(so), 8);
    so.pSODeclaration = &other_entry;
    other.entry(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_STREAM_OUTPUT, &so, sizeof(so), 8);
    auto rec = capture_bytes(s.bytes);
    auto other_rec = capture_bytes(other.bytes);
    if (!rec || !other_rec) { check(false, "stream-output streams captured"); return; }
    std::memset(name, 'X', sizeof(name) - 1);
    entry.SemanticName = nullptr;
    stride = 0;
    D3D12_STREAM_OUTPUT_DESC copy{}, other_copy{};
    std::memcpy(&copy, rec->bytes().data() + 8, sizeof(copy));  // first payload: u32 type, 8-aligned
    std::memcpy(&other_copy, other_rec->bytes().data() + 8, sizeof(other_copy));
    check(copy.pSODeclaration != &entry && copy.NumEntries == 1 && copy.pSODeclaration[0].SemanticName != name &&
              std::strcmp(copy.pSODeclaration[0].SemanticName, "SV_Position") == 0 &&
              copy.pSODeclaration[0].ComponentCount == 4 && copy.pBufferStrides[0] == 16,
          "stream-output entries, their semantic names and strides are copied");
    check(copy.pSODeclaration && copy.pSODeclaration == other_copy.pSODeclaration,
          "equal stream-output entries with different padding share one declaration");
}

void test_release_prunes_interner() {
    // Every earlier test's records are gone, so nothing may stay interned.
    check(rt::d3d12_xs::interned_blob_count() == 0, "no blob stays interned once every record is destroyed");

    D3D12_INPUT_ELEMENT_DESC elements[1] = {make_element(kSemantic)};
    std::vector<unsigned char> base = base_stream(elements, 1);
    auto r1 = capture_bytes(base);
    auto r2 = capture_bytes(base);
    // VS bytecode, the semantic name and the element array: three distinct blobs for two records.
    check(r1 && r2 && rt::d3d12_xs::interned_blob_count() == 3, "two equal records intern three blobs");
    r1.reset();
    check(rt::d3d12_xs::interned_blob_count() == 3, "the blobs outlive one of their two records");
    r2.reset();
    check(rt::d3d12_xs::interned_blob_count() == 0, "the last record's release prunes its blobs");

    auto again = capture_bytes(base);
    check(again && std::memcmp(vs_of(*again).pShaderBytecode, kVsCode, sizeof(kVsCode)) == 0 &&
              rt::d3d12_xs::interned_blob_count() == 3,
          "re-capturing after a full release interns fresh copies");
}

void test_concurrent_capture_and_release() {
    // Race captures against last-owner deletion for the same bytecodes.
    constexpr int kThreads = 8;
    constexpr int kVariants = 3;
    constexpr int kIterations = 4000;
    std::vector<unsigned char> codes[kVariants], streams[kVariants];
    for (int v = 0; v < kVariants; ++v) {
        codes[v].assign(512, static_cast<unsigned char>(0x40 + v));
        streams[v] = vs_stream(codes[v]);
    }
    std::atomic<int> bad{0};
    std::barrier start(kThreads);
    std::vector<std::thread> workers;
    for (int t = 0; t < kThreads; ++t) {
        workers.emplace_back([&, t]() {
            start.arrive_and_wait();
            for (int i = 0; i < kIterations; ++i) {
                const int v = (t * 7 + i) % kVariants;
                auto rec = capture_bytes(streams[v]);
                if (!rec) { ++bad; continue; }
                const D3D12_SHADER_BYTECODE vs = vs_of(*rec);
                if (vs.BytecodeLength != codes[v].size() ||
                    std::memcmp(vs.pShaderBytecode, codes[v].data(), codes[v].size()) != 0)
                    ++bad;
                // Drop this record before the next capture so a variant can lose its last owner mid-loop.
            }
        });
    }
    for (auto& w : workers) w.join();
    check(bad.load() == 0, "concurrent capture and release keep every copy intact");
    check(rt::d3d12_xs::interned_blob_count() == 0, "concurrent capture and release leave nothing interned");
}

}  // namespace

int main() {
    std::printf("pso_stream_test: derived pipeline state streams\n");
    test_layout_matches_console_offsets();
    test_capture_and_deep_copy();
    test_derive_single_and_all_tags();
    test_stencil_mask_record();
    test_refusals();
    test_blob_dedup();
    test_distinct_blobs_do_not_alias();
    test_stream_output_names_copied();
    test_release_prunes_interner();
    test_concurrent_capture_and_release();
    std::printf("%s: %d failure(s)\n", g_failures ? "FAILED" : "OK", g_failures);
    return g_failures ? 1 : 0;
}
