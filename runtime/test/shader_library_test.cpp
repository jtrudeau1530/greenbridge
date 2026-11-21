// Shader library pieces without a GPU: SHA-256 (FIPS 180-4 vectors), zlib inflate (stored, fixed and dynamic
// Huffman blocks, compressed from synthetic data with Python's zlib), the JSON reader, FSSM extraction from a
// synthetic DXBC container, and index lookup by content hash. No game data.
#include "../src/shader_library.h"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#define CHECK(condition) do {                                    \
    if (!(condition)) {                                          \
        std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); \
        return 1;                                                \
    }                                                            \
} while (0)

namespace {

std::vector<std::uint8_t> bytes(const std::string& s) { return {s.begin(), s.end()}; }

// The dynamic-Huffman sample: an LCG over a skewed alphabet (the same generator produced `dynamic` below).
std::vector<std::uint8_t> skewed(std::size_t n) {
    static constexpr char alphabet[] = "eeeeettaaoinshrdlu";
    std::vector<std::uint8_t> out;
    std::uint32_t x = 12345;
    for (std::size_t i = 0; i < n; ++i) {
        x = (x * 1103515245u + 12345u) & 0x7fffffffu;
        out.push_back(std::uint8_t(alphabet[(x >> 16) % 18]));
    }
    return out;
}

const std::vector<std::uint8_t> stored = {
    0x78, 0x01, 0x01, 0x28, 0x00, 0xd7, 0xff, 0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b,
    0x0c, 0x0d, 0x0e, 0x0f, 0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18, 0x19, 0x1a, 0x1b, 0x1c, 0x1d, 0x1e,
    0x1f, 0x20, 0x21, 0x22, 0x23, 0x24, 0x25, 0x26, 0x27, 0x29, 0xcc, 0x03, 0x0d};
const std::vector<std::uint8_t> fixed = {0x78, 0x01, 0xcb, 0x48, 0xcd, 0xc9, 0xc9, 0x57, 0x40, 0x26, 0x73,
                                         0x52, 0xf3, 0xf3, 0x32, 0x53, 0x12, 0x01, 0x7a, 0xaf, 0x09, 0x79};
const std::vector<std::uint8_t> dynamic = {
    0x78, 0xda, 0x1d, 0x92, 0xdb, 0x0a, 0xc0, 0x30, 0x08, 0x43, 0xbf, 0x55, 0x30, 0xa0, 0x20, 0x0a, 0x5e, 0xfe, 0x7f,
    0xe9, 0xf6, 0x54, 0x66, 0x6c, 0x4e, 0xb4, 0x13, 0xb6, 0x22, 0x26, 0x40, 0x08, 0xaa, 0xc0, 0xc3, 0x40, 0x0e, 0x17,
    0x1e, 0xe5, 0x09, 0x5f, 0x44, 0xc9, 0xab, 0x01, 0x10, 0x71, 0xa4, 0x40, 0x25, 0x25, 0xa2, 0xb1, 0x28, 0x31, 0x44,
    0x88, 0xcb, 0x62, 0x7c, 0x29, 0xda, 0x38, 0x43, 0x52, 0x78, 0xd2, 0x5b, 0x23, 0x47, 0x99, 0xb8, 0x61, 0xd8, 0x05,
    0x1f, 0xa4, 0x42, 0xf3, 0x26, 0x84, 0x17, 0xe4, 0xf0, 0x5e, 0x83, 0x81, 0x92, 0x35, 0x7a, 0x05, 0x2d, 0x5a, 0xce,
    0xb1, 0x3d, 0xab, 0xa5, 0xca, 0x0a, 0x95, 0x7b, 0x43, 0x11, 0xdd, 0xe0, 0x69, 0xc3, 0x5b, 0x37, 0x92, 0x2c, 0xd7,
    0x79, 0xd6, 0x92, 0xaf, 0x20, 0x43, 0x73, 0x61, 0x1f, 0x13, 0xcc, 0xa1, 0x55, 0xd9, 0x47, 0xd3, 0x89, 0x55, 0x6a,
    0x01, 0x06, 0x9a, 0x2e, 0x8b, 0x30, 0xba, 0x63, 0xaf, 0x56, 0xf7, 0x3c, 0x49, 0x0e, 0xd4, 0x90, 0xae, 0x2f, 0x65,
    0x70, 0xb7, 0xd9, 0xe7, 0x32, 0xcf, 0xef, 0x18, 0x3d, 0xd8, 0x59, 0xde, 0xc1, 0xf4, 0x63, 0xb2, 0x4b, 0x90, 0xb1,
    0x54, 0xff, 0x87, 0xa0, 0x53, 0x68, 0xdf, 0x57, 0xcc, 0x9c, 0x79, 0x3e, 0x8b, 0x74, 0xb8, 0xef, 0xac, 0x33, 0x25,
    0xc7, 0x44, 0x8c, 0x95, 0x29, 0x3d, 0xfb, 0x23, 0xee, 0xcb, 0xc8, 0x51, 0x48, 0x50, 0x4e, 0x93, 0x78, 0x38, 0xc4,
    0xbf, 0x67, 0x48, 0x8c, 0x37, 0x53, 0xb2, 0x5e, 0xa3, 0x83, 0xfc, 0x19, 0x30, 0xe5, 0x3f, 0xa1, 0x51, 0xb1, 0xe0,
    0x24, 0x8a, 0xb7, 0x20, 0x8d, 0xda, 0x79, 0xcb, 0xb2, 0xb7, 0x3a, 0x7e, 0xc9, 0xc4, 0x69, 0x5a, 0x06, 0x3a, 0xc5,
    0xfd, 0x33, 0x01, 0x33, 0x72, 0xb9, 0xc9, 0x09, 0x44, 0x38, 0xa4, 0xeb, 0xf7, 0x27, 0x6a, 0xe5, 0x11, 0x61, 0x99,
    0xe1, 0xa2, 0xc2, 0x75, 0x8d, 0x0d, 0xf6, 0x1e, 0x40, 0x6c, 0xeb, 0xbc, 0xfd, 0x73, 0xf1, 0x74, 0x81, 0x31, 0x73,
    0xf2, 0x81, 0x64, 0xff, 0xfb, 0x27, 0x60, 0xbf, 0xd1, 0x31, 0x02, 0xfe, 0x78, 0x7c, 0x3d, 0x44, 0x17, 0x72, 0x10,
    0xd3, 0xe8, 0xe7, 0x36, 0xe7, 0x77, 0x1f, 0x7a, 0xbd, 0xf9, 0x80};

void put32(std::vector<std::uint8_t>& out, std::uint32_t v) {
    for (int i = 0; i < 4; ++i) out.push_back(std::uint8_t(v >> (8 * i)));
}

// A shader container shaped like the game's: DXBC with a DXIL part holding an XDAT table whose 0x0D entry is
// XBCH + a zlib stream (tools/fssm_unpack.py documents the layout).
std::vector<std::uint8_t> container(const std::vector<std::uint8_t>& zlib, std::uint32_t unpacked) {
    std::vector<std::uint8_t> xbch = {'X', 'B', 'C', 'H'};
    put32(xbch, 1);
    put32(xbch, std::uint32_t(zlib.size()));
    put32(xbch, 0);
    put32(xbch, unpacked);
    put32(xbch, 0);
    xbch.insert(xbch.end(), zlib.begin(), zlib.end());
    std::vector<std::uint8_t> dxil(16, 0);  // stands in for the DxilProgramHeader stub
    const std::uint32_t xdat_at = std::uint32_t(dxil.size());
    dxil.insert(dxil.end(), {'X', 'D', 'A', 'T'});
    put32(dxil, 2);   // entries
    put32(dxil, 12);  // header size
    const std::uint32_t first = xdat_at + 12 + 2 * 12;
    put32(dxil, 0x12);  // a 16-byte hash entry, ignored
    put32(dxil, first);
    put32(dxil, 16);
    put32(dxil, 0x0d);
    put32(dxil, first + 16);
    put32(dxil, std::uint32_t(xbch.size()));
    dxil.insert(dxil.end(), 16, 0xab);
    dxil.insert(dxil.end(), xbch.begin(), xbch.end());

    std::vector<std::uint8_t> out = {'D', 'X', 'B', 'C'};
    out.resize(28, 0);  // hash, version, total size (unchecked)
    put32(out, 2);       // part count
    put32(out, 40);      // PRIV part offset
    put32(out, 40 + 8 + 4);
    out.insert(out.end(), {'P', 'R', 'I', 'V'});
    put32(out, 4);
    put32(out, 0);
    out.insert(out.end(), {'D', 'X', 'I', 'L'});
    put32(out, std::uint32_t(dxil.size()));
    out.insert(out.end(), dxil.begin(), dxil.end());
    return out;
}

}  // namespace

int main() {
    using namespace leonida;

    CHECK(sha256_hex({}) == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
    CHECK(sha256_hex(bytes("abc")) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    CHECK(sha256_hex(bytes("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq")) ==
          "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1");

    std::vector<std::uint8_t> out;
    CHECK(zlib_inflate(stored, out) && out.size() == 40);
    for (std::uint8_t i = 0; i < 40; ++i) CHECK(out[i] == i);
    CHECK(zlib_inflate(fixed, out) && out == bytes("hello hello hello leonida"));
    CHECK(zlib_inflate(dynamic, out) && out == skewed(600));
    CHECK(sha256_hex(out) == "9cd6848f765bff2a9f42a8b42f6164042dad67d3e9f82e0044ddc7d1ecdeebb2");
    auto broken = dynamic;
    broken.back() ^= 1;  // Adler-32
    CHECK(!zlib_inflate(broken, out));
    broken = dynamic;
    broken.resize(100);  // truncated
    CHECK(!zlib_inflate(broken, out));
    CHECK(!zlib_inflate(dynamic, out, 599));  // larger than the limit
    CHECK(!zlib_inflate(bytes("not zlib"), out));

    Json json;
    // "s" holds an escaped quote and a \u escape for U+00E9, which the reader must turn into UTF-8.
    const std::string text = "{\"a\": {\"stage\": \"cs\", \"n\": [1, 2.5, -3e2], \"ok\": true, "
                             "\"s\": \"x\\\"\\u00e9\", \"z\": null}}";
    const std::string expected_s = std::string("x\"") + char(0xc3) + char(0xa9);
    CHECK(JsonReader(text).parse(json));
    const Json* a = json.find("a");
    CHECK(a && a->find("stage")->string == "cs" && a->find("n")->array[2].number == -300 && a->find("ok")->boolean);
    CHECK(a->find("s")->string == expected_s && a->find("z")->kind == Json::Kind::null);
    CHECK(!JsonReader("{\"a\": }").parse(json));
    CHECK(!JsonReader("[1, 2").parse(json));
    CHECK(!JsonReader("{} extra").parse(json));

    const auto shader = container(dynamic, 600);
    CHECK(dxbc_part(shader, "PRIV").size() == 4 && !dxbc_part(shader, "DXIL").empty() && dxbc_part(shader, "RTS0").empty());
    CHECK(fssm_program(shader) == skewed(600));
    auto fssm = bytes("FSSM and a native program");
    CHECK(fssm_program(fssm) == fssm);  // a bare FSSM blob passes through
    CHECK(fssm_program(container(dynamic, 601)).empty());  // stated size wrong
    CHECK(fssm_program(bytes("DXBC")).empty());

    // An index with one findable shader, one without a hash, and the module next to it.
    const auto dir = std::filesystem::temp_directory_path() / "leonida_shader_library_test";
    std::filesystem::create_directories(dir);
    const std::string hash = sha256_hex(skewed(600));
    {
        std::ofstream index(dir / "index.json", std::ios::binary);
        index << "{\"cs_test\": {\"stage\": \"cs\", \"fssm_sha256\": \"" << hash
              << "\", \"bindings\": [2], \"push_constant_bytes\": 144, \"local_size\": [64, 1, 1], "
                 "\"uses_code_base\": true, \"wave64\": false},\n"
                 " \"ps_unhashed\": {\"stage\": \"ps\", \"bindings\": []}}";
        std::ofstream spv(dir / "cs_test.spv", std::ios::binary);
        const std::uint32_t words[5] = {0x07230203, 0x00010500, 0, 1, 0};
        spv.write(reinterpret_cast<const char*>(words), sizeof(words));
    }
    ShaderLibrary library;
    std::string error;
    CHECK(library.load(dir.string(), &error) && library.size() == 1);
    const ShaderEntry* entry = library.find("cs", sha256_hex(fssm_program(shader)));
    CHECK(entry && entry->name == "cs_test" && entry->binding_count == 1 && entry->push_constant_bytes == 144);
    CHECK(entry->local_size[0] == 64 && entry->uses_code_base && !entry->wave64);
    CHECK(!library.find("ps", hash));
    CHECK(library.spirv(*entry).size() == 5);
    CHECK(!library.load((dir / "missing").string(), &error) && !error.empty());
    std::filesystem::remove_all(dir);

    std::printf("SHA-256, inflate (stored/fixed/dynamic), JSON, FSSM extraction and index lookup OK\n");
    return 0;
}
