#pragma once

// Finds the translated SPIR-V for a shader the game hands to D3D12 (docs/d3d12_vulkan.md section 3.6).
//
// A pipeline's bytecode is a DXBC container. Its DXIL part holds an XDAT table whose entry 0x0D is an XBCH header
// followed by a zlib stream; the stream inflates to the FSSM blob, the native RDNA2 program the translator read
// (tools/fssm_unpack.py does the same in Python). The SHA-256 of that blob is the `fssm_sha256` field the translator
// writes into `spirv/index.json`, next to `<name>.spv`. This header needs no Windows, Vulkan or third-party code:
// it carries its own inflate (RFC 1950/1951), SHA-256 (FIPS 180-4) and a minimal JSON reader.

#include <array>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iterator>
#include <map>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace leonida {

// ---------------------------------------------------------------------------------------------------------------
// zlib inflate

namespace detail {

class BitReader {
public:
    explicit BitReader(std::span<const std::uint8_t> in) : in_(in) {}

    std::uint32_t bits(int need) {
        std::uint64_t value = buffer_;
        while (count_ < need) {
            if (pos_ >= in_.size()) {
                failed_ = true;
                return 0;
            }
            value |= std::uint64_t(in_[pos_++]) << count_;
            count_ += 8;
        }
        buffer_ = value >> need;
        count_ -= need;
        return std::uint32_t(value & ((std::uint64_t(1) << need) - 1));
    }

    void align() {
        buffer_ = 0;
        count_ = 0;
    }

    bool byte(std::uint8_t& out) {
        if (pos_ >= in_.size()) return failed_ = true, false;
        out = in_[pos_++];
        return true;
    }

    std::size_t position() const { return pos_; }
    bool failed() const { return failed_; }

private:
    std::span<const std::uint8_t> in_;
    std::size_t pos_ = 0;
    std::uint64_t buffer_ = 0;
    int count_ = 0;
    bool failed_ = false;
};

struct Huffman {
    std::array<std::int16_t, 16> count{};
    std::array<std::int16_t, 288> symbol{};
};

// Canonical Huffman table from code lengths. Returns false for an over-subscribed set.
inline bool build_huffman(Huffman& h, const std::int16_t* lengths, int n) {
    h.count.fill(0);
    for (int i = 0; i < n; ++i) ++h.count[lengths[i]];
    if (h.count[0] == n) return true;  // no codes: valid only if never used
    int left = 1;
    for (int len = 1; len < 16; ++len) {
        left <<= 1;
        left -= h.count[len];
        if (left < 0) return false;
    }
    std::array<std::int16_t, 16> offsets{};
    for (int len = 1; len < 15; ++len) offsets[len + 1] = std::int16_t(offsets[len] + h.count[len]);
    for (int i = 0; i < n; ++i)
        if (lengths[i]) h.symbol[offsets[lengths[i]]++] = std::int16_t(i);
    return true;
}

inline int decode(BitReader& in, const Huffman& h) {
    int code = 0, first = 0, index = 0;
    for (int len = 1; len < 16; ++len) {
        code |= int(in.bits(1));
        const int count = h.count[len];
        if (code - count < first) return h.symbol[index + (code - first)];
        index += count;
        first += count;
        first <<= 1;
        code <<= 1;
        if (in.failed()) return -1;
    }
    return -1;
}

inline bool inflate_codes(BitReader& in, std::vector<std::uint8_t>& out, const Huffman& lengths,
                          const Huffman& distances, std::size_t limit) {
    static constexpr std::int16_t length_base[29] = {3,  4,  5,  6,  7,  8,  9,  10, 11,  13,  15,  17,  19,  23, 27,
                                                     31, 35, 43, 51, 59, 67, 83, 99, 115, 131, 163, 195, 227, 258};
    static constexpr std::int16_t length_extra[29] = {0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2,
                                                      2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0};
    static constexpr std::int16_t distance_base[30] = {1,    2,    3,    4,    5,    7,     9,     13,    17,  25,
                                                       33,   49,   65,   97,   129,  193,   257,   385,   513, 769,
                                                       1025, 1537, 2049, 3073, 4097, 6145,  8193,  12289, 16385, 24577};
    static constexpr std::int16_t distance_extra[30] = {0, 0, 0, 0, 1, 1, 2, 2,  3,  3,  4,  4,  5,  5,  6,
                                                        6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13};
    for (;;) {
        int symbol = decode(in, lengths);
        if (symbol < 0 || in.failed()) return false;
        if (symbol < 256) {
            if (out.size() >= limit) return false;
            out.push_back(std::uint8_t(symbol));
            continue;
        }
        if (symbol == 256) return true;
        symbol -= 257;
        if (symbol >= 29) return false;
        const std::size_t length = std::size_t(length_base[symbol]) + in.bits(length_extra[symbol]);
        const int d = decode(in, distances);
        if (d < 0 || d >= 30) return false;
        const std::size_t distance = std::size_t(distance_base[d]) + in.bits(distance_extra[d]);
        if (in.failed() || distance > out.size() || out.size() + length > limit) return false;
        const std::size_t from = out.size() - distance;
        for (std::size_t i = 0; i < length; ++i) out.push_back(out[from + i]);
    }
}

} // namespace detail

// Inflates a zlib stream (RFC 1950: header, deflate data, Adler-32). Returns false for a malformed stream, a
// preset dictionary, a checksum mismatch, or output beyond `limit` bytes.
inline bool zlib_inflate(std::span<const std::uint8_t> in, std::vector<std::uint8_t>& out,
                         std::size_t limit = std::size_t(1) << 30) {
    out.clear();
    if (in.size() < 6) return false;
    const unsigned cmf = in[0], flg = in[1];
    if ((cmf & 0x0f) != 8 || (cmf >> 4) > 7 || ((cmf << 8) | flg) % 31 != 0 || (flg & 0x20)) return false;
    detail::BitReader reader(in.subspan(2));
    for (bool last = false; !last;) {
        last = reader.bits(1) != 0;
        const std::uint32_t type = reader.bits(2);
        if (reader.failed()) return false;
        if (type == 0) {  // stored
            reader.align();
            std::uint8_t b[4];
            for (auto& x : b)
                if (!reader.byte(x)) return false;
            const unsigned length = b[0] | (b[1] << 8), complement = b[2] | (b[3] << 8);
            if (length != (~complement & 0xffff) || out.size() + length > limit) return false;
            for (unsigned i = 0; i < length; ++i) {
                std::uint8_t x;
                if (!reader.byte(x)) return false;
                out.push_back(x);
            }
        } else if (type == 1) {  // fixed Huffman
            static const auto tables = [] {
                std::pair<detail::Huffman, detail::Huffman> t;
                std::int16_t lengths[288];
                for (int i = 0; i < 144; ++i) lengths[i] = 8;
                for (int i = 144; i < 256; ++i) lengths[i] = 9;
                for (int i = 256; i < 280; ++i) lengths[i] = 7;
                for (int i = 280; i < 288; ++i) lengths[i] = 8;
                detail::build_huffman(t.first, lengths, 288);
                for (int i = 0; i < 30; ++i) lengths[i] = 5;
                detail::build_huffman(t.second, lengths, 30);
                return t;
            }();
            if (!detail::inflate_codes(reader, out, tables.first, tables.second, limit)) return false;
        } else if (type == 2) {  // dynamic Huffman
            const int literal_count = int(reader.bits(5)) + 257, distance_count = int(reader.bits(5)) + 1;
            const int code_count = int(reader.bits(4)) + 4;
            if (literal_count > 286 || distance_count > 30) return false;
            static constexpr int order[19] = {16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15};
            std::int16_t lengths[320] = {};
            for (int i = 0; i < code_count; ++i) lengths[order[i]] = std::int16_t(reader.bits(3));
            detail::Huffman code_lengths;
            if (!detail::build_huffman(code_lengths, lengths, 19)) return false;
            for (int i = 0; i < literal_count + distance_count;) {
                const int symbol = detail::decode(reader, code_lengths);
                if (symbol < 0) return false;
                if (symbol < 16) {
                    lengths[i++] = std::int16_t(symbol);
                    continue;
                }
                std::int16_t value = 0;
                int repeat;
                if (symbol == 16) {
                    if (i == 0) return false;
                    value = lengths[i - 1];
                    repeat = 3 + int(reader.bits(2));
                } else if (symbol == 17) {
                    repeat = 3 + int(reader.bits(3));
                } else {
                    repeat = 11 + int(reader.bits(7));
                }
                if (i + repeat > literal_count + distance_count) return false;
                while (repeat--) lengths[i++] = value;
            }
            if (lengths[256] == 0) return false;  // no end-of-block code
            detail::Huffman literal, distance;
            if (!detail::build_huffman(literal, lengths, literal_count) ||
                !detail::build_huffman(distance, lengths + literal_count, distance_count))
                return false;
            if (!detail::inflate_codes(reader, out, literal, distance, limit)) return false;
        } else {
            return false;
        }
        if (reader.failed()) return false;
    }
    // Adler-32, big-endian, after the deflate data's last whole byte.
    const std::size_t at = 2 + reader.position();
    if (at + 4 > in.size()) return false;
    std::uint32_t a = 1, b = 0;
    for (const std::uint8_t x : out) {
        a = (a + x) % 65521;
        b = (b + a) % 65521;
    }
    const std::uint32_t expected = (std::uint32_t(in[at]) << 24) | (std::uint32_t(in[at + 1]) << 16) |
                                   (std::uint32_t(in[at + 2]) << 8) | in[at + 3];
    return ((b << 16) | a) == expected;
}

// ---------------------------------------------------------------------------------------------------------------
// SHA-256

inline std::string sha256_hex(std::span<const std::uint8_t> data) {
    static constexpr std::uint32_t k[64] = {
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
        0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
        0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
        0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
        0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
        0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
        0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};
    std::uint32_t h[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                          0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
    const auto rotr = [](std::uint32_t x, int n) { return (x >> n) | (x << (32 - n)); };
    std::vector<std::uint8_t> message(data.begin(), data.end());
    const std::uint64_t bits = std::uint64_t(data.size()) * 8;
    message.push_back(0x80);
    while (message.size() % 64 != 56) message.push_back(0);
    for (int i = 7; i >= 0; --i) message.push_back(std::uint8_t(bits >> (8 * i)));
    for (std::size_t block = 0; block < message.size(); block += 64) {
        std::uint32_t w[64];
        for (int i = 0; i < 16; ++i)
            w[i] = (std::uint32_t(message[block + 4 * i]) << 24) | (std::uint32_t(message[block + 4 * i + 1]) << 16) |
                   (std::uint32_t(message[block + 4 * i + 2]) << 8) | message[block + 4 * i + 3];
        for (int i = 16; i < 64; ++i) {
            const std::uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
            const std::uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16] + s0 + w[i - 7] + s1;
        }
        std::uint32_t a = h[0], b = h[1], c = h[2], d = h[3], e = h[4], f = h[5], g = h[6], hh = h[7];
        for (int i = 0; i < 64; ++i) {
            const std::uint32_t t1 = hh + (rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25)) + ((e & f) ^ (~e & g)) + k[i] + w[i];
            const std::uint32_t t2 = (rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22)) + ((a & b) ^ (a & c) ^ (b & c));
            hh = g;
            g = f;
            f = e;
            e = d + t1;
            d = c;
            c = b;
            b = a;
            a = t1 + t2;
        }
        h[0] += a; h[1] += b; h[2] += c; h[3] += d; h[4] += e; h[5] += f; h[6] += g; h[7] += hh;
    }
    static constexpr char hex[] = "0123456789abcdef";
    std::string out;
    for (const std::uint32_t word : h)
        for (int shift = 28; shift >= 0; shift -= 4) out.push_back(hex[(word >> shift) & 0xf]);
    return out;
}

// ---------------------------------------------------------------------------------------------------------------
// JSON (the subset the translator's index uses, which is all of JSON except number precision beyond double)

struct Json {
    enum class Kind { null, boolean, number, string, array, object } kind = Kind::null;
    bool boolean = false;
    double number = 0;
    std::string string;
    std::vector<Json> array;
    std::vector<std::pair<std::string, Json>> object;

    const Json* find(std::string_view key) const {
        for (const auto& [k, v] : object)
            if (k == key) return &v;
        return nullptr;
    }
};

class JsonReader {
public:
    explicit JsonReader(std::string_view text) : text_(text) {}

    bool parse(Json& out) {
        if (!value(out, 0)) return false;
        space();
        return pos_ == text_.size();
    }

private:
    void space() {
        while (pos_ < text_.size() && (text_[pos_] == ' ' || text_[pos_] == '\n' || text_[pos_] == '\r' ||
                                       text_[pos_] == '\t'))
            ++pos_;
    }

    bool literal(std::string_view word) {
        if (text_.substr(pos_, word.size()) != word) return false;
        pos_ += word.size();
        return true;
    }

    bool value(Json& out, int depth) {
        if (depth > 64) return false;
        space();
        if (pos_ >= text_.size()) return false;
        const char c = text_[pos_];
        if (c == '{') {
            out.kind = Json::Kind::object;
            ++pos_;
            space();
            if (pos_ < text_.size() && text_[pos_] == '}') return ++pos_, true;
            for (;;) {
                space();
                std::string key;
                if (!string(key)) return false;
                space();
                if (pos_ >= text_.size() || text_[pos_++] != ':') return false;
                Json item;
                if (!value(item, depth + 1)) return false;
                out.object.emplace_back(std::move(key), std::move(item));
                space();
                if (pos_ >= text_.size()) return false;
                if (text_[pos_] == ',') {
                    ++pos_;
                    continue;
                }
                return text_[pos_++] == '}';
            }
        }
        if (c == '[') {
            out.kind = Json::Kind::array;
            ++pos_;
            space();
            if (pos_ < text_.size() && text_[pos_] == ']') return ++pos_, true;
            for (;;) {
                Json item;
                if (!value(item, depth + 1)) return false;
                out.array.push_back(std::move(item));
                space();
                if (pos_ >= text_.size()) return false;
                if (text_[pos_] == ',') {
                    ++pos_;
                    continue;
                }
                return text_[pos_++] == ']';
            }
        }
        if (c == '"') {
            out.kind = Json::Kind::string;
            return string(out.string);
        }
        if (literal("true")) return out.kind = Json::Kind::boolean, out.boolean = true, true;
        if (literal("false")) return out.kind = Json::Kind::boolean, out.boolean = false, true;
        if (literal("null")) return out.kind = Json::Kind::null, true;
        const std::size_t start = pos_;
        while (pos_ < text_.size() && std::strchr("+-0123456789.eE", text_[pos_])) ++pos_;
        if (pos_ == start) return false;
        out.kind = Json::Kind::number;
        out.number = std::stod(std::string(text_.substr(start, pos_ - start)));
        return true;
    }

    bool string(std::string& out) {
        if (pos_ >= text_.size() || text_[pos_++] != '"') return false;
        while (pos_ < text_.size()) {
            const char c = text_[pos_++];
            if (c == '"') return true;
            if (c != '\\') {
                out.push_back(c);
                continue;
            }
            if (pos_ >= text_.size()) return false;
            const char e = text_[pos_++];
            switch (e) {
            case '"': case '\\': case '/': out.push_back(e); break;
            case 'b': out.push_back('\b'); break;
            case 'f': out.push_back('\f'); break;
            case 'n': out.push_back('\n'); break;
            case 'r': out.push_back('\r'); break;
            case 't': out.push_back('\t'); break;
            case 'u': {
                if (pos_ + 4 > text_.size()) return false;
                const unsigned code = unsigned(std::stoul(std::string(text_.substr(pos_, 4)), nullptr, 16));
                pos_ += 4;
                if (code < 0x80) {
                    out.push_back(char(code));
                } else if (code < 0x800) {
                    out.push_back(char(0xc0 | (code >> 6)));
                    out.push_back(char(0x80 | (code & 0x3f)));
                } else {  // surrogate pairs are kept as two 3-byte sequences; index names are ASCII
                    out.push_back(char(0xe0 | (code >> 12)));
                    out.push_back(char(0x80 | ((code >> 6) & 0x3f)));
                    out.push_back(char(0x80 | (code & 0x3f)));
                }
                break;
            }
            default: return false;
            }
        }
        return false;
    }

    std::string_view text_;
    std::size_t pos_ = 0;
};

// ---------------------------------------------------------------------------------------------------------------
// DXBC containers and the FSSM program

// The payload of one DXBC part ("DXIL", "PRIV", ...), or an empty span.
inline std::span<const std::uint8_t> dxbc_part(std::span<const std::uint8_t> container, const char (&fourcc)[5]) {
    const auto u32 = [&](std::size_t at) -> std::uint32_t {
        if (at + 4 > container.size()) return 0;
        std::uint32_t v;
        std::memcpy(&v, container.data() + at, 4);
        return v;
    };
    if (container.size() < 32 || std::memcmp(container.data(), "DXBC", 4) != 0) return {};
    const std::uint32_t parts = u32(28);
    for (std::uint32_t i = 0; i < parts; ++i) {
        const std::uint32_t at = u32(32 + 4 * std::size_t(i));
        if (std::size_t(at) + 8 > container.size()) return {};
        const std::uint32_t size = u32(at + 4);
        if (std::size_t(at) + 8 + size > container.size()) return {};
        if (std::memcmp(container.data() + at, fourcc, 4) == 0) return container.subspan(at + 8, size);
    }
    return {};
}

// The FSSM blob of a shader: inflated from the container's DXIL/XDAT/XBCH entry, or the bytecode itself when it is
// already a bare FSSM blob. Empty when neither applies.
inline std::vector<std::uint8_t> fssm_program(std::span<const std::uint8_t> bytecode) {
    if (bytecode.size() >= 4 && std::memcmp(bytecode.data(), "FSSM", 4) == 0)
        return {bytecode.begin(), bytecode.end()};
    const auto dxil = dxbc_part(bytecode, "DXIL");
    const auto u32 = [&](std::size_t at) -> std::uint32_t {
        if (at + 4 > dxil.size()) return 0;
        std::uint32_t v;
        std::memcpy(&v, dxil.data() + at, 4);
        return v;
    };
    std::size_t xdat = 0;
    for (; xdat + 4 <= dxil.size(); ++xdat)
        if (std::memcmp(dxil.data() + xdat, "XDAT", 4) == 0) break;
    if (xdat + 12 > dxil.size()) return {};
    const std::uint32_t count = u32(xdat + 4);
    for (std::uint32_t i = 0; i < count; ++i) {
        const std::size_t entry = xdat + 12 + 12 * std::size_t(i);
        if (entry + 12 > dxil.size()) return {};
        const std::uint32_t tag = u32(entry), offset = u32(entry + 4), size = u32(entry + 8);
        if (tag != 0x0d) continue;
        if (std::size_t(offset) + size > dxil.size() || size < 24) return {};
        const auto blob = dxil.subspan(offset, size);
        if (std::memcmp(blob.data(), "XBCH", 4) != 0) return {};
        std::uint32_t unpacked;
        std::memcpy(&unpacked, blob.data() + 16, 4);
        std::vector<std::uint8_t> out;
        if (!zlib_inflate(blob.subspan(24), out, unpacked) || out.size() != unpacked) return {};
        return out;
    }
    return {};
}

// ---------------------------------------------------------------------------------------------------------------
// The translator's spirv/index.json

struct ShaderEntry {
    std::string name;
    std::string stage;  // "vs", "ps", "cs", "ms", "gs"
    std::string fssm_sha256;
    std::size_t binding_count = 0;  // set-0 bindings the module declares
    std::uint32_t push_constant_bytes = 0;
    std::array<std::uint32_t, 3> local_size{1, 1, 1};
    bool uses_code_base = false;
    bool wave64 = false;
};

class ShaderLibrary {
public:
    // Reads <dir>/index.json. Entries without fssm_sha256 cannot be found by content and are skipped.
    bool load(const std::string& dir, std::string* error = nullptr) {
        const auto fail = [&](const std::string& why) {
            if (error) *error = why;
            return false;
        };
        std::ifstream file(dir + "/index.json", std::ios::binary);
        if (!file) return fail("cannot open " + dir + "/index.json");
        const std::string text((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
        Json root;
        if (!JsonReader(text).parse(root) || root.kind != Json::Kind::object) return fail("index.json is not a JSON object");
        dir_ = dir;
        entries_.clear();
        for (const auto& [name, meta] : root.object) {
            const Json* hash = meta.find("fssm_sha256");
            const Json* stage = meta.find("stage");
            if (!hash || hash->kind != Json::Kind::string || !stage || stage->kind != Json::Kind::string) continue;
            ShaderEntry e;
            e.name = name;
            e.stage = stage->string;
            e.fssm_sha256 = hash->string;
            if (const Json* b = meta.find("bindings"); b && b->kind == Json::Kind::array) e.binding_count = b->array.size();
            if (const Json* p = meta.find("push_constant_bytes"); p && p->kind == Json::Kind::number)
                e.push_constant_bytes = std::uint32_t(p->number);
            if (const Json* l = meta.find("local_size"); l && l->kind == Json::Kind::array)
                for (std::size_t i = 0; i < l->array.size() && i < 3; ++i) e.local_size[i] = std::uint32_t(l->array[i].number);
            if (const Json* c = meta.find("uses_code_base"); c && c->kind == Json::Kind::boolean) e.uses_code_base = c->boolean;
            if (const Json* w = meta.find("wave64"); w && w->kind == Json::Kind::boolean) e.wave64 = w->boolean;
            entries_[{e.stage, e.fssm_sha256}] = std::move(e);
        }
        return true;
    }

    bool loaded() const { return !dir_.empty(); }
    std::size_t size() const { return entries_.size(); }

    const ShaderEntry* find(std::string_view stage, std::string_view fssm_sha256) const {
        const auto it = entries_.find({std::string(stage), std::string(fssm_sha256)});
        return it == entries_.end() ? nullptr : &it->second;
    }

    // The module's words, or empty when <dir>/<name>.spv is missing or not SPIR-V.
    std::vector<std::uint32_t> spirv(const ShaderEntry& entry) const {
        std::ifstream file(dir_ + "/" + entry.name + ".spv", std::ios::binary);
        if (!file) return {};
        const std::string bytes((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
        if (bytes.size() < 20 || bytes.size() % 4) return {};
        std::vector<std::uint32_t> words(bytes.size() / 4);
        std::memcpy(words.data(), bytes.data(), bytes.size());
        return words[0] == 0x07230203 ? words : std::vector<std::uint32_t>{};
    }

private:
    std::string dir_;
    std::map<std::pair<std::string, std::string>, ShaderEntry> entries_;
};

} // namespace leonida
