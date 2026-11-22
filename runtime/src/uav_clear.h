#pragma once

// ClearUnorderedAccessViewUint / ClearUnorderedAccessViewFloat values as the bytes of one element of a view's
// format, and those bytes as the VkClearColorValue that writes them into an image of another format of the same
// size (a typeless resource viewed through a typed UAV). Needs the Vulkan headers only.
//
// The conversion rules are D3D's (Microsoft's documentation of ID3D11DeviceContext::ClearUnorderedAccessViewUint
// and ...Float, which the D3D12 methods keep):
// - Uint copies the low n bits of value i into channel i, where n is the channel's width, with no conversion,
//   whatever the format's type: 0x3f800000 into an R32_FLOAT channel is 1.0, 0xff into an R8_UNORM one is 1.0.
//   A raw or structured buffer view takes value 0 for every 32-bit word.
// - Float converts to the format: UNORM and SNORM clamp to [0, 1] / [-1, 1] and round to nearest (NaN is 0);
//   16-bit and the packed 11/10-bit floats round to nearest even (the small floats have no sign: negative is 0).
//   D3D11 calls Float on an integer view invalid and D3D12 does not say; here the value is converted to the
//   integer (rounded toward zero, clamped, NaN is 0) so the result is defined. A raw or structured buffer view
//   takes value 0's bits.
//
// An image is cleared with the value that decodes those bytes in the image's own format, which is exact for every
// format here except the SNORM pattern 0x80..00 (Vulkan writes -1.0 as 0x81..), and NaN payloads, which a driver
// may canonicalise.

#ifndef VK_NO_PROTOTYPES
#define VK_NO_PROTOTYPES
#endif
#include <vulkan/vulkan.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>

namespace leonida::uav_clear {

enum class Kind : std::uint8_t { unorm, snorm, uint, sint, sfloat, ufloat };

// One channel: where its bits are in the element (little-endian, from bit 0 of byte 0) and how many.
struct Channel {
    std::uint8_t offset = 0;
    std::uint8_t bits = 0;  // 0: the format has no such channel
};

struct Layout {
    VkFormat format;
    Kind kind;
    std::uint8_t bytes;             // of one element
    std::array<Channel, 4> rgba;    // R, G, B, A
};

// The formats D3D12 allows for typed UAVs (D3D12_FORMAT_SUPPORT2_UAV_TYPED_STORE), as their Vulkan formats.
inline constexpr Layout layouts[] = {
    {VK_FORMAT_R32G32B32A32_SFLOAT, Kind::sfloat, 16, {{{0, 32}, {32, 32}, {64, 32}, {96, 32}}}},
    {VK_FORMAT_R32G32B32A32_UINT, Kind::uint, 16, {{{0, 32}, {32, 32}, {64, 32}, {96, 32}}}},
    {VK_FORMAT_R32G32B32A32_SINT, Kind::sint, 16, {{{0, 32}, {32, 32}, {64, 32}, {96, 32}}}},
    {VK_FORMAT_R16G16B16A16_SFLOAT, Kind::sfloat, 8, {{{0, 16}, {16, 16}, {32, 16}, {48, 16}}}},
    {VK_FORMAT_R16G16B16A16_UNORM, Kind::unorm, 8, {{{0, 16}, {16, 16}, {32, 16}, {48, 16}}}},
    {VK_FORMAT_R16G16B16A16_SNORM, Kind::snorm, 8, {{{0, 16}, {16, 16}, {32, 16}, {48, 16}}}},
    {VK_FORMAT_R16G16B16A16_UINT, Kind::uint, 8, {{{0, 16}, {16, 16}, {32, 16}, {48, 16}}}},
    {VK_FORMAT_R16G16B16A16_SINT, Kind::sint, 8, {{{0, 16}, {16, 16}, {32, 16}, {48, 16}}}},
    {VK_FORMAT_R32G32_SFLOAT, Kind::sfloat, 8, {{{0, 32}, {32, 32}, {}, {}}}},
    {VK_FORMAT_R32G32_UINT, Kind::uint, 8, {{{0, 32}, {32, 32}, {}, {}}}},
    {VK_FORMAT_R32G32_SINT, Kind::sint, 8, {{{0, 32}, {32, 32}, {}, {}}}},
    {VK_FORMAT_A2B10G10R10_UNORM_PACK32, Kind::unorm, 4, {{{0, 10}, {10, 10}, {20, 10}, {30, 2}}}},
    {VK_FORMAT_A2B10G10R10_UINT_PACK32, Kind::uint, 4, {{{0, 10}, {10, 10}, {20, 10}, {30, 2}}}},
    {VK_FORMAT_B10G11R11_UFLOAT_PACK32, Kind::ufloat, 4, {{{0, 11}, {11, 11}, {22, 10}, {}}}},
    {VK_FORMAT_R8G8B8A8_UNORM, Kind::unorm, 4, {{{0, 8}, {8, 8}, {16, 8}, {24, 8}}}},
    {VK_FORMAT_R8G8B8A8_SNORM, Kind::snorm, 4, {{{0, 8}, {8, 8}, {16, 8}, {24, 8}}}},
    {VK_FORMAT_R8G8B8A8_UINT, Kind::uint, 4, {{{0, 8}, {8, 8}, {16, 8}, {24, 8}}}},
    {VK_FORMAT_R8G8B8A8_SINT, Kind::sint, 4, {{{0, 8}, {8, 8}, {16, 8}, {24, 8}}}},
    {VK_FORMAT_B8G8R8A8_UNORM, Kind::unorm, 4, {{{16, 8}, {8, 8}, {0, 8}, {24, 8}}}},
    {VK_FORMAT_R16G16_SFLOAT, Kind::sfloat, 4, {{{0, 16}, {16, 16}, {}, {}}}},
    {VK_FORMAT_R16G16_UNORM, Kind::unorm, 4, {{{0, 16}, {16, 16}, {}, {}}}},
    {VK_FORMAT_R16G16_SNORM, Kind::snorm, 4, {{{0, 16}, {16, 16}, {}, {}}}},
    {VK_FORMAT_R16G16_UINT, Kind::uint, 4, {{{0, 16}, {16, 16}, {}, {}}}},
    {VK_FORMAT_R16G16_SINT, Kind::sint, 4, {{{0, 16}, {16, 16}, {}, {}}}},
    {VK_FORMAT_R32_SFLOAT, Kind::sfloat, 4, {{{0, 32}, {}, {}, {}}}},
    {VK_FORMAT_R32_UINT, Kind::uint, 4, {{{0, 32}, {}, {}, {}}}},
    {VK_FORMAT_R32_SINT, Kind::sint, 4, {{{0, 32}, {}, {}, {}}}},
    {VK_FORMAT_R8G8_UNORM, Kind::unorm, 2, {{{0, 8}, {8, 8}, {}, {}}}},
    {VK_FORMAT_R8G8_SNORM, Kind::snorm, 2, {{{0, 8}, {8, 8}, {}, {}}}},
    {VK_FORMAT_R8G8_UINT, Kind::uint, 2, {{{0, 8}, {8, 8}, {}, {}}}},
    {VK_FORMAT_R8G8_SINT, Kind::sint, 2, {{{0, 8}, {8, 8}, {}, {}}}},
    {VK_FORMAT_R16_SFLOAT, Kind::sfloat, 2, {{{0, 16}, {}, {}, {}}}},
    {VK_FORMAT_R16_UNORM, Kind::unorm, 2, {{{0, 16}, {}, {}, {}}}},
    {VK_FORMAT_R16_SNORM, Kind::snorm, 2, {{{0, 16}, {}, {}, {}}}},
    {VK_FORMAT_R16_UINT, Kind::uint, 2, {{{0, 16}, {}, {}, {}}}},
    {VK_FORMAT_R16_SINT, Kind::sint, 2, {{{0, 16}, {}, {}, {}}}},
    {VK_FORMAT_R8_UNORM, Kind::unorm, 1, {{{0, 8}, {}, {}, {}}}},
    {VK_FORMAT_R8_SNORM, Kind::snorm, 1, {{{0, 8}, {}, {}, {}}}},
    {VK_FORMAT_R8_UINT, Kind::uint, 1, {{{0, 8}, {}, {}, {}}}},
    {VK_FORMAT_R8_SINT, Kind::sint, 1, {{{0, 8}, {}, {}, {}}}},
};

inline constexpr const Layout* layout(VkFormat format) {
    for (const Layout& l : layouts)
        if (l.format == format) return &l;
    return nullptr;
}

// Up to one element of the largest format.
struct Element {
    std::array<std::uint8_t, 16> bytes{};
    std::uint32_t size = 0;
};

namespace detail {

inline constexpr std::uint32_t mask(unsigned bits) { return bits >= 32 ? 0xffffffffu : (1u << bits) - 1; }

inline void put(Element& e, Channel c, std::uint32_t value) {
    value &= mask(c.bits);
    for (unsigned i = 0; i < c.bits; ++i)
        if (value >> i & 1) e.bytes[(c.offset + i) / 8] |= std::uint8_t(1u << ((c.offset + i) % 8));
}

inline std::uint32_t get(const Element& e, Channel c) {
    std::uint32_t value = 0;
    for (unsigned i = 0; i < c.bits; ++i)
        value |= std::uint32_t(e.bytes[(c.offset + i) / 8] >> ((c.offset + i) % 8) & 1) << i;
    return value;
}

// A float32 as a float with `mantissa` fraction bits, `exponent` exponent bits and an optional sign bit (IEEE
// half: 10, 5, signed; the packed formats: 6 or 5, 5, unsigned), rounded to nearest even. Too large is infinity.
inline std::uint32_t small_float(float value, unsigned mantissa, unsigned exponent, bool sign) {
    const std::uint32_t f = std::bit_cast<std::uint32_t>(value);
    const std::uint32_t negative = f >> 31;
    const std::uint32_t max_exponent = mask(exponent);
    const std::uint32_t sign_bit = sign ? negative << (mantissa + exponent) : 0;
    if (std::isnan(value)) return sign_bit | (max_exponent << mantissa) | (1u << (mantissa - 1));
    if (!sign && negative) return 0;
    const std::uint32_t magnitude = f & 0x7fffffffu;
    const int bias = int(max_exponent >> 1);
    const int e = int(magnitude >> 23) - 127 + bias;  // the exponent field in the small format
    if (e >= int(max_exponent)) return sign_bit | (max_exponent << mantissa);  // overflow, or infinity
    const std::uint32_t full = (magnitude & 0x7fffffu) | 0x800000u;           // 24-bit significand
    // A normal result keeps `mantissa` bits; a subnormal one shifts further right.
    const int shift = (e > 0 ? 0 : 1 - e) + int(23 - mantissa);
    if (shift > 24) return sign_bit;  // rounds to zero
    std::uint32_t kept = full >> shift;
    const std::uint32_t rest = full & mask(unsigned(shift));
    const std::uint32_t half = 1u << (shift - 1);
    if (rest > half || (rest == half && (kept & 1))) ++kept;  // may carry into the exponent, which is right
    const std::uint32_t biased = e > 0 ? (std::uint32_t(e) << mantissa) + (kept - (1u << mantissa)) : kept;
    return sign_bit | std::min(biased, max_exponent << mantissa);
}

inline float from_small_float(std::uint32_t bits, unsigned mantissa, unsigned exponent, bool sign) {
    const bool negative = sign && (bits >> (mantissa + exponent) & 1);
    const std::uint32_t e = bits >> mantissa & mask(exponent);
    const std::uint32_t m = bits & mask(mantissa);
    const int bias = int(mask(exponent) >> 1);
    float value;
    if (e == mask(exponent))
        value = m ? std::numeric_limits<float>::quiet_NaN() : std::numeric_limits<float>::infinity();
    else if (e == 0)
        value = std::ldexp(float(m), 1 - bias - int(mantissa));
    else
        value = std::ldexp(float(m | (1u << mantissa)), int(e) - bias - int(mantissa));
    return negative ? -value : value;
}

inline std::uint32_t to_float_bits(float value, const Layout& l, unsigned bits) {
    if (l.kind == Kind::ufloat) return small_float(value, bits - 5, 5, false);
    if (bits == 16) return small_float(value, 10, 5, true);
    return std::bit_cast<std::uint32_t>(value);
}

inline float from_float_bits(std::uint32_t raw, const Layout& l, unsigned bits) {
    if (l.kind == Kind::ufloat) return from_small_float(raw, bits - 5, 5, false);
    if (bits == 16) return from_small_float(raw, 10, 5, true);
    return std::bit_cast<float>(raw);
}

// Float to an integer channel: rounded toward zero, clamped, NaN is 0.
inline std::uint32_t to_integer(float value, bool is_signed, unsigned bits) {
    if (std::isnan(value)) return 0;
    if (is_signed) {
        const double low = -std::ldexp(1.0, int(bits) - 1), high = std::ldexp(1.0, int(bits) - 1) - 1;
        return std::uint32_t(std::int64_t(std::clamp(std::trunc(double(value)), low, high)));
    }
    return std::uint32_t(std::clamp(std::trunc(double(value)), 0.0, std::ldexp(1.0, int(bits)) - 1));
}

}  // namespace detail

// The element a clear writes into a view of `format`. `values` are the call's four UINTs, or the bits of its four
// FLOATs when is_float. False: not a typed-UAV format.
inline bool encode(VkFormat format, bool is_float, const std::array<std::uint32_t, 4>& values, Element& out) {
    const Layout* l = layout(format);
    if (!l) return false;
    out = {};
    out.size = l->bytes;
    for (unsigned i = 0; i < 4; ++i) {
        const Channel c = l->rgba[i];
        if (!c.bits) continue;
        if (!is_float) {
            detail::put(out, c, values[i]);
            continue;
        }
        const float v = std::bit_cast<float>(values[i]);
        const double scale = double(detail::mask(c.bits));
        switch (l->kind) {
        case Kind::unorm:
            detail::put(out, c, std::isnan(v) ? 0 : std::uint32_t(std::lround(std::clamp(double(v), 0.0, 1.0) * scale)));
            break;
        case Kind::snorm: {
            const double max = double(detail::mask(c.bits - 1));
            detail::put(out, c, std::isnan(v) ? 0 : std::uint32_t(std::lround(std::clamp(double(v), -1.0, 1.0) * max)));
            break;
        }
        case Kind::uint: detail::put(out, c, detail::to_integer(v, false, c.bits)); break;
        case Kind::sint: detail::put(out, c, detail::to_integer(v, true, c.bits)); break;
        case Kind::sfloat:
        case Kind::ufloat: detail::put(out, c, detail::to_float_bits(v, *l, c.bits)); break;
        }
    }
    return true;
}

// The clear value that writes `element` into an image of `format` (same element size). False: not a format here,
// or a different size.
inline bool decode(VkFormat format, const Element& element, VkClearColorValue& out) {
    const Layout* l = layout(format);
    if (!l || l->bytes != element.size) return false;
    out = {};
    for (unsigned i = 0; i < 4; ++i) {
        const Channel c = l->rgba[i];
        if (!c.bits) continue;
        const std::uint32_t raw = detail::get(element, c);
        switch (l->kind) {
        case Kind::uint: out.uint32[i] = raw; break;
        case Kind::sint: {
            const std::uint32_t sign = 1u << (c.bits - 1);
            out.int32[i] = c.bits >= 32 ? std::int32_t(raw) : std::int32_t((raw ^ sign)) - std::int32_t(sign);
            break;
        }
        case Kind::unorm: out.float32[i] = float(double(raw) / double(detail::mask(c.bits))); break;
        case Kind::snorm: {
            const std::uint32_t sign = 1u << (c.bits - 1);
            const std::int32_t s = std::int32_t(raw ^ sign) - std::int32_t(sign);
            out.float32[i] = std::max(-1.0f, float(double(s) / double(sign - 1)));
            break;
        }
        case Kind::sfloat:
        case Kind::ufloat: out.float32[i] = detail::from_float_bits(raw, *l, c.bits); break;
        }
    }
    return true;
}

// A raw or structured buffer view: every 32-bit word gets value 0 (its bits, for Float).
inline Element raw_element(const std::array<std::uint32_t, 4>& values) {
    Element e{};
    e.size = 4;
    std::memcpy(e.bytes.data(), &values[0], 4);
    return e;
}

}  // namespace leonida::uav_clear
