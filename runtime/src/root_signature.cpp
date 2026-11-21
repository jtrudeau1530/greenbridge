// Portable root-signature packer, see root_signature.h.
#include "root_signature.h"

namespace rt::rootsig {

PackResult pack(const RootSignature& sig) {
    PackResult out;
    if (sig.params.size() > 64) {
        out.error = "root signature has more than 64 root parameters";
        return out;
    }
    uint32_t cost = 0;
    for (const RootParameter& p : sig.params) {
        if (p.type == ParamType::Constants && p.num_values == 0) {
            out.error = "root constant parameter needs at least 1 dword";
            return out;
        }
        const uint32_t dwords = p.type == ParamType::Table          ? 2
                                : p.type == ParamType::Descriptor   ? 2
                                                                    : p.num_values;
        cost += param_cost(p.type, p.num_values);
        if (cost > kMaxD3d12Cost) {
            out.error = "root signature exceeds the 64-dword D3D12 cost limit";
            return out;
        }
        if (out.dword_count + dwords > kMaxUserSgprs) {
            // Spill layout is unknown (see __XB_GetRootSignatureSpillPointer):
            // refuse loudly instead of silently truncating every draw.
            out.error = "root signature needs more than 32 user SGPRs (spill unsupported)";
            return out;
        }
        const uint32_t param_index =
            static_cast<uint32_t>(&p - sig.params.data());
        for (uint32_t i = 0; i < dwords; ++i) {
            out.dwords[out.dword_count] = 0;  // filled at bind time
            out.provenance.push_back({param_index, i});
            ++out.dword_count;
        }
    }
    out.d3d12_cost = cost;
    out.ok = true;
    return out;
}

namespace {

void put_u32(std::vector<uint8_t>& out, uint32_t v) {
    out.push_back(static_cast<uint8_t>(v));
    out.push_back(static_cast<uint8_t>(v >> 8));
    out.push_back(static_cast<uint8_t>(v >> 16));
    out.push_back(static_cast<uint8_t>(v >> 24));
}

void put_f32(std::vector<uint8_t>& out, float v) {
    uint32_t bits = 0;
    static_assert(sizeof(bits) == sizeof(v));
    __builtin_memcpy(&bits, &v, sizeof(bits));
    put_u32(out, bits);
}

class Reader {
public:
    Reader(const uint8_t* data, size_t size) : data_(data), size_(size) {}

    [[nodiscard]] bool u32(uint32_t& v) {
        if (pos_ + 4 > size_) return false;
        v = static_cast<uint32_t>(data_[pos_]) |
            (static_cast<uint32_t>(data_[pos_ + 1]) << 8) |
            (static_cast<uint32_t>(data_[pos_ + 2]) << 16) |
            (static_cast<uint32_t>(data_[pos_ + 3]) << 24);
        pos_ += 4;
        return true;
    }

    [[nodiscard]] bool f32(float& v) {
        uint32_t bits = 0;
        if (!u32(bits)) return false;
        __builtin_memcpy(&v, &bits, sizeof(v));
        return true;
    }

    [[nodiscard]] bool at_end() const { return pos_ == size_; }

private:
    const uint8_t* data_ = nullptr;
    size_t size_ = 0;
    size_t pos_ = 0;
};

constexpr uint32_t kCanonicalMagic = 0x3153524Cu;  // "LRS1" little-endian
constexpr uint32_t kCanonicalVersion = 1;

}  // namespace

std::vector<uint8_t> encode_canonical(const RootSignature& sig) {
    std::vector<uint8_t> out;
    put_u32(out, kCanonicalMagic);
    put_u32(out, kCanonicalVersion);
    put_u32(out, static_cast<uint32_t>(sig.version));
    put_u32(out, static_cast<uint32_t>(sig.params.size()));
    put_u32(out, static_cast<uint32_t>(sig.static_samplers.size()));
    put_u32(out, sig.flags);
    for (const RootParameter& p : sig.params) {
        put_u32(out, static_cast<uint32_t>(p.type));
        put_u32(out, static_cast<uint32_t>(p.visibility));
        put_u32(out, p.num_values);
        put_u32(out, p.shader_register);
        put_u32(out, p.register_space);
        put_u32(out, p.descriptor_register);
        put_u32(out, p.descriptor_space);
        put_u32(out, static_cast<uint32_t>(p.ranges.size()));
        for (const DescriptorRange& r : p.ranges) {
            put_u32(out, r.range_type);
            put_u32(out, r.num_descriptors);
        }
    }
    for (const StaticSampler& s : sig.static_samplers) {
        put_u32(out, s.filter);
        put_u32(out, s.address_u);
        put_u32(out, s.address_v);
        put_u32(out, s.address_w);
        put_f32(out, s.mip_lod_bias);
        put_u32(out, s.max_anisotropy);
        put_u32(out, s.comparison_func);
        put_u32(out, s.border_color);
        put_f32(out, s.min_lod);
        put_f32(out, s.max_lod);
        put_u32(out, s.shader_register);
        put_u32(out, s.register_space);
        put_u32(out, static_cast<uint32_t>(s.visibility));
    }
    return out;
}

DecodeResult decode_canonical(const uint8_t* data, size_t size) {
    DecodeResult out;
    if (data == nullptr && size != 0) {
        out.error = "null data with nonzero size";
        return out;
    }
    Reader r(data == nullptr ? reinterpret_cast<const uint8_t*>("") : data, size);
    uint32_t magic = 0, version = 0, nparams = 0, nsamplers = 0;
    if (!r.u32(magic) || magic != kCanonicalMagic) {
        out.error = "bad canonical magic";
        return out;
    }
    if (!r.u32(version) || version != kCanonicalVersion) {
        out.error = "unsupported canonical version";
        return out;
    }
    uint32_t ver = 0;
    if (!r.u32(ver) || !r.u32(nparams) || !r.u32(nsamplers) || !r.u32(out.sig.flags)) {
        out.error = "truncated canonical header";
        return out;
    }
    if (ver != 1 && ver != 2) {
        out.error = "unknown root-signature version";
        return out;
    }
    if (nparams > 64) {
        out.error = "more than 64 root parameters";
        return out;
    }
    out.sig.version = ver == 1 ? Version::V1_0 : Version::V1_1;
    out.sig.params.reserve(nparams);
    for (uint32_t i = 0; i < nparams; ++i) {
        RootParameter p;
        uint32_t type = 0, vis = 0, nranges = 0;
        if (!r.u32(type) || !r.u32(vis) || !r.u32(p.num_values) || !r.u32(p.shader_register) ||
            !r.u32(p.register_space) || !r.u32(p.descriptor_register) ||
            !r.u32(p.descriptor_space) || !r.u32(nranges)) {
            out.error = "truncated canonical parameter";
            return out;
        }
        if (type > 2) {
            out.error = "unknown parameter type";
            return out;
        }
        p.type = static_cast<ParamType>(type);
        p.visibility = static_cast<Visibility>(vis);
        if (nranges > 1024) {
            out.error = "absurd range count";
            return out;
        }
        p.ranges.reserve(nranges);
        for (uint32_t j = 0; j < nranges; ++j) {
            DescriptorRange range;
            if (!r.u32(range.range_type) || !r.u32(range.num_descriptors)) {
                out.error = "truncated canonical range";
                return out;
            }
            p.ranges.push_back(range);
        }
        out.sig.params.push_back(std::move(p));
    }
    out.sig.static_samplers.reserve(nsamplers);
    for (uint32_t i = 0; i < nsamplers; ++i) {
        StaticSampler s;
        uint32_t vis = 0;
        if (!r.u32(s.filter) || !r.u32(s.address_u) || !r.u32(s.address_v) || !r.u32(s.address_w) ||
            !r.f32(s.mip_lod_bias) || !r.u32(s.max_anisotropy) || !r.u32(s.comparison_func) ||
            !r.u32(s.border_color) || !r.f32(s.min_lod) || !r.f32(s.max_lod) ||
            !r.u32(s.shader_register) || !r.u32(s.register_space) || !r.u32(vis)) {
            out.error = "truncated canonical sampler";
            return out;
        }
        s.visibility = static_cast<Visibility>(vis);
        out.sig.static_samplers.push_back(s);
    }
    if (!r.at_end()) {
        out.error = "trailing bytes after canonical root signature";
        return out;
    }
    out.ok = true;
    return out;
}

// RTS0 payload hypothesis: u32 version (1 = 1.0, 2 = 1.1), u32 num_params,
// u32 num_static_samplers, u32 flags, then per-parameter fields in d3d12.h
// order. Static samplers follow the same field order as D3D12_STATIC_SAMPLER_DESC.
// Anything else (v1.2 blobs, extra dwords, truncation) is a hard error.
DecodeResult parse_rts0(const uint8_t* data, size_t size) {
    DecodeResult out;
    Reader r(data == nullptr ? reinterpret_cast<const uint8_t*>("") : data, size);
    uint32_t ver = 0, nparams = 0, nsamplers = 0, flags = 0;
    if (data == nullptr && size != 0) {
        out.error = "null data with nonzero size";
        return out;
    }
    if (!r.u32(ver) || !r.u32(nparams) || !r.u32(nsamplers) || !r.u32(flags)) {
        out.error = "truncated RTS0 header";
        return out;
    }
    if (ver != 1 && ver != 2) {
        out.error = "unknown RTS0 version (not 1.0/1.1)";
        return out;
    }
    if (nparams > 64) {
        out.error = "RTS0 with more than 64 root parameters";
        return out;
    }
    out.sig.version = ver == 1 ? Version::V1_0 : Version::V1_1;
    out.sig.flags = flags;
    out.sig.params.reserve(nparams);
    for (uint32_t i = 0; i < nparams; ++i) {
        RootParameter p;
        uint32_t type = 0, vis = 0;
        if (!r.u32(type) || !r.u32(vis)) {
            out.error = "truncated RTS0 parameter";
            return out;
        }
        if (type > 2) {
            out.error = "unknown RTS0 parameter type";
            return out;
        }
        p.type = static_cast<ParamType>(type);
        p.visibility = static_cast<Visibility>(vis);
        if (p.type == ParamType::Constants) {
            if (!r.u32(p.shader_register) || !r.u32(p.register_space) || !r.u32(p.num_values)) {
                out.error = "truncated RTS0 constants";
                return out;
            }
        } else if (p.type == ParamType::Descriptor) {
            if (!r.u32(p.descriptor_register) || !r.u32(p.descriptor_space)) {
                out.error = "truncated RTS0 descriptor";
                return out;
            }
        } else {
            uint32_t nranges = 0;
            if (!r.u32(nranges)) {
                out.error = "truncated RTS0 table";
                return out;
            }
            if (nranges > 1024) {
                out.error = "absurd RTS0 range count";
                return out;
            }
            p.ranges.reserve(nranges);
            for (uint32_t j = 0; j < nranges; ++j) {
                DescriptorRange range;
                uint32_t base_reg = 0, space = 0, offset = 0;
                if (!r.u32(range.range_type) || !r.u32(range.num_descriptors) || !r.u32(base_reg) ||
                    !r.u32(space) || !r.u32(offset)) {
                    out.error = "truncated RTS0 range";
                    return out;
                }
                (void)base_reg;
                (void)space;
                (void)offset;
                p.ranges.push_back(range);
            }
        }
        out.sig.params.push_back(std::move(p));
    }
    out.sig.static_samplers.reserve(nsamplers);
    for (uint32_t i = 0; i < nsamplers; ++i) {
        StaticSampler s;
        uint32_t vis = 0;
        if (!r.u32(s.filter) || !r.u32(s.address_u) || !r.u32(s.address_v) || !r.u32(s.address_w) ||
            !r.f32(s.mip_lod_bias) || !r.u32(s.max_anisotropy) || !r.u32(s.comparison_func) ||
            !r.u32(s.border_color) || !r.f32(s.min_lod) || !r.f32(s.max_lod) ||
            !r.u32(s.shader_register) || !r.u32(s.register_space) || !r.u32(vis)) {
            out.error = "truncated RTS0 sampler";
            return out;
        }
        s.visibility = static_cast<Visibility>(vis);
        out.sig.static_samplers.push_back(s);
    }
    if (!r.at_end()) {
        out.error = "trailing bytes after RTS0 root signature";
        return out;
    }
    out.ok = true;
    return out;
}

}  // namespace rt::rootsig
