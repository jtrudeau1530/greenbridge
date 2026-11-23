// xg_xs.dll: Xbox Series texture-swizzling and texture-computer APIs.
//
// XGCreateTextureComputer provides a "texture computer" that reformats texture data
// between tiled and linear layouts. XGSuggestSwizzleMode suggests the optimal swizzle
// mode for a texture description. Both are bounded by the console's GPU tiling hardware.
//
// On a PC there is no such hardware and no corresponding API. The shim returns
// E_NOTIMPL for the texture computer (the title must fall back to a CPU path or
// skip this optimization) and suggests the simplest swizzle mode for the swizzle query.
//
// Signatures from the public GDK reference (docs.microsoft.com/xbox/gdk):
//   XGCreateTextureComputer / XGSuggestSwizzleMode are documented in the Xbox GDK.

#include <windows.h>

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>
#include <utility>

#include "shims.h"
#include "startup_trace.h"
#include "title_textures.h"

namespace {

// ---- XGSuggestSwizzleMode ----
//
// HRESULT XGSuggestSwizzleMode(const XG_TEXTURE_DESC* desc,
//                               XG_SWIZZLE_MODE* mode)
//
// Suggests the optimal swizzle mode for a given texture description.
// Without Xbox tiling hardware, we suggest the linear mode.
//
// XG_SWIZZLE_MODE values (from the GDK header xg.h / xg_xs.h):
//   This is a partially-documented enum. Common values include:
//     0 = XG_SWIZZLE_STRIP   (strip/linear, available on all series)
//   Higher values select hardware-specific tiling modes.

HRESULT __stdcall XGSuggestSwizzleMode(const void* desc, uint32_t* mode) {
    if (!desc || !mode) return E_POINTER;
    *mode = 0;  // XG_SWIZZLE_STRIP (linear, no tiling)
    // Log the call once for diagnostics
    static std::atomic<bool> warned{false};
    if (!warned.exchange(true, std::memory_order_relaxed)) {
        std::fprintf(stderr,
                     "[rt] xg_xs: XGSuggestSwizzleMode returning linear mode 0; "
                     "the title's texture tiling is not applied on the PC path (logged once)\n");
    }
    if (rt::startup_trace::enabled()) {
        static std::mutex m;
        static bool first = true;
        std::lock_guard l(m);
        if (first) {
            first = false;
            rt::startup_trace::event("xg_xs",
                                     "\"api\":\"XGSuggestSwizzleMode\",\"mode\":0");
        }
    }
    return S_OK;
}

// ---- XGCreateTextureComputer ----
//
// HRESULT XGCreateTextureComputer(XG_TEXTURE_COMPUTER_DESC* desc,
//                                  ID3D12Device* device,
//                                  REFIID riid,
//                                  void** textureComputer)
//
// Creates a texture computer that can reformat texture data (tile/detile).
// On a PC without the console's display hardware, this is not supported.
// We return E_NOTIMPL so the title falls back to its CPU path.

// Boot stand-in (RT-5): the retail title calls XGCreateTextureComputer(desc, &computer) with two arguments and then
// calls the computer's methods (method 8 first, while loading). The stand-in answers every method with S_OK and does
// nothing, logging the first call of each method with its first words, so loading goes on; texture data is not tiled.
namespace xgstub {
constexpr unsigned kSlots = 48;
struct Computer {
    void** vtable;
    std::atomic<unsigned long> refs{1};
    std::uint32_t desc[14]{};  // the resource description it was made for
};
std::atomic<bool> first[kSlots];
// The resource description's words: D3D12_RESOURCE_DESC (dimension, alignment, width, height, depth | mips, format, ...).
struct Shape {
    std::uint32_t dimension, height, depth, mips, format;
    std::uint64_t width;
};
Shape shape_of(const std::uint32_t* w) {
    Shape s{};
    s.dimension = w[0];
    std::memcpy(&s.width, w + 4, 8);
    s.height = w[6];
    s.depth = w[7] & 0xFFFF;
    s.mips = w[7] >> 16;
    s.format = w[8];
    if (!s.mips) s.mips = 1;
    if (!s.depth) s.depth = 1;
    if (!s.height) s.height = 1;
    return s;
}
// Bytes per block and block edge of a DXGI format (the formats a title streams).
void block_of(std::uint32_t f, std::uint32_t& bytes, std::uint32_t& edge) {
    edge = 1;
    if ((f >= 70 && f <= 72) || (f >= 79 && f <= 81)) { bytes = 8; edge = 4; return; }                      // BC1, BC4
    if ((f >= 73 && f <= 78) || (f >= 82 && f <= 84) || (f >= 94 && f <= 99)) { bytes = 16; edge = 4; return; }  // others
    if (f >= 1 && f <= 4) { bytes = 16; return; }
    if (f >= 5 && f <= 8) { bytes = 12; return; }
    if (f >= 9 && f <= 22) { bytes = 8; return; }
    if ((f >= 48 && f <= 59) || f == 85 || f == 86 || f == 115) { bytes = 2; return; }
    if (f >= 60 && f <= 66) { bytes = 1; return; }
    bytes = 4;
}
// GetResourceTiling (method 10): total tiles, packed mips, standard 64 KiB tile shape, then one entry per subresource.
void resource_tiling(const std::uint32_t* words, std::uint64_t total, std::uint64_t packed, std::uint64_t tile,
                     std::uint64_t count, std::uint32_t first, std::uint64_t tilings) {
    const Shape s = shape_of(words);
    std::uint32_t bytes = 4, edge = 1;
    block_of(s.format, bytes, edge);
    std::uint32_t tw = 128, th = 128, td = 1;
    if (s.dimension == 4) {
        td = 16;
        switch (bytes) {
            case 1: tw = 64; th = 32; td = 32; break;
            case 2: tw = 32; th = 32; td = 32; break;
            case 4: tw = 32; th = 32; td = 16; break;
            case 8: tw = 32; th = 16; td = 16; break;
            default: tw = 16; th = 16; td = 16; break;
        }
    } else {
        switch (bytes) {
            case 1: tw = 256; th = 256; break;
            case 2: tw = 256; th = 128; break;
            case 4: tw = 128; th = 128; break;
            case 8: tw = 128; th = 64; break;
            default: tw = 64; th = 64; break;
        }
        tw *= edge;
        th *= edge;
    }
    const std::uint32_t layers = s.dimension == 4 ? 1 : s.depth;
    std::uint32_t standard = 0;
    while (standard < s.mips && (s.width >> standard) >= tw && (s.height >> standard) >= th) ++standard;
    const std::uint32_t packed_mips = s.mips - standard;
    std::uint32_t capacity = 0;
    if (count) capacity = *reinterpret_cast<std::uint32_t*>(count);
    std::uint32_t next = 0, written = 0;
    for (std::uint32_t layer = 0; layer < layers; ++layer) {
        for (std::uint32_t mip = 0; mip < s.mips; ++mip) {
            const std::uint32_t sub = layer * s.mips + mip;
            std::uint32_t w = 0, h = 0, d = 0, start = 0xFFFFFFFFu;
            if (mip < standard) {
                const auto up = [](std::uint64_t v, std::uint32_t t) { return static_cast<std::uint32_t>(((v ? v : 1) + t - 1) / t); };
                w = up(s.width >> mip, tw);
                h = up(s.height >> mip, th);
                d = s.dimension == 4 ? up(s.depth >> mip, td) : 1;
                start = next;
                next += w * h * d;
            }
            if (tilings && sub >= first && written < capacity) {
                auto* t = reinterpret_cast<std::uint32_t*>(tilings + std::uint64_t(written) * 12);
                t[0] = w;
                t[1] = (h & 0xFFFF) | (d << 16);
                t[2] = start;
                ++written;
            }
        }
        if (packed_mips && layer == 0 && packed) {
            auto* b = reinterpret_cast<std::uint8_t*>(packed);
            b[0] = static_cast<std::uint8_t>(standard);
            b[1] = static_cast<std::uint8_t>(packed_mips);
            const std::uint32_t tiles = 1, at = next;
            std::memcpy(b + 4, &tiles, 4);
            std::memcpy(b + 8, &at, 4);
        }
        if (packed_mips) next += 1;
    }
    if (!packed_mips && packed) {
        auto* b = reinterpret_cast<std::uint8_t*>(packed);
        std::memset(b, 0, 12);
        b[0] = static_cast<std::uint8_t>(standard);
    }
    if (total) *reinterpret_cast<std::uint32_t*>(total) = next;
    if (tile) {
        auto* t = reinterpret_cast<std::uint32_t*>(tile);
        t[0] = tw;
        t[1] = th;
        t[2] = td;
    }
    if (count) *reinterpret_cast<std::uint32_t*>(count) = written;
    static std::atomic<int> n{0};
    if (n.fetch_add(1) < 6)
        std::fprintf(stderr, "[xg] tiling dim %u %llux%ux%u mips %u format %u: %u tiles of %ux%u, %u standard + %u packed mips\n",
                     s.dimension, static_cast<unsigned long long>(s.width), s.height, s.depth, s.mips, s.format, next, tw, th,
                     standard, packed_mips);
}

template <unsigned Slot>
std::uint64_t WINAPI method(Computer* self, std::uint64_t a1, std::uint64_t a2, std::uint64_t a3, std::uint64_t a4, std::uint64_t a5, std::uint64_t a6) {
    if constexpr (Slot == 0) {  // QueryInterface: itself for anything
        if (!a2) return E_POINTER;
        *reinterpret_cast<void**>(a2) = self;
        self->refs.fetch_add(1);
        return S_OK;
    } else if constexpr (Slot == 1) {
        return self->refs.fetch_add(1) + 1;
    } else if constexpr (Slot == 2) {
        return self->refs.fetch_sub(1) - 1;  // never freed: a handful exist at most
    } else {
        if constexpr (Slot == 10) resource_tiling(self->desc, a1, a2, a3, a4, static_cast<std::uint32_t>(a5), a6);
        // CopyIntoSubresource(base, offset, subresource, source, row pitch, slice pitch)
        if constexpr (Slot == 8)
            rt::title_textures::note_linear_upload(a1, static_cast<std::uint32_t>(a3), reinterpret_cast<const void*>(a4),
                                                   static_cast<std::uint32_t>(a5), static_cast<std::uint32_t>(a6), self->desc);
        static std::atomic<int> shown{0};
        if (shown.fetch_add(1) < 14)
            std::fprintf(stderr, "[xg] %p method %u (0x%llx 0x%llx 0x%llx 0x%llx 0x%llx 0x%llx)\n", (void*)self, Slot, (unsigned long long)a1,
                         (unsigned long long)a2, (unsigned long long)a3, (unsigned long long)a4, (unsigned long long)a5, (unsigned long long)a6);
        if (!first[Slot].exchange(true))
            std::fprintf(stderr, "[rt] xg_xs: texture computer method %u first call (0x%llx 0x%llx 0x%llx 0x%llx): S_OK, no-op\n",
                         Slot, (unsigned long long)a1, (unsigned long long)a2, (unsigned long long)a3,
                         (unsigned long long)a4);
        return S_OK;
    }
}
template <std::size_t... I>
void fill(void** t, std::index_sequence<I...>) { ((t[I] = reinterpret_cast<void*>(&method<static_cast<unsigned>(I)>)), ...); }
void** vtable() {
    static void** v = [] { static void* t[kSlots]; fill(t, std::make_index_sequence<kSlots>{}); return t; }();
    return v;
}
bool writable(const void* p) {
    MEMORY_BASIC_INFORMATION mbi{};
    return p && VirtualQuery(p, &mbi, sizeof(mbi)) && mbi.State == MEM_COMMIT &&
           (mbi.Protect & (PAGE_READWRITE | PAGE_EXECUTE_READWRITE | PAGE_WRITECOPY));
}
}  // namespace xgstub

HRESULT __stdcall XGCreateTextureComputer(const void* desc, void** out2, const void* a3, void** out4) {
    static std::atomic<int> logged{0};
    if (logged.fetch_add(1, std::memory_order_relaxed) < 3)
        std::fprintf(stderr, "[rt] xg_xs: XGCreateTextureComputer(0x%p, 0x%p, 0x%p, 0x%p): stand-in computer\n", desc,
                     static_cast<void*>(out2), a3, static_cast<void*>(out4));
    void** out = xgstub::writable(out2) ? out2 : (xgstub::writable(out4) ? out4 : nullptr);
    if (!out) return E_POINTER;
    auto* c = new xgstub::Computer{};
    c->vtable = xgstub::vtable();
    std::memcpy(c->desc, desc, sizeof(c->desc));
    *out = c;
    if (logged.load() < 14) {
        const auto* w = reinterpret_cast<const std::uint32_t*>(desc);
        std::fprintf(stderr, "[xg] computer %p desc", (void*)c);
        for (int i = 0; i < 18; ++i) std::fprintf(stderr, " %x", w[i]);
        std::fprintf(stderr, "\n");
    }
    return S_OK;
}

}  // namespace

SHIM_TABLE(xg_xs,
    {"xg_xs.dll", "XGCreateTextureComputer", reinterpret_cast<void*>(&XGCreateTextureComputer)},
    {"xg_xs.dll", "XGSuggestSwizzleMode", reinterpret_cast<void*>(&XGSuggestSwizzleMode)},
);