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
};
std::atomic<bool> first[kSlots];
template <unsigned Slot>
std::uint64_t WINAPI method(Computer* self, std::uint64_t a1, std::uint64_t a2, std::uint64_t a3, std::uint64_t a4) {
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
    *out = c;
    return S_OK;
}

}  // namespace

SHIM_TABLE(xg_xs,
    {"xg_xs.dll", "XGCreateTextureComputer", reinterpret_cast<void*>(&XGCreateTextureComputer)},
    {"xg_xs.dll", "XGSuggestSwizzleMode", reinterpret_cast<void*>(&XGSuggestSwizzleMode)},
);