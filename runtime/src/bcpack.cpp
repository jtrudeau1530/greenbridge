// Native BCPack decoder: a line-for-line port of tools/bcpack.py with `history=None` (see bcpack.h).
// Function names follow the Python ones; where the two differ, the Python decoder is the reference.

#include "bcpack.h"

#include <cstdlib>
#include <cstdio>
#include <algorithm>
#include <array>
#include <bit>
#include <cstring>
#include <map>
#include <set>
#include <string>
#include <utility>

namespace leonida::bcpack {
namespace {

#include "bcpack_tables.inc"

constexpr std::int64_t rans_l = std::int64_t{1} << 24;
constexpr int max_width = 10;
constexpr int prefetch_budget = 8;
constexpr int max_virtual_zeros = 8;
constexpr std::size_t element_bytes = 8;

using Lanes = std::vector<std::vector<std::uint16_t>>;
using Labels = std::set<std::string>;

struct Failure {
    Status status;
    std::string message;
};

[[noreturn]] void fail(std::string message) { throw Failure{Status::error, std::move(message)}; }
[[noreturn]] void refuse(std::string message) { throw Failure{Status::provisional, std::move(message)}; }

std::string str(long long v) { return std::to_string(v); }

std::string join(const std::vector<std::string>& items, std::size_t first, std::size_t last) {
    std::string out;
    for (std::size_t i = first; i < last && i < items.size(); ++i) {
        if (!out.empty()) out += ", ";
        out += items[i];
    }
    return out;
}

const char* codec_name(int codec) {
    switch (codec) {
    case 1: return "BC1";
    case 3: return "BC3";
    case 4: return "BC4";
    case 5: return "BC5";
    case 6: return "BC6H";
    case 7: return "BC7";
    default: return nullptr;
    }
}

// Default per-lane symbol widths (bits) for each codec (bcpack.LANE_WIDTHS).
constexpr std::uint8_t widths_bc1[] = {8, 8, 5, 6, 5, 5, 6, 5, 8, 8, 8, 8};
constexpr std::uint8_t widths_bc3[] = {4, 4, 8, 6, 6, 6, 6, 5, 6, 5, 8, 8};
constexpr std::uint8_t widths_bc4[] = {8, 8, 8, 8, 6, 6, 6, 6, 6, 6, 6, 6};
constexpr std::uint8_t widths_bc5[] = {4, 4, 8, 8, 6, 6, 6, 6, 6, 6, 6, 6};
constexpr std::uint8_t widths_bc6h[] = {4, 4, 8, 8, 8, 5, 5, 5, 6, 6, 6, 8, 8, 8, 8, 5};
constexpr std::uint8_t widths_bc7[] = {4, 4, 8, 8, 8, 8, 8, 7, 7, 5, 6, 8, 8, 8, 8, 5};

std::span<const std::uint8_t> lane_widths(int codec) {
    switch (codec) {
    case 1: return widths_bc1;
    case 3: return widths_bc3;
    case 4: return widths_bc4;
    case 5: return widths_bc5;
    case 6: return widths_bc6h;
    case 7: return widths_bc7;
    default: return {};
    }
}

// --- Entropy layer -------------------------------------------------------------------------------------------------

// 16-bit rANS banded frequency model for one (symbol_width, q) pair (bcpack._Model).
struct Model {
    int w = 0;
    const std::uint16_t* freq = nullptr;
    std::uint32_t base[11] = {};
    std::uint32_t total = 0;

    int lookup(std::uint32_t cf) const {
        int p = w;
        while (p > 0 && cf < base[p]) --p;
        return p == 0 ? 0 : (1 << (p - 1)) + static_cast<int>((cf - base[p]) / freq[p]);
    }

    // The next state, or -1 for any negative one: Python keeps the exact negative value, but a negative state
    // only ever compares below RANS_L and stays negative, so its sign is all that matters.
    std::int64_t advance(int s, std::int64_t x) const {
        const int p = static_cast<int>(std::bit_width(static_cast<unsigned>(s)));
        std::int64_t f = freq[p];
        std::int64_t c = base[p] + f * (s - (p == 0 ? 0 : (1 << (p - 1))));
        if (c == 0) {
            f -= 1;
            c = 1;
        }
        const std::int64_t next = f * (x >> 16) + (x & 0xFFFF) - c;
        return next < 0 ? -1 : next;
    }
};

const Model& get_model(int w, int q) {
    static const auto models = [] {
        std::array<Model, 160> all{};
        for (int wi = 1; wi <= 10; ++wi) {
            for (int qi = 0; qi < 16; ++qi) {
                Model& m = all[(wi - 1) * 16 + qi];
                m.w = wi;
                m.freq = prob_freq[wi - 1][qi];
                std::uint32_t acc = 0;
                for (int p = 0; p <= wi; ++p) {
                    m.base[p] = acc;
                    acc += m.freq[p] * (p == 0 ? 1u : 1u << (p - 1));
                }
                m.total = acc;
            }
        }
        return all;
    }();
    const Model& m = models[(w - 1) * 16 + q];
    if (m.total != 65536)
        fail("probability table (" + str(w) + ") does not sum to 65536 (got " + str(m.total) + ")");
    return m;
}

// A state shifted left by 8 bits, with a byte merged in; negative states stay negative (see Model::advance).
std::int64_t shift_state(std::int64_t st, int byte = 0) { return st < 0 ? -1 : ((st << 8) | byte); }

// bcpack._decode_lanes_raw: the interleaved rANS lane array into one symbol list per lane.
std::size_t decode_lanes_raw(std::span<const std::uint8_t> raw, std::span<const std::uint8_t> widths,
                             std::int64_t nsym, Lanes& out) {
    enum { hdr0, hdr1, hdr2, proc };
    const int nl = static_cast<int>(widths.size());
    const std::int64_t L = rans_l;
    std::vector<int> ctrl(nl, hdr0), w(nl, 0), q(nl, 0), h3(nl, 0), flush(nl, 0), vz(nl, 0), inbuf(nl, 0),
        sym(nl, 0), slotval(nl, 0);
    std::vector<std::int64_t> out_rem(nl, nsym), in_rem(nl, 0), p0(nl, 0), p1(nl, 0);
    std::vector<char> done(nl, nsym <= 0), full(nl, 0), held(nl, 0), slotfull(nl, 0), stall(nl, 0);
    std::vector<const Model*> model(nl, nullptr);
    out.assign(nl, {});
    for (auto& lane : out) lane.reserve(nsym > 0 ? static_cast<std::size_t>(nsym) : 0);
    std::size_t pos = 0;
    const std::size_t nraw = raw.size();

    auto lane_fail = [&](const char* msg, int lane) {
        fail("lane " + str(lane) + " (width " + str(w[lane]) + ", q " + str(q[lane]) + ") at input byte " +
             str(static_cast<long long>(pos)) + ": " + msg);
    };

    while (true) {
        bool all_done = true;
        for (int i = 0; i < nl; ++i) {
            if (!(done[i] && !slotfull[i])) {
                all_done = false;
                break;
            }
        }
        if (all_done) return pos;

        // Phase 1: state update, prefetch arbitration (at most 8 bytes/cycle), and symbol release.
        int left = prefetch_budget;
        for (int i = 0; i < nl; ++i) {
            if (done[i]) {
                stall[i] = 1;
                continue;
            }
            if (held[i] && slotfull[i]) {
                stall[i] = 1;
                continue;
            }
            stall[i] = 0;
            if (ctrl[i] == proc) p0[i] = held[i] ? model[i]->advance(sym[i], p1[i]) : p1[i];
            if (!full[i] && (ctrl[i] != proc || in_rem[i] > 0)) {
                if (left == 0) {
                    stall[i] = 1;
                    continue;
                }
                if (pos >= nraw) lane_fail("input ran out", i);
                inbuf[i] = raw[pos];
                ++pos;
                full[i] = 1;
                --left;
            }
            if (held[i]) {
                slotfull[i] = 1;
                slotval[i] = sym[i];
                held[i] = 0;
            }
        }

        // Phase 2: header parsing, byte merge, virtual zero fill on flush, and symbol lookup.
        for (int i = 0; i < nl; ++i) {
            if (done[i] || stall[i]) continue;
            const int c = ctrl[i];
            if (c == hdr0) {
                if (!full[i]) continue;
                const int h = inbuf[i];
                q[i] = h & 15;
                flush[i] = (h >> 4) & 1;
                h3[i] = (h >> 5) & 1;
                w[i] = widths[i] - (h >> 6);
                full[i] = 0;
                ctrl[i] = hdr1;
                vz[i] = 0;
            } else if (c == hdr1) {
                if (!full[i]) continue;
                full[i] = 0;
                if (!(1 <= w[i] && w[i] <= max_width)) lane_fail("symbol width out of range", i);
                model[i] = &get_model(w[i], q[i]);
                if (h3[i]) {
                    in_rem[i] = inbuf[i];
                    ctrl[i] = hdr2;
                } else {
                    in_rem[i] = inbuf[i] + 1;
                    ctrl[i] = proc;
                }
            } else if (c == hdr2) {
                if (!full[i]) continue;
                in_rem[i] = ((static_cast<std::int64_t>(inbuf[i]) << 8) | (in_rem[i] & 0xFF)) + 257;
                ctrl[i] = proc;
                full[i] = 0;
            } else {
                std::int64_t st = p0[i];
                if (st < L) {
                    if (in_rem[i] > 0) {
                        if (full[i]) {
                            st = shift_state(st, inbuf[i]);
                            --in_rem[i];
                            full[i] = 0;
                            if (in_rem[i] == 0 && flush[i]) {
                                while (st < L && st != 0 && vz[i] < max_virtual_zeros) {
                                    st = shift_state(st);
                                    ++vz[i];
                                }
                            }
                        }
                    } else if (flush[i]) {
                        if (st == 0 || vz[i] >= max_virtual_zeros)
                            lane_fail("fragment did not end at its home state", i);
                        while (st < L && st != 0 && vz[i] < max_virtual_zeros) {
                            st = shift_state(st);
                            ++vz[i];
                        }
                    } else {
                        p1[i] = st;
                        if (out_rem[i] > 0)
                            ctrl[i] = hdr0;
                        else
                            done[i] = 1;
                        continue;
                    }
                }
                const bool nxt = st > L && out_rem[i] > 0;
                if (nxt) {
                    const int s = model[i]->lookup(static_cast<std::uint32_t>(st & 0xFFFF));
                    sym[i] = s;
                    --out_rem[i];
                    if (s >> w[i]) lane_fail("symbol outside its width", i);
                }
                held[i] = nxt;
                p1[i] = st;
                if (in_rem[i] == 0) {
                    if (flush[i]) {
                        if (st == L) {
                            if (out_rem[i] > 0) {
                                ctrl[i] = hdr0;
                                p0[i] = 0;
                                p1[i] = 0;
                            } else {
                                done[i] = 1;
                            }
                        } else if (st > L && !nxt) {
                            lane_fail("fragment did not end at its home state", i);
                        }
                    } else if (out_rem[i] == 0 && !nxt) {
                        done[i] = 1;
                    }
                } else if (st == L || (out_rem[i] == 0 && !nxt && st > L)) {
                    lane_fail("fragment did not end at its home state", i);
                }
            }
        }

        // Phase 3: emit one synchronized element when every active lane has filled its slot.
        bool ready = true;
        bool anyfull = false;
        for (int i = 0; i < nl; ++i) {
            if (slotfull[i]) {
                anyfull = true;
            } else if (!(done[i] && !held[i])) {
                ready = false;
                break;
            }
        }
        if (ready && anyfull) {
            for (int i = 0; i < nl; ++i) {
                if (slotfull[i]) {
                    out[i].push_back(static_cast<std::uint16_t>(slotval[i]));
                    slotfull[i] = 0;
                }
            }
        }
        for (int i = 0; i < nl; ++i) {
            if (done[i] && held[i] && !slotfull[i]) {
                slotfull[i] = 1;
                slotval[i] = sym[i];
                held[i] = 0;
            }
        }
    }
}

std::size_t run_decode_lanes(std::span<const std::uint8_t> raw, int codec, std::size_t out_size, Lanes& lanes) {
    const auto widths = lane_widths(codec);
    if (widths.empty()) fail("unsupported BCPack codec id: " + str(codec));
    const std::size_t bb = block_bytes(codec);
    if (out_size % bb != 0)
        fail("decoded size " + str(static_cast<long long>(out_size)) + " is not a whole number of " +
             codec_name(codec) + " blocks");
    const std::size_t used = decode_lanes_raw(raw, widths, static_cast<std::int64_t>(out_size / element_bytes), lanes);
    for (std::size_t i = used; i < raw.size(); ++i)
        if (raw[i]) fail("non-zero data left over after the last lane finished");
    return used;
}

// --- Shared helpers --------------------------------------------------------------------------------------------------

int unzig(int v) { return (v & 1) ? -((v + 1) >> 1) : (v >> 1); }

int inter(int a, int b, int n) {
    int d = 0;
    for (int k = 0; k < n; ++k) d |= (((a >> k) & 1) << (2 * k)) | (((b >> k) & 1) << (2 * k + 1));
    return d;
}

int inter8(int a, int b) { return inter(a, b, 4); }

// 4-bit de-interleave of 8-bit index-pair symbols: even bits, odd bits (bcpack._DEINT).
constexpr auto deint = [] {
    std::array<std::array<std::uint8_t, 256>, 2> t{};
    for (int s = 0; s < 256; ++s) {
        t[0][s] = static_cast<std::uint8_t>(((s >> 0) & 1) | (((s >> 2) & 1) << 1) | (((s >> 4) & 1) << 2) |
                                            (((s >> 6) & 1) << 3));
        t[1][s] = static_cast<std::uint8_t>(((s >> 1) & 1) | (((s >> 3) & 1) << 1) | (((s >> 5) & 1) << 2) |
                                            (((s >> 7) & 1) << 3));
    }
    return t;
}();

int pymod(int a, int m) { return ((a % m) + m) % m; }

// Maximum undetermined blocks permitted in provisional mode (1 in 1,024, min 2).
std::size_t loose_limit_strict(std::size_t nblocks) { return std::max<std::size_t>(2, nblocks / 1024); }
std::size_t loose_limit(std::size_t nblocks) {
    static const bool any = std::getenv("LEONIDA_BCPACK_HISTORY") != nullptr;  // boot: a partly wrong texture beats none
    return any ? nblocks : loose_limit_strict(nblocks);
}

bool any_nonzero(const std::vector<std::uint16_t>& lane) {
    return std::any_of(lane.begin(), lane.end(), [](std::uint16_t v) { return v != 0; });
}

void put_u16(std::uint8_t* at, int v) {
    at[0] = static_cast<std::uint8_t>(v & 0xFF);
    at[1] = static_cast<std::uint8_t>((v >> 8) & 0xFF);
}

void put_le(std::uint8_t* at, std::uint64_t v, int nbytes) {
    for (int k = 0; k < nbytes; ++k) at[k] = static_cast<std::uint8_t>((v >> (8 * k)) & 0xFF);
}

// ORs the low `nbits` of `v` into a 128-bit little-endian block at bit `pos`; bits past 128 are dropped.
void put_bits(std::uint8_t* block, int pos, std::uint64_t v, int nbits) {
    for (int k = 0; k < nbits; ++k) {
        const int bit = pos + k;
        if (bit >= 128) break;
        if ((v >> k) & 1) block[bit >> 3] |= static_cast<std::uint8_t>(1u << (bit & 7));
    }
}

// The four BC1-style colour index bytes from the quadrant symbols of pixels TL, TR, BL, BR.
void put_quadrants(std::uint8_t* at, int qa, int qb, int qc, int qd) {
    at[0] = static_cast<std::uint8_t>((qa & 15) | ((qb & 15) << 4));
    at[1] = static_cast<std::uint8_t>((qa >> 4) | ((qb >> 4) << 4));
    at[2] = static_cast<std::uint8_t>((qc & 15) | ((qd & 15) << 4));
    at[3] = static_cast<std::uint8_t>((qc >> 4) | ((qd >> 4) << 4));
}

std::string sorted_list(const std::set<int>& items) {  // Python's repr of sorted(set_of_ints)
    std::string out = "[";
    for (int v : items) {
        if (out.size() > 1) out += ", ";
        out += str(v);
    }
    return out + "]";
}

// --- BC1 / BC3 / BC4 / BC5 ------------------------------------------------------------------------------------------

void bc1_blocks(const Lanes& lanes, std::size_t nblocks, Labels* provisional, std::uint8_t* out) {
    if (any_nonzero(lanes[1])) fail("BC1 lane 1 is non-zero (unexpected)");
    constexpr int mods[6] = {32, 64, 32, 32, 64, 32};
    std::vector<std::array<int, 6>> hist(nblocks);
    const auto& l0 = lanes[0];
    for (std::size_t bi = 0; bi < nblocks; ++bi) {
        const long long i = static_cast<long long>(bi);
        const int v = l0[bi];
        const int d = (v & 1) ? 256 - ((v + 1) >> 1) : (v >> 1);
        std::array<int, 6> base{};
        if (d == 0) {
        } else if (i - d >= 0) {
            base = hist[i - d];
        } else {
            const int entry = static_cast<int>(256 + i - d);
            const int state = bc1_history_state[entry];
            if (state == 1 || (state == 2 && provisional)) {
                for (int k = 0; k < 6; ++k) base[k] = bc1_history[entry][k];
                if (state == 2) provisional->insert("BC1 history " + str(entry));
            } else {
                refuse("BC1 block " + str(i) + " references " + (state == 2 ? "unconfirmed" : "unknown") +
                       " built-in history entry " + str(entry));
            }
        }
        std::array<int, 6> e{};
        for (int k = 0; k < 6; ++k) e[k] = pymod(base[k] + unzig(lanes[2 + k][bi]), mods[k]);
        hist[bi] = e;
        std::uint8_t* at = out + bi * 8;
        put_u16(at, (e[0] << 11) | (e[1] << 5) | e[2]);
        put_u16(at + 2, (e[3] << 11) | (e[4] << 5) | e[5]);
        put_quadrants(at + 4, bc1_quadrant[lanes[8][bi]], bc1_quadrant[lanes[9][bi]], bc1_quadrant[lanes[10][bi]],
                      bc1_quadrant[lanes[11][bi]]);
    }
}

// 48 bits of 3-bit indices from four rows of (left, right) pair symbols.
std::uint64_t pair_rows(const std::uint8_t (*pairs)[2], const std::uint16_t* const rows_a[4],
                        const std::uint16_t* const rows_b[4]) {
    std::uint64_t bits = 0;
    for (int r = 0; r < 4; ++r) {
        const auto& left = pairs[*rows_a[r]];
        const auto& right = pairs[*rows_b[r]];
        bits |= static_cast<std::uint64_t>(left[0] | (left[1] << 3) | (right[0] << 6) | (right[1] << 9)) << (12 * r);
    }
    return bits;
}

void bc3_blocks(const Lanes& lanes, std::size_t nblocks, Labels* provisional, std::uint8_t* out) {
    if (any_nonzero(lanes[1])) fail("BC3 lane 1 is non-zero (unexpected)");
    constexpr int mods[8] = {256, 256, 32, 64, 32, 32, 64, 32};
    constexpr std::array<int, 8> none = {128, 128, 0, 0, 0, 0, 0, 0};
    std::vector<std::array<int, 8>> hist(nblocks);
    std::vector<char> lost(nblocks, 0);
    std::set<int> unknown;
    const auto &l0 = lanes[0], &alpha = lanes[2], &lr = lanes[7], &lg = lanes[8], &lb = lanes[9];
    const auto &top = lanes[10], &bottom = lanes[11];
    for (std::size_t bi = 0; bi < nblocks; ++bi) {
        const long long i = static_cast<long long>(bi);
        const std::size_t a = 2 * bi, b = 2 * bi + 1;
        const int d = unzig(inter8(l0[a], l0[b])) & 255;
        std::array<int, 8> base = none;
        if (d == 0) {
        } else if (i - d >= 0) {
            base = hist[i - d];
            lost[bi] = lost[i - d];
        } else {
            const int entry = static_cast<int>(256 + i - d);
            const int state = bc3_history_state[entry];
            if (state == 1) {
                for (int k = 0; k < 8; ++k) base[k] = bc3_history[entry][k];
            } else if (state == 2) {
                if (!provisional)
                    refuse("BC3 block " + str(i) + " references unconfirmed built-in history entry " + str(entry));
                for (int k = 0; k < 8; ++k) base[k] = bc3_history[entry][k];
                provisional->insert("BC3 history " + str(entry));
            } else {
                if (!provisional)
                    refuse("BC3 block " + str(i) + " references unknown built-in history entry " + str(entry));
                lost[bi] = 1;
                unknown.insert(entry);
            }
        }
        const int dz[8] = {alpha[a], alpha[b], lr[a], lg[a], lb[a], lr[b], lg[b], lb[b]};
        std::array<int, 8> e{};
        for (int k = 0; k < 8; ++k) e[k] = pymod(base[k] + unzig(dz[k]), mods[k]);
        hist[bi] = e;
        const std::uint16_t* rows_a[4] = {&lanes[3][a], &lanes[4][a], &lanes[5][a], &lanes[6][a]};
        const std::uint16_t* rows_b[4] = {&lanes[3][b], &lanes[4][b], &lanes[5][b], &lanes[6][b]};
        std::uint8_t* at = out + bi * 16;
        at[0] = static_cast<std::uint8_t>(e[0]);
        at[1] = static_cast<std::uint8_t>(e[1]);
        put_le(at + 2, pair_rows(bc3_alpha_pair, rows_a, rows_b), 6);
        put_u16(at + 8, (e[2] << 11) | (e[3] << 5) | e[4]);
        put_u16(at + 10, (e[5] << 11) | (e[6] << 5) | e[7]);
        put_quadrants(at + 12, bc3_colour_quadrant[top[a]], bc3_colour_quadrant[top[b]],
                      bc3_colour_quadrant[bottom[a]], bc3_colour_quadrant[bottom[b]]);
    }
    if (!unknown.empty()) {
        const auto total = static_cast<std::size_t>(std::count(lost.begin(), lost.end(), 1));
        if (total > loose_limit(nblocks))
            refuse("BC3 chunk has " + str(static_cast<long long>(total)) + "/" + str(static_cast<long long>(nblocks)) +
                   " blocks from unknown built-in history entries");
        provisional->insert("BC3 unknown history " + sorted_list(unknown) + " (" +
                            str(static_cast<long long>(total)) + " blocks)");
    }
}

void bc4_blocks(const Lanes& lanes, std::size_t nblocks, Labels* provisional, std::uint8_t* out) {
    if (any_nonzero(lanes[1])) fail("BC4 lane 1 is non-zero (unexpected)");
    std::vector<std::array<int, 2>> hist(nblocks);
    const auto &l0 = lanes[0], &l2 = lanes[2], &l3 = lanes[3];
    for (std::size_t bi = 0; bi < nblocks; ++bi) {
        const long long i = static_cast<long long>(bi);
        const int v = l0[bi];
        const int d = (v & 1) ? 256 - ((v + 1) >> 1) : (v >> 1);
        std::array<int, 2> base = {128, 128};
        if (d == 0) {
        } else if (i - d >= 0) {
            base = hist[i - d];
        } else {
            const int entry = static_cast<int>(256 + i - d);
            const int state = bc4_history_state[entry];
            if (state == 1 || (state == 2 && provisional)) {
                base = {bc4_history[entry][0], bc4_history[entry][1]};
                if (state == 2) provisional->insert("BC4 history " + str(entry));
            } else {
                refuse("BC4 block " + str(i) + " references " + (state == 2 ? "unconfirmed" : "unknown") +
                       " built-in history entry " + str(entry));
            }
        }
        const int a0 = (base[0] + unzig(l2[bi])) & 255;
        const int a1 = (base[1] + unzig(l3[bi])) & 255;
        hist[bi] = {a0, a1};
        const std::uint16_t* rows_a[4] = {&lanes[4][bi], &lanes[6][bi], &lanes[8][bi], &lanes[10][bi]};
        const std::uint16_t* rows_b[4] = {&lanes[5][bi], &lanes[7][bi], &lanes[9][bi], &lanes[11][bi]};
        std::uint8_t* at = out + bi * 8;
        at[0] = static_cast<std::uint8_t>(a0);
        at[1] = static_cast<std::uint8_t>(a1);
        put_le(at + 2, pair_rows(bc4_pair, rows_a, rows_b), 6);
    }
}

void bc5_blocks(const Lanes& lanes, std::size_t nblocks, Labels* provisional, std::uint8_t* out) {
    if (any_nonzero(lanes[1])) fail("BC5 lane 1 is non-zero (unexpected)");
    constexpr std::array<int, 4> none = {128, 128, 128, 128};
    std::vector<std::array<int, 4>> hist(nblocks);
    std::vector<char> lost(nblocks, 0);
    std::set<int> unknown;
    const auto &l0 = lanes[0], &red = lanes[2], &green = lanes[3];
    for (std::size_t bi = 0; bi < nblocks; ++bi) {
        const long long i = static_cast<long long>(bi);
        const std::size_t a = 2 * bi, b = 2 * bi + 1;
        const int d = unzig(inter8(l0[a], l0[b])) & 255;
        std::array<int, 4> base = none;
        if (d == 0) {
        } else if (i - d >= 0) {
            base = hist[i - d];
            lost[bi] = lost[i - d];
        } else {
            const int entry = static_cast<int>(256 + i - d);
            const int state = bc5_history_state[entry];
            if (state == 1) {
                for (int k = 0; k < 4; ++k) base[k] = bc5_history[entry][k];
            } else if (state == 2) {
                if (!provisional)
                    refuse("BC5 block " + str(i) + " references unconfirmed built-in history entry " + str(entry));
                for (int k = 0; k < 4; ++k) base[k] = bc5_history[entry][k];
                provisional->insert("BC5 history " + str(entry));
            } else {
                if (!provisional)
                    refuse("BC5 block " + str(i) + " references unknown built-in history entry " + str(entry));
                lost[bi] = 1;
                unknown.insert(entry);
            }
        }
        const std::array<int, 4> e = {(base[0] + unzig(red[a])) & 255, (base[1] + unzig(red[b])) & 255,
                                      (base[2] + unzig(green[a])) & 255, (base[3] + unzig(green[b])) & 255};
        hist[bi] = e;
        std::uint8_t* at = out + bi * 16;
        for (int ch = 0; ch < 2; ++ch) {
            const std::uint16_t* rows_a[4];
            const std::uint16_t* rows_b[4];
            for (int r = 0; r < 4; ++r) {
                rows_a[r] = &lanes[4 + 4 * ch + r][a];
                rows_b[r] = &lanes[4 + 4 * ch + r][b];
            }
            at[8 * ch] = static_cast<std::uint8_t>(e[2 * ch]);
            at[8 * ch + 1] = static_cast<std::uint8_t>(e[2 * ch + 1]);
            put_le(at + 8 * ch + 2, pair_rows(bc5_pair, rows_a, rows_b), 6);
        }
    }
    if (!unknown.empty()) {
        const auto total = static_cast<std::size_t>(std::count(lost.begin(), lost.end(), 1));
        if (total > loose_limit(nblocks))
            refuse("BC5 chunk has " + str(static_cast<long long>(total)) + "/" + str(static_cast<long long>(nblocks)) +
                   " blocks from unknown built-in history entries");
        provisional->insert("BC5 unknown history " + sorted_list(unknown) + " (" +
                            str(static_cast<long long>(total)) + " blocks)");
    }
}

// --- BC6H -----------------------------------------------------------------------------------------------------------

// BC6H initial history as bcpack_tables_bptc.load_bc6h(None, with_confirmed_bits=True) returns it: all unknown.
struct Bc6hHistory {
    std::uint16_t endpoints[256][3][4] = {};
    std::uint8_t indices[256][16] = {};
    std::uint8_t endpoint_class[256] = {};
    std::uint8_t index_class[256] = {};
    std::uint8_t endpoint_confirmed_bits[256][3][4] = {};
    std::uint8_t index_confirmed_bits[256][16] = {};
};

const Bc6hHistory& builtin_bc6h_history();

constexpr const char* history_hint = " (pass --history <path>)";

// Map BCPack's base endpoint `w` into the D3D11/Khronos BC6H bit order (bcpack._bc6h_stream_w).
int bc6h_stream_w(int mode, int w) {
    if (mode == 13) return (w & 0x3FF) | (((w >> 10) & 1) << 11) | (((w >> 11) & 1) << 10);
    if (mode == 14) {
        int out = w & 0x3FF;
        for (int b = 10; b < 16; ++b) out |= ((w >> b) & 1) << (25 - b);
        return out;
    }
    return w;
}

// First-reference order and per-entry counts of the built-in history entries a chunk's blocks resolve to
// (the insertion-ordered `ends` / `idxs` dicts of bcpack._bc6h_blocks).
struct RootCounts {
    std::array<int, 256> count{};
    std::vector<int> order;
    void add(int root) {
        if (count[root]++ == 0) order.push_back(root);
    }
};

void bc6h_blocks(const Lanes& lanes, std::size_t nblocks, Labels* provisional, std::uint8_t* out) {
    if (lanes[0].size() != 2 * nblocks) fail("BC6H lane length does not match the block count");
    const Bc6hHistory& hist = builtin_bc6h_history();
    using Endpoints = std::array<std::array<int, 4>, 3>;
    const Endpoints none_e = {{{0x8000, 0x200, 0x20, 0x20}, {0x8000, 0x200, 0x20, 0x20}, {0x8000, 0x200, 0x20, 0x20}}};
    std::array<int, 16> none_i;
    none_i.fill(8);
    std::vector<Endpoints> st_e(nblocks);
    std::vector<std::array<int, 16>> st_i(nblocks);
    std::vector<int> eroot(nblocks, -1), iroot(nblocks, -1);
    RootCounts ends, idxs;
    std::array<bool, 256> weak_e{}, weak_i{};
    std::size_t violations = 0;
    const auto &l0 = lanes[0], &l1 = lanes[1], &l15 = lanes[15];

    for (std::size_t bi = 0; bi < nblocks; ++bi) {
        const long long i = static_cast<long long>(bi);
        const std::size_t a = 2 * bi, b = 2 * bi + 1;
        const int m_sym = l15[a];
        if (m_sym > 13) fail("BC6H block " + str(i) + " mode symbol " + str(m_sym) + " outside 0..13");
        const int mode = m_sym + 1;
        const auto& info = bc6h_mode_info[mode];
        const int code = info[0], trans = info[1], base = info[2], reg = info[6], header = info[7];
        const int deltas[3] = {info[3], info[4], info[5]};
        const int part = l15[b] < 31 ? l15[b] : 31;

        const int di = inter8(l1[a], l1[b]);
        const int de = (di + unzig(inter8(l0[a], l0[b]))) & 255;
        Endpoints eref;
        if (de == 0) {
            eref = none_e;
        } else if (i >= de) {
            eref = st_e[i - de];
            eroot[bi] = eroot[i - de];
        } else {
            const int entry = static_cast<int>(256 + i - de);
            for (int c = 0; c < 3; ++c)
                for (int k = 0; k < 4; ++k) eref[c][k] = hist.endpoints[entry][c][k];
            eroot[bi] = entry;
            // Only direct initial-history reads consume the calibrated bits.
            for (int c = 0; c < 3; ++c) {
                const int dc = deltas[c];
                const int need[4] = {base, dc, reg == 2 ? dc : 0, reg == 2 ? dc : 0};
                for (int qi = 0; qi < 4; ++qi)
                    if (hist.endpoint_confirmed_bits[entry][c][qi] < need[qi]) weak_e[entry] = true;
            }
        }
        if (eroot[bi] >= 0) ends.add(eroot[bi]);

        std::array<int, 16> iref;
        if (di == 0) {
            iref = none_i;
        } else if (i >= di) {
            iref = st_i[i - di];
            iroot[bi] = iroot[i - di];
        } else {
            const int entry = static_cast<int>(256 + i - di);
            for (int p = 0; p < 16; ++p) iref[p] = hist.indices[entry][p];
            iroot[bi] = entry;
            for (int p = 0; p < 16; ++p)
                if (hist.index_confirmed_bits[entry][p] < (reg == 2 ? 3 : 4)) weak_i[entry] = true;
        }
        if (iroot[bi] >= 0) idxs.add(iroot[bi]);

        const int mw = (1 << base) - 1;
        const int w_shift = 16 - base;
        std::array<std::array<int, 4>, 3> f_block{};
        Endpoints st_block{};
        for (int c = 0; c < 3; ++c) {
            const int dc = deltas[c];
            const int mD = (1 << dc) - 1;
            const int sh = dc < 6 ? 6 - dc : 0;
            const int dz0 = unzig(inter(lanes[2 + c][a], lanes[2 + c][b], 8));
            const int dz1 = unzig(inter(lanes[5 + c][a], lanes[5 + c][b], 5));
            const int dz2 = unzig(lanes[8 + c][a]);
            const int dz3 = unzig(lanes[8 + c][b]);
            const auto& r = eref[c];
            const int w = ((r[0] >> w_shift) + dz0) & mw;
            const int x = ((r[1] >> (10 - dc)) + dz1) & mD;
            const int y = ((r[2] >> sh) + dz2) & mD;
            const int z = ((r[3] >> sh) + dz3) & mD;
            const int x10 = x << (10 - dc);
            if (reg == 2) {
                st_block[c] = {w << w_shift, x10, y << sh, z << sh};
                f_block[c] = {bc6h_stream_w(mode, w), x, y, z};
            } else {
                const int v = (w & mD) ^ (trans == 1 ? (1 << (dc - 1)) : 0);
                const int low_y = dc >= 6 ? (v >> (dc - 6)) : (v << sh);
                st_block[c] = {w << w_shift, x10, low_y, x10 >> 4};
                f_block[c] = {bc6h_stream_w(mode, w), x, 0, 0};
            }
        }
        st_e[bi] = st_block;

        const int ibits = reg == 2 ? 3 : 4;
        const int imask = (1 << ibits) - 1;
        const int ishift = 4 - ibits;
        std::array<int, 16> ix_block{}, si_block{};
        for (int yy = 0; yy < 4; ++yy) {
            const int sa = lanes[11 + yy][a];
            const int sb = lanes[11 + yy][b];
            int dv[4];
            if (yy < 3) {
                dv[0] = deint[1][sa], dv[1] = deint[0][sa], dv[2] = deint[1][sb], dv[3] = deint[0][sb];
            } else {
                dv[0] = deint[0][sa], dv[1] = deint[1][sa], dv[2] = deint[0][sb], dv[3] = deint[1][sb];
            }
            for (int off = 0; off < 4; ++off) {
                const int p = yy * 4 + off;
                const int ix = ((iref[p] >> ishift) + unzig(dv[off])) & imask;
                ix_block[p] = ix;
                si_block[p] = ix << ishift;
            }
        }
        st_i[bi] = si_block;

        const int second = reg == 2 ? bc7_anchor2[part] : -1;
        if (ix_block[0] & (reg == 2 ? 4 : 8)) ++violations;
        if (reg == 2 && (ix_block[second] & 4)) ++violations;

        std::uint8_t* blk = out + 16 * bi;
        std::memset(blk, 0, 16);
        put_bits(blk, 0, static_cast<std::uint64_t>(code), 5);
        for (int k = 0; k < bc6h_plan_count[mode]; ++k) {
            const auto& item = bc6h_plan[mode][k];
            const int c = item[0], qi = item[1], bit = item[2], pos = item[3];
            const int value = c < 0 ? part : f_block[c][qi];
            if ((value >> bit) & 1) put_bits(blk, pos, 1, 1);
        }
        int at = header;
        for (int p = 0; p < 16; ++p) {
            const int nb = (p == 0 || p == second) ? ibits - 1 : ibits;
            put_bits(blk, at, static_cast<std::uint64_t>(ix_block[p] & ((1 << nb) - 1)), nb);
            at += nb;
        }
    }

    if (!provisional) {
        std::vector<int> unconf_e, unconf_i;
        for (int k : ends.order)
            if (hist.endpoint_class[k] != 3 || weak_e[k]) unconf_e.push_back(k);
        for (int k : idxs.order)
            if (hist.index_class[k] != 3 || weak_i[k]) unconf_i.push_back(k);
        if (!unconf_e.empty() || !unconf_i.empty()) {
            const bool endpoint = !unconf_e.empty();
            const int k = endpoint ? unconf_e[0] : unconf_i[0];
            const int cls = endpoint ? hist.endpoint_class[k] : hist.index_class[k];
            refuse(std::string("BC6H chunk references ") + (cls == 0 ? "unknown" : "unconfirmed") + " built-in " +
                   (endpoint ? "endpoint" : "index") + " history entry " + str(k) + history_hint);
        }
        if (violations)
            fail("BC6H chunk has " + str(static_cast<long long>(violations)) +
                 " anchor pixel(s) with an inconsistent index");
        return;
    }

    std::array<bool, 256> unknown_e{}, unknown_i{};
    std::set<std::string> missing_set;
    for (int k : ends.order)
        if (hist.endpoint_class[k] == 0) unknown_e[k] = true, missing_set.insert("endpoint " + str(k));
    for (int k : idxs.order)
        if (hist.index_class[k] == 0) unknown_i[k] = true, missing_set.insert("index " + str(k));
    std::size_t lost = 0;
    for (std::size_t bi = 0; bi < nblocks; ++bi)
        if ((eroot[bi] >= 0 && unknown_e[eroot[bi]]) || (iroot[bi] >= 0 && unknown_i[iroot[bi]])) ++lost;
    const std::vector<std::string> missing(missing_set.begin(), missing_set.end());
    if (lost > loose_limit(nblocks))
        refuse("BC6H chunk has " + str(static_cast<long long>(lost)) + "/" + str(static_cast<long long>(nblocks)) +
               " blocks from unknown built-in history entries (" + join(missing, 0, 6) +
               (missing.size() > 6 ? ", ..." : "") + ")" + history_hint);
    if (violations > loose_limit(nblocks))
        refuse("BC6H chunk has " + str(static_cast<long long>(violations)) +
               " anchor pixels with an inconsistent index");
    if (!missing.empty())
        provisional->insert("BC6H unknown history " + join(missing, 0, missing.size()) + " (" +
                            str(static_cast<long long>(lost)) + " blocks)");
    if (violations)
        provisional->insert("BC6H anchor index inconsistent in " + str(static_cast<long long>(violations)) + " pixels");
    for (int k = 0; k < 256; ++k) {
        if (weak_e[k] && hist.endpoint_class[k] == 3)
            provisional->insert("BC6H endpoint history " + str(k) + " (unconfirmed bits)");
        if (weak_i[k] && hist.index_class[k] == 3)
            provisional->insert("BC6H index history " + str(k) + " (unconfirmed bits)");
    }
    const std::pair<const char*, const RootCounts*> kinds[2] = {{"endpoint", &ends}, {"index", &idxs}};
    for (int kind = 0; kind < 2; ++kind) {
        const std::uint8_t* cls = kind == 0 ? hist.endpoint_class : hist.index_class;
        std::vector<int> keys = kinds[kind].second->order;
        std::sort(keys.begin(), keys.end());
        std::vector<std::string> fitted, rough;
        for (int k : keys) {
            if (cls[k] == 1) fitted.push_back(str(k));
            if (cls[k] == 2) rough.push_back(str(k));
        }
        const std::string name = std::string("BC6H ") + kinds[kind].first + " history ";
        if (!fitted.empty()) provisional->insert(name + join(fitted, 0, fitted.size()));
        if (!rough.empty()) provisional->insert(name + join(rough, 0, rough.size()) + " (last-resort)");
    }
}

// --- BC7 ------------------------------------------------------------------------------------------------------------

constexpr int b7_ns[8] = {3, 2, 3, 2, 1, 1, 1, 2};
constexpr int b7_cb[8] = {4, 6, 5, 7, 5, 7, 7, 5};
constexpr int b7_ab[8] = {0, 0, 0, 0, 6, 8, 7, 5};
constexpr int b7_pk[8] = {1, 2, 0, 1, 0, 0, 1, 1};  // 0 = none, 1 = per-endpoint p-bit, 2 = shared per subset
constexpr int b7_ib[8] = {3, 3, 2, 2, 3, 2, 4, 2};  // index bits of the primary index set (lanes 11..14)
constexpr int b7_pb[8] = {4, 6, 6, 6, 0, 0, 0, 6};  // partition field bits
constexpr int b7_lane[3][4] = {{2, 3, 4, 5}, {6, 7, 8, 9}, {10, 5, 9, -1}};
constexpr int b7_anchor = 8;
constexpr int b7_pbit = 16;
constexpr int b7_signature = 32;
constexpr int b7_min_depth = 4;

// BC7 initial history as bcpack_tables_bptc.load_bc7(None) returns it: depth 0 (unknown) everywhere.
struct Bc7History {
    std::uint8_t endpoints[256][24] = {};
    std::uint8_t endpoint_bits[256][24] = {};
    std::uint8_t endpoint_confirmed_bits[256][24] = {};
    std::uint8_t indices[256][32] = {};
    std::uint8_t index_bits[256][32] = {};
    std::uint8_t index_confirmed_bits[256][32] = {};
    std::set<std::pair<int, int>> last_resort;  // (entry, slot)
};

// LEONIDA_BCPACK_HISTORY: a local file with the initial history calibrated from the player's own pools
// (tools/bcpack_history_calib.py); absent, every entry is unknown.
struct LoadedHistory {
    Bc7History bc7;
    Bc6hHistory bc6h;
};
const LoadedHistory& loaded_history() {
    static const LoadedHistory loaded = [] {
        LoadedHistory h;
        const char* path = std::getenv("LEONIDA_BCPACK_HISTORY");
        FILE* f = path ? std::fopen(path, "rb") : nullptr;
        if (!f) return h;
        char magic[4];
        bool ok = std::fread(magic, 1, 4, f) == 4 && std::memcmp(magic, "LBH1", 4) == 0;
        const auto take = [&](void* to, std::size_t n) { ok = ok && std::fread(to, 1, n, f) == n; };
        take(h.bc7.endpoints, sizeof(h.bc7.endpoints));
        take(h.bc7.endpoint_bits, sizeof(h.bc7.endpoint_bits));
        take(h.bc7.endpoint_confirmed_bits, sizeof(h.bc7.endpoint_confirmed_bits));
        take(h.bc7.indices, sizeof(h.bc7.indices));
        take(h.bc7.index_bits, sizeof(h.bc7.index_bits));
        take(h.bc7.index_confirmed_bits, sizeof(h.bc7.index_confirmed_bits));
        take(h.bc6h.endpoints, sizeof(h.bc6h.endpoints));
        take(h.bc6h.indices, sizeof(h.bc6h.indices));
        take(h.bc6h.endpoint_class, sizeof(h.bc6h.endpoint_class));
        take(h.bc6h.index_class, sizeof(h.bc6h.index_class));
        take(h.bc6h.endpoint_confirmed_bits, sizeof(h.bc6h.endpoint_confirmed_bits));
        take(h.bc6h.index_confirmed_bits, sizeof(h.bc6h.index_confirmed_bits));
        std::uint32_t n = 0;
        take(&n, 4);
        for (std::uint32_t i = 0; ok && i < n; ++i) {
            std::uint16_t pair[2];
            take(pair, 4);
            if (ok) h.bc7.last_resort.insert({pair[0], pair[1]});
        }
        std::fclose(f);
        std::fprintf(stderr, "[rt] bcpack: initial history %s from %s\n", ok ? "loaded" : "NOT loaded", path);
        return ok ? h : LoadedHistory{};
    }();
    return loaded;
}
const Bc7History& builtin_bc7_history() { return loaded_history().bc7; }
const Bc6hHistory& builtin_bc6h_history() { return loaded_history().bc6h; }

int bc7_rank(int a, int b) { return (a < 32 && b < 16) ? bc7_rank_of[a * 16 + b] : -1; }

bool bc7_inferred_rank(int rank) {
    return std::find(std::begin(bc7_inferred), std::end(bc7_inferred), rank) != std::end(bc7_inferred);
}

// Leading bits read from a stored endpoint byte for channels (R, G, B, A).
std::array<int, 4> bc7_field_bits(int mode, int rot) {
    std::array<int, 4> w = {b7_cb[mode], b7_cb[mode], b7_cb[mode], b7_ab[mode]};
    if (rot) w[rot - 1] = b7_ab[mode], w[3] = b7_cb[mode];
    const int pk = b7_pk[mode];
    std::array<int, 4> out{};
    for (int c = 0; c < 4; ++c) out[c] = w[c] ? w[c] + ((pk && c == 0) ? 1 : 0) : 0;
    return out;
}

int bc7_delta_e(const Lanes& lanes, std::size_t i, int di) {
    return (di + unzig(inter8(lanes[0][2 * i], lanes[0][2 * i + 1]))) & 255;
}

int bc7_delta_i(const Lanes& lanes, std::size_t i) { return inter8(lanes[1][2 * i], lanes[1][2 * i + 1]); }

// Signature and built-in history coverage (bcpack._bc7_check). Returns the (kind, entry) pairs, kind 0 =
// endpoint and 1 = index, too shallow to use, in provisional mode.
std::vector<std::pair<int, int>> bc7_check(const Lanes& lanes, std::size_t nblocks, const Bc7History& hist,
                                           Labels* provisional) {
    const auto& l15 = lanes[15];
    std::set<int> ranks;
    for (std::size_t i = 0; i < nblocks; ++i) {
        const int a = l15[2 * i], b = l15[2 * i + 1];
        const int rank = bc7_rank(a, b);
        if (rank < 0 || bc7_config[rank][0] > 7 || bc7_config[rank][1] == 255)
            fail("BC7 signature (" + str(a) + ", " + str(b) + ") is outside the signature table");
        ranks.insert(rank);
    }
    for (int rank : ranks) {
        if (!bc7_inferred_rank(rank)) continue;
        if (!provisional) refuse("BC7 signature rank " + str(rank) + " is inferred-only (unconfirmed)");
        provisional->insert("BC7 signature " + str(rank));
    }

    std::map<int, std::array<std::array<int, 4>, 3>> need_e;
    std::map<int, std::array<int, 2>> need_i;
    for (std::size_t bi = 0; bi < std::min<std::size_t>(nblocks, 256); ++bi) {
        const int i = static_cast<int>(bi);
        const int rank = bc7_rank_of[l15[2 * bi] * 16 + l15[2 * bi + 1]];
        const int mode = bc7_config[rank][0], rot = bc7_config[rank][2];
        const int di = bc7_delta_i(lanes, bi);
        const int de = bc7_delta_e(lanes, bi, di);
        if (de > i) {
            const auto bits = bc7_field_bits(mode, rot);
            auto& row = need_e[256 + i - de];
            for (int sub = 0; sub < b7_ns[mode]; ++sub)
                for (int c = 0; c < 4; ++c) row[sub][c] = std::max(row[sub][c], bits[c]);
        }
        if (di > i) {
            auto& row = need_i[256 + i - di];
            row[0] = std::max(row[0], b7_ib[mode]);
            if (mode == 4 || mode == 5) row[1] = 2;
        }
    }

    std::vector<std::pair<int, int>> unknown;
    for (int kind = 0; kind < 2; ++kind) {
        const char* kinds = kind == 0 ? "endpoint" : "index";
        std::vector<std::pair<int, std::vector<std::pair<int, int>>>> needs;  // entry -> (slot, bits)
        if (kind == 0) {
            for (const auto& [k, want] : need_e) {
                std::vector<std::pair<int, int>> slots;
                for (int sub = 0; sub < 3; ++sub)
                    for (int e = 0; e < 2; ++e)
                        for (int c = 0; c < 4; ++c)
                            if (want[sub][c]) slots.emplace_back(sub * 8 + e * 4 + c, want[sub][c]);
                needs.emplace_back(k, std::move(slots));
            }
        } else {
            for (const auto& [k, want] : need_i) {
                std::vector<std::pair<int, int>> slots;
                for (int st = 0; st < 2; ++st)
                    for (int p = 0; p < 16; ++p)
                        if (want[st]) slots.emplace_back(st * 16 + p, want[st]);
                needs.emplace_back(k, std::move(slots));
            }
        }
        for (const auto& [k, slots] : needs) {
            const std::uint8_t* bk = kind == 0 ? hist.endpoint_bits[k] : hist.index_bits[k];
            const std::uint8_t* ck = kind == 0 ? hist.endpoint_confirmed_bits[k] : hist.index_confirmed_bits[k];
            bool weak = false, missing = false;
            int shallow = 8;
            for (const auto& [j, b] : slots) {
                if (bk[j] < b) {
                    missing = true;
                    shallow = std::min<int>(shallow, bk[j]);
                }
                if (ck[j] < b) weak = true;
            }
            if (missing) {
                if (!provisional)
                    refuse(std::string("BC7 chunk references unknown built-in ") + kinds + " history entry " + str(k) +
                           history_hint);
                if (kind == 0 && shallow >= b7_min_depth)
                    provisional->insert("BC7 endpoint history " + str(k) + " (top " + str(shallow) + " bits known)");
                else
                    unknown.emplace_back(kind, k);
            } else if (weak) {
                if (!provisional)
                    refuse(std::string("BC7 chunk references unconfirmed built-in ") + kinds + " history entry " +
                           str(k));
                provisional->insert(std::string("BC7 ") + kinds + " history " + str(k));
            }
            if (kind == 0 && provisional) {
                for (const auto& [j, b] : slots) {
                    if (hist.last_resort.count({k, j})) {
                        provisional->insert("BC7 endpoint history " + str(k) + " (last-resort)");
                        break;
                    }
                }
            }
        }
    }
    return unknown;
}

// Rebuilds the 16-byte BC7 blocks and per-block consistency status flags (bcpack._bc7_rebuild).
std::vector<int> bc7_rebuild(const Lanes& lanes, std::size_t nblocks, const Bc7History& hist, std::uint8_t* out) {
    std::vector<int> status(nblocks, 0);
    std::vector<std::array<int, 24>> eps(nblocks);
    std::vector<std::array<int, 32>> ixs(nblocks);
    const std::array<int, 24> zero_e{};
    const std::array<int, 32> zero_i{};
    const std::vector<std::uint16_t>* rows1[4] = {&lanes[11], &lanes[12], &lanes[13], &lanes[14]};
    const std::vector<std::uint16_t>* rows2[4] = {&lanes[6], &lanes[7], &lanes[8], &lanes[10]};
    const auto& l15 = lanes[15];

    for (std::size_t bi = 0; bi < nblocks; ++bi) {
        const long long i = static_cast<long long>(bi);
        const std::size_t a = 2 * bi, b = 2 * bi + 1;
        const int rank = bc7_rank(l15[a], l15[b]);
        if (rank < 0 || bc7_config[rank][0] > 7) {
            status[bi] = b7_signature;
            eps[bi] = zero_e;
            ixs[bi] = zero_i;
            continue;
        }
        const int mode = bc7_config[rank][0], part = bc7_config[rank][1], rot = bc7_config[rank][2],
                  isel = bc7_config[rank][3];
        int st = 0;
        const int di = bc7_delta_i(lanes, bi);
        const int de = bc7_delta_e(lanes, bi, di);
        std::array<int, 24> rep = zero_e;
        if (de != 0) {
            if (i >= de)
                rep = eps[i - de];
            else
                for (int k = 0; k < 24; ++k) rep[k] = hist.endpoints[256 + i - de][k];
        }
        std::array<int, 32> rix = zero_i;
        if (di != 0) {
            if (i >= di)
                rix = ixs[i - di];
            else
                for (int k = 0; k < 32; ++k) rix[k] = hist.indices[256 + i - di][k];
        }

        const int ns = b7_ns[mode], cb = b7_cb[mode], ab = b7_ab[mode], pk = b7_pk[mode];
        int w[4] = {cb, cb, cb, ab};
        if (rot) w[rot - 1] = ab, w[3] = cb;

        int val[3][2][4] = {};
        std::array<int, 24> sep{};
        int pb[3][2] = {};
        for (int sub = 0; sub < ns; ++sub) {
            const int base_sub = sub * 8;
            for (int c = 0; c < 4; ++c) {
                const int lane_idx = b7_lane[sub][c];
                const int wc = w[c];
                if (!wc || lane_idx < 0) continue;
                const int bits = wc + ((pk && c == 0) ? 1 : 0);
                const int shift = 8 - bits;
                const int mask = (1 << bits) - 1;
                const auto& ln = lanes[lane_idx];
                const int v0 = ((rep[base_sub + c] >> shift) + unzig(ln[a])) & mask;
                const int v1 = ((rep[base_sub + 4 + c] >> shift) + unzig(ln[b])) & mask;
                val[sub][0][c] = v0;
                val[sub][1][c] = v1;
                sep[base_sub + c] = v0 << shift;
                sep[base_sub + 4 + c] = v1 << shift;
            }
            if (pk == 1) {
                pb[sub][0] = val[sub][0][0] & 1;
                pb[sub][1] = val[sub][1][0] & 1;
            } else if (pk == 2) {
                const int p0 = val[sub][0][0] & 1;
                pb[sub][0] = p0;
                pb[sub][1] = p0;
                if ((val[sub][1][0] & 1) != p0) st |= b7_pbit;
            }
        }
        for (int sub = ns; sub < 3; ++sub)
            for (int k = 0; k < 8; ++k) sep[sub * 8 + k] = sep[k];
        eps[bi] = sep;

        const int b1 = b7_ib[mode];
        const int b1_shift = 4 - b1;
        const int b1_mask = (1 << b1) - 1;
        const bool two = mode == 4 || mode == 5;
        std::array<int, 16> s1{}, s2{};
        std::array<int, 32> six{};
        for (int y = 0; y < 4; ++y) {
            const int sa = (*rows1[y])[a], sb = (*rows1[y])[b];
            int dv[4];
            if (y < 3)
                dv[0] = deint[1][sa], dv[1] = deint[0][sa], dv[2] = deint[1][sb], dv[3] = deint[0][sb];
            else
                dv[0] = deint[0][sa], dv[1] = deint[1][sa], dv[2] = deint[0][sb], dv[3] = deint[1][sb];
            for (int off = 0; off < 4; ++off) {
                const int p = y * 4 + off;
                const int iv = ((rix[p] >> b1_shift) + unzig(dv[off])) & b1_mask;
                s1[p] = iv;
                six[p] = iv << b1_shift;
            }
            if (two) {
                const int s2a = (*rows2[y])[a], s2b = (*rows2[y])[b];
                int ev[4];
                if (y < 3)
                    ev[0] = deint[1][s2a], ev[1] = deint[0][s2a], ev[2] = deint[1][s2b], ev[3] = deint[0][s2b];
                else
                    ev[0] = deint[0][s2a], ev[1] = deint[1][s2a], ev[2] = deint[0][s2b], ev[3] = deint[1][s2b];
                for (int off = 0; off < 4; ++off) {
                    const int p = y * 4 + off;
                    const int iv2 = ((rix[16 + p] >> 2) + unzig(ev[off])) & 3;
                    s2[p] = iv2;
                    six[16 + p] = iv2 << 2;
                }
            }
        }
        ixs[bi] = six;

        auto all_zero = [](const auto& arr) { return std::all_of(arr.begin(), arr.end(), [](int v) { return !v; }); };
        if (mode == 0 && part == 0 && all_zero(sep) && all_zero(s1)) {
            status[bi] = st;
            continue;
        }

        // Subset of each pixel and the anchor pixel of each subset.
        int sm[16] = {};
        if (ns == 2)
            for (int p = 0; p < 16; ++p) sm[p] = (bc7_part2_packed[part] >> p) & 1;
        else if (ns == 3)
            for (int p = 0; p < 16; ++p) sm[p] = (bc7_part3_packed[part] >> (2 * p)) & 3;
        const int anchors[3] = {0, ns == 2 ? bc7_anchor2[part] : (ns == 3 ? bc7_anchor3a[part] : 0),
                                ns == 3 ? bc7_anchor3b[part] : 0};
        const std::array<int, 16>* pri = &s1;
        const std::array<int, 16>* sec = nullptr;
        int ib = b1, ib2 = 0;
        if (mode == 4) {
            pri = &s2, ib = 2, sec = &s1, ib2 = 3;
        } else if (mode == 5) {
            pri = &s1, ib = 2, sec = &s2, ib2 = 2;
        }

        std::uint8_t* blk = out + 16 * bi;
        std::memset(blk, 0, 16);
        put_bits(blk, mode, 1, 1);
        int pos = mode + 1;
        const int pbits = b7_pb[mode];
        if (pbits) {
            put_bits(blk, pos, static_cast<std::uint64_t>(part & ((1 << pbits) - 1)), pbits);
            pos += pbits;
        }
        if (mode == 4 || mode == 5) {
            put_bits(blk, pos, static_cast<std::uint64_t>(rot & 3), 2);
            pos += 2;
        }
        if (mode == 4) {
            put_bits(blk, pos, static_cast<std::uint64_t>(isel & 1), 1);
            pos += 1;
        }
        const int cb_mask = (1 << cb) - 1;
        for (int c = 0; c < 3; ++c) {
            const int tc = (rot && c == rot - 1) ? 3 : c;
            const bool drop = pk && tc == 0;
            for (int sub = 0; sub < ns; ++sub) {
                for (int e = 0; e < 2; ++e) {
                    const int v = val[sub][e][tc];
                    put_bits(blk, pos, static_cast<std::uint64_t>((drop ? v >> 1 : v) & cb_mask), cb);
                    pos += cb;
                }
            }
        }
        if (ab) {
            const int ab_mask = (1 << ab) - 1;
            const int tc = rot ? rot - 1 : 3;
            const bool drop = pk && tc == 0;
            for (int sub = 0; sub < ns; ++sub) {
                for (int e = 0; e < 2; ++e) {
                    const int v = val[sub][e][tc];
                    put_bits(blk, pos, static_cast<std::uint64_t>((drop ? v >> 1 : v) & ab_mask), ab);
                    pos += ab;
                }
            }
        }
        if (pk == 1) {
            for (int sub = 0; sub < ns; ++sub) {
                put_bits(blk, pos++, static_cast<std::uint64_t>(pb[sub][0] & 1), 1);
                put_bits(blk, pos++, static_cast<std::uint64_t>(pb[sub][1] & 1), 1);
            }
        } else if (pk == 2) {
            for (int sub = 0; sub < ns; ++sub) put_bits(blk, pos++, static_cast<std::uint64_t>(pb[sub][0] & 1), 1);
        }

        const int ib_mask = (1 << ib) - 1;
        const int ib_amask = (1 << (ib - 1)) - 1;
        for (int p = 0; p < 16; ++p) {
            const bool anchor = p == anchors[ns > 1 ? sm[p] : 0];
            const int pv = (*pri)[p];
            if (anchor) {
                if (pv >> (ib - 1)) st |= b7_anchor;
                put_bits(blk, pos, static_cast<std::uint64_t>(pv & ib_amask), ib - 1);
                pos += ib - 1;
            } else {
                put_bits(blk, pos, static_cast<std::uint64_t>(pv & ib_mask), ib);
                pos += ib;
            }
        }
        if (sec) {
            const int ib2_mask = (1 << ib2) - 1;
            const int ib2_amask = (1 << (ib2 - 1)) - 1;
            for (int p = 0; p < 16; ++p) {
                const int sv = (*sec)[p];
                if (p == 0) {
                    if (sv >> (ib2 - 1)) st |= b7_anchor;
                    put_bits(blk, pos, static_cast<std::uint64_t>(sv & ib2_amask), ib2 - 1);
                    pos += ib2 - 1;
                } else {
                    put_bits(blk, pos, static_cast<std::uint64_t>(sv & ib2_mask), ib2);
                    pos += ib2;
                }
            }
        }
        if (pos != 128) st |= b7_signature;
        status[bi] = st;
    }
    return status;
}

void bc7_blocks(const Lanes& lanes, std::size_t nblocks, Labels* provisional, std::uint8_t* out) {
    if (lanes[0].size() != 2 * nblocks) fail("BC7 lane length does not match the block count");
    const Bc7History& hist = builtin_bc7_history();
    const auto unknown = bc7_check(lanes, nblocks, hist, provisional);
    if (!unknown.empty()) {
        std::vector<int> er(nblocks, 0), ir(nblocks, 0);
        for (std::size_t bi = 0; bi < nblocks; ++bi) {
            const long long i = static_cast<long long>(bi);
            const int di = bc7_delta_i(lanes, bi);
            const int de = bc7_delta_e(lanes, bi, di);
            er[bi] = de == 0 ? -1 : (de > i ? static_cast<int>(256 + i - de) : er[i - de]);
            ir[bi] = di == 0 ? -1 : (di > i ? static_cast<int>(256 + i - di) : ir[i - di]);
        }
        std::array<bool, 256> unknown_e{}, unknown_i{};
        std::vector<std::string> names;
        std::vector<std::string> labels;
        for (const auto& [kind, k] : unknown) {
            const auto& roots = kind == 0 ? er : ir;
            const auto count = std::count(roots.begin(), roots.end(), k);
            (kind == 0 ? unknown_e : unknown_i)[k] = true;
            const std::string name = std::string(kind == 0 ? "endpoint " : "index ") + str(k);
            names.push_back(name);
            labels.push_back("BC7 unknown " + name.substr(0, name.find(' ')) + " history " + str(k) + " (" +
                             str(static_cast<long long>(count)) + " blocks)");
        }
        std::size_t total = 0;
        for (std::size_t bi = 0; bi < nblocks; ++bi)
            if ((er[bi] >= 0 && unknown_e[er[bi]]) || (ir[bi] >= 0 && unknown_i[ir[bi]])) ++total;
        if (total > loose_limit(nblocks))
            refuse("BC7 chunk has " + str(static_cast<long long>(total)) + "/" + str(static_cast<long long>(nblocks)) +
                   " blocks from unknown built-in history entries (" + join(names, 0, 6) +
                   (names.size() > 6 ? ", ..." : "") + ")" + history_hint);
        provisional->insert(labels.begin(), labels.end());
    }
    const auto status = bc7_rebuild(lanes, nblocks, hist, out);
    std::size_t bad = 0, first = 0;
    for (std::size_t bi = 0; bi < nblocks; ++bi) {
        if (status[bi]) {
            if (!bad) first = bi;
            ++bad;
        }
    }
    if (bad) {
        const std::string where = str(static_cast<long long>(bad)) + "/" + str(static_cast<long long>(nblocks));
        if (!(status[first] & (b7_anchor | b7_pbit)))
            fail("BC7 chunk failed signature check (" + where + " blocks; first: block " +
                 str(static_cast<long long>(first)) + ")");
        if (!provisional || provisional->empty())
            fail("BC7 chunk failed anchor/p-bit consistency check (" + where + " blocks; first: block " +
                 str(static_cast<long long>(first)) + ")");
        if (bad > loose_limit(nblocks))
            refuse("BC7 chunk failed anchor/p-bit consistency check in " + where + " blocks");
        provisional->insert("BC7 consistency check failed in " + str(static_cast<long long>(bad)) + " blocks");
    }
}

}  // namespace

std::size_t block_bytes(int codec) {
    switch (codec) {
    case 1:
    case 4: return 8;
    case 3:
    case 5:
    case 6:
    case 7: return 16;
    default: return 0;
    }
}

Result decode_lanes(std::span<const std::uint8_t> raw, int codec, std::size_t out_size,
                    std::vector<std::vector<std::uint16_t>>& lanes, std::size_t* bytes_used) {
    Result result;
    try {
        const std::size_t used = run_decode_lanes(raw, codec, out_size, lanes);
        if (bytes_used) *bytes_used = used;
    } catch (const Failure& f) {
        result.status = f.status;
        result.message = f.message;
    }
    return result;
}

Result decode(std::span<const std::uint8_t> raw, int codec, std::span<std::uint8_t> out, bool allow_provisional) {
    Result result;
    const char* name = codec_name(codec);
    if (!name) {
        result.status = Status::unsupported;
        result.message = "BCPack block rebuilder for codec " + str(codec) + " (" + str(codec) +
                         ") is not implemented yet";
        return result;
    }
    try {
        Lanes lanes;
        run_decode_lanes(raw, codec, out.size(), lanes);
        const std::size_t nblocks = out.size() / block_bytes(codec);
        Labels labels;
        Labels* provisional = allow_provisional ? &labels : nullptr;
        std::fill(out.begin(), out.end(), std::uint8_t{0});
        switch (codec) {
        case 1: bc1_blocks(lanes, nblocks, provisional, out.data()); break;
        case 3: bc3_blocks(lanes, nblocks, provisional, out.data()); break;
        case 4: bc4_blocks(lanes, nblocks, provisional, out.data()); break;
        case 5: bc5_blocks(lanes, nblocks, provisional, out.data()); break;
        case 6: bc6h_blocks(lanes, nblocks, provisional, out.data()); break;
        case 7: bc7_blocks(lanes, nblocks, provisional, out.data()); break;
        }
        result.provisional.assign(labels.begin(), labels.end());
    } catch (const Failure& f) {
        result.status = f.status;
        result.message = f.message;
    }
    return result;
}

}  // namespace leonida::bcpack
