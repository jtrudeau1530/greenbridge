"""Clean-room BCPack bitstream decoder (Python standard library only).

Implements the Xbox Series DirectStorage BCPack entropy layer (interleaved
12/16-lane 16-bit rANS with 8-byte/cycle prefetch scheduling and exponential
frequency tables) for all six active modes (BC1, BC3, BC4, BC5, BC6H, BC7) and
block rebuilders for BC1, BC3, BC4 and BC5 (codecs 1, 3, 4 and 5).

Codec identifiers follow `DSTORAGE_BCPACK_MODE` (matching `tools/ttpf_mips.py`
and `docs/textures.md`):
    1 = BC1, 3 = BC3, 4 = BC4, 5 = BC5, 6 = BC6H, 7 = BC7.

Strict decoding (`decode(..., allow_provisional=False)`, the default) refuses
any chunk that depends on an unconfirmed built-in history entry. Pass
`allow_provisional=True` (or use `decode_with_info`) to allow provisional
entries and inspect which ones were used.

See `docs/textures.md` for the bitstream specification and provenance.
"""
import os
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bcpack_tables as _tables  # noqa: E402

# rANS and lane-array constants
RANS_L = 1 << 24
MAX_WIDTH = 10
PREFETCH_BUDGET = 8
MAX_VIRTUAL_ZEROS = 8

CODEC_NAMES = {
    1: "BC1",
    3: "BC3",
    4: "BC4",
    5: "BC5",
    6: "BC6H",
    7: "BC7",
}

# Bytes represented by one lane symbol-tuple (BC3/BC5/BC6H/BC7 use 2 elements per 16-byte block).
ELEMENT_BYTES = 8

BLOCK_BYTES = {
    1: 8,
    3: 16,
    4: 8,
    5: 16,
    6: 16,
    7: 16,
}

# Default per-lane symbol widths (bits) for each DSTORAGE_BCPACK_MODE.
LANE_WIDTHS = {
    1: (8, 8, 5, 6, 5, 5, 6, 5, 8, 8, 8, 8),
    3: (4, 4, 8, 6, 6, 6, 6, 5, 6, 5, 8, 8),
    4: (8, 8, 8, 8, 6, 6, 6, 6, 6, 6, 6, 6),
    5: (4, 4, 8, 8, 6, 6, 6, 6, 6, 6, 6, 6),
    6: (4, 4, 8, 8, 8, 5, 5, 5, 6, 6, 6, 8, 8, 8, 8, 5),
    7: (4, 4, 8, 8, 8, 8, 8, 7, 7, 5, 6, 8, 8, 8, 8, 5),
}

ALL_CODECS = frozenset(CODEC_NAMES)
SUPPORTED_REBUILD_CODECS = frozenset({1, 3, 4, 5})


class BcpackError(ValueError):
    """Raised when a BCPack stream fails structural or rANS validation."""


class BcpackProvisionalError(BcpackError):
    """Raised when strict decoding encounters an unconfirmed or unknown history entry."""


class BcpackUnsupportedCodec(NotImplementedError):
    """Raised when a BCPack codec has no block rebuilder yet."""


class _Model:
    """16-bit rANS banded frequency model for one (symbol_width, q) pair."""

    __slots__ = ("w", "freq", "base")

    def __init__(self, w, freq):
        self.w = w
        self.freq = freq
        base = []
        acc = 0
        for p, f in enumerate(freq):
            base.append(acc)
            acc += f * (1 if p == 0 else 1 << (p - 1))
        if acc != 65536:
            raise BcpackError(f"probability table ({w}) does not sum to 65536 (got {acc})")
        self.base = tuple(base)

    def lookup(self, cf):
        p = self.w
        base = self.base
        while p > 0 and cf < base[p]:
            p -= 1
        return 0 if p == 0 else (1 << (p - 1)) + (cf - base[p]) // self.freq[p]

    def advance(self, s, x):
        p = s.bit_length()
        f = self.freq[p]
        c = self.base[p] + f * (s - (0 if p == 0 else 1 << (p - 1)))
        if c == 0:
            f, c = f - 1, 1
        return f * (x >> 16) + (x & 0xFFFF) - c


_MODEL_CACHE = {}


def _get_model(w, q):
    key = (w, q)
    m = _MODEL_CACHE.get(key)
    if m is None:
        freq = _tables.PROB_TABLES.get(key)
        if freq is None:
            return None
        m = _Model(w, freq)
        _MODEL_CACHE[key] = m
    return m


def _decode_lanes_raw(raw, widths, nsym):
    """Decode the interleaved rANS lane array into one symbol list per lane.

    Returns `(lanes, bytes_used)`. Raises `BcpackError` if any fragment fails
    to land at the exact rANS lower bound `RANS_L`, runs out of input, or
    produces an out-of-range symbol.
    """
    nl = len(widths)
    HDR0, HDR1, HDR2, PROC = 0, 1, 2, 3
    L = RANS_L
    ctrl = [HDR0] * nl
    out_rem = [nsym] * nl
    w = [0] * nl
    q = [0] * nl
    h3 = [0] * nl
    flush = [0] * nl
    done = [nsym <= 0] * nl
    full = [False] * nl
    held = [False] * nl
    vz = [0] * nl
    inbuf = [0] * nl
    sym = [0] * nl
    in_rem = [0] * nl
    p0 = [0] * nl
    p1 = [0] * nl
    model = [None] * nl
    slotfull = [False] * nl
    slotval = [0] * nl
    stall = [False] * nl
    out = [[] for _ in range(nl)]
    pos = 0
    nraw = len(raw)

    def fail(msg, lane):
        raise BcpackError(
            f"lane {lane} (width {w[lane]}, q {q[lane]}) at input byte {pos}: {msg}"
        )

    while True:
        if all(done[i] and not slotfull[i] for i in range(nl)):
            return out, pos

        # Phase 1: state update, prefetch arbitration (at most 8 bytes/cycle),
        # and symbol release for non-stalled lanes.
        left = PREFETCH_BUDGET
        for i in range(nl):
            if done[i]:
                stall[i] = True
                continue
            if held[i] and slotfull[i]:
                stall[i] = True
                continue
            stall[i] = False
            if ctrl[i] == PROC:
                p0[i] = model[i].advance(sym[i], p1[i]) if held[i] else p1[i]
            if not full[i] and (ctrl[i] != PROC or in_rem[i] > 0):
                if left == 0:
                    stall[i] = True
                    continue
                if pos >= nraw:
                    fail("input ran out", i)
                inbuf[i] = raw[pos]
                pos += 1
                full[i] = True
                left -= 1
            if held[i]:
                slotfull[i] = True
                slotval[i] = sym[i]
                held[i] = False

        # Phase 2: header parsing, byte merge, virtual zero fill on flush, and symbol lookup.
        for i in range(nl):
            if done[i] or stall[i]:
                continue
            c = ctrl[i]
            if c == HDR0:
                if not full[i]:
                    continue
                h = inbuf[i]
                q[i] = h & 15
                flush[i] = (h >> 4) & 1
                h3[i] = (h >> 5) & 1
                w[i] = widths[i] - (h >> 6)
                full[i] = False
                ctrl[i] = HDR1
                vz[i] = 0
            elif c == HDR1:
                if not full[i]:
                    continue
                full[i] = False
                wi = w[i]
                if not 1 <= wi <= MAX_WIDTH:
                    fail("symbol width out of range", i)
                m = _get_model(wi, q[i])
                if m is None:
                    fail("probability table for this width and q is not known", i)
                model[i] = m
                if h3[i]:
                    in_rem[i] = inbuf[i]
                    ctrl[i] = HDR2
                else:
                    in_rem[i] = inbuf[i] + 1
                    ctrl[i] = PROC
            elif c == HDR2:
                if not full[i]:
                    continue
                in_rem[i] = ((inbuf[i] << 8) | (in_rem[i] & 0xFF)) + 257
                ctrl[i] = PROC
                full[i] = False
            else:
                st = p0[i]
                if st < L:
                    if in_rem[i] > 0:
                        if full[i]:
                            st = (st << 8) | inbuf[i]
                            in_rem[i] -= 1
                            full[i] = False
                            if in_rem[i] == 0 and flush[i]:
                                while st < L and st != 0 and vz[i] < MAX_VIRTUAL_ZEROS:
                                    st <<= 8
                                    vz[i] += 1
                    elif flush[i]:
                        if st == 0 or vz[i] >= MAX_VIRTUAL_ZEROS:
                            fail("fragment did not end at its home state", i)
                        while st < L and st != 0 and vz[i] < MAX_VIRTUAL_ZEROS:
                            st <<= 8
                            vz[i] += 1
                    else:
                        p1[i] = st
                        if out_rem[i] > 0:
                            ctrl[i] = HDR0
                        else:
                            done[i] = True
                        continue
                nxt = st > L and out_rem[i] > 0
                if nxt:
                    s = model[i].lookup(st & 0xFFFF)
                    sym[i] = s
                    out_rem[i] -= 1
                    if s >> w[i]:
                        fail("symbol outside its width", i)
                held[i] = nxt
                p1[i] = st
                if in_rem[i] == 0:
                    if flush[i]:
                        if st == L:
                            if out_rem[i] > 0:
                                ctrl[i] = HDR0
                                p0[i] = 0
                                p1[i] = 0
                            else:
                                done[i] = True
                        elif st > L and not nxt:
                            fail("fragment did not end at its home state", i)
                    elif out_rem[i] == 0 and not nxt:
                        done[i] = True
                elif st == L or (out_rem[i] == 0 and not nxt and st > L):
                    fail("fragment did not end at its home state", i)

        # Phase 3: emit one synchronized element when every active lane has filled its slot.
        ready = True
        anyfull = False
        for i in range(nl):
            if slotfull[i]:
                anyfull = True
            elif not (done[i] and not held[i]):
                ready = False
                break
        if ready and anyfull:
            for i in range(nl):
                if slotfull[i]:
                    out[i].append(slotval[i])
                    slotfull[i] = False
        for i in range(nl):
            if done[i] and held[i] and not slotfull[i]:
                slotfull[i] = True
                slotval[i] = sym[i]
                held[i] = False


def decode_lanes(raw, codec, out_size):
    """Entropy layer: decode `raw` into per-lane symbol lists and bytes consumed.

    Works for all six active codecs (`1, 3, 4, 5, 6, 7`). Trailing non-zero
    bytes after the last lane finishes are rejected.
    """
    if codec not in LANE_WIDTHS:
        raise BcpackError(f"unsupported BCPack codec id: {codec}")
    if out_size < 0 or out_size % BLOCK_BYTES[codec] != 0:
        raise BcpackError(
            f"decoded size {out_size} is not a whole number of {CODEC_NAMES[codec]} blocks"
        )
    nsym = out_size // ELEMENT_BYTES
    data = memoryview(raw)
    lanes, used = _decode_lanes_raw(data, LANE_WIDTHS[codec], nsym)
    if any(data[used:]):
        raise BcpackError("non-zero data left over after the last lane finished")
    return lanes, used


def _unzig(v):
    return -((v + 1) >> 1) if (v & 1) else (v >> 1)


def _inter8(a, b):
    d = 0
    for k in range(4):
        d |= (((a >> k) & 1) << (2 * k)) | (((b >> k) & 1) << (2 * k + 1))
    return d


def _loose_limit(nblocks):
    """Maximum undetermined blocks permitted in provisional mode (1 in 1,024, min 2)."""
    return max(2, nblocks // 1024)


def _bc1_blocks(lanes, nblocks, provisional):
    if any(lanes[1]):
        raise BcpackError("BC1 lane 1 is non-zero (unexpected)")
    quad = _tables.BC1_QUADRANT
    confirmed = _tables.BC1_HISTORY_CONFIRMED
    loose = _tables.BC1_HISTORY_PROVISIONAL
    out = bytearray(nblocks * 8)
    hist = [None] * nblocks
    mods = (32, 64, 32, 32, 64, 32)
    l0 = lanes[0]
    cols = lanes[2:8]
    q0, q1, q2, q3 = lanes[8:12]
    for i in range(nblocks):
        v = l0[i]
        d = 256 - ((v + 1) >> 1) if (v & 1) else (v >> 1)
        if d == 0:
            base = (0, 0, 0, 0, 0, 0)
        elif i - d >= 0:
            base = hist[i - d]
        else:
            entry = 256 + i - d
            base = confirmed.get(entry)
            if base is None:
                if provisional is not None and entry in loose:
                    base = loose[entry]
                    provisional.add(f"BC1 history {entry}")
                else:
                    state = "unconfirmed" if entry in loose else "unknown"
                    raise BcpackProvisionalError(
                        f"BC1 block {i} references {state} built-in history entry {entry}"
                    )
        e0 = (base[0] + _unzig(cols[0][i])) % mods[0]
        e1 = (base[1] + _unzig(cols[1][i])) % mods[1]
        e2 = (base[2] + _unzig(cols[2][i])) % mods[2]
        e3 = (base[3] + _unzig(cols[3][i])) % mods[3]
        e4 = (base[4] + _unzig(cols[4][i])) % mods[4]
        e5 = (base[5] + _unzig(cols[5][i])) % mods[5]
        hist[i] = (e0, e1, e2, e3, e4, e5)
        c0 = (e0 << 11) | (e1 << 5) | e2
        c1 = (e3 << 11) | (e4 << 5) | e5
        a, b, c, dd = quad[q0[i]], quad[q1[i]], quad[q2[i]], quad[q3[i]]
        # Quadrant byte: pixels TL, TR, BL, BR in bits 0-1, 2-3, 4-5, 6-7.
        struct.pack_into(
            "<HHBBBB",
            out,
            i * 8,
            c0,
            c1,
            (a & 15) | ((b & 15) << 4),
            (a >> 4) | ((b >> 4) << 4),
            (c & 15) | ((dd & 15) << 4),
            (c >> 4) | ((dd >> 4) << 4),
        )
    return bytes(out)


_BC3_MODS = (256, 256, 32, 64, 32, 32, 64, 32)
_BC3_NONE = (128, 128, 0, 0, 0, 0, 0, 0)


def _bc3_blocks(lanes, nblocks, provisional):
    if any(lanes[1]):
        raise BcpackError("BC3 lane 1 is non-zero (unexpected)")
    pairs = _tables.BC3_ALPHA_PAIR
    quads = _tables.BC3_COLOUR_QUADRANT
    confirmed = _tables.BC3_HISTORY_CONFIRMED
    loose = _tables.BC3_HISTORY_PROVISIONAL
    out = bytearray(nblocks * 16)
    hist = [None] * nblocks
    lost = [False] * nblocks
    l0, alpha = lanes[0], lanes[2]
    rows = lanes[3:7]
    lr, lg, lb = lanes[7], lanes[8], lanes[9]
    top, bottom = lanes[10], lanes[11]
    unknown = set()
    mods = _BC3_MODS
    for i in range(nblocks):
        a, b = 2 * i, 2 * i + 1
        d = _unzig(_inter8(l0[a], l0[b])) & 255
        if d == 0:
            base = _BC3_NONE
        elif i - d >= 0:
            base = hist[i - d]
            lost[i] = lost[i - d]
        else:
            entry = 256 + i - d
            base = confirmed.get(entry)
            if base is None and entry in loose:
                if provisional is None:
                    raise BcpackProvisionalError(
                        f"BC3 block {i} references unconfirmed built-in history entry {entry}"
                    )
                base = loose[entry]
                provisional.add(f"BC3 history {entry}")
            if base is None:
                if provisional is None:
                    raise BcpackProvisionalError(
                        f"BC3 block {i} references unknown built-in history entry {entry}"
                    )
                base = _BC3_NONE
                lost[i] = True
                unknown.add(entry)
        dz = (alpha[a], alpha[b], lr[a], lg[a], lb[a], lr[b], lg[b], lb[b])
        e = tuple((base[k] + _unzig(dz[k])) % mods[k] for k in range(8))
        hist[i] = e
        bits = 0
        for r in range(4):
            left, right = pairs[rows[r][a]], pairs[rows[r][b]]
            bits |= (left[0] | (left[1] << 3) | (right[0] << 6) | (right[1] << 9)) << (12 * r)
        at = i * 16
        out[at] = e[0]
        out[at + 1] = e[1]
        out[at + 2:at + 8] = bits.to_bytes(6, "little")
        c0 = (e[2] << 11) | (e[3] << 5) | e[4]
        c1 = (e[5] << 11) | (e[6] << 5) | e[7]
        qa, qb = quads[top[a]], quads[top[b]]
        qc, qd = quads[bottom[a]], quads[bottom[b]]
        struct.pack_into(
            "<HHBBBB",
            out,
            at + 8,
            c0,
            c1,
            (qa & 15) | ((qb & 15) << 4),
            (qa >> 4) | ((qb >> 4) << 4),
            (qc & 15) | ((qd & 15) << 4),
            (qc >> 4) | ((qd >> 4) << 4),
        )
    if unknown:
        total = sum(lost)
        if total > _loose_limit(nblocks):
            raise BcpackProvisionalError(
                f"BC3 chunk has {total}/{nblocks} blocks from unknown built-in history entries"
            )
        provisional.add(f"BC3 unknown history {sorted(unknown)} ({total} blocks)")
    return bytes(out)


def _bc4_blocks(lanes, nblocks, provisional):
    if any(lanes[1]):
        raise BcpackError("BC4 lane 1 is non-zero (unexpected)")
    pairs = _tables.BC4_PAIR
    confirmed = _tables.BC4_HISTORY_CONFIRMED
    loose = _tables.BC4_HISTORY_PROVISIONAL
    out = bytearray(nblocks * 8)
    hist = [None] * nblocks
    l0, l2, l3 = lanes[0], lanes[2], lanes[3]
    rows = lanes[4:12]
    for i in range(nblocks):
        v = l0[i]
        d = 256 - ((v + 1) >> 1) if (v & 1) else (v >> 1)
        if d == 0:
            base = (128, 128)
        elif i - d >= 0:
            base = hist[i - d]
        else:
            entry = 256 + i - d
            base = confirmed.get(entry)
            if base is None:
                if provisional is not None and entry in loose:
                    base = loose[entry]
                    provisional.add(f"BC4 history {entry}")
                else:
                    state = "unconfirmed" if entry in loose else "unknown"
                    raise BcpackProvisionalError(
                        f"BC4 block {i} references {state} built-in history entry {entry}"
                    )
        a0 = (base[0] + _unzig(l2[i])) & 255
        a1 = (base[1] + _unzig(l3[i])) & 255
        hist[i] = (a0, a1)
        bits = 0
        for r in range(4):
            left, right = pairs[rows[2 * r][i]], pairs[rows[2 * r + 1][i]]
            bits |= (left[0] | (left[1] << 3) | (right[0] << 6) | (right[1] << 9)) << (12 * r)
        at = i * 8
        out[at] = a0
        out[at + 1] = a1
        out[at + 2:at + 8] = bits.to_bytes(6, "little")
    return bytes(out)


def _bc5_blocks(lanes, nblocks, provisional):
    if any(lanes[1]):
        raise BcpackError("BC5 lane 1 is non-zero (unexpected)")
    pairs = _tables.BC5_PAIR
    confirmed = _tables.BC5_HISTORY_CONFIRMED
    loose = _tables.BC5_HISTORY_PROVISIONAL
    out = bytearray(nblocks * 16)
    hist = [None] * nblocks
    lost = [False] * nblocks
    l0, red, green = lanes[0], lanes[2], lanes[3]
    rows = lanes[4:12]
    unknown = set()
    for i in range(nblocks):
        a, b = 2 * i, 2 * i + 1
        d = _unzig(_inter8(l0[a], l0[b])) & 255
        if d == 0:
            base = (128, 128, 128, 128)
        elif i - d >= 0:
            base = hist[i - d]
            lost[i] = lost[i - d]
        else:
            entry = 256 + i - d
            base = confirmed.get(entry)
            if base is None and entry in loose:
                if provisional is None:
                    raise BcpackProvisionalError(
                        f"BC5 block {i} references unconfirmed built-in history entry {entry}"
                    )
                base = loose[entry]
                provisional.add(f"BC5 history {entry}")
            if base is None:
                if provisional is None:
                    raise BcpackProvisionalError(
                        f"BC5 block {i} references unknown built-in history entry {entry}"
                    )
                base = (128, 128, 128, 128)
                lost[i] = True
                unknown.add(entry)
        e = (
            (base[0] + _unzig(red[a])) & 255,
            (base[1] + _unzig(red[b])) & 255,
            (base[2] + _unzig(green[a])) & 255,
            (base[3] + _unzig(green[b])) & 255,
        )
        hist[i] = e
        at = i * 16
        for ch in range(2):
            bits = 0
            for r in range(4):
                left, right = pairs[rows[4 * ch + r][a]], pairs[rows[4 * ch + r][b]]
                bits |= (left[0] | (left[1] << 3) | (right[0] << 6) | (right[1] << 9)) << (12 * r)
            out[at + 8 * ch] = e[2 * ch]
            out[at + 8 * ch + 1] = e[2 * ch + 1]
            out[at + 8 * ch + 2:at + 8 * ch + 8] = bits.to_bytes(6, "little")
    if unknown:
        total = sum(lost)
        if total > _loose_limit(nblocks):
            raise BcpackProvisionalError(
                f"BC5 chunk has {total}/{nblocks} blocks from unknown built-in history entries"
            )
        provisional.add(f"BC5 unknown history {sorted(unknown)} ({total} blocks)")
    return bytes(out)


_REBUILDERS = {
    1: _bc1_blocks,
    3: _bc3_blocks,
    4: _bc4_blocks,
    5: _bc5_blocks,
}


def decode_with_info(raw, codec, out_size, *, allow_provisional=False):
    """Decode one BCPack chunk, returning `(decoded_bytes, provisional_labels)`.

    When `allow_provisional` is False (strict mode), any unconfirmed or unknown
    built-in history reference raises `BcpackProvisionalError` and the returned
    set is always empty.
    """
    rebuilder = _REBUILDERS.get(codec)
    if rebuilder is None:
        name = CODEC_NAMES.get(codec, str(codec))
        raise BcpackUnsupportedCodec(f"BCPack block rebuilder for codec {codec} ({name}) is not implemented yet")
    lanes, _ = decode_lanes(raw, codec, out_size)
    nblocks = out_size // BLOCK_BYTES[codec]
    prov = set() if allow_provisional else None
    data = rebuilder(lanes, nblocks, prov)
    return data, (prov if prov is not None else set())


def decode(raw, codec, out_size, *, allow_provisional=False):
    """Decode one BCPack chunk into `out_size` bytes of standard BCn blocks."""
    data, _ = decode_with_info(raw, codec, out_size, allow_provisional=allow_provisional)
    return data
