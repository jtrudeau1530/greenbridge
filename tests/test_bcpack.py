"""Tests for tools/bcpack.py and tools/bcpack_tables.py using synthetic streams.

Builds genuine BCPack rANS streams from scratch with a self-contained test
encoder — no game files or texture bytes are used.

    python -I tests/test_bcpack.py
"""
import importlib.util
import os
import struct
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import bcpack  # noqa: E402
import bcpack_tables as tables  # noqa: E402


# --- Synthetic BCPack rANS encoder (test helper only) ----------------------

def _zig(v):
    """Inverse of bcpack._unzig for signed integer deltas."""
    return (-2 * v - 1) if v < 0 else (2 * v)


def _deint4(v):
    """Inverse of bcpack._inter8: split 8-bit integer into two 4-bit nibbles."""
    a = b = 0
    for k in range(4):
        a |= ((v >> (2 * k)) & 1) << k
        b |= ((v >> (2 * k + 1)) & 1) << k
    return a, b


def _encode_rans_syms(syms, model, x):
    """Reverse-encode `syms` starting from end state `x`, returning `(new_x, bytes_reversed)`."""
    buf = []
    for s in reversed(syms):
        p = s.bit_length()
        f = model.freq[p]
        c = model.base[p] + f * (s - (0 if p == 0 else 1 << (p - 1)))
        if c == 0:
            f, c = f - 1, 1
        limit = f << 16
        while x >= limit:
            buf.append(x & 0xFF)
            x >>= 8
        q_div, r_mod = divmod(x, f)
        x = (q_div << 16) + c + r_mod
    return x, buf


def _make_hdr(default_w, w, q, flush, n):
    z = default_w - w
    assert 0 <= z <= 3 and n >= 1
    if n <= 256:
        return [((z & 3) << 6) | ((1 if flush else 0) << 4) | (q & 15), n - 1]
    n2 = n - 257
    return [
        ((z & 3) << 6) | (1 << 5) | ((1 if flush else 0) << 4) | (q & 15),
        n2 & 0xFF,
        (n2 >> 8) & 0xFF,
    ]


def _encode_fragment(syms, default_w, w, q, *, flush=True, strip_zeros=True):
    """Encode `syms` into one standalone flush fragment (header + rANS payload)."""
    assert flush
    model = bcpack._get_model(w, q)
    assert model is not None
    x, buf = _encode_rans_syms(syms, model, bcpack.RANS_L)
    payload = [(x >> 24) & 0xFF, (x >> 16) & 0xFF, (x >> 8) & 0xFF, x & 0xFF] + list(reversed(buf))
    if strip_zeros:
        stripped = 0
        while len(payload) > 1 and payload[-1] == 0 and stripped < bcpack.MAX_VIRTUAL_ZEROS:
            payload.pop()
            stripped += 1
    return bytes(_make_hdr(default_w, w, q, True, len(payload)) + payload), 0


def _encode_lane_stream(fragments, default_w):
    """Chain one or more `(syms, w, q, flush)` fragments for a single lane.

    The final fragment must have `flush=True`. For a non-flush fragment followed
    by another fragment, the boundary state is shifted below `RANS_L` so the
    decoder finishes fragment `i` at `st < RANS_L` and renormalizes from fragment `i+1`.
    """
    encoded = [None] * len(fragments)
    x = bcpack.RANS_L
    for idx in range(len(fragments) - 1, -1, -1):
        syms, w, q, flush = fragments[idx]
        model = bcpack._get_model(w, q)
        assert model is not None
        if flush:
            x = bcpack.RANS_L
        x, buf = _encode_rans_syms(syms, model, x)
        prev_flush = True if idx == 0 else fragments[idx - 1][3]
        if prev_flush:
            # This fragment starts from state 0: prepend all 4 initial state bytes.
            payload = [(x >> 24) & 0xFF, (x >> 16) & 0xFF, (x >> 8) & 0xFF, x & 0xFF] + list(reversed(buf))
            if flush:
                stripped = 0
                while len(payload) > 1 and payload[-1] == 0 and stripped < bcpack.MAX_VIRTUAL_ZEROS:
                    payload.pop()
                    stripped += 1
        else:
            # Previous fragment is non-flush: emit renormalization bytes into this
            # fragment until `x < RANS_L` so the previous fragment hands off `x < RANS_L`.
            while x >= bcpack.RANS_L:
                buf.append(x & 0xFF)
                x >>= 8
            payload = list(reversed(buf))
        encoded[idx] = bytes(_make_hdr(default_w, w, q, flush, len(payload)) + payload)
    return b"".join(encoded)


def interleave_lanes(lane_streams, widths, nsym):
    """Interleave per-lane byte streams according to the BCPack 8-byte/cycle scheduler."""
    nl = len(widths)
    HDR0, HDR1, HDR2, PROC = 0, 1, 2, 3
    L = bcpack.RANS_L
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
    stall = [False] * nl
    lpos = [0] * nl
    interleaved = bytearray()
    max_cycles = max(1000, nsym * 64 + sum(len(s) for s in lane_streams) * 8)

    for _ in range(max_cycles):
        if all(done[i] and not slotfull[i] for i in range(nl)):
            for i in range(nl):
                assert lpos[i] == len(lane_streams[i]), f"lane {i} left {len(lane_streams[i]) - lpos[i]} bytes"
            return bytes(interleaved)
        left = bcpack.PREFETCH_BUDGET
        for i in range(nl):
            if done[i] or (held[i] and slotfull[i]):
                stall[i] = True
                continue
            stall[i] = False
            if ctrl[i] == PROC:
                p0[i] = model[i].advance(sym[i], p1[i]) if held[i] else p1[i]
            if not full[i] and (ctrl[i] != PROC or in_rem[i] > 0):
                if left == 0:
                    stall[i] = True
                    continue
                b = lane_streams[i][lpos[i]]
                lpos[i] += 1
                interleaved.append(b)
                inbuf[i] = b
                full[i] = True
                left -= 1
            if held[i]:
                slotfull[i] = True
                held[i] = False
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
                model[i] = bcpack._get_model(w[i], q[i])
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
                                while st < L and st != 0 and vz[i] < bcpack.MAX_VIRTUAL_ZEROS:
                                    st <<= 8
                                    vz[i] += 1
                    elif flush[i]:
                        while st < L and st != 0 and vz[i] < bcpack.MAX_VIRTUAL_ZEROS:
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
                    sym[i] = model[i].lookup(st & 0xFFFF)
                    out_rem[i] -= 1
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
                    elif out_rem[i] == 0 and not nxt:
                        done[i] = True
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
                slotfull[i] = False
        for i in range(nl):
            if done[i] and held[i] and not slotfull[i]:
                slotfull[i] = True
                held[i] = False
    raise AssertionError("interleave_lanes exceeded max_cycles")


def encode_lanes(lanes, codec, *, q=0):
    """Encode per-lane symbol lists into a complete BCPack chunk."""
    widths = bcpack.LANE_WIDTHS[codec]
    nsym = len(lanes[0])
    streams = []
    for i, (syms, dw) in enumerate(zip(lanes, widths)):
        qi = q[i] if isinstance(q, (list, tuple)) else q
        frag, _ = _encode_fragment(syms, dw, dw, qi, flush=True)
        streams.append(frag)
    return interleave_lanes(streams, widths, nsym)


def encode_bc1_blocks(blocks, *, ref_dists=None):
    """Encode raw BC1 blocks (8 bytes each) into a valid codec-1 BCPack chunk."""
    n = len(blocks) // 8
    inv_quad = {v: i for i, v in enumerate(tables.BC1_QUADRANT)}
    mods = (32, 64, 32, 32, 64, 32)
    lanes = [[] for _ in range(12)]
    hist = []
    for i in range(n):
        c0, c1, r0, r1, r2, r3 = struct.unpack_from("<HHBBBB", blocks, i * 8)
        e = ((c0 >> 11) & 31, (c0 >> 5) & 63, c0 & 31,
             (c1 >> 11) & 31, (c1 >> 5) & 63, c1 & 31)
        d = 0 if ref_dists is None else ref_dists[i]
        if d == 0:
            base = (0, 0, 0, 0, 0, 0)
        elif i - d >= 0:
            base = hist[i - d]
        else:
            entry = 256 + i - d
            base = tables.BC1_HISTORY_CONFIRMED.get(entry) or tables.BC1_HISTORY_PROVISIONAL.get(entry) or (0, 0, 0, 0, 0, 0)
        signed_d = d if d < 128 else d - 256
        lanes[0].append(_zig(signed_d))
        lanes[1].append(0)
        for k in range(6):
            delta = (e[k] - base[k]) % mods[k]
            if delta >= mods[k] // 2:
                delta -= mods[k]
            lanes[2 + k].append(_zig(delta))
        hist.append(e)
        qa = (r0 & 15) | ((r1 & 15) << 4)
        qb = (r0 >> 4) | ((r1 >> 4) << 4)
        qc = (r2 & 15) | ((r3 & 15) << 4)
        qd = (r2 >> 4) | ((r3 >> 4) << 4)
        lanes[8].append(inv_quad[qa])
        lanes[9].append(inv_quad[qb])
        lanes[10].append(inv_quad[qc])
        lanes[11].append(inv_quad[qd])
    return encode_lanes(lanes, 1)


# --- Unit tests ------------------------------------------------------------

class ProbTablesTest(unittest.TestCase):
    def test_all_160_tables_valid_and_invertible(self):
        self.assertEqual(len(tables.PROB_TABLES), 160)
        for w in range(1, 11):
            for q in range(16):
                m = bcpack._get_model(w, q)
                self.assertIsNotNone(m)
                # Verify boundary symbols (0, 1, mid, max) invert cleanly.
                for s in {0, 1, (1 << (w - 1)), (1 << w) - 1}:
                    p = s.bit_length()
                    f = m.freq[p]
                    c = m.base[p] + f * (s - (0 if p == 0 else 1 << (p - 1)))
                    if c == 0:
                        f, c = f - 1, 1
                    x = (256 << 16) + c
                    self.assertEqual(m.lookup(x & 0xFFFF), s)
                    self.assertEqual(m.advance(s, x), 256 * f)


class EntropyLayerTest(unittest.TestCase):
    def test_round_trip_12_and_16_lanes(self):
        for codec in (1, 3, 4, 5, 6, 7):
            widths = bcpack.LANE_WIDTHS[codec]
            nsym = 24
            lanes = [[((i * 13 + j * 7) & ((1 << dw) - 1)) for j in range(nsym)]
                     for i, dw in enumerate(widths)]
            qs = [(i * 3) % 15 for i in range(len(widths))]
            raw = encode_lanes(lanes, codec, q=qs)
            decoded, used = bcpack.decode_lanes(raw + b"\x00\x00\x00", codec, nsym * 8)
            self.assertEqual(used, len(raw))
            self.assertEqual(decoded, lanes)

    def test_multi_fragment_flush_and_non_flush_and_3byte_header(self):
        codec = 1
        widths = bcpack.LANE_WIDTHS[codec]
        # 400 symbols forces a >256-byte fragment (3-byte header, h3=1) on 8-bit lanes.
        nsym = 400
        lanes = [[((i * 29 + j * 17) & ((1 << dw) - 1)) for j in range(nsym)]
                 for i, dw in enumerate(widths)]
        streams = []
        for i, (syms, dw) in enumerate(zip(lanes, widths)):
            if i == 0:
                # Two flush fragments with reduced width on the second half.
                s1 = [x & 0x3F for x in syms[:100]]
                s2 = [x & 0x1F for x in syms[100:]]
                lanes[0] = s1 + s2
                streams.append(_encode_lane_stream([
                    (s1, dw - 2, 2, True),
                    (s2, dw - 3, 5, True),
                ], dw))
            elif i == 2:
                # Chained non-flush -> flush fragments with a model change mid-stream.
                streams.append(_encode_lane_stream([
                    (syms[:150], dw, 1, False),
                    (syms[150:], dw, 4, True),
                ], dw))
            else:
                # Single large flush fragment (3-byte header on 8-bit lanes).
                streams.append(_encode_lane_stream([(syms, dw, 0, True)], dw))
        raw = interleave_lanes(streams, widths, nsym)
        decoded, used = bcpack.decode_lanes(raw, codec, nsym * 8)
        self.assertEqual(used, len(raw))
        self.assertEqual(decoded, lanes)

    def test_rejects_corrupt_and_trailing_data(self):
        lanes = [[0] * 8 for _ in range(12)]
        raw = encode_lanes(lanes, 1)
        with self.assertRaises(bcpack.BcpackError):
            bcpack.decode_lanes(raw + b"\x01", 1, 64)
        with self.assertRaises(bcpack.BcpackError):
            bcpack.decode_lanes(raw[:-2], 1, 64)
        flipped = bytearray(raw)
        flipped[-1] ^= 0xFF
        with self.assertRaises(bcpack.BcpackError):
            bcpack.decode_lanes(bytes(flipped), 1, 64)
        with self.assertRaises(bcpack.BcpackError):
            bcpack.decode_lanes(raw, 99, 64)
        with self.assertRaises(bcpack.BcpackError):
            bcpack.decode_lanes(raw, 1, 60)

    def test_unsupported_codec_rebuild_raises(self):
        lanes = [[0] * 4 for _ in range(16)]
        raw = encode_lanes(lanes, 7)
        # Entropy layer succeeds for BC7...
        dec, _ = bcpack.decode_lanes(raw, 7, 32)
        self.assertEqual(dec, lanes)
        # ...while block rebuild raises BcpackUnsupportedCodec in PR 1.
        with self.assertRaises(bcpack.BcpackUnsupportedCodec):
            bcpack.decode(raw, 7, 32)


class BlockRebuildTest(unittest.TestCase):
    def test_bc1_confirmed_and_provisional_history(self):
        # Block 0: d=0 (zero base); Block 1: dynamic ref to block 0 (d=1);
        # Block 2: confirmed built-in preset 1 (256 + 2 - d == 1 => d=257? no, 256+2-255=3, use entry 255 -> d=3).
        blocks = b"".join([
            struct.pack("<HHBBBB", 0xF800, 0x07E0, 0x12, 0x34, 0x56, 0x78),
            struct.pack("<HHBBBB", 0xF81F, 0x001F, 0xAA, 0xBB, 0xCC, 0xDD),
            struct.pack("<HHBBBB", 0xFFFF, 0x0000, 0x00, 0x55, 0xAA, 0xFF),
        ])
        # Block 2 with d=3 -> 256 + 2 - 3 = 255 (confirmed in BC1_HISTORY_CONFIRMED).
        self.assertIn(255, tables.BC1_HISTORY_CONFIRMED)
        raw = encode_bc1_blocks(blocks, ref_dists=[0, 1, 3])
        out, prov = bcpack.decode_with_info(raw, 1, len(blocks))
        self.assertEqual(out, blocks)
        self.assertEqual(prov, set())

        # Reference provisional entry 6 from block 0: 256 + 0 - d = 6 => d = 250.
        self.assertIn(6, tables.BC1_HISTORY_PROVISIONAL)
        raw_prov = encode_bc1_blocks(blocks[:8], ref_dists=[250])
        with self.assertRaises(bcpack.BcpackProvisionalError):
            bcpack.decode(raw_prov, 1, 8)
        out_p, prov_p = bcpack.decode_with_info(raw_prov, 1, 8, allow_provisional=True)
        self.assertEqual(out_p, blocks[:8])
        self.assertEqual(prov_p, {"BC1 history 6"})

    def test_bc4_round_trip(self):
        inv_pair = {p: i for i, p in enumerate(tables.BC4_PAIR)}
        # Build 2 BC4 blocks: block 0 uses d=0 (base 128,128), block 1 refs block 0 (d=1).
        b0_bits = sum(((r + p) & 7) << (3 * (4 * r + p)) for r in range(4) for p in range(4))
        b1_bits = sum(((7 - r - p) & 7) << (3 * (4 * r + p)) for r in range(4) for p in range(4))
        expected = (
            bytes([130, 120]) + b0_bits.to_bytes(6, "little") +
            bytes([140, 110]) + b1_bits.to_bytes(6, "little")
        )
        lanes = [[] for _ in range(12)]
        for i, (d, base, (a0, a1), bits) in enumerate([
            (0, (128, 128), (130, 120), b0_bits),
            (1, (130, 120), (140, 110), b1_bits),
        ]):
            lanes[0].append(_zig(d))
            lanes[1].append(0)
            lanes[2].append(_zig(a0 - base[0]))
            lanes[3].append(_zig(a1 - base[1]))
            for r in range(4):
                row = (bits >> (12 * r)) & 0xFFF
                lanes[4 + 2 * r].append(inv_pair[(row & 7, (row >> 3) & 7)])
                lanes[4 + 2 * r + 1].append(inv_pair[((row >> 6) & 7, (row >> 9) & 7)])
        raw = encode_lanes(lanes, 4)
        self.assertEqual(bcpack.decode(raw, 4, len(expected)), expected)

        # Provisional BC4 history entry (entry 25 -> block 0 uses d = 231).
        self.assertIn(25, tables.BC4_HISTORY_PROVISIONAL)
        prov_a0, prov_a1 = tables.BC4_HISTORY_PROVISIONAL[25]
        lanes_p = [[_zig(231 - 256)], [0], [0], [0]] + [[0]] * 8
        raw_p = encode_lanes(lanes_p, 4)
        with self.assertRaises(bcpack.BcpackProvisionalError):
            bcpack.decode(raw_p, 4, 8)
        out_p, info_p = bcpack.decode_with_info(raw_p, 4, 8, allow_provisional=True)
        self.assertEqual((out_p[0], out_p[1]), (prov_a0, prov_a1))
        self.assertEqual(info_p, {"BC4 history 25"})

    def test_bc5_round_trip(self):
        inv_pair = {p: i for i, p in enumerate(tables.BC5_PAIR)}
        # Block 0 uses confirmed BC5 entry if present, plus block 1 dynamic ref.
        confirmed_entry = next(iter(sorted(tables.BC5_HISTORY_CONFIRMED)))
        base0 = tables.BC5_HISTORY_CONFIRMED[confirmed_entry]
        d0 = 256 - confirmed_entry  # block 0: 256 + 0 - d0 == confirmed_entry
        e0 = ((base0[0] + 2) & 255, (base0[1] - 3) & 255, (base0[2] + 1) & 255, (base0[3] - 1) & 255)
        e1 = ((e0[0] + 5) & 255, (e0[1] - 4) & 255, (e0[2] + 3) & 255, (e0[3] + 2) & 255)
        bits = [
            (0x0123456789AB, 0xFEDCBA987654),
            (0x112233445566, 0x778899AABBCC),
        ]
        expected = bytearray()
        for e, (br, bg) in zip((e0, e1), bits):
            expected += bytes([e[0], e[1]]) + br.to_bytes(6, "little")
            expected += bytes([e[2], e[3]]) + bg.to_bytes(6, "little")
        lanes = [[] for _ in range(12)]
        for d, base, e, (br, bg) in ((d0, base0, e0, bits[0]), (1, e0, e1, bits[1])):
            sd = d if d < 128 else d - 256
            da, db = _deint4(_zig(sd))
            lanes[0].extend([da, db])
            lanes[1].extend([0, 0])
            for lane_idx, (c_a, c_b, b_a, b_b) in ((2, (e[0], e[1], base[0], base[1])),
                                                   (3, (e[2], e[3], base[2], base[3]))):
                d_a = (c_a - b_a + 128) % 256 - 128
                d_b = (c_b - b_b + 128) % 256 - 128
                lanes[lane_idx].extend([_zig(d_a), _zig(d_b)])
            for ch, bval in enumerate((br, bg)):
                for r in range(4):
                    row = (bval >> (12 * r)) & 0xFFF
                    lanes[4 + 4 * ch + r].extend([
                        inv_pair[(row & 7, (row >> 3) & 7)],
                        inv_pair[((row >> 6) & 7, (row >> 9) & 7)],
                    ])
        raw = encode_lanes(lanes, 5)
        out, prov = bcpack.decode_with_info(raw, 5, len(expected))
        self.assertEqual(out, bytes(expected))
        self.assertEqual(prov, set())

    def test_bc3_round_trip(self):
        inv_pair = {p: i for i, p in enumerate(tables.BC3_ALPHA_PAIR)}
        inv_quad = {v: i for i, v in enumerate(tables.BC3_COLOUR_QUADRANT)}
        # Block 0 with d=0: base is (128, 128, 0, 0, 0, 0, 0, 0).
        a0, a1 = 135, 120
        abits = 0x0123456789AB
        c0 = (5 << 11) | (10 << 5) | 3
        c1 = (2 << 11) | (6 << 5) | 1
        r0, r1, r2, r3 = 0x12, 0x34, 0x56, 0x78
        expected = bytes([a0, a1]) + abits.to_bytes(6, "little") + struct.pack("<HHBBBB", c0, c1, r0, r1, r2, r3)
        lanes = [[] for _ in range(12)]
        lanes[0].extend([0, 0])
        lanes[1].extend([0, 0])
        lanes[2].extend([_zig(a0 - 128), _zig(a1 - 128)])
        for r in range(4):
            row = (abits >> (12 * r)) & 0xFFF
            lanes[3 + r].extend([
                inv_pair[(row & 7, (row >> 3) & 7)],
                inv_pair[((row >> 6) & 7, (row >> 9) & 7)],
            ])
        lanes[7].extend([_zig(5), _zig(2)])
        lanes[8].extend([_zig(10), _zig(6)])
        lanes[9].extend([_zig(3), _zig(1)])
        qa = (r0 & 15) | ((r1 & 15) << 4)
        qb = (r0 >> 4) | ((r1 >> 4) << 4)
        qc = (r2 & 15) | ((r3 & 15) << 4)
        qd = (r2 >> 4) | ((r3 >> 4) << 4)
        lanes[10].extend([inv_quad[qa], inv_quad[qb]])
        lanes[11].extend([inv_quad[qc], inv_quad[qd]])
        raw = encode_lanes(lanes, 3)
        out, prov = bcpack.decode_with_info(raw, 3, len(expected))
        self.assertEqual(out, expected)
        self.assertEqual(prov, set())


if __name__ == "__main__":
    unittest.main()
