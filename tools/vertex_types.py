"""Measure the component types of the vertex streams in `42 40 00 00` blocks (follow-up to ASSET-57).

A stream can hold several blocks back to back, with their meshlet groups after the whole run in the same order
(group i belongs to block i); every paired block is measured.

docs/restofworld.md measures each tag's bytes per vertex on `42 40 00 00` blocks: 8 for 0x42, 4 for 0x21, 0x25,
0x26, 0x27 and 0x28, 16 for 0x8c. This tool tests what those bytes are, using checks that need no textures and no
other file: the vertex count and the triangle edges come from the meshlet group that follows the streams (validated
the same way as tools/meshlet_page.py), and each candidate interpretation is scored over those edges.

Block header (same slot-word layout as the `63 30 00 00` blocks, measured in docs/story.md):
    +0x00..+0x58  23 slot words: low byte = tag, (word >> 8) * 2 = stream offset from the block start
    +0x5C         u32 flags
    +0x60         float3 position bias      (rpf9-explorer RageModelDecoder reads these two for tag 0x42)
    +0x6C         u32
    +0x70         float3 position scale
    +0x80         slot 0 stream

Checks per tag (each one a count over vertices or triangle edges; no value leaves the tool):
    0x42 slot 0   packed position: x = bits 8-31 of the low u32, y = bits 0-7 | bits 32-47 << 8, z = bits 48-63,
                  position = bias + scale * (x, y, z). Counts finite results and edge lengths against the block size.
    0x28          two u16 per vertex. Wrap test: an edge whose two values differ by more than 0x8000 read as unsigned
                  but not as signed (or the reverse) shows which reading is continuous. Half-float test: NaN/Inf and
                  |v| > 1024 counts. Range: share of vertices above 1.0 at u16 / 8192 (tiling).
                  Mapping test: per triangle, |UV area| / |position area|; the spread (median absolute deviation of
                  the log ratio within a block) is lower for the reading that maps the surface evenly.
    4-byte tags   per byte: share of blocks where the byte is constant, share of vertices with byte 3 = 0 or 255,
                  edge smoothness (mean |difference| over triangle edges / mean |difference| over random vertex
                  pairs; < 1 means the value follows the surface), unit length read as snorm8 xyz and as
                  10:10:10:2 snorm, and each NORMAL_CANDIDATES decoding against the face normals. Tag 0x21 is
                  the octahedral normal of normal_0x21 (the mesh shaders' decode, chosen by bit 31).

Usage:
    python -I tools/vertex_types.py <archive.rpf|dir> [...] [--limit N] [--json out.json]
"""
import argparse
import json
import math
import mmap
import os
import random
import re
import struct
import sys
import zlib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from meshlet_page import PageError, parse_group  # noqa: E402  (same group validation as the 63 30 pages)

MAGIC = 0x4042
HEADER = 0x80
SLOTS = 23
TRI_SAMPLE = 256  # triangles per block for the mapping and face-normal tests
ZHDR = re.compile(rb"\x78[\x01\x5e\x9c\xda]")


class BlockError(ValueError):
    pass


def stride(tag):
    """Bytes per vertex measured for each tag in docs/restofworld.md (ASSET-57): (tag >> 4) * 2 fits all of them."""
    return (tag >> 4) * 2


def parse_slots(data, at=0):
    if at + HEADER > len(data):
        raise BlockError("header truncated")
    words = struct.unpack_from("<24I", data, at)
    if words[0] != MAGIC:
        raise BlockError("no 42 40 00 00 at slot 0")
    slots = {i: (w & 0xFF, (w >> 8) * 2) for i, w in enumerate(words[:SLOTS]) if w}
    order = sorted(slots, key=lambda i: slots[i][1])
    if slots[0][1] != HEADER or order[0] != 0:
        raise BlockError("slot 0 is not the first stream at +0x80")
    for a, b in zip(order, order[1:]):
        if slots[b][1] <= slots[a][1]:
            raise BlockError("slot offsets are not increasing")
    return slots, order, words[23]


def vertex_count(slots, order):
    """Vertex count from the first span (slot 0, 8 bytes per vertex), rounded down past 16-byte padding."""
    span = slots[order[1]][1] - slots[order[0]][1] if len(order) > 1 else 0
    if span <= 0:
        raise BlockError("single-stream block")
    return span // 8


def skip_padding(data, pos):
    """Round up to 16 bytes, then step over zero words."""
    pos = (pos + 15) & ~15
    while pos + 4 <= len(data) and struct.unpack_from("<I", data, pos)[0] == 0:
        pos += 4
    return pos


def block_end(data, at, slots, order, n):
    """End of a block: its last stream, sized by that tag's width, padded to 16 bytes, over zero words."""
    last = order[-1]
    return skip_padding(data, at + slots[last][1] + stride(slots[last][0]) * n)


def find_group(data, at, slots, order, n):
    """The meshlet group right after a block's last stream."""
    pos = block_end(data, at, slots, order, n)
    try:
        group, _ = parse_group(data, pos)
    except PageError as exc:
        raise BlockError(f"no group: {exc}")
    return pos, group


def block_run(data, at=0):
    """The `42 40 00 00` blocks stored back to back from `at`, as (offset, slots, order, count), and the offset
    after the last one. Each next header starts exactly at the previous block's end."""
    run = []
    while True:
        slots, order, _ = parse_slots(data, at)
        n = vertex_count(slots, order)
        run.append((at, slots, order, n))
        at = block_end(data, at, slots, order, n)
        if at + 4 > len(data) or struct.unpack_from("<I", data, at)[0] != MAGIC:
            return run, at


def measure_stream(data, T, rng, errors):
    """Measure every block of the run that starts a stream. The meshlet groups follow the whole run in the same
    order: group i belongs to block i, and each pair must still pass the dense-coverage check."""
    run, pos = block_run(data)
    add(T, "run_blocks", len(run))
    for i, (at, _, _, _) in enumerate(run):
        try:
            group, end = parse_group(data, pos)
        except PageError as exc:
            if i == 0:
                raise BlockError(f"no group: {exc}")
            add(errors, "no group in the stream for a later block of the run", len(run) - i)
            return
        try:
            measure_block(data, at, T, rng, (pos, group))
        except BlockError as exc:
            add(errors, str(exc).split(":")[0])
        pos = skip_padding(data, end)


def group_edges(data, gat, group):
    """Triangle vertex triples (block vertex indices) of a validated group."""
    meshlets = group["meshlets"]
    rec_at = gat + 12
    tri_at = rec_at + 16 * meshlets
    ref_at = tri_at + 4 * group["triangles"]
    refs = struct.unpack_from(f"<{group['vertex_refs']}H", data, ref_at)
    tris = []
    vbase = tbase = 0  # running sums: the record's u16 fields wrap past 65,535 (parse_group checks them mod 2^16)
    for m in range(meshlets):
        w0 = struct.unpack_from("<I", data, rec_at + 16 * m)[0]
        vcount, tcount = (w0 & 0x7F) + 1, ((w0 >> 7) & 0x7F) + 1
        for t in struct.unpack_from(f"<{tcount}I", data, tri_at + 4 * tbase):
            tris.append((refs[vbase + (t & 0x7F)], refs[vbase + ((t >> 7) & 0x7F)], refs[vbase + ((t >> 14) & 0x7F)]))
        vbase += vcount
        tbase += tcount
    return tris


def positions(data, at, slots, n):
    bias = struct.unpack_from("<3f", data, at + 0x60)
    scale = struct.unpack_from("<3f", data, at + 0x70)
    base = at + slots[0][1]
    out = []
    for v in range(n):
        q = struct.unpack_from("<Q", data, base + 8 * v)[0]
        x, y, z = (q >> 8) & 0xFFFFFF, (((q >> 32) & 0xFFFF) << 8) | (q & 0xFF), q >> 48
        out.append((bias[0] + scale[0] * x, bias[1] + scale[1] * y, bias[2] + scale[2] * z))
    return out, bias, scale


def face_normal(a, b, c):
    ux, uy, uz = b[0] - a[0], b[1] - a[1], b[2] - a[2]
    vx, vy, vz = c[0] - a[0], c[1] - a[1], c[2] - a[2]
    n = (uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx)
    l = math.sqrt(n[0] ** 2 + n[1] ** 2 + n[2] ** 2)
    return (n[0] / l, n[1] / l, n[2] / l) if l > 1e-12 else None


def _norm(x, y, z):
    l = math.sqrt(x * x + y * y + z * z)
    return (x / l, y / l, z / l) if l > 1e-6 else None


def _oct(x, y):
    """Octahedral unit vector from two values in [-1, 1]."""
    z = 1 - abs(x) - abs(y)
    if z < 0:
        x, y = (1 - abs(y)) * (1 if x >= 0 else -1), (1 - abs(x)) * (1 if y >= 0 else -1)
    return _norm(x, y, z)


def normal_0x21(w):
    """Unit normal from one tag-0x21 u32, decoded as the mesh shaders decode it (docs/restofworld.md).

    Bit 31 set: octahedral x = bits 0-9, y = bits 10-19, each unorm10 remapped by v * 2 / 1023 - 1.
    Bit 31 clear: octahedral x = byte 0, y = byte 1, each unorm8 remapped by v / 127.5 - 1.
    The other bits are not part of the normal."""
    if w >> 31:
        return _oct((w & 1023) * 2 / 1023 - 1, ((w >> 10) & 1023) * 2 / 1023 - 1)
    return _oct((w & 255) / 127.5 - 1, ((w >> 8) & 255) / 127.5 - 1)


def _s8(b):
    return max((b - 256 if b > 127 else b) / 127.0, -1.0)


def _u8(b):
    return b / 127.5 - 1.0


def _qt(x, y, z, w, col):
    """Column `col` of the rotation matrix of quaternion (x, y, z, w) (QTangent: col 2 = normal, col 0 = tangent)."""
    l = math.sqrt(x * x + y * y + z * z + w * w)
    if l < 1e-6:
        return None
    x, y, z, w = x / l, y / l, z / l, w / l
    if col == 2:
        return (2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y))
    if col == 1:
        return (2 * (x * y - w * z), 1 - 2 * (x * x + z * z), 2 * (y * z + w * x))
    return (1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y))


def _snorm10_components(word: int) -> tuple[float, float, float]:
    channels = [(word >> shift) & 0x3FF for shift in (0, 10, 20)]
    x, y, z = (max(-1.0, (v - 1024 if v >= 512 else v) / 511.0) for v in channels)
    return x, y, z


NORMAL_CANDIDATES = {
    **{f"qtan_snorm8_col{c}": (lambda c: lambda r, v: _qt(_s8(r[4 * v]), _s8(r[4 * v + 1]), _s8(r[4 * v + 2]), _s8(r[4 * v + 3]), c))(c) for c in (0, 1, 2)},
    **{f"qtan_unorm8_col{c}": (lambda c: lambda r, v: _qt(_u8(r[4 * v]), _u8(r[4 * v + 1]), _u8(r[4 * v + 2]), _u8(r[4 * v + 3]), c))(c) for c in (0, 1, 2)},
    **{f"qtan_10102_col{c}": (lambda c: lambda r, v: (lambda w: _qt(*[((w >> s) & 1023) / 511.5 - 1 for s in (0, 10, 20)], ((w >> 30) & 3) / 1.5 - 1, c))(struct.unpack_from("<I", r, 4 * v)[0]))(c) for c in (0, 1, 2)},
    "snorm8_xyz": lambda r, v: _norm(_s8(r[4 * v]), _s8(r[4 * v + 1]), _s8(r[4 * v + 2])),
    "unorm8_xyz": lambda r, v: _norm(_u8(r[4 * v]), _u8(r[4 * v + 1]), _u8(r[4 * v + 2])),
    "snorm10_xyz": lambda r, v: _norm(*_snorm10_components(struct.unpack_from("<I", r, 4 * v)[0])),
    "unorm10_xyz": lambda r, v: _norm(*[(w / 511.5 - 1) for w in
                                        (((struct.unpack_from("<I", r, 4 * v)[0]) >> s) & 1023 for s in (0, 10, 20))]),
    "oct_snorm8_b01": lambda r, v: _oct(_s8(r[4 * v]), _s8(r[4 * v + 1])),
    "oct_unorm8_b01": lambda r, v: _oct(_u8(r[4 * v]), _u8(r[4 * v + 1])),
    "oct_snorm8_b23": lambda r, v: _oct(_s8(r[4 * v + 2]), _s8(r[4 * v + 3])),
    "oct_unorm8_b23": lambda r, v: _oct(_u8(r[4 * v + 2]), _u8(r[4 * v + 3])),
    "oct_snorm16": lambda r, v: _oct(*[max(x / 32767.0, -1.0) for x in struct.unpack_from("<2h", r, 4 * v)]),
    "oct_unorm16": lambda r, v: _oct(*[x / 32767.5 - 1 for x in struct.unpack_from("<2H", r, 4 * v)]),
    "oct_bit31_10_or_8": lambda r, v: normal_0x21(struct.unpack_from("<I", r, 4 * v)[0]),
}


def half(h):
    s, e, m = h >> 15, (h >> 10) & 31, h & 1023
    if e == 31:
        return math.nan
    v = (m / 1024.0) * 2.0 ** -14 if e == 0 else (1 + m / 1024.0) * 2.0 ** (e - 15)
    return -v if s else v


def tri_area(a, b, c):
    ux, uy, uz = b[0] - a[0], b[1] - a[1], b[2] - a[2]
    vx, vy, vz = c[0] - a[0], c[1] - a[1], c[2] - a[2]
    return 0.5 * math.sqrt((uy * vz - uz * vy) ** 2 + (uz * vx - ux * vz) ** 2 + (ux * vy - uy * vx) ** 2)


def uv_area(a, b, c):
    return 0.5 * abs((b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1]))


def mad_log(ratios):
    ls = sorted(math.log(r) for r in ratios if r > 0 and math.isfinite(r))
    if len(ls) < 8:
        return None
    med = ls[len(ls) // 2]
    dev = sorted(abs(x - med) for x in ls)
    return dev[len(dev) // 2]


def add(d, k, v=1):
    d[k] = d.get(k, 0) + v


def measure_block(data, at, T, rng, found=None):
    slots, order, flags = parse_slots(data, at)
    n = vertex_count(slots, order)
    gat, group = found or find_group(data, at, slots, order, n)
    if group["vertex_span"] != n or group["unique_refs"] != n:
        raise BlockError("group does not cover the block densely")
    tris = group_edges(data, gat, group)
    edges = set()
    for a, b, c in tris:
        for p, q in ((a, b), (b, c), (c, a)):
            edges.add((p, q) if p < q else (q, p))
    edges = list(edges)
    pairs = [(rng.randrange(n), rng.randrange(n)) for _ in range(min(len(edges), 4096))]
    add(T, "blocks")
    add(T, "vertices", n)
    add(T, "edges", len(edges))
    pos, bias, scale = positions(data, at, slots, n)
    if all(math.isfinite(c) for c in bias + scale) and min(scale) >= 0:
        add(T, "pos_header_finite")
    ext = max(max(p[k] for p in pos) - min(p[k] for p in pos) for k in range(3)) if pos else 0
    if ext > 0 and all(math.dist(pos[p], pos[q]) <= ext * 1.0001 for p, q in edges):
        add(T, "pos_edges_within_block")
    # the mapping and face-normal tests run on a fixed sample of triangles per block (seeded rng, so a rerun
    # gives the same counts); the edge, wrap and per-byte counts above and below stay exhaustive
    stris = tris if len(tris) <= TRI_SAMPLE else rng.sample(tris, TRI_SAMPLE)
    pos_areas = [tri_area(pos[a], pos[b], pos[c]) for a, b, c in stris]
    # +0x6C = which model axis the 16-bit slot holds (CodeX GTA6 viewer: k=2 x,y,z; k=1 (y,z,x); k=0 (z,y,x));
    # normals are compared in model space
    k = struct.unpack_from("<I", data, at + 0x6C)[0]
    mpos = [(p[1], p[2], p[0]) if k == 1 else (p[2], p[1], p[0]) if k == 0 else p for p in pos]
    face_normals = [face_normal(mpos[a], mpos[b], mpos[c]) for a, b, c in stris]
    if edges:
        e = sum(math.dist(pos[p], pos[q]) for p, q in edges) / len(edges)
        r = sum(math.dist(pos[p], pos[q]) for p, q in pairs) / max(len(pairs), 1)
        if r > 0:
            T.setdefault("pos_edge_vs_random", []).append(e / r)
    hw = T.setdefault("header_words", {})
    add(hw.setdefault("u32_6c", {}), f"0x{struct.unpack_from('<I', data, at + 0x6C)[0]:x}")
    add(hw.setdefault("f32_7c", {}), repr(round(struct.unpack_from('<f', data, at + 0x7C)[0], 4)))
    add(hw.setdefault("u32_5c", {}), f"0x{flags:x}")
    for i, (tag, off) in slots.items():
        key = f"slot{i}_tag{tag:02x}"
        add(T.setdefault("slot_tags", {}), key)
        if i == 0:
            continue
        base = at + off
        st = stride(tag)
        tt = T.setdefault("tags", {}).setdefault(f"slot{i}:{tag:02x}", {})
        add(tt, "blocks")
        add(tt, "vertices", n)
        if tag == 0x28:
            u = struct.unpack_from(f"<{2 * n}H", data, base)
            wrap_u = wrap_s = 0
            for p, q in edges:
                for k in (0, 1):
                    a, b = u[2 * p + k], u[2 * q + k]
                    du = abs(a - b)
                    ds = abs((a - 65536 if a >= 32768 else a) - (b - 65536 if b >= 32768 else b))
                    wrap_u += du > 0x8000
                    wrap_s += ds > 0x8000
            add(tt, "edge_jumps_unsigned", wrap_u)
            add(tt, "edge_jumps_signed", wrap_s)
            hv = [half(x) for x in u]
            add(tt, "half_nan_or_inf", sum(1 for x in hv if not math.isfinite(x)))
            add(tt, "half_abs_over_1024", sum(1 for x in hv if math.isfinite(x) and abs(x) > 1024))
            add(tt, "u16_8192_over_1", sum(1 for v in range(n) if u[2 * v] > 8192 or u[2 * v + 1] > 8192))
            sv = [x - 65536 if x >= 32768 else x for x in u]
            add(tt, "s16_abs_over_8192", sum(1 for x in sv if abs(x) > 8192))
            for name in ("s16", "u16", "half"):
                vals = [x / 8192.0 for x in sv] if name == "s16" else [x / 8192.0 for x in u] if name == "u16" else hv
                uvs = [(vals[2 * v], vals[2 * v + 1]) for v in range(n)]
                if name == "half" and not all(math.isfinite(c) for uv in uvs for c in uv):
                    add(tt, "mapping_half_unusable")
                    continue
                r = [uv_area(uvs[a], uvs[b], uvs[c]) / pa for (a, b, c), pa in zip(stris, pos_areas) if pa > 1e-9]
                m = mad_log(r)
                if m is not None:
                    T.setdefault("mapping_spread", {}).setdefault(name, []).append(m)
        elif st == 4:
            raw = data[base:base + 4 * n]
            for k in range(4):
                col = raw[k::4]
                if len(set(col)) == 1:
                    add(tt, f"byte{k}_constant_blocks")
                e = sum(abs(col[p] - col[q]) for p, q in edges) / max(len(edges), 1)
                r = sum(abs(col[p] - col[q]) for p, q in pairs) / max(len(pairs), 1)
                if r > 0:
                    tt.setdefault(f"byte{k}_smooth", []).append(e / r)
            add(tt, "byte3_0", raw[3::4].count(0))
            add(tt, "byte3_255", raw[3::4].count(255))
            unit8 = unit10 = 0
            for v in range(n):
                x, y, z = (b - 256 if b > 127 else b for b in raw[4 * v:4 * v + 3])
                unit8 += abs(math.sqrt(x * x + y * y + z * z) / 127.0 - 1) < 0.03
                w = struct.unpack_from("<I", raw, 4 * v)[0]
                c = _snorm10_components(w)
                unit10 += abs(math.sqrt(sum(x * x for x in c)) - 1) < 0.03
            add(tt, "unit_snorm8", unit8)
            add(tt, "unit_snorm10", unit10)
            for name, dec in NORMAL_CANDIDATES.items():
                good = tot = 0
                for (a, b, c), fn in zip(stris, face_normals):
                    if fn is None:
                        continue
                    for v in (a, b, c):
                        nv = dec(raw, v)
                        if nv is None:
                            continue
                        tot += 1
                        good += abs(nv[0] * fn[0] + nv[1] * fn[1] + nv[2] * fn[2]) > 0.8
                if tot:
                    tt.setdefault(f"facenormal_{name}", []).append(good / tot)
    return n


def iter_blocks(path, limit):
    """Yield (archive, stream offset, inflated bytes) for zlib streams in an RPF9 data area starting 42 40 00 00.

    The RPF9 archive contract matches rpf9_names.scan_archive: at least 8 bytes, magic b'RPF9' or b'9FPR', and
    8 <= data_start <= size (an empty but valid header-only archive stays valid)."""
    found = 0
    with open(path, "rb") as fh:
        if os.fstat(fh.fileno()).st_size < 8:
            raise ValueError("not an RPF9 archive")
        m = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            size = len(m)
            if m[:4] not in (b"RPF9", b"9FPR"):
                raise ValueError("not an RPF9 archive")
            data_start = struct.unpack_from("<I", m, 4)[0]
            if not 8 <= data_start <= size:
                raise ValueError("RPF9 data start is outside the archive")
            pos = data_start
            while pos < size:
                match = ZHDR.search(m, pos)
                if not match:
                    break
                s = match.start()
                d = zlib.decompressobj()
                try:
                    head = d.decompress(m[s:s + 4096], 4)
                except zlib.error:
                    pos = s + 1
                    continue
                if head != b"\x42\x40\x00\x00":
                    pos = s + 1
                    continue
                d = zlib.decompressobj()
                parts, p = [], s
                try:
                    while not d.eof and p < size:
                        chunk = m[p:p + (1 << 20)]
                        parts.append(d.decompress(chunk))
                        p += len(chunk)
                        if d.unused_data or d.eof:
                            break
                except zlib.error:
                    pos = s + 1
                    continue
                if not d.eof:
                    pos = s + 1
                    continue
                yield os.path.basename(path), s, b"".join(parts) + d.flush()
                found += 1
                if limit and found >= limit:
                    return
                pos = p - len(d.unused_data)
        finally:
            m.close()


def summarise(T):
    out = {k: v for k, v in T.items() if k not in ("tags", "mapping_spread", "pos_edge_vs_random", "header_words")}
    pe = sorted(T.get("pos_edge_vs_random", []))
    out["pos_edge_vs_random_median"] = round(pe[len(pe) // 2], 4) if pe else None
    out["header_words"] = {k: dict(sorted(v.items(), key=lambda kv: -kv[1])[:6]) for k, v in T.get("header_words", {}).items()}
    tags = {}
    for tag, tt in sorted(T.get("tags", {}).items()):
        row = {}
        for k, v in tt.items():
            if isinstance(v, list):
                v = sorted(v)
                row[k + "_median"] = round(v[len(v) // 2], 3) if v else None
            else:
                row[k] = v
        tags[tag] = row
    out["tags"] = tags
    spread = {}
    for name, v in T.get("mapping_spread", {}).items():
        v = sorted(v)
        spread[name] = {"blocks": len(v), "median": round(v[len(v) // 2], 3), "p90": round(v[len(v) * 9 // 10], 3)}
    out["mapping_spread"] = spread
    return out


def same_output_path(first, second):
    if os.path.normcase(os.path.realpath(first)) == os.path.normcase(os.path.realpath(second)):
        return True
    try:
        return os.path.samefile(first, second)
    except OSError:
        return False


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("inputs", nargs="+", help=".rpf archives or folders of them")
    ap.add_argument("--limit", type=int, default=0, help="stop after this many 42 40 00 00 streams per archive")
    ap.add_argument("--json", help="write the totals here")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args(argv)
    if args.limit < 0:
        print("error: --limit must be >= 0", file=sys.stderr)
        return 2
    archives = []
    for p in args.inputs:
        if not os.path.exists(p):
            print(f"error: {p} does not exist", file=sys.stderr)
            return 2
        if os.path.isdir(p):
            for root, directories, files in os.walk(p):
                directories.sort()
                archives += [os.path.join(root, f) for f in sorted(files) if f.lower().endswith(".rpf")]
        else:
            archives.append(p)
    # The report is written after the whole scan, so check its path first: an archive given as --json used to be
    # replaced by the report (exit 0), and a directory ended the run in a traceback.
    if args.json:
        if os.path.isdir(args.json):
            print(f"error: --json {args.json} is a directory", file=sys.stderr)
            return 2
        for path in archives:
            if same_output_path(args.json, path):
                print(f"error: --json would overwrite the input {path}", file=sys.stderr)
                return 2
    rng = random.Random(args.seed)
    T = {"streams": 0}
    errors = {}
    for path in archives:
        try:
            for _, _, data in iter_blocks(path, args.limit):
                T["streams"] += 1
                try:
                    measure_stream(data, T, rng, errors)
                except BlockError as exc:
                    add(errors, str(exc).split(":")[0])
                except (struct.error, IndexError):
                    add(errors, "truncated")
        except (ValueError, OSError) as exc:
            print(f"error: {path}: {exc}", file=sys.stderr)
            return 2
        print(f"{os.path.basename(path)}: {T['streams']} streams so far, {T.get('blocks', 0)} blocks measured", file=sys.stderr)
    out = summarise(T)
    out["skipped"] = errors
    print(json.dumps(out, indent=1))
    if args.json:
        try:
            os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
            with open(args.json, "w", encoding="utf-8", newline="\n") as fh:
                json.dump(out, fh, indent=1)
        except OSError as exc:
            print(f"error: cannot write {args.json}: {exc.strerror or exc}", file=sys.stderr)
            return 2
    return 0


if __name__ == "__main__":
    try:  # a redirected stdout is strict ANSI code page under -I; write '?' for what it cannot encode
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    sys.exit(main())
