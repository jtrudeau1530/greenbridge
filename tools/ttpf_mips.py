"""Measure and decode the TTPF (RDS5) compact mip chain, including the packed tail.

The compact profile in `TtpfReader` is the AMD Gfx10 mip-tail layout: mips smaller
than half a tile share one tile at offset 0, and the rest are standalone, smallest
first. `TtpfReader` only trusts a hand-validated tuple table and stops at the
descriptor's `nonTail` field, which is not the layout boundary. This tool computes
the layout from the descriptor and validates it against the native allocation span,
then decodes the mips that are zlib/raw (no BCPack backend is used or guessed).

Read-only: no texture bytes are printed or committed. `scan` emits aggregate
counters only. `layout` prints descriptor fields and offsets. `check` prints the
BC block-average error against a 2x2 downsample of the parent mip; it decodes
BC1, BC3, BC4_UNORM and BC5_UNORM (BC2, the SNORM BC4/5 variants, BC6H and BC7
are sized but not decoded, and uncompressed tiling is not validated). `export`
writes a full BC chain only when every source mip is zlib/raw.

    python -I tools/ttpf_mips.py scan <game root or pool paths...> [--json out]
    python -I tools/ttpf_mips.py layout <pool.ttpf> <index>
    python -I tools/ttpf_mips.py check <pool.ttpf> <index>
    python -I tools/ttpf_mips.py export <pool.ttpf> <index> <out.dds> [--mips N] [--force]

Standard library only. The RDS5/ETPF checks mirror rpf9-explorer's `TtpfReader`
(MIT); the tile patterns and tail formula are the AMD Gfx10 4 KiB/64 KiB layouts
already used by that reader and recorded in docs/textures.md.
"""
import argparse
import glob
import json
import mmap
import os
import struct
import sys
import zlib

MAX_OUTPUT = 256 * 1024 * 1024
MAX_TEXTURES = 250_000
ETPF_VERSION = 634

# DXGI format -> element bytes, as RageTextureDecoder.ElementBytes. DXGI 11
# (R16G16B16A16_UNORM) is 8 bytes but is left out here as well: its uncompressed
# tiling has not been validated, so `scan` reports it as a format gap.
ELEMENT_BYTES = {
    2: 16, 3: 16, 4: 16, 6: 12, 7: 12, 8: 12, 10: 8, 12: 8, 13: 8, 14: 8,
    16: 8, 17: 8, 18: 8, 24: 4, 25: 4, 26: 4, 28: 4, 29: 4, 30: 4, 31: 4, 32: 4,
    34: 4, 35: 4, 36: 4, 37: 4, 38: 4, 41: 4, 42: 4, 43: 4, 49: 2, 50: 2, 51: 2,
    52: 2, 54: 2, 56: 2, 57: 2, 58: 2, 59: 2, 61: 1, 62: 1, 63: 1, 64: 1, 65: 1,
    71: 8, 72: 8, 74: 16, 75: 16, 77: 16, 78: 16, 80: 8, 81: 8, 83: 16, 84: 16,
    87: 4, 88: 4, 91: 4, 93: 4, 95: 16, 96: 16, 98: 16, 99: 16,
}
BC_FORMATS = frozenset(range(70, 85)) | frozenset(range(94, 100))

# AMD Gfx10 intra-tile bit order (bit index -> source bit), from TtpfReader.
PATTERNS_4K = {
    0: (1, 2, 3, 4, -1, -2, -3, -4, -5, 5, -6, 6),
    1: (0, 1, 2, 3, -1, -2, -3, 4, -4, 5, -5, 6),
    2: (0, 0, 1, 2, -1, -2, -3, 3, -4, 4, -5, 5),
    3: (0, 0, 0, 1, -1, -2, 2, 3, -3, 4, -4, 5),
    4: (0, 0, 0, 0, -1, -2, 1, 2, -3, 3, -4, 4),
}
PATTERNS_64K = {
    0: (1, 2, 3, -2, -1, -3, 4, -4, -5, 5, -6, 6, -7, 7, -8, 8),
    1: (0, 1, 2, 3, -1, -2, -3, 4, -4, 5, -5, 6, -6, 7, -7, 8),
    2: (0, 0, 1, 2, -1, -2, -3, 3, -4, 4, -5, 5, -6, 6, -7, 7),
    3: (0, 0, 0, 1, -1, 2, 3, -2, -3, 4, -4, 5, -5, 6, -6, 7),
    4: (0, 0, 0, 0, 1, -1, 2, -2, -3, 3, -4, 4, -5, 5, -6, 6),
}

# The profile tuples `TtpfReader.ValidatedCompactProfile` accepts. Reproduced only
# so `scan` can show the reader's published skip/partial split next to the layout
# result; decoding does not depend on this table.
VALIDATED_COMPACT = {
    (4, 1024, 1024, 9, 3, 8, 0x503, 0x00B00106),
    (6, 512, 512, 8, 3, 8, 0x041, 0x00B800D4),
    (4, 256, 256, 7, 1, 8, 0x103, 0),
    (4, 512, 512, 8, 3, 16, 0x482, 0x015000F5),
    (4, 1024, 512, 8, 3, 8, 0x482, 0x015000F5),
    (4, 512, 1024, 8, 3, 8, 0x483, 0x013000F5),
    (4, 128, 256, 6, 1, 8, 0x083, 0),
    (4, 256, 512, 7, 1, 8, 0x167, 0),
    (4, 512, 256, 7, 1, 8, 0x166, 0),
    (4, 256, 256, 7, 1, 16, 0x166, 0),
    (4, 1024, 512, 8, 3, 16, 0x504, 0x900106),
    (4, 1024, 256, 7, 3, 16, 0x484, 0x11000F5),
    (6, 1024, 256, 7, 3, 8, 0x42, 0x9800D4),
    (6, 64, 1024, 5, 3, 8, 0x83, 0x3800B4),
    (4, 512, 1024, 8, 3, 16, 0x503, 0xB00106),
    (4, 512, 128, 6, 1, 16, 0x168, 0),
    (6, 256, 1024, 7, 3, 8, 0x83, 0x3800D4),
    (4, 1024, 128, 6, 1, 8, 0x168, 0),
    (4, 2048, 256, 7, 3, 16, 0x508, 0x100106),
    (4, 2048, 512, 8, 3, 8, 0x504, 0x900106),
    (6, 512, 256, 7, 3, 16, 0x42, 0x9800D4),
    (6, 1024, 128, 6, 3, 16, 0x84, 0x1800D4),
    (4, 512, 128, 6, 1, 8, 0x104, 0),
    (4, 256, 1024, 7, 3, 16, 0x483, 0x13000F5),
    (6, 1024, 64, 5, 3, 16, 0x83, 0x3800B4),
    (6, 256, 512, 7, 3, 16, 0x41, 0xB800D4),
    (4, 128, 128, 6, 1, 16, 0x82, 0),
    (4, 512, 64, 5, 1, 16, 0x108, 0),
    (4, 256, 128, 6, 1, 16, 0x104, 0),
    (6, 128, 1024, 6, 3, 16, 0x83, 0x3800D4),
    (4, 512, 64, 5, 1, 8, 0x84, 0),
    (4, 128, 512, 6, 1, 16, 0x167, 0),
    (4, 256, 128, 6, 1, 8, 0x82, 0),
    (4, 1024, 64, 5, 1, 8, 0x108, 0),
    (6, 1024, 32, 4, 3, 16, 0x82, 0x5800B4),
    (4, 128, 512, 6, 1, 8, 0x107, 0),
    (4, 512, 2048, 8, 3, 8, 0x507, 0x300106),
    (4, 256, 2048, 7, 3, 8, 0x507, 0x3000F5),
    (4, 512, 32, 4, 1, 16, 0x107, 0),
    (6, 128, 1024, 6, 3, 8, 0x83, 0x3800B4),
    (6, 64, 1024, 5, 3, 16, 0x83, 0x3800B4),
    (4, 256, 256, 5, 1, 16, 0x166, 0),
    (4, 128, 256, 6, 1, 16, 0x103, 0),
    (4, 256, 64, 5, 1, 16, 0x84, 0),
    (4, 16, 256, 3, 1, 16, 0x83, 0),
    (4, 256, 32, 4, 1, 16, 0x84, 0),
    (4, 64, 512, 5, 1, 16, 0x107, 0),
    (4, 2048, 256, 7, 3, 8, 0x484, 0x11000F5),
    (4, 64, 512, 5, 1, 8, 0x107, 0),
    (4, 64, 256, 5, 1, 8, 0x83, 0),
    (6, 256, 512, 9, 3, 16, 0x41, 0xB800D4),
    (6, 32, 1024, 4, 3, 16, 0x82, 0x5800B4),
    (4, 512, 512, 3, 3, 8, 0x502, 0xD00006),
    (36, 1024, 1024, 9, 3, 8, 0x503, 0xB00106),
    (36, 256, 256, 7, 1, 8, 0x103, 0),
}
VALIDATED_AUX = dict(mips=3, non_tail=3, meta=0x1A8, extra=0x01080008, width=1024,
                     height_a=512, fmt_a=61, height_b=1024, bytes_b=8)


class LayoutError(ValueError):
    pass


class BcpackRequired(NotImplementedError):
    pass


def u32(b, o):
    return struct.unpack_from("<I", b, o)[0]


def u64(b, o):
    return struct.unpack_from("<Q", b, o)[0]


def align(v, n):
    return (v + n - 1) // n * n


def tile_mode(mode):
    """Descriptor mode is a flag bit plus a 4-bit Gfx10 tile mode at bit 1."""
    return (mode >> 1) & 0xF


def tile_bytes(mode):
    # Tile mode 0 is SW_64KB_D; 1-3 are SW_4KB_S; 9-12 appear only on arrays.
    return 65536 if tile_mode(mode) == 0 else 4096


def element_bytes(fmt):
    if fmt not in ELEMENT_BYTES:
        raise LayoutError(f"no element-size mapping for DXGI format {fmt}")
    return ELEMENT_BYTES[fmt]


def surface_layout(width, height, fmt, mips, tile=4096):
    """AMD Gfx10 surface layout: standalone mips smallest-first, then one tail tile.

    Returns a dict with the allocation size, the first tail mip index, block
    dimensions and a per-mip list. `offset` is relative to the allocation base.
    """
    eb = element_bytes(fmt)
    log = {1: 0, 2: 1, 4: 2, 8: 3, 16: 4}[eb]
    px = 4 if fmt in BC_FORMATS else 1
    levels = 1
    v = max(width, height)
    while v > 1:
        levels += 1
        v >>= 1
    if mips < 1 or mips > levels:
        raise LayoutError(f"invalid mip count {mips} for {width}x{height}")
    half_bits = 6 if tile == 4096 else 8
    max_tail = 8 if tile == 4096 else 12
    bw = 1 << (half_bits - log // 2)
    bh = 1 << (half_bits - (log + 1) // 2)
    mw, mh = {0: (16, 16), 1: (16, 8), 2: (8, 8), 3: (8, 4)}.get(log, (4, 4))
    ew = (width + px - 1) // px
    eh = (height + px - 1) // px
    tail = mips
    sizes = []
    out = []
    for i in range(mips):
        wc = max(1, (ew + (1 << i) - 1) >> i)
        hc = max(1, (eh + (1 << i) - 1) >> i)
        if mips > 1 and wc <= bw // 2 and hc <= bh and mips - i <= max_tail:
            tail = i
            break
        pw = align(wc, bw)
        ph = align(hc, bh)
        sizes.append(pw * ph * eb)
        out.append(dict(index=i, blocks_w=wc, blocks_h=hc, padded_w=pw, padded_h=ph,
                        pixels_w=max(1, width >> i), pixels_h=max(1, height >> i),
                        offset=0, region=pw * ph * eb, tile_x=0, tile_y=0, in_tail=False))
    off = tile if tail < mips else 0
    for i in range(tail - 1, -1, -1):
        out[i]["offset"] = off
        off += sizes[i]
    for i in range(tail, mips):
        entry = max_tail - 1 - (i - tail)
        moff = (16 << entry) if entry > 6 else entry << 8
        x = y = 0
        for bit in range(6):
            x |= ((moff >> (9 + 2 * bit)) & 1) << bit
            y |= ((moff >> (8 + 2 * bit)) & 1) << bit
        wc = max(1, (ew + (1 << i) - 1) >> i)
        hc = max(1, (eh + (1 << i) - 1) >> i)
        out.append(dict(index=i, blocks_w=wc, blocks_h=hc, padded_w=bw, padded_h=bh,
                        pixels_w=max(1, width >> i), pixels_h=max(1, height >> i),
                        offset=0, region=tile, tile_x=x * mw, tile_y=y * mh,
                        in_tail=True, tail_offset=moff))
    return dict(size=off, tile=tile, first_tail=tail, block_w=bw, block_h=bh,
                element_bytes=eb, mips=out)


def detile(layout, mip, tiled):
    """Copy the mip's elements out of the tiled allocation into block-linear order."""
    pat = (PATTERNS_4K if layout["tile"] == 4096 else PATTERNS_64K)[
        {1: 0, 2: 1, 4: 2, 8: 3, 16: 4}[layout["element_bytes"]]]
    bw, bh = layout["block_w"], layout["block_h"]
    eb = layout["element_bytes"]
    out = bytearray(mip["blocks_w"] * mip["blocks_h"] * eb)
    for y in range(mip["blocks_h"]):
        for x in range(mip["blocks_w"]):
            off = ((y // bh) * (mip["padded_w"] // bw) + x // bw) * layout["tile"]
            for bit, s in enumerate(pat):
                if s > 0:
                    off |= (((x + mip["tile_x"]) >> (s - 1)) & 1) << bit
                elif s < 0:
                    off |= (((y + mip["tile_y"]) >> (-s - 1)) & 1) << bit
            at = (y * mip["blocks_w"] + x) * eb
            out[at:at + eb] = tiled[off:off + eb]
    return bytes(out)


# --- BC decoding (color only; enough for the downsample check) -------------

def _rgb565(c):
    return ((c >> 11) & 31) * 255.0 / 31, ((c >> 5) & 63) * 255.0 / 63, (c & 31) * 255.0 / 31


def decode_bc1_block(data, at):
    c0, c1, idx = struct.unpack_from("<HHI", data, at)
    p0, p1 = _rgb565(c0), _rgb565(c1)
    if c0 > c1:
        pal = [p0, p1, tuple((2 * a + b) / 3 for a, b in zip(p0, p1)),
               tuple((a + 2 * b) / 3 for a, b in zip(p0, p1))]
    else:
        pal = [p0, p1, tuple((a + b) / 2 for a, b in zip(p0, p1)), (0.0, 0.0, 0.0)]
    return [pal[(idx >> (2 * (y * 4 + x))) & 3] for y in range(4) for x in range(4)]


def decode_bc4_block(data, at):
    e0, e1 = data[at], data[at + 1]
    bits = int.from_bytes(data[at + 2:at + 8], "little")
    if e0 > e1:
        pal = [e0, e1] + [((7 - i) * e0 + i * e1) // 7 for i in range(1, 7)]
    else:
        pal = [e0, e1] + [((5 - i) * e0 + i * e1) // 5 for i in range(1, 5)] + [0, 255]
    vals = [pal[(bits >> (3 * (y * 4 + x))) & 7] for y in range(4) for x in range(4)]
    return [(float(v),) * 3 for v in vals]


def decode_pixels(fmt, data, blocks_w, blocks_h):
    """Return one (r, g, b) per pixel, or None for a format this tool cannot decode."""
    px = []
    for by in range(blocks_h):
        rows = [[None] * (blocks_w * 4) for _ in range(4)]
        for bx in range(blocks_w):
            at = (by * blocks_w + bx) * 16
            if fmt in (71, 72):
                blk = decode_bc1_block(data, (by * blocks_w + bx) * 8)
            elif fmt in (77, 78):
                blk = decode_bc1_block(data, at + 8)
            elif fmt in (80,):
                blk = decode_bc4_block(data, (by * blocks_w + bx) * 8, False)
            elif fmt in (83,):
                a = decode_bc4_block(data, at, False)
                b = decode_bc4_block(data, at + 8, False)
                blk = [(p[0], q[0], 0.0) for p, q in zip(a, b)]
            else:
                return None
            for y in range(4):
                for x in range(4):
                    rows[y][bx * 4 + x] = blk[y * 4 + x]
        px.extend(rows)
    return [p for row in px for p in row]


def block_means(pixels, blocks_w, blocks_h):
    out = []
    for by in range(blocks_h):
        for bx in range(blocks_w):
            acc = [0.0, 0.0, 0.0]
            for y in range(4):
                for x in range(4):
                    q = pixels[(by * 4 + y) * (blocks_w * 4) + bx * 4 + x]
                    for k in range(3):
                        acc[k] += q[k]
            out.append((acc[0] / 16, acc[1] / 16, acc[2] / 16))
    return out


def downsample2(blocks, blocks_w, blocks_h):
    ow, oh = max(1, blocks_w // 2), max(1, blocks_h // 2)
    out = []
    for y in range(oh):
        for x in range(ow):
            acc = [0.0, 0.0, 0.0]
            n = 0
            for dy in range(2):
                for dx in range(2):
                    bx = min(blocks_w - 1, 2 * x + dx)
                    by = min(blocks_h - 1, 2 * y + dy)
                    q = blocks[by * blocks_w + bx]
                    for k in range(3):
                        acc[k] += q[k]
                    n += 1
            out.append(tuple(a / n for a in acc))
    return out


def mean_abs_error(a, b):
    if not a:
        return None
    return sum(abs(x - y) for pa, pb in zip(a, b) for x, y in zip(pa, pb)) / (3 * len(a))


# --- RDS5 / ETPF container ------------------------------------------------

class Pool:
    """Read-only RDS5 texture pool, mirroring TtpfReader's catalog checks."""

    def __init__(self, path):
        self.path = path
        self._fh = open(path, "rb")
        try:
            self.data = mmap.mmap(self._fh.fileno(), 0, access=mmap.ACCESS_READ)
        except Exception:
            self._fh.close()
            raise
        d = self.data
        if len(d) < 256 or d[:4] != b"RDS5":
            raise ValueError("not an RDS5 container or truncated header")
        count = u32(d, 4)
        size_word = u64(d, 8)
        self.virtual_size = size_word & 0x00FFFFFFFFFFFFFF
        if (count < 3 or count > 1_000_002 or (size_word >> 56) != 22 or
                self.virtual_size < 4096 or u32(d, 16) not in
                ((count + 31) // 32, (count + 31) // 32 + 1)):
            raise ValueError("unsupported RDS5 index layout")
        self.chunk_count = count - 2
        table_end = 24 + (count - 1) * 8
        self.etpf = align(table_end, 256)
        if table_end > len(d) - 16:
            raise ValueError("RDS5 index exceeds its source")
        table = d[24:table_end]
        if ((u64(table, len(table) - 8) & 0x3FFFFFF) * 256 != len(d) or
                ((u64(table, len(table) - 8) >> 30) & 0x3FFFFF) * 4096 != self.virtual_size):
            raise ValueError("RDS5 sentinel does not match its lengths")
        self.chunks = []
        for i in range(self.chunk_count):
            v, nxt = u64(table, i * 8), u64(table, (i + 1) * 8)
            start, end = (v & 0x3FFFFFF) * 256, (nxt & 0x3FFFFFF) * 256
            virt, vend = ((v >> 30) & 0x3FFFFF) * 4096, ((nxt >> 30) & 0x3FFFFF) * 4096
            pad = v >> 52
            codec = (v >> 26) & 15
            packed, size = end - start - pad, vend - virt
            if (start < self.etpf + 16 or end > len(d) or packed < 1 or packed > MAX_OUTPUT or
                    size < 1 or size > MAX_OUTPUT or vend > self.virtual_size or
                    (codec == 0 and packed != size)):
                raise ValueError(f"RDS5 chunk {i} range is invalid")
            self.chunks.append((virt, size, start, packed, codec, pad))
        eh = d[self.etpf:self.etpf + 16]
        if eh[:4] != b"ETPF" or u32(eh, 4) != ETPF_VERSION:
            raise ValueError("unsupported embedded ETPF version")
        self.textures = u32(eh, 8)
        area = u32(eh, 12)
        metadata = 16 + area + self.textures * 32
        padded = align(metadata, 4096)
        alignment = u32(d, 20)
        source_padding = (alignment >> 28) * 256
        if (self.textures < 1 or self.textures > MAX_TEXTURES or area < self.textures * 16 or
                (alignment & 0x0FFFFFFF) != 0 or padded != self.chunks[0][0] or
                self.etpf + padded + source_padding != self.chunks[0][2]):
            raise ValueError("ETPF tables do not match the data boundary")
        self.area = area
        self.records = d[self.etpf + 16:self.etpf + 16 + area]
        self.ids = d[self.etpf + 16 + area:self.etpf + 16 + area + self.textures * 32]
        self._starts = [c[0] for c in self.chunks]

    def close(self):
        self.data.close()
        self._fh.close()

    def descriptor(self, index):
        if not 0 <= index < self.textures:
            raise IndexError(index)
        a, b, c, d = struct.unpack_from("<IIII", self.records, index * 16)
        return dict(
            index=index, width=(a & 0x3FFF) + 1, height=((a >> 14) & 0x3FFF) + 1,
            mips=(a >> 28) + (b & 7), packed_mips=a >> 28, non_tail=b & 7,
            format=(b >> 3) & 255, depth=((b >> 11) & 1023) + 1,
            dimension=(b >> 21) & 3, mode=b >> 23, meta=c >> 21, extra=d,
            base=(c & 0x1FFFFF) * 4096, asset_key_at=(
                u64(self.records, self.textures * 16 + index * 8)
                if (self.textures * 16 + index * 8 + 8 <= len(self.records))
                else None),
        )

    def chunks_for(self, start, size):
        import bisect
        end = start + size
        i = max(0, bisect.bisect_right(self._starts, start) - 1)
        out = []
        covered = start
        while i < len(self.chunks) and self.chunks[i][0] < end:
            virt, csize, src, packed, codec, pad = self.chunks[i]
            if virt > covered:
                raise ValueError("texture crosses an unmapped virtual range")
            out.append(self.chunks[i])
            covered = min(end, virt + csize)
            i += 1
        if covered != end:
            raise ValueError("texture range is not completely mapped")
        return out

    def codecs(self, start, size):
        return {c[4] for c in self.chunks_for(start, size)}

    def readable(self, start, size):
        return self.codecs(start, size) <= {0, 8}

    def read_range(self, start, size, cache=None):
        cache = {} if cache is None else cache
        out = bytearray(size)
        for virt, csize, src, packed, codec, pad in self.chunks_for(start, size):
            key = (src, packed, codec)
            if key not in cache:
                raw = self.data[src:src + packed]
                if codec == 0:
                    cache[key] = bytes(raw)
                elif codec == 8:
                    cache[key] = zlib.decompress(raw)
                else:
                    raise BcpackRequired("range touches a BCPack chunk")
                if len(cache[key]) != csize:
                    raise ValueError("decoded chunk length disagrees with its range")
            lo, hi = max(start, virt), min(start + size, virt + csize)
            out[lo - start:hi - start] = cache[key][lo - virt:hi - virt]
        return bytes(out)

    def span(self, index, base):
        for j in range(index + 1, self.textures):
            cand = (u32(self.records, j * 16 + 8) & 0x1FFFFF) * 4096
            if cand < base:
                return None
            if cand > base:
                return cand - base
        return self.virtual_size - base


def compact_row(desc):
    """The reader's compact-profile gate (mode families 4/6/36 and aux mode 2)."""
    return desc["mode"] in (4, 6, 36) or (desc["mode"] == 2 and (desc["meta"] or desc["extra"]))


def validated_compact(desc, eb):
    return (desc["mode"], desc["width"], desc["height"], desc["mips"], desc["non_tail"],
            eb, desc["meta"], desc["extra"]) in VALIDATED_COMPACT


def validated_aux(desc):
    a = VALIDATED_AUX
    return (desc["mips"] == a["mips"] and desc["non_tail"] == a["non_tail"] and
            desc["meta"] == a["meta"] and desc["extra"] == a["extra"] and
            desc["width"] == a["width"] and
            ((desc["height"] == a["height_a"] and desc["format"] == a["fmt_a"]) or
             (desc["height"] == a["height_b"] and desc["format"] in BC_FORMATS and
              ELEMENT_BYTES.get(desc["format"]) == a["bytes_b"])))


def analyze(pool, index):
    """Descriptor + layout + per-mip codec availability for one texture."""
    d = pool.descriptor(index)
    out = dict(descriptor=d, tile_mode=tile_mode(d["mode"]), tile=tile_bytes(d["mode"]))
    if d["dimension"] != 1 or d["depth"] != 1:
        out["blocked"] = "array/volume descriptor"
        return out
    try:
        layout = surface_layout(d["width"], d["height"], d["format"], d["mips"],
                                out["tile"])
    except LayoutError as e:
        out["blocked"] = str(e)
        return out
    out["layout"] = layout
    if d["base"] == 0 or (d["base"] >> 12) == 0x1FFFFF:
        out["blocked"] = "no mapped allocation"
        return out
    span = pool.span(index, d["base"])
    out["allocation_span"] = span
    out["span_matches"] = span == layout["size"]
    if not out["span_matches"]:
        out["blocked"] = "layout does not match the allocation span"
        return out
    avail = []
    for mip in layout["mips"]:
        try:
            avail.append(pool.readable(d["base"] + mip["offset"], mip["region"]))
        except ValueError:
            avail = []
            break
    out["available"] = avail
    if avail:
        limit = d["non_tail"] if compact_row(d) else d["mips"]
        limit = max(0, min(limit, len(avail)))
        base_mip = next((i for i in range(limit) if avail[i]), None)
        exported = 0
        if base_mip is not None:
            while base_mip + exported < limit and avail[base_mip + exported]:
                exported += 1
        out["reader_limit"] = limit
        out["reader_base_mip"] = base_mip
        out["reader_exported"] = exported
        out["full_chain"] = bool(avail) and all(avail)
        out["recoverable_mips"] = sum(1 for a in avail[exported:] if a) if base_mip == 0 else 0
    return out


# --- commands --------------------------------------------------------------

def find_pools(arg):
    if os.path.isfile(arg):
        return [arg]
    pools = sorted(glob.glob(os.path.join(arg, "data", "ttdb", "**", "*.ttpf"), recursive=True))
    pools += sorted(glob.glob(os.path.join(arg, "pack", "*", "data", "ttdb", "**", "*.ttpf"),
                                recursive=True))
    return pools


def scan_pools(paths, as_json=None):
    total = dict(pools=0, entries=0, arrays=0, unsupported_format=0, format11=0,
                 layout_mismatch=0, reader_full=0, reader_partial=0,
                 reader_skip_bcpack=0, whitelist_gated=0, gated_layout_ok=0,
                 gated_full_chain=0, gated_full_mips=0, gated_full_bytes=0,
                 gated_tail_only=0, gated_by_mode={},
                 compact_partial_rows=0, compact_missing_mips=0,
                 compact_decodable_mips=0, compact_decodable_bytes=0,
                 compact_missing_standalone=0, compact_missing_tail=0,
                 compact_rows_measured=0, compact_non_tail_before_tail=0,
                 compact_standalone_beyond_limit=0,
                 compact_standalone_decodable_beyond_limit=0,
                 compact_full_rows=0, tile_modes={})
    for path in paths:
        pool = Pool(path)
        try:
            total["pools"] += 1
            for i in range(pool.textures):
                d = pool.descriptor(i)
                total["entries"] += 1
                tm = str(tile_mode(d["mode"]))
                total["tile_modes"][tm] = total["tile_modes"].get(tm, 0) + 1
                if d["dimension"] != 1 or d["depth"] != 1:
                    total["arrays"] += 1
                    continue
                # The reader's gate: modes 4/6/36 and aux mode 2 need a validated tuple;
                # every other non-0/2 mode is rejected by the same branch.
                aux = d["mode"] == 2 and (d["meta"] or d["extra"])
                if d["mode"] in (0, 2):
                    gated = d["mode"] == 2 and aux and not validated_aux(d)
                else:
                    eb = ELEMENT_BYTES.get(d["format"])
                    gated = not (d["mode"] in (4, 6, 36) and eb is not None and
                                 validated_compact(d, eb))
                try:
                    layout = surface_layout(d["width"], d["height"], d["format"],
                                            d["mips"], tile_bytes(d["mode"]))
                except LayoutError:
                    total["unsupported_format"] += 1
                    if d["format"] == 11:
                        total["format11"] += 1
                    if gated:
                        total["whitelist_gated"] += 1
                    continue
                span = pool.span(i, d["base"])
                if d["base"] == 0 or span != layout["size"]:
                    total["layout_mismatch"] += 1
                    continue
                avail = []
                for mip in layout["mips"]:
                    try:
                        avail.append(pool.readable(d["base"] + mip["offset"], mip["region"]))
                    except ValueError:
                        avail = []
                        break
                if not avail:
                    total["layout_mismatch"] += 1
                    continue
                if gated:
                    total["whitelist_gated"] += 1
                    total["gated_layout_ok"] += 1
                    key = str(d["mode"])
                    total["gated_by_mode"][key] = total["gated_by_mode"].get(key, 0) + 1
                    if all(avail):
                        total["gated_full_chain"] += 1
                        total["gated_full_mips"] += d["mips"]
                        total["gated_full_bytes"] += sum(
                            mip["blocks_w"] * mip["blocks_h"] * layout["element_bytes"]
                            for mip in layout["mips"])
                    elif (layout["first_tail"] < d["mips"] and
                          all(avail[k] for k in range(layout["first_tail"], d["mips"]))):
                        total["gated_tail_only"] += 1
                    continue
                is_compact = compact_row(d)
                limit = min(d["non_tail"], len(avail)) if is_compact else len(avail)
                base_mip = next((k for k in range(limit) if avail[k]), None)
                exported = 0
                if base_mip is not None:
                    while base_mip + exported < limit and avail[base_mip + exported]:
                        exported += 1
                if exported == 0:
                    total["reader_skip_bcpack"] += 1
                elif exported < d["mips"]:
                    total["reader_partial"] += 1
                    if is_compact and base_mip == 0 and exported == limit:
                        total["compact_partial_rows"] += 1
                        missing = d["mips"] - exported
                        total["compact_missing_mips"] += missing
                        dec = 0
                        for k in range(exported, d["mips"]):
                            if avail[k]:
                                dec += 1
                                if k < layout["first_tail"]:
                                    total["compact_missing_standalone"] += 1
                                else:
                                    total["compact_missing_tail"] += 1
                        total["compact_decodable_mips"] += dec
                        total["compact_decodable_bytes"] += sum(
                            layout["mips"][k]["blocks_w"] * layout["mips"][k]["blocks_h"] *
                            layout["element_bytes"] for k in range(exported, d["mips"])
                            if avail[k])
                        if dec == missing:
                            total["compact_full_rows"] += 1
                else:
                    total["reader_full"] += 1
                if d["mode"] in (4, 6, 36):
                    total["compact_rows_measured"] += 1
                    if d["non_tail"] < layout["first_tail"]:
                        total["compact_non_tail_before_tail"] += 1
                        total["compact_standalone_beyond_limit"] += (
                            layout["first_tail"] - d["non_tail"])
                        total["compact_standalone_decodable_beyond_limit"] += sum(
                            1 for k in range(d["non_tail"], layout["first_tail"]) if avail[k])
        finally:
            pool.close()
    report = dict(total=total)
    if as_json:
        with open(as_json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=1)
    print(json.dumps(report, indent=1))


def describe(pool, index):
    a = analyze(pool, index)
    d = a["descriptor"]
    out = dict(index=index, width=d["width"], height=d["height"], format=d["format"],
               mips=d["mips"], non_tail=d["non_tail"], mode=d["mode"],
               tile_mode=a["tile_mode"], tile=a["tile"], base=d["base"],
               meta=d["meta"], extra=d["extra"])
    if "layout" in a:
        out["allocation_size"] = a["layout"]["size"]
        out["allocation_span"] = a.get("allocation_span")
        out["span_matches"] = a.get("span_matches")
        out["first_tail_mip"] = a["layout"]["first_tail"]
        out["mips_detail"] = [
            dict(index=m["index"], blocks=[m["blocks_w"], m["blocks_h"]], offset=m["offset"],
                 region=m["region"], tail=m["in_tail"], tile_origin=[m["tile_x"], m["tile_y"]])
            for m in a["layout"]["mips"]]
    if "available" in a:
        out["available"] = a["available"]
        out["reader_limit"] = a["reader_limit"]
        out["reader_exported"] = a["reader_exported"]
        out["full_chain"] = a["full_chain"]
    if a.get("blocked"):
        out["blocked"] = a["blocked"]
    return out


def check(pool, index):
    a = analyze(pool, index)
    if "layout" not in a:
        print(json.dumps(dict(index=index, blocked=a.get("blocked")), indent=1))
        return 2
    layout = a["layout"]
    d = a["descriptor"]
    cache = {}
    try:
        tiled = pool.read_range(d["base"], layout["size"], cache)
    except BcpackRequired:
        print(json.dumps(dict(index=index, blocked="BCPack in allocation"), indent=1))
        return 2
    decoded, means = [], []
    for mip in layout["mips"]:
        data = detile(layout, mip, tiled[mip["offset"]:mip["offset"] + mip["region"]])
        px = decode_pixels(d["format"], data, mip["blocks_w"], mip["blocks_h"])
        if px is None:
            decoded.append(None)
            means.append(None)
        else:
            decoded.append(px)
            means.append(block_means(px, mip["blocks_w"], mip["blocks_h"]))
    if all(px is None for px in decoded):
        print(json.dumps(dict(index=index, blocked=(
            f"format {d['format']} is not one of the BC formats this tool decodes "
            "(71/72/77/78/80/83)")), indent=1))
        return 2
    errs = []
    for k in range(1, d["mips"]):
        if means[k - 1] is None or means[k] is None:
            errs.append(None)
            continue
        pm = layout["mips"][k - 1]
        cm = layout["mips"][k]
        errs.append(round(mean_abs_error(downsample2(means[k - 1], pm["blocks_w"], pm["blocks_h"]),
                                         means[k]), 2))
    control = None
    if layout["first_tail"] < d["mips"] and decoded[layout["first_tail"] - 1] is not None:
        tm = layout["mips"][layout["first_tail"]]
        bad = dict(tm, tile_x=0, tile_y=0)
        data = detile(layout, bad, tiled[:layout["tile"]])
        px = decode_pixels(d["format"], data, tm["blocks_w"], tm["blocks_h"])
        if px is not None:
            pm = layout["mips"][layout["first_tail"] - 1]
            control = round(mean_abs_error(
                downsample2(means[layout["first_tail"] - 1], pm["blocks_w"], pm["blocks_h"]),
                block_means(px, tm["blocks_w"], tm["blocks_h"])), 2)
    result = dict(index=index, width=d["width"], height=d["height"], format=d["format"],
                  mips=d["mips"], non_tail=d["non_tail"], first_tail=layout["first_tail"],
                  tile=layout["tile"], chain_error=errs, max_error=max(
                      (e for e in errs if e is not None), default=None),
                  tail_origin_control=control, full_chain=a.get("full_chain"))
    print(json.dumps(result, indent=1))
    return 0 if all(e is not None and e <= 6 for e in errs) else 1


def dds_bytes(width, height, fmt, mips, linear, payload):
    hdr = bytearray(128)
    hdr[0:4] = b"DDS "
    flags = 0x81007 if mips == 1 else 0xA1007
    struct.pack_into("<IIIIIII", hdr, 4, 124, flags, height, width, linear, 0, mips)
    struct.pack_into("<II4s", hdr, 76, 32, 4, b"DX10")
    struct.pack_into("<II", hdr, 108, 0x1000 if mips == 1 else 0x401008, 0)
    return bytes(hdr) + struct.pack("<IIIII", fmt, 3, 0, 1, 0) + payload


def export(pool, index, out_path, mips=None, force=False):
    a = analyze(pool, index)
    if "layout" not in a or "available" not in a:
        print(f"texture {index}: {a.get('blocked', 'layout unavailable')}", file=sys.stderr)
        return 2
    d = a["descriptor"]
    if d["format"] not in BC_FORMATS:
        print(f"texture {index}: format {d['format']} is uncompressed; this tool does not "
              "validate uncompressed tiling, so it will not write a DDS", file=sys.stderr)
        return 2
    if not a["full_chain"]:
        print(f"texture {index}: not every source mip is zlib/raw; refusing to mix BCPack",
              file=sys.stderr)
        return 2
    count = d["mips"] if mips is None else mips
    if count < 1 or count > d["mips"]:
        print(f"texture {index}: --mips must be 1..{d['mips']}", file=sys.stderr)
        return 2
    layout = a["layout"]
    cache = {}
    payload = bytearray()
    for mip in layout["mips"][:count]:
        tiled = pool.read_range(d["base"] + mip["offset"], mip["region"], cache)
        payload += detile(layout, mip, tiled)
    if 148 + len(payload) > MAX_OUTPUT:
        print("DDS exceeds the 256 MiB output limit", file=sys.stderr)
        return 2
    if os.path.exists(out_path) and not force:
        print(f"{out_path} exists; pass --force to replace it", file=sys.stderr)
        return 2
    top = layout["mips"][0]
    linear = (top["blocks_w"] * top["blocks_h"] * layout["element_bytes"]
              if d["format"] in BC_FORMATS else d["width"] * layout["element_bytes"])
    with open(out_path, "wb") as f:
        f.write(dds_bytes(d["width"], d["height"], d["format"], count, linear, payload))
    print(f"texture {index}: {d['width']}x{d['height']} fmt {d['format']} {count} mips "
          f"({148 + len(payload)} bytes) -> {out_path}")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    ps = sub.add_parser("scan", help="aggregate counters over pools")
    ps.add_argument("paths", nargs="+")
    ps.add_argument("--json")
    pl = sub.add_parser("layout", help="descriptor fields and mip offsets")
    pl.add_argument("pool")
    pl.add_argument("index", type=int)
    pc = sub.add_parser("check", help="decode a chain and report downsample error")
    pc.add_argument("pool")
    pc.add_argument("index", type=int)
    pe = sub.add_parser("export", help="write the full zlib/raw chain as DDS")
    pe.add_argument("pool")
    pe.add_argument("index", type=int)
    pe.add_argument("out")
    pe.add_argument("--mips", type=int)
    pe.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd == "scan":
        pools = []
        for p in a.paths:
            pools += find_pools(p)
        scan_pools(pools, a.json)
        return 0
    pool = Pool(a.pool)
    try:
        if a.cmd == "layout":
            print(json.dumps(describe(pool, a.index), indent=1))
            return 0
        if a.cmd == "check":
            return check(pool, a.index)
        return export(pool, a.index, a.out, a.mips, a.force)
    finally:
        pool.close()


if __name__ == "__main__":
    sys.exit(main())
