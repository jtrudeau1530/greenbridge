"""Tests for tools/ttpf_mips.py: tile modes, mip-tail layout, BC decode, metric, and a
synthetic RDS5/ETPF pool round trip. No game data is used.

    python -I tests/test_ttpf_mips.py
"""
import importlib.util
import json
import os
import struct
import sys
import tempfile
import unittest
import zlib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOL = os.path.join(ROOT, "tools", "ttpf_mips.py")
spec = importlib.util.spec_from_file_location("ttpf_mips", TOOL)
ttpf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ttpf)


def rgb565(r, g, b):
    return ((r * 31 // 255) << 11) | ((g * 63 // 255) << 5) | (b * 31 // 255)


def solid_bc1(color):
    c = rgb565(*color)
    return struct.pack("<HHI", c, c, 0)


def retile_to(layout, mip, linear, out):
    pat = (ttpf.PATTERNS_4K if layout["tile"] == 4096 else ttp64)[
        {1: 0, 2: 1, 4: 2, 8: 3, 16: 4}[layout["element_bytes"]]]
    eb = layout["element_bytes"]
    bw, bh = layout["block_w"], layout["block_h"]
    for y in range(mip["blocks_h"]):
        for x in range(mip["blocks_w"]):
            off = ((y // bh) * (mip["padded_w"] // bw) + x // bw) * layout["tile"]
            for bit, s in enumerate(pat):
                if s > 0:
                    off |= (((x + mip["tile_x"]) >> (s - 1)) & 1) << bit
                elif s < 0:
                    off |= (((y + mip["tile_y"]) >> (-s - 1)) & 1) << bit
            at = (y * mip["blocks_w"] + x) * eb
            out[mip["offset"] + off:mip["offset"] + off + eb] = linear[at:at + eb]


ttpf64 = {
    0: (1, 2, 3, -2, -1, -3, 4, -4, -5, 5, -6, 6, -7, 7, -8, 8),
    1: (0, 1, 2, 3, -1, -2, -3, 4, -4, 5, -5, 6, -6, 7, -7, 8),
    2: (0, 0, 1, 2, -1, -2, -3, 3, -4, 4, -5, 5, -6, 6, -7, 7),
    3: (0, 0, 0, 1, -1, 2, 3, -2, -3, 4, -4, 5, -5, 6, -6, 7),
    4: (0, 0, 0, 0, 1, -1, 2, -2, -3, 3, -4, 4, -5, 5, -6, 6),
}


def solid_chain(layout, fmt=71):
    """Build block-linear bytes for every mip; each block is one solid BC1 colour."""
    eb = layout["element_bytes"]
    out = {}
    # Smooth gradient at the top level; every lower mip is the 2x2 block mean.
    top = layout["mips"][0]
    colours = {}
    for y in range(top["blocks_h"]):
        for x in range(top["blocks_w"]):
            colours[(0, x, y)] = (min(255, x * 255 // max(1, top["blocks_w"] - 1)),
                                  min(255, y * 255 // max(1, top["blocks_h"] - 1)), 128)
    for m in range(1, len(layout["mips"])):
        pm = layout["mips"][m - 1]
        cm = layout["mips"][m]
        for y in range(cm["blocks_h"]):
            for x in range(cm["blocks_w"]):
                acc = [0, 0, 0]
                n = 0
                for dy in range(2):
                    for dx in range(2):
                        px, py = min(pm["blocks_w"] - 1, 2 * x + dx), min(pm["blocks_h"] - 1, 2 * y + dy)
                        q = colours[(m - 1, px, py)]
                        for k in range(3):
                            acc[k] += q[k]
                        n += 1
                colours[(m, x, y)] = tuple(a // n for a in acc)
    for mip in layout["mips"]:
        buf = bytearray(mip["blocks_w"] * mip["blocks_h"] * eb)
        for y in range(mip["blocks_h"]):
            for x in range(mip["blocks_w"]):
                at = (y * mip["blocks_w"] + x) * eb
                buf[at:at + eb] = solid_bc1(colours[(mip["index"], x, y)])[:eb].ljust(eb, b"\0")
        out[mip["index"]] = bytes(buf)
    return out


def build_pool(path, width=256, height=256, mips=7, non_tail=1, fmt=71, mode=4,
               meta=0x103, extra=0, packed=6):
    layout = ttpf.surface_layout(width, height, fmt, mips, ttpf.tile_bytes(mode))
    linear = solid_chain(layout, fmt)
    payload = bytearray(layout["size"])
    for mip in layout["mips"]:
        retile_to(layout, mip, linear[mip["index"]], payload)
    packed_payload = zlib.compress(bytes(payload), 9)
    pad = (-len(packed_payload)) % 256
    count = 3
    table_end = 24 + (count - 1) * 8
    etpf = ttpf.align(table_end, 256)
    metadata = 16 + 24 + 32
    padded = ttpf.align(metadata, 4096)
    source = etpf + padded
    chunk_end = source + len(packed_payload) + pad
    virt = padded
    vsize = virt + layout["size"]
    assert vsize % 4096 == 0
    entry = (source // 256) | (8 << 26) | ((virt // 4096) << 30) | (pad << 52)
    sentinel = (chunk_end // 256) | ((vsize // 4096) << 30)
    header = bytearray(24)
    header[0:4] = b"RDS5"
    struct.pack_into("<I", header, 4, count)
    struct.pack_into("<Q", header, 8, vsize | (22 << 56))
    struct.pack_into("<I", header, 16, (count + 31) // 32)
    struct.pack_into("<I", header, 20, 0)
    a = (width - 1) | ((height - 1) << 14) | ((packed & 0xF) << 28)
    b = non_tail | (fmt << 3) | (1 << 21) | (mode << 23)
    c = (virt // 4096) | (meta << 21)
    record = struct.pack("<IIII", a, b, c, extra) + struct.pack("<Q", 0x1234567890ABCDEF)
    blob = bytearray(chunk_end)
    blob[0:24] = header
    struct.pack_into("<QQ", blob, 24, entry, sentinel)
    blob[etpf:etpf + 4] = b"ETPF"
    struct.pack_into("<III", blob, etpf + 4, 634, 1, 24)
    blob[etpf + 16:etpf + 16 + 24] = record
    blob[source:source + len(packed_payload)] = packed_payload
    with open(path, "wb") as f:
        f.write(blob)
    return layout


class TileModeTest(unittest.TestCase):
    def test_mode_is_flag_plus_tile_mode(self):
        self.assertEqual([ttpf.tile_mode(m) for m in (0, 1, 2, 4, 5, 6, 7, 32, 34, 36, 38)],
                         [0, 0, 1, 2, 2, 3, 3, 0, 1, 2, 3])
        self.assertEqual([ttpf.tile_bytes(m) for m in (0, 1, 2, 4, 32, 34)],
                         [65536, 65536, 4096, 4096, 65536, 4096])


class LayoutTest(unittest.TestCase):
    def test_known_bc1_chain_sizes(self):
        lay = ttpf.surface_layout(256, 256, 71, 7, 4096)
        self.assertEqual(lay["first_tail"], 2)
        self.assertEqual(lay["size"], 45056)
        self.assertEqual([m["region"] for m in lay["mips"]],
                         [32768, 8192, 4096, 4096, 4096, 4096, 4096])
        self.assertEqual([m["offset"] for m in lay["mips"][:2]], [12288, 4096])
        # 16x16 pixels = 4x4 blocks at in-tile origin (16, 0)
        self.assertEqual((lay["mips"][2]["blocks_w"], lay["mips"][2]["blocks_h"]), (16, 16))
        self.assertEqual((lay["mips"][2]["tile_x"], lay["mips"][2]["tile_y"]), (16, 0))

    def test_validated_1024_chain(self):
        lay = ttpf.surface_layout(1024, 1024, 71, 9, 4096)
        self.assertEqual(lay["first_tail"], 4)
        self.assertEqual(lay["size"], 700416)
        self.assertEqual([m["region"] for m in lay["mips"][:4]],
                         [524288, 131072, 32768, 8192])
        self.assertEqual(lay["mips"][4]["tile_x"], 16)
        self.assertEqual(lay["mips"][4]["tile_y"], 0)

    def test_64k_tile_mode(self):
        lay = ttpf.surface_layout(1024, 1024, 98, 9, 65536)
        self.assertEqual(lay["size"], 1441792)
        self.assertEqual(lay["first_tail"], 3)
        self.assertEqual(lay["tile"], 65536)

    def test_bad_mip_count(self):
        with self.assertRaises(ttpf.LayoutError):
            ttpf.surface_layout(64, 64, 71, 20, 4096)


class BcDecodeTest(unittest.TestCase):
    def test_bc1_solid_block(self):
        block = solid_bc1((255, 0, 0))
        px = ttpf.decode_pixels(71, block, 1, 1)
        self.assertEqual(len(px), 16)
        self.assertGreater(px[0][0], 250)
        self.assertLess(px[0][1], 8)

    def test_bc1_four_colour_mode(self):
        # c0=white > c1=black: index 0 -> white, index 3 -> (c0 + 2*c1)/3
        white = ttpf.decode_pixels(71, struct.pack("<HHI", 0xFFFF, 0x0000, 0), 1, 1)
        self.assertGreater(white[0][0], 250)
        dark = ttpf.decode_pixels(71, struct.pack("<HHI", 0xFFFF, 0x0000, 0xFFFFFFFF), 1, 1)
        self.assertLess(dark[0][0], 100)

    def test_metric_rewards_the_true_downsample(self):
        parent = [(0.0, 0.0, 0.0), (255.0, 0.0, 0.0), (0.0, 255.0, 0.0), (0.0, 0.0, 255.0)]
        child = [(63.75, 63.75, 63.75)]
        self.assertLess(ttpf.mean_abs_error(ttpf.downsample2(parent, 2, 2), child), 1.0)
        wrong = [(0.0, 255.0, 0.0)]
        self.assertGreater(ttpf.mean_abs_error(ttpf.downsample2(parent, 2, 2), wrong), 50)


class SyntheticPoolTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = os.path.join(self.tmp.name, "synthetic.ttpf")
        self.layout = build_pool(self.path)
        self.pool = ttpf.Pool(self.path)
        self.addCleanup(self.pool.close)

    def test_descriptor_and_layout(self):
        d = self.pool.descriptor(0)
        self.assertEqual((d["width"], d["height"], d["format"], d["mips"], d["non_tail"]),
                         (256, 256, 71, 7, 1))
        self.assertEqual(d["mode"], 4)
        a = ttpf.analyze(self.pool, 0)
        self.assertTrue(a["span_matches"])
        self.assertTrue(a["full_chain"])
        self.assertEqual(a["layout"]["first_tail"], 2)

    def test_detile_round_trip(self):
        a = ttpf.analyze(self.pool, 0)
        layout = a["layout"]
        tiled = self.pool.read_range(a["descriptor"]["base"], layout["size"])
        linear = solid_chain(layout)
        for mip in layout["mips"]:
            data = ttpf.detile(layout, mip, tiled[mip["offset"]:mip["offset"] + mip["region"]])
            self.assertEqual(data, linear[mip["index"]])

    def test_chain_metric(self):
        import io
        from contextlib import redirect_stdout
        buf = io.StringIO()
        with redirect_stdout(buf):
            code = ttpf.check(self.pool, 0)
        result = json.loads(buf.getvalue())
        self.assertEqual(code, 0, result)
        self.assertLess(result["max_error"], 4.0)
        # a wrong in-tile origin for the first tail mip must be much worse
        self.assertGreater(result["tail_origin_control"], 5 * result["max_error"])

    def test_export_full_chain(self):
        out = os.path.join(self.tmp.name, "chain.dds")
        import io
        from contextlib import redirect_stdout
        with redirect_stdout(io.StringIO()):
            self.assertEqual(ttpf.export(self.pool, 0, out), 0)
        with open(out, "rb") as f:
            data = f.read()
        self.assertEqual(data[:4], b"DDS ")
        mips = struct.unpack_from("<I", data, 28)[0]
        self.assertEqual(mips, 7)
        expected = 148 + sum(m["blocks_w"] * m["blocks_h"] * 8 for m in self.layout["mips"])
        self.assertEqual(len(data), expected)
        # payload is exactly the concatenated block-linear chain
        linear = solid_chain(self.layout)
        self.assertEqual(data[148:], b"".join(linear[m["index"]] for m in self.layout["mips"]))

    def test_scan_reports_the_whitelist_gate(self):
        # The synthetic tuple is validated, so it is not gated: the partial export is
        # the descriptor non_tail limit, not the 319-byte profile table.
        out = os.path.join(self.tmp.name, "scan.json")
        import io
        from contextlib import redirect_stdout
        with redirect_stdout(io.StringIO()):
            ttpf.scan_pools([self.path], out)
        with open(out, encoding="utf-8") as f:
            result = json.load(f)["total"]
        self.assertEqual(result["entries"], 1)
        self.assertEqual(result["whitelist_gated"], 0)
        self.assertEqual(result["reader_partial"], 1)
        self.assertEqual(result["compact_partial_rows"], 1)
        self.assertEqual(result["compact_missing_mips"], 6)
        self.assertEqual(result["compact_decodable_mips"], 6)
        self.assertEqual(result["compact_full_rows"], 1)
        self.assertEqual(result["gated_accounted"], 0)
        self.assertEqual(result["gated_tuple_distinct"], 0)
        self.assertEqual((result["after_full"], result["after_partial"],
                          result["after_skip"]), (1, 0, 0))

    def test_scan_counts_an_unlisted_tuple_as_gated(self):
        path = os.path.join(self.tmp.name, "unlisted.ttpf")
        build_pool(path, meta=0x599)
        out = os.path.join(self.tmp.name, "scan2.json")
        import io
        from contextlib import redirect_stdout
        with redirect_stdout(io.StringIO()):
            ttpf.scan_pools([path], out)
        with open(out, encoding="utf-8") as f:
            total = json.load(f)["total"]
        self.assertEqual(total["whitelist_gated"], 1)
        self.assertEqual(total["gated_layout_ok"], 1)
        self.assertEqual(total["gated_full_chain"], 1)
        self.assertEqual(total["gated_full_mips"], 7)
        self.assertEqual(total["gated_partial"], 0)
        self.assertEqual(total["gated_no_contiguous"], 0)
        self.assertEqual(total["gated_no_layout"], 0)
        self.assertEqual(total["gated_unknown_format"], 0)
        self.assertEqual(total["gated_other_layout_error"], 0)
        self.assertEqual(total["gated_span_mismatch"], 0)
        self.assertEqual(total["gated_unreadable_range"], 0)
        self.assertEqual(total["gated_accounted"], total["whitelist_gated"])
        self.assertEqual(total["gated_tuple_distinct"], 1)
        self.assertEqual((total["after_full"], total["after_partial"],
                          total["after_skip"]), (1, 0, 0))

    def test_scan_reports_the_rejected_tuple(self):
        path = os.path.join(self.tmp.name, "tuple.ttpf")
        build_pool(path, meta=0x599)
        out = os.path.join(self.tmp.name, "scan3.json")
        import io
        from contextlib import redirect_stdout
        with redirect_stdout(io.StringIO()):
            ttpf.scan_pools([path], out, include_tuples=True)
        with open(out, encoding="utf-8") as f:
            total = json.load(f)["total"]
        self.assertEqual(total["gated_tuples"],
                         [{"tuple": [4, 256, 256, 7, 1, 8], "count": 1}])


if __name__ == "__main__":
    unittest.main()
