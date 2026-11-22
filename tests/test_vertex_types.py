"""Synthetic-data tests for tools/vertex_types.py.

    python -I tests/test_vertex_types.py

Every byte here is built by the test. No game file, value or name is used.
"""
import contextlib
import io
import json
import os
import random
import struct
import subprocess
import sys
import tempfile
import unittest
import zlib
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOL = os.path.join(ROOT, "tools", "vertex_types.py")
sys.path.insert(0, os.path.join(ROOT, "tools"))

import vertex_types as vt  # noqa: E402

GRID = 10  # GRID x GRID vertices, 2 * (GRID - 1)^2 triangles


def pack_position(x, y, z):
    """Inverse of the 0x42 slot-0 packing: x bits 8-31, y low byte bits 0-7 and high 16 bits 32-47, z bits 48-63."""
    return (y & 0xFF) | (x << 8) | ((y >> 8) << 32) | (z << 48)


def grid_triangles():
    tris = []
    for j in range(GRID - 1):
        for i in range(GRID - 1):
            a, b, c, d = j * GRID + i, j * GRID + i + 1, (j + 1) * GRID + i, (j + 1) * GRID + i + 1
            tris += [(a, b, c), (b, d, c)]
    return tris


def group_for(tris, n):
    """A meshlet group: each meshlet references all n vertices and takes up to 128 of the triangles."""
    assert n <= 128
    chunks = [tris[i:i + 128] for i in range(0, len(tris), 128)]
    recs, words, refs = bytearray(), bytearray(), bytearray()
    vbase = tbase = 0
    for m, part in enumerate(chunks):
        recs += struct.pack("<4I", (n - 1) | ((len(part) - 1) << 7), vbase | (tbase << 16), 0x7FF7FF, 0x7FF)
        for a, b, c in part:
            words += struct.pack("<I", a | (b << 7) | (c << 14) | (m << 21))
        refs += struct.pack(f"<{n}H", *range(n))
        vbase += n
        tbase += len(part)
    return struct.pack("<3I", 0x80000000 | len(chunks), tbase, vbase) + bytes(recs + words + refs)


def block(uv_offset=-5, colour=(10, 20, 30, 255)):
    """A 42 40 00 00 block on a flat grid: slots 0 (0x42 position), 1 (0x26), 2 (0x21 = snorm8 +z), 3 (0x28)."""
    n = GRID * GRID
    streams = [
        (0, 0x42, b"".join(struct.pack("<Q", pack_position(100 * (v % GRID), 100 * (v // GRID), 0)) for v in range(n))),
        (1, 0x26, bytes(colour) * n),
        (2, 0x21, bytes((0, 0, 127, 0)) * n),
        (3, 0x28, b"".join(struct.pack("<2h", 1000 * (v % GRID + uv_offset), 1000 * (v // GRID + uv_offset))
                           for v in range(n))),
    ]
    head = bytearray(vt.HEADER)
    off = vt.HEADER
    body = bytearray()
    for slot, tag, data in streams:
        struct.pack_into("<I", head, 4 * slot, tag | ((off // 2) << 8))
        body += data
        off += len(data)
    struct.pack_into("<I", head, 0x5C, 0x200)
    struct.pack_into("<3f", head, 0x60, 0.0, 0.0, 0.0)
    struct.pack_into("<I", head, 0x6C, 2)
    struct.pack_into("<3f", head, 0x70, 0.01, 0.01, 0.01)
    struct.pack_into("<f", head, 0x7C, 4.0)
    buf = bytearray(head + body)
    buf += b"\0" * (-len(buf) % 16)
    buf += group_for(grid_triangles(), n)
    buf += b"\0" * (-len(buf) % 16)
    return bytes(buf)


def measure(data):
    T = {}
    vt.measure_block(data, 0, T, random.Random(1))
    return T, vt.summarise(T)


class BlockTests(unittest.TestCase):
    def test_stride_rule(self):
        self.assertEqual([vt.stride(t) for t in (0x42, 0x21, 0x25, 0x26, 0x27, 0x28, 0x8C, 0x63)],
                         [8, 4, 4, 4, 4, 4, 16, 12])

    def test_slots_and_count(self):
        slots, order, flags = vt.parse_slots(block())
        self.assertEqual(order, [0, 1, 2, 3])
        self.assertEqual(slots[0], (0x42, 0x80))
        self.assertEqual(flags, 0x200)
        self.assertEqual(vt.vertex_count(slots, order), GRID * GRID)

    def test_rejects_other_magic(self):
        data = bytearray(block())
        data[0] = 0x63
        with self.assertRaises(vt.BlockError):
            vt.parse_slots(bytes(data))

    def test_signed_uv_is_continuous(self):
        T, out = measure(block(uv_offset=-5))
        uv = out["tags"]["slot3:28"]
        self.assertEqual(uv["edge_jumps_signed"], 0)
        self.assertGreater(uv["edge_jumps_unsigned"], 0)
        self.assertEqual(T["header_words"]["f32_7c"], {"4.0": 1})

    def test_positive_uv_has_no_jumps_either_way(self):
        _, out = measure(block(uv_offset=0))
        uv = out["tags"]["slot3:28"]
        self.assertEqual((uv["edge_jumps_signed"], uv["edge_jumps_unsigned"]), (0, 0))

    def test_mapping_spread_is_zero_on_an_even_grid(self):
        _, out = measure(block())
        self.assertEqual(out["mapping_spread"]["s16"]["median"], 0.0)

    def test_constant_colour_and_known_normal(self):
        _, out = measure(block())
        col = out["tags"]["slot1:26"]
        for k in range(4):
            self.assertEqual(col[f"byte{k}_constant_blocks"], 1)
        self.assertEqual(col["byte3_255"], GRID * GRID)
        nrm = out["tags"]["slot2:21"]
        self.assertEqual(nrm["facenormal_snorm8_xyz_median"], 1.0)
        self.assertEqual(nrm["unit_snorm8"], GRID * GRID)

    def test_positions_follow_the_surface(self):
        _, out = measure(block())
        self.assertLess(out["pos_edge_vs_random_median"], 0.5)
        self.assertEqual(out["pos_header_finite"], 1)


class ArchiveTests(unittest.TestCase):
    def test_end_to_end_on_a_fake_archive(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "fake.rpf")
            body = zlib.compress(block())
            with open(path, "wb") as fh:
                fh.write(b"RPF9" + struct.pack("<I", 16) + b"\0" * 8 + body + zlib.compress(b"\x63\x30\x00\x00" * 8))
            out_json = os.path.join(tmp, "out.json")
            r = subprocess.run([sys.executable, "-I", TOOL, path, "--json", out_json], capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr)
            with open(out_json, encoding="utf-8") as fh:
                out = json.load(fh)
            self.assertEqual((out["streams"], out["blocks"]), (1, 1))
            self.assertEqual(out["tags"]["slot3:28"]["edge_jumps_signed"], 0)

    def test_directory_scan_uses_stable_archive_order(self):
        with tempfile.TemporaryDirectory() as root:
            def walk(path):
                directories = ["z", "a"]
                yield path, directories, []
                for directory in directories:
                    yield os.path.join(path, directory), [], ["one.rpf"]

            with (mock.patch.object(vt.os, "walk", side_effect=walk),
                  mock.patch.object(vt, "iter_blocks", return_value=()) as blocks,
                  contextlib.redirect_stdout(io.StringIO()),
                  contextlib.redirect_stderr(io.StringIO())):
                self.assertEqual(vt.main([root]), 0)

            self.assertEqual(blocks.call_args_list, [
                mock.call(os.path.join(root, "a", "one.rpf"), 0),
                mock.call(os.path.join(root, "z", "one.rpf"), 0),
            ])

    def test_negative_limit_is_an_error(self):
        r = subprocess.run([sys.executable, "-I", TOOL, TOOL, "--limit", "-1"], capture_output=True, text=True)
        self.assertEqual(r.returncode, 2)


if __name__ == "__main__":
    unittest.main()
