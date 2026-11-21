"""gamedata_maps.py must decode MHIU/PAMH headers and arrays from synthetic payloads.

    python -I tests/test_gamedata_maps.py

Every fixture here is synthetic. The MHIU fixture has a zero-filled prefix and a square grid; the
PAMH fixture has a valid row table and u16 grid. The archive test builds a carve-shaped directory
and checks the CLI, including a MHIU split across two 64 KiB streams.
"""
import json
import os
import struct
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOL = os.path.join(ROOT, "tools", "gamedata_maps.py")


def mhiu_payload(grid: bytes, prefix: int = 0, prefix_byte: int = 0) -> bytes:
    data = bytes([prefix_byte]) * prefix + grid
    head = bytearray(44)
    head[:4] = b"MHIU"
    struct.pack_into("<I", head, 0x04, 0x08000001)
    struct.pack_into("<I", head, 0x08, 2048)
    struct.pack_into("<6f", head, 0x0C, -10.0, -20.0, -30.0, 40.0, 50.0, 60.0)
    struct.pack_into("<I", head, 0x24, len(data))
    struct.pack_into("<I", head, 0x28, 0)
    return bytes(head) + data


def pamh_payload(width: int, height: int) -> bytes:
    rows = b"".join(struct.pack("<HHI", 0, width, i * width) for i in range(height))
    grid = struct.pack(f"<{width * height}H", *range(100, 100 + width * height))
    data = rows + grid
    head = bytearray(44)
    head[:4] = b"PAMH"
    struct.pack_into("<I", head, 0x04, 0x00000201)
    struct.pack_into("<I", head, 0x08, 0)
    struct.pack_into("<HH", head, 0x0C, width, height)
    struct.pack_into("<6f", head, 0x10, -10.0, -20.0, -30.0, 40.0, 50.0, 60.0)
    struct.pack_into("<I", head, 0x28, len(data))
    return bytes(head) + data


class GamedataMapsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = self.tmp.name
        self.carve = os.path.join(self.root, "carve")
        os.makedirs(self.carve)
        self.rows = []
        self.next_off = 0x1000

    def tearDown(self):
        self.tmp.cleanup()

    def add(self, payload: bytes, chunks=None):
        parts = chunks if chunks is not None else [payload]
        for part in parts:
            off = self.next_off
            self.next_off += 0x1000
            with open(os.path.join(self.carve, f"{off:012x}.bin"), "wb") as f:
                f.write(part)
            self.rows.append({"off": off, "stored": len(part), "size": len(part),
                              "head": part[:16].hex(), "pre": ""})

    def finish(self):
        with open(os.path.join(self.carve, "index.json"), "w", encoding="utf-8") as f:
            json.dump({"summary": {"archive": "synthetic", "files": len(self.rows)},
                       "files": self.rows}, f)

    def run_tool(self):
        out = os.path.join(self.root, "maps.json")
        p = subprocess.run([sys.executable, "-I", TOOL, self.carve, "--json", out],
                           cwd=self.root, capture_output=True, text=True,
                           encoding="utf-8", errors="replace")
        data = None
        if os.path.exists(out):
            with open(out, encoding="utf-8") as f:
                data = json.load(f)
        return p, data

    def test_decodes_mhiu_and_pamh(self):
        self.add(mhiu_payload(bytes(range(1, 17)), prefix=3))
        self.add(pamh_payload(3, 2))
        self.finish()

        p, data = self.run_tool()
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        mhiu, pamh = data["payloads"]
        self.assertEqual(mhiu["type"], "MHIU")
        self.assertEqual(mhiu["grid"]["width"], 4)
        self.assertEqual(mhiu["grid"]["height"], 4)
        self.assertEqual(mhiu["grid"]["cells"], 16)
        self.assertEqual(mhiu["grid"]["prefix_bytes"], 3)
        self.assertEqual(mhiu["header"]["bounds"]["min"], [-10.0, -20.0, -30.0])
        self.assertEqual(pamh["type"], "PAMH")
        self.assertEqual(pamh["grid"]["width"], 3)
        self.assertEqual(pamh["grid"]["height"], 2)
        self.assertEqual(pamh["grid"]["cells"], 6)
        self.assertEqual(pamh["grid"]["row_records"], 2)
        self.assertIn("MHIU", p.stdout)
        self.assertIn("PAMH", p.stdout)

    def test_groups_mhiu_continuation_stream(self):
        grid = b"\x07" * (300 * 300)
        payload = mhiu_payload(grid, prefix=7)
        self.add(payload, [payload[:65536], payload[65536:]])
        self.finish()

        p, data = self.run_tool()
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        mhiu = data["payloads"][0]
        self.assertEqual(mhiu["chunks"], 2)
        self.assertEqual(mhiu["grid"]["width"], 300)
        self.assertEqual(mhiu["grid"]["height"], 300)

    def test_rejects_nonzero_prefix(self):
        self.add(mhiu_payload(bytes(range(1, 17)), prefix=3, prefix_byte=1))
        self.finish()

        p, data = self.run_tool()
        self.assertEqual(p.returncode, 1)
        self.assertIn("square grid", p.stderr + p.stdout)
        self.assertIsNone(data)


if __name__ == "__main__":
    unittest.main()
