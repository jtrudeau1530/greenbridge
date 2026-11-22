"""Integration check for a built rpf9dump. Set RPF9DUMP_DLL to its Release DLL to run it."""

import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
import zlib

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_ttpf_mips import pack_etpf_record, write_rds5_pool


def write_pool(pool):
    # An unmapped catalog entry and two distinct, single-mip BC1 textures, one per zlib chunk.
    records = [pack_etpf_record(16, 16, 1, 0, 71, 1, 1, 2, base) for base in (0, 4096, 8192)]
    write_rds5_pool(pool, records, [b"a" * 4096, b"b" * 4096])


def export(pool, out):
    return subprocess.run(["dotnet", os.environ["RPF9DUMP_DLL"], "ttpf", pool, out],
                          capture_output=True, text=True)


def textures(folder):
    return json.loads((folder / "manifest.json").read_text(encoding="utf-8"))["Textures"]


@unittest.skipUnless(os.environ.get("RPF9DUMP_DLL"), "requires a built rpf9dump")
class TtpfExportResumeTest(unittest.TestCase):
    def test_manifest_is_published_only_after_all_extractable_outputs_exist(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pool = root / "pool.ttpf"
            write_pool(pool)

            clean = export(pool, root / "control")
            self.assertEqual(clean.returncode, 0, clean.stdout + clean.stderr)
            control = root / "control" / pool.stem
            rows = textures(control)
            self.assertEqual(len(rows), 3)
            self.assertIsNone(rows[0]["File"])
            self.assertTrue(rows[0]["Limitation"])
            first, second = (row["File"] for row in rows[1:])
            expected = [(control / name).read_bytes() for name in (first, second)]
            self.assertNotEqual(expected[0], expected[1])

            out = root / "retry"
            folder = out / pool.stem
            blocked = folder / second
            blocked.mkdir(parents=True)
            failed = export(pool, out)
            self.assertNotEqual(failed.returncode, 0, failed.stdout + failed.stderr)
            self.assertFalse((folder / "manifest.json").exists())
            sibling = folder / first
            self.assertEqual(sibling.read_bytes(), expected[0])
            os.utime(sibling, (1_000_000_000, 1_000_000_000))
            saved_mtime = sibling.stat().st_mtime_ns

            blocked.rmdir()
            retry = export(pool, out)
            self.assertEqual(retry.returncode, 0, retry.stdout + retry.stderr)
            self.assertEqual(textures(folder), rows)
            self.assertEqual([sibling.read_bytes(), blocked.read_bytes()], expected)
            self.assertEqual(sibling.stat().st_mtime_ns, saved_mtime)

    def test_undecodable_texture_is_a_limitation_and_siblings_still_publish(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pool = root / "pool.ttpf"
            write_pool(pool)
            # Swap the last chunk for a well-formed zlib stream that inflates to half of its catalogued
            # 4096 bytes. It keeps its slot; the size difference moves into the entry's pad field.
            blob = bytearray(pool.read_bytes())
            entry, sentinel = struct.unpack_from("<QQ", blob, 24 + 8)
            start, end = (entry & 0x3FFFFFF) * 256, (sentinel & 0x3FFFFFF) * 256
            short = zlib.compress(b"b" * 2048, 9)
            pad = end - start - len(short)
            self.assertTrue(0 <= pad < 256)
            blob[start:end] = short + bytes(pad)
            struct.pack_into("<Q", blob, 24 + 8, (entry & ((1 << 52) - 1)) | (pad << 52))
            pool.write_bytes(blob)

            out = root / "out"
            folder = out / pool.stem
            first = export(pool, out)
            self.assertEqual(first.returncode, 0, first.stdout + first.stderr)
            rows = textures(folder)
            self.assertEqual([row["File"] is None for row in rows], [True, False, True])
            self.assertIn("shorter", rows[2]["Limitation"])
            self.assertEqual(sorted(p.name for p in folder.glob("*.dds")), [rows[1]["File"]])
            sibling = folder / rows[1]["File"]
            os.utime(sibling, (1_000_000_000, 1_000_000_000))
            state = lambda: (sibling.read_bytes(), sibling.stat().st_mtime_ns, (folder / "manifest.json").read_bytes())
            saved = state()

            again = export(pool, out)
            self.assertEqual(again.returncode, 0, again.stdout + again.stderr)
            self.assertEqual(state(), saved)
            self.assertEqual(sorted(p.name for p in folder.glob("*.dds")), [rows[1]["File"]])


if __name__ == "__main__":
    unittest.main()
