"""tex_headers.py report paths: never overwrite a carve manifest or a scanned .bin stream.

Run with ``python -I tests/test_tex_headers_report_paths.py``. Set ``LEONIDA_TEX_HEADERS`` to another
copy of the tool to use as a negative control (main's version writes the report over the .bin it just
scanned). All inputs are hand-built; no game data is used.
"""
import json
import os
import pathlib
import struct
import subprocess
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parent.parent
TOOL = pathlib.Path(os.environ.get("LEONIDA_TEX_HEADERS", str(ROOT / "tools" / "tex_headers.py")))


def make_carve(folder):
    """One synthetic BC1 texture: a CPU descriptor blob followed by its GPU data blob."""
    carve = pathlib.Path(folder) / "carve"
    arch = carve / "common"
    arch.mkdir(parents=True)
    cpu_off, gpu_off = 0x1000, 0x2000
    qwords = [0] * 32
    qwords[0] = 0
    qwords[1] = 0x123456789ABCDEF0
    k = 10
    qwords[k - 2] = 0xCAFEBABE00000001
    qwords[k - 1] = 0x980
    qwords[k] = 0x60000000 + 0x1000
    qwords[k + 4] = struct.unpack("<Q", struct.pack("<HHHBB", 512, 256, 1, 0x42, 71))[0]
    qwords[k + 5] = (9 << 16) | 1
    cpu = struct.pack(f"<{len(qwords)}Q", *qwords)
    gpu = b"\x00" * (0x1000 + 4096)
    cpu_path = arch / f"{cpu_off:012x}.bin"
    gpu_path = arch / f"{gpu_off:012x}.bin"
    cpu_path.write_bytes(cpu)
    gpu_path.write_bytes(gpu)
    index = {"files": [
        {"off": cpu_off, "head": cpu[:16].hex(), "size": len(cpu)},
        {"off": gpu_off, "head": gpu[:16].hex(), "size": len(gpu)},
    ]}
    (arch / "index.json").write_text(json.dumps(index), encoding="utf-8")
    return carve, cpu_path, gpu_path


class ReportPathTest(unittest.TestCase):
    def run_tool(self, *args):
        return subprocess.run([sys.executable, "-I", str(TOOL), *map(str, args)],
                              cwd=ROOT, capture_output=True, text=True, timeout=60)

    def test_out_cannot_overwrite_a_scanned_blob(self):
        for which in ("cpu", "gpu"):
            with self.subTest(blob=which), tempfile.TemporaryDirectory() as td:
                carve, cpu_path, gpu_path = make_carve(td)
                blob = cpu_path if which == "cpu" else gpu_path
                before = blob.read_bytes()
                result = self.run_tool(carve, "--out", blob)
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertIn("would overwrite an input blob", result.stderr)
                self.assertNotIn("Traceback", result.stderr)
                self.assertEqual(blob.read_bytes(), before)

    def test_out_cannot_be_a_hard_link_to_a_scanned_blob(self):
        with tempfile.TemporaryDirectory() as td:
            carve, cpu_path, _ = make_carve(td)
            alias = pathlib.Path(td) / "report.json"
            os.link(cpu_path, alias)
            before = cpu_path.read_bytes()
            self.assertTrue(os.path.samefile(cpu_path, alias))
            result = self.run_tool(carve, "--out", alias)
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertIn("would overwrite an input blob", result.stderr)
            self.assertEqual(cpu_path.read_bytes(), before)

    def test_out_cannot_be_a_directory(self):
        with tempfile.TemporaryDirectory() as td:
            carve, _, _ = make_carve(td)
            outdir = pathlib.Path(td) / "outdir"
            outdir.mkdir()
            result = self.run_tool(carve, "--out", outdir)
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertIn("is a directory", result.stderr)

    def test_report_inside_the_carve_folder_still_works(self):
        with tempfile.TemporaryDirectory() as td:
            carve, cpu_path, gpu_path = make_carve(td)
            cpu, gpu = cpu_path.read_bytes(), gpu_path.read_bytes()
            out = carve / "common" / "textures_index.json"
            result = self.run_tool(carve, "--out", out)
            self.assertEqual(result.returncode, 0, result.stderr)
            rows = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(len(rows), 1)
            self.assertEqual((rows[0]["w"], rows[0]["h"], rows[0]["fmt_name"]),
                             (512, 256, "BC1_UNORM"))
            self.assertEqual(cpu_path.read_bytes(), cpu)
            self.assertEqual(gpu_path.read_bytes(), gpu)


if __name__ == "__main__":
    unittest.main()
