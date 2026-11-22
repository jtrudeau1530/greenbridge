"""Exercise PS5 tooling with generated metadata, headers, and audio geometry."""

import json
from pathlib import Path
import struct
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
import wave
import zlib


TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))

from ps5_app0 import executable_header
from ps5_audio import require_readable_awc, verify_wav
from ps5_content import begin_output, publish_output, read_compute_capture
from ps5_playgo import path_hash
from ps5_runtime import check_modules, module_directories


def elf(kind=0xFE10, entry=0x400078):
    ident = b"\x7fELF\x02\x01\x01" + bytes(9)
    header = struct.pack("<16sHHIQQQIHHHHHH", ident, kind, 62, 1, entry, 64, 0, 0, 64, 56, 1, 0, 0, 0)
    load = struct.pack("<IIQQQQQQ", 1, 5, 0, 0x400000, 0, 121, 121, 4096)
    return header + load + b"\xc3"


def write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8", newline="\n")


def write_playgo(reference, name):
    plgx = bytearray(256)
    plgx[:4] = b"plgx"
    struct.pack_into("<H", plgx, 4, 0x1000)
    struct.pack_into("<H", plgx, 10, 1)
    struct.pack_into("<H", plgx, 32, 1)
    attrs = bytearray(32)
    struct.pack_into("<I", attrs, 4, 1)
    sections = [attrs, bytes(4), b"Generated chunk\0", bytes(16), b"", b"", b""]
    for offset, section in zip(range(192, 248, 8), sections):
        struct.pack_into("<II", plgx, offset, len(plgx), len(section))
        plgx.extend(section)
    struct.pack_into("<I", plgx, 16, len(plgx))
    (reference / "playgo-chunk.dat").write_bytes(plgx)

    flt = bytearray(24)
    struct.pack_into("<I", flt, 0, 1)
    struct.pack_into("<II", flt, 8, 56, 8)
    flt.extend(struct.pack("<4sIIIQQQ", b"\x7fFLT", 0, 0, 1, 0x92CA8AAB26A24F51, 0x09BBB761A41BC44D, path_hash(name)))
    (reference / "playgo-hash-table.dat").write_bytes(flt)
    (reference / "playgo-ficm.dat").write_bytes(struct.pack("<4IH", 1, 0, 16, 2, 0))


def capture(spirv):
    identity = struct.pack("<7I", 64, 64, 32, 1, 2, 3, 4)
    name = b"Generated GPU"
    counts = struct.pack("<4I", 0, 0, 0, len(spirv) // 4)
    return b"APS5COMP\x01\0\0\0" + identity + bytes(64) + struct.pack("<I", len(name)) + name + counts + spirv


class PS5PortTests(unittest.TestCase):
    def setUp(self):
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.app0 = self.root / "app0"
        self.app0.mkdir()

    def run_cli(self, command, *args, status=0):
        result = subprocess.run(
            [sys.executable, "-I", str(TOOLS / "ps5_port.py"), command, str(self.app0), *map(str, args)],
            text=True, capture_output=True, cwd=self.root, timeout=30
        )
        self.assertEqual(result.returncode, status, result.stdout + result.stderr)
        return json.loads(result.stdout)

    def test_playgo_comparison_reports_coverage_and_a_missing_named_file(self):
        reference = self.root / "reference"
        reference.mkdir()
        (self.app0 / "sce_sys").mkdir()
        (self.app0 / "eboot.bin").write_bytes(elf())
        param = {"titleId": "PPSA99999", "contentId": "GENERATED", "contentVersion": "01.000.000"}
        write_json(self.app0 / "sce_sys/param.json", param)
        write_json(reference / "param.json", param)
        name = "generated.txt"
        asset = self.app0 / name
        asset.write_bytes(b"generated")
        write_playgo(reference, name)
        catalog = self.root / "catalog.json"
        matches = [{"candidate": name, "hash": f"{path_hash(name):016x}", "chunk": 0}]
        write_json(catalog, {"versions": {"01.000.000": {"matching_paths": matches}}})
        args = ("--reference-sce-sys", reference, "--path-catalog", catalog)
        found = self.run_cli("check", *args)["reference_comparison"]
        self.assertTrue(found["identity_matches"])
        self.assertEqual(found["matched_path_hashes"], 1)
        self.assertEqual(found["catalog_paths_present"], [{"path": name, "bytes": 9}])
        asset.unlink()
        missing = self.run_cli("check", *args, status=3)["reference_comparison"]
        self.assertEqual(missing["catalog_paths_missing_empty_or_ambiguous"], [name])

    def test_assets_distinguishes_recognized_headers_from_invalid_files(self):
        ihdr = b"IHDR" + struct.pack(">IIBBBBB", 2, 3, 8, 6, 0, 0, 0)
        png = b"\x89PNG\r\n\x1a\n" + struct.pack(">I", 13) + ihdr + struct.pack(">I", zlib.crc32(ihdr))
        (self.app0 / "generated.png").write_bytes(png)
        rpf9 = b"9FPR" + struct.pack("<3I", 64, 64, 0) + bytes(48)
        (self.app0 / "generated.rpf").write_bytes(rpf9)
        (self.app0 / "truncated.ttpf").write_bytes(b"RDS5")
        (self.app0 / "unknown.png").write_bytes(b"not a PNG")
        result = self.run_cli("assets")
        self.assertEqual(result["format_counts"], {"PNG": 1, "RPF9": 1})
        self.assertEqual(result["invalid_headers"][0]["path"], "truncated.ttpf")
        self.assertEqual(result["unconfirmed_candidates"][0]["path"], "unknown.png")

    def test_wav_geometry_and_protected_awc_input(self):
        wav = self.root / "generated.wav"
        with wave.open(str(wav), "wb") as output:
            output.setparams((1, 2, 48000, 0, "NONE", "not compressed"))
            output.writeframes(b"\0\0" * 5)
        verify_wav(wav, 5, 48000, 1)
        with self.assertRaisesRegex(ValueError, "playback geometry"):
            verify_wav(wav, 6, 48000, 1)
        awc = self.root / "generated.awc"
        awc.write_bytes(b"ADAT" + struct.pack("<I", 0xFF080001))
        with self.assertRaisesRegex(ValueError, "requires readable input"):
            require_readable_awc(awc)

    def test_compute_capture_extracts_spirv_and_rejects_truncated_records(self):
        raw = struct.pack("<5I", 0x07230203, 0x10000, 0, 1, 0)
        file = self.root / "generated.aps5cp"
        data = capture(raw)
        file.write_bytes(data)
        shader, metadata = read_compute_capture(file)
        self.assertEqual(shader, raw)
        self.assertEqual(metadata["gpu"], "Generated GPU")
        self.assertEqual(metadata["required_subgroup_size"], 32)
        file.write_bytes(data[:-1])
        with self.assertRaisesRegex(ValueError, "Truncated compute capture"):
            read_compute_capture(file)

    def test_output_is_published_only_to_a_new_external_directory(self):
        output = self.root / "output"
        with self.assertRaisesRegex(ValueError, "outside"):
            begin_output(self.app0, self.app0 / "nested", {})
        report = {}
        output, temp = begin_output(self.app0, output, report)
        self.assertFalse(output.exists())
        self.assertEqual(Path(report["incomplete_directory"]), temp)
        (temp / "staged.txt").write_bytes(b"generated")
        publish_output(output, temp, report)
        self.assertEqual((output / "staged.txt").read_bytes(), b"generated")
        self.assertNotIn("incomplete_directory", report)
        with self.assertRaises(FileExistsError):
            begin_output(self.app0, output, {})

    def test_module_check_requires_readable_shared_objects(self):
        path = self.app0 / "sce_module"
        path.mkdir()
        shared = path / "generated.prx"
        shared.write_bytes(elf(kind=0xFE18, entry=0))
        executable_header(shared, module=True)
        selected, custom = module_directories(self.app0, [])
        self.assertEqual(custom, [])
        self.assertEqual(check_modules(self.app0, [shared], selected), [Path("sce_module/generated.prx")])
        with self.assertRaisesRegex(ValueError, "Duplicate module"):
            module_directories(self.app0, ["sce_module"])
        shared.write_bytes(elf())
        with self.assertRaisesRegex(ValueError, "shared module"):
            check_modules(self.app0, [shared], selected)


if __name__ == "__main__":
    unittest.main()
