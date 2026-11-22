"""Check the PS5 input mode with generated headers and metadata."""

import json
from pathlib import Path
import struct
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest


MAKE = Path(__file__).resolve().parents[1] / "make_game.py"


def elf(machine=62, entry=0x400078):
    ident = b"\x7fELF\x02\x01\x01" + bytes(9)
    header = struct.pack("<16sHHIQQQIHHHHHH", ident, 0xFE10, machine, 1, entry, 64, 0, 0, 64, 56, 1, 0, 0, 0)
    load = struct.pack("<IIQQQQQQ", 1, 5, 0, 0x400000, 0, 121, 121, 4096)
    return header + load + b"\xc3"


class PS5App0Tests(unittest.TestCase):
    def setUp(self):
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.app0 = Path(temp.name) / "app0"
        (self.app0 / "sce_sys").mkdir(parents=True)
        (self.app0 / "sce_sys" / "param.json").write_text(
            json.dumps({"titleId": "PPSA99999"}), encoding="utf-8", newline="\n"
        )
        self.eboot = self.app0 / "eboot.bin"
        self.eboot.write_bytes(elf())

    def check(self, *extra):
        return self.run_make("--platform", "ps5", "--game", str(self.app0), "--check", *extra)

    def run_make(self, *arguments):
        return subprocess.run(
            [sys.executable, "-I", str(MAKE), *arguments], cwd=self.app0.parent,
            capture_output=True, text=True, timeout=30
        )

    def test_readable_elf_and_title_metadata(self):
        before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in self.app0.rglob("*") if p.is_file()}
        result = self.check()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("PS5 app0 check passed for PPSA99999", result.stdout)
        after = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in self.app0.rglob("*") if p.is_file()}
        self.assertEqual(before, after)
        self.assertEqual(list(self.app0.parent.iterdir()), [self.app0])

    def test_missing_eboot(self):
        self.eboot.unlink()
        result = self.check()
        self.assertEqual(result.returncode, 1)
        self.assertIn("missing eboot.bin", result.stderr)

    def test_self_explains_required_input(self):
        self.eboot.write_bytes(b"\x54\x14\xf5\xee" + bytes(60))
        result = self.check()
        self.assertEqual(result.returncode, 1)
        self.assertIn("readable ELF", result.stderr)

    def test_non_x86_and_truncated_headers_fail(self):
        for data, reason in ((elf(machine=183), "x86-64"), (elf()[:80], "table outside the file"),
                             (elf(entry=0), "no executable, file-backed entry point")):
            with self.subTest(reason=reason):
                self.eboot.write_bytes(data)
                result = self.check()
                self.assertEqual(result.returncode, 1)
                self.assertIn(reason, result.stderr)

    def test_ps4_title_id_fails(self):
        (self.app0 / "sce_sys" / "param.json").write_text(
            json.dumps({"titleId": "CUSA99999"}), encoding="utf-8", newline="\n"
        )
        result = self.check()
        self.assertEqual(result.returncode, 1)
        self.assertIn("missing a PS5 title ID", result.stderr)

    def test_xbox_stages_are_unavailable_in_ps5_mode(self):
        result = self.check("--stages", "runtime")
        self.assertEqual(result.returncode, 2)
        self.assertIn("only the read-only app0 check", result.stderr)

    def test_xbox_remains_the_default(self):
        options = ("--game", str(self.app0), "--check")
        default = self.run_make(*options)
        explicit = self.run_make("--platform", "xbox", *options)
        self.assertEqual((default.returncode, default.stderr), (explicit.returncode, explicit.stderr))
        self.assertNotEqual(default.returncode, 0)


if __name__ == "__main__":
    unittest.main()
