"""The disassembler must stop with a reason instead of writing empty .s files and calling them clean.

Issue: the official Windows LLVM build ships no llvm-mc and has no AMDGPU target. Before this check, a
"llvm-mc" that existed but could not disassemble produced empty output, counted 0 invalid words, and every
shader was reported "clean" (and make_game.py translated nothing). The probe now catches it once, up front.

    python -I tests/test_shader_disasm.py
"""
import json
import os
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOL = os.path.join(ROOT, "tools", "shader_disasm.py")

# Two instructions: s_nop 0 then s_endpgm. The probe sends only the s_endpgm, so a fake tool can tell the
# one-instruction probe from the real shader by the input it receives.
CODE = b"\x00\x00\x80\xbf\x00\x00\x81\xbf"
PROBE = "0x00 0x00 0x81 0xbf"
ASM = "\ts_nop 0 ; encoding: [0x00,0x00,0x80,0xbf]\n\ts_endpgm ; encoding: [0x00,0x00,0x81,0xbf]\n"


def fssm() -> bytes:
    """A minimal FSSM blob: the code offset at 0xC0, then s_nop + s_endpgm at offset + 0x28."""
    blob = bytearray(0x200)
    blob[0:4] = b"FSSM"
    blob[0xC0:0xC4] = (0x100).to_bytes(4, "little")
    blob[0x128:0x12C] = CODE
    return bytes(blob)


class DisasmFailureTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = self.tmp.name
        self.fssm = os.path.join(self.dir, "one.fssm")
        with open(self.fssm, "wb") as f:
            f.write(fssm())

    def tearDown(self):
        self.tmp.cleanup()

    def run_tool(self, llvm_mc: str, out: str) -> subprocess.CompletedProcess:
        env = dict(os.environ, LLVM_MC=llvm_mc)
        return subprocess.run([sys.executable, "-I", TOOL, self.fssm, "--out", out, "--jobs", "1"],
                              capture_output=True, text=True, env=env)

    def fake(self, body: str) -> str:
        path = os.path.join(self.dir, "fake_mc.py")
        with open(path, "w", encoding="utf-8") as f:
            f.write(body)
        return f'"{sys.executable}" "{path}"'  # a command line, not just a path: LLVM_MC must accept both

    def test_missing_tool(self):
        out = os.path.join(self.dir, "out")
        p = self.run_tool("definitely-not-an-llvm-mc", out)
        self.assertEqual(p.returncode, 1)
        self.assertIn("could not be started", p.stderr)

    def test_no_amdgpu_target(self):
        mc = self.fake("import sys\n"
                       "sys.stderr.write('LLVM ERROR: No available targets are compatible with triple "
                       "\\\"amdgcn--amdpal\\\"\\n')\n"
                       "sys.exit(1)\n")
        p = self.run_tool(mc, os.path.join(self.dir, "out"))
        self.assertEqual(p.returncode, 1)
        self.assertIn("No available targets", p.stderr)
        self.assertIn("AMDGPU target", p.stderr)

    def test_empty_output_is_a_failure_and_writes_nothing(self):
        mc = self.fake("import sys\n"
                       "if sys.stdin.read().strip() == %r:\n"
                       "    print('\\ts_endpgm ; encoding: [0x00,0x00,0x81,0xbf]')\n" % PROBE)
        out = os.path.join(self.dir, "out")
        p = self.run_tool(mc, out)
        self.assertEqual(p.returncode, 1)
        self.assertIn("produced no disassembly", p.stderr)
        self.assertEqual(sorted(os.listdir(out)), [])

    def test_llvm_error_discards_partial_disassembly(self):
        mc = self.fake("import sys\n"
                       "sys.stdin.read()\n"
                       "if '-show-encoding' in sys.argv:\n"
                       "    print('s_nop 0')\n"
                       "    sys.stderr.write('error: disassembly stopped\\n')\n"
                       "    sys.exit(1)\n"
                       "print('s_endpgm')\n")
        out = os.path.join(self.dir, "out")
        p = self.run_tool(mc, out)
        self.assertEqual(p.returncode, 1, p.stdout)
        self.assertIn("'failed': 1", p.stdout)
        self.assertIn(self.fssm, p.stderr)
        self.assertIn("error: disassembly stopped", p.stderr)
        self.assertEqual(len(p.stderr.splitlines()), 1)
        self.assertEqual(sorted(os.listdir(out)), [])

    def test_no_code_is_an_error_when_no_listings_are_written(self):
        with open(self.fssm, "wb") as f:
            f.write(b"FSSM\x01\x01" + bytes(320))
        mc = self.fake("import sys\n"
                       "sys.stdin.read()\n"
                       "print('s_endpgm')\n")
        out = os.path.join(self.dir, "out")
        p = self.run_tool(mc, out)
        self.assertEqual(p.returncode, 1, p.stdout)
        self.assertIn("'no s_endpgm': 1", p.stdout)
        self.assertIn(self.fssm, p.stderr)
        self.assertEqual(len(p.stderr.splitlines()), 1)
        self.assertEqual(sorted(os.listdir(out)), [])

    def test_a_real_disassembly_is_written(self):
        mc = self.fake("import sys\n"
                       "probe = sys.stdin.read().strip() == %r\n"
                       "print('\\ts_endpgm ; encoding: [0x00,0x00,0x81,0xbf]' if probe else %r)\n"
                       % (PROBE, ASM))
        out = os.path.join(self.dir, "out")
        p = self.run_tool(mc, out)
        self.assertEqual(p.returncode, 0, p.stderr)
        written = os.listdir(out)
        self.assertEqual(written, ["one.s"])
        with open(os.path.join(out, "one.s"), encoding="utf-8") as f:
            body = f.read()
        self.assertIn("s_endpgm", body)
        self.assertIn("0 invalid words", body)


class DisasmInputTest(unittest.TestCase):
    """Short blobs, bad --jobs/--limit and an empty folder end with a reason, not a traceback."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = self.tmp.name
        self.src = os.path.join(self.dir, "fssm")
        os.makedirs(self.src)
        mc = os.path.join(self.dir, "fake_mc.py")
        with open(mc, "w", encoding="utf-8") as f:
            f.write("import sys\n"
                    "probe = sys.stdin.read().strip() == %r\n"
                    "print('\\ts_endpgm ; encoding: [0x00,0x00,0x81,0xbf]' if probe else %r)\n" % (PROBE, ASM))
        self.mc = f'"{sys.executable}" "{mc}"'

    def tearDown(self):
        self.tmp.cleanup()

    def run_tool(self, *args: str) -> subprocess.CompletedProcess:
        env = dict(os.environ, LLVM_MC=self.mc)
        return subprocess.run([sys.executable, "-I", TOOL, *args], capture_output=True, text=True, env=env)

    def write(self, name: str, data: bytes) -> None:
        with open(os.path.join(self.src, name), "wb") as f:
            f.write(data)

    def test_code_range_scans_a_blob_shorter_than_the_header(self):
        sys.path.insert(0, os.path.join(ROOT, "tools"))
        try:
            import shader_disasm
        finally:
            sys.path.pop(0)
        self.assertEqual(shader_disasm.code_range(b"FSSM" + CODE), (None, 12, "scan"))
        self.assertEqual(shader_disasm.code_range(b"FSSM"), (None, None, "scan"))
        self.assertEqual(shader_disasm.code_range(b""), (None, None, "scan"))

    def test_a_short_blob_does_not_stop_the_others(self):
        self.write("a_good.fssm", fssm())
        self.write("b_short.fssm", b"FSSM" + CODE)   # 12 bytes: no room for the code offset at 0xC0
        self.write("c_good.fssm", fssm())
        out = os.path.join(self.dir, "out")
        p = self.run_tool(self.src, "--out", out, "--jobs", "1")
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertNotIn("Traceback", p.stderr)
        self.assertEqual(sorted(os.listdir(out)), ["a_good.s", "c_good.s"])
        self.assertIn("'clean': 2", p.stdout)
        self.assertIn("'scan failed': 1", p.stdout)

    def test_directory_with_fssm_suffix_does_not_count_toward_limit(self):
        os.mkdir(os.path.join(self.src, "a_cache.fssm"))
        self.write("b_shader.fssm", fssm())
        out = os.path.join(self.dir, "out")

        result = self.run_tool(self.src, "--out", out, "--limit", "1")

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(os.listdir(out), ["b_shader.s"])

    def test_unpack_index_selects_current_blobs(self):
        self.write("current.fssm", fssm())
        self.write("obsolete.fssm", fssm())
        index = os.path.join(self.dir, "fssm_index.json")
        with open(index, "w", encoding="utf-8") as f:
            json.dump([{"name": "current"}], f)
        out = os.path.join(self.dir, "out")
        result = self.run_tool(self.src, "--out", out, "--unpack-index", index)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(os.listdir(out), ["current.s"])
        self.assertTrue(os.path.isfile(os.path.join(self.src, "obsolete.fssm")))

    def test_jobs_and_limit_must_be_at_least_one(self):
        self.write("one.fssm", fssm())
        out = os.path.join(self.dir, "out")
        for flag, value in (("--jobs", "0"), ("--jobs", "-2"), ("--limit", "0"), ("--limit", "-1")):
            with self.subTest(flag=flag, value=value):
                p = self.run_tool(self.src, "--out", out, flag, value)
                self.assertEqual(p.returncode, 2, p.stderr)
                self.assertIn("must be at least 1", p.stderr)
        self.assertFalse(os.path.exists(out))

    def test_a_folder_without_fssm_files_is_an_error(self):
        for args in ((self.src,), (self.src, "--out", os.path.join(self.dir, "out"))):
            with self.subTest(args=args):
                p = self.run_tool(*args)
                self.assertEqual(p.returncode, 1, p.stderr)
                self.assertIn("no .fssm files", p.stderr)
                self.assertNotIn("Traceback", p.stderr)

    def test_a_missing_source_is_an_error(self):
        p = self.run_tool(os.path.join(self.dir, "nope"), "--out", os.path.join(self.dir, "out"))
        self.assertEqual(p.returncode, 1, p.stderr)
        self.assertIn("is not a file or folder", p.stderr)
        self.assertNotIn("Traceback", p.stderr)

    def test_an_out_that_is_a_file_is_an_error(self):
        self.write("one.fssm", fssm())
        out = os.path.join(self.dir, "out")
        with open(out, "w", encoding="utf-8") as f:
            f.write("keep")
        p = self.run_tool(self.src, "--out", out)
        self.assertEqual(p.returncode, 1, p.stderr)
        self.assertIn("cannot create the output folder", p.stderr)
        self.assertNotIn("Traceback", p.stderr)
        with open(out, encoding="utf-8") as f:
            self.assertEqual(f.read(), "keep")


if __name__ == "__main__":
    unittest.main()
