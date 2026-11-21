"""The disassembler must stop with a reason instead of writing empty .s files and calling them clean.

Issue: the official Windows LLVM build ships no llvm-mc and has no AMDGPU target. Before this check, a
"llvm-mc" that existed but could not disassemble produced empty output, counted 0 invalid words, and every
shader was reported "clean" (and make_game.py translated nothing). The probe now catches it once, up front.

    python -I tests/test_shader_disasm.py
"""
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


if __name__ == "__main__":
    unittest.main()
