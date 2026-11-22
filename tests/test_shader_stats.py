"""Unit tests for scripts/shader_stats.py (standard library only, no GPU).

    python -I tests/test_shader_stats.py
"""
import os
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(ROOT, "scripts", "shader_stats.py")

# ps_flat's listing as llvm-mc prints it: 8 instructions, highest VGPR v5.
PS_FLAT = """; ps_flat  code 0x0..0x0 (synthetic), gfx1013, 0 invalid words
\tv_interp_mov_f32_e32 v2, p0, attr1.x    ; encoding: [0x02,0x04,0x0a,0xc8]
\tv_add_nc_u32_e32 v3, 1, v2              ; encoding: [0x81,0x04,0x06,0x4a]
\tv_interp_p1_f32_e32 v4, v0, attr0.x     ; encoding: [0x00,0x00,0x10,0xc8]
\tv_interp_p2_f32_e32 v4, v1, attr0.x     ; encoding: [0x01,0x00,0x11,0xc8]
\tv_interp_p1_f32_e32 v5, v0, attr0.y     ; encoding: [0x00,0x01,0x14,0xc8]
\tv_interp_p2_f32_e32 v5, v1, attr0.y     ; encoding: [0x01,0x01,0x15,0xc8]
\texp mrt0 v2, v3, v4, v5 done vm         ; encoding: [0x0f,0x18,0x00,0xf8,0x02,0x03,0x04,0x05]
\ts_endpgm                                ; encoding: [0x00,0x00,0x81,0xbf]
"""


class ShaderStats(unittest.TestCase):
    def run_stats(self, rows, details=True):
        with tempfile.TemporaryDirectory() as tmp:
            disasm = os.path.join(tmp, "disasm")
            os.makedirs(disasm)
            with open(os.path.join(disasm, "ps_flat.s"), "w", encoding="utf-8") as f:
                f.write(PS_FLAT)
            csv_path = os.path.join(tmp, "stats.csv")
            with open(csv_path, "w", encoding="utf-8") as f:
                f.write("name,stage,statistic,value\n" + "".join(f"{r}\n" for r in rows))
            cmd = [sys.executable, "-I", SCRIPT, csv_path, disasm] + (["--details"] if details else [])
            return subprocess.run(cmd, capture_output=True, text=True)

    def test_growth_registers_and_scratch(self):
        r = self.run_stats(["ps_flat,ps,Instructions,16", "ps_flat,ps,VGPRs,8", "ps_flat,ps,SGPRs,20",
                            "ps_flat,ps,Scratch size,1024", "ps_flat,ps,Spilled VGPRs,2",
                            "ps_flat,ps,Subgroups per SIMD,16", "ps_flat,ps,Driver pipeline hash,abc"])
        self.assertEqual(r.returncode, 0, r.stderr)
        line = next(ln for ln in r.stdout.splitlines() if ln.startswith("ps_flat"))
        self.assertEqual(line.split(), ["ps_flat", "ps", "8", "16", "2.00", "6", "8", "20", "1024", "2", "16"])
        self.assertIn("ps: 1 shaders, code growth median 2.00x", r.stdout)
        self.assertIn("scratch in 1, spills in 1", r.stdout)

    def test_shaders_without_a_listing_are_left_out(self):
        r = self.run_stats(["ps_other,ps,Instructions,10"], details=False)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("no shader is in both", r.stderr)


if __name__ == "__main__":
    unittest.main()
