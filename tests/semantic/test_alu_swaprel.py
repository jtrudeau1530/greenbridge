"""Value tests for v_swaprel_b32 (GFX10 VOP1 opcode 104), run through spirv_mini_interp.

  tmp = VGPR[D + M0[25:16]];  VGPR[D + M0[25:16]] = VGPR[S0 + M0[9:0]];  VGPR[S0 + M0[9:0]] = tmp

The same two M0 fields as v_movrelsd_2_b32; the other M0 bits are ignored. Each case loads a distinct sentinel into
v1..v30 (0x40000000 + n, normal floats, so the export round trip can't change them), sets M0, runs one v_swaprel_b32
and exports four registers through exp mrt0. Expected words come from a plain-Python register-file model.
Standard library only; no GPU, no game files.
"""
import os
import struct
import sys
import tempfile
import unittest
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.dont_write_bytecode = True
sys.path.insert(0, os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools"))
sys.path.insert(0, HERE)

import rdna2spirv  # noqa: E402
import spirv_mini_interp as si  # noqa: E402

STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)


def sentinel(n):
    return 0x40000000 + n


def model(d, s0, m0):
    regs = {n: sentinel(n) for n in range(1, 31)}
    a, b = d + ((m0 >> 16) & 0x3FF), s0 + (m0 & 0x3FF)
    regs[a], regs[b] = regs[b], regs[a]
    return regs


def run(d, s0, m0, export, extra=()):
    with tempfile.TemporaryDirectory() as tmp:
        os.mkdir(os.path.join(tmp, "disasm"))
        with open(os.path.join(tmp, "ps_swaprel.dxbc"), "wb") as f:
            f.write(STUB_DXBC)
        lines = [f"v_mov_b32 v{n}, {sentinel(n):#x}" for n in range(1, 31)] + [f"s_mov_b32 m0, {m0:#x}", *extra,
                 f"v_swaprel_b32 v{d}, v{s0}", f"exp mrt0 {', '.join(f'v{r}' for r in export)} done vm", "s_endpgm"]
        src = os.path.join(tmp, "disasm", "ps_swaprel.s")
        with open(src, "w", encoding="utf-8") as f:
            f.write("; ps_swaprel\n" + "".join(f"\t{ln}\n" for ln in lines))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ResourceWarning)
            spv = rdna2spirv.translate(src)
    return list(si.run_fragment(spv)["%out_mrt0"])


class SwaprelTests(unittest.TestCase):
    def check(self, d, s0, m0):
        regs = model(d, s0, m0)
        a, b = d + ((m0 >> 16) & 0x3FF), s0 + (m0 & 0x3FF)
        export = [a, b, a + 1, b - 1 if b > 1 else b + 1]
        self.assertEqual(run(d, s0, m0, export), [regs[r] for r in export])

    def test_fields(self):
        for d, s0, m0 in ((1, 2, 0), (3, 10, 0x0005_0002), (20, 1, 0x0002_0007), (5, 5, 0x0001_0000)):
            with self.subTest(d=d, s0=s0, m0=hex(m0)):
                self.check(d, s0, m0)

    def test_other_m0_bits_are_ignored(self):
        # bits 15:10 and 31:26 set: same result as M0 = (3 << 16) | 4
        self.check(2, 6, 0xFC03_FC04)

    def test_both_sides_name_one_register(self):
        # D + M0[25:16] == S0 + M0[9:0]: the register is swapped with itself and keeps its value
        self.assertEqual(run(4, 6, 0x0003_0001, [7, 6, 8, 1]), [sentinel(7), sentinel(6), sentinel(8), sentinel(1)])

    def test_model(self):
        regs = model(1, 2, 0x0002_0005)         # D slot 1 + 2 = 3, S0 slot 2 + 5 = 7
        self.assertEqual((regs[3], regs[7], regs[4]), (sentinel(7), sentinel(3), sentinel(4)))

    def test_non_vgpr_operand_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            os.mkdir(os.path.join(tmp, "disasm"))
            with open(os.path.join(tmp, "ps_swaprel.dxbc"), "wb") as f:
                f.write(STUB_DXBC)
            src = os.path.join(tmp, "disasm", "ps_swaprel.s")
            with open(src, "w", encoding="utf-8") as f:
                f.write("; ps_swaprel\n\tv_swaprel_b32 v1, s2\n\texp mrt0 v1, v1, v1, v1 done vm\n\ts_endpgm\n")
            with self.assertRaises(rdna2spirv.Unsupported):
                rdna2spirv.translate(src)

if __name__ == "__main__":
    unittest.main()
