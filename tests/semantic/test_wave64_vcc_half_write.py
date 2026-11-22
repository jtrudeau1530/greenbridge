"""Wave64: a 32-bit write to vcc_lo or vcc_hi changes only that half of the VCC lane mask.

RDNA2 ISA: VCC is one 64-bit register; VCC_LO (SGPR 106) is VCC[31:0] and VCC_HI (SGPR 107) is VCC[63:32], each
a separately addressable 32-bit scalar operand. In wave64, v_cndmask_b32 reads VCC[lane] for lanes 0-63 and VCCZ
(s_cbranch_vccz) is set when all 64 bits are 0. So `s_mov_b32 vcc_lo, 0` must leave lanes 32-63 of VCC as they
were, and `s_mov_b32 vcc_hi, 0` must clear lanes 32-63 and keep lanes 0-31.

Each case builds VCC in a 64-thread compute shader from two push-constant words (bit i of s4 for lanes 0-31, of s5
for lanes 32-63), writes one half with a 0 / -1 constant (a mask value in every representation), then stores what
every lane sees: v_cndmask_b32 on VCC (its own bit) and the s_cbranch_vccz outcome (the wave's). Translated by
tools/rdna2spirv.py and run on all 64 lanes with tests/semantic/spirv_mini_interp.run_workgroup; no llvm-mc, GPU
or game data. The listing carries `; encoding:` bytes (zeros; only their length is read) so the branch target is
placed.
"""
import gc
import os
import struct
import sys
import tempfile
import unittest
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TOOLS_DIR = os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools")

sys.dont_write_bytecode = True
for p in (HERE, TOOLS_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

import rdna2spirv  # noqa: E402
import spirv_mini_interp as si  # noqa: E402

M32 = 0xFFFFFFFF
STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)
VSHARP = [0, 0, 0xFFFFFFFF, 0x31016FAC]  # s[0:3]: raw buffer (OOB_SELECT 3), any offset in range

# (instruction, size in bytes). VCC lane i = bit (i & 31) of (s4 if i < 32 else s5).
BUILD_VCC = [
    ("v_mov_b32_e32 v11, s5", 4),
    ("v_cmp_lt_u32_e64 vcc, v0, 32", 8),
    ("v_cndmask_b32_e64 v1, v11, s4, vcc", 8),
    ("v_and_b32_e32 v2, 31, v0", 4),
    ("v_lshrrev_b32_e32 v3, v2, v1", 4),
    ("v_and_b32_e32 v3, 1, v3", 4),
    ("v_cmp_ne_u32_e64 vcc, 0, v3", 8),
    ("s_and_b64 s[12:13], vcc, exec", 4),     # also what marks the program wave64 (a 64-bit op on VCC/EXEC)
]
# out[lane] = VCC[lane] ? 5 : 3; out[64 + lane] = 7 if s_cbranch_vccz was taken, else 9.
OBSERVE = [
    ("v_cndmask_b32_e64 v6, 3, 5, vcc", 8),
    ("v_lshlrev_b32_e32 v5, 2, v0", 4),
    ("buffer_store_dword v6, v5, s[0:3], 0 offen", 8),
    ("s_mov_b32 s20, 7", 4),
    ("s_cbranch_vccz 1", 4),
    ("s_mov_b32 s20, 9", 4),
    ("v_mov_b32_e32 v6, s20", 4),
    ("buffer_store_dword v6, v5, s[0:3], 0 offen offset:256", 8),
    ("s_endpgm", 4),
]


def translate_cs(name, lines):
    with tempfile.TemporaryDirectory() as tmp:
        os.mkdir(os.path.join(tmp, "disasm"))
        os.mkdir(os.path.join(tmp, "fssm"))
        src = os.path.join(tmp, "disasm", name + ".s")
        with open(src, "w", encoding="utf-8") as f:
            f.write(f"; {name} synthetic\n")
            for text, size in lines:
                f.write(f"\t{text} ; encoding: [{','.join(['0x00'] * size)}]\n")
        with open(os.path.join(tmp, name + ".dxbc"), "wb") as f:
            f.write(STUB_DXBC)
        with open(os.path.join(tmp, "fssm", name + ".fssm"), "wb") as f:
            # COMPUTE_PGM_RSRC2 0x8090: 8 user SGPRs (s[0:3] V#, s4..s7 push constants); 64x1x1 threads.
            f.write(bytes(0x138) + struct.pack("<4I", 0x8090, 64, 1, 1))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ResourceWarning)
            spv = rdna2spirv.translate(src)
            gc.collect()
        return spv


def run(spv, lo, hi):
    pc = VSHARP + [lo & M32, hi & M32, 0, 0] + [0] * 24
    mem = si.run_workgroup(spv, 64, 64, inputs_for=lambda i: {"%wg_id": (0, 0, 0), "%local_id": (i, 0, 0),
                                                            "%sg_lane": i}, push_constants=pc, memory={})
    return [mem[4 * l] for l in range(64)], [mem[256 + 4 * l] for l in range(64)]


def expect(lo, hi):
    vcc = (lo & M32) | (hi & M32) << 32
    return [5 if vcc >> l & 1 else 3 for l in range(64)], [7 if vcc == 0 else 9] * 64


# The half write under test, and the new (lo, hi) halves of VCC as a function of the old ones.
CASES = {
    "mov_vcc_lo_0": ("s_mov_b32 vcc_lo, 0", lambda lo, hi: (0, hi)),
    "mov_vcc_hi_0": ("s_mov_b32 vcc_hi, 0", lambda lo, hi: (lo, 0)),
    "mov_vcc_lo_all": ("s_mov_b32 vcc_lo, -1", lambda lo, hi: (M32, hi)),
    "mov_vcc_hi_all": ("s_mov_b32 vcc_hi, -1", lambda lo, hi: (lo, M32)),
    "and_vcc_lo_0": ("s_and_b32 vcc_lo, vcc_lo, 0", lambda lo, hi: (0, hi)),
    "or_vcc_hi_all": ("s_or_b32 vcc_hi, vcc_hi, -1", lambda lo, hi: (lo, M32)),
}
MASKS = [(0x0000FFFF, 0xFFFF0000), (0, 1), (1, 0), (0x80000006, 0x00010000), (0, 0), (M32, M32)]


class Wave64VccHalfWriteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spv = {k: translate_cs("cs_w64_vcc_" + k, BUILD_VCC + [(ln, 4)] + OBSERVE)
                   for k, (ln, _) in CASES.items()}

    def test_reference(self):
        self.assertEqual(expect(0, 1)[1][0], 9)
        self.assertEqual(expect(0, 0)[1][0], 7)
        self.assertEqual(expect(0, 1)[0][32], 5)

    def test_half_write_keeps_other_half(self):
        for name, (line, after) in CASES.items():
            for lo, hi in MASKS:
                with self.subTest(case=line, lo=hex(lo), hi=hex(hi)):
                    got = run(self.spv[name], lo, hi)
                    self.assertEqual(got, expect(*after(lo, hi)))


if __name__ == "__main__":
    unittest.main()
