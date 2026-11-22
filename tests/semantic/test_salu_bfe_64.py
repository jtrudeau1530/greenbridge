"""Semantic interpreter tests for RDNA2 s_bfe_u64 and s_bfe_i64.

RDNA2 64-bit scalar bitfield extract instructions:
  - s_bfe_u64 D, S0, S1: Unsigned 64-bit bitfield extract across SGPR pairs.
  - s_bfe_i64 D, S0, S1: Signed 64-bit bitfield extract with sign extension.

Operand layout and semantics:
  - S0 is the 64-bit source operand (SGPR pair s[k:k+1], immediate, or constant).
  - S1 is the packed 32-bit control word:
      offset = S1[5:0]   (bits [5:0], 0..63).
      width  = S1[22:16] (bits [22:16], 0..127).
      Upper bits [15:6] and [31:23] of S1 are ignored.
  - Width limit / clamping:
      If width == 0: result is 0, SCC = 0.
      When offset + width > 64, the extracted field stops at bit 63:
      effective_width = min(width, 64 - offset).
  - s_bfe_u64 zero-extends the extracted field into the 64-bit destination pair D[63:0].
  - s_bfe_i64 sign-extends the extracted field from bit (effective_width - 1)
    across the entire 64-bit destination pair.
  - Condition code (SCC):
      SCC = 1 if D != 0 else 0 (both s_bfe_u64 and s_bfe_i64).
      SCC reflects the logical OR of both destination dwords (D.lo != 0 or D.hi != 0).
"""
from __future__ import annotations

import os
import random
import struct
import sys
import tempfile
import unittest
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TOOLS_DIR = os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools")

sys.dont_write_bytecode = True
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import rdna2spirv  # noqa: E402
import spirv_mini_interp as si  # noqa: E402

MASK32 = 0xFFFFFFFF
MASK64 = (1 << 64) - 1
M64 = MASK64
SIGN_BIT_64 = 1 << 63
STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)
OPS = ("s_bfe_u64", "s_bfe_i64")


def pack_bfe64_ctl(offset: int, width: int) -> int:
    """Pack 64-bit bitfield offset [5:0] and width [22:16] into an RDNA scalar control word."""
    return ((width & 0x7F) << 16) | (offset & 0x3F)


def unpack_bfe64_ctl(ctl: int) -> tuple[int, int]:
    """Unpack offset [5:0] and width [22:16] from an RDNA 64-bit scalar control word."""
    offset = ctl & 0x3F
    width = (ctl >> 16) & 0x7F
    return offset, width


def ref_s_bfe_u64(src: int, ctl: int) -> tuple[int, int]:
    """Pure-Python reference oracle for RDNA2 s_bfe_u64.

    Returns:
        (result, scc): 64-bit unsigned extracted field and SCC flag (1 if res != 0 else 0).
    """
    src &= MASK64
    offset, width = unpack_bfe64_ctl(ctl)
    if width == 0 or offset >= 64:
        return 0, 0
    eff_width = min(width, 64 - offset)
    mask = (1 << eff_width) - 1
    res = (src >> offset) & mask
    scc = 1 if res != 0 else 0
    return res & MASK64, scc


def ref_s_bfe_i64(src: int, ctl: int) -> tuple[int, int]:
    """Pure-Python reference oracle for RDNA2 s_bfe_i64.

    Returns:
        (result, scc): 64-bit sign-extended extracted field and SCC flag (1 if res != 0 else 0).
    """
    src &= MASK64
    offset, width = unpack_bfe64_ctl(ctl)
    if width == 0 or offset >= 64:
        return 0, 0
    eff_width = min(width, 64 - offset)
    mask = (1 << eff_width) - 1
    val = (src >> offset) & mask
    sign_bit = 1 << (eff_width - 1)
    if val & sign_bit:
        val -= (1 << eff_width)
    res = val & MASK64
    scc = 1 if res != 0 else 0
    return res, scc


# Legacy aliases matching existing suite
def ctl(off: int, width: int) -> int:
    return pack_bfe64_ctl(off, width)


def ref(op: str, x: int, ctl_val: int) -> int:
    if op == "s_bfe_u64":
        return ref_s_bfe_u64(x, ctl_val)[0]
    if op == "s_bfe_i64":
        return ref_s_bfe_i64(x, ctl_val)[0]
    raise ValueError(f"Unknown 64-bit bfe opcode: {op}")


class ScalarAlu64Interpreter:
    """Deterministic semantic interpreter for RDNA2 64-bit scalar ALU bitfield instructions.

    Models SGPR register file (106 dwords) and 1-bit Scalar Condition Code (SCC).
    """

    def __init__(self, num_sgprs: int = 106) -> None:
        self.num_sgprs = num_sgprs
        self.sgpr = [0] * num_sgprs
        self.scc = 0

    def reset(self) -> None:
        self.sgpr = [0] * self.num_sgprs
        self.scc = 0

    def get_sgpr(self, r: int) -> int:
        return self.sgpr[r] & MASK32

    def set_sgpr(self, r: int, val: int) -> None:
        self.sgpr[r] = val & MASK32

    def get_pair(self, r: int) -> int:
        lo = self.get_sgpr(r)
        hi = self.get_sgpr(r + 1)
        return (lo | (hi << 32)) & MASK64

    def set_pair(self, r: int, val: int) -> None:
        self.set_sgpr(r, val & MASK32)
        self.set_sgpr(r + 1, (val >> 32) & MASK32)

    def resolve_src64(self, opnd: int | dict | str) -> int:
        """Resolve 64-bit scalar operand (SGPR pair, immediate, or int)."""
        if isinstance(opnd, int):
            return opnd & MASK64
        if isinstance(opnd, dict):
            k = opnd.get("k")
            if k == "imm":
                return opnd["v"] & MASK64
            if k == "sgpr":
                return self.get_pair(opnd["r"])
            raise ValueError(f"Unknown 64-bit operand dict: {opnd}")
        if isinstance(opnd, str):
            if opnd.startswith("s[") and opnd.endswith("]"):
                parts = opnd[2:-1].split(":")
                return self.get_pair(int(parts[0]))
            if opnd.startswith("s"):
                return self.get_pair(int(opnd[1:]))
            return int(opnd, 0) & MASK64
        raise TypeError(f"Unsupported 64-bit operand type: {type(opnd)}")

    def resolve_src32(self, opnd: int | dict | str) -> int:
        """Resolve 32-bit scalar operand (control word, single SGPR, immediate)."""
        if isinstance(opnd, int):
            return opnd & MASK32
        if isinstance(opnd, dict):
            k = opnd.get("k")
            if k == "imm":
                return opnd["v"] & MASK32
            if k == "sgpr":
                return self.get_sgpr(opnd["r"])
            raise ValueError(f"Unknown 32-bit operand dict: {opnd}")
        if isinstance(opnd, str):
            if opnd.startswith("s"):
                return self.get_sgpr(int(opnd[1:]))
            return int(opnd, 0) & MASK32
        raise TypeError(f"Unsupported 32-bit operand type: {type(opnd)}")

    def resolve_dst_pair(self, dst: int | dict | str) -> int:
        """Resolve destination SGPR pair base index."""
        if isinstance(dst, int):
            return dst
        if isinstance(dst, dict) and dst.get("k") == "sgpr":
            return dst["r"]
        if isinstance(dst, str):
            if dst.startswith("s[") and dst.endswith("]"):
                parts = dst[2:-1].split(":")
                return int(parts[0])
            if dst.startswith("s"):
                return int(dst[1:])
        raise ValueError(f"Unsupported destination pair: {dst}")

    def step(self, op: str, dst: int | dict | str, s0: int | dict | str, s1: int | dict | str) -> tuple[int, int]:
        """Execute a single 64-bit scalar ALU bitfield instruction."""
        # Read source values BEFORE writing destination to correctly handle aliased pairs
        v0 = self.resolve_src64(s0)
        v1 = self.resolve_src32(s1)
        d_idx = self.resolve_dst_pair(dst)

        if op == "s_bfe_u64":
            res, scc = ref_s_bfe_u64(v0, v1)
            self.set_pair(d_idx, res)
            self.scc = scc
            return res, scc
        if op == "s_bfe_i64":
            res, scc = ref_s_bfe_i64(v0, v1)
            self.set_pair(d_idx, res)
            self.scc = scc
            return res, scc
        if op == "s_mov_b32":
            self.set_sgpr(d_idx, v0 & MASK32)
            return v0 & MASK32, self.scc
        raise NotImplementedError(f"Instruction {op} not implemented in 64-bit interpreter")

    def run_sequence(self, instructions: list[tuple[str, int | dict | str, int | dict | str, int | dict | str]]) -> tuple[int, int]:
        """Run a sequence of instructions, returning the last (res, scc)."""
        last_res = 0
        last_scc = 0
        for op, dst, s0, s1 in instructions:
            last_res, last_scc = self.step(op, dst, s0, s1)
        return last_res, last_scc


class ReferenceOracleTests(unittest.TestCase):
    """Validation of pure-Python 64-bit bitfield reference models and oracle behavior."""

    def test_pack_unpack_ctl_64(self) -> None:
        ctl_val = pack_bfe64_ctl(42, 60)
        off, wid = unpack_bfe64_ctl(ctl_val)
        self.assertEqual(off, 42)
        self.assertEqual(wid, 60)
        self.assertEqual(ctl_val, (60 << 16) | 42)

        # Upper bits [15:6] and [31:23] are ignored on unpack
        dirty = ctl_val | 0xFF80FFC0
        off_dirty, wid_dirty = unpack_bfe64_ctl(dirty)
        self.assertEqual(off_dirty, 42)
        self.assertEqual(wid_dirty, 60)

    def test_ref_s_bfe_u64_basic_oracle(self) -> None:
        # Extract 16 bits from offset 16 (within lower dword)
        ctl_val = pack_bfe64_ctl(16, 16)
        res, scc = ref_s_bfe_u64(0xDEADBEEFCAFE0000, ctl_val)
        self.assertEqual(res, 0xCAFE)
        self.assertEqual(scc, 1)

        # Extract 16 bits from offset 48 (within upper dword)
        ctl_hi = pack_bfe64_ctl(48, 16)
        res_hi, scc_hi = ref_s_bfe_u64(0xDEADBEEFCAFE0000, ctl_hi)
        self.assertEqual(res_hi, 0xDEAD)
        self.assertEqual(scc_hi, 1)

    def test_ref_s_bfe_u64_zero_width_oracle(self) -> None:
        # Zero width must yield 0 and SCC=0 regardless of input or offset
        for off in (0, 16, 31, 32, 48, 63):
            ctl_val = pack_bfe64_ctl(off, 0)
            res, scc = ref_s_bfe_u64(MASK64, ctl_val)
            self.assertEqual(res, 0)
            self.assertEqual(scc, 0)

    def test_ref_s_bfe_u64_full_64bit_oracle(self) -> None:
        ctl_val = pack_bfe64_ctl(0, 64)
        val = 0x0123456789ABCDEF
        res, scc = ref_s_bfe_u64(val, ctl_val)
        self.assertEqual(res, val)
        self.assertEqual(scc, 1)

    def test_ref_s_bfe_i64_positive_oracle(self) -> None:
        # 4-bit field with value 7 (msb 0) -> positive 7
        ctl_val = pack_bfe64_ctl(8, 4)
        res, scc = ref_s_bfe_i64(0x700, ctl_val)
        self.assertEqual(res, 7)
        self.assertEqual(scc, 1)

    def test_ref_s_bfe_i64_negative_oracle(self) -> None:
        # 4-bit field with value 8 (msb 1) -> negative -8 (0xFFFFFFFFFFFFFFF8)
        ctl_val = pack_bfe64_ctl(8, 4)
        res, scc = ref_s_bfe_i64(0x800, ctl_val)
        self.assertEqual(res, (-8) & MASK64)
        self.assertEqual(scc, 1)

        # 8-bit field with value 0x80 -> -128
        ctl_8 = pack_bfe64_ctl(16, 8)
        res_8, scc_8 = ref_s_bfe_i64(0x00800000, ctl_8)
        self.assertEqual(res_8, (-128) & MASK64)
        self.assertEqual(scc_8, 1)

    def test_ref_s_bfe_i64_single_bit_sign_extension(self) -> None:
        # 1-bit field with bit=1 -> sign extends to all 64 bits set (-1)
        ctl_val = pack_bfe64_ctl(5, 1)
        res_neg, scc_neg = ref_s_bfe_i64(1 << 5, ctl_val)
        self.assertEqual(res_neg, MASK64)
        self.assertEqual(scc_neg, 1)

        # 1-bit field with bit=0 -> 0, SCC=0
        res_zero, scc_zero = ref_s_bfe_i64(0, ctl_val)
        self.assertEqual(res_zero, 0)
        self.assertEqual(scc_zero, 0)

    def test_ref_s_bfe_clamping_past_bit_63_oracle(self) -> None:
        # Offset 60, width 8 runs to bit 68 -> clamped to 4 bits
        ctl_val = pack_bfe64_ctl(60, 8)
        src = 0xF0000000_00000000
        res_u, scc_u = ref_s_bfe_u64(src, ctl_val)
        self.assertEqual(res_u, 0xF)
        self.assertEqual(scc_u, 1)

        # s_bfe_i64 sign-extends 0xF (4-bit -1) across all 64 bits
        res_i, scc_i = ref_s_bfe_i64(src, ctl_val)
        self.assertEqual(res_i, MASK64)
        self.assertEqual(scc_i, 1)

    def test_interp_s_bfe_u64_basic(self) -> None:
        interp = ScalarAlu64Interpreter()
        interp.set_pair(2, 0x01234567_89ABCDEF)
        interp.set_sgpr(4, pack_bfe64_ctl(32, 32))
        res, scc = interp.step("s_bfe_u64", "s[0:1]", "s[2:3]", "s4")
        self.assertEqual(res, 0x01234567)
        self.assertEqual(interp.get_pair(0), 0x01234567)
        self.assertEqual(scc, 1)
        self.assertEqual(interp.scc, 1)

    def test_interp_s_bfe_i64_basic(self) -> None:
        interp = ScalarAlu64Interpreter()
        interp.set_pair(2, 0x80000000_00000000)
        interp.set_sgpr(4, pack_bfe64_ctl(60, 8))
        res, scc = interp.step("s_bfe_i64", "s[0:1]", "s[2:3]", "s4")
        self.assertEqual(res, (-8) & MASK64)
        self.assertEqual(interp.get_pair(0), (-8) & MASK64)
        self.assertEqual(scc, 1)
        self.assertEqual(interp.scc, 1)


VALUES = [0, M64, 1 << 63, 0x0123456789ABCDEF, 0xFEDCBA9876543210, 0x8000000080000000, 0x00000000FFFFFFFF]
CONTROLS = [ctl(0, 0), ctl(0, 1), ctl(0, 64), ctl(0, 127), ctl(1, 63), ctl(31, 2), ctl(32, 32), ctl(33, 31),
            ctl(60, 8), ctl(63, 1), ctl(63, 64), ctl(4, 40), 0xFFC0FFFF, 0x00100020]

_TMP_DIR = None
_SRC_PATH = None


def setUpModule():
    global _TMP_DIR, _SRC_PATH
    _TMP_DIR = tempfile.TemporaryDirectory()
    disasm_dir = os.path.join(_TMP_DIR.name, "disasm")
    os.mkdir(disasm_dir)
    _SRC_PATH = os.path.join(disasm_dir, "ps_salu_bfe64.s")
    with open(os.path.join(_TMP_DIR.name, "ps_salu_bfe64.dxbc"), "wb") as f:
        f.write(STUB_DXBC)


def tearDownModule():
    global _TMP_DIR
    if _TMP_DIR is not None:
        _TMP_DIR.cleanup()
        _TMP_DIR = None


def run_lines(lines) -> tuple:
    """Run the lines, return (s[0:1] as one 64-bit value, SCC)."""
    if _TMP_DIR is None:
        setUpModule()
    tail = ["s_cselect_b32 s20, 0x40000001, 0x40000000"]
    for i, (r, sh) in enumerate(((0, 0), (0, 16), (1, 0), (1, 16))):
        tail += [f"s_lshr_b32 s{10 + i}, s{r}, {sh}", f"s_and_b32 s{10 + i}, s{10 + i}, 0xffff",
                 f"s_or_b32 s{10 + i}, s{10 + i}, 0x40000000", f"v_mov_b32 v{i}, s{10 + i}"]
    tail.append("v_mov_b32 v4, s20")
    asm = ("; test_salu_bfe64\n" + "".join(f"\t{ln}\n" for ln in list(lines) + tail)
           + "\texp mrt0 v0, v1, v2, v3 vm\n\texp mrt1 v4, v4, v4, v4 done vm\n\ts_endpgm\n")
    with open(_SRC_PATH, "w", encoding="utf-8") as f:
        f.write(asm)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)
        spv = rdna2spirv.translate(_SRC_PATH)
    out = si.run_fragment(spv)
    w, scc = out["%out_mrt0"], out["%out_mrt1"][0]
    assert all(x >> 16 == 0x4000 for x in w) and scc >> 1 == 0x20000000, ([hex(x) for x in w], hex(scc))
    return sum((x & 0xFFFF) << (16 * i) for i, x in enumerate(w)), scc & 1


def run_op(op: str, x: int, c: int, scc_in: int = 0) -> tuple:
    return run_lines([f"s_mov_b32 s2, {x & 0xFFFFFFFF:#x}", f"s_mov_b32 s3, {x >> 32:#x}", f"s_mov_b32 s4, {c:#x}",
                      f"s_mov_b32 s5, {scc_in}", "s_cmp_eq_u32 s5, 1", f"{op} s[0:1], s[2:3], s4"])


class Bfe64Tests(unittest.TestCase):
    def test_reference(self):
        self.assertEqual(ref("s_bfe_u64", 0x0123456789ABCDEF, ctl(32, 32)), 0x01234567)
        self.assertEqual(ref("s_bfe_i64", 1 << 63, ctl(60, 8)), (-8) & M64)           # field past bit 63: >> 60
        self.assertEqual(ref("s_bfe_u64", M64, ctl(0, 127)), M64)                     # width cut to 64
        self.assertEqual(ref("s_bfe_i64", 0x80, ctl(4, 4)), (-8) & M64)
        self.assertEqual(ref("s_bfe_u64", M64, ctl(5, 0)), 0)

    def test_values_and_scc(self):
        for op in OPS:
            for x in VALUES:
                for c in CONTROLS:
                    want = ref(op, x, c)
                    with self.subTest(op=op, x=hex(x), ctl=hex(c)):
                        self.assertEqual(run_op(op, x, c, 1 - int(want != 0)), (want, int(want != 0)))

    def test_seeded(self):
        rng = random.Random(0xBFE64)
        for _ in range(30):
            op, x = rng.choice(OPS), rng.getrandbits(64)
            c = ctl(rng.randrange(64), rng.randrange(128)) if rng.random() < 0.8 else rng.getrandbits(32)
            with self.subTest(op=op, x=hex(x), ctl=hex(c)):
                self.assertEqual(run_op(op, x, c)[0], ref(op, x, c))

    def test_lane_mask_source_is_rejected(self):
        with self.assertRaises(rdna2spirv.Unsupported):
            run_lines(["s_mov_b32 s4, 0x100000", "s_bfe_u64 s[0:1], exec, s4"])


if __name__ == "__main__":
    unittest.main()
