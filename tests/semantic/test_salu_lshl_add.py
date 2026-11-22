"""Semantic interpreter tests for RDNA2 s_lshl1_add_u32 ... s_lshl4_add_u32.

RDNA2 scalar shift-and-add instructions:
  - s_lshl1_add_u32 D, S0, S1: D = ((S0 << 1) + S1) & 0xFFFFFFFF
  - s_lshl2_add_u32 D, S0, S1: D = ((S0 << 2) + S1) & 0xFFFFFFFF
  - s_lshl3_add_u32 D, S0, S1: D = ((S0 << 3) + S1) & 0xFFFFFFFF
  - s_lshl4_add_u32 D, S0, S1: D = ((S0 << 4) + S1) & 0xFFFFFFFF

Semantics and Condition Code (SCC):
  - D is the lower 32 bits of the scaled addition: ((S0 << N) + S1) mod 2^32.
  - SCC represents unsigned overflow of the sum taken in 64 bits (RDNA ISA pseudocode):
      tmp = (64'U(S0) << N) + 64'U(S1)
      SCC = 1 if tmp >= 2^32 else 0
    Bits shifted out of S0 (upper N bits of S0) count as carry just like an addition
    carry out of the 32-bit addition.
  - Pre-existing SCC state is ignored and unconditionally overwritten.

Each test case verifies mathematical semantics against pure-Python reference models
and executes synthetic shaders with tools/rdna2spirv.py in tests/semantic/spirv_mini_interp.py.
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
TWO_POW_32 = 1 << 32
STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)
VALUES = [
    0,
    1,
    0x0FFFFFFF,
    0x10000000,
    0x1FFFFFFF,
    0x3FFFFFFF,
    0x40000000,
    0x7FFFFFFF,
    0x80000000,
    0xFFFFFFFF,
    0x12345678,
    0xF0000000,
]


def ref_s_lshl_add_u32(n: int, s0: int, s1: int) -> tuple[int, int]:
    """Pure-Python reference oracle for RDNA2 s_lshlN_add_u32 (N in {1, 2, 3, 4}).

    Computes 64-bit unsigned temporary sum:
      tmp = (u64(s0) << n) + u64(s1)
      result = tmp & 0xFFFFFFFF
      scc = 1 if tmp >= 2^32 else 0

    Returns:
      (result, scc): 32-bit unsigned result and SCC overflow flag.
    """
    if n not in (1, 2, 3, 4):
        raise ValueError(f"Shift amount N must be in {1, 2, 3, 4}, got {n}")
    s0_u32 = s0 & MASK32
    s1_u32 = s1 & MASK32
    tmp = (s0_u32 << n) + s1_u32
    res = tmp & MASK32
    scc = 1 if (tmp >= TWO_POW_32) else 0

    # Cross-check against hardware decomposition:
    # shifted_out_bits | addition_carry == scc
    out_bits = s0_u32 >> (32 - n)
    sh_low = (s0_u32 << n) & MASK32
    carry = 1 if (sh_low + s1_u32 > MASK32) else 0
    assert (1 if (out_bits != 0 or carry != 0) else 0) == scc

    return res, scc


def ref_s_lshl1_add_u32(s0: int, s1: int) -> tuple[int, int]:
    """Reference oracle for s_lshl1_add_u32: D = ((S0 << 1) + S1) & 0xFFFFFFFF."""
    return ref_s_lshl_add_u32(1, s0, s1)


def ref_s_lshl2_add_u32(s0: int, s1: int) -> tuple[int, int]:
    """Reference oracle for s_lshl2_add_u32: D = ((S0 << 2) + S1) & 0xFFFFFFFF."""
    return ref_s_lshl_add_u32(2, s0, s1)


def ref_s_lshl3_add_u32(s0: int, s1: int) -> tuple[int, int]:
    """Reference oracle for s_lshl3_add_u32: D = ((S0 << 3) + S1) & 0xFFFFFFFF."""
    return ref_s_lshl_add_u32(3, s0, s1)


def ref_s_lshl4_add_u32(s0: int, s1: int) -> tuple[int, int]:
    """Reference oracle for s_lshl4_add_u32: D = ((S0 << 4) + S1) & 0xFFFFFFFF."""
    return ref_s_lshl_add_u32(4, s0, s1)


def ref(n: int, x: int, y: int) -> tuple[int, int]:
    """Backwards-compatible reference wrapper."""
    return ref_s_lshl_add_u32(n, x, y)


class ScalarAluInterpreter:
    """Minimal deterministic software emulator for scalar ALU state."""

    def __init__(self, num_sgprs: int = 106) -> None:
        self.sgprs = [0] * num_sgprs
        self.scc: int = 0

    def reset(self) -> None:
        for i in range(len(self.sgprs)):
            self.sgprs[i] = 0
        self.scc = 0

    def get_sgpr(self, reg: int | str) -> int:
        idx = int(reg[1:]) if isinstance(reg, str) and reg.startswith("s") else int(reg)
        return self.sgprs[idx] & MASK32

    def set_sgpr(self, reg: int | str, val: int) -> None:
        idx = int(reg[1:]) if isinstance(reg, str) and reg.startswith("s") else int(reg)
        self.sgprs[idx] = val & MASK32

    def parse_operand(self, op: str | int) -> int:
        if isinstance(op, int):
            return op & MASK32
        op = op.strip()
        if op.startswith("s"):
            return self.get_sgpr(op)
        if op.startswith("0x") or op.startswith("0X"):
            return int(op, 16) & MASK32
        return int(op) & MASK32

    def step(self, op: str, *args: str | int) -> tuple[int, int]:
        """Execute a single SALU instruction and update register state."""
        if op == "s_mov_b32":
            dst, src = args[0], args[1]
            val = self.parse_operand(src)
            self.set_sgpr(dst, val)
            return val, self.scc

        if op == "s_cmp_eq_u32":
            s0, s1 = self.parse_operand(args[0]), self.parse_operand(args[1])
            self.scc = 1 if s0 == s1 else 0
            return 0, self.scc

        if op == "s_cselect_b32":
            dst, s0, s1 = args[0], self.parse_operand(args[1]), self.parse_operand(args[2])
            val = s0 if self.scc != 0 else s1
            self.set_sgpr(dst, val)
            return val, self.scc

        if op == "s_addc_u32":
            dst, s0, s1 = args[0], self.parse_operand(args[1]), self.parse_operand(args[2])
            tot = s0 + s1 + self.scc
            res = tot & MASK32
            self.scc = 1 if tot > MASK32 else 0
            self.set_sgpr(dst, res)
            return res, self.scc

        if op in ("s_lshl1_add_u32", "s_lshl2_add_u32", "s_lshl3_add_u32", "s_lshl4_add_u32"):
            n = int(op[6])
            dst = args[0]
            s0 = self.parse_operand(args[1])
            s1 = self.parse_operand(args[2])
            res, scc = ref_s_lshl_add_u32(n, s0, s1)
            self.set_sgpr(dst, res)
            self.scc = scc
            return res, self.scc

        raise ValueError(f"Unsupported instruction in ScalarAluInterpreter: {op}")


_TMP_DIR = None
_SRC_PATH = None


def setUpModule():
    global _TMP_DIR, _SRC_PATH
    _TMP_DIR = tempfile.TemporaryDirectory()
    disasm_dir = os.path.join(_TMP_DIR.name, "disasm")
    os.mkdir(disasm_dir)
    _SRC_PATH = os.path.join(disasm_dir, "ps_salu_lshl_add.s")
    with open(os.path.join(_TMP_DIR.name, "ps_salu_lshl_add.dxbc"), "wb") as f:
        f.write(STUB_DXBC)


def tearDownModule():
    global _TMP_DIR
    if _TMP_DIR is not None:
        _TMP_DIR.cleanup()
        _TMP_DIR = None


def run_lines(lines) -> tuple:
    """Run the lines, export v0 and v1, return those two words."""
    if _TMP_DIR is None:
        setUpModule()
    asm = "; test_salu_lshl_add\n" + "".join(f"\t{ln}\n" for ln in lines) + "\texp mrt0 v0, v1, v0, v0 done vm\n\ts_endpgm\n"
    with open(_SRC_PATH, "w", encoding="utf-8") as f:
        f.write(asm)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)
        spv = rdna2spirv.translate(_SRC_PATH)
    return tuple(si.run_fragment(spv)["%out_mrt0"][:2])


def run_op(n: int, x: int, y: int, scc_in: int) -> tuple:
    """SCC = scc_in, then s0 = s_lshlN_add_u32(s2, s3); return (s0, SCC after)."""
    return run_lines([
        f"s_mov_b32 s2, {x:#x}",
        f"s_mov_b32 s3, {y:#x}",
        f"s_mov_b32 s4, {scc_in}",
        "s_cmp_eq_u32 s4, 1",
        f"s_lshl{n}_add_u32 s0, s2, s3",
        "s_cselect_b32 s1, 1, 0",
        "v_mov_b32 v0, s0",
        "v_mov_b32 v1, s1",
    ])


class ReferenceOracleTests(unittest.TestCase):
    """Validation of pure-Python reference oracles and mathematical invariants."""

    def test_reference_oracle_zero_identity(self):
        for n in (1, 2, 3, 4):
            with self.subTest(n=n):
                res, scc = ref_s_lshl_add_u32(n, 0, 0)
                self.assertEqual(res, 0)
                self.assertEqual(scc, 0)

    def test_reference_oracle_address_scaling(self):
        base_addr = 0x2000
        index = 25
        # Scale 1: 2 bytes (halfword) -> 2000 + 50 = 2050
        self.assertEqual(ref_s_lshl1_add_u32(index, base_addr), (base_addr + 50, 0))
        # Scale 2: 4 bytes (dword) -> 2000 + 100 = 2100
        self.assertEqual(ref_s_lshl2_add_u32(index, base_addr), (base_addr + 100, 0))
        # Scale 3: 8 bytes (qword) -> 2000 + 200 = 2200
        self.assertEqual(ref_s_lshl3_add_u32(index, base_addr), (base_addr + 200, 0))
        # Scale 4: 16 bytes (dqword) -> 2000 + 400 = 2400
        self.assertEqual(ref_s_lshl4_add_u32(index, base_addr), (base_addr + 400, 0))

    def test_reference_oracle_shifted_out_decomposition(self):
        # Bit shifted out of S0 alone sets SCC even with zero addend
        self.assertEqual(ref_s_lshl1_add_u32(0x80000000, 0), (0, 1))
        self.assertEqual(ref_s_lshl2_add_u32(0x40000000, 0), (0, 1))
        self.assertEqual(ref_s_lshl3_add_u32(0x20000000, 0), (0, 1))
        self.assertEqual(ref_s_lshl4_add_u32(0x10000000, 0), (0, 1))

    def test_reference_oracle_add_carry_alone(self):
        # Addition overflow alone sets SCC even without shifted-out bits
        self.assertEqual(ref_s_lshl1_add_u32(1, 0xFFFFFFFE), (0, 1))
        self.assertEqual(ref_s_lshl2_add_u32(1, 0xFFFFFFFC), (0, 1))
        self.assertEqual(ref_s_lshl3_add_u32(1, 0xFFFFFFF8), (0, 1))
        self.assertEqual(ref_s_lshl4_add_u32(1, 0xFFFFFFF0), (0, 1))

    def test_reference_oracle_both_shift_and_add_carry(self):
        # Both shifted out bits and addition carry present
        res1, scc1 = ref_s_lshl1_add_u32(0x80000001, 0xFFFFFFFF)
        self.assertEqual(res1, 1)
        self.assertEqual(scc1, 1)

        res4, scc4 = ref_s_lshl4_add_u32(0xF0000001, 0xFFFFFFFF)
        self.assertEqual(res4, 0x0000000F)
        self.assertEqual(scc4, 1)

    def test_reference_oracle_invalid_shift_raises(self):
        with self.assertRaises(ValueError):
            ref_s_lshl_add_u32(0, 10, 20)
        with self.assertRaises(ValueError):
            ref_s_lshl_add_u32(5, 10, 20)

    def test_scalar_interpreter_basic_pipeline(self):
        interp = ScalarAluInterpreter()
        interp.step("s_mov_b32", "s2", 0x10)
        interp.step("s_mov_b32", "s3", 0x20)
        res, scc = interp.step("s_lshl2_add_u32", "s0", "s2", "s3")
        self.assertEqual(res, 0x10 * 4 + 0x20)
        self.assertEqual(scc, 0)
        self.assertEqual(interp.get_sgpr("s0"), 0x60)
        self.assertEqual(interp.scc, 0)


class LshlAddTests(unittest.TestCase):
    def test_reference(self):
        self.assertEqual(ref(2, 3, 4), (16, 0))
        self.assertEqual(ref(1, 0x80000000, 0), (0, 1))                 # the shifted-out bit alone sets SCC
        self.assertEqual(ref(4, 0x0FFFFFFF, 0x10), (0, 1))              # carry out of the addition alone
        self.assertEqual(ref(3, 0x1FFFFFFF, 7), (0xFFFFFFFF, 0))        # fits exactly

    def test_values_and_scc(self):
        for n in (1, 2, 3, 4):
            for x in VALUES:
                for y in VALUES[::3]:
                    want = ref(n, x, y)
                    with self.subTest(n=n, x=hex(x), y=hex(y)):
                        self.assertEqual(run_op(n, x, y, 1 - want[1]), want)

    def test_seeded(self):
        rng = random.Random(0x5A)
        for _ in range(40):
            n, x, y = rng.randint(1, 4), rng.getrandbits(32), rng.getrandbits(32)
            with self.subTest(n=n, x=hex(x), y=hex(y)):
                self.assertEqual(run_op(n, x, y, rng.getrandbits(1)), ref(n, x, y))

    def test_scc_drives_s_addc(self):
        # A 64-bit index scaled by 4 into a base: the carry from s_lshl2_add_u32 feeds the high word.
        got = run_lines(["s_mov_b32 s2, 0x40000001", "s_mov_b32 s3, 0xfffffffc", "s_mov_b32 s5, 7",
                         "s_lshl2_add_u32 s0, s2, s3", "s_addc_u32 s1, s5, 0", "v_mov_b32 v0, s0", "v_mov_b32 v1, s1"])
        self.assertEqual(got, (0, 8))


if __name__ == "__main__":
    unittest.main()
