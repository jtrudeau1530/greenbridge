"""Semantic reference oracle and verification for RDNA2 64-bit inverted scalar logic.

Covers:
  - s_andn2_b64: D = S0 & (~S1)
  - s_orn2_b64:  D = S0 | (~S1)
  - s_nand_b64:  D = ~(S0 & S1)
  - s_nor_b64:   D = ~(S0 | S1)
  - s_xnor_b64:  D = ~(S0 ^ S1)
  - Condition code SCC: SCC = 1 if D != 0 else 0

Tests standard library bitwise arithmetic, truth table exhaustion, boundary patterns,
four-corner SCC zero/nonzero matrix, and non-commutative operand asymmetry.
"""
from __future__ import annotations

import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TOOLS_DIR = os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools")

sys.dont_write_bytecode = True
if HERE not in sys.path:
    sys.path.insert(0, HERE)
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)

MASK32 = 0xFFFFFFFF
MASK64 = 0xFFFFFFFFFFFFFFFF


def ref_andn2_b64(s0: int, s1: int) -> int:
    """RDNA2 s_andn2_b64: D = S0 & (~S1)."""
    return (s0 & (~s1 & MASK64)) & MASK64


def ref_orn2_b64(s0: int, s1: int) -> int:
    """RDNA2 s_orn2_b64: D = S0 | (~S1)."""
    return (s0 | (~s1 & MASK64)) & MASK64


def ref_nand_b64(s0: int, s1: int) -> int:
    """RDNA2 s_nand_b64: D = ~(S0 & S1)."""
    return (~(s0 & s1)) & MASK64


def ref_nor_b64(s0: int, s1: int) -> int:
    """RDNA2 s_nor_b64: D = ~(S0 | S1)."""
    return (~(s0 | s1)) & MASK64


def ref_xnor_b64(s0: int, s1: int) -> int:
    """RDNA2 s_xnor_b64: D = ~(S0 ^ S1)."""
    return (~(s0 ^ s1)) & MASK64


def ref_scc(val64: int) -> int:
    """RDNA2 SCC condition flag: 1 if 64-bit result != 0, else 0."""
    return 1 if (val64 & MASK64) != 0 else 0


REF_OPS = {
    "s_andn2_b64": ref_andn2_b64,
    "s_orn2_b64": ref_orn2_b64,
    "s_nand_b64": ref_nand_b64,
    "s_nor_b64": ref_nor_b64,
    "s_xnor_b64": ref_xnor_b64,
}


class SaluInvertedLogicOracleTests(unittest.TestCase):
    """Reference mathematical oracle tests for 64-bit inverted scalar logic."""

    def test_truth_table_bit_by_bit(self):
        """Verify 1-bit truth table combinations across all 5 inverted operations."""
        truth_tables = {
            "s_andn2_b64": {(0, 0): 0, (0, 1): 0, (1, 0): 1, (1, 1): 0},
            "s_orn2_b64":  {(0, 0): 1, (0, 1): 0, (1, 0): 1, (1, 1): 1},
            "s_nand_b64":  {(0, 0): 1, (0, 1): 1, (1, 0): 1, (1, 1): 0},
            "s_nor_b64":   {(0, 0): 1, (0, 1): 0, (1, 0): 0, (1, 1): 0},
            "s_xnor_b64":  {(0, 0): 1, (0, 1): 0, (1, 0): 0, (1, 1): 1},
        }
        for op, table in truth_tables.items():
            fn = REF_OPS[op]
            for (b0, b1), expected_bit in table.items():
                with self.subTest(op=op, b0=b0, b1=b1):
                    s0 = MASK64 if b0 else 0
                    s1 = MASK64 if b1 else 0
                    want = MASK64 if expected_bit else 0
                    got = fn(s0, s1)
                    self.assertEqual(got, want)
                    self.assertEqual(ref_scc(got), 1 if expected_bit else 0)

    def test_symmetry_and_asymmetry(self):
        """Test commutativity for nand/nor/xnor and non-commutativity for andn2/orn2."""
        a = 0xF0F0F0F0_0F0F0F0F
        b = 0x33333333_CCCCCCCC
        for op in ("s_nand_b64", "s_nor_b64", "s_xnor_b64"):
            fn = REF_OPS[op]
            with self.subTest(op=op, prop="commutative"):
                self.assertEqual(fn(a, b), fn(b, a))

        for op in ("s_andn2_b64", "s_orn2_b64"):
            fn = REF_OPS[op]
            with self.subTest(op=op, prop="non-commutative"):
                self.assertNotEqual(fn(a, b), fn(b, a))

    def test_boundary_values(self):
        """Test edge cases with 0, MASK64, MSB, LSB, and alternating bit patterns."""
        boundaries = [
            0x00000000_00000000,
            0xFFFFFFFF_FFFFFFFF,
            0x80000000_00000000,
            0x00000000_00000001,
            0x55555555_55555555,
            0xAAAAAAAA_AAAAAAAA,
            0x00000000_FFFFFFFF,
            0xFFFFFFFF_00000000,
            0x12345678_9ABCDEF0,
            0xEDCBA987_6543210F,
        ]
        for op, fn in REF_OPS.items():
            for s0 in boundaries:
                for s1 in boundaries:
                    with self.subTest(op=op, s0=hex(s0), s1=hex(s1)):
                        res = fn(s0, s1)
                        self.assertGreaterEqual(res, 0)
                        self.assertLessEqual(res, MASK64)
                        scc = ref_scc(res)
                        self.assertIn(scc, (0, 1))
                        self.assertEqual(scc, 1 if res != 0 else 0)

    def test_scc_four_corners(self):
        """Verify SCC accurately distinguishes all four (lo, hi) zero/nonzero pairs."""
        self.assertEqual(ref_scc(0x00000000_00000000), 0)
        self.assertEqual(ref_scc(0x00000000_00000001), 1)
        self.assertEqual(ref_scc(0x00000000_80000000), 1)
        self.assertEqual(ref_scc(0x00000001_00000000), 1)
        self.assertEqual(ref_scc(0x80000000_00000000), 1)
        self.assertEqual(ref_scc(0x12345678_9ABCDEF0), 1)
        self.assertEqual(ref_scc(0xFFFFFFFF_FFFFFFFF), 1)

    def test_zero_result_edge_cases(self):
        """Verify inputs that explicitly evaluate to 0 clear SCC."""
        x = 0xA5A5A5A5_5A5A5A5A
        self.assertEqual(ref_andn2_b64(x, x), 0)
        self.assertEqual(ref_scc(ref_andn2_b64(x, x)), 0)

        self.assertEqual(ref_andn2_b64(0, x), 0)
        self.assertEqual(ref_scc(ref_andn2_b64(0, x)), 0)

        self.assertEqual(ref_orn2_b64(0, MASK64), 0)
        self.assertEqual(ref_scc(ref_orn2_b64(0, MASK64)), 0)

        self.assertEqual(ref_nand_b64(MASK64, MASK64), 0)
        self.assertEqual(ref_scc(ref_nand_b64(MASK64, MASK64)), 0)

        not_x = (~x) & MASK64
        self.assertEqual(ref_nor_b64(x, not_x), 0)
        self.assertEqual(ref_scc(ref_nor_b64(x, not_x)), 0)

        self.assertEqual(ref_nor_b64(MASK64, 0), 0)
        self.assertEqual(ref_scc(ref_nor_b64(MASK64, 0)), 0)

        self.assertEqual(ref_xnor_b64(x, not_x), 0)
        self.assertEqual(ref_scc(ref_xnor_b64(x, not_x)), 0)


class ScalarState:
    """Models 64-bit RDNA2 scalar execution state: SGPRs, EXEC, VCC, SCC, and NULL."""

    def __init__(self, num_sgprs: int = 106) -> None:
        self.sgprs = [0] * num_sgprs
        self.exec: int = MASK64
        self.vcc: int = 0
        self.scc: int = 0

    def read_sgpr64(self, idx: int) -> int:
        lo = self.sgprs[idx] & MASK32
        hi = self.sgprs[idx + 1] & MASK32
        return (hi << 32) | lo

    def write_sgpr64(self, idx: int, val: int) -> None:
        self.sgprs[idx] = val & MASK32
        self.sgprs[idx + 1] = (val >> 32) & MASK32

    def resolve_src64(self, src: int | str) -> int:
        if isinstance(src, int):
            return src & MASK64
        if src == "exec":
            return self.exec & MASK64
        if src == "vcc":
            return self.vcc & MASK64
        if src == "null":
            return 0
        if src.startswith("s[") and ":" in src:
            idx = int(src[2:].split(":")[0])
            return self.read_sgpr64(idx)
        raise ValueError(f"Unknown 64-bit source operand: {src}")

    def write_dst64(self, dst: str, val: int) -> None:
        val = val & MASK64
        if dst == "exec":
            self.exec = val
        elif dst == "vcc":
            self.vcc = val
        elif dst == "null":
            pass
        elif dst.startswith("s[") and ":" in dst:
            idx = int(dst[2:].split(":")[0])
            self.write_sgpr64(idx, val)
        else:
            raise ValueError(f"Unknown 64-bit destination operand: {dst}")

    def execute(self, op: str, dst: str, src0: int | str, src1: int | str) -> int:
        """Executes one 64-bit inverted scalar logic operation and updates SCC."""
        if op not in REF_OPS:
            raise ValueError(f"Unsupported op: {op}")
        s0_val = self.resolve_src64(src0)
        s1_val = self.resolve_src64(src1)
        res = REF_OPS[op](s0_val, s1_val)
        self.write_dst64(dst, res)
        self.scc = ref_scc(res)
        return res


class SaluInterpreterTests(unittest.TestCase):
    """Interpreter simulation tests covering SCC updates, operand aliasing, and wave64 exec masks."""

    def test_interpreter_all_ops_against_reference(self):
        state = ScalarState()
        v0 = 0x01234567_89ABCDEF
        v1 = 0xFEEDBEEF_CAFED00D
        state.write_sgpr64(0, v0)
        state.write_sgpr64(2, v1)

        for op, ref_fn in REF_OPS.items():
            with self.subTest(op=op):
                res = state.execute(op, "s[4:5]", "s[0:1]", "s[2:3]")
                expected = ref_fn(v0, v1)
                self.assertEqual(res, expected)
                self.assertEqual(state.read_sgpr64(4), expected)
                self.assertEqual(state.scc, ref_scc(expected))

    def test_destination_aliasing_exact(self):
        state = ScalarState()
        a = 0x11223344_55667788
        b = 0x88776655_44332211

        state.write_sgpr64(0, a)
        state.write_sgpr64(2, b)
        state.execute("s_andn2_b64", "s[0:1]", "s[0:1]", "s[2:3]")
        self.assertEqual(state.read_sgpr64(0), ref_andn2_b64(a, b))

        state.write_sgpr64(0, a)
        state.write_sgpr64(2, b)
        state.execute("s_andn2_b64", "s[2:3]", "s[0:1]", "s[2:3]")
        self.assertEqual(state.read_sgpr64(2), ref_andn2_b64(a, b))

        state.write_sgpr64(0, a)
        state.execute("s_andn2_b64", "s[0:1]", "s[0:1]", "s[0:1]")
        self.assertEqual(state.read_sgpr64(0), 0)
        self.assertEqual(state.scc, 0)

        state.write_sgpr64(0, a)
        state.execute("s_xnor_b64", "s[0:1]", "s[0:1]", "s[0:1]")
        self.assertEqual(state.read_sgpr64(0), MASK64)
        self.assertEqual(state.scc, 1)

    def test_destination_aliasing_overlapping(self):
        state = ScalarState()
        state.sgprs[0] = 0xAAAAAAAA
        state.sgprs[1] = 0xBBBBBBBB
        state.sgprs[2] = 0xCCCCCCCC
        state.sgprs[3] = 0xDDDDDDDD

        s0_val = state.read_sgpr64(0)
        s1_val = state.read_sgpr64(2)
        expected = ref_andn2_b64(s0_val, s1_val)

        state.execute("s_andn2_b64", "s[1:2]", "s[0:1]", "s[2:3]")

        self.assertEqual(state.sgprs[0], 0xAAAAAAAA)
        self.assertEqual(state.read_sgpr64(1), expected)
        self.assertEqual(state.sgprs[3], 0xDDDDDDDD)

    def test_inline_constants_and_literals(self):
        state = ScalarState()
        val = 0x01234567_89ABCDEF
        state.write_sgpr64(0, val)

        state.execute("s_andn2_b64", "s[2:3]", "s[0:1]", 0)
        self.assertEqual(state.read_sgpr64(2), val)
        self.assertEqual(state.scc, 1)

        state.execute("s_andn2_b64", "s[2:3]", "s[0:1]", MASK64)
        self.assertEqual(state.read_sgpr64(2), 0)
        self.assertEqual(state.scc, 0)

        state.execute("s_orn2_b64", "s[2:3]", "s[0:1]", 0)
        self.assertEqual(state.read_sgpr64(2), MASK64)
        self.assertEqual(state.scc, 1)

    def test_scc_updates_matrix(self):
        state = ScalarState()
        cases = [
            (0, 0, 0),
            (1, 0, 1),
            (0x80000000, 0, 1),
            (0, 1, 1),
            (0, 0x80000000, 1),
            (0x12345678, 0x9ABCDEF0, 1),
        ]
        for lo, hi, expected_scc in cases:
            with self.subTest(lo=hex(lo), hi=hex(hi)):
                val = (hi << 32) | lo
                state.write_sgpr64(0, val)
                state.execute("s_andn2_b64", "s[2:3]", "s[0:1]", 0)
                self.assertEqual(state.read_sgpr64(2), val)
                self.assertEqual(state.scc, expected_scc)

    def test_special_register_null(self):
        state = ScalarState()
        state.write_sgpr64(0, 0x12345678_9ABCDEF0)
        state.write_sgpr64(2, 0x12345678_9ABCDEF0)

        state.execute("s_andn2_b64", "null", "s[0:1]", "s[2:3]")
        self.assertEqual(state.scc, 0)
        self.assertEqual(state.read_sgpr64(0), 0x12345678_9ABCDEF0)
        self.assertEqual(state.read_sgpr64(2), 0x12345678_9ABCDEF0)

        state.execute("s_nor_b64", "null", "s[0:1]", 0)
        self.assertEqual(state.scc, 1)

    def test_wave64_branch_divergence_exec_pruning(self):
        state = ScalarState()
        state.exec = MASK64
        state.vcc = 0x55555555_55555555

        then_mask = state.vcc & state.exec
        state.execute("s_andn2_b64", "exec", "exec", "vcc")
        else_mask = state.exec

        self.assertEqual(then_mask & else_mask, 0)
        self.assertEqual(then_mask | else_mask, MASK64)
        self.assertEqual(else_mask, 0xAAAAAAAA_AAAAAAAA)
        self.assertEqual(state.scc, 1)

    def test_wave64_all_lanes_disabled_scc(self):
        state = ScalarState()
        state.exec = MASK64
        state.vcc = MASK64

        state.execute("s_andn2_b64", "exec", "exec", "vcc")
        self.assertEqual(state.exec, 0)
        self.assertEqual(state.scc, 0)

        for lane in range(64):
            with self.subTest(single_lane=lane):
                state.exec = 1 << lane
                state.vcc = 0
                state.execute("s_andn2_b64", "exec", "exec", "vcc")
                self.assertEqual(state.exec, 1 << lane)
                self.assertEqual(state.scc, 1)

    def test_wave64_vcc_mask_manipulation(self):
        state = ScalarState()
        state.vcc = 0x00000000_FFFFFFFF
        state.execute("s_nor_b64", "vcc", "vcc", 0)
        self.assertEqual(state.vcc, 0xFFFFFFFF_00000000)
        self.assertEqual(state.scc, 1)

        state.vcc = 0x0F0F0F0F_0F0F0F0F
        state.execute("s_xnor_b64", "vcc", "vcc", MASK64)
        self.assertEqual(state.vcc, 0x0F0F0F0F_0F0F0F0F)
        self.assertEqual(state.scc, 1)


if __name__ == "__main__":
    unittest.main()
