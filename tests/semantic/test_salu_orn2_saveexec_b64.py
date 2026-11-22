from __future__ import annotations

import unittest
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

# RDNA2 Architectural Bitmasks & Constants
MASK32: int = 0xFFFF_FFFF
MASK64: int = 0xFFFF_FFFF_FFFF_FFFF
EXEC_ALL_ONES: int = 0xFFFF_FFFF_FFFF_FFFF
EXEC_ALL_ZEROS: int = 0x0000_0000_0000_0000
SCC_SET: int = 1
SCC_CLEARED: int = 0


def u64_to_u32_pair(val: int) -> Tuple[int, int]:
    """Split a 64-bit scalar value into (lo, hi) 32-bit dwords."""
    return val & MASK32, (val >> 32) & MASK32


def u32_pair_to_u64(lo: int, hi: int) -> int:
    """Combine two 32-bit dwords into a 64-bit scalar register value."""
    return ((hi & MASK32) << 32) | (lo & MASK32)


def oracle_s_orn2_saveexec_b64(src0: int, exec_in: int) -> Tuple[int, int, int]:
    """Reference mathematical oracle for RDNA2 s_orn2_saveexec_b64.

    Instruction Definition:
        SOP1 / Wave64 Execution Flow Control:
        S_ORN2_SAVEEXEC_B64 SDST, SSRC0

    Architectural Operation:
        SDST = EXEC;
        EXEC = SSRC0 | ~EXEC;
        SCC  = (EXEC != 0);

    Parameters:
        src0: 64-bit scalar source operand (SGPR pair, literal constant, or inline constant).
        exec_in: 64-bit current EXEC mask register value before execution.

    Returns:
        Tuple of (saved_exec, new_exec, scc):
            saved_exec: the pre-instruction EXEC mask destined for SDST.
            new_exec: the updated EXEC mask after applying OR with bitwise NOT of previous EXEC.
            scc: scalar condition code flag (1 if new_exec is non-zero, else 0).
    """
    saved_exec = exec_in & MASK64
    not_exec = (~exec_in) & MASK64
    new_exec = (src0 | not_exec) & MASK64
    scc = SCC_SET if new_exec != 0 else SCC_CLEARED
    return saved_exec, new_exec, scc


@dataclass
class ArchitecturalModel:
    """Architectural register model capturing the hardware execution state."""

    sgprs: Dict[int, int] = field(default_factory=dict)
    exec_mask: int = EXEC_ALL_ONES
    scc: int = SCC_CLEARED

    def get_sgpr32(self, idx: int) -> int:
        return self.sgprs.get(idx, 0) & MASK32

    def set_sgpr32(self, idx: int, val: int) -> None:
        self.sgprs[idx] = val & MASK32

    def get_sgpr64(self, idx: int) -> int:
        lo = self.get_sgpr32(idx)
        hi = self.get_sgpr32(idx + 1)
        return u32_pair_to_u64(lo, hi)

    def set_sgpr64(self, idx: int, val: int) -> None:
        lo, hi = u64_to_u32_pair(val)
        self.set_sgpr32(idx, lo)
        self.set_sgpr32(idx + 1, hi)

    def step_orn2_saveexec(self, dst_idx: Optional[int], src0_val: int) -> Tuple[int, int, int]:
        saved, new_exec, scc = oracle_s_orn2_saveexec_b64(src0_val, self.exec_mask)
        if dst_idx is not None:
            self.set_sgpr64(dst_idx, saved)
        self.exec_mask = new_exec
        self.scc = scc
        return saved, new_exec, scc


class TestReferenceOracleAndArchitecturalModel(unittest.TestCase):
    """Commit 1: Reference mathematical oracle and architectural model tests."""

    def test_truth_table_bit_by_bit(self) -> None:
        """Exhaustively verify 1-bit combinations of (src0, exec_in).

        Table for D = src0 | ~exec:
        src0 | exec_in | ~exec | src0 | ~exec | saved | new_exec | scc
          0  |    0    |   1   |      1       |   0   |    1     |  1
          0  |    1    |   0   |      0       |   1   |    0     |  0
          1  |    0    |   1   |      1       |   0   |    1     |  1
          1  |    1    |   0   |      1       |   1   |    1     |  1
        """
        cases = [
            # (src0_bit, exec_bit, expected_saved, expected_new_exec, expected_scc)
            (0, 0, 0, 1, 1),
            (0, 1, 1, 0, 0),
            (1, 0, 0, 1, 1),
            (1, 1, 1, 1, 1),
        ]
        for s0_bit, ex_bit, exp_saved, exp_new, exp_scc in cases:
            with self.subTest(s0_bit=s0_bit, ex_bit=ex_bit):
                s0 = MASK64 if s0_bit else 0
                ex = MASK64 if ex_bit else 0
                want_saved = MASK64 if exp_saved else 0
                want_new = MASK64 if exp_new else 0
                saved, new_exec, scc = oracle_s_orn2_saveexec_b64(s0, ex)
                self.assertEqual(saved, want_saved)
                self.assertEqual(new_exec, want_new)
                self.assertEqual(scc, exp_scc)

    def test_all_zeros_and_all_ones(self) -> None:
        """Verify boundary extremes: all zeros and all ones in both operands."""
        # src0 = 0, exec = 0 -> ~exec = all 1s -> new_exec = all 1s, scc = 1, saved = 0
        saved, new_exec, scc = oracle_s_orn2_saveexec_b64(EXEC_ALL_ZEROS, EXEC_ALL_ZEROS)
        self.assertEqual(saved, EXEC_ALL_ZEROS)
        self.assertEqual(new_exec, EXEC_ALL_ONES)
        self.assertEqual(scc, SCC_SET)

        # src0 = 0, exec = all 1s -> ~exec = 0 -> new_exec = 0, scc = 0, saved = all 1s
        saved, new_exec, scc = oracle_s_orn2_saveexec_b64(EXEC_ALL_ZEROS, EXEC_ALL_ONES)
        self.assertEqual(saved, EXEC_ALL_ONES)
        self.assertEqual(new_exec, EXEC_ALL_ZEROS)
        self.assertEqual(scc, SCC_CLEARED)

        # src0 = all 1s, exec = 0 -> new_exec = all 1s, scc = 1, saved = 0
        saved, new_exec, scc = oracle_s_orn2_saveexec_b64(EXEC_ALL_ONES, EXEC_ALL_ZEROS)
        self.assertEqual(saved, EXEC_ALL_ZEROS)
        self.assertEqual(new_exec, EXEC_ALL_ONES)
        self.assertEqual(scc, SCC_SET)

        # src0 = all 1s, exec = all 1s -> new_exec = all 1s, scc = 1, saved = all 1s
        saved, new_exec, scc = oracle_s_orn2_saveexec_b64(EXEC_ALL_ONES, EXEC_ALL_ONES)
        self.assertEqual(saved, EXEC_ALL_ONES)
        self.assertEqual(new_exec, EXEC_ALL_ONES)
        self.assertEqual(scc, SCC_SET)

    def test_alternating_bit_patterns(self) -> None:
        """Test alternating checkerboard patterns 0x5555... and 0xAAAA..."""
        p5 = 0x5555_5555_5555_5555
        pA = 0xAAAA_AAAA_AAAA_AAAA

        # src0 = p5, exec = p5: ~exec = pA, new_exec = p5 | pA = all 1s
        saved, new_exec, scc = oracle_s_orn2_saveexec_b64(p5, p5)
        self.assertEqual(saved, p5)
        self.assertEqual(new_exec, EXEC_ALL_ONES)
        self.assertEqual(scc, SCC_SET)

        # src0 = p5, exec = pA: ~exec = p5, new_exec = p5 | p5 = p5
        saved, new_exec, scc = oracle_s_orn2_saveexec_b64(p5, pA)
        self.assertEqual(saved, pA)
        self.assertEqual(new_exec, p5)
        self.assertEqual(scc, SCC_SET)

    def test_walking_ones_and_zeros(self) -> None:
        """Verify walking ones across all 64 bit lanes."""
        for i in range(64):
            bit = 1 << i
            # Single active lane in src0 with empty exec:
            saved, new_exec, scc = oracle_s_orn2_saveexec_b64(bit, EXEC_ALL_ZEROS)
            self.assertEqual(saved, EXEC_ALL_ZEROS)
            self.assertEqual(new_exec, EXEC_ALL_ONES)
            self.assertEqual(scc, SCC_SET)

            # Single active lane in exec with empty src0:
            saved, new_exec, scc = oracle_s_orn2_saveexec_b64(0, bit)
            self.assertEqual(saved, bit)
            # new_exec = 0 | ~bit = all bits except bit i
            self.assertEqual(new_exec, (~bit) & MASK64)
            self.assertEqual(scc, SCC_SET)

    def test_scc_zero_uniqueness(self) -> None:
        """Prove that SCC == 0 occurs if and only if src0 == 0 and exec_in == ALL_ONES."""
        # Sole zero condition
        saved, new_exec, scc = oracle_s_orn2_saveexec_b64(0, EXEC_ALL_ONES)
        self.assertEqual(new_exec, 0)
        self.assertEqual(scc, SCC_CLEARED)

        # Any single bit set in src0 prevents zero
        for i in range(64):
            _, new_exec, scc = oracle_s_orn2_saveexec_b64(1 << i, EXEC_ALL_ONES)
            self.assertNotEqual(new_exec, 0)
            self.assertEqual(scc, SCC_SET)

        # Any single bit cleared in exec_in prevents zero (since ~exec will have that bit set)
        for i in range(64):
            exec_val = EXEC_ALL_ONES ^ (1 << i)
            _, new_exec, scc = oracle_s_orn2_saveexec_b64(0, exec_val)
            self.assertNotEqual(new_exec, 0)
            self.assertEqual(scc, SCC_SET)

    def test_architectural_model_execution(self) -> None:
        """Verify architectural model state transitions."""
        model = ArchitecturalModel(exec_mask=0x1234_5678_9ABC_DEF0)
        saved, new_ex, scc = model.step_orn2_saveexec(dst_idx=4, src0_val=0xF000_0000_0000_000F)

        # Check saved EXEC placed into SGPR pair s[4:5]
        self.assertEqual(saved, 0x1234_5678_9ABC_DEF0)
        self.assertEqual(model.get_sgpr64(4), 0x1234_5678_9ABC_DEF0)

        # Check hardware state updated
        expected_new = (0xF000_0000_0000_000F | (~0x1234_5678_9ABC_DEF0 & MASK64)) & MASK64
        self.assertEqual(model.exec_mask, expected_new)
        self.assertEqual(model.scc, SCC_SET if expected_new != 0 else SCC_CLEARED)


if __name__ == "__main__":
    unittest.main()
