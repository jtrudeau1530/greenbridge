from __future__ import annotations

"""Semantic reference oracle and architectural model for RDNA2 s_andn2_saveexec_b64.

Instruction specification:
    s_andn2_saveexec_b64 SDST, SSRC0 (SOP1 format)
    
Architecture:
    - Target: AMD RDNA2 / GFX10 Scalar ALU (SALU)
    - SDST: 64-bit scalar destination register pair (e.g. s[0:1])
    - SSRC0: 64-bit scalar source operand (SGPR pair, literal constant, or inline constant)
    - Implicit operand / destination: EXEC (64-bit execution mask register)
    - Implicit destination flag: SCC (Scalar Condition Code, bit 0 of status register)

Operation:
    SDST = EXEC
    EXEC = SSRC0 & ~EXEC
    SCC = 1 if (EXEC != 0) else 0

Wavefront control-flow contract:
    - Saves current active lane execution mask to SDST.
    - Updates EXEC with lanes that are set in SSRC0 and were previously INACTIVE (~EXEC).
    - SCC reflects whether any lane in the new EXEC mask is active.
    - Used in compiler lowering of conditional branching (if-else inversion, loop exit, divergence).
"""

import sys
import unittest
from typing import Tuple

MASK32 = 0xFFFFFFFF
MASK64 = 0xFFFFFFFFFFFFFFFF


def u32(val: int) -> int:
    """Normalize integer to unsigned 32-bit dword."""
    return val & MASK32


def u64(val: int) -> int:
    """Normalize integer to unsigned 64-bit qword."""
    return val & MASK64


def split_dwords(val64: int) -> Tuple[int, int]:
    """Split 64-bit integer into (low_dword, high_dword)."""
    val = val64 & MASK64
    return val & MASK32, (val >> 32) & MASK32


def join_dwords(lo: int, hi: int) -> int:
    """Join (low_dword, high_dword) into 64-bit unsigned integer."""
    return ((hi & MASK32) << 32) | (lo & MASK32)


def ref_andn2_b64(s0: int, s1: int) -> int:
    """Bit-exact reference for 64-bit AND-NOT operation: D = S0 & (~S1)."""
    return (s0 & (~s1 & MASK64)) & MASK64


def ref_scc(val64: int) -> int:
    """RDNA2 SCC condition flag update: 1 if 64-bit result is non-zero, else 0."""
    return 1 if (val64 & MASK64) != 0 else 0


def ref_andn2_saveexec_b64(ssrc0: int, current_exec: int) -> Tuple[int, int, int]:
    """Bit-exact reference mathematical oracle for RDNA2 s_andn2_saveexec_b64.
    
    Returns:
        (saved_exec, new_exec, scc)
        - saved_exec: previous 64-bit EXEC written to SDST
        - new_exec: updated 64-bit EXEC written to EXEC register
        - scc: updated 1-bit SCC condition code (1 if new_exec != 0 else 0)
    """
    saved_exec = current_exec & MASK64
    new_exec = ref_andn2_b64(ssrc0, saved_exec)
    scc = ref_scc(new_exec)
    return saved_exec, new_exec, scc


class RegisterPair:
    """Architectural model of an aligned 64-bit scalar register pair s[N:N+1]."""

    def __init__(self, lo: int = 0, hi: int = 0) -> None:
        self.lo = lo & MASK32
        self.hi = hi & MASK32

    @classmethod
    def from_u64(cls, val: int) -> RegisterPair:
        lo, hi = split_dwords(val)
        return cls(lo, hi)

    @property
    def as_u64(self) -> int:
        return join_dwords(self.lo, self.hi)

    def set_u64(self, val: int) -> None:
        self.lo, self.hi = split_dwords(val)

    def __repr__(self) -> str:
        return f"RegisterPair(lo={self.lo:#010x}, hi={self.hi:#010x}, u64={self.as_u64:#018x})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, RegisterPair):
            return NotImplemented
        return self.lo == other.lo and self.hi == other.hi


class ExecRegister:
    """Architectural model of the 64-bit wave64 EXEC execution mask register."""

    def __init__(self, val: int = MASK64) -> None:
        self.val = val & MASK64

    @property
    def lo(self) -> int:
        return self.val & MASK32

    @property
    def hi(self) -> int:
        return (self.val >> 32) & MASK32

    def is_lane_active(self, lane: int) -> bool:
        if not (0 <= lane < 64):
            raise IndexError(f"Lane index {lane} out of bounds [0, 63]")
        return bool(self.val & (1 << lane))

    def active_lane_count(self) -> int:
        return bin(self.val).count("1")

    def __repr__(self) -> str:
        return f"ExecRegister(val={self.val:#018x}, active_lanes={self.active_lane_count()})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ExecRegister):
            return NotImplemented
        return self.val == other.val


class SccRegister:
    """Architectural model of the 1-bit Scalar Condition Code (SCC) status flag."""

    def __init__(self, value: int = 0) -> None:
        self.value = 1 if value else 0

    def update(self, result_u64: int) -> None:
        self.value = 1 if (result_u64 & MASK64) != 0 else 0

    def __repr__(self) -> str:
        return f"SccRegister({self.value})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SccRegister):
            return NotImplemented
        return self.value == other.value


class TestSaluAndn2SaveexecB64Oracle(unittest.TestCase):
    """Unit test suite for the reference mathematical oracle and architectural models."""

    def test_1bit_truth_table_exhaustive(self):
        """Verify the complete truth table of AND-NOT across all bit lanes.
        
        Truth table for S0 & ~S1:
        S0=0, S1=0 -> 0 & ~0 = 0 & 1 = 0
        S0=0, S1=1 -> 0 & ~1 = 0 & 0 = 0
        S0=1, S1=0 -> 1 & ~0 = 1 & 1 = 1
        S0=1, S1=1 -> 1 & ~1 = 1 & 0 = 0
        """
        table = {
            (0, 0): 0,
            (0, 1): 0,
            (1, 0): 1,
            (1, 1): 0,
        }
        for (b0, b1), expected_bit in table.items():
            for bit_pos in range(64):
                s0 = (1 << bit_pos) if b0 else 0
                exec_val = (1 << bit_pos) if b1 else 0
                saved, new_exec, scc = ref_andn2_saveexec_b64(s0, exec_val)
                expected_new_exec = (1 << bit_pos) if expected_bit else 0
                self.assertEqual(saved, exec_val)
                self.assertEqual(new_exec, expected_new_exec)
                self.assertEqual(scc, 1 if expected_bit else 0)

    def test_de_morgan_duality(self):
        """Verify De Morgan's equivalence: A & ~B == ~(~A | B)."""
        test_patterns = [
            (0x123456789ABCDEF0, 0x0FEDCBA987654321),
            (0xAAAAAAAAAAAAAAAA, 0x5555555555555555),
            (0xFFFFFFFF00000000, 0x00000000FFFFFFFF),
            (0x0000000000000000, 0xFFFFFFFFFFFFFFFF),
            (0xFFFFFFFFFFFFFFFF, 0x0000000000000000),
        ]
        for a, b in test_patterns:
            direct = ref_andn2_b64(a, b)
            de_morgan = (~((~a & MASK64) | b)) & MASK64
            self.assertEqual(direct, de_morgan)

    def test_non_commutativity(self):
        """Verify that S0 & ~S1 != S1 & ~S0 for asymmetric inputs."""
        a = 0xF0F0F0F0F0F0F0F0
        b = 0x3333333333333333
        res_ab = ref_andn2_b64(a, b)
        res_ba = ref_andn2_b64(b, a)
        self.assertNotEqual(res_ab, res_ba)
        # Specifically:
        # A & ~B = 0xF0... & 0xCC... = 0xC0C0C0C0C0C0C0C0
        # B & ~A = 0x33... & 0x0F... = 0x0303030303030303
        self.assertEqual(res_ab, 0xC0C0C0C0C0C0C0C0)
        self.assertEqual(res_ba, 0x0303030303030303)

    def test_self_annihilation_and_nullification(self):
        """Verify edge case mathematical identities."""
        patterns = [
            0x0000000000000000,
            0xFFFFFFFFFFFFFFFF,
            0x5555555555555555,
            0x123456789ABCDEF0,
        ]
        for x in patterns:
            # A & ~A == 0
            self.assertEqual(ref_andn2_b64(x, x), 0)
            self.assertEqual(ref_scc(ref_andn2_b64(x, x)), 0)
            # 0 & ~A == 0
            self.assertEqual(ref_andn2_b64(0, x), 0)
            # A & ~0 == A
            self.assertEqual(ref_andn2_b64(x, 0), x)
            # A & ~MASK64 == 0
            self.assertEqual(ref_andn2_b64(x, MASK64), 0)

    def test_saveexec_contract(self):
        """Verify that saved_exec rigorously captures old EXEC before modification."""
        test_execs = [
            0x0000000000000000,
            0x0000000000000001,
            0x8000000000000000,
            0xFFFFFFFF00000000,
            0x00000000FFFFFFFF,
            0x5555555555555555,
            0xFFFFFFFFFFFFFFFF,
        ]
        for old_exec in test_execs:
            for ssrc0 in [0, MASK64, 0xAAAAAAAAAAAAAAAA, old_exec]:
                saved, new_exec, scc = ref_andn2_saveexec_b64(ssrc0, old_exec)
                self.assertEqual(saved, old_exec)
                self.assertEqual(new_exec, ref_andn2_b64(ssrc0, old_exec))
                self.assertEqual(scc, 1 if new_exec != 0 else 0)

    def test_scc_four_quadrants(self):
        """Verify SCC accurately distinguishes lo-dword and hi-dword non-zero states."""
        # 1. both lo and hi zero -> SCC = 0
        self.assertEqual(ref_scc(0x0000000000000000), 0)
        # 2. lo non-zero, hi zero -> SCC = 1
        self.assertEqual(ref_scc(0x0000000000000001), 1)
        self.assertEqual(ref_scc(0x0000000080000000), 1)
        # 3. lo zero, hi non-zero -> SCC = 1
        self.assertEqual(ref_scc(0x0000000100000000), 1)
        self.assertEqual(ref_scc(0x8000000000000000), 1)
        # 4. both lo and hi non-zero -> SCC = 1
        self.assertEqual(ref_scc(0x123456789ABCDEF0), 1)
        self.assertEqual(ref_scc(0xFFFFFFFFFFFFFFFF), 1)

    def test_architectural_register_pair_and_exec(self):
        """Verify RegisterPair, ExecRegister, and SccRegister modeling integrity."""
        rp = RegisterPair.from_u64(0x123456789ABCDEF0)
        self.assertEqual(rp.lo, 0x9ABCDEF0)
        self.assertEqual(rp.hi, 0x12345678)
        self.assertEqual(rp.as_u64, 0x123456789ABCDEF0)

        rp.set_u64(0xAAAAAAAAAAAAAAAA)
        self.assertEqual(rp.lo, 0xAAAAAAAA)
        self.assertEqual(rp.hi, 0xAAAAAAAA)

        exec_reg = ExecRegister(0x0000000000000003)
        self.assertTrue(exec_reg.is_lane_active(0))
        self.assertTrue(exec_reg.is_lane_active(1))
        self.assertFalse(exec_reg.is_lane_active(2))
        self.assertEqual(exec_reg.active_lane_count(), 2)

        scc = SccRegister()
        self.assertEqual(scc.value, 0)
        scc.update(0x0000000100000000)
        self.assertEqual(scc.value, 1)
        scc.update(0x0)
        self.assertEqual(scc.value, 0)


if __name__ == "__main__":
    unittest.main()
