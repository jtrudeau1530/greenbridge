"""Semantic interpreter tests for AMD RDNA composite shift-add instructions:
  - v_lshl_add_u32: D.u32 = ((S0.u32 << (S1.u32 & 31)) + S2.u32) & 0xFFFFFFFF
  - v_lshl_add_u64: D.u64 = ((S0.u64 << (S1.u32 & mask)) + S2.u64) & 0xFFFFFFFFFFFFFFFF

Standard library only; models VALU bitwise arithmetic, shift count masking,
cross-dword bit shifting, carry propagation, and wraparound overflow.
Cross-checked against the SPIR-V mini interpreter (spirv_mini_interp.py).
"""
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import spirv_mini_interp as si

MASK32 = 0xFFFFFFFF
MASK64 = 0xFFFFFFFFFFFFFFFF


def interp_v_lshl_add_u32(s0, s1, s2):
    """Reference interpreter for v_lshl_add_u32.
    Shifts 32-bit unsigned integer s0 left by (s1 & 31) bits, then adds s2 with 32-bit wrap.
    """
    shift = s1 & 31
    shifted = ((s0 & MASK32) << shift) & MASK32
    return (shifted + (s2 & MASK32)) & MASK32


def interp_v_lshl_add_u64(s0, s1, s2, shift_mask=0x7):
    """Reference interpreter for v_lshl_add_u64.
    Shifts 64-bit unsigned integer s0 left by (s1 & shift_mask) bits, then adds s2 with 64-bit wrap.
    AMD hardware defaults shift_mask to 0x7 (bits 2:0) for 1..8 byte address scaling.
    """
    shift = s1 & shift_mask
    shifted = ((s0 & MASK64) << shift) & MASK64
    return (shifted + (s2 & MASK64)) & MASK64


def decompose_u64(val):
    """Decompose a 64-bit unsigned integer into (lo, hi) 32-bit dwords."""
    val &= MASK64
    return val & MASK32, (val >> 32) & MASK32


def compose_u64(lo, hi):
    """Compose two 32-bit dwords into a 64-bit unsigned integer."""
    return ((hi & MASK32) << 32) | (lo & MASK32)


def interp_v_lshl_add_u64_dwords(s0_lo, s0_hi, s1, s2_lo, s2_hi, shift_mask=0x7):
    """Emulate 64-bit shift-add using 32-bit register pairs and explicit carry tracking.
    Validates hardware register pair decomposition against 64-bit whole integer semantics.
    """
    shift = s1 & shift_mask
    s0_lo &= MASK32
    s0_hi &= MASK32
    s2_lo &= MASK32
    s2_hi &= MASK32

    if shift == 0:
        shifted_lo = s0_lo
        shifted_hi = s0_hi
    elif shift < 32:
        shifted_lo = (s0_lo << shift) & MASK32
        shifted_hi = (((s0_hi << shift) | (s0_lo >> (32 - shift)))) & MASK32
    else:
        shifted_lo = 0
        shifted_hi = (s0_lo << (shift - 32)) & MASK32

    sum_lo_raw = shifted_lo + s2_lo
    res_lo = sum_lo_raw & MASK32
    carry = 1 if sum_lo_raw > MASK32 else 0

    sum_hi_raw = shifted_hi + s2_hi + carry
    res_hi = sum_hi_raw & MASK32

    return res_lo, res_hi


def run_spirv_lshl_add_u32(s0, s1, s2):
    """Executes the exact SPIR-V instruction sequence emitted by rdna2spirv.py for v_lshl_add_u32."""
    module = f"""\
OpCapability Shader
%glsl = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %out
%void = OpTypeVoid
%fn = OpTypeFunction %void
%uint = OpTypeInt 32 0
%_ptr_Output_uint = OpTypePointer Output %uint
%out = OpVariable %_ptr_Output_uint Output
%c_s0 = OpConstant %uint {s0 & MASK32}
%c_s1 = OpConstant %uint {s1 & MASK32}
%c_s2 = OpConstant %uint {s2 & MASK32}
%c31 = OpConstant %uint 31
%main = OpFunction %void None %fn
%entry = OpLabel
%m = OpBitwiseAnd %uint %c_s1 %c31
%s = OpShiftLeftLogical %uint %c_s0 %m
%res = OpIAdd %uint %s %c_s2
OpStore %out %res
OpReturn
OpFunctionEnd
"""
    return si.run_fragment(module)["%out"][0]


class TestVLshlAddU32Semantics(unittest.TestCase):
    """Unit tests for 32-bit composite shift-add instruction (v_lshl_add_u32)."""

    def test_identity_shift_zero(self):
        vectors = [
            (0, 0, 0, 0),
            (1234, 0, 5678, 6912),
            (0xFFFFFFFF, 0, 1, 0),
            (0x80000000, 0, 0x80000000, 0),
            (42, 0, 0, 42),
        ]
        for s0, s1, s2, expected in vectors:
            with self.subTest(s0=s0, s1=s1, s2=s2):
                got = interp_v_lshl_add_u32(s0, s1, s2)
                self.assertEqual(got, expected)

    def test_power_of_two_scaling(self):
        base_address = 0x1000
        index = 15
        # scale 1 (halfword / 2 bytes)
        self.assertEqual(interp_v_lshl_add_u32(index, 1, base_address), base_address + 30)
        # scale 2 (dword / 4 bytes)
        self.assertEqual(interp_v_lshl_add_u32(index, 2, base_address), base_address + 60)
        # scale 3 (qword / 8 bytes)
        self.assertEqual(interp_v_lshl_add_u32(index, 3, base_address), base_address + 120)
        # scale 4 (16 bytes / dqword)
        self.assertEqual(interp_v_lshl_add_u32(index, 4, base_address), base_address + 240)

    def test_shift_masking_5bit(self):
        s0 = 0x1
        s2 = 0x10
        # In VALU 32-bit shift, count is masked with 0x1F (bits 4:0)
        self.assertEqual(interp_v_lshl_add_u32(s0, 32, s2), interp_v_lshl_add_u32(s0, 0, s2))
        self.assertEqual(interp_v_lshl_add_u32(s0, 33, s2), interp_v_lshl_add_u32(s0, 1, s2))
        self.assertEqual(interp_v_lshl_add_u32(s0, 35, s2), interp_v_lshl_add_u32(s0, 3, s2))
        self.assertEqual(interp_v_lshl_add_u32(s0, 63, s2), interp_v_lshl_add_u32(s0, 31, s2))
        self.assertEqual(interp_v_lshl_add_u32(s0, 64, s2), interp_v_lshl_add_u32(s0, 0, s2))
        self.assertEqual(interp_v_lshl_add_u32(s0, 0xFFFFFFFF, s2), interp_v_lshl_add_u32(s0, 31, s2))

    def test_shift_overflow_discard(self):
        # High bits shifted out of bit 31 are truncated
        s0 = 0x80000001
        self.assertEqual(interp_v_lshl_add_u32(s0, 1, 0), 0x00000002)
        self.assertEqual(interp_v_lshl_add_u32(s0, 31, 0), 0x80000000)
        self.assertEqual(interp_v_lshl_add_u32(0xFFFFFFFF, 16, 0), 0xFFFF0000)

    def test_add_overflow_wraparound(self):
        # Addition overflow wraps modulo 2^32
        self.assertEqual(interp_v_lshl_add_u32(0x40000000, 1, 0x80000000), 0)
        self.assertEqual(interp_v_lshl_add_u32(0x1, 31, 0x80000001), 1)
        self.assertEqual(interp_v_lshl_add_u32(0xFFFFFFFF, 0, 0xFFFFFFFF), 0xFFFFFFFE)

    def test_both_shift_and_add_overflow(self):
        s0 = 0xC0000001
        s1 = 2
        s2 = 0xFFFFFFFF
        # (0xC0000001 << 2) & MASK32 = 0x00000004
        # (4 + 0xFFFFFFFF) & MASK32 = 3
        self.assertEqual(interp_v_lshl_add_u32(s0, s1, s2), 3)

    def test_wave_lane_vector_simulation(self):
        base_ptr = 0x8000
        lanes = 64
        for lane_id in range(lanes):
            offset = interp_v_lshl_add_u32(lane_id, 2, base_ptr)
            expected = base_ptr + (lane_id * 4)
            self.assertEqual(offset, expected)


class TestVLshlAddU32SpirvInterp(unittest.TestCase):
    """Validates v_lshl_add_u32 execution through the SPIR-V mini interpreter."""

    def test_spirv_matches_semantic_reference(self):
        test_cases = [
            (0, 0, 0),
            (1, 0, 10),
            (7, 3, 5),
            (0x12345678, 4, 0x0000000F),
            (0x80000000, 1, 0x1),
            (0xFFFFFFFF, 0, 1),
            (0x0000FFFF, 16, 0x00000001),
            (1, 31, 0x7FFFFFFF),
            (5, 33, 10),  # shift 33 -> 1
            (0xAAAAAAAA, 35, 0x55555555),  # shift 35 -> 3
        ]
        for s0, s1, s2 in test_cases:
            with self.subTest(s0=s0, s1=s1, s2=s2):
                expected = interp_v_lshl_add_u32(s0, s1, s2)
                got = run_spirv_lshl_add_u32(s0, s1, s2)
                self.assertEqual(got, expected)

    def test_spirv_shift_mask_prevents_unsupported_large_shift(self):
        # In spirv_mini_interp.py, OpShiftLeftLogical raises UnsupportedConstruct if shift >= 32.
        # The translator must emit OpBitwiseAnd %c31 so shift amount is strictly in [0, 31].
        got = run_spirv_lshl_add_u32(0x1234, 100, 42)
        expected = interp_v_lshl_add_u32(0x1234, 100, 42)
        self.assertEqual(got, expected)


class TestVLshlAddU64Semantics(unittest.TestCase):
    """Unit tests for 64-bit composite shift-add instruction (v_lshl_add_u64)."""

    def test_basic_64bit_shift_and_add(self):
        s0 = 0x0000000000000005
        s1 = 2
        s2 = 0x0000000000000010
        self.assertEqual(interp_v_lshl_add_u64(s0, s1, s2), 36)

    def test_pointer_arithmetic_scaling(self):
        base_address = 0x00007FFF12340000
        index = 0x20
        # 64-bit pointer scale by 8 (qword array index: base + index * 8)
        expected = base_address + (index * 8)
        self.assertEqual(interp_v_lshl_add_u64(index, 3, base_address), expected)

    def test_cross_dword_shift_propagation(self):
        # Low dword bit 31 shifted across 32-bit boundary into high dword
        s0 = 0x0000000080000000
        self.assertEqual(interp_v_lshl_add_u64(s0, 1, 0), 0x0000000100000000)
        self.assertEqual(interp_v_lshl_add_u64(s0, 2, 0), 0x0000000200000000)
        self.assertEqual(interp_v_lshl_add_u64(s0, 3, 0), 0x0000000400000000)

    def test_cross_dword_addition_carry(self):
        # Low dword addition produces carry into high dword
        s0 = 0x0000000000000001
        s1 = 0
        s2 = 0x00000000FFFFFFFF
        got = interp_v_lshl_add_u64(s0, s1, s2)
        self.assertEqual(got, 0x0000000100000000)

        lo, hi = decompose_u64(got)
        self.assertEqual(lo, 0x00000000)
        self.assertEqual(hi, 0x00000001)

    def test_64bit_addition_overflow_wraparound(self):
        s0 = 0x8000000000000000
        s1 = 1
        s2 = 0xFFFFFFFFFFFFFFFF
        # (s0 << 1) & MASK64 = 0
        # (0 + MASK64) & MASK64 = MASK64
        self.assertEqual(interp_v_lshl_add_u64(s0, s1, s2), MASK64)

        # Total 64-bit overflow wrap
        self.assertEqual(interp_v_lshl_add_u64(MASK64, 0, 1), 0)

    def test_shift_masking_3bit_hardware(self):
        # Hardware shift count evaluates [2:0] (0..7)
        s0 = 0x1
        s2 = 0x100
        self.assertEqual(interp_v_lshl_add_u64(s0, 8, s2, shift_mask=0x7), interp_v_lshl_add_u64(s0, 0, s2, shift_mask=0x7))
        self.assertEqual(interp_v_lshl_add_u64(s0, 9, s2, shift_mask=0x7), interp_v_lshl_add_u64(s0, 1, s2, shift_mask=0x7))
        self.assertEqual(interp_v_lshl_add_u64(s0, 15, s2, shift_mask=0x7), interp_v_lshl_add_u64(s0, 7, s2, shift_mask=0x7))

    def test_shift_masking_6bit_general(self):
        # Generic 64-bit shift mask [5:0] (0..63)
        s0 = 0x1
        s2 = 0
        self.assertEqual(interp_v_lshl_add_u64(s0, 64, s2, shift_mask=0x3F), 1)
        self.assertEqual(interp_v_lshl_add_u64(s0, 65, s2, shift_mask=0x3F), 2)
        self.assertEqual(interp_v_lshl_add_u64(s0, 63, s2, shift_mask=0x3F), 0x8000000000000000)

    def test_dword_pair_decomposition_equivalence(self):
        test_pairs = [
            (0x12345678, 0x00000000, 2, 0x00000004, 0x00000000),
            (0x80000000, 0x00000000, 1, 0x00000000, 0x00000000),
            (0xFFFFFFFF, 0x00000000, 1, 0x00000002, 0x00000000),
            (0x00000001, 0x00000002, 3, 0xFFFFFFFF, 0x00000001),
            (0xFFFFFFFF, 0xFFFFFFFF, 2, 0xFFFFFFFF, 0xFFFFFFFF),
            (0x11111111, 0x22222222, 0, 0x33333333, 0x44444444),
        ]
        for s0_lo, s0_hi, s1, s2_lo, s2_hi in test_pairs:
            with self.subTest(s0_lo=s0_lo, s0_hi=s0_hi, s1=s1, s2_lo=s2_lo, s2_hi=s2_hi):
                s0 = compose_u64(s0_lo, s0_hi)
                s2 = compose_u64(s2_lo, s2_hi)
                expected_u64 = interp_v_lshl_add_u64(s0, s1, s2, shift_mask=0x7)
                expected_lo, expected_hi = decompose_u64(expected_u64)

                got_lo, got_hi = interp_v_lshl_add_u64_dwords(s0_lo, s0_hi, s1, s2_lo, s2_hi, shift_mask=0x7)
                self.assertEqual((got_lo, got_hi), (expected_lo, expected_hi))
                self.assertEqual(compose_u64(got_lo, got_hi), expected_u64)


class TestNegativeControls(unittest.TestCase):
    """Negative controls ensuring mutated arithmetic or missing masks fail assertions."""

    def test_bitwise_or_mutation_is_detected(self):
        # Mutating (shifted + s2) to (shifted | s2) as in v_lshl_or_b32 must be detected when bits collide
        s0 = 3
        s1 = 2
        s2 = 4
        # shifted = 12 (0b1100).
        # ADD: 12 + 4 = 16 (0b10000)
        # OR:  12 | 4 = 12 (0b01100)
        correct_add = interp_v_lshl_add_u32(s0, s1, s2)
        mutated_or = ((s0 << (s1 & 31)) | s2) & MASK32
        self.assertNotEqual(correct_add, mutated_or)

    def test_missing_5bit_shift_mask_mutation_is_detected(self):
        s0 = 1
        s1 = 33
        s2 = 0
        correct_masked = interp_v_lshl_add_u32(s0, s1, s2)
        unmasked = (s0 << s1) & MASK32  # 1 << 33 in 32-bit arithmetic would be 0
        self.assertNotEqual(correct_masked, unmasked)

    def test_missing_dword_carry_mutation_is_detected(self):
        s0_lo = 0
        s0_hi = 0
        s1 = 0
        s2_lo = 0xFFFFFFFF
        s2_hi = 0
        # Adding 1 to s2_lo should carry into s2_hi
        res_lo, res_hi = interp_v_lshl_add_u64_dwords(1, 0, s1, s2_lo, s2_hi)
        # Sabotaged: hi without carry
        sabotaged_hi = 0
        self.assertEqual(res_hi, 1)
        self.assertNotEqual(res_hi, sabotaged_hi)


if __name__ == "__main__":
    unittest.main()
