from __future__ import annotations

"""Regression and semantic tests for bitwise Boolean algebra and ALU bitop folding.

Features covered:
- RDNA2 bitwise Boolean logic folding and canonicalization:
  - Idempotence: x & x == x, x | x == x.
  - Annihilator (dominance): x & 0 == 0, x | 0xFFFFFFFF == 0xFFFFFFFF.
  - Identity: x & 0xFFFFFFFF == x, x | 0 == x, x ^ 0 == x.
  - Self-inverse: x ^ x == 0.
  - Involution: ~(~x) == x.
  - Absorption: x | (x & y) == x, x & (x | y) == x.
  - Commutativity, associativity, distributivity, De Morgan's laws, and complementarity.
- Translator constant folding for shift/bfe/align operands:
  - Constant shift count modulo 32 mask elimination.
  - Constant v_bfe_u32/i32 offset and width extraction directly to OpBitField[US]Extract.
  - Constant v_alignbyte_b32 and v_alignbit_b32 zero/nonzero folding.
  - dedupe_types early termination and type substitution.
"""

import os
import re
import struct
import sys
import tempfile
import unittest
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TOOLS_DIR = os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools")

sys.dont_write_bytecode = True
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import rdna2spirv  # noqa: E402
import spirv_mini_interp as si  # noqa: E402
import support  # noqa: E402

MASK32 = 0xFFFFFFFF
STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)

# Canonical bitwise test vectors spanning full 32-bit word domain
CANONICAL_TEST_VECTORS = [
    0x00000000,
    0x00000001,
    0x00000002,
    0x0000007F,
    0x00000080,
    0x000000FF,
    0x00000100,
    0x00007FFF,
    0x00008000,
    0x0000FFFF,
    0x00010000,
    0x00FFFFFF,
    0x01000000,
    0x55555555,
    0xAAAAAAAA,
    0x33333333,
    0xCCCCCCCC,
    0x0F0F0F0F,
    0xF0F0F0F0,
    0x00FF00FF,
    0xFF00FF00,
    0x0000FFFF,
    0xFFFF0000,
    0x12345678,
    0x9ABCDEF0,
    0xDEADBEEF,
    0xCAFEBABE,
    0x80000000,
    0x7FFFFFFF,
    0xFFFFFFFE,
    0xFFFFFFFF,
]

SINGLE_BIT_SET = [1 << i for i in range(32)]
SINGLE_BIT_CLR = [(~(1 << i)) & MASK32 for i in range(32)]


def ref_u32(val: int) -> int:
    """Normalize integer to 32-bit unsigned domain."""
    return val & MASK32


def ref_and(a: int, b: int) -> int:
    """Plain-Python 32-bit reference for v_and_b32 / s_and_b32."""
    return (a & b) & MASK32


def ref_or(a: int, b: int) -> int:
    """Plain-Python 32-bit reference for v_or_b32 / s_or_b32."""
    return (a | b) & MASK32


def ref_xor(a: int, b: int) -> int:
    """Plain-Python 32-bit reference for v_xor_b32 / s_xor_b32."""
    return (a ^ b) & MASK32


def ref_not(a: int) -> int:
    """Plain-Python 32-bit reference for v_not_b32 / s_not_b32."""
    return (~a) & MASK32


def ref_andn2(a: int, b: int) -> int:
    """Plain-Python 32-bit reference for s_andn2_b32 (a & ~b)."""
    return (a & (~b & MASK32)) & MASK32


def ref_orn2(a: int, b: int) -> int:
    """Plain-Python 32-bit reference for s_orn2_b32 (a | ~b)."""
    return (a | (~b & MASK32)) & MASK32


def ref_nand(a: int, b: int) -> int:
    """Plain-Python 32-bit reference for s_nand_b32 (~(a & b))."""
    return (~(a & b)) & MASK32


def ref_nor(a: int, b: int) -> int:
    """Plain-Python 32-bit reference for s_nor_b32 (~(a | b))."""
    return (~(a | b)) & MASK32


def ref_xnor(a: int, b: int) -> int:
    """Plain-Python 32-bit reference for s_xnor_b32 (~(a ^ b))."""
    return (~(a ^ b)) & MASK32


class BitwiseAlgebraOracle:
    """Formal mathematical oracle verifying Boolean algebraic identities."""

    @staticmethod
    def check_idempotence_and(x: int) -> bool:
        """x & x == x"""
        return ref_and(x, x) == ref_u32(x)

    @staticmethod
    def check_idempotence_or(x: int) -> bool:
        """x | x == x"""
        return ref_or(x, x) == ref_u32(x)

    @staticmethod
    def check_annihilator_and(x: int) -> bool:
        """x & 0 == 0"""
        return ref_and(x, 0) == 0

    @staticmethod
    def check_annihilator_or(x: int) -> bool:
        """x | 0xFFFFFFFF == 0xFFFFFFFF"""
        return ref_or(x, MASK32) == MASK32

    @staticmethod
    def check_identity_and(x: int) -> bool:
        """x & 0xFFFFFFFF == x"""
        return ref_and(x, MASK32) == ref_u32(x)

    @staticmethod
    def check_identity_or(x: int) -> bool:
        """x | 0 == x"""
        return ref_or(x, 0) == ref_u32(x)

    @staticmethod
    def check_identity_xor(x: int) -> bool:
        """x ^ 0 == x"""
        return ref_xor(x, 0) == ref_u32(x)

    @staticmethod
    def check_self_inverse_xor(x: int) -> bool:
        """x ^ x == 0"""
        return ref_xor(x, x) == 0

    @staticmethod
    def check_involution_not(x: int) -> bool:
        """~(~x) == x"""
        return ref_not(ref_not(x)) == ref_u32(x)

    @staticmethod
    def check_absorption_or_and(x: int, y: int) -> bool:
        """x | (x & y) == x"""
        return ref_or(x, ref_and(x, y)) == ref_u32(x)

    @staticmethod
    def check_absorption_and_or(x: int, y: int) -> bool:
        """x & (x | y) == x"""
        return ref_and(x, ref_or(x, y)) == ref_u32(x)

    @staticmethod
    def check_commutativity_and(x: int, y: int) -> bool:
        """x & y == y & x"""
        return ref_and(x, y) == ref_and(y, x)

    @staticmethod
    def check_commutativity_or(x: int, y: int) -> bool:
        """x | y == y | x"""
        return ref_or(x, y) == ref_or(y, x)

    @staticmethod
    def check_commutativity_xor(x: int, y: int) -> bool:
        """x ^ y == y ^ x"""
        return ref_xor(x, y) == ref_xor(y, x)

    @staticmethod
    def check_associativity_and(x: int, y: int, z: int) -> bool:
        """(x & y) & z == x & (y & z)"""
        return ref_and(ref_and(x, y), z) == ref_and(x, ref_and(y, z))

    @staticmethod
    def check_associativity_or(x: int, y: int, z: int) -> bool:
        """(x | y) | z == x | (y | z)"""
        return ref_or(ref_or(x, y), z) == ref_or(x, ref_or(y, z))

    @staticmethod
    def check_associativity_xor(x: int, y: int, z: int) -> bool:
        """(x ^ y) ^ z == x ^ (y ^ z)"""
        return ref_xor(ref_xor(x, y), z) == ref_xor(x, ref_xor(y, z))

    @staticmethod
    def check_distributivity_and_over_or(x: int, y: int, z: int) -> bool:
        """x & (y | z) == (x & y) | (x & z)"""
        return ref_and(x, ref_or(y, z)) == ref_or(ref_and(x, y), ref_and(x, z))

    @staticmethod
    def check_distributivity_or_over_and(x: int, y: int, z: int) -> bool:
        """x | (y & z) == (x | y) & (x | z)"""
        return ref_or(x, ref_and(y, z)) == ref_and(ref_or(x, y), ref_or(x, z))

    @staticmethod
    def check_demorgan_and(x: int, y: int) -> bool:
        """~(x & y) == (~x) | (~y)"""
        return ref_not(ref_and(x, y)) == ref_or(ref_not(x), ref_not(y))

    @staticmethod
    def check_demorgan_or(x: int, y: int) -> bool:
        """~(x | y) == (~x) & (~y)"""
        return ref_not(ref_or(x, y)) == ref_and(ref_not(x), ref_not(y))

    @staticmethod
    def check_complement_and(x: int) -> bool:
        """x & (~x) == 0"""
        return ref_and(x, ref_not(x)) == 0

    @staticmethod
    def check_complement_or(x: int) -> bool:
        """x | (~x) == 0xFFFFFFFF"""
        return ref_or(x, ref_not(x)) == MASK32

    @staticmethod
    def check_complement_xor(x: int) -> bool:
        """x ^ (~x) == 0xFFFFFFFF"""
        return ref_xor(x, ref_not(x)) == MASK32


class BitwiseAlgebraOracleTests(unittest.TestCase):
    """Test suite executing the reference oracle against exhaustive vector sets."""

    def test_idempotence(self):
        for vec in CANONICAL_TEST_VECTORS + SINGLE_BIT_SET + SINGLE_BIT_CLR:
            with self.subTest(vec=hex(vec)):
                self.assertTrue(BitwiseAlgebraOracle.check_idempotence_and(vec))
                self.assertTrue(BitwiseAlgebraOracle.check_idempotence_or(vec))

    def test_annihilator(self):
        for vec in CANONICAL_TEST_VECTORS + SINGLE_BIT_SET + SINGLE_BIT_CLR:
            with self.subTest(vec=hex(vec)):
                self.assertTrue(BitwiseAlgebraOracle.check_annihilator_and(vec))
                self.assertTrue(BitwiseAlgebraOracle.check_annihilator_or(vec))

    def test_identity(self):
        for vec in CANONICAL_TEST_VECTORS + SINGLE_BIT_SET + SINGLE_BIT_CLR:
            with self.subTest(vec=hex(vec)):
                self.assertTrue(BitwiseAlgebraOracle.check_identity_and(vec))
                self.assertTrue(BitwiseAlgebraOracle.check_identity_or(vec))
                self.assertTrue(BitwiseAlgebraOracle.check_identity_xor(vec))

    def test_self_inverse(self):
        for vec in CANONICAL_TEST_VECTORS + SINGLE_BIT_SET + SINGLE_BIT_CLR:
            with self.subTest(vec=hex(vec)):
                self.assertTrue(BitwiseAlgebraOracle.check_self_inverse_xor(vec))

    def test_involution(self):
        for vec in CANONICAL_TEST_VECTORS + SINGLE_BIT_SET + SINGLE_BIT_CLR:
            with self.subTest(vec=hex(vec)):
                self.assertTrue(BitwiseAlgebraOracle.check_involution_not(vec))

    def test_absorption(self):
        vectors = CANONICAL_TEST_VECTORS[:12]
        for x in vectors:
            for y in vectors:
                with self.subTest(x=hex(x), y=hex(y)):
                    self.assertTrue(BitwiseAlgebraOracle.check_absorption_or_and(x, y))
                    self.assertTrue(BitwiseAlgebraOracle.check_absorption_and_or(x, y))

    def test_commutativity(self):
        vectors = CANONICAL_TEST_VECTORS[:10]
        for x in vectors:
            for y in vectors:
                with self.subTest(x=hex(x), y=hex(y)):
                    self.assertTrue(BitwiseAlgebraOracle.check_commutativity_and(x, y))
                    self.assertTrue(BitwiseAlgebraOracle.check_commutativity_or(x, y))
                    self.assertTrue(BitwiseAlgebraOracle.check_commutativity_xor(x, y))

    def test_associativity(self):
        vectors = [0x0, 0xFFFFFFFF, 0x55555555, 0xAAAAAAAA, 0x12345678, 0x80000000]
        for x in vectors:
            for y in vectors:
                for z in vectors:
                    with self.subTest(x=hex(x), y=hex(y), z=hex(z)):
                        self.assertTrue(BitwiseAlgebraOracle.check_associativity_and(x, y, z))
                        self.assertTrue(BitwiseAlgebraOracle.check_associativity_or(x, y, z))
                        self.assertTrue(BitwiseAlgebraOracle.check_associativity_xor(x, y, z))

    def test_distributivity(self):
        vectors = [0x0, 0xFFFFFFFF, 0x55555555, 0xAAAAAAAA, 0x12345678, 0x80000000]
        for x in vectors:
            for y in vectors:
                for z in vectors:
                    with self.subTest(x=hex(x), y=hex(y), z=hex(z)):
                        self.assertTrue(BitwiseAlgebraOracle.check_distributivity_and_over_or(x, y, z))
                        self.assertTrue(BitwiseAlgebraOracle.check_distributivity_or_over_and(x, y, z))

    def test_demorgan_laws(self):
        vectors = CANONICAL_TEST_VECTORS[:12]
        for x in vectors:
            for y in vectors:
                with self.subTest(x=hex(x), y=hex(y)):
                    self.assertTrue(BitwiseAlgebraOracle.check_demorgan_and(x, y))
                    self.assertTrue(BitwiseAlgebraOracle.check_demorgan_or(x, y))

    def test_complementarity(self):
        for vec in CANONICAL_TEST_VECTORS + SINGLE_BIT_SET + SINGLE_BIT_CLR:
            with self.subTest(vec=hex(vec)):
                self.assertTrue(BitwiseAlgebraOracle.check_complement_and(vec))
                self.assertTrue(BitwiseAlgebraOracle.check_complement_or(vec))
                self.assertTrue(BitwiseAlgebraOracle.check_complement_xor(vec))

    def test_composite_folding_identities(self):
        vectors = CANONICAL_TEST_VECTORS[:12]
        for x in vectors:
            # x ^ 0xFFFFFFFF == ~x
            self.assertEqual(ref_xor(x, MASK32), ref_not(x))
            for y in vectors:
                # (x | y) & (x | ~y) == x
                lhs1 = ref_and(ref_or(x, y), ref_or(x, ref_not(y)))
                self.assertEqual(lhs1, ref_u32(x))
                # (x & y) | (x & ~y) == x
                lhs2 = ref_or(ref_and(x, y), ref_and(x, ref_not(y)))
                self.assertEqual(lhs2, ref_u32(x))
                # x ^ (x & y) == x & ~y
                lhs3 = ref_xor(x, ref_and(x, y))
                self.assertEqual(lhs3, ref_andn2(x, y))


class BitopConstantFoldingTests(unittest.TestCase):
    def translate_lines(self, lines: list[str]) -> str:
        body = "\n".join("\t" + line for line in lines)
        asm = f"; bitop_test\n{body}\n\texp mrt0 v0, v0, v0, v0 done vm\n\ts_endpgm\n"
        with tempfile.TemporaryDirectory() as tmp:
            d_dir = os.path.join(tmp, "disasm")
            os.mkdir(d_dir)
            s_path = os.path.join(d_dir, "ps_bitop.s")
            with open(s_path, "w", encoding="utf-8") as f:
                f.write(asm)
            with open(os.path.join(tmp, "ps_bitop.dxbc"), "wb") as f:
                f.write(STUB_DXBC)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ResourceWarning)
                return rdna2spirv.translate(s_path)

    def test_constant_shift_eliminates_runtime_mask(self):
        lines = [
            "v_mov_b32 v1, 0x1234",
            "v_lshlrev_b32 v0, 4, v1",
        ]
        spv = self.translate_lines(lines)
        self.assertNotIn("OpBitwiseAnd %uint", spv)
        self.assertIn("OpShiftLeftLogical %uint", spv)
        res = si.run_fragment(spv)["%out_mrt0"][0]
        self.assertEqual(res, 0x12340)

    def test_constant_bfe_eliminates_runtime_bounds(self):
        lines = [
            "v_mov_b32 v1, 0x12345678",
            "v_bfe_u32 v0, v1, 8, 12",
        ]
        spv = self.translate_lines(lines)
        self.assertNotIn("OpBitwiseAnd %uint", spv)
        self.assertNotIn("OpISub %uint", spv)
        self.assertNotIn("OpULessThan %bool", spv)
        self.assertIn("OpBitFieldUExtract %uint", spv)
        res = si.run_fragment(spv)["%out_mrt0"][0]
        self.assertEqual(res, 0x456)

    def test_constant_alignbyte_zero_shift_folds_directly(self):
        lines = [
            "v_mov_b32 v1, 0xDEADBEEF",
            "v_mov_b32 v2, 0x12345678",
            "v_alignbyte_b32 v0, v1, v2, 0",
        ]
        spv = self.translate_lines(lines)
        self.assertNotIn("OpShiftRightLogical", spv)
        self.assertNotIn("OpShiftLeftLogical", spv)
        self.assertNotIn("OpBitwiseOr", spv)
        res = si.run_fragment(spv)["%out_mrt0"][0]
        self.assertEqual(res, 0x12345678)

    def test_constant_alignbyte_nonzero_shift_emits_direct_shifts(self):
        lines = [
            "v_mov_b32 v1, 0x8899AABB",
            "v_mov_b32 v2, 0x11223344",
            "v_alignbyte_b32 v0, v1, v2, 1",
        ]
        spv = self.translate_lines(lines)
        self.assertNotIn("OpBitwiseAnd %uint", spv)
        self.assertNotIn("OpISub %uint", spv)
        self.assertIn("OpShiftRightLogical %uint", spv)
        self.assertIn("OpShiftLeftLogical %uint", spv)
        self.assertIn("OpBitwiseOr %uint", spv)
        res = si.run_fragment(spv)["%out_mrt0"][0]
        self.assertEqual(res, 0xBB112233)

    def test_constant_alignbit_zero_shift_folds_directly(self):
        lines = [
            "v_mov_b32 v1, 0xDEADBEEF",
            "v_mov_b32 v2, 0x12345678",
            "v_alignbit_b32 v0, v1, v2, 0",
        ]
        spv = self.translate_lines(lines)
        self.assertNotIn("OpShiftRightLogical", spv)
        self.assertNotIn("OpShiftLeftLogical", spv)
        self.assertNotIn("OpBitwiseOr", spv)
        res = si.run_fragment(spv)["%out_mrt0"][0]
        self.assertEqual(res, 0x12345678)

    def test_constant_alignbit_nonzero_shift_emits_direct_shifts(self):
        lines = [
            "v_mov_b32 v1, 0x12345678",
            "v_mov_b32 v2, 0x9ABCDEF0",
            "v_alignbit_b32 v0, v1, v2, 16",
        ]
        spv = self.translate_lines(lines)
        self.assertNotIn("OpBitwiseAnd %uint", spv)
        self.assertNotIn("OpISub %uint", spv)
        self.assertIn("OpShiftRightLogical %uint", spv)
        self.assertIn("OpShiftLeftLogical %uint", spv)
        self.assertIn("OpBitwiseOr %uint", spv)
        res = si.run_fragment(spv)["%out_mrt0"][0]
        self.assertEqual(res, 0x56789ABC)

    def test_dedupe_types_merges_duplicate_types_and_substitutes(self):
        spv_sample = (
            "OpCapability Shader\n"
            "%uint = OpTypeInt 32 0\n"
            "%v4uint = OpTypeVector %uint 4\n"
            "%v4uint_st = OpTypeVector %uint 4\n"
            "%ptr_v4uint = OpTypePointer StorageBuffer %v4uint\n"
            "%ptr_v4uint_st = OpTypePointer StorageBuffer %v4uint_st\n"
            "%fn = OpTypeFunction %void\n"
            "%main = OpFunction %void None %fn\n"
            "%entry = OpLabel\n"
            "%var = OpVariable %ptr_v4uint_st StorageBuffer\n"
            "%val = OpLoad %v4uint_st %var\n"
            "OpReturn\n"
            "OpFunctionEnd"
        )
        deduped = rdna2spirv.dedupe_types(spv_sample)
        self.assertNotIn("%v4uint_st =", deduped)
        self.assertNotIn("%ptr_v4uint_st =", deduped)
        self.assertIn("%var = OpVariable %ptr_v4uint StorageBuffer", deduped)
        self.assertIn("%val = OpLoad %v4uint %var", deduped)


if __name__ == "__main__":
    unittest.main()
