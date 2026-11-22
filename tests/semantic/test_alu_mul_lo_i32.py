"""Semantic interpreter tests for AMD RDNA2 vector 32-bit integer low multiplication:
  - v_mul_lo_u32: D.u32 = (S0.u32 * S1.u32) & 0xFFFFFFFF

In two's complement binary arithmetic, the low 32 bits of 32-bit integer multiplication
are mathematically identical regardless of whether operands are treated as signed (i32)
or unsigned (u32), because (a * b) mod 2^32 is isomorphic across signed and unsigned
representations. RDNA2 therefore implements both signed and unsigned 32-bit low-dword
multiplication with the single VALU instruction v_mul_lo_u32, translated to SPIR-V OpIMul.

Standard library only. No game assets, archives, or shaders.
"""
from __future__ import annotations

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
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import rdna2spirv  # noqa: E402
import spirv_mini_interp as si  # noqa: E402

MASK32 = 0xFFFFFFFF
SIGN32 = 0x80000000
MASK64 = 0xFFFFFFFFFFFFFFFF
STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)

SAMPLE_VALUES = (
    0,
    1,
    2,
    3,
    7,
    15,
    16,
    31,
    32,
    64,
    100,
    255,
    256,
    0x1000,
    0x10000,
    0x55555555,
    0xAAAAAAAA,
    0x12345678,
    0x9ABCDEF0,
    0x7FFFFFFE,
    0x7FFFFFFF,
    0x80000000,
    0x80000001,
    0xFFFFFFFE,
    0xFFFFFFFF,
)

SAMPLE_PAIRS = (
    (0, 0),
    (0, 1),
    (0, 0xFFFFFFFF),
    (1, 1),
    (1, 0xFFFFFFFF),
    (2, 3),
    (7, 9),
    (10, 10),
    (16, 64),
    (0x10000, 0x10000),
    (0x7FFFFFFF, 1),
    (0x7FFFFFFF, 2),
    (0x7FFFFFFF, 0x7FFFFFFF),
    (0x80000000, 1),
    (0x80000000, 2),
    (0x80000000, 0x80000000),
    (0x80000000, 0xFFFFFFFF),
    (0xFFFFFFFF, 0xFFFFFFFF),
    (0xFFFFFFFF, 5),
    (0x12345678, 0x9ABCDEF0),
    (0xAAAA5555, 0x5555AAAA),
    (0xDEADBEEF, 0xCAFEBABE),
    (0x80000001, 0x7FFFFFFE),
    (100, 0xFFFFFF9C),  # 100 * -100
)


def u32(x: int) -> int:
    """Mask integer to 32 bits unsigned."""
    return x & MASK32


def s32(x: int) -> int:
    """Convert integer to 32-bit signed two's complement integer."""
    val = x & MASK32
    return val - (1 << 32) if (val & SIGN32) else val


def ref_mul_lo_u32(a: int, b: int) -> int:
    """Reference oracle for v_mul_lo_u32: (u32(a) * u32(b)) & 0xFFFFFFFF."""
    return (u32(a) * u32(b)) & MASK32


def ref_mul_lo_i32(a: int, b: int) -> int:
    """Signed reference for low 32-bit multiplication: (s32(a) * s32(b)) & 0xFFFFFFFF."""
    return (s32(a) * s32(b)) & MASK32


def ref_mul_64(a: int, b: int, signed: bool = False) -> tuple[int, int]:
    """Returns (lo, hi) halves of full 64-bit product."""
    if signed:
        prod = s32(a) * s32(b)
        prod_u = prod & MASK64
    else:
        prod_u = (u32(a) * u32(b)) & MASK64
    return prod_u & MASK32, (prod_u >> 32) & MASK32


_TMP_DIR: tempfile.TemporaryDirectory | None = None
_SRC_PATH: str | None = None


def setUpModule() -> None:
    global _TMP_DIR, _SRC_PATH
    _TMP_DIR = tempfile.TemporaryDirectory()
    disasm_dir = os.path.join(_TMP_DIR.name, "disasm")
    os.mkdir(disasm_dir)
    _SRC_PATH = os.path.join(disasm_dir, "ps_mul_lo.s")
    with open(os.path.join(_TMP_DIR.name, "ps_mul_lo.dxbc"), "wb") as f:
        f.write(STUB_DXBC)


def tearDownModule() -> None:
    global _TMP_DIR, _SRC_PATH
    if _TMP_DIR is not None:
        gc.collect()
        _TMP_DIR.cleanup()
        _TMP_DIR = None
        _SRC_PATH = None


def run_mul_lo_shader(
    a: int,
    b: int,
    op: str = "v_mul_lo_u32",
    inline_s0: bool = False,
    inline_s1: bool = False,
    s0_is_sgpr: bool = False,
) -> int:
    """Run `op v0, s0/v1, v2` and return the 32-bit word written to mrt0[0]."""
    if _TMP_DIR is None:
        setUpModule()
    lines = []
    if s0_is_sgpr:
        lines.append(f"\ts_mov_b32 s0, {u32(a):#x}")
        op0 = "s0"
    elif inline_s0:
        op0 = f"{u32(a):#x}"
    else:
        lines.append(f"\tv_mov_b32 v1, {u32(a):#x}")
        op0 = "v1"

    if inline_s1:
        op1 = f"{u32(b):#x}"
    else:
        lines.append(f"\tv_mov_b32 v2, {u32(b):#x}")
        op1 = "v2"

    lines.append(f"\t{op} v0, {op0}, {op1}")
    asm = (
        "; test_alu_mul_lo_i32\n"
        + "\n".join(lines)
        + "\n\texp mrt0 v0, v0, v0, v0 done vm\n\ts_endpgm\n"
    )
    with open(_SRC_PATH, "w", encoding="utf-8") as f:
        f.write(asm)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)
        spv = rdna2spirv.translate(_SRC_PATH)
        gc.collect()
    return si.run_fragment(spv)["%out_mrt0"][0]


def run_spirv_imul(a: int, b: int) -> int:
    """Direct execution of SPIR-V OpIMul instruction in spirv_mini_interp."""
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
%ca = OpConstant %uint {u32(a)}
%cb = OpConstant %uint {u32(b)}
%main = OpFunction %void None %fn
%entry = OpLabel
%res = OpIMul %uint %ca %cb
OpStore %out %res
OpReturn
OpFunctionEnd
"""
    return si.run_fragment(module)["%out"][0]


class TestMulLoReferenceOracle(unittest.TestCase):
    """Mathematical validation of 32-bit integer low multiplication reference oracle."""

    def test_zero_multiplier(self) -> None:
        """Any 32-bit integer multiplied by 0 equals 0."""
        for val in SAMPLE_VALUES:
            with self.subTest(val=hex(val)):
                self.assertEqual(ref_mul_lo_u32(val, 0), 0)
                self.assertEqual(ref_mul_lo_u32(0, val), 0)
                self.assertEqual(ref_mul_lo_i32(val, 0), 0)
                self.assertEqual(ref_mul_lo_i32(0, val), 0)

    def test_unity_multiplier(self) -> None:
        """Any 32-bit integer multiplied by 1 preserves its value."""
        for val in SAMPLE_VALUES:
            with self.subTest(val=hex(val)):
                self.assertEqual(ref_mul_lo_u32(val, 1), u32(val))
                self.assertEqual(ref_mul_lo_u32(1, val), u32(val))
                self.assertEqual(ref_mul_lo_i32(val, 1), u32(val))
                self.assertEqual(ref_mul_lo_i32(1, val), u32(val))

    def test_negation_identity(self) -> None:
        """Multiplying by -1 (0xFFFFFFFF) yields two's complement negation modulo 2^32."""
        for val in SAMPLE_VALUES:
            with self.subTest(val=hex(val)):
                expected = (-u32(val)) & MASK32
                self.assertEqual(ref_mul_lo_u32(val, -1), expected)
                self.assertEqual(ref_mul_lo_u32(val, 0xFFFFFFFF), expected)
                self.assertEqual(ref_mul_lo_i32(val, -1), expected)
                self.assertEqual(ref_mul_lo_i32(val, 0xFFFFFFFF), expected)

    def test_commutativity(self) -> None:
        """Low multiplication is commutative: a * b == b * a."""
        for a, b in SAMPLE_PAIRS:
            with self.subTest(a=hex(a), b=hex(b)):
                self.assertEqual(ref_mul_lo_u32(a, b), ref_mul_lo_u32(b, a))
                self.assertEqual(ref_mul_lo_i32(a, b), ref_mul_lo_i32(b, a))

    def test_associativity_mod32(self) -> None:
        """Low multiplication is associative modulo 2^32: (a * b) * c == a * (b * c)."""
        triplets = [
            (3, 5, 7),
            (100, 200, 300),
            (0x1000, 0x100, 0x10),
            (0x12345678, 3, 5),
            (0x7FFFFFFF, 2, 4),
            (0xFFFFFFFF, 2, 3),
            (0x80000001, 0x10001, 7),
        ]
        for a, b, c in triplets:
            with self.subTest(a=hex(a), b=hex(b), c=hex(c)):
                left = ref_mul_lo_u32(ref_mul_lo_u32(a, b), c)
                right = ref_mul_lo_u32(a, ref_mul_lo_u32(b, c))
                self.assertEqual(left, right)

    def test_distributivity_over_addition(self) -> None:
        """Low multiplication distributes over addition modulo 2^32: a * (b + c) == a * b + a * c."""
        triplets = [
            (5, 12, 18),
            (0x1000, 0x2000, 0x3000),
            (0xFFFFFFFF, 1, 1),
            (0x7FFFFFFF, 0x80000000, 1),
            (0x12345678, 0x9ABCDEF0, 0x55555555),
        ]
        for a, b, c in triplets:
            with self.subTest(a=hex(a), b=hex(b), c=hex(c)):
                sum_bc = (b + c) & MASK32
                prod_distributed = (ref_mul_lo_u32(a, b) + ref_mul_lo_u32(a, c)) & MASK32
                self.assertEqual(ref_mul_lo_u32(a, sum_bc), prod_distributed)

    def test_power_of_two_shift_equivalence(self) -> None:
        """Multiplication by 2^k is equivalent to left shift by k modulo 2^32."""
        for val in (1, 3, 7, 42, 0x12345678, 0x80000001, 0xFFFFFFFF):
            for k in range(32):
                with self.subTest(val=hex(val), shift=k):
                    multiplier = 1 << k
                    expected = (u32(val) << k) & MASK32
                    self.assertEqual(ref_mul_lo_u32(val, multiplier), expected)

    def test_hand_calculated_constants(self) -> None:
        """Rigorous validation against hand-calculated boundary results."""
        cases = [
            (0, 0, 0),
            (1, 1, 1),
            # 0xFFFFFFFF * 0xFFFFFFFF = (-1) * (-1) = 1 in two's complement
            (0xFFFFFFFF, 0xFFFFFFFF, 1),
            # 0x7FFFFFFF * 2 = 0xFFFFFFFE (INT32_MAX * 2)
            (0x7FFFFFFF, 2, 0xFFFFFFFE),
            # 0x80000000 * 2 = 0x1_00000000 -> lo = 0
            (0x80000000, 2, 0),
            # 0x80000000 * 0x80000000 = (-2^31)^2 = 2^62 = 0x40000000_00000000 -> lo = 0
            (0x80000000, 0x80000000, 0),
            # 0x80000000 * 0xFFFFFFFF = (-2^31) * (-1) = 2^31 = 0x80000000
            (0x80000000, 0xFFFFFFFF, 0x80000000),
            # 0x10000 * 0x10000 = 2^16 * 2^16 = 2^32 = 0x1_00000000 -> lo = 0
            (0x10000, 0x10000, 0),
            # 0x10001 * 0x10001 = (2^16 + 1)^2 = 2^32 + 2*2^16 + 1 -> lo = 0x00020001
            (0x10001, 0x10001, 0x00020001),
            # 100 * -1 = -100 = 0xFFFFFF9C
            (100, -1, 0xFFFFFF9C),
            # 100 * 0xFFFFFFFF = 0xFFFFFF9C
            (100, 0xFFFFFFFF, 0xFFFFFF9C),
            # 0x12345678 * 2 = 0x2468ACF0
            (0x12345678, 2, 0x2468ACF0),
        ]
        for a, b, expected in cases:
            with self.subTest(a=hex(a), b=hex(b), expected=hex(expected)):
                self.assertEqual(ref_mul_lo_u32(a, b), expected)
                self.assertEqual(ref_mul_lo_i32(a, b), expected)


class TestMulLoTranslator(unittest.TestCase):
    """End-to-end tests translating RDNA2 v_mul_lo_u32 and executing in spirv_mini_interp."""

    def test_basic_execution(self) -> None:
        """Test v_mul_lo_u32 shader execution against the reference oracle."""
        test_pairs = [
            (0, 0),
            (1, 1),
            (2, 3),
            (10, 20),
            (0x10000, 0x10000),
            (0x7FFFFFFF, 2),
            (0x80000000, 2),
            (0xFFFFFFFF, 0xFFFFFFFF),
            (0x12345678, 0x9ABCDEF0),
            (100, -1),
        ]
        for a, b in test_pairs:
            with self.subTest(a=hex(a), b=hex(b)):
                expected = ref_mul_lo_u32(a, b)
                self.assertEqual(run_mul_lo_shader(a, b), expected)
                self.assertEqual(run_mul_lo_shader(b, a), expected)

    def test_inline_constants(self) -> None:
        """Test inline constants in src0 and src1 positions."""
        test_cases = [
            (5, 7, 35),
            (100, -1, 0xFFFFFF9C),
            (0x7FFFFFFF, 2, 0xFFFFFFFE),
            (0x80000000, 2, 0),
            (0xFFFFFFFF, 5, (-5) & MASK32),
        ]
        for a, b, expected in test_cases:
            with self.subTest(a=a, b=b, pos="src1"):
                self.assertEqual(run_mul_lo_shader(a, b, inline_s1=True), expected)
            with self.subTest(a=a, b=b, pos="src0"):
                self.assertEqual(run_mul_lo_shader(a, b, inline_s0=True), expected)

    def test_sgpr_scalar_source(self) -> None:
        """Test mixed scalar SGPR and vector VGPR inputs."""
        test_cases = [
            (42, 10, 420),
            (0x1234, 0x5678, ref_mul_lo_u32(0x1234, 0x5678)),
            (0x80000000, 2, 0),
            (0xFFFFFFFF, 0xFFFFFFFF, 1),
        ]
        for s0_val, v1_val, expected in test_cases:
            with self.subTest(s0=hex(s0_val), v1=hex(v1_val)):
                got = run_mul_lo_shader(s0_val, v1_val, s0_is_sgpr=True)
                self.assertEqual(got, expected)

    def test_opcode_aliases(self) -> None:
        """Test opcode aliases v_mul_lo_u32_e32 and v_mul_lo_u32_e64."""
        aliases = ("v_mul_lo_u32", "v_mul_lo_u32_e32", "v_mul_lo_u32_e64")
        a = 0x12345678
        b = 0x00000002
        expected = ref_mul_lo_u32(a, b)
        for op in aliases:
            with self.subTest(op=op):
                self.assertEqual(run_mul_lo_shader(a, b, op=op), expected)

    def test_spirv_lowering_instruction(self) -> None:
        """Verifies that rdna2spirv emits OpIMul %uint for v_mul_lo_u32."""
        if _TMP_DIR is None:
            setUpModule()
        lines = [
            "\tv_mov_b32 v1, 15",
            "\tv_mov_b32 v2, 4",
            "\tv_mul_lo_u32 v0, v1, v2",
            "\texp mrt0 v0, v0, v0, v0 done vm",
            "\ts_endpgm",
        ]
        asm = "; test_spirv_lowering\n" + "\n".join(lines) + "\n"
        with open(_SRC_PATH, "w", encoding="utf-8") as f:
            f.write(asm)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ResourceWarning)
            spv = rdna2spirv.translate(_SRC_PATH)
        self.assertIn("OpIMul %uint", spv)

    def test_spirv_direct_imul(self) -> None:
        """Direct execution of OpIMul in SPIR-V interpreter matches reference."""
        for a, b in SAMPLE_PAIRS:
            with self.subTest(a=hex(a), b=hex(b)):
                self.assertEqual(run_spirv_imul(a, b), ref_mul_lo_u32(a, b))


if __name__ == "__main__":
    unittest.main()
