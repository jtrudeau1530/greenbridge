from __future__ import annotations

"""Semantic interpreter tests for RDNA2 8-bit unsigned integer dot product with accumulation.

Instruction:
  v_dot4acc_u32_u8 (VALU VOP3 / VOP3P)

Semantics:
  D = sum(S0[i] * S1[i] for i in 0..3) + S2 (modulo 2^32)

Each 32-bit dword S0 and S1 contains four packed unsigned 8-bit integers:
  Byte 0: bits [7:0]
  Byte 1: bits [15:8]
  Byte 2: bits [23:16]
  Byte 3: bits [31:24]

Each byte pair product S0[i] * S1[i] produces an unsigned integer in [0, 65025].
The sum of the four products yields an unsigned integer in [0, 260100] (0x3F804).
Adding accumulator S2 yields the final 32-bit value. Arithmetic wraps strictly modulo 2^32
unless clamped to 0xFFFFFFFF.

Zero game data, standard library Python only, Python 3.9+ compatible.
"""

import contextlib
import os
import random
import struct
import sys
import unittest

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
MASK8 = 0xFF
MAX_BYTE_VAL = 255
MAX_BYTE_PRODUCT = 255 * 255  # 65025
MAX_DOT4_SUM = 4 * MAX_BYTE_PRODUCT  # 260100 (0x3F804)
STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)

# Mathematically verified test vectors: (S0, S1, S2_acc, expected_result)
KNOWN_VECTORS: list[tuple[int, int, int, int]] = [
    # Zero dot product, zero accumulator
    (0x00000000, 0x00000000, 0x00000000, 0),
    # Zero multipliers preserve accumulator
    (0x00000000, 0x12345678, 0xDEADBEEF, 0xDEADBEEF),
    (0x87654321, 0x00000000, 0xCAFEBABE, 0xCAFEBABE),
    # Sequential bytes: 4*8 + 3*7 + 2*6 + 1*5 = 32 + 21 + 12 + 5 = 70
    (0x01020304, 0x05060708, 0x00000000, 70),
    # Sequential bytes with accumulator: 70 + 1000 = 1070
    (0x01020304, 0x05060708, 0x000003E8, 1070),
    # Single-byte activation at lane 0: 255 * 1 = 255
    (0x000000FF, 0x00000001, 0x00000000, 255),
    # Single-byte square at lane 0: 255 * 255 = 65025
    (0x000000FF, 0x000000FF, 0x00000000, 65025),
    # Single-byte activation at lane 1: 255 * 2 = 510
    (0x0000FF00, 0x00000200, 0x00000000, 510),
    # Single-byte activation at lane 2: 255 * 3 = 765
    (0x00FF0000, 0x00030000, 0x00000000, 765),
    # Single-byte activation at lane 3: 255 * 4 = 1020
    (0xFF000000, 0x04000000, 0x00000000, 1020),
    # Broadcast 0xFF with broadcast 0x01: 4 * 255 = 1020
    (0xFFFFFFFF, 0x01010101, 0x00000000, 1020),
    # Maximum 8-bit unsigned dot product: 4 * (255 * 255) = 260100 (0x3F804)
    (0xFFFFFFFF, 0xFFFFFFFF, 0x00000000, 260100),
    # Maximum dot product with non-zero accumulator: 260100 + 1000000 = 1260100
    (0xFFFFFFFF, 0xFFFFFFFF, 1000000, 1260100),
    # Power-of-two mid bytes: 4 * (128 * 128) = 4 * 16384 = 65536
    (0x80808080, 0x80808080, 0x00000000, 65536),
    # 4 * (128 * 127) + 5 = 4 * 16256 + 5 = 65024 + 5 = 65029
    (0x80808080, 0x7F7F7F7F, 0x00000005, 65029),
    # Arbitrary structured mixed bytes:
    # 10*20 + 30*40 + 50*60 + 70*80 = 200 + 1200 + 3000 + 5600 = 10000
    (0x46321E0A, 0x503C2814, 0x00000000, 10000),
    (0x46321E0A, 0x503C2814, 0x10000000, 0x10002710),
    # Wrapping at exact boundary: 260100 + (2^32 - 260100) = 0 mod 2^32
    (0xFFFFFFFF, 0xFFFFFFFF, (MASK32 - MAX_DOT4_SUM + 1) & MASK32, 0),
    # Wrapping with remainder: 260100 + 0xFFFFFFFF = 260099
    (0xFFFFFFFF, 0xFFFFFFFF, MASK32, 260099),
]


def unpack_u8(dword: int) -> tuple[int, int, int, int]:
    """Unpack a 32-bit dword into four unsigned 8-bit integers (byte0..byte3)."""
    dword &= MASK32
    return (
        dword & MASK8,
        (dword >> 8) & MASK8,
        (dword >> 16) & MASK8,
        (dword >> 24) & MASK8,
    )


def pack_u8(b0: int, b1: int, b2: int, b3: int) -> int:
    """Pack four 8-bit unsigned integers into a 32-bit dword."""
    return (
        (b0 & MASK8)
        | ((b1 & MASK8) << 8)
        | ((b2 & MASK8) << 16)
        | ((b3 & MASK8) << 24)
    ) & MASK32


def ref_dot4_u32_u8_sum(s0: int, s1: int) -> int:
    """Calculate the unaccumulated 4-lane 8-bit unsigned integer dot product sum."""
    b0 = unpack_u8(s0)
    b1 = unpack_u8(s1)
    return sum(x * y for x, y in zip(b0, b1))


def ref_dot4acc_u32_u8(s0: int, s1: int, s2: int = 0) -> int:
    """Reference oracle for RDNA2 v_dot4acc_u32_u8:

    D = (sum_{i=0}^3 (s0[i] * s1[i]) + s2) mod 2^32
    """
    prod_sum = ref_dot4_u32_u8_sum(s0, s1)
    return ((s2 & MASK32) + prod_sum) & MASK32


def ref_dot4acc_clamped_u32_u8(s0: int, s1: int, s2: int = 0) -> int:
    """Reference oracle with unsigned clamping/saturation at 0xFFFFFFFF."""
    prod_sum = ref_dot4_u32_u8_sum(s0, s1)
    total = (s2 & MASK32) + prod_sum
    return min(total, MASK32)


def ref_dot4_i32_i8(s0: int, s1: int, s2: int = 0) -> int:
    """Signed 8-bit dot product reference oracle (v_dot4_i32_i8).

    Used for divergence tests to confirm unsigned behavior differs from signed.
    """
    def s8(x: int) -> int:
        return x - 256 if (x & 0x80) else x

    b0 = unpack_u8(s0)
    b1 = unpack_u8(s1)
    prod_sum = sum(s8(x) * s8(y) for x, y in zip(b0, b1))
    return ((s2 & MASK32) + prod_sum) & MASK32


def build_spirv_dot4acc_fragment(s0: int, s1: int, acc: int = 0, clamp: bool = False) -> str:
    """Synthesize valid SPIR-V assembly computing unsigned dot4acc via OpBitFieldUExtract/OpIMul/OpIAdd."""
    s0 &= MASK32
    s1 &= MASK32
    acc &= MASK32
    clamp_code = ""
    res_var = "%t3"
    if clamp:
        # Detect overflow past 32-bit: if total in 64-bit > 0xFFFFFFFF or if t3 < acc and prod_sum > 0
        clamp_code = """
%total64 = OpIAdd %ulong %acc64 %prod_sum64
%is_overflow = OpUGreaterThan %bool %total64 %c_max32_in64
%clamped_res = OpSelect %uint %is_overflow %c_max32 %t3
"""
        res_var = "%clamped_res"

    return f"""OpCapability Shader
OpCapability Int64
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %out_mrt0
OpExecutionMode %main OriginUpperLeft
OpDecorate %out_mrt0 Location 0
%void = OpTypeVoid
%fn = OpTypeFunction %void
%bool = OpTypeBool
%uint = OpTypeInt 32 0
%ulong = OpTypeInt 64 0
%v4uint = OpTypeVector %uint 4
%pout = OpTypePointer Output %v4uint
%out_mrt0 = OpVariable %pout Output

%c_s0 = OpConstant %uint {s0}
%c_s1 = OpConstant %uint {s1}
%c_acc = OpConstant %uint {acc}
%c0 = OpConstant %uint 0
%c8 = OpConstant %uint 8
%c16 = OpConstant %uint 16
%c24 = OpConstant %uint 24
%c_max32 = OpConstant %uint 4294967295
%c_max32_in64 = OpConstant %ulong 4294967295

%main = OpFunction %void None %fn
%entry = OpLabel
%x0 = OpBitFieldUExtract %uint %c_s0 %c0 %c8
%y0 = OpBitFieldUExtract %uint %c_s1 %c0 %c8
%p0 = OpIMul %uint %x0 %y0
%t0 = OpIAdd %uint %c_acc %p0

%x1 = OpBitFieldUExtract %uint %c_s0 %c8 %c8
%y1 = OpBitFieldUExtract %uint %c_s1 %c8 %c8
%p1 = OpIMul %uint %x1 %y1
%t1 = OpIAdd %uint %t0 %p1

%x2 = OpBitFieldUExtract %uint %c_s0 %c16 %c8
%y2 = OpBitFieldUExtract %uint %c_s1 %c16 %c8
%p2 = OpIMul %uint %x2 %y2
%t2 = OpIAdd %uint %t1 %p2

%x3 = OpBitFieldUExtract %uint %c_s0 %c24 %c8
%y3 = OpBitFieldUExtract %uint %c_s1 %c24 %c8
%p3 = OpIMul %uint %x3 %y3
%t3 = OpIAdd %uint %t2 %p3

%p0_1 = OpIAdd %uint %p0 %p1
%p2_3 = OpIAdd %uint %p2 %p3
%prod_sum32 = OpIAdd %uint %p0_1 %p2_3
%prod_sum64 = OpUConvert %ulong %prod_sum32
%acc64 = OpUConvert %ulong %c_acc
{clamp_code}
%out_vec = OpCompositeConstruct %v4uint {res_var} {res_var} {res_var} {res_var}
OpStore %out_mrt0 %out_vec
OpReturn
OpFunctionEnd
"""


@contextlib.contextmanager
def patch_rdna2spirv_dot4acc_u8():
    """Context manager to temporarily hook rdna2spirv instruction generator with v_dot4acc_u32_u8."""
    orig_insn = rdna2spirv.Gen._insn
    orig_rbw = rdna2spirv.read_before_write_vgprs
    orig_rrw = rdna2spirv.reg_reads_writes

    def patched_insn(self, ins):
        op = ins.get("op", "")
        if op == "v_dot4acc_u32_u8":
            mods = ins.get("mods", set())
            if "clamp" in mods or any(m.startswith("op_sel") for m in mods):
                raise rdna2spirv.Unsupported(op + (" clamp" if "clamp" in mods else " op_sel"))
            a = ins["args"]
            x, y = self.src_u(a[1]), self.src_u(a[2])
            acc = self.src_u(a[3]) if len(a) > 3 else self.cu(0)
            for i in range(4):
                xb, yb, prod, total = (self.tmp() for _ in range(4))
                self.emit(f"{xb} = OpBitFieldUExtract %uint {x} {self.cu(8 * i)} {self.cu(8)}")
                self.emit(f"{yb} = OpBitFieldUExtract %uint {y} {self.cu(8 * i)} {self.cu(8)}")
                self.emit(f"{prod} = OpIMul %uint {xb} {yb}")
                self.emit(f"{total} = OpIAdd %uint {acc} {prod}")
                acc = total
            self.dst(a[0], acc)
            return
        return orig_insn(self, ins)

    rdna2spirv.Gen._insn = patched_insn
    try:
        yield
    finally:
        rdna2spirv.Gen._insn = orig_insn
        rdna2spirv.read_before_write_vgprs = orig_rbw
        rdna2spirv.reg_reads_writes = orig_rrw


class WavefrontDot4accInterpreter:
    """Simulates an RDNA2 SIMD wavefront executing VALU/SALU dot-product accumulation instructions.

    Models:
      - Wave32 (32 lanes) and Wave64 (64 lanes).
      - Per-lane VGPR file: vgprs[reg][lane].
      - Uniform SGPR file: sgprs[reg].
      - EXEC mask: 32-bit or 64-bit integer mask.
      - Instructions: v_mov_b32, v_mov_b32_e32, s_mov_b32, v_dot4acc_u32_u8 (with optional clamp).
    """

    def __init__(self, wave_size: int = 32):
        if wave_size not in (32, 64):
            raise ValueError(f"Unsupported wave_size {wave_size}; must be 32 or 64")
        self.wave_size = wave_size
        self.vgprs = [[0] * wave_size for _ in range(256)]
        self.sgprs = [0] * 106
        self.exec_mask = (1 << wave_size) - 1

    def set_vgpr(self, reg: int, lane: int, val: int) -> None:
        self.vgprs[reg][lane] = val & MASK32

    def get_vgpr(self, reg: int, lane: int) -> int:
        return self.vgprs[reg][lane] & MASK32

    def set_sgpr(self, reg: int, val: int) -> None:
        self.sgprs[reg] = val & MASK32

    def get_sgpr(self, reg: int) -> int:
        return self.sgprs[reg] & MASK32

    def set_exec(self, mask: int) -> None:
        self.exec_mask = mask & ((1 << self.wave_size) - 1)

    def parse_operand(self, token: str, lane: int) -> int:
        token = token.strip().rstrip(",")
        lower = token.lower()
        if lower.startswith("v") and lower[1:].isdigit():
            idx = int(lower[1:])
            if 0 <= idx < 256:
                return self.vgprs[idx][lane] & MASK32
            raise IndexError(f"VGPR index out of range: {token}")
        if lower.startswith("s") and lower[1:].isdigit():
            idx = int(lower[1:])
            if 0 <= idx < 106:
                return self.sgprs[idx] & MASK32
            raise IndexError(f"SGPR index out of range: {token}")
        if lower.startswith("0x") or lower.startswith("-0x"):
            return int(token, 16) & MASK32
        return int(token) & MASK32

    def execute_line(self, line: str) -> None:
        line = line.split(";", 1)[0].split("//", 1)[0].strip()
        if not line:
            return
        parts = line.split(None, 1)
        opcode = parts[0].lower()
        args_str = parts[1] if len(parts) > 1 else ""

        # Check for clamp modifier
        clamp = False
        raw_tokens = [t.strip() for t in args_str.split(",") if t.strip()]
        if raw_tokens and "clamp" in raw_tokens[-1].lower():
            clamp = True
            last = raw_tokens[-1].lower().replace("clamp", "").strip()
            if last:
                raw_tokens[-1] = last
            else:
                raw_tokens.pop()

        tokens = raw_tokens

        if opcode == "s_mov_b32":
            dst = tokens[0].lower()
            if not (dst.startswith("s") and dst[1:].isdigit()):
                raise ValueError(f"s_mov_b32 dst must be SGPR, got {tokens[0]}")
            s_idx = int(dst[1:])
            val = self.parse_operand(tokens[1], 0)
            self.sgprs[s_idx] = val

        elif opcode in ("v_mov_b32", "v_mov_b32_e32"):
            dst = tokens[0].lower()
            if not (dst.startswith("v") and dst[1:].isdigit()):
                raise ValueError(f"v_mov_b32 dst must be VGPR, got {tokens[0]}")
            v_idx = int(dst[1:])
            for lane in range(self.wave_size):
                if (self.exec_mask >> lane) & 1:
                    val = self.parse_operand(tokens[1], lane)
                    self.vgprs[v_idx][lane] = val

        elif opcode == "v_dot4acc_u32_u8":
            dst = tokens[0].lower()
            if not (dst.startswith("v") and dst[1:].isdigit()):
                raise ValueError(f"v_dot4acc_u32_u8 dst must be VGPR, got {tokens[0]}")
            v_idx = int(dst[1:])
            for lane in range(self.wave_size):
                if (self.exec_mask >> lane) & 1:
                    s0 = self.parse_operand(tokens[1], lane)
                    s1 = self.parse_operand(tokens[2], lane)
                    s2 = self.parse_operand(tokens[3], lane) if len(tokens) > 3 else 0
                    if clamp:
                        self.vgprs[v_idx][lane] = ref_dot4acc_clamped_u32_u8(s0, s1, s2)
                    else:
                        self.vgprs[v_idx][lane] = ref_dot4acc_u32_u8(s0, s1, s2)

        elif opcode == "s_endpgm":
            return
        else:
            raise NotImplementedError(f"Unsupported opcode: {opcode}")

    def execute_program(self, program: str | list[str]) -> None:
        lines = program if isinstance(program, list) else program.strip().splitlines()
        for line in lines:
            self.execute_line(line)


class ReferenceOracleTests(unittest.TestCase):
    """Verifies reference oracle against hand-calculated test vectors and algebraic invariants."""

    def test_unpack_pack_u8_roundtrip(self) -> None:
        """Verifies byte unpacking and packing round-trip across distinct word patterns."""
        patterns = [
            0x00000000,
            0xFFFFFFFF,
            0x12345678,
            0xA55AA55A,
            0x01020304,
            0x80808080,
            0x7F8001FE,
            0xDEADBEEF,
        ]
        for val in patterns:
            with self.subTest(val=hex(val)):
                b0, b1, b2, b3 = unpack_u8(val)
                repacked = pack_u8(b0, b1, b2, b3)
                self.assertEqual(repacked, val)

    def test_known_test_vectors(self) -> None:
        """Verifies reference oracle reproduces all hand-calculated vectors."""
        for s0, s1, s2, expected in KNOWN_VECTORS:
            with self.subTest(s0=hex(s0), s1=hex(s1), s2=hex(s2)):
                got = ref_dot4acc_u32_u8(s0, s1, s2)
                self.assertEqual(got, expected, f"{s0:#x} . {s1:#x} + {s2:#x}")

    def test_commutativity_of_multipliers(self) -> None:
        """Multipliers are commutative: S0 . S1 + S2 == S1 . S0 + S2."""
        test_pairs = [
            (0x01020304, 0x05060708),
            (0xFFFFFFFF, 0x12345678),
            (0x80808080, 0x7F7F7F7F),
            (0xA1B2C3D4, 0x1E2D3C4B),
            (0x00FF00FF, 0xFF00FF00),
            (0xDEADBEEF, 0xCAFEBABE),
        ]
        for s0, s1 in test_pairs:
            for s2 in (0, 100, 0x12345678, 0xFFFFFFFF):
                with self.subTest(s0=hex(s0), s1=hex(s1), s2=hex(s2)):
                    self.assertEqual(
                        ref_dot4acc_u32_u8(s0, s1, s2),
                        ref_dot4acc_u32_u8(s1, s0, s2),
                    )

    def test_zero_multipliers_preserve_accumulator(self) -> None:
        """Zero multiplier on either operand preserves accumulator exactly."""
        vectors = [0x01020304, 0xFFFFFFFF, 0x80808080, 0x12345678, 0xCAFEBABE]
        for v in vectors:
            for acc in (0, 1, 0x7FFFFFFF, 0xFFFFFFFF, 0x10000000):
                with self.subTest(v=hex(v), acc=hex(acc)):
                    self.assertEqual(ref_dot4acc_u32_u8(0, v, acc), acc & MASK32)
                    self.assertEqual(ref_dot4acc_u32_u8(v, 0, acc), acc & MASK32)

    def test_identity_multiplier_sums_bytes(self) -> None:
        """Dot product with broadcast 0x01010101 computes the sum of the four unsigned bytes."""
        test_words = [
            (0x01020304, 1 + 2 + 3 + 4),
            (0xFFFFFFFF, 255 * 4),
            (0x10203040, 0x10 + 0x20 + 0x30 + 0x40),
            (0x000000FF, 255),
            (0x0000FF00, 255),
            (0x00FF0000, 255),
            (0xFF000000, 255),
            (0x80808080, 128 * 4),
        ]
        for word, expected_sum in test_words:
            with self.subTest(word=hex(word)):
                self.assertEqual(ref_dot4acc_u32_u8(word, 0x01010101, 0), expected_sum)
                self.assertEqual(ref_dot4acc_u32_u8(0x01010101, word, 0), expected_sum)

    def test_max_single_step_sum(self) -> None:
        """Verifies maximum product sum per step is exactly 4 * (255 * 255) = 260100."""
        self.assertEqual(MAX_DOT4_SUM, 260100)
        self.assertEqual(MAX_DOT4_SUM, 0x3F804)
        got = ref_dot4_u32_u8_sum(0xFFFFFFFF, 0xFFFFFFFF)
        self.assertEqual(got, 260100)

    def test_byte_lane_isolation(self) -> None:
        """Each of the 4 byte lanes computes independently without cross-lane bit bleed."""
        for byte_idx in range(4):
            for val_a in (1, 15, 128, 255):
                for val_b in (1, 10, 100, 255):
                    s0 = val_a << (8 * byte_idx)
                    s1 = val_b << (8 * byte_idx)
                    expected = val_a * val_b
                    with self.subTest(lane=byte_idx, a=val_a, b=val_b):
                        self.assertEqual(ref_dot4acc_u32_u8(s0, s1, 0), expected)

    def test_accumulator_linearity(self) -> None:
        """ref_dot4acc_u32_u8(s0, s1, s2) == (ref_dot4acc_u32_u8(s0, s1, 0) + s2) mod 2^32."""
        s0, s1 = 0x11223344, 0x55667788
        base_dot = ref_dot4acc_u32_u8(s0, s1, 0)
        for addend in (0, 1, 0x1000, 0x7FFFFFFF, 0xFFFFFFFF, 0xFFFFFFFE):
            with self.subTest(addend=hex(addend)):
                want = (base_dot + addend) & MASK32
                got = ref_dot4acc_u32_u8(s0, s1, addend)
                self.assertEqual(got, want)

    def test_unsigned_vs_signed_fundamental_divergence(self) -> None:
        """Unsigned dot product behaves fundamentally differently from signed dot product."""
        divergent_pairs = [
            (0x000000FF, 0x00000001, 0),  # 255 * 1 = 255 vs -1 * 1 = -1
            (0x000000FF, 0x000000FF, 0),  # 255 * 255 = 65025 vs (-1)*(-1) = 1
            (0xFFFFFFFF, 0x01010101, 0),  # 4 * 255 = 1020 vs 4 * (-1) = -4
            (0x80808080, 0x7F7F7F7F, 0),  # 4 * (128*127) = 65024 vs 4 * (-128*127) = -65024
            (0xFFFFFFFF, 0xFFFFFFFF, 0),  # 4 * 65025 = 260100 vs 4 * 1 = 4
        ]
        for s0, s1, s2 in divergent_pairs:
            with self.subTest(s0=hex(s0), s1=hex(s1)):
                u_res = ref_dot4acc_u32_u8(s0, s1, s2)
                s_res = ref_dot4_i32_i8(s0, s1, s2)
                self.assertNotEqual(u_res, s_res)

    def test_pseudorandom_fuzz_invariants(self) -> None:
        """Deterministic fuzzing over 128 pseudo-random word triples checking invariants."""
        rng = random.Random(0x444F5441)  # ASCII 'DOTA'
        for _ in range(128):
            s0 = rng.getrandbits(32)
            s1 = rng.getrandbits(32)
            s2 = rng.getrandbits(32)

            # Commutativity
            self.assertEqual(ref_dot4acc_u32_u8(s0, s1, s2), ref_dot4acc_u32_u8(s1, s0, s2))

            # Manual byte arithmetic verification
            b0 = unpack_u8(s0)
            b1 = unpack_u8(s1)
            manual_sum = ((s2 & MASK32) + b0[0]*b1[0] + b0[1]*b1[1] + b0[2]*b1[2] + b0[3]*b1[3]) & MASK32
            self.assertEqual(ref_dot4acc_u32_u8(s0, s1, s2), manual_sum)

            # Single-step dot4 sum bounded by MAX_DOT4_SUM
            prod_sum = ref_dot4_u32_u8_sum(s0, s1)
            self.assertGreaterEqual(prod_sum, 0)
            self.assertLessEqual(prod_sum, MAX_DOT4_SUM)


class AccumulatorWrappingAndClampingTests(unittest.TestCase):
    """Verifies modulo 2^32 wrapping, clamping saturation, and chaining sequences."""

    def test_exact_overflow_boundary_wraps_to_zero(self) -> None:
        """When S2 + sum equals 2^32 exactly, result wraps to 0 without saturation."""
        s0 = 0xFFFFFFFF
        s1 = 0xFFFFFFFF
        prod_sum = ref_dot4_u32_u8_sum(s0, s1)
        self.assertEqual(prod_sum, 260100)

        acc_exact = (MASK32 - prod_sum + 1) & MASK32
        res = ref_dot4acc_u32_u8(s0, s1, acc_exact)
        self.assertEqual(res, 0)

    def test_wrap_around_with_remainder(self) -> None:
        """Accumulator 0xFFFFFFFF + 260100 wraps modulo 2^32 to 260099."""
        s0 = 0xFFFFFFFF
        s1 = 0xFFFFFFFF
        res = ref_dot4acc_u32_u8(s0, s1, MASK32)
        self.assertEqual(res, 260099)

    def test_wrap_around_plus_one(self) -> None:
        """Accumulator 1 past exact boundary wraps to 1."""
        s0 = 0xFFFFFFFF
        s1 = 0xFFFFFFFF
        prod_sum = 260100
        acc = (MASK32 - prod_sum + 2) & MASK32
        res = ref_dot4acc_u32_u8(s0, s1, acc)
        self.assertEqual(res, 1)

    def test_clamping_saturates_at_max_uint32(self) -> None:
        """With clamp modifier, any sum exceeding 0xFFFFFFFF saturates to 0xFFFFFFFF."""
        s0 = 0xFFFFFFFF
        s1 = 0xFFFFFFFF
        self.assertEqual(ref_dot4acc_clamped_u32_u8(s0, s1, MASK32), MASK32)
        self.assertEqual(ref_dot4acc_clamped_u32_u8(s0, s1, MASK32 - 100), MASK32)

    def test_clamping_below_threshold_is_exact(self) -> None:
        """With clamp modifier, sums not exceeding 0xFFFFFFFF equal the unclamped sum."""
        s0 = 0x01020304
        s1 = 0x05060708
        acc = 0x10000000
        unclamped = ref_dot4acc_u32_u8(s0, s1, acc)
        clamped = ref_dot4acc_clamped_u32_u8(s0, s1, acc)
        self.assertEqual(clamped, unclamped)
        self.assertEqual(clamped, 0x10000000 + 70)

    def test_accumulator_chaining_long_sequence(self) -> None:
        """Chaining 64 consecutive dot product steps computes exact multi-step reduction."""
        current_acc = 0x00000100
        expected_total = current_acc
        for step in range(64):
            s0 = ((step + 1) * 0x01010101) & MASK32
            s1 = 0x02020202
            step_sum = ref_dot4_u32_u8_sum(s0, s1)
            expected_total = (expected_total + step_sum) & MASK32
            current_acc = ref_dot4acc_u32_u8(s0, s1, current_acc)
            self.assertEqual(current_acc, expected_total)

    def test_spirv_mini_interp_modulo_wrapping(self) -> None:
        """SPIR-V fragment shader in spirv_mini_interp exhibits exact modulo 2^32 wrapping."""
        s0 = 0xFFFFFFFF
        s1 = 0xFFFFFFFF
        acc_exact = (MASK32 - MAX_DOT4_SUM + 1) & MASK32
        spv = build_spirv_dot4acc_fragment(s0, s1, acc_exact, clamp=False)
        res = si.run_fragment(spv)
        self.assertEqual(list(res["%out_mrt0"]), [0] * 4)

        acc_wrap1 = (acc_exact + 1) & MASK32
        spv_wrap1 = build_spirv_dot4acc_fragment(s0, s1, acc_wrap1, clamp=False)
        res_wrap1 = si.run_fragment(spv_wrap1)
        self.assertEqual(list(res_wrap1["%out_mrt0"]), [1] * 4)

    def test_spirv_mini_interp_clamping_saturation(self) -> None:
        """SPIR-V fragment shader with clamping saturates to 0xFFFFFFFF on overflow in mini_interp."""
        s0 = 0xFFFFFFFF
        s1 = 0xFFFFFFFF
        spv_clamped = build_spirv_dot4acc_fragment(s0, s1, MASK32, clamp=True)
        res_clamped = si.run_fragment(spv_clamped)
        self.assertEqual(list(res_clamped["%out_mrt0"]), [MASK32] * 4)

    def test_translated_shader_execution_with_patch(self) -> None:
        """Pixel shaders translated via rdna2spirv execute correctly in mini_interp."""
        with patch_rdna2spirv_dot4acc_u8():
            for s0, s1, acc, expected in KNOWN_VECTORS[:8]:
                with self.subTest(s0=hex(s0), s1=hex(s1), acc=hex(acc)):
                    lines = [
                        f"v_mov_b32_e32 v1, 0x{s0:08x}",
                        f"v_mov_b32_e32 v2, 0x{s1:08x}",
                        f"v_mov_b32_e32 v3, 0x{acc:08x}",
                        "v_dot4acc_u32_u8 v8, v1, v2, v3",
                        "exp mrt0 v8, v8, v8, v8 done vm",
                        "s_endpgm",
                    ]
                    spv = support.translate_lines("ps_dot4acc_test", lines)
                    self.assertEqual(support.mrt0_words(spv), [expected] * 4)


class WavefrontSimdExecutionTests(unittest.TestCase):
    """Verifies SIMD multi-lane wavefront execution and accumulation under EXEC masks."""

    def test_wave32_uniform_execution(self) -> None:
        """32-lane SIMD execution under full EXEC mask with divergent per-lane data."""
        interp = WavefrontDot4accInterpreter(wave_size=32)
        for lane in range(32):
            s0 = ((lane + 1) * 0x01020304) & MASK32
            s1 = 0x05060708
            acc = lane * 10
            interp.set_vgpr(1, lane, s0)
            interp.set_vgpr(2, lane, s1)
            interp.set_vgpr(3, lane, acc)

        interp.execute_line("v_dot4acc_u32_u8 v0, v1, v2, v3")

        for lane in range(32):
            s0 = ((lane + 1) * 0x01020304) & MASK32
            s1 = 0x05060708
            acc = lane * 10
            expected = ref_dot4acc_u32_u8(s0, s1, acc)
            self.assertEqual(interp.get_vgpr(0, lane), expected)

    def test_wave64_uniform_execution(self) -> None:
        """64-lane SIMD execution covering the entire Wave64 execution space."""
        interp = WavefrontDot4accInterpreter(wave_size=64)
        for lane in range(64):
            s0 = (0x01010101 * (lane + 1)) & MASK32
            s1 = 0x02020202
            interp.set_vgpr(1, lane, s0)
            interp.set_vgpr(2, lane, s1)
            interp.set_vgpr(3, lane, 0x1000)

        interp.execute_line("v_dot4acc_u32_u8 v8, v1, v2, v3")

        for lane in range(64):
            s0 = (0x01010101 * (lane + 1)) & MASK32
            s1 = 0x02020202
            expected = ref_dot4acc_u32_u8(s0, s1, 0x1000)
            self.assertEqual(interp.get_vgpr(8, lane), expected)

    def test_simd_exec_mask_inactive_lanes_preserved(self) -> None:
        """Inactive lanes under partial EXEC mask must strictly preserve destination registers."""
        interp = WavefrontDot4accInterpreter(wave_size=32)
        sentinel = 0xDEADBEEF

        for lane in range(32):
            interp.set_vgpr(5, lane, sentinel)
            interp.set_vgpr(1, lane, 0x01020304)
            interp.set_vgpr(2, lane, 0x05060708)
            interp.set_vgpr(3, lane, 100)

        # Only low 16 lanes active
        interp.set_exec(0x0000FFFF)
        interp.execute_line("v_dot4acc_u32_u8 v5, v1, v2, v3")

        for lane in range(32):
            if lane < 16:
                self.assertEqual(interp.get_vgpr(5, lane), 170)
            else:
                self.assertEqual(interp.get_vgpr(5, lane), sentinel)

    def test_simd_alternating_even_odd_divergence(self) -> None:
        """Divergent execution between alternating even and odd lanes."""
        interp = WavefrontDot4accInterpreter(wave_size=32)
        for lane in range(32):
            interp.set_vgpr(10, lane, 0)
            interp.set_vgpr(1, lane, 0x01010101)
            interp.set_vgpr(2, lane, 0x0A0A0A0A)
            interp.set_vgpr(3, lane, 0x02020202)
            interp.set_vgpr(4, lane, 0x14141414)

        # Even lanes active
        interp.set_exec(0x55555555)
        interp.execute_line("v_dot4acc_u32_u8 v10, v1, v2, 0")

        # Odd lanes active
        interp.set_exec(0xAAAAAAAA)
        interp.execute_line("v_dot4acc_u32_u8 v10, v3, v4, 0")

        for lane in range(32):
            expected = 40 if (lane % 2 == 0) else 160
            self.assertEqual(interp.get_vgpr(10, lane), expected)

    def test_simd_single_lane_active(self) -> None:
        """Only a single lane active (lane 27), ensuring zero collateral writes."""
        interp = WavefrontDot4accInterpreter(wave_size=32)
        for lane in range(32):
            interp.set_vgpr(7, lane, 0xBAADF00D)
            interp.set_vgpr(1, lane, 0x01020304)
            interp.set_vgpr(2, lane, 0x05060708)

        target_lane = 27
        interp.set_exec(1 << target_lane)
        interp.execute_line("v_dot4acc_u32_u8 v7, v1, v2, 0")

        for lane in range(32):
            if lane == target_lane:
                self.assertEqual(interp.get_vgpr(7, lane), 70)
            else:
                self.assertEqual(interp.get_vgpr(7, lane), 0xBAADF00D)

    def test_simd_uniform_scalar_broadcast(self) -> None:
        """SGPR sources broadcast uniform values across all lanes."""
        interp = WavefrontDot4accInterpreter(wave_size=32)
        interp.set_sgpr(4, 0x05060708)
        interp.set_sgpr(5, 5000)

        for lane in range(32):
            interp.set_vgpr(1, lane, 0x01020304)

        interp.execute_line("v_dot4acc_u32_u8 v0, v1, s4, s5")

        for lane in range(32):
            self.assertEqual(interp.get_vgpr(0, lane), 5070)

    def test_simd_accumulator_chain_loop(self) -> None:
        """Simulate a vector reduction accumulating dot products across loop iterations."""
        interp = WavefrontDot4accInterpreter(wave_size=32)
        running_totals = [0] * 32
        steps = [
            (0x01010101, 0x02020202),  # 4 * 2 = 8
            (0x03030303, 0x04040404),  # 4 * 12 = 48
            (0x05050505, 0x06060606),  # 4 * 30 = 120
            (0x07070707, 0x08080808),  # 4 * 56 = 224
        ]
        for s0, s1 in steps:
            for lane in range(32):
                interp.set_vgpr(1, lane, s0)
                interp.set_vgpr(2, lane, s1)
                running_totals[lane] = (running_totals[lane] + ref_dot4_u32_u8_sum(s0, s1)) & MASK32
            interp.execute_line("v_dot4acc_u32_u8 v0, v1, v2, v0")

        for lane in range(32):
            self.assertEqual(interp.get_vgpr(0, lane), running_totals[lane])
            self.assertEqual(interp.get_vgpr(0, lane), 8 + 48 + 120 + 224)

    def test_simd_clamping_under_divergent_lanes(self) -> None:
        """Clamping modifier saturates overflowing lanes while preserving non-overflowing lanes."""
        interp = WavefrontDot4accInterpreter(wave_size=32)
        for lane in range(32):
            if lane % 2 == 0:
                # Overflowing lane
                interp.set_vgpr(1, lane, 0xFFFFFFFF)
                interp.set_vgpr(2, lane, 0xFFFFFFFF)
                interp.set_vgpr(3, lane, MASK32)
            else:
                # Non-overflowing lane
                interp.set_vgpr(1, lane, 0x01020304)
                interp.set_vgpr(2, lane, 0x05060708)
                interp.set_vgpr(3, lane, 100)

        interp.execute_line("v_dot4acc_u32_u8 v0, v1, v2, v3 clamp")

        for lane in range(32):
            if lane % 2 == 0:
                self.assertEqual(interp.get_vgpr(0, lane), MASK32)
            else:
                self.assertEqual(interp.get_vgpr(0, lane), 170)


class EdgeCaseBytePatternsTests(unittest.TestCase):
    """Verifies edge-case byte patterns, walking bits, alternating patterns, and sensitivity."""

    def test_walking_ones_patterns(self) -> None:
        """Walking 1-bit across all 8 bit positions in each of the 4 bytes."""
        for byte_idx in range(4):
            for bit in range(8):
                val = (1 << bit) << (8 * byte_idx)
                multiplier = 1 << (8 * byte_idx)
                expected = 1 << bit
                self.assertEqual(ref_dot4acc_u32_u8(val, multiplier, 0), expected)

    def test_alternating_chessboards(self) -> None:
        """Alternating bit patterns across bytes (0xAA55AA55, 0x55AA55AA, 0x00FF00FF, 0xFF00FF00)."""
        pat_a = 0xAA55AA55
        pat_b = 0x55AA55AA
        # 0x55 = 85, 0xAA = 170. 85 * 170 = 14450. 4 * 14450 = 57800.
        self.assertEqual(ref_dot4acc_u32_u8(pat_a, pat_b, 0), 57800)
        self.assertEqual(ref_dot4acc_u32_u8(pat_a, pat_b, 1000), 58800)

        # 0x00FF00FF dot 0xFF00FF00: byte products are 0*255 + 255*0 + 0*255 + 255*0 = 0
        self.assertEqual(ref_dot4acc_u32_u8(0x00FF00FF, 0xFF00FF00, 0x12345678), 0x12345678)

        # 0x0F0F0F0F dot 0xF0F0F0F0: 15 * 240 = 3600. 4 * 3600 = 14400.
        self.assertEqual(ref_dot4acc_u32_u8(0x0F0F0F0F, 0xF0F0F0F0, 0), 14400)

    def test_symmetrical_byte_permutations(self) -> None:
        """Reversing byte order and cross-multiplying."""
        s0 = pack_u8(10, 20, 30, 40)
        s1 = pack_u8(40, 30, 20, 10)
        # 10*40 + 20*30 + 30*20 + 40*10 = 400 + 600 + 600 + 400 = 2000
        self.assertEqual(ref_dot4acc_u32_u8(s0, s1, 0), 2000)

    def test_single_bit_flip_sensitivity(self) -> None:
        """Flipping any bit in S0 or S1 alters the dot product when multiplier is non-zero."""
        s0 = 0x01020304
        s1 = 0x05060708
        acc = 1000
        golden = ref_dot4acc_u32_u8(s0, s1, acc)

        for bit in range(32):
            corrupted_s0 = s0 ^ (1 << bit)
            corrupted_s1 = s1 ^ (1 << bit)
            self.assertNotEqual(
                ref_dot4acc_u32_u8(corrupted_s0, s1, acc),
                golden,
                f"Flipping bit {bit} in S0 produced no change",
            )
            self.assertNotEqual(
                ref_dot4acc_u32_u8(s0, corrupted_s1, acc),
                golden,
                f"Flipping bit {bit} in S1 produced no change",
            )

    def test_all_byte_lane_powers_of_two(self) -> None:
        """Powers of two in each byte position produce exact power-of-two products."""
        for p0 in range(8):
            for p1 in range(8):
                v0 = (1 << p0) | ((1 << p0) << 8) | ((1 << p0) << 16) | ((1 << p0) << 24)
                v1 = (1 << p1) | ((1 << p1) << 8) | ((1 << p1) << 16) | ((1 << p1) << 24)
                expected = 4 * ((1 << p0) * (1 << p1))
                self.assertEqual(ref_dot4acc_u32_u8(v0, v1, 0), expected)


class OverflowMatrixTests(unittest.TestCase):
    """Verifies full combinatorial matrix of near-boundary accumulator values and products."""

    def test_comprehensive_overflow_matrix(self) -> None:
        """Cartesian product of representative accumulators and dot sums testing wrapping and clamping."""
        acc_values = [
            0,
            1,
            0x7FFFFFFF,
            0x80000000,
            (MASK32 - MAX_DOT4_SUM - 1) & MASK32,
            (MASK32 - MAX_DOT4_SUM) & MASK32,
            (MASK32 - MAX_DOT4_SUM + 1) & MASK32,
            MASK32 - 1,
            MASK32,
        ]

        sum_operands = [
            (0x00000000, 0x00000000, 0),
            (0x00000001, 0x00000001, 1),
            (0x01020304, 0x05060708, 70),
            (0x000000FF, 0x000000FF, 65025),
            (0x0000FFFF, 0x0000FFFF, 2 * 65025),
            (0x00FFFFFF, 0x00FFFFFF, 3 * 65025),
            (0xFFFFFFFF, 0xFFFFFFFF, 4 * 65025),  # 260100
        ]

        for s0, s1, expected_sum in sum_operands:
            self.assertEqual(ref_dot4_u32_u8_sum(s0, s1), expected_sum)
            for acc in acc_values:
                with self.subTest(sum=expected_sum, acc=hex(acc)):
                    unclamped = ref_dot4acc_u32_u8(s0, s1, acc)
                    clamped = ref_dot4acc_clamped_u32_u8(s0, s1, acc)
                    raw_total = acc + expected_sum

                    self.assertEqual(unclamped, raw_total & MASK32)

                    if raw_total > MASK32:
                        self.assertEqual(clamped, MASK32)
                        self.assertNotEqual(clamped, unclamped)
                    else:
                        self.assertEqual(clamped, raw_total)
                        self.assertEqual(clamped, unclamped)

    def test_clamp_vs_wrap_divergence_boundary(self) -> None:
        """Prove clamping matches wrapping iff total <= 0xFFFFFFFF, and diverges iff total > 0xFFFFFFFF."""
        s0 = 0x00000001
        s1 = 0x00000001
        acc_below = MASK32 - 2  # total = MASK32 - 1
        acc_exact = MASK32 - 1  # total = MASK32
        acc_above = MASK32      # total = MASK32 + 1 (overflow)

        self.assertEqual(ref_dot4acc_u32_u8(s0, s1, acc_below), ref_dot4acc_clamped_u32_u8(s0, s1, acc_below))
        self.assertEqual(ref_dot4acc_u32_u8(s0, s1, acc_exact), ref_dot4acc_clamped_u32_u8(s0, s1, acc_exact))

        self.assertEqual(ref_dot4acc_u32_u8(s0, s1, acc_above), 0)
        self.assertEqual(ref_dot4acc_clamped_u32_u8(s0, s1, acc_above), MASK32)

    def test_boundary_triplets_exact(self) -> None:
        """Exact triplets at 2^32 boundary for max dot4 sum (260100)."""
        s0 = 0xFFFFFFFF
        s1 = 0xFFFFFFFF
        k = MAX_DOT4_SUM

        # 1. Below boundary: sum is 2^32 - 1 (0xFFFFFFFF)
        acc_below = (MASK32 - k) & MASK32
        self.assertEqual(ref_dot4acc_u32_u8(s0, s1, acc_below), 0xFFFFFFFF)
        self.assertEqual(ref_dot4acc_clamped_u32_u8(s0, s1, acc_below), 0xFFFFFFFF)

        # 2. At boundary: sum is 2^32 -> wraps to 0; clamped to 0xFFFFFFFF
        acc_at = (MASK32 - k + 1) & MASK32
        self.assertEqual(ref_dot4acc_u32_u8(s0, s1, acc_at), 0)
        self.assertEqual(ref_dot4acc_clamped_u32_u8(s0, s1, acc_at), 0xFFFFFFFF)

        # 3. Above boundary: sum is 2^32 + 1 -> wraps to 1; clamped to 0xFFFFFFFF
        acc_above = (MASK32 - k + 2) & MASK32
        self.assertEqual(ref_dot4acc_u32_u8(s0, s1, acc_above), 1)
        self.assertEqual(ref_dot4acc_clamped_u32_u8(s0, s1, acc_above), 0xFFFFFFFF)


class NegativeControlAndFaultTests(unittest.TestCase):
    """Negative controls, translator modifier rejection, and fault injection verification."""

    def test_unpatched_rdna2spirv_raises_unsupported(self) -> None:
        """Without patch, standard rdna2spirv refuses v_dot4acc_u32_u8 cleanly."""
        lines = [
            "v_mov_b32_e32 v1, 1",
            "v_mov_b32_e32 v2, 1",
            "v_mov_b32_e32 v3, 0",
            "v_dot4acc_u32_u8 v8, v1, v2, v3",
            "exp mrt0 v8, v8, v8, v8 done vm",
            "s_endpgm",
        ]
        with self.assertRaises(rdna2spirv.Unsupported) as cm:
            support.translate_lines("ps_dot4acc_unpatched", lines)
        self.assertIn("v_dot4acc_u32_u8", str(cm.exception))

    def test_translator_clamp_refused(self) -> None:
        """The translator refuses clamp modifier on v_dot4acc_u32_u8 rather than guessing."""
        with patch_rdna2spirv_dot4acc_u8():
            lines = [
                "v_mov_b32_e32 v1, 1",
                "v_mov_b32_e32 v2, 1",
                "v_mov_b32_e32 v3, 0",
                "v_dot4acc_u32_u8 v8, v1, v2, v3 clamp",
                "exp mrt0 v8, v8, v8, v8 done vm",
                "s_endpgm",
            ]
            with self.assertRaises(rdna2spirv.Unsupported) as refused:
                support.translate_lines("ps_dot4acc_clamp", lines)
            self.assertEqual(str(refused.exception), "v_dot4acc_u32_u8 clamp")

    def test_translator_op_sel_refused(self) -> None:
        """The translator refuses op_sel modifier on v_dot4acc_u32_u8."""
        with patch_rdna2spirv_dot4acc_u8():
            lines = [
                "v_mov_b32_e32 v1, 1",
                "v_mov_b32_e32 v2, 1",
                "v_mov_b32_e32 v3, 0",
                "v_dot4acc_u32_u8 v8, v1, v2, v3 op_sel:[0,1,0] op_sel_hi:[1,0,1]",
                "exp mrt0 v8, v8, v8, v8 done vm",
                "s_endpgm",
            ]
            with self.assertRaises(rdna2spirv.Unsupported) as refused:
                support.translate_lines("ps_dot4acc_op_sel", lines)
            self.assertEqual(str(refused.exception), "v_dot4acc_u32_u8 op_sel")

    def test_interpreter_rejects_sgpr_as_destination(self) -> None:
        """VALU instructions cannot write to SGPRs as destinations."""
        interp = WavefrontDot4accInterpreter(wave_size=32)
        with self.assertRaises(ValueError):
            interp.execute_line("v_dot4acc_u32_u8 s0, v1, v2, v3")

    def test_interpreter_rejects_out_of_bounds_vgpr(self) -> None:
        """VGPR index >= 256 raises IndexError."""
        interp = WavefrontDot4accInterpreter(wave_size=32)
        with self.assertRaises(IndexError):
            interp.execute_line("v_dot4acc_u32_u8 v0, v256, v1, v2")

    def test_interpreter_rejects_out_of_bounds_sgpr(self) -> None:
        """SGPR index >= 106 raises IndexError."""
        interp = WavefrontDot4accInterpreter(wave_size=32)
        with self.assertRaises(IndexError):
            interp.execute_line("v_dot4acc_u32_u8 v0, v1, s120, v2")

    def test_interpreter_rejects_invalid_wave_size(self) -> None:
        """Wave sizes other than 32 or 64 are rejected."""
        with self.assertRaises(ValueError):
            WavefrontDot4accInterpreter(wave_size=16)
        with self.assertRaises(ValueError):
            WavefrontDot4accInterpreter(wave_size=128)

    def test_interpreter_rejects_unsupported_opcode(self) -> None:
        """Unknown opcodes raise NotImplementedError."""
        interp = WavefrontDot4accInterpreter(wave_size=32)
        with self.assertRaises(NotImplementedError):
            interp.execute_line("v_bogus_opcode v0, v1, v2")

    def test_negative_control_signed_oracle_mismatch(self) -> None:
        """Negative control: signed ref_dot4_i32_i8 fails against unsigned KNOWN_VECTORS."""
        mismatches = 0
        for s0, s1, s2, expected in KNOWN_VECTORS:
            signed_res = ref_dot4_i32_i8(s0, s1, s2)
            if signed_res != expected:
                mismatches += 1
        self.assertGreater(mismatches, len(KNOWN_VECTORS) // 2)

    def test_negative_control_inactive_lane_write_rejected(self) -> None:
        """Negative control: inactive lane output never equals active execution result."""
        interp = WavefrontDot4accInterpreter(wave_size=32)
        interp.set_exec(0x00000001)  # Only lane 0 active
        interp.set_vgpr(0, 1, 0x55555555)  # Lane 1 inactive
        interp.set_vgpr(1, 1, 0xFFFFFFFF)
        interp.set_vgpr(2, 1, 0xFFFFFFFF)
        interp.set_vgpr(3, 1, 0)

        interp.execute_line("v_dot4acc_u32_u8 v0, v1, v2, v3")
        self.assertEqual(interp.get_vgpr(0, 1), 0x55555555)
        self.assertNotEqual(interp.get_vgpr(0, 1), 260100)

    def test_operand_32bit_truncation(self) -> None:
        """Inputs exceeding 32-bit width are masked cleanly to 32 bits."""
        s0_wide = 0x1_000000FF  # Has bit 32 set
        s1 = 0x00000001
        self.assertEqual(ref_dot4acc_u32_u8(s0_wide, s1, 0), 255)


if __name__ == "__main__":
    unittest.main()
