from __future__ import annotations

"""Semantic interpreter tests for AMD RDNA2 packed 8-bit linear interpolation (v_lerp_u8).

AMD RDNA VALU Instruction Set Architecture (ISA):
  v_lerp_u8 dst, src0, src1, src2

Operation:
  Packs four 8-bit unsigned linear interpolations into a 32-bit vector register:
  For each byte lane i in {0, 1, 2, 3}:
    s0_i = (src0 >> (8 * i)) & 0xFF
    s1_i = (src1 >> (8 * i)) & 0xFF
    s2_i = (src2 >> (8 * i)) & 0x01   # only bit 0 of each byte lane contributes
    sum_i = s0_i + s1_i + s2_i        # range: 0 .. 511 (9 bits)
    dst_i = sum_i >> 1                # range: 0 .. 255 (8 bits)

Arithmetic Properties:
  - Range isolation: max sum 255 + 255 + 1 = 511. 511 >> 1 = 255 (0xFF).
    Result is strictly bound within [0, 255] for every byte lane. No cross-byte
    carry or overflow can occur.
  - Rounding control:
    - s2 bit 0 = 0: floor / truncation of the unsigned average.
    - s2 bit 0 = 1: ceiling / round-half-up of the unsigned average.
  - Identity property:
    - When s0_i == s1_i == k, ((k + k + r) >> 1) == (2k + r) >> 1 == k for both r in {0, 1}.
  - Symmetry / Commutativity:
    - src0 and src1 are interchangeable: v_lerp_u8(a, b, c) == v_lerp_u8(b, a, c).
  - Upper bit discard:
    - Bits [7:1] of each byte lane in src2 are strictly ignored by hardware and translator.

SIMD Multi-Lane Execution:
  RDNA2 executes VALU instructions in SIMD wavefronts of 32 lanes (Wave32) or 64
  lanes (Wave64). Each active lane executes packed byte interpolation independently.
  Inactive lanes (where bit `i` of the `EXEC` mask is 0) preserve their prior VGPR
  contents unmodified.

SPIR-V Lowering in rdna2spirv:
  Lowers to 4 parallel lane extractions via OpBitFieldUExtract (offset 0, 8, 16, 24 with
  widths 8 for src0/src1 and width 1 for src2), two OpIAdd operations, OpShiftRightLogical
  by 1, and OpBitwiseOr reassembly into a 32-bit dword.

Standard library only: Python 3.9+ compatible, zero game data, synthetic shaders only.
"""
import gc
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
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import rdna2spirv  # noqa: E402
import spirv_mini_interp as si  # noqa: E402

MASK8 = 0xFF
MASK32 = 0xFFFFFFFF
MASK64 = 0xFFFFFFFFFFFFFFFF
STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)

_TMP_DIR: tempfile.TemporaryDirectory | None = None
_SRC_PATH: str | None = None


def setUpModule() -> None:
    global _TMP_DIR, _SRC_PATH
    _TMP_DIR = tempfile.TemporaryDirectory()
    disasm_dir = os.path.join(_TMP_DIR.name, "disasm")
    os.mkdir(disasm_dir)
    _SRC_PATH = os.path.join(disasm_dir, "ps_lerp_u8.s")
    with open(os.path.join(_TMP_DIR.name, "ps_lerp_u8.dxbc"), "wb") as f:
        f.write(STUB_DXBC)


def tearDownModule() -> None:
    global _TMP_DIR, _SRC_PATH
    if _TMP_DIR is not None:
        gc.collect()
        _TMP_DIR.cleanup()
        _TMP_DIR = None
        _SRC_PATH = None


# ---------------------------------------------------------------------------
# Reference Oracles & Byte Packing Primitives
# ---------------------------------------------------------------------------

def unpack_bytes_u32(val: int) -> tuple[int, int, int, int]:
    """Unpack a 32-bit unsigned word into 4 byte lanes (b0, b1, b2, b3)."""
    val &= MASK32
    return (
        val & MASK8,
        (val >> 8) & MASK8,
        (val >> 16) & MASK8,
        (val >> 24) & MASK8,
    )


def pack_bytes_u32(b0: int, b1: int, b2: int, b3: int) -> int:
    """Pack 4 unsigned 8-bit bytes into a 32-bit unsigned word."""
    return (
        (b0 & MASK8)
        | ((b1 & MASK8) << 8)
        | ((b2 & MASK8) << 16)
        | ((b3 & MASK8) << 24)
    )


def ref_lerp_u8_byte(s0: int, s1: int, s2_round: int) -> int:
    """Calculate single byte lane lerp: ((s0 + s1 + (s2 & 1)) >> 1) & 0xFF.

    The intermediate sum ranges from 0 to 511. Right shifting by 1 produces
    a value in [0, 255] with no carry out of the 8-bit boundary.
    """
    b0 = s0 & MASK8
    b1 = s1 & MASK8
    r = s2_round & 1
    return (b0 + b1 + r) >> 1


def ref_lerp_u8_word(s0: int, s1: int, s2: int) -> int:
    """Full 32-bit packed word reference oracle for v_lerp_u8 across all 4 lanes."""
    b0_0, b0_1, b0_2, b0_3 = unpack_bytes_u32(s0)
    b1_0, b1_1, b1_2, b1_3 = unpack_bytes_u32(s1)
    b2_0, b2_1, b2_2, b2_3 = unpack_bytes_u32(s2)

    out0 = ref_lerp_u8_byte(b0_0, b1_0, b2_0)
    out1 = ref_lerp_u8_byte(b0_1, b1_1, b2_1)
    out2 = ref_lerp_u8_byte(b0_2, b1_2, b2_2)
    out3 = ref_lerp_u8_byte(b0_3, b1_3, b2_3)

    return pack_bytes_u32(out0, out1, out2, out3)


def ref(x: int, y: int, z: int) -> int:
    """Compatibility alias matching the existing test suite signature."""
    return ref_lerp_u8_word(x, y, z)


def ref_lerp_fixed_point_blend(s0: int, s1: int, round_nearest: bool = True) -> int:
    """Alternative formulation modeling fixed-point average with round-half-up bias."""
    r_word = 0x01010101 if round_nearest else 0
    return ref_lerp_u8_word(s0, s1, r_word)


# ---------------------------------------------------------------------------
# SIMD Wave Multi-Lane Reference Interpreter
# ---------------------------------------------------------------------------

class SimdWaveLerpInterpreter:
    """Multi-lane SIMD wave reference interpreter modeling RDNA2 Wave32/Wave64 execution.

    Models:
      - Wave sizes: Wave32 (32 lanes) and Wave64 (64 lanes)
      - Per-lane VGPR state: 256 vector registers per SIMD lane
      - Uniform SGPR state: 106 scalar registers shared across the wavefront
      - 64-bit EXEC mask: controls active vs inactive lanes
      - Inactive lane preservation: disabled lanes retain previous VGPR contents
      - Instruction execution: v_lerp_u8, v_mov_b32, s_mov_b32, s_mov_b64
    """

    def __init__(self, wave_size: int = 64) -> None:
        if wave_size not in (32, 64):
            raise ValueError(f"Unsupported wave size {wave_size}; must be 32 or 64")
        self.wave_size = wave_size
        self.vgprs: list[list[int]] = [[0] * 256 for _ in range(wave_size)]
        self.sgprs: list[int] = [0] * 106
        self.exec_mask: int = (1 << wave_size) - 1

    def set_exec_mask(self, mask: int) -> None:
        """Set the active lane mask."""
        limit = (1 << self.wave_size) - 1
        self.exec_mask = mask & limit

    def get_exec_mask(self) -> int:
        return self.exec_mask

    def is_lane_active(self, lane: int) -> bool:
        return bool((self.exec_mask >> lane) & 1)

    def set_lane_vgpr(self, lane: int, reg: int, val: int) -> None:
        self.vgprs[lane][reg] = val & MASK32

    def get_lane_vgpr(self, lane: int, reg: int) -> int:
        return self.vgprs[lane][reg] & MASK32

    def set_sgpr(self, reg: int, val: int) -> None:
        self.sgprs[reg] = val & MASK32

    def get_sgpr(self, reg: int) -> int:
        return self.sgprs[reg] & MASK32

    def _resolve_operand(self, op: str | int, lane: int) -> int:
        """Resolve a register operand or immediate literal for a given SIMD lane."""
        if isinstance(op, int):
            return op & MASK32
        op_str = op.strip().lower()
        if op_str.startswith("v"):
            reg_idx = int(op_str[1:])
            return self.vgprs[lane][reg_idx] & MASK32
        if op_str.startswith("s"):
            reg_idx = int(op_str[1:])
            return self.sgprs[reg_idx] & MASK32
        return int(op_str, 0) & MASK32

    def v_lerp_u8(self, vdst: int, src0: str | int, src1: str | int, src2: str | int) -> None:
        """Execute packed 8-bit linear interpolation across all active SIMD lanes."""
        for lane in range(self.wave_size):
            if (self.exec_mask >> lane) & 1:
                v0 = self._resolve_operand(src0, lane)
                v1 = self._resolve_operand(src1, lane)
                v2 = self._resolve_operand(src2, lane)
                self.vgprs[lane][vdst] = ref_lerp_u8_word(v0, v1, v2)

    def v_mov_b32(self, vdst: int, src: str | int) -> None:
        """Execute 32-bit vector move across active lanes."""
        for lane in range(self.wave_size):
            if (self.exec_mask >> lane) & 1:
                self.vgprs[lane][vdst] = self._resolve_operand(src, lane)

    def s_mov_b32(self, sdst: int, src: str | int) -> None:
        """Execute uniform scalar move."""
        self.sgprs[sdst] = self._resolve_operand(src, 0)

    def s_mov_b64(self, sdst: int, val: int) -> None:
        """Execute 64-bit scalar move into a register pair."""
        self.sgprs[sdst] = val & MASK32
        self.sgprs[sdst + 1] = (val >> 32) & MASK32

    def run_instruction(self, line: str) -> None:
        """Parse and execute a single RDNA assembly instruction."""
        clean = line.split(";")[0].strip()
        if not clean:
            return
        parts = clean.replace(",", " ").split()
        op = parts[0].lower()
        args = parts[1:]

        if op == "v_lerp_u8":
            dst = int(args[0].lower().replace("v", ""))
            self.v_lerp_u8(dst, args[1], args[2], args[3])
        elif op == "v_mov_b32":
            dst = int(args[0].lower().replace("v", ""))
            self.v_mov_b32(dst, args[1])
        elif op == "s_mov_b32":
            dst_str = args[0].lower()
            if dst_str == "exec_lo":
                val = self._resolve_operand(args[1], 0)
                self.set_exec_mask((self.exec_mask & ~0xFFFFFFFF) | (val & 0xFFFFFFFF))
            elif dst_str == "exec_hi":
                val = self._resolve_operand(args[1], 0)
                self.set_exec_mask((self.exec_mask & 0xFFFFFFFF) | ((val & 0xFFFFFFFF) << 32))
            else:
                dst = int(dst_str.replace("s", ""))
                self.s_mov_b32(dst, args[1])
        elif op == "s_mov_b64":
            dst_str = args[0].lower()
            val = int(args[1], 0) if args[1].startswith("0x") or args[1].isdigit() or (args[1].startswith("-") and args[1][1:].isdigit()) else self._resolve_operand(args[1], 0)
            if dst_str == "exec":
                self.set_exec_mask(val)
            else:
                dst = int(dst_str.replace("s[", "").replace("]", "").split(":")[0].replace("s", ""))
                self.s_mov_b64(dst, val)
        else:
            raise NotImplementedError(f"Instruction {op} not implemented in wave interpreter")

    def run_program(self, lines: list[str]) -> None:
        """Execute a sequence of assembly instructions in order."""
        for line in lines:
            self.run_instruction(line)


# ---------------------------------------------------------------------------
# Synthetic Shader Execution Harness
# ---------------------------------------------------------------------------

VALUES = [
    0,
    0xFFFFFFFF,
    0x01010101,
    0xFF00FF00,
    0x00FF00FF,
    0x80808080,
    0x7F7F7F7F,
    0x12345678,
    0xFEDCBA98,
]
ROUND = [0, 0xFFFFFFFF, 0x01010101, 0x01000100, 0xFEFEFEFE]


def run_lines(lines: list[str]) -> int:
    """Run assembly lines, exporting v0 through mrt0 as two 16-bit halves."""
    if _TMP_DIR is None or _SRC_PATH is None:
        setUpModule()
    assert _SRC_PATH is not None

    augmented = list(lines) + [
        "v_and_b32 v5, 0xffff, v0",
        "v_lshrrev_b32 v6, 16, v0",
        "v_or_b32 v5, 0x40000000, v5",
        "v_or_b32 v6, 0x40000000, v6",
    ]
    asm = (
        "; test_lerp_u8\n"
        + "".join(f"\t{ln}\n" for ln in augmented)
        + "\texp mrt0 v5, v6, v5, v5 done vm\n\ts_endpgm\n"
    )
    with open(_SRC_PATH, "w", encoding="utf-8") as f:
        f.write(asm)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)
        spv = rdna2spirv.translate(_SRC_PATH)
    lo, hi = si.run_fragment(spv)["%out_mrt0"][:2]
    assert lo >> 16 == 0x4000 and hi >> 16 == 0x4000, (hex(lo), hex(hi))
    return (lo & 0xFFFF) | ((hi & 0xFFFF) << 16)


def run_op(x: int, y: int, z: int) -> int:
    """Execute v_lerp_u8 on three 32-bit register inputs."""
    return run_lines([
        f"v_mov_b32 v1, {x:#x}",
        f"v_mov_b32 v2, {y:#x}",
        f"v_mov_b32 v3, {z:#x}",
        "v_lerp_u8 v0, v1, v2, v3",
    ])


# ---------------------------------------------------------------------------
# Commit 1 Tests: Reference Oracle and Mathematical Bounds
# ---------------------------------------------------------------------------

class TestLerpU8ReferenceOracle(unittest.TestCase):
    """Unit tests for the reference oracle, byte lane packing, and arithmetic bounds."""

    def test_unpack_and_pack_identity(self) -> None:
        """Unpacking and repacking 32-bit words must preserve all bits."""
        samples = [0, 0xFFFFFFFF, 0x12345678, 0x80706050, 0x00FF00FF, 0xAA55AA55]
        for val in samples:
            b0, b1, b2, b3 = unpack_bytes_u32(val)
            repacked = pack_bytes_u32(b0, b1, b2, b3)
            self.assertEqual(val, repacked)

    def test_byte_level_basic_arithmetic(self) -> None:
        """Single-byte lane calculations with floor and ceiling rounding."""
        # 0 + 0 + 0 >> 1 = 0
        self.assertEqual(ref_lerp_u8_byte(0, 0, 0), 0)
        self.assertEqual(ref_lerp_u8_byte(0, 0, 1), 0)

        # 1 + 2 = 3; 3 >> 1 = 1; (3 + 1) >> 1 = 2
        self.assertEqual(ref_lerp_u8_byte(1, 2, 0), 1)
        self.assertEqual(ref_lerp_u8_byte(1, 2, 1), 2)

        # 255 + 255 = 510; 510 >> 1 = 255; (510 + 1) >> 1 = 255
        self.assertEqual(ref_lerp_u8_byte(255, 255, 0), 255)
        self.assertEqual(ref_lerp_u8_byte(255, 255, 1), 255)

        # 254 + 255 = 509; 509 >> 1 = 254; (509 + 1) >> 1 = 255
        self.assertEqual(ref_lerp_u8_byte(254, 255, 0), 254)
        self.assertEqual(ref_lerp_u8_byte(254, 255, 1), 255)

        # 0 + 255 = 255; 255 >> 1 = 127; (255 + 1) >> 1 = 128
        self.assertEqual(ref_lerp_u8_byte(0, 255, 0), 127)
        self.assertEqual(ref_lerp_u8_byte(0, 255, 1), 128)

    def test_byte_identity_property(self) -> None:
        """When s0 == s1, the output must equal s0 regardless of the round bit."""
        for val in range(256):
            self.assertEqual(ref_lerp_u8_byte(val, val, 0), val)
            self.assertEqual(ref_lerp_u8_byte(val, val, 1), val)

    def test_byte_commutativity(self) -> None:
        """s0 and s1 operands must be commutative across all byte combinations."""
        test_samples = [0, 1, 2, 42, 127, 128, 254, 255]
        for a in test_samples:
            for b in test_samples:
                for r in (0, 1):
                    self.assertEqual(
                        ref_lerp_u8_byte(a, b, r),
                        ref_lerp_u8_byte(b, a, r),
                        f"Non-commutative at ({a}, {b}, {r})",
                    )

    def test_round_bit_behavior(self) -> None:
        """Round bit 1 increments output by exactly 1 when sum is odd, 0 when sum is even."""
        for a in range(256):
            for b in (0, 1, 127, 128, 254, 255):
                sum_ab = a + b
                res0 = ref_lerp_u8_byte(a, b, 0)
                res1 = ref_lerp_u8_byte(a, b, 1)
                if sum_ab % 2 == 1:
                    self.assertEqual(res1, res0 + 1)
                else:
                    self.assertEqual(res1, res0)

    def test_s2_upper_bits_ignored(self) -> None:
        """Bits 1..7 of each byte in src2 must have zero impact on the computation."""
        patterns = [0xFE, 0xFC, 0x80, 0xAA, 0x54, 0x02]
        for p in patterns:
            # When bit 0 is 0, any upper bits must still act as round 0
            self.assertEqual(ref_lerp_u8_byte(10, 20, p), ref_lerp_u8_byte(10, 20, 0))
            # When bit 0 is 1, any upper bits must still act as round 1
            self.assertEqual(ref_lerp_u8_byte(10, 20, p | 1), ref_lerp_u8_byte(10, 20, 1))

        # Word-level validation: masking S2 with 0x01010101 gives identical result
        s0 = 0x12345678
        s1 = 0x9ABCDEF0
        s2 = 0xFEDCBA98
        self.assertEqual(
            ref_lerp_u8_word(s0, s1, s2),
            ref_lerp_u8_word(s0, s1, s2 & 0x01010101),
        )

    def test_byte_lane_isolation_no_carry_leakage(self) -> None:
        """Carry out of lane i (up to 511) must not bleed into lane i+1."""
        # Lane 0 has 255 + 255 + 1 = 511 -> 255. Lane 1 has 0.
        s0 = 0x000000FF
        s1 = 0x000000FF
        s2 = 0x00000001
        res = ref_lerp_u8_word(s0, s1, s2)
        self.assertEqual(res & 0xFF, 0xFF)
        self.assertEqual((res >> 8) & 0xFF, 0x00)

        # Lane 1 has max sum, Lane 2 and Lane 0 have 0.
        s0 = 0x0000FF00
        s1 = 0x0000FF00
        s2 = 0x00000100
        res = ref_lerp_u8_word(s0, s1, s2)
        self.assertEqual(res & 0xFF, 0x00)
        self.assertEqual((res >> 8) & 0xFF, 0xFF)
        self.assertEqual((res >> 16) & 0xFF, 0x00)

        # Lane 2 has max sum.
        s0 = 0x00FF0000
        s1 = 0x00FF0000
        s2 = 0x00010000
        res = ref_lerp_u8_word(s0, s1, s2)
        self.assertEqual((res >> 8) & 0xFF, 0x00)
        self.assertEqual((res >> 16) & 0xFF, 0xFF)
        self.assertEqual((res >> 24) & 0xFF, 0x00)

        # Lane 3 has max sum.
        s0 = 0xFF000000
        s1 = 0xFF000000
        s2 = 0x01000000
        res = ref_lerp_u8_word(s0, s1, s2)
        self.assertEqual((res >> 16) & 0xFF, 0x00)
        self.assertEqual((res >> 24) & 0xFF, 0xFF)


# ---------------------------------------------------------------------------
# Commit 2 Tests: Multi-Lane SIMD Wave Interpreter
# ---------------------------------------------------------------------------

class TestSimdWaveLerpInterpreter(unittest.TestCase):
    """Verifies SIMD wave execution, lane divergence, and inactive lane preservation under EXEC mask."""

    def test_wave64_uniform_execution(self) -> None:
        """All 64 lanes active with uniform operands produce identical results."""
        wave = SimdWaveLerpInterpreter(wave_size=64)
        wave.set_exec_mask(MASK64)

        s0 = 0x80402010
        s1 = 0x20406080
        s2 = 0x01000100

        expected = ref_lerp_u8_word(s0, s1, s2)

        for lane in range(64):
            wave.set_lane_vgpr(lane, 1, s0)
            wave.set_lane_vgpr(lane, 2, s1)
            wave.set_lane_vgpr(lane, 3, s2)

        wave.v_lerp_u8(vdst=0, src0="v1", src1="v2", src2="v3")

        for lane in range(64):
            self.assertEqual(wave.get_lane_vgpr(lane, 0), expected)

    def test_wave32_divergent_operands(self) -> None:
        """Wave32 with unique operands per lane produces lane-isolated outputs."""
        wave = SimdWaveLerpInterpreter(wave_size=32)
        wave.set_exec_mask(0xFFFFFFFF)

        expected_results = []
        for lane in range(32):
            s0 = pack_bytes_u32(lane * 4, (lane * 7) & 0xFF, 128, 255 - lane)
            s1 = pack_bytes_u32(255 - lane * 4, lane, (lane * 3) & 0xFF, lane)
            s2 = pack_bytes_u32(lane & 1, (lane >> 1) & 1, 1, 0)

            wave.set_lane_vgpr(lane, 1, s0)
            wave.set_lane_vgpr(lane, 2, s1)
            wave.set_lane_vgpr(lane, 3, s2)
            expected_results.append(ref_lerp_u8_word(s0, s1, s2))

        wave.v_lerp_u8(vdst=0, src0="v1", src1="v2", src2="v3")

        for lane in range(32):
            self.assertEqual(wave.get_lane_vgpr(lane, 0), expected_results[lane])

    def test_wave64_exec_mask_inactive_lanes_preserved(self) -> None:
        """Inactive lanes under EXEC mask must retain their previous VGPR contents."""
        wave = SimdWaveLerpInterpreter(wave_size=64)
        canary = 0xDEADBEEF

        # Pre-populate all lanes with canary in destination v0
        for lane in range(64):
            wave.set_lane_vgpr(lane, 0, canary)
            wave.set_lane_vgpr(lane, 1, 0x11223344)
            wave.set_lane_vgpr(lane, 2, 0x55667788)
            wave.set_lane_vgpr(lane, 3, 0x01010101)

        # Only lower 32 lanes active: EXEC = 0x00000000FFFFFFFF
        active_mask = 0xFFFFFFFF
        wave.set_exec_mask(active_mask)

        wave.v_lerp_u8(vdst=0, src0="v1", src1="v2", src2="v3")

        expected_active = ref_lerp_u8_word(0x11223344, 0x55667788, 0x01010101)

        # Active lanes must be updated
        for lane in range(32):
            self.assertEqual(wave.get_lane_vgpr(lane, 0), expected_active)

        # Inactive lanes must be unmodified
        for lane in range(32, 64):
            self.assertEqual(wave.get_lane_vgpr(lane, 0), canary)

    def test_wave32_checkerboard_divergence(self) -> None:
        """Checkerboard active mask (even lanes active, odd inactive)."""
        wave = SimdWaveLerpInterpreter(wave_size=32)
        canary = 0xCAFEBABE

        for lane in range(32):
            wave.set_lane_vgpr(lane, 0, canary)
            wave.set_lane_vgpr(lane, 1, 0x00FF00FF)
            wave.set_lane_vgpr(lane, 2, 0xFF00FF00)
            wave.set_lane_vgpr(lane, 3, 0)

        # Even lanes active: 0x55555555
        checkerboard = 0x55555555
        wave.set_exec_mask(checkerboard)

        wave.v_lerp_u8(vdst=0, src0="v1", src1="v2", src2="v3")

        expected_active = ref_lerp_u8_word(0x00FF00FF, 0xFF00FF00, 0)

        for lane in range(32):
            if lane % 2 == 0:
                self.assertEqual(wave.get_lane_vgpr(lane, 0), expected_active)
            else:
                self.assertEqual(wave.get_lane_vgpr(lane, 0), canary)

    def test_wave64_sparse_exec_masks(self) -> None:
        """Sparse lane activation: only lane 0, lane 31, and lane 63 active."""
        wave = SimdWaveLerpInterpreter(wave_size=64)
        sparse_mask = (1 << 0) | (1 << 31) | (1 << 63)
        wave.set_exec_mask(sparse_mask)

        for lane in range(64):
            wave.set_lane_vgpr(lane, 0, 0x12340000 | lane)
            wave.set_lane_vgpr(lane, 1, 0x10101010)
            wave.set_lane_vgpr(lane, 2, 0x20202020)
            wave.set_lane_vgpr(lane, 3, 0)

        wave.v_lerp_u8(vdst=0, src0="v1", src1="v2", src2="v3")

        expected_val = ref_lerp_u8_word(0x10101010, 0x20202020, 0)
        for lane in range(64):
            if lane in (0, 31, 63):
                self.assertEqual(wave.get_lane_vgpr(lane, 0), expected_val)
            else:
                self.assertEqual(wave.get_lane_vgpr(lane, 0), 0x12340000 | lane)

    def test_wave_all_lanes_disabled(self) -> None:
        """When EXEC is 0, no VGPR in any lane is modified."""
        wave = SimdWaveLerpInterpreter(wave_size=64)
        wave.set_exec_mask(0)

        for lane in range(64):
            wave.set_lane_vgpr(lane, 0, 0x55555555)

        wave.v_lerp_u8(vdst=0, src0=0xFFFFFFFF, src1=0xFFFFFFFF, src2=1)

        for lane in range(64):
            self.assertEqual(wave.get_lane_vgpr(lane, 0), 0x55555555)

    def test_wave_mixed_operands_sgpr_and_inline(self) -> None:
        """Operand resolution combining SGPR source, VGPR source, and inline constant."""
        wave = SimdWaveLerpInterpreter(wave_size=32)
        wave.set_exec_mask(0xFFFFFFFF)

        wave.set_sgpr(4, 0x40404040)
        for lane in range(32):
            wave.set_lane_vgpr(lane, 1, 0x60606060)

        # src0=s4, src1=v1, src2=inline constant 1
        wave.v_lerp_u8(vdst=0, src0="s4", src1="v1", src2=1)

        expected = ref_lerp_u8_word(0x40404040, 0x60606060, 1)
        for lane in range(32):
            self.assertEqual(wave.get_lane_vgpr(lane, 0), expected)

    def test_wave_program_execution(self) -> None:
        """Assembly program sequence execution on wave interpreter."""
        wave = SimdWaveLerpInterpreter(wave_size=32)
        program = [
            "s_mov_b32 s2, 0x10203040",
            "v_mov_b32 v1, 0x30405060",
            "v_lerp_u8 v0, s2, v1, 1",
        ]
        wave.run_program(program)
        expected = ref_lerp_u8_word(0x10203040, 0x30405060, 1)
        for lane in range(32):
            self.assertEqual(wave.get_lane_vgpr(lane, 0), expected)

    def test_hierarchical_interpolation_pipeline(self) -> None:
        """Hierarchical multi-stage bilinear-style byte interpolation pipeline."""
        wave = SimdWaveLerpInterpreter(wave_size=32)
        wave.set_exec_mask(0xFFFFFFFF)

        # Blend A and B into v3 with round 0
        # Blend C and D into v4 with round 0
        # Blend v3 and v4 into v0 with round 1
        va = 0x10203040
        vb = 0x50607080
        vc = 0x20304050
        vd = 0x60708090

        for lane in range(32):
            wave.set_lane_vgpr(lane, 1, va)
            wave.set_lane_vgpr(lane, 2, vb)
            wave.set_lane_vgpr(lane, 5, vc)
            wave.set_lane_vgpr(lane, 6, vd)

        wave.v_lerp_u8(vdst=3, src0="v1", src1="v2", src2=0)
        wave.v_lerp_u8(vdst=4, src0="v5", src1="v6", src2=0)
        wave.v_lerp_u8(vdst=0, src0="v3", src1="v4", src2=1)

        ab = ref_lerp_u8_word(va, vb, 0)
        cd = ref_lerp_u8_word(vc, vd, 0)
        expected_final = ref_lerp_u8_word(ab, cd, 1)

        for lane in range(32):
            self.assertEqual(wave.get_lane_vgpr(lane, 3), ab)
            self.assertEqual(wave.get_lane_vgpr(lane, 4), cd)
            self.assertEqual(wave.get_lane_vgpr(lane, 0), expected_final)


# ---------------------------------------------------------------------------
# Synthetic Shader Execution Tests
# ---------------------------------------------------------------------------

class TestLerpU8SyntheticShaderTranslation(unittest.TestCase):
    """End-to-end SPIR-V translation and spirv_mini_interp execution tests."""

    def test_reference_known_vectors(self) -> None:
        """Verify baseline test vectors against ref_lerp_u8_word."""
        self.assertEqual(ref_lerp_u8_word(0xFF, 0xFF, 1), 0xFF)
        self.assertEqual(ref_lerp_u8_word(0x01, 0x02, 0), 0x01)
        self.assertEqual(ref_lerp_u8_word(0x01, 0x02, 1), 0x02)
        self.assertEqual(ref_lerp_u8_word(0x00000300, 0x00000400, 0x00000100), 0x00000400)
        self.assertEqual(ref_lerp_u8_word(0x01, 0x02, 0xFE), 0x01)

    def test_edge_values(self) -> None:
        """Run grid of edge values through rdna2spirv and spirv_mini_interp."""
        for x in VALUES:
            for y in VALUES:
                for z in ROUND:
                    with self.subTest(x=hex(x), y=hex(y), z=hex(z)):
                        self.assertEqual(run_op(x, y, z), ref_lerp_u8_word(x, y, z))

    def test_seeded(self) -> None:
        """Randomized 32-bit word verification against spirv_mini_interp."""
        rng = random.Random(0x1E8)
        for _ in range(40):
            x, y, z = rng.getrandbits(32), rng.getrandbits(32), rng.getrandbits(32)
            with self.subTest(x=hex(x), y=hex(y), z=hex(z)):
                self.assertEqual(run_op(x, y, z), ref_lerp_u8_word(x, y, z))

    def test_scalar_and_inline_operands(self) -> None:
        """Exercise SGPR scalar source and inline constant operand forms."""
        lines = [
            "s_mov_b32 s2, 0x10203040",
            "v_mov_b32 v1, 0x30405060",
            "v_lerp_u8 v0, s2, v1, 1",
        ]
        self.assertEqual(run_lines(lines), ref_lerp_u8_word(0x10203040, 0x30405060, 1))

    def test_clamp_is_rejected_not_guessed(self) -> None:
        """VOP3 clamp modifier is unsupported on v_lerp_u8 and must raise Unsupported."""
        with self.assertRaises(rdna2spirv.Unsupported):
            run_lines(["v_mov_b32 v1, 1", "v_lerp_u8 v0, v1, v1, v1 clamp"])


if __name__ == "__main__":
    unittest.main()
