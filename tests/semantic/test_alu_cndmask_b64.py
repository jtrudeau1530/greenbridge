"""Semantic reference tests for RDNA2 64-bit vector conditional select (v_cndmask_b64).

RDNA2 Vector Arithmetic Logic Unit (VALU) architecture:
  In AMD RDNA2 (GFX10) instruction set architecture, vector registers (VGPRs) are 32-bit wide.
  Operations on 64-bit vector quantities (integers, double-precision floats, or 64-bit pointers)
  operate across aligned pairs of adjacent registers: v[d:d+1] from source pairs v[s0:s0+1] and
  v[s1:s1+1].

  v_cndmask_b64 conditionally selects 64-bit values per-lane based on a condition bit:
    For active lane i in the wavefront (0 <= i < WaveSize, WaveSize in {32, 64}):
      cond_bit = (condition_mask >> i) & 1
      dst[i]   = src1[i] if cond_bit != 0 else src0[i]

  Low-level dword decomposition:
    Because each 64-bit VGPR pair consists of low dword v[2k] and high dword v[2k+1]:
      dst_lo[i] = src1_lo[i] if cond_bit != 0 else src0_lo[i]
      dst_hi[i] = src1_hi[i] if cond_bit != 0 else src0_hi[i]
    Both halves are governed by the exact same per-lane condition bit without divergence between
    the low and high dwords.

Standard library only: Python 3.9+ compatible, zero game data, zero external dependencies.
"""
from __future__ import annotations

import math
import os
import random
import struct
import sys
import unittest

MASK32 = 0xFFFFFFFF
MASK64 = 0xFFFFFFFFFFFFFFFF


def u32(val: int) -> int:
    """Normalize integer to unsigned 32-bit range [0, 2^32 - 1]."""
    return val & MASK32


def u64(val: int) -> int:
    """Normalize integer to unsigned 64-bit range [0, 2^64 - 1]."""
    return val & MASK64


def s32(val: int) -> int:
    """Interpret a 32-bit unsigned word as a signed two's-complement integer."""
    val &= MASK32
    return val - (1 << 32) if (val & 0x80000000) else val


def s64(val: int) -> int:
    """Interpret a 64-bit unsigned word as a signed two's-complement integer."""
    val &= MASK64
    return val - (1 << 64) if (val & (1 << 63)) else val


def split_dwords(val64: int) -> tuple[int, int]:
    """Split an unsigned 64-bit integer into (low_dword, high_dword)."""
    val = val64 & MASK64
    return val & MASK32, (val >> 32) & MASK32


def join_dwords(lo: int, hi: int) -> int:
    """Join (low_dword, high_dword) into an unsigned 64-bit integer."""
    return ((hi & MASK32) << 32) | (lo & MASK32)


def f64_to_u64(val: float) -> int:
    """Pack an IEEE-754 double precision float into an unsigned 64-bit integer."""
    return struct.unpack("<Q", struct.pack("<d", val))[0]


def u64_to_f64(bits: int) -> float:
    """Unpack an unsigned 64-bit integer into an IEEE-754 double precision float."""
    return struct.unpack("<d", struct.pack("<Q", bits & MASK64))[0]


def ref_cndmask_b64(src0: int, src1: int, cond: int | bool) -> int:
    """Reference oracle for 64-bit conditional select (v_cndmask_b64) on a single lane.

    In RDNA2 ISA:
      D.u64 = cond ? S1.u64 : S0.u64
    Condition is true (non-zero) selects src1; false (zero) selects src0.
    """
    return (src1 if cond else src0) & MASK64


def ref_cndmask_b64_dwords(
    s0_lo: int, s0_hi: int, s1_lo: int, s1_hi: int, cond: int | bool
) -> tuple[int, int]:
    """Reference oracle for pair-wise 32-bit dword selection.

    Demonstrates that 64-bit vector conditional select operates identically to parallel
    32-bit conditional select on low and high dwords using the identical lane condition.
    """
    bit = 1 if cond else 0
    lo = (s1_lo if bit else s0_lo) & MASK32
    hi = (s1_hi if bit else s0_hi) & MASK32
    return lo, hi


def ref_cndmask_b64_decomposed(src0: int, src1: int, cond: int | bool) -> int:
    """Compute 64-bit conditional select by decomposing into 32-bit dwords and reassembling."""
    s0_lo, s0_hi = split_dwords(src0)
    s1_lo, s1_hi = split_dwords(src1)
    res_lo, res_hi = ref_cndmask_b64_dwords(s0_lo, s0_hi, s1_lo, s1_hi, cond)
    return join_dwords(res_lo, res_hi)


def ref_simd_cndmask_b64(
    src0_lanes: list[int],
    src1_lanes: list[int],
    mask: int,
    wave_size: int = 64,
    exec_mask: int | None = None,
    prev_dst: list[int] | None = None,
) -> list[int]:
    """Reference oracle for SIMD vector conditional select across a wavefront.

    Parameters:
      src0_lanes: 64-bit values for src0 across wave_size lanes.
      src1_lanes: 64-bit values for src1 across wave_size lanes.
      mask: condition bitmask (32-bit or 64-bit integer, bit i determines selection for lane i).
      wave_size: 32 (wave32) or 64 (wave64).
      exec_mask: optional execution mask; inactive lanes (exec_bit == 0) preserve prev_dst.
      prev_dst: previous destination values across lanes, preserved when lane is inactive.

    Returns:
      list of 64-bit unsigned integers representing the destination register pair across lanes.
    """
    if wave_size not in (32, 64):
        raise ValueError(f"Invalid wave_size: {wave_size}. Must be 32 or 64.")
    if len(src0_lanes) != wave_size or len(src1_lanes) != wave_size:
        raise ValueError(
            f"Expected {wave_size} lane inputs, got {len(src0_lanes)} and {len(src1_lanes)}"
        )
    if prev_dst is None:
        prev_dst = [0] * wave_size
    elif len(prev_dst) != wave_size:
        raise ValueError(f"Expected {wave_size} prev_dst entries, got {len(prev_dst)}")

    effective_exec = ((1 << wave_size) - 1) if exec_mask is None else (exec_mask & ((1 << wave_size) - 1))
    results: list[int] = []

    for i in range(wave_size):
        if not ((effective_exec >> i) & 1):
            results.append(prev_dst[i] & MASK64)
            continue
        cond_bit = (mask >> i) & 1
        val = ref_cndmask_b64(src0_lanes[i], src1_lanes[i], cond_bit)
        results.append(val)

    return results


class SIMDVectorCNDMask64Simulator:
    """Deterministic SIMD execution model for RDNA2 64-bit vector conditional select.

    Simulates VALU register state across a wavefront (32 or 64 lanes):
      - VGPR file: 256 vector registers, each lane holding a 32-bit unsigned dword.
      - SGPR file: 106 scalar registers, each holding a 32-bit scalar dword.
      - VCC: vector condition code (64-bit mask; in Wave32 only bits 0..31 are used).
      - EXEC: execution mask (64-bit mask; in Wave32 only bits 0..31 are used).
    """

    def __init__(self, wave_size: int = 64) -> None:
        if wave_size not in (32, 64):
            raise ValueError(f"Invalid wave_size: {wave_size}. Must be 32 or 64.")
        self.wave_size = wave_size
        self.num_vgprs = 256
        self.num_sgprs = 106
        self.vgpr = [[0] * self.wave_size for _ in range(self.num_vgprs)]
        self.sgpr = [0] * self.num_sgprs
        self.vcc = 0
        self.exec = (1 << wave_size) - 1

    def reset(self) -> None:
        """Reset all registers, VCC, and EXEC mask to defaults."""
        self.vgpr = [[0] * self.wave_size for _ in range(self.num_vgprs)]
        self.sgpr = [0] * self.num_sgprs
        self.vcc = 0
        self.exec = (1 << self.wave_size) - 1

    def get_vgpr_u32(self, reg: int, lane: int) -> int:
        if not (0 <= reg < self.num_vgprs):
            raise IndexError(f"VGPR index {reg} out of range [0, {self.num_vgprs})")
        if not (0 <= lane < self.wave_size):
            raise IndexError(f"Lane {lane} out of range [0, {self.wave_size})")
        return self.vgpr[reg][lane] & MASK32

    def set_vgpr_u32(self, reg: int, lane: int, val: int) -> None:
        if not (0 <= reg < self.num_vgprs):
            raise IndexError(f"VGPR index {reg} out of range [0, {self.num_vgprs})")
        if not (0 <= lane < self.wave_size):
            raise IndexError(f"Lane {lane} out of range [0, {self.wave_size})")
        self.vgpr[reg][lane] = val & MASK32

    def get_vgpr_u64(self, reg_base: int, lane: int) -> int:
        if not (0 <= reg_base < self.num_vgprs - 1):
            raise IndexError(f"64-bit VGPR base {reg_base} out of range [0, {self.num_vgprs - 1})")
        lo = self.get_vgpr_u32(reg_base, lane)
        hi = self.get_vgpr_u32(reg_base + 1, lane)
        return join_dwords(lo, hi)

    def set_vgpr_u64(self, reg_base: int, lane: int, val64: int) -> None:
        if not (0 <= reg_base < self.num_vgprs - 1):
            raise IndexError(f"64-bit VGPR base {reg_base} out of range [0, {self.num_vgprs - 1})")
        lo, hi = split_dwords(val64)
        self.set_vgpr_u32(reg_base, lane, lo)
        self.set_vgpr_u32(reg_base + 1, lane, hi)

    def load_vgpr_pair_wave(self, reg_base: int) -> list[int]:
        """Read 64-bit values across all lanes for a vector register pair."""
        return [self.get_vgpr_u64(reg_base, lane) for lane in range(self.wave_size)]

    def store_vgpr_pair_wave(self, reg_base: int, values: list[int]) -> None:
        """Write 64-bit values across all lanes for a vector register pair."""
        if len(values) != self.wave_size:
            raise ValueError(f"Expected {self.wave_size} values, got {len(values)}")
        for lane in range(self.wave_size):
            self.set_vgpr_u64(reg_base, lane, values[lane])

    def get_sgpr_u32(self, reg: int) -> int:
        if not (0 <= reg < self.num_sgprs):
            raise IndexError(f"SGPR index {reg} out of range [0, {self.num_sgprs})")
        return self.sgpr[reg] & MASK32

    def set_sgpr_u32(self, reg: int, val: int) -> None:
        if not (0 <= reg < self.num_sgprs):
            raise IndexError(f"SGPR index {reg} out of range [0, {self.num_sgprs})")
        self.sgpr[reg] = val & MASK32

    def get_sgpr_u64(self, reg_base: int) -> int:
        if not (0 <= reg_base < self.num_sgprs - 1):
            raise IndexError(f"64-bit SGPR base {reg_base} out of range [0, {self.num_sgprs - 1})")
        lo = self.get_sgpr_u32(reg_base)
        hi = self.get_sgpr_u32(reg_base + 1)
        return join_dwords(lo, hi)

    def set_sgpr_u64(self, reg_base: int, val64: int) -> None:
        if not (0 <= reg_base < self.num_sgprs - 1):
            raise IndexError(f"64-bit SGPR base {reg_base} out of range [0, {self.num_sgprs - 1})")
        lo, hi = split_dwords(val64)
        self.set_sgpr_u32(reg_base, lo)
        self.set_sgpr_u32(reg_base + 1, hi)

    def resolve_condition_mask(self, cond_src: str | int) -> int:
        """Resolve condition source operand into a bitmask for the current wavefront."""
        if isinstance(cond_src, int):
            return cond_src & ((1 << self.wave_size) - 1)
        if not isinstance(cond_src, str):
            raise TypeError(f"Invalid condition operand type: {type(cond_src)}")

        cond = cond_src.strip().lower()
        if cond in ("vcc", "vcc_lo"):
            if self.wave_size == 32:
                return self.vcc & MASK32
            return (self.vcc & MASK64) if cond == "vcc" else (self.vcc & MASK32)
        if cond == "vcc_hi":
            return (self.vcc >> 32) & MASK32
        if cond.startswith("s[") and cond.endswith("]"):
            inner = cond[2:-1]
            parts = inner.split(":")
            if len(parts) != 2:
                raise ValueError(f"Invalid SGPR pair descriptor: {cond_src}")
            base_reg = int(parts[0])
            end_reg = int(parts[1])
            if end_reg != base_reg + 1:
                raise ValueError(f"SGPR pair must be consecutive registers: {cond_src}")
            return self.get_sgpr_u64(base_reg) & ((1 << self.wave_size) - 1)
        if cond.startswith("s"):
            reg_idx = int(cond[1:])
            val = self.get_sgpr_u32(reg_idx)
            return val & ((1 << self.wave_size) - 1)
        raise ValueError(f"Unknown condition mask source: {cond_src}")

    def step_v_cndmask_b64(
        self,
        dst_base: int,
        src0_base: int,
        src1_base: int,
        cond_src: str | int = "vcc",
    ) -> list[int]:
        """Execute 64-bit conditional select across active lanes.

        Two-phase lockstep execution:
          Phase 1 (Read): reads all source operand dwords and evaluates condition bits.
          Phase 2 (Write): writes destination dwords only for lanes enabled by EXEC.
        """
        if not (0 <= dst_base < self.num_vgprs - 1):
            raise IndexError(f"dst_base {dst_base} out of range [0, {self.num_vgprs - 1})")
        if not (0 <= src0_base < self.num_vgprs - 1):
            raise IndexError(f"src0_base {src0_base} out of range [0, {self.num_vgprs - 1})")
        if not (0 <= src1_base < self.num_vgprs - 1):
            raise IndexError(f"src1_base {src1_base} out of range [0, {self.num_vgprs - 1})")

        mask = self.resolve_condition_mask(cond_src)
        active_exec = self.exec & ((1 << self.wave_size) - 1)

        # Phase 1: Read and evaluate
        computed: list[tuple[int, int] | None] = []
        for lane in range(self.wave_size):
            if not ((active_exec >> lane) & 1):
                computed.append(None)
                continue
            cond_bit = (mask >> lane) & 1
            s0_val = self.get_vgpr_u64(src0_base, lane)
            s1_val = self.get_vgpr_u64(src1_base, lane)
            res = ref_cndmask_b64(s0_val, s1_val, cond_bit)
            computed.append(split_dwords(res))

        # Phase 2: Write
        for lane in range(self.wave_size):
            dwords = computed[lane]
            if dwords is not None:
                lo, hi = dwords
                self.set_vgpr_u32(dst_base, lane, lo)
                self.set_vgpr_u32(dst_base + 1, lane, hi)

        return self.load_vgpr_pair_wave(dst_base)


class TestCndmaskB64ReferenceOracle(unittest.TestCase):
    """Unit tests for single-lane and SIMD 64-bit conditional select reference oracles."""

    def test_basic_truth_table(self) -> None:
        """Verify standard selector truth table: 0 selects src0, 1 selects src1."""
        val0 = 0x01234567_89ABCDEF
        val1 = 0xFEDCBA98_76543210
        self.assertEqual(ref_cndmask_b64(val0, val1, 0), val0)
        self.assertEqual(ref_cndmask_b64(val0, val1, False), val0)
        self.assertEqual(ref_cndmask_b64(val0, val1, 1), val1)
        self.assertEqual(ref_cndmask_b64(val0, val1, True), val1)
        self.assertEqual(ref_cndmask_b64(val0, val1, 42), val1)

    def test_64bit_distinct_dwords(self) -> None:
        """Test combinations where only the low, only the high, or both dwords differ."""
        cases = [
            # Only low dwords differ
            (0x11112222_33334444, 0x11112222_AAAABBBB),
            # Only high dwords differ
            (0xAAAA0000_12345678, 0xBBBB0000_12345678),
            # Both dwords differ completely
            (0x00000000_00000000, 0xFFFFFFFF_FFFFFFFF),
            (0x55555555_55555555, 0xAAAAAAAA_AAAAAAAA),
            (0x12345678_9ABCDEF0, 0x0FEDCBA9_87654321),
        ]
        for s0, s1 in cases:
            with self.subTest(s0=hex(s0), s1=hex(s1)):
                self.assertEqual(ref_cndmask_b64(s0, s1, 0), s0)
                self.assertEqual(ref_cndmask_b64(s0, s1, 1), s1)
                self.assertEqual(
                    ref_cndmask_b64_decomposed(s0, s1, 0),
                    ref_cndmask_b64(s0, s1, 0),
                )
                self.assertEqual(
                    ref_cndmask_b64_decomposed(s0, s1, 1),
                    ref_cndmask_b64(s0, s1, 1),
                )

    def test_edge_bit_patterns(self) -> None:
        """Test boundary conditions, sign bits (bit 63, bit 31), and extreme integer limits."""
        patterns = [
            0,
            1,
            MASK64,
            1 << 63,
            (1 << 63) - 1,
            1 << 31,
            (1 << 31) - 1,
            1 << 32,
            MASK32,
            0x7FFFFFFF_FFFFFFFF,
            0x80000000_00000000,
            0xFFFFFFFF_00000000,
            0x00000000_FFFFFFFF,
        ]
        for i, p0 in enumerate(patterns):
            for j, p1 in enumerate(patterns):
                self.assertEqual(ref_cndmask_b64(p0, p1, 0), p0)
                self.assertEqual(ref_cndmask_b64(p0, p1, 1), p1)

    def test_float64_special_bitcasts(self) -> None:
        """Test 64-bit conditional select on IEEE-754 double precision bit patterns."""
        f_values = [
            0.0,
            -0.0,
            1.0,
            -1.0,
            math.pi,
            math.e,
            float("inf"),
            float("-inf"),
            float("nan"),
            1e-308,
            1e308,
            sys.float_info.min,
            sys.float_info.max,
        ]
        u64_patterns = [f64_to_u64(f) for f in f_values]

        for u0 in u64_patterns:
            for u1 in u64_patterns:
                self.assertEqual(ref_cndmask_b64(u0, u1, 0), u0)
                self.assertEqual(ref_cndmask_b64(u0, u1, 1), u1)

    def test_algebraic_identities(self) -> None:
        """Verify algebraic identities of 64-bit conditional selection."""
        rng = random.Random(1337)
        for _ in range(50):
            x = rng.getrandbits(64)
            y = rng.getrandbits(64)

            # Identity: selecting between identical values returns the value
            self.assertEqual(ref_cndmask_b64(x, x, 0), x)
            self.assertEqual(ref_cndmask_b64(x, x, 1), x)

            # Inversion: swapping operands and inverting condition bit yields identical result
            self.assertEqual(ref_cndmask_b64(x, y, 0), ref_cndmask_b64(y, x, 1))
            self.assertEqual(ref_cndmask_b64(x, y, 1), ref_cndmask_b64(y, x, 0))

            # Idempotence: re-selecting result with same condition returns itself
            r0 = ref_cndmask_b64(x, y, 0)
            self.assertEqual(ref_cndmask_b64(r0, y, 0), r0)
            r1 = ref_cndmask_b64(x, y, 1)
            self.assertEqual(ref_cndmask_b64(x, r1, 1), r1)

    def test_dword_decomposition_equivalence(self) -> None:
        """Verify bit-exact match between full 64-bit select and dword-wise decomposition."""
        rng = random.Random(42)
        for _ in range(200):
            s0 = rng.getrandbits(64)
            s1 = rng.getrandbits(64)
            for c in (0, 1):
                want = ref_cndmask_b64(s0, s1, c)
                got = ref_cndmask_b64_decomposed(s0, s1, c)
                self.assertEqual(want, got)
                s0_lo, s0_hi = split_dwords(s0)
                s1_lo, s1_hi = split_dwords(s1)
                lo, hi = ref_cndmask_b64_dwords(s0_lo, s0_hi, s1_lo, s1_hi, c)
                self.assertEqual(lo, want & MASK32)
                self.assertEqual(hi, (want >> 32) & MASK32)

    def test_simd_oracle_wave32(self) -> None:
        """Validate SIMD reference oracle across 32 lanes."""
        rng = random.Random(32)
        s0_lanes = [rng.getrandbits(64) for _ in range(32)]
        s1_lanes = [rng.getrandbits(64) for _ in range(32)]
        mask = 0b10101010_11001100_11110000_01010101

        res = ref_simd_cndmask_b64(s0_lanes, s1_lanes, mask, wave_size=32)
        self.assertEqual(len(res), 32)
        for lane in range(32):
            bit = (mask >> lane) & 1
            expected = s1_lanes[lane] if bit else s0_lanes[lane]
            self.assertEqual(res[lane], expected)

    def test_simd_oracle_wave64(self) -> None:
        """Validate SIMD reference oracle across 64 lanes."""
        rng = random.Random(64)
        s0_lanes = [rng.getrandbits(64) for _ in range(64)]
        s1_lanes = [rng.getrandbits(64) for _ in range(64)]
        mask = 0xCAFEBABEDEADBEEF

        res = ref_simd_cndmask_b64(s0_lanes, s1_lanes, mask, wave_size=64)
        self.assertEqual(len(res), 64)
        for lane in range(64):
            bit = (mask >> lane) & 1
            expected = s1_lanes[lane] if bit else s0_lanes[lane]
            self.assertEqual(res[lane], expected)


class TestSIMDMultiLaneDivergenceAndMaskRouting(unittest.TestCase):
    """Test multi-lane divergence and VCC / SGPR mask routing on SIMD wavefronts."""

    def test_wave32_uniform_all_zeros_and_all_ones(self) -> None:
        """Verify uniform condition across 32 lanes (all 0s select src0, all 1s select src1)."""
        sim = SIMDVectorCNDMask64Simulator(wave_size=32)
        s0_vals = [0x11111111_00000000 + i for i in range(32)]
        s1_vals = [0x99999999_88888888 - i for i in range(32)]
        sim.store_vgpr_pair_wave(2, s0_vals)
        sim.store_vgpr_pair_wave(4, s1_vals)

        # All zeros condition
        sim.vcc = 0
        out_zeros = sim.step_v_cndmask_b64(0, 2, 4, cond_src="vcc")
        self.assertEqual(out_zeros, s0_vals)

        # All ones condition
        sim.vcc = 0xFFFFFFFF
        out_ones = sim.step_v_cndmask_b64(0, 2, 4, cond_src="vcc")
        self.assertEqual(out_ones, s1_vals)

    def test_wave64_uniform_all_zeros_and_all_ones(self) -> None:
        """Verify uniform condition across 64 lanes (all 0s select src0, all 1s select src1)."""
        sim = SIMDVectorCNDMask64Simulator(wave_size=64)
        s0_vals = [0x22222222_10000000 + i for i in range(64)]
        s1_vals = [0xEEEEEEEE_20000000 - i for i in range(64)]
        sim.store_vgpr_pair_wave(6, s0_vals)
        sim.store_vgpr_pair_wave(8, s1_vals)

        # All zeros
        sim.vcc = 0
        out_zeros = sim.step_v_cndmask_b64(4, 6, 8, cond_src="vcc")
        self.assertEqual(out_zeros, s0_vals)

        # All ones
        sim.vcc = MASK64
        out_ones = sim.step_v_cndmask_b64(4, 6, 8, cond_src="vcc")
        self.assertEqual(out_ones, s1_vals)

    def test_multi_lane_divergence_checkerboard_wave32(self) -> None:
        """Verify alternating even/odd lane divergence pattern across 32 lanes."""
        sim = SIMDVectorCNDMask64Simulator(wave_size=32)
        s0_vals = [0xAAAA0000_00000000 | i for i in range(32)]
        s1_vals = [0x55550000_00000000 | i for i in range(32)]
        sim.store_vgpr_pair_wave(0, s0_vals)
        sim.store_vgpr_pair_wave(2, s1_vals)

        # Mask: even bits 0 (src0), odd bits 1 (src1) -> 0xAAAAAAAA
        mask_even0_odd1 = 0xAAAAAAAA
        sim.vcc = mask_even0_odd1
        out = sim.step_v_cndmask_b64(4, 0, 2, cond_src="vcc")
        for lane in range(32):
            expected = s1_vals[lane] if (lane % 2 == 1) else s0_vals[lane]
            self.assertEqual(out[lane], expected)

        # Mask: even bits 1 (src1), odd bits 0 (src0) -> 0x55555555
        mask_even1_odd0 = 0x55555555
        sim.vcc = mask_even1_odd0
        out = sim.step_v_cndmask_b64(4, 0, 2, cond_src="vcc")
        for lane in range(32):
            expected = s1_vals[lane] if (lane % 2 == 0) else s0_vals[lane]
            self.assertEqual(out[lane], expected)

    def test_multi_lane_divergence_checkerboard_wave64(self) -> None:
        """Verify alternating even/odd lane divergence pattern across 64 lanes."""
        sim = SIMDVectorCNDMask64Simulator(wave_size=64)
        s0_vals = [0x12340000_00000000 | (i << 16) for i in range(64)]
        s1_vals = [0x98760000_00000000 | (i << 16) for i in range(64)]
        sim.store_vgpr_pair_wave(10, s0_vals)
        sim.store_vgpr_pair_wave(12, s1_vals)

        sim.vcc = 0xAAAAAAAAAAAAAAAA
        out = sim.step_v_cndmask_b64(14, 10, 12, cond_src="vcc")
        for lane in range(64):
            expected = s1_vals[lane] if (lane % 2 == 1) else s0_vals[lane]
            self.assertEqual(out[lane], expected)

        sim.vcc = 0x5555555555555555
        out = sim.step_v_cndmask_b64(14, 10, 12, cond_src="vcc")
        for lane in range(64):
            expected = s1_vals[lane] if (lane % 2 == 0) else s0_vals[lane]
            self.assertEqual(out[lane], expected)

    def test_single_lane_active_wave32_and_wave64(self) -> None:
        """Verify that a single active condition bit modifies exclusively that lane."""
        for wave_size, test_lanes in ((32, [0, 1, 7, 15, 16, 30, 31]), (64, [0, 1, 31, 32, 47, 62, 63])):
            sim = SIMDVectorCNDMask64Simulator(wave_size=wave_size)
            s0_vals = [0x33333333_00000000 | i for i in range(wave_size)]
            s1_vals = [0x77777777_00000000 | i for i in range(wave_size)]
            sim.store_vgpr_pair_wave(0, s0_vals)
            sim.store_vgpr_pair_wave(2, s1_vals)

            for target_lane in test_lanes:
                with self.subTest(wave_size=wave_size, target_lane=target_lane):
                    mask = 1 << target_lane
                    sim.vcc = mask
                    out = sim.step_v_cndmask_b64(4, 0, 2, cond_src="vcc")
                    for lane in range(wave_size):
                        expected = s1_vals[lane] if lane == target_lane else s0_vals[lane]
                        self.assertEqual(out[lane], expected)

    def test_single_lane_inactive_wave64(self) -> None:
        """Verify that all lanes except one single inactive lane select src1."""
        sim = SIMDVectorCNDMask64Simulator(wave_size=64)
        s0_vals = [0x44444444_00000000 | i for i in range(64)]
        s1_vals = [0x88888888_00000000 | i for i in range(64)]
        sim.store_vgpr_pair_wave(20, s0_vals)
        sim.store_vgpr_pair_wave(22, s1_vals)

        for inactive_lane in (0, 15, 31, 32, 48, 63):
            with self.subTest(inactive_lane=inactive_lane):
                sim.vcc = MASK64 ^ (1 << inactive_lane)
                out = sim.step_v_cndmask_b64(24, 20, 22, cond_src="vcc")
                for lane in range(64):
                    expected = s0_vals[lane] if lane == inactive_lane else s1_vals[lane]
                    self.assertEqual(out[lane], expected)

    def test_subgroup_warp_partitions_wave64(self) -> None:
        """Verify contiguous block partitions (half-waves and quad-waves) across 64 lanes."""
        sim = SIMDVectorCNDMask64Simulator(wave_size=64)
        s0_vals = [0x10000000_00000000 | i for i in range(64)]
        s1_vals = [0x20000000_00000000 | i for i in range(64)]
        sim.store_vgpr_pair_wave(0, s0_vals)
        sim.store_vgpr_pair_wave(2, s1_vals)

        # Low half (0..31) select src0, high half (32..63) select src1
        sim.vcc = 0xFFFFFFFF_00000000
        out = sim.step_v_cndmask_b64(4, 0, 2, cond_src="vcc")
        self.assertEqual(out[:32], s0_vals[:32])
        self.assertEqual(out[32:], s1_vals[32:])

        # Low half select src1, high half select src0
        sim.vcc = 0x00000000_FFFFFFFF
        out = sim.step_v_cndmask_b64(4, 0, 2, cond_src="vcc")
        self.assertEqual(out[:32], s1_vals[:32])
        self.assertEqual(out[32:], s0_vals[32:])

        # Quad-wave blocks: lanes 0..15 (0), 16..31 (1), 32..47 (0), 48..63 (1)
        sim.vcc = 0xFFFF0000_FFFF0000
        out = sim.step_v_cndmask_b64(4, 0, 2, cond_src="vcc")
        for lane in range(64):
            quarter = (lane // 16) % 2
            expected = s1_vals[lane] if quarter == 1 else s0_vals[lane]
            self.assertEqual(out[lane], expected)

    def test_vcc_implicit_and_explicit_routing_wave32(self) -> None:
        """Verify routing via implicit vcc and explicit vcc_lo in Wave32 mode."""
        sim = SIMDVectorCNDMask64Simulator(wave_size=32)
        s0_vals = [0x11110000_00000000 | i for i in range(32)]
        s1_vals = [0x22220000_00000000 | i for i in range(32)]
        sim.store_vgpr_pair_wave(0, s0_vals)
        sim.store_vgpr_pair_wave(2, s1_vals)

        mask32 = 0x5A5A5A5A
        # High bits in VCC must be ignored by Wave32
        sim.vcc = 0xDEADBEEF_00000000 | mask32

        out_vcc = sim.step_v_cndmask_b64(4, 0, 2, cond_src="vcc")
        out_vcc_lo = sim.step_v_cndmask_b64(6, 0, 2, cond_src="vcc_lo")
        self.assertEqual(out_vcc, out_vcc_lo)
        for lane in range(32):
            bit = (mask32 >> lane) & 1
            expected = s1_vals[lane] if bit else s0_vals[lane]
            self.assertEqual(out_vcc[lane], expected)

    def test_vcc_implicit_and_explicit_routing_wave64(self) -> None:
        """Verify routing via vcc (full 64-bit), vcc_lo, and vcc_hi in Wave64 mode."""
        sim = SIMDVectorCNDMask64Simulator(wave_size=64)
        s0_vals = [0xAAAA0000_00000000 | i for i in range(64)]
        s1_vals = [0xBBBB0000_00000000 | i for i in range(64)]
        sim.store_vgpr_pair_wave(0, s0_vals)
        sim.store_vgpr_pair_wave(2, s1_vals)

        lo_mask = 0x12345678
        hi_mask = 0x9ABCDEF0
        sim.vcc = join_dwords(lo_mask, hi_mask)

        # Full 64-bit VCC
        out_full = sim.step_v_cndmask_b64(4, 0, 2, cond_src="vcc")
        for lane in range(32):
            bit = (lo_mask >> lane) & 1
            self.assertEqual(out_full[lane], s1_vals[lane] if bit else s0_vals[lane])
        for lane in range(32, 64):
            bit = (hi_mask >> (lane - 32)) & 1
            self.assertEqual(out_full[lane], s1_vals[lane] if bit else s0_vals[lane])

        # Explicit vcc_lo: in wave64, upper 32 lanes receive zero condition (select src0)
        out_lo = sim.step_v_cndmask_b64(6, 0, 2, cond_src="vcc_lo")
        for lane in range(32):
            bit = (lo_mask >> lane) & 1
            self.assertEqual(out_lo[lane], s1_vals[lane] if bit else s0_vals[lane])
        for lane in range(32, 64):
            self.assertEqual(out_lo[lane], s0_vals[lane])

        # Explicit vcc_hi: upper 32 bits of VCC placed into low 32 lanes, upper lanes zero
        out_hi = sim.step_v_cndmask_b64(8, 0, 2, cond_src="vcc_hi")
        for lane in range(32):
            bit = (hi_mask >> lane) & 1
            self.assertEqual(out_hi[lane], s1_vals[lane] if bit else s0_vals[lane])
        for lane in range(32, 64):
            self.assertEqual(out_hi[lane], s0_vals[lane])

    def test_sgpr_pair_mask_routing_wave64(self) -> None:
        """Verify that an SGPR pair (e.g. s[4:5]) routes condition bits accurately to 64 lanes."""
        sim = SIMDVectorCNDMask64Simulator(wave_size=64)
        s0_vals = [0xC0FFEE00_00000000 | i for i in range(64)]
        s1_vals = [0xFEED0000_00000000 | i for i in range(64)]
        sim.store_vgpr_pair_wave(0, s0_vals)
        sim.store_vgpr_pair_wave(2, s1_vals)

        mask64 = 0x80000000_00000001 | 0x00000001_80000000
        sim.set_sgpr_u64(4, mask64)

        out = sim.step_v_cndmask_b64(4, 0, 2, cond_src="s[4:5]")
        for lane in range(64):
            bit = (mask64 >> lane) & 1
            self.assertEqual(out[lane], s1_vals[lane] if bit else s0_vals[lane])

    def test_sgpr_single_mask_routing_wave32(self) -> None:
        """Verify that a single 32-bit SGPR routes condition bits accurately in Wave32 mode."""
        sim = SIMDVectorCNDMask64Simulator(wave_size=32)
        s0_vals = [0x01010101_00000000 | i for i in range(32)]
        s1_vals = [0x02020202_00000000 | i for i in range(32)]
        sim.store_vgpr_pair_wave(0, s0_vals)
        sim.store_vgpr_pair_wave(2, s1_vals)

        mask32 = 0x0F0F0F0F
        sim.set_sgpr_u32(8, mask32)

        out = sim.step_v_cndmask_b64(4, 0, 2, cond_src="s8")
        for lane in range(32):
            bit = (mask32 >> lane) & 1
            self.assertEqual(out[lane], s1_vals[lane] if bit else s0_vals[lane])

    def test_random_divergence_validation_against_oracle(self) -> None:
        """Validate simulated SIMD wave execution against reference oracle for 30 random cases."""
        rng = random.Random(999)
        for wave_size in (32, 64):
            sim = SIMDVectorCNDMask64Simulator(wave_size=wave_size)
            for _ in range(15):
                s0_vals = [rng.getrandbits(64) for _ in range(wave_size)]
                s1_vals = [rng.getrandbits(64) for _ in range(wave_size)]
                mask = rng.getrandbits(wave_size)

                sim.store_vgpr_pair_wave(0, s0_vals)
                sim.store_vgpr_pair_wave(2, s1_vals)
                sim.vcc = mask

                out_sim = sim.step_v_cndmask_b64(4, 0, 2, cond_src="vcc")
                out_oracle = ref_simd_cndmask_b64(s0_vals, s1_vals, mask, wave_size=wave_size)
                self.assertEqual(out_sim, out_oracle)


if __name__ == "__main__":
    unittest.main()
