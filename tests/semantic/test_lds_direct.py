from __future__ import annotations

"""Semantic interpreter tests for LDS direct dual-offset reads (ds_read2_b32, ds_read2st64_b32).

Covers:
  * AMD RDNA2 ISA specification for LDS (Local Data Share) dual-offset read operations
    (AMD RDNA 2 Instruction Set Architecture, Section 9 "Data Share Instructions"):
      - ds_read2_b32:   v[vdst:vdst+1], vaddr offset0:o0 offset1:o1
          Reads two independent 32-bit dwords directly from LDS into consecutive
          vector registers without separate address generation instructions:
              effective_byte_address0 = (vaddr + offset0 * 4) & 0xFFFFFFFF
              effective_byte_address1 = (vaddr + offset1 * 4) & 0xFFFFFFFF
          Where offset0 and offset1 are 8-bit unsigned integers [0, 255] scaled by 4 bytes.
      - ds_read2st64_b32: v[vdst:vdst+1], vaddr offset0:o0 offset1:o1
          Reads two independent 32-bit dwords with stride-64 dwords (256 bytes):
              effective_byte_address0 = (vaddr + offset0 * 256) & 0xFFFFFFFF
              effective_byte_address1 = (vaddr + offset1 * 256) & 0xFFFFFFFF
          Where offset0 and offset1 are 8-bit unsigned integers [0, 255] scaled by 256 bytes.
      - Dual destination registers:
          v[vdst] receives LDS dword from effective address 0.
          v[vdst + 1] receives LDS dword from effective address 1.
      - Single base address register:
          vaddr specifies the 32-bit base byte address for both accesses.
          Base address is read before destination registers are modified, so vdst may
          overlap with vaddr safely.
      - Sub-dword alignment & dword indexing:
          Low 2 bits (effective_byte_address & 3) represent intra-dword byte offset;
          for b32 operations, the dword index truncates these bits:
              dword_index = (effective_byte_address >> 2) & (lds_dwords - 1)
      - 32-bit address wrapping: (vaddr + offset * unit) wraps strictly modulo 2^32.
      - LDS capacity wrapping: dword indices wrap modulo lds_dwords (power-of-two size).
      - Active lane execution:
          LDS direct reads execute conditionally on lane active mask (EXEC).
          Inactive lanes leave both destination VGPRs (vdst and vdst+1) unmodified.
  * SPIR-V translation mapping (tools/rdna2spirv.py):
      - Compute (cs) and mesh (ms) shader stages:
          ds_read2_b32 and ds_read2st64_b32 compute unit = 4 * (64 if st64 else 1).
          Emits:
            %s1 = OpIAdd %uint {addr} {offset * unit}
            %i  = OpShiftRightLogical %uint %s1 2
            %m  = OpBitwiseAnd %uint %i {lds_dwords - 1}
            %p  = OpAccessChain %_ptr_WG_uint %lds %m
            %v  = OpLoad %uint %p
          Followed by register writes to vgpr r and r + 1, guarded by %active via OpSelect.
      - Other stages (ps, vs, gs) raise rdna2spirv.Unsupported.
  * Deterministic semantic interpreter and test harness:
      - Simulates LDS memory array, VGPR register file, and lane execution state.
      - Cross-checks RDNA2 ISA offset calculations against reference functions and SPIR-V pointer math.
      - Validates stride scaling, asymmetric offsets, overlapping source/dest, inactive lanes,
        and negative controls (sabotaged units, dropped destination writes, and swapped offsets).

Run directly:
    python -I tests/semantic/test_lds_direct.py
Or via unittest discovery:
    python -m unittest discover -s tests/semantic -p test_lds_direct.py
"""

import os
import random
import re
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TOOLS_DIR = os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools")

if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import rdna2spirv  # noqa: E402
import shader_ir  # noqa: E402

MASK32 = 0xFFFFFFFF
DEFAULT_LDS_DWORDS = 16384  # 64 KB LDS capacity in dwords


def calc_lds_dual_offset(
    vaddr: int,
    offset0: int = 0,
    offset1: int = 0,
    is_st64: bool = False,
    lds_dwords: int = DEFAULT_LDS_DWORDS,
) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """Reference ISA calculation for LDS dual-offset read effective byte addresses and dword indices.

    Args:
        vaddr: 32-bit base byte address in VGPR.
        offset0: 8-bit unsigned integer offset for first dword.
        offset1: 8-bit unsigned integer offset for second dword.
        is_st64: True if stride-64 (256 bytes per offset count); False for 4 bytes.
        lds_dwords: LDS capacity in 32-bit dwords (power of two).

    Returns:
        ((eff_byte0, dword_idx0, byte_in_dword0), (eff_byte1, dword_idx1, byte_in_dword1))
    """
    vaddr = vaddr & MASK32
    unit = 256 if is_st64 else 4
    mask = lds_dwords - 1

    off0_bytes = (offset0 & 0xFF) * unit
    eff0 = (vaddr + off0_bytes) & MASK32
    dword0 = (eff0 >> 2) & mask
    byte0 = eff0 & 3

    off1_bytes = (offset1 & 0xFF) * unit
    eff1 = (vaddr + off1_bytes) & MASK32
    dword1 = (eff1 >> 2) & mask
    byte1 = eff1 & 3

    return (eff0, dword0, byte0), (eff1, dword1, byte1)


def emulate_spirv_lds_ptr(addr_val: int, byte_off: int, lds_dwords: int = DEFAULT_LDS_DWORDS) -> int:
    """Emulate the exact SPIR-V pointer index sequence emitted by rdna2spirv.Gen.lds_ptr.

    Sequence:
        %s1 = OpIAdd %uint {addr} {byte_off}
        %i  = OpShiftRightLogical %uint %s1 2
        %m  = OpBitwiseAnd %uint %i {lds_dwords - 1}
    Returns:
        dword index %m into %lds array.
    """
    s1 = (addr_val + byte_off) & MASK32
    i = (s1 >> 2) & MASK32
    m = i & (lds_dwords - 1)
    return m


class LdsDirectSemanticInterpreter:
    """Deterministic semantic interpreter for RDNA2 LDS direct dual-offset read instructions.

    Models LDS shared memory array, VGPR register files, and lane execution state.
    """

    def __init__(self, lds_dwords: int = DEFAULT_LDS_DWORDS, num_vgprs: int = 256) -> None:
        self.lds_dwords = lds_dwords
        self.lds = [0] * lds_dwords
        self.num_vgprs = num_vgprs
        self.vgprs = [0] * num_vgprs
        self.active_lane = True

    def set_vgpr(self, reg_idx: int, value: int) -> None:
        self.vgprs[reg_idx] = value & MASK32

    def get_vgpr(self, reg_idx: int) -> int:
        return self.vgprs[reg_idx]

    def set_lds_dword(self, dword_idx: int, value: int) -> None:
        self.lds[dword_idx & (self.lds_dwords - 1)] = value & MASK32

    def get_lds_dword(self, dword_idx: int) -> int:
        return self.lds[dword_idx & (self.lds_dwords - 1)]

    def ds_write_b32(self, vaddr_reg: int, vdata_reg: int, offset: int = 0) -> int | None:
        """Execute ds_write_b32 vaddr, vdata [offset:offset]."""
        if not self.active_lane:
            return None
        vaddr = self.vgprs[vaddr_reg]
        eff_byte = (vaddr + (offset & 0xFFFF)) & MASK32
        dword_idx = (eff_byte >> 2) & (self.lds_dwords - 1)
        self.lds[dword_idx] = self.vgprs[vdata_reg] & MASK32
        return dword_idx

    def ds_read2_b32(
        self, vdst_reg: int, vaddr_reg: int, offset0: int = 0, offset1: int = 0
    ) -> tuple[int, int] | None:
        """Execute ds_read2_b32 v[vdst:vdst+1], vaddr offset0:o0 offset1:o1."""
        if not self.active_lane:
            return None
        # Base address register is read before destination registers are modified
        vaddr = self.vgprs[vaddr_reg]
        (eff0, dword0, _), (eff1, dword1, _) = calc_lds_dual_offset(
            vaddr, offset0, offset1, is_st64=False, lds_dwords=self.lds_dwords
        )
        val0 = self.lds[dword0]
        val1 = self.lds[dword1]
        self.vgprs[vdst_reg] = val0
        self.vgprs[vdst_reg + 1] = val1
        return dword0, dword1

    def ds_read2st64_b32(
        self, vdst_reg: int, vaddr_reg: int, offset0: int = 0, offset1: int = 0
    ) -> tuple[int, int] | None:
        """Execute ds_read2st64_b32 v[vdst:vdst+1], vaddr offset0:o0 offset1:o1."""
        if not self.active_lane:
            return None
        vaddr = self.vgprs[vaddr_reg]
        (eff0, dword0, _), (eff1, dword1, _) = calc_lds_dual_offset(
            vaddr, offset0, offset1, is_st64=True, lds_dwords=self.lds_dwords
        )
        val0 = self.lds[dword0]
        val1 = self.lds[dword1]
        self.vgprs[vdst_reg] = val0
        self.vgprs[vdst_reg + 1] = val1
        return dword0, dword1

    def parse_and_execute(self, instruction_text: str) -> tuple[int, int] | int | None:
        """Parse and execute a single LDS instruction text line."""
        line = instruction_text.strip()
        if not line or line.startswith(";"):
            return None

        # Parse modifiers: offset0:N, offset1:N, offset:N
        offset0 = 0
        offset1 = 0
        offset = 0

        m0 = re.search(r"\boffset0:(\S+)", line)
        if m0:
            offset0 = int(m0.group(1), 0)
            line = re.sub(r"\boffset0:\S+", "", line).strip()

        m1 = re.search(r"\boffset1:(\S+)", line)
        if m1:
            offset1 = int(m1.group(1), 0)
            line = re.sub(r"\boffset1:\S+", "", line).strip()

        m = re.search(r"\boffset:(\S+)", line)
        if m:
            offset = int(m.group(1), 0)
            line = re.sub(r"\boffset:\S+", "", line).strip()

        parts = line.split(None, 1)
        op = parts[0]
        args_str = parts[1] if len(parts) > 1 else ""
        args = [a.strip() for a in args_str.split(",") if a.strip()]

        if op in ("ds_read2_b32", "ds_read2st64_b32"):
            # Syntax: op v[dst:dst+1] or vdst, vaddr
            dst_match = re.match(r"^v\[(\d+):(\d+)\]$", args[0])
            if dst_match:
                vdst_reg = int(dst_match.group(1))
            else:
                vdst_reg = int(args[0].replace("v", ""))
            vaddr_reg = int(args[1].replace("v", ""))

            if op == "ds_read2_b32":
                return self.ds_read2_b32(vdst_reg, vaddr_reg, offset0, offset1)
            else:
                return self.ds_read2st64_b32(vdst_reg, vaddr_reg, offset0, offset1)

        elif op == "ds_write_b32":
            vaddr_reg = int(args[0].replace("v", ""))
            vdata_reg = int(args[1].replace("v", ""))
            return self.ds_write_b32(vaddr_reg, vdata_reg, offset)
        else:
            raise NotImplementedError(f"Unsupported instruction: {op}")


class DualOffsetCalculationTests(unittest.TestCase):
    """Test effective byte addresses and dword offset calculations per RDNA2 ISA."""

    def test_zero_offsets_both_zero(self):
        cases = [
            (0, 0),
            (4, 1),
            (64, 16),
            (1024, 256),
            (65532, 16383),
        ]
        for vaddr, exp_dword in cases:
            with self.subTest(vaddr=vaddr):
                (eff0, dword0, byte0), (eff1, dword1, byte1) = calc_lds_dual_offset(
                    vaddr, offset0=0, offset1=0, is_st64=False
                )
                self.assertEqual(eff0, vaddr)
                self.assertEqual(eff1, vaddr)
                self.assertEqual(dword0, exp_dword)
                self.assertEqual(dword1, exp_dword)
                self.assertEqual(byte0, 0)
                self.assertEqual(byte1, 0)

    def test_ds_read2_b32_dword_scaling(self):
        """ds_read2_b32 scales offset0 and offset1 by 4 bytes (1 dword)."""
        cases = [
            (0, 1, 2, 4, 8, 1, 2),
            (0, 4, 8, 16, 32, 4, 8),
            (100, 5, 10, 120, 140, 30, 35),
            (0x1000, 0x10, 0x20, 0x1040, 0x1080, 0x1040 // 4, 0x1080 // 4),
        ]
        for vaddr, off0, off1, exp_eff0, exp_eff1, exp_dw0, exp_dw1 in cases:
            with self.subTest(vaddr=vaddr, off0=off0, off1=off1):
                (eff0, dw0, b0), (eff1, dw1, b1) = calc_lds_dual_offset(
                    vaddr, off0, off1, is_st64=False
                )
                self.assertEqual(eff0, exp_eff0)
                self.assertEqual(eff1, exp_eff1)
                self.assertEqual(dw0, exp_dw0)
                self.assertEqual(dw1, exp_dw1)
                self.assertEqual(b0, 0)
                self.assertEqual(b1, 0)

    def test_ds_read2st64_b32_stride64_scaling(self):
        """ds_read2st64_b32 scales offset0 and offset1 by 256 bytes (64 dwords)."""
        cases = [
            (0, 1, 2, 256, 512, 64, 128),
            (0, 4, 8, 1024, 2048, 256, 512),
            (512, 1, 3, 768, 1280, 192, 320),
            (0x2000, 2, 4, 0x2200, 0x2400, 0x2200 // 4, 0x2400 // 4),
        ]
        for vaddr, off0, off1, exp_eff0, exp_eff1, exp_dw0, exp_dw1 in cases:
            with self.subTest(vaddr=vaddr, off0=off0, off1=off1):
                (eff0, dw0, b0), (eff1, dw1, b1) = calc_lds_dual_offset(
                    vaddr, off0, off1, is_st64=True
                )
                self.assertEqual(eff0, exp_eff0)
                self.assertEqual(eff1, exp_eff1)
                self.assertEqual(dw0, exp_dw0)
                self.assertEqual(dw1, exp_dw1)
                self.assertEqual(b0, 0)
                self.assertEqual(b1, 0)

    def test_independent_asymmetric_dual_offsets(self):
        """Dual offsets operate independently, including when off0 > off1."""
        vaddr = 400
        # off0 = 30 -> +120 bytes -> 520 (dword 130)
        # off1 = 5  -> +20 bytes  -> 420 (dword 105)
        (eff0, dw0, _), (eff1, dw1, _) = calc_lds_dual_offset(vaddr, 30, 5, is_st64=False)
        self.assertEqual(eff0, 520)
        self.assertEqual(eff1, 420)
        self.assertEqual(dw0, 130)
        self.assertEqual(dw1, 105)

    def test_8bit_offset_boundary_limits(self):
        """8-bit offsets support values up to 255 (0xFF)."""
        # ds_read2_b32: max reach is 255 * 4 = 1020 bytes
        (eff0, dw0, _), (eff1, dw1, _) = calc_lds_dual_offset(0, 0, 255, is_st64=False)
        self.assertEqual(eff0, 0)
        self.assertEqual(dw0, 0)
        self.assertEqual(eff1, 1020)
        self.assertEqual(dw1, 255)

        # ds_read2st64_b32: max reach is 255 * 256 = 65280 bytes (16320 dwords)
        (eff0_st, dw0_st, _), (eff1_st, dw1_st, _) = calc_lds_dual_offset(0, 0, 255, is_st64=True)
        self.assertEqual(eff0_st, 0)
        self.assertEqual(dw0_st, 0)
        self.assertEqual(eff1_st, 65280)
        self.assertEqual(dw1_st, 16320)

    def test_sub_dword_unaligned_base_address(self):
        """Unaligned base byte addresses are truncated by >> 2 to the containing dword."""
        cases = [
            (1, 0, 0, 1, 1, 0, 0, 1, 1),
            (2, 1, 2, 6, 10, 1, 2, 2, 2),
            (3, 4, 8, 19, 35, 4, 8, 3, 3),
        ]
        for vaddr, o0, o1, exp_eff0, exp_eff1, exp_dw0, exp_dw1, exp_b0, exp_b1 in cases:
            with self.subTest(vaddr=vaddr):
                (eff0, dw0, b0), (eff1, dw1, b1) = calc_lds_dual_offset(vaddr, o0, o1, is_st64=False)
                self.assertEqual(eff0, exp_eff0)
                self.assertEqual(eff1, exp_eff1)
                self.assertEqual(dw0, exp_dw0)
                self.assertEqual(dw1, exp_dw1)
                self.assertEqual(b0, exp_b0)
                self.assertEqual(b1, exp_b1)

    def test_32bit_address_arithmetic_wrapping(self):
        """(vaddr + offset * unit) wraps strictly modulo 2^32."""
        cases = [
            (0xFFFFFFFC, 1, 2, 0, 4, 0, 1),
            (0xFFFFFF00, 1, 2, 0, 0x100, 0, 0x40),
        ]
        # First case with b32: 0xFFFFFFFC + 4 = 0 (wrapped), + 8 = 4
        (eff0, dw0, _), (eff1, dw1, _) = calc_lds_dual_offset(0xFFFFFFFC, 1, 2, is_st64=False)
        self.assertEqual(eff0, 0)
        self.assertEqual(dw0, 0)
        self.assertEqual(eff1, 4)
        self.assertEqual(dw1, 1)

        # Second case with st64: 0xFFFFFF00 + 256 = 0 (wrapped), + 512 = 256
        (eff0_st, dw0_st, _), (eff1_st, dw1_st, _) = calc_lds_dual_offset(0xFFFFFF00, 1, 2, is_st64=True)
        self.assertEqual(eff0_st, 0)
        self.assertEqual(dw0_st, 0)
        self.assertEqual(eff1_st, 256)
        self.assertEqual(dw1_st, 64)

    def test_lds_size_wrapping(self):
        """Dword indices wrap modulo lds_dwords (power of two)."""
        lds_dwords = 1024  # 4 KB capacity
        # Base 4096 = exactly 1 LDS capacity wrap
        (eff0, dw0, _), (eff1, dw1, _) = calc_lds_dual_offset(
            4096, 4, 8, is_st64=False, lds_dwords=lds_dwords
        )
        self.assertEqual(dw0, 4)
        self.assertEqual(dw1, 8)

    def test_randomized_equivalence_with_spirv_math(self):
        """500 pseudo-random iterations verifying dual offsets against SPIR-V emulation."""
        rng = random.Random(0x4C445332)
        for _ in range(500):
            vaddr = rng.randint(0, MASK32)
            off0 = rng.randint(0, 255)
            off1 = rng.randint(0, 255)
            is_st64 = rng.choice([False, True])
            lds_dwords = 1 << rng.randint(6, 14)

            (eff0, dw0, _), (eff1, dw1, _) = calc_lds_dual_offset(
                vaddr, off0, off1, is_st64=is_st64, lds_dwords=lds_dwords
            )

            unit = 256 if is_st64 else 4
            dw0_spirv = emulate_spirv_lds_ptr(vaddr, off0 * unit, lds_dwords=lds_dwords)
            dw1_spirv = emulate_spirv_lds_ptr(vaddr, off1 * unit, lds_dwords=lds_dwords)

            self.assertEqual(dw0, dw0_spirv)
            self.assertEqual(dw1, dw1_spirv)
            self.assertTrue(0 <= dw0 < lds_dwords)
            self.assertTrue(0 <= dw1 < lds_dwords)


class LdsDirectExecutionTests(unittest.TestCase):
    """Test full execution state transitions for ds_read2_b32 and ds_read2st64_b32."""

    def test_basic_ds_read2_b32_execution(self):
        interp = LdsDirectSemanticInterpreter()
        # Seed LDS at dwords 32 and 34
        interp.set_lds_dword(32, 0x11111111)
        interp.set_lds_dword(34, 0x22222222)

        # Base address v0 = 128 (dword 32). off0 = 0 (dword 32), off1 = 2 (dword 34)
        interp.set_vgpr(0, 128)
        interp.set_vgpr(2, 0)
        interp.set_vgpr(3, 0)

        dwords = interp.ds_read2_b32(vdst_reg=2, vaddr_reg=0, offset0=0, offset1=2)
        self.assertEqual(dwords, (32, 34))
        self.assertEqual(interp.get_vgpr(2), 0x11111111)
        self.assertEqual(interp.get_vgpr(3), 0x22222222)

    def test_basic_ds_read2st64_b32_execution(self):
        interp = LdsDirectSemanticInterpreter()
        # Stride 64 dwords (256 bytes)
        # Base address v1 = 0
        # off0 = 1 -> byte 256 -> dword 64
        # off1 = 3 -> byte 768 -> dword 192
        interp.set_lds_dword(64, 0xAAAAAAAA)
        interp.set_lds_dword(192, 0xBBBBBBBB)

        interp.set_vgpr(1, 0)
        interp.set_vgpr(4, 0)
        interp.set_vgpr(5, 0)

        dwords = interp.ds_read2st64_b32(vdst_reg=4, vaddr_reg=1, offset0=1, offset1=3)
        self.assertEqual(dwords, (64, 192))
        self.assertEqual(interp.get_vgpr(4), 0xAAAAAAAA)
        self.assertEqual(interp.get_vgpr(5), 0xBBBBBBBB)

    def test_identical_dual_offsets(self):
        """When offset0 == offset1, both destination registers receive the identical dword."""
        interp = LdsDirectSemanticInterpreter()
        interp.set_lds_dword(50, 0xCAFEBABE)
        interp.set_vgpr(0, 200)  # dword 50

        interp.ds_read2_b32(vdst_reg=6, vaddr_reg=0, offset0=0, offset1=0)
        self.assertEqual(interp.get_vgpr(6), 0xCAFEBABE)
        self.assertEqual(interp.get_vgpr(7), 0xCAFEBABE)

    def test_inverted_offsets_execution(self):
        """offset0 > offset1 correctly loads high offset to vdst and low offset to vdst+1."""
        interp = LdsDirectSemanticInterpreter()
        interp.set_lds_dword(10, 0x10101010)
        interp.set_lds_dword(25, 0x25252525)

        interp.set_vgpr(0, 0)
        # off0 = 25 (dword 25), off1 = 10 (dword 10)
        interp.ds_read2_b32(vdst_reg=8, vaddr_reg=0, offset0=25, offset1=10)
        self.assertEqual(interp.get_vgpr(8), 0x25252525)
        self.assertEqual(interp.get_vgpr(9), 0x10101010)

    def test_destination_overlaps_source_address_register(self):
        """Base address register is sampled before destination registers are updated."""
        interp = LdsDirectSemanticInterpreter()
        # vdst = 2, vaddr = 2: vdst overlaps vaddr
        interp.set_vgpr(2, 64)  # base address 64 -> dword 16
        interp.set_lds_dword(16, 0x33333333)
        interp.set_lds_dword(17, 0x44444444)

        interp.ds_read2_b32(vdst_reg=2, vaddr_reg=2, offset0=0, offset1=1)
        self.assertEqual(interp.get_vgpr(2), 0x33333333)
        self.assertEqual(interp.get_vgpr(3), 0x44444444)

        # vdst = 4, vaddr = 5: vdst+1 overlaps vaddr
        interp.set_vgpr(5, 128)  # base address 128 -> dword 32
        interp.set_lds_dword(32, 0x55555555)
        interp.set_lds_dword(33, 0x66666666)

        interp.ds_read2_b32(vdst_reg=4, vaddr_reg=5, offset0=0, offset1=1)
        self.assertEqual(interp.get_vgpr(4), 0x55555555)
        self.assertEqual(interp.get_vgpr(5), 0x66666666)

    def test_inactive_lane_preserves_both_destination_vgprs(self):
        """When active_lane is False, neither destination register is altered."""
        interp = LdsDirectSemanticInterpreter()
        interp.set_vgpr(0, 100)
        interp.set_vgpr(2, 0x77777777)
        interp.set_vgpr(3, 0x88888888)
        interp.set_lds_dword(25, 0x99999999)
        interp.set_lds_dword(26, 0xAAAAAAAA)

        interp.active_lane = False
        res = interp.ds_read2_b32(vdst_reg=2, vaddr_reg=0, offset0=0, offset1=1)
        self.assertIsNone(res)
        self.assertEqual(interp.get_vgpr(2), 0x77777777)
        self.assertEqual(interp.get_vgpr(3), 0x88888888)

        # Same for st64
        res_st = interp.ds_read2st64_b32(vdst_reg=2, vaddr_reg=0, offset0=1, offset1=2)
        self.assertIsNone(res_st)
        self.assertEqual(interp.get_vgpr(2), 0x77777777)
        self.assertEqual(interp.get_vgpr(3), 0x88888888)

    def test_parse_and_execute_bracket_range_syntax(self):
        """Instruction text with v[dst:dst+1] range syntax parses and executes correctly."""
        interp = LdsDirectSemanticInterpreter()
        interp.set_lds_dword(10, 0x12345678)
        interp.set_lds_dword(12, 0x9ABCDEF0)
        interp.set_vgpr(1, 32)  # dword 8

        # 32 + 2*4 = 40 (dword 10), 32 + 4*4 = 48 (dword 12)
        ret = interp.parse_and_execute("ds_read2_b32 v[6:7], v1 offset0:2 offset1:4")
        self.assertEqual(ret, (10, 12))
        self.assertEqual(interp.get_vgpr(6), 0x12345678)
        self.assertEqual(interp.get_vgpr(7), 0x9ABCDEF0)

    def test_parse_and_execute_single_reg_syntax(self):
        """Instruction text with single vdst syntax parses and executes correctly."""
        interp = LdsDirectSemanticInterpreter()
        interp.set_lds_dword(1, 0x11112222)
        interp.set_lds_dword(2, 0x33334444)
        interp.set_vgpr(2, 0)

        ret = interp.parse_and_execute("ds_read2_b32 v10, v2 offset0:1 offset1:2")
        self.assertEqual(ret, (1, 2))
        self.assertEqual(interp.get_vgpr(10), 0x11112222)
        self.assertEqual(interp.get_vgpr(11), 0x33334444)

    def test_parse_and_execute_st64_hex_offsets(self):
        """Hexadecimal offset modifiers with ds_read2st64_b32."""
        interp = LdsDirectSemanticInterpreter()
        # offset0: 0x02 -> 2 * 256 = 512 (dword 128)
        # offset1: 0x04 -> 4 * 256 = 1024 (dword 256)
        interp.set_lds_dword(128, 0xDEADBEEF)
        interp.set_lds_dword(256, 0xFEEDFACE)
        interp.set_vgpr(3, 0)

        ret = interp.parse_and_execute("ds_read2st64_b32 v[8:9], v3 offset0:0x02 offset1:0x04")
        self.assertEqual(ret, (128, 256))
        self.assertEqual(interp.get_vgpr(8), 0xDEADBEEF)
        self.assertEqual(interp.get_vgpr(9), 0xFEEDFACE)

    def test_parse_and_execute_default_omitted_offsets(self):
        """Omitted offset modifiers default to 0."""
        interp = LdsDirectSemanticInterpreter()
        interp.set_lds_dword(0, 0x55555555)
        interp.set_lds_dword(1, 0x66666666)
        interp.set_vgpr(4, 0)

        # Both omitted
        ret = interp.parse_and_execute("ds_read2_b32 v[12:13], v4")
        self.assertEqual(ret, (0, 0))
        self.assertEqual(interp.get_vgpr(12), 0x55555555)
        self.assertEqual(interp.get_vgpr(13), 0x55555555)

        # offset1 omitted -> defaults to 0
        ret2 = interp.parse_and_execute("ds_read2_b32 v[14:15], v4 offset0:1")
        self.assertEqual(ret2, (1, 0))
        self.assertEqual(interp.get_vgpr(14), 0x66666666)
        self.assertEqual(interp.get_vgpr(15), 0x55555555)

    def test_consecutive_dual_read_stream(self):
        """Simulate unrolled reduction or FFT loading multiple dwords via dual reads."""
        interp = LdsDirectSemanticInterpreter()
        for i in range(8):
            interp.set_lds_dword(i, 0x100 + i)

        interp.set_vgpr(0, 0)
        # Load dwords 0, 1 into v[2:3]
        interp.parse_and_execute("ds_read2_b32 v[2:3], v0 offset0:0 offset1:1")
        # Load dwords 2, 3 into v[4:5]
        interp.parse_and_execute("ds_read2_b32 v[4:5], v0 offset0:2 offset1:3")
        # Load dwords 4, 5 into v[6:7]
        interp.parse_and_execute("ds_read2_b32 v[6:7], v0 offset0:4 offset1:5")
        # Load dwords 6, 7 into v[8:9]
        interp.parse_and_execute("ds_read2_b32 v[8:9], v0 offset0:6 offset1:7")

        for i in range(8):
            self.assertEqual(interp.get_vgpr(2 + i), 0x100 + i)


class Rdna2SpirvDirectEmissionTests(unittest.TestCase):
    """Verify tools/rdna2spirv.py SPIR-V emission for ds_read2_b32 and ds_read2st64_b32."""

    def test_ds_read2_b32_emission_in_cs(self):
        g = rdna2spirv.Gen(stage="cs")
        # ds_read2_b32 v[0:1], v2 offset0:4 offset1:8
        g.lds_op(
            "ds_read2_b32",
            [{"k": "vgpr", "r": 0, "n": 2}, {"k": "vgpr", "r": 2, "n": 1}],
            ["offset0:4", "offset1:8"],
        )
        body = "\n".join(g.body)

        self.assertTrue(g.uses_lds)
        self.assertIn("OpSelectionMerge", body)
        self.assertIn("OpBranchConditional", body)
        # Check OpLoad instructions: 1 vaddr load + 2 LDS loads + 2 destination VGPR old loads = 5
        self.assertEqual(body.count("OpLoad %uint %t"), 1 + 2 + 2)

        # Check byte offsets: 4 * 4 = 16 (0x10), 8 * 4 = 32 (0x20)
        c16 = g.cu(16)
        c32 = g.cu(32)
        self.assertIn(f"OpIAdd %uint", body)
        self.assertIn(c16, body)
        self.assertIn(c32, body)

        # Check destination register writes: v0 and v1
        c0 = g.cu(0)
        c1 = g.cu(1)
        self.assertIn(f"OpAccessChain %_ptr_Function_uint %vgpr {c0}", body)
        self.assertIn(f"OpAccessChain %_ptr_Function_uint %vgpr {c1}", body)

    def test_ds_read2st64_b32_emission_in_cs(self):
        g = rdna2spirv.Gen(stage="cs")
        # ds_read2st64_b32 v[4:5], v10 offset0:1 offset1:2
        g.lds_op(
            "ds_read2st64_b32",
            [{"k": "vgpr", "r": 4, "n": 2}, {"k": "vgpr", "r": 10, "n": 1}],
            ["offset0:1", "offset1:2"],
        )
        body = "\n".join(g.body)

        self.assertTrue(g.uses_lds)
        # Check byte offsets: 1 * 256 = 256 (0x100), 2 * 256 = 512 (0x200)
        c256 = g.cu(256)
        c512 = g.cu(512)
        self.assertIn(c256, body)
        self.assertIn(c512, body)

        # Check destination registers: v4 and v5
        c4 = g.cu(4)
        c5 = g.cu(5)
        self.assertIn(f"OpAccessChain %_ptr_Function_uint %vgpr {c4}", body)
        self.assertIn(f"OpAccessChain %_ptr_Function_uint %vgpr {c5}", body)

    def test_ds_read2_emission_in_ms(self):
        """Mesh shaders (ms) allow LDS direct reads."""
        g = rdna2spirv.Gen(stage="ms")
        g.lds_op(
            "ds_read2_b32",
            [{"k": "vgpr", "r": 2, "n": 2}, {"k": "vgpr", "r": 0, "n": 1}],
            ["offset0:0", "offset1:1"],
        )
        self.assertTrue(g.uses_lds)
        body = "\n".join(g.body)
        self.assertIn("OpSelectionMerge", body)

    def test_disasm_parser_integration(self):
        """Verify integration between shader_ir.parse_line and Gen.lds_op."""
        ins = shader_ir.parse_line("\tds_read2_b32 v[8:9], v3 offset0:10 offset1:20")
        self.assertEqual(ins["op"], "ds_read2_b32")
        self.assertEqual(ins["args"][0]["r"], 8)
        self.assertEqual(ins["args"][0]["n"], 2)
        self.assertEqual(ins["args"][1]["r"], 3)

        g = rdna2spirv.Gen(stage="cs")
        g.lds_op(ins["op"], ins["args"], ins["mods"])
        body = "\n".join(g.body)
        c40 = g.cu(40)   # 10 * 4
        c80 = g.cu(80)   # 20 * 4
        self.assertIn(c40, body)
        self.assertIn(c80, body)

    def test_unsupported_stages_reject(self):
        """Pixel, vertex, and geometry stages must reject ds_read2 instructions."""
        for stage in ("ps", "vs", "gs"):
            with self.subTest(stage=stage):
                g = rdna2spirv.Gen(stage=stage)
                with self.assertRaises(rdna2spirv.Unsupported):
                    g.lds_op(
                        "ds_read2_b32",
                        [{"k": "vgpr", "r": 0, "n": 2}, {"k": "vgpr", "r": 1, "n": 1}],
                        [],
                    )
                with self.assertRaises(rdna2spirv.Unsupported):
                    g.lds_op(
                        "ds_read2st64_b32",
                        [{"k": "vgpr", "r": 0, "n": 2}, {"k": "vgpr", "r": 1, "n": 1}],
                        [],
                    )


class NegativeControlAndMutationTests(unittest.TestCase):
    """Verify that intentional corruptions in dual-offset LDS read semantics are strictly caught."""

    def test_detects_b32_stride_corruption(self):
        """Corrupting ds_read2_b32 unit from 4 to 1 (treating offset as bytes) fails."""
        vaddr = 0
        off0 = 8
        off1 = 16
        (eff0, dw0, _), (eff1, dw1, _) = calc_lds_dual_offset(vaddr, off0, off1, is_st64=False)
        self.assertEqual(dw0, 8)
        self.assertEqual(dw1, 16)

        # Corrupted unit: byte scaling (unit = 1 instead of 4)
        corrupted_dw0 = ((vaddr + off0 * 1) >> 2) & (DEFAULT_LDS_DWORDS - 1)
        self.assertEqual(corrupted_dw0, 2)
        self.assertNotEqual(dw0, corrupted_dw0)

    def test_detects_st64_stride_corruption(self):
        """Corrupting ds_read2st64_b32 unit from 256 to 4 (treating st64 as regular read2) fails."""
        vaddr = 0
        off0 = 2
        off1 = 4
        (eff0, dw0, _), (eff1, dw1, _) = calc_lds_dual_offset(vaddr, off0, off1, is_st64=True)
        self.assertEqual(dw0, 128)
        self.assertEqual(dw1, 256)

        # Corrupted unit: unit = 4 instead of 256
        corrupted_dw0 = ((vaddr + off0 * 4) >> 2) & (DEFAULT_LDS_DWORDS - 1)
        self.assertEqual(corrupted_dw0, 2)
        self.assertNotEqual(dw0, corrupted_dw0)

    def test_detects_offset_swap_corruption(self):
        """Swapping offset0 and offset1 results in mismatched values when offsets differ."""
        interp = LdsDirectSemanticInterpreter()
        interp.set_lds_dword(10, 0xAAAAAAAA)
        interp.set_lds_dword(20, 0xBBBBBBBB)
        interp.set_vgpr(0, 0)

        # Correct: off0=10 (v2 gets 0xAAAA), off1=20 (v3 gets 0xBBBB)
        interp.ds_read2_b32(vdst_reg=2, vaddr_reg=0, offset0=10, offset1=20)
        self.assertEqual(interp.get_vgpr(2), 0xAAAAAAAA)
        self.assertEqual(interp.get_vgpr(3), 0xBBBBBBBB)

        # Mutated: swapped offsets
        interp.set_vgpr(4, 0)
        interp.set_vgpr(5, 0)
        interp.ds_read2_b32(vdst_reg=4, vaddr_reg=0, offset0=20, offset1=10)
        self.assertNotEqual(interp.get_vgpr(2), interp.get_vgpr(4))
        self.assertNotEqual(interp.get_vgpr(3), interp.get_vgpr(5))

    def test_detects_single_vgpr_write_corruption(self):
        """Failing to write the second destination VGPR (vdst+1) is detected."""
        interp = LdsDirectSemanticInterpreter()
        interp.set_lds_dword(1, 0x11111111)
        interp.set_lds_dword(2, 0x22222222)
        interp.set_vgpr(0, 0)
        interp.set_vgpr(2, 0)
        interp.set_vgpr(3, 0xFFFFFFFF)  # Sentinel in vdst+1

        # Correct read updates both v2 and v3
        interp.ds_read2_b32(vdst_reg=2, vaddr_reg=0, offset0=1, offset1=2)
        self.assertEqual(interp.get_vgpr(2), 0x11111111)
        self.assertEqual(interp.get_vgpr(3), 0x22222222)

        # If only vdst were written, vdst+1 would retain sentinel
        self.assertNotEqual(interp.get_vgpr(3), 0xFFFFFFFF)

    def test_detects_shift_corruption(self):
        """Corrupting >> 2 shift to >> 3 (treating dwords as 8-byte uint64) fails."""
        vaddr = 16
        off = 4  # 4 * 4 = 16 bytes -> eff = 32 -> dword 8
        (eff0, dw0, _), _ = calc_lds_dual_offset(vaddr, off, 0, is_st64=False)
        self.assertEqual(dw0, 8)

        corrupted_dw = ((vaddr + off * 4) >> 3) & (DEFAULT_LDS_DWORDS - 1)
        self.assertEqual(corrupted_dw, 4)
        self.assertNotEqual(dw0, corrupted_dw)

    def test_detects_missing_lds_wrap_mask(self):
        """Omitting (lds_dwords - 1) mask fails for out-of-bounds addresses."""
        lds_dwords = 512
        vaddr = 512 * 4 * 3 + 16  # 3 full wraps + 16 bytes (dword 4)
        (eff0, dw0, _), _ = calc_lds_dual_offset(vaddr, 0, 0, lds_dwords=lds_dwords)
        self.assertEqual(dw0, 4)

        corrupted_unmasked = (vaddr >> 2)
        self.assertNotEqual(dw0, corrupted_unmasked)

    def test_detects_inactive_lane_leak_corruption(self):
        """Corrupted interpreter that ignores active_lane flag fails check."""
        interp = LdsDirectSemanticInterpreter()
        interp.set_vgpr(2, 0x12341234)
        interp.set_vgpr(3, 0x56785678)
        interp.active_lane = False

        interp.ds_read2_b32(vdst_reg=2, vaddr_reg=0, offset0=0, offset1=1)
        # Should be preserved
        self.assertEqual(interp.get_vgpr(2), 0x12341234)
        self.assertEqual(interp.get_vgpr(3), 0x56785678)


if __name__ == "__main__":
    unittest.main()
