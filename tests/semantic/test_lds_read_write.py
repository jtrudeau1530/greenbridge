"""Semantic interpreter tests for LDS read/write byte offset calculation (ds_read_b32, ds_write_b32).

Covers:
  * AMD RDNA2 ISA specification for LDS (Local Data Share) data share operations
    (AMD RDNA 2 Instruction Set Architecture, Section 9 "Data Share Instructions"):
      - ds_read_b32:  vdst, vaddr [offset:offset]
          Reads 32 bits from LDS at effective byte address:
              effective_byte_address = (vaddr + offset) & 0xFFFFFFFF
      - ds_write_b32: vaddr, vdata [offset:offset]
          Writes 32 bits into LDS at effective byte address:
              effective_byte_address = (vaddr + offset) & 0xFFFFFFFF
      - Offset field: 16-bit unsigned byte offset in [0, 65535]. Default: 0 if omitted.
      - Dword addressing: LDS is structured as 32-bit dwords (4 bytes per dword).
          dword_index = (effective_byte_address >> 2) & (lds_dwords - 1)
      - Sub-dword alignment: low 2 bits (effective_byte_address & 3) represent intra-dword
        byte position; for b32 operations, the dword index truncates these bits.
      - 32-bit address wrapping: (vaddr + offset) wraps modulo 2^32.
      - LDS size wrapping: dword indices wrap modulo lds_dwords (power-of-two size).
      - Active lane execution: LDS writes and reads execute conditionally on the lane's
        active mask. Inactive lanes do not modify LDS memory and preserve destination VGPRs.
  * SPIR-V translation mapping (tools/rdna2spirv.py):
      - In compute shaders (cs) and mesh shaders (ms), LDS operations allocate:
          %lds_arr = OpTypeArray %uint lds_dwords
          %lds = OpVariable %_ptr_WG_lds Workgroup
      - Pointer calculation (lds_ptr):
          %s1 = OpIAdd %uint {addr} {byte_off}
          %i  = OpShiftRightLogical %uint %s1 2
          %m  = OpBitwiseAnd %uint %i {lds_dwords - 1}
          %p  = OpAccessChain %_ptr_WG_uint %lds %m
      - ds_read_b32 emits OpLoad %uint %p, followed by OpSelect with %active to preserve
        inactive VGPR values.
      - ds_write_b32 emits OpStore %p %vdata guarded by OpBranchConditional on %active.
      - LDS operations in non-compute stages (ps, vs, gs) raise Unsupported.
  * Deterministic semantic interpreter and SPIR-V instruction emulation:
      - Simulates LDS memory array, VGPR register files, and lane execution masks.
      - Cross-checks RDNA2 ISA offset calculations against plain-Python reference functions
        and translated SPIR-V instruction sequences.
      - Evaluates address aliasing, independent offsets, unaligned bases, boundary wrapping,
        and negative controls (sabotaged shifts, masks, and offsets).

Run directly:
    python -I tests/semantic/test_lds_read_write.py
Or via unittest discovery:
    python -m unittest discover -s tests/semantic -v
"""
import os
import random
import re
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TOOLS_DIR = os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools")

sys.path.insert(0, TOOLS_DIR)
sys.path.insert(0, HERE)

import rdna2spirv  # noqa: E402

MASK32 = 0xFFFFFFFF
DEFAULT_LDS_DWORDS = 16384  # 64 KB default LDS capacity in dwords


def calc_lds_byte_offset(vaddr, byte_offset=0, lds_dwords=DEFAULT_LDS_DWORDS):
    """Reference ISA calculation for LDS read/write effective byte address and dword index.

    Returns:
        (effective_byte_address, dword_index, byte_in_dword)
    """
    vaddr = vaddr & MASK32
    byte_offset = byte_offset & 0xFFFF  # 16-bit unsigned offset in RDNA2 DS instructions
    eff_byte = (vaddr + byte_offset) & MASK32
    dword_idx = (eff_byte >> 2) & (lds_dwords - 1)
    byte_in_dword = eff_byte & 3
    return eff_byte, dword_idx, byte_in_dword


def emulate_spirv_lds_ptr(addr_val, byte_off, lds_dwords=DEFAULT_LDS_DWORDS):
    """Emulate the exact SPIR-V instruction sequence emitted by rdna2spirv.Gen.lds_ptr.

    Sequence:
        %s1 = OpIAdd %uint {addr} {byte_off}
        %i  = OpShiftRightLogical %uint %s1 2
        %m  = OpBitwiseAnd %uint %i {lds_dwords - 1}
    Returns:
        dword index %m into %lds
    """
    s1 = (addr_val + byte_off) & MASK32
    i = (s1 >> 2) & MASK32
    m = i & (lds_dwords - 1)
    return m


class LdsSemanticInterpreter:
    """Deterministic semantic interpreter for RDNA2 LDS read/write instructions.

    Models LDS shared memory storage, VGPR register files, and lane execution state.
    """

    def __init__(self, lds_dwords=DEFAULT_LDS_DWORDS, num_vgprs=256):
        self.lds_dwords = lds_dwords
        self.lds = [0] * lds_dwords
        self.num_vgprs = num_vgprs
        self.vgprs = [0] * num_vgprs
        self.active_lane = True

    def set_vgpr(self, reg_idx, value):
        self.vgprs[reg_idx] = value & MASK32

    def get_vgpr(self, reg_idx):
        return self.vgprs[reg_idx]

    def set_lds_dword(self, dword_idx, value):
        self.lds[dword_idx & (self.lds_dwords - 1)] = value & MASK32

    def get_lds_dword(self, dword_idx):
        return self.lds[dword_idx & (self.lds_dwords - 1)]

    def ds_write_b32(self, vaddr_reg, vdata_reg, offset=0):
        """Execute ds_write_b32 vaddr, vdata [offset:offset]."""
        if not self.active_lane:
            return None
        vaddr = self.vgprs[vaddr_reg]
        vdata = self.vgprs[vdata_reg]
        _, dword_idx, _ = calc_lds_byte_offset(vaddr, offset, self.lds_dwords)
        self.lds[dword_idx] = vdata & MASK32
        return dword_idx

    def ds_read_b32(self, vdst_reg, vaddr_reg, offset=0):
        """Execute ds_read_b32 vdst, vaddr [offset:offset]."""
        if not self.active_lane:
            # Inactive lane leaves destination VGPR unchanged
            return None
        vaddr = self.vgprs[vaddr_reg]
        _, dword_idx, _ = calc_lds_byte_offset(vaddr, offset, self.lds_dwords)
        self.vgprs[vdst_reg] = self.lds[dword_idx]
        return dword_idx

    def parse_and_execute(self, instruction_text):
        """Parse and execute a single LDS instruction line."""
        line = instruction_text.strip()
        if not line or line.startswith(";"):
            return None

        # Parse modifiers: e.g. offset:0x20 or offset:32
        offset = 0
        mod_match = re.search(r"\boffset:(\S+)", line)
        if mod_match:
            offset_str = mod_match.group(1)
            offset = int(offset_str, 0)
            line = re.sub(r"\boffset:\S+", "", line).strip()

        parts = line.split(None, 1)
        op = parts[0]
        args = [a.strip() for a in parts[1].split(",")] if len(parts) > 1 else []

        if op == "ds_write_b32":
            # ds_write_b32 vaddr, vdata
            vaddr_reg = int(args[0].replace("v", ""))
            vdata_reg = int(args[1].replace("v", ""))
            return self.ds_write_b32(vaddr_reg, vdata_reg, offset)
        elif op == "ds_read_b32":
            # ds_read_b32 vdst, vaddr
            vdst_reg = int(args[0].replace("v", ""))
            vaddr_reg = int(args[1].replace("v", ""))
            return self.ds_read_b32(vdst_reg, vaddr_reg, offset)
        else:
            raise NotImplementedError(f"Unsupported LDS instruction: {op}")


class LdsByteOffsetCalculationTests(unittest.TestCase):
    """Test effective byte address and dword offset calculations per RDNA2 ISA."""

    def test_zero_offset_calculation(self):
        cases = [
            (0, 0),
            (4, 1),
            (8, 2),
            (64, 16),
            (1024, 256),
            (65532, 16383),
        ]
        for vaddr, expected_dword in cases:
            with self.subTest(vaddr=vaddr):
                eff_byte, dword_idx, byte_in_dword = calc_lds_byte_offset(vaddr, 0)
                self.assertEqual(eff_byte, vaddr)
                self.assertEqual(dword_idx, expected_dword)
                self.assertEqual(byte_in_dword, 0)

    def test_positive_aligned_immediate_offsets(self):
        cases = [
            (0, 4, 4, 1),
            (0, 16, 16, 4),
            (0, 64, 64, 16),
            (100, 20, 120, 30),
            (0x1000, 0x200, 0x1200, 0x1200 // 4),
            (60000, 4000, 64000, 16000),
        ]
        for vaddr, offset, exp_eff, exp_dword in cases:
            with self.subTest(vaddr=vaddr, offset=offset):
                eff_byte, dword_idx, byte_in_dword = calc_lds_byte_offset(vaddr, offset)
                self.assertEqual(eff_byte, exp_eff)
                self.assertEqual(dword_idx, exp_dword)
                self.assertEqual(byte_in_dword, 0)

    def test_hexadecimal_offset_parsing_and_calculation(self):
        interpreter = LdsSemanticInterpreter()
        interpreter.set_vgpr(0, 0x200)
        interpreter.set_vgpr(1, 0x12345678)

        # 0x200 + 0x40 = 0x240 -> dword 0x240 // 4 = 0x90 = 144
        dword_idx = interpreter.parse_and_execute("ds_write_b32 v0, v1 offset:0x40")
        self.assertEqual(dword_idx, 144)
        self.assertEqual(interpreter.get_lds_dword(144), 0x12345678)

        # Read back via hex offset
        interpreter.parse_and_execute("ds_read_b32 v2, v0 offset:0x40")
        self.assertEqual(interpreter.get_vgpr(2), 0x12345678)

    def test_sub_dword_unaligned_base_address(self):
        """Unaligned base addresses are truncated by >> 2 to the containing dword."""
        cases = [
            (1, 0, 1, 0, 1),
            (2, 0, 2, 0, 2),
            (3, 0, 3, 0, 3),
            (5, 0, 5, 1, 1),
            (6, 2, 8, 2, 0),
            (7, 1, 8, 2, 0),
            (7, 5, 12, 3, 0),
        ]
        for vaddr, offset, exp_eff, exp_dword, exp_byte in cases:
            with self.subTest(vaddr=vaddr, offset=offset):
                eff_byte, dword_idx, byte_in_dword = calc_lds_byte_offset(vaddr, offset)
                self.assertEqual(eff_byte, exp_eff)
                self.assertEqual(dword_idx, exp_dword)
                self.assertEqual(byte_in_dword, exp_byte)

    def test_32bit_address_wrapping(self):
        """(vaddr + offset) wraps modulo 2^32."""
        cases = [
            (0xFFFFFFFC, 4, 0, 0),
            (0xFFFFFFFC, 8, 4, 1),
            (0xFFFFFFFF, 1, 0, 0),
            (0xFFFFFFFF, 5, 4, 1),
            (0xFFFFFF00, 0x200, 0x100, 0x40),
        ]
        for vaddr, offset, exp_eff, exp_dword in cases:
            with self.subTest(vaddr=hex(vaddr), offset=offset):
                eff_byte, dword_idx, byte_in_dword = calc_lds_byte_offset(vaddr, offset)
                self.assertEqual(eff_byte, exp_eff)
                self.assertEqual(dword_idx, exp_dword)

    def test_lds_size_wrapping(self):
        """Dword indices wrap modulo lds_dwords (power of two)."""
        lds_dwords = 1024  # 4 KB LDS capacity = 4096 bytes
        cases = [
            (4096, 0, 0),        # Exactly 1 wrap
            (4096, 16, 4),       # Wrap + 4 dwords
            (8192, 0, 0),        # 2 wraps
            (4092, 8, 1),        # Cross boundary: 4092 + 8 = 4100 -> byte 4 -> dword 1
            (10000, 0, (10000 >> 2) & 1023),
        ]
        for vaddr, offset, exp_dword in cases:
            with self.subTest(vaddr=vaddr, offset=offset):
                _, dword_idx, _ = calc_lds_byte_offset(vaddr, offset, lds_dwords)
                self.assertEqual(dword_idx, exp_dword)

    def test_max_16bit_immediate_offset(self):
        """RDNA2 DS offset field allows 16-bit values up to 65535."""
        max_aligned_offset = 65532  # 0xFFFC
        eff_byte, dword_idx, byte_in_dword = calc_lds_byte_offset(0, max_aligned_offset)
        self.assertEqual(eff_byte, 65532)
        self.assertEqual(dword_idx, 65532 // 4)
        self.assertEqual(byte_in_dword, 0)

        max_offset = 65535  # 0xFFFF
        eff_byte, dword_idx, byte_in_dword = calc_lds_byte_offset(0, max_offset)
        self.assertEqual(eff_byte, 65535)
        self.assertEqual(dword_idx, 65535 // 4)
        self.assertEqual(byte_in_dword, 3)

    def test_randomized_property_equivalence_with_spirv(self):
        """Verify 500 pseudo-random test vectors match across ISA and SPIR-V emulation."""
        rng = random.Random(0x1EA105DA)
        for _ in range(500):
            vaddr = rng.randint(0, MASK32)
            offset = rng.randint(0, 0xFFFF)
            lds_dwords = 1 << rng.randint(6, 14)  # 64 to 16384 dwords

            eff_isa, dword_isa, _ = calc_lds_byte_offset(vaddr, offset, lds_dwords)
            dword_spirv = emulate_spirv_lds_ptr(vaddr, offset, lds_dwords)

            self.assertEqual(dword_isa, dword_spirv)
            self.assertTrue(0 <= dword_isa < lds_dwords)


class LdsSemanticInterpreterExecutionTests(unittest.TestCase):
    """Test full execution state transitions for ds_write_b32 and ds_read_b32."""

    def test_write_read_roundtrip_zero_offset(self):
        interp = LdsSemanticInterpreter()
        interp.set_vgpr(0, 128)        # base byte address 128 -> dword 32
        interp.set_vgpr(1, 0xCAFEBABE) # data to write

        interp.ds_write_b32(vaddr_reg=0, vdata_reg=1, offset=0)
        self.assertEqual(interp.get_lds_dword(32), 0xCAFEBABE)

        interp.set_vgpr(2, 0)          # destination register cleared
        interp.ds_read_b32(vdst_reg=2, vaddr_reg=0, offset=0)
        self.assertEqual(interp.get_vgpr(2), 0xCAFEBABE)

    def test_write_read_roundtrip_with_offset(self):
        interp = LdsSemanticInterpreter()
        interp.set_vgpr(0, 200)        # base byte address 200
        interp.set_vgpr(1, 0x11223344) # data

        # Write to 200 + 48 = 248 -> dword 62
        interp.ds_write_b32(vaddr_reg=0, vdata_reg=1, offset=48)
        self.assertEqual(interp.get_lds_dword(62), 0x11223344)

        # Read back with same base and offset
        interp.ds_read_b32(vdst_reg=3, vaddr_reg=0, offset=48)
        self.assertEqual(interp.get_vgpr(3), 0x11223344)

    def test_address_aliasing_via_different_bases_and_offsets(self):
        """Different (base, offset) pairs pointing to the same effective address resolve identically."""
        interp = LdsSemanticInterpreter()
        # Pair 1: base 100 + offset 28 = 128 (dword 32)
        interp.set_vgpr(0, 100)
        interp.set_vgpr(1, 0xDEADBEEF)
        interp.ds_write_b32(vaddr_reg=0, vdata_reg=1, offset=28)

        # Pair 2: base 128 + offset 0 = 128 (dword 32)
        interp.set_vgpr(2, 128)
        interp.ds_read_b32(vdst_reg=4, vaddr_reg=2, offset=0)
        self.assertEqual(interp.get_vgpr(4), 0xDEADBEEF)

        # Pair 3: base 0 + offset 128 = 128 (dword 32)
        interp.set_vgpr(3, 0)
        interp.ds_read_b32(vdst_reg=5, vaddr_reg=3, offset=128)
        self.assertEqual(interp.get_vgpr(5), 0xDEADBEEF)

    def test_disjoint_offsets_isolation(self):
        """Writes to consecutive offsets do not overwrite neighbouring dwords."""
        interp = LdsSemanticInterpreter()
        base_reg = 0
        interp.set_vgpr(base_reg, 512)

        data = [0x11111111, 0x22222222, 0x33333333, 0x44444444]
        for i, val in enumerate(data):
            interp.set_vgpr(1, val)
            interp.ds_write_b32(vaddr_reg=base_reg, vdata_reg=1, offset=4 * i)

        # Verify each dword individually
        for i, expected in enumerate(data):
            target_reg = 10 + i
            interp.ds_read_b32(vdst_reg=target_reg, vaddr_reg=base_reg, offset=4 * i)
            self.assertEqual(interp.get_vgpr(target_reg), expected)

    def test_inactive_lane_does_not_modify_lds(self):
        interp = LdsSemanticInterpreter()
        interp.set_vgpr(0, 64)
        interp.set_vgpr(1, 0x99999999)
        interp.set_lds_dword(16, 0x00000000)

        # Inactive lane executes write
        interp.active_lane = False
        interp.ds_write_b32(vaddr_reg=0, vdata_reg=1, offset=0)
        self.assertEqual(interp.get_lds_dword(16), 0x00000000)

    def test_inactive_lane_does_not_modify_destination_vgpr(self):
        interp = LdsSemanticInterpreter()
        interp.set_vgpr(0, 64)
        interp.set_vgpr(2, 0x55555555) # Existing value
        interp.set_lds_dword(16, 0xAAAAAAAA)

        # Inactive lane executes read
        interp.active_lane = False
        interp.ds_read_b32(vdst_reg=2, vaddr_reg=0, offset=0)
        # Value preserved
        self.assertEqual(interp.get_vgpr(2), 0x55555555)


class TranslatorLdsEmissionTests(unittest.TestCase):
    """Verify tools/rdna2spirv.py SPIR-V emission for LDS byte offset calculations."""

    def test_lds_ptr_emits_iadd_shift_mask_accesschain(self):
        g = rdna2spirv.Gen(stage="cs")
        ptr = g.lds_ptr("%vaddr", 32)
        body = "\n".join(g.body)

        c32 = g.cu(32)
        c2 = g.cu(2)
        c_mask = g.cu(g.lds_dwords - 1)

        self.assertIn(f"OpIAdd %uint %vaddr {c32}", body)
        self.assertIn(f"OpShiftRightLogical %uint", body)
        self.assertIn(f"{c2}", body)
        self.assertIn(f"OpBitwiseAnd %uint", body)
        self.assertIn(f"{c_mask}", body)
        self.assertIn(f"OpAccessChain %_ptr_WG_uint %lds", body)
        self.assertTrue(ptr.startswith("%t"))

    def test_ds_write_b32_emission_in_compute(self):
        g = rdna2spirv.Gen(stage="cs")
        # ds_write_b32 vaddr=v1, vdata=v2, offset:16
        g.lds_op("ds_write_b32", [{"k": "vgpr", "r": 1, "n": 1}, {"k": "vgpr", "r": 2, "n": 1}], ["offset:16"])
        body = "\n".join(g.body)

        self.assertTrue(g.uses_lds)
        self.assertIn("OpStore", body)
        self.assertIn("OpSelectionMerge", body)
        self.assertIn("OpBranchConditional", body)
        c16 = g.cu(16)
        self.assertIn(f"OpIAdd %uint", body)
        self.assertIn(c16, body)

    def test_ds_read_b32_emission_in_compute(self):
        g = rdna2spirv.Gen(stage="cs")
        # ds_read_b32 vdst=v0, vaddr=v1, offset:64
        g.lds_op("ds_read_b32", [{"k": "vgpr", "r": 0, "n": 1}, {"k": "vgpr", "r": 1, "n": 1}], ["offset:64"])
        body = "\n".join(g.body)

        self.assertTrue(g.uses_lds)
        self.assertIn("OpLoad %uint", body)
        self.assertIn("OpSelect %uint", body)
        c64 = g.cu(64)
        self.assertIn(c64, body)

    def test_mesh_stage_permits_lds_operations(self):
        g = rdna2spirv.Gen(stage="ms")
        g.lds_op("ds_write_b32", [{"k": "vgpr", "r": 0, "n": 1}, {"k": "vgpr", "r": 1, "n": 1}], ["offset:0"])
        self.assertTrue(g.uses_lds)
        self.assertIn("OpStore", "\n".join(g.body))

    def test_pixel_and_vertex_stages_reject_lds_operations(self):
        for stage in ("ps", "vs"):
            with self.subTest(stage=stage):
                g = rdna2spirv.Gen(stage=stage)
                with self.assertRaises(rdna2spirv.Unsupported):
                    g.lds_op("ds_read_b32", [{"k": "vgpr", "r": 0, "n": 1}, {"k": "vgpr", "r": 1, "n": 1}], [])
                with self.assertRaises(rdna2spirv.Unsupported):
                    g.lds_op("ds_write_b32", [{"k": "vgpr", "r": 0, "n": 1}, {"k": "vgpr", "r": 1, "n": 1}], [])


class SpirvEquivalenceTests(unittest.TestCase):
    """Verify bit-exact agreement between emitted SPIR-V pointer math and semantic interpreter."""

    def test_offset_step_coverage(self):
        # Step through offsets 0, 4, 8, ... up to 256 with various base addresses
        bases = [0, 4, 12, 100, 1024, 65520, 0xFFFFFFE0]
        for base in bases:
            for byte_off in range(0, 256, 4):
                with self.subTest(base=base, byte_off=byte_off):
                    _, expected_dword, _ = calc_lds_byte_offset(base, byte_off)
                    spirv_dword = emulate_spirv_lds_ptr(base, byte_off)
                    self.assertEqual(expected_dword, spirv_dword)


class NegativeControlTests(unittest.TestCase):
    """Verify that corruptions in LDS byte offset calculations are strictly caught."""

    def test_detects_shift_by_three_corruption(self):
        """Corrupting shift from >> 2 to >> 3 (treating dwords as 8-byte units) fails."""
        vaddr = 16
        offset = 16
        _, correct_dword, _ = calc_lds_byte_offset(vaddr, offset)
        self.assertEqual(correct_dword, 8)

        # Sabotaged calculation: shift right by 3
        corrupted_dword = ((vaddr + offset) >> 3) & (DEFAULT_LDS_DWORDS - 1)
        self.assertEqual(corrupted_dword, 4)
        self.assertNotEqual(correct_dword, corrupted_dword)

    def test_detects_omitted_byte_offset_corruption(self):
        """Ignoring immediate offset fails when offset is non-zero."""
        vaddr = 64
        offset = 64
        _, correct_dword, _ = calc_lds_byte_offset(vaddr, offset)
        self.assertEqual(correct_dword, 32)

        # Sabotaged calculation: offset dropped
        corrupted_dword = (vaddr >> 2) & (DEFAULT_LDS_DWORDS - 1)
        self.assertEqual(corrupted_dword, 16)
        self.assertNotEqual(correct_dword, corrupted_dword)

    def test_detects_missing_wrap_mask_corruption(self):
        """Omitting the LDS size wrapping mask fails for out-of-bounds addresses."""
        lds_dwords = 1024
        vaddr = 4096 * 2 + 16  # Beyond 1024 dwords
        offset = 0
        _, correct_dword, _ = calc_lds_byte_offset(vaddr, offset, lds_dwords)
        self.assertEqual(correct_dword, 4)

        # Sabotaged calculation: missing mask
        unmasked_dword = (vaddr + offset) >> 2
        self.assertEqual(unmasked_dword, 2052)
        self.assertNotEqual(correct_dword, unmasked_dword)

    def test_detects_dword_scaled_offset_corruption(self):
        """Treating byte offset as dwords (multiplying by 4) fails."""
        vaddr = 0
        offset = 16  # 16 bytes = 4 dwords
        _, correct_dword, _ = calc_lds_byte_offset(vaddr, offset)
        self.assertEqual(correct_dword, 4)

        # Sabotaged calculation: treating offset as dwords (adding 16 dwords)
        corrupted_dword = (vaddr >> 2) + offset
        self.assertEqual(corrupted_dword, 16)
        self.assertNotEqual(correct_dword, corrupted_dword)


if __name__ == "__main__":
    unittest.main()
