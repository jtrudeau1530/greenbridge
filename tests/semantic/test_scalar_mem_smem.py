"""Semantic interpreter tests for scalar memory read offsets and descriptor address calculation (s_load_dword).

Covers AMD RDNA2 / GFX10 SMEM (Scalar Memory) ISA semantics and tools/rdna2spirv.py translation:
  * 64-bit base pointer reconstruction from SGPR pair s[p:p+1]
  * Buffer Resource Descriptor (V#) 128-bit layout and base address extraction:
      - Dword 0: Base address low 32 bits [31:0]
      - Dword 1: Base address high 16 bits [47:32] in bits [15:0]; bits [31:16] are format/stride/flags
      - Proper masking (dword1 & 0xFFFF) preventing descriptor flags from polluting the 64-bit pointer
  * Read offset calculations:
      - Zero offset, immediate byte offsets, and SGPR dynamic offsets
      - 32-bit carry propagation across the 4GB boundary into the upper 32 bits
      - 64-bit address wrapping modulo 2^64
  * Multi-dword scalar loads (s_load_dword, s_load_dwordx2, x4, x8, x16) with sequential 4-byte address strides
  * Destination register targeting: SGPR ranges and special registers (vcc_lo, vcc_hi, vcc pair, m0)
  * tools/rdna2spirv.py translator validation (Gen.scalar_load instruction sequences and operand validation)
  * spirv_mini_interp strictness validation (rejecting unmodelled 64-bit types and PhysicalStorageBuffer)
"""
import os
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
from shader_ir import parse_line  # noqa: E402
import spirv_mini_interp as si  # noqa: E402

MASK32 = 0xFFFFFFFF
MASK64 = 0xFFFFFFFFFFFFFFFF


def compute_smem_pointer_address(base_lo: int, base_hi: int, offset: int) -> int:
    """Compute 64-bit effective address for s_load_dword from 64-bit pointer and offset."""
    base_64 = ((base_hi & MASK32) << 32) | (base_lo & MASK32)
    return (base_64 + (offset & MASK32)) & MASK64


def compute_smem_descriptor_address(desc_dw0: int, desc_dw1: int, offset: int) -> int:
    """Compute 64-bit effective address for s_buffer_load_dword from buffer descriptor V# and offset.

    Per AMD RDNA2 ISA:
      - dword0: Base address [31:0]
      - dword1: Base address [47:32] in bits [15:0]. Bits [31:16] are format/stride metadata.
    """
    base_hi = desc_dw1 & 0xFFFF
    base_64 = (base_hi << 32) | (desc_dw0 & MASK32)
    return (base_64 + (offset & MASK32)) & MASK64


class SmemMemoryBank:
    """Deterministic simulated memory bank for SMEM load validation."""

    def __init__(self):
        self._mem = {}

    def write_dword(self, byte_addr: int, val: int) -> None:
        self._mem[byte_addr & MASK64] = val & MASK32

    def read_dword(self, byte_addr: int) -> int:
        return self._mem.get(byte_addr & MASK64, 0)


class SmemReferenceInterpreter:
    """Reference execution engine for RDNA2 scalar memory loads."""

    def __init__(self, memory: SmemMemoryBank):
        self.mem = memory
        self.sgpr = [0] * 106
        self.vcc_lo = 0
        self.vcc_hi = 0
        self.m0 = 0

    def load_register(self, name: str, reg_idx: int = 0) -> int:
        if name == "sgpr":
            return self.sgpr[reg_idx] & MASK32
        if name == "vcc_lo":
            return self.vcc_lo & MASK32
        if name == "vcc_hi":
            return self.vcc_hi & MASK32
        if name == "m0":
            return self.m0 & MASK32
        raise ValueError(f"unknown register {name}")

    def store_register(self, name: str, reg_idx: int, val: int) -> None:
        val &= MASK32
        if name == "sgpr":
            self.sgpr[reg_idx] = val
        elif name == "vcc_lo":
            self.vcc_lo = val
        elif name == "vcc_hi":
            self.vcc_hi = val
        elif name == "m0":
            self.m0 = val
        else:
            raise ValueError(f"unknown register {name}")

    def execute_s_load(self, count: int, dest_reg: str, dest_idx: int,
                       base_lo_idx: int, offset: int) -> int:
        lo = self.load_register("sgpr", base_lo_idx)
        hi = self.load_register("sgpr", base_lo_idx + 1)
        addr = compute_smem_pointer_address(lo, hi, offset)
        for i in range(count):
            dword_val = self.mem.read_dword(addr + i * 4)
            if dest_reg == "sgpr":
                self.store_register("sgpr", dest_idx + i, dword_val)
            elif dest_reg == "vcc":
                if i == 0:
                    self.store_register("vcc_lo", 0, dword_val)
                elif i == 1:
                    self.store_register("vcc_hi", 0, dword_val)
            elif dest_reg in ("vcc_lo", "vcc_hi", "m0"):
                self.store_register(dest_reg, 0, dword_val)
        return addr

    def execute_s_buffer_load(self, count: int, dest_reg: str, dest_idx: int,
                              desc_dw0_idx: int, offset: int) -> int:
        dw0 = self.load_register("sgpr", desc_dw0_idx)
        dw1 = self.load_register("sgpr", desc_dw0_idx + 1)
        addr = compute_smem_descriptor_address(dw0, dw1, offset)
        for i in range(count):
            dword_val = self.mem.read_dword(addr + i * 4)
            if dest_reg == "sgpr":
                self.store_register("sgpr", dest_idx + i, dword_val)
            elif dest_reg == "vcc":
                if i == 0:
                    self.store_register("vcc_lo", 0, dword_val)
                elif i == 1:
                    self.store_register("vcc_hi", 0, dword_val)
            elif dest_reg in ("vcc_lo", "vcc_hi", "m0"):
                self.store_register(dest_reg, 0, dword_val)
        return addr


class SmemAddressOracleTests(unittest.TestCase):
    """Verifies SMEM 64-bit effective address calculations and V# descriptor masking."""

    def test_s_load_pointer_address_basic(self):
        cases = [
            (0x1000, 0x0, 0, 0x1000),
            (0x1000, 0x0, 16, 0x1010),
            (0x1000, 0x0, 0x40, 0x1040),
            (0x2000, 0x1, 0x100, 0x100002100),
            (0x00000000, 0x00000008, 0x200, 0x800000200),
        ]
        for lo, hi, off, expected in cases:
            with self.subTest(lo=hex(lo), hi=hex(hi), off=hex(off)):
                self.assertEqual(compute_smem_pointer_address(lo, hi, off), expected)

    def test_s_load_pointer_carry_into_upper_32_bits(self):
        # Adding offset to low dword crossing 0xFFFFFFFF boundary must carry into hi dword.
        cases = [
            (0xFFFFFFFC, 0x00000001, 8, 0x0000000200000004),
            (0xFFFFFFFF, 0x00000000, 1, 0x0000000100000000),
            (0xFFFFFFF0, 0x00000005, 0x20, 0x0000000600000010),
            (0xFFFFFFFF, 0x0000000F, 0xFFFFFFFF, 0x00000010FFFFFFFE),
        ]
        for lo, hi, off, expected in cases:
            with self.subTest(lo=hex(lo), hi=hex(hi), off=hex(off)):
                self.assertEqual(compute_smem_pointer_address(lo, hi, off), expected)

    def test_s_load_pointer_wrap_modulo_64(self):
        # 64-bit overflow wraps around to 0
        self.assertEqual(compute_smem_pointer_address(0xFFFFFFF0, 0xFFFFFFFF, 0x20), 0x0000000000000010)
        self.assertEqual(compute_smem_pointer_address(0xFFFFFFFF, 0xFFFFFFFF, 1), 0)

    def test_s_buffer_load_descriptor_masking(self):
        # Bits [31:16] of dword 1 contain format/stride/cache attributes;
        # only bits [15:0] participate in base address formation.
        cases = [
            # dw0, dw1, offset, expected_addr
            (0x00001000, 0x00000002, 0x10, 0x0000000200001010),
            # dw1 has 0x8004 in bits [31:16] (typical buffer resource metadata)
            (0x00001000, 0x80040002, 0x10, 0x0000000200001010),
            # dw1 has all upper 16 bits set (0xFFFF0000)
            (0x12345678, 0xFFFF0007, 0x0, 0x0000000712345678),
            # dw1 has format bits set, address bits 0
            (0x00000040, 0x3F000000, 0x20, 0x0000000000000060),
            # Carry from dw0 into masked dw1
            (0xFFFFFFF8, 0x80000001, 0x10, 0x0000000200000008),
        ]
        for dw0, dw1, off, expected in cases:
            with self.subTest(dw0=hex(dw0), dw1=hex(dw1), off=hex(off)):
                self.assertEqual(compute_smem_descriptor_address(dw0, dw1, off), expected)


class SmemExecutionTests(unittest.TestCase):
    """Verifies multi-dword scalar loads, sequential strides, and register targets."""

    def setUp(self):
        self.bank = SmemMemoryBank()
        self.interp = SmemReferenceInterpreter(self.bank)

    def test_s_load_dword_single(self):
        # Base s[2:3] = 0x1_00000000, offset = 0x20 -> address 0x1_00000020
        self.interp.store_register("sgpr", 2, 0x00000000)
        self.interp.store_register("sgpr", 3, 0x00000001)
        self.bank.write_dword(0x100000020, 0xCAFEBABE)

        addr = self.interp.execute_s_load(1, "sgpr", 4, 2, 0x20)
        self.assertEqual(addr, 0x100000020)
        self.assertEqual(self.interp.load_register("sgpr", 4), 0xCAFEBABE)

    def test_s_load_dwordx2_sequential_stride(self):
        self.interp.store_register("sgpr", 6, 0x00001000)
        self.interp.store_register("sgpr", 7, 0x00000000)
        self.bank.write_dword(0x1000, 0x11112222)
        self.bank.write_dword(0x1004, 0x33334444)

        addr = self.interp.execute_s_load(2, "sgpr", 8, 6, 0)
        self.assertEqual(addr, 0x1000)
        self.assertEqual(self.interp.load_register("sgpr", 8), 0x11112222)
        self.assertEqual(self.interp.load_register("sgpr", 9), 0x33334444)

    def test_s_load_dwordx4_sequential_stride(self):
        self.interp.store_register("sgpr", 0, 0x00002000)
        self.interp.store_register("sgpr", 1, 0x00000000)
        expected_words = [0xA0000001, 0xA0000002, 0xA0000003, 0xA0000004]
        for i, val in enumerate(expected_words):
            self.bank.write_dword(0x2010 + i * 4, val)

        addr = self.interp.execute_s_load(4, "sgpr", 12, 0, 0x10)
        self.assertEqual(addr, 0x2010)
        for i, val in enumerate(expected_words):
            self.assertEqual(self.interp.load_register("sgpr", 12 + i), val)

    def test_s_load_dwordx8_sequential_stride(self):
        self.interp.store_register("sgpr", 0, 0x00004000)
        self.interp.store_register("sgpr", 1, 0x00000000)
        for i in range(8):
            self.bank.write_dword(0x4000 + i * 4, 0xB0000000 | i)

        addr = self.interp.execute_s_load(8, "sgpr", 16, 0, 0)
        self.assertEqual(addr, 0x4000)
        for i in range(8):
            self.assertEqual(self.interp.load_register("sgpr", 16 + i), 0xB0000000 | i)

    def test_s_load_dwordx16_sequential_stride(self):
        self.interp.store_register("sgpr", 0, 0x00008000)
        self.interp.store_register("sgpr", 1, 0x00000000)
        for i in range(16):
            self.bank.write_dword(0x8000 + i * 4, 0xC0000000 | i)

        addr = self.interp.execute_s_load(16, "sgpr", 32, 0, 0)
        self.assertEqual(addr, 0x8000)
        for i in range(16):
            self.assertEqual(self.interp.load_register("sgpr", 32 + i), 0xC0000000 | i)

    def test_s_buffer_load_dword_with_descriptor_masking(self):
        # V# descriptor: dw0 = 0x5000, dw1 = 0x80040003 (stride=0x8004, hi_addr=3)
        self.interp.store_register("sgpr", 8, 0x00005000)
        self.interp.store_register("sgpr", 9, 0x80040003)
        self.interp.store_register("sgpr", 10, 0x00010000)  # num_records
        self.interp.store_register("sgpr", 11, 0x00140000)  # flags

        expected_addr = 0x0000000300005040
        self.bank.write_dword(expected_addr, 0x99887766)

        addr = self.interp.execute_s_buffer_load(1, "sgpr", 20, 8, 0x40)
        self.assertEqual(addr, expected_addr)
        self.assertEqual(self.interp.load_register("sgpr", 20), 0x99887766)

    def test_special_destination_registers(self):
        self.interp.store_register("sgpr", 0, 0x1000)
        self.interp.store_register("sgpr", 1, 0x0)
        self.bank.write_dword(0x1000, 0x12345678)
        self.bank.write_dword(0x1004, 0x9ABCDEF0)

        # Single dword to vcc_lo
        self.interp.execute_s_load(1, "vcc_lo", 0, 0, 0)
        self.assertEqual(self.interp.load_register("vcc_lo"), 0x12345678)

        # Single dword to vcc_hi
        self.interp.execute_s_load(1, "vcc_hi", 0, 0, 4)
        self.assertEqual(self.interp.load_register("vcc_hi"), 0x9ABCDEF0)

        # Single dword to m0
        self.interp.execute_s_load(1, "m0", 0, 0, 0)
        self.assertEqual(self.interp.load_register("m0"), 0x12345678)

        # Two dwords to vcc pair (low to vcc_lo, high to vcc_hi)
        self.interp.execute_s_load(2, "vcc", 0, 0, 0)
        self.assertEqual(self.interp.load_register("vcc_lo"), 0x12345678)
        self.assertEqual(self.interp.load_register("vcc_hi"), 0x9ABCDEF0)


class SmemTranslatorGenTests(unittest.TestCase):
    """Tests tools/rdna2spirv.py Gen.scalar_load SPIR-V emission structure and validation rules."""

    def test_s_load_dword_emits_64bit_shift_or_add(self):
        g = rdna2spirv.Gen("ps")
        ins = parse_line("s_load_dword s0, s[2:3], 0x10")
        g.scalar_load(ins["op"], ins["args"], ins["mods"])
        text = "\n".join(g.body)

        self.assertIn("OpShiftLeftLogical %ulong", text)
        self.assertIn("OpBitwiseOr %ulong", text)
        self.assertIn("OpIAdd %ulong", text)
        self.assertIn("OpConvertUToPtr %_ptr_PSB_arr", text)
        self.assertIn("OpAccessChain %_ptr_PSB_uint", text)
        self.assertIn("OpLoad %uint", text)
        self.assertIn("Aligned 4", text)
        self.assertTrue(g.uses_psb)

    def test_s_buffer_load_dword_emits_cffff_mask(self):
        g = rdna2spirv.Gen("ps")
        ins = parse_line("s_buffer_load_dword s4, s[8:11], 0x20")
        g.scalar_load(ins["op"], ins["args"], ins["mods"])
        text = "\n".join(g.body)

        self.assertIn("OpBitwiseAnd %uint", text)
        self.assertIn("%cffff", text)
        self.assertIn("OpConvertUToPtr %_ptr_PSB_arr", text)
        self.assertTrue(g.uses_psb)

    def test_multi_dword_emission_counts(self):
        cases = [
            ("s_load_dword s0, s[2:3], 0", 1),
            ("s_load_dwordx2 s[0:1], s[2:3], 0", 2),
            ("s_load_dwordx4 s[0:3], s[4:5], 0", 4),
            ("s_load_dwordx8 s[0:7], s[8:9], 0", 8),
            ("s_load_dwordx16 s[0:15], s[16:17], 0", 16),
            ("s_buffer_load_dword s0, s[4:7], 0", 1),
            ("s_buffer_load_dwordx4 s[0:3], s[8:11], 0", 4),
            ("s_buffer_load_dwordx8 s[0:7], s[8:11], 0", 8),
        ]
        for line, expected_n in cases:
            with self.subTest(line=line):
                g = rdna2spirv.Gen("ps")
                ins = parse_line(line)
                g.scalar_load(ins["op"], ins["args"], ins["mods"])
                psb_chains = [ln for ln in g.body if "OpAccessChain %_ptr_PSB_uint" in ln]
                psb_loads = [ln for ln in g.body if "OpLoad %uint" in ln and "Aligned 4" in ln]
                self.assertEqual(len(psb_chains), expected_n)
                self.assertEqual(len(psb_loads), expected_n)

    def test_special_destinations_emission(self):
        for line in ("s_buffer_load_dword vcc_lo, s[0:3], 0",
                     "s_buffer_load_dword vcc_hi, s[0:3], 4",
                     "s_buffer_load_dword m0, s[0:3], 8",
                     "s_load_dwordx2 vcc, s[0:1], 0"):
            with self.subTest(line=line):
                g = rdna2spirv.Gen("ps")
                ins = parse_line(line)
                g.scalar_load(ins["op"], ins["args"], ins["mods"])
                self.assertTrue(g.uses_psb)

    def test_unsupported_operands_raise_cleanly(self):
        # Base must have n=2 for s_load
        g = rdna2spirv.Gen("ps")
        ins_bad_sload_base = parse_line("s_load_dword s0, s[0:3], 0")
        with self.assertRaises(rdna2spirv.Unsupported) as ctx:
            g.scalar_load(ins_bad_sload_base["op"], ins_bad_sload_base["args"], ins_bad_sload_base["mods"])
        self.assertIn("s_load without 64-bit pointer", str(ctx.exception))

        # Base must have n=4 for s_buffer_load
        ins_bad_sbuf_base = parse_line("s_buffer_load_dword s0, s[0:1], 0")
        with self.assertRaises(rdna2spirv.Unsupported) as ctx:
            g.scalar_load(ins_bad_sbuf_base["op"], ins_bad_sbuf_base["args"], ins_bad_sbuf_base["mods"])
        self.assertIn("s_buffer_load without V#", str(ctx.exception))

        # Non-SGPR base must raise
        ins_vgpr_base = parse_line("s_load_dword s0, v[0:1], 0")
        with self.assertRaises(rdna2spirv.Unsupported):
            g.scalar_load(ins_vgpr_base["op"], ins_vgpr_base["args"], ins_vgpr_base["mods"])


class SmemMiniInterpStrictnessTests(unittest.TestCase):
    """Verifies that spirv_mini_interp rejects storage classes and constructs it does not model.

    As documented in tests/semantic/README.md, the mini-interpreter shares no code with the translator
    and raises UnsupportedConstruct for anything not explicitly modelled (such as physical-storage-buffer
    OpVariables, odd integer widths, or pointer-to-int casts), guaranteeing that translator regressions fail
    tests immediately. 64-bit integers, OpConvertUToPtr and physical-storage-buffer loads are modelled; their
    positive paths are covered in test_interp.py and test_value_semantics.py.
    """

    def test_rejects_unsupported_integer_widths(self):
        module = """\
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main"
%void = OpTypeVoid
%ushort = OpTypeInt 16 0
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
OpReturn
OpFunctionEnd
"""
        with self.assertRaises(si.UnsupportedConstruct) as ctx:
            si.run_fragment(module)
        self.assertIn("integer type that is not 32 or 64 bits", str(ctx.exception))

    def test_rejects_physical_storage_buffer_storage_class(self):
        module = """\
OpCapability Shader
OpCapability PhysicalStorageBufferAddresses
OpMemoryModel PhysicalStorageBuffer64 GLSL450
OpEntryPoint Fragment %main "main"
%void = OpTypeVoid
%uint = OpTypeInt 32 0
%_ptr_PSB = OpTypePointer PhysicalStorageBuffer %uint
%var = OpVariable %_ptr_PSB PhysicalStorageBuffer
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
OpReturn
OpFunctionEnd
"""
        with self.assertRaises(si.UnsupportedConstruct) as ctx:
            si.run_fragment(module)
        self.assertIn("storage class PhysicalStorageBuffer", str(ctx.exception))

    def test_rejects_convert_ptr_to_u_in_body(self):
        module = """\
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main"
%void = OpTypeVoid
%uint = OpTypeInt 32 0
%_ptr_Func = OpTypePointer Function %uint
%fn = OpTypeFunction %void
%c1 = OpConstant %uint 1
%main = OpFunction %void None %fn
%entry = OpLabel
%p = OpConvertUToPtr %_ptr_Func %c1
%u = OpConvertPtrToU %uint %p
OpReturn
OpFunctionEnd
"""
        with self.assertRaises(si.UnsupportedConstruct) as ctx:
            si.run_fragment(module)
        self.assertIn("OpConvertPtrToU", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
