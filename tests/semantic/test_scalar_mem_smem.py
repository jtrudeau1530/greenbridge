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
  * SMEM loads with a register offset and an immediate at once ("sOFF offset:IMM"): address = base + sOFF + IMM,
    checked by running the emitted code (SmemBodyEvaluator, SmemSoffsetImmTranslatorTests)
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


class SmemBodyEvaluator:
    """Runs the straight-line SPIR-V that Gen.scalar_load emits, against concrete SGPR values and a dword memory.

    spirv_mini_interp refuses 64-bit integers and PhysicalStorageBuffer by design, so the address a scalar load
    reads cannot be checked there. This evaluator knows only the handful of instructions scalar_load uses and
    raises on anything else, so a change in the emitted code cannot be approximated silently. The heap base (push
    constant dwords 32..33) is 0 unless given, which makes every address read an Xbox VA.
    """

    def __init__(self, gen, sgpr, memory, heap_base=0):
        self.consts = {name: value for value, name in gen.consts.items()}
        self.sgpr = dict(sgpr)
        self.memory = memory
        self.heap_base = heap_base
        self.ids = {}
        self.reads = []          # byte address of every PhysicalStorageBuffer dword load, in order
        for line in gen.body:
            self._step(line.split())

    def _value(self, name):
        if name == "%c64_32":
            return 32
        if name in self.ids:
            return self.ids[name]
        if name in self.consts:
            return self.consts[name]
        raise AssertionError(f"unknown id {name}")

    def _step(self, words):
        if words[0] == "OpStore":
            kind, index = self._value(words[1])
            if kind != "sgpr":
                raise AssertionError(f"store through {kind}")
            self.sgpr[index] = self._value(words[2])
            return
        if len(words) < 4 or words[1] != "=":
            raise AssertionError("unmodelled line: " + " ".join(words))
        result, op, ty, args = words[0], words[2], words[3], words[4:]
        if op == "OpAccessChain" and ty == "%_ptr_Function_uint" and args[0] == "%sgpr":
            value = ("sgpr", self._value(args[1]))
        elif op == "OpAccessChain" and ty == "%_ptr_PSB_uint":
            kind, address = self._value(args[0])
            if kind != "psb":
                raise AssertionError(f"PSB access chain on {kind}")
            value = ("mem", (address + 4 * self._value(args[1])) & MASK64)
        elif op == "OpLoad" and ty == "%uint" and args[0] == "%ud_base_lo_p":
            value = self.heap_base & MASK32
        elif op == "OpLoad" and ty == "%uint" and args[0] == "%ud_base_hi_p":
            value = (self.heap_base >> 32) & MASK32
        elif op == "OpLoad" and ty == "%uint":
            kind, where = self._value(args[0])
            if kind == "sgpr":
                value = self.sgpr.get(where, 0) & MASK32
            elif kind == "mem" and args[1:] == ["Aligned", "4"]:
                self.reads.append(where)
                value = self.memory.read_dword(where)
            else:
                raise AssertionError("unmodelled load: " + " ".join(words))
        elif op == "OpBitwiseAnd" and ty == "%uint":
            value = self._value(args[0]) & self._value(args[1])
        elif op == "OpIAdd" and ty == "%uint":
            value = (self._value(args[0]) + self._value(args[1])) & MASK32
        elif op == "OpUConvert" and ty == "%ulong":
            value = self._value(args[0]) & MASK32
        elif op == "OpShiftLeftLogical" and ty == "%ulong":
            value = (self._value(args[0]) << self._value(args[1])) & MASK64
        elif op == "OpBitwiseOr" and ty == "%ulong":
            value = self._value(args[0]) | self._value(args[1])
        elif op == "OpIAdd" and ty == "%ulong":
            value = (self._value(args[0]) + self._value(args[1])) & MASK64
        elif op == "OpConvertUToPtr" and ty == "%_ptr_PSB_arr":
            value = ("psb", self._value(args[0]))
        else:
            raise AssertionError("unmodelled instruction: " + " ".join(words))
        self.ids[result] = value


class SmemSoffsetImmTranslatorTests(unittest.TestCase):
    """SMEM loads that carry a register offset and an immediate at once: llvm-mc prints "sOFF offset:IMM".

    Rule: address = base + soffset + imm (+ 4*i for dword i).
    Measured on an AMD Radeon RX 6800 (gfx1030) by running s_buffer_load_dword/x2/x4/x8 as a raw compute kernel:
    soffset 8 + imm 4 reads the dword at +12. s_load_dword* with both offsets was not measured; it shares the SMEM
    encoding (soffset_en + imm), so the same rule is applied to it as an inference.
    The hardware also truncates base, soffset and imm each to a multiple of 4 and returns 0 past NUM_RECORDS; the
    translator does neither, so every case here is dword-aligned and in bounds.

    The instruction lines are hand-written. Each case runs the code Gen.scalar_load emits with SmemBodyEvaluator
    and asserts both the addresses read and the values that land in the destination SGPRs.
    """

    VSHARP_BASE = 0x0000000200001000        # s24 = 0x00001000, s25 = 0x80040002 (flags above bit 15 are not address)
    POINTER = 0x0000000300004000            # s[24:25] for s_load

    def run_load(self, line, pointer=False):
        ins = parse_line(line)
        g = rdna2spirv.Gen("ps")
        g.scalar_load(ins["op"], ins["args"], ins["mods"])
        base = self.POINTER if pointer else self.VSHARP_BASE
        bank = SmemMemoryBank()
        for off in range(0, 0x40, 4):       # every dword near the base holds its own offset, tagged
            bank.write_dword(base + off, 0x5EED0000 | off)
        sgpr = {40: 8}                      # the register offset used by every case
        if pointer:
            sgpr.update({24: 0x00004000, 25: 0x00000003})
        else:
            sgpr.update({24: 0x00001000, 25: 0x80040002, 26: 0x00010000, 27: 0x00140000})
        ev = SmemBodyEvaluator(g, sgpr, bank)
        return ev, [a - base for a in ev.reads], g

    def test_s_buffer_load_dword_soffset_plus_imm(self):
        # The measured case: soffset 8 + imm 4 reads the dword at +12 (not +8, which is what dropping the imm gives).
        ev, offsets, _ = self.run_load("s_buffer_load_dword s12, s[24:27], s40 offset:0x4")
        self.assertEqual(offsets, [12])
        self.assertEqual(ev.sgpr[12], 0x5EED000C)

    def test_s_load_dword_soffset_plus_imm(self):
        # Inferred from the shared SMEM encoding; not measured on hardware.
        ev, offsets, _ = self.run_load("s_load_dword s12, s[24:25], s40 offset:0x4", pointer=True)
        self.assertEqual(offsets, [12])
        self.assertEqual(ev.sgpr[12], 0x5EED000C)

    def test_multi_dword_soffset_plus_imm(self):
        cases = [
            ("s_buffer_load_dwordx2 s[12:13], s[24:27], s40 offset:0x4", False, 2, 0x0C),
            ("s_buffer_load_dwordx4 s[12:15], s[24:27], s40 offset:0x10", False, 4, 0x18),
            ("s_buffer_load_dwordx8 s[12:19], s[24:27], s40 offset:0x14", False, 8, 0x1C),
            ("s_load_dwordx2 s[12:13], s[24:25], s40 offset:0x4", True, 2, 0x0C),
            ("s_load_dwordx4 s[12:15], s[24:25], s40 offset:0x10", True, 4, 0x18),
        ]
        for line, pointer, n, first in cases:
            with self.subTest(line=line):
                ev, offsets, _ = self.run_load(line, pointer)
                self.assertEqual(offsets, [first + 4 * i for i in range(n)])
                self.assertEqual([ev.sgpr[12 + i] for i in range(n)], [0x5EED0000 | (first + 4 * i) for i in range(n)])

    def test_imm_only_and_soffset_only_addresses_unchanged(self):
        cases = [
            ("s_buffer_load_dword s12, s[24:27], 0x4", False, [0x04]),
            ("s_buffer_load_dword s12, s[24:27], s40", False, [0x08]),
            ("s_buffer_load_dwordx2 s[12:13], s[24:27], 0x10", False, [0x10, 0x14]),
            ("s_buffer_load_dwordx4 s[12:15], s[24:27], s40", False, [0x08, 0x0C, 0x10, 0x14]),
            ("s_load_dword s12, s[24:25], 0x4", True, [0x04]),
            ("s_load_dword s12, s[24:25], s40", True, [0x08]),
            ("s_load_dwordx2 s[12:13], s[24:25], 0", True, [0x00, 0x04]),
            ("s_load_dwordx2 s[12:13], s[24:25], s40", True, [0x08, 0x0C]),
        ]
        for line, pointer, want in cases:
            with self.subTest(line=line):
                ev, offsets, _ = self.run_load(line, pointer)
                self.assertEqual(offsets, want)
                self.assertEqual([ev.sgpr[12 + i] for i in range(len(want))], [0x5EED0000 | o for o in want])

    def test_zero_imm_with_soffset_reads_the_soffset_address(self):
        ev, offsets, _ = self.run_load("s_buffer_load_dword s12, s[24:27], s40 offset:0x0")
        self.assertEqual(offsets, [8])
        self.assertEqual(ev.sgpr[12], 0x5EED0008)

    def test_non_sgpr_scalar_register_offset_plus_imm(self):
        # llvm-mc also accepts m0, vcc_lo or null as the register offset. null reads as zero; m0 is not an SGPR slot,
        # so only check that the form still translates and that the immediate changes the code.
        ev, offsets, _ = self.run_load("s_buffer_load_dword s12, s[24:27], null offset:0x14")
        self.assertEqual(offsets, [0x14])
        self.assertEqual(ev.sgpr[12], 0x5EED0014)
        bodies = []
        for line in ("s_buffer_load_dword s12, s[24:27], m0", "s_buffer_load_dword s12, s[24:27], m0 offset:0x4"):
            ins = parse_line(line)
            g = rdna2spirv.Gen("ps")
            g.scalar_load(ins["op"], ins["args"], ins["mods"])
            bodies.append(g.body)
        self.assertNotEqual(bodies[0], bodies[1])

    def test_offset_modifier_changes_the_emitted_code(self):
        for line, pointer in (("s_buffer_load_dword s12, s[24:27], s40", False),
                              ("s_load_dword s12, s[24:25], s40", True)):
            with self.subTest(line=line):
                _, _, plain = self.run_load(line, pointer)
                _, _, both = self.run_load(line + " offset:0x4", pointer)
                self.assertNotEqual(plain.body, both.body)

    def test_heap_base_is_added_once(self):
        ins = parse_line("s_buffer_load_dword s12, s[24:27], s40 offset:0x4")
        g = rdna2spirv.Gen("ps")
        g.scalar_load(ins["op"], ins["args"], ins["mods"])
        ev = SmemBodyEvaluator(g, {24: 0x1000, 25: 0, 40: 8}, SmemMemoryBank(), heap_base=0x0000007F00000000)
        self.assertEqual(ev.reads, [0x0000007F00000000 + 0x1000 + 12])

    def test_offset_forms_the_assembler_rejects_raise(self):
        # llvm-mc accepts none of these; hand-written or damaged listings must not translate to a guessed address.
        for line in ("s_buffer_load_dword s12, s[24:27], 0x8 offset:0x4",       # immediate operand and offset modifier
                     "s_buffer_load_dword s12, s[24:27], s40 offset:zz",        # unparseable immediate
                     "s_buffer_load_dword s12, s[24:27], s40 offset:0x4 offset:0x8",
                     "s_buffer_load_dword s12, s[24:27], s40, 0x4",             # a fourth operand
                     "s_load_dword s12, s[24:25], 0x8 offset:0x4"):
            with self.subTest(line=line):
                ins = parse_line(line)
                g = rdna2spirv.Gen("ps")
                with self.assertRaises(rdna2spirv.Unsupported):
                    g.scalar_load(ins["op"], ins["args"], ins["mods"])


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
