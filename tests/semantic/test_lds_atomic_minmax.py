"""Semantic interpreter tests for LDS atomic min/max (ds_min_u32, ds_max_u32, ds_min_i32, ds_max_i32).

Standard library only; no GPU, no LLVM, no external tools, no game files.

    python -I tests/semantic/test_lds_atomic_minmax.py
    python -m unittest discover -s tests/semantic -v

Background & Specification Rules:
  * AMD RDNA2 / GFX10 Local Data Share (LDS) Architecture:
    - LDS is an on-chip software-managed scratchpad memory shared across all threads
      in a workgroup (threadgroup).
    - Unsigned Atomics:
      - `ds_min_u32 vaddr, vdata [offset:N]`:
          Atomically computes:
            addr = (vaddr + byte_offset)
            idx = (addr >> 2) & (lds_dwords - 1)
            LDS[idx] = min_u32(LDS[idx], vdata)
          Non-returning form: the prior value at LDS[idx] is discarded.
      - `ds_min_rtn_u32 vdst, vaddr, vdata [offset:N]`:
          Returning form: writes prior LDS[idx] into destination VGPR vdst and
          stores min_u32(old, vdata) in LDS[idx].
      - `ds_max_u32 vaddr, vdata [offset:N]` / `ds_max_rtn_u32 vdst, vaddr, vdata [offset:N]`:
          Unsigned maximum, computing max_u32(LDS[idx], vdata).
    - Signed Atomics (Two's-Complement 32-bit):
      - `ds_min_i32 vaddr, vdata [offset:N]` / `ds_min_rtn_i32 vdst, vaddr, vdata [offset:N]`:
          Interprets operands as signed two's-complement 32-bit integers in [-2^31, 2^31 - 1],
          computing min_i32(LDS[idx], vdata). E.g. -1 (0xFFFFFFFF) < 0 (0x00000000).
      - `ds_max_i32 vaddr, vdata [offset:N]` / `ds_max_rtn_i32 vdst, vaddr, vdata [offset:N]`:
          Signed maximum, computing max_i32(LDS[idx], vdata).
    - Active lane masking:
      - LDS atomic instructions execute under execution mask control (%exec_active).
        Only active lanes participate in memory modifications and destination register writes.
    - Hardware wrapping & alignment:
      - Byte addresses are shifted right by 2, dropping unaligned low bits [1:0].
      - The resulting dword index is masked with (lds_dwords - 1), wrapping out-of-bounds
        offsets within the allocated power-of-two LDS slice.
  * SPIR-V Representation:
    - LDS is declared as an array in the Workgroup storage class:
        %lds_arr = OpTypeArray %uint <lds_dwords>
        %_ptr_WG_lds = OpTypePointer Workgroup %lds_arr
        %_ptr_WG_uint = OpTypePointer Workgroup %uint
        %lds = OpVariable %_ptr_WG_lds Workgroup
    - Atomic min/max instructions map directly to SPIR-V core opcodes:
        ds_min_u32 / ds_min_rtn_u32 -> OpAtomicUMin %uint %ptr %scope %semantics %val
        ds_max_u32 / ds_max_rtn_u32 -> OpAtomicUMax %uint %ptr %scope %semantics %val
        ds_min_i32 / ds_min_rtn_i32 -> OpAtomicSMin %uint %ptr %scope %semantics %val
        ds_max_i32 / ds_max_rtn_i32 -> OpAtomicSMax %uint %ptr %scope %semantics %val
      where %scope is Workgroup (2) and %semantics is Relaxed (0).
"""
from __future__ import annotations

import os
import re
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TOOLS_DIR = os.path.join(ROOT, "tools")
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)

import rdna2spirv  # noqa: E402

MASK = 0xFFFFFFFF
SIGN_BIT = 0x80000000
INT32_MIN_SIGNED = -0x80000000
INT32_MAX_SIGNED = 0x7FFFFFFF


class UnsupportedConstruct(NotImplementedError):
    """The module uses an unmodelled SPIR-V opcode, type, or storage class."""


class WorkgroupBoundsError(ValueError):
    """Workgroup dimension or memory bounds violation."""


_IGNORED = frozenset({
    "OpCapability", "OpExtInstImport", "OpMemoryModel", "OpEntryPoint", "OpExecutionMode",
    "OpDecorate", "OpMemberDecorate", "OpTypeVoid", "OpTypeFunction", "OpLabel",
    "OpReturn", "OpFunctionEnd", "OpName", "OpSource", "OpSelectionMerge",
})

_LINE = re.compile(r"^(?:(%\S+)\s*=\s*)?(Op\w+)\s*(.*)$")


def to_signed_i32(val: int) -> int:
    """Interprets a 32-bit unsigned bit pattern as a signed two's-complement integer."""
    u = val & MASK
    return u - 0x100000000 if u & SIGN_BIT else u


def to_unsigned_u32(val: int) -> int:
    """Converts a signed or unsigned integer to a 32-bit unsigned bit pattern."""
    return val & MASK


def reference_atomic_umin(initial: int, val: int) -> int:
    """Reference 32-bit unsigned atomic minimum."""
    u_init = initial & MASK
    u_val = val & MASK
    return min(u_init, u_val)


def reference_atomic_umax(initial: int, val: int) -> int:
    """Reference 32-bit unsigned atomic maximum."""
    u_init = initial & MASK
    u_val = val & MASK
    return max(u_init, u_val)


def reference_atomic_smin(initial: int, val: int) -> int:
    """Reference 32-bit signed atomic minimum returning 32-bit bit pattern."""
    s_init = to_signed_i32(initial)
    s_val = to_signed_i32(val)
    return to_unsigned_u32(min(s_init, s_val))


def reference_atomic_smax(initial: int, val: int) -> int:
    """Reference 32-bit signed atomic maximum returning 32-bit bit pattern."""
    s_init = to_signed_i32(initial)
    s_val = to_signed_i32(val)
    return to_unsigned_u32(max(s_init, s_val))


def reference_lds_index(base_addr: int, byte_offset: int, lds_dwords: int) -> int:
    """Reference dword index calculation with hardware wrapping."""
    effective_byte = (base_addr + byte_offset) & MASK
    dword_idx = effective_byte >> 2
    return dword_idx & (lds_dwords - 1)


class LdsAtomicMinMaxInterp:
    """Strict semantic interpreter for compute SPIR-V modules exercising LDS min/max atomics."""

    def __init__(self, spvasm: str):
        self.spvasm = spvasm
        self.types: dict[str, tuple] = {}
        self.constants: dict[str, int | bool] = {}
        self.ptr_types: dict[str, tuple[str, str]] = {}
        self.var_class: dict[str, str] = {}
        self.local_size: tuple[int, int, int] = (1, 1, 1)
        self.lds_var: str | None = None
        self.lds_size: int = 0
        self.body: list[tuple[str | None, str, list[str], str]] = []
        self._parse()

    def _parse(self) -> None:
        in_body = False
        for raw in self.spvasm.splitlines():
            line = raw.strip()
            if not line:
                continue
            m = _LINE.match(line)
            if not m:
                raise UnsupportedConstruct(f"Cannot parse line: {line!r}")
            res, op, rest = m.groups()
            a = rest.split()
            if in_body:
                self.body.append((res, op, a, line))
                continue
            if op == "OpFunction":
                in_body = True
                continue
            if op == "OpExecutionMode" and len(a) >= 5 and a[1] == "LocalSize":
                lx, ly, lz = int(a[2]), int(a[3]), int(a[4])
                if lx < 1 or ly < 1 or lz < 1 or (lx * ly * lz) > 1024:
                    raise WorkgroupBoundsError(f"Invalid LocalSize {lx}x{ly}x{lz}")
                self.local_size = (lx, ly, lz)
                continue
            if op in _IGNORED:
                continue
            if op == "OpTypeInt":
                if int(a[0]) != 32:
                    raise UnsupportedConstruct("Only 32-bit int supported")
                self.types[res] = ("int", int(a[0]), int(a[1]))
            elif op == "OpTypeBool":
                self.types[res] = ("bool",)
            elif op == "OpTypeVector":
                self.types[res] = ("vec", a[0], int(a[1]))
            elif op == "OpTypeArray":
                size = int(self.constants[a[1]])
                self.types[res] = ("arr", a[0], size)
            elif op == "OpTypePointer":
                self.ptr_types[res] = (a[0], a[1])
            elif op == "OpConstant":
                self.constants[res] = int(a[1], 0) & MASK
            elif op == "OpConstantTrue":
                self.constants[res] = True
            elif op == "OpConstantFalse":
                self.constants[res] = False
            elif op == "OpVariable":
                cls, pointee = self.ptr_types[a[0]]
                if a[1] != cls:
                    raise UnsupportedConstruct("OpVariable storage class mismatch")
                self.var_class[res] = cls
                if cls == "Workgroup":
                    if self.lds_var is not None:
                        raise UnsupportedConstruct("Multiple Workgroup variables")
                    self.lds_var = res
                    arr_ty = self.types[pointee]
                    if arr_ty[0] != "arr":
                        raise UnsupportedConstruct("Workgroup variable must be an array")
                    self.lds_size = arr_ty[2]
                elif cls not in ("Output", "Function"):
                    raise UnsupportedConstruct(f"Unsupported storage class {cls}")
            else:
                raise UnsupportedConstruct(f"Header opcode {op}")

    def run(
        self,
        num_invocations: int | None = None,
        seed_lds: dict[int, int] | None = None,
        lane_inputs: list[dict[str, int]] | None = None,
    ) -> tuple[list[int], list[dict[str, int]]]:
        """Executes the workgroup and returns (final_lds, outputs_per_lane)."""
        lx, ly, lz = self.local_size
        total_invocations = lx * ly * lz
        count = total_invocations if num_invocations is None else min(num_invocations, total_invocations)

        lds = [0] * self.lds_size
        if seed_lds:
            for idx, val in seed_lds.items():
                if 0 <= idx < self.lds_size:
                    lds[idx] = val & MASK

        lane_outputs: list[dict[str, int]] = []
        for lidx in range(count):
            inputs = lane_inputs[lidx] if lane_inputs and lidx < len(lane_inputs) else {}
            out = self._run_lane(lidx, lds, inputs)
            lane_outputs.append(out)

        return list(lds), lane_outputs

    def _run_lane(self, lidx: int, lds: list[int], inputs: dict[str, int]) -> dict[str, int]:
        values: dict[str, int | bool] = dict(self.constants)
        values.update(inputs)
        var_box: dict[str, list[int]] = {}
        pointers: dict[str, tuple[str, list[int], int | None]] = {}
        outputs: list[str] = []

        if self.lds_var:
            pointers[self.lds_var] = ("lds", lds, None)

        for res, cls in self.var_class.items():
            if cls == "Output":
                var_box[res] = [0]
                pointers[res] = ("out", var_box[res], 0)
                outputs.append(res)
            elif cls == "Function":
                var_box[res] = [0]
                pointers[res] = ("fn", var_box[res], 0)

        def val(x: str) -> int | bool:
            if x in values:
                return values[x]
            raise UnsupportedConstruct(f"Undefined identifier {x}")

        for res, op, a, _ in self.body:
            if op in _IGNORED:
                continue
            if op in ("OpBranch", "OpBranchConditional"):
                continue
            if op == "OpVariable":
                cls, _ = self.ptr_types[a[0]]
                if cls != "Function":
                    raise UnsupportedConstruct("Non-Function variable in body")
                var_box[res] = [int(val(a[2])) if len(a) > 2 else 0]
                pointers[res] = ("fn", var_box[res], 0)
            elif op == "OpAccessChain":
                base = a[1]
                if base not in pointers:
                    raise UnsupportedConstruct(f"OpAccessChain on non-pointer {base}")
                ptag = pointers[base][0]
                if ptag == "lds":
                    idx = int(val(a[2]))
                    if not (0 <= idx < len(lds)):
                        raise WorkgroupBoundsError(f"LDS access out of bounds: {idx} >= {len(lds)}")
                    pointers[res] = ("lds", lds, idx)
                elif ptag in ("out", "fn"):
                    pointers[res] = pointers[base]
                else:
                    raise UnsupportedConstruct(f"Access chain on {ptag}")
            elif op == "OpLoad":
                ptr = pointers[a[1]]
                ptag = ptr[0]
                idx = ptr[2]
                assert idx is not None
                if ptag in ("lds", "out", "fn"):
                    values[res] = ptr[1][idx] & MASK
                else:
                    raise UnsupportedConstruct(f"OpLoad from {ptag}")
            elif op == "OpStore":
                ptr = pointers[a[0]]
                v = int(val(a[1])) & MASK
                ptag = ptr[0]
                idx = ptr[2]
                assert idx is not None
                if ptag in ("lds", "out", "fn"):
                    ptr[1][idx] = v
                else:
                    raise UnsupportedConstruct(f"OpStore to {ptag}")
            elif op == "OpAtomicUMin":
                # %res = OpAtomicUMin %type %ptr %scope %semantics %val
                ptr = pointers[a[1]]
                ptag = ptr[0]
                if ptag != "lds":
                    raise UnsupportedConstruct(f"OpAtomicUMin on non-LDS pointer {ptag}")
                operand = int(val(a[4])) & MASK
                idx = ptr[2]
                assert idx is not None
                old = ptr[1][idx] & MASK
                ptr[1][idx] = min(old, operand)
                values[res] = old
            elif op == "OpAtomicUMax":
                # %res = OpAtomicUMax %type %ptr %scope %semantics %val
                ptr = pointers[a[1]]
                ptag = ptr[0]
                if ptag != "lds":
                    raise UnsupportedConstruct(f"OpAtomicUMax on non-LDS pointer {ptag}")
                operand = int(val(a[4])) & MASK
                idx = ptr[2]
                assert idx is not None
                old = ptr[1][idx] & MASK
                ptr[1][idx] = max(old, operand)
                values[res] = old
            elif op == "OpAtomicSMin":
                # %res = OpAtomicSMin %type %ptr %scope %semantics %val
                ptr = pointers[a[1]]
                ptag = ptr[0]
                if ptag != "lds":
                    raise UnsupportedConstruct(f"OpAtomicSMin on non-LDS pointer {ptag}")
                operand = int(val(a[4])) & MASK
                idx = ptr[2]
                assert idx is not None
                old = ptr[1][idx] & MASK
                s_old = to_signed_i32(old)
                s_op = to_signed_i32(operand)
                ptr[1][idx] = to_unsigned_u32(min(s_old, s_op))
                values[res] = old
            elif op == "OpAtomicSMax":
                # %res = OpAtomicSMax %type %ptr %scope %semantics %val
                ptr = pointers[a[1]]
                ptag = ptr[0]
                if ptag != "lds":
                    raise UnsupportedConstruct(f"OpAtomicSMax on non-LDS pointer {ptag}")
                operand = int(val(a[4])) & MASK
                idx = ptr[2]
                assert idx is not None
                old = ptr[1][idx] & MASK
                s_old = to_signed_i32(old)
                s_op = to_signed_i32(operand)
                ptr[1][idx] = to_unsigned_u32(max(s_old, s_op))
                values[res] = old
            elif op == "OpIAdd":
                values[res] = (int(val(a[1])) + int(val(a[2]))) & MASK
            elif op == "OpISub":
                values[res] = (int(val(a[1])) - int(val(a[2]))) & MASK
            elif op == "OpShiftRightLogical":
                sh = int(val(a[2]))
                if sh >= 32:
                    raise UnsupportedConstruct("Shift >= 32")
                values[res] = (int(val(a[1])) >> sh) & MASK
            elif op == "OpShiftLeftLogical":
                sh = int(val(a[2]))
                if sh >= 32:
                    raise UnsupportedConstruct("Shift >= 32")
                values[res] = (int(val(a[1])) << sh) & MASK
            elif op == "OpBitwiseAnd":
                values[res] = int(val(a[1])) & int(val(a[2])) & MASK
            elif op == "OpBitwiseOr":
                values[res] = (int(val(a[1])) | int(val(a[2]))) & MASK
            elif op == "OpBitwiseXor":
                values[res] = (int(val(a[1])) ^ int(val(a[2]))) & MASK
            elif op == "OpSelect":
                cond = bool(val(a[1]))
                values[res] = val(a[2]) if cond else val(a[3])
            elif op == "OpCopyObject":
                values[res] = val(a[1])
            elif op == "OpBitcast":
                values[res] = int(val(a[1])) & MASK
            else:
                raise UnsupportedConstruct(f"Opcode {op} not modelled in body")

        return {name: var_box[name][0] & MASK for name in outputs}


class TestReferenceMinMaxMath(unittest.TestCase):
    """Verifies 32-bit unsigned and signed min/max arithmetic truth tables and edge cases."""

    def test_unsigned_min_basic_and_edges(self):
        cases = [
            (10, 20, 10),
            (20, 10, 10),
            (0, 0, 0),
            (0, 100, 0),
            (100, 0, 0),
            (0, 0xFFFFFFFF, 0),
            (0x7FFFFFFF, 0x80000000, 0x7FFFFFFF),
            (0x80000000, 0xFFFFFFFF, 0x80000000),
            (0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF),
            (0xFFFFFFFE, 0xFFFFFFFF, 0xFFFFFFFE),
        ]
        for a, b, expected in cases:
            with self.subTest(a=hex(a), b=hex(b), op="umin"):
                self.assertEqual(reference_atomic_umin(a, b), expected)

    def test_unsigned_max_basic_and_edges(self):
        cases = [
            (10, 20, 20),
            (20, 10, 20),
            (0, 0, 0),
            (0, 100, 100),
            (100, 0, 100),
            (0, 0xFFFFFFFF, 0xFFFFFFFF),
            (0x7FFFFFFF, 0x80000000, 0x80000000),
            (0x80000000, 0xFFFFFFFF, 0xFFFFFFFF),
            (0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF),
            (0xFFFFFFFE, 0xFFFFFFFF, 0xFFFFFFFF),
        ]
        for a, b, expected in cases:
            with self.subTest(a=hex(a), b=hex(b), op="umax"):
                self.assertEqual(reference_atomic_umax(a, b), expected)

    def test_signed_min_basic_and_edges(self):
        cases = [
            (10, 20, 10),
            (20, 10, 10),
            (0, 0, 0),
            (0, 100, 0),
            (0, 0xFFFFFFFF, 0xFFFFFFFF),  # 0 vs -1 -> min is -1
            (100, 0xFFFFFFFF, 0xFFFFFFFF),  # 100 vs -1 -> min is -1
            (0xFFFFFFF6, 0xFFFFFFEC, 0xFFFFFFEC),  # -10 vs -20 -> min is -20
            (0x7FFFFFFF, 0x80000000, 0x80000000),  # INT32_MAX vs INT32_MIN -> min is INT32_MIN
            (0x80000000, 0, 0x80000000),  # INT32_MIN vs 0 -> min is INT32_MIN
            (0xFFFFFFFF, 0x00000001, 0xFFFFFFFF),  # -1 vs 1 -> min is -1
        ]
        for a, b, expected in cases:
            with self.subTest(a=hex(a), b=hex(b), op="smin"):
                self.assertEqual(reference_atomic_smin(a, b), expected)

    def test_signed_max_basic_and_edges(self):
        cases = [
            (10, 20, 20),
            (20, 10, 20),
            (0, 0, 0),
            (0, 100, 100),
            (0, 0xFFFFFFFF, 0),  # 0 vs -1 -> max is 0
            (100, 0xFFFFFFFF, 100),  # 100 vs -1 -> max is 100
            (0xFFFFFFF6, 0xFFFFFFEC, 0xFFFFFFF6),  # -10 vs -20 -> max is -10
            (0x7FFFFFFF, 0x80000000, 0x7FFFFFFF),  # INT32_MAX vs INT32_MIN -> max is INT32_MAX
            (0x80000000, 0, 0),  # INT32_MIN vs 0 -> max is 0
            (0xFFFFFFFF, 0x00000001, 0x00000001),  # -1 vs 1 -> max is 1
        ]
        for a, b, expected in cases:
            with self.subTest(a=hex(a), b=hex(b), op="smax"):
                self.assertEqual(reference_atomic_smax(a, b), expected)

    def test_signed_vs_unsigned_divergence(self):
        # Crucial architectural contrast: identical bit patterns evaluate to opposite extremes
        divergent_pairs = [
            # (a, b): umin, smin, umax, smax
            (0, 0xFFFFFFFF, 0, 0xFFFFFFFF, 0xFFFFFFFF, 0),
            (0x7FFFFFFF, 0x80000000, 0x7FFFFFFF, 0x80000000, 0x80000000, 0x7FFFFFFF),
            (1, 0xFFFFFFFE, 1, 0xFFFFFFFE, 0xFFFFFFFE, 1),
            (0x80000001, 5, 5, 0x80000001, 0x80000001, 5),
        ]
        for a, b, exp_umin, exp_smin, exp_umax, exp_smax in divergent_pairs:
            with self.subTest(a=hex(a), b=hex(b)):
                self.assertEqual(reference_atomic_umin(a, b), exp_umin)
                self.assertEqual(reference_atomic_smin(a, b), exp_smin)
                self.assertEqual(reference_atomic_umax(a, b), exp_umax)
                self.assertEqual(reference_atomic_smax(a, b), exp_smax)
                # Ensure they actually diverge
                self.assertNotEqual(exp_umin, exp_smin)
                self.assertNotEqual(exp_umax, exp_smax)

    def test_accumulative_min_max_series(self):
        # Accumulate arbitrary sequence of signed and unsigned values
        vals = [50, 200, 10, 500, 3, 1000, 25]
        u_min = vals[0]
        u_max = vals[0]
        for v in vals[1:]:
            u_min = reference_atomic_umin(u_min, v)
            u_max = reference_atomic_umax(u_max, v)
        self.assertEqual(u_min, 3)
        self.assertEqual(u_max, 1000)

        # Signed sequence with positive and negative numbers
        s_vals = [10, -50, 100, -5, -500, 200, 0]
        s_min = s_vals[0] & MASK
        s_max = s_vals[0] & MASK
        for v in s_vals[1:]:
            s_min = reference_atomic_smin(s_min, v & MASK)
            s_max = reference_atomic_smax(s_max, v & MASK)
        self.assertEqual(to_signed_i32(s_min), -500)
        self.assertEqual(to_signed_i32(s_max), 200)


class TestLdsAddressingAndOffsets(unittest.TestCase):
    """Verifies byte offset math, dword truncation, and power-of-two modulo wrapping."""

    def test_aligned_dword_addressing(self):
        lds_dwords = 256
        for dword in (0, 1, 2, 64, 127, 255):
            byte_addr = dword * 4
            self.assertEqual(reference_lds_index(byte_addr, 0, lds_dwords), dword)

    def test_unaligned_byte_truncation(self):
        lds_dwords = 128
        for byte_off in (0, 1, 2, 3):
            self.assertEqual(reference_lds_index(16, byte_off, lds_dwords), 4)
        for unaligned_base in (16, 17, 18, 19):
            self.assertEqual(reference_lds_index(unaligned_base, 0, lds_dwords), 4)

    def test_static_offset_addition(self):
        lds_dwords = 512
        base = 32  # dword 8
        offset = 64  # +16 dwords
        self.assertEqual(reference_lds_index(base, offset, lds_dwords), 24)

    def test_boundary_wrapping(self):
        lds_dwords = 64
        self.assertEqual(reference_lds_index(256, 0, lds_dwords), 0)
        self.assertEqual(reference_lds_index(256, 4, lds_dwords), 1)
        self.assertEqual(reference_lds_index(0xFFFFFFFC, 0, lds_dwords), 63)


class TestSpirvAtomicUnsignedMinMaxInterpreter(unittest.TestCase):
    """Executes SPIR-V compute kernels exercising OpAtomicUMin and OpAtomicUMax."""

    def test_ds_min_u32_non_returning(self):
        spvasm = """
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
%uint = OpTypeInt 32 0
%c64 = OpConstant %uint 64
%c2 = OpConstant %uint 2
%c0 = OpConstant %uint 0
%c3 = OpConstant %uint 3
%c50 = OpConstant %uint 50
%lds_arr = OpTypeArray %uint %c64
%_ptr_WG_lds = OpTypePointer Workgroup %lds_arr
%_ptr_WG_uint = OpTypePointer Workgroup %uint
%lds = OpVariable %_ptr_WG_lds Workgroup
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
%ptr = OpAccessChain %_ptr_WG_uint %lds %c3
%dummy = OpAtomicUMin %uint %ptr %c2 %c0 %c50
OpReturn
OpFunctionEnd
"""
        interp = LdsAtomicMinMaxInterp(spvasm)
        # Case 1: Initial 200 min 50 -> 50
        final_lds, _ = interp.run(seed_lds={3: 200})
        self.assertEqual(final_lds[3], 50)
        # Case 2: Initial 20 min 50 -> 20 (stays smaller)
        final_lds, _ = interp.run(seed_lds={3: 20})
        self.assertEqual(final_lds[3], 20)

    def test_ds_min_rtn_u32_returning(self):
        spvasm = """
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main" %out
OpExecutionMode %main LocalSize 1 1 1
%uint = OpTypeInt 32 0
%_ptr_Output_uint = OpTypePointer Output %uint
%out = OpVariable %_ptr_Output_uint Output
%c64 = OpConstant %uint 64
%c2 = OpConstant %uint 2
%c0 = OpConstant %uint 0
%c5 = OpConstant %uint 5
%c400 = OpConstant %uint 400
%lds_arr = OpTypeArray %uint %c64
%_ptr_WG_lds = OpTypePointer Workgroup %lds_arr
%_ptr_WG_uint = OpTypePointer Workgroup %uint
%lds = OpVariable %_ptr_WG_lds Workgroup
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
%ptr = OpAccessChain %_ptr_WG_uint %lds %c5
%old = OpAtomicUMin %uint %ptr %c2 %c0 %c400
OpStore %out %old
OpReturn
OpFunctionEnd
"""
        interp = LdsAtomicMinMaxInterp(spvasm)
        final_lds, lane_out = interp.run(seed_lds={5: 1000})
        # Prior value was 1000
        self.assertEqual(lane_out[0]["%out"], 1000)
        # Memory became 400
        self.assertEqual(final_lds[5], 400)

    def test_ds_max_u32_non_returning(self):
        spvasm = """
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
%uint = OpTypeInt 32 0
%c32 = OpConstant %uint 32
%c2 = OpConstant %uint 2
%c0 = OpConstant %uint 0
%c2_slot = OpConstant %uint 2
%c150 = OpConstant %uint 150
%lds_arr = OpTypeArray %uint %c32
%_ptr_WG_lds = OpTypePointer Workgroup %lds_arr
%_ptr_WG_uint = OpTypePointer Workgroup %uint
%lds = OpVariable %_ptr_WG_lds Workgroup
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
%ptr = OpAccessChain %_ptr_WG_uint %lds %c2_slot
%dummy = OpAtomicUMax %uint %ptr %c2 %c0 %c150
OpReturn
OpFunctionEnd
"""
        interp = LdsAtomicMinMaxInterp(spvasm)
        # Initial 50 max 150 -> 150
        final_lds, _ = interp.run(seed_lds={2: 50})
        self.assertEqual(final_lds[2], 150)
        # Initial 300 max 150 -> 300 (stays larger)
        final_lds, _ = interp.run(seed_lds={2: 300})
        self.assertEqual(final_lds[2], 300)

    def test_ds_max_rtn_u32_returning(self):
        spvasm = """
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main" %out
OpExecutionMode %main LocalSize 1 1 1
%uint = OpTypeInt 32 0
%_ptr_Output_uint = OpTypePointer Output %uint
%out = OpVariable %_ptr_Output_uint Output
%c32 = OpConstant %uint 32
%c2 = OpConstant %uint 2
%c0 = OpConstant %uint 0
%c4_slot = OpConstant %uint 4
%c800 = OpConstant %uint 800
%lds_arr = OpTypeArray %uint %c32
%_ptr_WG_lds = OpTypePointer Workgroup %lds_arr
%_ptr_WG_uint = OpTypePointer Workgroup %uint
%lds = OpVariable %_ptr_WG_lds Workgroup
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
%ptr = OpAccessChain %_ptr_WG_uint %lds %c4_slot
%old = OpAtomicUMax %uint %ptr %c2 %c0 %c800
OpStore %out %old
OpReturn
OpFunctionEnd
"""
        interp = LdsAtomicMinMaxInterp(spvasm)
        final_lds, lane_out = interp.run(seed_lds={4: 300})
        # Prior value was 300
        self.assertEqual(lane_out[0]["%out"], 300)
        # Memory became 800
        self.assertEqual(final_lds[4], 800)

    def test_unsigned_min_max_boundaries(self):
        # Updating 0xFFFFFFFF with 0 via OpAtomicUMin -> 0
        spvasm = """
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main" %out
OpExecutionMode %main LocalSize 1 1 1
%uint = OpTypeInt 32 0
%_ptr_Output_uint = OpTypePointer Output %uint
%out = OpVariable %_ptr_Output_uint Output
%c16 = OpConstant %uint 16
%c2 = OpConstant %uint 2
%c0 = OpConstant %uint 0
%c0_slot = OpConstant %uint 0
%lds_arr = OpTypeArray %uint %c16
%_ptr_WG_lds = OpTypePointer Workgroup %lds_arr
%_ptr_WG_uint = OpTypePointer Workgroup %uint
%lds = OpVariable %_ptr_WG_lds Workgroup
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
%ptr = OpAccessChain %_ptr_WG_uint %lds %c0_slot
%old = OpAtomicUMin %uint %ptr %c2 %c0 %c0
OpStore %out %old
OpReturn
OpFunctionEnd
"""
        interp = LdsAtomicMinMaxInterp(spvasm)
        final_lds, lane_out = interp.run(seed_lds={0: 0xFFFFFFFF})
        self.assertEqual(lane_out[0]["%out"], 0xFFFFFFFF)
        self.assertEqual(final_lds[0], 0)


class TestSpirvAtomicSignedMinMaxInterpreter(unittest.TestCase):
    """Executes SPIR-V compute kernels exercising OpAtomicSMin and OpAtomicSMax."""

    def test_ds_min_i32_non_returning(self):
        spvasm = """
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
%uint = OpTypeInt 32 0
%c64 = OpConstant %uint 64
%c2 = OpConstant %uint 2
%c0 = OpConstant %uint 0
%c3 = OpConstant %uint 3
%c_neg5 = OpConstant %uint 0xFFFFFFFB
%lds_arr = OpTypeArray %uint %c64
%_ptr_WG_lds = OpTypePointer Workgroup %lds_arr
%_ptr_WG_uint = OpTypePointer Workgroup %uint
%lds = OpVariable %_ptr_WG_lds Workgroup
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
%ptr = OpAccessChain %_ptr_WG_uint %lds %c3
%dummy = OpAtomicSMin %uint %ptr %c2 %c0 %c_neg5
OpReturn
OpFunctionEnd
"""
        interp = LdsAtomicMinMaxInterp(spvasm)
        # Initial 10 smin -5 -> -5 (0xFFFFFFFB)
        final_lds, _ = interp.run(seed_lds={3: 10})
        self.assertEqual(final_lds[3], 0xFFFFFFFB)

        # Initial -10 (0xFFFFFFF6) smin -5 -> -10 (stays smaller)
        final_lds, _ = interp.run(seed_lds={3: 0xFFFFFFF6})
        self.assertEqual(final_lds[3], 0xFFFFFFF6)

    def test_ds_min_rtn_i32_returning(self):
        spvasm = """
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main" %out
OpExecutionMode %main LocalSize 1 1 1
%uint = OpTypeInt 32 0
%_ptr_Output_uint = OpTypePointer Output %uint
%out = OpVariable %_ptr_Output_uint Output
%c64 = OpConstant %uint 64
%c2 = OpConstant %uint 2
%c0 = OpConstant %uint 0
%c1 = OpConstant %uint 1
%c_neg1 = OpConstant %uint 0xFFFFFFFF
%lds_arr = OpTypeArray %uint %c64
%_ptr_WG_lds = OpTypePointer Workgroup %lds_arr
%_ptr_WG_uint = OpTypePointer Workgroup %uint
%lds = OpVariable %_ptr_WG_lds Workgroup
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
%ptr = OpAccessChain %_ptr_WG_uint %lds %c1
%old = OpAtomicSMin %uint %ptr %c2 %c0 %c_neg1
OpStore %out %old
OpReturn
OpFunctionEnd
"""
        interp = LdsAtomicMinMaxInterp(spvasm)
        # 50 smin -1 -> -1 (0xFFFFFFFF), returning 50
        final_lds, lane_out = interp.run(seed_lds={1: 50})
        self.assertEqual(lane_out[0]["%out"], 50)
        self.assertEqual(final_lds[1], 0xFFFFFFFF)

    def test_ds_max_i32_non_returning(self):
        spvasm = """
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
%uint = OpTypeInt 32 0
%c32 = OpConstant %uint 32
%c2 = OpConstant %uint 2
%c0 = OpConstant %uint 0
%c2_slot = OpConstant %uint 2
%c_neg10 = OpConstant %uint 0xFFFFFFF6
%lds_arr = OpTypeArray %uint %c32
%_ptr_WG_lds = OpTypePointer Workgroup %lds_arr
%_ptr_WG_uint = OpTypePointer Workgroup %uint
%lds = OpVariable %_ptr_WG_lds Workgroup
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
%ptr = OpAccessChain %_ptr_WG_uint %lds %c2_slot
%dummy = OpAtomicSMax %uint %ptr %c2 %c0 %c_neg10
OpReturn
OpFunctionEnd
"""
        interp = LdsAtomicMinMaxInterp(spvasm)
        # Initial -50 (0xFFFFFFCE) smax -10 (0xFFFFFFF6) -> -10
        final_lds, _ = interp.run(seed_lds={2: 0xFFFFFFCE})
        self.assertEqual(final_lds[2], 0xFFFFFFF6)

        # Initial 5 smax -10 -> 5 (stays larger)
        final_lds, _ = interp.run(seed_lds={2: 5})
        self.assertEqual(final_lds[2], 5)

    def test_ds_max_rtn_i32_returning(self):
        spvasm = """
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main" %out
OpExecutionMode %main LocalSize 1 1 1
%uint = OpTypeInt 32 0
%_ptr_Output_uint = OpTypePointer Output %uint
%out = OpVariable %_ptr_Output_uint Output
%c32 = OpConstant %uint 32
%c2 = OpConstant %uint 2
%c0 = OpConstant %uint 0
%c6_slot = OpConstant %uint 6
%c_int32_max = OpConstant %uint 0x7FFFFFFF
%lds_arr = OpTypeArray %uint %c32
%_ptr_WG_lds = OpTypePointer Workgroup %lds_arr
%_ptr_WG_uint = OpTypePointer Workgroup %uint
%lds = OpVariable %_ptr_WG_lds Workgroup
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
%ptr = OpAccessChain %_ptr_WG_uint %lds %c6_slot
%old = OpAtomicSMax %uint %ptr %c2 %c0 %c_int32_max
OpStore %out %old
OpReturn
OpFunctionEnd
"""
        interp = LdsAtomicMinMaxInterp(spvasm)
        # Seed with INT32_MIN (0x80000000), smax with INT32_MAX (0x7FFFFFFF) -> INT32_MAX, returning INT32_MIN
        final_lds, lane_out = interp.run(seed_lds={6: 0x80000000})
        self.assertEqual(lane_out[0]["%out"], 0x80000000)
        self.assertEqual(final_lds[6], 0x7FFFFFFF)


class TestSpirvSignedUnsignedDivergenceInKernels(unittest.TestCase):
    """Direct comparative test proving signed and unsigned atomics diverge on identical inputs."""

    def test_umin_vs_smin_with_zero_and_all_ones(self):
        umin_spvasm = """
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main" %out
OpExecutionMode %main LocalSize 1 1 1
%uint = OpTypeInt 32 0
%_ptr_Output_uint = OpTypePointer Output %uint
%out = OpVariable %_ptr_Output_uint Output
%c16 = OpConstant %uint 16
%c2 = OpConstant %uint 2
%c0 = OpConstant %uint 0
%c0_slot = OpConstant %uint 0
%c_all_ones = OpConstant %uint 0xFFFFFFFF
%lds_arr = OpTypeArray %uint %c16
%_ptr_WG_lds = OpTypePointer Workgroup %lds_arr
%_ptr_WG_uint = OpTypePointer Workgroup %uint
%lds = OpVariable %_ptr_WG_lds Workgroup
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
%ptr = OpAccessChain %_ptr_WG_uint %lds %c0_slot
%old = OpAtomicUMin %uint %ptr %c2 %c0 %c_all_ones
OpStore %out %old
OpReturn
OpFunctionEnd
"""
        smin_spvasm = umin_spvasm.replace("OpAtomicUMin", "OpAtomicSMin")

        # Initial value 0:
        # UMin: min_u32(0, 0xFFFFFFFF) = 0
        interp_u = LdsAtomicMinMaxInterp(umin_spvasm)
        lds_u, out_u = interp_u.run(seed_lds={0: 0})
        self.assertEqual(lds_u[0], 0)
        self.assertEqual(out_u[0]["%out"], 0)

        # SMin: min_i32(0, -1) = -1 (0xFFFFFFFF)
        interp_s = LdsAtomicMinMaxInterp(smin_spvasm)
        lds_s, out_s = interp_s.run(seed_lds={0: 0})
        self.assertEqual(lds_s[0], 0xFFFFFFFF)
        self.assertEqual(out_s[0]["%out"], 0)

    def test_umax_vs_smax_with_int32_min_max(self):
        umax_spvasm = """
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main" %out
OpExecutionMode %main LocalSize 1 1 1
%uint = OpTypeInt 32 0
%_ptr_Output_uint = OpTypePointer Output %uint
%out = OpVariable %_ptr_Output_uint Output
%c16 = OpConstant %uint 16
%c2 = OpConstant %uint 2
%c0 = OpConstant %uint 0
%c0_slot = OpConstant %uint 0
%c_int32_min = OpConstant %uint 0x80000000
%lds_arr = OpTypeArray %uint %c16
%_ptr_WG_lds = OpTypePointer Workgroup %lds_arr
%_ptr_WG_uint = OpTypePointer Workgroup %uint
%lds = OpVariable %_ptr_WG_lds Workgroup
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
%ptr = OpAccessChain %_ptr_WG_uint %lds %c0_slot
%old = OpAtomicUMax %uint %ptr %c2 %c0 %c_int32_min
OpStore %out %old
OpReturn
OpFunctionEnd
"""
        smax_spvasm = umax_spvasm.replace("OpAtomicUMax", "OpAtomicSMax")

        # Initial value 0x7FFFFFFF:
        # UMax: max_u32(0x7FFFFFFF, 0x80000000) = 0x80000000 (unsigned 2147483648 > 2147483647)
        interp_u = LdsAtomicMinMaxInterp(umax_spvasm)
        lds_u, out_u = interp_u.run(seed_lds={0: 0x7FFFFFFF})
        self.assertEqual(lds_u[0], 0x80000000)

        # SMax: max_i32(+2147483647, -2147483648) = +2147483647 (0x7FFFFFFF)
        interp_s = LdsAtomicMinMaxInterp(smax_spvasm)
        lds_s, out_s = interp_s.run(seed_lds={0: 0x7FFFFFFF})
        self.assertEqual(lds_s[0], 0x7FFFFFFF)


class TestChainedSequentialAtomics(unittest.TestCase):
    """Executes sequential chains of atomic min/max updates on a single LDS slot."""

    def test_chained_umin_reduction(self):
        spvasm = """
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main" %out
OpExecutionMode %main LocalSize 1 1 1
%uint = OpTypeInt 32 0
%_ptr_Output_uint = OpTypePointer Output %uint
%out = OpVariable %_ptr_Output_uint Output
%c16 = OpConstant %uint 16
%c2 = OpConstant %uint 2
%c0 = OpConstant %uint 0
%c4_slot = OpConstant %uint 4
%c300 = OpConstant %uint 300
%c150 = OpConstant %uint 150
%c200 = OpConstant %uint 200
%c50 = OpConstant %uint 50
%lds_arr = OpTypeArray %uint %c16
%_ptr_WG_lds = OpTypePointer Workgroup %lds_arr
%_ptr_WG_uint = OpTypePointer Workgroup %uint
%lds = OpVariable %_ptr_WG_lds Workgroup
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
%ptr = OpAccessChain %_ptr_WG_uint %lds %c4_slot
%r1 = OpAtomicUMin %uint %ptr %c2 %c0 %c300
%r2 = OpAtomicUMin %uint %ptr %c2 %c0 %c150
%r3 = OpAtomicUMin %uint %ptr %c2 %c0 %c200
%r4 = OpAtomicUMin %uint %ptr %c2 %c0 %c50
OpStore %out %r4
OpReturn
OpFunctionEnd
"""
        interp = LdsAtomicMinMaxInterp(spvasm)
        final_lds, lane_out = interp.run(seed_lds={4: 1000})
        # r1 saw 1000 (LDS became 300)
        # r2 saw 300 (LDS became 150)
        # r3 saw 150 (LDS stayed 150 because 200 > 150)
        # r4 saw 150 (LDS became 50)
        self.assertEqual(lane_out[0]["%out"], 150)
        self.assertEqual(final_lds[4], 50)

    def test_chained_smax_reduction(self):
        spvasm = """
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main" %out
OpExecutionMode %main LocalSize 1 1 1
%uint = OpTypeInt 32 0
%_ptr_Output_uint = OpTypePointer Output %uint
%out = OpVariable %_ptr_Output_uint Output
%c16 = OpConstant %uint 16
%c2 = OpConstant %uint 2
%c0 = OpConstant %uint 0
%c2_slot = OpConstant %uint 2
%c_neg100 = OpConstant %uint 0xFFFFFF9C
%c_neg20 = OpConstant %uint 0xFFFFFFEC
%c_neg50 = OpConstant %uint 0xFFFFFFCE
%c_pos10 = OpConstant %uint 10
%lds_arr = OpTypeArray %uint %c16
%_ptr_WG_lds = OpTypePointer Workgroup %lds_arr
%_ptr_WG_uint = OpTypePointer Workgroup %uint
%lds = OpVariable %_ptr_WG_lds Workgroup
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
%ptr = OpAccessChain %_ptr_WG_uint %lds %c2_slot
%r1 = OpAtomicSMax %uint %ptr %c2 %c0 %c_neg100
%r2 = OpAtomicSMax %uint %ptr %c2 %c0 %c_neg20
%r3 = OpAtomicSMax %uint %ptr %c2 %c0 %c_neg50
%r4 = OpAtomicSMax %uint %ptr %c2 %c0 %c_pos10
OpStore %out %r4
OpReturn
OpFunctionEnd
"""
        interp = LdsAtomicMinMaxInterp(spvasm)
        final_lds, lane_out = interp.run(seed_lds={2: 0x80000000})  # INT32_MIN
        # r1 saw INT32_MIN (LDS became -100)
        # r2 saw -100 (LDS became -20)
        # r3 saw -20 (LDS stayed -20 because -50 < -20)
        # r4 saw -20 (LDS became +10)
        self.assertEqual(lane_out[0]["%out"], 0xFFFFFFEC)  # -20
        self.assertEqual(final_lds[2], 10)


class TestWorkgroupMultiLaneReductions(unittest.TestCase):
    """Verifies parallel workgroup reductions and atomic convergence across multiple lanes."""

    def test_multi_lane_unsigned_min_reduction(self):
        # 8 lanes cooperate to find the unsigned minimum
        spvasm = """
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main" %out
OpExecutionMode %main LocalSize 8 1 1
%uint = OpTypeInt 32 0
%_ptr_Output_uint = OpTypePointer Output %uint
%out = OpVariable %_ptr_Output_uint Output
%c16 = OpConstant %uint 16
%c2 = OpConstant %uint 2
%c0 = OpConstant %uint 0
%c0_slot = OpConstant %uint 0
%lds_arr = OpTypeArray %uint %c16
%_ptr_WG_lds = OpTypePointer Workgroup %lds_arr
%_ptr_WG_uint = OpTypePointer Workgroup %uint
%lds = OpVariable %_ptr_WG_lds Workgroup
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
%ptr = OpAccessChain %_ptr_WG_uint %lds %c0_slot
%old = OpAtomicUMin %uint %ptr %c2 %c0 %c_val
OpStore %out %old
OpReturn
OpFunctionEnd
"""
        interp = LdsAtomicMinMaxInterp(spvasm)
        lane_values = [500, 320, 110, 850, 42, 600, 99, 150]
        lane_inputs = [{"%c_val": v} for v in lane_values]

        final_lds, lane_outs = interp.run(num_invocations=8, seed_lds={0: 1000}, lane_inputs=lane_inputs)
        # Minimum of seed 1000 and all lane_values is 42
        self.assertEqual(final_lds[0], 42)
        # Returned values must form a monotonically non-increasing sequence
        ret_vals = [o["%out"] for o in lane_outs]
        self.assertEqual(ret_vals[0], 1000)
        for i in range(1, len(ret_vals)):
            self.assertLessEqual(ret_vals[i], ret_vals[i - 1])

    def test_multi_lane_signed_max_reduction(self):
        # 8 lanes propose signed values, finding signed maximum
        spvasm = """
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main" %out
OpExecutionMode %main LocalSize 8 1 1
%uint = OpTypeInt 32 0
%_ptr_Output_uint = OpTypePointer Output %uint
%out = OpVariable %_ptr_Output_uint Output
%c16 = OpConstant %uint 16
%c2 = OpConstant %uint 2
%c0 = OpConstant %uint 0
%c1_slot = OpConstant %uint 1
%lds_arr = OpTypeArray %uint %c16
%_ptr_WG_lds = OpTypePointer Workgroup %lds_arr
%_ptr_WG_uint = OpTypePointer Workgroup %uint
%lds = OpVariable %_ptr_WG_lds Workgroup
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
%ptr = OpAccessChain %_ptr_WG_uint %lds %c1_slot
%old = OpAtomicSMax %uint %ptr %c2 %c0 %c_val
OpStore %out %old
OpReturn
OpFunctionEnd
"""
        interp = LdsAtomicMinMaxInterp(spvasm)
        # Signed values: -50, -200, -5, +30, -10, +75, -1, 0
        signed_values = [-50, -200, -5, 30, -10, 75, -1, 0]
        lane_inputs = [{"%c_val": to_unsigned_u32(v)} for v in signed_values]

        final_lds, _ = interp.run(
            num_invocations=8,
            seed_lds={1: to_unsigned_u32(-100)},
            lane_inputs=lane_inputs,
        )
        self.assertEqual(to_signed_i32(final_lds[1]), 75)

    def test_wave32_workgroup_bounding_box_reduction(self):
        # 32 lanes (a full wave) compute 2D bounding box (min_x, max_x) in LDS[0] and LDS[1]
        spvasm = """
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 32 1 1
%uint = OpTypeInt 32 0
%c64 = OpConstant %uint 64
%c2 = OpConstant %uint 2
%c0 = OpConstant %uint 0
%c0_min_slot = OpConstant %uint 0
%c1_max_slot = OpConstant %uint 1
%lds_arr = OpTypeArray %uint %c64
%_ptr_WG_lds = OpTypePointer Workgroup %lds_arr
%_ptr_WG_uint = OpTypePointer Workgroup %uint
%lds = OpVariable %_ptr_WG_lds Workgroup
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
%ptr_min = OpAccessChain %_ptr_WG_uint %lds %c0_min_slot
%ptr_max = OpAccessChain %_ptr_WG_uint %lds %c1_max_slot
%r_min = OpAtomicUMin %uint %ptr_min %c2 %c0 %c_coord
%r_max = OpAtomicUMax %uint %ptr_max %c2 %c0 %c_coord
OpReturn
OpFunctionEnd
"""
        interp = LdsAtomicMinMaxInterp(spvasm)
        # Coordinate per lane: lane 0 -> 100, lane 1 -> 105, ..., lane 31 -> 255
        lane_inputs = [{"%c_coord": 100 + i * 5} for i in range(32)]
        final_lds, _ = interp.run(
            num_invocations=32,
            seed_lds={0: 0xFFFFFFFF, 1: 0},
            lane_inputs=lane_inputs,
        )
        self.assertEqual(final_lds[0], 100)
        self.assertEqual(final_lds[1], 100 + 31 * 5)


class TestTranslatorIntegration(unittest.TestCase):
    """Verifies that tools/rdna2spirv.py emits conforming SPIR-V for ds_min/max opcodes."""

    def _make_mock_gen(self, stage="cs"):
        class MockGen:
            def __init__(self):
                self.stage = stage
                self.uses_lds = False
                self.lds_dwords = 256
                self.lds_bound = None       # not bounds-checked (SHADER-309): the wrapped access
                self.wave64 = True
                self._tmp = 0
                self.emitted = []
                self.stored_dsts = []

            def tmp(self):
                self._tmp += 1
                return f"%t{self._tmp}"

            def cu(self, v):
                return f"%c{v}"

            def active_b(self):
                return "%exec_active"

            def src_u(self, arg):
                return f"%v_{arg['r']}"

            def dst(self, d, v):
                self.stored_dsts.append((d, v))

            def emit(self, line):
                self.emitted.append(line)

            LDS_ATOMICS = rdna2spirv.Gen.LDS_ATOMICS
            mod_int = staticmethod(rdna2spirv.Gen.mod_int)
            lds_ptr = rdna2spirv.Gen.lds_ptr
            lds_access = rdna2spirv.Gen.lds_access
            lds_atomic = rdna2spirv.Gen.lds_atomic
            lds_op = rdna2spirv.Gen.lds_op

        return MockGen()

    def test_lds_atomics_table_contains_minmax_opcodes(self):
        expected_mappings = {
            "ds_min_u32": "OpAtomicUMin",
            "ds_max_u32": "OpAtomicUMax",
            "ds_min_i32": "OpAtomicSMin",
            "ds_max_i32": "OpAtomicSMax",
        }
        for opcode, spirv_inst in expected_mappings.items():
            with self.subTest(opcode=opcode):
                self.assertIn(opcode, rdna2spirv.Gen.LDS_ATOMICS)
                self.assertEqual(rdna2spirv.Gen.LDS_ATOMICS[opcode], spirv_inst)

    def test_ds_min_u32_non_returning_emission(self):
        gen = self._make_mock_gen(stage="cs")
        # ds_min_u32 v0, v1, offset:8 (args: addr=v0, data=v1)
        gen.lds_op("ds_min_u32", [{"r": 0}, {"r": 1}], ["offset:8"])
        self.assertTrue(gen.uses_lds)
        emitted_str = "\n".join(gen.emitted)

        self.assertIn("OpBranchConditional %exec_active", emitted_str)
        self.assertIn("OpIAdd %uint %v_0 %c8", emitted_str)
        self.assertIn("OpShiftRightLogical %uint", emitted_str)
        self.assertIn("OpBitwiseAnd %uint", emitted_str)
        self.assertIn("OpAccessChain %_ptr_WG_uint %lds", emitted_str)
        self.assertIn("OpAtomicUMin %uint", emitted_str)
        self.assertIn("%c2 %c0 %v_1", emitted_str)
        # Non-returning form must NOT store destination
        self.assertEqual(len(gen.stored_dsts), 0)

    def test_ds_min_rtn_u32_returning_emission(self):
        gen = self._make_mock_gen(stage="cs")
        # ds_min_rtn_u32 v5, v0, v1, offset:12 (args: dst=v5, addr=v0, data=v1)
        gen.lds_op("ds_min_rtn_u32", [{"r": 5}, {"r": 0}, {"r": 1}], ["offset:12"])
        self.assertTrue(gen.uses_lds)
        emitted_str = "\n".join(gen.emitted)

        self.assertIn("OpAtomicUMin %uint", emitted_str)
        self.assertIn("%c2 %c0 %v_1", emitted_str)
        self.assertEqual(len(gen.stored_dsts), 1)
        dest_reg, val_id = gen.stored_dsts[0]
        self.assertEqual(dest_reg["r"], 5)
        self.assertTrue(val_id.startswith("%t"))

    def test_ds_max_u32_emission(self):
        gen = self._make_mock_gen(stage="cs")
        gen.lds_op("ds_max_rtn_u32", [{"r": 2}, {"r": 3}, {"r": 4}], [])
        self.assertTrue(gen.uses_lds)
        emitted_str = "\n".join(gen.emitted)
        self.assertIn("OpAtomicUMax %uint", emitted_str)
        self.assertEqual(len(gen.stored_dsts), 1)
        self.assertEqual(gen.stored_dsts[0][0]["r"], 2)

    def test_ds_min_i32_signed_emission(self):
        gen = self._make_mock_gen(stage="cs")
        gen.lds_op("ds_min_rtn_i32", [{"r": 7}, {"r": 1}, {"r": 2}], ["offset:16"])
        self.assertTrue(gen.uses_lds)
        emitted_str = "\n".join(gen.emitted)
        self.assertIn("OpAtomicSMin %uint", emitted_str)
        self.assertIn("%c2 %c0 %v_2", emitted_str)
        self.assertEqual(len(gen.stored_dsts), 1)
        self.assertEqual(gen.stored_dsts[0][0]["r"], 7)

    def test_ds_max_i32_signed_emission(self):
        gen = self._make_mock_gen(stage="cs")
        gen.lds_op("ds_max_rtn_i32", [{"r": 9}, {"r": 4}, {"r": 5}], ["offset:4"])
        self.assertTrue(gen.uses_lds)
        emitted_str = "\n".join(gen.emitted)
        self.assertIn("OpAtomicSMax %uint", emitted_str)
        self.assertIn("%c2 %c0 %v_5", emitted_str)
        self.assertEqual(len(gen.stored_dsts), 1)
        self.assertEqual(gen.stored_dsts[0][0]["r"], 9)

    def test_lds_minmax_outside_compute_raises_unsupported(self):
        for invalid_stage in ("ps", "vs", "gs"):
            for op in ("ds_min_u32", "ds_max_u32", "ds_min_i32", "ds_max_i32"):
                with self.subTest(stage=invalid_stage, op=op):
                    gen = self._make_mock_gen(stage=invalid_stage)
                    with self.assertRaises(rdna2spirv.Unsupported):
                        gen.lds_op(op, [{"r": 0}, {"r": 1}], [])


class TestNegativeControlsAndBounds(unittest.TestCase):
    """Verifies bounds enforcement, invalid opcodes, and malformed constructs."""

    def test_unsupported_storage_class_rejected(self):
        spvasm = """
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
%uint = OpTypeInt 32 0
%_ptr_UC = OpTypePointer UniformConstant %uint
%u = OpVariable %_ptr_UC UniformConstant
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
OpReturn
OpFunctionEnd
"""
        with self.assertRaises(UnsupportedConstruct):
            LdsAtomicMinMaxInterp(spvasm)

    def test_unsupported_opcode_in_body_rejected(self):
        spvasm = """
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
%uint = OpTypeInt 32 0
%c16 = OpConstant %uint 16
%lds_arr = OpTypeArray %uint %c16
%_ptr_WG_lds = OpTypePointer Workgroup %lds_arr
%lds = OpVariable %_ptr_WG_lds Workgroup
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
%bad = OpSDiv %uint %c16 %c16
OpReturn
OpFunctionEnd
"""
        interp = LdsAtomicMinMaxInterp(spvasm)
        with self.assertRaises(UnsupportedConstruct):
            interp.run()

    def test_invalid_workgroup_size_rejected(self):
        for dims in [(0, 1, 1), (1025, 1, 1), (32, 33, 1)]:
            with self.subTest(dims=dims):
                spvasm = f"""
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize {dims[0]} {dims[1]} {dims[2]}
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
OpReturn
OpFunctionEnd
"""
                with self.assertRaises(WorkgroupBoundsError):
                    LdsAtomicMinMaxInterp(spvasm)

    def test_atomic_on_non_lds_pointer_rejected(self):
        spvasm = """
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
%uint = OpTypeInt 32 0
%_ptr_fn_uint = OpTypePointer Function %uint
%c2 = OpConstant %uint 2
%c0 = OpConstant %uint 0
%c5 = OpConstant %uint 5
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
%local_v = OpVariable %_ptr_fn_uint Function
%bad = OpAtomicUMin %uint %local_v %c2 %c0 %c5
OpReturn
OpFunctionEnd
"""
        interp = LdsAtomicMinMaxInterp(spvasm)
        with self.assertRaises(UnsupportedConstruct):
            interp.run()

    def test_lds_out_of_bounds_access_chain_rejected(self):
        spvasm = """
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
%uint = OpTypeInt 32 0
%c8 = OpConstant %uint 8
%c2 = OpConstant %uint 2
%c0 = OpConstant %uint 0
%c100_slot = OpConstant %uint 100
%lds_arr = OpTypeArray %uint %c8
%_ptr_WG_lds = OpTypePointer Workgroup %lds_arr
%_ptr_WG_uint = OpTypePointer Workgroup %uint
%lds = OpVariable %_ptr_WG_lds Workgroup
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
%ptr = OpAccessChain %_ptr_WG_uint %lds %c100_slot
%dummy = OpAtomicUMin %uint %ptr %c2 %c0 %c0
OpReturn
OpFunctionEnd
"""
        interp = LdsAtomicMinMaxInterp(spvasm)
        with self.assertRaises(WorkgroupBoundsError):
            interp.run()


if __name__ == "__main__":
    unittest.main()
