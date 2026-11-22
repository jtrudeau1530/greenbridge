"""Semantic interpreter tests for LDS atomic subtraction (ds_sub_u32, ds_sub_rtn_u32).

Standard library only; no GPU, no LLVM, no external tools, no game files.

    python -I tests/semantic/test_lds_atomic_sub.py
    python -m unittest discover -s tests/semantic -p test_lds_atomic_sub.py

Background & Specification Rules:
  * AMD RDNA2 / GFX10 Local Data Share (LDS) Architecture:
    - LDS is an on-chip software-managed scratchpad memory shared by all threads
      in a workgroup (threadgroup).
    - `ds_sub_u32 vaddr, vdata [offset:N]`:
        Atomically computes:
          addr = (vaddr + byte_offset)
          idx = (addr >> 2) & (lds_dwords - 1)
          LDS[idx] = (LDS[idx] - vdata) & 0xFFFFFFFF
        Non-returning form: the prior value at LDS[idx] is discarded.
    - Exact formula:
        LDS[addr] = (LDS[addr] - val) mod 2^32
    - Modulo 2^32 arithmetic:
        All subtractions wrap strictly modulo 2^32 without trapping or saturating.
  * SPIR-V Representation:
    - LDS is declared as an array in the Workgroup storage class:
        %lds_arr = OpTypeArray %uint <lds_dwords>
        %_ptr_WG_lds = OpTypePointer Workgroup %lds_arr
        %_ptr_WG_uint = OpTypePointer Workgroup %uint
        %lds = OpVariable %_ptr_WG_lds Workgroup
    - Atomic subtractions map directly to OpAtomicISub:
        %old = OpAtomicISub %uint %ptr %scope %semantics %val
        where %scope is Workgroup (2) and %semantics is Relaxed / None (0).
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


class UnsupportedConstruct(NotImplementedError):
    """The module uses an unmodelled SPIR-V opcode, type, or storage class."""


class WorkgroupBoundsError(ValueError):
    """Workgroup dimension or memory bounds violation."""


_IGNORED = frozenset({
    "OpCapability", "OpExtInstImport", "OpMemoryModel", "OpEntryPoint", "OpExecutionMode",
    "OpDecorate", "OpMemberDecorate", "OpTypeVoid", "OpTypeFunction",
    "OpReturn", "OpFunctionEnd", "OpName", "OpSource", "OpSelectionMerge",
})

_LINE = re.compile(r"^(?:(%\S+)\s*=\s*)?(Op\w+)\s*(.*)$")


def reference_atomic_sub(initial: int, subtrahend: int) -> int:
    """Reference 32-bit unsigned wrapping atomic subtraction."""
    return (initial - subtrahend) & MASK


def reference_lds_index(base_addr: int, byte_offset: int, lds_dwords: int) -> int:
    """Reference dword index calculation with hardware wrapping."""
    effective_byte = (base_addr + byte_offset) & MASK
    dword_idx = effective_byte >> 2
    return dword_idx & (lds_dwords - 1)


class LdsAtomicSubInterp:
    """Strict semantic interpreter for compute SPIR-V modules exercising LDS atomic subtraction."""

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
        self.label_indices: dict[str, int] = {}
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
                if op == "OpLabel" and res:
                    self.label_indices[res] = len(self.body)
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
        lane_inputs: list[dict[str, int | bool]] | None = None,
        exec_mask: int | None = None,
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

        lane_outputs = []
        for lidx in range(count):
            inputs = dict(lane_inputs[lidx]) if lane_inputs and lidx < len(lane_inputs) else {}
            if exec_mask is not None:
                inputs["%exec_active"] = bool((exec_mask >> lidx) & 1)
            out = self._run_lane(lidx, lds, inputs)
            lane_outputs.append(out)

        return list(lds), lane_outputs

    def _run_lane(self, lidx: int, lds: list[int], inputs: dict[str, int | bool]) -> dict[str, int]:
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

        pc = 0
        total_insns = len(self.body)
        while pc < total_insns:
            res, op, a, _ = self.body[pc]
            pc += 1

            if op in _IGNORED or op == "OpLabel":
                continue
            if op == "OpReturn":
                break
            if op == "OpBranch":
                target = a[0]
                if target not in self.label_indices:
                    raise UnsupportedConstruct(f"Unknown branch label {target}")
                pc = self.label_indices[target]
                continue
            if op == "OpBranchConditional":
                cond = bool(val(a[0]))
                target = a[1] if cond else a[2]
                if target not in self.label_indices:
                    raise UnsupportedConstruct(f"Unknown branch target {target}")
                pc = self.label_indices[target]
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
                if ptag == "lds":
                    values[res] = ptr[1][ptr[2]] & MASK
                elif ptag in ("out", "fn"):
                    values[res] = ptr[1][ptr[2]] & MASK
            elif op == "OpStore":
                ptr = pointers[a[0]]
                v = int(val(a[1])) & MASK
                ptag = ptr[0]
                if ptag == "lds":
                    ptr[1][ptr[2]] = v
                elif ptag in ("out", "fn"):
                    ptr[1][ptr[2]] = v
            elif op == "OpAtomicISub":
                # %res = OpAtomicISub %type %ptr %scope %semantics %val
                ptr = pointers[a[1]]
                ptag = ptr[0]
                if ptag != "lds":
                    raise UnsupportedConstruct(f"OpAtomicISub on non-LDS pointer {ptag}")
                subtrahend = int(val(a[4])) & MASK
                old = ptr[1][ptr[2]] & MASK
                ptr[1][ptr[2]] = (old - subtrahend) & MASK
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


class TestAtomicSubReferenceMath(unittest.TestCase):
    """Verifies 32-bit unsigned wrapping arithmetic and edge cases for atomic subtractions."""

    def test_basic_subtraction(self):
        self.assertEqual(reference_atomic_sub(100, 30), 70)
        self.assertEqual(reference_atomic_sub(0, 0), 0)
        self.assertEqual(reference_atomic_sub(50, 50), 0)
        self.assertEqual(reference_atomic_sub(100, 0), 100)

    def test_zero_identity(self):
        # Subtracting zero is an identity operation
        for val in (0, 1, 42, 0x12345678, 0x80000000, 0xFFFFFFFF):
            with self.subTest(val=hex(val)):
                self.assertEqual(reference_atomic_sub(val, 0), val)

    def test_monotonic_decrement_series(self):
        # Simulates multiple atomic decrements from 1000 down by 10 each step
        current = 1000
        for _ in range(50):
            current = reference_atomic_sub(current, 10)
        self.assertEqual(current, 500)

    def test_large_operand_subtraction(self):
        # Large values without underflow
        self.assertEqual(reference_atomic_sub(0x80000000, 0x1000), 0x7FFFF000)
        self.assertEqual(reference_atomic_sub(0xFFFFFFFF, 0x7FFFFFFF), 0x80000000)


class TestSpirvAtomicSubNonReturning(unittest.TestCase):
    """Executes SPIR-V compute kernels exercising non-returning ds_sub_u32 operations."""

    def test_ds_sub_u32_non_returning_semantic(self):
        # ds_sub_u32: updates LDS[target], returns nothing to registers
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
%c25 = OpConstant %uint 25
%lds_arr = OpTypeArray %uint %c64
%_ptr_WG_lds = OpTypePointer Workgroup %lds_arr
%_ptr_WG_uint = OpTypePointer Workgroup %uint
%lds = OpVariable %_ptr_WG_lds Workgroup
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
%ptr = OpAccessChain %_ptr_WG_uint %lds %c3
%dummy = OpAtomicISub %uint %ptr %c2 %c0 %c25
OpReturn
OpFunctionEnd
"""
        interp = LdsAtomicSubInterp(spvasm)
        final_lds, _ = interp.run(seed_lds={3: 100})
        # Initial 100 - subtrahend 25 = 75
        self.assertEqual(final_lds[3], 75)
        # Other slots stay pristine
        self.assertEqual(final_lds[0], 0)
        self.assertEqual(final_lds[2], 0)
        self.assertEqual(final_lds[4], 0)

    def test_multiple_independent_subtractions(self):
        # Subtractions on distinct slots do not interfere with each other
        spvasm = """
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
%uint = OpTypeInt 32 0
%c32 = OpConstant %uint 32
%c2 = OpConstant %uint 2
%c0 = OpConstant %uint 0
%c1 = OpConstant %uint 1
%c5 = OpConstant %uint 5
%c9 = OpConstant %uint 9
%c10 = OpConstant %uint 10
%c20 = OpConstant %uint 20
%c30 = OpConstant %uint 30
%lds_arr = OpTypeArray %uint %c32
%_ptr_WG_lds = OpTypePointer Workgroup %lds_arr
%_ptr_WG_uint = OpTypePointer Workgroup %uint
%lds = OpVariable %_ptr_WG_lds Workgroup
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
%p1 = OpAccessChain %_ptr_WG_uint %lds %c1
%d1 = OpAtomicISub %uint %p1 %c2 %c0 %c10
%p5 = OpAccessChain %_ptr_WG_uint %lds %c5
%d5 = OpAtomicISub %uint %p5 %c2 %c0 %c20
%p9 = OpAccessChain %_ptr_WG_uint %lds %c9
%d9 = OpAtomicISub %uint %p9 %c2 %c0 %c30
OpReturn
OpFunctionEnd
"""
        interp = LdsAtomicSubInterp(spvasm)
        final_lds, _ = interp.run(seed_lds={1: 50, 5: 100, 9: 200})
        self.assertEqual(final_lds[1], 40)
        self.assertEqual(final_lds[5], 80)
        self.assertEqual(final_lds[9], 170)

    def test_repeated_subtractions_same_slot(self):
        # Sequential subtractions cumulatively decrement the targeted slot
        spvasm = """
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
%uint = OpTypeInt 32 0
%c16 = OpConstant %uint 16
%c2 = OpConstant %uint 2
%c0 = OpConstant %uint 0
%c2_slot = OpConstant %uint 2
%c1 = OpConstant %uint 1
%c2_val = OpConstant %uint 2
%c3_val = OpConstant %uint 3
%lds_arr = OpTypeArray %uint %c16
%_ptr_WG_lds = OpTypePointer Workgroup %lds_arr
%_ptr_WG_uint = OpTypePointer Workgroup %uint
%lds = OpVariable %_ptr_WG_lds Workgroup
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
%ptr = OpAccessChain %_ptr_WG_uint %lds %c2_slot
%d1 = OpAtomicISub %uint %ptr %c2 %c0 %c1
%d2 = OpAtomicISub %uint %ptr %c2 %c0 %c2_val
%d3 = OpAtomicISub %uint %ptr %c2 %c0 %c3_val
OpReturn
OpFunctionEnd
"""
        interp = LdsAtomicSubInterp(spvasm)
        final_lds, _ = interp.run(seed_lds={2: 20})
        # 20 - 1 - 2 - 3 = 14
        self.assertEqual(final_lds[2], 14)


class TestTranslatorSubIntegration(unittest.TestCase):
    """Verifies that tools/rdna2spirv.py emits conforming SPIR-V for ds_sub_u32."""

    def _make_mock_gen(self, stage: str = "cs"):
        class MockGen:
            def __init__(self):
                self.stage = stage
                self.uses_lds = False
                self.lds_dwords = 256
                self.wave64 = True
                self._tmp = 0
                self.emitted: list[str] = []
                self.stored_dsts: list[tuple[dict, str]] = []

            def tmp(self) -> str:
                self._tmp += 1
                return f"%t{self._tmp}"

            def cu(self, v: int) -> str:
                return f"%c{v}"

            def active_b(self) -> str:
                return "%exec_active"

            def src_u(self, arg: dict) -> str:
                return f"%v_{arg['r']}"

            def dst(self, d: dict, v: str) -> None:
                self.stored_dsts.append((d, v))

            def emit(self, line: str) -> None:
                self.emitted.append(line)

            LDS_ATOMICS = rdna2spirv.Gen.LDS_ATOMICS
            mod_int = staticmethod(rdna2spirv.Gen.mod_int)
            lds_ptr = rdna2spirv.Gen.lds_ptr
            lds_op = rdna2spirv.Gen.lds_op

        return MockGen()

    def test_lds_atomics_table_contains_ds_sub(self):
        # Table maps non-returning opcode name to OpAtomicISub
        self.assertIn("ds_sub_u32", rdna2spirv.Gen.LDS_ATOMICS)
        self.assertEqual(rdna2spirv.Gen.LDS_ATOMICS["ds_sub_u32"], "OpAtomicISub")

    def test_ds_sub_u32_non_returning_emission(self):
        gen = self._make_mock_gen(stage="cs")
        # ds_sub_u32 v0, v1, offset:8 (args: addr=v0, data=v1)
        gen.lds_op("ds_sub_u32", [{"r": 0}, {"r": 1}], ["offset:8"])
        self.assertTrue(gen.uses_lds)
        emitted_str = "\n".join(gen.emitted)

        # Must branch on active mask
        self.assertIn("OpBranchConditional %exec_active", emitted_str)
        # Must compute pointer with offset 8 and mask 255
        self.assertIn("OpIAdd %uint %v_0 %c8", emitted_str)
        self.assertIn("OpShiftRightLogical %uint", emitted_str)
        self.assertIn("OpBitwiseAnd %uint", emitted_str)
        self.assertIn("OpAccessChain %_ptr_WG_uint %lds", emitted_str)
        # Must emit OpAtomicISub with scope Workgroup (2) and Relaxed (0)
        self.assertIn("OpAtomicISub %uint", emitted_str)
        self.assertIn("%c2 %c0 %v_1", emitted_str)
        # Non-returning form must NOT call dst()
        self.assertEqual(len(gen.stored_dsts), 0)


if __name__ == "__main__":
    unittest.main()
