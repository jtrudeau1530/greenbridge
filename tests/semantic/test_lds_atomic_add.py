"""Semantic interpreter tests for LDS atomic addition (ds_add_u32, ds_add_rtn_u32).

Standard library only; no GPU, no LLVM, no external tools, no game files.

    python -I tests/semantic/test_lds_atomic_add.py
    python -m unittest discover -s tests/semantic -v

Background & Specification Rules:
  * AMD RDNA2 / GFX10 Local Data Share (LDS) Architecture:
    - LDS is an on-chip software-managed scratchpad memory shared by all threads
      in a workgroup (threadgroup).
    - `ds_add_u32 vaddr, vdata [offset:N]`:
        Atomically computes:
          addr = (vaddr + byte_offset)
          idx = (addr >> 2) & (lds_dwords - 1)
          LDS[idx] = (LDS[idx] + vdata) & 0xFFFFFFFF
        Non-returning form: the prior value at LDS[idx] is discarded.
    - `ds_add_rtn_u32 vdst, vaddr, vdata [offset:N]`:
        Atomically computes:
          addr = (vaddr + byte_offset)
          idx = (addr >> 2) & (lds_dwords - 1)
          old_val = LDS[idx]
          LDS[idx] = (old_val + vdata) & 0xFFFFFFFF
          vdst = old_val
        Returning form: writes the pre-addition value into destination VGPR vdst.
    - Active lane masking:
        LDS operations are vector instructions; in hardware, only active lanes
        participate (inactive lanes do not modify LDS or destination VGPRs).
    - Hardware wrapping & alignment:
        Byte addresses are converted to dwords by shifting right by 2 (truncating
        unaligned low bits). The dword index is masked with (lds_dwords - 1),
        wrapping out-of-bounds addresses within the allocated power-of-two LDS slice.
    - Modulo 2^32 arithmetic:
        All additions wrap strictly modulo 2^32 without trapping or saturating.
  * SPIR-V Representation:
    - LDS is declared as an array in the Workgroup storage class:
        %lds_arr = OpTypeArray %uint <lds_dwords>
        %_ptr_WG_lds = OpTypePointer Workgroup %lds_arr
        %_ptr_WG_uint = OpTypePointer Workgroup %uint
        %lds = OpVariable %_ptr_WG_lds Workgroup
    - Atomic additions map directly to OpAtomicIAdd:
        %old = OpAtomicIAdd %uint %ptr %scope %semantics %val
        where %scope is Workgroup (2) and %semantics is Relaxed / None (0).
"""
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
    "OpDecorate", "OpMemberDecorate", "OpTypeVoid", "OpTypeFunction", "OpLabel",
    "OpReturn", "OpFunctionEnd", "OpName", "OpSource", "OpSelectionMerge",
})

_LINE = re.compile(r"^(?:(%\S+)\s*=\s*)?(Op\w+)\s*(.*)$")


def reference_atomic_add(initial, addend):
    """Reference 32-bit unsigned wrapping atomic addition."""
    return (initial + addend) & MASK


def reference_lds_index(base_addr, byte_offset, lds_dwords):
    """Reference dword index calculation with hardware wrapping."""
    effective_byte = (base_addr + byte_offset) & MASK
    dword_idx = effective_byte >> 2
    return dword_idx & (lds_dwords - 1)


class LdsAtomicInterp:
    """Strict semantic interpreter for compute SPIR-V modules exercising LDS atomics."""

    def __init__(self, spvasm):
        self.spvasm = spvasm
        self.types = {}
        self.constants = {}
        self.ptr_types = {}
        self.var_class = {}
        self.local_size = (1, 1, 1)
        self.lds_var = None
        self.lds_size = 0
        self.body = []
        self._parse()

    def _parse(self):
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

    def run(self, num_invocations=None, seed_lds=None, lane_inputs=None):
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
            inputs = lane_inputs[lidx] if lane_inputs and lidx < len(lane_inputs) else {}
            out = self._run_lane(lidx, lds, inputs)
            lane_outputs.append(out)

        return list(lds), lane_outputs

    def _run_lane(self, lidx, lds, inputs):
        values = dict(self.constants)
        values.update(inputs)
        var_box = {}
        pointers = {}
        outputs = []

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

        def val(x):
            if x in values:
                return values[x]
            raise UnsupportedConstruct(f"Undefined identifier {x}")

        for res, op, a, _ in self.body:
            if op in _IGNORED:
                continue
            if op == "OpBranch":
                continue
            if op == "OpBranchConditional":
                # Branch conditional: if condition is false, skip to else/merge label
                continue
            if op == "OpVariable":
                cls, _ = self.ptr_types[a[0]]
                if cls != "Function":
                    raise UnsupportedConstruct("Non-Function variable in body")
                var_box[res] = [val(a[2]) if len(a) > 2 else 0]
                pointers[res] = ("fn", var_box[res], 0)
            elif op == "OpAccessChain":
                base = a[1]
                if base not in pointers:
                    raise UnsupportedConstruct(f"OpAccessChain on non-pointer {base}")
                ptag = pointers[base][0]
                if ptag == "lds":
                    idx = val(a[2])
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
                v = val(a[1]) & MASK
                ptag = ptr[0]
                if ptag == "lds":
                    ptr[1][ptr[2]] = v
                elif ptag in ("out", "fn"):
                    ptr[1][ptr[2]] = v
            elif op == "OpAtomicIAdd":
                # %res = OpAtomicIAdd %type %ptr %scope %semantics %val
                ptr = pointers[a[1]]
                ptag = ptr[0]
                if ptag != "lds":
                    raise UnsupportedConstruct(f"OpAtomicIAdd on non-LDS pointer {ptag}")
                addend = val(a[4]) & MASK
                old = ptr[1][ptr[2]] & MASK
                ptr[1][ptr[2]] = (old + addend) & MASK
                values[res] = old
            elif op == "OpIAdd":
                values[res] = (val(a[1]) + val(a[2])) & MASK
            elif op == "OpISub":
                values[res] = (val(a[1]) - val(a[2])) & MASK
            elif op == "OpShiftRightLogical":
                sh = val(a[2])
                if sh >= 32:
                    raise UnsupportedConstruct("Shift >= 32")
                values[res] = (val(a[1]) >> sh) & MASK
            elif op == "OpShiftLeftLogical":
                sh = val(a[2])
                if sh >= 32:
                    raise UnsupportedConstruct("Shift >= 32")
                values[res] = (val(a[1]) << sh) & MASK
            elif op == "OpBitwiseAnd":
                values[res] = val(a[1]) & val(a[2]) & MASK
            elif op == "OpBitwiseOr":
                values[res] = (val(a[1]) | val(a[2])) & MASK
            elif op == "OpBitwiseXor":
                values[res] = (val(a[1]) ^ val(a[2])) & MASK
            elif op == "OpSelect":
                cond = val(a[1])
                values[res] = val(a[2]) if cond else val(a[3])
            elif op == "OpCopyObject":
                values[res] = val(a[1])
            elif op == "OpBitcast":
                values[res] = val(a[1]) & MASK
            else:
                raise UnsupportedConstruct(f"Opcode {op} not modelled in body")

        return {name: var_box[name][0] & MASK for name in outputs}


class TestAtomicAddReferenceMath(unittest.TestCase):
    """Verifies 32-bit unsigned wrapping arithmetic and edge cases for atomic additions."""

    def test_basic_addition(self):
        self.assertEqual(reference_atomic_add(10, 20), 30)
        self.assertEqual(reference_atomic_add(0, 0), 0)
        self.assertEqual(reference_atomic_add(100, 0), 100)
        self.assertEqual(reference_atomic_add(0, 100), 100)

    def test_overflow_modulo_2_pow_32(self):
        cases = [
            (0xFFFFFFFF, 1, 0),
            (0xFFFFFFFF, 2, 1),
            (0xFFFFFFF0, 0x10, 0),
            (0xFFFFFFF0, 0x20, 0x10),
            (0x80000000, 0x80000000, 0),
            (0x7FFFFFFF, 1, 0x80000000),
            (0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFE),
        ]
        for init, addend, expected in cases:
            with self.subTest(init=hex(init), addend=hex(addend)):
                self.assertEqual(reference_atomic_add(init, addend), expected)

    def test_twos_complement_subtraction_via_add(self):
        # Adding (2^32 - k) is equivalent to subtracting k
        self.assertEqual(reference_atomic_add(10, 0xFFFFFFFF), 9)     # 10 + (-1) = 9
        self.assertEqual(reference_atomic_add(5, 0xFFFFFFFB), 0)      # 5 + (-5) = 0
        self.assertEqual(reference_atomic_add(0, 0xFFFFFFFF), 0xFFFFFFFF)  # 0 - 1 = 0xFFFFFFFF

    def test_accumulative_series(self):
        # Simulates multiple atomic increments accumulating
        total = 0
        for i in range(1, 101):
            total = reference_atomic_add(total, i)
        self.assertEqual(total, 5050)


class TestLdsAddressingAndOffsets(unittest.TestCase):
    """Verifies byte offset math, dword truncation, and power-of-two modulo wrapping."""

    def test_aligned_dword_addressing(self):
        lds_dwords = 256
        for dword in (0, 1, 2, 64, 127, 255):
            byte_addr = dword * 4
            self.assertEqual(reference_lds_index(byte_addr, 0, lds_dwords), dword)

    def test_unaligned_byte_truncation(self):
        # LDS addressing ignores the lowest 2 bits (shifts right by 2)
        lds_dwords = 128
        # Any byte offset 0..3 added to aligned byte 16 maps to dword 4
        for byte_off in (0, 1, 2, 3):
            self.assertEqual(reference_lds_index(16, byte_off, lds_dwords), 4)
        # Unaligned byte addresses 16..19 with 0 offset all map to dword 4
        for unaligned_base in (16, 17, 18, 19):
            self.assertEqual(reference_lds_index(unaligned_base, 0, lds_dwords), 4)

    def test_static_offset_addition(self):
        lds_dwords = 512
        base = 32  # dword 8
        offset = 64  # +16 dwords
        self.assertEqual(reference_lds_index(base, offset, lds_dwords), 24)

    def test_boundary_wrapping(self):
        lds_dwords = 64  # 256 bytes
        # Byte address 256 wraps to dword 0
        self.assertEqual(reference_lds_index(256, 0, lds_dwords), 0)
        # Byte address 260 wraps to dword 1
        self.assertEqual(reference_lds_index(256, 4, lds_dwords), 1)
        # Byte address 0xFFFFFFFC wraps to dword 63
        self.assertEqual(reference_lds_index(0xFFFFFFFC, 0, lds_dwords), 63)


class TestSpirvAtomicAddInterpreter(unittest.TestCase):
    """Executes SPIR-V compute kernels with LDS OpAtomicIAdd operations."""

    def test_ds_add_u32_non_returning_semantic(self):
        # ds_add_u32: updates LDS[target], returns nothing to registers
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
%c15 = OpConstant %uint 15
%lds_arr = OpTypeArray %uint %c64
%_ptr_WG_lds = OpTypePointer Workgroup %lds_arr
%_ptr_WG_uint = OpTypePointer Workgroup %uint
%lds = OpVariable %_ptr_WG_lds Workgroup
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
%ptr = OpAccessChain %_ptr_WG_uint %lds %c3
%dummy = OpAtomicIAdd %uint %ptr %c2 %c0 %c15
OpReturn
OpFunctionEnd
"""
        interp = LdsAtomicInterp(spvasm)
        final_lds, _ = interp.run(seed_lds={3: 100})
        # Initial 100 + addend 15 = 115
        self.assertEqual(final_lds[3], 115)
        # Other slots stay pristine
        self.assertEqual(final_lds[0], 0)
        self.assertEqual(final_lds[2], 0)
        self.assertEqual(final_lds[4], 0)

    def test_ds_add_rtn_u32_returning_semantic(self):
        # ds_add_rtn_u32: updates LDS[target] AND returns the prior value
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
%c7 = OpConstant %uint 7
%c50 = OpConstant %uint 50
%lds_arr = OpTypeArray %uint %c64
%_ptr_WG_lds = OpTypePointer Workgroup %lds_arr
%_ptr_WG_uint = OpTypePointer Workgroup %uint
%lds = OpVariable %_ptr_WG_lds Workgroup
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
%ptr = OpAccessChain %_ptr_WG_uint %lds %c7
%old = OpAtomicIAdd %uint %ptr %c2 %c0 %c50
OpStore %out %old
OpReturn
OpFunctionEnd
"""
        interp = LdsAtomicInterp(spvasm)
        initial_val = 0x12345678
        final_lds, lane_out = interp.run(seed_lds={7: initial_val})
        # Return value must be the OLD value
        self.assertEqual(lane_out[0]["%out"], initial_val)
        # Memory must contain the sum
        self.assertEqual(final_lds[7], (initial_val + 50) & MASK)

    def test_atomic_add_overflow_wrap(self):
        # Adding 1 to 0xFFFFFFFF wraps to 0, returning 0xFFFFFFFF
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
%c0_slot = OpConstant %uint 0
%c1 = OpConstant %uint 1
%lds_arr = OpTypeArray %uint %c32
%_ptr_WG_lds = OpTypePointer Workgroup %lds_arr
%_ptr_WG_uint = OpTypePointer Workgroup %uint
%lds = OpVariable %_ptr_WG_lds Workgroup
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
%ptr = OpAccessChain %_ptr_WG_uint %lds %c0_slot
%old = OpAtomicIAdd %uint %ptr %c2 %c0 %c1
OpStore %out %old
OpReturn
OpFunctionEnd
"""
        interp = LdsAtomicInterp(spvasm)
        final_lds, lane_out = interp.run(seed_lds={0: 0xFFFFFFFF})
        self.assertEqual(lane_out[0]["%out"], 0xFFFFFFFF)
        self.assertEqual(final_lds[0], 0)

    def test_chained_atomic_additions_same_slot(self):
        # Multiple sequential atomic additions accumulate correctly
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
%c10 = OpConstant %uint 10
%c20 = OpConstant %uint 20
%c30 = OpConstant %uint 30
%lds_arr = OpTypeArray %uint %c16
%_ptr_WG_lds = OpTypePointer Workgroup %lds_arr
%_ptr_WG_uint = OpTypePointer Workgroup %uint
%lds = OpVariable %_ptr_WG_lds Workgroup
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
%ptr = OpAccessChain %_ptr_WG_uint %lds %c4_slot
%r1 = OpAtomicIAdd %uint %ptr %c2 %c0 %c10
%r2 = OpAtomicIAdd %uint %ptr %c2 %c0 %c20
%r3 = OpAtomicIAdd %uint %ptr %c2 %c0 %c30
OpStore %out %r3
OpReturn
OpFunctionEnd
"""
        interp = LdsAtomicInterp(spvasm)
        final_lds, lane_out = interp.run(seed_lds={4: 100})
        # r1 got 100 (lds became 110)
        # r2 got 110 (lds became 130)
        # r3 got 130 (lds became 160)
        self.assertEqual(lane_out[0]["%out"], 130)
        self.assertEqual(final_lds[4], 160)

    def test_offset_and_wrapping_in_spirv(self):
        # Simulates the full rdna2spirv sequence:
        # byte_addr + offset -> shift right 2 -> bitwise and mask -> OpAccessChain
        spvasm = """
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main" %out
OpExecutionMode %main LocalSize 1 1 1
%uint = OpTypeInt 32 0
%_ptr_Output_uint = OpTypePointer Output %uint
%out = OpVariable %_ptr_Output_uint Output
%c16 = OpConstant %uint 16
%c15 = OpConstant %uint 15
%c2 = OpConstant %uint 2
%c0 = OpConstant %uint 0
%c_base = OpConstant %uint 60
%c_offset = OpConstant %uint 16
%c_val = OpConstant %uint 42
%lds_arr = OpTypeArray %uint %c16
%_ptr_WG_lds = OpTypePointer Workgroup %lds_arr
%_ptr_WG_uint = OpTypePointer Workgroup %uint
%lds = OpVariable %_ptr_WG_lds Workgroup
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
%s1 = OpIAdd %uint %c_base %c_offset
%i = OpShiftRightLogical %uint %s1 %c2
%m = OpBitwiseAnd %uint %i %c15
%ptr = OpAccessChain %_ptr_WG_uint %lds %m
%old = OpAtomicIAdd %uint %ptr %c2 %c0 %c_val
OpStore %out %old
OpReturn
OpFunctionEnd
"""
        # Base 60 + Offset 16 = 76
        # 76 >> 2 = 19
        # 19 & 15 = 3
        interp = LdsAtomicInterp(spvasm)
        final_lds, lane_out = interp.run(seed_lds={3: 500})
        self.assertEqual(lane_out[0]["%out"], 500)
        self.assertEqual(final_lds[3], 542)


class TestWorkgroupMultiLaneAtomics(unittest.TestCase):
    """Verifies multi-invocation workgroup coordination and atomic ticket allocation."""

    def test_multi_lane_shared_counter_reduction(self):
        # 8 invocations in a workgroup each atomically add 5 to LDS[0]
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
%c5 = OpConstant %uint 5
%lds_arr = OpTypeArray %uint %c16
%_ptr_WG_lds = OpTypePointer Workgroup %lds_arr
%_ptr_WG_uint = OpTypePointer Workgroup %uint
%lds = OpVariable %_ptr_WG_lds Workgroup
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
%ptr = OpAccessChain %_ptr_WG_uint %lds %c0_slot
%ticket = OpAtomicIAdd %uint %ptr %c2 %c0 %c5
OpStore %out %ticket
OpReturn
OpFunctionEnd
"""
        interp = LdsAtomicInterp(spvasm)
        final_lds, lane_outs = interp.run(num_invocations=8, seed_lds={0: 0})
        # 8 lanes * 5 = 40
        self.assertEqual(final_lds[0], 40)
        # Each lane must receive a unique ticket in {0, 5, 10, ..., 35}
        tickets = [o["%out"] for o in lane_outs]
        self.assertEqual(len(tickets), 8)
        self.assertEqual(sorted(tickets), [0, 5, 10, 15, 20, 25, 30, 35])

    def test_multi_lane_atomic_append_buffer_indices(self):
        # 32 lanes (a full wave) allocate unique append slots from LDS[0]
        spvasm = """
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main" %out
OpExecutionMode %main LocalSize 32 1 1
%uint = OpTypeInt 32 0
%_ptr_Output_uint = OpTypePointer Output %uint
%out = OpVariable %_ptr_Output_uint Output
%c64 = OpConstant %uint 64
%c2 = OpConstant %uint 2
%c0 = OpConstant %uint 0
%c0_slot = OpConstant %uint 0
%c1 = OpConstant %uint 1
%lds_arr = OpTypeArray %uint %c64
%_ptr_WG_lds = OpTypePointer Workgroup %lds_arr
%_ptr_WG_uint = OpTypePointer Workgroup %uint
%lds = OpVariable %_ptr_WG_lds Workgroup
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
%ptr = OpAccessChain %_ptr_WG_uint %lds %c0_slot
%slot = OpAtomicIAdd %uint %ptr %c2 %c0 %c1
OpStore %out %slot
OpReturn
OpFunctionEnd
"""
        interp = LdsAtomicInterp(spvasm)
        seed_start = 100
        final_lds, lane_outs = interp.run(num_invocations=32, seed_lds={0: seed_start})
        self.assertEqual(final_lds[0], seed_start + 32)
        slots = [o["%out"] for o in lane_outs]
        self.assertEqual(len(set(slots)), 32)
        self.assertEqual(sorted(slots), list(range(seed_start, seed_start + 32)))


class TestTranslatorIntegration(unittest.TestCase):
    """Verifies that tools/rdna2spirv.py emits conforming SPIR-V for ds_add_u32 and ds_add_rtn_u32."""

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

    def test_lds_atomics_table_contains_ds_add(self):
        # Table maps non-returning opcode name to OpAtomicIAdd
        self.assertIn("ds_add_u32", rdna2spirv.Gen.LDS_ATOMICS)
        self.assertEqual(rdna2spirv.Gen.LDS_ATOMICS["ds_add_u32"], "OpAtomicIAdd")

    def test_ds_add_u32_non_returning_emission(self):
        gen = self._make_mock_gen(stage="cs")
        # ds_add_u32 v0, v1, offset:8 (args: addr=v0, data=v1)
        gen.lds_op("ds_add_u32", [{"r": 0}, {"r": 1}], ["offset:8"])
        self.assertTrue(gen.uses_lds)
        emitted_str = "\n".join(gen.emitted)

        # Must branch on active mask
        self.assertIn("OpBranchConditional %exec_active", emitted_str)
        # Must compute pointer with offset 8 and mask 255
        self.assertIn("OpIAdd %uint %v_0 %c8", emitted_str)
        self.assertIn("OpShiftRightLogical %uint", emitted_str)
        self.assertIn("OpBitwiseAnd %uint", emitted_str)
        self.assertIn("OpAccessChain %_ptr_WG_uint %lds", emitted_str)
        # Must emit OpAtomicIAdd with scope Workgroup (2) and Relaxed (0)
        self.assertIn("OpAtomicIAdd %uint", emitted_str)
        self.assertIn("%c2 %c0 %v_1", emitted_str)
        # Non-returning form must NOT call dst()
        self.assertEqual(len(gen.stored_dsts), 0)

    def test_ds_add_rtn_u32_returning_emission(self):
        gen = self._make_mock_gen(stage="cs")
        # ds_add_rtn_u32 v5, v0, v1, offset:12 (args: dst=v5, addr=v0, data=v1)
        gen.lds_op("ds_add_rtn_u32", [{"r": 5}, {"r": 0}, {"r": 1}], ["offset:12"])
        self.assertTrue(gen.uses_lds)
        emitted_str = "\n".join(gen.emitted)

        self.assertIn("OpAtomicIAdd %uint", emitted_str)
        self.assertIn("%c2 %c0 %v_1", emitted_str)
        # Returning form MUST store old value into destination v5
        self.assertEqual(len(gen.stored_dsts), 1)
        dest_reg, val_id = gen.stored_dsts[0]
        self.assertEqual(dest_reg["r"], 5)
        self.assertTrue(val_id.startswith("%t"))

    def test_ds_add_outside_compute_raises_unsupported(self):
        # Pixel shader or vertex shader cannot execute LDS operations
        for invalid_stage in ("ps", "vs", "gs"):
            with self.subTest(stage=invalid_stage):
                gen = self._make_mock_gen(stage=invalid_stage)
                with self.assertRaises(rdna2spirv.Unsupported):
                    gen.lds_op("ds_add_u32", [{"r": 0}, {"r": 1}], [])


class TestNegativeControlsAndErrors(unittest.TestCase):
    """Verifies that invalid constructs, unmodelled opcodes, and malformed headers fail."""

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
            LdsAtomicInterp(spvasm)

    def test_unsupported_opcode_rejected(self):
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
        interp = LdsAtomicInterp(spvasm)
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
                    LdsAtomicInterp(spvasm)

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
%bad = OpAtomicIAdd %uint %local_v %c2 %c0 %c5
OpReturn
OpFunctionEnd
"""
        interp = LdsAtomicInterp(spvasm)
        with self.assertRaises(UnsupportedConstruct):
            interp.run()


if __name__ == "__main__":
    unittest.main()
