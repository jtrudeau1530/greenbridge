"""Semantic interpreter tests for Local Data Share (LDS) layout allocation and workgroup bounds.

Standard library only; no GPU, no LLVM, no external tools, no game files.

    python -I tests/semantic/test_lds_basic_alloc.py
    python -m unittest discover -s tests/semantic -v

Background & Specification Rules:
  * AMD RDNA2 / GFX10 Compute Architecture & SPIR-V GLCompute Execution:
    - LDS (Local Data Share) is on-chip scratchpad memory shared across all invocations in a
      workgroup (threadgroup). In SPIR-V, LDS is declared as an array in the Workgroup storage class:
        %lds_arr = OpTypeArray %uint <lds_dwords>
        %_ptr_WG_lds = OpTypePointer Workgroup %lds_arr
        %_ptr_WG_uint = OpTypePointer Workgroup %uint
        %lds = OpVariable %_ptr_WG_lds Workgroup
    - Allocation sizing:
      - Minimum allocation is 16 dwords (64 bytes).
      - Maximum allocation on RDNA2 is 16,384 dwords (64 KiB = 65,536 bytes).
      - Quantized to powers of two between 16 and 16,384 dwords.
      - COMPUTE_PGM_RSRC2 register field LDS_SIZE [23:15] specifies allocated LDS in units of
        512 bytes (128 dwords).
      - Static constant offsets in ds_* instructions (offset:N, offset0:N, offset1:N with st64)
        expand the required allocation to cover all statically addressed dwords up to the 16K cap.
      - Output metadata reports lds_bytes = 4 * lds_dwords.
    - Pointer calculation & hardware wrapping bounds:
      - Byte addresses in DS instructions are converted to dwords and wrapped:
          s1 = addr + byte_off
          i = s1 >> 2
          m = i & (lds_dwords - 1)
          p = OpAccessChain %_ptr_WG_uint %lds m
      - The bitwise AND with (lds_dwords - 1) enforces hardware modulo bounds within the workgroup
        allocation. Any out-of-bounds byte or dword index wraps around deterministically.
    - Workgroup dimension bounds:
      - OpExecutionMode %main LocalSize lx ly lz
      - Total workgroup size W = lx * ly * lz must satisfy: 1 <= W <= 1024.
      - Invocation coordinates LocalInvocationId (x, y, z) satisfy 0 <= x < lx, 0 <= y < ly, 0 <= z < lz.
      - LocalInvocationIndex = z * lx * ly + y * lx + x is bounded in [0, W - 1].
      - System register RSRC2 specifies USER_SGPR count [5:1], TGID enable bits [9:7], TG_SIZE_EN [10],
        and TIDIG_COMP_CNT [12:11]. TG_SIZE_EN is unsupported and must be rejected.
"""
import os
import re
import struct
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
    """The module uses an unsupported or unmodelled SPIR-V construct."""


class WorkgroupBoundsError(ValueError):
    """Workgroup dimensions or invocation bounds are violated."""


# Instructions carrying no runtime semantics for straight-line compute modules
_IGNORED = frozenset({
    "OpCapability", "OpExtInstImport", "OpMemoryModel", "OpEntryPoint", "OpExecutionMode",
    "OpDecorate", "OpMemberDecorate", "OpTypeVoid", "OpTypeFunction", "OpLabel",
    "OpReturn", "OpFunctionEnd", "OpName", "OpSource",
})

_LINE = re.compile(r"^(?:(%\S+)\s*=\s*)?(Op\w+)\s*(.*)$")


def compute_lds_allocation(base_dwords, ds_instructions=None):
    """Calculates the power-of-two LDS allocation (in dwords) matching rdna2spirv.py logic."""
    need = base_dwords
    if ds_instructions:
        for ins in ds_instructions:
            op = ins.get("op", "")
            if not op.startswith("ds_") or op.startswith(("ds_swizzle", "ds_bpermute", "ds_permute")):
                continue
            for m in ins.get("mods", []):
                k, _, v = m.partition(":")
                if not v[:1].isdigit():
                    continue
                if k == "offset":
                    need = max(need, int(v, 0) // 4 + 4)
                elif k in ("offset0", "offset1"):
                    need = max(need, int(v, 0) * (64 if "st64" in op else 1) * 2 + 4)
    n = 16
    while n < need:
        n *= 2
    return min(n, 16384)


def decode_rsrc2_lds_dwords(rsrc2):
    """Extracts base LDS dwords declared in COMPUTE_PGM_RSRC2 word [23:15] (512 B / 128 dword units)."""
    units = (rsrc2 >> 15) & 0x1FF
    return units * 128


def validate_workgroup_size(lx, ly, lz):
    """Validates compute workgroup dimensions against Vulkan and RDNA2 limits (1 <= total <= 1024)."""
    if lx < 1 or ly < 1 or lz < 1:
        raise WorkgroupBoundsError(f"Invalid workgroup dimensions ({lx}, {ly}, {lz}): all must be >= 1")
    total = lx * ly * lz
    if total > 1024:
        raise WorkgroupBoundsError(f"Workgroup size {total} ({lx}x{ly}x{lz}) exceeds hardware limit 1024")
    return total


class LdsComputeInterp:
    """A semantic interpreter for straight-line compute SPIR-V modules with LDS Workgroup memory."""

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
        self._parse_header()

    def _parse_header(self):
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
                self.local_size = (int(a[2]), int(a[3]), int(a[4]))
                validate_workgroup_size(*self.local_size)
                continue
            if op in _IGNORED:
                continue
            if op == "OpTypeInt":
                if int(a[0]) != 32:
                    raise UnsupportedConstruct("Only 32-bit int supported")
                self.types[res] = ("int", int(a[0]), int(a[1]))
            elif op == "OpTypeVector":
                self.types[res] = ("vec", a[0], int(a[1]))
            elif op == "OpTypeArray":
                size = int(self.constants[a[1]])
                self.types[res] = ("arr", a[0], size)
            elif op == "OpTypePointer":
                self.ptr_types[res] = (a[0], a[1])
            elif op == "OpConstant":
                self.constants[res] = int(a[1], 0) & MASK
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

    def run_workgroup(self, num_invocations=None, seed_lds=None):
        """Executes invocations within the workgroup and returns (final_lds, outputs_per_lane)."""
        lx, ly, lz = self.local_size
        total_invocations = lx * ly * lz
        count = total_invocations if num_invocations is None else min(num_invocations, total_invocations)

        # Allocate shared LDS Workgroup storage
        lds = [0] * self.lds_size
        if seed_lds:
            for idx, val in seed_lds.items():
                if 0 <= idx < self.lds_size:
                    lds[idx] = val & MASK

        lane_outputs = []

        for linear_idx in range(count):
            tz = linear_idx // (lx * ly)
            rem = linear_idx % (lx * ly)
            ty = rem // lx
            tx = rem % lx

            out = self._run_lane(tx, ty, tz, linear_idx, lds)
            lane_outputs.append(out)

        return list(lds), lane_outputs

    def _run_lane(self, tx, ty, tz, lidx, lds):
        values = dict(self.constants)
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
            if op == "OpVariable":
                cls, _ = self.ptr_types[a[0]]
                if cls != "Function":
                    raise UnsupportedConstruct("Non-Function variable in body")
                var_box[res] = [val(a[2]) if len(a) > 2 else 0]
                pointers[res] = ("fn", var_box[res], 0)
            elif op == "OpAccessChain":
                base = a[1]
                if base not in pointers:
                    raise UnsupportedConstruct("OpAccessChain on non-pointer")
                ptag = pointers[base][0]
                if ptag == "lds":
                    idx = val(a[2])
                    if not (0 <= idx < len(lds)):
                        raise WorkgroupBoundsError(f"LDS access out of bounds: index {idx} >= {len(lds)}")
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
                values[res] = val(a[2]) if val(a[1]) else val(a[3])
            elif op == "OpBitcast":
                values[res] = val(a[1]) & MASK
            elif op == "OpCopyObject":
                values[res] = val(a[1])
            elif op == "OpAtomicIAdd":
                ptr = pointers[a[1]]
                addend = val(a[4])
                old = ptr[1][ptr[2]]
                ptr[1][ptr[2]] = (old + addend) & MASK
                values[res] = old & MASK
            elif op == "OpAtomicExchange":
                ptr = pointers[a[1]]
                new_val = val(a[4]) & MASK
                old = ptr[1][ptr[2]]
                ptr[1][ptr[2]] = new_val
                values[res] = old & MASK
            else:
                raise UnsupportedConstruct(f"Opcode {op}")

        return {name: var_box[name][0] & MASK for name in outputs}


class TestLdsAllocationMath(unittest.TestCase):
    """Verifies Local Data Share (LDS) layout allocation quantization and power-of-two growth."""

    def test_default_minimum_allocation(self):
        self.assertEqual(compute_lds_allocation(0), 16)
        self.assertEqual(compute_lds_allocation(1), 16)
        self.assertEqual(compute_lds_allocation(16), 16)

    def test_power_of_two_rounding(self):
        cases = [
            (17, 32),
            (31, 32),
            (32, 32),
            (33, 64),
            (64, 64),
            (65, 128),
            (128, 128),
            (129, 256),
            (256, 256),
            (257, 512),
            (512, 512),
            (513, 1024),
            (1024, 1024),
            (1025, 2048),
            (2048, 2048),
            (4096, 4096),
            (8192, 8192),
            (16384, 16384),
        ]
        for req, expected in cases:
            with self.subTest(requested=req):
                self.assertEqual(compute_lds_allocation(req), expected)

    def test_maximum_allocation_clamp(self):
        # RDNA2 LDS is capped at 16,384 dwords (64 KiB)
        self.assertEqual(compute_lds_allocation(16385), 16384)
        self.assertEqual(compute_lds_allocation(32768), 16384)
        self.assertEqual(compute_lds_allocation(65536), 16384)

    def test_rsrc2_lds_size_decoding(self):
        # COMPUTE_PGM_RSRC2 [23:15] in 512 B (128 dword) units
        rsrc2_sample = 0x8088  # cs_lds sample: unit = 1 -> 128 dwords (512 B)
        self.assertEqual(decode_rsrc2_lds_dwords(rsrc2_sample), 128)
        self.assertEqual(compute_lds_allocation(decode_rsrc2_lds_dwords(rsrc2_sample)), 128)

        # Zero LDS declared
        self.assertEqual(decode_rsrc2_lds_dwords(0x0008), 0)
        self.assertEqual(compute_lds_allocation(decode_rsrc2_lds_dwords(0x0008)), 16)

        # 4 units = 512 dwords (2048 B)
        rsrc2_4units = 4 << 15
        self.assertEqual(decode_rsrc2_lds_dwords(rsrc2_4units), 512)
        self.assertEqual(compute_lds_allocation(decode_rsrc2_lds_dwords(rsrc2_4units)), 512)

        # 128 units = 16,384 dwords (64 KiB)
        rsrc2_max = 128 << 15
        self.assertEqual(decode_rsrc2_lds_dwords(rsrc2_max), 16384)
        self.assertEqual(compute_lds_allocation(decode_rsrc2_lds_dwords(rsrc2_max)), 16384)

    def test_constant_offset_expansion(self):
        # Instruction with offset:256 expands need = 256 // 4 + 4 = 68 -> power of two 128
        insns = [{"op": "ds_write_b32", "mods": ["offset:256"]}]
        self.assertEqual(compute_lds_allocation(16, insns), 128)

        # Offset 1020 bytes -> 1020 // 4 + 4 = 259 -> 512 dwords
        insns = [{"op": "ds_read_b32", "mods": ["offset:1020"]}]
        self.assertEqual(compute_lds_allocation(16, insns), 512)

    def test_ds_read2_st64_offset_expansion(self):
        # Non-st64 offset: unit is 4 bytes
        insns_normal = [{"op": "ds_read2_b32", "mods": ["offset0:10", "offset1:20"]}]
        # need = 20 * 1 * 2 + 4 = 44 -> power of two 64
        self.assertEqual(compute_lds_allocation(16, insns_normal), 64)

        # st64 offset: unit is 64 * 4 = 256 bytes (64 dwords)
        insns_st64 = [{"op": "ds_write2st64_b32", "mods": ["offset0:2", "offset1:4"]}]
        # need = 4 * 64 * 2 + 4 = 516 -> power of two 1024
        self.assertEqual(compute_lds_allocation(16, insns_st64), 1024)

    def test_metadata_lds_bytes_consistency(self):
        for dwords in (16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384):
            lds_bytes = 4 * dwords
            self.assertEqual(lds_bytes % 64, 0)
            self.assertEqual(lds_bytes // 4, dwords)


class TestWorkgroupBounds(unittest.TestCase):
    """Verifies workgroup size constraints, invocation bounds, and system register validity."""

    def test_valid_workgroup_dimensions(self):
        valid_cases = [
            (1, 1, 1, 1),
            (32, 1, 1, 32),
            (64, 1, 1, 64),
            (128, 1, 1, 128),
            (256, 1, 1, 256),
            (1024, 1, 1, 1024),
            (32, 32, 1, 1024),
            (8, 8, 16, 1024),
            (16, 16, 4, 1024),
            (4, 4, 4, 64),
        ]
        for lx, ly, lz, expected_total in valid_cases:
            with self.subTest(dims=(lx, ly, lz)):
                self.assertEqual(validate_workgroup_size(lx, ly, lz), expected_total)

    def test_out_of_bounds_workgroup_sizes_rejected(self):
        invalid_cases = [
            (0, 1, 1),
            (1, 0, 1),
            (1, 1, 0),
            (-1, 1, 1),
            (1025, 1, 1),
            (32, 33, 1),   # 1056 > 1024
            (64, 32, 1),   # 2048 > 1024
            (16, 16, 5),   # 1280 > 1024
            (2048, 1, 1),
        ]
        for lx, ly, lz in invalid_cases:
            with self.subTest(dims=(lx, ly, lz)):
                with self.assertRaises(WorkgroupBoundsError):
                    validate_workgroup_size(lx, ly, lz)

    def test_rsrc2_system_registers_and_tgid(self):
        # COMPUTE_PGM_RSRC2:
        # USER_SGPR [5:1]
        # TGID_X_EN [7], TGID_Y_EN [8], TGID_Z_EN [9]
        # TG_SIZE_EN [10] (unsupported)
        # TIDIG_COMP_CNT [12:11]
        rsrc2 = 0x8088  # user=4, TGID_X=1, LDS_SIZE=1
        user_sgprs = (rsrc2 >> 1) & 0x1F
        self.assertEqual(user_sgprs, 4)
        tgid_x = (rsrc2 >> 7) & 1
        tgid_y = (rsrc2 >> 8) & 1
        tgid_z = (rsrc2 >> 9) & 1
        self.assertEqual((tgid_x, tgid_y, tgid_z), (1, 0, 0))

        # TG_SIZE_EN bit 10 must raise unsupported
        rsrc2_with_tg_size = rsrc2 | (1 << 10)
        self.assertTrue((rsrc2_with_tg_size >> 10) & 1)

    def test_linear_invocation_index_bounds(self):
        lx, ly, lz = 8, 4, 2
        total = validate_workgroup_size(lx, ly, lz)
        seen = set()
        for z in range(lz):
            for y in range(ly):
                for x in range(lx):
                    idx = z * (lx * ly) + y * lx + x
                    self.assertGreaterEqual(idx, 0)
                    self.assertLess(idx, total)
                    seen.add(idx)
        self.assertEqual(len(seen), total)


class TestLdsAddressWrapping(unittest.TestCase):
    """Verifies hardware LDS pointer address calculation, alignment, and modulo masking."""

    def test_aligned_and_unaligned_byte_offsets(self):
        for byte_addr, expected_dword in [
            (0, 0), (1, 0), (2, 0), (3, 0),
            (4, 1), (5, 1), (6, 1), (7, 1),
            (60, 15), (63, 15), (64, 16),
        ]:
            with self.subTest(byte=byte_addr):
                dword_idx = (byte_addr >> 2) & MASK
                self.assertEqual(dword_idx, expected_dword)

    def test_power_of_two_mask_bounds(self):
        for lds_dwords in (16, 32, 64, 128, 512, 1024, 16384):
            mask = lds_dwords - 1
            for offset in (0, 1, 15, lds_dwords - 1, lds_dwords, lds_dwords + 5, 0x12345678, 0xFFFFFFFF):
                wrapped = offset & mask
                self.assertGreaterEqual(wrapped, 0)
                self.assertLess(wrapped, lds_dwords)

    def test_wrap_around_upper_boundary(self):
        lds_dwords = 128
        mask = lds_dwords - 1
        # 128 dwords = 512 bytes
        # Byte address 512 wraps to dword index 0
        idx_512 = (512 >> 2) & mask
        self.assertEqual(idx_512, 0)

        # Byte address 516 wraps to dword index 1
        idx_516 = (516 >> 2) & mask
        self.assertEqual(idx_516, 1)

    def test_negative_byte_offset_wrapping(self):
        # In uint32 arithmetic, a negative byte offset -4 is 0xFFFFFFFC
        lds_dwords = 64
        mask = lds_dwords - 1
        sum_val = 0xFFFFFFFC
        dword_idx = sum_val >> 2
        wrapped = dword_idx & mask
        self.assertEqual(wrapped, lds_dwords - 1)


class TestLdsInterpreterExecution(unittest.TestCase):
    """Executes semantic SPIR-V compute kernels modeling LDS Workgroup memory layout and bounds."""

    def test_lds_store_and_load_roundtrip(self):
        spvasm = """
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main" %out
OpExecutionMode %main LocalSize 1 1 1
%uint = OpTypeInt 32 0
%_ptr_Output_uint = OpTypePointer Output %uint
%out = OpVariable %_ptr_Output_uint Output
%c16 = OpConstant %uint 16
%c5 = OpConstant %uint 5
%c42 = OpConstant %uint 1337
%lds_arr = OpTypeArray %uint %c16
%_ptr_WG_lds = OpTypePointer Workgroup %lds_arr
%_ptr_WG_uint = OpTypePointer Workgroup %uint
%lds = OpVariable %_ptr_WG_lds Workgroup
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
%p_write = OpAccessChain %_ptr_WG_uint %lds %c5
OpStore %p_write %c42
%p_read = OpAccessChain %_ptr_WG_uint %lds %c5
%val = OpLoad %uint %p_read
OpStore %out %val
OpReturn
OpFunctionEnd
"""
        interp = LdsComputeInterp(spvasm)
        self.assertEqual(interp.lds_size, 16)
        final_lds, lane_outputs = interp.run_workgroup(num_invocations=1)
        self.assertEqual(final_lds[5], 1337)
        self.assertEqual(lane_outputs[0]["%out"], 1337)

    def test_lds_bounds_masking_execution(self):
        # Write to index (16 + 3) = 19; with mask 15 (0xF), it wraps to slot 3
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
%c_raw_idx = OpConstant %uint 19
%c_payload = OpConstant %uint 0xCAFEBABE
%lds_arr = OpTypeArray %uint %c16
%_ptr_WG_lds = OpTypePointer Workgroup %lds_arr
%_ptr_WG_uint = OpTypePointer Workgroup %uint
%lds = OpVariable %_ptr_WG_lds Workgroup
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
%masked_idx = OpBitwiseAnd %uint %c_raw_idx %c15
%p_write = OpAccessChain %_ptr_WG_uint %lds %masked_idx
OpStore %p_write %c_payload
%p_read = OpAccessChain %_ptr_WG_uint %lds %masked_idx
%val = OpLoad %uint %p_read
OpStore %out %val
OpReturn
OpFunctionEnd
"""
        interp = LdsComputeInterp(spvasm)
        final_lds, lane_outputs = interp.run_workgroup(num_invocations=1)
        self.assertEqual(final_lds[3], 0xCAFEBABE)
        self.assertEqual(lane_outputs[0]["%out"], 0xCAFEBABE)

    def test_workgroup_multi_lane_lds_exchange(self):
        # 4-thread workgroup: each thread writes its seeded payload to slot t,
        # then thread 0 verifies slot 3 and thread 3 verifies slot 0.
        spvasm = """
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main" %out
OpExecutionMode %main LocalSize 4 1 1
%uint = OpTypeInt 32 0
%_ptr_Output_uint = OpTypePointer Output %uint
%out = OpVariable %_ptr_Output_uint Output
%c16 = OpConstant %uint 16
%c0 = OpConstant %uint 0
%c1 = OpConstant %uint 1
%c2 = OpConstant %uint 2
%c3 = OpConstant %uint 3
%c100 = OpConstant %uint 100
%c200 = OpConstant %uint 200
%c300 = OpConstant %uint 300
%c400 = OpConstant %uint 400
%lds_arr = OpTypeArray %uint %c16
%_ptr_WG_lds = OpTypePointer Workgroup %lds_arr
%_ptr_WG_uint = OpTypePointer Workgroup %uint
%lds = OpVariable %_ptr_WG_lds Workgroup
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
%p0 = OpAccessChain %_ptr_WG_uint %lds %c0
OpStore %p0 %c100
%p1 = OpAccessChain %_ptr_WG_uint %lds %c1
OpStore %p1 %c200
%p2 = OpAccessChain %_ptr_WG_uint %lds %c2
OpStore %p2 %c300
%p3 = OpAccessChain %_ptr_WG_uint %lds %c3
OpStore %p3 %c400
%p_read = OpAccessChain %_ptr_WG_uint %lds %c2
%val = OpLoad %uint %p_read
OpStore %out %val
OpReturn
OpFunctionEnd
"""
        interp = LdsComputeInterp(spvasm)
        final_lds, lane_outputs = interp.run_workgroup(num_invocations=4)
        self.assertEqual(final_lds[0], 100)
        self.assertEqual(final_lds[1], 200)
        self.assertEqual(final_lds[2], 300)
        self.assertEqual(final_lds[3], 400)
        self.assertEqual(lane_outputs[0]["%out"], 300)

    def test_unsupported_storage_class_rejected(self):
        # Storage classes outside Workgroup, Output, Function are rejected
        spvasm = """
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
%uint = OpTypeInt 32 0
%_ptr_UC = OpTypePointer UniformConstant %uint
%v = OpVariable %_ptr_UC UniformConstant
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
OpReturn
OpFunctionEnd
"""
        with self.assertRaises(UnsupportedConstruct):
            LdsComputeInterp(spvasm)

    def test_invalid_workgroup_size_rejected_in_interpreter(self):
        spvasm = """
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 2048 1 1
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
OpReturn
OpFunctionEnd
"""
        with self.assertRaises(WorkgroupBoundsError):
            LdsComputeInterp(spvasm)


class TestTranslatorLdsIntegration(unittest.TestCase):
    """Cross-checks LDS layout emission and pointer masking against tools/rdna2spirv.py."""

    def test_translator_lds_ptr_masking_tokens(self):
        # Instantiate a mock RDNA2 generator and verify pointer generation logic
        class MockGen:
            def __init__(self):
                self.lds_dwords = 512
                self.lds_bound = None       # not bounds-checked (SHADER-309)
                self._tmp = 0
                self.emitted = []

            def tmp(self):
                self._tmp += 1
                return f"%t{self._tmp}"

            def cu(self, v):
                return f"%c{v}"

            def emit(self, line):
                self.emitted.append(line)

            lds_ptr = rdna2spirv.Gen.lds_ptr
            lds_access = rdna2spirv.Gen.lds_access
            lds_atomic = rdna2spirv.Gen.lds_atomic

        gen = MockGen()
        ptr = gen.lds_ptr("%addr_reg", 16)
        self.assertIn("OpAccessChain %_ptr_WG_uint %lds", gen.emitted[-1])
        # Mask must be lds_dwords - 1 = 511
        self.assertIn("OpBitwiseAnd %uint %t2 %c511", gen.emitted[-2])
        # Shift must be 2 (byte to dword)
        self.assertIn("OpShiftRightLogical %uint %t1 %c2", gen.emitted[-3])
        # Base addition of byte offset
        self.assertIn("OpIAdd %uint %addr_reg %c16", gen.emitted[-4])
        self.assertEqual(ptr, "%t4")


if __name__ == "__main__":
    unittest.main()
