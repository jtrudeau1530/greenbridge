"""Semantic interpreter tests for whole-wave subgroup ballot equality and ballot comparison.

Standard library only; models SPIR-V GroupNonUniform (subgroup) operations across wave32 and wave64
subgroups according to the SPIR-V 1.3+ and Vulkan Subgroup specifications.

Covers:
  * Whole-wave subgroup ballot generation (OpGroupNonUniformBallot) for wave32 and wave64.
  * Whole-wave ballot equality (OpGroupNonUniformAllEqual, component-wise ballot comparisons,
    and OpGroupNonUniformAll lane-mask reductions matching tools/rdna2spirv.py wave_equal).
  * Whole-wave ballot comparison (OpGroupNonUniformAny, OpGroupNonUniformAll,
    OpGroupNonUniformBallotBitCount for population count, OpGroupNonUniformBallotFindLSB / FindMSB,
    OpGroupNonUniformBallotBitExtract, and ballot bitwise operations).
  * Subgroup execution masks (active / inactive lanes under EXEC).
  * Strict validation and rejection of invalid scopes, types, and operands.
"""
import re
import struct
import unittest

MASK = 0xFFFFFFFF


class UnsupportedConstruct(NotImplementedError):
    """The module uses something this interpreter deliberately does not model."""


_IGNORED = frozenset({
    "OpCapability", "OpExtInstImport", "OpMemoryModel", "OpEntryPoint", "OpExecutionMode",
    "OpDecorate", "OpMemberDecorate", "OpTypeVoid", "OpTypeFunction", "OpLabel", "OpReturn",
    "OpFunctionEnd", "OpName", "OpSource", "OpExtension",
})

_LINE = re.compile(r"^(?:(%\S+)\s*=\s*)?(Op\w+)\s*(.*)$")


def _s32(x):
    x &= MASK
    return x - (1 << 32) if x >> 31 else x


def f32_bits(x):
    try:
        return struct.unpack("<I", struct.pack("<f", x))[0]
    except OverflowError:
        return 0x7F800000 if x > 0 else 0xFF800000


class _Types:
    def __init__(self):
        self.t = {}

    def zero(self, name):
        k = self.t[name]
        if k[0] == "int":
            return 0
        if k[0] == "float":
            return 0.0
        if k[0] == "bool":
            return False
        if k[0] == "vec":
            return tuple(self.zero(k[1]) for _ in range(k[2]))
        if k[0] == "arr":
            return [self.zero(k[1]) for _ in range(k[2])]
        raise UnsupportedConstruct(f"zero value of type {name} ({k[0]})")


class SubgroupInterpreter:
    """Interprets a straight-line SPIR-V module across a whole wave/subgroup of invocations."""

    def __init__(self, text, subgroup_size=32, active_mask=None):
        if not (1 <= subgroup_size <= 128):
            raise ValueError(f"unsupported subgroup_size: {subgroup_size}")
        self.subgroup_size = subgroup_size
        full_mask = (1 << subgroup_size) - 1
        self.active_mask = full_mask if active_mask is None else (active_mask & full_mask)
        self.active_lanes = [i for i in range(subgroup_size) if (self.active_mask & (1 << i)) != 0]

        self.types = _Types()
        self.constants = {}
        self.ptr_types = {}
        self.var_class = {}
        self.builtin_vars = {}   # var_name -> builtin name (e.g. SubgroupLocalInvocationId)
        self.output_vars = []
        self.body = []

        self._parse(text)

    def _parse(self, text):
        in_body = False
        decorations = {}  # target_id -> {prop: val}

        for raw in text.splitlines():
            line = raw.strip()
            if not line:
                continue
            m = _LINE.match(line)
            if not m:
                raise UnsupportedConstruct(f"cannot parse line: {line!r}")
            res, op, rest = m.groups()
            a = rest.split()

            if in_body:
                self.body.append((res, op, a, line))
                continue

            if op == "OpFunction":
                in_body = True
                continue
            if op == "OpDecorate":
                target = a[0]
                prop = a[1]
                val = a[2] if len(a) > 2 else True
                decorations.setdefault(target, {})[prop] = val
                continue
            if op in _IGNORED:
                continue

            if op == "OpTypeInt":
                width = int(a[0])
                if width != 32:
                    raise UnsupportedConstruct(f"non-32-bit integer type width {width}")
                self.types.t[res] = ("int", width, int(a[1]))
            elif op == "OpTypeFloat":
                width = int(a[0])
                if width != 32:
                    raise UnsupportedConstruct(f"non-32-bit float type width {width}")
                self.types.t[res] = ("float",)
            elif op == "OpTypeBool":
                self.types.t[res] = ("bool",)
            elif op == "OpTypeVector":
                self.types.t[res] = ("vec", a[0], int(a[1]))
            elif op == "OpTypeArray":
                self.types.t[res] = ("arr", a[0], int(self.constants[a[1]]))
            elif op == "OpTypeStruct":
                self.types.t[res] = ("struct", a)
            elif op == "OpTypePointer":
                self.ptr_types[res] = (a[0], a[1])
            elif op == "OpConstant":
                k = self.types.t[a[0]]
                if k[0] == "int":
                    self.constants[res] = int(a[1]) & MASK
                elif k[0] == "float":
                    self.constants[res] = float(a[1])
                else:
                    raise UnsupportedConstruct(f"OpConstant of type {k[0]}")
            elif op == "OpConstantTrue":
                self.constants[res] = True
            elif op == "OpConstantFalse":
                self.constants[res] = False
            elif op == "OpConstantNull":
                self.constants[res] = self.types.zero(a[0])
            elif op == "OpConstantComposite":
                self.constants[res] = tuple(self.constants[x] for x in a[1:])
            elif op == "OpVariable":
                cls, pointee = self.ptr_types[a[0]]
                self.var_class[res] = cls
                if res in decorations and "BuiltIn" in decorations[res]:
                    self.builtin_vars[res] = decorations[res]["BuiltIn"]
                if cls == "Output":
                    self.output_vars.append((res, pointee))
            else:
                raise UnsupportedConstruct(f"{op} in module header")

        if not in_body:
            raise UnsupportedConstruct("no function body found")

    def run(self):
        """Execute the module across all active lanes. Returns {output_name: [words_per_lane]}."""
        # Per-lane value and variable environments
        lane_values = {i: dict(self.constants) for i in self.active_lanes}
        lane_var_box = {i: {} for i in self.active_lanes}
        lane_pointers = {i: {} for i in self.active_lanes}

        for i in self.active_lanes:
            for out_res, pointee in self.output_vars:
                box = [self.types.zero(pointee)]
                lane_var_box[i][out_res] = box
                lane_pointers[i][out_res] = (box, 0)

        def deref(lane, p):
            box, idx = p
            return box[idx]

        for res, op, a, line in self.body:
            if op in _IGNORED:
                continue

            # Subgroup collective / non-uniform operations
            if op.startswith("OpGroupNonUniform"):
                self._step_subgroup_op(res, op, a, lane_values)
                continue

            # Standard per-lane operations
            for i in self.active_lanes:
                vals = lane_values[i]
                ptrs = lane_pointers[i]
                boxes = lane_var_box[i]

                def v(x):
                    if x in vals:
                        return vals[x]
                    raise UnsupportedConstruct(f"use of undefined id {x} in lane {i}")

                if op == "OpVariable":
                    cls, pointee = self.ptr_types[a[0]]
                    if cls != "Function":
                        raise UnsupportedConstruct(f"non-Function variable {res} in body")
                    init = v(a[2]) if len(a) > 2 else self.types.zero(pointee)
                    box = [list(init) if isinstance(init, list) else init]
                    boxes[res] = box
                    ptrs[res] = (box, 0)

                elif op == "OpAccessChain":
                    base = a[1]
                    if base not in ptrs:
                        raise UnsupportedConstruct(f"OpAccessChain on non-pointer {base}")
                    p = ptrs[base]
                    for idx_id in a[2:]:
                        idx = vals[idx_id]
                        container = deref(i, p)
                        if not isinstance(container, list) or not 0 <= idx < len(container):
                            raise UnsupportedConstruct("OpAccessChain out of range or not a container")
                        p = (container, idx)
                    ptrs[res] = p

                elif op == "OpLoad":
                    ptr_id = a[1]
                    if ptr_id in self.builtin_vars:
                        builtin = self.builtin_vars[ptr_id]
                        if builtin == "SubgroupLocalInvocationId":
                            vals[res] = i
                        elif builtin == "SubgroupSize":
                            vals[res] = self.subgroup_size
                        elif builtin == "SubgroupEqMask":
                            mask_val = 1 << i
                            vals[res] = (mask_val & MASK, (mask_val >> 32) & MASK, 0, 0)
                        else:
                            raise UnsupportedConstruct(f"unsupported builtin {builtin}")
                    else:
                        p = ptrs[ptr_id]
                        loaded = deref(i, p)
                        vals[res] = list(loaded) if isinstance(loaded, list) else loaded

                elif op == "OpStore":
                    p = ptrs[a[0]]
                    p[0][p[1]] = v(a[1])

                elif op == "OpCopyObject":
                    vals[res] = v(a[1])

                elif op == "OpSelect":
                    cond = v(a[1])
                    if not isinstance(cond, bool):
                        raise UnsupportedConstruct("OpSelect with non-bool condition")
                    vals[res] = v(a[2]) if cond else v(a[3])

                elif op == "OpBitcast":
                    k = self.types.t[a[0]]
                    src = v(a[1])
                    if k[0] == "int" and isinstance(src, int):
                        vals[res] = src & MASK
                    elif k[0] == "int" and isinstance(src, float):
                        vals[res] = f32_bits(src)
                    elif k[0] == "float" and isinstance(src, int):
                        vals[res] = struct.unpack("<f", struct.pack("<I", src & MASK))[0]
                    else:
                        raise UnsupportedConstruct("OpBitcast between unsupported types")

                elif op == "OpCompositeExtract":
                    comp = v(a[1])
                    for idx_str in a[2:]:
                        comp = comp[int(idx_str)]
                    vals[res] = comp

                elif op == "OpCompositeConstruct":
                    vals[res] = tuple(v(x) for x in a[1:])

                elif op == "OpIAdd":
                    vals[res] = (v(a[1]) + v(a[2])) & MASK

                elif op == "OpISub":
                    vals[res] = (v(a[1]) - v(a[2])) & MASK

                elif op == "OpBitwiseAnd":
                    vals[res] = (v(a[1]) & v(a[2])) & MASK

                elif op == "OpBitwiseOr":
                    vals[res] = (v(a[1]) | v(a[2])) & MASK

                elif op == "OpBitwiseXor":
                    vals[res] = (v(a[1]) ^ v(a[2])) & MASK

                elif op == "OpNot":
                    vals[res] = (~v(a[1])) & MASK

                elif op == "OpShiftLeftLogical":
                    sh = v(a[2])
                    if sh >= 32:
                        raise UnsupportedConstruct("shift amount >= 32")
                    vals[res] = (v(a[1]) << sh) & MASK

                elif op == "OpShiftRightLogical":
                    sh = v(a[2])
                    if sh >= 32:
                        raise UnsupportedConstruct("shift amount >= 32")
                    vals[res] = (v(a[1]) >> sh) & MASK

                elif op == "OpShiftRightArithmetic":
                    sh = v(a[2])
                    if sh >= 32:
                        raise UnsupportedConstruct("shift amount >= 32")
                    s = _s32(v(a[1]))
                    vals[res] = (s >> sh) & MASK

                elif op in ("OpBitFieldUExtract", "OpBitFieldSExtract"):
                    base, off, cnt = v(a[1]), v(a[2]), v(a[3])
                    if off + cnt > 32:
                        raise UnsupportedConstruct("bitfield extract beyond 32 bits")
                    field = (base >> off) & ((1 << cnt) - 1)
                    if op == "OpBitFieldSExtract" and cnt and (field >> (cnt - 1)):
                        field -= (1 << cnt)
                    vals[res] = field & MASK

                elif op == "OpBitFieldInsert":
                    base, ins, off, cnt = v(a[1]), v(a[2]), v(a[3]), v(a[4])
                    if off + cnt > 32:
                        raise UnsupportedConstruct("bitfield insert beyond 32 bits")
                    mk = ((1 << cnt) - 1) << off
                    vals[res] = ((base & ~mk) | ((ins << off) & mk)) & MASK

                elif op == "OpIEqual":
                    vals[res] = (v(a[1]) == v(a[2]))

                elif op == "OpINotEqual":
                    vals[res] = (v(a[1]) != v(a[2]))

                elif op == "OpULessThan":
                    vals[res] = (v(a[1]) < v(a[2]))

                elif op == "OpSLessThan":
                    vals[res] = (_s32(v(a[1])) < _s32(v(a[2])))

                elif op == "OpULessThanEqual":
                    vals[res] = (v(a[1]) <= v(a[2]))

                elif op == "OpSLessThanEqual":
                    vals[res] = (_s32(v(a[1])) <= _s32(v(a[2])))

                elif op == "OpUGreaterThan":
                    vals[res] = (v(a[1]) > v(a[2]))

                elif op == "OpSGreaterThan":
                    vals[res] = (_s32(v(a[1])) > _s32(v(a[2])))

                elif op == "OpUGreaterThanEqual":
                    vals[res] = (v(a[1]) >= v(a[2]))

                elif op == "OpSGreaterThanEqual":
                    vals[res] = (_s32(v(a[1])) >= _s32(v(a[2])))

                elif op == "OpLogicalEqual":
                    vals[res] = (bool(v(a[1])) == bool(v(a[2])))

                elif op == "OpLogicalNotEqual":
                    vals[res] = (bool(v(a[1])) != bool(v(a[2])))

                elif op == "OpLogicalAnd":
                    vals[res] = (bool(v(a[1])) and bool(v(a[2])))

                elif op == "OpLogicalOr":
                    vals[res] = (bool(v(a[1])) or bool(v(a[2])))

                elif op == "OpLogicalNot":
                    vals[res] = (not bool(v(a[1])))

                elif op == "OpExtInst":
                    if len(a) < 3 or a[1] != "%glsl":
                        raise UnsupportedConstruct("extended instruction set")
                    name = a[2]
                    args = [v(x) for x in a[3:]]
                    if name == "UMin":
                        vals[res] = min(args)
                    elif name == "UMax":
                        vals[res] = max(args)
                    elif name == "SMin":
                        vals[res] = min(args, key=_s32)
                    elif name == "SMax":
                        vals[res] = max(args, key=_s32)
                    elif name == "FindUMsb":
                        val = args[0] & MASK
                        vals[res] = val.bit_length() - 1 if val != 0 else 0xFFFFFFFF
                    else:
                        raise UnsupportedConstruct(f"GLSL.std.450 {name}")
                else:
                    raise UnsupportedConstruct(f"unsupported body opcode {op}")

        # Collect outputs
        result = {}
        for out_res, _ in self.output_vars:
            per_lane_words = []
            for i in self.active_lanes:
                raw_val = lane_var_box[i][out_res][0]
                flat = raw_val if isinstance(raw_val, tuple) else (raw_val,)
                per_lane_words.append(tuple(f32_bits(x) if isinstance(x, float) else x & MASK for x in flat))
            result[out_res] = per_lane_words
        return result

    def _step_subgroup_op(self, res, op, a, lane_values):
        scope_arg = a[1]
        scope_val = self.constants.get(scope_arg)
        if scope_val != 3:
            raise UnsupportedConstruct(f"subgroup operation requires Execution Scope 3 (Subgroup), got {scope_val}")

        if op == "OpGroupNonUniformBallot":
            # a = [%v4uint, %scope, %pred]
            pred_id = a[2]
            bits = 0
            for i in self.active_lanes:
                pred_val = lane_values[i].get(pred_id)
                if not isinstance(pred_val, bool):
                    raise UnsupportedConstruct(f"OpGroupNonUniformBallot predicate must be bool, got {type(pred_val)}")
                if pred_val:
                    bits |= (1 << i)
            # Result is uvec4 (4 x 32-bit uint)
            bal_vec = (
                bits & MASK,
                (bits >> 32) & MASK,
                (bits >> 64) & MASK,
                (bits >> 96) & MASK,
            )
            for i in self.active_lanes:
                lane_values[i][res] = bal_vec

        elif op == "OpGroupNonUniformAll":
            # a = [%bool, %scope, %pred]
            pred_id = a[2]
            all_true = True
            for i in self.active_lanes:
                pred_val = lane_values[i].get(pred_id)
                if not isinstance(pred_val, bool):
                    raise UnsupportedConstruct(f"OpGroupNonUniformAll predicate must be bool, got {type(pred_val)}")
                if not pred_val:
                    all_true = False
                    break
            for i in self.active_lanes:
                lane_values[i][res] = all_true

        elif op == "OpGroupNonUniformAny":
            # a = [%bool, %scope, %pred]
            pred_id = a[2]
            any_true = False
            for i in self.active_lanes:
                pred_val = lane_values[i].get(pred_id)
                if not isinstance(pred_val, bool):
                    raise UnsupportedConstruct(f"OpGroupNonUniformAny predicate must be bool, got {type(pred_val)}")
                if pred_val:
                    any_true = True
                    break
            for i in self.active_lanes:
                lane_values[i][res] = any_true

        elif op == "OpGroupNonUniformAllEqual":
            # a = [%bool, %scope, %val]
            val_id = a[2]
            if not self.active_lanes:
                is_equal = True
            else:
                first_val = lane_values[self.active_lanes[0]].get(val_id)
                is_equal = all(lane_values[i].get(val_id) == first_val for i in self.active_lanes[1:])
            for i in self.active_lanes:
                lane_values[i][res] = is_equal

        elif op == "OpGroupNonUniformBallotBitCount":
            # a = [%uint, %scope, %group_op, %ballot]
            group_op = a[2]
            bal_id = a[3]
            for i in self.active_lanes:
                bal_val = lane_values[i].get(bal_id)
                if not isinstance(bal_val, (tuple, list)) or len(bal_val) < 4:
                    raise UnsupportedConstruct("OpGroupNonUniformBallotBitCount ballot operand must be uvec4")
                b_int = (bal_val[0] & MASK) | ((bal_val[1] & MASK) << 32) | \
                        ((bal_val[2] & MASK) << 64) | ((bal_val[3] & MASK) << 96)
                # Count bits
                if group_op == "Reduce":
                    count = bin(b_int).count("1")
                elif group_op == "InclusiveScan":
                    mask_i = (1 << (i + 1)) - 1
                    count = bin(b_int & mask_i).count("1")
                elif group_op == "ExclusiveScan":
                    mask_i = (1 << i) - 1
                    count = bin(b_int & mask_i).count("1")
                else:
                    raise UnsupportedConstruct(f"unsupported group operation {group_op}")
                lane_values[i][res] = count & MASK

        elif op == "OpGroupNonUniformBallotFindLSB":
            # a = [%uint, %scope, %ballot]
            bal_id = a[2]
            for i in self.active_lanes:
                bal_val = lane_values[i].get(bal_id)
                if not isinstance(bal_val, (tuple, list)) or len(bal_val) < 4:
                    raise UnsupportedConstruct("OpGroupNonUniformBallotFindLSB ballot operand must be uvec4")
                b_int = (bal_val[0] & MASK) | ((bal_val[1] & MASK) << 32) | \
                        ((bal_val[2] & MASK) << 64) | ((bal_val[3] & MASK) << 96)
                if b_int == 0:
                    lane_values[i][res] = 0xFFFFFFFF
                else:
                    lsb = (b_int & -b_int).bit_length() - 1
                    lane_values[i][res] = lsb

        elif op == "OpGroupNonUniformBallotFindMSB":
            # a = [%uint, %scope, %ballot]
            bal_id = a[2]
            for i in self.active_lanes:
                bal_val = lane_values[i].get(bal_id)
                if not isinstance(bal_val, (tuple, list)) or len(bal_val) < 4:
                    raise UnsupportedConstruct("OpGroupNonUniformBallotFindMSB ballot operand must be uvec4")
                b_int = (bal_val[0] & MASK) | ((bal_val[1] & MASK) << 32) | \
                        ((bal_val[2] & MASK) << 64) | ((bal_val[3] & MASK) << 96)
                if b_int == 0:
                    lane_values[i][res] = 0xFFFFFFFF
                else:
                    msb = b_int.bit_length() - 1
                    lane_values[i][res] = msb

        elif op == "OpGroupNonUniformBallotBitExtract":
            # a = [%bool, %scope, %ballot, %index]
            bal_id = a[2]
            idx_id = a[3]
            for i in self.active_lanes:
                bal_val = lane_values[i].get(bal_id)
                idx = lane_values[i].get(idx_id)
                if not isinstance(bal_val, (tuple, list)) or len(bal_val) < 4:
                    raise UnsupportedConstruct("ballot operand must be uvec4")
                b_int = (bal_val[0] & MASK) | ((bal_val[1] & MASK) << 32) | \
                        ((bal_val[2] & MASK) << 64) | ((bal_val[3] & MASK) << 96)
                lane_values[i][res] = bool((b_int >> idx) & 1) if 0 <= idx < 128 else False

        elif op in ("OpGroupNonUniformBitwiseOr", "OpGroupNonUniformBitwiseAnd", "OpGroupNonUniformBitwiseXor"):
            # a = [%uint, %scope, %group_op, %val]
            group_op = a[2]
            val_id = a[3]
            if group_op != "Reduce":
                raise UnsupportedConstruct(f"unsupported bitwise group operation {group_op}")
            folded = 0 if op != "OpGroupNonUniformBitwiseAnd" else MASK
            for idx, i in enumerate(self.active_lanes):
                val = lane_values[i].get(val_id)
                if op == "OpGroupNonUniformBitwiseOr":
                    folded |= val
                elif op == "OpGroupNonUniformBitwiseAnd":
                    folded &= val
                elif op == "OpGroupNonUniformBitwiseXor":
                    folded = val if idx == 0 else (folded ^ val)
            for i in self.active_lanes:
                lane_values[i][res] = folded & MASK

        elif op == "OpGroupNonUniformShuffle":
            # a = [%type, %scope, %val, %id]
            val_id = a[2]
            src_id = a[3]
            results = {}
            for i in self.active_lanes:
                src_lane = lane_values[i].get(src_id)
                if src_lane in lane_values:
                    results[i] = lane_values[src_lane].get(val_id)
                else:
                    results[i] = 0
            for i in self.active_lanes:
                lane_values[i][res] = results[i]

        else:
            raise UnsupportedConstruct(f"unsupported subgroup opcode {op}")


MODULE_HEADER = """\
OpCapability Shader
OpCapability GroupNonUniform
OpCapability GroupNonUniformVote
OpCapability GroupNonUniformBallot
OpCapability GroupNonUniformArithmetic
OpCapability GroupNonUniformShuffle
%glsl = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %out %sg_lane %sg_size
OpDecorate %sg_lane BuiltIn SubgroupLocalInvocationId
OpDecorate %sg_size BuiltIn SubgroupSize
%void = OpTypeVoid
%fn = OpTypeFunction %void
%uint = OpTypeInt 32 0
%float = OpTypeFloat 32
%bool = OpTypeBool
%v4uint = OpTypeVector %uint 4
%_ptr_Output_uint = OpTypePointer Output %uint
%_ptr_Output_v4uint = OpTypePointer Output %v4uint
%_ptr_Input_uint = OpTypePointer Input %uint
%out = OpVariable %_ptr_Output_uint Output
%sg_lane = OpVariable %_ptr_Input_uint Input
%sg_size = OpVariable %_ptr_Input_uint Input
%c0 = OpConstant %uint 0
%c1 = OpConstant %uint 1
%c2 = OpConstant %uint 2
%c3 = OpConstant %uint 3
%c4 = OpConstant %uint 4
%c8 = OpConstant %uint 8
%c15 = OpConstant %uint 15
%c16 = OpConstant %uint 16
%c31 = OpConstant %uint 31
%c32 = OpConstant %uint 32
%c47 = OpConstant %uint 47
%c63 = OpConstant %uint 63
%c64 = OpConstant %uint 64
%cFF = OpConstant %uint 4294967295
%c55 = OpConstant %uint 1431655765
%cAA = OpConstant %uint 2863311530
%t = OpConstantTrue %bool
%f = OpConstantFalse %bool
"""

MODULE_FOOTER = "OpReturn\nOpFunctionEnd\n"


def run_subgroup_words(body, subgroup_size=32, active_mask=None, decls=""):
    """Run `body` across all active invocations; returns list of %out uint words per lane."""
    text = (MODULE_HEADER + decls +
            "%main = OpFunction %void None %fn\n%entry = OpLabel\n" +
            body + MODULE_FOOTER)
    interp = SubgroupInterpreter(text, subgroup_size=subgroup_size, active_mask=active_mask)
    return [w[0] for w in interp.run()["%out"]]


def run_subgroup_uniform_word(body, subgroup_size=32, active_mask=None, decls=""):
    """Run `body`, asserting every active invocation writes the exact same uint word, and return it."""
    words = run_subgroup_words(body, subgroup_size=subgroup_size, active_mask=active_mask, decls=decls)
    if not words:
        raise AssertionError("no active invocations produced words")
    first = words[0]
    for idx, w in enumerate(words[1:], 1):
        if w != first:
            raise AssertionError(f"divergent output across subgroup: lane 0 got {first:#x}, lane {idx} got {w:#x}")
    return first


def run_subgroup_uvec4(body, subgroup_size=32, active_mask=None, decls=""):
    """Run `body` writing a %v4uint to %out_vec; return list of 4-tuples per active lane."""
    header = MODULE_HEADER.replace("%out = OpVariable %_ptr_Output_uint Output\n",
                                   "%out_vec = OpVariable %_ptr_Output_v4uint Output\n")
    text = (header + decls +
            "%main = OpFunction %void None %fn\n%entry = OpLabel\n" +
            body + MODULE_FOOTER)
    interp = SubgroupInterpreter(text, subgroup_size=subgroup_size, active_mask=active_mask)
    return interp.run()["%out_vec"]


class SubgroupBallotGenerationTests(unittest.TestCase):
    """Whole-wave subgroup ballot generation via OpGroupNonUniformBallot."""

    def test_ballot_all_active_true_wave32(self):
        body = """\
%bal = OpGroupNonUniformBallot %v4uint %c3 %t
OpStore %out_vec %bal
"""
        outputs = run_subgroup_uvec4(body, subgroup_size=32)
        expected = (0xFFFFFFFF, 0, 0, 0)
        for i, val in enumerate(outputs):
            self.assertEqual(val, expected, f"lane {i} expected full wave32 ballot")

    def test_ballot_all_active_true_wave64(self):
        body = """\
%bal = OpGroupNonUniformBallot %v4uint %c3 %t
OpStore %out_vec %bal
"""
        outputs = run_subgroup_uvec4(body, subgroup_size=64)
        expected = (0xFFFFFFFF, 0xFFFFFFFF, 0, 0)
        for i, val in enumerate(outputs):
            self.assertEqual(val, expected, f"lane {i} expected full wave64 ballot")

    def test_ballot_all_active_false(self):
        body = """\
%bal = OpGroupNonUniformBallot %v4uint %c3 %f
OpStore %out_vec %bal
"""
        outputs = run_subgroup_uvec4(body, subgroup_size=32)
        expected = (0, 0, 0, 0)
        for i, val in enumerate(outputs):
            self.assertEqual(val, expected, f"lane {i} expected all-zero ballot")

    def test_ballot_single_lane_predicates(self):
        for target_lane in (0, 1, 15, 31, 32, 47, 63):
            with self.subTest(target_lane=target_lane):
                body = f"""\
%lane = OpLoad %uint %sg_lane
%c_target = OpCopyObject %uint %c{target_lane}
%pred = OpIEqual %bool %lane %c_target
%bal = OpGroupNonUniformBallot %v4uint %c3 %pred
OpStore %out_vec %bal
"""
                outputs = run_subgroup_uvec4(body, subgroup_size=64)
                w0 = (1 << target_lane) if target_lane < 32 else 0
                w1 = (1 << (target_lane - 32)) if target_lane >= 32 else 0
                expected = (w0, w1, 0, 0)
                for i, val in enumerate(outputs):
                    self.assertEqual(val, expected, f"lane {i} mismatch for target {target_lane}")

    def test_ballot_checkerboard_even_odd(self):
        body_even = """\
%lane = OpLoad %uint %sg_lane
%bit0 = OpBitwiseAnd %uint %lane %c1
%pred = OpIEqual %bool %bit0 %c0
%bal = OpGroupNonUniformBallot %v4uint %c3 %pred
OpStore %out_vec %bal
"""
        outputs_even = run_subgroup_uvec4(body_even, subgroup_size=64)
        expected_even = (0x55555555, 0x55555555, 0, 0)
        for val in outputs_even:
            self.assertEqual(val, expected_even)

        body_odd = """\
%lane = OpLoad %uint %sg_lane
%bit0 = OpBitwiseAnd %uint %lane %c1
%pred = OpIEqual %bool %bit0 %c1
%bal = OpGroupNonUniformBallot %v4uint %c3 %pred
OpStore %out_vec %bal
"""
        outputs_odd = run_subgroup_uvec4(body_odd, subgroup_size=64)
        expected_odd = (0xAAAAAAAA, 0xAAAAAAAA, 0, 0)
        for val in outputs_odd:
            self.assertEqual(val, expected_odd)

    def test_ballot_inactive_lanes_under_exec_mask(self):
        # Even if predicate evaluates to true, inactive lanes must not set their ballot bit.
        active_mask = 0x00000000000000FF  # only lanes 0..7 active
        body = """\
%bal = OpGroupNonUniformBallot %v4uint %c3 %t
OpStore %out_vec %bal
"""
        outputs = run_subgroup_uvec4(body, subgroup_size=64, active_mask=active_mask)
        expected = (0x000000FF, 0, 0, 0)
        self.assertEqual(len(outputs), 8)
        for val in outputs:
            self.assertEqual(val, expected)


class SubgroupBallotEqualityTests(unittest.TestCase):
    """Subgroup ballot equality and whole-wave mask comparisons."""

    def test_all_equal_uniform_scalars(self):
        body = """\
%res_bool = OpGroupNonUniformAllEqual %bool %c3 %c42
%res = OpSelect %uint %res_bool %c1 %c0
OpStore %out %res
"""
        word = run_subgroup_uniform_word(body, subgroup_size=64, decls="%c42 = OpConstant %uint 42\n")
        self.assertEqual(word, 1)

    def test_all_equal_divergent_single_lane(self):
        for diff_lane in (0, 15, 31, 32, 63):
            with self.subTest(diff_lane=diff_lane):
                body = f"""\
%lane = OpLoad %uint %sg_lane
%is_diff = OpIEqual %bool %lane %c{diff_lane}
%val = OpSelect %uint %is_diff %c2 %c1
%res_bool = OpGroupNonUniformAllEqual %bool %c3 %val
%res = OpSelect %uint %res_bool %c1 %c0
OpStore %out %res
"""
                word = run_subgroup_uniform_word(body, subgroup_size=64)
                self.assertEqual(word, 0, f"differing lane {diff_lane} must break AllEqual")

    def test_all_equal_uniform_ballot_vectors(self):
        # OpGroupNonUniformBallot returns the exact same uvec4 to every invocation.
        # OpGroupNonUniformAllEqual on that ballot vector must return True.
        body = """\
%lane = OpLoad %uint %sg_lane
%pred = OpULessThan %bool %lane %c16
%bal = OpGroupNonUniformBallot %v4uint %c3 %pred
%res_bool = OpGroupNonUniformAllEqual %bool %c3 %bal
%res = OpSelect %uint %res_bool %c1 %c0
OpStore %out %res
"""
        word = run_subgroup_uniform_word(body, subgroup_size=64)
        self.assertEqual(word, 1)

    def test_all_equal_with_inactive_lanes(self):
        # Lane 10 holds a differing value, but lane 10 is inactive in EXEC mask.
        active_mask = (1 << 64) - 1 - (1 << 10)
        body = """\
%lane = OpLoad %uint %sg_lane
%is_10 = OpIEqual %bool %lane %c10
%val = OpSelect %uint %is_10 %c2 %c1
%res_bool = OpGroupNonUniformAllEqual %bool %c3 %val
%res = OpSelect %uint %res_bool %c1 %c0
OpStore %out %res
"""
        word = run_subgroup_uniform_word(body, subgroup_size=64, active_mask=active_mask,
                                         decls="%c10 = OpConstant %uint 10\n")
        self.assertEqual(word, 1, "inactive lanes must not participate in AllEqual")

    def test_ballot_componentwise_equality_matching(self):
        # Compare two independently generated ballots representing identical wave predicates.
        body = """\
%lane = OpLoad %uint %sg_lane
%p1 = OpULessThan %bool %lane %c32
%p2 = OpULessThanEqual %bool %lane %c31
%b1 = OpGroupNonUniformBallot %v4uint %c3 %p1
%b2 = OpGroupNonUniformBallot %v4uint %c3 %p2
%w0_1 = OpCompositeExtract %uint %b1 0
%w0_2 = OpCompositeExtract %uint %b2 0
%w1_1 = OpCompositeExtract %uint %b1 1
%w1_2 = OpCompositeExtract %uint %b2 1
%eq0 = OpIEqual %bool %w0_1 %w0_2
%eq1 = OpIEqual %bool %w1_1 %w1_2
%eq = OpLogicalAnd %bool %eq0 %eq1
%res = OpSelect %uint %eq %c1 %c0
OpStore %out %res
"""
        word = run_subgroup_uniform_word(body, subgroup_size=64)
        self.assertEqual(word, 1)

    def test_ballot_componentwise_equality_mismatch(self):
        # Two ballots differing by lane 32 in the upper half of wave64.
        body = """\
%lane = OpLoad %uint %sg_lane
%p1 = OpULessThan %bool %lane %c32
%p2 = OpULessThan %bool %lane %c33
%b1 = OpGroupNonUniformBallot %v4uint %c3 %p1
%b2 = OpGroupNonUniformBallot %v4uint %c3 %p2
%w0_1 = OpCompositeExtract %uint %b1 0
%w0_2 = OpCompositeExtract %uint %b2 0
%w1_1 = OpCompositeExtract %uint %b1 1
%w1_2 = OpCompositeExtract %uint %b2 1
%eq0 = OpIEqual %bool %w0_1 %w0_2
%eq1 = OpIEqual %bool %w1_1 %w1_2
%eq = OpLogicalAnd %bool %eq0 %eq1
%res = OpSelect %uint %eq %c1 %c0
OpStore %out %res
"""
        word = run_subgroup_uniform_word(body, subgroup_size=64, decls="%c33 = OpConstant %uint 33\n")
        self.assertEqual(word, 0)

    def test_wave_equal_translator_pattern_all_match(self):
        # Models rdna2spirv.py wave_equal(mine, equal=True):
        # s_cmp_eq of lane masks requires every lane to agree -> OpGroupNonUniformAll.
        body = """\
%lane = OpLoad %uint %sg_lane
%x = OpBitwiseAnd %uint %lane %c1
%y = OpBitwiseAnd %uint %lane %c1
%mine = OpIEqual %bool %x %y
%wave_eq = OpGroupNonUniformAll %bool %c3 %mine
%res = OpSelect %uint %wave_eq %c1 %c0
OpStore %out %res
"""
        word = run_subgroup_uniform_word(body, subgroup_size=64)
        self.assertEqual(word, 1)

    def test_wave_equal_translator_pattern_single_lane_mismatch(self):
        # If even one lane differs, wave_equal with equal=True must return 0.
        body = """\
%lane = OpLoad %uint %sg_lane
%x = OpBitwiseAnd %uint %lane %c1
%is_63 = OpIEqual %bool %lane %c63
%y = OpSelect %uint %is_63 %c2 %x
%mine = OpIEqual %bool %x %y
%wave_eq = OpGroupNonUniformAll %bool %c3 %mine
%res = OpSelect %uint %wave_eq %c1 %c0
OpStore %out %res
"""
        word = run_subgroup_uniform_word(body, subgroup_size=64)
        self.assertEqual(word, 0)

    def test_wave_equal_translator_pattern_inequality_vote(self):
        # Models rdna2spirv.py wave_equal(mine, equal=False) for s_cmp_lg:
        # returns True if any lane differs -> OpGroupNonUniformAny.
        body_no_diff = """\
%lane = OpLoad %uint %sg_lane
%x = OpCopyObject %uint %lane
%y = OpCopyObject %uint %lane
%mine = OpINotEqual %bool %x %y
%wave_ne = OpGroupNonUniformAny %bool %c3 %mine
%res = OpSelect %uint %wave_ne %c1 %c0
OpStore %out %res
"""
        self.assertEqual(run_subgroup_uniform_word(body_no_diff, subgroup_size=64), 0)

        body_one_diff = """\
%lane = OpLoad %uint %sg_lane
%x = OpCopyObject %uint %lane
%is_0 = OpIEqual %bool %lane %c0
%y = OpSelect %uint %is_0 %c1 %lane
%mine = OpINotEqual %bool %x %y
%wave_ne = OpGroupNonUniformAny %bool %c3 %mine
%res = OpSelect %uint %wave_ne %c1 %c0
OpStore %out %res
"""
        self.assertEqual(run_subgroup_uniform_word(body_one_diff, subgroup_size=64), 1)


class SubgroupBallotComparisonTests(unittest.TestCase):
    """Subgroup ballot comparison, voting, bit counting, and bit extraction."""

    def test_vote_all(self):
        body_unanimous = """\
%res_bool = OpGroupNonUniformAll %bool %c3 %t
%res = OpSelect %uint %res_bool %c1 %c0
OpStore %out %res
"""
        self.assertEqual(run_subgroup_uniform_word(body_unanimous, subgroup_size=32), 1)

        body_not_unanimous = """\
%lane = OpLoad %uint %sg_lane
%pred = OpULessThan %bool %lane %c31
%res_bool = OpGroupNonUniformAll %bool %c3 %pred
%res = OpSelect %uint %res_bool %c1 %c0
OpStore %out %res
"""
        self.assertEqual(run_subgroup_uniform_word(body_not_unanimous, subgroup_size=32), 0)

    def test_vote_any(self):
        body_none = """\
%res_bool = OpGroupNonUniformAny %bool %c3 %f
%res = OpSelect %uint %res_bool %c1 %c0
OpStore %out %res
"""
        self.assertEqual(run_subgroup_uniform_word(body_none, subgroup_size=64), 0)

        body_one = """\
%lane = OpLoad %uint %sg_lane
%pred = OpIEqual %bool %lane %c47
%res_bool = OpGroupNonUniformAny %bool %c3 %pred
%res = OpSelect %uint %res_bool %c1 %c0
OpStore %out %res
"""
        self.assertEqual(run_subgroup_uniform_word(body_one, subgroup_size=64), 1)

    def test_scc_from_mask_semantics(self):
        # Models rdna2spirv.py scc_from_mask: OpGroupNonUniformAny on tmp_bool_nz.
        # SCC is 1 if wave mask is non-zero, 0 if all zeros.
        body_zero = """\
%nz = OpINotEqual %bool %c0 %c0
%any = OpGroupNonUniformAny %bool %c3 %nz
%scc = OpSelect %uint %any %c1 %c0
OpStore %out %scc
"""
        self.assertEqual(run_subgroup_uniform_word(body_zero, subgroup_size=64), 0)

        body_nonzero = """\
%lane = OpLoad %uint %sg_lane
%is_32 = OpIEqual %bool %lane %c32
%nz = OpSelect %uint %is_32 %c1 %c0
%b_nz = OpINotEqual %bool %nz %c0
%any = OpGroupNonUniformAny %bool %c3 %b_nz
%scc = OpSelect %uint %any %c1 %c0
OpStore %out %scc
"""
        self.assertEqual(run_subgroup_uniform_word(body_nonzero, subgroup_size=64), 1)

    def test_ballot_bit_count_reduce(self):
        for size, count_expected in ((32, 32), (64, 64)):
            with self.subTest(size=size):
                body = """\
%bal = OpGroupNonUniformBallot %v4uint %c3 %t
%cnt = OpGroupNonUniformBallotBitCount %uint %c3 Reduce %bal
OpStore %out %cnt
"""
                self.assertEqual(run_subgroup_uniform_word(body, subgroup_size=size), count_expected)

    def test_ballot_bit_count_checkerboard(self):
        # Checkerboard has 16 bits set in wave32, 32 bits set in wave64.
        body = """\
%lane = OpLoad %uint %sg_lane
%bit0 = OpBitwiseAnd %uint %lane %c1
%pred = OpIEqual %bool %bit0 %c0
%bal = OpGroupNonUniformBallot %v4uint %c3 %pred
%cnt = OpGroupNonUniformBallotBitCount %uint %c3 Reduce %bal
OpStore %out %cnt
"""
        self.assertEqual(run_subgroup_uniform_word(body, subgroup_size=32), 16)
        self.assertEqual(run_subgroup_uniform_word(body, subgroup_size=64), 32)

    def test_ballot_bit_count_scans(self):
        # InclusiveScan at lane i gives number of active set bits up to lane i.
        # ExclusiveScan gives number of active set bits strictly before lane i.
        body_inclusive = """\
%bal = OpGroupNonUniformBallot %v4uint %c3 %t
%cnt = OpGroupNonUniformBallotBitCount %uint %c3 InclusiveScan %bal
OpStore %out %cnt
"""
        words = run_subgroup_words(body_inclusive, subgroup_size=32)
        for i, cnt in enumerate(words):
            self.assertEqual(cnt, i + 1, f"lane {i} inclusive scan mismatch")

        body_exclusive = """\
%bal = OpGroupNonUniformBallot %v4uint %c3 %t
%cnt = OpGroupNonUniformBallotBitCount %uint %c3 ExclusiveScan %bal
OpStore %out %cnt
"""
        words_ex = run_subgroup_words(body_exclusive, subgroup_size=32)
        for i, cnt in enumerate(words_ex):
            self.assertEqual(cnt, i, f"lane {i} exclusive scan mismatch")

    def test_ballot_find_lsb_and_msb(self):
        for lanes_set in ((0,), (15,), (31,), (32,), (63,), (5, 20, 45)):
            with self.subTest(lanes_set=lanes_set):
                # Construct condition: lane in lanes_set
                cond_lines = ["%lane = OpLoad %uint %sg_lane", "%p_acc = OpCopyObject %bool %f"]
                for idx, ln in enumerate(lanes_set):
                    cond_lines.append(f"%m_{idx} = OpIEqual %bool %lane %c{ln}")
                    cond_lines.append(f"%p_{idx} = OpLogicalOr %bool %p_acc %m_{idx}")
                    cond_lines.append(f"%p_acc = OpCopyObject %bool %p_{idx}")
                cond_str = "\n".join(cond_lines)

                body_lsb = f"""\
{cond_str}
%bal = OpGroupNonUniformBallot %v4uint %c3 %p_acc
%lsb = OpGroupNonUniformBallotFindLSB %uint %c3 %bal
OpStore %out %lsb
"""
                expected_lsb = min(lanes_set)
                self.assertEqual(run_subgroup_uniform_word(body_lsb, subgroup_size=64,
                                                           decls="%c5 = OpConstant %uint 5\n%c20 = OpConstant %uint 20\n%c45 = OpConstant %uint 45\n"),
                                 expected_lsb)

                body_msb = f"""\
{cond_str}
%bal = OpGroupNonUniformBallot %v4uint %c3 %p_acc
%msb = OpGroupNonUniformBallotFindMSB %uint %c3 %bal
OpStore %out %msb
"""
                expected_msb = max(lanes_set)
                self.assertEqual(run_subgroup_uniform_word(body_msb, subgroup_size=64,
                                                           decls="%c5 = OpConstant %uint 5\n%c20 = OpConstant %uint 20\n%c45 = OpConstant %uint 45\n"),
                                 expected_msb)

    def test_ballot_bit_extract(self):
        # Each lane extracts its own bit from the ballot, verifying it matches the predicate.
        body = """\
%lane = OpLoad %uint %sg_lane
%bit0 = OpBitwiseAnd %uint %lane %c1
%pred = OpIEqual %bool %bit0 %c1
%bal = OpGroupNonUniformBallot %v4uint %c3 %pred
%extracted = OpGroupNonUniformBallotBitExtract %bool %c3 %bal %lane
%res = OpSelect %uint %extracted %c1 %c0
OpStore %out %res
"""
        words = run_subgroup_words(body, subgroup_size=64)
        for i, w in enumerate(words):
            self.assertEqual(w, i & 1, f"lane {i} bit extract mismatch")

    def test_ballot_bitwise_reductions(self):
        # Test OpGroupNonUniformBitwiseOr across lanes (each lane brings (1 << (lane % 32))).
        body_or = """\
%lane = OpLoad %uint %sg_lane
%sh = OpBitwiseAnd %uint %lane %c31
%val = OpShiftLeftLogical %uint %c1 %sh
%or_fold = OpGroupNonUniformBitwiseOr %uint %c3 Reduce %val
OpStore %out %or_fold
"""
        self.assertEqual(run_subgroup_uniform_word(body_or, subgroup_size=32), 0xFFFFFFFF)

        # Test OpGroupNonUniformBitwiseAnd across lanes (each lane brings 0xFFFFFFF0 | (lane & 1)).
        # Bit 0 will be 0 because lane 0 brings 0. Bits 4..31 will be 1.
        body_and = """\
%lane = OpLoad %uint %sg_lane
%b0 = OpBitwiseAnd %uint %lane %c1
%val = OpBitwiseOr %uint %cF0 %b0
%and_fold = OpGroupNonUniformBitwiseAnd %uint %c3 Reduce %val
OpStore %out %and_fold
"""
        self.assertEqual(run_subgroup_uniform_word(body_and, subgroup_size=32, decls="%cF0 = OpConstant %uint 4294967280\n"),
                         0xFFFFFFF0)


class SubgroupBallotPairReconstructionTests(unittest.TestCase):
    """Reconstructing whole-wave 64-bit masks via ballot_pair pattern in tools/rdna2spirv.py."""

    def test_reconstruct_wave32_mask(self):
        # rdna2spirv.py:
        # bal = OpGroupNonUniformBallot %v4uint_bal %c3 pred
        # lo = OpCompositeExtract %uint bal 0
        body = """\
%lane = OpLoad %uint %sg_lane
%pred = OpULessThan %bool %lane %c15
%bal = OpGroupNonUniformBallot %v4uint %c3 %pred
%lo = OpCompositeExtract %uint %bal 0
OpStore %out %lo
"""
        self.assertEqual(run_subgroup_uniform_word(body, subgroup_size=32), (1 << 15) - 1)

    def test_reconstruct_wave64_mask(self):
        # lo = bal[0], hi = bal[1]. Check both halves match the expected 64-bit pattern.
        # Lanes 0..15 on, lanes 32..47 on.
        body = """\
%lane = OpLoad %uint %sg_lane
%b_lo = OpULessThan %bool %lane %c16
%gte_32 = OpUGreaterThanEqual %bool %lane %c32
%lt_48 = OpULessThan %bool %lane %c48
%b_hi = OpLogicalAnd %bool %gte_32 %lt_48
%pred = OpLogicalOr %bool %b_lo %b_hi
%bal = OpGroupNonUniformBallot %v4uint %c3 %pred
%lo = OpCompositeExtract %uint %bal 0
%hi = OpCompositeExtract %uint %bal 1
OpStore %out_vec %bal
"""
        outputs = run_subgroup_uvec4(body, subgroup_size=64, decls="%c48 = OpConstant %uint 48\n")
        expected = (0x0000FFFF, 0x0000FFFF, 0, 0)
        for val in outputs:
            self.assertEqual(val, expected)


class SubgroupValidationTests(unittest.TestCase):
    """Strict validation and boundary tests matching spirv_mini_interp design."""

    def test_invalid_execution_scope_rejected(self):
        for bad_scope in (1, 2, 4):  # Device=1, Workgroup=2, Invocation=4
            with self.subTest(bad_scope=bad_scope):
                body = f"""\
%bad_scope = OpConstant %uint {bad_scope}
%bal = OpGroupNonUniformBallot %v4uint %bad_scope %t
OpStore %out %c0
"""
                with self.assertRaises(UnsupportedConstruct):
                    run_subgroup_words(body, subgroup_size=32)

    def test_ballot_non_bool_predicate_rejected(self):
        body = """\
%bal = OpGroupNonUniformBallot %v4uint %c3 %c1
OpStore %out %c0
"""
        with self.assertRaises(UnsupportedConstruct):
            run_subgroup_words(body, subgroup_size=32)

    def test_vote_non_bool_predicate_rejected(self):
        body = """\
%res = OpGroupNonUniformAll %bool %c3 %c1
OpStore %out %c0
"""
        with self.assertRaises(UnsupportedConstruct):
            run_subgroup_words(body, subgroup_size=32)

    def test_empty_ballot_find_lsb_msb_returns_sentinel(self):
        body = """\
%bal = OpGroupNonUniformBallot %v4uint %c3 %f
%lsb = OpGroupNonUniformBallotFindLSB %uint %c3 %bal
OpStore %out %lsb
"""
        self.assertEqual(run_subgroup_uniform_word(body, subgroup_size=32), 0xFFFFFFFF)

        body_msb = """\
%bal = OpGroupNonUniformBallot %v4uint %c3 %f
%msb = OpGroupNonUniformBallotFindMSB %uint %c3 %bal
OpStore %out %msb
"""
        self.assertEqual(run_subgroup_uniform_word(body_msb, subgroup_size=32), 0xFFFFFFFF)

    def test_unsupported_group_operation_rejected(self):
        body = """\
%bal = OpGroupNonUniformBallot %v4uint %c3 %t
%cnt = OpGroupNonUniformBallotBitCount %uint %c3 ClusteredReduce %bal
OpStore %out %cnt
"""
        with self.assertRaises(UnsupportedConstruct):
            run_subgroup_words(body, subgroup_size=32)


if __name__ == "__main__":
    unittest.main()
