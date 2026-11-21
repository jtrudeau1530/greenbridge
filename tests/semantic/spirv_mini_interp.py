"""A deliberately tiny interpreter for the SPIR-V assembly that tools/rdna2spirv.py emits for branch-free pixel
shaders and simple structured compute shaders. It exists so tests can check *computed values*, not just that the
module validates.

Design rules (see tests/semantic/README.md):
  * Written from the SPIR-V and GLSL.std.450 specifications. It shares no code with the translator.
  * Strict: any opcode, extended instruction, type or storage class not listed below raises
    UnsupportedConstruct. Nothing is skipped or approximated silently, so a translator change that starts
    emitting something new makes the test error out instead of passing.
  * One invocation (one lane). Control flow is the forward-only discard test and the forward merge/dynamic-bounds
    branches the translator emits: OpBranch / OpBranchConditional to a later label plus OpPhi at the merge block.
    Backward branches, loops, calls and switches are rejected. Function/Output/Input variables live in Python
    boxes; PushConstant reads can be seeded with a word list; physical-storage-buffer loads/stores use a flat
    address -> word dictionary. Workgroup storage and images are rejected.
  * A default run sees zero push constants, no inputs and empty memory, so tests that do not care about them
    keep the old behaviour.

Values: uint/int are Python ints in [0, 2**32) or [0, 2**64) for 64-bit types (signedness is applied by the
operation, as in SPIR-V); float is a Python float rounded to binary32; vectors are tuples; arrays are lists;
bool is bool.
"""
import math
import re
import struct

MASK = 0xFFFFFFFF


class UnsupportedConstruct(NotImplementedError):
    """The module uses something this interpreter deliberately does not model."""


# Instructions that carry no runtime semantics for a single-entry-point, straight-line module.
_IGNORED = frozenset({
    "OpCapability", "OpExtension", "OpExtInstImport", "OpMemoryModel", "OpEntryPoint", "OpExecutionMode",
    "OpDecorate", "OpMemberDecorate", "OpTypeVoid", "OpTypeFunction", "OpLabel", "OpSelectionMerge",
    "OpReturn", "OpFunctionEnd", "OpName", "OpSource",
})

_LINE = re.compile(r"^(?:(%\S+)\s*=\s*)?(Op\w+)\s*(.*)$")


def f32(x):
    """Round a Python float to binary32 (overflow to infinity)."""
    try:
        return struct.unpack("<f", struct.pack("<f", x))[0]
    except OverflowError:
        return float("inf") if x > 0 else float("-inf")


def f32_bits(x):
    try:
        return struct.unpack("<I", struct.pack("<f", x))[0]
    except OverflowError:
        return 0x7F800000 if x > 0 else 0xFF800000


def bits_f32(b):
    return struct.unpack("<f", struct.pack("<I", b & MASK))[0]


def half_to_float(h):
    return struct.unpack("<e", struct.pack("<H", h & 0xFFFF))[0]


def float_to_half(x):
    try:
        return struct.unpack("<H", struct.pack("<e", x))[0]   # round to nearest even
    except OverflowError:
        return 0x7C00 if x > 0 else 0xFC00


def _s32(x):
    x &= MASK
    return x - (1 << 32) if x >> 31 else x


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

    def size(self, name):
        """Byte size of a concrete type. Runtime arrays have no size of their own; the element size is used."""
        k = self.t[name]
        if k[0] in ("int", "float"):
            return 4
        if k[0] == "vec":
            return 4 * k[2]
        if k[0] == "arr":
            return self.size(k[1]) * k[2]
        if k[0] == "rarr":
            return self.size(k[1])
        if k[0] == "struct":
            return sum(self.size(t) for t in k[1])
        raise UnsupportedConstruct(f"size of type {name} ({k[0]})")

    def bits(self, name):
        k = self.t[name]
        if k[0] != "int":
            raise UnsupportedConstruct(f"integer width of type {name} ({k[0]})")
        return k[1]


def run_fragment(text, inputs=None, push_constants=None, memory=None):
    """Execute the entry point of `text` once and return the words written to every Output variable.

    `inputs` maps an Input variable name (for example "%in_attr1") to its value (a tuple for vectors).
    `push_constants` is a list of 32-bit words at byte offsets 0, 4, 8, ... of the PushConstant block
    (missing words read as zero). `memory` is a mutable {byte_address: word} mapping used for
    PhysicalStorageBuffer loads and stores.

    Returns {output_variable_name: tuple_of_32bit_words}. Float outputs are returned as their bit patterns.
    """
    inputs = inputs or {}
    push_constants = list(push_constants or [])
    memory = {} if memory is None else memory
    types = _Types()
    values = {}                 # id -> value (constants and results)
    constants = {}              # id -> value, kept apart so zero-initialised arrays can be built
    ptr_types = {}              # pointer type id -> (storage class, pointee type id)
    var_class = {}              # variable id -> storage class
    var_box = {}                # variable id -> [value]
    pointers = {}               # id -> (container list, index) / ("pc", offset) / ("psb", address)
    pc_type = {}                # id -> pointee type id of a PushConstant pointer
    in_body = False
    body = []
    outputs = []

    def val(x):
        if x in values:
            return values[x]
        raise UnsupportedConstruct(f"use of undefined id {x}")

    def walk(type_id, indices):
        """Byte offset and element type after selecting `indices` in an aggregate type."""
        off, cur = 0, type_id
        for idx in indices:
            k = types.t[cur]
            if k[0] == "struct":
                for m in range(idx):
                    off += types.size(k[1][m])
                cur = k[1][idx]
            elif k[0] in ("arr", "rarr", "vec"):
                off += types.size(k[1]) * idx
                cur = k[1]
            else:
                raise UnsupportedConstruct(f"OpAccessChain into {k[0]}")
        return off, cur

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
            body.append((res, op, a, line))
            continue
        if op == "OpFunction":
            in_body = True
            continue
        if op in _IGNORED:
            continue
        if op == "OpTypeInt":
            bits = int(a[0])
            if bits not in (32, 64):
                raise UnsupportedConstruct("integer type that is not 32 or 64 bits")
            types.t[res] = ("int", bits, int(a[1]))
        elif op == "OpTypeFloat":
            if int(a[0]) != 32:
                raise UnsupportedConstruct("non-32-bit float type")
            types.t[res] = ("float",)
        elif op == "OpTypeBool":
            types.t[res] = ("bool",)
        elif op == "OpTypeVector":
            types.t[res] = ("vec", a[0], int(a[1]))
        elif op == "OpTypeArray":
            types.t[res] = ("arr", a[0], int(constants[a[1]]))
        elif op == "OpTypeRuntimeArray":
            types.t[res] = ("rarr", a[0])
        elif op == "OpTypeStruct":
            types.t[res] = ("struct", a)
        elif op == "OpTypePointer":
            ptr_types[res] = (a[0], a[1])
        elif op == "OpConstant":
            k = types.t[a[0]]
            if k[0] == "float":
                constants[res] = values[res] = f32(float(a[1]))
            elif k[0] == "int":
                constants[res] = values[res] = int(a[1]) & ((1 << k[1]) - 1)
            else:
                raise UnsupportedConstruct(f"OpConstant of type {k[0]}")
        elif op == "OpConstantTrue":
            constants[res] = values[res] = True
        elif op == "OpConstantFalse":
            constants[res] = values[res] = False
        elif op == "OpConstantNull":
            constants[res] = values[res] = types.zero(a[0])
        elif op == "OpVariable":
            cls, pointee = ptr_types[a[0]]
            if a[1] != cls:
                raise UnsupportedConstruct("OpVariable storage class mismatch")
            var_class[res] = cls
            if cls == "PushConstant":
                var_box[res] = [None]                  # reads come from push_constants, see OpLoad
                pc_type[res] = pointee
            elif cls == "Input":
                init = inputs.get(res, types.zero(pointee))
                var_box[res] = [list(init) if isinstance(init, (list, tuple)) else init]
            elif cls in ("Function", "Output"):
                if len(a) > 2:
                    init = values[a[2]]
                    var_box[res] = [list(init) if isinstance(init, list) else init]
                else:
                    var_box[res] = [types.zero(pointee)]
                if cls == "Output":
                    outputs.append(res)
            else:
                raise UnsupportedConstruct(f"storage class {cls}")
            pointers[res] = ("pc", 0) if cls == "PushConstant" else (var_box[res], 0)
        else:
            raise UnsupportedConstruct(f"{op} in module header")
    if not in_body:
        raise UnsupportedConstruct("no function body")

    def deref(p):
        box, i = p
        return box[i]

    def wmask(type_id):
        return (1 << types.bits(type_id)) - 1

    def signed(v, bits):
        v &= (1 << bits) - 1
        return v - (1 << bits) if v >> (bits - 1) else v

    labels = {res: idx for idx, (res, op, _, _) in enumerate(body) if op == "OpLabel" and res}
    pc = 0
    killed = False
    cur_label = prev_label = None
    while pc < len(body):
        res, op, a, line = body[pc]
        pc += 1
        if op == "OpLabel":
            prev_label, cur_label = cur_label, res
            continue
        if op in _IGNORED:
            continue
        if op == "OpKill":
            killed = True
            break
        if op == "OpBranch":
            if a[0] not in labels or labels[a[0]] < pc:
                raise UnsupportedConstruct("non-forward branch")
            pc = labels[a[0]]
            continue
        if op == "OpBranchConditional":
            cond = val(a[0])
            if not isinstance(cond, bool):
                raise UnsupportedConstruct("OpBranchConditional with a non-bool condition")
            target = a[1] if cond else a[2]
            if target not in labels or labels[target] < pc:
                raise UnsupportedConstruct("non-forward branch")
            pc = labels[target]
            continue
        if op == "OpPhi":
            incoming = [a[i] for i in range(1, len(a) - 1, 2) if a[i + 1] == prev_label]
            if len(incoming) != 1:
                raise UnsupportedConstruct("OpPhi with an unknown predecessor")
            values[res] = val(incoming[0])
            continue
        if op == "OpVariable":
            cls, pointee = ptr_types[a[0]]
            if cls != "Function":
                raise UnsupportedConstruct("non-Function variable in body")
            var_box[res] = [list(values[a[2]]) if len(a) > 2 and isinstance(values[a[2]], list)
                            else (values[a[2]] if len(a) > 2 else types.zero(pointee))]
            pointers[res] = (var_box[res], 0)
        elif op == "OpAccessChain":
            base = a[1]
            if base not in pointers:
                raise UnsupportedConstruct("OpAccessChain on non-pointer")
            p = pointers[base]
            if p[0] == "pc":
                off, cur = walk(pc_type[base], [values[i] for i in a[2:]])
                pointers[res] = ("pc", p[1] + off)
                pc_type[res] = cur
                var_class[res] = "PushConstant"
                continue
            if p[0] == "psb":
                if len(a) != 3:
                    raise UnsupportedConstruct("multi-index physical-storage-buffer access chain")
                elem = ptr_types[a[0]][1]
                if types.t[elem][0] == "rarr":
                    stride = types.size(types.t[elem][1])
                elif types.t[elem][0] in ("int", "float", "vec"):
                    stride = types.size(elem)
                else:
                    raise UnsupportedConstruct(f"physical-storage-buffer access chain into {types.t[elem][0]}")
                pointers[res] = ("psb", p[1] + stride * values[a[2]])
                continue
            for idx_id in a[2:]:
                idx = values[idx_id]
                container = deref(p)
                if not isinstance(container, (list, tuple)) or not 0 <= idx < len(container):
                    raise UnsupportedConstruct("OpAccessChain into non-array or out of range")
                p = (container, idx)
            pointers[res] = p
        elif op == "OpLoad":
            p = pointers[a[1]]
            if p[0] == "pc":
                w = p[1] // 4
                values[res] = push_constants[w] if 0 <= w < len(push_constants) else 0
            elif p[0] == "psb":
                values[res] = memory.get(p[1], 0)
            else:
                v = deref(p)
                values[res] = list(v) if isinstance(v, list) else v
        elif op == "OpStore":
            p = pointers[a[0]]
            if p[0] == "pc":
                raise UnsupportedConstruct("store to PushConstant")
            if p[0] == "psb":
                memory[p[1]] = val(a[1])
            else:
                p[0][p[1]] = val(a[1])
        elif op == "OpSelect":
            if not isinstance(val(a[1]), bool):
                raise UnsupportedConstruct("OpSelect with a non-scalar-bool condition")
            values[res] = val(a[2]) if val(a[1]) else val(a[3])
        elif op == "OpCopyObject":
            values[res] = val(a[1])
        elif op == "OpBitcast":
            k, v = types.t[a[0]], val(a[1])
            if k[0] == "float" and isinstance(v, int):
                values[res] = bits_f32(v)
            elif k[0] == "int" and isinstance(v, float):
                values[res] = f32_bits(v)
            elif k[0] == "int" and isinstance(v, int):
                values[res] = v & wmask(a[0])
            else:
                raise UnsupportedConstruct("OpBitcast between these types")
        elif op == "OpCompositeExtract":
            v = val(a[1])
            for i in a[2:]:
                v = v[int(i)]
            values[res] = v
        elif op == "OpCompositeConstruct":
            values[res] = tuple(val(x) for x in a[1:])
        elif op == "OpBitwiseAnd":
            values[res] = (val(a[1]) & val(a[2])) & wmask(a[0])
        elif op == "OpBitwiseOr":
            values[res] = (val(a[1]) | val(a[2])) & wmask(a[0])
        elif op == "OpBitwiseXor":
            values[res] = (val(a[1]) ^ val(a[2])) & wmask(a[0])
        elif op == "OpNot":
            values[res] = (~val(a[1])) & wmask(a[0])
        elif op == "OpShiftLeftLogical":
            bits = types.bits(a[0])
            if val(a[2]) >= bits:
                raise UnsupportedConstruct(f"shift amount >= {bits} (undefined in SPIR-V)")
            values[res] = (val(a[1]) << val(a[2])) & wmask(a[0])
        elif op == "OpShiftRightLogical":
            bits = types.bits(a[0])
            if val(a[2]) >= bits:
                raise UnsupportedConstruct(f"shift amount >= {bits} (undefined in SPIR-V)")
            values[res] = (val(a[1]) & wmask(a[0])) >> val(a[2])
        elif op == "OpShiftRightArithmetic":
            bits = types.bits(a[0])
            if val(a[2]) >= bits:
                raise UnsupportedConstruct(f"shift amount >= {bits} (undefined in SPIR-V)")
            values[res] = (signed(val(a[1]), bits) >> val(a[2])) & wmask(a[0])
        elif op == "OpIAdd":
            values[res] = (val(a[1]) + val(a[2])) & wmask(a[0])
        elif op == "OpISub":
            values[res] = (val(a[1]) - val(a[2])) & wmask(a[0])
        elif op == "OpIMul":
            values[res] = (val(a[1]) * val(a[2])) & wmask(a[0])
        elif op == "OpUConvert":
            values[res] = val(a[1]) & wmask(a[0])
        elif op == "OpBitCount":
            values[res] = (val(a[1]) & wmask(a[0])).bit_count()
        elif op == "OpISubBorrow":
            x, y = val(a[1]) & MASK, val(a[2]) & MASK
            values[res] = ((x - y) & MASK, 1 if x < y else 0)
        elif op == "OpIEqual":
            values[res] = val(a[1]) == val(a[2])
        elif op == "OpINotEqual":
            values[res] = val(a[1]) != val(a[2])
        elif op == "OpSLessThan":
            values[res] = _s32(val(a[1])) < _s32(val(a[2]))
        elif op == "OpULessThan":
            values[res] = val(a[1]) < val(a[2])
        elif op == "OpLogicalAnd":
            x, y = val(a[1]), val(a[2])
            if not (isinstance(x, bool) and isinstance(y, bool)):
                raise UnsupportedConstruct("OpLogicalAnd on non-bool")
            values[res] = x and y
        elif op == "OpLogicalOr":
            x, y = val(a[1]), val(a[2])
            if not (isinstance(x, bool) and isinstance(y, bool)):
                raise UnsupportedConstruct("OpLogicalOr on non-bool")
            values[res] = x or y
        elif op == "OpLogicalNot":
            values[res] = not val(a[1])
        elif op in ("OpFAdd", "OpFSub", "OpFMul"):
            x, y = val(a[1]), val(a[2])
            if not (isinstance(x, float) and isinstance(y, float)):
                raise UnsupportedConstruct(f"{op} on non-float")
            values[res] = f32(x + y if op == "OpFAdd" else x - y if op == "OpFSub" else x * y)
        elif op == "OpFNegate":
            values[res] = f32(-val(a[1]))
        elif op in ("OpFOrdEqual", "OpFOrdLessThan", "OpFOrdGreaterThan", "OpFOrdGreaterThanEqual",
                    "OpFOrdLessThanEqual", "OpFUnordNotEqual"):
            x, y = val(a[1]), val(a[2])
            if not (isinstance(x, float) and isinstance(y, float)):
                raise UnsupportedConstruct(f"{op} on non-float")
            if op == "OpFUnordNotEqual":
                values[res] = not (x == y)          # true when unordered, unlike OpFOrdNotEqual
            else:
                values[res] = {"OpFOrdEqual": x == y, "OpFOrdLessThan": x < y,
                               "OpFOrdGreaterThan": x > y, "OpFOrdGreaterThanEqual": x >= y,
                               "OpFOrdLessThanEqual": x <= y}[op]
        elif op in ("OpBitFieldUExtract", "OpBitFieldSExtract"):
            base, off, cnt = val(a[1]), val(a[2]), val(a[3])
            if off + cnt > 32:
                raise UnsupportedConstruct("bit-field extract beyond 32 bits (undefined in SPIR-V)")
            field = (base >> off) & ((1 << cnt) - 1)
            if op == "OpBitFieldSExtract" and cnt and field >> (cnt - 1):
                field -= 1 << cnt
            values[res] = field & MASK
        elif op == "OpBitFieldInsert":
            base, ins, off, cnt = (val(x) for x in a[1:5])
            if off + cnt > 32:
                raise UnsupportedConstruct("bit-field insert beyond 32 bits (undefined in SPIR-V)")
            mk = ((1 << cnt) - 1) << off
            values[res] = ((base & ~mk) | ((ins << off) & mk)) & MASK
        elif op == "OpConvertUToPtr":
            values[res] = ("psb", val(a[1]))
            pointers[res] = values[res]
        elif op == "OpExtInst":
            if len(a) < 3 or a[1] != "%glsl":
                raise UnsupportedConstruct("extended instruction set")
            name, args = a[2], [val(x) for x in a[3:]]
            if name == "UnpackHalf2x16":
                values[res] = (half_to_float(args[0]), half_to_float(args[0] >> 16))
            elif name == "PackHalf2x16":
                values[res] = float_to_half(args[0][0]) | (float_to_half(args[0][1]) << 16)
            elif name == "Fma":
                # Exact product and sum in double, one rounding to binary32. Not a true fused operation in
                # general, but exact for the f16-derived operands the tests use (see tests/semantic/README.md).
                x, y, z = args
                if isinstance(x, tuple):
                    values[res] = tuple(f32(p * q + s) for p, q, s in zip(x, y, z))
                else:
                    values[res] = f32(x * y + z)
            elif name in ("Sin", "Cos"):
                values[res] = f32(math.sin(args[0]) if name == "Sin" else math.cos(args[0]))
            elif name == "FAbs":
                values[res] = f32(abs(args[0]))
            elif name == "UMin":
                values[res] = min(args)
            elif name == "SMax":
                values[res] = max(args, key=_s32)
            elif name == "FindSMsb":
                x = _s32(args[0])
                if x in (0, -1):
                    values[res] = MASK
                elif x > 0:
                    values[res] = x.bit_length() - 1
                else:
                    values[res] = (~x).bit_length() - 1
            else:
                raise UnsupportedConstruct(f"GLSL.std.450 {name}")
        else:
            raise UnsupportedConstruct(op)

    if killed:
        return {}
    result = {}
    for name in outputs:
        v = var_box[name][0]
        flat = v if isinstance(v, tuple) else (v,)
        result[name] = tuple(f32_bits(x) if isinstance(x, float) else x & MASK for x in flat)
    return result
