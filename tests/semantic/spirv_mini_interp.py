"""A deliberately tiny interpreter for the straight-line SPIR-V assembly that tools/rdna2spirv.py emits for
branch-free pixel shaders. It exists so tests can check *computed values*, not just that the module validates.

Design rules (see tests/semantic/README.md):
  * Written from the SPIR-V and GLSL.std.450 specifications. It shares no code with the translator.
  * Strict: any opcode, extended instruction, type or storage class not listed below raises
    UnsupportedConstruct. Nothing is skipped or approximated silently, so a translator change that starts
    emitting something new makes the test error out instead of passing.
  * Single invocation (one lane), no control flow, no memory beyond Function/Output variables.
    PushConstant data reads as zero (the shaders under test never read it, but the translator copies the
    32 user-data dwords at entry); writing to it is rejected.

Values: uint/int are Python ints in [0, 2**32) (signedness is applied by the operation, as in SPIR-V);
float is a Python float rounded to binary32; vectors are tuples; arrays are lists; bool is bool.
"""
import re
import struct

MASK = 0xFFFFFFFF


class UnsupportedConstruct(NotImplementedError):
    """The module uses something this interpreter deliberately does not model."""


# Instructions that carry no runtime semantics for a single-entry-point, straight-line module.
_IGNORED = frozenset({
    "OpCapability", "OpExtInstImport", "OpMemoryModel", "OpEntryPoint", "OpExecutionMode", "OpDecorate",
    "OpMemberDecorate", "OpTypeVoid", "OpTypeFunction", "OpLabel", "OpReturn", "OpFunctionEnd",
    "OpName", "OpSource",
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


def run_fragment(text):
    """Execute the entry point of `text` once and return the words written to every Output variable.

    Returns {output_variable_name: tuple_of_32bit_words}. Float outputs are returned as their bit patterns.
    """
    types = _Types()
    values = {}                 # id -> value (constants and results)
    constants = {}              # id -> value, kept apart so zero-initialised arrays can be built
    ptr_types = {}              # pointer type id -> (storage class, pointee type id)
    var_class = {}              # variable id -> storage class
    var_box = {}                # variable id -> [value]
    pointers = {}               # id -> (container list, index) for results of OpAccessChain / variables
    in_body = False
    body = []
    outputs = []

    def val(x):
        if x in values:
            return values[x]
        raise UnsupportedConstruct(f"use of undefined id {x}")

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
            types.t[res] = ("int", int(a[0]), int(a[1]))
            if int(a[0]) != 32:
                raise UnsupportedConstruct("non-32-bit integer type")
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
        elif op == "OpTypeStruct":
            types.t[res] = ("struct", a)
        elif op == "OpTypePointer":
            ptr_types[res] = (a[0], a[1])
        elif op == "OpConstant":
            k = types.t[a[0]]
            if k[0] == "float":
                constants[res] = values[res] = f32(float(a[1]))
            elif k[0] == "int":
                constants[res] = values[res] = int(a[1]) & MASK
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
                var_box[res] = [None]                  # reads give zero, see _load
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
            pointers[res] = (var_box[res], 0)
        else:
            raise UnsupportedConstruct(f"{op} in module header")
    if not in_body:
        raise UnsupportedConstruct("no function body")

    def deref(p):
        box, i = p
        return box[i]

    for res, op, a, line in body:
        if op in _IGNORED:
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
            if var_class.get(base) == "PushConstant":
                pointers[res] = ("pc", None)
                var_class[res] = "PushConstant"
                continue
            p = pointers[base]
            for idx_id in a[2:]:
                idx = values[idx_id]
                container = deref(p)
                if not isinstance(container, list) or not 0 <= idx < len(container):
                    raise UnsupportedConstruct("OpAccessChain into non-array or out of range")
                p = (container, idx)
            pointers[res] = p
        elif op == "OpLoad":
            p = pointers[a[1]]
            if p[0] == "pc":
                values[res] = 0                       # push constants read as zero
            else:
                v = deref(p)
                values[res] = list(v) if isinstance(v, list) else v
        elif op == "OpStore":
            p = pointers[a[0]]
            if p[0] == "pc":
                raise UnsupportedConstruct("store to PushConstant")
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
                values[res] = v & MASK
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
            values[res] = val(a[1]) & val(a[2])
        elif op == "OpBitwiseOr":
            values[res] = val(a[1]) | val(a[2])
        elif op == "OpShiftLeftLogical":
            if val(a[2]) >= 32:
                raise UnsupportedConstruct("shift amount >= 32 (undefined in SPIR-V)")
            values[res] = (val(a[1]) << val(a[2])) & MASK
        elif op == "OpIAdd":
            values[res] = (val(a[1]) + val(a[2])) & MASK
        elif op == "OpISub":
            values[res] = (val(a[1]) - val(a[2])) & MASK
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
            elif name == "UMin":
                values[res] = min(args)
            elif name == "SMax":
                values[res] = max(args, key=_s32)
            else:
                raise UnsupportedConstruct(f"GLSL.std.450 {name}")
        else:
            raise UnsupportedConstruct(op)

    result = {}
    for name in outputs:
        v = var_box[name][0]
        flat = v if isinstance(v, tuple) else (v,)
        result[name] = tuple(f32_bits(x) if isinstance(x, float) else x & MASK for x in flat)
    return result
