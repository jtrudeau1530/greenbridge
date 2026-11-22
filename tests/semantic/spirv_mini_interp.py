"""A deliberately tiny interpreter for the SPIR-V assembly that tools/rdna2spirv.py emits for branch-free pixel
shaders and simple structured compute shaders. It exists so tests can check *computed values*, not just that the
module validates.

Design rules (see tests/semantic/README.md):
  * Written from the SPIR-V and GLSL.std.450 specifications. It shares no code with the translator.
  * Strict: any opcode, extended instruction, type or storage class not listed below raises
    UnsupportedConstruct. Nothing is skipped or approximated silently, so a translator change that starts
    emitting something new makes the test error out instead of passing.
  * One invocation (one lane), or with run_workgroup several in lockstep, which meet only at
    subgroup operations (Shuffle, Ballot, Any, All, BitwiseOr Reduce; BallotBitExtract and BallotFindLSB
    are per lane). Control flow is the forward-only discard test and the forward merge/dynamic-bounds
    branches the translator emits: OpBranch / OpBranchConditional to a later label plus OpPhi at the merge block.
    Backward branches, loops, calls and switches are rejected. Function/Output/Input variables live in Python
    boxes; PushConstant reads can be seeded with a word list; physical-storage-buffer loads/stores (and 32-bit
    atomics, which with one invocation are read-combine-write) use a flat address -> word dictionary. Workgroup
    storage and images are rejected.
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

_ATOMICS = frozenset({
    "OpAtomicIAdd", "OpAtomicISub", "OpAtomicUMin", "OpAtomicUMax", "OpAtomicSMin", "OpAtomicSMax", "OpAtomicAnd",
    "OpAtomicOr", "OpAtomicXor", "OpAtomicExchange", "OpAtomicCompareExchange",
})

_LINE = re.compile(r"^(?:(%\S+)\s*=\s*)?(Op\w+)\s*(.*)$")


class _Undef:
    """An undefined value (SPIR-V: a shift by at least the bit width). Pure arithmetic on it gives another
    undefined value; any other use (a store, a branch or select condition, a subgroup operation, an output)
    raises, so a result that does not depend on it, such as the unselected side of OpSelect, is still exact."""

    def __repr__(self):
        return "<undefined>"


UNDEF = _Undef()

# Operations whose result is undefined when an operand is (they have no side effect and no control).
_PURE = frozenset({
    "OpBitwiseAnd", "OpBitwiseOr", "OpBitwiseXor", "OpNot", "OpShiftLeftLogical", "OpShiftRightLogical",
    "OpShiftRightArithmetic", "OpIAdd", "OpISub", "OpIMul", "OpUConvert", "OpBitCount", "OpIEqual",
    "OpINotEqual", "OpSLessThan", "OpULessThan", "OpUGreaterThan", "OpUGreaterThanEqual", "OpCopyObject",
    "OpBitcast",
    "OpBitFieldUExtract", "OpBitFieldSExtract", "OpBitFieldInsert",
})


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
    A module with a subgroup operation is rejected here; see run_workgroup.
    """
    gen = _invocation(text, inputs, push_constants, memory)
    try:
        req = next(gen)
    except StopIteration as done:
        return done.value
    gen.close()
    raise UnsupportedConstruct(f"{req[0]} in a single invocation (use run_workgroup)")


class UndefinedResult(Exception):
    """A subgroup operation whose result SPIR-V leaves undefined (for example a shuffle from a lane that is not in
    the subgroup). Raised instead of inventing a value."""


def run_workgroup(text, invocations, subgroup_size, inputs_for=None, push_constants=None, memory=None,
                  out_of_range="raise"):
    """Execute `invocations` invocations of `text` in lockstep and return the shared `memory`.

    Invocation i is subgroup i // subgroup_size, lane i % subgroup_size (a 1-D workgroup split into subgroups in
    invocation order). When `invocations` is not a multiple of `subgroup_size` the last subgroup is partial: its
    missing lanes are not running. `inputs_for(i)` gives invocation i's Input values (see run_fragment); push
    constants and the physical-address memory are shared. Every invocation must reach every subgroup operation,
    in the same order (the translator emits them in uniform control flow and models EXEC with a variable, so a
    lane that is off in EXEC still executes them).

    Subgroup operations (Subgroup scope), over the running lanes of the caller's subgroup:
      * OpGroupNonUniformShuffle returns the value of lane Id. An Id at or past `subgroup_size` is undefined in
        SPIR-V: `out_of_range="raise"` raises UndefinedResult; `out_of_range="wrap"` takes lane Id mod
        subgroup_size instead, a model of one observed driver behaviour and not of the specification (Leonida
        issue #94: on a 32-wide NVIDIA subgroup lane 40 read lane 8). An Id inside the subgroup whose lane is not
        running gives an undefined value (raises only if used, see _Undef).
      * OpGroupNonUniformBallot sets bit l for each running lane l whose predicate is true.
      * OpGroupNonUniformAny / All: the predicate of any / every running lane.
      * OpGroupNonUniformBitwiseOr Reduce: the OR of every running lane's value.
    """
    if out_of_range not in ("raise", "wrap"):
        raise ValueError(f"out_of_range: {out_of_range!r}")
    memory = {} if memory is None else memory
    gens = [_invocation(text, (inputs_for(i) if inputs_for else None), push_constants, memory, step=True)
            for i in range(invocations)]
    pending = [None] * invocations             # what each invocation yielded last; None once it has returned

    def advance(i, value):
        try:
            pending[i] = gens[i].send(value)
        except StopIteration:
            pending[i] = None

    for i in range(invocations):
        advance(i, None)
    while True:
        # One instruction per invocation per round, in invocation order: invocations at the same instruction
        # execute it one after the other before any goes on, as the lanes of a SIMD wave do, so a load that
        # follows a store in program order sees every lane's store and none from a later instruction.
        while any(req is not None and req[0] == "tick" for req in pending):
            for i in range(invocations):
                if pending[i] is not None and pending[i][0] == "tick":
                    advance(i, None)
        live = [i for i in range(invocations) if pending[i] is not None]
        if not live:
            break
        if len(live) != invocations or len({pending[i][1] for i in live}) != 1:
            raise UnsupportedConstruct("subgroup operation not reached by every invocation at the same point")
        results = []
        for i in range(invocations):
            kind, _, _value, ident = pending[i]
            base = (i // subgroup_size) * subgroup_size
            group = [pending[j] for j in range(base, min(base + subgroup_size, invocations))]
            if kind == "OpGroupNonUniformBallot":
                # Bit l of the 128-bit mask = lane l of this subgroup is running with a true predicate.
                bits = sum(1 << l for l, req in enumerate(group) if req[2])
                results.append(tuple((bits >> (32 * w)) & MASK for w in range(4)))
            elif kind == "OpGroupNonUniformAny":
                results.append(any(req[2] for req in group))
            elif kind == "OpGroupNonUniformAll":
                results.append(all(req[2] for req in group))
            elif kind == "OpGroupNonUniformBitwiseOr":
                acc = 0
                for req in group:
                    acc |= req[2]
                results.append(acc)
            elif kind == "OpGroupNonUniformShuffle":
                if not 0 <= ident < subgroup_size:
                    if out_of_range == "raise":
                        raise UndefinedResult(f"invocation {i}: shuffle from lane {ident} of a "
                                              f"{subgroup_size}-wide subgroup")
                    ident %= subgroup_size
                results.append(group[ident][2] if ident < len(group) else UNDEF)
            else:
                raise UnsupportedConstruct(kind)
        for i in range(invocations):
            advance(i, results[i])
    return memory


_TICK = ("tick", None, None, None)


def _invocation(text, inputs, push_constants, memory, step=False):
    """One invocation as a generator. It yields (opcode, instruction index, value, id) at each subgroup operation
    and expects that operation's result back; with `step` it also yields _TICK before each instruction. It
    returns run_fragment's result."""
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
    id_type = {}                # constant / result id -> its type id (the width of a signed comparison)
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
                id_type[res] = a[0]
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

    def sval(x):
        """An integer operand as signed, at the width of its type (a scalar int; 32 if the type is unknown)."""
        t = types.t.get(id_type.get(x))
        if t is not None and t[0] != "int":
            raise UnsupportedConstruct(f"signed comparison of a {t[0]}")
        return signed(val(x), t[1] if t else 32)

    labels = {res: idx for idx, (res, op, _, _) in enumerate(body) if op == "OpLabel" and res}
    pc = 0
    killed = False
    cur_label = prev_label = None
    while pc < len(body):
        if step:
            yield _TICK
        res, op, a, line = body[pc]
        pc += 1
        if res and a:
            id_type[res] = a[0]
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
        if op in _PURE and any(values.get(x) is UNDEF for x in a[1:]):
            values[res] = UNDEF
            continue
        used = a[1:2] if op == "OpSelect" else a      # OpSelect may pass the chosen undefined value on
        if any(values.get(x) is UNDEF for x in used):
            raise UnsupportedConstruct(f"{op} uses an undefined value: {line}")
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
        elif op in _ATOMICS:
            # One invocation, so an atomic is read, combine, write; the result is the original value (SPIR-V spec,
            # 3.42.18). Operands: result type, pointer, scope, semantics(, unequal semantics), value(, comparator).
            p = pointers[a[1]]
            if p[0] != "psb":
                raise UnsupportedConstruct(f"{op} outside PhysicalStorageBuffer")
            if types.bits(a[0]) != 32:
                raise UnsupportedConstruct(f"{op} on a non-32-bit integer")
            old = memory.get(p[1], 0)
            if op == "OpAtomicCompareExchange":
                new = val(a[5]) if old == val(a[6]) & MASK else old
            else:
                v = val(a[4]) & MASK
                new = {"OpAtomicIAdd": old + v, "OpAtomicISub": old - v, "OpAtomicUMin": min(old, v),
                       "OpAtomicUMax": max(old, v), "OpAtomicSMin": old if _s32(old) <= _s32(v) else v,
                       "OpAtomicSMax": old if _s32(old) >= _s32(v) else v, "OpAtomicAnd": old & v,
                       "OpAtomicOr": old | v, "OpAtomicXor": old ^ v, "OpAtomicExchange": v}[op]
            memory[p[1]] = new & MASK
            values[res] = old
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
                values[res] = UNDEF             # undefined in SPIR-V; raises only if it is used
                continue
            values[res] = (val(a[1]) << val(a[2])) & wmask(a[0])
        elif op == "OpShiftRightLogical":
            bits = types.bits(a[0])
            if val(a[2]) >= bits:
                values[res] = UNDEF             # undefined in SPIR-V; raises only if it is used
                continue
            values[res] = (val(a[1]) & wmask(a[0])) >> val(a[2])
        elif op == "OpShiftRightArithmetic":
            bits = types.bits(a[0])
            if val(a[2]) >= bits:
                values[res] = UNDEF             # undefined in SPIR-V; raises only if it is used
                continue
            values[res] = (signed(val(a[1]), bits) >> val(a[2])) & wmask(a[0])
        elif op == "OpIAdd":
            values[res] = (val(a[1]) + val(a[2])) & wmask(a[0])
        elif op == "OpISub":
            values[res] = (val(a[1]) - val(a[2])) & wmask(a[0])
        elif op == "OpIMul":
            values[res] = (val(a[1]) * val(a[2])) & wmask(a[0])
        elif op == "OpUConvert":
            values[res] = val(a[1]) & wmask(a[0])
        elif op == "OpSConvert":
            # Sign-extend from the operand's width, then wrap to the result's (SPIR-V spec, OpSConvert).
            if types.t.get(id_type.get(a[1]), ("?",))[0] != "int":
                raise UnsupportedConstruct("OpSConvert of an operand with no known integer type")
            values[res] = sval(a[1]) & wmask(a[0])
        elif op == "OpBitReverse":
            bits = types.bits(a[0])
            values[res] = int(f"{val(a[1]) & wmask(a[0]):0{bits}b}"[::-1], 2)
        elif op in ("OpConvertUToF", "OpConvertSToF"):
            x = val(a[1])
            if isinstance(x, bool) or not isinstance(x, int) or x > MASK:
                raise UnsupportedConstruct(f"{op} on a non-32-bit-integer operand")
            values[res] = f32(float(_s32(x) if op == "OpConvertSToF" else x))
        elif op == "OpBitCount":
            values[res] = (val(a[1]) & wmask(a[0])).bit_count()
        elif op == "OpISubBorrow":
            x, y = val(a[1]) & MASK, val(a[2]) & MASK
            values[res] = ((x - y) & MASK, 1 if x < y else 0)
        elif op in ("OpUMulExtended", "OpSMulExtended"):
            # Result struct {low, high} of the full 64-bit product (SPIR-V spec, 32-bit scalars only).
            x, y = val(a[1]) & MASK, val(a[2]) & MASK
            if op == "OpSMulExtended":
                x, y = _s32(x), _s32(y)
            p = (x * y) & 0xFFFFFFFFFFFFFFFF
            values[res] = (p & MASK, p >> 32)
        elif op == "OpIAddCarry":
            x, y = val(a[1]) & MASK, val(a[2]) & MASK
            values[res] = ((x + y) & MASK, (x + y) >> 32)
        elif op == "OpIEqual":
            values[res] = val(a[1]) == val(a[2])
        elif op == "OpINotEqual":
            values[res] = val(a[1]) != val(a[2])
        elif op in ("OpSLessThan", "OpSLessThanEqual", "OpSGreaterThan", "OpSGreaterThanEqual"):
            x, y = sval(a[1]), sval(a[2])
            values[res] = {"OpSLessThan": x < y, "OpSLessThanEqual": x <= y,
                           "OpSGreaterThan": x > y, "OpSGreaterThanEqual": x >= y}[op]
        elif op == "OpULessThan":
            values[res] = val(a[1]) < val(a[2])
        elif op == "OpULessThanEqual":
            values[res] = val(a[1]) <= val(a[2])
        elif op == "OpUGreaterThan":
            values[res] = val(a[1]) > val(a[2])
        elif op == "OpUGreaterThanEqual":
            values[res] = val(a[1]) >= val(a[2])
        elif op == "OpGroupNonUniformShuffle":
            # OpGroupNonUniformShuffle <result type> <scope> <value> <id>; the scope must be Subgroup (3).
            if types.t[a[0]][0] != "int" or val(a[1]) != 3:
                raise UnsupportedConstruct("OpGroupNonUniformShuffle other than a Subgroup-scope integer")
            values[res] = yield (op, pc - 1, val(a[2]), val(a[3]))
        elif op == "OpGroupNonUniformBallot":
            # OpGroupNonUniformBallot <uvec4 type> <scope> <bool predicate>
            k = types.t[a[0]]
            if k[0] != "vec" or k[2] != 4 or val(a[1]) != 3 or not isinstance(val(a[2]), bool):
                raise UnsupportedConstruct("OpGroupNonUniformBallot other than a Subgroup-scope uvec4")
            values[res] = yield (op, pc - 1, val(a[2]), None)
        elif op in ("OpGroupNonUniformAny", "OpGroupNonUniformAll"):
            # <bool> <scope> <bool predicate>
            if val(a[1]) != 3 or not isinstance(val(a[2]), bool):
                raise UnsupportedConstruct(f"{op} other than a Subgroup-scope bool")
            values[res] = yield (op, pc - 1, val(a[2]), None)
        elif op == "OpGroupNonUniformBitwiseOr":
            # <type> <scope> <group operation> <value>: only Reduce over a 32-bit integer.
            if val(a[1]) != 3 or a[2] != "Reduce" or types.t[a[0]][:2] != ("int", 32):
                raise UnsupportedConstruct("OpGroupNonUniformBitwiseOr other than a Subgroup-scope 32-bit Reduce")
            values[res] = yield (op, pc - 1, val(a[3]) & MASK, None)
        elif op == "OpGroupNonUniformBallotFindLSB":
            # <uint> <scope> <uvec4 ballot>: the lowest set bit; undefined when no bit is set.
            ballot = val(a[2])
            if val(a[1]) != 3:
                raise UnsupportedConstruct("OpGroupNonUniformBallotFindLSB scope")
            bits = sum(w << (32 * k) for k, w in enumerate(ballot))
            values[res] = (bits & -bits).bit_length() - 1 if bits else UNDEF
        elif op == "OpGroupNonUniformBallotBitExtract":
            # OpGroupNonUniformBallotBitExtract <bool> <scope> <uvec4 ballot> <index>: needs no other lane.
            ballot, idx = val(a[2]), val(a[3])
            if val(a[1]) != 3 or not 0 <= idx < 128:
                raise UnsupportedConstruct("OpGroupNonUniformBallotBitExtract scope or index")
            values[res] = bool((ballot[idx // 32] >> (idx % 32)) & 1)
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
        elif op in ("OpConvertFToU", "OpConvertFToS"):
            x, bits = val(a[1]), types.bits(a[0])
            if not isinstance(x, float) or not math.isfinite(x):
                raise UnsupportedConstruct(f"{op} of a non-finite value (undefined in SPIR-V)")
            n = math.trunc(x)
            lo, hi = (0, 1 << bits) if op == "OpConvertFToU" else (-(1 << (bits - 1)), 1 << (bits - 1))
            if not lo <= n < hi:
                raise UnsupportedConstruct(f"{op} out of range (undefined in SPIR-V)")
            values[res] = n & wmask(a[0])
        elif op in ("OpFOrdEqual", "OpFOrdNotEqual", "OpFOrdLessThan", "OpFOrdGreaterThan",
                    "OpFOrdGreaterThanEqual", "OpFOrdLessThanEqual", "OpFUnordEqual", "OpFUnordNotEqual",
                    "OpFUnordLessThan", "OpFUnordGreaterThan", "OpFUnordGreaterThanEqual", "OpFUnordLessThanEqual"):
            x, y = val(a[1]), val(a[2])
            if not (isinstance(x, float) and isinstance(y, float)):
                raise UnsupportedConstruct(f"{op} on non-float")
            # Ordered: false if either operand is NaN. Unordered: true if either is NaN. Python's own comparisons
            # are already ordered (NaN compares false), and -0.0 == 0.0 as in SPIR-V.
            unordered = math.isnan(x) or math.isnan(y)
            rel = {"Equal": x == y, "NotEqual": x != y and not unordered, "LessThan": x < y,
                   "GreaterThan": x > y, "GreaterThanEqual": x >= y, "LessThanEqual": x <= y}[op.removeprefix("OpFUnord").removeprefix("OpFOrd")]
            values[res] = (rel or unordered) if op.startswith("OpFUnord") else rel
        elif op in ("OpIsNan", "OpIsInf"):
            x = val(a[1])
            if not isinstance(x, float):
                raise UnsupportedConstruct(f"{op} on non-float")
            values[res] = math.isnan(x) if op == "OpIsNan" else math.isinf(x)
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
            elif name == "SAbs":
                # Two's complement: the most negative value is its own absolute value (as on the hardware).
                values[res] = abs(_s32(args[0])) & MASK
            elif name in ("FMax", "FClamp"):
                if any(math.isnan(v) for v in args) or (name == "FClamp" and args[1] > args[2]):
                    raise UnsupportedConstruct(f"GLSL.std.450 {name} with these operands (undefined)")
                v = max(args[0], args[1])
                values[res] = f32(v if name == "FMax" else min(v, args[2]))
            elif name == "RoundEven":
                x = args[0]
                values[res] = f32(float(round(x))) if math.isfinite(x) else x    # round() breaks ties to even
            elif name == "FMin":
                if any(math.isnan(x) for x in args):
                    raise UnsupportedConstruct(f"GLSL.std.450 {name} of NaN (undefined)")
                values[res] = f32(min(args))
            elif name == "UMin":
                values[res] = min(args)
            elif name == "UMax":
                values[res] = max(args)
            elif name == "SMax":
                values[res] = max(args, key=_s32)
            elif name == "SMin":
                values[res] = min(args, key=_s32)
            elif name in ("UClamp", "SClamp"):
                # min(max(x, minVal), maxVal) (GLSL.std.450; undefined if minVal > maxVal, which raises here).
                key = _s32 if name == "SClamp" else (lambda v: v)
                x, lo, hi = args
                if key(lo) > key(hi):
                    raise UnsupportedConstruct(f"{name} with minVal > maxVal (undefined)")
                values[res] = min(max(x, lo, key=key), hi, key=key)
            elif name == "FindILsb":
                # Bit number of the least significant 1 bit, -1 if the value is 0 (GLSL.std.450).
                x = args[0] & MASK
                values[res] = (x & -x).bit_length() - 1 & MASK if x else MASK
            elif name == "FindUMsb":
                # Bit number of the most significant 1 bit, -1 if the value is 0 (GLSL.std.450).
                x = args[0] & MASK
                values[res] = x.bit_length() - 1 if x else MASK
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
