"""Unit and 64/128-lane workgroup execution tests for tools/wave64_emulate_opt.py (--wave64-emulate).

Verifies both exact optimizations introduced for Issue #884:
  1. Backward CFG SCC liveness: dead `scc_from_mask` / `scc_any_half` votes on lane-mask ops are elided
     under --wave64-emulate, while live SCC writes (direct, cross-block, loop-carried, carry-in) stay exact.
  2. Safe lane-local `s_cbranch_execz` / `s_cbranch_execnz` region folding + continuation chaining: single-entry,
     single-exit regions with no cross-lane ops, no barriers, no memory/LDS writes, no EXEC-widening, and no
     live-out scalars fold inline behind `if (%active)` (and chain single-predecessor continuations), keeping
     all 64 lanes of each wave in the same scheduler round.

Includes a strict workgroup SPIR-V interpreter with real OpControlBarrier uniformity checks, shared Workgroup
memory (%x_buf and %lds), 32-bit Workgroup atomics, OpSwitch, loops, and PhysicalStorageBuffer loads/stores so
the translated --wave64-emulate SPIR-V is executed bit-for-bit across 64-thread (1-wave) and 128-thread
(2-wave) workgroups and compared against an independent RDNA2 ISA reference.

Run from the repository root:  python -I tests/semantic/test_wave64_emulate_opt.py
"""
import os
import re
import struct
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import support  # noqa: E402

rdna2spirv = support.rdna2spirv
wave64_emulate_opt = rdna2spirv.wave64_emulate_opt

MASK32 = 0xFFFFFFFF
VSHARP = [0, 0, 0xFFFFFFFF, 0x31016FAC]  # raw buffer covering all offsets


# ---- Strict SPIR-V workgroup + barrier interpreter (for --wave64-emulate modules) ----
_SPV_LINE = re.compile(r"^(?:(%\w+)\s*=\s*)?(\w+)(?:\s+(.*))?$")
_SPV_IGNORED = frozenset({
    "OpCapability", "OpExtension", "OpExtInstImport", "OpMemoryModel", "OpEntryPoint",
    "OpExecutionMode", "OpDecorate", "OpMemberDecorate", "OpTypeVoid", "OpTypeFunction",
    "OpSelectionMerge", "OpLoopMerge", "OpFunctionEnd",
})


def _strip_comments(text):
    out = []
    for raw in text.splitlines():
        s = raw.split(";", 1)[0].strip()
        if s:
            out.append(s)
    return out


def _s32(v):
    v &= MASK32
    return v - 0x100000000 if v & 0x80000000 else v


def _parse_module(text):
    types = {}
    ptr_types = {}
    consts = {}
    id_type = {}
    globals_ = []
    body = []
    in_body = False
    first_fn = None

    def zero_of(tid):
        k = types[tid]
        if k[0] == "int":
            return 0
        if k[0] == "float":
            return 0.0
        if k[0] == "bool":
            return False
        if k[0] == "vec":
            return tuple(zero_of(k[1]) for _ in range(k[2]))
        if k[0] == "arr":
            return [zero_of(k[1]) for _ in range(k[2])]
        if k[0] == "struct":
            return [zero_of(m) for m in k[1]]
        raise NotImplementedError(f"zero_of {k}")

    for line in _strip_comments(text):
        m = _SPV_LINE.match(line)
        if not m:
            raise AssertionError(f"cannot parse SPIR-V line: {line!r}")
        res, op, rest = m.groups()
        a = rest.split() if rest else []
        if in_body:
            body.append((res, op, a, line))
            continue
        if op == "OpFunction":
            in_body = True
            first_fn = res
            continue
        if op in _SPV_IGNORED:
            continue
        if op == "OpTypeInt":
            types[res] = ("int", int(a[0]), int(a[1]))
        elif op == "OpTypeFloat":
            types[res] = ("float", int(a[0]))
        elif op == "OpTypeBool":
            types[res] = ("bool",)
        elif op == "OpTypeVector":
            types[res] = ("vec", a[0], int(a[1]))
        elif op == "OpTypeArray":
            types[res] = ("arr", a[0], int(consts[a[1]]))
        elif op == "OpTypeRuntimeArray":
            types[res] = ("rarr", a[0])
        elif op == "OpTypeStruct":
            types[res] = ("struct", a)
        elif op == "OpTypePointer":
            ptr_types[res] = (a[0], a[1])
        elif op == "OpConstant":
            k = types[a[0]]
            if k[0] == "int":
                consts[res] = int(a[1]) & ((1 << k[1]) - 1)
                id_type[res] = a[0]
            elif k[0] == "float":
                consts[res] = float(a[1])
                id_type[res] = a[0]
            else:
                raise NotImplementedError(f"OpConstant {k}")
        elif op == "OpConstantTrue":
            consts[res] = True
        elif op == "OpConstantFalse":
            consts[res] = False
        elif op == "OpConstantNull":
            consts[res] = zero_of(a[0])
        elif op == "OpVariable":
            cls, pointee = ptr_types[a[0]]
            assert a[1] == cls, f"storage class mismatch: {line}"
            globals_.append((res, cls, pointee, a[2] if len(a) > 2 else None))
        else:
            raise NotImplementedError(f"header op {op}: {line}")

    assert in_body and first_fn is not None
    labels = {res: idx for idx, (res, op, _, _) in enumerate(body) if op == "OpLabel" and res}
    fn_start = {first_fn: 0}
    for idx, (res, op, _, _) in enumerate(body):
        if op == "OpFunction":
            fn_start[res] = idx + 1
    return {
        "types": types, "ptr_types": ptr_types, "consts": consts, "id_type": id_type,
        "globals": globals_, "body": body, "labels": labels, "main_pc": fn_start.get("%main", 0),
        "zero_of": zero_of,
    }


def _check_spirv_structure(text):
    """Static structural sanity checks on translated SPIR-V text (unique labels, valid targets, no subgroup ops)."""
    lines = _strip_comments(text)
    in_fn = False
    seen_labels = set()
    defined_ids = set()
    referenced_labels = []
    prev_term = True
    for line in lines:
        m = _SPV_LINE.match(line)
        assert m, f"unparseable SPIR-V line: {line!r}"
        res, op, rest = m.groups()
        a = rest.split() if rest else []
        if op == "OpFunction":
            in_fn = True
            prev_term = True
            continue
        if op == "OpFunctionEnd":
            assert prev_term, "function ended without block terminator"
            in_fn = False
            continue
        if not in_fn:
            if res:
                defined_ids.add(res)
            continue
        if op == "OpLabel":
            assert prev_term, f"OpLabel {res} reached without preceding terminator"
            assert res not in seen_labels, f"duplicate OpLabel {res}"
            seen_labels.add(res)
            prev_term = False
            continue
        assert not prev_term, f"instruction after terminator before OpLabel: {line}"
        if op == "OpBranch":
            referenced_labels.append(a[0])
            prev_term = True
        elif op == "OpBranchConditional":
            referenced_labels.extend((a[1], a[2]))
            prev_term = True
        elif op == "OpSwitch":
            referenced_labels.append(a[1])
            for i in range(3, len(a), 2):
                referenced_labels.append(a[i])
            prev_term = True
        elif op in ("OpReturn", "OpReturnValue", "OpKill", "OpUnreachable"):
            prev_term = True
        elif op == "OpSelectionMerge":
            referenced_labels.append(a[0])
        elif op == "OpLoopMerge":
            referenced_labels.extend((a[0], a[1]))
    for lbl in referenced_labels:
        assert lbl in seen_labels, f"branch/merge references undefined label {lbl}"


def run_emulated_workgroup(spvasm, local_size, push_constants=None, memory=None, max_steps=2_000_000):
    """Execute a --wave64-emulate compute shader across all `lx * ly * lz` invocations with real shared
    Workgroup memory and strict uniform OpControlBarrier checking. Returns `(memory, barrier_count)`."""
    _check_spirv_structure(spvasm)
    mod = _parse_module(spvasm)
    types = mod["types"]
    ptr_types = mod["ptr_types"]
    consts = mod["consts"]
    body = mod["body"]
    labels = mod["labels"]
    zero_of = mod["zero_of"]

    push_constants = list(push_constants or [])
    memory = {} if memory is None else memory
    lx, ly, lz = local_size
    coords = [(x, y, z) for z in range(lz) for y in range(ly) for x in range(lx)]
    n_inv = len(coords)

    # Shared Workgroup variables across all invocations in the workgroup.
    wg_boxes = {}
    for res, cls, pointee, init_id in mod["globals"]:
        if cls == "Workgroup":
            wg_boxes[res] = [zero_of(pointee)]

    def walk_struct_offset(tid, idxs):
        off = 0
        cur = tid
        for idx in idxs:
            k = types[cur]
            if k[0] == "struct":
                for m_t in k[1][:idx]:
                    off += _type_bytes(m_t)
                cur = k[1][idx]
            elif k[0] == "arr":
                off += idx * _type_bytes(k[1])
                cur = k[1]
            else:
                raise NotImplementedError(f"walk {k}")
        return off, cur

    def _type_bytes(tid):
        k = types[tid]
        if k[0] == "int":
            return k[1] // 8
        if k[0] == "float":
            return k[1] // 8
        if k[0] == "vec":
            return k[2] * _type_bytes(k[1])
        if k[0] == "arr":
            return k[2] * _type_bytes(k[1])
        if k[0] == "struct":
            return sum(_type_bytes(m) for m in k[1])
        raise NotImplementedError(f"size {k}")

    def make_inv(idx, coord):
        vals = dict(consts)
        id_t = dict(mod["id_type"])
        ptrs = {}
        pc_t = {}
        for res, cls, pointee, init_id in mod["globals"]:
            if cls == "PushConstant":
                ptrs[res] = ("pc", 0)
                pc_t[res] = pointee
            elif cls == "Workgroup":
                ptrs[res] = (wg_boxes[res], 0)
            elif cls == "Input":
                if res == "%local_id":
                    box = [coord]
                elif res == "%wg_id":
                    box = [(0, 0, 0)]
                elif res == "%local_index":
                    box = [idx]
                else:
                    box = [zero_of(pointee)]
                ptrs[res] = (box, 0)
            else:
                raise NotImplementedError(f"global storage class {cls}")
        return {
            "pc": mod["main_pc"], "vals": vals, "id_t": id_t, "ptrs": ptrs, "pc_t": pc_t,
            "cur_lbl": None, "prev_lbl": None, "done": False, "barrier": None,
        }

    def step_until_barrier(inv, step_budget):
        vals, id_t, ptrs, pc_t = inv["vals"], inv["id_t"], inv["ptrs"], inv["pc_t"]
        pc = inv["pc"]
        steps = 0
        while pc < len(body):
            steps += 1
            if steps > step_budget:
                raise AssertionError(f"exceeded step budget {step_budget} (infinite loop?)")
            res, op, a, line = body[pc]
            pc += 1
            if res and a:
                id_t[res] = a[0]
            if op == "OpLabel":
                inv["prev_lbl"], inv["cur_lbl"] = inv["cur_lbl"], res
                continue
            if op in _SPV_IGNORED:
                continue
            if op == "OpControlBarrier":
                inv["pc"] = pc
                inv["barrier"] = pc - 1
                return steps
            if op == "OpReturn":
                inv["pc"] = pc
                inv["done"] = True
                return steps
            if op == "OpBranch":
                pc = labels[a[0]]
                continue
            if op == "OpBranchConditional":
                cond = vals[a[0]]
                assert isinstance(cond, bool), f"non-bool branch cond: {line}"
                pc = labels[a[1] if cond else a[2]]
                continue
            if op == "OpSwitch":
                sel = vals[a[0]]
                tgt = a[1]
                for k in range(2, len(a), 2):
                    if int(a[k]) == sel:
                        tgt = a[k + 1]
                        break
                pc = labels[tgt]
                continue
            if op == "OpPhi":
                matched = [a[k] for k in range(1, len(a) - 1, 2) if a[k + 1] == inv["prev_lbl"]]
                assert len(matched) == 1, f"OpPhi prev_lbl={inv['prev_lbl']!r} not in {line}"
                vals[res] = vals[matched[0]]
                continue
            if op == "OpVariable":
                cls, pointee = ptr_types[a[0]]
                assert cls == "Function"
                if len(a) > 2:
                    iv = vals[a[2]]
                    box = [list(iv) if isinstance(iv, list) else iv]
                else:
                    box = [zero_of(pointee)]
                ptrs[res] = (box, 0)
            elif op == "OpAccessChain":
                p = ptrs[a[1]]
                if p[0] == "pc":
                    off, cur = walk_struct_offset(pc_t[a[1]], [vals[i] for i in a[2:]])
                    ptrs[res] = ("pc", p[1] + off)
                    pc_t[res] = cur
                elif p[0] == "psb":
                    elem = ptr_types[a[0]][1]
                    stride = _type_bytes(types[elem][1] if types[elem][0] == "rarr" else elem)
                    ptrs[res] = ("psb", p[1] + stride * vals[a[2]])
                else:
                    cur_p = p
                    for idx_id in a[2:]:
                        idx = vals[idx_id]
                        container = cur_p[0][cur_p[1]]
                        assert 0 <= idx < len(container), f"OOB index {idx} in {line}"
                        cur_p = (container, idx)
                    ptrs[res] = cur_p
            elif op == "OpLoad":
                p = ptrs[a[1]]
                if p[0] == "pc":
                    w = p[1] // 4
                    vals[res] = push_constants[w] if 0 <= w < len(push_constants) else 0
                elif p[0] == "psb":
                    assert p[1] % 4 == 0, f"unaligned PSB load {p[1]:#x}"
                    vals[res] = memory.get(p[1], 0)
                else:
                    v = p[0][p[1]]
                    vals[res] = list(v) if isinstance(v, list) else v
            elif op == "OpStore":
                p = ptrs[a[0]]
                v = vals[a[1]]
                if p[0] == "psb":
                    assert p[1] % 4 == 0, f"unaligned PSB store {p[1]:#x}"
                    memory[p[1]] = v & MASK32
                else:
                    p[0][p[1]] = v
            elif op in ("OpAtomicOr", "OpAtomicIAdd", "OpAtomicAnd", "OpAtomicXor", "OpAtomicExchange"):
                p = ptrs[a[1]]
                old = memory.get(p[1], 0) if p[0] == "psb" else p[0][p[1]]
                v = vals[a[4]] & MASK32
                if op == "OpAtomicOr":
                    new = old | v
                elif op == "OpAtomicIAdd":
                    new = (old + v) & MASK32
                elif op == "OpAtomicAnd":
                    new = old & v
                elif op == "OpAtomicXor":
                    new = old ^ v
                else:
                    new = v
                if p[0] == "psb":
                    memory[p[1]] = new & MASK32
                else:
                    p[0][p[1]] = new & MASK32
                vals[res] = old
            elif op == "OpSelect":
                vals[res] = vals[a[2]] if vals[a[1]] else vals[a[3]]
            elif op == "OpCopyObject":
                vals[res] = vals[a[1]]
            elif op == "OpCompositeExtract":
                v = vals[a[1]]
                for i in a[2:]:
                    v = v[int(i)]
                vals[res] = v
            elif op == "OpCompositeConstruct":
                vals[res] = tuple(vals[x] for x in a[1:])
            elif op == "OpConvertUToPtr":
                ptrs[res] = ("psb", vals[a[1]])
                vals[res] = ptrs[res]
            elif op == "OpUConvert":
                bits = types[a[0]][1]
                vals[res] = vals[a[1]] & ((1 << bits) - 1)
            elif op == "OpBitcast":
                k = types[a[0]]
                v = vals[a[1]]
                if k[0] == "int":
                    vals[res] = v & ((1 << k[1]) - 1) if isinstance(v, int) else struct.unpack("<I", struct.pack("<f", v))[0]
                elif k[0] == "float":
                    vals[res] = struct.unpack("<f", struct.pack("<I", v & MASK32))[0] if isinstance(v, int) else v
                else:
                    raise NotImplementedError(f"OpBitcast {line}")
            elif op == "OpBitwiseAnd":
                vals[res] = (vals[a[1]] & vals[a[2]]) & ((1 << types[a[0]][1]) - 1)
            elif op == "OpBitwiseOr":
                vals[res] = (vals[a[1]] | vals[a[2]]) & ((1 << types[a[0]][1]) - 1)
            elif op == "OpBitwiseXor":
                vals[res] = (vals[a[1]] ^ vals[a[2]]) & ((1 << types[a[0]][1]) - 1)
            elif op == "OpNot":
                vals[res] = (~vals[a[1]]) & ((1 << types[a[0]][1]) - 1)
            elif op == "OpShiftLeftLogical":
                bits = types[a[0]][1]
                vals[res] = (vals[a[1]] << vals[a[2]]) & ((1 << bits) - 1)
            elif op == "OpShiftRightLogical":
                bits = types[a[0]][1]
                vals[res] = (vals[a[1]] & ((1 << bits) - 1)) >> vals[a[2]]
            elif op == "OpIAdd":
                bits = types[a[0]][1]
                vals[res] = (vals[a[1]] + vals[a[2]]) & ((1 << bits) - 1)
            elif op == "OpISub":
                bits = types[a[0]][1]
                vals[res] = (vals[a[1]] - vals[a[2]]) & ((1 << bits) - 1)
            elif op == "OpIMul":
                bits = types[a[0]][1]
                vals[res] = (vals[a[1]] * vals[a[2]]) & ((1 << bits) - 1)
            elif op == "OpUMod":
                vals[res] = vals[a[1]] % vals[a[2]]
            elif op == "OpUDiv":
                vals[res] = vals[a[1]] // vals[a[2]]
            elif op == "OpIAddCarry":
                s = (vals[a[1]] & MASK32) + (vals[a[2]] & MASK32)
                vals[res] = (s & MASK32, 1 if s > MASK32 else 0)
            elif op == "OpISubBorrow":
                x, y = vals[a[1]] & MASK32, vals[a[2]] & MASK32
                vals[res] = ((x - y) & MASK32, 1 if x < y else 0)
            elif op == "OpIEqual":
                vals[res] = vals[a[1]] == vals[a[2]]
            elif op == "OpINotEqual":
                vals[res] = vals[a[1]] != vals[a[2]]
            elif op == "OpULessThan":
                vals[res] = vals[a[1]] < vals[a[2]]
            elif op == "OpULessThanEqual":
                vals[res] = vals[a[1]] <= vals[a[2]]
            elif op == "OpUGreaterThan":
                vals[res] = vals[a[1]] > vals[a[2]]
            elif op == "OpUGreaterThanEqual":
                vals[res] = vals[a[1]] >= vals[a[2]]
            elif op == "OpSLessThan":
                vals[res] = _s32(vals[a[1]]) < _s32(vals[a[2]])
            elif op == "OpSGreaterThan":
                vals[res] = _s32(vals[a[1]]) > _s32(vals[a[2]])
            elif op == "OpLogicalNot":
                vals[res] = not vals[a[1]]
            elif op == "OpLogicalAnd":
                vals[res] = vals[a[1]] and vals[a[2]]
            elif op == "OpLogicalOr":
                vals[res] = vals[a[1]] or vals[a[2]]
            elif op == "OpBitCount":
                vals[res] = (vals[a[1]] & MASK32).bit_count()
            elif op == "OpBitFieldUExtract":
                base, off, cnt = vals[a[1]], vals[a[2]], vals[a[3]]
                vals[res] = (base >> off) & ((1 << cnt) - 1) if cnt else 0
            elif op == "OpExtInst":
                ext_op = a[2]
                if ext_op == "UMin":
                    vals[res] = min(vals[a[3]], vals[a[4]])
                elif ext_op == "UMax":
                    vals[res] = max(vals[a[3]], vals[a[4]])
                elif ext_op == "FindILsb":
                    v = vals[a[3]] & MASK32
                    vals[res] = (v & -v).bit_length() - 1 if v else MASK32
                elif ext_op == "FindUMsb":
                    v = vals[a[3]] & MASK32
                    vals[res] = v.bit_length() - 1 if v else MASK32
                else:
                    raise NotImplementedError(f"OpExtInst {ext_op}: {line}")
            else:
                raise NotImplementedError(f"unsupported op {op}: {line}")
        inv["pc"] = pc
        inv["done"] = True
        return steps

    invs = [make_inv(i, c) for i, c in enumerate(coords)]
    total_steps = 0
    barriers = 0
    while True:
        for inv in invs:
            if not inv["done"]:
                inv["barrier"] = None
                total_steps += step_until_barrier(inv, max_steps - total_steps)
        if all(inv["done"] for inv in invs):
            break
        # Strict uniform barrier check: every invocation must stop at the exact same OpControlBarrier.
        if any(inv["done"] for inv in invs):
            raise AssertionError("non-uniform OpControlBarrier: some invocations returned while others waited")
        sites = {inv["barrier"] for inv in invs}
        if len(sites) != 1 or None in sites:
            raise AssertionError(f"non-uniform OpControlBarrier sites: {sites}")
        barriers += 1
    return memory, barriers


def _translate_encoded(name, insns, local_size=(64, 1, 1), emulate=True, rsrc2=0x8088):
    """Translate a list of `(text, nbytes)` instructions with synthetic encoding placeholders."""
    lines = [f"; {name}  synthetic", "; mattr: +wavefrontsize64,-wavefrontsize32"]
    for txt, nb in insns:
        enc = ",".join(["0x00"] * nb)
        lines.append(f"\t{txt:52s} ; encoding: [{enc}]")
    prev_emu = rdna2spirv.WAVE64_EMULATE
    rdna2spirv.WAVE64_EMULATE = emulate
    try:
        with tempfile.TemporaryDirectory() as td:
            os.mkdir(os.path.join(td, "disasm"))
            os.mkdir(os.path.join(td, "fssm"))
            s_path = os.path.join(td, "disasm", name + ".s")
            f_path = os.path.join(td, "fssm", name + ".fssm")
            with open(s_path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")
            lx, ly, lz = local_size
            with open(f_path, "wb") as f:
                f.write(bytes(0x138) + struct.pack("<4I", rsrc2, lx, ly, lz))
            spv = rdna2spirv.translate(s_path)
            gen = rdna2spirv.translate.last_gen
            return spv, gen
    finally:
        rdna2spirv.WAVE64_EMULATE = prev_emu


def _switch_case_count(gen):
    return sum(1 for b in gen.blocks if b is not None) + len(gen.x_segs)


def _screenprobes_insns(n_steps, with_loads=False):
    insns = [("v_mov_b32_e32 v4, v0", 4)]
    for _ in range(n_steps):
        insns.append(("v_cmp_lt_u32_e64 s[10:11], 16, v4", 8))
        insns.append(("s_and_saveexec_b64 s[12:13], s[10:11]", 4))
        insns.append(("s_cbranch_execz 4" if with_loads else "s_cbranch_execz 2", 4))
        if with_loads:
            insns.append(("v_lshlrev_b32_e32 v8, 2, v0", 4))
            insns.append(("buffer_load_dword v5, v8, s[0:3], 0 offen", 8))
            insns.append(("v_add_nc_u32_e32 v4, v5, v4", 4))
        else:
            insns.append(("v_add_nc_u32_e32 v4, 1, v4", 4))
        insns.append(("v_xor_b32_e32 v4, 7, v4", 4))
        insns.append(("s_andn2_b64 exec, s[12:13], exec", 4))
        insns.append(("s_cbranch_execz 2", 4))
        insns.append(("v_add_nc_u32_e32 v4, 3, v4", 4))
        insns.append(("v_xor_b32_e32 v4, 5, v4", 4))
        insns.append(("s_or_b64 exec, exec, s[12:13]", 4))
    insns.append(("v_lshlrev_b32_e32 v9, 2, v0", 4))
    insns.append(("v_add_nc_u32_e32 v9, 1024, v9", 4))
    insns.append(("buffer_store_dword v4, v9, s[0:3], 0 offen", 8))
    insns.append(("s_endpgm", 4))
    return insns


class DeadSccLivenessTests(unittest.TestCase):
    def test_dead_scc_mask_ops_elide_votes_while_live_scc_is_exact(self):
        # First s_and_saveexec_b64 has DEAD SCC (immediately overwritten by second s_and_saveexec_b64);
        # second s_and_saveexec_b64 has LIVE SCC (read by s_cselect_b32), and only lane 50 is active!
        insns = [
            ("v_cmp_lt_u32_e64 s[6:7], 20, v0", 8),
            ("s_and_saveexec_b64 s[8:9], s[6:7]", 4),      # DEAD SCC -> no vote!
            ("s_or_b64 exec, exec, s[8:9]", 4),            # DEAD SCC -> no vote!
            ("v_cmp_eq_u32_e64 s[10:11], 50, v0", 8),
            ("s_and_saveexec_b64 s[12:13], s[10:11]", 4),  # LIVE SCC -> must vote across all 64 lanes!
            ("s_cselect_b32 s14, 111, 222", 4),
            ("s_mov_b64 exec, -1", 4),
            ("v_mov_b32_e32 v5, s14", 4),
            ("v_lshlrev_b32_e32 v6, 2, v0", 4),
            ("buffer_store_dword v5, v6, s[0:3], 0 offen", 8),
            ("s_endpgm", 4),
        ]
        spv, gen = _translate_encoded("cs_scc_live_dead", insns, (64, 1, 1), emulate=True)
        # Only ONE mask-op SCC vote remains (1 OpAtomicOr, not 3).
        self.assertEqual(spv.count("OpAtomicOr"), 1)
        self.assertEqual(len(gen.x_segs), 1)
        mem, _ = run_emulated_workgroup(spv, (64, 1, 1), push_constants=VSHARP + [0] * 28)
        for lane in range(64):
            self.assertEqual(mem[lane * 4], 111, f"lane {lane} missed live wave64 SCC from lane 50")

    def test_cross_block_live_scc_preserved(self):
        # SCC written in block 0 by s_and_saveexec_b64, read in block 1 by s_cbranch_scc0.
        insns = [
            ("s_mov_b32 s14, 0", 4),
            ("v_cmp_eq_u32_e64 s[6:7], 45, v0", 8),
            ("s_and_saveexec_b64 s[8:9], s[6:7]", 4),      # LIVE SCC across branch!
            ("s_cbranch_scc0 1", 4),                       # not taken because lane 45 is active
            ("s_mov_b32 s14, 77", 4),
            ("s_mov_b64 exec, -1", 4),
            ("v_mov_b32_e32 v5, s14", 4),
            ("v_lshlrev_b32_e32 v6, 2, v0", 4),
            ("buffer_store_dword v5, v6, s[0:3], 0 offen", 8),
            ("s_endpgm", 4),
        ]
        spv, _ = _translate_encoded("cs_scc_xblock", insns, (64, 1, 1), emulate=True)
        self.assertEqual(spv.count("OpAtomicOr"), 1)
        mem, _ = run_emulated_workgroup(spv, (64, 1, 1), push_constants=VSHARP + [0] * 28)
        for lane in range(64):
            self.assertEqual(mem[lane * 4], 77, f"lane {lane}: expected 77 from cross-block live SCC")

    def test_non_emulated_unchanged_keeps_wqm_scc_vote(self):
        # Verify WAVE64_EMULATE=False keeps dead-SCC votes (required by test_translate.py WQM_MASK).
        insns = [
            ("s_mov_b64 exec, -1", 4),
            ("v_cmp_gt_u32_e32 vcc, 16, v0", 4),
            ("s_wqm_b64 s[6:7], vcc", 4),
            ("s_bcnt1_i32_b64 s8, s[6:7]", 4),
            ("s_endpgm", 4),
        ]
        spv_native, _ = _translate_encoded("cs_wqm_native_keep", insns, (64, 1, 1), emulate=False)
        self.assertIn("OpGroupNonUniformAny", spv_native)
        spv_emu, _ = _translate_encoded("cs_wqm_emu_dead_scc", insns, (64, 1, 1), emulate=True)
        self.assertNotIn("OpGroupNonUniform", spv_emu)
        # Only s_bcnt1's wave_mask uses OpAtomicOr (2 words); s_wqm's dead SCC vote is elided.
        self.assertEqual(spv_emu.count("OpAtomicOr"), 2)


class LocalExecFoldTests(unittest.TestCase):
    def test_screenprobes_chain_64_and_128_threads_bit_exact(self):
        # 16 sequential ScreenProbes-shaped if/else steps with buffer loads + final store.
        insns = _screenprobes_insns(16, with_loads=True)
        init_mem = {i * 4: ((i * 17 + 3) & 0xFF) for i in range(128)}
        for wg in ((64, 1, 1), (128, 1, 1)):
            with self.subTest(wg=wg):
                spv, gen = _translate_encoded(f"cs_sp_{wg[0]}", insns, wg, emulate=True)
                # All 16*4=64 blocks collapse into 1 block + 1 memory-order split before the final store!
                self.assertEqual(_switch_case_count(gen), 2)
                self.assertEqual(spv.count("OpAtomicOr"), 0)
                self.assertNotIn("OpGroupNonUniform", spv)
                mem, _ = run_emulated_workgroup(spv, wg, push_constants=VSHARP + [0] * 28, memory=dict(init_mem))
                for lane in range(wg[0]):
                    v = lane
                    ld = init_mem[lane * 4]
                    for _ in range(16):
                        if 16 < v:
                            v = ((v + ld) & MASK32) ^ 7
                        else:
                            v = ((v + 3) & MASK32) ^ 5
                    self.assertEqual(mem[1024 + lane * 4], v, f"wg={wg} lane={lane}")

    def test_nested_exec_and_inner_saveexec_fold_exact(self):
        # Outer s_cbranch_execz contains a nested s_and_saveexec_b64 + s_cbranch_execz + s_or_b64 restore.
        insns = [
            ("v_mov_b32_e32 v4, v0", 4),
            ("v_cmp_lt_u32_e64 s[6:7], 15, v0", 8),        # lanes 16..63
            ("s_and_saveexec_b64 s[8:9], s[6:7]", 4),
            ("s_cbranch_execz 7", 4),                      # outer skip -> restore
            ("v_add_nc_u32_e32 v4, 100, v4", 4),
            ("v_cmp_lt_u32_e64 s[10:11], 40, v0", 8),      # inner: lanes 41..63
            ("s_and_saveexec_b64 s[12:13], s[10:11]", 4),
            ("s_cbranch_execz 1", 4),                      # inner skip -> inner restore
            ("v_add_nc_u32_e32 v4, 1000, v4", 4),
            ("s_or_b64 exec, exec, s[12:13]", 4),          # inner restore (from zero_pair s[12:13])
            ("v_add_nc_u32_e32 v4, 10, v4", 4),
            ("s_or_b64 exec, exec, s[8:9]", 4),            # outer restore
            ("v_lshlrev_b32_e32 v9, 2, v0", 4),
            ("buffer_store_dword v4, v9, s[0:3], 0 offen", 8),
            ("s_endpgm", 4),
        ]
        spv, gen = _translate_encoded("cs_nested_fold", insns, (64, 1, 1), emulate=True)
        self.assertEqual(_switch_case_count(gen), 1)
        self.assertEqual(spv.count("OpAtomicOr"), 0)
        mem, _ = run_emulated_workgroup(spv, (64, 1, 1), push_constants=VSHARP + [0] * 28)
        for lane in range(64):
            want = lane
            if lane > 15:
                want += 100
                if lane > 40:
                    want += 1000
                want += 10
            self.assertEqual(mem[lane * 4], want, f"lane {lane}")

    def test_negative_live_out_scalar_not_folded(self):
        # Like cs_w64x_branch: s9 is written inside s_cbranch_execz and read after join -> MUST NOT fold!
        insns = [
            ("s_mov_b32 s9, 0", 4),
            ("s_mov_b32 exec_lo, 0", 4),
            ("s_mov_b32 exec_hi, 0xff", 4),                # lanes 32..39 active; lane 0 inactive
            ("s_cbranch_execz 1", 4),                      # must vote across all 64 lanes!
            ("s_mov_b32 s9, 5", 4),
            ("s_mov_b64 exec, -1", 4),
            ("v_mov_b32_e32 v5, s9", 4),
            ("v_lshlrev_b32_e32 v6, 2, v0", 4),
            ("buffer_store_dword v5, v6, s[0:3], 0 offen", 8),
            ("s_endpgm", 4),
        ]
        spv, _ = _translate_encoded("cs_neg_live_scalar", insns, (64, 1, 1), emulate=True)
        self.assertEqual(spv.count("OpAtomicOr"), 1)
        mem, _ = run_emulated_workgroup(spv, (64, 1, 1), push_constants=VSHARP + [0] * 28)
        for lane in range(64):
            self.assertEqual(mem[lane * 4], 5, f"lane {lane}: stale scalar s9 after s_cbranch_execz")

    def test_negative_store_and_crosslane_and_exec_widen_not_folded(self):
        cases = [
            ("store", [
                ("s_and_saveexec_b64 s[8:9], vcc", 4),
                ("s_cbranch_execz 2", 4),
                ("buffer_store_dword v0, v0, s[0:3], 0 offen", 8),
                ("s_or_b64 exec, exec, s[8:9]", 4),
                ("s_endpgm", 4),
            ]),
            ("crosslane", [
                ("s_and_saveexec_b64 s[8:9], vcc", 4),
                ("s_cbranch_execz 1", 4),
                ("v_readlane_b32 s10, v0, 40", 4),
                ("s_or_b64 exec, exec, s[8:9]", 4),
                ("s_endpgm", 4),
            ]),
            ("exec_widen", [
                ("s_and_saveexec_b64 s[8:9], vcc", 4),
                ("s_cbranch_execz 2", 4),
                ("s_mov_b64 exec, -1", 4),                 # widens EXEC! must not fold
                ("v_add_nc_u32_e32 v4, 1, v4", 4),
                ("s_or_b64 exec, exec, s[8:9]", 4),
                ("s_endpgm", 4),
            ]),
        ]
        for label, insns in cases:
            with self.subTest(case=label):
                spv, _ = _translate_encoded(f"cs_neg_{label}", insns, (64, 1, 1), emulate=True)
                self.assertGreaterEqual(spv.count("OpAtomicOr"), 1, f"{label} should NOT fold s_cbranch_execz")

    def test_write_before_folded_read_orders_across_halves(self):
        # Every lane stores `100 + lane` at `lane * 4`; then ONLY lanes >= 32 enter a folded s_cbranch_execz
        # and read partner lane `(lane ^ 32) * 4`. The uniform x_split before `if (%active)` must order the write!
        insns = [
            ("v_lshlrev_b32_e32 v7, 2, v0", 4),
            ("v_add_nc_u32_e32 v6, 100, v0", 4),
            ("buffer_store_dword v6, v7, s[0:3], 0 offen", 8),
            ("v_mov_b32_e32 v9, 0", 4),
            ("v_cmp_le_u32_e64 s[6:7], 32, v0", 8),        # lanes 32..63 only
            ("s_and_saveexec_b64 s[8:9], s[6:7]", 4),
            ("s_cbranch_execz 4", 4),
            ("v_xor_b32_e32 v8, 32, v0", 4),
            ("v_lshlrev_b32_e32 v8, 2, v8", 4),
            ("buffer_load_dword v9, v8, s[0:3], 0 offen", 8),
            ("s_or_b64 exec, exec, s[8:9]", 4),
            ("v_add_nc_u32_e32 v10, 512, v7", 4),
            ("buffer_store_dword v9, v10, s[0:3], 0 offen", 8),
            ("s_endpgm", 4),
        ]
        spv, gen = _translate_encoded("cs_raw_before_fold", insns, (64, 1, 1), emulate=True)
        # Folded (0 OpAtomicOr), with 2 uniform memory-order splits: write->read and read->write.
        self.assertEqual(spv.count("OpAtomicOr"), 0)
        self.assertEqual(len(gen.x_segs), 2)
        mem, _ = run_emulated_workgroup(spv, (64, 1, 1), push_constants=VSHARP + [0] * 28)
        for lane in range(64):
            want = (100 + (lane ^ 32)) if lane >= 32 else 0
            self.assertEqual(mem[512 + lane * 4], want, f"lane {lane}")

    def test_multiwave_128_folded_exec_stays_synchronized_for_later_readlane_and_barrier(self):
        # 128 threads (2 waves): every thread runs a folded s_cbranch_execz, followed by v_readlane(lane 40)
        # and an s_barrier + LDS read from the other wave. Proves zero round desynchronization across waves!
        insns = [
            ("v_mov_b32_e32 v4, v0", 4),
            ("v_and_b32_e32 v2, 1, v0", 4),
            ("v_cmp_eq_u32_e64 s[6:7], 1, v2", 8),         # odd lanes only
            ("s_and_saveexec_b64 s[8:9], s[6:7]", 4),
            ("s_cbranch_execz 1", 4),                      # folded!
            ("v_add_nc_u32_e32 v4, 1000, v4", 4),
            ("s_or_b64 exec, exec, s[8:9]", 4),
            ("v_readlane_b32 s10, v4, 41", 4),             # cross-half readlane in each wave (odd lane -> +1000)
            ("v_lshlrev_b32_e32 v5, 2, v0", 4),
            ("ds_write_b32 v5, v4", 8),
            ("s_barrier", 4),
            ("v_xor_b32_e32 v6, 64, v0", 4),               # read partner thread in the OTHER wave!
            ("v_lshlrev_b32_e32 v6, 2, v6", 4),
            ("ds_read_b32 v7, v6", 8),
            ("v_add_nc_u32_e32 v8, s10, v7", 4),
            ("buffer_store_dword v8, v5, s[0:3], 0 offen", 8),
            ("s_endpgm", 4),
        ]
        spv, _ = _translate_encoded("cs_mw128_sync", insns, (128, 1, 1), emulate=True)
        self.assertEqual(spv.count("OpAtomicOr"), 0)
        mem, barriers = run_emulated_workgroup(spv, (128, 1, 1), push_constants=VSHARP + [0] * 28)
        self.assertGreaterEqual(barriers, 3)
        for tid in range(128):
            wave_base = (tid // 64) * 64
            s10_want = wave_base + 41 + 1000
            partner = tid ^ 64
            v7_want = partner + (1000 if (partner & 1) else 0)
            self.assertEqual(mem[tid * 4], s10_want + v7_want, f"tid {tid}")

    def test_folded_region_with_internal_loop_exact(self):
        # Multi-block folded s_cbranch_execz containing an internal scalar-counted loop (`sub_tree` loop path).
        # Lanes >= 32 add 5 four times (+20); lanes < 32 skip the entire loop. s15 is dead at the join.
        insns = [
            ("v_mov_b32_e32 v4, v0", 4),
            ("v_cmp_le_u32_e64 s[6:7], 32, v0", 8),        # lanes 32..63
            ("s_and_saveexec_b64 s[8:9], s[6:7]", 4),
            ("s_cbranch_execz 5", 4),                      # skip 5 dwords (5 x 4B insns) to restore
            ("s_mov_b32 s15, 4", 4),
            ("v_add_nc_u32_e32 v4, 5, v4", 4),             # loop header (pc +4)
            ("s_sub_u32 s15, s15, 1", 4),
            ("s_cmp_gt_u32 s15, 0", 4),
            ("s_cbranch_scc1 -4", 4),                      # back 4 dwords to v_add_nc_u32
            ("s_or_b64 exec, exec, s[8:9]", 4),
            ("v_lshlrev_b32_e32 v9, 2, v0", 4),
            ("buffer_store_dword v4, v9, s[0:3], 0 offen", 8),
            ("s_endpgm", 4),
        ]
        spv, gen = _translate_encoded("cs_fold_loop", insns, (64, 1, 1), emulate=True)
        self.assertEqual(_switch_case_count(gen), 1)
        self.assertEqual(spv.count("OpAtomicOr"), 0)
        self.assertGreaterEqual(spv.count("OpLoopMerge"), 2)  # outer scheduler loop + inner folded loop
        mem, _ = run_emulated_workgroup(spv, (64, 1, 1), push_constants=VSHARP + [0] * 28)
        for lane in range(64):
            want = lane + (20 if lane >= 32 else 0)
            self.assertEqual(mem[lane * 4], want, f"lane {lane}")

    def test_cs_w64x_saveexec_b64_fixture_emulated_exact(self):
        # Run the repo's real cs_w64x_saveexec_b64.s fixture under --wave64-emulate on the 64-lane barrier interpreter.
        fixture = os.path.join(HERE, "fixtures", "disasm", "cs_w64x_saveexec_b64.s")
        if not os.path.isfile(fixture):
            self.skipTest("cs_w64x_saveexec_b64.s fixture absent")
        prev_emu = rdna2spirv.WAVE64_EMULATE
        rdna2spirv.WAVE64_EMULATE = True
        try:
            with tempfile.TemporaryDirectory() as td:
                os.mkdir(os.path.join(td, "disasm"))
                os.mkdir(os.path.join(td, "fssm"))
                s_path = os.path.join(td, "disasm", "cs_w64x_saveexec_b64.s")
                f_path = os.path.join(td, "fssm", "cs_w64x_saveexec_b64.fssm")
                with open(fixture, encoding="utf-8") as src, open(s_path, "w", encoding="utf-8") as dst:
                    dst.write(src.read())
                with open(f_path, "wb") as f:
                    f.write(bytes(0x138) + struct.pack("<4I", 0x8088, 64, 1, 1))
                spv = rdna2spirv.translate(s_path)
        finally:
            rdna2spirv.WAVE64_EMULATE = prev_emu
        mem, _ = run_emulated_workgroup(spv, (64, 1, 1), push_constants=VSHARP + [0] * 28)
        got = [mem.get(i * 4, 0) for i in range(8)]
        self.assertEqual(got, [6160, 12304, 5120, 2088, 6168, 8192, 7176, 1837171])

    def test_write_before_folded_scalar_load_orders_cleanly(self):
        # #1587 interaction: buffer_store before a folded s_cbranch_execz containing s_load_dword must emit
        # a uniform x_split BEFORE `if (%active)`, never inside the divergent branch.
        insns = [
            ("s_mov_b32 s8, 0", 4),
            ("s_mov_b32 s9, 0", 4),
            ("v_lshlrev_b32_e32 v7, 2, v0", 4),
            ("v_add_nc_u32_e32 v6, 200, v0", 4),
            ("buffer_store_dword v6, v7, s[0:3], 0 offen", 8),
            ("v_mov_b32_e32 v9, 0", 4),
            ("v_cmp_le_u32_e64 s[6:7], 32, v0", 8),
            ("s_and_saveexec_b64 s[10:11], s[6:7]", 4),
            ("s_cbranch_execz 3", 4),
            ("s_load_dword s14, s[8:9], 0x0", 8),          # s14 dead at join; reads mem written by lane 0!
            ("v_mov_b32_e32 v9, s14", 4),
            ("s_or_b64 exec, exec, s[10:11]", 4),
            ("v_add_nc_u32_e32 v10, 512, v7", 4),
            ("buffer_store_dword v9, v10, s[0:3], 0 offen", 8),
            ("s_endpgm", 4),
        ]
        spv, gen = _translate_encoded("cs_raw_sload_fold", insns, (64, 1, 1), emulate=True)
        self.assertEqual(spv.count("OpAtomicOr"), 0)
        self.assertEqual(len(gen.x_segs), 2)               # write->s_load split before fold + s_load->store after
        mem, _ = run_emulated_workgroup(spv, (64, 1, 1), push_constants=VSHARP + [0] * 28)
        for lane in range(64):
            want = 200 if lane >= 32 else 0
            self.assertEqual(mem[512 + lane * 4], want, f"lane {lane}")


if __name__ == "__main__":
    unittest.main()
