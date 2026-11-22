"""Exact CFG/dataflow optimizations for --wave64-emulate (Issue #884).

Two conservative, bit-exact analyses over a compute/mesh shader's basic blocks:

1. `scc_live_outs(blocks, succ)` -> set of `id(ins)` whose SCC write can reach
   any SCC read before being unconditionally overwritten. Under --wave64-emulate,
   mask ops (`s_and_saveexec_b64`, `s_andn2_b64`, `s_or_b64`, `s_not_b64`,
   `s_wqm_b64`, ...) otherwise pay a full cross-lane `x_round` + `x_split`
   (workgroup barrier + OpSwitch case) to compute SCC even when SCC is dead.

2. `find_local_exec_folds(blocks, raw_edges, succ, lane_masks, scc_live)` ->
   `(folds, fold_chain, absorbed)`: identifies single-entry, single-exit
   regions guarded by `s_cbranch_execz` / `s_cbranch_execnz` whose body has:
     - no cross-lane operations, no `s_barrier`, no live-SCC mask votes,
     - no memory or LDS writes/atomics and no unmasked side effects,
     - no EXEC-widening writes (an inactive lane stays inactive across the region),
     - no scalar (`SGPR`, `VCC`, `SCC`, `M0`) written in the region that is live
       at the join block (`skip`),
     - reducible internal control flow (`cfg_structure.structure` succeeds).

   Each such region can be emitted inline as a structured `if (%active) { R }`
   inside its guarding block with zero `x_round` / `x_split` / `OpControlBarrier`,
   keeping all 64 lanes of the wave in the exact same scheduler round on arrival
   at `skip`. When `skip` has no predecessors outside `{B} | R`, `skip` is also
   chained inline after the fold, collapsing chains of thousands of unrolled
   `s_and_saveexec_b64; s_cbranch_execz` diamonds into a single scheduler block.

Standard library only.
"""
import collections
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
import cfg_structure  # noqa: E402
import lane_masks as _lane_masks  # noqa: E402

BRANCHES = frozenset({
    "s_branch", "s_cbranch_scc0", "s_cbranch_scc1",
    "s_cbranch_vccz", "s_cbranch_vccnz",
    "s_cbranch_execz", "s_cbranch_execnz",
})

# Instructions that implicitly read SCC (in addition to explicit `scc` operands).
_SCC_IMPLICIT_READS = frozenset({
    "s_cbranch_scc0", "s_cbranch_scc1",
    "s_cselect_b32", "s_cselect_b64",
    "s_cmov_b32", "s_cmov_b64",
    "s_addc_u32", "s_subb_u32",
})

# Instructions verified in tools/rdna2spirv.py to unconditionally overwrite %scc.
# Conservatively omitting an op only keeps SCC live across it (never unsafe).
_SAVEEXEC_KINDS = ("and", "or", "xor", "andn1", "andn2", "orn1", "orn2", "nand", "nor", "xnor")
_ALL_SAVEEXEC_WREXEC = frozenset(
    [f"s_{k}_saveexec_b{w}" for k in _SAVEEXEC_KINDS for w in (32, 64)]
    + [f"s_{k}_wrexec_b{w}" for k in ("andn1", "andn2") for w in (32, 64)]
)

_SCC_UNCONDITIONAL_WRITES = _ALL_SAVEEXEC_WREXEC | frozenset({
    "s_and_b64", "s_or_b64", "s_xor_b64", "s_andn2_b64", "s_orn2_b64",
    "s_nand_b64", "s_nor_b64", "s_xnor_b64", "s_not_b64",
    "s_and_b32", "s_or_b32", "s_xor_b32", "s_andn2_b32", "s_orn2_b32",
    "s_nand_b32", "s_nor_b32", "s_xnor_b32", "s_not_b32",
    "s_add_u32", "s_addc_u32", "s_sub_u32", "s_subb_u32",
    "s_add_i32", "s_sub_i32", "s_addk_i32",
    "s_lshl1_add_u32", "s_lshl2_add_u32", "s_lshl3_add_u32", "s_lshl4_add_u32",
    "s_min_u32", "s_max_u32", "s_min_i32", "s_max_i32",
    "s_abs_i32", "s_absdiff_i32",
    "s_lshl_b32", "s_lshr_b32", "s_ashr_i32", "s_lshl_b64",
    "s_bfe_u32", "s_bfe_i32", "s_bfe_u64", "s_bfe_i64",
    "s_bcnt0_i32_b32", "s_bcnt1_i32_b32", "s_bcnt0_i32_b64", "s_bcnt1_i32_b64",
    "s_wqm_b32", "s_wqm_b64", "s_quadmask_b32", "s_quadmask_b64",
    "s_cmp_eq_u32", "s_cmp_lg_u32", "s_cmp_gt_u32", "s_cmp_ge_u32", "s_cmp_lt_u32", "s_cmp_le_u32",
    "s_cmp_eq_i32", "s_cmp_lg_i32", "s_cmp_gt_i32", "s_cmp_ge_i32", "s_cmp_lt_i32", "s_cmp_le_i32",
    "s_cmp_eq_u64", "s_cmp_lg_u64",
    "s_bitcmp0_b32", "s_bitcmp1_b32", "s_bitcmp0_b64", "s_bitcmp1_b64",
})

# Mask ops that emit a cross-lane vote in --wave64-emulate when SCC is live.
_MASK_SCC_VOTE_OPS = _ALL_SAVEEXEC_WREXEC | frozenset({
    "s_and_b64", "s_or_b64", "s_xor_b64", "s_andn2_b64", "s_orn2_b64",
    "s_nand_b64", "s_nor_b64", "s_xnor_b64", "s_not_b64",
    "s_wqm_b32", "s_wqm_b64",
})

# Scalar register bit positions for fast integer-bitmask backward liveness.
_BIT_VCC = 128
_BIT_SCC = 129
_BIT_M0 = 130
_ALL_SGPRS = (1 << 128) - 1

_IMPLICIT_VCC_READS = frozenset({
    "v_cndmask_b32", "v_addc_co_u32", "v_subb_co_u32", "v_subbrev_co_u32",
    "v_add_co_ci_u32", "v_sub_co_ci_u32", "v_subrev_co_ci_u32",
    "v_addc_u32", "v_subb_u32", "v_div_fmas_f32", "v_div_fmas_f64",
    "s_cbranch_vccz", "s_cbranch_vccnz",
})

_NO_DEST_PREFIXES = (
    "s_cmp", "s_bitcmp", "v_cmpx", "exp",
    "buffer_store", "global_store", "flat_store", "scratch_store", "tbuffer_store", "image_store",
    "ds_write", "ds_store", "s_store", "s_buffer_store",
    "s_cbranch", "s_branch", "s_waitcnt", "s_nop", "s_endpgm", "s_barrier", "s_sendmsg",
    "s_setprio", "s_inst_prefetch", "s_clause", "s_trap", "s_icache", "s_dcache", "s_sleep",
    "s_code_end", "s_denorm_mode", "s_round_mode", "s_ttracedata", "v_nop", "v_clrexcp",
    "v_pipeflush", "s_set", "s_sethalt", "s_setkill", "s_wakeup", "s_version",
    "buffer_gl", "buffer_wbinvl", "ds_gws", "ds_nop",
)


def base_op(op):
    for suffix in ("_e32", "_e64", "_sdwa", "_dpp"):
        if op.endswith(suffix):
            return op[:-len(suffix)]
    return op


def _active_insns(blk):
    """Instructions of a basic block up to and including its terminator (branch or s_endpgm)."""
    out = []
    for ins in blk:
        out.append(ins)
        op = base_op(ins["op"])
        if op == "s_endpgm" or op in BRANCHES:
            break
    return out


def reads_scc(ins):
    op = base_op(ins["op"])
    if op in _SCC_IMPLICIT_READS or op.startswith(("s_getreg", "s_setreg", "s_save_status")):
        return True
    for x in ins["args"]:
        if x.get("k") == "special" and x.get("t") == "scc":
            return True
    return False


def writes_scc(ins):
    return base_op(ins["op"]) in _SCC_UNCONDITIONAL_WRITES


def scc_live_outs(blocks, succ):
    """Return the set of `id(ins)` whose SCC result may be read before being unconditionally overwritten."""
    n = len(blocks)
    if not n:
        return set()
    preds = [[] for _ in range(n)]
    for b in range(n):
        for s in succ[b]:
            preds[s].append(b)
    use = [False] * n
    kill = [False] * n
    active = [_active_insns(blk) for blk in blocks]
    for b, insns in enumerate(active):
        k = False
        u = False
        for ins in insns:
            if not k and reads_scc(ins):
                u = True
                break
            if writes_scc(ins):
                k = True
        use[b] = u
        kill[b] = k
    live_in = list(use)
    live_out = [False] * n
    work = [b for b in range(n) if live_in[b]]
    while work:
        b = work.pop()
        for p in preds[b]:
            if not live_out[p]:
                live_out[p] = True
                if not kill[p] and not live_in[p]:
                    live_in[p] = True
                    work.append(p)
    out = set()
    for b, insns in enumerate(active):
        live = live_out[b]
        for ins in reversed(insns):
            if live:
                out.add(id(ins))
            live = (live and not writes_scc(ins)) or reads_scc(ins)
    return out


def _operand_scalar_mask(x):
    """Bitmask of scalar registers (s0..s127, vcc, scc, m0) named by operand `x`."""
    k = x.get("k")
    if k == "sgpr":
        r = x["r"]
        cnt = x.get("n", 1)
        m = 0
        for i in range(cnt):
            if 0 <= r + i < 128:
                m |= 1 << (r + i)
        return m
    if k == "special":
        t = x.get("t", "")
        if t.startswith("vcc"):
            return 1 << _BIT_VCC
        if t == "scc":
            return 1 << _BIT_SCC
        if t == "m0":
            return 1 << _BIT_M0
    return 0


def _scalar_rw(ins):
    """Return `(read_mask, may_write_mask, must_kill_mask)` over scalar bits 0..130 for `ins`."""
    op = base_op(ins["op"])
    a = ins["args"]
    mods = ins.get("mods") or ()
    # Dynamic SGPR indexing reads/writes arbitrary SGPRs + m0: never kill, conservatively read/write all SGPRs.
    if op.startswith("s_movrel"):
        return _ALL_SGPRS | (1 << _BIT_M0), _ALL_SGPRS, 0
    if op == "s_endpgm" or op in BRANCHES:
        rd = 0
        if op in ("s_cbranch_scc0", "s_cbranch_scc1"):
            rd |= 1 << _BIT_SCC
        elif op in ("s_cbranch_vccz", "s_cbranch_vccnz"):
            rd |= 1 << _BIT_VCC
        return rd, 0, 0
    # Split destination vs source operands conservatively.
    dests, srcs = [], list(a)
    if op.startswith(("s_cmp", "s_bitcmp")) or op.startswith("v_cmpx") or op.startswith(_NO_DEST_PREFIXES):
        dests, srcs = [], list(a)
    elif op.startswith("v_cmp"):
        if len(a) == 2:
            dests, srcs = [], list(a)
        elif a:
            dests, srcs = [a[0]], list(a[1:])
    elif op.startswith(("buffer_atomic", "global_atomic", "flat_atomic", "image_atomic")):
        dests = [a[0]] if ("glc" in mods and a) else []
        srcs = list(a)
    elif op.startswith("ds_"):
        returns = (op.startswith(("ds_read", "ds_bpermute", "ds_permute", "ds_swizzle", "ds_consume", "ds_append"))
                   or "_rtn" in op)
        dests, srcs = ([a[0]], list(a[1:])) if (returns and a) else ([], list(a))
    elif op in ("s_addk_i32", "s_mulk_i32", "s_cmovk_i32", "s_movk_i32") and a:
        # SOPK reads and writes a[0] (except s_movk_i32 which only writes a[0]).
        dests = [a[0]]
        srcs = list(a[1:]) if op == "s_movk_i32" else list(a)
    elif a:
        dests, srcs = [a[0]], list(a[1:])
        if op.startswith("v_") and len(a) > 2 and (
                "_co_" in op or op in ("v_mad_u64_u32", "v_mad_i64_i32", "v_div_scale_f32", "v_div_scale_f64")) \
                and a[1].get("k") in ("sgpr", "special"):
            dests, srcs = [a[0], a[1]], list(a[2:])

    rd = 0
    for x in srcs:
        rd |= _operand_scalar_mask(x)
    if op in _IMPLICIT_VCC_READS:
        rd |= 1 << _BIT_VCC
    if reads_scc(ins):
        rd |= 1 << _BIT_SCC
    if op.startswith(("ds_", "s_sendmsg", "v_interp")):
        rd |= 1 << _BIT_M0
    if op == "s_cmov_b32" or op == "s_cmov_b64":
        # Conditional move preserves destination when SCC == 0, so it also reads destination.
        for d in dests:
            rd |= _operand_scalar_mask(d)

    may_w = 0
    for d in dests:
        may_w |= _operand_scalar_mask(d)
    if op.startswith("v_cmp") and not op.startswith("v_cmpx") and len(a) == 2:
        may_w |= 1 << _BIT_VCC
    if writes_scc(ins):
        may_w |= 1 << _BIT_SCC
    elif op.startswith("s_") and not op.startswith(("s_mov", "s_cselect", "s_load", "s_buffer_load", "s_getpc",
                                                    "s_nop", "s_waitcnt", "s_clause", "s_inst_prefetch", "s_pack",
                                                    "s_mul_i32", "s_mul_hi", "s_mulk", "s_ff", "s_flbit", "s_bfm",
                                                    "s_get_waveid")):
        # Any uncategorized scalar op could touch SCC: include in may_w (never in must_kill).
        may_w |= 1 << _BIT_SCC

    # Only kill registers unconditionally overwritten (never conditional moves or half-VCC writes).
    must_kill = 0
    if op not in ("s_cmov_b32", "s_cmov_b64", "s_cmovk_i32"):
        for d in dests:
            if d.get("k") == "special" and d.get("t") in ("vcc_lo", "vcc_hi"):
                continue  # half-write preserves the other half in wave64
            must_kill |= _operand_scalar_mask(d)
        if op.startswith("v_cmp") and not op.startswith("v_cmpx") and len(a) == 2:
            must_kill |= 1 << _BIT_VCC
    if writes_scc(ins):
        must_kill |= 1 << _BIT_SCC
    return rd, may_w, must_kill


def _scalar_liveness(active_blocks, succ):
    """Exact conservative backward scalar liveness `live_in[b]` (int bitmask) for each block."""
    n = len(active_blocks)
    preds = [[] for _ in range(n)]
    for b in range(n):
        for s in succ[b]:
            preds[s].append(b)
    use = [0] * n
    kill = [0] * n
    for b, insns in enumerate(active_blocks):
        u = 0
        k = 0
        for ins in insns:
            rd, _, mk = _scalar_rw(ins)
            u |= rd & ~k
            k |= mk
        use[b] = u
        kill[b] = k
    live_in = list(use)
    live_out = [0] * n
    in_work = [bool(u) for u in live_in]
    work = collections.deque(b for b in range(n) if in_work[b])
    while work:
        b = work.popleft()
        in_work[b] = False
        bi = live_in[b]
        for p in preds[b]:
            new_out = live_out[p] | bi
            if new_out != live_out[p]:
                live_out[p] = new_out
                new_in = use[p] | (new_out & ~kill[p])
                if new_in != live_in[p]:
                    live_in[p] = new_in
                    if not in_work[p]:
                        in_work[p] = True
                        work.append(p)
    return live_in


def _postdominators(succ):
    """Immediate post-dominator `ipdom[b]` (Cooper-Harvey-Kennedy on reversed CFG with virtual exit `n`)."""
    n = len(succ)
    exit_ = n
    rev = [[] for _ in range(n + 1)]
    for b in range(n):
        if succ[b]:
            for s in succ[b]:
                rev[s].append(b)
        else:
            rev[exit_].append(b)
    # Iterative RPO from exit_ on reversed CFG.
    state = {exit_: 1}
    order = []
    stack = [(exit_, iter(rev[exit_]))]
    while stack:
        v, it = stack[-1]
        for w in it:
            if w not in state:
                state[w] = 1
                stack.append((w, iter(rev[w])))
                break
        else:
            state[v] = 2
            order.append(v)
            stack.pop()
    rpo = order[::-1]
    pos = {v: i for i, v in enumerate(rpo)}
    ipdom = {exit_: exit_}
    # Original successors (with [] -> [exit_]) are the predecessors in the reversed CFG.
    orig_succ = [list(s) if s else [exit_] for s in succ] + [[]]
    changed = True
    while changed:
        changed = False
        for v in rpo[1:]:
            new = None
            for p in orig_succ[v]:
                if p not in ipdom:
                    continue
                if new is None:
                    new = p
                    continue
                a, b = p, new
                while a != b:
                    while pos[a] > pos[b]:
                        a = ipdom[a]
                    while pos[b] > pos[a]:
                        b = ipdom[b]
                new = a
            if ipdom.get(v) != new:
                ipdom[v] = new
                changed = True
    return ipdom


def _holds_lane_mask(ins, x, lm_map):
    k = x.get("k")
    if k == "special":
        t = x.get("t", "")
        return t.startswith("exec") or (t.startswith("vcc") and "vcc" in lm_map)
    if k == "sgpr":
        return all(x["r"] + i in lm_map for i in range(x.get("n", 1)))
    return False


def _is_exec_op_arg(x):
    return x.get("k") == "special" and x.get("t", "") == "exec"


def _sgpr_pair(x):
    if x.get("k") == "sgpr" and x.get("n", 1) == 2:
        return x["r"]
    return None


def _classify_block_body(insns, lane_masks, scc_live, mask_writes):
    """Inspect non-terminator instructions of a block for lane-local region safety.

    Returns `(clean, mem_reads, may_write_scalars, exec_writes, sgpr_writes_all, saveexec_pairs)` where:
      - `clean`: True if no instruction does a cross-lane op, barrier, memory write, or unmasked side effect.
      - `mem_reads`: subset of {"lds", "mem"}.
      - `may_write_scalars`: int bitmask of scalar regs possibly written.
      - `exec_writes`: list of `(op, args)` that write EXEC.
      - `sgpr_writes_all`: dict mapping sgpr index -> count of writes inside this block.
      - `saveexec_pairs`: set of base SGPR indices `r` written by `s_and_saveexec_b64 s[r:r+1], ...`.
    """
    mem_reads = set()
    may_w = 0
    exec_writes = []
    sgpr_writes = collections.Counter()
    saveexec_pairs = set()

    for ins in insns:
        raw_op = ins["op"]
        op = base_op(raw_op)
        if op == "s_endpgm" or op in BRANCHES:
            break
        if id(ins) in mask_writes:
            # write_all_lanes VGPR store (parked lane mask): runs even when EXEC == 0, so must not be skipped.
            return False, frozenset(), 0, (), sgpr_writes, saveexec_pairs
        a = ins["args"]
        lm = lane_masks.get(id(ins), {})

        # 1. Strictly forbidden opcodes inside a lane-local folded region.
        if raw_op.endswith(("_dpp", "_dpp8")):
            return False, frozenset(), 0, (), sgpr_writes, saveexec_pairs
        if op in ("s_barrier", "s_sendmsg", "s_setpc_b64", "s_swappc_b64", "s_trap",
                  "s_sethalt", "s_setkill", "s_sleep", "s_setprio",
                  "s_round_mode", "s_denorm_mode",
                  "v_readlane_b32", "v_readfirstlane_b32", "v_writelane_b32",
                  "v_permlane16_b32", "v_permlanex16_b32",
                  "ds_bpermute_b32", "ds_permute_b32", "ds_swizzle_b32",
                  "s_wqm_b32", "s_wqm_b64", "s_quadmask_b32", "s_quadmask_b64"):
            return False, frozenset(), 0, (), sgpr_writes, saveexec_pairs
        if op.startswith(("s_movrel", "s_setreg", "s_getreg", "s_store", "s_buffer_store", "s_atomic", "exp")):
            return False, frozenset(), 0, (), sgpr_writes, saveexec_pairs
        if "store" in op or "atomic" in op:
            return False, frozenset(), 0, (), sgpr_writes, saveexec_pairs
        if op.startswith("ds_"):
            if not op.startswith("ds_read"):
                return False, frozenset(), 0, (), sgpr_writes, saveexec_pairs
            mem_reads.add("lds")
        elif op.startswith(("buffer_load", "tbuffer_load", "global_load", "flat_load", "scratch_load",
                            "s_load_", "s_buffer_load_",
                            "image_load", "image_sample", "image_gather", "image_bvh")):
            mem_reads.add("mem")

        # 2. Cross-lane mask readers (`wave_mask`, `wave_equal`, live-SCC `scc_from_mask`).
        if op in ("s_bcnt0_i32_b32", "s_bcnt1_i32_b32", "s_bcnt0_i32_b64", "s_bcnt1_i32_b64",
                  "s_ff0_i32_b32", "s_ff1_i32_b32", "s_ff0_i32_b64", "s_ff1_i32_b64"):
            if len(a) > 1 and _holds_lane_mask(ins, a[1], lm):
                return False, frozenset(), 0, (), sgpr_writes, saveexec_pairs
        elif op in ("v_mbcnt_lo_u32_b32", "v_mbcnt_hi_u32_b32"):
            if len(a) > 1 and a[1].get("k") != "imm" and _holds_lane_mask(ins, a[1], lm):
                return False, frozenset(), 0, (), sgpr_writes, saveexec_pairs
        elif op.startswith(("s_cmp", "s_bitcmp")):
            if any(_holds_lane_mask(ins, x, lm) for x in a):
                return False, frozenset(), 0, (), sgpr_writes, saveexec_pairs
        elif op in _MASK_SCC_VOTE_OPS and id(ins) in scc_live:
            # Would emit scc_from_mask -> x_round if operating on a lane mask.
            is_mask_dst = bool(a) and a[0].get("k") == "special" and a[0].get("t", "").startswith(("exec", "vcc"))
            if "saveexec" in op or "wrexec" in op or is_mask_dst or any(_holds_lane_mask(ins, x, lm) for x in a[1:]):
                return False, frozenset(), 0, (), sgpr_writes, saveexec_pairs

        # 3. Track scalar writes and EXEC writes.
        _, ins_mw, _ = _scalar_rw(ins)
        may_w |= ins_mw
        # Track individual SGPR writes for exact saved-EXEC-mask checking.
        s_bits = ins_mw & _ALL_SGPRS
        while s_bits:
            lsb = s_bits & -s_bits
            sgpr_writes[lsb.bit_length() - 1] += 1
            s_bits ^= lsb
        if op == "s_and_saveexec_b64" and a:
            pair = _sgpr_pair(a[0])
            if pair is not None:
                saveexec_pairs.add(pair)
        writes_exec = (op.startswith("v_cmpx") or "saveexec" in op or "wrexec" in op
                       or (bool(a) and a[0].get("k") == "special" and a[0].get("t", "").startswith("exec")))
        if writes_exec:
            exec_writes.append((op, a))

    return True, frozenset(mem_reads), may_w, exec_writes, sgpr_writes, saveexec_pairs


def _exec_writes_stay_inactive(region_blocks, blk_info):
    """Return True iff every EXEC write in `region_blocks` preserves `active == False` for inactive lanes."""
    total_sgpr_writes = collections.Counter()
    cand_saved = set()
    for u in region_blocks:
        total_sgpr_writes.update(blk_info[u][4])
        cand_saved.update(blk_info[u][5])
    # A pair s[r:r+1] is guaranteed to hold 0 for every lane entering R inactive if its ONLY write in R
    # is `s_and_saveexec_b64 s[r:r+1], ...` (and neither half is written anywhere else in R).
    zero_pairs = {r for r in cand_saved if total_sgpr_writes[r] == 1 and total_sgpr_writes[r + 1] == 1}
    for u in region_blocks:
        for op, a in blk_info[u][3]:
            if op.startswith("v_cmpx") or op in ("s_and_saveexec_b64", "s_and_saveexec_b32"):
                continue
            if not a or not _is_exec_op_arg(a[0]):
                return False  # partial exec_lo/exec_hi or unknown EXEC write
            if op == "s_and_b64" and len(a) == 3 and (_is_exec_op_arg(a[1]) or _is_exec_op_arg(a[2])):
                continue
            if op == "s_andn2_b64" and len(a) == 3:
                if _is_exec_op_arg(a[1]):
                    continue
                if _sgpr_pair(a[1]) in zero_pairs:
                    continue
                return False
            if op == "s_or_b64" and len(a) == 3:
                if _is_exec_op_arg(a[1]) and _sgpr_pair(a[2]) in zero_pairs:
                    continue
                if _is_exec_op_arg(a[2]) and _sgpr_pair(a[1]) in zero_pairs:
                    continue
                return False
            if op == "s_mov_b64" and len(a) == 2 and _sgpr_pair(a[1]) in zero_pairs:
                continue
            return False
    return True


def find_local_exec_folds(blocks, raw_edges, succ, lane_masks, scc_live, mask_writes=frozenset()):
    """Find safe lane-local `s_cbranch_execz`/`nz` regions to fold inline under --wave64-emulate.

    Returns `(folds, fold_chain, absorbed)`:
      - `folds`: dict `head_block -> {"run": int, "skip": int, "region": frozenset[int],
                 "order": list[int], "sub_tree": list | None, "mem_reads": frozenset[str]}`
                 containing only the maximal (outermost) folded regions.
      - `fold_chain`: dict `head_block -> skip_block` when `skip_block` has no predecessors outside
                 `{head_block} | region` and can be emitted directly after the fold in the same block.
      - `absorbed`: set of all block indices emitted inline (inside a folded region or chained as `skip`).
    """
    n = len(blocks)
    if n < 2:
        return {}, {}, set()

    active_blocks = [_active_insns(blk) for blk in blocks]
    preds = [set() for _ in range(n)]
    for b in range(n):
        for s in succ[b]:
            preds[s].add(b)

    ipdom = _postdominators(succ)
    scalar_live_in = _scalar_liveness(active_blocks, succ)

    blk_info = []
    terms = []
    for b, insns in enumerate(active_blocks):
        blk_info.append(_classify_block_body(insns, lane_masks, scc_live, mask_writes))
        last = insns[-1] if insns else None
        op = base_op(last["op"]) if last is not None else None
        if op == "s_endpgm":
            terms.append(("end", None))
        elif op in BRANCHES:
            terms.append(("br", op))
        else:
            terms.append(("fall", None))

    # Gather candidate s_cbranch_execz / s_cbranch_execnz blocks with single-entry/single-exit post-dominated regions.
    candidates = []
    for b in range(n):
        kind, op = terms[b]
        if kind != "br" or op not in ("s_cbranch_execz", "s_cbranch_execnz"):
            continue
        edges = raw_edges[b]
        if len(edges) < 2:
            continue
        tgt, fall = edges[0], edges[1]
        run, skip = (fall, tgt) if op == "s_cbranch_execz" else (tgt, fall)
        if run == skip or skip >= n or ipdom.get(b) != skip:
            continue
        # Collect region R reachable from `run` stopping at `skip`.
        region = set()
        work = [run]
        ok = True
        while work:
            u = work.pop()
            if u == skip or u in region:
                continue
            if u == b or u == 0 or not succ[u] or not blk_info[u][0]:
                ok = False
                break
            region.add(u)
            for s in succ[u]:
                if s != skip and s not in region:
                    work.append(s)
        if not ok or not region:
            continue
        # Single-entry check: no external edge into any block of R except B -> run.
        for u in region:
            allowed_ext = {b} if u == run else set()
            if not preds[u] <= (region | allowed_ext):
                ok = False
                break
        if not ok:
            continue
        # No scalar written in R may be live-in at `skip`.
        region_may_w = 0
        for u in region:
            region_may_w |= blk_info[u][2]
        if region_may_w & scalar_live_in[skip]:
            continue
        # EXEC must stay inactive for any lane that skipped R.
        if not _exec_writes_stay_inactive(region, blk_info):
            continue
        candidates.append((len(region), b, run, skip, frozenset(region)))

    # Validate innermost regions first so nested s_cbranch_execz/nz inside an outer region are already verified.
    candidates.sort(key=lambda item: (item[0], item[1]))
    valid = {}
    for _, b, run, skip, region in candidates:
        ok = True
        for u in region:
            tkind, top = terms[u]
            if tkind == "end":
                ok = False
                break
            if tkind == "br":
                if top in ("s_cbranch_vccz", "s_cbranch_vccnz"):
                    ok = False
                    break
                if top in ("s_cbranch_execz", "s_cbranch_execnz"):
                    if u not in valid or valid[u]["skip"] not in (region | {skip}):
                        ok = False
                        break
        if not ok:
            continue
        # Build local sub-CFG and verify reducibility (multi-block regions use cfg_structure.structure).
        if len(region) == 1:
            order = [run]
            sub_tree = None
        else:
            order = [run] + sorted(region - {run})
            idx_of = {u: i for i, u in enumerate(order)}
            sub_succ = [[idx_of[s] for s in succ[u] if s in idx_of] for u in order]
            sub_tree = cfg_structure.structure(sub_succ)
            if sub_tree is None:
                continue
        mem_reads = frozenset().union(*(blk_info[u][1] for u in region))
        valid[b] = {
            "run": run,
            "skip": skip,
            "region": region,
            "order": order,
            "sub_tree": sub_tree,
            "mem_reads": mem_reads,
        }

    # Select maximal (non-overlapping outermost) folded regions.
    selected = {}
    in_region = set()
    for _, b, _, _, _ in sorted(
            ((len(v["region"]), k, v["run"], v["skip"], v["region"]) for k, v in valid.items()),
            key=lambda x: (-x[0], x[1])):
        fd = valid[b]
        if b in in_region or (fd["region"] & in_region):
            continue
        selected[b] = fd
        in_region.update(fd["region"])

    # Chain single-predecessor `skip` continuations directly after their fold (cycle-free by construction).
    fold_chain = {}
    chained = set()
    parent_chain = {}

    def chain_root(x):
        while x in parent_chain:
            x = parent_chain[x]
        return x

    for b in sorted(selected):
        fd = selected[b]
        skip = fd["skip"]
        if skip == 0 or skip in in_region or skip in chained:
            continue
        if not preds[skip] <= ({b} | fd["region"]):
            continue
        if chain_root(skip) == chain_root(b):
            continue
        fold_chain[b] = skip
        parent_chain[skip] = b
        chained.add(skip)

    absorbed = in_region | chained
    return selected, fold_chain, absorbed
