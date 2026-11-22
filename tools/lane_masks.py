"""Which scalar registers hold a lane mask?

On hardware a lane mask is one register for the whole wave, a bit per lane. The translator runs one invocation
per lane and keeps only that lane's bit in the register (all ones or zero), which is all that and/or/not, selects
and EXEC need. A few instructions look at the mask as a whole instead: s_bcnt1 (how many lanes), s_ff1 (the first
lane), v_mbcnt (how many lanes below this one). For those the translator has to collect the other lanes' bits.
The same instructions also run on plain 64-bit data, such as a light mask read from a buffer, and nothing in the
instruction says which of the two it has, so this module works it out from what wrote the register.

    masks, writes = analyse(blocks, succ)   # blocks: lists of instructions; succ: successor block numbers per block
    masks[id(ins)]                      # dict mapping each register holding a lane mask on every path to ins
                                        # (SGPR number, "vcc", ("lane", vgpr, lane) parked in a VGPR lane, or
                                        # ("v", vgpr) passing through a VGPR) to its kind (MASK, MASK_LO, MASK_HI)
    id(ins) in writes                   # ins computes a mask into a VGPR (a plain copy does not count)

A register counts as a mask only if every path to the instruction wrote a mask to it. Anything else is treated
as data, which is what the translator did for all of them before.

Parked masks: a shader that runs out of SGPRs moves some into single lanes of a spare VGPR (v_writelane_b32) and
fetches them back later (v_readlane_b32). When the SGPR is a saved EXEC, the value that comes back has to be this
lane's bit again, not the bit of the lane it was parked in.

Masks through a VGPR: the compiler does some scalar work in the vector unit, such as an OR of three SGPRs
(v_mov_b32 / v_or3_b32, then v_readfirstlane_b32 to get the scalar back). On hardware every lane computes the same
whole mask and the first lane's copy is read. Here each lane computes its own bit, so it has to read its own
copy, and it has to compute it whether or not it is active. A plain copy of a mask (v_mov_b32) is different:
it moves an already-computed value, so it is an ordinary VALU write and inactive lanes keep the old VGPR
(SHADER-21 part 2).
"""

MASK = "mask"           # full lane mask (64-bit pair, vcc, or wave32 single register)
MASK_LO = "mask_lo"     # low 32-bit half of a 64-bit lane mask (lanes 0-31)
MASK_HI = "mask_hi"     # high 32-bit half of a 64-bit lane mask (lanes 32-63)
MASK_KINDS = (MASK, MASK_LO, MASK_HI)
EITHER = "either"       # a constant 0 or -1: no lanes or all lanes, and the same bits when read as data
# Absent from the state: data.

COPIES = ("s_mov_b", "s_not_b", "s_wqm_b")
BITWISE = ("s_and_b", "s_or_b", "s_xor_b", "s_andn2_b", "s_orn2_b", "s_nand_b", "s_nor_b", "s_xnor_b",
           "s_cselect_b")
NO_DESTINATION = ("s_cmp", "s_bitcmp", "v_cmpx", "s_set", "s_rfe")      # their first operand is a source
VECTOR_BITWISE = ("v_mov_b32", "v_or3_b32", "v_or_b32", "v_and_b32", "v_xor_b32", "v_xor3_b32", "v_not_b32")


def base_op(op):
    for suffix in ("_e32", "_e64", "_sdwa", "_dpp"):
        if op.endswith(suffix):
            return op[:-len(suffix)]
    return op


def kind(state, a):
    if a["k"] == "special":
        t = a.get("t", "")
        if t == "exec_lo":
            return MASK_LO
        if t == "exec_hi":
            return MASK_HI
        if t == "exec":
            return MASK
        if t == "vcc":
            return state.get("vcc")
        if t in ("vcc_lo", "vcc_hi"):
            vk = state.get("vcc")
            if vk in MASK_KINDS:
                return MASK_LO if t == "vcc_lo" else MASK_HI
            return vk if t == "vcc_lo" else None
        return None
    if a["k"] == "sgpr":
        n = a.get("n", 1)
        ks = [state.get(a["r"] + i) for i in range(n)]
        if None in ks:
            return None
        if n == 2 and any(k in MASK_KINDS for k in ks):
            return MASK
        return ks[0] if len(set(ks)) == 1 else None
    if a["k"] == "vgpr" and a.get("n", 1) == 1:
        return state.get(("v", a["r"]))
    if a["k"] == "imm" and a["v"] in (0, -1):
        return EITHER
    return None


def put(state, a, k):
    if a["k"] == "sgpr":
        n = a.get("n", 1)
        if k == MASK and n == 2:
            state[a["r"]] = MASK_LO
            state[a["r"] + 1] = MASK_HI
            return
        regs = [a["r"] + i for i in range(n)]
    elif a["k"] == "slist":
        regs = [r if isinstance(r, int) else "vcc" for r in a["r"]]
    elif a["k"] == "special" and a.get("t") in ("vcc", "vcc_lo"):
        regs = ["vcc"]
    elif a["k"] == "special" and a.get("t") == "vcc_hi":
        # A separate write to vcc_hi lands in %vcc_hi, so vcc no longer holds one lane mask.
        state.pop("vcc", None)
        return
    else:
        return
    for r in regs:
        if k is None:
            state.pop(r, None)
        else:
            state[r] = k


def lane_slot(vgpr, lane):
    """State key for a mask parked in one lane of a VGPR, or None unless both operands are plain."""
    if vgpr["k"] != "vgpr" or lane["k"] != "imm":
        return None
    return ("lane", vgpr["r"], lane["v"] & 63)


def step(state, ins):
    """Update `state` (register -> MASK | MASK_LO | MASK_HI | EITHER) for one instruction. True if it writes a mask to a VGPR."""
    op, a = base_op(ins["op"]), ins["args"]
    if not a or op.startswith(NO_DESTINATION):
        return
    if op == "v_readfirstlane_b32" and len(a) == 2:                             # sD = vS, the same in every lane
        put(state, a[0], kind(state, a[1]))
        return
    if op == "v_writelane_b32" and len(a) == 3 and lane_slot(a[0], a[2]):       # vD[lane] = sS
        k = kind(state, a[1])
        if k is None:
            state.pop(lane_slot(a[0], a[2]), None)
        else:
            state[lane_slot(a[0], a[2])] = k
        return
    if op == "v_readlane_b32" and len(a) == 3:                                  # sD = vS[lane]
        put(state, a[0], state.get(lane_slot(a[1], a[2])))
        return
    if op.startswith("v_cmp"):
        # Handle v_cmp before VGPR clobber (defensive for 2-operand forms with implicit vcc).
        put(state, {"k": "special", "t": "vcc"} if len(a) == 2 else a[0], MASK)
        return
    if a[0]["k"] == "vgpr":                             # any other write to a VGPR ends what was parked in it
        # A mask only when every source is one (or a constant 0 / -1) on every path: no EITHER for VGPRs, whose
        # plain writes skip inactive lanes. The sources are looked at first: the destination can be one of them.
        kinds = [kind(state, x) for x in a[1:]] if op in VECTOR_BITWISE and a[0].get("n", 1) == 1 else [None]
        written = range(a[0]["r"], a[0]["r"] + a[0].get("n", 1))
        for key in [k for k in state if isinstance(k, tuple) and k[1] in written]:
            del state[key]
        if None not in kinds and any(k in MASK_KINDS for k in kinds):
            ms = {k for k in kinds if k in MASK_KINDS}
            state[("v", a[0]["r"])] = ms.pop() if len(ms) == 1 else MASK
            # A plain copy (v_mov_b32) is an ordinary VALU write: inactive lanes keep the old
            # VGPR (SHADER-21 part 2). Only per-lane mask computations take the all-lanes store,
            # because whole-wave readers (wave_mask, v_readfirstlane) read every lane's bit.
            return op != "v_mov_b32"
    if "_saveexec_b" in op or "_wrexec_b" in op or op == "s_bfm_b64":
        put(state, a[0], MASK)
    elif op.startswith(COPIES) and len(a) > 1:
        k = kind(state, a[1])
        if k is None and op.endswith("b64") and a[0]["k"] == "special" and a[0].get("t") in ("vcc", "vcc_lo") and \
                a[1]["k"] == "imm":
            k = MASK
        put(state, a[0], k)
    elif op.startswith(BITWISE):
        kinds = [kind(state, x) for x in a[1:]]
        ms = {k for k in kinds if k in MASK_KINDS}
        if None not in kinds:
            put(state, a[0], (ms.pop() if len(ms) == 1 else MASK) if ms else EITHER)
        elif op.endswith("b64") and (ms or (a[0]["k"] == "special" and a[0].get("t") in ("vcc", "vcc_lo"))) and \
                all(k is not None or x["k"] == "imm" for k, x in zip(kinds, a[1:])):
            put(state, a[0], (ms.pop() if len(ms) == 1 else MASK) if ms else MASK)
        else:
            put(state, a[0], None)
    else:
        put(state, a[0], None)
        if ((op.startswith("v_") and "_co_" in op) or op in ("v_mad_u64_u32", "v_mad_i64_i32")) and len(a) > 1:
            put(state, a[1], MASK)                      # carry/sign lane mask of v_add_co_u32, v_mad_u64_u32, etc.


def parked(blocks, masks):
    """The (vgpr, lane) slots some v_readlane_b32 fetches a mask from, given analyse()'s result."""
    found = set()
    for blk in blocks:
        for ins in blk:
            a = ins["args"]
            if base_op(ins["op"]) == "v_readlane_b32" and len(a) == 3:
                slot = lane_slot(a[1], a[2])
                if slot in masks.get(id(ins), ()):
                    found.add(slot[1:])
    return sorted(found)


def analyse(blocks, succ, *, include_either=False):
    entry = [None] * len(blocks)                        # None: not reached yet
    if blocks:
        entry[0] = {}
    work = [0] if blocks else []
    while work:
        n = work.pop()
        state = dict(entry[n])
        for ins in blocks[n]:
            step(state, ins)
        for s in succ[n]:
            if entry[s] is None:
                merged = dict(state)
            else:                                       # a mask on both paths, or a mask on one and 0 / -1 on the other
                merged = {}
                for r in entry[s].keys() & state.keys():
                    a_k, b_k = entry[s][r], state[r]
                    if a_k == b_k:
                        merged[r] = a_k
                    elif a_k == EITHER:
                        merged[r] = b_k
                    elif b_k == EITHER:
                        merged[r] = a_k
                    else:
                        merged[r] = MASK
            if merged != entry[s]:
                entry[s] = merged
                work.append(s)
    out, writes, either = {}, set(), {}
    for n, blk in enumerate(blocks):
        state = dict(entry[n] or {})
        for ins in blk:
            out[id(ins)] = {r: k for r, k in state.items() if k in MASK_KINDS}
            if include_either:
                either[id(ins)] = {r for r, k in state.items() if k == EITHER}
            if step(state, ins):
                writes.add(id(ins))
    return (out, writes, either) if include_either else (out, writes)
