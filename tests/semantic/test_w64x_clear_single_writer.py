"""--wave64-emulate: every word of the exchange buffer has one writer between two barriers (SHADER-708).

The emulation's Workgroup buffer `%x_buf` starts undefined, so the scheduler (`x_scheduler` in tools/rdna2spirv.py)
zeroes its mask and schedule words, and the data slots of the lanes a partial last wave lacks, before the first
barrier, and at the top of every round it frees the mask words and counts of the slot the next round writes. Until
SHADER-708 those clears were plain `OpStore`s that several invocations aimed at one word: the first clear clamped
the index with `UMin`, so every invocation past the end of a range stored to its last word, and in a round every
invocation that had nothing to clear stored 0 to a shared "sink" word. Two stores to one word with no barrier
between them are a data race in SPIR-V even when both store the same value, and GPU-assisted validation reports
each as `SharedMemoryDataRace-RaceOnStore`.

What is checked here, on the SPIR-V the translator emits for a small hand-written wave64 compute program at several
workgroup sizes (one wave, several waves, a partial last wave, fewer invocations than words to clear, exactly as
many, a three-dimensional group):

  * The scheduler's head is run for every invocation of the workgroup with the small evaluator below (integer
    arithmetic and forward selections only; anything else is an error): before the first `OpControlBarrier`, and
    at the top of rounds 0 to 5, no word of `%x_buf` is stored by two invocations, every store writes 0, and the
    words stored are exactly the ones the layout says have to be cleared there (docs/design.md, "Wave64 emulation":
    2L data words, 6W mask words, 6 counts), so the same memory is zero at the first use as before the fix.
  * Every invocation reaches that first barrier, whatever it had to clear.
  * Every other plain store to `%x_buf` in the module is the lane's own data slot (round parity x L + the
    invocation's index).
  * The evaluator itself: with the guard of a clear removed from the text, it reports the shared word.
  * The results: the program run on 32-wide subgroups by `test_loop_ballot_barrier.run_compute_workgroup`, whose
    Workgroup memory starts as 0xDEADBEEF, gives the values counted by hand.
  * A mesh shader, whose place in the workgroup is LocalInvocationIndex: the same check of its scheduler's head.

That program is one block, so it only gets the round loop. The slot of the next round is freed in three more places,
and each of them is run here as well, a whole workgroup at a time, by the lockstep interpreter of
test_wave64_emulate_opt (`run_emulated_workgroup`). It raises when a word of `%x_buf` is stored by two invocations
between two barriers, or stored by one and loaded or updated atomically by another, and here the buffer starts as
0xDEADBEEF, so a word that is used without having been cleared gives a wrong result:

  * the helper functions of inline structured emulation (`%x_pre_vote_fn`, `%x_pre_data_fn`, `%x_pre_mask_fn`,
    `%x_pre_both_fn`), in a one-wave workgroup (the whole program is inline) and in one of several waves;
  * the local round loop of a region the waves go through differently (a "mini scheduler"), twice in one program
    and two of them back to back;
  * the round loop that takes over at an early `s_endpgm` (the "tail scheduler"), right after a mini scheduler.

Each loop goes around the three slots more than once, with a mask that shrinks from round to round, at workgroup
sizes with a partial last wave, a last wave of one invocation, one invocation in all, and three dimensions.

The programs are written here; no game data.
"""
import os
import re
import struct
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)  # python -I leaves the script's folder off sys.path

import test_loop_ballot_barrier as rounds  # noqa: E402  (translates a compute listing; runs a whole workgroup)
import test_wave64_emulate_opt as opt  # noqa: E402  (the lockstep interpreter of an emulated workgroup)

M32 = 0xFFFFFFFF

# Four exchanges, so the rounds go once around the three mask slots: the wave's lane count, its first thread, and
# how many of its lanes have thread & 3 equal to 3 and to 0. Each thread stores the four values.
PROGRAM = [
    "s_mov_b64 s[10:11], exec",
    "v_and_b32_e32 v1, 3, v0",
    "s_bcnt1_i32_b64 s6, exec",
    "v_readfirstlane_b32 s7, v0",
    "v_cmp_eq_u32_e32 vcc, 3, v1",
    "s_bcnt1_i32_b64 s9, vcc",
    "v_cmp_eq_u32_e32 vcc, 0, v1",
    "s_bcnt1_i32_b64 s8, vcc",
    "v_mov_b32_e32 v2, s6",
    "v_mov_b32_e32 v3, s7",
    "v_mov_b32_e32 v4, s9",
    "v_mov_b32_e32 v5, s8",
    "v_lshlrev_b32_e32 v6, 4, v0",
    "buffer_store_dword v2, v6, s[0:3], 0 offen offset:0",
    "buffer_store_dword v3, v6, s[0:3], 0 offen offset:4",
    "buffer_store_dword v4, v6, s[0:3], 0 offen offset:8",
    "buffer_store_dword v5, v6, s[0:3], 0 offen offset:12",
    "s_endpgm",
]

# Workgroup sizes: one wave; two; a partial wave; four waves, the last one partial; fewer invocations than the 12
# mask and schedule words of one wave; exactly 12; one invocation; and two three-dimensional groups.
SIZES = [(64, 1, 1), (128, 1, 1), (40, 1, 1), (200, 1, 1), (8, 1, 1), (12, 1, 1), (1, 1, 1), (4, 4, 4), (8, 5, 3)]

# A mesh shader of 64 vertices and 64 triangles at most: the wave's count of lanes with (lane & 7) == 5 and its first
# lane are the x and y of the positions it exports. Only its scheduler's head is looked at.
MESH_PROGRAM = [
    "v_mbcnt_lo_u32_b32 v0, -1, 0",
    "v_mbcnt_hi_u32_b32 v0, -1, v0",
    "v_and_b32_e32 v12, 7, v0",
    "v_cmp_eq_u32_e64 s[6:7], 5, v12",
    "s_bcnt1_i32_b64 s10, s[6:7]",
    "v_readfirstlane_b32 s11, v0",
    "v_cvt_f32_u32_e32 v1, s10",
    "v_cvt_f32_u32_e32 v2, s11",
    "v_mov_b32_e32 v6, 0",
    "v_mov_b32_e32 v7, 1.0",
    "s_mov_b32 m0, 0x1003",
    "s_sendmsg sendmsg(MSG_GS_ALLOC_REQ)",
    "v_cmpx_gt_u32_e32 3, v0",
    "exp pos0 v1, v2, v6, v7 done",
    "exp param0 v1, v2, v6, v7",
    "v_cmpx_eq_u32_e32 0, v0",
    "v_mov_b32_e32 v9, 0x200400",
    "exp prim v9, off, off, off done",
    "s_endpgm",
]
MESH_THREADS = [128, 100, 40, 1]

_DEF = re.compile(r"(%\w+) = (Op\w+)(?: (.*))?$")


def translate(size):
    threads = size[0] * size[1] * size[2]
    spv, gen = rounds._translate_cs("cs_w64x_clear_%dx%dx%d" % size, PROGRAM, threads=threads, emulate=True,
                                    size_xyz=size)
    assert gen.control_flow == "emulated wave64 rounds", gen.control_flow
    return spv


def mesh_container(threads):
    """A DXBC container holding only PSV0, the runtime info the translator takes a mesh shader's limits from: stage
    13 (mesh), `threads` x 1 x 1 invocations, at most 64 vertices and 64 primitives, triangles."""
    info = bytearray(48)
    struct.pack_into("<HH", info, 12, 64, 64)
    info[24], info[27] = 13, 2
    struct.pack_into("<III", info, 36, threads, 1, 1)
    chunk = b"PSV0" + struct.pack("<II", 4 + len(info), len(info)) + bytes(info)
    return b"DXBC" + bytes(16) + struct.pack("<IIII", 1, 36 + len(chunk), 1, 36) + chunk


def translate_mesh(threads):
    name = "ms_w64x_clear_%d" % threads
    with tempfile.TemporaryDirectory() as tmp:
        os.mkdir(os.path.join(tmp, "disasm"))
        src = os.path.join(tmp, "disasm", name + ".s")
        with open(src, "w", encoding="utf-8", newline="\n") as f:
            f.write(f"; {name} synthetic\n; mattr: +wavefrontsize64,-wavefrontsize32\n")
            for text, size in rounds._resolve_labels(MESH_PROGRAM):
                f.write(f"\t{text} ; encoding: [{','.join(['0x00'] * size)}]\n")
        with open(os.path.join(tmp, name + ".dxbc"), "wb") as f:
            f.write(mesh_container(threads))
        emulate, rounds.rdna2spirv.WAVE64_EMULATE = rounds.rdna2spirv.WAVE64_EMULATE, True
        try:
            spv = rounds.rdna2spirv.translate(src)
            gen = rounds.rdna2spirv.translate.last_gen
        finally:
            rounds.rdna2spirv.WAVE64_EMULATE = emulate
    assert (gen.stage, gen.control_flow) == ("ms", "emulated wave64 rounds"), (gen.stage, gen.control_flow)
    return spv


class Module:
    """The lines of the translated function, its labels, the module's integer constants and each id's definition."""

    def __init__(self, spv):
        text = [ln.strip() for ln in spv.splitlines()]
        self.consts = {"%true": True, "%false": False}
        for ln in text:
            m = re.match(r"(%\w+) = OpConstant %uint (\d+)$", ln)
            if m:
                self.consts[m.group(1)] = int(m.group(2))
        m = [re.match(r"%x_arr = OpTypeArray %uint (%\w+)$", ln) for ln in text]
        self.buf_words = self.consts[next(x for x in m if x).group(1)]
        first = text.index("%entry = OpLabel")
        self.lines = text[first:text.index("OpFunctionEnd", first)]
        self.labels, self.defs = {}, {}
        for n, ln in enumerate(self.lines):
            m = _DEF.match(ln)
            if m:
                self.defs[m.group(1)] = (m.group(2), (m.group(3) or "").split())
                if m.group(2) == "OpLabel":
                    self.labels[m.group(1)] = n
        # The scheduler's head: from the invocation's index to the first barrier, then the top of a round.
        self.start = next(n for n, ln in enumerate(self.lines)
                          if ln.startswith(("%x_lid = OpLoad", "%x_li = OpLoad")))
        self.barrier = next(n for n, ln in enumerate(self.lines) if ln.startswith("OpControlBarrier"))
        self.round = self.lines.index("%x_round = OpLabel")
        self.round_end = self.lines.index("%x_d = OpLoad %bool %done")

    def run(self, env, start, end, local_id, size, round_no=0):
        """One invocation from line `start` until it reaches line `end`: the (word, value) it stores to %x_buf.
        `env` holds its SSA values and is updated. Only what the scheduler's head is made of is understood."""
        stores, n, steps = [], start, 0
        while n != end:
            steps += 1
            if steps > 10000 or n >= len(self.lines):
                raise AssertionError(f"the invocation never reached line {end}: {self.lines[end]}")
            ln = self.lines[n]
            n += 1
            m = _DEF.match(ln)
            res, op, a = (m.group(1), m.group(2), (m.group(3) or "").split()) if m else (None, *_split(ln))
            v = lambda x: env[x]                                                      # noqa: E731
            if op in ("OpLabel", "OpSelectionMerge", "OpLoopMerge"):
                continue
            if op == "OpBranch":
                n = self.labels[a[0]]
            elif op == "OpBranchConditional":
                n = self.labels[a[1] if v(a[0]) else a[2]]
            elif op == "OpLoad" and a[1] == "%local_id":
                env[res] = tuple(local_id)
            elif op == "OpLoad" and a[1] == "%local_index":
                env[res] = local_id[0] + size[0] * (local_id[1] + size[1] * local_id[2])
            elif op == "OpLoad" and a[1] == "%x_it":
                env[res] = round_no
            elif op == "OpCompositeExtract":
                env[res] = v(a[1])[int(a[2])]
            elif op == "OpIAdd":
                env[res] = (v(a[1]) + v(a[2])) & M32
            elif op == "OpIMul":
                env[res] = (v(a[1]) * v(a[2])) & M32
            elif op == "OpUMod":
                env[res] = v(a[1]) % v(a[2])
            elif op == "OpBitwiseAnd":
                env[res] = v(a[1]) & v(a[2])
            elif op == "OpShiftRightLogical":
                env[res] = v(a[1]) >> v(a[2])
            elif op == "OpIEqual":
                env[res] = v(a[1]) == v(a[2])
            elif op == "OpULessThan":
                env[res] = v(a[1]) < v(a[2])
            elif op == "OpSelect":
                env[res] = v(a[2]) if v(a[1]) else v(a[3])
            elif op == "OpExtInst" and a[2] == "UMin":
                env[res] = min(v(a[3]), v(a[4]))
            elif op == "OpAccessChain" and a[1] == "%x_buf" and len(a) == 3:
                env[res] = ("x_buf", v(a[2]))
            elif op == "OpStore" and isinstance(env.get(a[0]), tuple) and env[a[0]][0] == "x_buf":
                stores.append((env[a[0]][1], v(a[1])))
            else:
                raise AssertionError(f"not expected in the scheduler's head: {ln}")
        return stores


def _split(line):
    op, *args = line.split()
    return op, args


def layout(threads):
    """docs/design.md, "Wave64 emulation": L data slots twice, two mask words per wave in three slots, two counts
    in three slots, one more word. Returns (L, waves, first mask word, first count word, words in the buffer)."""
    waves = -(-threads // 64)
    lanes = 64 * waves
    return lanes, waves, 2 * lanes, 2 * lanes + 6 * waves, 2 * lanes + 6 * waves + 7


def clears(spv, size, rounds_to_run=6):
    """Run the scheduler's head for every invocation of a workgroup of `size`. Returns one {word: [invocation, ...]}
    for the clear before the first barrier and one per round, and the values stored."""
    mod = Module(spv)
    first, per_round, values = {}, [{} for _ in range(rounds_to_run)], set()
    for z in range(size[2]):
        for y in range(size[1]):
            for x in range(size[0]):
                inv = x + size[0] * (y + size[1] * z)
                env = dict(mod.consts)
                for word, value in mod.run(env, mod.start, mod.barrier, (x, y, z), size):
                    first.setdefault(word, []).append(inv)
                    values.add(value)
                for r, seen in enumerate(per_round):
                    for word, value in mod.run(dict(env), mod.round, mod.round_end, (x, y, z), size, round_no=r):
                        seen.setdefault(word, []).append(inv)
                        values.add(value)
    return mod, first, per_round, values


def shared(writers):
    """The words more than one invocation stores to, with who does."""
    return {word: invs for word, invs in sorted(writers.items()) if len(set(invs)) > 1}


class ClearHasOneWriterPerWord(unittest.TestCase):
    def check_clears(self, spv, size):
        threads = size[0] * size[1] * size[2]
        mod, first, per_round, values = clears(spv, size)
        lanes, waves, mask, sched, words = layout(threads)
        self.assertEqual(mod.buf_words, words)
        self.assertEqual(shared(first), {}, "stored by several invocations before the first barrier")
        # Everything the first round may read or add to: all mask words and counts, and both data slots
        # of each lane the partial last wave lacks. The last word of the buffer is not one of them.
        want = set(range(mask, sched + 6)) | set(range(threads, lanes)) | set(range(lanes + threads, 2 * lanes))
        self.assertEqual(set(first), want)
        for r, seen in enumerate(per_round):
            self.assertEqual(shared(seen), {}, f"stored by several invocations at the top of round {r}")
            slot = (r + 1) % 3                  # the slot round r + 1 writes
            want = {mask + 2 * (slot * waves + w) + k for w in range(waves) for k in (0, 1)} | \
                {sched + 2 * slot, sched + 2 * slot + 1}
            self.assertEqual(set(seen), want, f"round {r}")
        self.assertEqual(values, {0})

    def test_no_word_is_cleared_by_two_invocations_and_the_same_words_are_cleared(self):
        for size in SIZES:
            with self.subTest(size=size):
                self.check_clears(translate(size), size)

    def test_a_mesh_shader_s_clears_have_one_writer_per_word(self):
        for threads in MESH_THREADS:
            with self.subTest(threads=threads):
                spv = translate_mesh(threads)
                self.assertIn("%x_li = OpLoad %uint %local_index", spv)
                self.check_clears(spv, (threads, 1, 1))

    def test_every_invocation_reaches_the_first_barrier(self):
        # The barrier has to stay in uniform control flow: each guard of a store is a selection of its own that
        # ends (store, then a branch to the merge block) before the barrier, which is the last thing before the
        # round loop. (Module.run, used by the test above, also raises for an invocation that never gets there.)
        for size in ((64, 1, 1), (40, 1, 1), (8, 1, 1)):
            with self.subTest(size=size):
                mod = Module(translate(size))
                head = mod.lines[mod.start:mod.barrier]
                merges = [ln.split()[1] for ln in head if ln.startswith("OpSelectionMerge")]
                self.assertTrue(merges, "the clear of a range that is not a multiple of the workgroup is guarded")
                for merge in merges:
                    at = head.index(f"OpSelectionMerge {merge} None")
                    cond = head[at + 1].split()
                    self.assertEqual((cond[0], cond[3]), ("OpBranchConditional", merge))
                    self.assertEqual(head[at + 2], f"{cond[2]} = OpLabel")
                    arm = head[at + 3:head.index(f"{merge} = OpLabel")]
                    self.assertEqual([ln.split()[0] for ln in arm if ln.startswith("Op")], ["OpStore", "OpBranch"])
                    self.assertEqual(arm[-1], f"OpBranch {merge}")
                self.assertEqual(mod.lines[mod.barrier + 1], "OpBranch %loop_head")

    def test_other_plain_stores_to_the_buffer_are_the_lane_s_own_slot(self):
        for size in ((64, 1, 1), (200, 1, 1), (8, 5, 3)):
            threads = size[0] * size[1] * size[2]
            with self.subTest(size=size):
                mod = Module(translate(size))
                lanes = layout(threads)[0]
                own = 0
                for n, ln in enumerate(mod.lines):
                    if not ln.startswith("OpStore") or mod.start <= n < mod.barrier or mod.round <= n < mod.round_end:
                        continue
                    op, a = mod.defs.get(ln.split()[1], ("", []))
                    if op != "OpAccessChain" or a[1] != "%x_buf":
                        continue
                    op, idx = mod.defs[a[2]]
                    self.assertEqual((op, idx[2]), ("OpIAdd", "%x_li"), ln)
                    op, base = mod.defs[idx[1]]
                    self.assertEqual((op, mod.consts[base[2]]), ("OpIMul", lanes), ln)
                    op, parity = mod.defs[base[1]]
                    self.assertEqual((op, mod.consts[parity[2]]), ("OpBitwiseAnd", 1), ln)
                    own += 1
                self.assertGreater(own, 0, "the program has a per-lane exchange")
                self.assertGreaterEqual(lanes, threads)

    def test_the_check_reports_a_clear_without_its_guard(self):
        size = (64, 1, 1)
        spv = translate(size)
        _, mask, sched = layout(64)[1:4]
        for guard, word, phase in (("OpBranchConditional %x_ql %x_qs %x_qe", sched + 2, 1),
                                   ("OpBranchConditional %x_ml %x_ms %x_me", mask + 2, 1),
                                   ("OpBranchConditional %x_z0_0o %x_z0_0s %x_z0_0e", None, 0)):
            with self.subTest(guard=guard):
                self.assertEqual(spv.count(guard), 1)
                bad = spv.replace(guard, guard.replace(guard.split()[1], "%true"))
                if phase == 0:
                    # Without the guard, invocations 12-63 store past the 12 words (and past the buffer), each to
                    # a word of its own, so it is the set of words stored that gives it away.
                    first = clears(bad, size)[1]
                    self.assertNotEqual(set(first), set(range(mask, sched + 6)))
                else:
                    seen = clears(bad, size)[2][0]              # round 0 frees slot 1
                    self.assertEqual(shared(seen), {word: list(range(64)), word + 1: list(range(64))})


class ResultsWithUndefinedWorkgroupMemory(unittest.TestCase):
    def test_values_on_32_wide_subgroups(self):
        for threads in (64, 128, 40, 200):
            with self.subTest(threads=threads):
                spv = translate((threads, 1, 1))
                mem = rounds.run_compute_workgroup(spv, threads, 32, push_constants=rounds.VSHARP)
                for t in range(threads):
                    base = t & ~63
                    wave = range(base, min(base + 64, threads))
                    want = (len(wave), base, sum(1 for u in wave if u & 3 == 3), sum(1 for u in wave if u & 3 == 0))
                    self.assertEqual(tuple(mem.get(16 * t + 4 * k, -1) for k in range(4)), want, f"thread {t}")


# ---- the other places a round starts: whole workgroups in the lockstep interpreter ----

FILL = 0xDEADBEEF                          # what %x_buf holds at launch here (undefined on a GPU)
PUSH = opt.VSHARP + [0] * 28               # s[0:3]: a raw buffer over every offset
RSRC2 = 0x8088 | 2 << 11                   # opt._translate_encoded's default plus the thread ids y and z in v1, v2
STORE = ["v_add_nc_u32_e32 v4, s21, v4",   # out[4 t] = the thread's value plus the wave's counts, out[4 t + 1] = s19
         "v_mov_b32_e32 v5, s19",
         "buffer_store_dword v4, v10, s[0:3], 0 offen offset:0",
         "buffer_store_dword v5, v10, s[0:3], 0 offen offset:4",
         "s_endpgm"]

SINGLE_SIZES = [(64, 1, 1), (40, 1, 1), (1, 1, 1), (4, 4, 4), (5, 3, 2)]
WAVES_SIZES = [(128, 1, 1), (200, 1, 1), (65, 1, 1), (8, 5, 3), (4, 4, 8)]


def setup(size):
    """v0 = the thread's index in the workgroup (it starts as the x of its id), v11 = its lane, v4 = its value
    (the index to start with), v10 = 16 x index, s21 = 0."""
    return ["s_mov_b64 exec, -1",
            f"v_mul_u32_u24_e32 v6, {size[0]}, v1",
            f"v_mul_u32_u24_e32 v7, {size[0] * size[1]}, v2",
            "v_add_nc_u32_e32 v0, v0, v6",
            "v_add_nc_u32_e32 v0, v0, v7",
            "v_and_b32_e32 v11, 63, v0",
            "v_lshlrev_b32_e32 v10, 4, v0",
            "v_mov_b32_e32 v4, v0",
            "s_mov_b32 s21, 0"]


def loop(tag, lane, counter):
    """While `counter` (an SGPR) is not 0: every thread adds the value of lane `lane` of its wave, s21 grows by
    the number of the wave's lanes below `counter` (a mask that shrinks by one lane a time), counter -= 1. Two
    exchanges a time around: a value and two mask words."""
    return [f"loop_{tag}:",
            f"v_readlane_b32 s18, v4, {lane}",
            f"v_cmp_gt_u32_e64 s[6:7], {counter}, v11",
            "s_bcnt1_i32_b64 s9, s[6:7]",
            "s_add_u32 s21, s21, s9",
            "v_add_nc_u32_e32 v4, s18, v4",
            f"s_sub_u32 {counter}, {counter}, 1",
            f"s_cmp_gt_u32 {counter}, 0",
            f"s_cbranch_scc1 loop_{tag}"]


def single_program(size):
    """One wave. Four times: the first lane's value and the wave's active mask (%x_pre_both_fn), lane 5's value
    (%x_pre_data_fn), a count of the lanes below 4, 3, 2, 1 (%x_pre_mask_fn); then a voted branch
    (%x_pre_vote_fn)."""
    return setup(size) + [
        "s_mov_b32 s20, 4",
        "loop:",
        "v_readfirstlane_b32 s17, v4",
        "v_readlane_b32 s18, v4, 5",
        "v_cmp_gt_u32_e64 s[6:7], s20, v11",
        "s_bcnt1_i32_b64 s9, s[6:7]",
        "s_add_u32 s21, s21, s9",
        "v_add_nc_u32_e32 v4, s17, v4",
        "v_add_nc_u32_e32 v4, s18, v4",
        "s_sub_u32 s20, s20, 1",
        "s_cmp_gt_u32 s20, 0",
        "s_cbranch_scc1 loop",
        "v_readfirstlane_b32 s19, v0",
        "v_cmp_gt_u32_e32 vcc, 2, v11",
        "s_cbranch_vccz skip",                  # not taken: lane 0 is below 2
        "s_add_u32 s21, s21, 100",
        "skip:"] + STORE


def hybrid_program(size):
    """Several waves: inline exchanges at the top level (the helpers with a wave parameter), a loop only the first
    wave runs (4 times) and a loop only the other waves run (5 times), each in a mini scheduler. The last block is
    four inline exchanges in a row: a count of the lanes below 8, two values, a count of the lanes below 2, which
    uses the mask slot of the first count again, freed by the helper of the exchange between them."""
    return setup(size) + [
        "v_readfirstlane_b32 s19, v0",          # the wave's first thread
        "s_cmp_lt_u32 s19, 64",
        "s_cbranch_scc0 second",
        "s_mov_b32 s20, 4"] + loop("a", 37, "s20") + [
        "second:",
        "v_readlane_b32 s22, v4, 3",
        "v_add_nc_u32_e32 v4, s22, v4",
        "s_cmp_ge_u32 s19, 64",
        "s_cbranch_scc0 last",
        "s_mov_b32 s20, 5"] + loop("b", 9, "s20") + [
        "last:",
        "v_cmp_gt_u32_e64 s[6:7], 8, v11",
        "s_bcnt1_i32_b64 s9, s[6:7]",
        "s_add_u32 s21, s21, s9",
        "v_readlane_b32 s25, v4, 1",
        "v_add_nc_u32_e32 v4, s25, v4",
        "v_readlane_b32 s25, v4, 2",
        "v_add_nc_u32_e32 v4, s25, v4",
        "v_cmp_gt_u32_e64 s[6:7], 2, v11",
        "s_bcnt1_i32_b64 s9, s[6:7]",
        "s_add_u32 s21, s21, s9"] + STORE


def tail_program(size):
    """Several waves: the first wave's loop (4 times) is a mini scheduler, which the other waves sit out, and
    everything from there on, with an s_endpgm on either side of the branch, is the tail scheduler: the first wave
    stores and ends in its first round, the other waves go around their loop twice after that."""
    return setup(size) + [
        "v_readfirstlane_b32 s19, v0",
        "v_readlane_b32 s22, v4, 3",
        "v_add_nc_u32_e32 v4, s22, v4",
        "s_cmp_lt_u32 s19, 64",
        "s_cbranch_scc0 other",
        "s_mov_b32 s20, 4"] + loop("a", 37, "s20") + STORE + [
        "other:",
        "s_mov_b32 s20, 2"] + loop("b", 9, "s20") + STORE


def loops_program(size):
    """Several waves (four at most), two loops back to back that every wave runs, wave w the first one w + 2
    times and the second one 4 - w times: two mini schedulers with nothing between them."""
    return setup(size) + [
        "v_readfirstlane_b32 s19, v0",
        "s_lshr_b32 s24, s19, 6",
        "s_add_u32 s20, s24, 2",
        "s_sub_u32 s23, 4, s24"] + loop("a", 37, "s20") + loop("b", 9, "s23") + STORE


def waves_of(size):
    threads = size[0] * size[1] * size[2]
    return [list(range(base, min(base + 64, threads))) for base in range(0, threads, 64)]


def run_loop(value, wave, lane, trips):
    """What loop() does to one wave: `value` (by thread) is updated; returns what s21 grows by."""
    counts = 0
    for left in range(trips, 0, -1):
        got = value[wave[lane]] if lane < len(wave) else 0        # a lane the wave lacks reads as 0
        counts += min(left, len(wave))
        for t in wave:
            value[t] += got
    return counts


def stored(value, s21, s19):
    out = {}
    for t, v in value.items():
        out[16 * t], out[16 * t + 4] = (v + s21[t]) & M32, s19[t]
    return out


def single_expect(size):
    wave, = waves_of(size)
    value, counts = {t: t for t in wave}, 100
    for left in (4, 3, 2, 1):
        got = value[0] + (value[5] if len(wave) > 5 else 0)
        counts += min(left, len(wave))
        for t in wave:
            value[t] += got
    return stored(value, dict.fromkeys(wave, counts), dict.fromkeys(wave, 0))


def waves_expect(kind, size):
    value, s21, s19 = {}, {}, {}
    for w, wave in enumerate(waves_of(size)):
        value.update((t, t) for t in wave)
        lane = lambda n: value[wave[n]] if len(wave) > n else 0                       # noqa: E731
        counts = 0
        if kind == "hybrid":
            if w == 0:
                counts += run_loop(value, wave, 37, 4)
            got = lane(3)
            for t in wave:
                value[t] += got
            if w:
                counts += run_loop(value, wave, 9, 5)
            for n in (1, 2):
                got = lane(n)
                for t in wave:
                    value[t] += got
            counts += min(8, len(wave)) + min(2, len(wave))
        elif kind == "tail":
            got = lane(3)
            for t in wave:
                value[t] += got
            counts += run_loop(value, wave, 37, 4) if w == 0 else run_loop(value, wave, 9, 2)
        else:
            counts += run_loop(value, wave, 37, w + 2)
            counts += run_loop(value, wave, 9, 4 - w)
        s21.update(dict.fromkeys(wave, counts))
        s19.update(dict.fromkeys(wave, wave[0]))
    return stored(value, s21, s19)


def run_workgroup(name, program, size):
    """Translate with emulation and run the whole workgroup in the lockstep interpreter, in both orders of the
    invocations between two barriers, %x_buf starting as FILL. Returns (gen, SPIR-V, memory, barriers passed)."""
    spv, gen = opt._translate_encoded("cs_w64x_%s_%dx%dx%d" % ((name,) + size), rounds._resolve_labels(program),
                                      size, emulate=True, rsrc2=RSRC2)
    runs = [opt.run_emulated_workgroup(spv, size, push_constants=PUSH, reverse_order=rev, x_buf_fill=FILL)
            for rev in (False, True)]
    assert runs[0] == runs[1], "the result depends on the order the invocations run in"
    return gen, spv, runs[0][0], runs[0][1]


def steps(gen):
    return [step[0] for step in gen.emulate_plan["steps"]]


class EveryRoundStartHasOneWriterPerWord(unittest.TestCase):
    def test_inline_helpers_of_a_single_wave(self):
        for size in SINGLE_SIZES:
            with self.subTest(size=size):
                gen, spv, memory, barriers = run_workgroup("single", single_program(size), size)
                self.assertEqual((gen.control_flow, gen.emulate_plan["mode"], gen.x_lanes),
                                 ("emulated wave64 structured", "inline_all", 64))
                self.assertEqual(sorted(gen.x_round_fns), ["both", "data", "mask", "vote"])
                self.assertNotIn("OpSwitch", spv)
                # The first clear, 4 x 3 exchanges in the loop, the first lane and the vote after it.
                self.assertGreaterEqual(barriers, 15)
                self.assertEqual(memory, single_expect(size))

    def test_inline_helpers_and_mini_schedulers_of_several_waves(self):
        for size in WAVES_SIZES:
            with self.subTest(size=size):
                gen, spv, memory, barriers = run_workgroup("hybrid", hybrid_program(size), size)
                self.assertEqual((gen.control_flow, gen.emulate_plan["mode"]),
                                 ("emulated wave64 structured", "structured_hybrid"))
                self.assertGreater(gen.x_lanes, 64)
                self.assertEqual(sorted(gen.x_round_fns), ["both", "data", "mask"])
                self.assertEqual([s for s in steps(gen) if s != "inline_sync_free"],
                                 ["inline_block", "mini_scheduler", "inline_block", "mini_scheduler", "inline_block"])
                # No barrier of its own in front of either mini scheduler: its first round follows the reads of
                # the inline exchange before it directly.
                self.assertEqual([step[-1] for step in gen.emulate_plan["steps"] if step[0] == "mini_scheduler"],
                                 [False, False])
                self.assertGreater(barriers, 20)
                self.assertEqual(memory, waves_expect("hybrid", size))

    def test_tail_scheduler_after_a_mini_scheduler(self):
        for size in WAVES_SIZES:
            with self.subTest(size=size):
                gen, spv, memory, barriers = run_workgroup("tail", tail_program(size), size)
                self.assertEqual(steps(gen), ["inline_block", "mini_scheduler", "tail_scheduler"])
                self.assertEqual([step[-1] for step in gen.emulate_plan["steps"][1:]], [False, False])
                self.assertIn("%x_round = OpLabel", spv)
                self.assertGreater(barriers, 20)
                self.assertEqual(memory, waves_expect("tail", size))

    def test_two_mini_schedulers_back_to_back(self):
        for size in WAVES_SIZES:
            with self.subTest(size=size):
                gen, spv, memory, barriers = run_workgroup("loops", loops_program(size), size)
                kinds = steps(gen)
                self.assertEqual(kinds.count("mini_scheduler"), 2)
                at = kinds.index("mini_scheduler")
                self.assertEqual(kinds[at:at + 2], ["mini_scheduler", "mini_scheduler"])
                self.assertNotIn("tail_scheduler", kinds)
                self.assertGreater(barriers, 20)
                self.assertEqual(memory, waves_expect("loops", size))

    def test_the_interpreter_reports_a_store_another_invocation_reads_or_repeats(self):
        size = (128, 1, 1)
        spv, gen = opt._translate_encoded("cs_w64x_tail_check", rounds._resolve_labels(tail_program(size)), size,
                                          emulate=True, rsrc2=RSRC2)
        run = lambda text: opt.run_emulated_workgroup(text, size, push_constants=PUSH)   # noqa: E731
        run(spv)
        # The tail scheduler frees the slot of round + 2 instead: that is the slot of round - 1, which the wave's
        # lanes read in this round. One writer still, but its store races with their loads.
        freed = f"%x_n1 = OpIAdd %uint %x_itv {gen.consts[1]}"
        self.assertEqual(spv.count(freed), 1)
        with self.assertRaisesRegex(AssertionError, r"%x_buf word \d+ is (stored|read) by invocation \d+ and "
                                                    r"(read|stored) by \d+ with no barrier"):
            run(spv.replace(freed, f"%x_n1 = OpIAdd %uint %x_itv {gen.consts[2]}"))
        # A clear without its guard, in the mini scheduler and in the tail scheduler: every lane stores.
        for guard in ("OpBranchConditional %xm0_ml %xm0_ms %xm0_me", "OpBranchConditional %xm0_ql %xm0_qs %xm0_qe",
                      "OpBranchConditional %x_ml %x_ms %x_me", "OpBranchConditional %x_ql %x_qs %x_qe"):
            with self.subTest(guard=guard):
                self.assertEqual(spv.count(guard), 1)
                with self.assertRaisesRegex(AssertionError, r"%x_buf word \d+ is stored by invocations \d+ and \d+"):
                    run(spv.replace(guard, guard.replace(guard.split()[1], "%true")))


if __name__ == "__main__":
    unittest.main()
