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

The programs are written here; no game data.
"""
import os
import re
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)  # python -I leaves the script's folder off sys.path

import test_loop_ballot_barrier as rounds  # noqa: E402  (translates a compute listing; runs a whole workgroup)

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

_DEF = re.compile(r"(%\w+) = (Op\w+)(?: (.*))?$")


def translate(size):
    threads = size[0] * size[1] * size[2]
    spv, gen = rounds._translate_cs("cs_w64x_clear_%dx%dx%d" % size, PROGRAM, threads=threads, emulate=True,
                                    size_xyz=size)
    assert gen.control_flow == "emulated wave64 rounds", gen.control_flow
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
    def test_no_word_is_cleared_by_two_invocations_and_the_same_words_are_cleared(self):
        for size in SIZES:
            threads = size[0] * size[1] * size[2]
            with self.subTest(size=size):
                mod, first, per_round, values = clears(translate(size), size)
                lanes, waves, mask, sched, words = layout(threads)
                self.assertEqual(mod.buf_words, words)
                self.assertEqual(shared(first), {}, "stored by several invocations before the first barrier")
                # Everything the first round may read or add to: all mask words and counts, and both data slots
                # of each lane the partial last wave lacks. The last word of the buffer is not one of them.
                want = set(range(mask, sched + 6)) | set(range(threads, lanes)) | \
                    set(range(lanes + threads, 2 * lanes))
                self.assertEqual(set(first), want)
                for r, seen in enumerate(per_round):
                    self.assertEqual(shared(seen), {}, f"stored by several invocations at the top of round {r}")
                    slot = (r + 1) % 3                  # the slot round r + 1 writes
                    want = {mask + 2 * (slot * waves + w) + k for w in range(waves) for k in (0, 1)} | \
                        {sched + 2 * slot, sched + 2 * slot + 1}
                    self.assertEqual(set(seen), want, f"round {r}")
                self.assertEqual(values, {0})

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


if __name__ == "__main__":
    unittest.main()
