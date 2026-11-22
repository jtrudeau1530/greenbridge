"""tools/crosslane_fuzz.py: the reference model of the cross-lane ops, checked against values worked out by hand
from the RDNA ISA rules in its docstring. No GPU, no llvm-mc, no game files.

    python -I tests/test_crosslane_fuzz.py
"""
import argparse
import contextlib
import io
import os
import random
import re
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

TOOLS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools")
TOOL = os.path.join(TOOLS, "crosslane_fuzz.py")
sys.path.insert(0, TOOLS)
import crosslane_fuzz as X  # noqa: E402


def wave(size=64):
    w = X.Wave(size, (1 << size) - 1)
    w.v[2] = [3 * l + 1 for l in range(size)]
    return w


def draw(gen, method, pred, tries=4000):
    for _ in range(tries):
        o = getattr(gen, method)()
        if o is not None and pred(o):
            return o
    raise AssertionError(f"{method}: no op matched")


def operands(insn):
    """Register numbers / literals of one instruction line, in order (modifiers after the operands dropped)."""
    head = insn.split(" row_")[0].split(" quad_perm")[0].split(" dpp8")[0]
    return [t.strip() for t in head.split(" ", 1)[1].split(",")]


class Model(unittest.TestCase):
    def test_readlane_ignores_exec(self):
        g = X.Gen(random.Random(1), 64, 64)
        o = draw(g, "op_readlane", lambda o: "s_mov_b32" not in o.text)
        dst, src, lane = operands(o.text)
        w = wave()
        w.v[int(src[1:])] = [5 * l for l in range(64)]
        w.exec = 1 << 63
        o.apply([w])
        self.assertEqual(w.s[int(dst[1:])], 5 * int(lane))

    def test_readfirstlane(self):
        g = X.Gen(random.Random(0), 64, 64)
        o = g.op_readfirstlane()
        dst, src = (int(t[1:]) for t in operands(o.text))
        w = wave()
        w.v[src] = [7 + l for l in range(64)]
        w.exec = 0                                      # EXEC 0: lane 0
        o.apply([w])
        self.assertEqual(w.s[dst], 7)
        w.exec = (1 << 40) | (1 << 50)                  # lowest active lane
        o.apply([w])
        self.assertEqual(w.s[dst], 47)

    def test_cmp_writes_zero_for_inactive_lanes(self):
        g = X.Gen(random.Random(2), 64, 64)
        o = draw(g, "op_cmp", lambda o: "_ne_u32_e64 s[" in o.text and len(set(operands(o.text)[1:])) == 2)
        dst, a, b = operands(o.text)
        w = wave()
        w.v[int(a[1:])] = list(range(64))
        w.v[int(b[1:])] = [64] * 64                     # a != b in every lane
        w.exec = 0xFFFF0000FFFF0000
        o.apply([w])
        self.assertEqual(w.mask(int(dst[2:].split(":")[0])), 0xFFFF0000FFFF0000)

    def test_mbcnt_lo_hi(self):
        g = X.Gen(random.Random(4), 64, 64)
        g.masks = [16]
        o = draw(g, "op_mbcnt", lambda o: "s16" in o.text)
        d = int(o.text.split()[1].strip(",")[1:])
        acc = int(operands(o.text.split("\n")[0])[2][1:])
        w = wave()
        w.v[acc] = [0] * 64
        w.set_mask(16, 0xF0F0F0F0F0F0F0F0)
        o.apply([w])
        self.assertEqual([w.vr(d)[l] for l in (0, 8, 40, 63)], [0, 4, 20, 31])

    def test_saveexec_and_restore(self):
        g = X.Gen(random.Random(3), 64, 64)
        g.masks = [16]
        save, restore = g.op_saveexec(), g.op_restore()
        w = wave()
        w.exec = 0x00000000FFFFFFFF
        w.set_mask(16, 0x0F0F0F0F0F0F0F0F)
        save.apply([w])
        self.assertEqual(w.exec, 0x0F0F0F0F)
        restore.apply([w])
        self.assertEqual(w.exec, 0x00000000FFFFFFFF)

    def test_writelane_wave32_uses_bits_4_0(self):
        g = X.Gen(random.Random(0), 32, 32)
        o = draw(g, "op_writelane", lambda o: o.tag == "writelane_w32hi")
        m = re.search(r"s_mov_b32 s28, (\d+)", o.text)
        d, src, sel = operands(o.text.split("\n")[-1].strip())
        enc = int(m.group(1)) if m else int(sel)
        self.assertGreaterEqual(enc, 32)
        w = wave(32)
        if src.startswith("s"):
            w.s[int(src[1:])] = 0xABCD
        val = 0xABCD if src.startswith("s") else int(src) & X.M32
        w.v[int(d[1:])] = [0x5555] * 32
        o.apply([w])
        self.assertEqual(w.v[int(d[1:])], [val if l == enc & 31 else 0x5555 for l in range(32)])

    def test_permlanex16_mapping(self):
        g = X.Gen(random.Random(6), 64, 64)
        o = draw(g, "op_permlane", lambda o: "v_permlanex16_b32" in o.text)
        insn = o.text.split("\n")[-1].strip()
        d, v = (int(t[1:]) for t in operands(insn)[:2])
        s2 = int(re.search(r"s_mov_b32 s29, 0x([0-9a-f]+)", o.text).group(1), 16)
        lit = re.search(r"s_mov_b32 s28, 0x([0-9a-f]+)", o.text) or re.search(r"0x([0-9a-f]+)", insn)
        s1 = int(lit.group(1), 16)                      # from s28, or the instruction's literal
        w = wave()
        w.v[v] = [500 + l for l in range(64)]
        o.apply([w])
        nib = (s1 | s2 << 32)
        for l in (0, 5, 20, 40, 63):
            src = (l & ~31) | ((l & 16) ^ 16) | (nib >> (4 * (l & 15)) & 15)
            self.assertEqual(w.vr(d)[l], 500 + src)

    def test_dpp_invalid_source_and_disabled_bank_are_not_written(self):
        g = X.Gen(random.Random(5), 64, 64, dpp=True)
        ops = [o for o in (g.op_dpp() for _ in range(6000)) if o.text.startswith("v_add_nc_u32_dpp")]
        shr = next(o for o in ops if " row_shr:1 " in o.text and "row_mask:0xf bank_mask:0xf" in o.text
                   and "bound_ctrl" not in o.text and "fi:1" not in o.text)
        bank = next(o for o in ops if "row_mask:0xf bank_mask:0x1" in o.text and not o.text.endswith("0x1f"))
        for o, unwritten, written in ((shr, (0, 16, 32, 48), (1, 17)), (bank, (4, 15, 20), (0, 3, 16))):
            d, a, b = (int(t[1:]) for t in operands(o.text))
            w = wave()
            w.v[a] = [100 + l for l in range(64)]
            w.v[b] = [1000 + l for l in range(64)]
            if d not in (a, b):
                w.v[d] = [7] * 64
            before = list(w.vr(d))
            o.apply([w])
            for l in unwritten:
                self.assertEqual(w.vr(d)[l], before[l], (o.text, l))
            for l in written:
                self.assertNotEqual(w.vr(d)[l], before[l], (o.text, l))

    def test_branch_skips_body_per_wave(self):
        g = X.Gen(random.Random(9), 64, 128)
        o = draw(g, "op_branch", lambda o: "s_cbranch_execz" in o.text and "v_readfirstlane" in o.text)
        w0, w1 = wave(), wave()
        w0.exec, w1.exec = 0, 1 << 33
        dst = int(re.search(r"v_readfirstlane_b32 s(\d+)", o.text).group(1))
        w0.s[dst] = w1.s[dst] = 0xDEAD
        o.apply([w0, w1])
        self.assertEqual(w0.s[dst], 0xDEAD)            # EXEC 0: this wave skips the body

    def test_programs_are_reproducible(self):
        a = X.make_program(42, 64, 20, allow_maskdata=False)
        b = X.make_program(42, 64, 20, allow_maskdata=False)
        self.assertEqual(a.source(), b.source())
        self.assertEqual(a.expected()[0], b.expected()[0])

    def test_self_test(self):
        self.assertEqual(X.self_test(), 0)

    def test_dpp_seeds_are_unchanged(self):
        # dpp_op was split out of Gen.op_dpp keeping the draw order; this program (5 DPP ops) is the one main
        # generated before the split, so a seed someone reported still means the same program.
        import hashlib
        p = X.make_program(77, 64, 28, dpp=True)
        self.assertEqual(sum("_dpp " in o.text for o in p.ops), 5)
        self.assertEqual(hashlib.sha1(p.source().encode()).hexdigest(), "7174158b8d064cbddca318d3d7f433afac072691")


def run_dpp(op, exec_mask, size=64, old=7):
    """Apply one DPP op to a wave whose src0 (v3, or v2 in place) is 100 + lane, src1 (v4) 1000 + lane and vD (v2)
    `old`; returns (vD, hypothesis flag)."""
    w = X.Wave(size, (1 << size) - 1)
    w.v[3] = [100 + l for l in range(size)]
    w.v[4] = [1000 + l for l in range(size)]
    if op.text.split()[2].rstrip(",") == "v2":      # in place: vD is src0
        w.v[2] = [100 + l for l in range(size)]
    else:
        w.v[2] = [old] * size
    w.exec = exec_mask
    hyp = op.fn(w)
    return w.vr(2), hyp


class DppSweep(unittest.TestCase):
    """dpp_op against values worked out by hand from the RDNA2 DPP rules, and what --dpp-sweep generates."""
    ODD = 0x5555555555555555

    def test_row_shr_invalid_source_is_unwritten_or_zero(self):
        v, _ = run_dpp(X.dpp_op(2, 3, 4, "mov", control=("row_shr", 1)), X.M64)
        self.assertEqual((v[0], v[1], v[15], v[16], v[17]), (7, 100, 114, 7, 116))
        v, _ = run_dpp(X.dpp_op(2, 3, 4, "mov", control=("row_shr", 1), bound=True), X.M64)
        self.assertEqual((v[0], v[16], v[17]), (0, 0, 116))

    def test_inactive_source_fi_and_bound_ctrl(self):
        # EXEC odd lanes off: lane 2 reads lane 1 (off). fi 0: bound_ctrl decides; fi 1: the inactive value is read.
        shr = lambda **kw: run_dpp(X.dpp_op(2, 3, 4, "mov", control=("row_shr", 1), **kw), self.ODD)[0]
        self.assertEqual(shr()[2], 7)
        self.assertEqual(shr(bound=True)[2], 0)
        self.assertEqual(shr(fi=True)[2], 101)
        self.assertEqual(shr(fi=True, bound=True)[2], 101)
        self.assertEqual(shr(fi=True)[1], 7)                    # lane 1 itself is off: never written

    def test_non_mov_unwritten_lane_keeps_old_vd(self):
        # #238: an invalid source without bound_ctrl leaves the lane alone, not op(old vD, src1).
        v, _ = run_dpp(X.dpp_op(2, 3, 4, "add", control=("row_shl", 2)), X.M64)
        self.assertEqual((v[0], v[13], v[14], v[15]), (102 + 1000, 115 + 1013, 7, 7))

    def test_row_and_bank_masks(self):
        v, _ = run_dpp(X.dpp_op(2, 3, 4, "mov", control=("quad_perm", (0, 0, 0, 0)), row_mask=0x2, bank_mask=0x5),
                       X.M64)
        written = [l for l in range(64) if v[l] != 7]
        self.assertEqual(written, [16, 17, 18, 19, 24, 25, 26, 27])
        self.assertEqual(v[18], 116)

    def test_in_place_reads_every_source_first(self):
        v, _ = run_dpp(X.dpp_op(2, 2, 4, "mov", control=("row_ror", 1)), X.M64)
        self.assertEqual((v[0], v[1], v[16]), (115, 100, 131))

    def test_dpp8_inactive_source(self):
        bcast7 = (7,) * 8
        v, hyp = run_dpp(X.dpp_op(2, 3, 4, "mov", dpp8=bcast7), self.ODD)
        self.assertEqual((v[0], v[8], v[1]), (0, 0, 7))            # lane 7 is off: 0 (the #238 hypothesis)
        self.assertTrue(hyp)
        v, hyp = run_dpp(X.dpp_op(2, 3, 4, "mov", dpp8=bcast7, fi=True), self.ODD)
        self.assertEqual((v[0], v[8]), (107, 115))
        self.assertFalse(hyp)

    def test_row_share_and_xmask_wave32(self):
        v, _ = run_dpp(X.dpp_op(2, 3, 4, "mov", control=("row_share", 5)), X.M32, size=32)
        self.assertEqual((v[0], v[15], v[16], v[31]), (105, 105, 121, 121))
        v, _ = run_dpp(X.dpp_op(2, 3, 4, "xor", control=("row_xmask", 3)), X.M32, size=32)
        self.assertEqual(v[5], (100 + 6) ^ 1005)

    def test_sweep_covers_every_control(self):
        controls = X.dpp16_sweep_controls()
        self.assertEqual(len(controls), 335)
        self.assertEqual(len(set(controls)), 335)
        for wave in (32, 64):
            progs = X.dpp_sweep_programs(wave)
            self.assertEqual(len(progs), 7206)
            self.assertEqual(len({p.name for p in progs}), len(progs))
            self.assertEqual({p.family for p in progs}, set(X.DPP16_CONTROLS) | {"masks", "dpp8"})
            texts = [p.ops[1].text for p in progs]
            for c, arg in controls:
                text = X.dpp16_select(c, arg)[0]
                self.assertTrue(any(f" {text} " in t for t in texts), text)
            for kind in X.DPP_OPS.values():
                self.assertTrue(any(t.startswith(kind + "_dpp v2, v2,") or t.startswith(kind + "_dpp v2, v2 ")
                                    for t in texts), kind)
            masks = {re.search(r"row_mask:(0x\w+) bank_mask:(0x\w+)", t).groups() for t in texts if "row_mask" in t}
            self.assertEqual(len(masks), 256)
            for p in progs:
                self.assertEqual(p.group, wave)
                self.assertEqual([o.tag for o in p.ops], ["exec", "dpp"])
                p.expected()                                     # the model runs every case

    def test_sweep_cli_needs_the_oracle_tools(self):
        env = dict(os.environ, LEONIDA_AMD_CLANG=os.path.join(TOOLS, "no-such-clang.exe"))
        r = subprocess.run([sys.executable, "-I", TOOL, "--oracle", "hw", "--dpp-sweep"],
                           capture_output=True, text=True, env=env)
        self.assertEqual(r.returncode, 77, r.stdout + r.stderr)

    def test_sweep_on_hardware(self):
        from hw_oracle import runner
        try:
            oracle = runner.open_oracle()
        except runner.Unavailable as e:
            self.skipTest(f"hardware oracle unavailable: {e}")
        try:
            stats, fails = {}, []
            progs = [p for wave in (32, 64) for p in X.dpp_sweep_programs(wave)[::97]]
            X.run_hw(oracle, progs, stats, fails)
        finally:
            oracle.close()
        self.assertEqual([(p.name, p.label, w) for p, _, w, _ in fails], [])
        self.assertEqual(sum(st["match"] for st in stats.values()), len(progs))


class Modes(unittest.TestCase):
    """--modes takes native, emulate or both. Anything else used to run every program natively, wave64 included."""

    def test_both_is_native_and_emulate(self):
        self.assertEqual(X.parse_modes("both"), ["native", "emulate"])
        self.assertEqual(X.parse_modes("native,both"), ["native", "emulate"])

    def test_list_keeps_order_and_drops_repeats(self):
        self.assertEqual(X.parse_modes(" emulate, native ,emulate,"), ["emulate", "native"])

    def test_unknown_or_empty_mode_is_rejected(self):
        for text in ("emulat", "native,wave64", "", " , "):
            with self.subTest(text=text), self.assertRaises(argparse.ArgumentTypeError):
                X.parse_modes(text)

    def test_cli_rejects_an_unknown_mode_before_running(self):
        r = subprocess.run([sys.executable, "-I", TOOL, "--modes", "emulat", "--self-test"],
                           capture_output=True, text=True)
        self.assertEqual(r.returncode, 2, r.stdout + r.stderr)
        self.assertIn("unknown mode 'emulat'", r.stderr)

    def test_emulate_with_wave32_is_a_usage_error(self):
        # emulate runs a wave64 program as two 32-lane halves, so --wave 32 selects nothing to run: it used to print
        # a bare `summary:` and exit 0. The refusal comes before the missing-harness check, so no build is needed.
        r = subprocess.run([sys.executable, "-I", TOOL, "--wave", "32", "--modes", "emulate", "--programs", "1",
                            "--exe", os.path.join(TOOLS, "no-such-build", "vk_compute_test")],
                           capture_output=True, text=True)
        self.assertEqual(r.returncode, 2, r.stdout + r.stderr)
        self.assertIn("--modes emulate needs wave64 programs", r.stderr)
        self.assertNotIn("vk_compute_test not found", r.stderr)
        self.assertEqual(r.stdout, "")

    def test_wave64_native_without_w64_native_is_a_usage_error(self):
        # A wave64 program runs natively only when --w64-native asks for the cross-half approximation; without it the
        # selection was empty and the run ended after a bare `summary:` with exit 0.
        r = subprocess.run([sys.executable, "-I", TOOL, "--wave", "64", "--modes", "native", "--programs", "1",
                            "--exe", os.path.join(TOOLS, "no-such-build", "vk_compute_test")],
                           capture_output=True, text=True)
        self.assertEqual(r.returncode, 2, r.stdout + r.stderr)
        self.assertIn("--w64-native", r.stderr)
        self.assertNotIn("vk_compute_test not found", r.stderr)
        self.assertEqual(r.stdout, "")

    def test_wave64_native_with_w64_native_reaches_the_harness_check(self):
        # Not a usage error any more: the run gets as far as looking for vk_compute_test.
        r = subprocess.run([sys.executable, "-I", TOOL, "--wave", "64", "--modes", "native", "--programs", "1",
                            "--w64-native", "--exe", os.path.join(TOOLS, "no-such-build", "vk_compute_test")],
                           capture_output=True, text=True)
        self.assertEqual(r.returncode, 1, r.stdout + r.stderr)
        self.assertIn("vk_compute_test not found", r.stderr)

    def test_wave64_and_both_still_accept_emulate(self):
        for wave in ("64", "both"):
            with self.subTest(wave=wave):
                r = subprocess.run([sys.executable, "-I", TOOL, "--wave", wave, "--modes", "emulate", "--self-test"],
                                   capture_output=True, text=True)
                self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
                self.assertIn("self-test OK", r.stdout)


class Counts(unittest.TestCase):
    """--programs and --ops cannot be below 1: a run of no programs ended in a bare `summary:` with no lines and exit
    0, which reads as every program passing, and --ops 0 built a program with no ops. The other fuzzers refuse both."""

    def run_tool(self, *args):
        return subprocess.run([sys.executable, "-I", TOOL, *args], capture_output=True, text=True, timeout=300)

    def test_zero_or_negative_programs_is_a_usage_error(self):
        for value in ("0", "-1", "-5"):
            with self.subTest(programs=value):
                r = self.run_tool("--programs", value, "--self-test")
                self.assertEqual(r.returncode, 2, r.stdout + r.stderr)
                self.assertIn(f"--programs: must be at least 1 (got {value})", r.stderr)
                self.assertEqual(r.stdout, "")

    def test_zero_ops_is_a_usage_error(self):
        r = self.run_tool("--ops", "0", "--self-test")
        self.assertEqual(r.returncode, 2, r.stdout + r.stderr)
        self.assertIn("--ops: must be at least 1 (got 0)", r.stderr)
        self.assertEqual(r.stdout, "")

    def test_the_count_is_checked_before_the_harness_is_looked_for(self):
        # Both are wrong here: a count below 1 is a usage error whatever the state of the build is.
        r = self.run_tool("--exe", os.path.join(TOOLS, "no-such-build", "vk_compute_test"), "--programs", "0")
        self.assertEqual(r.returncode, 2, r.stdout + r.stderr)
        self.assertNotIn("vk_compute_test not found", r.stderr)

    def test_a_count_of_one_still_runs(self):
        r = self.run_tool("--programs", "1", "--ops", "1", "--self-test")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("self-test OK", r.stdout)


class MissingHarness(unittest.TestCase):
    """Without vk_compute_test the translator oracle stops with one line before any program is generated. It used to
    translate for a while and then end in a FileNotFoundError traceback from the first GPU run."""

    def run_tool(self, *extra, env=None):
        with tempfile.TemporaryDirectory() as tmp:
            exe = os.path.join(tmp, "no-such-build", "vk_compute_test.exe")
            # LLVM_MC points nowhere too: the check has to come before anything that would need llvm-mc.
            env = dict(os.environ, LLVM_MC=os.path.join(tmp, "no-such-llvm-mc"), **(env or {}))
            r = subprocess.run([sys.executable, "-I", TOOL, "--exe", exe, "--programs", "2", *extra],
                               capture_output=True, text=True, env=env, timeout=300)
        return exe, r

    def test_missing_exe_is_one_line_before_anything_runs(self):
        exe, r = self.run_tool()
        self.assertEqual(r.returncode, 1, r.stdout + r.stderr)
        self.assertNotIn("Traceback", r.stderr)
        self.assertIn(f"vk_compute_test not found at {exe}", r.stderr)
        self.assertIn("--target vk_compute_test", r.stderr)
        self.assertEqual(r.stdout, "")

    def test_hardware_oracle_alone_does_not_need_it(self):
        _, r = self.run_tool("--oracle", "hw", env={"LEONIDA_AMD_CLANG": os.path.join(TOOLS, "no-such-clang.exe")})
        self.assertEqual(r.returncode, 77, r.stdout + r.stderr)
        self.assertNotIn("vk_compute_test not found", r.stderr)

    def test_self_test_does_not_need_it(self):
        _, r = self.run_tool("--self-test")
        self.assertNotIn("vk_compute_test not found", r.stderr)
        self.assertNotIn("Traceback", r.stderr)


class GpuEnv(unittest.TestCase):
    """FUZZ-13: vk_compute_test is asked for 32-wide subgroups for wave32 and emulated wave64 programs, as lds_fuzz.py
    and valu_fuzz.py do. On AMD the compute default is 64, and a wave32 group over 32 lanes ran as one subgroup."""

    def program(self, wave, group):
        return X.Program("p", wave, group, [], {}, [])

    def test_wave32_asks_for_32(self):
        env = X.gpu_env(self.program(32, 64), False, "d.bin", base={})
        self.assertEqual(env["VKC_SUBGROUP_SIZE"], "32")

    def test_emulated_wave64_asks_for_32(self):
        env = X.gpu_env(self.program(64, 64), True, "d.bin", base={})
        self.assertEqual(env["VKC_SUBGROUP_SIZE"], "32")

    def test_native_wave64_keeps_the_device_default(self):
        env = X.gpu_env(self.program(64, 64), False, "d.bin", base={"VKC_SUBGROUP_SIZE": "16"})
        self.assertEqual(env["VKC_SUBGROUP_SIZE"], "16")  # whatever the caller had, untouched
        self.assertNotIn("VKC_SUBGROUP_SIZE", X.gpu_env(self.program(64, 64), False, "d.bin", base={}))

    def test_buffer_size_and_dump(self):
        env = X.gpu_env(self.program(32, 96), False, "out/d.bin", base={"KEEP": "1"})
        self.assertEqual(env["VKC_OUT_WORDS"], str(max(96 * len(X.OUT_REGS), 64)))
        self.assertEqual(env["VKC_DUMP"], "out/d.bin")
        self.assertEqual(env["KEEP"], "1")
        self.assertEqual(X.gpu_env(self.program(32, 1), False, "d", base={})["VKC_OUT_WORDS"],
                         str(max(len(X.OUT_REGS), 64)))


class UnpinnedSubgroup(unittest.TestCase):
    """A wave32 run vk_compute_test couldn't pin to 32 lanes is a skip; an emulated wave64 one is still compared."""

    UNPINNED = "device: llvmpipe (LLVM 20.1.2, 256 bits) (subgroup 8..8)\ncompute subgroup size: 32 not available\n"

    def _run_gpu(self, prog, emulate, stdout):
        def fake(cmd, **kw):
            with open(kw["env"]["VKC_DUMP"], "wb") as f:
                f.write(bytes(4 * prog.group * len(X.OUT_REGS)))
            return subprocess.CompletedProcess(cmd, 0, stdout=stdout, stderr="")
        with tempfile.TemporaryDirectory() as work, mock.patch.object(X.subprocess, "run", side_effect=fake):
            return X.run_gpu("vk_compute_test", "p.spv", prog, work, emulate)

    def test_unpinned_wave32_raises(self):
        with self.assertRaises(X.Unpinned) as ctx:
            self._run_gpu(X.make_program(1, 32, 4), False, self.UNPINNED)
        self.assertIn("can't run it at 32 lanes", str(ctx.exception))

    def test_pinned_wave32_gives_its_words(self):
        prog = X.make_program(1, 32, 4)
        words = self._run_gpu(prog, False, "compute subgroup size: 32\n")
        self.assertEqual(len(words), prog.group * len(X.OUT_REGS))

    def test_emulated_wave64_gives_its_words(self):
        prog = X.make_program(1, 64, 4)
        words = self._run_gpu(prog, True, self.UNPINNED)
        self.assertEqual(len(words), prog.group * len(X.OUT_REGS))


class HardwareOracle(unittest.TestCase):
    """--oracle hw: the native kernel text (no GPU needed) and the exit-77 contract."""

    def test_hw_lines_resolve_branches_and_store_through_the_output_pointer(self):
        for seed in range(40):
            prog = X.make_program(seed, 64 if seed % 2 else 32, 24)
            lines = prog.hw_lines(guard=False)
            text = "\n".join(lines)
            self.assertNotIn("@", text)
            self.assertNotIn("buffer_store", text)
            self.assertEqual(sum(ln.startswith("global_store_b32 v22, v20, s[2:3] offset:") for ln in lines),
                             len(X.OUT_REGS))
            self.assertEqual(lines[-1], "s_endpgm")
            targets = re.findall(r"s_cbranch_\w+ (\.L\S+)", text)
            for t in targets:
                self.assertIn(t + ":", lines)
            self.assertEqual(len(targets), len(set(targets)))

    def test_guard_follows_every_instruction_but_the_end(self):
        prog = X.make_program(3, 32, 10)
        plain, guarded = prog.hw_lines(guard=False), prog.hw_lines(guard=True)
        labels = [ln for ln in plain if ln.endswith(":")]
        self.assertEqual(len(guarded), 2 * (len(plain) - len(labels)) - 1 + len(labels))
        self.assertEqual([ln for ln in guarded if ln != "s_nop 7"], plain)

    def test_native_kernel_geometry(self):
        from hw_oracle import asm
        prog = X.make_program(5, 64, 8)
        k = asm.Kernel(prog.hw_lines(), prog.wave, prog.group, prog.group * len(X.OUT_REGS), raw=True)
        text, names, _ = asm.module_text([k], "gfx1150", wave=64)
        self.assertIn(".amdhsa_wavefront_size32 0", text)
        self.assertIn(".wavefront_size: 64", text)
        text32, _, _ = asm.module_text([asm.Kernel(["s_endpgm"], 32, 32, 1, raw=True)], "gfx1150")
        self.assertIn(".amdhsa_wavefront_size32 1", text32)

    def test_tool_exits_77_without_the_oracle_tools(self):
        env = dict(os.environ, LEONIDA_AMD_CLANG=os.path.join(TOOLS, "no-such-clang.exe"))
        r = subprocess.run([sys.executable, "-I", TOOL, "--oracle", "hw", "--programs", "2"],
                           capture_output=True, text=True, env=env)
        self.assertEqual(r.returncode, 77, r.stdout + r.stderr)
        self.assertIn("unavailable", r.stdout)

    def test_unknown_oracle_is_a_usage_error(self):
        r = subprocess.run([sys.executable, "-I", TOOL, "--oracle", "gpu"], capture_output=True, text=True)
        self.assertEqual(r.returncode, 2)

    def test_on_hardware(self):
        from hw_oracle import runner
        try:
            oracle = runner.open_oracle()
        except runner.Unavailable as e:
            self.skipTest(f"hardware oracle unavailable: {e}")
        try:
            stats, fails = {}, []
            progs = [X.make_program(900 + i, (32, 64)[i % 2], 16, dpp=True) for i in range(12)]
            X.run_hw(oracle, progs, stats, fails)
        finally:
            oracle.close()
        self.assertEqual([(p.name, w) for p, _, w, _ in fails], [])
        self.assertEqual(sum(st["programs"] for st in stats.values()), 12)
        self.assertEqual(sum(st["match"] for st in stats.values()), 12)
        for st in stats.values():
            self.assertEqual((st["mismatch"], st["hyp_only"], st["rejected"]), (0, 0, 0))

class HarnessErrors(unittest.TestCase):
    """A program the GPU harness hangs or crashes on is one reported failure, not the end of the run."""

    def test_timeout_is_a_harness_error(self):
        prog = X.make_program(1, 32, 4)
        with tempfile.TemporaryDirectory() as work, \
                mock.patch.object(X.subprocess, "run", side_effect=subprocess.TimeoutExpired(["x"], 60)):
            with self.assertRaises(X.HarnessError) as ctx:
                X.run_gpu("vk_compute_test", "p.spv", prog, work, False)
        self.assertIn("timed out after 60 s", str(ctx.exception))

    def test_missing_dump_is_a_harness_error_with_the_exit_code(self):
        prog = X.make_program(1, 32, 4)
        done = subprocess.CompletedProcess(["x"], 3, stdout="", stderr="device lost\n")
        with tempfile.TemporaryDirectory() as work, mock.patch.object(X.subprocess, "run", return_value=done):
            with self.assertRaises(X.HarnessError) as ctx:
                X.run_gpu("vk_compute_test", "p.spv", prog, work, False)
        self.assertIn("no dump (exit 3): device lost", str(ctx.exception))

    def _main(self, run_gpu, argv):
        """main() with the assembler, translator and GPU replaced: every program 'translates'."""
        def translate(work, emulate):
            spdir = os.path.join(work, "spirv_emu" if emulate else "spirv_nat")
            os.makedirs(spdir, exist_ok=True)
            for name in names:
                open(os.path.join(spdir, name + ".spv"), "wb").close()
            return spdir, ""

        names = []
        out = io.StringIO()
        with tempfile.TemporaryDirectory() as build:
            exe = os.path.join(build, "vk_compute_test.exe")   # only has to exist: run_gpu is replaced
            open(exe, "wb").close()
            with mock.patch.object(X, "assemble", side_effect=lambda mc, p, *a: names.append(p.name)), \
                    mock.patch.object(X, "translate", side_effect=translate), \
                    mock.patch.object(X, "run_gpu", side_effect=run_gpu), \
                    mock.patch.object(sys, "argv", ["crosslane_fuzz.py", "--exe", exe] + argv), \
                    contextlib.redirect_stdout(out):
                code = X.main()
        return code, out.getvalue()

    @staticmethod
    def _expected_words(prog):
        exp, _ = prog.expected()
        words = [0] * (prog.group * len(X.OUT_REGS))
        for (tid, j), v in exp.items():
            words[tid * len(X.OUT_REGS) + j] = v
        return words

    def test_run_goes_on_after_a_hang_and_exits_2(self):
        calls = []

        def run_gpu(exe, spv, prog, work, emulate):
            calls.append(prog.name)
            if len(calls) == 2:
                raise X.HarnessError("vk_compute_test timed out after 60 s")
            return self._expected_words(prog)

        with tempfile.TemporaryDirectory() as keep:
            code, out = self._main(run_gpu, ["--programs", "4", "--wave", "32", "--modes", "native",
                                             "--no-maskdata", "--keep", keep])
            kept = os.listdir(keep)
        self.assertEqual(len(calls), 4)
        self.assertEqual(code, 2)
        self.assertIn(f"{calls[1]} [native] harness error: vk_compute_test timed out after 60 s", out)
        self.assertIn("programs 4, match 3, mismatch 0, hyp_only 0, rejected 0, harness_errors 1", out)
        self.assertEqual(kept, [f"{calls[1]}_native.asm"])

    def test_every_program_skipped_exits_2(self):
        def run_gpu(exe, spv, prog, work, emulate):
            raise X.Unpinned("the device can't run it at 32 lanes")

        code, out = self._main(run_gpu, ["--programs", "4", "--wave", "32", "--modes", "native"])
        self.assertEqual(code, 2)
        self.assertIn("harness_errors 0, skipped 4", out)
        self.assertIn("no program ran", out)
        self.assertNotIn("mismatch;", out)

    def test_skips_beside_matches_exit_0(self):
        def run_gpu(exe, spv, prog, work, emulate):
            if prog.wave == 32:
                raise X.Unpinned("the device can't run it at 32 lanes")
            return self._expected_words(prog)

        code, out = self._main(run_gpu, ["--programs", "4", "--wave", "both", "--no-maskdata"])
        self.assertEqual(code, 0, out)
        self.assertIn("wave32 native: programs 2, match 0, mismatch 0, hyp_only 0, rejected 0, harness_errors 0, "
                      "skipped 2", out)
        self.assertIn("wave64 emulate: programs 2, match 2", out)

    def test_clean_run_still_exits_0(self):
        code, out = self._main(lambda exe, spv, prog, work, emulate: self._expected_words(prog),
                               ["--programs", "3", "--wave", "32", "--modes", "native", "--no-maskdata"])
        self.assertEqual(code, 0)
        self.assertIn("match 3, mismatch 0, hyp_only 0, rejected 0, harness_errors 0", out)


if __name__ == "__main__":
    unittest.main()
