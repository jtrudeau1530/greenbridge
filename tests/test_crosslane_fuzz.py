"""tools/crosslane_fuzz.py: the reference model of the cross-lane ops, checked against values worked out by hand
from the RDNA ISA rules in its docstring. No GPU, no llvm-mc, no game files.

    python -I tests/test_crosslane_fuzz.py
"""
import os
import random
import re
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
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


if __name__ == "__main__":
    unittest.main()
