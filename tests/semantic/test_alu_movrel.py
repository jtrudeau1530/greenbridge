"""Value tests for the M0-relative VGPR moves (RDNA2 ISA, VOP1):

  v_movrels_b32   D = VGPR[S0.addr + M0]
  v_movreld_b32   VGPR[D.addr + M0] = S0
  v_movrelsd_b32  VGPR[D.addr + M0] = VGPR[S0.addr + M0]

Each case is a synthetic hand-written pixel-shader listing: v_mov_b32 loads a distinct sentinel into v1..v8 (and
v0 where a case needs it), s_mov_b32 sets M0, one movrel instruction runs, and four registers go out through
exp mrt0. The listing is translated by tools/rdna2spirv.py and executed by spirv_mini_interp.py. The exported
registers include the indexed one and its neighbours, so the tests show both which register was read and that
a write landed only in VGPR[D.addr + M0]. M0 = 0 and several non-zero M0 values are covered.

The expected words are hand-computed constants; a small plain-Python model of the register file (no translator)
is checked against the same constants, so a typo in either is caught. Standard library only.

Not tested:
- v_movrelsd_2_b32 (the GFX10 variant with separate source and destination offsets in M0) is not handled by the
  translator yet: it raises Unsupported (a follow-up for whoever implements it).
- An index past the VGPR file. The translator emits an unchecked OpAccessChain into its 256-entry %vgpr array,
  so S0.addr + M0 or D.addr + M0 >= 256 is undefined in SPIR-V (the mini interpreter raises
  UnsupportedConstruct). An index below 256 but beyond the VGPRs the shader uses reads or writes the translator's
  zero-initialised array; the hardware's range check against the wave's VGPR allocation is not modelled. No case
  relies on either.
"""
import gc
import os
import sys
import tempfile
import unittest
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from support import STUB_DXBC, mrt0_words, rdna2spirv  # noqa: E402  (support puts tools/ on sys.path)

MASK32 = 0xFFFFFFFF
NUM_VGPRS = 256         # size of the translator's %vgpr array


def sentinel(i):
    """The value v_mov_b32 loads into v<i> before the movrel instruction."""
    return 0xC0DE0000 | (i * 0x111)


SENTINELS = {i: sentinel(i) for i in range(1, 9)}
V0_VALUE = 0x5EED0000          # loaded into v0 by the movreld cases (the value that gets written)


def model(op, dst, src, m0, init):
    """Plain-Python register file: apply `op dst, src` with M0 = m0 to a copy of `init` ({reg: value})."""
    vgpr = [0] * NUM_VGPRS
    for r, v in init.items():
        vgpr[r] = v & MASK32
    if op == "v_movrels_b32":
        vgpr[dst] = vgpr[src + m0]
    elif op == "v_movreld_b32":
        vgpr[dst + m0] = vgpr[src]
    elif op == "v_movrelsd_b32":
        vgpr[dst + m0] = vgpr[src + m0]
    else:
        raise ValueError(op)
    return vgpr


S = SENTINELS
# (op, dst, src, m0, load v0?, exported registers, expected words) -- expected words worked out by hand.
CASES = [
    # movrels: D = VGPR[S0 + M0]
    ("v_movrels_b32", 0, 1, 0, False, (0, 1, 2, 3), (S[1], S[1], S[2], S[3])),     # M0 = 0 reads S0 itself
    ("v_movrels_b32", 0, 2, 3, False, (0, 2, 4, 5), (S[5], S[2], S[4], S[5])),     # v2 + 3 = v5
    ("v_movrels_b32", 0, 1, 7, False, (0, 1, 7, 8), (S[8], S[1], S[7], S[8])),     # v1 + 7 = v8
    ("v_movrels_b32", 9, 0, 4, True, (9, 0, 4, 5), (S[4], V0_VALUE, S[4], S[5])),  # base v0, M0 alone indexes
    # movreld: VGPR[D + M0] = S0; neighbours keep their sentinels
    ("v_movreld_b32", 1, 0, 0, True, (0, 1, 2, 3), (V0_VALUE, V0_VALUE, S[2], S[3])),
    ("v_movreld_b32", 3, 0, 2, True, (3, 4, 5, 6), (S[3], S[4], V0_VALUE, S[6])),  # writes v5, not v3
    ("v_movreld_b32", 1, 8, 6, False, (1, 6, 7, 8), (S[1], S[6], S[8], S[8])),     # v1 + 6 = v7 <- v8
    # movrelsd: VGPR[D + M0] = VGPR[S0 + M0]; M0 offsets both sides
    ("v_movrelsd_b32", 2, 1, 0, False, (1, 2, 3, 4), (S[1], S[1], S[3], S[4])),
    ("v_movrelsd_b32", 6, 1, 2, False, (6, 7, 8, 3), (S[6], S[7], S[3], S[3])),    # v8 <- v3
    ("v_movrelsd_b32", 1, 3, 4, False, (1, 3, 5, 7), (S[1], S[3], S[7], S[7])),    # v5 <- v7
]

_TMP = None


def setUpModule():
    global _TMP
    _TMP = tempfile.TemporaryDirectory()
    os.mkdir(os.path.join(_TMP.name, "disasm"))
    with open(os.path.join(_TMP.name, "ps_movrel.dxbc"), "wb") as f:
        f.write(STUB_DXBC)


def tearDownModule():
    global _TMP
    if _TMP is not None:
        gc.collect()
        _TMP.cleanup()
        _TMP = None


def translate_listing(body):
    """Write `body` (instruction lines) as ps_movrel.s, translate it and return the SPIR-V text."""
    lines = ["; test_movrel"] + [f"\t{line}" for line in body] + ["\ts_endpgm", ""]
    path = os.path.join(_TMP.name, "disasm", "ps_movrel.s")
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write("\n".join(lines))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)
        spv = rdna2spirv.translate(path)
        gc.collect()
    return spv


def initial_regs(load_v0):
    regs = dict(SENTINELS)
    if load_v0:
        regs[0] = V0_VALUE
    return regs


def run_case(op, dst, src, m0, load_v0, exported):
    body = [f"v_mov_b32 v{r}, {v:#x}" for r, v in sorted(initial_regs(load_v0).items())]
    body.append(f"s_mov_b32 m0, {m0}")
    body.append(f"{op} v{dst}, v{src}")
    body.append("exp mrt0 " + ", ".join(f"v{r}" for r in exported) + " done vm")
    return tuple(mrt0_words(translate_listing(body)))


def case_id(case):
    op, dst, src, m0 = case[:4]
    return f"{op} v{dst}, v{src} (m0={m0})"


class ModelMatchesHandConstants(unittest.TestCase):
    def test_model(self):
        for case in CASES:
            op, dst, src, m0, load_v0, exported, expected = case
            with self.subTest(case=case_id(case)):
                vgpr = model(op, dst, src, m0, initial_regs(load_v0))
                self.assertEqual(tuple(vgpr[r] for r in exported), expected)

    def test_sentinels_distinct(self):
        values = list(SENTINELS.values()) + [V0_VALUE]
        self.assertEqual(len(set(values)), len(values))


class TranslatedValues(unittest.TestCase):
    def check(self, op):
        cases = [c for c in CASES if c[0] == op]
        self.assertTrue(cases)
        for case in cases:
            _, dst, src, m0, load_v0, exported, expected = case
            with self.subTest(case=case_id(case)):
                got = run_case(op, dst, src, m0, load_v0, exported)
                self.assertEqual(got, expected, f"{case_id(case)}: got {[hex(w) for w in got]}, "
                                                f"expected {[hex(w) for w in expected]}")

    def test_v_movrels_b32(self):
        self.check("v_movrels_b32")

    def test_v_movreld_b32(self):
        self.check("v_movreld_b32")

    def test_v_movrelsd_b32(self):
        self.check("v_movrelsd_b32")


if __name__ == "__main__":
    unittest.main()
