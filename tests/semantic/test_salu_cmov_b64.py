"""Semantic tests for RDNA2 s_cmov_b64 on 64-bit scalar *data* pairs.

Rule: D[63:0] = SCC ? S0[63:0] : D[63:0]; SCC is read, not written (RDNA2 ISA SOP1,
LLVM AMDGPU SOPInstructions.td `S_CMOV_B64 : SOP1_64` under `Uses = [SCC]`).

Only plain data is supported. A lane-mask register holds one lane's bit in this
translator, so selecting it as 64-bit data would silently read back a per-lane
bool instead of the wave's mask (issue #210). Those operands stay rejected.

Every program here is hand-written; the translator and mini interpreter from
this repository do the work. No LLVM, SPIR-V tools, GPU or game files are used.

Run directly:
    python -I tests/semantic/test_salu_cmov_b64.py -v
"""
import os
import re
import sys
import tempfile
import unittest
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from support import STUB_DXBC, mrt0_words, rdna2spirv  # noqa: E402

MASK32 = 0xFFFFFFFF
MASK64 = 0xFFFFFFFFFFFFFFFF

_TMP_DIR = None
_PS_PATH = None


def setUpModule():
    global _TMP_DIR, _PS_PATH
    _TMP_DIR = tempfile.TemporaryDirectory()
    dis = os.path.join(_TMP_DIR.name, "disasm")
    os.mkdir(dis)
    _PS_PATH = os.path.join(dis, "ps_cmov64.s")
    with open(os.path.join(_TMP_DIR.name, "ps_cmov64.dxbc"), "wb") as f:
        f.write(STUB_DXBC)


def tearDownModule():
    global _TMP_DIR
    if _TMP_DIR is not None:
        _TMP_DIR.cleanup()
        _TMP_DIR = None


def translate_ps(lines):
    if _TMP_DIR is None:
        setUpModule()
    asm = "; ps_cmov64\n" + "".join(f"\t{ln}\n" for ln in lines)
    with open(_PS_PATH, "w", encoding="utf-8", newline="") as f:
        f.write(asm)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)
        return rdna2spirv.translate(_PS_PATH)


def run_ps(lines):
    spv = translate_ps(lines + ["exp mrt0 v0, v1, v2, v3 done vm", "s_endpgm"])
    return mrt0_words(spv)


def scc_line(scc):
    return "s_cmp_eq_u32 1, 1" if scc else "s_cmp_eq_u32 0, 1"


def load_pair_lines(reg, val64):
    return [f"s_mov_b32 s{reg}, {val64 & MASK32:#x}",
            f"s_mov_b32 s{reg + 1}, {(val64 >> 32) & MASK32:#x}"]


def export_pair_lines(reg):
    return [f"v_mov_b32 v0, s{reg}", f"v_mov_b32 v1, s{reg + 1}"]


def ref_cmov64(old, src, scc):
    return (src if scc else old) & MASK64


class DataPairTests(unittest.TestCase):
    def test_plain_data_pair_is_accepted(self):
        # The explicit non-mask case: both dwords come from plain immediate moves,
        # not from EXEC, VCC or a compare. D is an ordinary SGPR data pair.
        old = 0xDEADBEEF_11223344
        src = 0x55667788_90ABCDEF
        for scc in (0, 1):
            with self.subTest(scc=scc):
                want = ref_cmov64(old, src, scc)
                words = run_ps(
                    load_pair_lines(2, old)
                    + load_pair_lines(4, src)
                    + [scc_line(scc), "s_cmov_b64 s[2:3], s[4:5]",
                       "s_cselect_b32 s10, 1, 0"]
                    + export_pair_lines(2)
                    + ["v_mov_b32 v2, s10", "v_mov_b32 v3, 0"]
                )
                self.assertEqual(words, [want & MASK32, want >> 32, scc, 0])

    def test_edge_values(self):
        pairs = [
            (0, 0xFFFFFFFF_FFFFFFFF),
            (0x00000001_00000000, 0x80000000_7FFFFFFF),
            (0x7FFFFFFF_FFFFFFFF, 0x80000000_00000000),
            (0x00000002_80000001, 0x00000002_80000002),
        ]
        for old, src in pairs:
            for scc in (0, 1):
                with self.subTest(old=hex(old), src=hex(src), scc=scc):
                    want = ref_cmov64(old, src, scc)
                    words = run_ps(
                        load_pair_lines(2, old)
                        + load_pair_lines(4, src)
                        + [scc_line(scc), "s_cmov_b64 s[2:3], s[4:5]"]
                        + export_pair_lines(2)
                        + ["v_mov_b32 v2, 0", "v_mov_b32 v3, 0"]
                    )
                    self.assertEqual(words, [want & MASK32, want >> 32, 0, 0])

    def test_immediate_and_null_sources(self):
        old = 0x55667788_11223344
        # shader_ir leaves a literal wider than 32 bits as a raw operand, which
        # src64_data does not accept (the same holds for s_mov_b64); use the
        # immediate forms the parser actually delivers.
        for src_asm, src in (("0x12345678", 0x12345678), ("42", 42),
                             ("0", 0), ("-1", MASK64), ("-2", MASK64 - 1), ("null", 0)):
            for scc in (0, 1):
                with self.subTest(src=src_asm, scc=scc):
                    want = ref_cmov64(old, src, scc)
                    words = run_ps(
                        load_pair_lines(2, old)
                        + [scc_line(scc), f"s_cmov_b64 s[2:3], {src_asm}"]
                        + export_pair_lines(2)
                        + ["v_mov_b32 v2, 0", "v_mov_b32 v3, 0"]
                    )
                    self.assertEqual(words, [want & MASK32, want >> 32, 0, 0])

    def test_destination_aliasing(self):
        a64 = 0x55667788_11223344
        b64 = 0xDDEEFF00_99AABBCC
        # s[3:4] = (s4 << 32) | s3, with s4 = low(a64), s3 = low(b64).
        overlap_val = ((a64 & MASK32) << 32) | (b64 & MASK32)
        for source, src_val in (("s[4:5]", a64), ("s[3:4]", overlap_val)):
            for scc in (0, 1):
                with self.subTest(source=source, scc=scc):
                    want = ref_cmov64(a64, src_val, scc)
                    lines = load_pair_lines(3, b64) + load_pair_lines(4, a64)
                    lines += [scc_line(scc), f"s_cmov_b64 s[4:5], {source}"]
                    lines += export_pair_lines(4) + ["v_mov_b32 v2, 0", "v_mov_b32 v3, 0"]
                    words = run_ps(lines)
                    self.assertEqual(words, [want & MASK32, want >> 32, 0, 0])

    def test_constant_pairs_are_not_mistaken_for_masks(self):
        # 0 / -1 are EITHER in tools/lane_masks.py (the same bits as data), so
        # they must not be mistaken for lane masks here.
        for src_asm, src in (("-1", MASK64), ("0", 0)):
            for scc in (0, 1):
                with self.subTest(src=src_asm, scc=scc):
                    want = ref_cmov64(0x11223344_55667788, src, scc)
                    words = run_ps(
                        load_pair_lines(2, 0x11223344_55667788)
                        + [f"s_mov_b64 s[4:5], {src_asm}", scc_line(scc),
                           "s_cmov_b64 s[2:3], s[4:5]"]
                        + export_pair_lines(2)
                        + ["v_mov_b32 v2, 0", "v_mov_b32 v3, 0"]
                    )
                    self.assertEqual(words, [want & MASK32, want >> 32, 0, 0])

    def test_scc_is_not_written(self):
        old, src = 0x11111111_22222222, 0x33333333_44444444
        for scc in (0, 1):
            with self.subTest(scc=scc):
                want = ref_cmov64(old, src, scc)
                words = run_ps(
                    load_pair_lines(2, old)
                    + load_pair_lines(4, src)
                    + [scc_line(scc), "s_cmov_b64 s[2:3], s[4:5]",
                       "s_addc_u32 s10, 10, 20"]
                    + export_pair_lines(2)
                    + ["v_mov_b32 v2, s10", "v_mov_b32 v3, 0"]
                )
                self.assertEqual(words, [want & MASK32, want >> 32, 30 + scc, 0])


class RejectionTests(unittest.TestCase):
    MASK_SOURCE = "s_cmov_b64 lane-mask source"
    MASK_DEST = "s_cmov_b64 lane-mask destination"
    DEST = "s_cmov_b64 destination"

    def test_lane_mask_source_rejected_with_message(self):
        cmp = "v_cmp_ne_u32_e64 s[20:21], 0, v0"
        cases = [
            ([cmp, "s_mov_b64 s[22:23], s[20:21]"], "s[22:23]"),
            (["s_mov_b64 s[22:23], exec"], "s[22:23]"),
            ([], "exec"),
            # A zero-extended 32-bit mask: one dword is a mask, the other is
            # an EITHER constant. holds_lane_mask() alone misses this pair.
            ([cmp, "s_mov_b32 s2, s20", "s_mov_b32 s3, 0"], "s[2:3]"),
            ([cmp, "s_mov_b32 s2, 0", "s_mov_b32 s3, s20"], "s[2:3]"),
        ]
        for prologue, src in cases:
            with self.subTest(src=src, prologue=prologue[-1] if prologue else "none"):
                lines = prologue + [scc_line(1), f"s_cmov_b64 s[2:3], {src}",
                                    "exp mrt0 v0, v0, v0, v0 done vm", "s_endpgm"]
                with self.assertRaisesRegex(rdna2spirv.Unsupported, re.escape(self.MASK_SOURCE)):
                    translate_ps(lines)

    def test_mask_destination_rejected_with_message(self):
        cmp = "v_cmp_ne_u32_e64 s[20:21], 0, v0"
        cases = [
            [cmp, "s_cmov_b64 s[20:21], s[4:5]"],
            [cmp, "s_mov_b32 s2, s20", "s_mov_b32 s3, 0", "s_cmov_b64 s[2:3], s[4:5]"],
        ]
        for lines in cases:
            with self.subTest(lines=lines[-1]):
                full = lines + ["exp mrt0 v0, v0, v0, v0 done vm", "s_endpgm"]
                with self.assertRaisesRegex(rdna2spirv.Unsupported, re.escape(self.MASK_DEST)):
                    translate_ps(full)

    def test_invalid_destinations_rejected_with_message(self):
        for line in ("s_cmov_b64 exec, s[4:5]", "s_cmov_b64 vcc, s[4:5]",
                     "s_cmov_b64 m0, s[4:5]", "s_cmov_b64 s4, s[4:5]",
                     "s_cmov_b64 null, s[4:5]"):
            with self.subTest(line=line):
                full = [line, "exp mrt0 v0, v0, v0, v0 done vm", "s_endpgm"]
                with self.assertRaisesRegex(rdna2spirv.Unsupported, re.escape(self.DEST)):
                    translate_ps(full)


if __name__ == "__main__":
    unittest.main()
