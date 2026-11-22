"""MUBUF bounds checking per V# OOB_SELECT (word3 bits [29:28]), as measured natively on RDNA2 (gfx1036,
docs/buffer_bounds.md; Mesa's GFX11 column, not its GFX10 one).

With idx = vindex (0 without idxen), off = inst_offset + vaddr offset (0 without offen) and offs = off + soffset, a
dword access is out of range when:

    0 STRUCTURED_WITH_OFFSET   idx >= NUM_RECORDS or off + 4 > STRIDE
    1 STRUCTURED               idx >= NUM_RECORDS
    2 DISABLED                 NUM_RECORDS == 0
    3 RAW                      offs + 4 > NUM_RECORDS      (idx * STRIDE is in the address only)

An out-of-range load returns 0 without reading, an out-of-range store is dropped. The table below is written out by
hand from those rules (not computed), and every row runs twice: with word3 a literal the translator folds, and with
word3 or-ed together at run time, which it selects with OpSelect. The multi-dword tail of RAW loads is in
test_value_semantics.py (measured on an RX 6800); mode 0's is by analogy.
"""
import re
import unittest

import support
import spirv_mini_interp as si
import test_value_semantics as tv         # module only: importing its classes would collect them a second time

BASE = tv.VEC_REGION
OUT_V = 0x31016FAC                          # the output V# in s[0:3]: RAW, base 0
W3 = 0x01016FAC                             # word3 without OOB_SELECT: RESOURCE_LEVEL, FORMAT 32_FLOAT, DST_SEL xyzw
NONE = None

# (mode, stride, num_records, idx, off, soffset, inst_offset, in_range); idx/off NONE: no idxen / no offen
CASES = [
    # mode 0: idx < NR and off + 4 <= stride (off is vaddr + inst_offset, not soffset)
    (0, 8, 4, 1, 4, 0, 0, True),
    (0, 8, 4, 1, 8, 0, 0, False),      # off 8 + 4 > stride 8
    (0, 12, 4, 0, 8, 0, 0, True),      # the last dword of the record fits: 8 + 4 <= 12
    (0, 10, 4, 0, 8, 0, 0, False),     # ... and 8 + 4 > 10 does not
    (0, 8, 4, 4, 0, 0, 0, False),      # idx 4 >= NR 4
    (0, 8, 4, 3, 4, 0, 0, True),
    (0, 8, 4, 1, 0, 16, 0, True),      # soffset is not part of off
    (0, 8, 4, 1, 0, 0, 8, False),      # inst_offset is
    (0, 8, 4, NONE, 4, 0, 0, True),    # offen only: idx 0 < NR
    (0, 8, 4, NONE, 8, 0, 0, False),
    (0, 0, 4, 0, 0, 0, 0, False),      # stride 0: 0 + 4 > 0
    (0, 8, 0, 0, 0, 0, 0, False),      # NR 0
    # mode 1: idx < NR
    (1, 8, 4, 3, 12, 0, 0, True),      # off past the stride is not checked
    (1, 8, 4, 4, 0, 0, 0, False),
    (1, 8, 4, NONE, 16, 0, 0, True),   # no idxen: idx 0 < NR, however far off is
    (1, 8, 0, NONE, 0, 0, 0, False),
    (1, 0, 4, 3, 0, 0, 0, True),       # stride 0 still checks the index
    (1, 0, 4, 4, 0, 0, 0, False),
    # mode 2: only NR == 0 is out of range
    (2, 8, 4, 5, 0, 0, 0, True),
    (2, 0, 4, NONE, 0x20, 0, 0, True),
    (2, 8, 0, NONE, 0, 0, 0, False),
    # mode 3: offs + 4 <= NR; idx and stride do not take part
    (3, 0, 0x10, NONE, 0xC, 0, 0, True),
    (3, 0, 0x10, NONE, 0x10, 0, 0, False),
    (3, 0, 0xC, NONE, 8, 0, 0, True),    # the last dword fits: 8 + 4 <= 12
    (3, 0, 0xA, NONE, 8, 0, 0, False),   # a dword that only starts inside: 8 + 4 > 10
    (3, 8, 0x10, 7, 4, 0, 0, True),    # idx 7 >= NR 16 is irrelevant; address = 7 * 8 + 4
    (3, 8, 0x10, NONE, 0x10, 0, 0, False),
    (3, 8, 0x10, 0, 4, 8, 0, True),    # soffset counts: offs 12
    (3, 8, 0x10, 0, 4, 12, 0, False),  # offs 16
    (3, 0, 0x10, NONE, 0, 0, 8, True),
    (3, 0, 0x10, NONE, 0, 0, 0x10, False),
    (3, 8, 8, 3, 0, 0, 0, True),       # idx 3 >= NR 8 is not a check here
    (3, 0, 0, NONE, 0, 0, 0, False),
]


def operands(idx, off, soffset, inst):
    """vaddr operand text, modifiers and the v1/v2 setup for one case."""
    regs, mods = [], []
    if idx is not NONE:
        regs.append(idx)
        mods.append("idxen")
    if off is not NONE:
        regs.append(off)
        mods.append("offen")
    vaddr = {0: "off", 1: "v1", 2: "v[1:2]"}[len(regs)]
    setup = [f"v_mov_b32 v{1 + i}, {r:#x}" for i, r in enumerate(regs)]
    if soffset:
        setup.append(f"s_mov_b32 s8, {soffset:#x}")
    return vaddr, " ".join(mods + ([f"offset:{inst}"] if inst else [])), setup, ("s8" if soffset else "0")


def v_sharp(mode, stride, nr, runtime, fmt=22):
    """Lines that put the V# in s[4:7]; with runtime the mode is or-ed in, so the translator cannot see it."""
    w3 = 0x01000FAC | fmt << 12
    lines = [f"s_mov_b32 s4, {BASE:#x}", "s_mov_b32 s5, " + hex(stride << 16), f"s_mov_b32 s6, {nr:#x}"]
    if runtime:
        lines += [f"s_mov_b32 s7, {w3:#x}", f"s_mov_b32 s9, {mode << 28:#x}", "s_or_b32 s7, s7, s9"]
    else:
        lines += [f"s_mov_b32 s7, {w3 | mode << 28:#x}"]
    return lines


MODE_EXTRACT = re.compile(r"OpBitFieldUExtract %uint %t\d+ %c1c %c2\b")     # word3 >> 28 & 3 at run time


class OobSelect(unittest.TestCase):
    def translate(self, lines):
        return support.translate_lines("cs_oob_select_t", ["v_cmpx_eq_u32_e32 0, v0", *lines, "s_endpgm"],
                                       fssm=tv.CS_BOUNDS_FSSM)

    def run_spv(self, spv, preload=True):
        memory = tv.ReadLog({BASE + k: tv._pattern_word(k) for k in range(-4, 0x80)} if preload else {})
        support.run_fragment(spv, inputs={"%wg_id": (0, 0, 0), "%local_id": (0, 0, 0)},
                             push_constants=[0, 0, 0x100, OUT_V] + [0] * 28, memory=memory)
        return memory

    def load_case(self, case, runtime):
        mode, stride, nr, idx, off, soffset, inst, ok = case
        vaddr, mods, setup, soff = operands(idx, off, soffset, inst)
        lines = [*v_sharp(mode, stride, nr, runtime), *setup,
                 f"buffer_load_dword v2, {vaddr}, s[4:7], {soff} {mods}", "s_waitcnt vmcnt(0)",
                 "buffer_store_dword v2, off, s[0:3], 0"]
        spv = self.translate(lines)
        # the output store through s[0:3] reads its mode at run time; the probed V# only when it is not a literal
        self.assertEqual(len(MODE_EXTRACT.findall(spv)), 1 + runtime, case)
        memory = self.run_spv(spv)
        addr = (0 if idx is NONE else idx * stride) + (off or 0) + soffset + inst
        want = tv._pattern_word(addr) if ok else 0
        self.assertEqual(f"{dict.get(memory, 0, 0):#010x}", f"{want:#010x}", (case, runtime))
        self.assertEqual([a - BASE for a in memory.reads], [addr] if ok else [], (case, runtime))

    def test_loads_literal_v_sharp(self):
        for case in CASES:
            with self.subTest(case=case):
                self.load_case(case, runtime=False)

    def test_loads_runtime_v_sharp(self):
        for case in CASES:
            with self.subTest(case=case):
                self.load_case(case, runtime=True)

    def store_case(self, case, runtime):
        mode, stride, nr, idx, off, soffset, inst, ok = case
        vaddr, mods, setup, soff = operands(idx, off, soffset, inst)
        lines = [*v_sharp(mode, stride, nr, runtime), *setup, "v_mov_b32 v3, 0x11111111",
                 f"buffer_store_dword v3, {vaddr}, s[4:7], {soff} {mods}"]
        spv = self.translate(lines)
        self.assertEqual(len(MODE_EXTRACT.findall(spv)), int(runtime), case)
        memory = self.run_spv(spv, preload=False)
        addr = (0 if idx is NONE else idx * stride) + (off or 0) + soffset + inst
        self.assertEqual(dict(memory), {BASE + addr: 0x11111111} if ok else {}, (case, runtime))

    def test_stores_literal_v_sharp(self):
        for case in CASES:
            with self.subTest(case=case):
                self.store_case(case, runtime=False)

    def test_stores_runtime_v_sharp(self):
        for case in CASES:
            with self.subTest(case=case):
                self.store_case(case, runtime=True)

    def test_multi_dword_load_follows_the_mode(self):
        # mode 0 is per dword (by analogy with the measured RAW tail): x2 at idx 1, off 4. With stride 12 both dwords
        # end inside (8, 12); with stride 8 the second ends at 12 > 8 and reads 0
        for runtime in (False, True):
            for stride, want in ((12, [tv._pattern_word(16), tv._pattern_word(20)]), (8, [tv._pattern_word(12), 0])):
                lines = [*v_sharp(0, stride, 4, runtime), "v_mov_b32 v1, 1", "v_mov_b32 v2, 4",
                         "buffer_load_dwordx2 v[4:5], v[1:2], s[4:7], 0 idxen offen", "s_waitcnt vmcnt(0)",
                         "buffer_store_dwordx2 v[4:5], off, s[0:3], 0"]
                memory = self.run_spv(self.translate(lines))
                self.assertEqual([dict.get(memory, 4 * i, 0) for i in range(2)], want, (runtime, stride))
        # mode 1 checks the index once for the whole load
        for runtime in (False, True):
            lines = [*v_sharp(1, 8, 4, runtime), "v_mov_b32 v1, 1", "v_mov_b32 v2, 4",
                     "buffer_load_dwordx2 v[4:5], v[1:2], s[4:7], 0 idxen offen", "s_waitcnt vmcnt(0)",
                     "buffer_store_dwordx2 v[4:5], off, s[0:3], 0"]
            memory = self.run_spv(self.translate(lines))
            self.assertEqual([dict.get(memory, 4 * i, 0) for i in range(2)],
                             [tv._pattern_word(12), tv._pattern_word(16)], runtime)

    # measured on the Radeon 880M (gfx1150), a valid proxy for RDNA2 here (docs/buffer_bounds.md): a dword access
    # that straddles the edge is per dword for loads AND stores, a typed element is all or nothing
    STORE_CASES = [
        # (mode, stride, nr, idx, off, dwords, written dword count)
        (3, 0, 0x20, NONE, 0x14, 4, 3), (3, 0, 0x20, NONE, 0x1C, 4, 1), (3, 0, 0x20, NONE, 0x20, 4, 0),
        (3, 0, 0x20, NONE, 0x1C, 2, 1), (3, 0, 0x20, NONE, 0x10, 4, 4),
        (0, 16, 8, 1, 4, 4, 3), (0, 16, 8, 1, 12, 4, 1), (0, 16, 8, 1, 16, 4, 0), (0, 16, 8, 1, 8, 2, 2),
        (0, 16, 8, 1, 12, 2, 1),
        (1, 16, 8, 1, 16, 4, 4), (2, 16, 8, 1, 16, 4, 4),
    ]

    def test_multi_dword_stores_are_per_dword(self):
        for runtime in (False, True):
            for mode, stride, nr, idx, off, n, written in self.STORE_CASES:
                vaddr, mods, setup, soff = operands(idx, off, 0, 0)
                regs = f"v[3:{2 + n}]"
                lines = [*v_sharp(mode, stride, nr, runtime), *setup,
                         *[f"v_mov_b32 v{3 + k}, {0x11111111 * (k + 1):#x}" for k in range(n)],
                         f"buffer_store_dwordx{n} {regs}, {vaddr}, s[4:7], 0 {mods}"]
                memory = self.run_spv(self.translate(lines), preload=False)
                base = BASE + (0 if idx is NONE else idx * stride) + off
                want = {base + 4 * k: 0x11111111 * (k + 1) for k in range(written)}
                self.assertEqual(dict(memory), want, (mode, stride, nr, idx, off, n, runtime))

    FORMAT_CASES = [(3, 0, 0x20, NONE, 0x10, True), (3, 0, 0x20, NONE, 0x14, False), (3, 0, 0x20, NONE, 0x20, False),
                    (0, 16, 8, 1, 0, True), (0, 16, 8, 1, 4, False), (1, 16, 8, 1, 16, True)]

    def test_typed_element_is_all_or_nothing(self):
        # 32_32_32_32_UINT (75), a 16-byte element: in only if all 16 bytes end inside (buffer_load_format_xyzw and
        # buffer_store_format_xyzw alike)
        for runtime in (False, True):
            for mode, stride, nr, idx, off, ok in self.FORMAT_CASES:
                vaddr, mods, setup, soff = operands(idx, off, 0, 0)
                addr = (0 if idx is NONE else idx * stride) + off
                lines = [*v_sharp(mode, stride, nr, runtime, fmt=75), *setup,
                         f"buffer_load_format_xyzw v[3:6], {vaddr}, s[4:7], 0 {mods}", "s_waitcnt vmcnt(0)",
                         "buffer_store_dwordx4 v[3:6], off, s[0:3], 0"]
                memory = self.run_spv(self.translate(lines))
                got = [dict.get(memory, 4 * k, 0) for k in range(4)]
                want = [tv._pattern_word(addr + 4 * k) for k in range(4)] if ok else [0] * 4
                self.assertEqual(got, want, ("load", mode, stride, nr, idx, off, runtime))
                lines = [*v_sharp(mode, stride, nr, runtime, fmt=75), *setup,
                         *[f"v_mov_b32 v{3 + k}, {0x11111111 * (k + 1):#x}" for k in range(4)],
                         f"buffer_store_format_xyzw v[3:6], {vaddr}, s[4:7], 0 {mods}"]
                memory = self.run_spv(self.translate(lines), preload=False)
                want = {BASE + addr + 4 * k: 0x11111111 * (k + 1) for k in range(4)} if ok else {}
                self.assertEqual(dict(memory), want, ("store", mode, stride, nr, idx, off, runtime))

    def test_swizzled_literal_v_sharp_is_unsupported_in_every_mode(self):
        # SWIZZLE_ENABLE (WORD1 bit 31) and ADD_TID_ENABLE (WORD3 bit 23), Mesa gfx10-rsrc.json lines 326 and 337:
        # swizzled addressing and bounds are not modelled, so a literal V# that asks for them is rejected in every mode
        # instead of getting the unswizzled check. A run-time V# with those bits is translated unswizzled (known gap).
        import rdna2spirv
        for mode in range(4):
            for w1, w3 in ((1 << 31, 0), (0, 1 << 23)):
                with self.subTest(mode=mode, word1=hex(w1), word3=hex(w3)):
                    lines = ["s_mov_b32 s4, 0x1000", f"s_mov_b32 s5, {w1:#x}", "s_mov_b32 s6, 16",
                             f"s_mov_b32 s7, {0x01016FAC | mode << 28 | w3:#x}", "buffer_load_dword v2, off, s[4:7], 0"]
                    with self.assertRaises(rdna2spirv.Unsupported):
                        self.translate(lines)


if __name__ == "__main__":
    unittest.main()
