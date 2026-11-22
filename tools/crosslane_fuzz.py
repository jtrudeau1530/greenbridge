r"""Differential fuzzer for the translator's wave and cross-lane operations (FUZZ-7 in #156 / #374).

Generates seeded random compute programs built from cross-lane RDNA2 instructions (v_readlane, v_writelane,
v_readfirstlane, ballots through v_cmp / v_cmpx, s_bcnt1 / s_ff1 of masks, v_mbcnt, v_cndmask on a mask, masks
read as data, v_permlane16 / v_permlanex16, forward s_cbranch_execz / vccz / vccnz around them, and with --dpp
DPP16 / DPP8 sources) under random EXEC masks: constants per half, v_cmpx, nested s_and_saveexec and its
restore, partial waves (a group size that isn't a multiple of the wave size) and groups of two or more waves.
Each program runs through a Python model of the ISA (the reference, below) and through the translator on the
local GPU: llvm-mc assembles it, tools/rdna2spirv.py translates it (wave32 natively; wave64 with
--wave64-emulate, and natively with --w64-native), runtime/build/vk_compute_test runs it with VKC_OUT_WORDS /
VKC_DUMP, and every word every thread stores is compared. docs/shader_validation.md has the details.

    python -I tools/crosslane_fuzz.py --programs 200 --seed 1 [--wave 32|64|both] [--modes native,emulate]
    python -I tools/crosslane_fuzz.py --self-test        # the model against hand-computed values, no GPU
    python -I tools/crosslane_fuzz.py --oracle hw        # model vs the AMD GPU natively (wave32 and wave64); exit 77
                                                         # when tools/hw_oracle's tools are missing. --oracle translator,hw
                                                         # does both. No llvm-mc, Vulkan or vk_compute_test for hw alone.

Exit code: 0 when every program that ran matches (or differs only in a program with a hypothesis op), 1 on a
mismatch or an untranslated program, 2 when the only failures are harness errors: vk_compute_test timed out or
gave no dump for a program. Such a program is listed with the others (and kept with --keep) and the run goes on.
A missing vk_compute_test for the translator oracle is one line and exit 1, before any program is generated.
A wave32 program is skipped where vk_compute_test can't pin 32-lane subgroups ("compute subgroup size: 32 not
available", e.g. Mesa llvmpipe at 8 lanes), and a run where every program was skipped exits 2. Wave64 programs
translated with --wave64-emulate use no subgroup operations, so they are compared at any subgroup size.

Needs LLVM_MC (llvm-mc with AMDGPU, as for the shaders stage), the Vulkan SDK and a built vk_compute_test
(cmake --build runtime/build --target vk_compute_test). No game files.

Reference semantics (AMD "RDNA 2" ISA reference guide; Mesa ACO and LLVM where the guide is terse):
  * v_readlane_b32 sD, vS, lane: sD = vS of lane (lane & 63 in wave64, & 31 in wave32), EXEC ignored.
  * v_writelane_b32 vD, s, lane: vD of that lane = s, EXEC ignored.
  * v_readfirstlane_b32: vS of the lowest lane on in EXEC, lane 0 if EXEC is 0.
  * VOPC v_cmp: bit l = condition for active lanes, 0 for inactive ones (sD / VCC is written as a whole).
    v_cmpx: EXEC = EXEC & condition (inactive lanes stay off).
  * s_and_saveexec_bN sD, s: sD = EXEC; EXEC = s & EXEC.
  * v_mbcnt_lo_u32_b32 vD, s, acc: popcount(s & ((1 << lane) - 1)) + acc (all 32 bits for lane >= 32);
    v_mbcnt_hi_u32_b32: popcount(s & ((1 << (lane - 32)) - 1)) + acc for lane >= 32, acc for lane < 32.
  * v_permlane16_b32 vD, vS, s1, s2: lane l reads lane (l & ~15) | sel[l & 15], sel = nibbles of {s2, s1};
    v_permlanex16_b32 reads lane (l & ~31) | ((l & 16) ^ 16) | sel[l & 15]. Only written in active lanes.
    A source lane that is off in EXEC (op_sel FI = 0) is the DPP "invalid source" case: with BOUND_CTRL = 0 the
    lane keeps its old vD. That was measured on RDNA2 and RDNA3.5 (docs/shader_validation.md, #374), but the
    translator models it only with #580, so programs that reach it are still reported apart.
  * VALU results are written only in active lanes; SALU is wave-uniform.
"""
import argparse
import os
import random
import re
import struct
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "runtime", "test", "shaders"))
import mc_command  # noqa: E402

M32 = 0xFFFFFFFF
M64 = (1 << 64) - 1
DATA_V = (2, 3, 4, 5, 6, 7)            # data VGPRs, stored at the end
DATA_S = (8, 9, 10, 11, 12, 13, 14, 15)  # data SGPRs, stored at the end
OUT_REGS = [("v", r) for r in DATA_V] + [("s", r) for r in DATA_S]
SAVE = (22, 24, 26)                    # saveexec destinations (pairs in wave64)
ENTRY = 30                             # entry EXEC (pair in wave64)
SEL = 28                               # scratch SGPRs for lane selects / permlane selectors (28, 29)
PERMLANE_FETCH_INACTIVE = False
TRANSLATOR = [os.path.join(HERE, "rdna2spirv.py")]   # --translator        # --permlane-fetch-inactive: model FI = 1 (read inactive sources)


class Wave:
    def __init__(self, size, exists):
        self.size = size
        self.exists = exists           # bit mask of lanes that exist (partial last wave)
        self.v = {}                    # vgpr -> [lane values]
        self.s = {}                    # sgpr -> value
        self.exec = exists
        self.vcc = 0

    def vr(self, r):
        return self.v.setdefault(r, [0] * self.size)

    def sr(self, r):
        return self.s.get(r, 0)

    def active(self, lane):
        return self.exec >> lane & 1

    def mask(self, name):
        """A mask operand: 'vcc', 'exec' or an SGPR number (pair in wave64)."""
        if name == "vcc":
            return self.vcc
        if name == "exec":
            return self.exec
        lo = self.sr(name)
        return lo | (self.sr(name + 1) << 32 if self.size == 64 else 0)

    def set_mask(self, name, m):
        m &= (1 << self.size) - 1
        if name == "vcc":
            self.vcc = m
        elif name == "exec":
            self.exec = m
        else:
            self.s[name] = m & M32
            if self.size == 64:
                self.s[name + 1] = m >> 32

    def vwrite(self, r, vals):
        dst = self.vr(r)
        for lane in range(self.size):
            if self.active(lane) and self.exists >> lane & 1:
                dst[lane] = vals[lane] & M32


def popcount(x):
    return bin(x).count("1")


def mreg(wave, name):
    """Assembler name of a mask operand."""
    if name == "vcc":
        return "vcc" if wave == 64 else "vcc_lo"
    if name == "exec":
        return "exec" if wave == 64 else "exec_lo"
    return f"s[{name}:{name + 1}]" if wave == 64 else f"s{name}"


class Op:
    """One instruction (or a short fixed group) with its assembly and its effect on the model."""
    hyp = False                         # depends on a hypothesis about the ISA (reported apart)

    def __init__(self, text, fn, tag, hyp=False):
        self.text, self.fn, self.tag, self.hyp = text, fn, tag, hyp

    def apply(self, waves):
        hyp = False
        for w in waves:
            hyp |= bool(self.fn(w))
        return hyp


class Gen:
    def __init__(self, rng, wave, group, allow_maskdata=True, wave32_hi_lanes=True, dpp=False):
        self.rng, self.W, self.L = rng, wave, group
        self.nwaves = (group + wave - 1) // wave
        last = group - (self.nwaves - 1) * wave
        self.min_exist = last                       # lanes every wave has
        self.full = last == wave
        self.allow_maskdata, self.hi_lanes = allow_maskdata, wave32_hi_lanes
        self.dpp = dpp
        self.masks = []                             # mask registers written so far
        self.depth = 0
        self.in_branch, self.branch_id = False, 0

    def r(self, seq):
        return self.rng.choice(seq)

    def lane(self):
        return self.rng.randrange(self.min_exist)

    def vsrc(self):
        return self.r(DATA_V + (1,))

    def mask_name(self, write=False):
        pool = ["vcc", 16, 18, 20]
        if write:
            return self.r(pool)
        return self.r(self.masks) if self.masks else None

    # ---- op builders ------------------------------------------------------------------------------------
    def op_cmp(self):
        d = self.mask_name(write=True)
        a, cond = self.vsrc(), self.r(("lt", "gt", "eq", "ne", "le", "ge"))
        f = {"lt": lambda x, y: x < y, "gt": lambda x, y: x > y, "eq": lambda x, y: x == y,
             "ne": lambda x, y: x != y, "le": lambda x, y: x <= y, "ge": lambda x, y: x >= y}[cond]
        if d == "vcc" and self.rng.random() < 0.5:
            k = self.rng.randrange(65)
            text = f"v_cmp_{cond}_u32_e32 {mreg(self.W, 'vcc')}, {k}, v{a}"

            def fn(w):
                m = 0
                for lane in range(w.size):
                    if w.active(lane) and f(k, w.vr(a)[lane]):
                        m |= 1 << lane
                w.set_mask("vcc", m)
        else:
            b = self.vsrc()
            text = f"v_cmp_{cond}_u32_e64 {mreg(self.W, d)}, v{a}, v{b}"

            def fn(w):
                m = 0
                for lane in range(w.size):
                    if w.active(lane) and f(w.vr(a)[lane], w.vr(b)[lane]):
                        m |= 1 << lane
                w.set_mask(d, m)
        if d not in self.masks:
            self.masks.append(d)
        return Op(text, fn, "cmp")

    def op_cmpx(self):
        a, k = self.vsrc(), self.rng.randrange(65)
        cond = self.r(("gt", "lt", "ne"))
        f = {"gt": lambda x, y: x > y, "lt": lambda x, y: x < y, "ne": lambda x, y: x != y}[cond]
        text = f"v_cmpx_{cond}_u32_e32 {k}, v{a}"

        def fn(w):
            m = 0
            for lane in range(w.size):
                if w.active(lane) and f(k, w.vr(a)[lane]):
                    m |= 1 << lane
            w.exec = m
        return Op(text, fn, "exec")

    def op_exec_const(self):
        ex = (1 << self.min_exist) - 1
        lo = self.rng.getrandbits(32) & ex
        if self.rng.random() < 0.15:
            lo = 0
        if self.W == 32:
            text = f"s_mov_b32 exec_lo, 0x{lo:x}"

            def fn(w):
                w.exec = lo
            return Op(text, fn, "exec")
        hi = self.rng.getrandbits(32) & (ex >> 32)
        if self.rng.random() < 0.15:
            hi = 0
        text = f"s_mov_b32 exec_lo, 0x{lo:x}\n\ts_mov_b32 exec_hi, 0x{hi:x}"

        def fn(w):
            w.exec = lo | hi << 32
        return Op(text, fn, "exec")

    def op_saveexec(self):
        m = self.mask_name()
        if m is None or self.depth >= len(SAVE):
            return None
        d = SAVE[self.depth]
        self.depth += 1
        op = "s_and_saveexec_b64" if self.W == 64 else "s_and_saveexec_b32"
        text = f"{op} {mreg(self.W, d)}, {mreg(self.W, m)}"

        def fn(w):
            old = w.exec
            w.exec = w.mask(m) & old
            w.set_mask(d, old)
        return Op(text, fn, "exec")

    def op_restore(self):
        if self.depth == 0:
            return None
        self.depth -= 1
        d = SAVE[self.depth]
        op = "s_mov_b64 exec," if self.W == 64 else "s_mov_b32 exec_lo,"
        text = f"{op} {mreg(self.W, d)}"

        def fn(w):
            w.exec = w.mask(d) & w.exists
        return Op(text, fn, "exec")

    def op_readlane(self):
        d, v = self.r(DATA_S), self.vsrc()
        lane = self.lane()
        enc = lane
        if self.W == 32 and self.hi_lanes and self.rng.random() < 0.3:
            enc = lane + 32 * self.rng.randrange(1, 8)     # wave32 uses lane select bits [4:0]
        if self.rng.random() < 0.5:
            text = f"v_readlane_b32 s{d}, v{v}, {enc}" if enc <= 64 else \
                f"s_mov_b32 s{SEL}, {enc}\n\tv_readlane_b32 s{d}, v{v}, s{SEL}"
        else:
            text = f"s_mov_b32 s{SEL}, {enc}\n\tv_readlane_b32 s{d}, v{v}, s{SEL}"
        tag = "readlane" if enc == lane else "readlane_w32hi"

        def fn(w):
            w.s[d] = w.vr(v)[lane]
            if "s_mov_b32" in text:
                w.s[SEL] = enc
        return Op(text, fn, tag)

    def op_writelane(self):
        d = self.r(DATA_V)
        lane = self.lane() if self.rng.random() < 0.7 else self.rng.randrange(self.W)
        enc = lane
        if self.W == 32 and self.hi_lanes and self.rng.random() < 0.3:
            enc = lane + 32 * self.rng.randrange(1, 8)
        if self.rng.random() < 0.5:
            src = f"s{self.r(DATA_S)}"
        else:
            src = str(self.rng.randrange(-16, 65))
        if self.rng.random() < 0.5 and enc <= 64:
            text = f"v_writelane_b32 v{d}, {src}, {enc}"
        else:
            text = f"s_mov_b32 s{SEL}, {enc}\n\tv_writelane_b32 v{d}, {src}, s{SEL}"
        tag = "writelane" if enc == lane else "writelane_w32hi"

        def fn(w):
            val = w.sr(int(src[1:])) if src.startswith("s") else int(src) & M32
            if lane < w.size and w.exists >> lane & 1:
                w.vr(d)[lane] = val
            if "s_mov_b32" in text:
                w.s[SEL] = enc
        return Op(text, fn, tag)

    def op_readfirstlane(self):
        d, v = self.r(DATA_S), self.vsrc()
        text = f"v_readfirstlane_b32 s{d}, v{v}"

        def fn(w):
            e = w.exec
            first = (e & -e).bit_length() - 1 if e else 0
            w.s[d] = w.vr(v)[first]
        return Op(text, fn, "readfirstlane")

    def op_bcnt(self):
        m = self.mask_name() if self.rng.random() < 0.7 else "exec"
        if m is None:
            m = "exec"
        d = self.r(DATA_S)
        kind = self.r(("bcnt1", "ff1", "bcnt0"))
        sfx = "b64" if self.W == 64 else "b32"
        text = f"s_{kind}_i32_{sfx} s{d}, {mreg(self.W, m)}"

        def fn(w):
            x = w.mask(m)
            if kind == "bcnt1":
                w.s[d] = popcount(x)
            elif kind == "bcnt0":
                w.s[d] = self.W - popcount(x)
            else:
                w.s[d] = ((x & -x).bit_length() - 1) & M32 if x else M32
        return Op(text, fn, "bcnt")

    def op_mbcnt(self):
        d, acc = self.r(DATA_V), self.vsrc()
        m = self.mask_name() if self.rng.random() < 0.6 else "exec"
        if m is None:
            m = "exec"
        if self.W == 32:
            text = f"v_mbcnt_lo_u32_b32 v{d}, {mreg(32, m)}, v{acc}"
        elif m in ("exec", "vcc"):
            text = (f"v_mbcnt_lo_u32_b32 v{d}, {m}_lo, v{acc}\n\t"
                    f"v_mbcnt_hi_u32_b32 v{d}, {m}_hi, v{d}")
        else:
            text = f"v_mbcnt_lo_u32_b32 v{d}, s{m}, v{acc}\n\tv_mbcnt_hi_u32_b32 v{d}, s{m + 1}, v{d}"

        def fn(w):
            x = w.mask(m)
            vals = []
            for lane in range(w.size):
                vals.append(popcount(x & ((1 << lane) - 1)) + w.vr(acc)[lane])
            w.vwrite(d, vals)
        return Op(text, fn, "mbcnt")

    def op_mbcnt_data(self):
        """v_mbcnt with an ordinary SGPR (data, not a mask) as its mask operand."""
        d, acc, s = self.r(DATA_V), self.vsrc(), self.r(DATA_S)
        hi = self.W == 64 and self.rng.random() < 0.5
        name = "v_mbcnt_hi_u32_b32" if hi else "v_mbcnt_lo_u32_b32"
        text = f"{name} v{d}, s{s}, v{acc}"

        def fn(w):
            x = w.sr(s)
            vals = []
            for lane in range(w.size):
                if hi:
                    below = 0 if lane < 32 else popcount(x & ((1 << (lane - 32)) - 1))
                else:
                    below = popcount(x & ((1 << min(lane, 32)) - 1))
                vals.append(below + w.vr(acc)[lane])
            w.vwrite(d, vals)
        return Op(text, fn, "mbcnt_data")

    def op_cndmask(self):
        m = self.mask_name()
        if m is None:
            return None
        d, a, b = self.r(DATA_V), self.vsrc(), self.vsrc()
        text = f"v_cndmask_b32_e64 v{d}, v{a}, v{b}, {mreg(self.W, m)}"

        def fn(w):
            x = w.mask(m)
            w.vwrite(d, [w.vr(b)[l] if x >> l & 1 else w.vr(a)[l] for l in range(w.size)])
        return Op(text, fn, "cndmask")

    def op_maskdata(self):
        m = self.mask_name()
        if m is None or not self.allow_maskdata:
            return None
        d = self.r(DATA_V)
        half = self.rng.randrange(2) if self.W == 64 else 0
        if m == "vcc":
            src = "vcc_hi" if half else "vcc_lo"
        else:
            src = f"s{m + half}"
        text = f"v_mov_b32_e32 v{d}, {src}"

        def fn(w):
            x = w.mask(m) >> (32 * half) & M32
            w.vwrite(d, [x] * w.size)
        return Op(text, fn, "maskdata")

    def op_mask_salu(self):
        if len(self.masks) < 1:
            return None
        a, b = self.r(self.masks), self.r(self.masks)
        d = self.r([16, 18, 20])
        kind = self.r(("and", "or", "xor", "andn2"))
        sfx = "b64" if self.W == 64 else "b32"
        text = f"s_{kind}_{sfx} {mreg(self.W, d)}, {mreg(self.W, a)}, {mreg(self.W, b)}"
        if d not in self.masks:
            self.masks.append(d)

        def fn(w):
            x, y = w.mask(a), w.mask(b)
            r = {"and": x & y, "or": x | y, "xor": x ^ y, "andn2": x & ~y}[kind]
            w.set_mask(d, r)
        return Op(text, fn, "mask_salu")

    def op_permlane(self):
        if not self.full:
            return None
        d, v = self.r(DATA_V), self.vsrc()
        x16 = self.rng.random() < 0.5
        s1, s2 = self.rng.getrandbits(32), self.rng.getrandbits(32)
        name = "v_permlanex16_b32" if x16 else "v_permlane16_b32"
        if self.rng.random() < 0.5:     # VOP3 takes one literal: s1 as a literal, s2 from an SGPR
            text = f"s_mov_b32 s{SEL + 1}, 0x{s2:x}\n\t{name} v{d}, v{v}, 0x{s1:x}, s{SEL + 1}"
        else:
            text = (f"s_mov_b32 s{SEL}, 0x{s1:x}\n\ts_mov_b32 s{SEL + 1}, 0x{s2:x}\n\t"
                    f"{name} v{d}, v{v}, s{SEL}, s{SEL + 1}")
        sel = s1 | s2 << 32

        def fn(w):
            if text.count("s_mov_b32") == 2:
                w.s[SEL] = s1
            w.s[SEL + 1] = s2
            src = w.vr(v)
            old = list(w.vr(d))
            vals, hyp = list(old), False
            for lane in range(w.size):
                n = sel >> (4 * (lane & 15)) & 15
                sl = (lane & ~31) | ((lane & 16) ^ 16) | n if x16 else (lane & ~15) | n
                if not w.active(lane):
                    continue
                if w.active(sl) or PERMLANE_FETCH_INACTIVE:
                    vals[lane] = src[sl]
                if not w.active(sl):
                    hyp = not PERMLANE_FETCH_INACTIVE   # FI = 0, BOUND_CTRL = 0: keeps vD (measured; translator: #580)
            w.vwrite(d, vals)
            return hyp
        return Op(text, fn, "permlane")

    def op_dpp(self):
        """A VOP1/VOP2 op with a DPP16 or DPP8 src0 (--dpp). Model, from the RDNA 2 ISA guide's DPP section:
        DPP16 picks the source lane inside the 16-lane row (quad_perm, row_shl/shr/ror, row_mirror,
        row_half_mirror, row_share, row_xmask); a source out of the row, or one that is off in EXEC while fi = 0,
        is invalid: bound_ctrl 1 uses 0 as src0, bound_ctrl 0 leaves the lane unwritten. Lanes in a row or bank
        that row_mask / bank_mask turn off are not written. DPP8 picks a lane in the 8-lane group; an inactive
        source with fi = 0 reads 0 (as #238 implements it; reported apart as a hypothesis)."""
        if not self.dpp or not self.full:           # a source lane in the missing part of a wave is undefined
            return None
        d, a, b = self.r(DATA_V), self.vsrc(), self.vsrc()
        kind = self.r(("mov", "mov", "not", "add", "xor"))
        r = self.rng
        dpp8 = r.random() < 0.25
        fi = r.random() < 0.3
        bound = r.random() < 0.5
        row_mask = 0xF if r.random() < 0.6 else r.randrange(16)
        bank_mask = 0xF if r.random() < 0.6 else r.randrange(16)
        if dpp8:
            sel = [r.randrange(8) for _ in range(8)]
            ctl = f"dpp8:[{','.join(map(str, sel))}]" + (" fi:1" if fi else "")
            pick = lambda l: ((l & ~7) | sel[l & 7], True)
        else:
            c = r.choice(("quad_perm", "row_shl", "row_shr", "row_ror", "row_mirror", "row_half_mirror",
                          "row_share", "row_xmask"))
            if c == "quad_perm":
                q = [r.randrange(4) for _ in range(4)]
                ctl = f"quad_perm:[{','.join(map(str, q))}]"
                f = lambda x: ((x & ~3) | q[x & 3], True)
            elif c == "row_shl":
                n = r.randrange(1, 16)
                ctl, f = f"row_shl:{n}", (lambda x: (x + n, x + n < 16))
            elif c == "row_shr":
                n = r.randrange(1, 16)
                ctl, f = f"row_shr:{n}", (lambda x: (x - n, x >= n))
            elif c == "row_ror":
                n = r.randrange(1, 16)
                ctl, f = f"row_ror:{n}", (lambda x: ((x - n) & 15, True))
            elif c == "row_mirror":
                ctl, f = "row_mirror", (lambda x: (15 - x, True))
            elif c == "row_half_mirror":
                ctl, f = "row_half_mirror", (lambda x: ((x & 8) | (7 - (x & 7)), True))
            elif c == "row_share":
                n = r.randrange(16)
                ctl, f = f"row_share:{n}", (lambda x: (n, True))
            else:
                n = r.randrange(16)
                ctl, f = f"row_xmask:{n}", (lambda x: (x ^ n, True))

            def pick(l, f=f):
                s_, ok = f(l & 15)
                return (l & ~15) | (s_ & 15), ok
            ctl += f" row_mask:0x{row_mask:x} bank_mask:0x{bank_mask:x}" + (" bound_ctrl:1" if bound else "") +                 (" fi:1" if fi else "")
        name = {"mov": "v_mov_b32", "not": "v_not_b32", "add": "v_add_nc_u32", "xor": "v_xor_b32"}[kind]
        if kind in ("mov", "not"):
            text = f"{name}_dpp v{d}, v{a} {ctl}"
        else:
            text = f"{name}_dpp v{d}, v{a}, v{b} {ctl}"
        alu = {"mov": lambda x, y: x, "not": lambda x, y: ~x, "add": lambda x, y: x + y,
               "xor": lambda x, y: x ^ y}[kind]

        def fn(w):
            src, other = w.vr(a), w.vr(b)
            vals, hyp = list(w.vr(d)), False
            for lane in range(w.size):
                if not w.active(lane):
                    continue
                sl, ok = pick(lane)
                if dpp8:
                    if w.active(sl) or fi:
                        x = src[sl]
                    else:
                        x, hyp = 0, True
                else:
                    if not (row_mask >> (lane >> 4) & 1 and bank_mask >> ((lane >> 2) & 3) & 1):
                        continue
                    if ok and (w.active(sl) or fi):
                        x = src[sl]
                    elif bound:
                        x = 0
                    else:
                        continue
                vals[lane] = alu(x, other[lane]) & M32
            w.vwrite(d, vals)
            return hyp
        return Op(text, fn, "dpp")

    def op_valu(self):
        d, a, b = self.r(DATA_V), self.vsrc(), self.vsrc()
        kind = self.r(("add", "xor", "movs", "lshr"))
        if kind == "add":
            text, f = f"v_add_nc_u32_e32 v{d}, v{a}, v{b}", lambda w, l: w.vr(a)[l] + w.vr(b)[l]
        elif kind == "xor":
            text, f = f"v_xor_b32_e32 v{d}, v{a}, v{b}", lambda w, l: w.vr(a)[l] ^ w.vr(b)[l]
        elif kind == "lshr":
            k = self.rng.randrange(1, 17)
            text, f = f"v_lshrrev_b32_e32 v{d}, {k}, v{a}", lambda w, l: w.vr(a)[l] >> k
        else:
            s = self.r(DATA_S)
            text, f = f"v_mov_b32_e32 v{d}, s{s}", lambda w, l: w.sr(s)
        return Op(text, lambda w: w.vwrite(d, [f(w, l) for l in range(w.size)]), "valu")

    def op_salu(self):
        d, a, b = self.r(DATA_S), self.r(DATA_S), self.r(DATA_S)
        if self.rng.random() < 0.5:
            text = f"s_add_u32 s{d}, s{a}, s{b}"

            def fn(w):
                w.s[d] = (w.sr(a) + w.sr(b)) & M32
        else:
            text = f"s_xor_b32 s{d}, s{a}, s{b}"

            def fn(w):
                w.s[d] = w.sr(a) ^ w.sr(b)
        return Op(text, fn, "salu")

    def op_branch(self):
        """A forward branch over a few ops: s_cbranch_execz / vccz / vccnz (taken per wave, as on hardware)."""
        if self.in_branch:
            return None
        kind = self.r(("execz", "execz", "vccz", "vccnz"))
        if kind != "execz" and "vcc" not in self.masks:
            kind = "execz"
        self.in_branch = True
        body = self.program(self.rng.randrange(1, 5))
        self.in_branch = False
        k = self.branch_id
        self.branch_id += 1
        text = "\n\t".join([f"s_cbranch_{kind} @{k}"] + [o.text for o in body] + [f";@END{k}"])

        def fn(w):
            taken = {"execz": w.exec == 0, "vccz": w.vcc == 0, "vccnz": w.vcc != 0}[kind]
            hyp = False
            if not taken:
                for o in body:
                    hyp |= bool(o.fn(w))
            return hyp
        op = Op(text, fn, "branch")
        op.tags = {"branch"} | {t for o in body for t in getattr(o, "tags", {o.tag})}
        return op

    WEIGHTS = (("op_branch", 3), ("op_cmp", 5), ("op_cmpx", 2), ("op_exec_const", 2), ("op_saveexec", 2), ("op_restore", 2),
               ("op_readlane", 4), ("op_writelane", 3), ("op_readfirstlane", 3), ("op_bcnt", 3), ("op_mbcnt", 3),
               ("op_mbcnt_data", 1), ("op_cndmask", 2), ("op_maskdata", 1), ("op_mask_salu", 2),
               ("op_permlane", 3), ("op_valu", 3), ("op_salu", 1), ("op_dpp", 6))

    def program(self, n):
        table = [kw for kw in self.WEIGHTS if kw[0] != "op_dpp" or self.dpp]   # seeds stay stable without --dpp
        names = [k for k, _ in table]
        weights = [w for _, w in table]
        ops = []
        while len(ops) < n:
            op = getattr(self, self.rng.choices(names, weights)[0])()
            if op is not None:
                ops.append(op)
        return ops


def prologue(rng, wave):
    """Lane id in v1, data in v2-v7 and s8-s15, entry EXEC saved. Returns (text lines, init fn)."""
    lines = ["v_mbcnt_lo_u32_b32 v1, -1, 0"]
    if wave == 64:
        lines.append("v_mbcnt_hi_u32_b32 v1, -1, v1")
        lines.append(f"s_mov_b64 s[{ENTRY}:{ENTRY + 1}], exec")
    else:
        lines.append(f"s_mov_b32 s{ENTRY}, exec_lo")
    coef = []
    for r in DATA_V:
        a, b = rng.randrange(1, 65), rng.getrandbits(32)
        coef.append((r, a, b))
        lines.append(f"v_mad_u32_u24 v{r}, v0, {a}, 0x{b:x}")
    svals = [(r, rng.getrandbits(32)) for r in DATA_S]
    lines += [f"s_mov_b32 s{r}, 0x{v:x}" for r, v in svals]
    # Every mask register holds a mask on every path, as in compiled code: a branch may skip the v_cmp that
    # writes one, and a register that is undefined on one path has no defined value after the join.
    mov, ex = ("s_mov_b64", "exec") if wave == 64 else ("s_mov_b32", "exec_lo")
    lines += [f"{mov} {mreg(wave, m)}, 0" for m in ("vcc", 16, 18, 20)]
    lines += [f"{mov} {mreg(wave, m)}, {ex}" for m in SAVE]

    def init(waves):
        for wi, w in enumerate(waves):
            tid = [wi * w.size + l for l in range(w.size)]
            w.v[0] = tid
            w.v[1] = list(range(w.size))
            for r, a, b in coef:
                w.v[r] = [((t & 0xFFFFFF) * a + b) & M32 for t in tid]
            for r, v in svals:
                w.s[r] = v
            w.set_mask(ENTRY, w.exists)
            for m in ("vcc", 16, 18, 20):
                w.set_mask(m, 0)
            for m in SAVE:
                w.set_mask(m, w.exists)
    return lines, init


def epilogue(wave, hw=False):
    """hw=True stores with global_store_b32 through the output pointer in s[2:3] (hw_oracle) instead of a buffer."""
    k = len(OUT_REGS)
    lines = [f"s_mov_b64 exec, s[{ENTRY}:{ENTRY + 1}]" if wave == 64 else f"s_mov_b32 exec_lo, s{ENTRY}",
             f"v_mul_u32_u24_e32 v22, {4 * k}, v0"]
    for j, (kind, r) in enumerate(OUT_REGS):
        lines.append(f"v_mov_b32_e32 v20, {kind}{r}")
        lines.append(f"global_store_b32 v22, v20, s[2:3] offset:{4 * j}" if hw else
                     f"buffer_store_dword v20, v22, s[0:3], 0 offen offset:{4 * j}")
    lines.append("s_endpgm")
    return lines


class Program:
    def __init__(self, name, wave, group, pro, init, ops):
        self.name, self.wave, self.group, self.pro, self.init, self.ops = name, wave, group, pro, init, ops

    def source(self, ops=None):
        ops = self.ops if ops is None else ops
        mattr = "+wavefrontsize64,-wavefrontsize32" if self.wave == 64 else "+wavefrontsize32,-wavefrontsize64"
        head = [f"; {self.name}: crosslane_fuzz, wave{self.wave}, group {self.group}",
                f"; group: {self.group},1,1", f"; mattr: {mattr}"]
        body = self.pro + [o.text for o in ops] + epilogue(self.wave)
        return "\n".join(head + body) + "\n"

    def hw_lines(self, ops=None, guard=True):
        """The program as a native kernel body for tools/hw_oracle: branch placeholders become labels, the stores go
        through the output pointer, and with guard each instruction is followed by s_nop 7. The ISA model has no
        hazards; the real unit needs wait states after some VALU writes of SGPRs, VCC and EXEC (GFX10 and GFX11
        both), and compiled code gets them from the compiler."""
        ops = self.ops if ops is None else ops
        out = []
        for text in self.pro + [o.text for o in ops] + epilogue(self.wave, hw=True):
            for ln in (x.strip() for x in text.split("\n")):
                m = re.fullmatch(r";@END(\d+)", ln)
                if m:
                    out.append(f".L{self.name}_{m.group(1)}:")
                    continue
                ln = re.sub(r"^(s_cbranch_\w+) @(\d+)$", lambda b: f"{b.group(1)} .L{self.name}_{b.group(2)}", ln)
                out.append(ln)
                if guard and ln != "s_endpgm":
                    out.append("s_nop 7")
        return out

    def expected(self, ops=None):
        """{(thread, word): value}, plus the set of ops whose result depends on a hypothesis."""
        ops = self.ops if ops is None else ops
        n = (self.group + self.wave - 1) // self.wave
        waves = []
        for wi in range(n):
            cnt = min(self.wave, self.group - wi * self.wave)
            waves.append(Wave(self.wave, (1 << cnt) - 1))
        self.init(waves)
        hyp = set()
        for i, o in enumerate(ops):
            if o.apply(waves):
                hyp.add(i)
        out = {}
        for wi, w in enumerate(waves):
            for lane in range(w.size):
                if not w.exists >> lane & 1:
                    continue
                tid = wi * w.size + lane
                for j, (kind, r) in enumerate(OUT_REGS):
                    out[(tid, j)] = w.vr(r)[lane] if kind == "v" else w.sr(r)
        return out, hyp


def make_program(seed, wave, n_ops, **kw):
    rng = random.Random(seed)
    if wave == 64:
        group = rng.choice((64, 64, 128, 40, 100, 192))
    else:
        group = rng.choice((32, 32, 64, 20, 50, 96))
    g = Gen(rng, wave, group, **kw)
    pro, init = prologue(rng, wave)
    ops = g.program(n_ops)
    return Program(f"cs_xl{wave}_{seed}", wave, group, pro, init, ops)


# ---- running on the GPU ---------------------------------------------------------------------------------
def assemble(mc, prog, src, disasm, fssm):
    import compute_fixture
    mattr = "+wavefrontsize64,-wavefrontsize32" if prog.wave == 64 else "+wavefrontsize32,-wavefrontsize64"
    cmd = mc + ["-triple=amdgcn--amdpal", "-mcpu=gfx1013", f"-mattr={mattr}", "-show-encoding"]
    if "@" in src:
        src = resolve_branches(cmd, prog, src)
    r = mc_command.run(cmd, input=src, capture_output=True, text=True)
    if r.returncode or "error:" in r.stderr:
        raise RuntimeError(f"{prog.name} did not assemble:\n{r.stderr}\n{src}")
    code = "".join(ln for ln in r.stdout.splitlines(keepends=True) if not ln.lstrip().startswith("."))
    with open(os.path.join(disasm, prog.name + ".s"), "w", encoding="utf-8", newline="\n") as f:
        f.write(f"; {prog.name}  crosslane_fuzz\n{code}")
    asm_path = os.path.join(fssm, prog.name + ".asm")
    with open(asm_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(src)
    with open(os.path.join(fssm, prog.name + ".fssm"), "wb") as f:
        f.write(compute_fixture.compute_header(asm_path))
    os.remove(asm_path)


def resolve_branches(cmd, prog, src):
    """Replace each `s_cbranch_* @k` with the dword offset to its `;@ENDk` line: assemble once with offset 0,
    take each instruction's size from its encoding, and sum the sizes in between."""
    lines = src.splitlines()
    insns = [i for i, ln in enumerate(lines) if ln.strip() and not ln.strip().startswith(";")]
    first = "\n".join(re.sub(r"@\d+$", "0", ln) for ln in lines) + "\n"
    r = mc_command.run(cmd, input=first, capture_output=True, text=True)
    if r.returncode or "error:" in r.stderr:
        raise RuntimeError(f"{prog.name} did not assemble:\n{r.stderr}\n{first}")
    sizes = [len(re.findall(r"0x[0-9a-f]{2}", m)) // 4 for m in re.findall(r"encoding: \[([^\]]*)\]", r.stdout)]
    if len(sizes) != len(insns):
        raise RuntimeError(f"{prog.name}: {len(sizes)} encodings for {len(insns)} instructions")
    size_at = dict(zip(insns, sizes))
    for i, ln in enumerate(lines):
        m = re.search(r"@(\d+)$", ln.strip())
        if not m:
            continue
        end = next(j for j in range(i + 1, len(lines)) if lines[j].strip() == f";@END{m.group(1)}")
        lines[i] = ln.replace(f"@{m.group(1)}", str(sum(size_at.get(j, 0) for j in range(i + 1, end))))
    return "\n".join(lines) + "\n"


def translate(work, emulate):
    out = os.path.join(work, "spirv_emu" if emulate else "spirv_nat")
    cmd = [sys.executable, "-I", TRANSLATOR[0], os.path.join(work, "disasm"), "--out", out]
    if emulate:
        cmd.append("--wave64-emulate")
    r = subprocess.run(cmd, capture_output=True, text=True)
    return out, r.stdout + r.stderr


def gpu_env(prog, emulate, dump, base=None):
    """vk_compute_test's environment (FUZZ-6, tools/lds_fuzz.py): VKC_OUT_WORDS sizes the buffer, VKC_DUMP gets its
    raw bytes. A wave32 program, or a wave64 one translated with --wave64-emulate, is written for 32-wide subgroups,
    so it asks for them with VKC_SUBGROUP_SIZE=32, as lds_fuzz.py and valu_fuzz.py do (FUZZ-13): on AMD, whose
    compute default is 64, a group over 32 would otherwise run as one 64-wide subgroup and mismatch in wave 1.
    A native wave64 program keeps the device default (--w64-native runs it on whatever the GPU has)."""
    env = dict(os.environ if base is None else base, VKC_OUT_WORDS=str(max(prog.group * len(OUT_REGS), 64)),
               VKC_DUMP=dump)
    if prog.wave == 32 or emulate:
        env["VKC_SUBGROUP_SIZE"] = "32"
    return env


class HarnessError(RuntimeError):
    """vk_compute_test did not give a result for one program: it hung, crashed or wrote no dump."""


class Unpinned(RuntimeError):
    """vk_compute_test ran a wave32 program but couldn't pin the 32-lane subgroups it needs, so its words came from
    another wave size (Mesa llvmpipe: 8 lanes) and are no evidence either way: the program is skipped. Programs
    translated with --wave64-emulate use no subgroup operations, so they are judged at any size."""


GPU_TIMEOUT = 60    # seconds for one program; a translated program that never ends shows up as this


def run_gpu(exe, spv, prog, work, emulate):
    dump = os.path.join(work, "dump.bin")
    if os.path.exists(dump):
        os.remove(dump)                 # never compare a stale dump
    env = gpu_env(prog, emulate, dump)
    try:
        r = subprocess.run([exe, spv, "1", "dump"], capture_output=True, text=True, env=env, timeout=GPU_TIMEOUT)
    except subprocess.TimeoutExpired:
        raise HarnessError(f"vk_compute_test timed out after {GPU_TIMEOUT} s") from None
    import compute_fixture
    lanes = compute_fixture.unpinned_subgroup(r.stdout + r.stderr) if prog.wave == 32 else None
    if lanes:
        raise Unpinned(f"the device can't run it at {lanes} lanes (vk_compute_test: compute subgroup size: {lanes} "
                       "not available)")
    if not os.path.exists(dump):
        detail = " ".join(f"{r.stdout}{r.stderr}".split())
        raise HarnessError(f"vk_compute_test gave no dump (exit {r.returncode}){': ' + detail if detail else ''}")
    with open(dump, "rb") as f:
        data = f.read()
    return list(struct.unpack(f"<{len(data) // 4}I", data))


def compare(prog, got, ops=None):
    exp, hyp = prog.expected(ops)
    bad = []
    for (tid, j), v in sorted(exp.items()):
        g = got[tid * len(OUT_REGS) + j]
        if g != v:
            bad.append((tid, j, v, g))
    return bad, hyp


def describe(prog, bad, limit=6):
    lines = []
    for tid, j, v, g in bad[:limit]:
        kind, r = OUT_REGS[j]
        lines.append(f"    thread {tid} (wave {tid // prog.wave}, lane {tid % prog.wave}) {kind}{r}: "
                     f"expected 0x{v:08x}, got 0x{g:08x}")
    if len(bad) > limit:
        lines.append(f"    ... {len(bad) - limit} more")
    return "\n".join(lines)


def minimize(prog, mode, ctx):
    """Greedy: drop ops while the program still mismatches (same mode). Returns the reduced op list."""
    ops = list(prog.ops)
    i = len(ops) - 1
    while i >= 0:
        trial = ops[:i] + ops[i + 1:]
        if still_fails(prog, trial, mode, ctx):
            ops = trial
        i -= 1
    return ops


def still_fails(prog, ops, mode, ctx):
    mc, exe = ctx
    with tempfile.TemporaryDirectory(prefix="xlmin-") as work:
        for d in ("disasm", "fssm"):
            os.mkdir(os.path.join(work, d))
        try:
            assemble(mc, prog, prog.source(ops), os.path.join(work, "disasm"), os.path.join(work, "fssm"))
        except RuntimeError:
            return False
        spdir, _ = translate(work, mode == "emulate")
        spv = os.path.join(spdir, prog.name + ".spv")
        if not os.path.exists(spv):
            return False
        try:
            bad, hyp = compare(prog, run_gpu(exe, spv, prog, work, mode == "emulate"), ops)
        except (HarnessError, Unpinned):
            return False                # a reduction that hangs or crashes is not the mismatch being reduced
        return bool(bad) and not hyp


MODES = ("native", "emulate")


def parse_modes(text):
    """--modes: a comma list of native and emulate, or both. Any other word is an error: the run loop only knows
    these two, and an unknown one used to run every program natively, wave64 included."""
    modes = []
    for word in (w.strip() for w in text.split(",")):
        if not word:
            continue
        for mode in MODES if word == "both" else (word,):
            if mode not in MODES:
                raise argparse.ArgumentTypeError(f"unknown mode {word!r} (use native, emulate or both)")
            if mode not in modes:
                modes.append(mode)
    if not modes:
        raise argparse.ArgumentTypeError("no mode given (use native, emulate or both)")
    return modes


def _at_least_one(text):
    """--programs and --ops: a count below 1 ran nothing (or a program of no ops) and still printed a summary, which
    reads as a clean pass. Refuse it as a usage error instead, as the other fuzzers do."""
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be at least 1 (got {value})")
    return value


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--programs", type=_at_least_one, default=100,
                    help="programs per wave (at least 1; default 100)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--ops", type=_at_least_one, default=24, help="ops per program (at least 1; default 24)")
    ap.add_argument("--wave", choices=("32", "64", "both"), default="both")
    ap.add_argument("--modes", type=parse_modes, default="native,emulate",
                    help="native, emulate (wave64 only), or both (the default)")
    ap.add_argument("--w64-native", action="store_true",
                    help="also run wave64 programs without --wave64-emulate (on a 32-wide subgroup that is the "
                         "documented cross-half approximation, so mismatches are expected there)")
    ap.add_argument("--no-maskdata", action="store_true", help="leave out masks read as data (#293)")
    ap.add_argument("--no-w32-hi-lanes", action="store_true", help="wave32 lane selects stay below 32")
    ap.add_argument("--exe", default=os.path.join(ROOT, "runtime", "build", "vk_compute_test" +
                                                  (".exe" if os.name == "nt" else "")))
    ap.add_argument("--keep", help="copy failing programs (.asm) here")
    ap.add_argument("--minimize", action="store_true", help="reduce each failing program")
    ap.add_argument("--permlane-fetch-inactive", action="store_true",
                    help="model an inactive permlane source as read (FI = 1) instead of the hypothesis")
    ap.add_argument("--dpp", action="store_true", help="also generate DPP16 / DPP8 ops")
    ap.add_argument("--oracle", default="translator",
                    help="comma-separated: translator (default; needs llvm-mc, the Vulkan SDK and vk_compute_test), hw "
                         "(natively on the AMD GPU, wave32 and native wave64; tools/hw_oracle/), or translator,hw")
    ap.add_argument("--hw-arch", default=None, help="GPU architecture to assemble for (default: detected)")
    ap.add_argument("--no-hw-guard", action="store_true",
                    help="no s_nop 7 after each instruction in the native kernels (hazards then show up as mismatches)")
    ap.add_argument("--translator", help="rdna2spirv.py to test (default: the one next to this file)")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if not os.path.isfile(args.exe):   # a Visual Studio build puts it in a configuration subfolder
        folder, name = os.path.split(args.exe)
        for config in ("RelWithDebInfo", "Release"):
            if os.path.isfile(os.path.join(folder, config, name)):
                args.exe = os.path.join(folder, config, name)
                break
    global PERMLANE_FETCH_INACTIVE
    PERMLANE_FETCH_INACTIVE = args.permlane_fetch_inactive
    if args.translator:
        TRANSLATOR[0] = os.path.abspath(args.translator)
    if args.self_test:
        return self_test()
    backends = tuple(dict.fromkeys(t.strip() for t in args.oracle.split(",")))
    if not backends or any(t not in ("translator", "hw") for t in backends):
        ap.error("--oracle takes translator, hw or translator,hw")
    if args.wave == "32" and "emulate" in args.modes:
        # emulate runs a wave64 program as two 32-lane halves, so a wave32-only selection has nothing to run: the run
        # used to end after a bare `summary:` with exit 0, the same false pass #1452 fixed for a count below 1.
        ap.error("--modes emulate needs wave64 programs: with --wave 32 nothing runs (use --wave 64 or both)")
    if args.wave == "64" and "native" in args.modes and not args.w64_native:
        # A wave64 program only runs natively when --w64-native asks for the documented cross-half approximation, so
        # this combination was empty in the same silent way.
        ap.error("--wave 64 --modes native runs nothing without --w64-native; pass it to run the documented "
                 "cross-half approximation, or use --modes emulate")
    if "translator" in backends and not os.path.isfile(args.exe):   # before any program is generated or run
        sys.exit(f"vk_compute_test not found at {args.exe} or in its RelWithDebInfo/Release folder; build it first "
                 "(cmake --build <build dir> --target vk_compute_test) or give its path with --exe")
    oracle = None
    if "hw" in backends:
        from hw_oracle import runner
        try:
            oracle = runner.open_oracle(arch=args.hw_arch)
        except runner.Unavailable as e:
            print(f"hardware oracle unavailable: {e}")
            return 77
        print(f"hardware oracle: {oracle.device}")
    waves = (32, 64) if args.wave == "both" else (int(args.wave),)
    modes = args.modes
    progs = []
    for i in range(args.programs):
        wave = waves[i % len(waves)]
        progs.append(make_program(args.seed * 100003 + i, wave, args.ops, allow_maskdata=not args.no_maskdata,
                                  wave32_hi_lanes=not args.no_w32_hi_lanes, dpp=args.dpp))
    stats = {}
    fails = []
    try:
        if oracle:
            run_hw(oracle, progs, stats, fails, guard=not args.no_hw_guard)
        mc = None
        if "translator" in backends:
            import mc_command
            mc = mc_command.mc_command()
            run_translator(args, mc, progs, waves, modes, stats, fails)
        tr_status = {}
        for p, mode, what, bad in fails:
            if mode != "hw":
                tr_status.setdefault(p.name, []).append(f"{mode}: {what}")
        for p, mode, what, bad in fails:
            if mode == "hw" and bad and "translator" in backends:
                print(f"{p.name}: hardware and reference differ; translator: " +
                      ("; ".join(tr_status.get(p.name, [])) or "matches the reference"))
        for p, mode, what, bad in fails:
            tags = sorted({t for o in p.ops for t in getattr(o, 'tags', {o.tag})})
            print(f"{p.name} [{mode}] {what}; group {p.group}; ops: {', '.join(tags)}")
            if bad:
                print(describe(p, bad))
            if args.keep:
                os.makedirs(args.keep, exist_ok=True)
                with open(os.path.join(args.keep, f"{p.name}_{mode}.asm"), "w", encoding="utf-8", newline="\n") as f:
                    f.write(kept_source(p, mode, None, not args.no_hw_guard))
            if args.minimize and bad and what.startswith("mismatch"):
                if mode == "hw":
                    ops = minimize_hw(p, oracle, not args.no_hw_guard)
                elif mc is not None:
                    ops = minimize(p, mode, (mc, args.exe))
                else:
                    continue
                hyp_ops = p.expected(ops)[1]
                print("  minimized:\n\t" + "\n\t".join(o.text + ("   ; hypothesis op" if i in hyp_ops else "")
                                                         for i, o in enumerate(ops)))
                if args.keep:
                    with open(os.path.join(args.keep, f"{p.name}_{mode}_min.asm"), "w", encoding="utf-8", newline="\n") as f:
                        f.write(kept_source(p, mode, ops, not args.no_hw_guard))
    finally:
        if oracle:
            oracle.close()
    print("summary:")
    for key, st in sorted(stats.items()):
        print(f"  {key}: " + ", ".join(f"{k} {v}" for k, v in st.items()))
    if any(st["mismatch"] or st["rejected"] for st in stats.values()):
        return 1
    if any(st.get("harness_errors") for st in stats.values()):
        return 2
    if stats and not any(st["match"] or st["hyp_only"] for st in stats.values()):
        print("no program ran: every one was skipped")
        return 2
    return 0


def kept_source(prog, mode, ops, guard):
    """What --keep writes: for a hardware failure the kernel body that actually ran, else the translator input."""
    if mode == "hw":
        return "\n".join(prog.hw_lines(ops, guard)) + "\n"
    return prog.source(ops)


def hw_skip_reason(prog, arch):
    """Why a program cannot be compared on this GPU, or None. Nothing the generator emits differs between GFX10 and
    GFX11 in a way that changes a result (checked by running it; docs/shader_validation.md lists what was looked
    at), so the only skips are assembler rejections, which the runner reports."""
    return None


def run_hw(oracle, progs, stats, fails, guard=True):
    """Run every program natively (wave32 or wave64 as the program is, its real group size, one thread per lane)
    and compare all stored words with the model."""
    from hw_oracle import asm
    kernels, which = [], []
    for p in progs:
        st = stats.setdefault(f"wave{p.wave} hw", {"programs": 0, "match": 0, "mismatch": 0, "hyp_only": 0,
                                                    "hyp_confirmed": 0, "rejected": 0})
        st["programs"] += 1
        why = hw_skip_reason(p, oracle.arch)
        if why:
            st["rejected"] += 1
            fails.append((p, "hw", f"skipped: {why}", None))
            continue
        kernels.append(asm.Kernel(p.hw_lines(guard=guard), p.wave, p.group, p.group * len(OUT_REGS), raw=True))
        which.append(p)
    for p, got in zip(which, oracle.run(kernels)):
        st = stats[f"wave{p.wave} hw"]
        if isinstance(got, str):
            st["rejected"] += 1
            fails.append((p, "hw", got, None))
            continue
        bad, hyp = compare(p, got)
        if not bad:
            st["match"] += 1
            st["hyp_confirmed"] += bool(hyp)
            for kind in sorted({t for i in hyp for t in getattr(p.ops[i], "tags", {p.ops[i].tag})
                                if t in ("permlane", "dpp")}):
                st["hyp_" + kind] = st.get("hyp_" + kind, 0) + 1
        else:
            # On the hardware a failing hypothesis is a falsified claim, and it could hide a model bug elsewhere in
            # the program, so it is a mismatch like any other (hyp_only counts the ones that had a hypothesis op).
            st["mismatch"] += 1
            if hyp:
                st["hyp_only"] += 1
            fails.append((p, "hw", "mismatch" + (" (program has a hypothesis op)" if hyp else ""), bad))


def minimize_hw(prog, oracle, guard=True):
    """Drop ops while the program still differs on the hardware."""
    from hw_oracle import asm

    def fails_with(ops):
        got = oracle.run([asm.Kernel(prog.hw_lines(ops, guard), prog.wave, prog.group,
                                     prog.group * len(OUT_REGS), raw=True)])[0]
        if isinstance(got, str):
            return False
        return bool(compare(prog, got, ops)[0])
    ops = list(prog.ops)
    for i in range(len(ops) - 1, -1, -1):
        trial = ops[:i] + ops[i + 1:]
        if fails_with(trial):
            ops = trial
    return ops


def run_translator(args, mc, progs, waves, modes, stats, fails):
    with tempfile.TemporaryDirectory(prefix="crosslane-") as work:
        disasm, fssm = os.path.join(work, "disasm"), os.path.join(work, "fssm")
        os.mkdir(disasm)
        os.mkdir(fssm)
        for p in progs:
            assemble(mc, p, p.source(), disasm, fssm)
        for mode in modes:
            if mode == "emulate" and 64 not in waves:
                continue
            spdir, log = translate(work, mode == "emulate")
            for p in progs:
                if mode == "emulate" and p.wave != 64:
                    continue
                if mode == "native" and p.wave == 64 and not args.w64_native:
                    continue
                key = f"wave{p.wave} {mode}"
                st = stats.setdefault(key, {"programs": 0, "match": 0, "mismatch": 0, "hyp_only": 0,
                                            "rejected": 0, "harness_errors": 0, "skipped": 0})
                st["programs"] += 1
                spv = os.path.join(spdir, p.name + ".spv")
                if not os.path.exists(spv):
                    st["rejected"] += 1
                    fails.append((p, mode, "not translated", None))
                    continue
                try:
                    bad, hyp = compare(p, run_gpu(args.exe, spv, p, work, mode == "emulate"))
                except HarnessError as e:
                    # One program that hangs or crashes the harness used to end the run with a traceback and
                    # lose every result before it. It is reported like the other failures and the run goes on.
                    st["harness_errors"] += 1
                    fails.append((p, mode, f"harness error: {e}", None))
                    continue
                except Unpinned as e:
                    st["skipped"] += 1
                    print(f"SKIP {p.name} [{mode}]: {e}")
                    continue
                if not bad:
                    st["match"] += 1
                elif hyp:
                    st["hyp_only"] += 1
                    fails.append((p, mode, "mismatch (program has a hypothesis op)", bad))
                else:
                    st["mismatch"] += 1
                    fails.append((p, mode, "mismatch", bad))


def self_test():
    """The model against values worked out by hand (no GPU, no llvm-mc)."""
    w = Wave(64, M64)
    w.v[2] = [3 * l + 1 for l in range(64)]
    g = Gen(random.Random(0), 64, 64)
    # readlane: lane 40 of 3l+1 = 121, independent of EXEC.
    w.exec = 1 << 63
    Op("", lambda x: x.s.__setitem__(8, x.vr(2)[40]), "t").apply([w])
    assert w.s[8] == 121
    # readfirstlane with only lane 63 on: 190; with EXEC 0: lane 0 = 1.
    rf = g.op_readfirstlane()
    w.exec = 1 << 63
    rf.apply([w])
    d = int(rf.text.split()[1].strip(",")[1:])
    v = int(rf.text.split()[2][1:])
    assert w.s[d] == w.vr(v)[63]
    w.exec = 0
    rf.apply([w])
    assert w.s[d] == w.vr(v)[0]
    # mbcnt of EXEC = all ones gives the lane.
    w.exec = M64
    vals = [popcount(M64 & ((1 << l) - 1)) for l in range(64)]
    assert vals[0] == 0 and vals[40] == 40 and vals[63] == 63
    # permlanex16 with selectors 0: lane 5 reads lane 16, lane 20 reads lane 0, lane 40 reads lane 48.
    x16 = [((l & ~31) | ((l & 16) ^ 16)) for l in range(64)]
    assert x16[5] == 16 and x16[20] == 0 and x16[40] == 48
    print("self-test OK")
    return 0


if __name__ == "__main__":
    try:  # a redirected stdout is strict ANSI code page under -I; write '?' for what it cannot encode
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    sys.exit(main())
