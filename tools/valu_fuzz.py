r"""Differential fuzzer for the translator, two instruction families on the GPU (FUZZ-2, FUZZ-5), standard
library only.

    python -I tools/valu_fuzz.py --family f16  --programs 200 --seed 1 [--wave 64|32|both] [--emulate]
    python -I tools/valu_fuzz.py --family mem  ...
    python -I tools/valu_fuzz.py --family f16 --show 1:7:32      # print program 7 of seed 1 (wave32) and its model
    python -I tools/valu_fuzz.py --family f16 --model-only ...   # generate and model only (no GPU)

Families:
  f16   16-bit and packed VALU: f16 add/sub/mul/min/max/fma/fmac/med3, conversions, v_pk_* with op_sel/op_sel_hi,
        16-bit integer ops and v_pack_b32_f16, in the encodings GFX10 has (VOP2, VOP3, op_sel where llvm-mc
        -mcpu=gfx1013 accepts it). Which forms keep the other half of the destination follows the 16-bit row of the
        contract in docs/design.md (from Mesa ACO), so a mismatch means the translator disagrees with that contract.
  mem   memory: buffer loads and stores of dword, dwordx2, dwordx3 and dwordx4 (the byte and short forms are not generated),
        offen/idxen/offset addressing, a V# built in SGPRs with a small num_records (out-of-range loads read 0 and
        out-of-range stores are dropped), and s_load_dword / s_buffer_load_dword of what earlier stores wrote.

Each program is one 64-thread group (wave64, or two wave32s). Every thread stores 16 result VGPRs (v20..v35); the
mem family also dumps its scratch area. The model of each family is written here from the RDNA2 ISA and does not
import the translator. Values whose model result is NaN, infinite or denormal are marked uncertain and not
compared, so float rounding conventions can't masquerade as findings.

A mismatch is printed with the thread, register, the two values and the op that last wrote the register, plus a
--show line. --skip-op OP leaves an op out without changing the rest of the program, to set a known finding aside.
Exit code: 0 when every program that ran matches, 1 on a mismatch or an untranslated program, 2 on a harness error
or when nothing ran. Needs vk_compute_test's VKC_DUMP / VKC_OUT_WORDS (from FUZZ-6, #368).

Needs llvm-mc 21 or newer for --family f16 (docs/testing.md): llvm-mc 18 rejects op_sel on the 16-bit integer VOP3
forms, so no f16 program assembles and the run stops with exit code 2. --family mem assembles with 18.
"""
import argparse
import json
import os
import random
import struct
import subprocess
import sys
import shutil
import tempfile
from fractions import Fraction
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "runtime", "test", "shaders"))
import mc_command  # noqa: E402
from compute_fixture import compute_header, find_test_program  # noqa: E402

THREADS = 64
POOL = list(range(20, 36))  # result registers, stored at the end
ADDR, TMP, TMP2 = 6, 7, 8
M32 = 0xFFFFFFFF
M16 = 0xFFFF
RSRC2 = 0x8088  # 4 user SGPRs (the V#), workgroup id in s4, 512 bytes of LDS: as runtime/test/shaders/build.py


# =============================================================================================== numbers

def f32_bits(x):
    return struct.unpack("<I", struct.pack("<f", x))[0]


def bits_f32(b):
    return struct.unpack("<f", struct.pack("<I", b & M32))[0]


def f16_value(h):
    """Exact value of a finite f16 as a Fraction, or None for NaN/inf."""
    h &= M16
    sign = -1 if h & 0x8000 else 1
    e = (h >> 10) & 0x1F
    m = h & 0x3FF
    if e == 0x1F:
        return None
    if e == 0:
        return sign * Fraction(m, 1 << 24)
    return sign * Fraction(1024 + m, 1 << 10) * (Fraction(2) ** (e - 15))


def f32_value(b):
    b &= M32
    sign = -1 if b & 0x80000000 else 1
    e = (b >> 23) & 0xFF
    m = b & 0x7FFFFF
    if e == 0xFF:
        return None
    if e == 0:
        return sign * Fraction(m, 1 << 149)
    return sign * Fraction((1 << 23) + m, 1 << 23) * (Fraction(2) ** (e - 127))


def round_fraction(q, mant_bits, emin, emax, rtz=False):
    """Round an exact value to a binary format with `mant_bits` fraction bits, normal exponents [emin, emax].
    Returns (sign, exponent field, mantissa field, flags) where flags marks 'denorm' or 'overflow' results, which
    the fuzzer treats as uncertain."""
    if q == 0:
        return 0, 0, 0, set()
    sign = 1 if q < 0 else 0
    a = -q if q < 0 else q
    e = a.numerator.bit_length() - a.denominator.bit_length()
    if Fraction(2) ** e > a:
        e -= 1
    if e < emin:
        return sign, 0, 0, {"denorm"}
    scaled = a / (Fraction(2) ** (e - mant_bits))  # in [2^mant_bits, 2^(mant_bits+1))
    whole = scaled.numerator // scaled.denominator
    rem = scaled - whole
    if not rtz and (rem > Fraction(1, 2) or (rem == Fraction(1, 2) and whole & 1)):
        whole += 1
    if whole == 1 << (mant_bits + 1):
        whole >>= 1
        e += 1
    if e > emax:
        return sign, 0, 0, {"overflow"}
    return sign, e, whole - (1 << mant_bits), set()


def to_f16(q, rtz=False):
    """(bits, uncertain) for an exact value rounded to f16 (round to nearest even, or toward zero). A zero comes back
    as +0; callers that know the IEEE sign of an exact zero set it."""
    if q is None:
        return 0x7E00, True
    if q == 0:
        return 0, False
    s, e, m, flags = round_fraction(q, 10, -14, 15, rtz)
    if flags:
        return 0x7E00, True
    return (s << 15) | ((e + 15) << 10) | m, False


def to_f32(q):
    if q is None:
        return 0x7FC00000, True
    if q == 0:
        return 0, False
    s, e, m, flags = round_fraction(q, 23, -126, 127)
    if flags:
        return 0x7FC00000, True
    return (s << 31) | ((e + 127) << 23) | m, False


def s16(x):
    x &= M16
    return x - 0x10000 if x & 0x8000 else x


# ============================================================================================ framework

class Reg:
    """A 32-bit register per thread, with an 'uncertain' flag per 16-bit half."""
    __slots__ = ("v", "ulo", "uhi")

    def __init__(self, v, ulo=False, uhi=False):
        self.v, self.ulo, self.uhi = v & M32, ulo, uhi


class Family:
    """One instruction family: generator, assembly and model. Subclasses fill these in."""
    name = ""
    short = ""
    extra_words = 0   # dumped after the per-thread registers

    def __init__(self, seed, index, wave, skip=()):
        self.rng = random.Random(f"{self.short}:{seed}:{index}:{wave}")
        self.name = f"cs_{self.short}fuzz_s{seed}_{index}_w{wave}"
        self.wave = wave
        self.skip = set(skip)
        self.ops = []
        self.build()

    # -- to implement
    def build(self):
        raise NotImplementedError

    def asm_body(self):
        raise NotImplementedError

    def model(self):
        """Returns (regs, extra): regs[t][r] = Reg, extra = list of words or uncertain None."""
        raise NotImplementedError

    def describe(self, i):
        return f"op {i}: {self.ops[i]['text']}" if i is not None and i >= 0 else "the initial value"

    # -- shared
    def asm(self):
        w64 = self.wave == 64
        out = [f"; {self.name}: tools/valu_fuzz.py ({self.title})", "; group: 64,1,1", f"; rsrc2: {RSRC2:#x}"]
        if w64:
            out.append("; mattr: +wavefrontsize64,-wavefrontsize32")
        out.append("s_mov_b64 exec, -1" if w64 else "s_mov_b32 exec_lo, -1")
        out += self.asm_body()
        out.append("s_mov_b64 exec, -1" if w64 else "s_mov_b32 exec_lo, -1")
        out.append(f"v_mul_u32_u24_e32 v{ADDR}, {4 * len(POOL)}, v0")
        for i, r in enumerate(POOL):
            out.append(f"buffer_store_dword v{r}, v{ADDR}, s[0:3], 0 offen offset:{4 * i}")
        out += self.asm_tail()
        out.append("s_endpgm")
        return "\n".join(out) + "\n"

    def asm_tail(self):
        return []

    def out_words(self):
        return THREADS * len(POOL) + self.extra_words


# ===================================================================================== FUZZ-2: f16/packed

F16_BIN = {"v_add_f16", "v_sub_f16", "v_mul_f16", "v_max_f16", "v_min_f16"}
F16_TRI = {"v_fma_f16", "v_med3_f16"}
PK_F16 = {"v_pk_add_f16", "v_pk_mul_f16", "v_pk_max_f16", "v_pk_min_f16", "v_pk_fma_f16"}
PK_SHIFT = {"v_pk_lshlrev_b16", "v_pk_lshrrev_b16", "v_pk_ashrrev_i16"}
I16_VOP3 = {"v_add_nc_u16", "v_sub_nc_u16"}
I16_BOTH = {"v_mul_lo_u16", "v_lshlrev_b16", "v_lshrrev_b16", "v_ashrrev_i16", "v_max_u16", "v_min_u16",
            "v_max_i16", "v_min_i16"}
F16_OPS = sorted(F16_BIN | F16_TRI | PK_F16 | PK_SHIFT | I16_VOP3 | I16_BOTH |
                 {"v_fmac_f16", "v_cvt_f16_f32", "v_cvt_f32_f16", "v_pack_b32_f16", "v_cvt_pkrtz_f16_f32"})


def is_f16_denorm(h):
    return (h & 0x7C00) == 0 and (h & 0x3FF) != 0


def f16_sign(h):
    return 1 if h & 0x8000 else 0


def zero_sign_add(sa, sb):
    """IEEE round-to-nearest: an exact zero sum is -0 only when both addends are -0 (x + -x is +0)."""
    return 1 if sa and sb else 0


def f16_op(name, a, b, c=None):
    """f16 arithmetic on half bit patterns: (result bits, uncertain). Denormal inputs are uncertain too: whether a
    Vulkan driver keeps fp16 denormals is a device property (shaderDenormPreserveFloat16), not a translator rule.
    Exact zero results get their IEEE sign."""
    if any(x is not None and is_f16_denorm(x) for x in (a, b, c)):
        return 0x7E00, True
    va, vb = f16_value(a), f16_value(b)
    vc = f16_value(c) if c is not None else Fraction(0)
    if va is None or vb is None or vc is None:
        return 0x7E00, True
    sa, sb = f16_sign(a), f16_sign(b)
    sc = f16_sign(c) if c is not None else 0

    def rounded(q, zero_sign):
        bits, u = to_f16(q)
        if not u and q == 0:
            bits = zero_sign << 15
        return bits, u

    if name == "add":
        return rounded(va + vb, zero_sign_add(sa, sb) if va == 0 and vb == 0 else 0)
    if name == "sub":
        return rounded(va - vb, zero_sign_add(sa, 1 - sb) if va == 0 and vb == 0 else 0)
    if name == "mul":
        return rounded(va * vb, sa ^ sb)
    if name == "fma":
        prod_zero = va == 0 or vb == 0
        zs = zero_sign_add(sa ^ sb, sc) if prod_zero and vc == 0 else 0
        return rounded(va * vb + vc, zs)
    if name in ("max", "min"):
        if va == vb and (a & 0x8000) != (b & 0x8000):
            return 0, True  # +0 vs -0: which one min/max returns is not promised
        pick = (va >= vb) if name == "max" else (va <= vb)
        return (a if pick else b) & M16, False
    if name == "med3":
        vals = sorted([(va, a), (vb, b), (vc, c)], key=lambda x: x[0])
        if len({v for v, _ in vals}) < 3 and len({h & M16 for _, h in vals}) == 3:
            return 0, True
        return vals[1][1] & M16, False
    raise ValueError(name)


def i16_op(name, a, b):
    a &= M16
    b &= M16
    if name == "v_add_nc_u16":
        return (a + b) & M16
    if name == "v_sub_nc_u16":
        return (a - b) & M16
    if name == "v_mul_lo_u16":
        return (a * b) & M16
    if name == "v_lshlrev_b16":
        return (b << (a & 15)) & M16
    if name == "v_lshrrev_b16":
        return b >> (a & 15)
    if name == "v_ashrrev_i16":
        return (s16(b) >> (a & 15)) & M16
    if name == "v_max_u16":
        return max(a, b)
    if name == "v_min_u16":
        return min(a, b)
    if name == "v_max_i16":
        return max(a, b, key=s16)
    if name == "v_min_i16":
        return min(a, b, key=s16)
    raise ValueError(name)


class F16Family(Family):
    title = "FUZZ-2"
    short = "f16"

    def nice_half(self):
        rng = self.rng
        if rng.random() < 0.2:
            return rng.choice((0x3C00, 0x4000, 0x3800, 0xBC00, 0x4200, 0x3E00))  # 1, 2, 0.5, -1, 3, 1.5
        e = rng.randrange(-3, 7) + 15   # keeps sums and differences away from denormals
        return (rng.getrandbits(1) << 15) | (e << 10) | rng.getrandbits(10)

    def nice_f32(self):
        rng = self.rng
        e = rng.randrange(-3, 7) + 127
        return (rng.getrandbits(1) << 31) | (e << 23) | rng.getrandbits(23)

    def build(self):
        rng = self.rng
        self.init = {}
        for r in POOL:
            kind = rng.random()
            if kind < 0.15:
                self.init[r] = ("f32", self.nice_f32())
            elif kind < 0.3:
                self.init[r] = ("int", rng.getrandbits(32))
            else:
                self.init[r] = ("halves", self.nice_half() | (self.nice_half() << 16))
        ops = [o for o in F16_OPS if o not in self.skip]
        for _ in range(rng.randrange(6, 17)):
            self.ops.append(self.random_op(rng.choice(ops)))

    def random_op(self, name):
        rng = self.rng
        d = rng.choice(POOL)
        src = lambda: rng.choice(POOL)  # noqa: E731
        sel = lambda n: [rng.randrange(2) for _ in range(n)]  # noqa: E731
        op = {"name": name, "d": d}
        # GFX10 encodings, checked with llvm-mc -mcpu=gfx1013/gfx1030: the f16 VOP2 ops' VOP3 forms take no op_sel
        # (only the native VOP3 ops v_fma/v_med3_f16 do), and the 16-bit integer ops exist only as op_sel VOP3.
        if name in F16_BIN:
            op["enc"] = rng.choice(("e32", "e64"))
            op["s"] = [src(), src()]
            op["opsel"] = [0, 0, 0]
        elif name in I16_BOTH or name in I16_VOP3:
            op["enc"] = "e64"
            op["s"] = [src(), src()]
            op["opsel"] = sel(3)
        elif name in F16_TRI:
            op["enc"] = "e64"
            op["s"] = [src(), src(), src()]
            op["opsel"] = sel(4)
        elif name == "v_fmac_f16":
            op["enc"] = "e32"
            op["s"] = [src(), src()]
        elif name in ("v_cvt_f16_f32", "v_cvt_f32_f16"):
            op["enc"] = "e32"
            op["s"] = [src()]
        elif name in PK_F16 or name in PK_SHIFT:
            n = 3 if name == "v_pk_fma_f16" else 2
            op["s"] = [src() for _ in range(n)]
            op["opsel"], op["opsel_hi"] = sel(n), sel(n)
        elif name == "v_pack_b32_f16":
            op["s"] = [src(), src()]
            op["opsel"] = sel(2)
        elif name == "v_cvt_pkrtz_f16_f32":
            op["enc"] = "e32"
            op["s"] = [src(), src()]
        op["text"] = self.op_text(op)
        return op

    def op_text(self, op):
        n, d, s = op["name"], op["d"], op["s"]
        vs = ", ".join(f"v{x}" for x in s)
        if op.get("enc") == "e32":
            return f"{n}_e32 v{d}, {vs}"
        if n in F16_BIN:
            return f"{n}_e64 v{d}, {vs}"
        if n in PK_F16 or n in PK_SHIFT:
            return (f"{n} v{d}, {vs} op_sel:[{','.join(map(str, op['opsel']))}]"
                    f" op_sel_hi:[{','.join(map(str, op['opsel_hi']))}]")
        if n == "v_pack_b32_f16":
            return f"{n} v{d}, {vs} op_sel:[{op['opsel'][0]},{op['opsel'][1]}]"
        return f"{n}_e64 v{d}, {vs} op_sel:[{','.join(map(str, op['opsel']))}]"

    def asm_body(self):
        out = []
        for r, (_, v) in self.init.items():
            out.append(f"v_mov_b32_e32 v{r}, {v:#x}")
        # One register differs per thread, so thread mix-ups show: low half = a half derived from the thread id.
        out += [f"v_and_b32_e32 v{TMP}, 7, v0", f"v_lshlrev_b32_e32 v{TMP}, 7, v{TMP}",
                f"v_or_b32_e32 v{POOL[0]}, 0x3c00, v{TMP}"]   # 1.0 + thread-dependent mantissa bits
        out += [op["text"] for op in self.ops]
        return out

    def model(self):
        regs = []
        writers = []
        for t in range(THREADS):
            R = {r: Reg(v) for r, (_, v) in self.init.items()}
            R[POOL[0]] = Reg(0x3C00 | ((t & 7) << 7))  # the v_or_b32 below writes the whole register
            W = {r: -1 for r in POOL}
            for i, op in enumerate(self.ops):
                self.step(op, R)
                W[op["d"]] = i
            regs.append(R)
            writers.append(W)
        self.writer = writers
        return regs, []

    def step(self, op, R):
        n, d, s = op["name"], op["d"], op["s"]
        half = lambda r, hi: ((R[r].v >> 16) if hi else R[r].v) & M16  # noqa: E731
        unc = lambda r, hi: R[r].uhi if hi else R[r].ulo  # noqa: E731
        old = R[d]

        def write_half(value, hi, uncertain):
            if hi:
                R[d] = Reg((old.v & 0xFFFF) | (value << 16), old.ulo, uncertain)
            else:
                R[d] = Reg((old.v & 0xFFFF0000) | value, uncertain, old.uhi)

        if n in F16_BIN or n in F16_TRI:
            sel = op["opsel"]
            k = len(s)
            vals = [half(s[j], sel[j]) for j in range(k)]
            u = any(unc(s[j], sel[j]) for j in range(k))
            res, u2 = f16_op(n[2:-4], *vals)
            write_half(res, sel[k], u or u2)        # f16 ops keep the other half (contract: partial writes)
        elif n == "v_fmac_f16":
            res, u2 = f16_op("fma", half(s[0], 0), half(s[1], 0), half(d, 0))
            write_half(res, 0, u2 or unc(s[0], 0) or unc(s[1], 0) or unc(d, 0))
        elif n in I16_BOTH or n in I16_VOP3:
            sel = op["opsel"]
            res = i16_op(n, half(s[0], sel[0]), half(s[1], sel[1]))
            write_half(res, sel[2], unc(s[0], sel[0]) or unc(s[1], sel[1]))  # op_sel VOP3: keeps the other half
        elif n == "v_cvt_f16_f32":
            x = R[s[0]].v
            res, u2 = to_f16(f32_value(x)) if (x & 0x7F800000) else (0, True)  # f32 zero/denormal: uncertain
            write_half(res, 0, u2 or R[s[0]].ulo or R[s[0]].uhi)
        elif n == "v_cvt_f32_f16":
            h = half(s[0], 0)
            v = None if is_f16_denorm(h) else f16_value(h)
            bits, u2 = to_f32(v) if v is not None else (0x7FC00000, True)
            if v == 0:
                bits = f16_sign(h) << 31
            R[d] = Reg(bits, u2 or unc(s[0], 0), u2 or unc(s[0], 0))
        elif n == "v_cvt_pkrtz_f16_f32":
            fa, fb = R[s[0]].v, R[s[1]].v
            lo, ul = to_f16(f32_value(fa), rtz=True) if fa & 0x7F800000 else (0, True)
            hi, uh = to_f16(f32_value(fb), rtz=True) if fb & 0x7F800000 else (0, True)
            R[d] = Reg(lo | (hi << 16), ul or R[s[0]].ulo or R[s[0]].uhi, uh or R[s[1]].ulo or R[s[1]].uhi)
        elif n == "v_pack_b32_f16":
            sel = op["opsel"]
            lo, hi = half(s[0], sel[0]), half(s[1], sel[1])
            R[d] = Reg(lo | (hi << 16), unc(s[0], sel[0]), unc(s[1], sel[1]))
        elif n in PK_F16 or n in PK_SHIFT:
            k = len(s)
            out, us = [], []
            for lane_sel in (op["opsel"], op["opsel_hi"]):
                vals = [half(s[j], lane_sel[j]) for j in range(k)]
                u = any(unc(s[j], lane_sel[j]) for j in range(k))
                if n in PK_SHIFT:
                    a, b = vals
                    base = {"v_pk_lshlrev_b16": "v_lshlrev_b16", "v_pk_lshrrev_b16": "v_lshrrev_b16",
                            "v_pk_ashrrev_i16": "v_ashrrev_i16"}[n]
                    res, u2 = i16_op(base, a, b), False
                else:
                    res, u2 = f16_op(n[5:-4], *vals)
                out.append(res)
                us.append(u or u2)
            R[d] = Reg(out[0] | (out[1] << 16), us[0], us[1])
        else:
            raise ValueError(n)



# ============================================================================================== FUZZ-5: memory

SCR_BASE = 4096          # byte address (console VA) of the scratch area, right after the per-thread results
SLOT = 64                # bytes per thread in the scratch area
SCR_BYTES = THREADS * SLOT
MEM_OPS = ("store_raw", "load_raw", "store_idx", "load_idx", "load_untouched")
VRAW, VIDX = 44, 48      # s[44:47]: raw V# (stride 0), s[48:51]: structured V# (stride = SLOT)


class MemFamily(Family):
    """MUBUF loads/stores. Every thread owns a 64-byte slot of the scratch area and only stores to and loads from its
    own slot (or bytes no op stores to), so the result doesn't depend on thread order. The raw V#'s num_records is a
    multiple of the slot size, so each access is wholly in or wholly out of range (multi-dword accesses that straddle
    num_records are a known gap, docs/design.md "Buffer bounds checking"; --straddle would be needed to fuzz them)."""
    title = "FUZZ-5"
    short = "mem"
    extra_words = SCR_BYTES // 4

    def build(self):
        rng = self.rng
        self.init = {r: (rng.randrange(1, 1 << 24), rng.getrandbits(32)) for r in POOL}
        self.raw_slots = rng.choice((THREADS, THREADS, rng.randrange(1, THREADS)))   # threads below this are in range
        self.idx_slots = rng.choice((THREADS, THREADS, rng.randrange(1, THREADS)))
        self.soff = rng.choice((0, 0, 4, 8, 16))
        kinds = [k for k in MEM_OPS if k not in self.skip]
        for _ in range(rng.randrange(4, 13)):
            self.ops.append(self.random_op(rng.choice(kinds)))

    def random_op(self, kind):
        rng = self.rng
        n = rng.choice((1, 1, 2, 3, 4))
        # the access, plus soffset for raw ops, stays inside the thread's slot
        soff = self.soff if kind in ("store_raw", "load_raw") else 0
        top = SLOT - 4 * n - soff
        off = 4 * rng.randrange(0, top // 4 + 1)
        reg = rng.choice(POOL[: len(POOL) - n + 1])
        op = {"kind": kind, "n": n, "off": off, "r": reg, "soff": soff}
        if kind == "load_untouched":
            op["n"] = 1
            op["off"] = 4 * rng.randrange(0, SLOT // 4)
            op["soff"] = 0
        op["text"] = f"{kind} n={op['n']} offset={op['off']} v{reg}"
        return op

    def asm_body(self):
        out = []
        for r, (k, c) in self.init.items():
            out += [f"v_mul_u32_u24_e32 v{r}, {k:#x}, v0", f"v_xor_b32_e32 v{r}, {c:#x}, v{r}"]
        # raw V#: base SCR_BASE, stride 0, num_records = raw_slots * SLOT bytes; word 3 as the push-constant V#
        out += [f"s_mov_b32 s{VRAW}, {SCR_BASE:#x}", f"s_mov_b32 s{VRAW + 1}, 0",
                f"s_mov_b32 s{VRAW + 2}, {self.raw_slots * SLOT:#x}", f"s_mov_b32 s{VRAW + 3}, s3",
                # structured V#: same base, stride SLOT (dword1[29:16]), num_records = idx_slots elements, OOB_SELECT 1 (index check);
                # the raw one keeps the push-constant V#'s OOB_SELECT 3
                f"s_mov_b32 s{VIDX}, {SCR_BASE:#x}", f"s_mov_b32 s{VIDX + 1}, {SLOT << 16:#x}",
                f"s_mov_b32 s{VIDX + 2}, {self.idx_slots:#x}", f"s_mov_b32 s{VIDX + 3}, 0x11016fac",
                f"s_mov_b32 s52, {self.soff:#x}",
                f"v_mul_u32_u24_e32 v{ADDR}, {SLOT}, v0"]          # raw: byte offset of the thread's slot
        for op in self.ops:
            n, off, r = op["n"], op["off"], op["r"]
            regs = f"v{r}" if n == 1 else f"v[{r}:{r + n - 1}]"
            suf = "" if n == 1 else f"x{n}"
            if op["kind"] == "store_raw":
                out.append(f"buffer_store_dword{suf} {regs}, v{ADDR}, s[{VRAW}:{VRAW + 3}], s52 offen offset:{off}")
            elif op["kind"] == "load_raw":
                out.append(f"buffer_load_dword{suf} {regs}, v{ADDR}, s[{VRAW}:{VRAW + 3}], s52 offen offset:{off}")
            elif op["kind"] == "store_idx":
                out.append(f"buffer_store_dword{suf} {regs}, v0, s[{VIDX}:{VIDX + 3}], 0 idxen offset:{off}")
            elif op["kind"] == "load_idx":
                out.append(f"buffer_load_dword{suf} {regs}, v0, s[{VIDX}:{VIDX + 3}], 0 idxen offset:{off}")
            elif op["kind"] == "load_untouched":
                out.append(f"buffer_load_dword {regs}, v{ADDR}, s[{VRAW}:{VRAW + 3}], 0 offen offset:{off}")
            out.append("s_waitcnt vmcnt(0)")
        return out

    def asm_tail(self):
        # the scratch area is already in the output buffer (it starts at SCR_BASE); nothing more to store
        return []

    def untouched(self):
        """Byte offsets within a slot that no store op writes (same for every thread)."""
        hit = set()
        for op in self.ops:
            if op["kind"] in ("store_raw", "store_idx"):
                start = op["off"] + op["soff"]
                hit.update(range(start, start + 4 * op["n"]))
        return hit

    def model(self):
        mem = bytearray([0xEE]) * SCR_BYTES     # vk_compute_test fills the buffer with 0xEE
        regs = [{r: Reg(((t * (k & 0xFFFFFF)) & M32) ^ c) for r, (k, c) in self.init.items()} for t in range(THREADS)]
        writer = [{r: -1 for r in POOL} for _ in range(THREADS)]
        stored = self.untouched()
        for i, op in enumerate(self.ops):
            n, off, r, kind = op["n"], op["off"], op["r"], op["kind"]
            for t in range(THREADS):
                raw = kind in ("store_raw", "load_raw", "load_untouched")
                in_range = t < (self.raw_slots if raw else self.idx_slots)
                base = t * SLOT + off + op["soff"]
                if kind.startswith("store"):
                    if in_range:
                        for j in range(n):
                            struct.pack_into("<I", mem, base + 4 * j, regs[t][r + j].v)
                else:
                    for j in range(n):
                        v = struct.unpack_from("<I", mem, base + 4 * j)[0] if in_range else 0
                        regs[t][r + j] = Reg(v)
                        writer[t][r + j] = i
        if any(op["kind"] == "load_untouched" and set(range(op["off"], op["off"] + 4)) & stored for op in self.ops):
            pass  # a load_untouched may read bytes a later or earlier store wrote: it is still in the thread's own slot
        self.writer = writer
        extra = list(struct.unpack(f"<{SCR_BYTES // 4}I", bytes(mem)))
        return regs, extra


# ========================================================================================== runner

FAMILIES = {"f16": F16Family, "mem": MemFamily}


def programs(family, seed, count, waves, skip=()):
    cls = FAMILIES[family]
    return [cls(seed, i, w, skip) for i in range(count) for w in waves]


def compare(p, data):
    words = struct.unpack(f"<{len(data) // 4}I", data)
    want, extra = p.model()
    diffs = []
    for t in range(THREADS):
        for i, r in enumerate(POOL):
            got = words[t * len(POOL) + i]
            w = want[t][r]
            mask = (0 if w.ulo else 0xFFFF) | (0 if w.uhi else 0xFFFF0000)
            if (got ^ w.v) & mask:
                diffs.append(f"thread {t} v{r}: got {got:#010x}, want {w.v:#010x} (compared bits {mask:#010x}; "
                             f"last written by {p.describe(p.writer[t][r])})")
    base = THREADS * len(POOL)
    for i, w in enumerate(extra):
        if w is not None and words[base + i] != w:
            diffs.append(f"extra word {i}: got {words[base + i]:#010x}, want {w:#010x}")
    return diffs


def translate(progs, work, emulate):
    disasm, fssm, spirv = (work / d for d in ("disasm", "fssm", "spirv"))
    for d in (disasm, fssm, spirv):
        if d.exists():
            shutil.rmtree(d)
    for d in (disasm, fssm):
        d.mkdir(parents=True)
    mc = mc_command.mc_command()
    for p in progs:
        text = p.asm()
        asm_path = work / (p.name + ".asm")
        asm_path.write_text(text, encoding="utf-8")
        cmd = mc + ["-triple=amdgcn--amdpal", "-mcpu=gfx1013", "-show-encoding"]
        if p.wave == 64:
            cmd.append("-mattr=+wavefrontsize64,-wavefrontsize32")
        r = subprocess.run(cmd, input=text, capture_output=True, text=True)
        if r.returncode or "error:" in r.stderr:
            raise RuntimeError(f"{p.name} did not assemble:\n{r.stderr}")
        code = "".join(ln for ln in r.stdout.splitlines(keepends=True) if not ln.lstrip().startswith("."))
        (disasm / (p.name + ".s")).write_text(f"; {p.name}\n{code}", encoding="utf-8")
        (fssm / (p.name + ".fssm")).write_bytes(compute_header(asm_path))
    cmd = [sys.executable, "-I", os.path.join(HERE, "rdna2spirv.py"), str(disasm), "--out", str(spirv)]
    if emulate:
        cmd.append("--wave64-emulate")
    r = subprocess.run(cmd, capture_output=True, text=True)
    index = spirv / "index.json"
    if not index.exists():
        raise RuntimeError(f"rdna2spirv.py exited {r.returncode} without an index.json:\n{(r.stdout + r.stderr)[-2000:]}")
    return json.loads(index.read_text(encoding="utf-8")), (r.stdout + r.stderr)


def run_gpu(p, spv, build, index_entry, emulate, dump):
    exe = os.path.join(build, "vk_compute_test" + (".exe" if os.name == "nt" else ""))
    if not os.path.isfile(exe):
        sys.exit(f"vk_compute_test not found in {build} or its RelWithDebInfo/Release folder; "
                 "build it first (cmake --build <build dir> --target vk_compute_test)")
    env = dict(os.environ, VKC_OUT_WORDS=str(p.out_words()), VKC_DUMP=str(dump))
    args = [exe, str(spv), "1", "dump"]
    if p.wave == 64 and not emulate:
        args.append("--require-wave64")
    elif p.wave == 32 or emulate:
        env["VKC_SUBGROUP_SIZE"] = "32"
    if index_entry.get("subgroup_size") and not emulate:
        env["VKC_SUBGROUP_SIZE"] = str(index_entry["subgroup_size"])
    r = subprocess.run(args, capture_output=True, text=True, env=env)
    return r.returncode, (r.stdout + r.stderr).strip()


def show(family, seed, index, wave, skip=()):
    p = FAMILIES[family](seed, index, wave, skip)
    print(p.asm())
    regs, _ = p.model()
    print("; model, thread 0:", " ".join(f"v{r}={regs[0][r].v:#x}" for r in POOL))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--family", choices=sorted(FAMILIES), required=True)
    ap.add_argument("--programs", type=int, default=50)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--wave", choices=("64", "32", "both"), default="both")
    ap.add_argument("--emulate", action="store_true", help="translate with --wave64-emulate, run at subgroup 32")
    ap.add_argument("--build", default=os.path.join(ROOT, "runtime", "build"))
    ap.add_argument("--work", default=None, help="keep the generated .asm and SPIR-V here")
    ap.add_argument("--model-only", action="store_true")
    ap.add_argument("--show", default=None, metavar="SEED:INDEX[:WAVE]")
    ap.add_argument("--max-report", type=int, default=8)
    ap.add_argument("--skip-op", action="append", default=[], help="leave this op out (a known finding)")
    a = ap.parse_args()
    if a.programs < 1 or a.max_report < 1:
        ap.error("--programs and --max-report must be at least 1")
    if a.show is not None:
        parts = a.show.split(":")
        try:
            if len(parts) not in (2, 3):
                raise ValueError
            seed, index = int(parts[0]), int(parts[1])
            wave = int(parts[2]) if len(parts) == 3 else 64
        except ValueError:
            ap.error("--show expects SEED:INDEX[:WAVE] with INDEX >= 0 and WAVE in 32, 64")
        if index < 0 or wave not in (32, 64):
            ap.error("--show expects SEED:INDEX[:WAVE] with INDEX >= 0 and WAVE in 32, 64")
        show(a.family, seed, index, wave, tuple(a.skip_op))
        return 0
    waves = (64, 32) if a.wave == "both" else (int(a.wave),)
    progs = programs(a.family, a.seed, a.programs, waves, tuple(a.skip_op))
    if a.model_only:
        for p in progs:
            p.model()
        print(f"{len(progs)} {a.family} programs generated and modelled")
        return 0
    a.build = find_test_program(a.build, "vk_compute_test") or a.build   # a Visual Studio config subfolder
    work = Path(a.work).resolve() if a.work else Path(tempfile.mkdtemp(prefix=f"{a.family}_fuzz_"))
    work.mkdir(parents=True, exist_ok=True)
    try:
        index, log = translate(progs, work, a.emulate)
    except RuntimeError as e:
        print(e)
        return 2
    failed = untranslated = skipped = errors = 0
    skip_args = "".join(f" --skip-op {k}" for k in a.skip_op)
    for p in progs:
        spv = work / "spirv" / (p.name + ".spv")
        if p.name not in index or not spv.is_file():
            untranslated += 1
            reason = next((ln.strip() for ln in log.splitlines() if p.name in ln), "")
            print(f"NOT TRANSLATED {p.name} {reason[:160]}")
            continue
        dump = work / (p.name + ".bin")
        dump.unlink(missing_ok=True)
        rc, msg = run_gpu(p, spv, a.build, index[p.name], a.emulate, dump)
        if rc == 77:
            skipped += 1
            print(f"SKIP {p.name}: {msg.splitlines()[-1] if msg else 'unsupported'}")
            continue
        data = dump.read_bytes() if dump.is_file() else b""
        if rc != 0 or len(data) != 4 * p.out_words():
            errors += 1
            print(f"ERROR {p.name}: exit {rc}, {len(data)} of {4 * p.out_words()} dump bytes: {msg[-400:]}")
            continue
        diffs = compare(p, data)
        if diffs:
            failed += 1
            idx = p.name.split("_")[3]
            print(f"MISMATCH {p.name} ({len(diffs)} words); print it with --family {a.family} --show "
                  f"{a.seed}:{idx}:{p.wave}{skip_args}")
            for d in diffs[:a.max_report]:
                print("   ", d)
    mode = "emulated" if a.emulate else "native"
    ran = len(progs) - untranslated - skipped - errors
    print(f"{a.family}: {ran - failed}/{ran} programs that ran match ({mode}); {failed} mismatched, {skipped} skipped, "
          f"{untranslated} not translated, {errors} harness errors; work dir {work}")
    if errors or ran == 0:
        return 2
    return 1 if failed or untranslated else 0


if __name__ == "__main__":
    sys.exit(main())
