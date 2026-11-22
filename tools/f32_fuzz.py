r"""Differential fuzzer for the translator's 32-bit float VALU (FUZZ-3; standard library only).

Seeded programs of VOP1 / VOP2 / VOP3 float instructions (with neg/abs source modifiers, clamp and omod) run
through two independent sides and are compared bit-exact:

  * F32Model below: the RDNA2 ISA definitions ("RDNA 2" Instruction Set Architecture reference guide, chapter
    6 "Vector ALU Operations" and the VOP1/VOP2/VOP3/VOPC opcode tables), with exact rational arithmetic and one
    round-to-nearest-even per operation. It shares no code with tools/rdna2spirv.py.
  * tools/rdna2spirv.py -> tests/semantic/spirv_mini_interp.py (the translated SPIR-V, executed).

What a comparison can and cannot say:
  * NaN results match any NaN: payloads are not modelled (the interpreter passes floats through Python).
  * Denormal f32 operands or results are skipped with a reason: whether the game's shaders flush f32
    denormals is the MODE register question (SHADER-35), not a property of these instructions.
  * v_rcp/v_rsq/v_sqrt/v_exp/v_log are only fed inputs whose results are exact (0, inf, NaN, negatives, powers
    of two); the hardware's 1-ULP approximations are out of scope.
  * An input for which the translated SPIR-V is undefined (GLSL.std.450 FMax of NaN, OpFDiv by zero, Log2 of
    0, ...) while the ISA defines the result is reported as its own class, "undefined in SPIR-V": the
    interpreter refuses to guess, and so does this tool.

Usage:
    python -I tools/f32_fuzz.py [--programs N] [--seed S] [--ops v_add_f32,...] [--show N]
    python -I tools/f32_fuzz.py --list-ops
    python -I tools/f32_fuzz.py --self-test

Exit code: 0 when every program matches, 1 when any mismatches, 2 on a harness error. Nothing here reads game
files.
"""
import argparse
import gc
import math
import os
import random
import struct
import sys
import tempfile
import warnings
from fractions import Fraction

HERE = os.path.dirname(os.path.abspath(__file__))
sys.dont_write_bytecode = True
TOOLS_DIR = os.environ.get("LEONIDA_TOOLS_DIR") or HERE
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)
SEM_DIR = os.path.join(os.path.dirname(HERE), "tests", "semantic")
if SEM_DIR not in sys.path:
    sys.path.insert(0, SEM_DIR)

import rdna2spirv  # noqa: E402
from spirv_mini_interp import run_fragment  # noqa: E402

M32 = 0xFFFFFFFF
STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)
QNAN = 0x7FC00000
PINF, NINF = 0x7F800000, 0xFF800000


class NotModelled(Exception):
    """An input whose hardware result this tool does not claim to know; counted as a skip with the reason."""


class Denormal(NotModelled):
    """An f32 denormal reached an operand or a result: outside what this tool judges (see the module doc)."""


# ---- f32 bit helpers (exact) ----

def is_nan(b):
    return (b >> 23) & 0xFF == 0xFF and b & 0x7FFFFF != 0


def is_inf(b):
    return b & 0x7FFFFFFF == PINF


def is_zero(b):
    return b & 0x7FFFFFFF == 0


def is_denorm(b):
    return (b >> 23) & 0xFF == 0 and b & 0x7FFFFF != 0


def sign(b):
    return b >> 31


def to_frac(b):
    """Exact value of a finite f32."""
    e, m = (b >> 23) & 0xFF, b & 0x7FFFFF
    v = Fraction(m, 1 << 23) * Fraction(2) ** -126 if e == 0 else Fraction((1 << 23) | m, 1 << 23) * Fraction(2) ** (e - 127)
    return -v if sign(b) else v


def round_f32(v, neg_zero=False):
    """Fraction -> f32 bits, round to nearest even, overflow to inf. `neg_zero`: the sign of an exact zero."""
    if v == 0:
        return 0x80000000 if neg_zero else 0
    s = 0x80000000 if v < 0 else 0
    a = -v if v < 0 else v
    e = a.numerator.bit_length() - a.denominator.bit_length()
    if Fraction(2) ** e > a:
        e -= 1
    if e < -126:
        q = a / (Fraction(2) ** -149)                 # denormal: units of 2^-149
        n = _rne(q)
        if n == 0:
            return s
        if n < (1 << 23):
            return s | n
        return s | (1 << 23)                           # rounded up to the smallest normal
    q = a / (Fraction(2) ** (e - 23))                  # [2^23, 2^24)
    n = _rne(q)
    if n == 1 << 24:
        n >>= 1
        e += 1
    if e > 127:
        return s | PINF
    return s | ((e + 127) << 23) | (n & 0x7FFFFF)


def _rne(q):
    n = q.numerator // q.denominator
    rem = q - n
    if rem > Fraction(1, 2) or (rem == Fraction(1, 2) and n & 1):
        n += 1
    return n


def round_rtz_f16(b):
    """f32 bits -> f16 bits, round toward zero; finite overflow gives +-65504, inf stays inf, NaN stays NaN."""
    s = (b >> 16) & 0x8000
    if is_nan(b):
        return 0x7E00 | s
    if is_inf(b):
        return 0x7C00 | s
    a = abs(to_frac(b))
    if a >= 65504:
        return s | 0x7BFF
    if a < Fraction(2) ** -14:
        return s | int(a / Fraction(2) ** -24)        # f16 denormal (truncated)
    e = a.numerator.bit_length() - a.denominator.bit_length()
    if Fraction(2) ** e > a:
        e -= 1
    n = int(a / Fraction(2) ** (e - 10))
    return s | ((e + 15) << 10) | (n & 0x3FF)


def round_rne_f16(b):
    s = (b >> 16) & 0x8000
    if is_nan(b):
        return 0x7E00 | s
    if is_inf(b):
        return 0x7C00 | s
    a = abs(to_frac(b))
    if a == 0:
        return s
    if a < Fraction(2) ** -14:
        n = _rne(a / Fraction(2) ** -24)
        return s | n                                   # n == 0x400 is the smallest normal, same encoding
    e = a.numerator.bit_length() - a.denominator.bit_length()
    if Fraction(2) ** e > a:
        e -= 1
    n = _rne(a / Fraction(2) ** (e - 10))
    if n == 1 << 11:
        n >>= 1
        e += 1
    if e > 15:
        return s | 0x7C00
    return s | ((e + 15) << 10) | (n & 0x3FF)


def f16_to_f32(h):
    s = (h & 0x8000) << 16
    e, m = (h >> 10) & 0x1F, h & 0x3FF
    if e == 0x1F:
        return s | PINF | (m << 13)
    if e == 0:
        return round_f32(Fraction(m, 1 << 24) * (-1 if s else 1), neg_zero=bool(s))
    return s | ((e - 15 + 127) << 23) | (m << 13)


def bits(x):
    return struct.unpack("<I", struct.pack("<f", x))[0]


def neg(b):
    return b ^ 0x80000000


def fabs(b):
    return b & 0x7FFFFFFF


# ---- the ISA model ----

def add(x, y):
    if is_nan(x) or is_nan(y):
        return QNAN
    if is_inf(x) or is_inf(y):
        if is_inf(x) and is_inf(y) and sign(x) != sign(y):
            return QNAN
        return x if is_inf(x) else y
    v = to_frac(x) + to_frac(y)
    return round_f32(v, neg_zero=v == 0 and sign(x) and sign(y))


def mul(x, y):
    if is_nan(x) or is_nan(y):
        return QNAN
    s = sign(x) ^ sign(y)
    if is_inf(x) or is_inf(y):
        if is_zero(x) or is_zero(y):
            return QNAN
        return (s << 31) | PINF
    return round_f32(to_frac(x) * to_frac(y), neg_zero=bool(s))


def fma(x, y, z):
    if is_nan(x) or is_nan(y) or is_nan(z):
        return QNAN
    ps = sign(x) ^ sign(y)
    if is_inf(x) or is_inf(y):
        if is_zero(x) or is_zero(y):
            return QNAN
        if is_inf(z) and sign(z) != ps:
            return QNAN
        return (ps << 31) | PINF
    if is_inf(z):
        return z
    p = to_frac(x) * to_frac(y)
    v = p + to_frac(z)
    if v == 0:
        pz = p == 0
        return round_f32(v, neg_zero=bool(pz and ps and sign(z)))
    return round_f32(v)


def legacy_mul(x, y):
    """DX9 rules: +-0 times anything, inf and NaN included, is +0 (LLVM AMDGPUInstCombineIntrinsic, fmul_legacy)."""
    if is_zero(x) or is_zero(y):
        return 0
    return mul(x, y)


def cvt_int_to_f32(v):
    return round_f32(Fraction(v))


def rounding(x, how):
    if is_nan(x):
        return QNAN
    if is_inf(x) or is_zero(x):
        return x
    v = to_frac(x)
    r = {"floor": math.floor, "ceil": math.ceil, "trunc": math.trunc}.get(how)
    n = r(v) if r else _rne_signed(v)
    return round_f32(Fraction(n), neg_zero=bool(sign(x)))


def _rne_signed(v):
    return _rne(v) if v >= 0 else -_rne(-v)


def cvt_pk_u8(x, byte, base):
    """flt32_to_uint8: round to nearest even, clamp to [0, 255], NaN -> 0."""
    if is_nan(x):
        u = 0
    elif sign(x) or is_zero(x):
        u = 0
    elif is_inf(x):
        u = 255
    else:
        u = min(255, _rne(to_frac(x)))
    sh = 8 * (byte & 3)
    return (base & ~(0xFF << sh) & M32) | (u << sh)


def cube(op, x, y, z):
    """V_CUBEID/SC/TC/MA_F32 (ISA pseudocode; comparisons are float compares, so -0 < 0 is false)."""
    if is_nan(x) or is_nan(y) or is_nan(z):
        raise NotModelled("cube of NaN")
    ax, ay, az = (abs(to_frac(v)) if not is_inf(v) else math.inf for v in (x, y, z))
    lt0 = lambda v: sign(v) and not is_zero(v)        # noqa: E731
    if az >= ax and az >= ay:
        major = "z"
    elif ay >= ax:
        major = "y"
    else:
        major = "x"
    if op == "v_cubeid_f32":
        face = {"z": 5 if lt0(z) else 4, "y": 3 if lt0(y) else 2, "x": 1 if lt0(x) else 0}[major]
        return bits(float(face))
    if op == "v_cubesc_f32":
        return {"z": neg(x) if lt0(z) else x, "y": x, "x": z if lt0(x) else neg(z)}[major]
    if op == "v_cubetc_f32":
        return {"z": neg(y), "y": neg(z) if lt0(y) else z, "x": neg(y)}[major]
    return mul(bits(2.0), {"z": z, "y": y, "x": x}[major])


def cmp_class(x, mask):
    if is_nan(x):
        c = 1 if x & 0x400000 else 0
    elif is_inf(x):
        c = 2 if sign(x) else 9
    elif is_zero(x):
        c = 5 if sign(x) else 6
    elif is_denorm(x):
        c = 4 if sign(x) else 7
    else:
        c = 3 if sign(x) else 8
    return (mask >> c) & 1


def transcendental(op, x):
    """Exact-result inputs only (see gen): special values and powers of two."""
    if is_nan(x):
        return QNAN
    if op == "v_rcp_f32":
        if is_zero(x):
            return (x & 0x80000000) | PINF
        if is_inf(x):
            return x & 0x80000000
        return round_f32(1 / to_frac(x))
    if op == "v_sqrt_f32":
        if is_zero(x):
            return x
        if sign(x):
            return QNAN
        if is_inf(x):
            return x
        return _exact_pow2_op(x, lambda e: Fraction(2) ** (e // 2))
    if op == "v_rsq_f32":
        if is_zero(x):
            return (x & 0x80000000) | PINF
        if sign(x):
            return QNAN
        if is_inf(x):
            return 0
        return _exact_pow2_op(x, lambda e: Fraction(2) ** (-(e // 2)))
    if op == "v_exp_f32":
        if is_inf(x):
            return 0 if sign(x) else PINF
        v = to_frac(x)
        if v.denominator != 1:
            raise ValueError("v_exp_f32 of a non-integer")
        n = int(v)
        if n >= 128:
            return PINF
        if n < -126:
            raise Denormal("v_exp_f32 result")
        return round_f32(Fraction(2) ** n)
    if op == "v_log_f32":
        if is_zero(x):
            return NINF
        if sign(x):
            return QNAN
        if is_inf(x):
            return PINF
        return _exact_pow2_op(x, lambda e: Fraction(e))
    raise ValueError(op)


def _exact_pow2_op(x, f):
    v = to_frac(x)
    e = v.numerator.bit_length() - 1 if v.denominator == 1 else -(v.denominator.bit_length() - 1)
    if Fraction(2) ** e != v:
        raise ValueError("not a power of two")
    return round_f32(f(e))


OMOD = {"mul:2": 2, "mul:4": 4, "div:2": Fraction(1, 2)}


def apply_out_mods(r, mods):
    """VOP3 output modifiers: omod scales the rounded result, then clamp to [0, 1] (NaN -> 0)."""
    for m in mods:
        if m in OMOD:
            if not is_nan(r):
                r = mul(r, round_f32(Fraction(OMOD[m])))
    if "clamp" in mods:
        if is_nan(r) or r == 0x80000000:
            raise NotModelled("clamp of NaN or -0 (@shelbykb2's claim)")
        if sign(r):
            r = 0
        elif not is_zero(r) and (is_inf(r) or to_frac(r) > 1):
            r = bits(1.0)
    return r


# ---- op table: name -> (sources, kind, model) ----
# kinds: "f" float source (takes neg/abs), "u" integer source, "k" literal float constant (fmaak/madak).

def _op(srcs, fn, mods=True, sel="any"):
    return {"srcs": srcs, "fn": fn, "mods": mods, "sel": sel}


OPS = {
    "v_add_f32": _op("ff", lambda x, y: add(x, y)),
    "v_sub_f32": _op("ff", lambda x, y: add(x, neg(y))),
    "v_subrev_f32": _op("ff", lambda x, y: add(y, neg(x))),
    "v_mul_f32": _op("ff", mul),
    "v_mul_legacy_f32": _op("ff", legacy_mul),
    "v_mad_legacy_f32": _op("fff", lambda x, y, z: add(legacy_mul(x, y), z)),
    "v_mad_f32": _op("fff", lambda x, y, z: add(mul(x, y), z)),
    "v_fma_f32": _op("fff", fma),
    "v_fmac_f32": _op("ffd", fma),                       # D = S0 * S1 + D
    "v_mac_f32": _op("ffd", lambda x, y, z: add(mul(x, y), z)),
    "v_fmaak_f32": _op("ffk", fma, mods=False),
    "v_madak_f32": _op("ffk", lambda x, y, z: add(mul(x, y), z), mods=False),
    "v_cvt_f32_i32": _op("u", lambda x: cvt_int_to_f32(x - (1 << 32) if x >> 31 else x)),
    "v_cvt_f32_u32": _op("u", lambda x: cvt_int_to_f32(x)),
    "v_cvt_f32_ubyte0": _op("u", lambda x: cvt_int_to_f32(x & 0xFF)),
    "v_cvt_f32_ubyte1": _op("u", lambda x: cvt_int_to_f32((x >> 8) & 0xFF)),
    "v_cvt_f32_ubyte2": _op("u", lambda x: cvt_int_to_f32((x >> 16) & 0xFF)),
    "v_cvt_f32_ubyte3": _op("u", lambda x: cvt_int_to_f32(x >> 24)),
    "v_cvt_f32_f16": _op("h", lambda x: f16_to_f32(x & 0xFFFF)),
    "v_cvt_f16_f32": _op("f", None, mods=False),         # partial write, handled in step()
    "v_cvt_pkrtz_f16_f32": _op("ff", lambda x, y: round_rtz_f16(x) | (round_rtz_f16(y) << 16), mods=False),
    "v_cvt_pk_u8_f32": _op("fuu", cvt_pk_u8, mods=False),
    "v_floor_f32": _op("f", lambda x: rounding(x, "floor")),
    "v_ceil_f32": _op("f", lambda x: rounding(x, "ceil")),
    "v_trunc_f32": _op("f", lambda x: rounding(x, "trunc")),
    "v_rndne_f32": _op("f", lambda x: rounding(x, "rne")),
    "v_rcp_f32": _op("t", lambda x: transcendental("v_rcp_f32", x)),
    "v_rsq_f32": _op("t", lambda x: transcendental("v_rsq_f32", x)),
    "v_sqrt_f32": _op("t", lambda x: transcendental("v_sqrt_f32", x)),
    "v_exp_f32": _op("e", lambda x: transcendental("v_exp_f32", x)),
    "v_log_f32": _op("t", lambda x: transcendental("v_log_f32", x)),
    "v_cubeid_f32": _op("fff", lambda x, y, z: cube("v_cubeid_f32", x, y, z)),
    "v_cubesc_f32": _op("fff", lambda x, y, z: cube("v_cubesc_f32", x, y, z)),
    "v_cubetc_f32": _op("fff", lambda x, y, z: cube("v_cubetc_f32", x, y, z)),
    "v_cubema_f32": _op("fff", lambda x, y, z: cube("v_cubema_f32", x, y, z)),
    "v_cmp_class_f32": _op("fu", None, mods=False),      # writes VCC, handled in step()
}
VOP3_ONLY = {"v_mad_legacy_f32", "v_mad_f32", "v_fma_f32", "v_cubeid_f32", "v_cubesc_f32", "v_cubetc_f32",
             "v_cubema_f32", "v_cvt_pk_u8_f32", "v_cvt_pkrtz_f16_f32"}
# Ops the hardware gives an output modifier in their VOP3 form but whose float output is an integer or packed
# value: no omod/clamp generated for them.
NO_OUT_MODS = {"v_cvt_f16_f32", "v_cvt_pkrtz_f16_f32", "v_cvt_pk_u8_f32", "v_cmp_class_f32", "v_fmaak_f32",
               "v_madak_f32"}

# ---- operand values ----

SPECIAL = [0, 0x80000000, bits(1.0), bits(-1.0), bits(0.5), bits(-0.5), bits(2.0), bits(-2.0), PINF, NINF,
           QNAN, 0xFFC00000, 0x7F800001, bits(1.5), bits(2.5), bits(-2.5), bits(0.49999997), bits(254.5),
           bits(255.5), bits(256.0), bits(-1e30), bits(1e30), 0x7F7FFFFF, 0xFF7FFFFF, 0x00800000, 0x80800000,
           bits(65504.0), bits(65519.996), bits(65520.0), bits(3.0), bits(1.0000001), bits(0.75), bits(-0.75),
           bits(16777216.0), bits(8388608.5), bits(1e-30)]
INTS = [0, 1, 0xFFFFFFFF, 0x7FFFFFFF, 0x80000000, 0x01000001, 0x00FFFFFF, 0xFFFFFF7F, 255, 0x12345678,
        16777217, 0xFFFFFFFE, 0x40000001]
TRANS = [0, 0x80000000, PINF, NINF, QNAN, bits(1.0), bits(4.0), bits(0.25), bits(16.0), bits(2.0 ** -20),
         bits(2.0 ** 40), bits(-4.0), bits(-1.0), bits(-(2.0 ** -14)), bits(2.0 ** 126), bits(2.0 ** -126)]
EXPS = [NINF, PINF, QNAN, 0, 0x80000000, bits(1.0), bits(3.0), bits(-3.0), bits(127.0), bits(128.0),
        bits(-126.0), bits(-20.0), bits(10.0)]


def rand_float(rng):
    r = rng.random()
    if r < 0.4:
        return rng.choice(SPECIAL)
    if r < 0.7:                      # a normal number with a modest exponent: arithmetic stays normal
        e = rng.randrange(127 - 20, 127 + 21)
        return (rng.getrandbits(1) << 31) | (e << 23) | rng.getrandbits(23)
    if r < 0.85:                     # small integers and halves: exact cases, ties
        return bits(rng.randrange(-300, 301) / rng.choice((1, 2, 4)))
    e = rng.randrange(1, 255)
    return (rng.getrandbits(1) << 31) | (e << 23) | rng.getrandbits(23)


# ---- programs ----
# A program: seeds (v0..v7 = values, by v_mov_b32 of a literal) and one instruction writing v8 (v8 is seeded
# too, so accumulating ops read it). Observed: v8, the first two sources' registers, and v9 (VCC as 0/1).

SEED_REGS = tuple(range(0, 9))


def fmt_src(reg, m):
    s = f"v{reg}"
    if m.get("abs"):
        s = f"|{s}|"
    if m.get("neg"):
        s = "-" + s
    return s


def gen(rng, op):
    spec = OPS[op]
    vals = [rand_float(rng) for _ in SEED_REGS]
    srcs, mods_src, text_srcs = [], [], []
    e64 = op in VOP3_ONLY or (spec["mods"] and rng.random() < 0.5)
    k_lit = None
    regs = list(range(8))
    rng.shuffle(regs)
    for i, kind in enumerate(spec["srcs"]):
        if kind == "d":
            continue
        r = regs[i]
        if kind == "u":
            vals[r] = rng.choice(INTS) if rng.random() < 0.6 else rng.getrandbits(32)
            if op == "v_cvt_pk_u8_f32" and i == 1:
                vals[r] = rng.randrange(0, 8)
        elif kind == "h":
            vals[r] = rng.getrandbits(32) if rng.random() < 0.5 else rng.choice(
                (0x3C00, 0xBC00, 0x7C00, 0xFC00, 0x7E00, 0x0001, 0x8001, 0x03FF, 0x7BFF, 0x0400, 0x8000))
        elif kind == "t":
            vals[r] = rng.choice(TRANS)
        elif kind == "e":
            vals[r] = rng.choice(EXPS)
        elif kind == "k":
            k_lit = rand_float(rng)
            srcs.append(("k", k_lit))
            continue
        m = {}
        if e64 and kind in "fte" and spec["mods"]:
            m = {"neg": rng.random() < 0.25, "abs": rng.random() < 0.2}
        srcs.append((r, m))
        text_srcs.append(fmt_src(r, m))
    out_mods = []
    if e64 and spec["mods"] and op not in NO_OUT_MODS:
        if rng.random() < 0.3:
            out_mods.append(rng.choice(tuple(OMOD)))
        if rng.random() < 0.3:
            out_mods.append("clamp")
    return vals, srcs, e64, out_mods


def source_value(vals, s):
    r, m = s
    v = vals[r]
    if m.get("abs"):
        v = fabs(v)
    if m.get("neg"):
        v = neg(v)
    return v


def model(op, vals, srcs, out_mods):
    """Words of v8 (result) and v9 (VCC as 0/1)."""
    spec = OPS[op]
    ins = []
    for s in srcs:
        ins.append(s[1] if s[0] == "k" else source_value(vals, s))
    if "d" in spec["srcs"]:
        ins.append(vals[8])
    floats = [v for v, k in zip(ins, spec["srcs"].replace("d", "f")) if k in "fktde"]
    if any(is_denorm(v) for v in floats):
        raise Denormal("operand")
    vcc = 0
    if op == "v_cvt_f16_f32":
        r = (vals[8] & 0xFFFF0000) | round_rne_f16(ins[0])
        if is_denorm(ins[0]):
            raise Denormal("operand")
    elif op == "v_cmp_class_f32":
        if is_nan(ins[0]) and not ins[0] & 0x400000:
            # The translator reads the class from the bits after an OpBitcast to float and back, exact on a
            # GPU; the interpreter holds floats as Python floats, which quiets a signalling NaN on the way.
            raise NotModelled("signalling NaN through the interpreter's float bitcast")
        r, vcc = vals[8], cmp_class(ins[0], ins[1])
    else:
        r = spec["fn"](*ins)
        if spec["mods"] or op in ("v_fmaak_f32", "v_madak_f32"):
            if is_denorm(r):
                raise Denormal("result")
        if op in ("v_mad_f32", "v_mac_f32", "v_madak_f32", "v_mad_legacy_f32"):
            prod = legacy_mul(ins[0], ins[1]) if op == "v_mad_legacy_f32" else mul(ins[0], ins[1])
            if is_denorm(prod):
                raise Denormal("intermediate product")
        r = apply_out_mods(r, out_mods)
        if is_denorm(r):
            raise Denormal("result")
    return [r & M32, vcc]


def line(op, vals, srcs, e64, out_mods):
    spec = OPS[op]
    parts = []
    for s in srcs:
        parts.append(f"0x{s[1]:x}" if s[0] == "k" else fmt_src(*s))
    name = op
    if op == "v_cmp_class_f32":
        name += "_e64" if e64 else "_e32"
        return f"{name} vcc_lo, " + ", ".join(parts) if e64 else f"{name} vcc_lo, " + ", ".join(parts)
    if op not in VOP3_ONLY and op not in ("v_fmaak_f32", "v_madak_f32"):
        name += "_e64" if e64 else "_e32"
    if not e64 and op not in ("v_fmaak_f32", "v_madak_f32"):
        # VOP1/VOP2: src1 must be a VGPR, already true; no modifiers
        pass
    del spec
    return f"{name} v8, " + ", ".join(parts) + ("" if not out_mods else " " + " ".join(out_mods))


def listing(op, vals, srcs, e64, out_mods):
    out = ["; f32_fuzz case  code 0x0..0x1000"]
    for r in SEED_REGS:
        out.append(f"\tv_mov_b32_e32 v{r}, 0x{vals[r]:x}")
    out.append("\ts_mov_b32 vcc_lo, 0")
    out.append("\t" + line(op, vals, srcs, e64, out_mods))
    out.append("\tv_cndmask_b32_e64 v9, 0, 1, vcc_lo")
    obs = [s[0] for s in srcs if s[0] != "k"][:2] + [0, 1]
    out.append(f"\texp mrt0 v8, v9, v{obs[0]}, v{obs[1]} done vm")
    out.append("\ts_endpgm")
    return "\n".join(out) + "\n", obs[:2]


def run_translated(text, name="ps_f32_fuzz"):
    with tempfile.TemporaryDirectory(prefix="f32-fuzz-") as tmp:
        dis = os.path.join(tmp, "disasm")
        os.mkdir(dis)
        path = os.path.join(dis, name + ".s")
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        with open(os.path.join(tmp, name + ".dxbc"), "wb") as f:
            f.write(STUB_DXBC)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ResourceWarning)
                spv = rdna2spirv.translate(path)
                gc.collect()
        except rdna2spirv.Unsupported as e:
            return "skip", f"translator unsupported: {e}"
    try:
        out = run_fragment(spv)
    except Exception as e:  # noqa: BLE001 - the interpreter is half of the comparison
        msg = str(e)
        if "undefined" in msg:
            return "undefined", msg[:90]
        return "skip", f"interpreter: {type(e).__name__}: {msg[:90]}"
    if list(out) != ["%out_mrt0"]:
        return "skip", f"outputs {sorted(out)}"
    return "ok", [w & M32 for w in out["%out_mrt0"]]


def same(want, got):
    if is_nan(want) and is_nan(got):
        return True
    return want == got


def check(op, vals, srcs, e64, out_mods):
    try:
        want = model(op, vals, srcs, out_mods)
    except Denormal as e:
        return "skip", f"denormal {e}"
    except NotModelled as e:
        return "skip", str(e)
    text, obs = listing(op, vals, srcs, e64, out_mods)
    status, got = run_translated(text)
    if status != "ok":
        return status, got
    want_words = [want[0], want[1], vals[obs[0]], vals[obs[1]]]
    if all(same(w, g) or (k >= 2 and same(w | 0x400000 if is_nan(w) else w, g)) for k, (w, g) in
           enumerate(zip(want_words, got))):
        return "match", None
    return "mismatch", (want_words, got, text)


def fdesc(b):
    if is_nan(b):
        return f"0x{b:08x}(NaN)"
    try:
        return f"0x{b:08x}({struct.unpack('<f', struct.pack('<I', b))[0]!r})"
    except Exception:  # noqa: BLE001
        return f"0x{b:08x}"


def run_suite(programs, seed, ops, show=2, quiet=False):
    rng = random.Random(seed)
    counts = {"match": 0, "mismatch": 0, "undefined": 0, "skip": {}}
    by_op, undef_by_op, shown = {}, {}, {}
    for i in range(programs):
        op = rng.choice(ops)
        vals, srcs, e64, out_mods = gen(rng, op)
        status, detail = check(op, vals, srcs, e64, out_mods)
        if status == "skip":
            counts["skip"][detail] = counts["skip"].get(detail, 0) + 1
            continue
        counts[status] += 1
        if status == "undefined":
            key = (op, detail)
            undef_by_op[key] = undef_by_op.get(key, 0) + 1
            continue
        if status == "mismatch":
            by_op[op] = by_op.get(op, 0) + 1
            if shown.get(op, 0) < show and not quiet:
                shown[op] = shown.get(op, 0) + 1
                want, got, text = detail
                src_line = [ln for ln in text.splitlines() if ln.startswith("\t" + op)][0].strip()
                print(f"--- program {i}: {src_line}")
                used = sorted({s[0] for s in srcs if s[0] != "k"} | ({8} if "d" in OPS[op]["srcs"] or op in (
                    "v_cvt_f16_f32",) else set()))
                print("    inputs: " + " ".join(f"v{r}={fdesc(vals[r])}" for r in used))
                print("    model v8/vcc:      " + " ".join(fdesc(w) for w in want[:2]))
                print("    translated v8/vcc: " + " ".join(fdesc(w) for w in got[:2]))
    if not quiet:
        print(f"{programs} programs (seed {seed}): {counts['match']} match, {counts['mismatch']} mismatch, "
              f"{counts['undefined']} undefined in SPIR-V, {sum(counts['skip'].values())} skipped")
        for reason, n in sorted(counts["skip"].items(), key=lambda t: -t[1]):
            print(f"  skip x{n}: {reason}")
        for (op, reason), n in sorted(undef_by_op.items(), key=lambda t: -t[1]):
            print(f"  undefined x{n}: {op}: {reason}")
        for op, n in sorted(by_op.items(), key=lambda t: -t[1]):
            print(f"  mismatch x{n}: {op}")
    return counts, by_op, undef_by_op


def self_test():
    """The model and the harness must both be able to pass and to fail."""
    ok = True
    # Model spot checks with known answers.
    checks = [
        (add(bits(1.0), bits(2.0)), bits(3.0)),
        (add(0x80000000, 0x80000000), 0x80000000),
        (add(0x80000000, 0), 0),
        (mul(bits(-2.0), 0), 0x80000000),
        (legacy_mul(PINF, 0x80000000), 0),
        (fma(bits(1.0 + 2 ** -23), bits(1.0 - 2 ** -23), bits(-1.0)), bits(-(2.0 ** -46))),
        (round_rtz_f16(bits(70000.0)), 0x7BFF),
        (round_rtz_f16(bits(1.0 + 2 ** -11)), 0x3C00),
        (round_rne_f16(bits(65520.0)), 0x7C00),
        (rounding(bits(-0.5), "rne"), 0x80000000),
        (rounding(bits(2.5), "rne"), bits(2.0)),
        (cube("v_cubeid_f32", bits(1.0), bits(-3.0), bits(2.0)), bits(3.0)),
        (cmp_class(0xFFC00000, 1 << 1), 1),
        (apply_out_mods(bits(0.75), ["mul:2", "clamp"]), bits(1.0)),
        (round_f32(Fraction(1, 3)), 0x3EAAAAAB),
    ]
    for i, (got, want) in enumerate(checks):
        if got != want:
            print(f"model check {i} FAILED: 0x{got:08x} != 0x{want:08x}")
            ok = False
    vals = [bits(1.0), bits(2.0)] + [0] * 7
    status, d = check("v_add_f32", vals, [(0, {}), (1, {})], False, [])
    if status != "match":
        print(f"canary FAILED: v_add_f32 1+2 gave {status}: {d}")
        ok = False
    real = OPS["v_add_f32"]["fn"]
    OPS["v_add_f32"]["fn"] = lambda x, y: real(x, y) ^ 1
    try:
        status, _ = check("v_add_f32", vals, [(0, {}), (1, {})], False, [])
    finally:
        OPS["v_add_f32"]["fn"] = real
    if status != "mismatch":
        print("canary FAILED: a corrupted model result was not flagged")
        ok = False
    if ok:
        print("canaries: model checks ok, match ok, mismatch detected")
    return ok


def main(argv=None):
    ap = argparse.ArgumentParser(description="32-bit float VALU differential fuzzer for the translator.")
    ap.add_argument("--programs", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--ops", default=None, help="comma-separated op names (default: all)")
    ap.add_argument("--show", type=int, default=2, help="failures to print per op")
    ap.add_argument("--list-ops", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args(argv)
    if a.programs < 1:
        ap.error("--programs must be at least 1")
    if a.list_ops:
        print("\n".join(OPS))
        return 0
    if a.self_test:
        return 0 if self_test() else 1
    ops = tuple(OPS)
    if a.ops:
        ops = tuple(o.strip() for o in a.ops.split(",") if o.strip())
        unknown = [o for o in ops if o not in OPS]
        if unknown:
            ap.error(f"unknown ops: {', '.join(unknown)} (see --list-ops)")
    counts, _, _ = run_suite(a.programs, a.seed, ops, show=a.show)
    return 1 if counts["mismatch"] else 0


if __name__ == "__main__":
    sys.exit(main())
