"""MODE.FP_ROUND[3:2] (f16/f64 rounding) written with s_setreg_imm32_b32, run in the SPIR-V interpreter.

The game switches f16/f64 rounding to toward zero (value 3) and back to nearest even (value 0) inside programs.
Values are picked so the two modes give different bits: 1 + 0.75 * 2^-10 rounds up to the next f16 under nearest
even and down under toward zero; 1 - 2^-60 rounds to 1.0 in f64 under nearest even and to 1 - 2^-53 toward zero.
"""
import pytest

import rdna2spirv
import spirv_interp as si
from listing import translate_listing

RTZ = "s_setreg_imm32_b32 hwreg(HW_REG_MODE, 2, 2), 3\n"
RNE = "s_setreg_imm32_b32 hwreg(HW_REG_MODE, 2, 2), 0\n"
X = 0x3F801800             # f32 1 + 0.75 * 2^-10
NX = X | 0x80000000


def vgprs(tmp_path, asm):
    return si.run(translate_listing(tmp_path, asm)).vgpr


@pytest.mark.parametrize("prefix,src,want", [
    ("", X, 0x3C01),                  # no MODE write: nearest even, as before
    (RTZ, X, 0x3C00),
    (RTZ, NX, 0xBC00),
    (RNE, X, 0x3C01),
    (RTZ, 0x4788B800, 0x7BFF),        # 70000.0: toward zero saturates to 65504 instead of inf
    (RTZ, 0x7F800000, 0x7C00),        # inf stays inf
    (RTZ, 0x33C00000, 0x0001),        # 1.5 * 2^-24: f16 subnormal, truncated (nearest even gives 0x0002)
    ("", 0x33C00000, 0x0002),
    (RTZ, 0x33000000, 0x0000),        # 2^-25: below the smallest f16 subnormal
    (RTZ, 0xC7800000, 0xFBFF),        # -65536.0 saturates to -65504
])
def test_cvt_f16_f32(prefix, src, want, tmp_path):
    asm = f"v_mov_b32 v1, {src:#x}\n{prefix}v_cvt_f16_f32_e32 v2, v1\n"
    assert vgprs(tmp_path, asm)[2] & 0xFFFF == want


def test_mode_switches_back(tmp_path):
    asm = f"v_mov_b32 v1, {X:#x}\n{RTZ}v_cvt_f16_f32_e32 v2, v1\n{RNE}v_cvt_f16_f32_e32 v3, v1\n"
    v = vgprs(tmp_path, asm)
    assert (v[2] & 0xFFFF, v[3] & 0xFFFF) == (0x3C00, 0x3C01)


def test_pk_add_f16_both_halves(tmp_path):
    # lo: 1.0 + 1.5 * 2^-11 (0.75 f16 ulp past 1.0), hi: the same negated; toward zero gives 1.0 and -1.0
    one, small = 0x3C00, 0x1200          # 1.0, 1.5 * 2^-11
    asm = (f"v_mov_b32 v1, {(one | 0x8000) << 16 | one:#x}\nv_mov_b32 v2, {(small | 0x8000) << 16 | small:#x}\n"
           f"{RTZ}v_pk_add_f16 v3, v1, v2\n")
    assert vgprs(tmp_path, asm)[3] == 0xBC00 << 16 | 0x3C00


# 2 - 2^-24: an f32 tie that rounds up to 2.0, so toward zero must look at the exact sum (0x3FFF), not the f32 one.
TWO, TINY, NTINY = 0x4000, 0x0001, 0x8001


@pytest.mark.parametrize("prefix,asm,want", [
    ("", "v_sub_f16_e32 v3, v1, v2", 0x4000),
    (RTZ, "v_sub_f16_e32 v3, v1, v2", 0x3FFF),
    (RTZ, "v_add_f16_e32 v3, v1, v4", 0x3FFF),
    (RTZ, "v_add_f16_e32 v3, v1, v2", 0x4000),                 # 2 + 2^-24 toward zero is 2.0
    (RTZ, "v_fma_f16 v3, v1, v5, v4", 0x3FFF),                 # 2 * 1 - 2^-24
    (RTZ, "v_add_f16_e64 v3, v1, v4 op_sel:[0,0,1]", 0x3FFF << 16 | 0x5555),   # high destination half
])
def test_f16_sum_rounds_from_the_exact_value(prefix, asm, want, tmp_path):
    setup = (f"v_mov_b32 v1, {TWO:#x}\nv_mov_b32 v2, {TINY:#x}\nv_mov_b32 v4, {NTINY:#x}\nv_mov_b32 v5, 0x3c00\n"
             "v_mov_b32 v3, 0x5555\n")
    got = vgprs(tmp_path, setup + prefix + asm + "\n")[3]
    assert (got if want > 0xFFFF else got & 0xFFFF) == want


def test_pk_fma_f16_exact_toward_zero(tmp_path):
    # both halves: 2 * 1 + (-2^-24) -> 0x3FFF, and -(2 * 1) + 2^-24 -> 0xBFFF
    asm = (f"v_mov_b32 v1, {(TWO | 0x8000) << 16 | TWO:#x}\nv_mov_b32 v2, 0x3c003c00\n"
           f"v_mov_b32 v4, {TINY << 16 | NTINY:#x}\n{RTZ}v_pk_fma_f16 v3, v1, v2, v4\n")
    assert vgprs(tmp_path, asm)[3] == 0xBFFF << 16 | 0x3FFF


@pytest.mark.parametrize("prefix,want", [("", (0, 0x3FF00000)), (RTZ, (0xFFFFFFFF, 0x3FEFFFFF))])
def test_add_f64(prefix, want, tmp_path):
    asm = ("v_mov_b32 v0, 0\nv_mov_b32 v1, 0x3ff00000\nv_mov_b32 v2, 0\nv_mov_b32 v3, 0xbc300000\n"
           f"{prefix}v_add_f64 v[4:5], v[0:1], v[2:3]\n")
    v = vgprs(tmp_path, asm)
    assert (v[4], v[5]) == want


@pytest.mark.parametrize("prefix,want", [("", (0, 0x40000000)), (RTZ, (0xFFFFFFFF, 0x3FFFFFFF))])
def test_add_f64_mode_rounds_before_omod(prefix, want, tmp_path):
    # (1 - 2^-60) * 2: the sum rounds by MODE first (1 - 2^-53 toward zero), then omod doubles it exactly
    asm = ("v_mov_b32 v0, 0\nv_mov_b32 v1, 0x3ff00000\nv_mov_b32 v2, 0\nv_mov_b32 v3, 0xbc300000\n"
           f"{prefix}v_add_f64 v[4:5], v[0:1], v[2:3] mul:2\n")
    v = vgprs(tmp_path, asm)
    assert (v[4], v[5]) == want


def test_add_f64_overflow_saturates(tmp_path):
    asm = ("v_mov_b32 v0, -1\nv_mov_b32 v1, 0x7fefffff\n"
           f"{RTZ}v_add_f64 v[4:5], v[0:1], v[0:1]\n")
    v = vgprs(tmp_path, asm)
    assert (v[4], v[5]) == (0xFFFFFFFF, 0x7FEFFFFF)


@pytest.mark.parametrize("asm", [
    "s_setreg_imm32_b32 hwreg(HW_REG_MODE, 0, 2), 3\n",       # f32 rounding: not modelled
    "s_setreg_imm32_b32 hwreg(HW_REG_MODE, 4, 4), 15\n",      # denormal controls: not modelled
    "s_setreg_imm32_b32 hwreg(HW_REG_MODE, 2, 2), 1\n",       # round toward +inf
    "s_setreg_b32 hwreg(HW_REG_MODE, 2, 2), s0\n",            # value from a register
])
def test_unmodelled_mode_writes_are_refused(asm, tmp_path):
    with pytest.raises(rdna2spirv.Unsupported):
        translate_listing(tmp_path, asm)
