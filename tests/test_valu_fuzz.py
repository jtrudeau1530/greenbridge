"""CPU checks for tools/valu_fuzz.py (FUZZ-2, FUZZ-5, FUZZ-7): the reference models and the generators, no GPU.

The GPU side (vk_compute_test, rdna2spirv.py) is what the fuzzer compares against; these tests make sure the models it
compares with are right, so a mismatch is the translator's and not the fuzzer's.
"""
import os
import random
import shutil
import struct
import subprocess
import sys
from fractions import Fraction

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "tools"))
import valu_fuzz as vf  # noqa: E402


# ------------------------------------------------------------------ numbers

def test_f16_round_trips_every_finite_half():
    for h in range(0x10000):
        if (h & 0x7C00) == 0x7C00:
            continue
        v = vf.f16_value(h)
        bits, uncertain = vf.to_f16(v)
        if v == 0 or vf.is_f16_denorm(h):
            continue        # zeros come back as +0 (callers set the sign); denormals are uncertain by design
        assert not uncertain and bits == h, hex(h)


def test_to_f16_matches_pythons_round_to_nearest_even():
    rng = random.Random(1)
    for _ in range(20000):
        x = rng.uniform(-60000, 60000) * rng.choice((1, 1e-2, 1e-3))
        want = struct.unpack("<H", struct.pack("<e", x))[0]
        bits, uncertain = vf.to_f16(Fraction(x))
        if not uncertain:
            assert bits == want, (x, hex(bits), hex(want))


def test_to_f32_matches_pythons_round_to_nearest_even():
    rng = random.Random(2)
    for _ in range(20000):
        x = rng.uniform(-1e30, 1e30) * rng.choice((1, 1e-20, 1e-35))
        want = struct.unpack("<I", struct.pack("<f", x))[0]
        bits, uncertain = vf.to_f32(Fraction(x))
        if not uncertain:
            assert bits == want, (x, hex(bits), hex(want))


def test_one_is_not_zero():
    # a regression in the model itself: an unbiased exponent of 0 with a zero mantissa is +-1.0, not zero
    assert vf.to_f16(Fraction(1)) == (0x3C00, False)
    assert vf.to_f16(Fraction(-1)) == (0xBC00, False)
    assert vf.to_f32(Fraction(1)) == (0x3F800000, False)


def test_rtz():
    assert vf.to_f16(vf.f32_value(vf.f32_bits(-2.71828)), rtz=True) == (0xC16F, False)
    assert vf.to_f16(vf.f32_value(vf.f32_bits(-2.71828)))[0] == 0xC170      # nearest rounds away here


def test_signed_zeros():
    assert vf.f16_op("mul", 0xBC00, 0x0000) == (0x8000, False)    # -1 * +0 = -0
    assert vf.f16_op("add", 0x3C00, 0xBC00) == (0x0000, False)    # 1 + -1 = +0 under round to nearest
    assert vf.f16_op("add", 0x8000, 0x8000) == (0x8000, False)    # -0 + -0 = -0
    assert vf.f16_op("sub", 0x8000, 0x0000) == (0x8000, False)    # -0 - +0 = -0
    assert vf.f16_op("max", 0x0000, 0x8000)[1]                    # +0 vs -0: not promised, so uncertain


@pytest.mark.parametrize("a,b,c,true,double", [
    (0xC281, 0xBA7F, 0xD175, 0xD121, 0xD120),
    (0x3C97, 0xBCD9, 0x50EB, 0x50BF, 0x50BE),
    (0x3846, 0x38EA, 0x5321, 0x532B, 0x532C),
])
def test_fma_is_rounded_once(a, b, c, true, double):
    # the FUZZ-2 finding: f32 Fma then f16 gives `double`; a correctly rounded f16 fma gives `true`
    assert vf.f16_op("fma", a, b, c) == (true, False)
    f32 = struct.unpack("<f", struct.pack("<f", float(vf.f16_value(a) * vf.f16_value(b) + vf.f16_value(c))))[0]
    assert vf.to_f16(Fraction(f32))[0] == double


def test_denormal_inputs_are_uncertain():
    assert vf.f16_op("add", 0x0001, 0x3C00)[1]
    assert vf.f16_op("mul", 0x3C00, 0x83FF)[1]


def test_i16_ops():
    assert vf.i16_op("v_ashrrev_i16", 4, 0x8000) == 0xF800
    assert vf.i16_op("v_lshrrev_b16", 20, 0x8000) == 0x0800          # shift amount is & 15
    assert vf.i16_op("v_max_i16", 0x7FFF, 0x8000) == 0x7FFF
    assert vf.i16_op("v_max_u16", 0x7FFF, 0x8000) == 0x8000
    assert vf.i16_op("v_sub_nc_u16", 0, 1) == 0xFFFF


# ------------------------------------------------------------------ f16 family

def test_f16_partial_writes_follow_the_contract():
    p = vf.F16Family(1, 0, 32)
    R = {r: vf.Reg(0xAAAA0000 | 0x3C00) for r in vf.POOL}
    d, s0, s1 = vf.POOL[1], vf.POOL[2], vf.POOL[3]
    p.step({"name": "v_add_f16", "d": d, "s": [s0, s1], "enc": "e32", "opsel": [0, 0, 0]}, R)
    assert R[d].v == 0xAAAA4000                              # 1 + 1 = 2 in the low half, high half kept
    p.step({"name": "v_add_nc_u16", "d": d, "s": [s0, s1], "enc": "e64", "opsel": [1, 1, 1]}, R)
    assert R[d].v == ((0xAAAA + 0xAAAA) & 0xFFFF) << 16 | 0x4000   # op_sel dst bit: high half, low kept
    p.step({"name": "v_cvt_f32_f16", "d": d, "s": [s0], "enc": "e32"}, R)
    assert R[d].v == 0x3F800000                              # 32-bit result: whole register


def test_f16_generator_is_deterministic_and_uses_gfx10_encodings():
    a = [p.asm() for p in vf.programs("f16", 7, 20, (32, 64))]
    b = [p.asm() for p in vf.programs("f16", 7, 20, (32, 64))]
    assert a == b
    text = "\n".join(a)
    for name in ("v_add_f16", "v_sub_f16", "v_mul_f16", "v_max_f16", "v_min_f16", "v_cvt_f32_f16"):
        assert f"{name}_e64" not in text or "op_sel" not in "".join(
            ln for ln in text.splitlines() if ln.startswith(f"{name}_e64"))    # GFX10: no op_sel on these
    assert "v_subrev_f16" not in text                                          # claimed elsewhere (#374)


def test_skip_op_leaves_an_op_out():
    for p in vf.programs("f16", 3, 30, (32,), skip=("v_fma_f16", "v_pk_fma_f16", "v_fmac_f16")):
        assert not any(op["name"] in ("v_fma_f16", "v_pk_fma_f16", "v_fmac_f16") for op in p.ops)


# ------------------------------------------------------------------ wave family

def wave_with(op, wave=32):
    p = vf.WaveFamily(1, 0, wave)
    base = {"d": vf.POOL[0], "a": vf.POOL[1], "b": vf.POOL[2], "c": vf.POOL[3], "exec": p.full, "lane": 3,
            "cmp": "eq_u32", "cmask": 0, "mbmask": 0, "sel": list(range(16)), "mdk": "mov_lo", "text": "t"}
    base.update(op)
    p.ops = [base]
    return p


def test_mbcnt_counts_mask_bits_below_the_lane():
    p = wave_with({"kind": "mbcnt", "mbmask": 0b1011}, wave=32)
    regs, _ = p.model()
    a = vf.POOL[1]
    init = {t: regs[t][a].v for t in range(vf.THREADS)}
    assert [regs[t][vf.POOL[0]].v - init[t] for t in range(5)] == [0, 1, 2, 2, 3]
    assert regs[32][vf.POOL[0]].v - init[32] == 0          # second wave32 starts again at lane 0


def test_readfirstlane_uses_the_first_active_lane():
    p = wave_with({"kind": "readfirstlane", "exec": 0b1000}, wave=32)
    regs, _ = p.model()
    first = ((3 * (p.init[vf.POOL[1]][0] & 0xFFFFFF)) & vf.M32) ^ p.init[vf.POOL[1]][1]
    assert all(regs[t][vf.POOL[0]].v == first for t in range(32))


def test_permlanex16_reads_the_other_row():
    p = wave_with({"kind": "permlanex16", "sel": [0] * 16}, wave=32)
    regs, _ = p.model()
    a0 = lambda t: ((t * (p.init[vf.POOL[1]][0] & 0xFFFFFF)) & vf.M32) ^ p.init[vf.POOL[1]][1]  # noqa: E731
    assert regs[0][vf.POOL[0]].v == a0(16) and regs[16][vf.POOL[0]].v == a0(0)


def test_cmp_writes_zero_for_inactive_lanes():
    p = wave_with({"kind": "maskdata", "exec": 0b0101, "cmp": "ne_u32", "cmask": 0xFFFFFFFF}, wave=32)
    regs, _ = p.model()
    assert regs[0][vf.POOL[0]].v == 0b0101


# ------------------------------------------------------------------ memory family

def test_out_of_range_loads_read_zero_and_stores_drop():
    p = vf.MemFamily(1, 0, 32)
    p.raw_slots, p.soff = 2, 0
    r = vf.POOL[0]
    p.ops = [{"kind": "store_raw", "n": 1, "off": 0, "r": r, "soff": 0, "text": "s"},
             {"kind": "load_raw", "n": 1, "off": 0, "r": vf.POOL[1], "soff": 0, "text": "l"}]
    regs, mem = p.model()
    assert regs[0][vf.POOL[1]].v == regs[0][r].v          # in range: reads back what it stored
    assert regs[5][vf.POOL[1]].v == 0                     # out of range: 0
    assert mem[5 * vf.SLOT // 4] == 0xEEEEEEEE            # and its store was dropped


def test_mem_accesses_stay_in_the_threads_slot():
    for p in vf.programs("mem", 4, 40, (32,)):
        for op in p.ops:
            assert 0 <= op["off"] + op["soff"] and op["off"] + op["soff"] + 4 * op["n"] <= vf.SLOT


# ------------------------------------------------------------------ assembling (needs llvm-mc)

def _mc():
    try:
        sys.path.insert(0, os.path.join(ROOT, "runtime", "test", "shaders"))
        import mc_command
        cmd = mc_command.mc_command()
        subprocess.run(cmd + ["--version"], capture_output=True, check=True)
        return cmd
    except Exception:
        return None


@pytest.mark.skipif(_mc() is None, reason="llvm-mc with the AMDGPU target not found")
@pytest.mark.parametrize("family", sorted(vf.FAMILIES))
def test_programs_assemble_for_gfx1013(family):
    mc = _mc()
    for p in vf.programs(family, 1, 6, (32, 64)):
        cmd = mc + ["-triple=amdgcn--amdpal", "-mcpu=gfx1013", "-show-encoding"]
        if p.wave == 64:
            cmd.append("-mattr=+wavefrontsize64,-wavefrontsize32")
        r = subprocess.run(cmd, input=p.asm(), capture_output=True, text=True)
        assert r.returncode == 0 and "error:" not in r.stderr, r.stderr[-800:]


def test_model_only_runs_every_family(capsys, monkeypatch):
    for family in sorted(vf.FAMILIES):
        monkeypatch.setattr(sys, "argv", ["valu_fuzz.py", "--family", family, "--programs", "4", "--model-only"])
        assert vf.main() == 0
    assert "programs generated and modelled" in capsys.readouterr().out
