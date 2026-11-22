r"""SHADER-137: RDNA2 hardware probe for the integer contract rows checked only on the CPU.

Runs small, self-contained RDNA2 instruction sequences natively on an AMD GPU through tools/hw_oracle (wave32 and
wave64) and compares every output word with Leonida's live translator (tools/rdna2spirv.py ->
tests/semantic/spirv_mini_interp.py). Standard library only; no game files.

Groups covered (each references its contract row in docs/design.md):
    salu_absdiff_mulhi  s_absdiff_i32, s_mul_hi_i32, s_mul_hi_u32 (SCC written vs kept)
    salu_lshl_add       s_lshl1..4_add_u32 (64-bit carry / shift-out into SCC)
    salu_sext_flbit     s_sext_i32_i8, s_sext_i32_i16, s_flbit_i32 (SCC kept)
    salu_cmov_pack      s_cmov_b32, s_cmovk_i32, s_pack_ll/lh/hh_b32_b16 (SCC kept)
    salu_bfe_past31     s_bfe_u32, s_bfe_i32 with offset + width > 32 and width 0
    salu_64_data        s_brev_b64, s_bfe_u64/i64, s_bitset0/1_b64, s_flbit_i32_b64/i64
    salu_quad_bitrep    s_quadmask_b32/b64, s_bitreplicate_b64_b32
    salu_64_bitwise     s_and/or/xor/andn2/orn2/nand/nor/xnor/not_b64 on scalar data
    salu_movrels        s_movrels_b32, s_movrels_b64 (M0 indexing, SCC kept)
    valu_add_nc_clamp   v_add/sub/subrev_nc_u32, v_add/sub_nc_i32, v_add/sub_nc_u16 with clamp
    valu_add_co_clamp   v_add/sub/subrev_co[_ci]_u32 with clamp (saturation + carry)
    valu_mad64          v_mad_u64_u32, v_mad_i64_i32 (sdst bit 64 and inactive EXEC lane)
    valu_sad_lerp       v_sad_u8, v_sad_hi_u8, v_sad_u16, v_sad_u32, v_msad_u8, v_lerp_u8
    valu_qsad_mqsad     v_qsad_pk_u16_u8, v_mqsad_pk_u16_u8, v_mqsad_u32_u8
    valu_mul24_sh64     v_mul_hi_u32_u24, v_mul_hi_i32_i24, v_lshrrev_b64, v_ashrrev_i64
    valu_mad16          v_mad_u16, v_mad_i16 (clamp, op_sel, kept half)
    valu_cvt_16         v_cvt_f16_u16/i16, v_cvt_u16/i16_f16 (kept half, op_sel, NaN, inf)
    valu_cvt_flr_rpi    v_cvt_flr_i32_f32, v_cvt_rpi_i32_f32 (exact floor(x+0.5), signed NaN)
    valu_cvt_pknorm     v_cvt_pknorm_i16_f32, v_cvt_pknorm_u16_f32 (ties, saturation, NaN)

Usage:
    python -I tools/int_hw_probe.py                         # translator side only (CPU)
    python -I tools/int_hw_probe.py --oracle hw             # native AMD GPU only (wave32 + wave64)
    python -I tools/int_hw_probe.py --oracle both           # compare hardware vs translator
    python -I tools/int_hw_probe.py --oracle both --strict  # fail on known translator divergences too
    python -I tools/int_hw_probe.py --self-test

Exit codes:
    0   all unexpected comparisons match (and wave32 == wave64 on hardware)
    1   at least one unexpected mismatch, wave32/64 diff, or known divergence under --strict
    2   bad CLI arguments or harness error
    77  hardware oracle requested but unavailable on this machine
"""
import argparse
import json
import os
import struct
import sys
import tempfile
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.dont_write_bytecode = True
for p in (HERE, os.path.join(ROOT, "tests", "semantic")):
    if p not in sys.path:
        sys.path.insert(0, p)

import rdna2spirv  # noqa: E402
import spirv_mini_interp  # noqa: E402

STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)
SENTINEL = 0x5A5A5A5A

# Live translator divergences confirmed on AMD Radeon 680M (gfx1035, RDNA2, wave32 & wave64).
# Each entry maps case.name -> (tag, expected_hw_words, reason). Empty: the five measured on gfx1035 (VOP3 clamp
# on the carry-out add/sub ops, v_cvt_rpi_i32_f32's exact floor(x + 0.5), and v_cvt_flr/rpi's NaN saturation by
# sign) are translated as the hardware does since #1831 and #1856.
KNOWN_DIVERGENCES = {}


class Case:
    """One self-contained 4-word probe. `body` leaves results in v8..v11 and must not clobber s[0:3] or v0.
    For wave-size-dependent mask instructions, `{vcc}`, `{exec}`, `{smov}`, `{stmp}` expand to wave32/wave64 forms."""

    __slots__ = ("group", "name", "body", "note")

    def __init__(self, group, name, body, note=""):
        self.group = group
        self.name = name
        self.body = tuple(body)
        self.note = note

    def lines(self, wave=32):
        vcc = "vcc_lo" if wave == 32 else "vcc"
        exc = "exec_lo" if wave == 32 else "exec"
        smov = "s_mov_b32" if wave == 32 else "s_mov_b64"
        stmp = "s14" if wave == 32 else "s[14:15]"
        return [ln.format(vcc=vcc, exec=exc, smov=smov, stmp=stmp) for ln in self.body]


def _scc_seed(val):
    return "s_cmp_eq_u32 0, 0" if val else "s_cmp_eq_u32 0, 1"


def _salu32_pair(group, name, setup, op_line, note=""):
    """Run one 32-bit scalar op twice: once with SCC preset to 0 (v8=D, v9=SCC) and once with SCC preset to 1
    (v10=D, v11=SCC), so 'SCC kept' and 'SCC written' are both directly observed."""
    body = [
        *setup, _scc_seed(0), op_line,
        "s_cselect_b32 s30, 1, 0", "v_mov_b32 v8, s6", "v_mov_b32 v9, s30",
        *setup, _scc_seed(1), op_line,
        "s_cselect_b32 s30, 1, 0", "v_mov_b32 v10, s6", "v_mov_b32 v11, s30",
    ]
    return Case(group, name, body, note)


def _salu64_case(group, name, setup, op_line, scc_in, note=""):
    """Run one 64-bit scalar op with SCC preset to scc_in; output v8=s6 (lo), v9=s7 (hi), v10=SCC, v11=0."""
    body = [
        *setup, _scc_seed(scc_in), op_line,
        "s_cselect_b32 s30, 1, 0",
        "v_mov_b32 v8, s6", "v_mov_b32 v9, s7", "v_mov_b32 v10, s30", "v_mov_b32 v11, 0",
    ]
    return Case(group, name, body, note)


def _valu4_case(group, name, body, note=""):
    return Case(group, name, body, note)


def build_cases():
    cases = []

    # 1. s_absdiff_i32, s_mul_hi_i32, s_mul_hi_u32 (docs/design.md)
    cases += [
        _salu32_pair("salu_absdiff_mulhi", "s_absdiff_basic",
                     ["s_mov_b32 s4, 10", "s_mov_b32 s5, 25"],
                     "s_absdiff_i32 s6, s4, s5", "|10 - 25| = 15, SCC=1"),
        _salu32_pair("salu_absdiff_mulhi", "s_absdiff_equal",
                     ["s_mov_b32 s4, 0x80000000", "s_mov_b32 s5, 0x80000000"],
                     "s_absdiff_i32 s6, s4, s5", "equal -> 0, SCC=0"),
        _salu32_pair("salu_absdiff_mulhi", "s_absdiff_wrap_max_min",
                     ["s_mov_b32 s4, 0x7fffffff", "s_mov_b32 s5, 0x80000000"],
                     "s_absdiff_i32 s6, s4, s5", "0x7fffffff - 0x80000000 wraps to -1 -> abs 1, SCC=1"),
        _salu32_pair("salu_absdiff_mulhi", "s_absdiff_int_min",
                     ["s_mov_b32 s4, 0x80000000", "s_mov_b32 s5, 0"],
                     "s_absdiff_i32 s6, s4, s5", "diff = INT_MIN stays 0x80000000, SCC=1"),
        _salu32_pair("salu_absdiff_mulhi", "s_mul_hi_i32_min_min",
                     ["s_mov_b32 s4, 0x80000000", "s_mov_b32 s5, 0x80000000"],
                     "s_mul_hi_i32 s6, s4, s5", "(-2^31)*(-2^31) >> 32 = 0x40000000, SCC kept"),
        _salu32_pair("salu_absdiff_mulhi", "s_mul_hi_i32_neg_pos",
                     ["s_mov_b32 s4, 0xffffffff", "s_mov_b32 s5, 2"],
                     "s_mul_hi_i32 s6, s4, s5", "(-1)*2 >> 32 = 0xffffffff, SCC kept"),
        _salu32_pair("salu_absdiff_mulhi", "s_mul_hi_u32_max_max",
                     ["s_mov_b32 s4, 0xffffffff", "s_mov_b32 s5, 0xffffffff"],
                     "s_mul_hi_u32 s6, s4, s5", "(2^32-1)^2 >> 32 = 0xfffffffe, SCC kept"),
    ]

    # 2. s_lshl1..4_add_u32 (64-bit shift-out + add carry into SCC)
    for shift in (1, 2, 3, 4):
        op = f"s_lshl{shift}_add_u32"
        high_bit = 1 << (32 - shift)
        cases += [
            _salu32_pair("salu_lshl_add", f"{op}_no_carry",
                         ["s_mov_b32 s4, 0x00010000", "s_mov_b32 s5, 5"],
                         f"{op} s6, s4, s5", "no shift-out, no add carry -> SCC=0"),
            _salu32_pair("salu_lshl_add", f"{op}_shift_out_only",
                         [f"s_mov_b32 s4, 0x{high_bit:08x}", "s_mov_b32 s5, 0"],
                         f"{op} s6, s4, s5", "bit shifted out of S0 with S1=0 -> D=0, SCC=1"),
            _salu32_pair("salu_lshl_add", f"{op}_add_carry_only",
                         ["s_mov_b32 s4, 1", "s_mov_b32 s5, 0xffffffff"],
                         f"{op} s6, s4, s5", "add carry without shift-out -> SCC=1"),
        ]

    # 3. s_sext_i32_i8, s_sext_i32_i16, s_flbit_i32
    cases += [
        _salu32_pair("salu_sext_flbit", "s_sext_i8_neg",
                     ["s_mov_b32 s4, 0x12345680"], "s_sext_i32_i8 s6, s4", "0x80 -> 0xffffff80, SCC kept"),
        _salu32_pair("salu_sext_flbit", "s_sext_i8_pos",
                     ["s_mov_b32 s4, 0xffffff7f"], "s_sext_i32_i8 s6, s4", "0x7f -> 0x0000007f, SCC kept"),
        _salu32_pair("salu_sext_flbit", "s_sext_i16_neg",
                     ["s_mov_b32 s4, 0x00008001"], "s_sext_i32_i16 s6, s4", "0x8001 -> 0xffff8001, SCC kept"),
        _salu32_pair("salu_sext_flbit", "s_sext_i16_pos",
                     ["s_mov_b32 s4, 0xffff7fff"], "s_sext_i32_i16 s6, s4", "0x7fff -> 0x00007fff, SCC kept"),
        _salu32_pair("salu_sext_flbit", "s_flbit_i32_zero",
                     ["s_mov_b32 s4, 0"], "s_flbit_i32 s6, s4", "0 -> 0xffffffff, SCC kept"),
        _salu32_pair("salu_sext_flbit", "s_flbit_i32_neg1",
                     ["s_mov_b32 s4, 0xffffffff"], "s_flbit_i32 s6, s4", "-1 -> 0xffffffff, SCC kept"),
        _salu32_pair("salu_sext_flbit", "s_flbit_i32_pos1",
                     ["s_mov_b32 s4, 1"], "s_flbit_i32 s6, s4", "1 -> 30, SCC kept"),
        _salu32_pair("salu_sext_flbit", "s_flbit_i32_neg16",
                     ["s_mov_b32 s4, 0xfffffff0"], "s_flbit_i32 s6, s4", "-16 -> 27, SCC kept"),
        _salu32_pair("salu_sext_flbit", "s_flbit_i32_int_min",
                     ["s_mov_b32 s4, 0x80000000"], "s_flbit_i32 s6, s4", "INT_MIN -> 0, SCC kept"),
    ]

    # 4. s_cmov_b32, s_cmovk_i32, s_pack_ll/lh/hh_b32_b16
    cases += [
        _salu32_pair("salu_cmov_pack", "s_cmov_b32",
                     ["s_mov_b32 s4, 0x11223344", "s_mov_b32 s6, 0xaabbccdd"],
                     "s_cmov_b32 s6, s4", "SCC=0 keeps D, SCC=1 takes S0, SCC kept"),
        _salu32_pair("salu_cmov_pack", "s_cmovk_i32_neg",
                     ["s_mov_b32 s6, 0x12345678"],
                     "s_cmovk_i32 s6, -300", "SCC=0 keeps D, SCC=1 sign-extends imm16, SCC kept"),
        _salu32_pair("salu_cmov_pack", "s_pack_ll",
                     ["s_mov_b32 s4, 0x11112222", "s_mov_b32 s5, 0x33334444"],
                     "s_pack_ll_b32_b16 s6, s4, s5", "{S1.lo, S0.lo} = 0x44442222, SCC kept"),
        _salu32_pair("salu_cmov_pack", "s_pack_lh",
                     ["s_mov_b32 s4, 0x11112222", "s_mov_b32 s5, 0x33334444"],
                     "s_pack_lh_b32_b16 s6, s4, s5", "{S1.hi, S0.lo} = 0x33332222, SCC kept"),
        _salu32_pair("salu_cmov_pack", "s_pack_hh",
                     ["s_mov_b32 s4, 0x11112222", "s_mov_b32 s5, 0x33334444"],
                     "s_pack_hh_b32_b16 s6, s4, s5", "{S1.hi, S0.hi} = 0x33331111, SCC kept"),
    ]

    # 5. s_bfe_u32 / s_bfe_i32 past bit 31
    cases += [
        _salu32_pair("salu_bfe_past31", "s_bfe_i32_past31",
                     ["s_mov_b32 s4, 0xd2f90159", "s_mov_b32 s5, 0x7f001c"],
                     "s_bfe_i32 s6, s4, s5", "off=28 width=127 stops at bit 31 -> 0xfffffffd, SCC=1"),
        _salu32_pair("salu_bfe_past31", "s_bfe_u32_past31",
                     ["s_mov_b32 s4, 0xd2f90159", "s_mov_b32 s5, 0xff7f001c"],
                     "s_bfe_u32 s6, s4, s5", "upper bits ignored, stops at bit 31 -> 0x0000000d, SCC=1"),
        _salu32_pair("salu_bfe_past31", "s_bfe_i32_width0",
                     ["s_mov_b32 s4, 0xffffffff", "s_mov_b32 s5, 0x00000005"],
                     "s_bfe_i32 s6, s4, s5", "width=0 -> 0, SCC=0"),
    ]

    # 6. 64-bit scalar data ops (s_brev_b64, s_bfe_u64/i64, s_bitset0/1_b64, s_flbit_i32_b64/i64)
    cases += [
        _salu64_case("salu_64_data", "s_brev_b64_scc1",
                     ["s_mov_b32 s4, 0x80000001", "s_mov_b32 s5, 0x12345678"],
                     "s_brev_b64 s[6:7], s[4:5]", 1, "bit-reverse 64-bit pair, SCC kept"),
        _salu64_case("salu_64_data", "s_bfe_u64_cross_past63",
                     ["s_mov_b32 s4, 0x12345678", "s_mov_b32 s5, 0xd2f90159", "s_mov_b32 s8, 0x7f003c"],
                     "s_bfe_u64 s[6:7], s[4:5], s8", 0, "off=60 width=127 stops at bit 63 -> 0xd, SCC=1"),
        _salu64_case("salu_64_data", "s_bfe_i64_past63",
                     ["s_mov_b32 s4, 0x12345678", "s_mov_b32 s5, 0xd2f90159", "s_mov_b32 s8, 0x7f003c"],
                     "s_bfe_i64 s[6:7], s[4:5], s8", 0, "off=60 width=127 sign-extends 0xd -> -3, SCC=1"),
        _salu64_case("salu_64_data", "s_bfe_u64_width0",
                     ["s_mov_b32 s4, 0xffffffff", "s_mov_b32 s5, 0xffffffff", "s_mov_b32 s8, 0x00000010"],
                     "s_bfe_u64 s[6:7], s[4:5], s8", 1, "width=0 -> 0, SCC=0"),
        _salu64_case("salu_64_data", "s_bitset0_b64_hi",
                     ["s_mov_b32 s6, 0xffffffff", "s_mov_b32 s7, 0xffffffff", "s_mov_b32 s8, 63"],
                     "s_bitset0_b64 s[6:7], s8", 1, "clear bit 63, SCC kept"),
        _salu64_case("salu_64_data", "s_bitset1_b64_mod64",
                     ["s_mov_b32 s6, 0", "s_mov_b32 s7, 0", "s_mov_b32 s8, 97"],
                     "s_bitset1_b64 s[6:7], s8", 0, "set bit 97 & 63 = 33 (hi bit 1), SCC kept"),
        _salu32_pair("salu_64_data", "s_flbit_i32_b64_zero",
                     ["s_mov_b32 s4, 0", "s_mov_b32 s5, 0"],
                     "s_flbit_i32_b64 s6, s[4:5]", "0 -> -1, SCC kept"),
        _salu32_pair("salu_64_data", "s_flbit_i32_b64_lo",
                     ["s_mov_b32 s4, 0x00010000", "s_mov_b32 s5, 0"],
                     "s_flbit_i32_b64 s6, s[4:5]", "bit 16 -> 47 leading zeros, SCC kept"),
        _salu32_pair("salu_64_data", "s_flbit_i32_i64_neg",
                     ["s_mov_b32 s4, 0xfffffff0", "s_mov_b32 s5, 0xffffffff"],
                     "s_flbit_i32_i64 s6, s[4:5]", "-16 -> 59 sign bits, SCC kept"),
        _salu32_pair("salu_64_data", "s_flbit_i32_i64_neg1",
                     ["s_mov_b32 s4, 0xffffffff", "s_mov_b32 s5, 0xffffffff"],
                     "s_flbit_i32_i64 s6, s[4:5]", "-1 -> -1, SCC kept"),
    ]

    # 7. s_quadmask_b32/b64, s_bitreplicate_b64_b32
    cases += [
        _salu32_pair("salu_quad_bitrep", "s_quadmask_b32_nz",
                     ["s_mov_b32 s4, 0x10080041"],
                     "s_quadmask_b32 s6, s4", "quads 0,1,4,7 set -> 0x93, SCC=1"),
        _salu32_pair("salu_quad_bitrep", "s_quadmask_b32_zero",
                     ["s_mov_b32 s4, 0"],
                     "s_quadmask_b32 s6, s4", "0 -> 0, SCC=0"),
        _salu64_case("salu_quad_bitrep", "s_quadmask_b64_hi",
                     ["s_mov_b32 s4, 0x00000000", "s_mov_b32 s5, 0x80000001"],
                     "s_quadmask_b64 s[6:7], s[4:5]", 0, "quads 8,15 -> lo=0x8100 hi=0, SCC=1"),
        _salu64_case("salu_quad_bitrep", "s_bitreplicate_b64_b32",
                     ["s_mov_b32 s4, 0x80010005"],
                     "s_bitreplicate_b64_b32 s[6:7], s4", 1, "each bit doubled, SCC kept"),
    ]

    # 8. 64-bit scalar bitwise ops on scalar data
    for op in ("s_and_b64", "s_or_b64", "s_xor_b64", "s_andn2_b64", "s_orn2_b64",
               "s_nand_b64", "s_nor_b64", "s_xnor_b64"):
        cases.append(_salu64_case(
            "salu_64_bitwise", f"{op}_data",
            ["s_mov_b32 s4, 0xf0f000ff", "s_mov_b32 s5, 0x12345678",
             "s_mov_b32 s8, 0x0ff0ff00", "s_mov_b32 s9, 0x87654321"],
            f"{op} s[6:7], s[4:5], s[8:9]", 0, f"{op} on 64-bit scalar data + SCC",
        ))
    cases.append(_salu64_case(
        "salu_64_bitwise", "s_not_b64_all_ones",
        ["s_mov_b32 s4, 0xffffffff", "s_mov_b32 s5, 0xffffffff"],
        "s_not_b64 s[6:7], s[4:5]", 1, "~(-1) = 0, SCC=0",
    ))

    # 9. s_movrels_b32 / s_movrels_b64
    cases += [
        _salu32_pair("salu_movrels", "s_movrels_b32_m0_2",
                     ["s_mov_b32 s10, 0x11111111", "s_mov_b32 s11, 0x22222222",
                      "s_mov_b32 s12, 0x33333333", "s_mov_b32 m0, 2"],
                     "s_movrels_b32 s6, s10", "SGPR[10+2] = 0x33333333, SCC kept"),
        _salu64_case("salu_movrels", "s_movrels_b64_m0_2",
                     ["s_mov_b32 s10, 0x11111111", "s_mov_b32 s11, 0x22222222",
                      "s_mov_b32 s12, 0x33333333", "s_mov_b32 s13, 0x44444444", "s_mov_b32 m0, 2"],
                     "s_movrels_b64 s[6:7], s[10:11]", 1, "SGPR[12:13], SCC kept"),
    ]

    # 10. Integer clamp on v_add/sub/subrev_nc_u32, v_add/sub_nc_i32, v_add/sub_nc_u16
    cases += [
        _valu4_case("valu_add_nc_clamp", "v_add_sub_nc_u32_clamp", [
            "v_mov_b32 v1, 0xfffffff0", "v_mov_b32 v2, 0x30",
            "v_add_nc_u32_e64 v8, v1, v2 clamp",      # saturates to 0xffffffff
            "v_sub_nc_u32_e64 v9, v2, v1 clamp",      # saturates to 0
            "v_subrev_nc_u32_e64 v10, v1, v2 clamp",  # v2 - v1 -> saturates to 0
            "v_subrev_nc_u32_e64 v11, v2, v1 clamp",  # v1 - v2 = 0xffffffc0
        ], "u32 nc add/sub/subrev with clamp"),
        _valu4_case("valu_add_nc_clamp", "v_add_sub_nc_i32_clamp", [
            "v_mov_b32 v1, 0x7ffffff0", "v_mov_b32 v2, 0x30",
            "v_mov_b32 v3, 0x80000010", "v_mov_b32 v4, 0x30",
            "v_add_nc_i32_e64 v8, v1, v2 clamp",      # +overflow -> 0x7fffffff
            "v_sub_nc_i32_e64 v9, v3, v4 clamp",      # -overflow -> 0x80000000
            "v_add_nc_i32_e64 v10, v3, v3 clamp",     # neg+neg overflow -> 0x80000000
            "v_sub_nc_i32_e64 v11, v1, v3 clamp",     # pos-neg overflow -> 0x7fffffff
        ], "i32 nc add/sub with clamp"),
        _valu4_case("valu_add_nc_clamp", "v_add_sub_nc_u16_clamp", [
            "v_mov_b32 v1, 0x1234fff0", "v_mov_b32 v2, 0x56780030",
            f"v_mov_b32 v8, 0x{SENTINEL:08x}",
            f"v_mov_b32 v9, 0x{SENTINEL:08x}",
            f"v_mov_b32 v10, 0x{SENTINEL:08x}",
            f"v_mov_b32 v11, 0x{SENTINEL:08x}",
            "v_add_nc_u16_e64 v8, v1, v2 clamp",                       # lo saturates 0xffff, hi kept
            "v_sub_nc_u16_e64 v9, v2, v1 clamp",                       # lo saturates 0x0000, hi kept
            "v_add_nc_u16_e64 v10, v1, v2 op_sel:[0,0,1] clamp",       # write hi saturates 0xffff, lo kept
            "v_sub_nc_u16_e64 v11, v1, v2 op_sel:[1,1,0] clamp",       # 0x1234 - 0x5678 -> 0x0000, hi kept
        ], "u16 nc add/sub with clamp and op_sel (untouched half kept)"),
    ]

    # 11. v_add/sub/subrev_co[_ci]_u32 with clamp (known divergence: translator ignores clamp)
    cases += [
        _valu4_case("valu_add_co_clamp", "v_add_co_u32_clamp", [
            "v_mov_b32 v1, 0xffffffff", "v_mov_b32 v2, 0xffffffff",
            "v_add_co_u32 v8, {vcc}, v1, v2 clamp",
            "v_cndmask_b32_e64 v9, 0, 1, {vcc}",
            "v_mov_b32 v3, 5", "v_mov_b32 v4, 10",
            "v_sub_co_u32 v10, {vcc}, v3, v4 clamp",
            "v_cndmask_b32_e64 v11, 0, 1, {vcc}",
        ], "v_add_co_u32 / v_sub_co_u32 clamp: hardware saturates D and reports real carry/borrow"),
        _valu4_case("valu_add_co_clamp", "v_subrev_co_u32_clamp", [
            "v_mov_b32 v1, 10", "v_mov_b32 v2, 5",
            "v_subrev_co_u32 v8, {vcc}, v1, v2 clamp",
            "v_cndmask_b32_e64 v9, 0, 1, {vcc}",
            "v_mov_b32 v3, 0", "v_mov_b32 v4, 0",
            "{smov} {vcc}, -1",
            "v_subrev_co_ci_u32_e64 v10, {vcc}, v3, v4, {vcc} clamp",
            "v_cndmask_b32_e64 v11, 0, 1, {vcc}",
        ], "v_subrev_co_u32 / v_subrev_co_ci_u32 clamp: hardware saturates D to 0 and reports borrow=1"),
        _valu4_case("valu_add_co_clamp", "v_add_co_ci_u32_clamp", [
            "v_mov_b32 v1, 0xffffffff", "v_mov_b32 v2, 0",
            "{smov} {vcc}, -1",
            "v_add_co_ci_u32_e64 v8, {vcc}, v1, v2, {vcc} clamp",
            "v_cndmask_b32_e64 v9, 0, 1, {vcc}",
            "v_mov_b32 v3, 0", "v_mov_b32 v4, 0",
            "{smov} {vcc}, -1",
            "v_sub_co_ci_u32_e64 v10, {vcc}, v3, v4, {vcc} clamp",
            "v_cndmask_b32_e64 v11, 0, 1, {vcc}",
        ], "v_add_co_ci_u32 / v_sub_co_ci_u32 clamp: carry-in overflow saturates D and sets carry-out"),
    ]

    # 12. v_mad_u64_u32, v_mad_i64_i32 (sdst bit 64 and EXEC masking)
    cases += [
        _valu4_case("valu_mad64", "v_mad_u64_u32_carry", [
            "v_mov_b32 v1, 0xffffffff", "v_mov_b32 v2, 0xffffffff",
            "v_mov_b32 v4, 0xffffffff", "v_mov_b32 v5, 0xffffffff",
            "v_mad_u64_u32 v[8:9], {vcc}, v1, v2, v[4:5]",
            "v_cndmask_b32_e64 v10, 0, 1, {vcc}",
            "v_mov_b32 v11, 0",
        ], "u64 max*max + max64 -> carry=1"),
        _valu4_case("valu_mad64", "v_mad_i64_i32_pos_overflow", [
            "v_mov_b32 v1, 0x80000000", "v_mov_b32 v2, 0x80000000",
            "v_mov_b32 v4, 0xffffffff", "v_mov_b32 v5, 0x7fffffff",
            "v_mad_i64_i32 v[8:9], {vcc}, v1, v2, v[4:5]",
            "v_cndmask_b32_e64 v10, 0, 1, {vcc}",
            "v_mov_b32 v11, 0",
        ], "(-2^31)^2 + INT64_MAX > 0 (signed overflow, positive exact sum -> sdst=0)"),
        _valu4_case("valu_mad64", "v_mad_i64_i32_neg_sum", [
            "v_mov_b32 v1, 0xfffffffe", "v_mov_b32 v2, 3",
            "v_mov_b32 v4, 0", "v_mov_b32 v5, 0",
            "v_mad_i64_i32 v[8:9], {vcc}, v1, v2, v[4:5]",
            "v_cndmask_b32_e64 v10, 0, 1, {vcc}",
            "v_mov_b32 v11, 0",
        ], "(-2)*3 + 0 = -6 -> sdst=1 (bit 64 sign of exact 65-bit sum)"),
        _valu4_case("valu_mad64", "v_mad_u64_u32_inactive_lane", [
            "v_mov_b32 v1, 0xffffffff", "v_mov_b32 v2, 0xffffffff",
            "v_mov_b32 v4, 0xffffffff", "v_mov_b32 v5, 0xffffffff",
            "v_mov_b32 v8, 0x11111111", "v_mov_b32 v9, 0x22222222",
            "{smov} {vcc}, -1",
            "{smov} {stmp}, {exec}",
            "{smov} {exec}, 0",
            "v_mad_u64_u32 v[8:9], {vcc}, v1, v2, v[4:5]",
            "{smov} {exec}, {stmp}",
            "v_cndmask_b32_e64 v10, 0, 1, {vcc}",
            "v_mov_b32 v11, 0",
        ], "inactive EXEC lane: D untouched and sdst bit cleared to 0"),
    ]

    # 13. v_sad_u8, v_sad_hi_u8, v_sad_u16, v_sad_u32, v_msad_u8, v_lerp_u8
    cases += [
        _valu4_case("valu_sad_lerp", "v_sad_ops", [
            "v_mov_b32 v1, 0xff801001", "v_mov_b32 v2, 0x0080f002", "v_mov_b32 v3, 0xffffff00",
            "v_sad_u8 v8, v1, v2, v3",       # wraps 32-bit
            "v_sad_hi_u8 v9, v1, v2, v3",    # (byte_sad << 16) + S2
            "v_sad_u16 v10, v1, v2, v3",     # 2x16-bit SAD + S2
            "v_sad_u32 v11, v1, v2, v3",     # |v1 - v2| + S2
        ], "v_sad_u8 / v_sad_hi_u8 / v_sad_u16 / v_sad_u32 with 32-bit wrap"),
        _valu4_case("valu_sad_lerp", "v_msad_and_lerp", [
            "v_mov_b32 v1, 0xff801001", "v_mov_b32 v2, 0x00800005", "v_mov_b32 v3, 100",
            "v_msad_u8 v8, v1, v2, v3",      # skips bytes 1 and 3 where S1 byte == 0
            "v_mov_b32 v4, 0xff030200", "v_mov_b32 v5, 0xff040301", "v_mov_b32 v6, 0x01000100",
            "v_lerp_u8 v9, v4, v5, v6",      # per-byte (a+b+(c&1))>>1
            "v_mov_b32 v7, 0x00feffff",
            "v_lerp_u8 v10, v4, v5, v7",     # upper bits of S2 bytes ignored
            "v_mov_b32 v11, 0",
        ], "v_msad_u8 zero-byte mask and v_lerp_u8 rounding/no-carry"),
    ]

    # 14. v_qsad_pk_u16_u8, v_mqsad_pk_u16_u8, v_mqsad_u32_u8
    cases += [
        _valu4_case("valu_qsad_mqsad", "v_qsad_mqsad_pk_u16", [
            "v_mov_b32 v2, 0x40302010", "v_mov_b32 v3, 0x80706050",
            "v_mov_b32 v4, 0x00150025",      # S1 has two zero bytes for mqsad
            "v_mov_b32 v6, 0xfffe0005", "v_mov_b32 v7, 0xfff00100",
            "v_qsad_pk_u16_u8 v[8:9], v[2:3], v4, v[6:7]",
            "v_mqsad_pk_u16_u8 v[10:11], v[2:3], v4, v[6:7]",
        ], "v_qsad_pk_u16_u8 and v_mqsad_pk_u16_u8 with 16-bit wrap and zero S1 bytes"),
        _valu4_case("valu_qsad_mqsad", "v_mqsad_u32_u8", [
            "v_mov_b32 v2, 0x40302010", "v_mov_b32 v3, 0x80706050",
            "v_mov_b32 v4, 0x00150025",
            "v_mov_b32 v12, 0xfffffffe", "v_mov_b32 v13, 10",
            "v_mov_b32 v14, 20", "v_mov_b32 v15, 0xffffffff",
            "v_mqsad_u32_u8 v[8:11], v[2:3], v4, v[12:15]",
        ], "v_mqsad_u32_u8 four 32-bit accumulators with wrap"),
    ]

    # 15. v_mul_hi_u32_u24, v_mul_hi_i32_i24, v_lshrrev_b64, v_ashrrev_i64
    cases += [
        _valu4_case("valu_mul24_sh64", "v_mul_hi_24", [
            "v_mov_b32 v1, 0xabffffff", "v_mov_b32 v2, 0xcdffffff",
            "v_mul_hi_u32_u24 v8, v1, v2",   # upper 8 bits ignored
            "v_mul_hi_i32_i24 v9, v1, v2",   # (-1)*(-1) >> 32 = 0
            "v_mov_b32 v3, 0x12800000", "v_mov_b32 v4, 0x347fffff",
            "v_mul_hi_i32_i24 v10, v3, v4",  # (-2^23)*(2^23-1) >> 32
            "v_mul_hi_u32_u24 v11, v3, v4",
        ], "v_mul_hi_u32_u24 and v_mul_hi_i32_i24 (upper 8 bits ignored)"),
        _valu4_case("valu_mul24_sh64", "v_lshr_ashr_64", [
            "v_mov_b32 v2, 0x12345678", "v_mov_b32 v3, 0x87654321",
            "v_mov_b32 v1, 36",
            "v_lshrrev_b64 v[8:9], v1, v[2:3]",
            "v_ashrrev_i64 v[10:11], v1, v[2:3]",
        ], "v_lshrrev_b64 and v_ashrrev_i64 shift=36"),
        _valu4_case("valu_mul24_sh64", "v_lshr_ashr_64_mod64", [
            "v_mov_b32 v2, 0x12345678", "v_mov_b32 v3, 0x87654321",
            "v_mov_b32 v1, 68",              # 68 & 63 = 4
            "v_lshrrev_b64 v[8:9], v1, v[2:3]",
            "v_ashrrev_i64 v[10:11], v1, v[2:3]",
        ], "v_lshrrev_b64 and v_ashrrev_i64 shift masked to 6 bits"),
    ]

    # 16. v_mad_u16 / v_mad_i16 (clamp, op_sel, kept half)
    cases += [
        _valu4_case("valu_mad16", "v_mad_u16_i16_clamp_opsel", [
            "v_mov_b32 v1, 0x80000300", "v_mov_b32 v2, 0x00020200", "v_mov_b32 v3, 0x00010010",
            f"v_mov_b32 v8, 0x{SENTINEL:08x}",
            f"v_mov_b32 v9, 0x{SENTINEL:08x}",
            f"v_mov_b32 v10, 0x{SENTINEL:08x}",
            f"v_mov_b32 v11, 0x{SENTINEL:08x}",
            "v_mad_u16 v8, v1, v2, v3 clamp",                         # 0x300*0x200+0x10 = 0x60010 -> 0xffff
            "v_mad_u16 v9, v1, v2, v3",                               # wraps to 0x0010, hi kept
            "v_mad_i16 v10, v1, v2, v3 op_sel:[1,1,1,0] clamp",       # (-32768)*2 + 1 = -65535 -> 0x8000
            "v_mad_i16 v11, v1, v2, v3 op_sel:[0,0,0,1] clamp",       # 0x300*0x200+0x10 -> 0x7fff in hi, lo kept
        ], "v_mad_u16 / v_mad_i16 clamp, op_sel and untouched destination half"),
    ]

    # 17. v_cvt_f16_u16 / v_cvt_f16_i16 / v_cvt_u16_f16 / v_cvt_i16_f16 (kept half, NaN, inf)
    cases += [
        _valu4_case("valu_cvt_16", "v_cvt_u16_i16_f16_e64_nan_inf", [
            "v_mov_b32 v1, 0x00007e00",      # +qNaN
            "v_mov_b32 v2, 0x0000fe00",      # -qNaN
            "v_mov_b32 v3, 0x00007c00",      # +inf
            "v_mov_b32 v4, 0x0000fc00",      # -inf
            f"v_mov_b32 v8, 0x{SENTINEL:08x}",
            f"v_mov_b32 v9, 0x{SENTINEL:08x}",
            f"v_mov_b32 v10, 0x{SENTINEL:08x}",
            f"v_mov_b32 v11, 0x{SENTINEL:08x}",
            "v_cvt_u16_f16_e64 v8, v1",      # +qNaN -> 0, check high half
            "v_cvt_i16_f16_e64 v9, v2",      # -qNaN -> 0, check high half
            "v_cvt_u16_f16_e64 v10, v3",     # +inf -> 0xffff, check high half
            "v_cvt_i16_f16_e64 v11, v4",     # -inf -> 0x8000, check high half
        ], "v_cvt_u16/i16_f16_e64 on +-qNaN and +-inf (high half behavior)"),
        _valu4_case("valu_cvt_16", "v_cvt_f16_u16_i16_e64_keep", [
            "v_mov_b32 v1, 0x0000fff0",      # 65520 -> +inf (0x7c00)
            "v_mov_b32 v2, 0x00008000",      # -32768 -> -32768.0 (0xf800)
            "v_mov_b32 v3, 0x0000002a",      # 42 -> 42.0 (0x5140)
            "v_mov_b32 v4, 0x00007fff",      # 32767 -> 32768.0 (0x7800)
            f"v_mov_b32 v8, 0x{SENTINEL:08x}",
            f"v_mov_b32 v9, 0x{SENTINEL:08x}",
            f"v_mov_b32 v10, 0x{SENTINEL:08x}",
            f"v_mov_b32 v11, 0x{SENTINEL:08x}",
            "v_cvt_f16_u16_e64 v8, v1",
            "v_cvt_f16_i16_e64 v9, v2",
            "v_cvt_f16_u16_e64 v10, v3",
            "v_cvt_f16_i16_e64 v11, v4",
        ], "v_cvt_f16_u16/i16_e64 overflow to +inf and high half behavior"),
    ]

    # 18. v_cvt_flr_i32_f32 / v_cvt_rpi_i32_f32 (exact floor(x+0.5) + signed NaN; v_cvt_i32_f32 control)
    cases += [
        _valu4_case("valu_cvt_flr_rpi", "v_cvt_rpi_precision", [
            "v_mov_b32 v1, 0x3effffff",      # 0.49999997 (f32(x+0.5)==1.0, exact floor(x+0.5)==0)
            "v_mov_b32 v2, 0x4b000001",      # 8388609.0  (f32(x+0.5)==8388610.0, exact==8388609)
            "v_mov_b32 v3, 0x4b7fffff",      # 16777215.0 (f32(x+0.5)==16777216.0, exact==16777215)
            "v_mov_b32 v4, 0xbf000000",      # -0.5 -> floor(-0.5 + 0.5) = 0
            "v_cvt_rpi_i32_f32 v8, v1",
            "v_cvt_rpi_i32_f32 v9, v2",
            "v_cvt_rpi_i32_f32 v10, v3",
            "v_cvt_rpi_i32_f32 v11, v4",
        ], "v_cvt_rpi_i32_f32 exact floor(x+0.5) vs binary32 x+0.5"),
        _valu4_case("valu_cvt_flr_rpi", "v_cvt_flr_rpi_nan_inf", [
            "v_mov_b32 v1, 0x7fc00000",      # +qNaN
            "v_mov_b32 v2, 0xffc00000",      # -qNaN
            "v_cvt_flr_i32_f32 v8, v1",
            "v_cvt_flr_i32_f32 v9, v2",
            "v_cvt_rpi_i32_f32 v10, v1",
            "v_cvt_rpi_i32_f32 v11, v2",
        ], "v_cvt_flr_i32_f32 / v_cvt_rpi_i32_f32 on +qNaN and -qNaN"),
        _valu4_case("valu_cvt_flr_rpi", "v_cvt_i32_f32_nan_control", [
            "v_mov_b32 v1, 0x7fc00000",      # +qNaN
            "v_mov_b32 v2, 0xffc00000",      # -qNaN
            "v_mov_b32 v3, 0x7f800001",      # +sNaN
            "v_mov_b32 v4, 0xff800001",      # -sNaN
            "v_cvt_i32_f32 v8, v1",
            "v_cvt_i32_f32 v9, v2",
            "v_cvt_i32_f32 v10, v3",
            "v_cvt_i32_f32 v11, v4",
        ], "control: v_cvt_i32_f32 returns 0 for every NaN"),
        _valu4_case("valu_cvt_flr_rpi", "v_cvt_flr_edges", [
            "v_mov_b32 v1, 0xbf800001",      # -1.0000001 -> -2
            "v_mov_b32 v2, 0x7f800000",      # +inf -> INT_MAX
            "v_mov_b32 v3, 0xff800000",      # -inf -> INT_MIN
            "v_mov_b32 v4, 0x4f000000",      # +2^31 -> INT_MAX
            "v_cvt_flr_i32_f32 v8, v1",
            "v_cvt_flr_i32_f32 v9, v2",
            "v_cvt_flr_i32_f32 v10, v3",
            "v_cvt_flr_i32_f32 v11, v4",
        ], "v_cvt_flr_i32_f32 negative fraction, infinities and 2^31 saturation"),
    ]

    # 19. v_cvt_pknorm_i16_f32 / v_cvt_pknorm_u16_f32 (NaN, inf, out-of-range, ties)
    cases += [
        _valu4_case("valu_cvt_pknorm", "v_cvt_pknorm_nan_inf", [
            "v_mov_b32 v1, 0x7fc00000",      # +qNaN
            "v_mov_b32 v2, 0xffc00000",      # -qNaN
            "v_mov_b32 v3, 0x7f800000",      # +inf
            "v_mov_b32 v4, 0xff800000",      # -inf
            "v_cvt_pknorm_i16_f32 v8, v1, v2",
            "v_cvt_pknorm_u16_f32 v9, v1, v2",
            "v_cvt_pknorm_i16_f32 v10, v3, v4",
            "v_cvt_pknorm_u16_f32 v11, v3, v4",
        ], "v_cvt_pknorm_i16/u16_f32 on +-qNaN and +-inf"),
        _valu4_case("valu_cvt_pknorm", "v_cvt_pknorm_ties", [
            "v_mov_b32 v1, 0x3f000000",      # 0.5 -> 0.5*32767 = 16383.5 (RNE=16384), 0.5*65535 = 32767.5 (RNE=32768)
            "v_mov_b32 v2, 0xbf000000",      # -0.5 -> -16383.5 (RNE=-16384), u16 -> 0
            "v_mov_b32 v3, 0x3f800000",      # 1.0 -> 32767 / 65535
            "v_mov_b32 v4, 0xbf800000",      # -1.0 -> -32767 (0x8001) / 0
            "v_cvt_pknorm_i16_f32 v8, v1, v2",
            "v_cvt_pknorm_u16_f32 v9, v1, v2",
            "v_cvt_pknorm_i16_f32 v10, v3, v4",
            "v_cvt_pknorm_u16_f32 v11, v3, v4",
        ], "v_cvt_pknorm_i16/u16_f32 ties at +-0.5 and +-1.0 endpoints"),
    ]

    return cases


def _translate_case(work, case):
    os.makedirs(os.path.join(work, "disasm"), exist_ok=True)
    name = f"ps_int_hw_{case.name}"
    src = os.path.join(work, "disasm", name + ".s")
    body = case.lines(wave=32) + ["exp mrt0 v8, v9, v10, v11 done vm", "s_endpgm"]
    with open(src, "w", encoding="utf-8", newline="\n") as f:
        f.write(f"; {name}\n" + "".join(f"\t{ln}\n" for ln in body))
    with open(os.path.join(work, name + ".dxbc"), "wb") as f:
        f.write(STUB_DXBC)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)
        spv = rdna2spirv.translate(src)
    return list(spirv_mini_interp.run_fragment(spv)["%out_mrt0"])


def run_translator(cases):
    out = {}
    with tempfile.TemporaryDirectory(prefix="int_hw_probe_") as work:
        for case in cases:
            try:
                out[case.name] = _translate_case(work, case)
            except Exception as e:
                out[case.name] = f"error: {type(e).__name__}: {e}"
    return out


def run_hw(cases, arch=None, waves=(32, 64)):
    from hw_oracle import asm, runner  # noqa: E402
    oracle = runner.open_oracle(arch=arch)
    try:
        device = oracle.device
        per_wave = {}
        for wave in waves:
            kernels = [asm.Kernel(case.lines(wave=wave), wave=wave, threads=1, words=4) for case in cases]
            res = oracle.run(kernels, words=4)
            per_wave[wave] = {case.name: r for case, r in zip(cases, res)}
        return device, per_wave
    finally:
        oracle.close()


def _fmt_words(w):
    if isinstance(w, list):
        return " ".join(f"0x{x:08x}" for x in w)
    return str(w)


def self_test():
    cases = build_cases()
    names = [c.name for c in cases]
    assert len(names) == len(set(names)), "duplicate case names"
    for name in KNOWN_DIVERGENCES:
        assert name in names, f"unknown KNOWN_DIVERGENCES key: {name}"
    res = run_translator(cases)
    errors = {k: v for k, v in res.items() if not isinstance(v, list)}
    assert not errors, f"translator errors: {errors}"
    print(f"self-test: {len(cases)} cases across {len({c.group for c in cases})} groups translated and ran on CPU OK")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--oracle", choices=("translator", "hw", "both"), default="translator")
    ap.add_argument("--wave", choices=("32", "64", "both"), default="both")
    ap.add_argument("--group", action="append", default=[], help="run only named group(s)")
    ap.add_argument("--hw-arch", default=None, help="override GPU architecture for hw_oracle")
    ap.add_argument("--list-groups", action="store_true")
    ap.add_argument("--strict", action="store_true", help="treat known translator divergences as failures")
    ap.add_argument("--json", dest="json_out", default=None, help="write raw results to a JSON file")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args(argv)

    if args.self_test:
        return self_test()

    cases = build_cases()
    groups = []
    for c in cases:
        if c.group not in groups:
            groups.append(c.group)
    if args.list_groups:
        for g in groups:
            n = sum(1 for c in cases if c.group == g)
            print(f"{g:<22} ({n} cases)")
        return 0

    if args.group:
        unknown = [g for g in args.group if g not in groups]
        if unknown:
            print(f"unknown group(s): {', '.join(unknown)}", file=sys.stderr)
            return 2
        want = set(args.group)
        cases = [c for c in cases if c.group in want]

    waves = (32, 64) if args.wave == "both" else (int(args.wave),)
    tr_res = run_translator(cases) if args.oracle in ("translator", "both") else None
    hw_device, hw_res = None, None
    if args.oracle in ("hw", "both"):
        from hw_oracle import runner  # noqa: E402
        try:
            hw_device, hw_res = run_hw(cases, arch=args.hw_arch, waves=waves)
        except runner.Unavailable as e:
            print(f"hardware oracle unavailable: {e}", file=sys.stderr)
            return 77
        print(f"hardware oracle: {hw_device} (waves: {','.join(str(w) for w in waves)})")

    mismatches = 0
    known_diffs = 0
    wave_diffs = 0
    records = []
    for c in cases:
        tr = tr_res[c.name] if tr_res is not None else None
        h32 = hw_res[32][c.name] if hw_res and 32 in hw_res else None
        h64 = hw_res[64][c.name] if hw_res and 64 in hw_res else None
        hw_ref = h32 if h32 is not None else h64
        status = "OK"
        tag = None
        if h32 is not None and h64 is not None and h32 != h64:
            status = "WAVE_DIFF"
            wave_diffs += 1
        elif (tr is not None and not isinstance(tr, list)) or (hw_ref is not None and not isinstance(hw_ref, list)):
            status = "ERROR"
            mismatches += 1
        elif tr is not None and hw_ref is not None and tr != hw_ref:
            known = KNOWN_DIVERGENCES.get(c.name)
            if known and hw_ref == known[1]:
                status = "KNOWN_DIFF"
                tag = known[0]
                known_diffs += 1
            else:
                status = "MISMATCH"
                mismatches += 1
        records.append({
            "group": c.group, "name": c.name, "note": c.note, "status": status, "known_tag": tag,
            "translator": tr, "hw_w32": h32, "hw_w64": h64,
        })
        if status != "OK" or args.oracle != "both":
            label = f"{status}:{tag}" if tag else status
            print(f"[{label:<22}] {c.group}/{c.name}: {c.note}")
            if h32 is not None:
                print(f"                         hw_w32: {_fmt_words(h32)}")
            if h64 is not None and h64 != h32:
                print(f"                         hw_w64: {_fmt_words(h64)}")
            if tr is not None:
                print(f"                         transl: {_fmt_words(tr)}")

    total = len(cases)
    matched = total - mismatches - known_diffs - wave_diffs
    print(f"summary: {total} cases: {matched} match, {known_diffs} known divergence, "
          f"{mismatches} unexpected mismatch, {wave_diffs} wave32/64 diff")

    if args.json_out:
        out_path = os.path.abspath(args.json_out)
        if os.path.isdir(out_path):
            print(f"ERROR: --json path is a directory: {out_path}", file=sys.stderr)
            return 2
        parent = os.path.dirname(out_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(out_path, "w", encoding="utf-8", newline="\n") as f:
            json.dump({"device": hw_device, "waves": list(waves), "cases": records}, f, indent=2)
            f.write("\n")

    if mismatches or wave_diffs or (args.strict and known_diffs):
        return 1
    return 0


if __name__ == "__main__":
    try:  # a redirected stdout is strict ANSI code page under -I; write '?' for what it cannot encode
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    sys.exit(main())
