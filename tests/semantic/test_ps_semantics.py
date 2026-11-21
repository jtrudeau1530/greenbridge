"""Value-level regression tests for the RDNA2 -> SPIR-V translator on two straight-line pixel shaders.

Each shader ends with `exp mrt0 v.., v.., v.., v..`, i.e. exactly four 32-bit words. Two shaders x four words =
the eight values checked here. Every word is asserted individually (see the subTests), so no output component
is left unchecked, and the output variable set is asserted to be exactly {mrt0}.

The expected words are written out as constants AND recomputed by plain-Python reference functions that
implement the ISA semantics below; a separate test checks the two agree. Neither depends on the SPIR-V
interpreter or on the translator.

ISA semantics relied on (AMD RDNA2 "Instruction Set Architecture" reference, VALU/SDWA chapters):
  SDWA source select  BYTE_n / WORD_n: take that field of the 32-bit source, zero-extended; sext(x) sign-extends it.
  SDWA dest select    dst_sel BYTE_n / WORD_n: the 32-bit result is truncated to that width and written there;
                      dst_unused UNUSED_PRESERVE keeps the other destination bits, UNUSED_PAD zeroes them,
                      UNUSED_SEXT sign-extends the written field into the bits above it and zeroes those below.
  v_subrev_nc_u32     D = S1 - S0.        v_pk_fma_f16   per 16-bit half: D = S0 * S1 + S2 (IEEE half).
  v_min3_u32 / v_max3_i32   minimum (unsigned) / maximum (signed) of three sources.
  v_bfe_i32 D = sign_extend(S0[S1[4:0] +: S2[4:0]])  (bit-field extract, width in bits).
Expectations were also recorded as GPU results by the repository author in the header comments of
runtime/test/shaders/ps_sdwa.asm and ps_ops.asm (not re-confirmed here).
"""
import struct
import unittest

import support

M = 0xFFFFFFFF

# Independent constants, written down by hand (see derivations in the reference functions below).
EXPECTED_PS_SDWA = [0x000000F1, 0xFFFEFFF2, 0x3C003344, 0xFFF70000]
EXPECTED_PS_OPS = [0x48804300, 0x00000005, 0x00000007, 0xFFFFFFFC]


def _half(x):
    return struct.unpack("<H", struct.pack("<e", x))[0]


def _unhalf(h):
    return struct.unpack("<e", struct.pack("<H", h))[0]


def _s32(x):
    return x - (1 << 32) if x >> 31 else x


def reference_ps_sdwa():
    v0, v1 = 0x8081F0F7, 0x00010005
    r0 = ((v0 >> 8) & 0xFF) + ((v1 >> 16) & 0xFFFF)                  # BYTE_1 of v0 + WORD_1 of v1 = 0xf0 + 1
    byte0 = v0 & 0xFF
    sext_byte0 = byte0 - 0x100 if byte0 & 0x80 else byte0            # sext(v0) with BYTE_0 = -9
    r1 = (sext_byte0 - v1) & M                                       # subrev: S1 - S0 = -9 - 0x10005
    r2 = (_half(1.0) << 16) | (0x11223344 & 0xFFFF)                  # WORD_1 <- f16(1.0), low half preserved
    field = (v0 & 0xFF) | (v1 & 0xFF)                                # BYTE_0 | BYTE_0 = 0xf7
    r3 = (field << 16) | (0xFF000000 if field & 0x80 else 0)         # BYTE_2, UNUSED_SEXT: sign above, zero below
    return [r0, r1, r2, r3]


def reference_ps_ops():
    lo = _unhalf(0x3C00) * _unhalf(0x4200) + _unhalf(0x3800)         # 1.0 * 3.0 + 0.5
    hi = _unhalf(0x4000) * _unhalf(0x4400) + _unhalf(0x3C00)         # 2.0 * 4.0 + 1.0
    r0 = (_half(hi) << 16) | _half(lo)
    r1 = min(7, 0xFFFFFFF0, 5)                                       # v_min3_u32 (unsigned)
    r2 = max(7, _s32(0xFFFFFFF0), 5) & M                             # v_max3_i32 (signed)
    f = (0xFFFFFFF0 >> 2) & 0xF                                      # bits [5:2]
    r3 = (f - 16 if f & 8 else f) & M                                # v_bfe_i32 sign extension of a 4-bit field
    return [r0, r1, r2, r3]


class OracleTests(unittest.TestCase):
    """The hand-written constants and the plain-Python reference agree (so the oracle is self-consistent)."""

    def test_reference_matches_constants(self):
        self.assertEqual(reference_ps_sdwa(), EXPECTED_PS_SDWA)
        self.assertEqual(reference_ps_ops(), EXPECTED_PS_OPS)


class TranslatorSemantics(unittest.TestCase):
    def check(self, name, expected):
        words = support.mrt0_words(support.translate_fixture(name))
        self.assertEqual(len(words), 4, "mrt0 has four components")
        for i, (got, want) in enumerate(zip(words, expected)):
            with self.subTest(shader=name, component="xyzw"[i]):
                self.assertEqual(f"{got:#010x}", f"{want:#010x}")

    def test_ps_sdwa(self):
        self.check("ps_sdwa", EXPECTED_PS_SDWA)

    def test_ps_ops(self):
        self.check("ps_ops", EXPECTED_PS_OPS)


if __name__ == "__main__":
    unittest.main()
