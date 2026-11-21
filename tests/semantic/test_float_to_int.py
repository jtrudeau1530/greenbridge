"""VOP1 conversions with known binary32 inputs and integer output words."""
import struct
import unittest

import support


# Input bits, unsigned result, signed result (as an untyped 32-bit word).
CASES = (
    (0x00000000, 0, 0),                     # +0
    (0x80000000, 0, 0),                     # -0
    (0x00000001, 0, 0),                     # subnormal
    (0x3FC00000, 1, 1),                     # +1.5
    (0xBFC00000, 0, 0xFFFFFFFF),            # -1.5
    (0x4EFFFFFF, 0x7FFFFF80, 0x7FFFFF80),   # largest binary32 below +2^31
    (0x4F000000, 0x80000000, 0x7FFFFFFF),   # +2^31
    (0x4F7FFFFF, 0xFFFFFF00, 0x7FFFFFFF),   # largest binary32 below +2^32
    (0x4F800000, 0xFFFFFFFF, 0x7FFFFFFF),   # +2^32
    (0xCEFFFFFF, 0, 0x80000080),            # -(2^31 - 128)
    (0xCF000000, 0, 0x80000000),            # -2^31
    (0xCF000001, 0, 0x80000000),            # next binary32 below -2^31
    (0x7F800000, 0xFFFFFFFF, 0x7FFFFFFF),   # +infinity
    (0xFF800000, 0, 0x80000000),            # -infinity
    (0x7F800001, 0, 0),                     # first positive NaN encoding
    (0xFFC00000, 0, 0),                     # negative quiet NaN
)


class FloatToIntTest(unittest.TestCase):
    def test_saturating_conversions(self):
        # Five user SGPRs, with s4 carrying the input after the output V#.
        fssm = bytes(0x138) + struct.pack("<4I", 0x808A, 1, 1, 1)
        spv = support.translate_fixture("cs_cvt_f32_sat", fssm=fssm)
        for bits, unsigned, signed in CASES:
            with self.subTest(bits=f"{bits:08x}"):
                memory = {}
                support.run_fragment(
                    spv, inputs={"%wg_id": (0, 0, 0), "%local_id": (0, 0, 0)},
                    push_constants=[0, 0, 8, 0, bits] + [0] * 27, memory=memory)
                self.assertEqual(memory, {0: unsigned, 4: signed})


if __name__ == "__main__":
    unittest.main()
