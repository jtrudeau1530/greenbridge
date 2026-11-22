"""v_dot4_i32_i8 and v_dot4c_i32_i8: the dot product of the signed bytes of two dwords, plus an addend.

ISA rule (RDNA2): D = S0.b0*S1.b0 + S0.b1*S1.b1 + S0.b2*S1.b2 + S0.b3*S1.b3 + addend, every byte sign-extended, the
sum kept modulo 2^32. The addend is the third source for v_dot4_i32_i8 (VOP3P) and the old value of the destination
for v_dot4c_i32_i8 (VOP2). The 29 FSPX programs of the package's upscaler DLL (docs/fspx.md) use the pair 27,271
times.

The expected words below were measured on RDNA2 hardware (gfx1036, the integrated GPU of a Ryzen 9 7950X3D, through
tools/hw_oracle): each case ran `v_dot4_i32_i8 v8, v1, v2, v3`, `v_dot4c_i32_i8 v10, v1, v2` with v10 holding the
addend, and the same with the two sources swapped. All three gave the word in the table, which the plain-Python
reference `ref` reproduces. With `clamp` the hardware saturates v_dot4_i32_i8 to a signed 32-bit value
(0x7fffffff and 0x80000000 in the two overflowing cases); the translator refuses that form.

op_sel. The plain instruction has op_sel 0 and op_sel_hi 7. 89 words in 8 of those programs are v_dot4_i32_i8 with
op_sel 2 and op_sel_hi 5, which llvm-mc does not decode (docs/fspx.md). Run on the same hardware as two raw words
(0xCC165008 0x0C0E0501: v8 = v1 . v2 + v3 with those fields), that form gave the words of MEASURED_OP_SEL: the dot
product with the two 16-bit halves of the second source exchanged, as op_sel selects halves for the packed 16-bit
ops. `ref_op_sel` reproduces them; the translator refuses any op_sel on these two instructions until a listing can
carry it.

    python -m unittest discover -s tests/semantic -p test_alu_dot4_i8.py
"""
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import support  # noqa: E402

rdna2spirv = support.rdna2spirv

# (S0, S1, addend, result measured on gfx1036)
MEASURED = [
    (0x01020304, 0x05060708, 0x00000000, 0x00000046),   # 4*8 + 3*7 + 2*6 + 1*5 = 70
    (0x01020304, 0x05060708, 0x000003E8, 0x0000042E),   # 70 + 1000
    (0xFF80017F, 0x7F7F80FF, 0x00000000, 0xFFFFBF02),   # 127*-1 + 1*-128 + -128*127 + -1*127 = -16638
    (0xFF80017F, 0x7F7F80FF, 0x00010000, 0x0000BF02),
    (0x80808080, 0x80808080, 0x00000000, 0x00010000),   # 4 * (-128 * -128)
    (0x80808080, 0x7F7F7F7F, 0x00000005, 0xFFFF0205),   # 4 * (-128 * 127) + 5
    (0x7F7F7F7F, 0x7F7F7F7F, 0x7FFFFFFF, 0x8000FC03),   # wraps upward: no saturation without clamp
    (0x80808080, 0x7F7F7F7F, 0x80000000, 0x7FFF0200),   # wraps downward
    (0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0x00000003),   # 4 * (-1 * -1) - 1
    (0x00000000, 0x12345678, 0xDEADBEEF, 0xDEADBEEF),
]


# (S0, S1, addend, result measured on gfx1036 with op_sel 2 and op_sel_hi 5)
MEASURED_OP_SEL = [
    (0x04030201, 0x08070605, 0x00000000, 62),           # the plain form gives 70
    (0x04030201, 0x08070605, 0x000003E8, 1062),
    (0x7F01FE80, 0x02FD7F81, 0x00000000, 16382),        # the plain form gives 16253
    (0x11223344, 0x55667788, 0x00010000, 74750),        # the plain form gives 68358
    (0xFFFFFFFF, 0x01020304, 0x00000007, 0xFFFFFFFD),   # every byte of S0 the same: the order cannot show
]


def ref(a, b, addend):
    def s8(x):
        return x - 256 if x & 0x80 else x
    total = sum(s8((a >> (8 * i)) & 0xFF) * s8((b >> (8 * i)) & 0xFF) for i in range(4))
    return (total + addend) & 0xFFFFFFFF


def ref_op_sel(a, b, addend):
    """op_sel 2, op_sel_hi 5: the low half of the product reads the high half of S1 and the high half its low half."""
    return ref(a, (b & 0xFFFF) << 16 | b >> 16, addend)


def program(a, b, addend):
    """mrt0 = (v_dot4_i32_i8, v_dot4c_i32_i8, v_dot4c_i32_i8 with the sources swapped, v_dot4_i32_i8 with the
    sources and the addend taken from SGPRs and a literal, as the shaders do)."""
    return [
        f"v_mov_b32_e32 v1, 0x{a:08x}",
        f"v_mov_b32_e32 v2, 0x{b:08x}",
        f"v_mov_b32_e32 v3, 0x{addend:08x}",
        f"s_mov_b32 s4, 0x{addend:08x}",
        "v_dot4_i32_i8 v8, v1, v2, v3",
        f"v_mov_b32_e32 v9, 0x{addend:08x}",
        "v_dot4c_i32_i8 v9, v1, v2",
        f"v_mov_b32_e32 v10, 0x{addend:08x}",
        "v_dot4c_i32_i8 v10, v2, v1",
        f"v_dot4_i32_i8 v11, v1, 0x{b:08x}, s4",
        "exp mrt0 v8, v9, v10, v11 done vm",
        "s_endpgm",
    ]


class Dot4I8(unittest.TestCase):
    def test_the_reference_reproduces_the_hardware_words(self):
        for a, b, addend, measured in MEASURED:
            self.assertEqual(ref(a, b, addend), measured, f"{a:#x} . {b:#x} + {addend:#x}")

    def test_both_forms_give_the_hardware_words(self):
        for a, b, addend, measured in MEASURED:
            with self.subTest(a=f"{a:#010x}", b=f"{b:#010x}", addend=f"{addend:#010x}"):
                spv = support.translate_lines("ps_dot4_i8", program(a, b, addend))
                self.assertEqual(support.mrt0_words(spv), [measured] * 4)

    def test_the_accumulating_form_reads_its_destination(self):
        # v9 is never written before v_dot4c reads it: the old value is then the register's initial 0, not garbage
        # from an expression that dropped the addend. 70 is the dot product alone.
        lines = ["v_mov_b32_e32 v1, 0x01020304", "v_mov_b32_e32 v2, 0x05060708", "v_dot4c_i32_i8 v9, v1, v2",
                 "v_dot4c_i32_i8 v9, v1, v2", "exp mrt0 v9, v9, v9, v9 done vm", "s_endpgm"]
        self.assertEqual(support.mrt0_words(support.translate_lines("ps_dot4c_twice", lines)), [140] * 4)

    def test_the_bytes_are_signed(self):
        # 0xff * 0x01: -1 as a signed byte, 255 as an unsigned one.
        lines = ["v_mov_b32_e32 v1, 0x000000ff", "v_mov_b32_e32 v2, 0x00000001", "v_mov_b32_e32 v3, 0",
                 "v_dot4_i32_i8 v8, v1, v2, v3", "exp mrt0 v8, v8, v8, v8 done vm", "s_endpgm"]
        self.assertEqual(support.mrt0_words(support.translate_lines("ps_dot4_signed", lines)), [0xFFFFFFFF] * 4)

    def test_the_op_sel_reference_reproduces_the_hardware_words(self):
        for a, b, addend, measured in MEASURED_OP_SEL:
            self.assertEqual(ref_op_sel(a, b, addend), measured, f"{a:#x} . {b:#x} + {addend:#x}")
        # The table can tell the two readings apart: the plain rule gives another word in four of the five cases.
        self.assertEqual(sum(ref(a, b, c) != m for a, b, c, m in MEASURED_OP_SEL), 4)

    def test_op_sel_is_refused(self):
        lines = ["v_mov_b32_e32 v1, 1", "v_mov_b32_e32 v2, 1", "v_mov_b32_e32 v3, 0",
                 "v_dot4_i32_i8 v8, v1, v2, v3 op_sel:[0,1,0] op_sel_hi:[1,0,1]",
                 "exp mrt0 v8, v8, v8, v8 done vm", "s_endpgm"]
        with self.assertRaises(rdna2spirv.Unsupported) as refused:
            support.translate_lines("ps_dot4_op_sel", lines)
        self.assertEqual(str(refused.exception), "v_dot4_i32_i8 op_sel")

    def test_clamp_is_refused(self):
        lines = ["v_mov_b32_e32 v1, 1", "v_mov_b32_e32 v2, 1", "v_mov_b32_e32 v3, 0",
                 "v_dot4_i32_i8 v8, v1, v2, v3 clamp", "exp mrt0 v8, v8, v8, v8 done vm", "s_endpgm"]
        with self.assertRaises(rdna2spirv.Unsupported) as refused:
            support.translate_lines("ps_dot4_clamp", lines)
        self.assertEqual(str(refused.exception), "v_dot4_i32_i8 clamp")


if __name__ == "__main__":
    unittest.main()
