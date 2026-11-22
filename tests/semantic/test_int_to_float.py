from __future__ import annotations
import unittest
import struct
import math

class IntToFloatOracle:
    @staticmethod
    def i32_to_f32_bits(val: int) -> int:
        f = float(val)
        return struct.unpack('<I', struct.pack('<f', f))[0]

    @staticmethod
    def u32_to_f32_bits(val: int) -> int:
        # Unsigned 32-bit conversion via ctypes or direct float cast if Python handles it correctly
        # In Python, int is arbitrary precision. For unsigned 32-bit:
        if val < 0:
            val += 0x100000000
        f = float(val)
        return struct.unpack('<I', struct.pack('<f', f))[0]

    @staticmethod
    def i32_to_f64_bits(val: int) -> int:
        f = float(val)
        return struct.unpack('<Q', struct.pack('<d', f))[0]

class TestIntToFloatReferenceOracle(unittest.TestCase):
    def test_oracle_basic(self):
        self.assertEqual(IntToFloatOracle.i32_to_f32_bits(0), 0x00000000)
        self.assertEqual(IntToFloatOracle.u32_to_f32_bits(0), 0x00000000)
        self.assertEqual(IntToFloatOracle.i32_to_f32_bits(1), 0x3f800000)
        self.assertEqual(IntToFloatOracle.u32_to_f32_bits(1), 0x3f800000)
        self.assertEqual(IntToFloatOracle.i32_to_f32_bits(-1), 0xbf800000)

class TestSignedVsUnsigned32Bit(unittest.TestCase):
    def test_signed_vs_unsigned_boundaries(self):
        # 0x7fffffff (max positive signed 32-bit int)
        # 0x80000000 (min negative signed 32-bit int vs unsigned 2^31)
        max_i32 = 0x7fffffff
        min_i32 = -0x80000000
        u32_2_31 = 0x80000000
        max_u32 = 0xffffffff

        # Signed max_i32 as float32
        f_max_i32 = IntToFloatOracle.i32_to_f32_bits(max_i32)
        # Unsigned 2^31 as float32
        f_u32_2_31 = IntToFloatOracle.u32_to_f32_bits(u32_2_31)
        # Signed min_i32 as float32
        f_min_i32 = IntToFloatOracle.i32_to_f32_bits(min_i32)
        # Unsigned max_u32 as float32
        f_max_u32 = IntToFloatOracle.u32_to_f32_bits(max_u32)

        # Verify that min_i32 (-2^31) and u32_2_31 (2^31) convert correctly
        self.assertNotEqual(f_min_i32, f_u32_2_31)
        # -2^31 is negative, 2^31 is positive
        self.assertTrue((f_min_i32 & 0x80000000) != 0)
        self.assertTrue((f_u32_2_31 & 0x80000000) == 0)

if __name__ == '__main__':
    unittest.main()
