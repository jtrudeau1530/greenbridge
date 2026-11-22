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

if __name__ == '__main__':
    unittest.main()
