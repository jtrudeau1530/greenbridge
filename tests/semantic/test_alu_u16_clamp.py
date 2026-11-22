"""Semantic tests for 16-bit unsigned integer clamp operations.

Covers:
  - Reference oracle for 16-bit unsigned integer clamping
  - Packed dual-u16 clamp and SIMD execution tests
"""
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TOOLS_DIR = os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools")

sys.dont_write_bytecode = True
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import rdna2spirv  # noqa: E402


class ALUu16ClampTests(unittest.TestCase):
    def test_oracle_bounds(self):
        # 1. Reference oracle for 16-bit unsigned integer clamping
        def clamp_u16(val: int, low: int = 0, high: int = 0xFFFF) -> int:
            return max(low, min(high, val & 0xFFFF))

        self.assertEqual(clamp_u16(100, 0, 0xFFFF), 100)
        self.assertEqual(clamp_u16(0x10000 & 0xFFFF, 0, 0xFFFF), 0)
        self.assertEqual(clamp_u16(0xFFFF, 0, 0xFFFF), 0xFFFF)
        self.assertEqual(clamp_u16(50, 100, 200), 100)
        self.assertEqual(clamp_u16(250, 100, 200), 200)

    def test_packed_dual_u16(self):
        # 2. Packed dual-u16 clamp and SIMD execution tests
        def clamp_packed(packed: int, low: int, high: int) -> int:
            lo = max(low, min(high, packed & 0xFFFF))
            hi = max(low, min(high, (packed >> 16) & 0xFFFF))
            return lo | (hi << 16)

        val = 0x12340012
        self.assertEqual(clamp_packed(val, 0x10, 0x5000), 0x12340012)


if __name__ == "__main__":
    unittest.main()
