"""A bitcast back to a value's original type is the original id, not a second OpBitcast.

Gen.tof/Gen.tou remember each bitcast's source (Gen.bitcast_from), so tou(tof(u)) is u and tof(tou(f)) is f. That is
always valid SSA: wherever the bitcast's result is usable, its source is too, because the bitcast's own definition
uses it. A flush of a register read as float is the common case: flush_f32 reads the bits of the float it was given.
Synthetic only.

    python -I -m unittest discover -s tests/semantic -p test_bitcast_round_trip.py -v
"""
import os
import re
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import support  # noqa: E402
from support import rdna2spirv  # noqa: E402


def bitcast_sources(spvasm):
    """{result: (type, source)} of every OpBitcast in the module."""
    return {m.group(1): (m.group(2), m.group(3))
            for m in re.finditer(r"^\s*(%\S+) = OpBitcast %(\S+) (%\S+)$", spvasm, re.M)}


class BitcastRoundTripTest(unittest.TestCase):
    def test_helpers_return_the_source(self):
        g = rdna2spirv.Gen.__new__(rdna2spirv.Gen)
        g.body, g.n, g.bitcast_from, g.emulate = [], 0, {}, False
        f = g.tof("%t100")
        self.assertEqual(g.tou(f), "%t100")
        u = g.tou("%t200")
        self.assertEqual(g.tof(u), "%t200")
        self.assertEqual(len(g.body), 2)             # one bitcast each way, no reverse bitcast emitted

    def test_a_translated_module_has_no_bitcast_of_a_bitcast_back(self):
        lines = ["v_mov_b32 v1, 0x00000005", "v_mov_b32 v2, 0x3f000000",
                 "v_mad_f32 v3, v1, v2, v1", "v_mul_legacy_f32 v4, v1, v2",
                 "exp mrt0 v3, v4, v4, v4 done vm", "s_endpgm"]
        spv = support.translate_lines("ps_bitcast_round_trip", lines)
        casts = bitcast_sources(spv)
        back = [r for r, (ty, src) in casts.items() if src in casts and casts[src][0] != ty]
        self.assertEqual(back, [])
        support.mrt0_words(spv)                      # still runs


if __name__ == "__main__":
    unittest.main()
