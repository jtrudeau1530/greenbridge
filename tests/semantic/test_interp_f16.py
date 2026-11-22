"""v_interp_p2_f16 with the VOP3 clamp bit: the f16 result is clamped to [0, 1].

Mesa ACO declares v_interp_p2_f16 dst(F16), which takes output modifiers, and llvm-mc encodes `clamp` on it (but no
omod). The translator models the p1ll/p2 pair as one read of the interpolated attribute, so the interpreter's input
value stands for that attribute here. Only the low half of each result is checked: the high half v_interp_p2_f16
leaves is not pinned by this test.
"""
import gc
import os
import tempfile
import unittest
import warnings

import support
from test_value_semantics import PS_FLAT_DXBC   # attr0.xy and attr1.x, f32

# attr0 = (1.5, -0.25): x read with clamp (1.0), y with clamp (0.0), x again without it (1.5), y without (-0.25).
SHADER = """v_interp_p1ll_f16 v2, v0, attr0.x
v_interp_p2_f16 v3, v1, attr0.x, v2 clamp
v_interp_p1ll_f16 v4, v0, attr0.y
v_interp_p2_f16 v5, v1, attr0.y, v4 clamp
v_interp_p1ll_f16 v6, v0, attr0.x
v_interp_p2_f16 v7, v1, attr0.x, v6
v_interp_p1ll_f16 v8, v0, attr0.y
v_interp_p2_f16 v9, v1, attr0.y, v8
exp mrt0 v3, v5, v7, v9 done vm
s_endpgm
"""
EXPECTED_LOW_HALVES = [0x3C00, 0x0000, 0x3E00, 0xB400]


def translate(asm, name="ps_interp_f16"):
    """SPIR-V assembly for pixel shader `asm`, translated with the attr0/attr1 input signature of ps_flat."""
    with tempfile.TemporaryDirectory() as tmp:
        os.mkdir(os.path.join(tmp, "disasm"))
        with open(os.path.join(tmp, "disasm", name + ".s"), "w", encoding="utf-8") as f:
            f.write(f"; {name}\n" + "".join("\t" + ln + "\n" for ln in asm.splitlines() if ln.strip()))
        with open(os.path.join(tmp, name + ".dxbc"), "wb") as f:
            f.write(PS_FLAT_DXBC)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ResourceWarning)
            result = support.rdna2spirv.translate(os.path.join(tmp, "disasm", name + ".s"))
            gc.collect()        # release the .dxbc handle before the directory is removed (Windows)
    return result


class InterpP2F16Clamp(unittest.TestCase):
    def test_clamp_applies_to_the_f16_result(self):
        words = support.mrt0_words(translate(SHADER), inputs={"%in_attr0": (1.5, -0.25, 0.0, 1.0),
                                                             "%in_attr1": (0.0,) * 4})
        self.assertEqual([f"{w & 0xFFFF:#06x}" for w in words], [f"{w:#06x}" for w in EXPECTED_LOW_HALVES])


if __name__ == "__main__":
    unittest.main()
