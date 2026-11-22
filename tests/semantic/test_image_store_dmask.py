"""image_store dmask: the enabled components come from consecutive VGPRs, in component order; the others are 0.

RDNA2 ISA, MIMG DMASK: for writes, which components are written with data from VGPRs; enabled components come
from consecutive VGPRs (DMASK = 1001: red in VGPRn, alpha in VGPRn+1). docs/design.md "Typed image stores":
"writes the dmask components (missing ones = 0)".

Checks the texel Gen.image_store hands OpImageWrite (all three views), as a list of VGPR numbers or 0.
"""
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS_DIR = os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(os.path.dirname(os.path.dirname(HERE)), "tools")
sys.dont_write_bytecode = True
sys.path.insert(0, TOOLS_DIR)

import rdna2spirv  # noqa: E402
from shader_ir import parse_line  # noqa: E402


def texels(line):
    ins = parse_line(line)
    g = rdna2spirv.Gen("cs")
    dim = next(m for m in ins["mods"] if m.startswith("dim:"))
    g.image_store(ins["args"], ins["mods"], dim)
    consts = {name: value for value, name in g.consts.items()}
    src = {}          # SSA id -> ("v", n) for VGPR loads, through bitcasts
    out = {}
    for line_ in g.body:
        w = line_.split()
        if len(w) < 5 or w[1] != "=":
            continue
        res, op, ty, args = w[0], w[2], w[3], w[4:]
        if op == "OpAccessChain" and args[0] == "%vgpr":
            src[res] = ("p", consts[args[1]])
        elif op == "OpLoad" and src.get(args[0], (None,))[0] == "p":
            src[res] = ("v", src[args[0]][1])
        elif op == "OpBitcast" and args[0] in src:
            src[res] = src[args[0]]
        elif op == "OpBitcast" and args[0] in consts:      # a missing component, seen through tof / as_int
            consts[res] = consts[args[0]]
        elif op == "OpCompositeConstruct" and ty in ("%v4float", "%v4uint_st", "%v4int_st"):
            out[ty] = [src[x][1] if x in src and src[x][0] == "v" else consts.get(x) for x in args]
    return out


class ImageStoreDmask(unittest.TestCase):
    def check(self, line, want):
        got = texels(line)
        self.assertEqual(len(got), 3)
        for ty, t in got.items():
            with self.subTest(view=ty):
                self.assertEqual(t, want)

    def test_full_mask(self):
        self.check("image_store v[4:7], v[0:1], s[8:15] dmask:0xf dim:SQ_RSRC_IMG_2D unorm", [4, 5, 6, 7])

    def test_low_mask(self):
        self.check("image_store v[4:5], v[0:1], s[8:15] dmask:0x3 dim:SQ_RSRC_IMG_2D unorm", [4, 5, 0, 0])

    def test_blue_only(self):
        self.check("image_store v4, v[0:1], s[8:15] dmask:0x4 dim:SQ_RSRC_IMG_2D unorm", [0, 0, 4, 0])

    def test_red_and_alpha(self):
        self.check("image_store v[4:5], v[0:1], s[8:15] dmask:0x9 dim:SQ_RSRC_IMG_2D unorm", [4, 0, 0, 5])


if __name__ == "__main__":
    unittest.main()
