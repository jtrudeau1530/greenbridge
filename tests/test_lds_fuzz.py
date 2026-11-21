"""tools/lds_fuzz.py without a GPU: the reference model's instruction semantics and the generator's guarantees
(every access in bounds, no two lanes racing on the same bytes)."""
from pathlib import Path
import struct
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import lds_fuzz as f


class Swizzle(unittest.TestCase):
    # Offsets as llvm-mc 23 (-mcpu=gfx1013) encodes the swizzle() macros of ds_swizzle_b32.
    def test_quad_perm(self):
        offset = 0x801B                                  # swizzle(QUAD_PERM,3,2,1,0)
        self.assertEqual([f.swizzle_src(lane, offset) for lane in range(8)], [3, 2, 1, 0, 7, 6, 5, 4])

    def test_bitmask_or(self):
        self.assertEqual(f.swizzle_src(3, 0x020F), 19)  # swizzle(BITMASK_PERM,"1pppp"): bit 4 set
        self.assertEqual(f.swizzle_src(19, 0x000F), 3)  # swizzle(BITMASK_PERM,"0pppp"): bit 4 cleared

    def test_reverse_and_broadcast(self):
        self.assertEqual([f.swizzle_src(lane, 0x1C1F) for lane in range(8)], [7, 6, 5, 4, 3, 2, 1, 0])  # REVERSE,8
        self.assertEqual({f.swizzle_src(lane, 0x0078) for lane in range(8, 16)}, {11})               # BROADCAST,8,3

    def test_bitmask_stays_in_its_32_lane_group(self):
        self.assertEqual(f.swizzle_src(32 + 3, 0x1C1F), 32 + (3 ^ 7))     # REVERSE,8 in the upper group
        self.assertEqual(f.swizzle_src(32 + 3, 0x7C1F), 32 + (3 ^ 31))    # REVERSE,32: still lanes 32-63


class Atomics(unittest.TestCase):
    def test_signed_and_unsigned_min_max(self):
        neg = 0xFFFFFFFF
        self.assertEqual(f.atomic("min_i32", 1, neg), neg)
        self.assertEqual(f.atomic("min_u32", 1, neg), 1)
        self.assertEqual(f.atomic("max_i32", 1, neg), 1)
        self.assertEqual(f.atomic("max_u32", 1, neg), neg)

    def test_wrapping_add_sub(self):
        self.assertEqual(f.atomic("add", 0xFFFFFFFF, 2), 1)
        self.assertEqual(f.atomic("sub", 1, 2), 0xFFFFFFFF)


class Generator(unittest.TestCase):
    def programs(self):
        return f.programs(seed=11, count=60, waves=(64, 32))

    def test_deterministic(self):
        a, b = f.Program(3, 5, 64), f.Program(3, 5, 64)
        self.assertEqual(a.asm(), b.asm())
        self.assertEqual(a.model(), b.model())

    def test_accesses_in_bounds_and_race_free(self):
        for p in self.programs():
            for op in p.ops:
                if "addr" not in op:
                    continue
                stride, base, off = op["addr"]
                if op["kind"] in ("read2", "write2"):
                    unit = 256 if op["st64"] else 4
                    spans = [[(t * stride + base + o * unit, 4) for o in (op["o0"], op["o1"])]
                             for t in range(f.THREADS)]
                else:
                    size = {"read": lambda: f.READS[op["name"]][0], "write": lambda: f.WRITES[op["name"]][0]}.get(
                        op["kind"], lambda: 4)()
                    spans = [[(t * stride + base + off, size)] for t in range(f.THREADS)]
                for ss in spans:
                    for a, n in ss:
                        self.assertTrue(0 <= a and a + n <= f.LDS_BYTES, (p.name, op))
                if op["kind"] in ("write", "write2", "atomic_rtn", "wrxchg", "cmpst"):
                    self.assertTrue(p.disjoint(spans), (p.name, op))

    def test_only_order_independent_atomics_share_addresses(self):
        for p in self.programs():
            for op in p.ops:
                if op["kind"] == "atomic":
                    self.assertIn(op["op"], f.SHARED_ATOMICS)

    def test_no_exec_keeps_the_rest_of_the_program(self):
        a, b = f.Program(1, 32, 32), f.Program(1, 32, 32, no_exec=("bpermute",))
        self.assertEqual(len(a.ops), len(b.ops))
        for x, y in zip(a.ops, b.ops):
            if x["kind"] == "bpermute":
                self.assertIsNone(y["exec"])
            else:
                self.assertEqual(x, y)


class Model(unittest.TestCase):
    def test_initial_lds_pattern(self):
        p = f.Program(1, 0, 64)
        p.ops = []
        _, lds = p.model()
        for i in (0, 1, 1023):
            self.assertEqual(struct.unpack_from("<I", lds, 4 * i)[0], ((i * p.lds_mul) & f.M32) ^ p.lds_xor)

    def test_bpermute_inactive_source_reads_zero(self):
        p = f.Program(1, 0, 32)
        p.ops = [{"kind": "bpermute", "mul": 0, "add": 8, "offset": 0, "src": 20, "dst": 21,
                  "exec": 0xFFFF00FF}]
        regs, _ = p.model()
        self.assertEqual(regs[4][21], 0)                       # source lane 8 is off
        keep = f.Program(1, 0, 32)
        keep.ops = []
        self.assertEqual(regs[9][21], keep.model()[0][9][21])  # lane 9 is off itself: keeps its value

    def test_d16_hi_read_keeps_the_low_half(self):
        p = f.Program(1, 0, 64)
        p.ops = [{"kind": "read", "name": "ds_read_u16_d16_hi", "addr": (0, 0, 0), "dst": 20, "exec": None}]
        before = f.Program(1, 0, 64)
        before.ops = []
        old = before.model()[0][0][20]
        regs, lds = p.model()
        self.assertEqual(regs[0][20], (old & 0xFFFF) | (struct.unpack_from("<H", lds, 0)[0] << 16))

    def test_cmpst_stores_data2_when_data_matches(self):
        p = f.Program(1, 0, 64)
        p.ops = [{"kind": "cmpst", "addr": (4, 0, 0), "src": 20, "src2": 21, "dst": 22, "exec": None,
                  "cmp_hit": True}]
        regs, lds = p.model()
        for t in (0, 63):
            self.assertEqual(struct.unpack_from("<I", lds, 4 * t)[0], regs[t][21])
            self.assertEqual(regs[t][22], regs[t][20])            # returns the old word, which was the compare value


if __name__ == "__main__":
    unittest.main()
