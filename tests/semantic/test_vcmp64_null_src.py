"""RDNA2 null reads zero in either 64-bit integer VALU compare source."""
import itertools
import unittest

import support
import spirv_mini_interp as si

CONDITIONS = ("eq", "ne", "lt", "le", "gt", "ge")
VALUES = (0, 1, 1 << 32, 1 << 63, (1 << 64) - 1)


def reference(cond, ty, x, y):
    if ty == "i64":
        x, y = (v - (1 << 64) if v >> 63 else v for v in (x, y))
    return {"eq": x == y, "ne": x != y, "lt": x < y, "le": x <= y,
            "gt": x > y, "ge": x >= y}[cond]


def run_compare(cond, ty, form, cmpx, sources, value=0, inactive=False):
    op = f"v_cmp{'x' if cmpx else ''}_{cond}_{ty}_{form}"
    operands = ", ".join(sources)
    if not cmpx:
        operands = "vcc_lo, " + operands
    lines = [f"v_mov_b32 v2, {value & 0xFFFFFFFF:#x}", f"v_mov_b32 v3, {value >> 32:#x}",
             "s_mov_b32 vcc_lo, -1"]
    if inactive:
        lines.append("s_mov_b32 exec_lo, 0")
    lines += [f"{op} {operands}", "v_cndmask_b32 v0, 0, 1, vcc_lo",
              "exp mrt0 v0, v0, v0, v0 done vm", "s_endpgm"]
    return si.run_fragment(support.translate_lines("ps_vcmp64_null", lines)).get("%out_mrt0")


class CompareNullSources(unittest.TestCase):
    def check_sources(self, cmpx):
        for cond, ty, (form, slot), value in itertools.product(
                CONDITIONS, ("u64", "i64"), (("e32", 0), ("e64", 0), ("e64", 1)), VALUES):
            with self.subTest(cond=cond, ty=ty, form=form, slot=slot, value=hex(value)):
                sources = ["v[2:3]", "v[2:3]"]
                sources[slot] = "null"
                x, y = (0, value) if slot == 0 else (value, 0)
                expected = reference(cond, ty, x, y)
                result = run_compare(cond, ty, form, cmpx, sources, value)
                if cmpx:
                    self.assertEqual(result, (1, 1, 1, 1) if expected else None)
                else:
                    self.assertEqual(result, (int(expected),) * 4)

    def test_vcmp_sources(self):
        self.check_sources(False)

    def test_vcmpx_sources_and_vcc_preservation(self):
        self.check_sources(True)

    def test_both_null(self):
        for cond, ty, cmpx in itertools.product(CONDITIONS, ("u64", "i64"), (False, True)):
            with self.subTest(cond=cond, ty=ty, cmpx=cmpx):
                expected = reference(cond, ty, 0, 0)
                result = run_compare(cond, ty, "e64", cmpx, ["null", "null"])
                self.assertEqual(result, ((1,) * 4 if expected else None) if cmpx else (int(expected),) * 4)

    def test_inactive_lane_stays_inactive(self):
        for cond, ty, cmpx in itertools.product(CONDITIONS, ("u64", "i64"), (False, True)):
            with self.subTest(cond=cond, ty=ty, cmpx=cmpx):
                self.assertIsNone(run_compare(cond, ty, "e64", cmpx, ["null", "v[2:3]"], inactive=True))

    def test_non_pair_register_stays_unsupported(self):
        for cond, cmpx in itertools.product(CONDITIONS, (False, True)):
            with self.subTest(cond=cond, cmpx=cmpx):
                with self.assertRaises(support.rdna2spirv.Unsupported):
                    run_compare(cond, "u64", "e64", cmpx, ["null", "v2"])


if __name__ == "__main__":
    unittest.main()
