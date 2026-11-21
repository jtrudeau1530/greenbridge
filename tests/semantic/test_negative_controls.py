"""Negative controls: textually sabotage the translator's *output* and show the oracle notices.
This proves the value checks are sensitive without touching the translator. (Mutations of the translator's own
source are in test_mutations.py.)"""
import unittest

import support
import test_ps_semantics as sem

CONTROLS = [
    # (shader, expected words, text to replace, replacement, what the sabotage models)
    ("ps_sdwa", sem.EXPECTED_PS_SDWA, "OpBitFieldSExtract", "OpBitFieldUExtract", "sext() ignored in SDWA"),
    ("ps_ops", sem.EXPECTED_PS_OPS, "OpBitFieldSExtract", "OpBitFieldUExtract", "v_bfe_i32 treated as unsigned"),
    ("ps_ops", sem.EXPECTED_PS_OPS, "glsl UMin", "glsl SMax", "v_min3_u32 computing a (signed) maximum"),
    ("ps_ops", sem.EXPECTED_PS_OPS, "glsl SMax", "glsl UMin", "v_max3_i32 treated as unsigned"),
]


class NegativeControls(unittest.TestCase):
    def test_sabotaged_output_is_detected(self):
        for shader, expected, old, new, what in CONTROLS:
            with self.subTest(what=what):
                good = support.translate_fixture(shader)
                self.assertEqual(support.mrt0_words(good), expected, "control needs a passing baseline")
                self.assertIn(old, good, "sabotage target must exist in the translated shader")
                bad = good.replace(old, new)
                self.assertNotEqual(support.mrt0_words(bad), expected, f"undetected: {what}")


if __name__ == "__main__":
    unittest.main()
