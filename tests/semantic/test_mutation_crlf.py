"""Check all mutation anchors on full CRLF translator copies, independently of the child-run suite.

Each case reads repository source and changes only its own temporary file; it does not use the mutation
suite's shared private tools tree. The runner can execute both independent modules concurrently.
"""
import os
import tempfile
import unittest

from test_mutations import MUTATIONS, ROOT, TRANSLATOR, apply_mutation, phase_timer


class MutationCRLFTests(unittest.TestCase):
    @phase_timer("CRLF anchors")
    def test_anchors_match_crlf_translator_copies(self):
        # Regression: on a CRLF checkout of tools/ the four multiline anchors (v_bfe_i32, v_subrev_co_ci_u32,
        # v_cmpx_eq_u64, v_mul_legacy_f32) do not occur in the raw bytes and test_each_mutation_is_caught
        # stops before any mutation runs. apply_mutation must find every anchor in a CRLF copy, mutate it
        # once, keep the file on CRLF and leave it valid Python.
        with open(os.path.join(ROOT, "tools", TRANSLATOR), encoding="utf-8", newline="") as f:
            text = f.read().replace("\r\n", "\n")
        crlf_text = text.replace("\n", "\r\n")
        for label, fname, old, new, expect in MUTATIONS:
            if fname != TRANSLATOR:
                continue
            with self.subTest(mutation=label), tempfile.TemporaryDirectory() as tmp:
                path = os.path.join(tmp, TRANSLATOR)
                with open(path, "w", encoding="utf-8", newline="") as f:
                    f.write(crlf_text)
                apply_mutation(path, old, new)
                with open(path, encoding="utf-8", newline="") as f:
                    mutated = f.read()
                # the copy stays on CRLF: no bare LF survives, and the injected mutation text followed suit
                self.assertNotIn("\n", mutated.replace("\r\n", ""))
                self.assertIn(new.replace("\n", "\r\n"), mutated)
                self.assertNotIn(old.replace("\n", "\r\n"), mutated)
                compile(mutated, path, "exec")


if __name__ == "__main__":
    unittest.main()
