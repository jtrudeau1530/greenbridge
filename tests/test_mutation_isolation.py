"""The shared private mutation fixture restores bytes and never reuses unexpected child state."""
import contextlib
import importlib.util
import io
from pathlib import Path
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("mutation_fixture", ROOT / "tests/semantic/test_mutations.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class MutationIsolationTests(unittest.TestCase):
    def test_exact_bytes_restored_after_success_and_failure(self):
        for fail in (False, True):
            with self.subTest(fail=fail), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "module.py"
                original = b"before\r\nanchor\r\nafter\r\n"
                path.write_bytes(original)
                try:
                    with m.restored_mutation(path, "anchor", "changed"):
                        self.assertEqual(path.read_bytes(), original.replace(b"anchor", b"changed"))
                        if fail:
                            raise RuntimeError("child failed")
                except RuntimeError as e:
                    self.assertEqual(str(e), "child failed")
                self.assertEqual(path.read_bytes(), original)

    def test_missing_anchor_restores_exact_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "module.py"
            original = b"unmatched\r\n"
            path.write_bytes(original)
            with self.assertRaisesRegex(AssertionError, "exactly once"):
                with m.restored_mutation(path, "absent", "changed"):
                    self.fail("a missing anchor must not run the child")
            self.assertEqual(path.read_bytes(), original)

    def exercise(self, dirty=False):
        seen = []
        with tempfile.TemporaryDirectory() as tmp:
            tools = Path(tmp) / "tools"
            tools.mkdir()
            original = b"anchor\r\n"
            (tools / "module.py").write_bytes(original)
            (tools / "untouched.py").write_bytes(b"unchanged\n")
            mutations = [("first", "module.py", "anchor", "one", "test_expected"),
                         ("second", "module.py", "anchor", "two", "test_expected")]

            def child(directory):
                seen.append((Path(directory) / "module.py").read_bytes())
                if len(seen) == 1:
                    if dirty:
                        (Path(directory) / "unexpected.py").write_text("child state")
                    # A survived first mutation fails its assertion, but must not poison or hide the second.
                    return 0, ""
                return 1, "FAIL: test_expected\n"

            with mock.patch.object(m, "ROOT", tmp), mock.patch.object(m, "MUTATIONS", mutations), \
                    mock.patch.object(m, "run_semantics", side_effect=child), \
                    mock.patch.object(m.shutil, "copytree", wraps=m.shutil.copytree) as copy, \
                    contextlib.redirect_stdout(io.StringIO()):
                result = unittest.TestResult()
                unittest.TestSuite([m.MutationTests("test_each_mutation_is_caught")]).run(result)
                self.assertEqual(copy.call_count, 1)
            self.assertEqual((tools / "module.py").read_bytes(), original)
        return seen, result

    def test_failed_mutation_restores_source_and_continues(self):
        seen, result = self.exercise()
        self.assertEqual(seen, [b"one\r\n", b"two\r\n"])
        self.assertEqual(len(result.failures), 1)
        self.assertEqual(result.errors, [])
        self.assertIn("mutation not detected: first", result.failures[0][1])

    def test_unexpected_child_state_prevents_next_mutation(self):
        seen, result = self.exercise(dirty=True)
        self.assertEqual(seen, [b"one\r\n"])
        self.assertFalse(result.wasSuccessful())
        self.assertEqual(result.errors, [])
        self.assertTrue(all("do not reuse contaminated state" in text for _, text in result.failures))


if __name__ == "__main__":
    unittest.main()
