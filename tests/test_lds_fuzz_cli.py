import contextlib
import io
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import lds_fuzz as f


class CLIValidation(unittest.TestCase):
    def test_zero_programs_must_fail(self):
        args = ["lds_fuzz.py", "--programs", "0", "--model-only"]
        with patch.object(sys, "argv", args):
            with self.assertRaises(SystemExit) as result:
                f.main()
        self.assertEqual(result.exception.code, 2)

    def test_malformed_show_must_fail(self):
        args = ["lds_fuzz.py", "--show", "invalido"]
        with patch.object(sys, "argv", args):
            with self.assertRaises(SystemExit) as result:
                f.main()
        self.assertEqual(result.exception.code, 2)

    def test_invalid_counts(self):
        for option, value in [
            ("--programs", "0"),
            ("--programs", "-1"),
            ("--max-report", "0"),
            ("--max-report", "-2"),
        ]:
            with self.subTest(option=option, value=value):
                args = ["lds_fuzz.py", option, value, "--model-only"]
                with patch.object(sys, "argv", args):
                    with contextlib.redirect_stderr(io.StringIO()):
                        with self.assertRaises(SystemExit) as result:
                            f.main()
                self.assertEqual(result.exception.code, 2)

    def test_invalid_show_formats(self):
        for value in [
            "invalido", "1", "1:2:3:4",
            "1:-1", "1:0:16", "a:0",
            "1:b", "1:0:invalido", "1:0:",
        ]:
            with self.subTest(value=value):
                with patch.object(sys, "argv", ["lds_fuzz.py", "--show", value]):
                    with contextlib.redirect_stderr(io.StringIO()):
                        with self.assertRaises(SystemExit) as result:
                            f.main()
                self.assertEqual(result.exception.code, 2)

    def test_valid_show_formats(self):
        for value, seed, index, wave in [
            ("7:1", 7, 1, 64),
            ("7:1:32", 7, 1, 32),
            ("7:1:64", 7, 1, 64),
        ]:
            with self.subTest(value=value):
                with patch.object(sys, "argv", ["lds_fuzz.py", "--show", value]):
                    with patch.object(f, "show") as show_mock:
                        self.assertEqual(f.main(), 0)
                show_mock.assert_called_once_with(seed, index, wave, ())

    def test_positive_count_boundary(self):
        args = [
            "lds_fuzz.py", "--programs", "1",
            "--max-report", "1", "--wave", "32", "--model-only",
        ]
        output = io.StringIO()
        with patch.object(sys, "argv", args):
            with contextlib.redirect_stdout(output):
                self.assertEqual(f.main(), 0)
        self.assertIn("1 programs generated and modelled", output.getvalue())


if __name__ == "__main__":
    unittest.main()
