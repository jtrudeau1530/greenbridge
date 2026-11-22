"""bench_wave64.py: a configuration that fails after its first process is a failure, not a configuration that ran.
The builds and GPU runs are replaced by stubs, so this needs no GPU, LLVM or Vulkan SDK."""
import contextlib
import io
import os
import sys
import unittest
from unittest.mock import patch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import bench_wave64  # noqa: E402


def bench(fail):
    """Run main() with 2 processes per configuration; `fail` maps a configuration name to the process that fails."""
    calls = {}

    def run(tools, spv, groups, reps, driver, emulate):
        name = next(n for n, d, e in bench_wave64.CONFIGS if d == driver and e == emulate)
        calls[name] = calls.get(name, 0) + 1
        if fail.get(name) == calls[name]:
            return None, "exit 1: device lost", None
        return (1.0, 0.9), "same", "stub GPU"

    out = io.StringIO()
    with patch.object(bench_wave64, "build", lambda work, name, emulate, llvm_mc: f"{name}-{emulate}.spv"), \
            patch.object(bench_wave64, "segments", lambda spv: None), \
            patch.object(bench_wave64, "run", run), \
            patch.object(sys, "argv", ["bench_wave64.py", "--procs", "2"]), \
            contextlib.redirect_stdout(out):
        code = bench_wave64.main()
    return code, out.getvalue()


class PartialFailure(unittest.TestCase):
    def test_every_process_ran(self):
        code, out = bench({})
        self.assertEqual(code, 0, out)
        self.assertIn("emulated output equals the reference", out)

    def test_emulated_config_fails_on_its_second_process(self):
        code, out = bench({"NVIDIA --wave64-emulate": 2})
        self.assertEqual(code, 1, out)
        self.assertIn("NVIDIA --wave64-emulate: failed or unsupported", out)
        self.assertIn("incomplete comparison: no complete result from NVIDIA --wave64-emulate", out)
        self.assertNotIn("emulated output equals the reference", out)

    def test_reference_fails_on_its_second_process(self):
        code, out = bench({"AMD native": 2})
        self.assertEqual(code, 1, out)
        self.assertIn("no reference output from AMD native", out)


class ArgumentValidation(unittest.TestCase):
    def test_positive_only(self):
        for option in ("--groups", "--reps", "--procs"):
            for value in ("0", "-1"):
                with self.subTest(option=option, value=value), \
                        patch.object(sys, "argv", ["bench_wave64.py", option, value]), \
                        patch.object(bench_wave64, "build") as build:
                    with self.assertRaises(SystemExit) as e:
                        bench_wave64.main()
                    self.assertEqual(e.exception.code, 2)
                    build.assert_not_called()


if __name__ == "__main__":
    unittest.main()
