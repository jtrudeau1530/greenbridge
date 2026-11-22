"""A GPU test program that hangs fails its own test instead of stalling the suite: run_gpu_tests.run_cmd and
check.run give each program LEONIDA_GPU_TEST_TIMEOUT seconds (default 120, 0 = no limit), then kill it. The hang
is a sleeping Python child, so no GPU is needed."""
import os
from pathlib import Path
import sys
import time
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "runtime" / "test" / "shaders"))
import check
import run_gpu_tests

HANG = [sys.executable, "-c", "import time; time.sleep(60)"]
QUICK = [sys.executable, "-c", "print('ok')"]


class Limit(unittest.TestCase):
    def limit(self, value):
        env = {k: v for k, v in os.environ.items() if k != "LEONIDA_GPU_TEST_TIMEOUT"}
        if value is not None:
            env["LEONIDA_GPU_TEST_TIMEOUT"] = value
        with patch.dict(os.environ, env, clear=True):
            return check.gpu_test_timeout()

    def test_default_and_override(self):
        self.assertEqual(self.limit(None), check.DEFAULT_GPU_TEST_TIMEOUT)
        self.assertEqual(self.limit(""), check.DEFAULT_GPU_TEST_TIMEOUT)
        self.assertEqual(self.limit(" 2.5 "), 2.5)

    def test_zero_or_negative_means_no_limit(self):
        self.assertIsNone(self.limit("0"))
        self.assertIsNone(self.limit("-1"))

    def test_not_a_number_is_an_error(self):
        for bad in ("soon", "nan"):
            with self.subTest(bad=bad), self.assertRaises(SystemExit) as ctx:
                self.limit(bad)
            self.assertIn("LEONIDA_GPU_TEST_TIMEOUT", str(ctx.exception.code))


class HungProgram(unittest.TestCase):
    def setUp(self):
        patcher = patch.dict(os.environ, {"LEONIDA_GPU_TEST_TIMEOUT": "0.5"})
        patcher.start()
        self.addCleanup(patcher.stop)

    def assert_killed(self, call):
        start = time.monotonic()
        rc, msg = call()
        self.assertLess(time.monotonic() - start, 30)
        self.assertNotEqual(rc, 0)
        self.assertIn("timed out after 0.5 s", msg)
        self.assertIn(os.path.basename(sys.executable), msg)

    def test_run_gpu_tests_run_cmd(self):
        self.assert_killed(lambda: run_gpu_tests.run_cmd(HANG))

    def test_check_run(self):
        self.assert_killed(lambda: check.run(*HANG))

    def test_quick_program_is_unaffected(self):
        self.assertEqual(run_gpu_tests.run_cmd(QUICK), (0, "ok"))
        rc, out = check.run(*QUICK)
        self.assertEqual((rc, out.strip()), (0, "ok"))

    def test_translator_step_has_no_limit(self):
        seen = {}

        def fake_run(cmd, **kw):
            seen.update(kw)
            raise OSError("stop here")

        with patch.object(run_gpu_tests.subprocess, "run", side_effect=fake_run):
            run_gpu_tests.run_cmd(QUICK, gpu=False)
        self.assertIsNone(seen["timeout"])
        with patch.object(run_gpu_tests.subprocess, "run", side_effect=fake_run):
            run_gpu_tests.run_cmd(QUICK)
        self.assertEqual(seen["timeout"], 0.5)


if __name__ == "__main__":
    unittest.main()
