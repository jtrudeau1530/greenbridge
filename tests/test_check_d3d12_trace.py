"""How runtime/test/check_d3d12_trace.py runs its fixture, with synthetic stand-in programs: no Wine, GPU or game files."""
import importlib.util
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location("check_d3d12_trace", ROOT / "runtime" / "test" / "check_d3d12_trace.py")
check = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(check)

# A stand-in runner: `python -c RUNNER_CODE <fixture> [args]` prints a trace line for the fixture it was given.
ECHO_RUNNER = "import sys; print('[d3d12-trace] #1 ' + ' '.join(sys.argv[1:]), file=sys.stderr); print('other')"

# A stand-in for Wine that hangs: it starts a helper (as wineserver would), records the helper's pid, and waits.
HANGING_RUNNER = """
import subprocess, sys, time
helper = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
with open(sys.argv[1], "w") as f:
    f.write(str(helper.pid))
time.sleep(60)
"""


def gone(pid: int) -> bool:
    """The process has exited (a zombie waiting for its new parent counts)."""
    try:
        with open(f"/proc/{pid}/stat") as f:
            return f.read().rsplit(")", 1)[1].split()[0] == "Z"
    except FileNotFoundError:
        return True


class RunFixtureTest(unittest.TestCase):
    def setUp(self):
        self.saved = (list(check.RUNNER), check.TIMEOUT_S, check.RUNNER_TIMEOUT_S)

    def tearDown(self):
        check.RUNNER[:], check.TIMEOUT_S, check.RUNNER_TIMEOUT_S = self.saved

    def test_the_runner_is_put_in_front_and_trace_lines_are_kept(self):
        check.RUNNER[:] = [sys.executable, "-c", ECHO_RUNNER]
        self.assertEqual(check.run_fixture("fixture.exe", "--trace-d3d12"),
                         ["[d3d12-trace] #1 fixture.exe --trace-d3d12"])

    def test_a_missing_runner_is_a_one_line_failure(self):
        check.RUNNER[:] = [os.path.join(tempfile.gettempdir(), "no-such-wine", "wine")]
        with self.assertRaises(SystemExit) as caught:
            check.run_fixture("fixture.exe")
        self.assertRegex(str(caught.exception.code), r"^cannot start .*no-such-wine.*wine: ")

    def test_a_missing_fixture_is_a_one_line_failure(self):
        check.RUNNER[:] = []
        missing = os.path.join(tempfile.gettempdir(), "no-such-dir", "d3d12_gcl8_test.exe")
        with self.assertRaises(SystemExit) as caught:
            check.run_fixture(missing)
        self.assertTrue(str(caught.exception.code).startswith(f"cannot start {missing}: "))

    def test_a_nonzero_exit_keeps_the_output(self):
        check.RUNNER[:] = [sys.executable, "-c", "import sys; print('out'); print('err', file=sys.stderr); sys.exit(3)"]
        with self.assertRaises(SystemExit) as caught:
            check.run_fixture("fixture.exe")
        self.assertEqual(caught.exception.code, "fixture.exe exited with 3\nout\nerr\n")

    @unittest.skipUnless(os.name == "posix" and os.path.isdir("/proc"), "process groups and /proc")
    def test_a_timeout_stops_the_runner_and_what_it_started(self):
        with tempfile.TemporaryDirectory() as scratch:
            pid_file = os.path.join(scratch, "helper.pid")
            check.RUNNER[:] = [sys.executable, "-c", HANGING_RUNNER, pid_file]
            check.RUNNER_TIMEOUT_S = 2
            with self.assertRaises(SystemExit) as caught:
                check.run_fixture("fixture.exe")
            self.assertEqual(caught.exception.code, "fixture.exe did not finish within 2 s")
            with open(pid_file) as f:
                helper = int(f.read())
        deadline = time.monotonic() + 10
        while not gone(helper) and time.monotonic() < deadline:
            time.sleep(0.05)
        self.assertTrue(gone(helper), "the runner's helper outlived the timeout")


if __name__ == "__main__":
    unittest.main()
