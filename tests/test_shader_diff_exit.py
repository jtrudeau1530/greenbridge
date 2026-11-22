"""shader_diff.py exit codes: a harness error is 2, not a mismatch (no GPU, no game data)."""
import os
import sys
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import shader_diff as sd  # noqa: E402


class HarnessErrorExit(unittest.TestCase):
    def test_harness_error_alone_exits_2(self):
        def boom(*_a):
            raise RuntimeError("broken harness")
        with mock.patch.object(sd, "run_one", boom), mock.patch("sys.stdout"):
            self.assertEqual(sd.main(["--programs", "2"]), 2)

    def test_mismatch_still_exits_1(self):
        with mock.patch.object(sd, "run_one", lambda *_a: ("mismatch", "lane word 0")), mock.patch("sys.stdout"):
            self.assertEqual(sd.main(["--programs", "1"]), 1)


if __name__ == "__main__":
    unittest.main()
