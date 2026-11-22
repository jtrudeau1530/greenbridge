"""scripts/build_runtime.ps1's cl.exe fallback builds every GPU test harness the Python tools need (no build run).

The fallback is what runs when Visual Studio has no "C++ CMake tools" component. Without these programs,
check.py / run_gpu_tests.py stop with "missing vk_compute_test", and so do the fuzzers.

    python -I tests/test_build_runtime_fallback.py
"""
from pathlib import Path
import re
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "runtime" / "test" / "shaders"), str(ROOT / "tools")]
import check  # noqa: E402
import image_fuzz  # noqa: E402
import run_gpu_tests  # noqa: E402

SCRIPT = ROOT / "scripts" / "build_runtime.ps1"


def fallback_body(text: str) -> str:
    """The cl.exe branch: from the line-starting `} else {` after the CMake test to the end of its here-string."""
    start = re.search(r"^\} else \{", text, re.M).start()
    return text[start:text.index('"@', text.index('$body = @"', start))]


class FallbackBuildsTheHarnesses(unittest.TestCase):
    def setUp(self):
        self.body = fallback_body(SCRIPT.read_text(encoding="utf-8"))
        self.programs = set(re.findall(r"/Fe:([A-Za-z0-9_]+)\.exe", self.body))

    def test_the_harnesses_the_gpu_tools_require_are_built(self):
        needed = set(run_gpu_tests.REQUIRED_TEST_PROGRAMS) | set(check.TEST_PROGRAMS) | {image_fuzz.HARNESS}
        self.assertEqual(needed - self.programs, set(), "missing from build_runtime.ps1's cl.exe fallback")

    def test_every_test_source_it_compiles_exists(self):
        sources = re.findall(r'"\$src\\test\\([A-Za-z0-9_]+\.cpp)"', self.body)
        self.assertTrue(sources)
        for name in sources:
            with self.subTest(source=name):
                self.assertTrue((ROOT / "runtime" / "test" / name).is_file())

    def test_build_cmd_sets_utf8_codepage_for_ninja_showincludes(self):
        text = SCRIPT.read_text(encoding="utf-8")
        self.assertIn("chcp 65001 >nul", text)
        self.assertLess(text.index("chcp 65001 >nul"), text.index('call `"$vcvars`"'))


if __name__ == "__main__":
    unittest.main()
