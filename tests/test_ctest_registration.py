"""Every test executable the runtime builds must be registered with ctest.

``add_executable(foo_test ...)`` without an ``add_test(... COMMAND foo_test)`` builds
the test but never runs it -- in ``ctest``, in the Windows runtime job or under Wine.
This happened to ``d3d12_indirect_rt_test``; the only intentional exception is
``d3d12_clear_test`` (see the comment next to its target in runtime/CMakeLists.txt).
"""
import os
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CMAKE = Path(os.environ.get("LEONIDA_RUNTIME_CMAKE", str(ROOT / "runtime" / "CMakeLists.txt")))
SOURCE = Path(os.environ.get("LEONIDA_RUNTIME_TEST_SOURCE",
                             str(ROOT / "runtime" / "test" / "d3d12_indirect_rt_test.cpp")))

# d3d12_clear_test is deliberately run by hand until #521's tie rounding lands.
NOT_IN_CTEST = {"d3d12_clear_test"}


def built_test_executables(text):
    return {name for name in re.findall(r"add_executable\(\s*([A-Za-z0-9_]+)", text)
            if name.endswith("_test")}


def registered_commands(text):
    # add_test may span lines, e.g. add_test(NAME x\n  COMMAND foo_test ...).
    return set(re.findall(r"add_test\([^)]*?\bCOMMAND\s+\"?([A-Za-z0-9_]+)", text, re.S))


class CTestRegistrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cmake = CMAKE.read_text(encoding="utf-8")

    def test_all_test_executables_have_an_add_test(self):
        missing = built_test_executables(self.cmake) - registered_commands(self.cmake) - NOT_IN_CTEST
        self.assertEqual(missing, set(), f"built but never run by ctest: {sorted(missing)}")

    def test_the_exception_is_still_the_only_one(self):
        # An allowlist that grows unnoticed is how this bug happens again.
        self.assertEqual(NOT_IN_CTEST & built_test_executables(self.cmake), {"d3d12_clear_test"})

    def test_indirect_rt_skips_with_the_ctest_skip_code(self):
        self.assertTrue(
            "add_test(NAME d3d12_indirect_rt COMMAND d3d12_indirect_rt_test)" in self.cmake,
            "d3d12_indirect_rt_test is built but not registered with ctest",
        )
        self.assertTrue(
            "set_tests_properties(d3d12_indirect_rt PROPERTIES SKIP_RETURN_CODE 77)" in self.cmake,
            "d3d12_indirect_rt must skip with 77 on a machine without a Vulkan GPU",
        )

        lines = SOURCE.read_text(encoding="utf-8").splitlines()
        skips = []
        for i, line in enumerate(lines):
            if 'std::printf("SKIP:' in line:
                window = "\n".join(lines[i:i + 6])
                codes = re.findall(r"return (\d+);", window)
                skips.append(int(codes[0]) if codes else None)
        # no loader, no ICD, no Vulkan 1.3 GPU -- all three are a ctest SKIP, not a failure.
        self.assertEqual(len(skips), 3, f"expected three whole-test SKIP paths, got {len(skips)}")
        self.assertEqual(skips, [77, 77, 77], "SKIP paths must return 77 for SKIP_RETURN_CODE 77")


if __name__ == "__main__":
    unittest.main()
