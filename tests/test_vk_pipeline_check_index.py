"""vk_pipeline_check: a missing, unreadable or empty index.json is an error, not an empty corpus (TEST-257).

It used to exit 0 after checking nothing. The index is read before Vulkan is loaded, so the failing cases need no GPU
and no game data. A valid `{}` stays an intentionally empty index: it gets past the index (the run may then stop at
Vulkan on a machine without a device, which is not what is checked here). Skips when vk_pipeline_check isn't built.
Set LEONIDA_VK_PIPELINE_CHECK to use a binary elsewhere.
"""
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def find_binary():
    env = os.environ.get("LEONIDA_VK_PIPELINE_CHECK")
    if env:
        return Path(env) if Path(env).is_file() else None
    build = ROOT / "runtime" / "build"
    for p in (build / "linux" / "vk_pipeline_check", build / "vk_pipeline_check", build / "Release" / "vk_pipeline_check.exe"):
        if p.is_file():
            return p
    return None


class IndexTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.exe = find_binary()
        if cls.exe is None:
            raise unittest.SkipTest("vk_pipeline_check isn't built (cmake -S runtime -B runtime/build/linux)")

    def run_tool(self, index_text):
        with tempfile.TemporaryDirectory() as d:
            if index_text is not None:
                Path(d, "index.json").write_text(index_text, encoding="utf-8")
            return subprocess.run([str(self.exe), d], capture_output=True, text=True, timeout=60)

    def assert_index_error(self, r):
        self.assertEqual(r.returncode, 2, r.stderr)
        self.assertIn("cannot use", r.stderr)
        self.assertIn("index.json", r.stderr)

    def test_missing_index_is_an_error(self):
        self.assert_index_error(self.run_tool(None))

    def test_empty_index_is_an_error(self):
        self.assert_index_error(self.run_tool(""))

    def test_whitespace_only_index_is_an_error(self):
        self.assert_index_error(self.run_tool(" \n\t \n"))

    def test_malformed_index_is_still_an_error(self):
        self.assert_index_error(self.run_tool('{"a": '))

    def test_valid_empty_object_is_an_empty_corpus(self):
        r = self.run_tool("{}")
        self.assertNotIn("cannot use", r.stderr)
        self.assertIn("0 shaders in index", r.stderr)


if __name__ == "__main__":
    unittest.main()
