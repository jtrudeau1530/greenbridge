"""run_gpu_tests.py finds the test programs where the CMake generator put them: in the build folder (Makefiles,
Ninja) or in build/<Config> (Visual Studio generator, TEST-38). Fake files only, so this runs on any OS."""
from pathlib import Path
import os
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "runtime" / "test" / "shaders"))
sys.path.insert(0, str(ROOT / "scripts"))
import run_gpu_tests

PROGRAMS = ("vk_compute_test.exe", "vk_store_test.exe", "vk_harness.exe")


def touch(folder: Path, names=PROGRAMS, mtime=None):
    folder.mkdir(parents=True, exist_ok=True)
    for n in names:
        f = folder / n
        f.write_bytes(b"")
        if mtime is not None:
            os.utime(f, (mtime, mtime))


class ResolveBuildDir(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.build = Path(tmp.name) / "build"
        self.build.mkdir()

    def resolve(self):
        return run_gpu_tests.resolve_build_dir(str(self.build), PROGRAMS)

    def test_programs_directly_in_the_build_folder(self):
        touch(self.build)
        self.assertEqual(self.resolve(), str(self.build))

    def test_visual_studio_generator_subfolder(self):
        touch(self.build / "RelWithDebInfo")
        self.assertEqual(self.resolve(), str(self.build / "RelWithDebInfo"))

    def test_release_subfolder(self):
        touch(self.build / "Release")
        self.assertEqual(self.resolve(), str(self.build / "Release"))

    def test_build_folder_wins_over_a_subfolder(self):
        touch(self.build)
        touch(self.build / "Release")
        self.assertEqual(self.resolve(), str(self.build))

    def test_most_recently_built_configuration_wins(self):
        touch(self.build / "Debug", mtime=1_000_000)
        touch(self.build / "Release", mtime=3_000_000)
        touch(self.build / "RelWithDebInfo", mtime=2_000_000)
        self.assertEqual(self.resolve(), str(self.build / "Release"))

    def test_a_configuration_missing_a_program_is_skipped(self):
        touch(self.build / "Release", names=PROGRAMS[:2], mtime=9_000_000)    # newer but incomplete
        touch(self.build / "Debug", mtime=1_000_000)
        self.assertEqual(self.resolve(), str(self.build / "Debug"))

    def test_nothing_found_returns_the_folder_given(self):
        # so the "Test programs not found in <folder>" message names what the user passed
        self.assertEqual(self.resolve(), str(self.build))
        touch(self.build / "Release", names=PROGRAMS[:1])
        self.assertEqual(self.resolve(), str(self.build))

    def test_unknown_configuration_folder_is_not_searched(self):
        touch(self.build / "Custom")
        self.assertEqual(self.resolve(), str(self.build))


if __name__ == "__main__":
    unittest.main()
