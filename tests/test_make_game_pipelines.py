"""make_game.py: --jobs reaches vk_pipeline_check in the pipelines stage.

Standard library only; no GPU, no Vulkan SDK, no game data: the checker is mocked.
"""
import os
import sys
import tempfile
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import make_game  # noqa: E402


class PipelinesJobsTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = tmp.name
        self.game = os.path.join(self.root, "game")
        self.work = os.path.join(self.root, "work")
        self.out = os.path.join(self.root, "out")
        spirv = os.path.join(self.work, "shaders", "spirv")
        os.makedirs(self.game)
        os.makedirs(spirv)
        with open(os.path.join(spirv, "index.json"), "w", encoding="utf-8") as f:
            f.write("{}")

    def checker_args(self, call):
        with mock.patch.object(make_game, "pipeline_check_exe", return_value="vk_pipeline_check.exe"), \
                mock.patch.object(make_game, "run") as run, \
                mock.patch.object(make_game, "log", lambda *a: None):
            call()
        self.assertEqual(run.call_count, 1)
        return [str(a) for a in run.call_args.args[0]]

    def jobs_of(self, args):
        self.assertEqual(args.count("--jobs"), 1, args)
        return args[args.index("--jobs") + 1]

    def main_with(self, *extra):
        argv = ["make_game.py", "--game", self.game, "--work", self.work, "--out", self.out,
                "--stages", "pipelines", *extra]
        with mock.patch.object(sys, "argv", argv), mock.patch.object(make_game, "below_normal"), \
                mock.patch.object(make_game, "check", return_value={"expected_exe": "g.exe"}), \
                mock.patch.object(make_game, "preflight"):
            self.assertEqual(make_game.main(), 0)

    def test_the_stage_passes_its_jobs(self):
        args = self.checker_args(lambda: make_game.pipelines(self.work, self.out, 7))
        self.assertEqual(self.jobs_of(args), "7")
        self.assertEqual(args[1:3], [os.path.join(self.work, "shaders", "spirv"),
                                     os.path.join(self.out, "pipeline_cache.bin")])

    def test_jobs_on_the_command_line_reach_the_checker(self):
        self.assertEqual(self.jobs_of(self.checker_args(lambda: self.main_with("--jobs", "5"))), "5")

    def test_the_default_matches_the_checker_default(self):
        # vk_pipeline_check's own default is half the hardware threads, as is make_game.py's --jobs.
        want = str(max(1, (os.cpu_count() or 2) // 2))
        self.assertEqual(self.jobs_of(self.checker_args(lambda: self.main_with())), want)


if __name__ == "__main__":
    unittest.main()
