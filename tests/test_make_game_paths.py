"""Build destinations must stay outside the game folder, including through directory links.

    python -I tests/test_make_game_paths.py

Only empty temporary directories are used; no game files or build tools are needed.
"""
from contextlib import ExitStack
import io
import os
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import make_game


class BuildDestinationTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = tmp.name
        self.game = os.path.join(self.root, "game")
        os.mkdir(self.game)
        self.alias = os.path.join(self.root, "game-link")
        self.link(self.game, self.alias)

    def link(self, target, path):
        if os.name == "nt":
            # Junctions work without Developer Mode or symlink privileges on Windows.
            result = subprocess.run(["cmd", "/c", "mklink", "/J", path, target],
                                    capture_output=True, text=True)
            if result.returncode:
                self.skipTest("cannot create a directory junction: " + result.stderr)
            self.addCleanup(os.rmdir, path)
        else:
            os.symlink(target, path, target_is_directory=True)
            self.addCleanup(os.unlink, path)

    def run_build(self, game, out, work, rejected):
        argv = ["make_game.py", "--game", game, "--out", out, "--work", work,
                "--stages", "shaders"]
        with ExitStack() as stack:
            stack.enter_context(patch.object(sys, "argv", argv))
            stack.enter_context(patch("make_game.below_normal"))
            stack.enter_context(patch("make_game.check", return_value={}))
            stack.enter_context(patch("make_game.log"))
            preflight = stack.enter_context(patch("make_game.preflight"))
            shaders = stack.enter_context(patch("make_game.shaders"))
            stderr = stack.enter_context(patch("sys.stderr", new_callable=io.StringIO))
            if rejected:
                with self.assertRaises(SystemExit) as error:
                    make_game.main()
                self.assertEqual(error.exception.code, 1)
                self.assertIn("--out and --work must be outside the game folder", stderr.getvalue())
                preflight.assert_not_called()
                shaders.assert_not_called()
            else:
                self.assertEqual(make_game.main(), 0)
                preflight.assert_called_once()
                shaders.assert_called_once()

    def test_destinations_through_game_link_are_rejected(self):
        for option in ("out", "work"):
            for suffix in ("", "new", os.path.join("new", "nested")):
                with self.subTest(option=option, suffix=suffix):
                    destinations = {name: os.path.join(self.root, name) for name in ("out", "work")}
                    destinations[option] = os.path.join(self.alias, suffix)
                    self.run_build(self.game, **destinations, rejected=True)
                    self.assertFalse(os.path.exists(os.path.join(self.game, "new")))

    def test_game_link_cannot_hide_destinations_in_real_game_folder(self):
        for option in ("out", "work"):
            with self.subTest(option=option):
                destinations = {name: os.path.join(self.root, name) for name in ("out", "work")}
                destinations[option] = os.path.join(self.game, "new")
                self.run_build(self.alias, **destinations, rejected=True)

    def test_link_to_unrelated_output_folder_is_allowed(self):
        target = os.path.join(self.root, "output")
        os.mkdir(target)
        alias = os.path.join(self.root, "output-link")
        self.link(target, alias)
        self.run_build(self.game, os.path.join(alias, "new"),
                       os.path.join(self.root, "work"), rejected=False)


if __name__ == "__main__":
    unittest.main()
