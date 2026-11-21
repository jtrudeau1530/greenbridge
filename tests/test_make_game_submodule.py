"""Exercise submodule initialization with local Git repositories and a synthetic file."""
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import make_game  # noqa: E402


@unittest.skipUnless(shutil.which("git"), "requires Git")
class EnsureSubmoduleTest(unittest.TestCase):
    submodule = Path("third_party", "rpf9-explorer")

    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory(prefix="leonida submodule ")))
        self.enterContext(mock.patch.dict(os.environ, {
            "GIT_ALLOW_PROTOCOL": "file",
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_NOSYSTEM": "1",
        }))

        dependency = self.root / "dependency"
        self.git(self.root, "init", "-q", dependency)
        (dependency / "Core").mkdir()
        (dependency / "Core" / "fixture.txt").write_text("synthetic\n", encoding="utf-8")
        self.git(dependency, "add", "Core")
        self.git(dependency, "commit", "-qm", "Add fixture")

        self.source = self.root / "source"
        self.git(self.root, "init", "-q", self.source)
        self.git(self.source, "submodule", "add", "-q", dependency.as_posix(), self.submodule.as_posix())
        self.git(self.source, "commit", "-qm", "Add submodule")

    def git(self, cwd, *args):
        result = subprocess.run(["git", "-c", "user.name=Test", "-c", "user.email=test@example.invalid",
                                 *args], cwd=cwd, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def assert_initializes_submodule(self, checkout):
        fixture = checkout / self.submodule / "Core" / "fixture.txt"
        self.assertFalse(fixture.exists())
        with mock.patch.object(make_game, "ROOT", str(checkout)):
            make_game.ensure_submodule()
        self.assertEqual(fixture.read_text(encoding="utf-8"), "synthetic\n")

    def test_initializes_submodule_in_clone(self):
        checkout = self.root / "clone"
        self.git(self.root, "clone", "-q", "--no-recurse-submodules", self.source, checkout)
        self.assertTrue((checkout / ".git").is_dir())
        self.assert_initializes_submodule(checkout)

    def test_initializes_submodule_in_linked_worktree(self):
        checkout = self.root / "worktree"
        self.git(self.source, "worktree", "add", "-q", "--detach", checkout)
        self.assertTrue((checkout / ".git").is_file())
        self.assert_initializes_submodule(checkout)


if __name__ == "__main__":
    unittest.main()
