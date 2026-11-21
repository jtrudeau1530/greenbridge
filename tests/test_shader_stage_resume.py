"""Shader stage of make_game.py: resuming a stopped build, Windows-safe carve names, an empty unpack (#126).

Standard library only; no game data: the tools are mocked or fed hand-made inputs.
"""
import os
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
import make_game  # noqa: E402
import shader_carve  # noqa: E402

UNPACK = os.path.join(ROOT, "tools", "fssm_unpack.py")


class CarveFileNameTest(unittest.TestCase):
    def names(self, *shader_names):
        seen, used = {}, set()
        return [shader_carve.file_name(n, seen, used) for n in shader_names]

    def test_repeated_name_gets_a_suffix(self):
        self.assertEqual(self.names("ps_a", "ps_a", "ps_a"), ["ps_a.dxbc", "ps_a~1.dxbc", "ps_a~2.dxbc"])

    def test_names_differing_only_in_case_do_not_share_a_file(self):
        out = self.names("PS_Water", "ps_water")
        self.assertEqual(len({n.lower() for n in out}), 2, out)

    def test_names_that_sanitize_to_the_same_file_do_not_share_it(self):
        out = self.names("a:b", "a_b", "a?b")
        self.assertEqual(len({n.lower() for n in out}), 3, out)

    def test_suffix_never_lands_on_a_real_name(self):
        out = self.names("x~1", "x", "x")
        self.assertEqual(len({n.lower() for n in out}), 3, out)

    def test_windows_device_names_are_prefixed(self):
        for dev in ("CON", "nul", "Aux", "PRN", "COM1", "lpt9", "con.vs"):
            name = self.names(dev)[0]
            self.assertNotIn(name.split(".")[0].upper(), shader_carve.RESERVED, name)
        self.assertEqual(self.names("CONSOLE")[0], "CONSOLE.dxbc")


class EmptyUnpackTest(unittest.TestCase):
    def test_nothing_unpacked_fails_clearly_and_writes_no_index(self):
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "junk.dxbc"), "wb") as f:
                f.write(b"DXBC" + bytes(60))     # no parts: nothing to unpack
            r = subprocess.run([sys.executable, "-I", UNPACK, d], capture_output=True, text=True)
            self.assertEqual(r.returncode, 1, r.stdout + r.stderr)
            self.assertIn("ERROR", r.stderr)
            self.assertNotIn("Traceback", r.stderr)
            self.assertFalse(os.path.exists(os.path.join(d, "fssm_index.json")))

    def test_empty_folder_fails_clearly(self):
        with tempfile.TemporaryDirectory() as d:
            r = subprocess.run([sys.executable, "-I", UNPACK, d], capture_output=True, text=True)
            self.assertEqual(r.returncode, 1, r.stdout + r.stderr)
            self.assertNotIn("Traceback", r.stderr)


class ShaderStageResumeTest(unittest.TestCase):
    """Which steps shaders() runs for a given work folder. The tools are mocked; each mock writes what the real
    tool writes last, so the next run sees the same files a real one would."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.work = self.tmp.name
        self.dxbc = os.path.join(self.work, "shaders", "dxbc")
        self.spirv = os.path.join(self.work, "shaders", "spirv")
        self.ran = []

    def tearDown(self):
        self.tmp.cleanup()

    def touch(self, *parts):
        p = os.path.join(*parts)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        open(p, "w").close()

    def fake_py(self, tool, *args):
        self.ran.append(tool)
        if tool == "shader_carve.py":
            self.touch(self.dxbc, "index.json")
        elif tool == "fssm_unpack.py":
            self.touch(self.dxbc, "fssm_index.json")
        elif tool == "shader_disasm.py":
            os.makedirs(os.path.join(self.dxbc, "disasm"), exist_ok=True)

    def fake_translate(self, disasm, spirv, jobs):
        self.ran.append("translate")
        self.touch(spirv, "index.json")

    def run_stage(self):
        self.ran = []
        with mock.patch.object(make_game, "py", self.fake_py), \
                mock.patch.object(make_game, "need", lambda *a: None), \
                mock.patch.object(make_game, "translate", self.fake_translate), \
                mock.patch.object(make_game, "log", lambda *a: None):
            make_game.shaders(os.path.join(self.work, "game"), self.work, 2)
        return self.ran

    def test_fresh_folder_runs_every_step_then_nothing(self):
        self.assertEqual(self.run_stage(), ["shader_carve.py", "fssm_unpack.py", "shader_disasm.py", "translate"])
        self.assertEqual(self.run_stage(), [])

    def test_unpack_stopped_after_creating_fssm_folder_runs_again(self):
        self.touch(self.dxbc, "index.json")
        os.makedirs(os.path.join(self.dxbc, "fssm"))       # what a stopped fssm_unpack.py leaves
        self.assertEqual(self.run_stage(), ["fssm_unpack.py", "shader_disasm.py", "translate"])

    def test_disasm_stopped_after_many_files_runs_again(self):
        self.touch(self.dxbc, "index.json")
        self.touch(self.dxbc, "fssm_index.json")
        for i in range(1200):                               # past the old 1000-file threshold
            self.touch(self.dxbc, "disasm", f"s{i}.s")
        self.assertEqual(self.run_stage(), ["shader_disasm.py", "translate"])

    def test_translation_stopped_after_many_files_runs_again(self):
        self.run_stage()
        os.remove(os.path.join(self.spirv, "index.json"))   # stopped before --merge
        for i in range(1200):
            self.touch(self.spirv, f"s{i}.spv")
        self.assertEqual(self.run_stage(), ["translate"])

    def test_failed_disasm_leaves_no_marker(self):
        self.touch(self.dxbc, "index.json")
        self.touch(self.dxbc, "fssm_index.json")

        def failing_py(tool, *args):
            raise SystemExit(1)
        with mock.patch.object(make_game, "py", failing_py), \
                mock.patch.object(make_game, "need", lambda *a: None), \
                mock.patch.object(make_game, "log", lambda *a: None):
            with self.assertRaises(SystemExit):
                make_game.shaders(os.path.join(self.work, "game"), self.work, 2)
        self.assertFalse(os.path.exists(os.path.join(self.dxbc, "disasm.done")))

    def test_rerunning_a_step_reruns_the_steps_after_it(self):
        self.run_stage()
        os.remove(os.path.join(self.dxbc, "fssm_index.json"))
        self.assertEqual(self.run_stage(), ["fssm_unpack.py", "shader_disasm.py", "translate"])


if __name__ == "__main__":
    unittest.main()
