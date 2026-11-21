"""make_game.py --wave64-emulate auto|on|off (SHADER-11, #94): when the build translates with --wave64-emulate.

Standard library only; no GPU, no Vulkan SDK, no game data: vulkaninfo and the translator are mocked, and the
vulkaninfo text below is a hand-written excerpt of its format.

    python -I tests/test_make_game_wave64.py
"""
import os
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import make_game  # noqa: E402


def vulkaninfo_text(*max_sizes):
    """vulkaninfo's plain-text layout: one block per device, each with a maxSubgroupSize line."""
    blocks = []
    for i, size in enumerate(max_sizes):
        blocks.append(f"GPU{i}:\n"
                      "VkPhysicalDeviceSubgroupSizeControlProperties:\n"
                      "-----------------------------------------------\n"
                      f"\tminSubgroupSize              = {min(size, 32)}\n"
                      f"\tmaxSubgroupSize              = {size}\n"
                      "\tmaxComputeWorkgroupSubgroups = 1024\n")
    return "\n".join(blocks)


class SmallestMaxSubgroupTest(unittest.TestCase):
    def test_one_device(self):
        self.assertEqual(make_game.smallest_max_subgroup(vulkaninfo_text(32)), 32)
        self.assertEqual(make_game.smallest_max_subgroup(vulkaninfo_text(64)), 64)

    def test_several_devices_give_the_smallest(self):
        self.assertEqual(make_game.smallest_max_subgroup(vulkaninfo_text(64, 32)), 32)
        self.assertEqual(make_game.smallest_max_subgroup(vulkaninfo_text(64, 64)), 64)

    def test_json_output(self):
        self.assertEqual(make_game.smallest_max_subgroup('{"minSubgroupSize": 32, "maxSubgroupSize": 64}'), 64)

    def test_no_subgroup_size_gives_none(self):
        self.assertIsNone(make_game.smallest_max_subgroup(""))
        self.assertIsNone(make_game.smallest_max_subgroup("ERROR: vkCreateInstance failed\n"))
        self.assertIsNone(make_game.smallest_max_subgroup("\tminSubgroupSize = 32\n"))


class Wave64EmulateTest(unittest.TestCase):
    """wave64_emulate(): on and off are taken as given; auto asks vulkaninfo and falls back to on."""

    def decide(self, mode, which="vulkaninfo", stdout="", error=None):
        def fake_run(args, **kw):
            if error:
                raise error
            return subprocess.CompletedProcess(args, 0, stdout=stdout, stderr="")
        with mock.patch.object(make_game.shutil, "which", return_value=which), \
                mock.patch.object(make_game.subprocess, "run", side_effect=fake_run) as run, \
                mock.patch.object(make_game, "log", lambda *a: None):
            result = make_game.wave64_emulate(mode)
        return result, run.called

    def test_on_and_off_never_ask_vulkaninfo(self):
        self.assertEqual(self.decide("on", stdout=vulkaninfo_text(64)), (True, False))
        self.assertEqual(self.decide("off", stdout=vulkaninfo_text(32)), (False, False))

    def test_auto_32_wide_turns_it_on(self):
        self.assertEqual(self.decide("auto", stdout=vulkaninfo_text(32)), (True, True))

    def test_auto_64_wide_leaves_it_off(self):
        self.assertEqual(self.decide("auto", stdout=vulkaninfo_text(64)), (False, True))

    def test_auto_with_a_64_and_a_32_wide_gpu_turns_it_on(self):
        self.assertEqual(self.decide("auto", stdout=vulkaninfo_text(64, 32))[0], True)

    def test_auto_without_vulkaninfo_turns_it_on(self):
        self.assertEqual(self.decide("auto", which=None), (True, False))

    def test_auto_when_vulkaninfo_cannot_run_turns_it_on(self):
        self.assertTrue(self.decide("auto", error=OSError("cannot start"))[0])
        self.assertTrue(self.decide("auto", error=subprocess.TimeoutExpired("vulkaninfo", 120))[0])

    def test_auto_when_vulkaninfo_prints_no_subgroup_size_turns_it_on(self):
        self.assertTrue(self.decide("auto", stdout="ERROR: no Vulkan device\n")[0])


class TranslateFlagTest(unittest.TestCase):
    """translate() passes --wave64-emulate to every shard, and only when asked."""

    def shard_args(self, emulate):
        started = []

        class FakeProc:
            def __init__(self, args):
                started.append(args)

            def wait(self):
                return 0
        with tempfile.TemporaryDirectory() as spirv, \
                mock.patch.object(make_game.subprocess, "Popen", side_effect=FakeProc), \
                mock.patch.object(make_game, "py") as py, \
                mock.patch.object(make_game, "log", lambda *a: None):
            make_game.translate("disasm", spirv, 3, emulate=emulate)
        self.assertEqual(py.call_count, 1)                    # the --merge step
        self.assertNotIn("--wave64-emulate", py.call_args.args)
        return started

    def test_emulated_shards_get_the_flag(self):
        shards = self.shard_args(True)
        self.assertEqual(len(shards), 3)
        for args in shards:
            self.assertEqual(args.count("--wave64-emulate"), 1, args)

    def test_native_shards_do_not(self):
        for args in self.shard_args(False):
            self.assertNotIn("--wave64-emulate", args)


class ArgumentTest(unittest.TestCase):
    def test_bad_mode_is_rejected(self):
        p = subprocess.run([sys.executable, "-I", os.path.join(ROOT, "make_game.py"), "--wave64-emulate", "maybe"],
                           capture_output=True, text=True)
        self.assertEqual(p.returncode, 2)
        self.assertIn("invalid choice", p.stderr)

    def test_help_lists_the_modes(self):
        p = subprocess.run([sys.executable, "-I", os.path.join(ROOT, "make_game.py"), "--help"],
                           capture_output=True, text=True)
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertIn("--wave64-emulate {auto,on,off}", p.stdout)


if __name__ == "__main__":
    unittest.main()
