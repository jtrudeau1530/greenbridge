"""make_game.py forwards the translator's fallback flags (--no-float-controls, --bary-fallback,
--texel-offset-fallback, --mesh-compute) and records them beside the translation like the wave64 mode
(SHADER-99 follow-up to #664, @WHOISJARDAS's finding).

Standard library only; no GPU, no Vulkan SDK, no game data: vulkaninfo and the translator are mocked.

    python -I tests/test_make_game_translator_flags.py
"""
import argparse
import json
import os
import sys
import tempfile
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import make_game  # noqa: E402

ALL_FLAGS = ("--no-float-controls", "--bary-fallback", "--texel-offset-fallback", "--mesh-compute")


class FallbackFlagsTest(unittest.TestCase):
    """fallback_flags(): the flags the parsed arguments ask for, in FALLBACK_FLAGS order."""

    def args_with(self, *flags):
        return argparse.Namespace(**{attr: flag in flags for flag, attr in make_game.FALLBACK_FLAGS})

    def test_none_by_default(self):
        self.assertEqual(make_game.fallback_flags(self.args_with()), ())

    def test_each_flag_alone(self):
        for flag, _attr in make_game.FALLBACK_FLAGS:
            self.assertEqual(make_game.fallback_flags(self.args_with(flag)), (flag,))

    def test_all_flags_in_record_order(self):
        self.assertEqual(make_game.fallback_flags(self.args_with(*reversed(ALL_FLAGS))), ALL_FLAGS)


class TranslateFallbacksTest(unittest.TestCase):
    """translate() passes the fallback flags to every shard, and re-plans when they change."""

    def shards_in(self, spirv, fallbacks=(), jobs=2, pre_parts=0):
        started = []
        for i in range(pre_parts):
            with open(os.path.join(spirv, f"index.part{i}.json"), "w", encoding="utf-8") as f:
                json.dump({"old": True}, f)

        class FakeProc:
            def __init__(self, args):
                started.append(args)

            def wait(self):
                return 0

        with mock.patch.object(make_game.subprocess, "Popen", side_effect=FakeProc), \
                mock.patch.object(make_game, "py"), \
                mock.patch.object(make_game, "log", lambda *a: None):
            make_game.translate("disasm", spirv, jobs, fallbacks=fallbacks)
        return started

    def shards(self, fallbacks=(), jobs=2, pre_parts=0):
        with tempfile.TemporaryDirectory() as spirv:
            return self.shards_in(spirv, fallbacks=fallbacks, jobs=jobs, pre_parts=pre_parts)

    def test_shards_get_the_fallback_flags(self):
        for args in self.shards(fallbacks=("--no-float-controls", "--bary-fallback")):
            self.assertEqual(args.count("--no-float-controls"), 1, args)
            self.assertEqual(args.count("--bary-fallback"), 1, args)

    def test_no_fallback_flags_by_default(self):
        for args in self.shards():
            for flag in ALL_FLAGS:
                self.assertNotIn(flag, args)

    def test_the_same_flags_keep_finished_shards(self):
        with tempfile.TemporaryDirectory() as spirv:
            self.shards_in(spirv, fallbacks=ALL_FLAGS)           # writes the plan
            for i in range(2):                                   # as if the shards had finished
                with open(os.path.join(spirv, f"shard{i}.json"), "w", encoding="utf-8") as f:
                    json.dump({"index": i, "count": 2}, f)
                with open(os.path.join(spirv, f"index.part{i}.json"), "w", encoding="utf-8") as f:
                    json.dump({"old": True}, f)
            again = self.shards_in(spirv, fallbacks=ALL_FLAGS)
            self.assertEqual(again, [])

    def test_changing_fallbacks_discards_finished_shards(self):
        again = self.shards(fallbacks=("--mesh-compute", "--bary-fallback"), pre_parts=2)
        self.assertEqual(len(again), 2, again)       # the parts were translated with other flags


class RecordedFlagsTest(unittest.TestCase):
    """shaders() records the flags in spirv/translator_flags.txt and re-translates when they change."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.work = tmp.name
        self.dxbc = os.path.join(self.work, "shaders", "dxbc")
        self.spirv = os.path.join(self.work, "shaders", "spirv")

    def touch(self, *parts):
        p = os.path.join(*parts)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            if p == os.path.join(self.dxbc, "index.json"):
                json.dump([{"file": "shader.dxbc"}], f)
            elif p == os.path.join(self.dxbc, "fssm_index.json"):
                json.dump([{"name": "shader"}], f)
            elif p.endswith(".s"):
                f.write("synthetic listing\n")

    def run_stage(self, fallbacks=()):
        ran = []

        def fake_py(tool, *args):
            if tool == "shader_carve.py":
                self.touch(self.dxbc, "index.json")
            elif tool == "fssm_unpack.py":
                self.touch(self.dxbc, "fssm_index.json")
            elif tool == "shader_disasm.py":
                self.touch(self.dxbc, "disasm", "shader.s")

        def fake_translate(disasm, spirv, jobs, emulate=False, fallbacks=()):
            ran.append("translate")
            self.touch(spirv, "index.json")

        with mock.patch.object(make_game, "py", fake_py), \
                mock.patch.object(make_game, "need", lambda *a: None), \
                mock.patch.object(make_game, "translate", fake_translate), \
                mock.patch.object(make_game, "llvm_mc_signature", return_value="test llvm-mc 21"), \
                mock.patch.object(make_game, "log", lambda *a: None):
            make_game.shaders(os.path.join(self.work, "game"), self.work, 2, fallbacks=fallbacks)
        return ran

    def test_flags_are_recorded_beside_the_translation(self):
        self.assertEqual(self.run_stage(fallbacks=("--no-float-controls",)), ["translate"])
        with open(os.path.join(self.spirv, "translator_flags.txt"), encoding="utf-8") as f:
            self.assertEqual(f.read(), "--no-float-controls\n")
        self.assertEqual(self.run_stage(fallbacks=("--no-float-controls",)), [])

    def test_changing_flags_translates_again(self):
        self.run_stage(fallbacks=("--no-float-controls",))
        self.assertEqual(self.run_stage(fallbacks=("--no-float-controls", "--mesh-compute")), ["translate"])

    def test_a_folder_from_before_the_file_existed_stays_reusable_without_flags(self):
        self.run_stage()          # no flags: an old folder (no translator_flags.txt) behaves the same
        os.remove(os.path.join(self.spirv, "translator_flags.txt"))
        self.assertEqual(self.run_stage(), [])

    def test_flags_asked_on_a_folder_translated_without_them_retranslate(self):
        self.run_stage()
        os.remove(os.path.join(self.spirv, "translator_flags.txt"))
        self.assertEqual(self.run_stage(fallbacks=("--bary-fallback",)), ["translate"])


class MainWiringTest(unittest.TestCase):
    """main() parses the fallback flags and hands the same tuple to preflight() and shaders()."""

    def run_main(self, *extra_argv, per_vertex=None, max_push_constants=None):
        argv = ["make_game.py", "--stages", "shaders", *extra_argv]
        with mock.patch.object(sys, "argv", argv), \
                mock.patch("make_game.below_normal"), \
                mock.patch("make_game.choose_game", return_value=os.path.abspath("C:/dummy_game")), \
                mock.patch("make_game.check", return_value={"expected_exe": "game.exe"}), \
                mock.patch("make_game.wave64_settings", return_value=(False, True)), \
                mock.patch("make_game.runtime_gpu_per_vertex", return_value=per_vertex), \
                mock.patch("make_game.runtime_gpu_max_push_constants", return_value=max_push_constants), \
                mock.patch("make_game.preflight") as preflight, \
                mock.patch("make_game.shaders") as shaders, \
                mock.patch("make_game.log"):
            self.assertEqual(make_game.main(), 0)
        return preflight.call_args.kwargs["fallbacks"], shaders.call_args.kwargs["fallbacks"]

    def test_the_flags_reach_the_shaders_stage(self):
        pre, sh = self.run_main("--no-float-controls", "--bary-fallback", "--mesh-compute")
        self.assertEqual(pre, ("--no-float-controls", "--bary-fallback", "--mesh-compute"))
        self.assertEqual(sh, pre)

    def test_no_flags_by_default(self):
        pre, sh = self.run_main()
        self.assertEqual(pre, ())
        self.assertEqual(sh, ())

    def test_f16_interp_fallback_follows_the_device(self):
        """On by itself where vk_subgroup_info says the GPU can't take PerVertexKHR inputs (MoltenVK), not where
        it can or can't tell; asked for explicitly, it is on everywhere."""
        self.assertEqual(self.run_main(per_vertex=False), (("--f16-interp-fallback",),) * 2)
        self.assertEqual(self.run_main(per_vertex=True), ((), ()))
        self.assertEqual(self.run_main("--f16-interp-fallback", per_vertex=True), (("--f16-interp-fallback",),) * 2)

    def test_pc_tail_ubo_follows_the_device(self):
        """On by itself where maxPushConstantsSize is below what the shaders need (144, or 152 with
        --mesh-compute), not where it fits or can't be read; asked for explicitly, it is on everywhere."""
        self.assertEqual(self.run_main(max_push_constants=128), (("--pc-tail-ubo",),) * 2)
        self.assertEqual(self.run_main(max_push_constants=144), ((), ()))
        self.assertEqual(self.run_main("--mesh-compute", max_push_constants=144),
                         (("--mesh-compute", "--pc-tail-ubo"),) * 2)
        self.assertEqual(self.run_main("--mesh-compute", max_push_constants=152), (("--mesh-compute",),) * 2)
        self.assertEqual(self.run_main(max_push_constants=256), ((), ()))
        self.assertEqual(self.run_main(max_push_constants=None), ((), ()))
        self.assertEqual(self.run_main("--pc-tail-ubo", max_push_constants=256), (("--pc-tail-ubo",),) * 2)

    def test_runtime_gpu_max_push_constants_parses_probe(self):
        for raw, want in (({"max_push_constants_size": 128}, 128),
                          ({"max_push_constants_size": 256}, 256),
                          ({}, None),
                          ({"max_push_constants_size": 0}, None),
                          ({"max_push_constants_size": -1}, None),
                          ({"max_push_constants_size": True}, None),
                          ({"max_push_constants_size": "128"}, None)):
            with self.subTest(raw=raw), mock.patch("make_game.runtime_gpu_info", return_value=raw):
                self.assertEqual(make_game.runtime_gpu_max_push_constants(), want)


if __name__ == "__main__":
    unittest.main()
