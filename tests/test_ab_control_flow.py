"""Synthetic regression tests for tools/ab_control_flow.py; no game corpus or GPU is required.

    python -I tests/test_ab_control_flow.py
"""
import argparse
import contextlib
import importlib.util
import io
import os
import tempfile
import unittest
from types import SimpleNamespace

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


AB = load("ab_control_flow_under_test", os.path.join(ROOT, "tools", "ab_control_flow.py"))


class CandidateSelectionTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.disasm = os.path.join(self.tmp.name, "disasm")
        self.spirv = os.path.join(self.tmp.name, "spirv")
        os.makedirs(self.disasm)
        os.makedirs(self.spirv)

    def tearDown(self):
        self.tmp.cleanup()

    def add_shader(self, name, asm, with_spv=True):
        with open(os.path.join(self.disasm, name + ".s"), "w", encoding="utf-8") as f:
            f.write(asm)
        if with_spv:
            with open(os.path.join(self.spirv, name + ".spv"), "wb") as f:
                f.write(b"synthetic SPIR-V placeholder")

    @staticmethod
    def shader(stage="ps", bindings=None, features=None):
        return {"stage": stage, "bindings": bindings or [], "features": features or []}

    def test_automatic_selection_applies_every_metadata_and_file_filter(self):
        index = {
            "z_valid": self.shader(),
            "a_valid": self.shader(features=["shaderInt64"]),
            "wrong_stage": self.shader(stage="vs"),
            "wrong_binding": self.shader(bindings=[9]),
            "wrong_feature": self.shader(features=["unsupportedFeature"]),
            "one_branch": self.shader(),
            "writes_memory": self.shader(),
            "missing_spv": self.shader(),
        }
        two_branches = "\ts_cbranch_scc0 12\n\ts_branch 24\n"
        self.add_shader("z_valid", two_branches)
        self.add_shader("a_valid", two_branches)
        self.add_shader("one_branch", "\ts_branch 24\n")
        self.add_shader("writes_memory", two_branches + "\tbuffer_store_dword v0, v1, s[0:1]\n")
        self.add_shader("missing_spv", two_branches, with_spv=False)
        picked = AB.select_candidates(index, self.disasm, self.spirv, count=10)
        self.assertEqual(picked, ["a_valid", "z_valid"])

    def test_backward_branch_detection_and_loops_filter(self):
        self.assertFalse(AB.is_backward_branch("\ts_cbranch_scc0 12\n\ts_branch 40\n"))
        self.assertTrue(AB.is_backward_branch("\ts_cbranch_scc0 32768\n\ts_branch 40\n"))
        index = {"forward": self.shader(), "backward": self.shader()}
        self.add_shader("forward", "\ts_cbranch_scc0 12\n\ts_branch 40\n")
        self.add_shader("backward", "\ts_cbranch_scc0 32768\n\ts_branch 40\n")
        self.assertEqual(AB.select_candidates(index, self.disasm, self.spirv, loops="none"), ["forward"])
        self.assertEqual(AB.select_candidates(index, self.disasm, self.spirv, loops="only"), ["backward"])

    def test_names_file_overrides_automatic_selection(self):
        names = os.path.join(self.tmp.name, "names.txt")
        with open(names, "w", encoding="utf-8") as f:
            f.write("not_in_index\nfiltered_out\n")
        # Both names have synthetic disassemblies but no index entries or prebuilt SPIR-V: --names bypasses
        # automatic metadata and file selection, while preserving files the later translation stage needs.
        self.add_shader("not_in_index", "\ts_branch 12\n\ts_branch 24\n", with_spv=False)
        self.add_shader("filtered_out", "\tbuffer_store_dword v0, v1, s[0:1]\n", with_spv=False)
        self.assertEqual(AB.select_candidates({}, self.disasm, self.spirv, loops="only", names_file=names),
                         ["not_in_index", "filtered_out"])

    def test_count_limits_the_automatic_selection(self):
        index = {name: self.shader() for name in ("c", "b", "a")}
        for name in index:
            self.add_shader(name, "\ts_cbranch_scc0 12\n\ts_branch 40\n")
        self.assertEqual(AB.select_candidates(index, self.disasm, self.spirv, count=2), ["a", "b"])


    def test_count_below_one_is_refused(self):
        index = {"a": self.shader()}
        self.add_shader("a", "\ts_cbranch_scc0 12\n\ts_branch 40\n")
        for count in (0, -3):
            with self.subTest(count=count), self.assertRaises(ValueError):
                AB.select_candidates(index, self.disasm, self.spirv, count=count)


class CountArgumentTest(unittest.TestCase):
    def test_positive_int(self):
        self.assertEqual(AB.positive_int("150"), 150)
        for text in ("0", "-1", "x"):
            with self.subTest(text=text), self.assertRaises(argparse.ArgumentTypeError):
                AB.positive_int(text)

    def test_cli_refuses_counts_below_one_before_making_out(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "out")
            for flag in ("--count", "--runs", "--timeout"):
                with self.subTest(flag=flag), self.assertRaises(SystemExit) as ctx, \
                        contextlib.redirect_stderr(io.StringIO()) as err:
                    AB.main([tmp, tmp, out, flag, "0"])
                self.assertEqual(ctx.exception.code, 2)
                self.assertIn(flag, err.getvalue())
            self.assertFalse(os.path.exists(out))


class ResultClassificationTest(unittest.TestCase):
    def test_translation_failure(self):
        self.assertEqual(AB.classify_result(False, [b"same"], [b"same"]), "a translation is invalid")

    def test_dispatch_harness_failure(self):
        self.assertEqual(AB.classify_result(True, [None], [b"image"]), "dispatch form failed to run")

    def test_structured_harness_failure(self):
        self.assertEqual(AB.classify_result(True, [b"image"], [None]), "structured form failed to run")

    def test_both_forms_timeout(self):
        self.assertEqual(AB.classify_result(True, ["timeout"], ["timeout"]), "neither terminates")

    def test_only_one_form_times_out(self):
        self.assertEqual(AB.classify_result(True, ["timeout"], [b"image"]), "ONLY ONE FORM TERMINATES")

    def test_harness_failure_is_not_mislabeled_as_the_other_form_terminating(self):
        self.assertEqual(AB.classify_result(True, ["timeout"], [None]), "structured form failed to run")

    def test_inconsistent_repeated_runs_are_unstable(self):
        self.assertEqual(AB.classify_result(True, [b"first", b"second"], [b"image"]),
                         "unstable from run to run")

    def test_identical_images(self):
        self.assertEqual(AB.classify_result(True, [b"same", b"same"], [b"same", b"same"]), "identical")

    def test_different_images(self):
        self.assertEqual(AB.classify_result(True, [b"dispatch"], [b"structured"]), "DIFFERENT")


class OutputAndVertexShaderTest(unittest.TestCase):
    def test_outcomes_tsv_contents(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = AB.write_outcomes(tmp, [("shader_a", "identical"), ("shader_b", "DIFFERENT")])
            with open(path, encoding="utf-8") as f:
                self.assertEqual(f.read(), "shader_a\tidentical\nshader_b\tDIFFERENT\n")

    def test_vertex_shader_generation_assembles_and_validates_all_locations(self):
        with tempfile.TemporaryDirectory() as tmp:
            calls = []

            def fake_run(args, check):
                calls.append((args, check))
                if args[0] == "spirv-as":
                    with open(args[-1], "wb") as f:
                        f.write(b"synthetic assembled SPIR-V")
                return SimpleNamespace(returncode=0)

            result = AB.test_vertex_shader(tmp, runner=fake_run)
            self.assertTrue(os.path.isfile(result))
            self.assertEqual(len(calls), 2)
            self.assertEqual([call[0][0] for call in calls], ["spirv-as", "spirv-val"])
            self.assertTrue(all(check for _, check in calls))
            source = os.path.join(tmp, "test_vs.spvasm")
            with open(source, encoding="utf-8") as f:
                asm = f.read()
            self.assertIn("OpDecorate %o31 Location 31", asm)
            self.assertIn("%o31 = OpVariable %po Output", asm)
            self.assertIn("OpStore %o31 %zero", asm)


if __name__ == "__main__":
    unittest.main()
