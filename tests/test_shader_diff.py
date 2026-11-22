"""Tests for tools/shader_diff.py (no game data, no GPU, no llvm-mc).

Run from the repository root:
    python -m pytest tests/test_shader_diff.py -v
"""
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import shader_diff as sd


def test_program_generation_is_deterministic():
    import random
    rng1, rng2 = random.Random(99), random.Random(99)
    assert (sd.generate_program(rng1, 6)[0] ==
            sd.generate_program(rng2, 6)[0])


def test_compare_pair_flags_known_bad():
    bad = sd.compare_pair([1, 2, 3, 4], [1, 2, 3, 5])
    assert bad is not None and "lane word 3" in bad
    assert sd.compare_pair([1, 2, 3, 4], [1, 2, 3, 4]) is None


@pytest.mark.parametrize("op", ["v_add_f32", "v_add_f32_e32", "v_add_f32_e64"])
def test_compare_pair_both_nan_matches(op):
    qnan = 0x7FC00001
    writers = {reg: (op, 1) for reg in sd.EXPORT_REGS}
    assert sd.compare_pair([qnan] * 4, [0x7FC00002] * 4, writers=writers) is None
    bad = sd.compare_pair([qnan] * 4, [0x3F800000] * 4, writers=writers)
    assert bad is not None and "only one side is NaN" in bad
    assert f"last writer {op} at body line 1" in bad


@pytest.mark.parametrize("changed_word, replacement, width, expected", [
    (1, 0x7FC00000, 4, ("lane word 1", "0xffffffff", "0x7fc00000", "v_mov_b32 at body line 4")),
    (2, 3, 4, ("lane word 2", "v_mov_b32 at body line 5")),
    (None, None, 3, ("shape mismatch: cpu has 4 words, spv has 3",)),
])
def test_run_one_reports_the_first_actual_mismatch(monkeypatch, changed_word, replacement, width, expected):
    lines = ["v_mov_b32 v8, 0x7f800000", "v_mov_b32 v12, 0xff800000",
             "v_add_f32 v8, v8, v12", "v_mov_b32 v9, 0xffffffff",
             "v_mov_b32 v10, 1", "v_mov_b32 v11, 2"]
    writers = {8: ("v_add_f32", 3), 9: ("v_mov_b32", 4), 10: ("v_mov_b32", 5), 11: ("v_mov_b32", 6)}
    assert sd.run_one("ps_mixed_outputs", lines, writers) == ("match", "")
    run_fragment = sd.run_fragment

    def mutate_export(spvasm):
        outputs = run_fragment(spvasm)
        words = list(outputs["%out_mrt0"])
        words[0] = 0x7FC00001  # Different payload for a floating-point NaN is allowed.
        if changed_word is not None:
            words[changed_word] = replacement
        outputs["%out_mrt0"] = words[:width]
        return outputs

    monkeypatch.setattr(sd, "run_fragment", mutate_export)
    status, detail = sd.run_one("ps_mixed_outputs", lines, writers)
    assert status == "mismatch"
    for fragment in expected:
        assert fragment in detail
    assert "v_add_f32 at body line 3" not in detail


def test_unsupported_op_is_a_counted_skip():
    status, detail = sd.run_one(
        "ps_probe_skip", ["v_interp_p1_f32 v8, v9, v10"],
        {8: ("v_interp_p1_f32", 1)})
    assert status == "skip"
    assert detail


def test_small_fixed_seed_run_is_clean():
    counts, first_failure = sd.run_suite(5, seed=20261121, max_ops=4)
    assert counts["mismatch"] == 0, first_failure
    assert counts["harness-error"] == 0, first_failure
    assert counts["pass"] == 5, counts


def test_cli_reports_the_number_of_programs_run(capsys):
    assert sd.main(["--programs", "5", "--limit", "2"]) == 0
    assert capsys.readouterr().out.startswith("2 programs (seed 1): 2 pass,")


def test_self_test_passes():
    assert sd.self_test() is True
