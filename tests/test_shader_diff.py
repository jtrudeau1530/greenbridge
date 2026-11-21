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


def test_compare_pair_both_nan_matches():
    qnan = 0x7FC00001
    assert sd.compare_pair([qnan] * 4, [0x7FC00002] * 4) is None
    bad = sd.compare_pair([qnan] * 4, [0x3F800000] * 4)
    assert bad is not None and "NaN" in bad


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


def test_self_test_passes():
    assert sd.self_test() is True
