"""tools/user_sgpr_census.py on hand-written index.json data (no game files, no GPU)."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import user_sgpr_census as usc  # noqa: E402


def reads(*pairs):
    return [{"sgpr": s, "kind": k} for s, k in pairs]


INDEX = {
    "a.ps": {"stage": "ps", "user_sgpr_reads": reads((8, "ptr32_lo"), (9, "const32"))},
    "b.ps": {"stage": "ps", "user_sgpr_reads": reads((1, "const32"), (8, "ptr32_lo"), (11, "ptr32_lo"))},
    "c.ps": {"stage": "ps", "user_sgpr_reads": reads((9, "ptr32_lo"))},
    "d.cs": {"stage": "cs", "user_sgpr_reads": reads((4, "ptr32_lo"), (5, "const32"))},
    "e.gs": {"stage": "gs", "user_sgpr_reads": reads((2, "const32"), (8, "ptr64_lo"), (9, "ptr64_hi"))},
    "f.vs": {"stage": "vs", "user_sgpr_reads": []},
}


def test_lowest_pointer_per_stage():
    s = usc.census(INDEX)
    assert s["ps"]["shaders"] == 3
    assert s["ps"]["lowest_pointer_sgpr"] == {"s8": 2, "s9": 1}
    assert s["ps"]["most_common_lowest_pointer_sgpr"] == 8
    assert s["ps"]["lowest_pointer_kind"] == {"ptr32_lo": 3}
    assert s["cs"]["most_common_lowest_pointer_sgpr"] == 4
    assert s["gs"]["lowest_pointer_kind"] == {"ptr64_lo": 1}
    assert s["vs"]["most_common_lowest_pointer_sgpr"] is None


def test_const_reads_are_not_pointers():
    s = usc.census({"x": {"stage": "ps", "user_sgpr_reads": reads((0, "const32"), (3, "const32"))}})
    assert s["ps"]["lowest_pointer_sgpr"] == {}
    assert s["ps"]["shaders_reading_kind"] == {"const32": 1}


def model_two_dword_tables_from_s0():
    """Shaders packed as runtime/src/root_signature.h does today: tables are 64-bit VAs from s0."""
    return {
        "t0": {"stage": "ps", "user_sgpr_reads": reads((0, "ptr64_lo"), (1, "ptr64_hi"), (2, "ptr64_lo"), (3, "ptr64_hi"))},
        "t1": {"stage": "ps", "user_sgpr_reads": reads((0, "ptr64_lo"), (1, "ptr64_hi"), (2, "const32"))},
    }


def model_one_dword_tables_from_s8():
    """Shaders packed with a stage base (PS s8) and one-dword tables: 32-bit VAs, high half implicit zero."""
    return {
        "t0": {"stage": "ps", "user_sgpr_reads": reads((8, "ptr32_lo"), (9, "ptr32_lo"))},
        "t1": {"stage": "ps", "user_sgpr_reads": reads((1, "const32"), (8, "ptr32_lo"), (9, "const32"), (10, "ptr32_lo"))},
    }


def test_the_two_packing_models_are_told_apart():
    a = usc.census(model_two_dword_tables_from_s0())["ps"]
    b = usc.census(model_one_dword_tables_from_s8(), base={"ps": 8})["ps"]
    # two dwords per table: pointers two SGPRs apart, never a 32-bit pointer, high halves read
    assert a["pointer_gap"] == {2: 1}
    assert a["after_ptr32"] == {}
    assert a["most_common_lowest_pointer_sgpr"] == 0
    assert a["shaders_reading_kind"].get("ptr64_hi") == 2
    # one dword per table from s8: pointers adjacent, no high halves, a system read below the base
    assert b["pointer_gap"] == {1: 1, 2: 1}
    # t0: s8 -> s9 is a pointer, s9 -> s10 not read; t1: s8 -> s9 const32, s10 -> s11 not read
    assert b["after_ptr32"] == {"const32": 1, "not read": 2, "pointer": 1}
    assert b["most_common_lowest_pointer_sgpr"] == 8
    assert "ptr64_hi" not in b["shaders_reading_kind"]
    assert b["below_base"] == {"s1:const32": 1}


@pytest.mark.parametrize("output_kind", ["same-path", "relative-alias", "hard-link"])
def test_json_report_refuses_to_overwrite_index(tmp_path, capsys, output_kind):
    index = tmp_path / "index.json"
    original = json.dumps(INDEX).encode("utf-8")
    index.write_bytes(original)
    output = index
    if output_kind == "relative-alias":
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        output = subdir / ".." / index.name
    elif output_kind == "hard-link":
        output = tmp_path / "alias.json"
        try:
            output.hardlink_to(index)
        except OSError as err:
            pytest.skip(f"hard links unavailable: {err}")

    with pytest.raises(SystemExit) as exc:
        usc.main([str(index), "--json", str(output)])

    assert exc.value.code == 2
    assert "--json must not overwrite the input index" in capsys.readouterr().err
    assert index.read_bytes() == original


def test_json_report_accepts_a_separate_output(tmp_path):
    index = tmp_path / "index.json"
    output = tmp_path / "summary.json"
    index.write_text(json.dumps(INDEX), encoding="utf-8")

    assert usc.main([str(index), "--json", str(output)]) == 0
    assert json.loads(output.read_text(encoding="utf-8"))["ps"]["lowest_pointer_sgpr"] == {"s8": 2, "s9": 1}
    assert json.loads(index.read_text(encoding="utf-8")) == INDEX


def test_expect_passes_and_fails(tmp_path, capsys):
    p = tmp_path / "index.json"
    p.write_text(json.dumps(INDEX), encoding="utf-8")
    assert usc.main([str(p), "--expect", "ps=8", "cs=4"]) == 0
    assert usc.main([str(p), "--expect", "ps=0"]) == 1
    assert "expected s0, got s8" in capsys.readouterr().err


def test_cli_rejects_malformed_stage_assignment():
    with tempfile.TemporaryDirectory() as tmp:
        index = Path(tmp, "index.json")
        report = Path(tmp, "report.json")
        index.write_text(json.dumps(INDEX), encoding="utf-8")
        tool = os.path.join(ROOT, "tools", "user_sgpr_census.py")
        for option in ("--base", "--expect"):
            for assignment in ("ps", "ps=bad", "=8", "ps=-1"):
                result = subprocess.run(
                    [sys.executable, "-I", tool, str(index), "--json", str(report), option, assignment],
                    capture_output=True, text=True,
                )
                assert result.returncode == 2, result.stderr
                assert "expected STAGE=SGPR" in result.stderr
                assert "Traceback" not in result.stderr
                assert not report.exists()
