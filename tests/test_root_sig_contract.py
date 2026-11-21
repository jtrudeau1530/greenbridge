"""Contract tests for tools/root_sig_contract.py (no game data, no GPU).

Run from the repository root:  python -m pytest tests/test_root_sig_contract.py
Everything is hand-built: synthetic root signatures and synthetic index
entries, so these run on every CI leg including Windows without llvm-mc.
"""
import json
import os
import subprocess
import sys

import pytest

import root_sig_contract as rsc

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOL = os.path.join(ROOT, "tools", "root_sig_contract.py")

ROOT_SIG = {
    "version": "1.1",
    "flags": 0,
    "params": [
        {"type": "constants", "count": 4, "visibility": "all"},
        {"type": "cbv", "visibility": "vertex"},
        {"type": "table", "ranges": [{"kind": "srv", "count": 8}],
         "visibility": "pixel"},
    ],
}


def test_pack_layout_order_and_provenance():
    dwords, error = rsc.pack_layout(ROOT_SIG)
    assert error == ""
    assert len(dwords) == 8
    assert [d["param"] for d in dwords] == [0, 0, 0, 0, 1, 1, 2, 2]
    assert [d["sub"] for d in dwords] == [0, 1, 2, 3, 0, 1, 0, 1]
    assert dwords[0]["ptype"] == "constants"
    assert dwords[4]["ptype"] == "descriptor"
    assert dwords[6]["ptype"] == "table"


def test_pack_layout_static_samplers_free():
    sig = dict(ROOT_SIG, static_samplers=[{"filter": 21}])
    dwords, error = rsc.pack_layout(sig)
    assert error == "" and len(dwords) == 8


def test_pack_layout_spill_refused():
    sig = {"version": "1.1", "flags": 0,
           "params": [{"type": "cbv"} for _ in range(17)]}
    dwords, error = rsc.pack_layout(sig)
    assert dwords == [] and "spill" in error


def test_pack_layout_cost_limit():
    sig = {"version": "1.1", "flags": 0,
           "params": [{"type": "constants", "count": 65}]}
    dwords, error = rsc.pack_layout(sig)
    assert dwords == [] and "64-dword" in error


def test_pack_layout_unknown_type():
    sig = {"version": "1.1", "flags": 0, "params": [{"type": "ufo"}]}
    dwords, error = rsc.pack_layout(sig)
    assert dwords == [] and "unknown type" in error


def test_reconcile_compatible():
    dwords, _ = rsc.pack_layout(ROOT_SIG)
    reads = [{"sgpr": 0, "kind": "const32"},
             {"sgpr": 4, "kind": "ptr64_lo"},
             {"sgpr": 5, "kind": "ptr64_hi"},
             {"sgpr": 6, "kind": "tsharp"}]
    assert rsc.reconcile(reads, dwords) == []


@pytest.mark.parametrize(("sgpr", "kind", "needle"), [
    (0, "ptr64_lo", "incompatible"),  # s0 is a constant dword, not a VA
    (31, "const32", "8 dwords"),  # layout ends at s7
    (4, "ptr64_hi", "paired lo"),  # s4 is sub 0 of the CBV pair
    (0, "nope", "unknown read kind"),
])
def test_reconcile_contradictions(sgpr, kind, needle):
    dwords, _ = rsc.pack_layout(ROOT_SIG)
    issues = rsc.reconcile([{"sgpr": sgpr, "kind": kind}], dwords)
    assert len(issues) == 1 and needle in issues[0]


def test_cluster_groups_shapes():
    index = {
        "a": {"user_sgpr_reads": [{"sgpr": 0, "kind": "const32"}]},
        "b": {"user_sgpr_reads": [{"sgpr": 0, "kind": "const32"}]},
        "c": {"user_sgpr_reads": [{"sgpr": 4, "kind": "ptr64_lo"}]},
    }
    ranked = rsc.cluster(index)
    assert ranked[0][1] == 2 and ranked[1][1] == 1


def test_cli_hard_fail_on_contradiction(tmp_path):
    index = {"bad": {"user_sgpr_reads": [{"sgpr": 31, "kind": "const32"}]}}
    index_path = tmp_path / "index.json"
    index_path.write_text(json.dumps(index), encoding="utf-8")
    sig_path = tmp_path / "sig.json"
    sig_path.write_text(json.dumps(ROOT_SIG), encoding="utf-8")
    r = subprocess.run(
        [sys.executable, "-I", TOOL, str(index_path),
         "--rootsig", str(sig_path)],
        capture_output=True, text=True)
    assert r.returncode == 1
    assert "CONTRADICTION bad" in r.stdout
