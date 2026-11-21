r"""Reconcile a root signature against the translator's user-SGPR reads (stdlib only).

A shader is identified by its index.json entry (user_sgpr_reads from
tools/shader_meta.py); the candidate root signature is a hand-written JSON
file in this tool's format (see --example), never game data.

Usage:
    python -I tools/root_sig_contract.py <index.json> --rootsig <sig.json>
    python -I tools/root_sig_contract.py <index.json> --rootsig <sig.json> --cluster
    python -I tools/root_sig_contract.py --example

Exit code is 0 when every read is compatible, 1 on the first contradiction
(hard fail: a mismatch means every draw/dispatch with that pair is corrupt).
"""
import argparse
import json
import sys

# Packing order mirrors runtime/src/root_signature.{h,cpp}: constants take
# NumValues dwords, root descriptors and tables take 2 (VA lo/hi), static
# samplers take none.
KIND_TO_TYPES = {
    "const32": {"constants"},
    "ptr64_lo": {"descriptor", "table"},
    "ptr64_hi": {"descriptor", "table"},
    "ptr32_lo": {"descriptor", "table"},
    "vsharp": {"descriptor", "table"},
    "tsharp": {"descriptor", "table"},
    "ssharp": {"descriptor", "table"},
}


def pack_layout(sig):
    """Lay out root params into 32 dwords. Returns (dwords, error).

    dwords[i] = {"param": index, "sub": sub-dword, "ptype": ptype}.
    error is set (and dwords empty) on spill, cost overflow or bad input:
    refuse, never truncate.
    """
    dwords = []
    cost = 0
    params = [dict(p) for p in sig.get("params", [])]  # normalized copy; input untouched
    if len(params) > 64:
        return [], "more than 64 root parameters"
    for index, p in enumerate(params):
        ptype = p.get("type", "")
        if ptype == "constants":
            count = p.get("count", 0)
            if not isinstance(count, int) or count < 1:
                return [], f"param {index}: constants need count >= 1"
            width, cost_add = count, count
        elif ptype in ("cbv", "srv", "uav"):
            width, cost_add = 2, 2
            ptype = "descriptor"
        elif ptype == "table":
            width, cost_add = 2, 1
        else:
            return [], f"param {index}: unknown type {ptype!r}"
        cost += cost_add
        if cost > 64:
            return [], "exceeds the 64-dword D3D12 cost limit"
        if len(dwords) + width > 32:
            return [], "needs more than 32 user SGPRs (spill unsupported)"
        for sub in range(width):
            dwords.append({"param": index, "sub": sub, "ptype": ptype})
    return dwords, ""


def reconcile(shader_reads, dwords):
    """Check one shader's reads against the packed layout.

    Returns a list of contradiction strings (empty = compatible).
    """
    issues = []
    by_sgpr = {}
    for read in shader_reads:
        by_sgpr.setdefault(read["sgpr"], []).append(read["kind"])
    for sgpr in sorted(by_sgpr):
        kinds = by_sgpr[sgpr]
        if sgpr >= len(dwords):
            issues.append(f"s{sgpr} read but layout has {len(dwords)} dwords")
            continue
        slot = dwords[sgpr]
        for kind in kinds:
            allowed = KIND_TO_TYPES.get(kind)
            if allowed is None:
                issues.append(f"s{sgpr}: unknown read kind {kind!r}")
            elif slot["ptype"] not in allowed:
                issues.append(
                    f"s{sgpr} ({kind}) incompatible with param "
                    f"{slot['param']} ({slot['ptype']})")
            if kind == "ptr64_hi":
                lo = dwords[sgpr - 1] if sgpr > 0 else None
                if slot["sub"] != 1 or lo is None or lo["param"] != slot["param"] \
                        or lo["sub"] != 0:
                    issues.append(f"s{sgpr} (ptr64_hi) without paired lo dword")
    return issues


def cluster(shaders):
    """Group shaders by read-set shape. Returns [(shape, count, names)]."""
    groups = {}
    for name, entry in shaders.items():
        reads = entry.get("user_sgpr_reads", [])
        shape = tuple(sorted((r["sgpr"], r["kind"]) for r in reads))
        groups.setdefault(shape, []).append(name)
    ranked = sorted(groups.items(), key=lambda kv: -len(kv[1]))
    return [(shape, len(names), names) for shape, names in ranked]


def main():
    ap = argparse.ArgumentParser(description="Root-signature vs shader-reads reconciler.")
    ap.add_argument("index", nargs="?", help="spirv/index.json")
    ap.add_argument("--rootsig", default=None, help="hand-written root-signature JSON")
    ap.add_argument("--cluster", action="store_true", help="print read-set shape clustering")
    ap.add_argument("--example", action="store_true", help="print an empty hand-written sig template")
    a = ap.parse_args()
    if a.example:
        print(json.dumps({"version": "1.1", "flags": 0, "params": [
            {"type": "constants", "count": 4, "visibility": "all"},
            {"type": "cbv", "visibility": "vertex"},
            {"type": "table", "ranges": [{"kind": "srv", "count": 8}],
             "visibility": "pixel"}],
            "static_samplers": []}, indent=2))
        return 0
    if not a.index or not a.rootsig:
        ap.error("index.json and --rootsig are required (or --example)")
    with open(a.index, encoding="utf-8") as f:
        index = json.load(f)
    with open(a.rootsig, encoding="utf-8") as f:
        sig = json.load(f)
    dwords, error = pack_layout(sig)
    if error:
        print(f"root signature refused: {error}")
        return 1
    print(f"packed {len(dwords)} dwords from {len(sig.get('params', []))} params")
    bad = 0
    for name in sorted(index):
        reads = index[name].get("user_sgpr_reads", [])
        issues = reconcile(reads, dwords)
        if issues:
            bad += 1
            print(f"CONTRADICTION {name}:")
            for issue in issues:
                print(f"  - {issue}")
    if a.cluster:
        print("read-set shapes (shape -> shaders):")
        for shape, count, names in cluster(index)[:20]:
            print(f"  {count:6}  {shape}  e.g. {names[0]}")
    print(f"{len(index)} shaders: {len(index) - bad} compatible, {bad} contradictory")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
