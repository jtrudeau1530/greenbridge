r"""Census of user-SGPR reads per shader stage, from the translator's index.json (stdlib only).

Answers, from data the pipeline already produces, where the console puts root arguments and how wide a
descriptor-table pointer is (docs/d3d12_vulkan.md §3.5, the "least verifiable part"):

  * per stage: how many shaders read each user-SGPR kind (const32, ptr32_lo, ptr64_lo/hi, v#/t#/s#);
  * per stage: histogram of the LOWEST user SGPR read as a pointer (ptr32_lo / ptr64_lo), i.e. where the
    first pointer-sized root argument sits;
  * whether pointers are 32-bit (one SGPR, high half implicit zero) or 64-bit pairs;
  * the distance between consecutive pointer SGPRs in one shader, and what the SGPR right after a 32-bit pointer
    holds. These separate the two packing models: one dword per table puts pointers at distance 1 with no high
    halves; two dwords per table put a ptr64_hi right after every table pointer (distance 2 between tables);
  * SGPRs read below a stage base given with --base (reserved / system block candidates).

What it does NOT measure: logical root-parameter indices. It sees physical SGPRs only; mapping them to root
parameters needs a root signature, which the shaders do not carry.

It reads only spirv/index.json (user_sgpr_reads and stage from tools/shader_meta.py): no game bytes.

Usage:
    python -I tools/user_sgpr_census.py <spirv/index.json> [--json OUT]
    python -I tools/user_sgpr_census.py <spirv/index.json> --expect ps=8 vs=8 cs=4 ms=16
        (exit 1 if, for a listed stage, the most common lowest-pointer SGPR is not the expected one)
"""
import argparse
import json
import os
import sys
from collections import Counter, defaultdict

POINTER_KINDS = ("ptr32_lo", "ptr64_lo")


def census(index, base=None):
    """index: {name: {"stage": ..., "user_sgpr_reads": [{"sgpr": n, "kind": k}, ...]}} -> summary dict.
    base: optional {stage: first root-argument SGPR}; reads below it are reported per (sgpr, kind)."""
    base = base or {}
    below = defaultdict(Counter)
    shaders = Counter()
    kinds = defaultdict(Counter)            # stage -> kind -> shaders reading it
    lowest_pointer = defaultdict(Counter)   # stage -> sgpr -> shaders
    lowest_pointer_kind = defaultdict(Counter)
    pointer_gap = defaultdict(Counter)      # stage -> distance between consecutive pointer SGPRs -> count
    after_ptr32 = defaultdict(Counter)      # stage -> what the SGPR after a ptr32_lo is read as
    for entry in index.values():
        stage = entry.get("stage", "?")
        reads = entry.get("user_sgpr_reads") or []
        shaders[stage] += 1
        for k in {r.get("kind") for r in reads}:
            kinds[stage][k] += 1
        ptrs = sorted((r["sgpr"], r["kind"]) for r in reads if r.get("kind") in POINTER_KINDS and "sgpr" in r)
        at = defaultdict(set)
        for r in reads:
            if "sgpr" in r:
                at[r["sgpr"]].add(r.get("kind"))
                if stage in base and r["sgpr"] < base[stage]:
                    below[stage][f"s{r['sgpr']}:{r.get('kind')}"] += 1
        regs = [p for p, _ in ptrs]
        for x, y in zip(regs, regs[1:]):
            pointer_gap[stage][y - x] += 1
        for p, k in ptrs:
            if k == "ptr32_lo":
                nxt = at.get(p + 1, set())
                after_ptr32[stage]["ptr64_hi" if "ptr64_hi" in nxt else "pointer" if nxt & set(POINTER_KINDS)
                                   else "const32" if "const32" in nxt else "not read" if not nxt else "other"] += 1
        if ptrs:
            sgpr, kind = ptrs[0]
            lowest_pointer[stage][sgpr] += 1
            lowest_pointer_kind[stage][kind] += 1
    out = {}
    for stage in sorted(shaders):
        lp = lowest_pointer[stage]
        out[stage] = {
            "shaders": shaders[stage],
            "shaders_reading_kind": dict(sorted(kinds[stage].items())),
            "lowest_pointer_sgpr": {f"s{k}": v for k, v in sorted(lp.items())},
            "lowest_pointer_kind": dict(sorted(lowest_pointer_kind[stage].items())),
            "most_common_lowest_pointer_sgpr": lp.most_common(1)[0][0] if lp else None,
            "pointer_gap": dict(sorted(pointer_gap[stage].items())),
            "after_ptr32": dict(sorted(after_ptr32[stage].items())),
        }
        if stage in base:
            out[stage]["below_base"] = dict(sorted(below[stage].items()))
    return out


def stage_sgpr(text):
    stage, _, value = text.partition("=")
    message = f"expected STAGE=SGPR with a nonnegative SGPR: {text!r}"
    try:
        sgpr = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(message) from None
    if not stage or sgpr < 0:
        raise argparse.ArgumentTypeError(message)
    return stage, sgpr


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("index", help="spirv/index.json written by tools/rdna2spirv.py")
    ap.add_argument("--json", help="also write the summary here")
    ap.add_argument("--expect", nargs="*", type=stage_sgpr, default=[], metavar="STAGE=SGPR",
                    help="fail if the most common lowest-pointer SGPR of STAGE is not SGPR")
    ap.add_argument("--base", nargs="*", type=stage_sgpr, default=[], metavar="STAGE=SGPR",
                    help="report the reads below this first root-argument SGPR per stage")
    a = ap.parse_args(argv)
    if a.json:
        try:
            if os.path.samefile(a.index, a.json):
                ap.error("--json must not overwrite the input index")
        except FileNotFoundError:
            pass
    with open(a.index, encoding="utf-8") as f:
        index = json.load(f)
    summary = census(index, dict(a.base))
    for stage, s in summary.items():
        top = s["most_common_lowest_pointer_sgpr"]
        print(f"{stage}: {s['shaders']} shaders; lowest pointer SGPR "
              f"{('s%d' % top) if top is not None else '-'} {s['lowest_pointer_sgpr']}; kinds {s['lowest_pointer_kind']}; "
              f"pointer gaps {s['pointer_gap']}; after a ptr32 {s['after_ptr32']}"
              + (f"; below base {s['below_base']}" if "below_base" in s else ""))
    if a.json:
        with open(a.json, "w", encoding="utf-8", newline="\n") as f:
            json.dump(summary, f, indent=1, sort_keys=True)
    bad = []
    for stage, sgpr in a.expect:
        got = summary.get(stage, {}).get("most_common_lowest_pointer_sgpr")
        if got != sgpr:
            bad.append(f"{stage}: expected s{sgpr}, got {'s%d' % got if got is not None else 'none'}")
    for b in bad:
        print("MISMATCH", b, file=sys.stderr)
    return 1 if bad else 0


if __name__ == "__main__":
    try:  # a redirected stdout is strict ANSI code page under -I; write '?' for what it cannot encode
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    sys.exit(main())
