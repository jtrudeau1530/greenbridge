"""How efficiently the driver compiles the translated shaders, against the game's own RDNA code.

    python3 -I scripts/shader_stats.py <stats.csv> <disasm dir> [--details]

<stats.csv> comes from vk_pipeline_check <spirv dir> --stats <stats.csv> (the driver's statistics for each translated
shader); <disasm dir> holds the original <name>.s listings the translator read. For every shader in both it compares
the driver's instruction count with the original one (code growth) and its VGPRs with the highest VGPR the original
code uses, and it flags scratch memory and spills: a register file that didn't fit in registers and went to memory,
which is the expensive case (relative VGPR indexing, v_movrels/v_movreld, forces it). It prints a summary per stage and,
with --details, one line per shader. Statistic names are RADV/ACO's; another driver may name them differently.
"""
import csv
import os
import statistics
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))
import shader_ir  # noqa: E402


def original(path):
    """(instruction count, VGPRs used) of an original RDNA listing."""
    insns = shader_ir.parse_file(path)["insns"]
    top = max((a["r"] + a.get("n", 1) for i in insns for a in i["args"] if a.get("k") == "vgpr"), default=0)
    return len(insns), top


def load_stats(path):
    """{name: {"stage": .., statistic: value}} from vk_pipeline_check --stats."""
    out = defaultdict(dict)
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            entry = out[row["name"]]
            entry["stage"] = row["stage"]
            try:
                entry[row["statistic"]] = float(row["value"])
            except ValueError:
                entry[row["statistic"]] = row["value"]
    return out


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if len(args) != 2:
        sys.exit(__doc__)
    stats, disasm = load_stats(args[0]), args[1]
    rows = []
    for name, st in sorted(stats.items()):
        src = os.path.join(disasm, name + ".s")
        if not os.path.isfile(src) or "Instructions" not in st:
            continue
        n_orig, v_orig = original(src)
        rows.append({"name": name, "stage": st["stage"], "orig": n_orig, "aco": int(st["Instructions"]),
                     "growth": st["Instructions"] / max(n_orig, 1), "v_orig": v_orig, "vgprs": int(st.get("VGPRs", 0)),
                     "sgprs": int(st.get("SGPRs", 0)), "scratch": int(st.get("Scratch size", 0)),
                     "spills": int(st.get("Spilled VGPRs", 0) + st.get("Spilled SGPRs", 0)),
                     "waves": int(st.get("Subgroups per SIMD", 0))})
    if not rows:
        sys.exit("no shader is in both the statistics and the disasm folder")
    if "--details" in sys.argv:
        print(f"{'shader':26} {'stage':5} {'orig':>5} {'aco':>5} {'growth':>6} {'v_orig':>6} {'VGPRs':>5} "
              f"{'SGPRs':>5} {'scratch':>7} {'spills':>6} {'waves':>5}")
        for r in rows:
            print(f"{r['name']:26} {r['stage']:5} {r['orig']:5} {r['aco']:5} {r['growth']:6.2f} {r['v_orig']:6} "
                  f"{r['vgprs']:5} {r['sgprs']:5} {r['scratch']:7} {r['spills']:6} {r['waves']:5}")
        print()
    for stage in sorted({r["stage"] for r in rows}):
        rs = [r for r in rows if r["stage"] == stage]
        g = sorted(r["growth"] for r in rs)
        print(f"{stage}: {len(rs)} shaders, code growth median {statistics.median(g):.2f}x "
              f"(p90 {g[int(0.9 * (len(g) - 1))]:.2f}x, max {g[-1]:.2f}x); "
              f"VGPRs median {statistics.median(r['vgprs'] for r in rs):g} vs original {statistics.median(r['v_orig'] for r in rs):g}; "
              f"scratch in {sum(r['scratch'] > 0 for r in rs)}, spills in {sum(r['spills'] > 0 for r in rs)}; "
              f"waves/SIMD median {statistics.median(r['waves'] for r in rs):g}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
