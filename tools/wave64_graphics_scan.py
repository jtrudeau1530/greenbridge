"""Count cross-half-sensitive wave ops in wave64 graphics shaders over the whole corpus.

    python -I tools/wave64_graphics_scan.py <work>/shaders/dxbc/disasm [--out wave64_graphics_scan.json]

design.md's Wave64 emulation row defers graphics (VS/PS/GS) wave64: emulation covers only compute and mesh
(Vulkan graphics stages have no workgroup memory), so on a 32-wide-subgroup GPU every cross-half wave op in a
wave64 graphics shader sees only its own 32 lanes. This tool sizes that gap before any design work: it walks the
disassembly the shaders stage leaves in <work>/shaders/dxbc/disasm and reports, per stage, how many wave64
shaders contain each op whose result depends on lanes outside the executing 32-lane half:

- whole-wave mask words: `v_mbcnt_lo/hi_u32_b32` (the hi word folds in the other half's exec bits),
  `s_bcnt0/1_i32_b64`, `s_ff0/1_i32_b64` (64-bit mask results) — the translator's `wave_mask` path.
- lane-addressed exchanges: `v_readlane_b32`, `v_writelane_b32`, `v_permlane16_b32`, `v_permlanex16_b32`
  (a lane select >= 32 addresses the other half; selects are usually dynamic SGPRs, so the count is an upper
  bound) and `v_readfirstlane_b32` (ballot over the whole wave) — the `g_shuffle` / ballot paths.
- whole-wave branches on a 64-bit mask: `s_cbranch_execz/_execnz/_vccz/_vccnz` — the `scc_from_mask` /
  `branch_condition` paths. Mask *arithmetic* (`s_and_saveexec_b64` and friends) is per-lane and exact; only the
  branch taken from it splits.

Quad ops and `ds_swizzle`/`ds_bpermute` are left out on purpose: quads never cross the 32/32 half and the
translator models the permute ops per 32-lane half, matching hardware. Op set rationale: rdna2spirv.py's native
handlers (`wave_mask`, `g_vote`, `g_shuffle`, `scc_from_mask`); wave64 detection is that file's own
`WAVE64_RE`, and stages come from its `stage_of` (the carve index, not the file name), so the scan classifies
exactly what the translator classifies.

Standard library only. Nothing from the game is printed or written: counts only — names, code and data stay in
the work folder.
"""
import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
import rdna2spirv  # noqa: E402  (same directory; supplies WAVE64_RE and stage_of)

# op -> why it crosses the 32/32 half (kept short; the full rationale is in the docstring)
CROSS_HALF_OPS = {
    "v_mbcnt_lo_u32_b32": "whole-wave exec bit count",
    "v_mbcnt_hi_u32_b32": "whole-wave exec bit count",
    "s_bcnt0_i32_b64": "64-bit mask result",
    "s_bcnt1_i32_b64": "64-bit mask result",
    "s_ff0_i32_b64": "64-bit mask result",
    "s_ff1_i32_b64": "64-bit mask result",
    "v_readlane_b32": "lane select may reach >= 32",
    "v_writelane_b32": "lane select may reach >= 32",
    "v_permlane16_b32": "lane select may reach >= 32",
    "v_permlanex16_b32": "lane select may reach >= 32",
    "v_readfirstlane_b32": "ballot over the whole wave",
    "s_cbranch_execz": "branch on a whole-wave mask vote",
    "s_cbranch_execnz": "branch on a whole-wave mask vote",
    "s_cbranch_vccz": "branch on a whole-wave mask vote",
    "s_cbranch_vccnz": "branch on a whole-wave mask vote",
}
_OPS_RE = {op: re.compile(r"\b%s\b" % re.escape(op)) for op in CROSS_HALF_OPS}
GRAPHICS = ("vs", "ps", "gs")


def scan_directory(disasm):
    """Aggregate cross-half op counts over one carve disasm directory. Returns per-stage dicts; no names."""
    per_stage = {stage: {"shaders": 0, "wave64": 0, "with_cross_half": 0, "occurrences": Counter()}
                 for stage in GRAPHICS}
    other = {"shaders": 0, "wave64": 0}
    for name in sorted(os.listdir(disasm)):
        if not name.endswith(".s"):
            continue
        path = os.path.join(disasm, name)
        with open(path, encoding="utf-8", errors="replace") as f:
            text = f.read()
        stage = rdna2spirv.stage_of(name[:-2], path)
        # the translator treats mesh shaders as unconditionally wave64; markers decide for the rest
        wave64 = bool(rdna2spirv.WAVE64_RE.search(text)) or stage == "ms"
        if stage in per_stage:
            bucket = per_stage[stage]
            bucket["shaders"] += 1
            if not wave64:
                continue
            bucket["wave64"] += 1
            hits = {op: len(rx.findall(text)) for op, rx in _OPS_RE.items() if rx.search(text)}
            if hits:
                bucket["with_cross_half"] += 1
                bucket["occurrences"].update(hits)
        else:
            other["shaders"] += 1
            other["wave64"] += int(wave64)
    for stage in per_stage:
        per_stage[stage]["occurrences"] = dict(sorted(per_stage[stage]["occurrences"].items(),
                                                      key=lambda kv: -kv[1]))
    return {"per_stage": per_stage, "other_stages": other}


def main():
    ap = argparse.ArgumentParser(description="Count cross-half wave ops in wave64 graphics shaders (counts only)")
    ap.add_argument("src", help="<work>/shaders/dxbc/disasm from the shaders stage")
    ap.add_argument("--out", help="write the same aggregates to a JSON file")
    args = ap.parse_args()
    if not os.path.isdir(args.src):
        raise SystemExit("not a directory: %s" % args.src)
    report = scan_directory(args.src)
    for stage, b in report["per_stage"].items():
        occ = b["occurrences"]
        occ_line = ", ".join("%s x%d" % (op, n) for op, n in occ.items()) or "none"
        print("%s: %d shaders, %d wave64, %d with a cross-half op (%s)"
              % (stage, b["shaders"], b["wave64"], b["with_cross_half"], occ_line))
    o = report["other_stages"]
    print("other stages: %d shaders, %d wave64 (not counted)" % (o["shaders"], o["wave64"]))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=1)
        print("wrote %s" % args.out)


if __name__ == "__main__":
    main()
