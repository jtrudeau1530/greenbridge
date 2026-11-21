"""Profile the disassembled shader corpus: which RDNA instructions the translator must support.

Usage:
    python -I shader_profile.py <disasm dir> [--out profile.json]

Counts each mnemonic (number of shaders using it and total occurrences), groups them by encoding family,
classifies instructions (ALU vs memory vs branch), computes operation ratios, tracks register footprints,
and lists the dwords LLVM couldn't decode (likely Xbox-specific instructions) with how many shaders contain them.
In the JSON report, ratios.alu_to_memory is null when the corpus has ALU instructions and no memory instructions.
"""
import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict

FAMILIES = [("s_buffer_", "smem"), ("s_load", "smem"), ("s_", "scalar"), ("v_", "vector"), ("ds_", "lds"),
            ("buffer_", "buffer"), ("tbuffer_", "buffer"), ("image_", "image"), ("global_", "global"),
            ("flat_", "flat"), ("scratch_", "scratch"), ("exp", "export")]

VGPR_RANGE = re.compile(r"\bv\[(\d+):(\d+)\]")
VGPR_SINGLE = re.compile(r"\bv(\d+)\b")
SGPR_RANGE = re.compile(r"\bs\[(\d+):(\d+)\]")
SGPR_SINGLE = re.compile(r"\bs(\d+)\b")
SPECIAL_REG = re.compile(r"\b(vcc|vcc_lo|vcc_hi|exec|exec_lo|exec_hi|m0|scc|flat_scratch|flat_scratch_lo|flat_scratch_hi)\b")


def family(m):
    for p, f in FAMILIES:
        if m.startswith(p):
            return f
    return "other"


def classify_instruction(m: str) -> str:
    """Classify an RDNA mnemonic into 'alu', 'memory', 'branch', 'export', or 'other'."""
    if not m:
        return "other"
    m = m.lower()
    if m.startswith("exp"):
        return "export"
    if m == "s_branch" or m.startswith("s_cbranch_") or m in ("s_setpc_b64", "s_swappc_b64", "s_rfe_b64", "s_call_b64"):
        return "branch"
    if m.startswith(("s_buffer_", "s_load", "s_store_", "s_dcache_", "buffer_", "tbuffer_", "image_", "global_", "flat_", "scratch_", "ds_")):
        return "memory"
    if m.startswith("v_"):
        return "alu"
    if m.startswith("s_"):
        if m in ("s_barrier", "s_nop", "s_endpgm", "s_trap", "s_sleep", "s_icache_inv") or \
           m.startswith(("s_waitcnt", "s_sendmsg", "s_setreg_", "s_getreg_", "s_setprio")):
            return "other"
        return "alu"
    return "other"


def is_alu(m: str) -> bool:
    return classify_instruction(m) == "alu"


def is_memory(m: str) -> bool:
    return classify_instruction(m) == "memory"


def is_branch(m: str) -> bool:
    return classify_instruction(m) == "branch"


def count_categories(mnemonics) -> dict:
    """Count instruction occurrences by high-level category (alu, memory, branch, export, other)."""
    counts = Counter()
    if isinstance(mnemonics, dict):
        for m, count in mnemonics.items():
            counts[classify_instruction(m)] += count
    else:
        for m in mnemonics:
            counts[classify_instruction(m)] += 1
    total = sum(counts.values())
    return {
        "alu": counts["alu"],
        "memory": counts["memory"],
        "branch": counts["branch"],
        "export": counts["export"],
        "other": counts["other"],
        "total": total,
    }


def compute_ratios(counts: dict) -> dict:
    """Calculate ALU vs Memory vs Branch ratios relative to total instructions,
    and relative ratios like alu_to_memory."""
    total = counts.get("total", 0)
    if total <= 0:
        total = sum(counts.get(k, 0) for k in ("alu", "memory", "branch", "export", "other"))
    alu = counts.get("alu", 0)
    mem = counts.get("memory", 0)
    branch = counts.get("branch", 0)
    return {
        "alu_ratio": round(alu / total, 6) if total > 0 else 0.0,
        "memory_ratio": round(mem / total, 6) if total > 0 else 0.0,
        "branch_ratio": round(branch / total, 6) if total > 0 else 0.0,
        "alu_to_memory": round(alu / mem, 6) if mem > 0 else (float("inf") if alu > 0 else 0.0),
    }


def parse_registers(line: str) -> dict:
    """Extract all VGPRs, SGPRs, and special registers referenced in a single disassembly line."""
    code, _, _ = line.partition(";")
    vgprs = set()
    for lo_s, hi_s in VGPR_RANGE.findall(code):
        lo, hi = int(lo_s), int(hi_s)
        vgprs.update(range(lo, hi + 1))
    code_no_vr = VGPR_RANGE.sub("", code)
    for v_s in VGPR_SINGLE.findall(code_no_vr):
        vgprs.add(int(v_s))

    sgprs = set()
    for lo_s, hi_s in SGPR_RANGE.findall(code):
        lo, hi = int(lo_s), int(hi_s)
        sgprs.update(range(lo, hi + 1))
    code_no_sr = SGPR_RANGE.sub("", code)
    for s_s in SGPR_SINGLE.findall(code_no_sr):
        sgprs.add(int(s_s))

    specials = set(SPECIAL_REG.findall(code))
    return {"vgprs": vgprs, "sgprs": sgprs, "special": specials}


def register_footprint(lines_or_text) -> dict:
    """Calculate the register footprint (max index, allocation count, and unique sets)
    across disassembly lines or text."""
    if isinstance(lines_or_text, str):
        lines = lines_or_text.splitlines()
    else:
        lines = lines_or_text

    all_vgprs = set()
    all_sgprs = set()
    all_specials = set()

    for line in lines:
        if not line.startswith("\t"):
            continue
        regs = parse_registers(line)
        all_vgprs.update(regs["vgprs"])
        all_sgprs.update(regs["sgprs"])
        all_specials.update(regs["special"])

    max_vgpr = max(all_vgprs) if all_vgprs else -1
    max_sgpr = max(all_sgprs) if all_sgprs else -1

    return {
        "max_vgpr": max_vgpr,
        "max_sgpr": max_sgpr,
        "vgpr_count": max_vgpr + 1 if max_vgpr >= 0 else 0,
        "sgpr_count": max_sgpr + 1 if max_sgpr >= 0 else 0,
        "unique_vgprs": len(all_vgprs),
        "unique_sgprs": len(all_sgprs),
        "vgprs": sorted(all_vgprs),
        "sgprs": sorted(all_sgprs),
        "special": sorted(all_specials),
    }


def profile_shader(text: str) -> dict:
    """Profile a single shader's disassembly: mnemonics, categories, ratios, and register footprint."""
    lines = text.splitlines()
    mnemonics = Counter()
    for line in lines:
        if not line.startswith("\t"):
            continue
        code = line.strip().partition(";")[0].strip()
        if not code:
            continue
        m = code.split()[0]
        mnemonics[m] += 1

    cats = count_categories(mnemonics)
    ratios = compute_ratios(cats)
    reg_fp = register_footprint(lines)

    return {
        "total_instructions": cats["total"],
        "mnemonics": dict(mnemonics),
        "categories": cats,
        "ratios": ratios,
        "registers": reg_fp,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("--out")
    a = ap.parse_args()
    occ, users = Counter(), Counter()
    invalid_users = Counter()
    per_stage = defaultdict(Counter)
    n = 0
    for name in os.listdir(a.src):
        if not name.endswith(".s"):
            continue
        n += 1
        stage = name[:2].lower() if name[2:3] == "_" else "?"
        seen = set()
        with open(os.path.join(a.src, name), encoding="utf-8", errors="replace") as f:
            text = f.read()
        for line in text.splitlines():
            if not line.startswith("\t"):
                continue
            m = line.split()[0]
            occ[m] += 1
            seen.add(m)
        for m in seen:
            users[m] += 1
            per_stage[stage][m] += 1
        for w in set(re.findall(r"<stdin>:\d+:\d+: warning: invalid instruction encoding\n; (0x[0-9a-f]{2}(?: 0x[0-9a-f]{2}){3})", text)):
            invalid_users[w] += 1
    fam = defaultdict(lambda: [0, 0])
    for m, c in occ.items():
        fam[family(m)][0] += 1
        fam[family(m)][1] += c
    cats = count_categories(occ)
    ratios = compute_ratios(cats)
    if ratios["alu_to_memory"] == float("inf"):
        ratios["alu_to_memory"] = None
    out = {"shaders": n, "mnemonics": len(occ),
           "families": {k: {"mnemonics": v[0], "occurrences": v[1]} for k, v in sorted(fam.items())},
           "categories": cats,
           "ratios": ratios,
           "by_mnemonic": [{"m": m, "shaders": users[m], "count": occ[m]} for m, _ in users.most_common()]}
    path = a.out or os.path.join(a.src, "..", "profile.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, allow_nan=False)
    print(f"{n} shaders, {len(occ)} distinct mnemonics -> {os.path.abspath(path)}")
    for k, v in sorted(fam.items(), key=lambda kv: -kv[1][1]):
        print(f"  {k:8} {v[0]:4} mnemonics {v[1]:>12,} uses")
    print(f"  ALU: {cats['alu']} ({ratios['alu_ratio'] * 100:.1f}%)  Memory: {cats['memory']} ({ratios['memory_ratio'] * 100:.1f}%)  Branch: {cats['branch']} ({ratios['branch_ratio'] * 100:.1f}%)")
    cum, total = 0, sum(occ.values())
    for i, (m, c) in enumerate(occ.most_common(), 1):
        cum += c
        if total > 0 and cum / total >= 0.99:
            print(f"  {i} mnemonics cover 99% of all instructions")
            break
    print("  most widespread:", ", ".join(f"{m} ({users[m]})" for m, _ in users.most_common(12)))


if __name__ == "__main__":
    sys.exit(main())
