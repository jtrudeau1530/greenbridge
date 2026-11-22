"""Scan the shader corpus for features the D3D12 -> Vulkan layer may need (docs/d3d12_vulkan.md §8.2 #10 and #14).

    python -I tools/feature_scan.py <work>/shaders [--out feature_scan.json]

Reads what the shaders stage of make_game.py leaves in <work>/shaders, and prints counts only:

- dxbc/index.json and the .dxbc files: the DXIL program kind of every shader (hull, domain and amplification
  stages would show up here), the SFI0 feature flags (D3D_SHADER_REQUIRES_* from the Windows SDK's d3d12shader.h,
  among them WAVE_MMA and SHADING_RATE), and the output-signature (OSG1) stream index of every element: stream
  output to a stream other than 0 needs a geometry shader writing that stream.
- dxbc/disasm/*.s (shader_disasm.py): the native RDNA2 instructions that mean wave MMA (v_mfma*, v_wmma*,
  v_smfmac*) or GDS / ordered-count traffic (the hardware path NGG stream output uses on RDNA2).
- spirv/index.json (rdna2spirv.py): uses_code_base per stage. A graphics PSO can only need two code bases
  (§5.2) if two of its stages use one; this gives the per-stage counts. Which stages are paired in one PSO is in
  the executable, not in shaders.rpf.

Standard library only. Nothing from the game is printed or written: names, code and data stay in the work folder.
"""
import argparse
import json
import os
import re
import struct
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from shader_carve import parts as dxbc_part_ranges  # noqa: E402

# d3d12shader.h (Windows SDK 10.0.26100.0), D3D_SHADER_REQUIRES_*: the SFI0 part is this 64-bit mask.
SFI0_FLAGS = {
    0x1: "DOUBLES", 0x2: "EARLY_DEPTH_STENCIL", 0x4: "UAVS_AT_EVERY_STAGE", 0x8: "64_UAVS",
    0x10: "MINIMUM_PRECISION", 0x20: "11_1_DOUBLE_EXTENSIONS", 0x40: "11_1_SHADER_EXTENSIONS",
    0x80: "LEVEL_9_COMPARISON_FILTERING", 0x100: "TILED_RESOURCES", 0x200: "STENCIL_REF", 0x400: "INNER_COVERAGE",
    0x800: "TYPED_UAV_LOAD_ADDITIONAL_FORMATS", 0x1000: "ROVS",
    0x2000: "VIEWPORT_AND_RT_ARRAY_INDEX_FROM_ANY_SHADER_FEEDING_RASTERIZER", 0x4000: "WAVE_OPS",
    0x8000: "INT64_OPS", 0x10000: "VIEW_ID", 0x20000: "BARYCENTRICS", 0x40000: "NATIVE_16BIT_OPS",
    0x80000: "SHADING_RATE", 0x100000: "RAYTRACING_TIER_1_1", 0x200000: "SAMPLER_FEEDBACK",
    0x400000: "ATOMIC_INT64_ON_TYPED_RESOURCE", 0x800000: "ATOMIC_INT64_ON_GROUP_SHARED",
    0x1000000: "DERIVATIVES_IN_MESH_AND_AMPLIFICATION_SHADERS", 0x2000000: "RESOURCE_DESCRIPTOR_HEAP_INDEXING",
    0x4000000: "SAMPLER_DESCRIPTOR_HEAP_INDEXING", 0x8000000: "WAVE_MMA",
    0x10000000: "ATOMIC_INT64_ON_DESCRIPTOR_HEAP_RESOURCE", 0x20000000: "ADVANCED_TEXTURE_OPS",
    0x40000000: "WRITEABLE_MSAA_TEXTURES", 0x80000000: "SAMPLE_CMP_GRADIENT_OR_BIAS",
    0x100000000: "EXTENDED_COMMAND_INFO",
}
# The ones §8.2 #10 asks about; always printed, also when zero.
WATCHED = ("WAVE_MMA", "SHADING_RATE", "VIEW_ID", "ROVS", "STENCIL_REF", "INNER_COVERAGE", "SAMPLER_FEEDBACK",
           "RAYTRACING_TIER_1_1")
ALL_STAGES = ("vs", "ps", "gs", "hs", "ds", "cs", "ms", "as", "lib")

MMA = re.compile(r"^\s*(v_mfma\w*|v_wmma\w*|v_smfmac\w*)\b")
# Ordered count, GWS and the GS-register adds are the GDS-side instructions NGG stream output relies on. ds_append /
# ds_consume are included as the other counter-style DS ops (LDS unless they carry `gds`, which GDS below catches).
ORDERED = re.compile(r"^\s*(ds_ordered_count|ds_gws_\w+|ds_append|ds_consume|ds_add_gs_reg\w*|ds_sub_gs_reg\w*)\b")
GDS = re.compile(r"^\s*ds_\w+\b[^;]*\bgds\b")
REG_SINGLE = re.compile(r"^([sv])(\d+)$")
REG_RANGE = re.compile(r"^([sv])\[(\d+):(\d+)\]$")
CONST_MOV_OPS = frozenset(("s_mov_b32", "s_mov_b32_e32", "v_mov_b32", "v_mov_b32_e32", "v_mov_b32_e64"))


def parts(d: bytes) -> dict:
    """DXBC part fourcc -> payload."""
    if not d.startswith(b"DXBC"):
        raise ValueError("missing DXBC magic")
    return {tag: d[off:off + size] for tag, (off, size) in dxbc_part_ranges(d).items()}


def streams(sig: bytes) -> set:
    """Stream index of every element of an ISG1/OSG1 signature (32-byte elements, stream first).

    A part shorter than its declared element table, or with its first element inside the 8-byte
    count/offset header, carries no usable stream information: report no streams rather than let
    struct.error abort the whole count-only scan. Well-formed signatures are read as before.
    """
    if len(sig) < 8:
        return set()
    count, first = struct.unpack_from("<II", sig, 0)
    if first < 8 or first + 32 * count > len(sig):
        return set()
    return {struct.unpack_from("<I", sig, first + 32 * i)[0] for i in range(count)}


def scan_dxbc(dxbc_dir: str, index: list) -> dict:
    stages = Counter()
    flags = defaultdict(Counter)  # flag -> stage -> shaders
    gs_streams = Counter()
    non_zero_stream = Counter()  # stage -> shaders with an output element on a stream other than 0
    missing = 0
    for e in index:
        stage = e.get("stage") or "?"
        stages[stage] += 1
        path = os.path.join(dxbc_dir, e["file"])
        try:
            with open(path, "rb") as f:
                p = parts(f.read())
        except (OSError, ValueError):
            missing += 1
            continue
        if len(p.get("SFI0", b"")) >= 8:
            mask = struct.unpack_from("<Q", p["SFI0"], 0)[0]
            for bit, name in SFI0_FLAGS.items():
                if mask & bit:
                    flags[name][stage] += 1
        out = streams(p.get("OSG1", b""))
        if stage == "gs":
            gs_streams[len(out)] += 1
        if out - {0}:
            non_zero_stream[stage] += 1
    return {"stages": {s: stages.get(s, 0) for s in ALL_STAGES} | {k: v for k, v in stages.items() if k not in ALL_STAGES},
            "sfi0": {n: dict(flags[n]) for n in sorted(flags)},
            "gs_output_stream_count": dict(gs_streams), "output_on_stream_other_than_0": dict(non_zero_stream),
            "unreadable": missing}


def getpc_for_data(ops: list) -> bool:
    """s_getpc_b64 not followed within three instructions by s_setpc_b64: the address is used to read data
    embedded in the code (a code base), not for a far branch (getpc; add/sub; addc/subb; setpc)."""
    return any(op == "s_getpc_b64" and "s_setpc_b64" not in ops[i + 1:i + 4] for i, op in enumerate(ops))


def _parse_imm(tok: str):
    """Parse a decimal or 0x-hex integer immediate, or return None."""
    try:
        return int(tok, 0) & 0xFFFFFFFF
    except ValueError:
        return None


def _written_regs(tok: str) -> list:
    """Return individual sN/vN register names covered by a destination operand token."""
    m = REG_SINGLE.match(tok)
    if m:
        return [tok]
    m = REG_RANGE.match(tok)
    if m:
        prefix, lo, hi = m.group(1), int(m.group(2)), int(m.group(3))
        if lo <= hi <= lo + 32:
            return [f"{prefix}{i}" for i in range(lo, hi + 1)]
    return []


def _update_const_regs(const_reg: dict, op: str, operands: list) -> None:
    """Track single-register immediate loads and invalidate any destination register written."""
    if not operands:
        return
    dst = operands[0].split()[0] if operands[0].split() else ""
    if op in CONST_MOV_OPS and len(operands) >= 2 and REG_SINGLE.match(dst):
        src_tok = operands[1].split()[0] if operands[1].split() else ""
        imm = _parse_imm(src_tok)
        if imm is not None:
            const_reg[dst] = imm
            return
    for r in _written_regs(dst):
        const_reg.pop(r, None)


def scan_disasm(disasm_dir: str, stage_of: dict) -> dict:
    mma, ordered, gds, code_data = Counter(), Counter(), Counter(), Counter()
    mma_ops, ordered_ops = Counter(), Counter()
    vperm_shaders = Counter()
    vperm_insns = vperm_imm = vperm_creg = vperm_unres = 0
    vperm_sext_shaders = vperm_sext_insns = 0
    vperm_classes = {"0_7": 0, "8_11": 0, "12": 0, "13_plus": 0}
    files = 0
    for fn in os.listdir(disasm_dir):
        if not fn.endswith(".s"):
            continue
        files += 1
        stage = stage_of.get(fn[:-2], "?")
        seen_mma = seen_ord = seen_gds = seen_vperm = seen_vperm_sext = False
        ops = []
        const_reg = {}
        with open(os.path.join(disasm_dir, fn), encoding="utf-8", errors="replace") as f:
            for line in f:
                code = line.split(";", 1)[0].strip()
                head = code.split(None, 1)
                if head and not head[0].startswith("."):
                    op = head[0]
                    ops.append(op)
                    operands = [x.strip() for x in head[1].split(",")] if len(head) > 1 else []
                    if op in ("v_perm_b32", "v_perm_b32_e64") and len(operands) >= 4:
                        seen_vperm = True
                        vperm_insns += 1
                        sel_tok = operands[3].split()[0] if operands[3].split() else ""
                        sel_val = _parse_imm(sel_tok)
                        if sel_val is not None:
                            vperm_imm += 1
                        elif sel_tok in const_reg:
                            sel_val = const_reg[sel_tok]
                            vperm_creg += 1
                        else:
                            vperm_unres += 1
                        if sel_val is not None:
                            has_sext = False
                            for shift in (0, 8, 16, 24):
                                b = (sel_val >> shift) & 0xFF
                                if b <= 7:
                                    vperm_classes["0_7"] += 1
                                elif b <= 11:
                                    vperm_classes["8_11"] += 1
                                    has_sext = True
                                elif b == 12:
                                    vperm_classes["12"] += 1
                                else:
                                    vperm_classes["13_plus"] += 1
                            if has_sext:
                                vperm_sext_insns += 1
                                seen_vperm_sext = True
                    _update_const_regs(const_reg, op, operands)
                m = MMA.match(line)
                if m:
                    mma_ops[m.group(1)] += 1
                    seen_mma = True
                m = ORDERED.match(line)
                if m:
                    ordered_ops[m.group(1)] += 1
                    seen_ord = True
                if GDS.match(line):
                    seen_gds = True
        mma[stage] += seen_mma
        ordered[stage] += seen_ord
        gds[stage] += seen_gds
        vperm_shaders[stage] += seen_vperm
        vperm_sext_shaders += int(seen_vperm_sext)
        code_data[stage] += getpc_for_data(ops)
    nz = lambda c: {k: v for k, v in c.items() if v}
    return {"files": files, "wave_mma_shaders": nz(mma), "wave_mma_ops": dict(mma_ops),
            "ordered_or_gws_shaders": nz(ordered), "ordered_or_gws_ops": dict(ordered_ops), "gds_shaders": nz(gds),
            "getpc_for_data_shaders": nz(code_data),
            "v_perm_b32": {
                "shaders": nz(vperm_shaders),
                "insns": vperm_insns,
                "immediate": vperm_imm,
                "const_reg": vperm_creg,
                "unresolved": vperm_unres,
                "sign_extend_8_11_shaders": vperm_sext_shaders,
                "sign_extend_8_11_insns": vperm_sext_insns,
                "selector_classes": vperm_classes,
            }}


def scan_spirv(spirv_index: str) -> dict:
    with open(spirv_index, encoding="utf-8") as f:
        idx = json.load(f)
    entries = idx.values() if isinstance(idx, dict) else idx
    total, code_base = Counter(), Counter()
    for m in entries:
        if not isinstance(m, dict):
            continue
        stage = m.get("stage", "?")
        total[stage] += 1
        code_base[stage] += bool(m.get("uses_code_base"))
    return {"translated": dict(total), "uses_code_base": {k: v for k, v in code_base.items() if v}}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("shaders", help="<work>/shaders from make_game.py")
    ap.add_argument("--out", help="also write the counts as JSON")
    a = ap.parse_args()

    dxbc_dir = os.path.join(a.shaders, "dxbc")
    index_path = os.path.join(dxbc_dir, "index.json")
    if not os.path.isdir(a.shaders):
        what = "is not a folder" if os.path.exists(a.shaders) else "does not exist"
        print(f"ERROR: {a.shaders} {what}. Give the <work>/shaders folder make_game.py writes.", file=sys.stderr)
        return 2
    if not os.path.isfile(index_path):
        print(f"ERROR: {index_path} is missing. Run the shaders stage of make_game.py first.", file=sys.stderr)
        return 2
    try:
        with open(index_path, encoding="utf-8") as f:
            index = json.load(f)
    except (OSError, ValueError) as e:
        print(f"ERROR: cannot read {index_path}: {getattr(e, 'strerror', None) or e}", file=sys.stderr)
        return 2
    if not isinstance(index, list):
        print(f"ERROR: {index_path} is not a shader list. Run the shaders stage of make_game.py again.",
              file=sys.stderr)
        return 2
    result = {"shaders": len(index), "dxbc": scan_dxbc(dxbc_dir, index)}
    stage_of = {os.path.splitext(e["file"])[0]: e.get("stage") or "?" for e in index}
    stage_of |= {e["name"]: e.get("stage") or "?" for e in index}
    disasm = os.path.join(dxbc_dir, "disasm")
    if os.path.isdir(disasm):
        result["disasm"] = scan_disasm(disasm, stage_of)
    spirv = os.path.join(a.shaders, "spirv", "index.json")
    if os.path.isfile(spirv):
        result["spirv"] = scan_spirv(spirv)

    d = result["dxbc"]
    print(f"{result['shaders']} shaders")
    print("DXIL stages:", ", ".join(f"{s} {n}" for s, n in d["stages"].items()))
    for name in WATCHED:
        print(f"SFI0 {name}: {sum(d['sfi0'].get(name, {}).values())} {d['sfi0'].get(name, {})}")
    print("SFI0, every flag set anywhere:", {n: sum(v.values()) for n, v in d["sfi0"].items()})
    print("GS output-signature stream counts:", d["gs_output_stream_count"],
          "| shaders with an output on a stream other than 0:", d["output_on_stream_other_than_0"] or 0)
    if "disasm" in result:
        s = result["disasm"]
        print(f"disasm: {s['files']} files | wave MMA shaders {s['wave_mma_shaders'] or 0} | ordered-count/GWS shaders "
              f"{s['ordered_or_gws_shaders'] or 0} {s['ordered_or_gws_ops'] or ''} | GDS shaders {s['gds_shaders'] or 0}")
        print("s_getpc_b64 used for code-embedded data (not a far branch), by stage:", s["getpc_for_data_shaders"] or 0)
        vp = s["v_perm_b32"]
        print(f"v_perm_b32: {sum(vp['shaders'].values())} shaders {vp['shaders'] or 0} | {vp['insns']} insns "
              f"(imm {vp['immediate']}, const_reg {vp['const_reg']}, unresolved {vp['unresolved']}) | "
              f"selectors 8-11: {vp['sign_extend_8_11_shaders']} shaders / {vp['sign_extend_8_11_insns']} insns | "
              f"bytes {vp['selector_classes']}")
    else:
        print("disasm: not there yet (run the shaders stage)")
    if "spirv" in result:
        s = result["spirv"]
        print("uses_code_base by stage:", s["uses_code_base"], "of", s["translated"])
    else:
        print("spirv/index.json: not there yet (run the shaders stage)")
    if d["unreadable"]:
        print(f"{d['unreadable']} .dxbc files could not be read")
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
