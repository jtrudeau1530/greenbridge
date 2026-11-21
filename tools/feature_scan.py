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
ORDERED = re.compile(r"^\s*(ds_ordered_count|ds_gws_\w+|ds_append|ds_consume|ds_add_gs_reg\w*|ds_sub_gs_reg\w*)\b")
GDS = re.compile(r"^\s*ds_\w+\b[^;]*\bgds\b")


def parts(d: bytes) -> dict:
    """DXBC part fourcc -> payload."""
    out = {}
    for k in range(struct.unpack_from("<I", d, 28)[0]):
        off = struct.unpack_from("<I", d, 32 + 4 * k)[0]
        size = struct.unpack_from("<I", d, off + 4)[0]
        out[d[off:off + 4].decode("latin-1")] = d[off + 8:off + 8 + size]
    return out


def streams(sig: bytes) -> set:
    """Stream index of every element of an ISG1/OSG1 signature (32-byte elements, stream first)."""
    if len(sig) < 8:
        return set()
    count, first = struct.unpack_from("<II", sig, 0)
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
        except (OSError, struct.error):
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


def scan_disasm(disasm_dir: str, stage_of: dict) -> dict:
    mma, ordered, gds, code_data = Counter(), Counter(), Counter(), Counter()
    mma_ops, ordered_ops = Counter(), Counter()
    files = 0
    for fn in os.listdir(disasm_dir):
        if not fn.endswith(".s"):
            continue
        files += 1
        stage = stage_of.get(fn[:-2], "?")
        seen_mma = seen_ord = seen_gds = False
        ops = []
        with open(os.path.join(disasm_dir, fn), encoding="utf-8", errors="replace") as f:
            for line in f:
                head = line.split(None, 1)
                if head and not head[0].startswith((";", ".")):
                    ops.append(head[0])
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
        code_data[stage] += getpc_for_data(ops)
    nz = lambda c: {k: v for k, v in c.items() if v}
    return {"files": files, "wave_mma_shaders": nz(mma), "wave_mma_ops": dict(mma_ops),
            "ordered_or_gws_shaders": nz(ordered), "ordered_or_gws_ops": dict(ordered_ops), "gds_shaders": nz(gds),
            "getpc_for_data_shaders": nz(code_data)}


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
    with open(os.path.join(dxbc_dir, "index.json"), encoding="utf-8") as f:
        index = json.load(f)
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
