"""Count pixel-shader interpolation modes from the DXBC container and the FSSM blob, and check both against the code.

SHADER-181.  Nothing in a pixel shader's code says which barycentric pair it reads: every `v_interp` uses v0/v1.
On AMD hardware the pair (center, centroid or sample; perspective or linear) is chosen by SPI_PS_INPUT_ENA and
SPI_PS_INPUT_ADDR, which the driver sets.  This tool reads the two places that record the mode per input:

* **PSV0** (DXBC pipeline-state validation part).  After PSVRuntimeInfo, the resource table, the string table and the
  semantic-index table come the signature elements (16 bytes each, inputs first):

      +0  u32 semantic name (string-table offset)    +8  u8 rows      +11 u8 semantic kind
      +4  u32 semantic-index table index             +9  u8 start row +12 u8 component type
                                                     +10 u8 cols (bits 0-3), start column (bits 4-5)
                                                     +13 u8 interpolation mode

  The mode is DXIL's InterpolationMode: 0 undefined, 1 constant, 2 linear, 3 linear centroid, 4 noperspective,
  5 noperspective centroid, 6 linear sample, 7 noperspective sample.  ISG1 has no mode field, and the DXIL part of
  these containers carries no bitcode, so PSV0 is the only DXBC-side source.
* **The FSSM input table** (`fssm_header.interpolation_modes`): flags bits 0-1, 1 = flat, 2 = linear.

The code check reads `<dir>/disasm/<name>.s`: a table position read by `v_interp_mov_f32` must be flat, one read by
`v_interp_p1/p2` or an f16 interp form must be linear, as `rdna2spirv.check_interpolation_modes` requires.

Usage:
    python -I ps_interp_census.py <dir holding the .dxbc containers, fssm/ and disasm/> [--json] [--limit N]

Counts and layouts only: the tool prints no shader names unless --list-disagreements is given.
"""
import argparse
import json
import os
import re
import struct
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fssm_header  # noqa: E402

PSV_MODES = {0: "undefined", 1: "constant", 2: "linear", 3: "linear_centroid", 4: "noperspective",
             5: "noperspective_centroid", 6: "linear_sample", 7: "noperspective_sample"}
# PSV semantic kinds (DXIL::SemanticKind) that a pixel shader receives through v_interp.
KIND_ARBITRARY, KIND_POSITION, KIND_CLIP, KIND_CULL, KIND_FRONT_FACE = 0, 3, 6, 7, 13
INTERPOLATED_KINDS = (KIND_ARBITRARY, KIND_CLIP, KIND_CULL)
STAGE_PIXEL = 0
INTERP = re.compile(r"^\s*(v_interp_\w+?)(?:_e32|_e64)?\s[^;]*?\battr(\d+)\.")


def parts(data):
    """DXBC parts by fourcc; {} when the container is malformed."""
    out = {}
    if len(data) < 32 or data[:4] != b"DXBC":
        return out
    count = struct.unpack_from("<I", data, 28)[0]
    if 32 + 4 * count > len(data):
        return out
    for k in range(count):
        off = struct.unpack_from("<I", data, 32 + 4 * k)[0]
        if off + 8 > len(data):
            return {}
        size = struct.unpack_from("<I", data, off + 4)[0]
        out[data[off:off + 4]] = data[off + 8:off + 8 + size]
    return out


def psv_inputs(psv):
    """Return (stage, inputs) from a PSV0 part; inputs are dicts with name, index, kind, rows, start_row, mode.

    stage is None when PSVRuntimeInfo is too short to carry one (version 0), and then inputs is [].
    Raises ValueError on a truncated part.
    """
    try:
        pos = 0
        info_size, = struct.unpack_from("<I", psv, pos)
        pos += 4
        info = psv[pos:pos + info_size]
        pos += info_size
        if info_size < 36:
            return None, []
        stage, n_in, n_out, n_pc = info[24], info[28], info[29], info[30]
        n_res, = struct.unpack_from("<I", psv, pos)
        pos += 4
        if n_res:
            res_size, = struct.unpack_from("<I", psv, pos)
            pos += 4 + res_size * n_res
        str_size, = struct.unpack_from("<I", psv, pos)
        pos += 4
        strings = psv[pos:pos + str_size]
        pos += str_size
        n_idx, = struct.unpack_from("<I", psv, pos)
        pos += 4
        indexes = struct.unpack_from(f"<{n_idx}I", psv, pos)
        pos += 4 * n_idx
        inputs = []
        if n_in + n_out + n_pc:
            el_size, = struct.unpack_from("<I", psv, pos)
            pos += 4
            for k in range(n_in):
                e = psv[pos + el_size * k:pos + el_size * (k + 1)]
                if len(e) < 14:
                    raise ValueError("truncated signature element")
                name_off, sem = struct.unpack_from("<II", e, 0)
                end = strings.index(b"\0", name_off)
                inputs.append({"name": strings[name_off:end].decode("latin-1"), "index": indexes[sem],
                               "rows": e[8], "start_row": e[9], "kind": e[11], "mode": e[13]})
        return stage, inputs
    except (struct.error, IndexError) as e:
        raise ValueError(f"truncated PSV0: {e}") from None


def code_reads(listing):
    """{table position: {"mov", "interp"}} from a disassembly listing."""
    reads = {}
    for line in listing.splitlines():
        m = INTERP.match(line)
        if m:
            reads.setdefault(int(m.group(2)), set()).add("mov" if m.group(1) == "v_interp_mov_f32" else "interp")
    return reads


def census(root, limit=None):
    names = sorted(n[:-5] for n in os.listdir(root) if n.endswith(".dxbc"))
    t = {"containers": 0, "pixel_shaders": 0, "psv_attribute_modes": Counter(), "psv_position_modes": Counter(),
         "psv_shader_mode_sets": Counter(), "fssm_modes": Counter(), "fssm_missing": 0, "fssm_empty_table": 0,
         "code": Counter(), "code_shaders": Counter(), "disasm_missing": 0, "unreadable": 0}
    disagree = []
    for name in names[:limit] if limit is not None else names:
        t["containers"] += 1
        try:
            with open(os.path.join(root, name + ".dxbc"), "rb") as f:
                psv = parts(f.read()).get(b"PSV0")
            stage, inputs = psv_inputs(psv) if psv is not None else (None, [])
        except (OSError, ValueError):
            t["unreadable"] += 1
            continue
        if stage != STAGE_PIXEL:
            continue
        t["pixel_shaders"] += 1
        modes = set()
        for e in inputs:
            label = PSV_MODES.get(e["mode"], f"mode{e['mode']}")
            if e["kind"] in INTERPOLATED_KINDS:
                t["psv_attribute_modes"][label] += 1
                modes.add(label)
            elif e["kind"] == KIND_POSITION:
                t["psv_position_modes"][label] += 1
        t["psv_shader_mode_sets"]["+".join(sorted(modes)) or "none"] += 1
        try:
            with open(os.path.join(root, "fssm", name + ".fssm"), "rb") as f:
                table = fssm_header.interpolation_modes(f.read())
        except OSError:
            t["fssm_missing"] += 1
            continue
        if not table:
            t["fssm_empty_table"] += 1
        for m in table:
            t["fssm_modes"][fssm_header.mode_name(m)] += 1
        try:
            with open(os.path.join(root, "disasm", name + ".s"), encoding="utf-8", errors="replace") as f:
                reads = code_reads(f.read())
        except OSError:
            t["disasm_missing"] += 1
            continue
        ok = True
        for n, kinds in sorted(reads.items()):
            if n >= len(table):
                t["code"]["read past the table"] += 1
                continue
            want = fssm_header.MODE_FLAT if kinds == {"mov"} else fssm_header.MODE_LINEAR
            agrees = len(kinds) == 1 and table[n] == want
            t["code"][f"{fssm_header.mode_name(table[n])} read by {'+'.join(sorted(kinds))}"] += 1
            ok = ok and agrees
        for n in range(len(table)):
            if n not in reads:
                t["code"][f"{fssm_header.mode_name(table[n])} unread"] += 1
        t["code_shaders"]["agree" if ok else "disagree"] += 1
        if not ok:
            disagree.append(name)
    return t, disagree


def main(argv=None):
    ap = argparse.ArgumentParser(description="Count PS interpolation modes (PSV0, FSSM) and check them against the code.")
    ap.add_argument("dir", help="folder with the .dxbc containers and their fssm/ and disasm/ subfolders")
    ap.add_argument("--json", action="store_true", help="print one JSON document")
    ap.add_argument("--limit", type=int, help="stop after this many containers")
    ap.add_argument("--list-disagreements", action="store_true", help="name the shaders whose code disagrees")
    args = ap.parse_args(argv)
    if not os.path.isdir(args.dir):
        print(f"ERROR: {args.dir} is not a folder", file=sys.stderr)
        return 2
    if args.limit is not None and args.limit < 0:
        print("ERROR: --limit must be 0 or more", file=sys.stderr)
        return 2
    t, disagree = census(args.dir, args.limit)
    if args.json:
        doc = {k: dict(sorted(v.items())) if isinstance(v, Counter) else v for k, v in t.items()}
        if args.list_disagreements:
            doc["disagreements"] = disagree
        json.dump(doc, sys.stdout, indent=1)
        sys.stdout.write("\n")
        return 0
    print(f"{t['containers']} containers, {t['pixel_shaders']} pixel shaders ({t['unreadable']} unreadable)")
    for key in ("psv_attribute_modes", "psv_position_modes", "psv_shader_mode_sets", "fssm_modes", "code",
                "code_shaders"):
        print(f"{key}: " + (", ".join(f"{k} {v}" for k, v in t[key].most_common()) or "none"))
    print(f"fssm missing {t['fssm_missing']}, empty table {t['fssm_empty_table']}, disasm missing {t['disasm_missing']}")
    if args.list_disagreements:
        for name in disagree:
            print("disagrees:", name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
