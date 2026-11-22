#!/usr/bin/env python3
"""Give set 0's descriptor arrays fixed, non-overlapping Metal argument-buffer IDs (SHADER-161).

SPIRV-Cross (1.4.363.0) puts every runtime descriptor array of an argument buffer at one ID: with Leonida's
set 0 (sampled images at binding 0, samplers at 1, more image arrays at 3+) it comments all but one out as
"Overlapping binding", so the shader can't reach them. This rewrites a translated module for a native Metal
path only (the Vulkan module and layout stay as they are):

- each set-0 runtime descriptor array becomes a sized array with that binding's capacity from
  runtime/src/vulkan_bindless.h (`bindings`), and
- each set-0 Binding becomes the first ID of its range: the capacities of all lower bindings added up.

Convert the result with `spirv-cross --msl --msl-argument-buffers --msl-argument-buffer-tier 1
--msl-decoration-binding`, so the IDs are the same in every shader, whichever bindings it uses. The argument
buffer then holds binding b's element i at ID `metal_id(b) + i` (`--print-layout` prints the table).

Usage: spirv_metal_set0.py in.spv out.spv     |     spirv_metal_set0.py --print-layout
Needs spirv-dis and spirv-as (SPIRV-Tools) on PATH, or SPIRV_DIS / SPIRV_AS.
"""
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BINDLESS_H = os.path.join(ROOT, "runtime", "src", "vulkan_bindless.h")


def capacities(path=BINDLESS_H):
    """Descriptor counts by binding number, from the `bindings` table of vulkan_bindless.h."""
    src = open(path, encoding="utf-8").read()
    m = re.search(r"std::array<Binding,\s*(\d+)>\s*bindings\s*=\s*\{\{(.*?)\}\};", src, re.S)
    if not m:
        raise SystemExit(f"{path}: no `bindings` table")
    counts = [int(c) for c in re.findall(r"\{VK_DESCRIPTOR_TYPE_\w+,\s*(\d+)\}", m.group(2))]
    if len(counts) != int(m.group(1)):
        raise SystemExit(f"{path}: `bindings` declares {m.group(1)} entries, found {len(counts)}")
    return counts


def metal_ids(counts):
    """First argument-buffer ID of each binding: the capacities of the lower bindings added up."""
    ids, total = [], 0
    for c in counts:
        ids.append(total)
        total += c
    return ids


def rewrite(text, counts):
    """Rewrite the disassembly (spirv-dis --raw-id is not needed; names are kept) of one module."""
    ids = metal_ids(counts)
    lines = text.splitlines()
    sets, binds = {}, {}
    for ln in lines:
        m = re.match(r"\s*OpDecorate (%\S+) DescriptorSet (\d+)$", ln)
        if m:
            sets[m.group(1)] = int(m.group(2))
        m = re.match(r"\s*OpDecorate (%\S+) Binding (\d+)$", ln)
        if m:
            binds[m.group(1)] = int(m.group(2))
    set0 = {v: b for v, b in binds.items() if sets.get(v) == 0}
    for v, b in set0.items():
        if b >= len(counts):
            raise SystemExit(f"{v}: binding {b} isn't in vulkan_bindless.h")

    defs = {}       # result id -> instruction text
    for ln in lines:
        m = re.match(r"\s*(%\S+) = (.*)$", ln)
        if m:
            defs[m.group(1)] = m.group(2)
    uint = next((r for r, d in defs.items() if d == "OpTypeInt 32 0"), None)

    new_types = []      # inserted right after the variable's pointer type is declared
    retype = {}         # variable -> new pointer type
    for v, b in sorted(set0.items(), key=lambda kv: kv[1]):
        var = re.match(r"OpVariable (%\S+) UniformConstant$", defs.get(v, ""))
        if not var:
            continue    # the hash table (a storage buffer) and other non-array resources keep their type
        ptr = re.match(r"OpTypePointer UniformConstant (%\S+)$", defs[var.group(1)])
        rt = re.match(r"OpTypeRuntimeArray (%\S+)$", defs.get(ptr.group(1), "")) if ptr else None
        if not rt:
            continue
        if uint is None:
            raise SystemExit("module has no 32-bit uint type for the array lengths")
        cap, arr, nptr = f"%msl_cap_b{b}", f"%msl_arr_b{b}", f"%msl_ptr_b{b}"
        new_types.append((var.group(1), [f"{cap} = OpConstant {uint} {counts[b]}",
                                         f"{arr} = OpTypeArray {rt.group(1)} {cap}",
                                         f"{nptr} = OpTypePointer UniformConstant {arr}"]))
        retype[v] = nptr

    out = []
    for ln in lines:
        m = re.match(r"(\s*OpDecorate (%\S+) Binding )(\d+)$", ln)
        if m and m.group(2) in set0:
            ln = f"{m.group(1)}{ids[int(m.group(3))]}"
        m = re.match(r"(\s*(%\S+) = OpVariable )(%\S+)( UniformConstant)$", ln)
        if m and m.group(2) in retype:
            ln = f"{m.group(1)}{retype[m.group(2)]}{m.group(4)}"
        out.append(ln)
        m = re.match(r"\s*(%\S+) = OpTypePointer", ln)
        if m:
            for after, decl in new_types:
                if after == m.group(1):
                    out.extend(decl)
    return "\n".join(out) + "\n"


def main(argv):
    counts = capacities()
    if argv[1:] == ["--print-layout"]:
        for b, (c, i) in enumerate(zip(counts, metal_ids(counts))):
            print(f"binding {b:2}: {c:5} descriptors at IDs {i}..{i + c - 1}")
        return 0
    if len(argv) != 3:
        print(__doc__.strip().splitlines()[-2], file=sys.stderr)
        return 2
    dis = os.environ.get("SPIRV_DIS", "spirv-dis")
    asm = os.environ.get("SPIRV_AS", "spirv-as")
    text = subprocess.run([dis, argv[1]], check=True, capture_output=True, text=True).stdout
    env = re.search(r"; Version: (\d+)\.(\d+)", text)
    target = f"spv{env.group(1)}.{env.group(2)}" if env else "vulkan1.2"
    subprocess.run([asm, "--target-env", target, "-", "-o", argv[2]],
                   input=rewrite(text, counts), check=True, text=True)
    return 0


if __name__ == "__main__":
    try:  # a redirected stdout is strict ANSI code page under -I; write '?' for what it cannot encode
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    sys.exit(main(sys.argv))
