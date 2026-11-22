#!/usr/bin/env python3
"""Pass-through geometry shaders that feed rdna2spirv.py --bary-fallback pixel shaders.

On a device without VK_KHR_fragment_shader_barycentric, `rdna2spirv.py --bary-fallback` declares the raw barycentric
pair a pixel shader reads as an ordinary vec3 input, at the location its index.json entry gives as
"bary_fallback_input". This module writes the stage that fills it: a geometry shader that copies Position and the
pixel shader's vec4 inputs from each vertex unchanged and adds (1,0,0), (0,1,0), (0,0,1) at the triangle's three
vertices. Default (perspective) interpolation of those gives the perspective-correct barycentrics, the value
BaryCoordKHR has, so the pixel shader's .y/.z are AMD's i/j.

    python -I tools/bary_fallback_gs.py <spirv dir>      # <name>.bary_gs.spv per such PS in index.json

Standard library only; build_for_index assembles and validates with spirv-as / spirv-val from PATH.
"""
import json
import os
import subprocess
import sys

SUFFIX = ".bary_gs"   # <ps name>.bary_gs.spvasm / .spv next to the pixel shader


def gs_spvasm(locations, bary_location):
    """SPIR-V assembly for a triangle pass-through GS: Position and the vec4 inputs at `locations`, copied per vertex,
    plus the one-hot barycentric vec3 at `bary_location`."""
    locations = sorted(set(int(n) for n in locations))
    if bary_location in locations:
        raise ValueError(f"bary location {bary_location} collides with a pass-through location")
    ins = [f"%in{n}" for n in locations]
    outs = [f"%out{n}" for n in locations]
    iface = " ".join(["%in_pos", "%out_pos", *ins, *outs, "%out_bary"])
    lines = [
        "OpCapability Geometry",
        "OpMemoryModel Logical GLSL450",
        f'OpEntryPoint Geometry %main "main" {iface}',
        "OpExecutionMode %main Triangles",
        "OpExecutionMode %main Invocations 1",
        "OpExecutionMode %main OutputTriangleStrip",
        "OpExecutionMode %main OutputVertices 3",
        "OpDecorate %in_pos BuiltIn Position",
        "OpDecorate %out_pos BuiltIn Position",
        *[f"OpDecorate %in{n} Location {n}" for n in locations],
        *[f"OpDecorate %out{n} Location {n}" for n in locations],
        f"OpDecorate %out_bary Location {int(bary_location)}",
        "%void = OpTypeVoid",
        "%fn = OpTypeFunction %void",
        "%uint = OpTypeInt 32 0",
        "%int = OpTypeInt 32 1",
        "%float = OpTypeFloat 32",
        "%v3float = OpTypeVector %float 3",
        "%v4float = OpTypeVector %float 4",
        "%uint_3 = OpConstant %uint 3",
        "%arr = OpTypeArray %v4float %uint_3",
        "%ptr_in_arr = OpTypePointer Input %arr",
        "%ptr_in_v4 = OpTypePointer Input %v4float",
        "%ptr_out_v4 = OpTypePointer Output %v4float",
        "%ptr_out_v3 = OpTypePointer Output %v3float",
        "%f0 = OpConstant %float 0",
        "%f1 = OpConstant %float 1",
        "%bary0 = OpConstantComposite %v3float %f1 %f0 %f0",
        "%bary1 = OpConstantComposite %v3float %f0 %f1 %f0",
        "%bary2 = OpConstantComposite %v3float %f0 %f0 %f1",
        *[f"%int_{v} = OpConstant %int {v}" for v in range(3)],
        "%in_pos = OpVariable %ptr_in_arr Input",
        "%out_pos = OpVariable %ptr_out_v4 Output",
        *[f"{v} = OpVariable %ptr_in_arr Input" for v in ins],
        *[f"{v} = OpVariable %ptr_out_v4 Output" for v in outs],
        "%out_bary = OpVariable %ptr_out_v3 Output",
        "%main = OpFunction %void None %fn",
        "%entry = OpLabel",
    ]
    for v in range(3):
        for src, dst in (("%in_pos", "%out_pos"), *zip(ins, outs)):
            p, x = f"%p{v}_{src[1:]}", f"%x{v}_{src[1:]}"
            lines += [f"{p} = OpAccessChain %ptr_in_v4 {src} %int_{v}", f"{x} = OpLoad %v4float {p}",
                      f"OpStore {dst} {x}"]
        lines += [f"OpStore %out_bary %bary{v}", "OpEmitVertex"]
    lines += ["OpEndPrimitive", "OpReturn", "OpFunctionEnd"]
    return "\n".join(lines) + "\n"


def write_for_index(spirv_dir):
    """Write <name>.bary_gs.spvasm for every pixel shader in <spirv_dir>/index.json that has a "bary_fallback_input".
    Returns the written paths (empty when there is no index or no such shader)."""
    index_path = os.path.join(spirv_dir, "index.json")
    if not os.path.isfile(index_path):
        return []
    with open(index_path, encoding="utf-8") as f:
        index = json.load(f)
    written = []
    for name, meta in sorted(index.items()):
        fb = meta.get("bary_fallback_input") if isinstance(meta, dict) else None
        if not fb:
            continue
        path = os.path.join(spirv_dir, name + SUFFIX + ".spvasm")
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write(gs_spvasm(meta.get("inputs", []), fb["location"]))
        written.append(path)
    return written


def build_for_index(spirv_dir):
    """write_for_index, then assemble each file to <name>.bary_gs.spv and validate it (spirv-as / spirv-val on PATH,
    Vulkan 1.2 as for the translator). Returns the .spv paths; raises RuntimeError with the tool output on failure."""
    built = []
    for asm in write_for_index(spirv_dir):
        spv = asm[: -len(".spvasm")] + ".spv"
        for cmd in (["spirv-as", "--target-env", "vulkan1.2", asm, "-o", spv],
                    ["spirv-val", "--target-env", "vulkan1.2", spv]):
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode:
                raise RuntimeError(f"{cmd[0]} failed for {asm}:\n{r.stdout}{r.stderr}")
        built.append(spv)
    return built


def main(argv):
    if len(argv) != 2:
        sys.exit(__doc__)
    for path in build_for_index(argv[1]):
        print(path)


if __name__ == "__main__":
    main(sys.argv)
