#!/usr/bin/env python3
"""Pass-through geometry shaders that feed rdna2spirv.py --bary-fallback pixel shaders.

On a device without VK_KHR_fragment_shader_barycentric, `rdna2spirv.py --bary-fallback` declares the raw barycentric
pair a pixel shader reads as an ordinary vec3 input, at the location its index.json entry gives as
"bary_fallback_input". This module writes the stage that fills it: a geometry shader that copies Position and the
pixel shader's vec4 inputs from each vertex unchanged and adds (1,0,0), (0,1,0), (0,0,1) at the triangle's three
vertices. Default (perspective) interpolation of those gives the perspective-correct barycentrics, the value
BaryCoordKHR has, so the pixel shader's .y/.z are AMD's i/j.

    python -I tools/bary_fallback_gs.py <spirv dir>      # <name>.bary_gs.spv per such PS in index.json

    python -I tools/bary_fallback_gs.py <spirv dir> --pair VS PS

The paired helper also forwards the VS's ClipDistance array. Its clip-count suffix distinguishes it from a
PS-only helper and other clip layouts. Standard library only; build_for_index assembles and validates with spirv-as / spirv-val from PATH.
"""
import argparse
import json
import os
import subprocess
import sys

SUFFIX = ".bary_gs"   # <ps name>.bary_gs.spvasm / .spv next to the pixel shader


def gs_spvasm(locations, bary_location, clip_count: int = 0):
    """SPIR-V assembly for a triangle pass-through GS: Position and the vec4 inputs at `locations`, copied per vertex,
    plus the one-hot barycentric vec3 at `bary_location`."""
    locations = sorted(set(int(n) for n in locations))
    if bary_location in locations:
        raise ValueError(f"bary location {bary_location} collides with a pass-through location")
    ins = [f"%in{n}" for n in locations]
    outs = [f"%out{n}" for n in locations]
    if type(clip_count) is not int or not 0 <= clip_count <= 8:
        raise ValueError("clip distance count must be an integer from 0 to 8")
    clip_iface = ["%in_clip", "%out_clip"] if clip_count else []
    iface = " ".join(["%in_pos", "%out_pos", *ins, *outs, "%out_bary", *clip_iface])
    lines = [
        "OpCapability Geometry",
        *(["OpCapability ClipDistance"] if clip_count else []),
        "OpMemoryModel Logical GLSL450",
        f'OpEntryPoint Geometry %main "main" {iface}',
        "OpExecutionMode %main Triangles",
        "OpExecutionMode %main Invocations 1",
        "OpExecutionMode %main OutputTriangleStrip",
        "OpExecutionMode %main OutputVertices 3",
        "OpDecorate %in_pos BuiltIn Position",
        "OpDecorate %out_pos BuiltIn Position",
        *(["OpDecorate %in_clip BuiltIn ClipDistance", "OpDecorate %out_clip BuiltIn ClipDistance"]
          if clip_count else []),
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
        *([f"%clip_count = OpConstant %uint {clip_count}",
           "%clip = OpTypeArray %float %clip_count",
           "%triangle_clip = OpTypeArray %clip %uint_3",
           "%ptr_in_triangle_clip = OpTypePointer Input %triangle_clip",
           "%ptr_in_clip = OpTypePointer Input %clip",
           "%ptr_out_clip = OpTypePointer Output %clip"] if clip_count else []),
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
        *(["%in_clip = OpVariable %ptr_in_triangle_clip Input", "%out_clip = OpVariable %ptr_out_clip Output"]
          if clip_count else []),
        "%main = OpFunction %void None %fn",
        "%entry = OpLabel",
    ]
    for v in range(3):
        for src, dst in (("%in_pos", "%out_pos"), *zip(ins, outs)):
            p, x = f"%p{v}_{src[1:]}", f"%x{v}_{src[1:]}"
            lines += [f"{p} = OpAccessChain %ptr_in_v4 {src} %int_{v}", f"{x} = OpLoad %v4float {p}",
                      f"OpStore {dst} {x}"]
        if clip_count:
            lines += [f"%clip_ptr{v} = OpAccessChain %ptr_in_clip %in_clip %int_{v}",
                      f"%distances{v} = OpLoad %clip %clip_ptr{v}", f"OpStore %out_clip %distances{v}"]
        lines += [f"OpStore %out_bary %bary{v}", "OpEmitVertex"]
    lines += ["OpEndPrimitive", "OpReturn", "OpFunctionEnd"]
    return "\n".join(lines) + "\n"


def write_for_index(spirv_dir, pair: tuple[str, str] | None = None):
    """Write helpers for all fallback PS entries, or one clip-count-specific helper for an explicit (VS, PS) pair."""
    index_path = os.path.join(spirv_dir, "index.json")
    if pair is None and not os.path.isfile(index_path):
        return []
    with open(index_path, encoding="utf-8") as f:
        index = json.load(f)
    clip_count = 0
    names = sorted(index)
    suffix = ""
    if pair is not None:
        vertex, pixel = pair
        producer = index.get(vertex, {})
        consumer = index.get(pixel, {})
        if producer.get("stage") != "vs" or consumer.get("stage") != "ps" or not consumer.get("bary_fallback_input"):
            raise ValueError("--pair requires a translated VS and a barycentric-fallback PS from the index")
        caps = set(producer.get("capabilities", []))
        unsupported = caps & {"CullDistance", "ShaderLayer", "ShaderViewportIndex"}
        if unsupported:
            raise ValueError("paired helper cannot forward " + ", ".join(sorted(unsupported)))
        if "ClipDistance" in caps:
            clip_count = producer.get("clip_distance_count")
            if type(clip_count) is not int or not 1 <= clip_count <= 8:
                raise ValueError("missing or invalid VS clip_distance_count; regenerate the translation index")
        names = [pixel]
        suffix = f".clip{clip_count}"
    written = []
    for name in names:
        meta = index[name]
        fb = meta.get("bary_fallback_input") if isinstance(meta, dict) else None
        if not fb:
            continue
        path = os.path.join(spirv_dir, name + SUFFIX + suffix + ".spvasm")
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write(gs_spvasm(meta.get("inputs", []), fb["location"], clip_count))
        written.append(path)
    return written


def build_for_index(spirv_dir, pair: tuple[str, str] | None = None):
    """write_for_index, then assemble each file to <name>.bary_gs.spv and validate it (spirv-as / spirv-val on PATH,
    Vulkan 1.2 as for the translator). Returns the .spv paths; raises RuntimeError with the tool output on failure."""
    built = []
    for asm in write_for_index(spirv_dir, pair):
        spv = asm[: -len(".spvasm")] + ".spv"
        for cmd in (["spirv-as", "--target-env", "vulkan1.2", asm, "-o", spv],
                    ["spirv-val", "--target-env", "vulkan1.2", spv]):
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode:
                raise RuntimeError(f"{cmd[0]} failed for {asm}:\n{r.stdout}{r.stderr}")
        built.append(spv)
    return built


def main(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("spirv_dir")
    parser.add_argument("--pair", nargs=2, metavar=("VS", "PS"), help="build for these index entries")
    args = parser.parse_args(argv[1:])
    try:
        paths = build_for_index(args.spirv_dir, tuple(args.pair) if args.pair else None)
    except (OSError, ValueError, RuntimeError) as err:
        parser.exit(1, f"{err}\n")
    for path in paths:
        print(path)


if __name__ == "__main__":
    main(sys.argv)
