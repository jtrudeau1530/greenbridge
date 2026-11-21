"""GPU A/B of control flow: render branchy pixel shaders translated both ways, as one loop around a switch
(--dispatch-loop) and as real loops and conditionals (the default), and compare the two images byte for byte.

    python -I tools/ab_control_flow.py <repo dir> <work>/shaders <out dir> [--count 150] [--loops only] [--runs 3]

Needs a finished shaders stage (the disassembly and index.json in <work>/shaders), spirv-as and spirv-val on PATH,
and runtime/build/vk_texture_test (or --harness). Shaders are picked from the corpus: pixel stage, at least two
branches, only the bindings and features the harness provides, and nothing that writes memory or images.

What the comparison needs to mean anything, each learned from a run that looked like a translator bug:
  - Every pixel input has to be defined. The vertex shader made here writes all 32 locations; with the game's
    pass-through one, which writes location 0 only, a shader that uses another input as a buffer index reads
    from wherever the driver's leftover value points, differently in each form.
  - A shader that exports no colour is compared by which pixels it keeps: the colour of those is undefined.
  - Shaders with loops go to a software driver first (--harness with a Linux build under Mesa llvmpipe). One
    that never ends is killed after --timeout there; on a real GPU it resets the driver. Feed the ones that
    finish back with --names.
The heap is zeros, so most images are a single colour: this shows the two forms take the same path to the same
result, not that the result is right. Outcomes go to <out dir>/outcomes.tsv.
"""
import argparse
import collections
import os
import re
import struct
import subprocess
import sys

# Left out: anything that writes memory or images, LDS, and the rare instructions the harness has no setup for.
# Loads are fine: with empty user data they read zeros from the start of the harness's heap, the same for both.
NEEDS = re.compile(r"^\t(image_store|image_atomic|buffer_store|tbuffer_|global_store|global_atomic|ds_|s_swappc|"
                   r"s_getpc|s_sendmsg|s_sleep)", re.M)
BRANCH = re.compile(r"^\ts_c?branch", re.M)
BACK = re.compile(r"^\ts_c?branch\S* (\d+)", re.M)
CLEAR = struct.pack("<4f", -1, -1, -1, -1)      # what vk_texture_test clears the image to
LOCATIONS = 32      # every pixel-shader input location gets a defined value


def test_vertex_shader(out_dir):
    """A full-screen triangle that writes every input location a pixel shader can read: location 0 is
    (u, v, 0, 1) with u and v running 0..1 across the image, the rest are zero. The game's pass-through vertex
    shader writes location 0 only, which leaves every other pixel input undefined, and a shader that uses one as
    a buffer index then reads from wherever that lands."""
    nl = "\n"
    asm = f"""OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Vertex %main "main" %pos %vid {" ".join(f"%o{i}" for i in range(LOCATIONS))}
OpDecorate %pos BuiltIn Position
OpDecorate %vid BuiltIn VertexIndex
{nl.join(f"OpDecorate %o{i} Location {i}" for i in range(LOCATIONS))}
%void = OpTypeVoid
%fn = OpTypeFunction %void
%float = OpTypeFloat 32
%int = OpTypeInt 32 1
%v4 = OpTypeVector %float 4
%po = OpTypePointer Output %v4
%pi = OpTypePointer Input %int
%pos = OpVariable %po Output
%vid = OpVariable %pi Input
{nl.join(f"%o{i} = OpVariable %po Output" for i in range(LOCATIONS))}
%f0 = OpConstant %float 0
%f1 = OpConstant %float 1
%f2 = OpConstant %float 2
%fm1 = OpConstant %float -1
%i1 = OpConstant %int 1
%i2 = OpConstant %int 2
%zero = OpConstantComposite %v4 %f0 %f0 %f0 %f0
%main = OpFunction %void None %fn
%l = OpLabel
%i = OpLoad %int %vid
%a = OpShiftLeftLogical %int %i %i1
%b = OpBitwiseAnd %int %a %i2
%c = OpBitwiseAnd %int %i %i2
%u = OpConvertSToF %float %b
%v = OpConvertSToF %float %c
%x = OpFMul %float %u %f2
%x2 = OpFAdd %float %x %fm1
%y = OpFMul %float %v %f2
%y2 = OpFAdd %float %y %fm1
%p = OpCompositeConstruct %v4 %x2 %y2 %f0 %f1
OpStore %pos %p
%t = OpCompositeConstruct %v4 %u %v %f0 %f1
OpStore %o0 %t
{nl.join(f"OpStore %o{i} %zero" for i in range(1, LOCATIONS))}
OpReturn
OpFunctionEnd
"""
    src, dst = os.path.join(out_dir, "test_vs.spvasm"), os.path.join(out_dir, "test_vs.spv")
    os.makedirs(out_dir, exist_ok=True)
    with open(src, "w", encoding="utf-8") as f:
        f.write(asm)
    subprocess.run(["spirv-as", "--target-env", "vulkan1.2", src, "-o", dst], check=True)
    subprocess.run(["spirv-val", "--target-env", "vulkan1.2", dst], check=True)
    return dst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("repo")
    ap.add_argument("shaders")
    ap.add_argument("out")
    ap.add_argument("--count", type=int, default=150)
    ap.add_argument("--harness", help="path to vk_texture_test (default: the Windows build in the repo)")
    ap.add_argument("--loops", choices=("none", "only"), default="none",
                    help="'only': shaders with backward branches. Use a software driver for those: one that never "
                         "terminates is killed after --timeout seconds, where a real GPU would reset its driver")
    ap.add_argument("--timeout", type=int, default=60)
    ap.add_argument("--names", help="run exactly the shaders listed in this file (one per line), no selection")
    ap.add_argument("--runs", type=int, default=1, help="render each form this many times; all must agree")
    a = ap.parse_args()
    disasm = os.path.join(a.shaders, "dxbc", "disasm")
    old = os.path.join(a.shaders, "spirv")
    new = os.path.join(a.out, "new")
    os.makedirs(new, exist_ok=True)
    # The texture test's harness: it binds a heap, a descriptor table, a fallback texture and a sampler.
    exe = "vk_texture_test.exe" if os.name == "nt" else "vk_texture_test"
    harness = a.harness or os.path.join(a.repo, "runtime", "build", exe)
    vs = test_vertex_shader(a.out)

    import json
    index = json.load(open(os.path.join(old, "index.json"), encoding="utf-8"))
    hosted = {"bufferDeviceAddress", "runtimeDescriptorArray", "shaderInt64"}     # what the harness enables
    picked, loops = [], 0
    names = sorted(n for n, m in index.items() if m.get("stage") == "ps" and set(m.get("bindings", [])) <= {0, 1, 2}
                   and set(m.get("features", [])) <= hosted)
    print(f"{len(names)} pixel shaders use only the bindings and features the harness provides")
    if a.names:
        with open(a.names, encoding="utf-8") as f:
            picked = [x for x in f.read().split("\n") if x]
        names = []
    for n in names:
        text = open(os.path.join(disasm, n + ".s"), encoding="utf-8", errors="replace").read()
        if NEEDS.search(text) or len(BRANCH.findall(text)) < 2 or not os.path.isfile(os.path.join(old, n + ".spv")):
            continue
        has_loop = any(int(x) >= 0x8000 for x in BACK.findall(text))     # a backward branch: a loop
        if has_loop != (a.loops == "only"):
            continue
        picked.append(n)
        if len(picked) >= a.count:
            break
    branches = sorted(len(BRANCH.findall(open(os.path.join(disasm, n + ".s"), encoding="utf-8", errors="replace").read()))
                      for n in picked)
    print(f"{len(picked)} branchy pixel shaders picked ({'with loops' if a.loops == 'only' else 'loop-free'}); "
          f"branches per shader: median {branches[len(branches) // 2] if branches else 0}, "
          f"max {branches[-1] if branches else 0}")

    # Both forms from the current translator, so the control-flow arrangement is the only difference.
    loopdir = os.path.join(a.out, "dispatch")
    os.makedirs(loopdir, exist_ok=True)
    res = collections.Counter()
    differ = []
    outcomes = []

    def note(n, what):
        res[what] += 1
        outcomes.append((n, what))
    tool = os.path.join(a.repo, "tools", "rdna2spirv.py")
    for n in picked:
        src = os.path.join(disasm, n + ".s")
        r1 = subprocess.run([sys.executable, "-I", tool, src, "--out", new], capture_output=True, text=True)
        r2 = subprocess.run([sys.executable, "-I", tool, src, "--out", loopdir, "--dispatch-loop"],
                            capture_output=True, text=True)
        if "1 pass spirv-val" not in r1.stdout or "1 pass spirv-val" not in r2.stdout:
            note(n, "a translation is invalid")
            continue
        imgs = []
        unstable = False
        # A shader that exports no colour (depth-only passes with an alpha test) leaves the colour of the pixels
        # it keeps undefined: garbage on a GPU, different every run. What it does decide is which pixels survive,
        # so compare that: a killed pixel still holds the clear colour.
        colour = "%out_mrt0" in open(os.path.join(new, n + ".spvasm"), encoding="utf-8").read()
        res["exports no colour: compared by surviving pixels"] += not colour
        for tag, fs in (("old", os.path.join(loopdir, n + ".spv")), ("new", os.path.join(new, n + ".spv"))):
            raw = os.path.join(a.out, f"{tag}.raw")
            got = []
            for _ in range(a.runs):
                if os.path.isfile(raw):
                    os.remove(raw)
                try:
                    h = subprocess.run([harness, vs, fs, raw, "8", "92", "c00000", "14500000", "0"],
                                       capture_output=True, text=True, timeout=a.timeout)
                    img = open(raw, "rb").read() if h.returncode == 0 and os.path.isfile(raw) else None
                    if img is not None and not colour:
                        img = bytes(img[i:i + 16] != CLEAR for i in range(0, len(img), 16))
                    got.append(img)
                except subprocess.TimeoutExpired:
                    got.append("timeout")
                    break
            unstable |= any(g != got[0] for g in got)
            imgs.append(got[0])
        if unstable:
            note(n, "unstable from run to run")
            continue
        if "timeout" in imgs:
            note(n, "neither terminates" if imgs[0] == imgs[1] else "ONLY ONE FORM TERMINATES")
            if imgs[0] != imgs[1]:
                differ.append(n)
        elif imgs[0] is None or imgs[1] is None:
            note(n, "dispatch form failed to run" if imgs[0] is None else "structured form failed to run")
        elif imgs[0] == imgs[1]:
            note(n, "identical")
            step = 16 if colour else 1
            res["identical and not all one colour"] += len(set(imgs[1][i:i + step] for i in range(0, len(imgs[1]), step))) > 1
        else:
            note(n, "DIFFERENT")
            differ.append(n)
    print(dict(res))
    for n in differ[:8]:
        print("   differs:", n[:70])
    with open(os.path.join(a.out, "outcomes.tsv"), "w", encoding="utf-8") as f:
        f.write("".join(f"{n}\t{o}\n" for n, o in outcomes))
    return 0


if __name__ == "__main__":
    sys.exit(main())
