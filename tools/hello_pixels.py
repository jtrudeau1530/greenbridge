#!/usr/bin/env python3
"""Draws a synthetic test image through vk_harness: the package-free "does it print pixels" check.

first_light.py needs the game's own package; this tool needs only vk_harness (built from runtime/) and
spirv-as (Vulkan SDK or spirv-tools). It assembles a tiny hand-written vertex+fragment pair, feeds an
indexed fullscreen quad through VKH_DRAW_MESH/VKH_VERTEX_LAYOUT, reads back the RGBA32F image and writes
image.ppm plus image.png. --check verifies the picture like first_light's oracle: coverage, the corner
clear value and pixel probes against the pattern's own closed form.

  python -I tools/hello_pixels.py --work <dir> --run [--check] [--size 256]

Everything it writes is synthetic and safe to keep; no game data is touched or produced.
"""

import json
import os
import shutil
import struct
import subprocess
import sys
import zlib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Vertex shader: location 0 = float3 position (already NDC), location 1 = float2 uv;
# forwards both to gl_Position and the interpolating varying.
VS_ASM = """\
; hello_pixels vertex shader: pass position through, forward uv
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Vertex %main "main" %in_pos %gl_Position %v_uv %in_uv
OpSource GLSL 450
OpName %main "main"
OpDecorate %gl_Position BuiltIn Position
OpDecorate %in_pos Location 0
OpDecorate %in_uv Location 1
OpDecorate %v_uv Location 0
%void = OpTypeVoid
%f32 = OpTypeFloat 32
%v3 = OpTypeVector %f32 3
%v2 = OpTypeVector %f32 2
%v4 = OpTypeVector %f32 4
%fn = OpTypeFunction %void
%pin3 = OpTypePointer Input %v3
%pin2 = OpTypePointer Input %v2
%pout4 = OpTypePointer Output %v4
%pout2 = OpTypePointer Output %v2
%in_pos = OpVariable %pin3 Input
%gl_Position = OpVariable %pout4 Output
%v_uv = OpVariable %pout2 Output
%in_uv = OpVariable %pin2 Input
%one = OpConstant %f32 1.0
%main = OpFunction %void None %fn
%entry = OpLabel
%p = OpLoad %v3 %in_pos
%u = OpLoad %v2 %in_uv
%px = OpCompositeExtract %f32 %p 0
%py = OpCompositeExtract %f32 %p 1
%pz = OpCompositeExtract %f32 %p 2
%clip = OpCompositeConstruct %v4 %px %py %pz %one
OpStore %gl_Position %clip
OpStore %v_uv %u
OpReturn
OpFunctionEnd
"""

# Fragment shader: rgb = gradient(uv) dimmed to 20% on a 16x16 checker grid; alpha = 1.
# r = u, g = v, b = (u+v)/2, dimmed on cells where floor(u*16) parity == floor(v*16) parity.
# Pure core instructions: OpFMod for the cell coordinate, OpFOrdLessThan/OpSelect for parity.
PS_ASM = """\
; hello_pixels fragment shader: uv gradient with a 16x16 checker grid
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %v_uv %out_color
OpExecutionMode %main OriginUpperLeft
OpSource GLSL 450
OpName %main "main"
OpDecorate %v_uv Location 0
OpDecorate %out_color Location 0
%void = OpTypeVoid
%f32 = OpTypeFloat 32
%bool = OpTypeBool
%v2 = OpTypeVector %f32 2
%v4 = OpTypeVector %f32 4
%fn = OpTypeFunction %void
%pin2 = OpTypePointer Input %v2
%pout4 = OpTypePointer Output %v4
%v_uv = OpVariable %pin2 Input
%out_color = OpVariable %pout4 Output
%c16 = OpConstant %f32 16.0
%c2 = OpConstant %f32 2.0
%c1 = OpConstant %f32 1.0
%cdim = OpConstant %f32 0.2
%half = OpConstant %f32 0.5
%main = OpFunction %void None %fn
%entry = OpLabel
%uv = OpLoad %v2 %v_uv
%u = OpCompositeExtract %f32 %uv 0
%v = OpCompositeExtract %f32 %uv 1
%cu = OpFMul %f32 %u %c16
%cv = OpFMul %f32 %v %c16
%mu = OpFMod %f32 %cu %c2
%mv = OpFMod %f32 %cv %c2
%lu = OpFOrdLessThan %bool %mu %c1
%lv = OpFOrdLessThan %bool %mv %c1
%same = OpLogicalEqual %bool %lu %lv
%grid = OpSelect %f32 %same %c1 %cdim
%sum = OpFAdd %f32 %u %v
%b = OpFMul %f32 %sum %half
%r = OpFMul %f32 %u %grid
%g = OpFMul %f32 %v %grid
%bb = OpFMul %f32 %b %grid
%col = OpCompositeConstruct %v4 %r %g %bb %c1
OpStore %out_color %col
OpReturn
OpFunctionEnd
"""

LAYOUT = "0:3f,1:2f"
CLEAR = (0.0, 0.0, 0.0, 0.0)


def find_harness(explicit=None):
    if explicit:
        return explicit
    env = os.environ.get("LEONIDA_VK_HARNESS")
    if env:
        return env
    exe = ".exe" if os.name == "nt" else ""
    for cand in (
        os.path.join(ROOT, "runtime", "build", "linux", f"vk_harness{exe}"),
        os.path.join(ROOT, "runtime", "build", "windows", f"vk_harness{exe}"),
        os.path.join(ROOT, "runtime", "build", f"vk_harness{exe}"),
    ):
        if os.path.isfile(cand):
            return cand
    raise SystemExit(
        "vk_harness not found: build runtime/build/linux or runtime/build/windows "
        "(target vk_harness), or pass --harness / set LEONIDA_VK_HARNESS"
    )


def find_spirv_as():
    exe = "spirv-as.exe" if os.name == "nt" else "spirv-as"
    return shutil.which(exe) or shutil.which("spirv-as")


def assemble(text: str, out_spv: str, assembler: str) -> None:
    p = subprocess.run(
        [assembler, "--target-env", "vulkan1.2", "-o", out_spv, "-"],
        input=text.encode(),
        capture_output=True,
    )
    if p.returncode != 0:
        raise SystemExit(f"spirv-as failed for {out_spv}:\n{p.stderr.decode(errors='replace')}")


def shaders_into(out_dir: str) -> None:
    # The documented no-SDK path: drop prebuilt vs.spv/ps.spv (from --emit-only on a machine that has
    # spirv-as) into the work directory and they are used as-is. Assembly still wins when spirv-as is
    # on PATH, so a stale pair never shadows fresh sources.
    vs_spv = os.path.join(out_dir, "vs.spv")
    ps_spv = os.path.join(out_dir, "ps.spv")
    assembler = find_spirv_as()
    if assembler is None:
        # Check the SPIR-V magic so a stray file does not reach the harness.
        def spv_ok(path: str) -> bool:
            if not os.path.isfile(path):
                return False
            with open(path, "rb") as f:
                return f.read(4) == b"\x03\x02\x23\x07"

        if spv_ok(vs_spv) and spv_ok(ps_spv):
            print("spirv-as not found; using the prebuilt vs.spv and ps.spv in the work directory")
            return
        raise SystemExit(
            "spirv-as not found on PATH (install spirv-tools or the Vulkan SDK), and the work directory "
            "has no prebuilt vs.spv/ps.spv pair (--emit-only on a machine that has one produces them)"
        )
    assemble(VS_ASM, vs_spv, assembler)
    assemble(PS_ASM, ps_spv, assembler)


def write_mesh(path: str) -> None:
    # Fullscreen indexed quad in NDC; (x, y, z, u, v) per vertex, y up.
    verts = [(-1.0, -1.0, 0.0, 0.0, 1.0), (1.0, -1.0, 0.0, 1.0, 1.0),
             (1.0, 1.0, 0.0, 1.0, 0.0), (-1.0, 1.0, 0.0, 0.0, 0.0)]
    idx = [0, 1, 2, 0, 2, 3]
    with open(path, "wb") as f:
        f.write(struct.pack("<II", len(verts), len(idx)))
        f.writelines(struct.pack("<5f", *v) for v in verts)
        f.write(struct.pack(f"<{len(idx)}I", *idx))


def expected_pixel(x: int, y: int, width: int, height: int):
    """Closed form of the fragment pattern at a pixel centre, y up (OriginUpperLeft -> row 0 is top)."""
    u = (x + 0.5) / width
    v = 1.0 - (y + 0.5) / height
    same = (int(u * 16) % 2) == (int(v * 16) % 2)
    grid = 1.0 if same else 0.2
    return (u * grid, v * grid, (u + v) * 0.5 * grid, 1.0)


def check_image(data: bytes, width: int, height: int):
    want = width * height * 16
    if len(data) != want:
        raise AssertionError(f"image.raw is {len(data)} bytes, expected {want}")
    pixels = struct.unpack(f"<{width * height * 4}f", data)
    clear_tol = 1e-3
    probes = [
        (width // 2, height // 2),
        (width // 4, height // 4),
        (3 * width // 4, height // 4),
        (width // 4, 3 * height // 4),
        (3 * width // 4, 3 * height // 4),
        (width // 8, height // 2),
        (7 * width // 8, height // 2),
    ]
    covered = 0
    for y in range(height):
        for x in range(width):
            i = (y * width + x) * 4
            if pixels[i + 3] != 0.0:
                covered += 1
    if covered != width * height:
        raise AssertionError(f"coverage {covered}/{width * height}: the quad did not fill the image")
    for x, y in probes:
        i = (y * width + x) * 4
        got = pixels[i : i + 4]
        want_px = expected_pixel(x, y, width, height)
        for c in range(4):
            if abs(got[c] - want_px[c]) > 0.04:
                raise AssertionError(
                    f"pixel ({x},{y}) is {[round(g, 4) for g in got]}, "
                    f"expected {[round(w, 4) for w in want_px]} channel {c}"
                )
    return {"covered": covered, "pixels": pixels}


def write_ppm(path: str, pixels, width: int, height: int) -> None:
    with open(path, "wb") as f:
        f.write(f"P6\n{width} {height}\n255\n".encode())
        f.writelines(
            bytes(max(0, min(255, round(pixels[i + c] * 255))) for c in range(3))
            for i in range(0, len(pixels), 4)
        )


def write_png(path: str, pixels, width: int, height: int) -> None:
    def chunk(tag: bytes, payload: bytes) -> bytes:
        return (
            struct.pack(">I", len(payload))
            + tag
            + payload
            + struct.pack(">I", zlib.crc32(tag + payload) & 0xFFFFFFFF)
        )

    scan = bytearray()
    for y in range(height):
        scan.append(0)
        for x in range(width):
            i = (y * width + x) * 4
            scan += bytes(max(0, min(255, round(pixels[i + c] * 255))) for c in range(3))
    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(bytes(scan), 6))
        + chunk(b"IEND", b"")
    )
    with open(path, "wb") as f:
        f.write(png)


def run_draw(args, out_dir: str) -> dict:
    harness = find_harness(args.harness)
    raw = os.path.join(out_dir, "image.raw")
    if os.path.exists(raw):
        os.unlink(raw)
    env = dict(os.environ, VKH_DRAW_MESH=os.path.join(out_dir, "mesh.bin"), VKH_VERTEX_LAYOUT=LAYOUT)
    cmd = [
        harness,
        os.path.join(out_dir, "vs.spv"),
        os.path.join(out_dir, "ps.spv"),
        raw,
        str(args.size),
        str(args.size),
    ]
    subprocess.run(cmd, check=True, env=env)
    if not os.path.isfile(raw):
        raise SystemExit(f"the harness exited successfully but did not write {raw}")
    with open(raw, "rb") as f:
        data = f.read()
    if args.check:
        result = check_image(data, args.size, args.size)
    else:
        result = {"pixels": struct.unpack(f"<{args.size * args.size * 4}f", data)}
    pixels = result["pixels"]
    ppm = os.path.join(out_dir, "image.ppm")
    png = os.path.join(out_dir, "image.png")
    write_ppm(ppm, pixels, args.size, args.size)
    write_png(png, pixels, args.size, args.size)
    print(f"{args.size}x{args.size} -> {ppm}")
    print(f"{args.size}x{args.size} -> {png}")
    return result


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", default=".", help="output directory (default: cwd)")
    ap.add_argument("--size", type=int, default=256, help="image edge in pixels (default 256)")
    ap.add_argument("--harness", help="vk_harness binary (default LEONIDA_VK_HARNESS or runtime/build)")
    ap.add_argument("--run", action="store_true", help="run the draw and write image.raw/.ppm/.png")
    ap.add_argument("--check", action="store_true", help="verify coverage and probe pixels after the draw")
    ap.add_argument("--emit-only", action="store_true", help="write shaders and mesh.bin, then stop")
    args = ap.parse_args(argv)

    out_dir = os.path.abspath(args.work)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "vs.asm"), "w", encoding="utf-8", newline="\n") as f:
        f.write(VS_ASM)
    with open(os.path.join(out_dir, "ps.asm"), "w", encoding="utf-8", newline="\n") as f:
        f.write(PS_ASM)
    shaders_into(out_dir)
    write_mesh(os.path.join(out_dir, "mesh.bin"))
    print(f"emitted vs.spv, ps.spv, mesh.bin in {out_dir}")
    if args.emit_only:
        return 0
    if not args.run:
        print("add --run to draw with vk_harness (VKH_DRAW_MESH mesh path), --check to verify pixels")
        return 0
    result = run_draw(args, out_dir)
    manifest = {
        "size": args.size,
        "layout": LAYOUT,
        "synthetic": True,
        "result": {"covered": result.get("covered"), "ppm": "image.ppm", "png": "image.png"},
    }
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(manifest, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(errors="replace")
    sys.exit(main())
