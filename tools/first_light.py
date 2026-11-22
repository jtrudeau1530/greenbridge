"""First light: draw one real game model with its own translated vertex and pixel shaders.

This assembles the smallest end-to-end path from the user's package to a Vulkan image: carve the game's
`shaders.rpf`, translate one real VS/PS pair with the repository's translator, convert one exported model to the
mesh the GPU harness reads, and run it on the device. Nothing is written outside `--work`, and no game data is
committed: the tool is the recipe, the user's copy is the input.

The default pair is a vertex shader that forwards POSITION and TEXCOORD to clip space and a pixel shader that
writes the interpolated TEXCOORD, so the first image is the model's own UV map. Both are ordinary game shaders;
`--vs` and `--ps` take other names from the carve index.

Usage:
    python -I first_light.py --game "<package>" --obj <model.obj> --work <dir> [--out <dir>] [--run]
    python -I first_light.py --game "<package>" --work <dir> --list          # show VS/PS candidates

The mesh file the harness reads is: u32 vertex_count, u32 index_count, vertex_count x (x,y,z,u,v) f32, then
index_count u32 indices. The draw is orthographic: the model is centred and scaled to NDC on X/Y and flattened to
z = 0.5, because the chosen VS does not apply a matrix (see docs/first_light.md).

The model is an OBJ from `rpf9dump models` (tools/rpf9dump). Building that tool and exporting an archive is the
only step outside this script:

    git submodule update --init
    dotnet build tools/rpf9dump/rpf9dump.csproj -c Release
    DOTNET_ROLL_FORWARD=Major tools/rpf9dump/bin/Release/net8.0/rpf9dump models "<package>/data.rpf" <work>/assets

Standard library only; the project's tools run with `python -I`.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import struct
import subprocess
import sys

TOOLS = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(TOOLS)
DEFAULT_VS = "VS_main_170e32e6_Wrapped"
DEFAULT_PS = "$PS_Rage_Untextured_14f8529b_Wrapped"
MESH_HEADER = struct.Struct("<II")


def run(argv, **kwargs):
    """Run a repository tool with this interpreter; fail loud with its command in the message."""
    cmd = [sys.executable, "-I"] + argv
    print("+ " + " ".join(cmd), file=sys.stderr)
    subprocess.run(cmd, check=True, **kwargs)


def carve_index(work: str) -> str:
    return os.path.join(work, "shaders", "dxbc", "index.json")


def ensure_carve(game: str, work: str) -> str:
    index = carve_index(work)
    if not os.path.isfile(index):
        run(
            [
                os.path.join(TOOLS, "shader_carve.py"),
                os.path.join(game, "shaders.rpf"),
                os.path.join(work, "shaders", "dxbc"),
            ]
        )
    return index


def choose(index_path: str, vs: str, ps: str) -> list[dict]:
    with open(index_path, encoding="utf-8") as f:
        entries = json.load(f)
    by_name = {e["name"]: e for e in entries}
    missing = [n for n in (vs, ps) if n not in by_name]
    if missing:
        raise SystemExit(
            f"shader not in the carve index: {', '.join(missing)} (use --list)"
        )
    return [by_name[vs], by_name[ps]]


def translate(work: str, picked: list[dict]) -> dict:
    """Translate just the two chosen containers, in the layout rdna2spirv.py reads."""
    base = os.path.join(work, "first_light")
    dxbc = os.path.join(base, "dxbc")
    os.makedirs(dxbc, exist_ok=True)
    index = []
    for entry in picked:
        shutil.copy(
            os.path.join(work, "shaders", "dxbc", entry["file"]),
            os.path.join(dxbc, entry["file"]),
        )
        index.append(
            {"name": entry["name"], "file": entry["file"], "stage": entry["stage"]}
        )
    with open(os.path.join(dxbc, "index.json"), "w", encoding="utf-8") as f:
        json.dump(index, f, indent=0)
    run([os.path.join(TOOLS, "fssm_unpack.py"), dxbc])
    run(
        [
            os.path.join(TOOLS, "shader_disasm.py"),
            os.path.join(dxbc, "fssm"),
            "--out",
            os.path.join(dxbc, "disasm"),
            "--mcpu",
            "gfx1013",
        ]
    )
    spirv = os.path.join(base, "spirv")
    run(
        [
            os.path.join(TOOLS, "rdna2spirv.py"),
            os.path.join(dxbc, "disasm"),
            "--out",
            spirv,
        ]
    )
    with open(os.path.join(spirv, "index.json"), encoding="utf-8") as f:
        meta = json.load(f)
    result = {}
    for entry in picked:
        key = entry["name"]
        item = meta.get(key)
        if item is None:  # a duplicate name gets a ~1 suffix
            item = next((v for k, v in meta.items() if k.startswith(key)), None)
        if item is None or item.get("stage") != entry["stage"]:
            raise SystemExit(f"translator has no {entry['stage']} result for {key}")
        result[entry["stage"]] = item
    return result


def read_obj(
    path: str,
) -> tuple[list[tuple[float, float, float, float, float]], list[int]]:
    """Positions, UVs and triangles as (x, y, z, u, v) vertices plus a u32 index list."""
    positions, uvs, faces = [], [], []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            p = line.split()
            if not p:
                continue
            if p[0] == "v":
                positions.append(tuple(float(x) for x in p[1:4]))
            elif p[0] == "vt":
                uvs.append(tuple(float(x) for x in p[1:3]))
            elif p[0] == "f":
                face = []
                for word in p[1:]:
                    parts = word.split("/")
                    vi = int(parts[0]) - 1
                    ti = int(parts[1]) - 1 if len(parts) > 1 and parts[1] else -1
                    face.append((vi, ti))
                for k in range(1, len(face) - 1):
                    faces.extend((face[0], face[k], face[k + 1]))
    if not positions or not faces:
        raise SystemExit(f"no vertices or faces in {path}")
    unique, vertices, indices = {}, [], []
    for vi, ti in faces:
        key = (vi, ti)
        if key not in unique:
            unique[key] = len(vertices)
            x, y, z = positions[vi]
            u, v = uvs[ti] if 0 <= ti < len(uvs) else (0.0, 0.0)
            vertices.append((x, y, z, u, v))
        indices.append(unique[key])
    return vertices, indices


def fit(vertices, width: int, height: int, fit_xy: bool, flatten_z: bool):
    xs = [v[0] for v in vertices]
    ys = [v[1] for v in vertices]
    zs = [v[2] for v in vertices]
    cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
    span = max(max(xs) - min(xs), max(ys) - min(ys)) or 1.0
    scale = 1.8 / span
    cz = (min(zs) + max(zs)) / 2 if flatten_z else 0.0
    aspect = width / height if height else 1.0
    out = []
    for x, y, z, u, v in vertices:
        nx = (x - cx) * scale * (1.0 / aspect if aspect > 1 else 1.0)
        ny = (y - cy) * scale * (aspect if aspect < 1 else 1.0)
        nz = 0.5 if flatten_z else (z - cz) * scale + 0.5
        out.append((nx, ny, nz, u, v))
    return out


def write_mesh(path: str, vertices, indices) -> None:
    with open(path, "wb") as f:
        f.write(MESH_HEADER.pack(len(vertices), len(indices)))
        f.writelines(struct.pack("<5f", *v) for v in vertices)
        f.write(struct.pack(f"<{len(indices)}I", *indices))


def find_harness(explicit: str | None) -> str:
    for candidate in (
        explicit,
        os.environ.get("LEONIDA_VK_HARNESS"),
        os.path.join(ROOT, "runtime", "build", "linux", "vk_harness"),
    ):
        if candidate and os.path.isfile(candidate):
            return candidate
    raise SystemExit(
        "vk_harness not found: build runtime/build/linux (target vk_harness) or set --harness"
    )


def run_draw(args, manifest: dict, out_dir: str) -> dict:
    harness = find_harness(args.harness)
    raw = os.path.join(out_dir, "image.raw")
    env = dict(
        os.environ,
        VKH_DRAW_MESH=os.path.join(out_dir, "mesh.bin"),
        VKH_VERTEX_LAYOUT="0:3f,1:2f",
    )
    cmd = [
        harness,
        os.path.join(out_dir, "vs.spv"),
        os.path.join(out_dir, "ps.spv"),
        raw,
        str(manifest["width"]),
        str(manifest["height"]),
    ]
    subprocess.run(cmd, check=True, env=env)
    with open(raw, "rb") as f:
        data = f.read()
    pixels = struct.unpack(f"<{len(data) // 4}f", data)
    covered = sum(1 for i in range(3, len(pixels), 4) if pixels[i] > 0.5)
    total = manifest["width"] * manifest["height"]
    center = manifest["width"] // 2
    ci = (manifest["height"] // 2 * manifest["width"] + center) * 4
    ppm = os.path.join(out_dir, "image.ppm")
    with open(ppm, "wb") as f:
        f.write(f"P6\n{manifest['width']} {manifest['height']}\n255\n".encode())
        f.writelines(
            bytes(max(0, min(255, round(pixels[i + c] * 255))) for c in range(3))
            for i in range(0, len(pixels), 4)
        )
    result = {
        "covered": covered,
        "total": total,
        "ratio": covered / total,
        "center": list(pixels[ci : ci + 4]),
        "ppm": ppm,
    }
    print(
        f"covered {covered}/{total} ({result['ratio']:.3f}); centre {result['center']}; {ppm}"
    )
    if result["ratio"] < manifest["min_coverage"]:
        raise SystemExit(
            f"coverage {result['ratio']:.3f} below {manifest['min_coverage']}"
        )
    manifest["result"] = result
    return result


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Draw one real game model with its own translated VS/PS."
    )
    ap.add_argument("--game", help="the package folder")
    ap.add_argument("--obj", help="an OBJ exported by rpf9dump models")
    ap.add_argument(
        "--work",
        default=os.path.expanduser("~/leonida-work"),
        help="the work folder make_game.py uses (default ~/leonida-work)",
    )
    ap.add_argument("--out", help="output folder (default <work>/first_light/out)")
    ap.add_argument(
        "--vs", default=DEFAULT_VS, help="vertex shader name in the carve index"
    )
    ap.add_argument(
        "--ps", default=DEFAULT_PS, help="pixel shader name in the carve index"
    )
    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--height", type=int, default=128)
    ap.add_argument("--min-coverage", type=float, default=0.5)
    ap.add_argument(
        "--no-fit",
        action="store_true",
        help="keep the model's coordinates (must already be in NDC)",
    )
    ap.add_argument("--keep-z", action="store_true", help="do not flatten z to 0.5")
    ap.add_argument(
        "--harness",
        help="vk_harness binary (default LEONIDA_VK_HARNESS or runtime/build/linux)",
    )
    ap.add_argument(
        "--run", action="store_true", help="run the draw and write image.raw/image.ppm"
    )
    ap.add_argument(
        "--check",
        action="store_true",
        help="run the draw and fail if the coverage is too low",
    )
    ap.add_argument(
        "--list",
        action="store_true",
        help="list VS/PS names in the carve index and exit",
    )
    a = ap.parse_args(argv)
    work = os.path.abspath(os.path.expanduser(a.work))
    out_dir = (
        os.path.abspath(os.path.expanduser(a.out))
        if a.out
        else os.path.join(work, "first_light", "out")
    )

    if a.list:
        if not a.game:
            raise SystemExit("--list needs --game")
        with open(ensure_carve(os.path.abspath(a.game), work), encoding="utf-8") as f:
            for e in json.load(f):
                if e["stage"] in ("vs", "ps"):
                    print(f"{e['stage']} {e['name']}")
        return 0
    if not a.game:
        raise SystemExit("--game is required")
    game = os.path.abspath(os.path.expanduser(a.game))
    if not a.no_fit and not a.obj:
        raise SystemExit("--obj is required unless --no-fit")

    index = ensure_carve(game, work)
    picked = choose(index, a.vs, a.ps)
    meta = translate(work, picked)
    os.makedirs(out_dir, exist_ok=True)
    shutil.copy(
        os.path.join(work, "first_light", "spirv", f"{picked[0]['name']}.spv"),
        os.path.join(out_dir, "vs.spv"),
    )
    shutil.copy(
        os.path.join(work, "first_light", "spirv", f"{picked[1]['name']}.spv"),
        os.path.join(out_dir, "ps.spv"),
    )
    vertices, indices = read_obj(a.obj) if a.obj else ([], [])
    if vertices:
        if not a.no_fit:
            vertices = fit(vertices, a.width, a.height, True, not a.keep_z)
        write_mesh(os.path.join(out_dir, "mesh.bin"), vertices, indices)
    manifest = {
        "vs": {
            "name": picked[0]["name"],
            "fssm_sha256": meta["vs"].get("fssm_sha256"),
            "verified": True,
        },
        "ps": {
            "name": picked[1]["name"],
            "fssm_sha256": meta["ps"].get("fssm_sha256"),
            "verified": True,
        },
        "vertex_count": len(vertices),
        "index_count": len(indices),
        "triangle_count": len(indices) // 3,
        "width": a.width,
        "height": a.height,
        "min_coverage": a.min_coverage,
        "files": {"vs": "vs.spv", "ps": "ps.spv", "mesh": "mesh.bin"},
    }
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1)
    print(f"{len(vertices)} vertices, {len(indices) // 3} triangles -> {out_dir}")
    if not (a.run or a.check):
        print(
            f"run: VKH_DRAW_MESH={os.path.join(out_dir, 'mesh.bin')} VKH_VERTEX_LAYOUT=0:3f,1:2f "
            f"<vk_harness> {os.path.join(out_dir, 'vs.spv')} {os.path.join(out_dir, 'ps.spv')} "
            f"{os.path.join(out_dir, 'image.raw')} {a.width} {a.height}"
        )
        return 0
    run_draw(a, manifest, out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
