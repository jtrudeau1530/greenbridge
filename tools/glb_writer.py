"""Pure standard-library Wavefront OBJ parser and binary glTF 2.0 (.glb) writer.

Usage:
    python -I tools/glb_writer.py <in.obj> <out.glb> [--up z|y]

Reads a Wavefront OBJ file (positions, UVs, normals, faces, groups, objects, materials),
converts coordinates from RAGE Z-up to glTF Y-up (or keeps Y-up), computes smooth
area-weighted vertex normals when explicit normals are absent, and serializes a strict
glTF 2.0 binary (.glb) container with optional deduplicated embedded PNG textures.
"""
from __future__ import annotations

import argparse
import array
import hashlib
import itertools
import json
import math
import os
import struct
import sys

GLB_MAGIC = b"glTF"
GLB_VERSION = 2
CHUNK_TYPE_JSON = 0x4E4F534A  # b"JSON"
CHUNK_TYPE_BIN = 0x004E4942   # b"BIN\x00"

TARGET_ARRAY_BUFFER = 34962
TARGET_ELEMENT_ARRAY_BUFFER = 34963

COMPONENT_UNSIGNED_SHORT = 5123
COMPONENT_UNSIGNED_INT = 5125
COMPONENT_FLOAT = 5126

MODE_TRIANGLES = 4

DEFAULT_CLAY_COLOR = [0.75, 0.75, 0.75, 1.0]
DEFAULT_TEXTURED_COLOR = [1.0, 1.0, 1.0, 1.0]
DEFAULT_METALLIC = 0.0
DEFAULT_ROUGHNESS = 0.85

DEFAULT_SAMPLER = {
    "magFilter": 9729,  # LINEAR
    "minFilter": 9987,  # LINEAR_MIPMAP_LINEAR
    "wrapS": 10497,     # REPEAT
    "wrapT": 10497,     # REPEAT
}

_NEEDS_BYTESWAP = sys.byteorder != "little"
_UINT16_CODE = "H" if array.array("H").itemsize == 2 else None
_UINT32_CODE = "I" if array.array("I").itemsize == 4 else ("L" if array.array("L").itemsize == 4 else None)
_FLOAT32_CODE = "f" if array.array("f").itemsize == 4 else None


class GlbError(ValueError):
    """Raised when OBJ/primitive data or GLB binary validation fails."""


def _clean_zero(v: float) -> float:
    return 0.0 if v == 0.0 else v


def _normalize3(nx: float, ny: float, nz: float) -> tuple[float, float, float]:
    if not (math.isfinite(nx) and math.isfinite(ny) and math.isfinite(nz)):
        return (0.0, 1.0, 0.0)
    length = math.sqrt(nx * nx + ny * ny + nz * nz)
    if length <= 0.0 or not math.isfinite(length):
        return (0.0, 1.0, 0.0)
    inv = 1.0 / length
    return (_clean_zero(nx * inv), _clean_zero(ny * inv), _clean_zero(nz * inv))


def _compute_smooth_normals(
    positions: list[tuple[float, float, float]],
    indices: list[int],
) -> list[tuple[float, float, float]]:
    """Compute area-weighted smooth vertex normals with (0, 1, 0) fallback."""
    n = len(positions)
    ax = [0.0] * n
    ay = [0.0] * n
    az = [0.0] * n

    for t in range(0, len(indices) - 2, 3):
        i0 = indices[t]
        i1 = indices[t + 1]
        i2 = indices[t + 2]
        p0 = positions[i0]
        p1 = positions[i1]
        p2 = positions[i2]
        e1x = p1[0] - p0[0]
        e1y = p1[1] - p0[1]
        e1z = p1[2] - p0[2]
        e2x = p2[0] - p0[0]
        e2y = p2[1] - p0[1]
        e2z = p2[2] - p0[2]
        # Cross product magnitude equals 2 * triangle area -> exact area weighting.
        cx = e1y * e2z - e1z * e2y
        cy = e1z * e2x - e1x * e2z
        cz = e1x * e2y - e1y * e2x
        if not (math.isfinite(cx) and math.isfinite(cy) and math.isfinite(cz)):
            continue
        ax[i0] += cx
        ay[i0] += cy
        az[i0] += cz
        ax[i1] += cx
        ay[i1] += cy
        az[i1] += cz
        ax[i2] += cx
        ay[i2] += cy
        az[i2] += cz

    normals: list[tuple[float, float, float]] = []
    append = normals.append
    for i in range(n):
        append(_normalize3(ax[i], ay[i], az[i]))
    return normals


def _resolve_obj_index(token: str, count: int) -> int | None:
    if not token:
        return None
    try:
        idx = int(token)
    except ValueError:
        return None
    if idx > 0:
        resolved = idx - 1
    elif idx < 0:
        resolved = count + idx
    else:
        return None
    if 0 <= resolved < count:
        return resolved
    return None


def parse_obj(
    text_or_path: str,
    *,
    is_path: bool = True,
    up_axis: str = "z",
    offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
    scale: float = 1.0,
) -> list[dict]:
    """Parse Wavefront OBJ geometry into glTF-ready primitives.

    Groups primitives in order of appearance by (group_index, material_name),
    converts coordinates to glTF Y-up, flips V for top-left UV origin,
    deduplicates vertices per primitive, and synthesizes smooth area-weighted
    normals when explicit normals are missing.
    """
    up = up_axis.lower()
    if up not in ("z", "y"):
        raise GlbError(f"unsupported up_axis: {up_axis!r} (expected 'z' or 'y')")
    if len(offset) != 3:
        raise GlbError(f"offset must have 3 components, got {len(offset)}")
    ox, oy, oz = float(offset[0]), float(offset[1]), float(offset[2])
    sc = float(scale)
    if not (math.isfinite(sc) and math.isfinite(ox) and math.isfinite(oy) and math.isfinite(oz)):
        raise GlbError("scale and offset must be finite numbers")

    if is_path:
        with open(text_or_path, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
    else:
        text = text_or_path

    z_up = up == "z"
    raw_v: list[tuple[float, float, float] | None] = []
    raw_vt: list[tuple[float, float]] = []
    raw_vn: list[tuple[float, float, float]] = []

    raw_v_append = raw_v.append
    raw_vt_append = raw_vt.append
    raw_vn_append = raw_vn.append

    group_index = 0
    group_name = ""
    saw_g = False
    group_has_faces = False
    material_name = ""

    # Preserve order of appearance for (group_index, material_name) primitives.
    builders: dict[tuple[int, str], dict] = {}

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line[0] == "#":
            continue

        # Fast dispatch on the leading keyword.
        parts = line.split()
        cmd = parts[0]

        if cmd == "v":
            if "#" in line:
                parts = line.split("#", 1)[0].split()
            if len(parts) < 4:
                raw_v_append(None)
                continue
            try:
                x = float(parts[1])
                y = float(parts[2])
                z = float(parts[3])
            except ValueError:
                raw_v_append(None)
                continue
            if z_up:
                px = x * sc + ox
                py = z * sc + oy
                pz = (-y) * sc + oz
            else:
                px = x * sc + ox
                py = y * sc + oy
                pz = z * sc + oz
            if math.isfinite(px) and math.isfinite(py) and math.isfinite(pz):
                raw_v_append((_clean_zero(px), _clean_zero(py), _clean_zero(pz)))
            else:
                raw_v_append(None)

        elif cmd == "vt":
            if "#" in line:
                parts = line.split("#", 1)[0].split()
            if len(parts) < 2:
                raw_vt_append((0.0, 0.0))
                continue
            try:
                u = float(parts[1])
                v = float(parts[2]) if len(parts) >= 3 else 0.0
            except ValueError:
                raw_vt_append((0.0, 0.0))
                continue
            v_gltf = 1.0 - v
            if not math.isfinite(u):
                u = 0.0
            if not math.isfinite(v_gltf):
                v_gltf = 0.0
            raw_vt_append((_clean_zero(u), _clean_zero(v_gltf)))

        elif cmd == "vn":
            if "#" in line:
                parts = line.split("#", 1)[0].split()
            if len(parts) < 4:
                raw_vn_append((0.0, 1.0, 0.0))
                continue
            try:
                nx = float(parts[1])
                ny = float(parts[2])
                nz = float(parts[3])
            except ValueError:
                raw_vn_append((0.0, 1.0, 0.0))
                continue
            if z_up:
                nx, ny, nz = nx, nz, -ny
            raw_vn_append(_normalize3(nx, ny, nz))

        elif cmd == "g":
            if saw_g or group_has_faces:
                group_index += 1
            saw_g = True
            group_has_faces = False
            group_name = line[len(cmd):].split("#", 1)[0].strip()

        elif cmd == "o":
            if group_has_faces:
                group_index += 1
            saw_g = False
            group_has_faces = False
            group_name = line[len(cmd):].split("#", 1)[0].strip()

        elif cmd == "usemtl":
            material_name = line[len(cmd):].split("#", 1)[0].strip()

        elif cmd == "f":
            if "#" in line:
                parts = line.split("#", 1)[0].split()
            if len(parts) < 4:
                continue

            nv = len(raw_v)
            nvt = len(raw_vt)
            nvn = len(raw_vn)
            corners: list[tuple[int, int | None, int | None]] = []
            valid_face = True

            for tok in parts[1:]:
                slashes = tok.split("/")
                ns = len(slashes)
                if ns > 3:
                    valid_face = False
                    break

                vi = _resolve_obj_index(slashes[0], nv)
                if vi is None or raw_v[vi] is None:
                    valid_face = False
                    break

                vti: int | None = None
                if ns >= 2 and slashes[1]:
                    vti = _resolve_obj_index(slashes[1], nvt)
                    if vti is None:
                        valid_face = False
                        break

                vni: int | None = None
                if ns == 3 and slashes[2]:
                    vni = _resolve_obj_index(slashes[2], nvn)
                    if vni is None:
                        valid_face = False
                        break

                corners.append((vi, vti, vni))

            if not valid_face or len(corners) < 3:
                continue

            # Fan-triangulate polygon (c0, c_i, c_{i+1}), skipping degenerate triangles
            # that repeat the same vertex position index.
            c0 = corners[0]
            tris: list[tuple[tuple[int, int | None, int | None],
                             tuple[int, int | None, int | None],
                             tuple[int, int | None, int | None]]] = []
            for i in range(1, len(corners) - 1):
                c1 = corners[i]
                c2 = corners[i + 1]
                if c0[0] == c1[0] or c1[0] == c2[0] or c0[0] == c2[0]:
                    continue
                tris.append((c0, c1, c2))

            if not tris:
                continue

            group_has_faces = True
            key = (group_index, material_name)
            builder = builders.get(key)
            if builder is None:
                builder = {
                    "name": group_name,
                    "material": material_name,
                    "group_index": group_index,
                    "vert_map": {},
                    "positions": [],
                    "normals": [],
                    "uvs": [],
                    "indices": [],
                    "has_any_uv": False,
                    "has_missing_normal": False,
                }
                builders[key] = builder

            vert_map: dict[tuple[int, int | None, int | None], int] = builder["vert_map"]
            b_pos: list[tuple[float, float, float]] = builder["positions"]
            b_nrm: list[tuple[float, float, float] | None] = builder["normals"]
            b_uv: list[tuple[float, float] | None] = builder["uvs"]
            b_idx: list[int] = builder["indices"]

            for tri in tris:
                for corner in tri:
                    local_idx = vert_map.get(corner)
                    if local_idx is None:
                        local_idx = len(b_pos)
                        vert_map[corner] = local_idx
                        vi, vti, vni = corner
                        pos = raw_v[vi]
                        assert pos is not None
                        b_pos.append(pos)
                        if vti is not None:
                            builder["has_any_uv"] = True
                            b_uv.append(raw_vt[vti])
                        else:
                            b_uv.append(None)
                        if vni is not None:
                            b_nrm.append(raw_vn[vni])
                        else:
                            builder["has_missing_normal"] = True
                            b_nrm.append(None)
                    b_idx.append(local_idx)

    primitives: list[dict] = []
    for builder in builders.values():
        indices = builder["indices"]
        positions = builder["positions"]
        if not indices or not positions:
            continue

        if builder["has_missing_normal"]:
            computed = _compute_smooth_normals(positions, indices)
            raw_normals = builder["normals"]
            normals = [
                raw_normals[i] if raw_normals[i] is not None else computed[i]
                for i in range(len(positions))
            ]
        else:
            normals = builder["normals"]

        if builder["has_any_uv"]:
            uvs = [uv if uv is not None else (0.0, 0.0) for uv in builder["uvs"]]
        else:
            uvs = []

        primitives.append({
            "name": builder["name"],
            "material": builder["material"],
            "group_index": builder["group_index"],
            "positions": positions,
            "normals": normals,
            "uvs": uvs,
            "indices": indices,
        })

    return primitives


def compute_bounds(primitives: list[dict]) -> list[float] | None:
    """Return [minx, miny, minz, maxx, maxy, maxz] over all finite positions, or None."""
    minx = math.inf
    miny = math.inf
    minz = math.inf
    maxx = -math.inf
    maxy = -math.inf
    maxz = -math.inf
    found = False

    for prim in primitives:
        for pos in prim.get("positions") or ():
            if len(pos) < 3:
                continue
            x, y, z = float(pos[0]), float(pos[1]), float(pos[2])
            if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(z)):
                continue
            found = True
            if x < minx:
                minx = x
            if y < miny:
                miny = y
            if z < minz:
                minz = z
            if x > maxx:
                maxx = x
            if y > maxy:
                maxy = y
            if z > maxz:
                maxz = z

    if not found:
        return None
    return [
        _clean_zero(minx),
        _clean_zero(miny),
        _clean_zero(minz),
        _clean_zero(maxx),
        _clean_zero(maxy),
        _clean_zero(maxz),
    ]


def _pack_f32(values: list[tuple[float, ...]], components: int) -> tuple[bytes, array.array]:
    if _FLOAT32_CODE is not None:
        arr = array.array(_FLOAT32_CODE, itertools.chain.from_iterable(values))
        if len(arr) != len(values) * components:
            raise GlbError(f"expected {components} floats per vertex")
        if not all(map(math.isfinite, arr)):
            raise GlbError("non-finite float32 vertex attribute")
        raw = arr
        if _NEEDS_BYTESWAP:
            swapped = array.array(_FLOAT32_CODE, arr)
            swapped.byteswap()
            return swapped.tobytes(), raw
        return raw.tobytes(), raw

    flat: list[float] = []
    for item in values:
        if len(item) != components:
            raise GlbError(f"expected {components} floats per vertex")
        for c in item:
            v = float(c)
            if not math.isfinite(v):
                raise GlbError("non-finite float32 vertex attribute")
            flat.append(v)
    packed = struct.pack(f"<{len(flat)}f", *flat)
    unpacked = struct.unpack(f"<{len(flat)}f", packed)
    if not all(map(math.isfinite, unpacked)):
        raise GlbError("non-finite float32 vertex attribute")
    return packed, array.array("d", unpacked)


def _pack_indices(indices: list[int], vert_count: int) -> tuple[bytes, int]:
    if len(indices) % 3 != 0:
        raise GlbError(f"triangle index count must be a multiple of 3, got {len(indices)}")
    lo = min(indices)
    hi = max(indices)
    if lo < 0 or hi >= vert_count:
        raise GlbError(f"index out of range [0, {vert_count - 1}]: min={lo}, max={hi}")

    if vert_count <= 65535:
        comp_type = COMPONENT_UNSIGNED_SHORT
        if _UINT16_CODE is not None:
            arr = array.array(_UINT16_CODE, indices)
            if _NEEDS_BYTESWAP:
                arr.byteswap()
            return arr.tobytes(), comp_type
        return struct.pack(f"<{len(indices)}H", *indices), comp_type

    comp_type = COMPONENT_UNSIGNED_INT
    if _UINT32_CODE is not None:
        arr = array.array(_UINT32_CODE, indices)
        if _NEEDS_BYTESWAP:
            arr.byteswap()
        return arr.tobytes(), comp_type
    return struct.pack(f"<{len(indices)}I", *indices), comp_type


def _append_buffer_view(
    bin_buf: bytearray,
    buffer_views: list[dict],
    payload: bytes,
    *,
    target: int | None = None,
) -> int:
    pad = (-len(bin_buf)) % 4
    if pad:
        bin_buf.extend(b"\x00" * pad)
    offset = len(bin_buf)
    bin_buf.extend(payload)
    end_pad = (-len(bin_buf)) % 4
    if end_pad:
        bin_buf.extend(b"\x00" * end_pad)

    bv: dict = {
        "buffer": 0,
        "byteOffset": offset,
        "byteLength": len(payload),
    }
    if target is not None:
        bv["target"] = target
    bv_idx = len(buffer_views)
    buffer_views.append(bv)
    return bv_idx


def _parse_base_color_factor(value: object, default: list[float]) -> list[float]:
    if value is None:
        return list(default)
    if not isinstance(value, (list, tuple)) or len(value) not in (3, 4):
        raise GlbError("base_color_factor must be a 3- or 4-element float sequence")
    out = [float(c) for c in value]
    if len(out) == 3:
        out.append(1.0)
    if not all(math.isfinite(c) and 0.0 <= c <= 1.0 for c in out):
        raise GlbError("base_color_factor values must be finite numbers in [0.0, 1.0]")
    return [_clean_zero(c) for c in out]


def _build_glb_with_stats(
    primitives: list[dict],
    materials: dict[str, dict] | None = None,
    *,
    generator: str = "Leonida glb_writer",
) -> tuple[bytes, dict]:
    non_empty: list[dict] = []
    for prim in primitives:
        positions = prim.get("positions") or ()
        indices = prim.get("indices") or ()
        if len(positions) > 0 and len(indices) > 0:
            non_empty.append(prim)

    if not non_empty:
        raise GlbError("cannot build GLB: no non-empty primitives")

    bin_buf = bytearray()
    buffer_views: list[dict] = []
    accessors: list[dict] = []
    gltf_materials: list[dict] = []
    images: list[dict] = []
    textures: list[dict] = []

    mat_index_by_key: dict[tuple[str | int, bool], int] = {}
    default_mat_index: int | None = None
    tex_index_by_sha256: dict[bytes, int] = {}

    def get_texture_index(png_bytes: bytes) -> int:
        digest = hashlib.sha256(png_bytes).digest()
        existing = tex_index_by_sha256.get(digest)
        if existing is not None:
            return existing
        bv_idx = _append_buffer_view(bin_buf, buffer_views, png_bytes, target=None)
        img_idx = len(images)
        images.append({
            "bufferView": bv_idx,
            "mimeType": "image/png",
        })
        tex_idx = len(textures)
        textures.append({
            "sampler": 0,
            "source": img_idx,
        })
        tex_index_by_sha256[digest] = tex_idx
        return tex_idx

    def _lookup_material_spec(prim: dict) -> tuple[str | int, dict] | None:
        if materials is None:
            return None
        mat_key = prim.get("material", "")
        if mat_key is None:
            mat_key = ""
        if isinstance(materials, dict):
            if mat_key in materials:
                return mat_key, materials[mat_key] or {}
            if "material_index" in prim:
                midx = prim["material_index"]
                if isinstance(midx, int) and not isinstance(midx, bool) and midx in materials:
                    return midx, materials[midx] or {}
        elif isinstance(materials, (list, tuple)):
            midx = prim.get("material_index")
            if isinstance(midx, int) and not isinstance(midx, bool) and 0 <= midx < len(materials):
                return midx, materials[midx] or {}
        return None

    def get_material_index(prim: dict, *, has_uv: bool) -> int:
        nonlocal default_mat_index
        found = _lookup_material_spec(prim)
        if found is not None:
            resolved_key, spec = found
            raw_png = spec.get("png_bytes")
            has_png = has_uv and isinstance(raw_png, (bytes, bytearray, memoryview)) and len(raw_png) > 0
            cache_key = (resolved_key, has_png)
            existing = mat_index_by_key.get(cache_key)
            if existing is not None:
                return existing
            default_color = DEFAULT_TEXTURED_COLOR if has_png else DEFAULT_CLAY_COLOR
            base_color = _parse_base_color_factor(spec.get("base_color_factor"), default_color)

            pbr: dict = {
                "baseColorFactor": base_color,
                "metallicFactor": DEFAULT_METALLIC,
                "roughnessFactor": DEFAULT_ROUGHNESS,
            }
            if has_png:
                tex_idx = get_texture_index(bytes(raw_png))
                pbr["baseColorTexture"] = {"index": tex_idx}

            mat_obj: dict = {"pbrMetallicRoughness": pbr}
            mat_name = spec.get("name")
            if mat_name is None and isinstance(resolved_key, str):
                mat_name = resolved_key
            if isinstance(mat_name, str) and mat_name:
                mat_obj["name"] = mat_name
            if spec.get("double_sided"):
                mat_obj["doubleSided"] = True

            mat_idx = len(gltf_materials)
            gltf_materials.append(mat_obj)
            mat_index_by_key[cache_key] = mat_idx
            return mat_idx

        if default_mat_index is None:
            default_mat_index = len(gltf_materials)
            gltf_materials.append({
                "name": "default_clay",
                "pbrMetallicRoughness": {
                    "baseColorFactor": list(DEFAULT_CLAY_COLOR),
                    "metallicFactor": DEFAULT_METALLIC,
                    "roughnessFactor": DEFAULT_ROUGHNESS,
                },
            })
        return default_mat_index

    gltf_primitives: list[dict] = []
    total_verts = 0
    total_tris = 0

    for prim in non_empty:
        positions = list(prim["positions"])
        indices = list(prim["indices"])
        vert_count = len(positions)
        total_verts += vert_count
        total_tris += len(indices) // 3

        normals_in = prim.get("normals")
        if normals_in is None or len(normals_in) == 0:
            normals = _compute_smooth_normals(positions, indices)
        elif len(normals_in) == vert_count:
            normals = [
                _normalize3(float(n[0]), float(n[1]), float(n[2]))
                for n in normals_in
            ]
        else:
            raise GlbError(
                f"normals length ({len(normals_in)}) does not match positions length ({vert_count})"
            )

        # POSITION
        pos_bytes, pos_f32 = _pack_f32(positions, 3)
        xs = pos_f32[0::3]
        ys = pos_f32[1::3]
        zs = pos_f32[2::3]
        pos_min = [_clean_zero(min(xs)), _clean_zero(min(ys)), _clean_zero(min(zs))]
        pos_max = [_clean_zero(max(xs)), _clean_zero(max(ys)), _clean_zero(max(zs))]
        pos_bv = _append_buffer_view(bin_buf, buffer_views, pos_bytes, target=TARGET_ARRAY_BUFFER)
        pos_acc = len(accessors)
        accessors.append({
            "bufferView": pos_bv,
            "componentType": COMPONENT_FLOAT,
            "count": vert_count,
            "type": "VEC3",
            "min": pos_min,
            "max": pos_max,
        })

        # NORMAL
        nrm_bytes, _ = _pack_f32(normals, 3)
        nrm_bv = _append_buffer_view(bin_buf, buffer_views, nrm_bytes, target=TARGET_ARRAY_BUFFER)
        nrm_acc = len(accessors)
        accessors.append({
            "bufferView": nrm_bv,
            "componentType": COMPONENT_FLOAT,
            "count": vert_count,
            "type": "VEC3",
        })

        attributes: dict[str, int] = {
            "POSITION": pos_acc,
            "NORMAL": nrm_acc,
        }

        # TEXCOORD_0 (only when non-empty uvs match vertex count)
        uvs_in = prim.get("uvs")
        if uvs_in is not None and len(uvs_in) > 0:
            if len(uvs_in) != vert_count:
                raise GlbError(
                    f"uvs length ({len(uvs_in)}) does not match positions length ({vert_count})"
                )
            clean_uvs = [
                (
                    _clean_zero(float(uv[0])) if math.isfinite(float(uv[0])) else 0.0,
                    _clean_zero(float(uv[1])) if math.isfinite(float(uv[1])) else 0.0,
                )
                for uv in uvs_in
            ]
            uv_bytes, _ = _pack_f32(clean_uvs, 2)
            uv_bv = _append_buffer_view(bin_buf, buffer_views, uv_bytes, target=TARGET_ARRAY_BUFFER)
            uv_acc = len(accessors)
            accessors.append({
                "bufferView": uv_bv,
                "componentType": COMPONENT_FLOAT,
                "count": vert_count,
                "type": "VEC2",
            })
            attributes["TEXCOORD_0"] = uv_acc

        # INDICES
        idx_bytes, idx_comp_type = _pack_indices(indices, vert_count)
        idx_bv = _append_buffer_view(bin_buf, buffer_views, idx_bytes, target=TARGET_ELEMENT_ARRAY_BUFFER)
        idx_acc = len(accessors)
        accessors.append({
            "bufferView": idx_bv,
            "componentType": idx_comp_type,
            "count": len(indices),
            "type": "SCALAR",
        })

        mat_idx = get_material_index(prim, has_uv="TEXCOORD_0" in attributes)

        gltf_primitives.append({
            "attributes": attributes,
            "indices": idx_acc,
            "material": mat_idx,
            "mode": MODE_TRIANGLES,
        })

    # Ensure BIN buffer is 4-byte aligned.
    bin_pad = (-len(bin_buf)) % 4
    if bin_pad:
        bin_buf.extend(b"\x00" * bin_pad)

    mesh_obj: dict = {"primitives": gltf_primitives}
    if len(non_empty) == 1:
        sole_name = non_empty[0].get("name")
        if isinstance(sole_name, str) and sole_name:
            mesh_obj["name"] = sole_name

    gltf: dict = {
        "asset": {
            "version": "2.0",
            "generator": generator,
        },
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0}],
        "meshes": [mesh_obj],
        "materials": gltf_materials,
        "accessors": accessors,
        "bufferViews": buffer_views,
        "buffers": [{"byteLength": len(bin_buf)}],
    }

    if textures:
        gltf["samplers"] = [dict(DEFAULT_SAMPLER)]
        gltf["images"] = images
        gltf["textures"] = textures

    json_bytes = json.dumps(gltf, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")
    json_pad = (-len(json_bytes)) % 4
    if json_pad:
        json_bytes += b" " * json_pad

    total_length = 12 + 8 + len(json_bytes) + 8 + len(bin_buf)
    header = struct.pack("<4sII", GLB_MAGIC, GLB_VERSION, total_length)
    json_hdr = struct.pack("<II", len(json_bytes), CHUNK_TYPE_JSON)
    bin_hdr = struct.pack("<II", len(bin_buf), CHUNK_TYPE_BIN)
    glb_bytes = b"".join((header, json_hdr, json_bytes, bin_hdr, bytes(bin_buf)))

    bounds = compute_bounds(non_empty)
    assert bounds is not None
    stats = {
        "bytes": len(glb_bytes),
        "triangles": total_tris,
        "vertices": total_verts,
        "primitives": len(non_empty),
        "textures": len(textures),
        "bounds": bounds,
    }
    return glb_bytes, stats


def build_glb(
    primitives: list[dict],
    materials: dict[str, dict] | None = None,
    *,
    generator: str = "Leonida glb_writer",
) -> bytes:
    """Serialize glTF-ready primitives and optional PNG materials into binary .glb bytes."""
    glb_bytes, _ = _build_glb_with_stats(primitives, materials, generator=generator)
    return glb_bytes


def write_glb(
    path: str,
    primitives: list[dict],
    materials: dict[str, dict] | None = None,
) -> dict:
    """Atomically write a binary .glb file and return geometry/texture summary stats."""
    glb_bytes, stats = _build_glb_with_stats(primitives, materials)
    out_path = os.fspath(path)
    parent = os.path.dirname(os.path.abspath(out_path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    tmp_path = out_path + ".tmp"
    try:
        with open(tmp_path, "wb") as f:
            f.write(glb_bytes)
        os.replace(tmp_path, out_path)
    except BaseException:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except OSError:
            pass
        raise
    return stats


def read_glb_summary(data: bytes) -> dict:
    """Validate binary glTF 2.0 container headers/chunks and return parsed summary."""
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise GlbError("GLB data must be bytes-like")
    buf = memoryview(data)
    actual_len = len(buf)
    if actual_len < 20:
        raise GlbError(f"GLB data too short: {actual_len} bytes")

    magic, version, total_length = struct.unpack_from("<4sII", buf, 0)
    if magic != GLB_MAGIC:
        raise GlbError(f"invalid GLB magic: {magic!r}")
    if version != GLB_VERSION:
        raise GlbError(f"unsupported GLB version: {version}")
    if total_length != actual_len:
        raise GlbError(f"GLB total length mismatch: header={total_length}, actual={actual_len}")
    if total_length % 4 != 0:
        raise GlbError(f"GLB total length not 4-byte aligned: {total_length}")

    json_len, json_type = struct.unpack_from("<II", buf, 12)
    if json_type != CHUNK_TYPE_JSON:
        raise GlbError(f"first GLB chunk must be JSON (0x{CHUNK_TYPE_JSON:08X}), got 0x{json_type:08X}")
    if json_len % 4 != 0:
        raise GlbError(f"JSON chunk length not 4-byte aligned: {json_len}")
    if 20 + json_len > total_length:
        raise GlbError(f"JSON chunk length {json_len} exceeds total length {total_length}")

    json_raw = bytes(buf[20:20 + json_len])
    try:
        gltf = json.loads(json_raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GlbError(f"invalid JSON chunk: {exc}") from exc
    if not isinstance(gltf, dict):
        raise GlbError("GLB JSON root must be a JSON object")

    offset = 20 + json_len
    bin_bytes = 0
    if offset < total_length:
        if offset + 8 > total_length:
            raise GlbError("truncated BIN chunk header")
        bin_len, bin_type = struct.unpack_from("<II", buf, offset)
        if bin_type != CHUNK_TYPE_BIN:
            raise GlbError(f"second GLB chunk must be BIN (0x{CHUNK_TYPE_BIN:08X}), got 0x{bin_type:08X}")
        if bin_len % 4 != 0:
            raise GlbError(f"BIN chunk length not 4-byte aligned: {bin_len}")
        if offset + 8 + bin_len > total_length:
            raise GlbError(f"BIN chunk length {bin_len} exceeds total length {total_length}")
        bin_bytes = bin_len
        offset += 8 + bin_len

        while offset < total_length:
            if offset + 8 > total_length:
                raise GlbError("truncated trailing GLB chunk header")
            chunk_len, _chunk_type = struct.unpack_from("<II", buf, offset)
            if chunk_len % 4 != 0:
                raise GlbError(f"trailing chunk length not 4-byte aligned: {chunk_len}")
            if offset + 8 + chunk_len > total_length:
                raise GlbError("trailing chunk exceeds GLB total length")
            offset += 8 + chunk_len

    buf0_limit = bin_bytes
    buffers = gltf.get("buffers")
    if isinstance(buffers, list) and buffers:
        buf0 = buffers[0]
        if isinstance(buf0, dict) and "uri" not in buf0:
            declared = buf0.get("byteLength", 0)
            if isinstance(declared, bool) or not isinstance(declared, int) or declared < 0 or declared > bin_bytes:
                raise GlbError(f"buffers[0].byteLength ({declared}) exceeds BIN chunk ({bin_bytes})")
            buf0_limit = declared

    buffer_views = gltf.get("bufferViews")
    if isinstance(buffer_views, list):
        for i, bv in enumerate(buffer_views):
            if not isinstance(bv, dict):
                raise GlbError(f"bufferViews[{i}] is not an object")
            if bv.get("buffer", 0) == 0:
                bv_off = bv.get("byteOffset", 0)
                bv_len = bv.get("byteLength", -1)
                if (
                    isinstance(bv_off, bool)
                    or isinstance(bv_len, bool)
                    or not isinstance(bv_off, int)
                    or not isinstance(bv_len, int)
                    or bv_off < 0
                    or bv_len < 0
                ):
                    raise GlbError(f"invalid byteOffset/byteLength in bufferViews[{i}]")
                if bv_off + bv_len > buf0_limit:
                    raise GlbError(
                        f"bufferViews[{i}] exceeds buffer 0 ({bv_off}+{bv_len} > {buf0_limit})"
                    )

    return {
        "version": version,
        "total_bytes": total_length,
        "json": gltf,
        "bin_bytes": bin_bytes,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Convert a Wavefront OBJ file to binary glTF 2.0 (.glb)."
    )
    parser.add_argument("in_obj", help="Input Wavefront .obj file path")
    parser.add_argument("out_glb", help="Output binary .glb file path")
    parser.add_argument(
        "--up",
        choices=("z", "y"),
        default="z",
        help="Source up axis: 'z' converts RAGE Z-up to glTF Y-up (default), 'y' keeps Y-up",
    )
    args = parser.parse_args(argv)

    try:
        primitives = parse_obj(args.in_obj, is_path=True, up_axis=args.up)
        stats = write_glb(args.out_glb, primitives)
    except (GlbError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(
        f"wrote {args.out_glb}: {stats['bytes']} bytes, "
        f"{stats['primitives']} primitives, {stats['vertices']} verts, "
        f"{stats['triangles']} tris"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
