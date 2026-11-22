"""Build streamed spatial world.json and .glb chunks from rpf9dump asset exports.

Consumes rpf9dump model manifests (models/manifest.json + .obj + <stem>_tex/*.dds) and
optional TTPF texture manifests (manifest.json + *.dds), groups geometry into spatial
or grid-laid-out XZ chunks, embeds decoded PNG textures into binary glTF 2.0 (.glb)
chunks, and writes world.json atomically for viewer/serve.py + viewer/index.html.
Chunks are built and verified in a staging directory first, so a build that fails or is
interrupted leaves the previous world.json and the chunks it references as they were.

Usage:
    python -I tools/world_build.py <assets_dir> --out <world_dir> [options]
"""
from __future__ import annotations

import argparse
import array
from concurrent.futures import ThreadPoolExecutor
import json
import math
import os
import re
import shutil
import statistics
import sys
import tempfile
import threading

_TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
if _TOOLS_DIR not in sys.path:
    sys.path.insert(0, _TOOLS_DIR)

from file_mode import published_mode  # noqa: E402

try:
    import dds_png
except ImportError:  # pragma: no cover - sibling module may not be present during --help
    dds_png = None

try:
    import glb_writer
except ImportError:  # pragma: no cover - sibling module may not be present during --help
    glb_writer = None


GRID_GAP = 4.0
SPAWN_SPAN_CAP = 1200.0


def _get_dds_png():
    global dds_png
    if dds_png is None:
        import dds_png as _dds_png
        dds_png = _dds_png
    return dds_png


def _get_glb_writer():
    global glb_writer
    if glb_writer is None:
        import glb_writer as _glb_writer
        glb_writer = _glb_writer
    return glb_writer


def _load_json_dict(path: str) -> dict | None:
    try:
        with open(path, "r", encoding="utf-8-sig") as f:
            data = json.load(f)
    except (OSError, ValueError, TypeError):
        return None
    return data if isinstance(data, dict) else None


def _list_field(record: dict, key: str) -> list:
    """Read an optional JSON array, treating other types as absent."""
    value = record.get(key)
    return value if isinstance(value, list) else []


def _is_model_manifest(path: str) -> bool:
    data = _load_json_dict(path)
    if data is None:
        return False
    items = data.get("Items")
    if not isinstance(items, list):
        return False
    # rpf9dump textures mode writes textures/manifest.json with Items containing
    # {"Dir": ..., "Textures": [...]} and no "File" field. Exclude those.
    parent_name = os.path.basename(os.path.dirname(os.path.abspath(path))).lower()
    if parent_name == "textures":
        return False
    if items:
        has_file = any(isinstance(it, dict) and "File" in it for it in items)
        all_tex_dicts = all(
            isinstance(it, dict) and "Dir" in it and "File" not in it
            for it in items
        )
        if not has_file and all_tex_dicts:
            return False
    return True


def discover_model_manifests(assets_dir: str) -> list[str]:
    """Find all rpf9dump model manifest.json files under assets_dir.

    Handles:
      - a single models/ directory containing manifest.json
      - an archive directory containing models/manifest.json
      - a whole <work>/assets tree containing */models/manifest.json
    Returns sorted unique absolute paths.
    """
    abs_root = os.path.abspath(assets_dir)
    if not os.path.exists(abs_root):
        return []

    found: dict[str, str] = {}

    if os.path.isfile(abs_root):
        if _is_model_manifest(abs_root):
            found[os.path.normcase(abs_root)] = abs_root
        return sorted(found.values())

    # 1. Direct models/ directory (manifest.json directly inside assets_dir).
    direct = os.path.join(abs_root, "manifest.json")
    if os.path.isfile(direct) and _is_model_manifest(direct):
        abs_direct = os.path.abspath(direct)
        found[os.path.normcase(abs_direct)] = abs_direct

    # 2. Archive directory (models/manifest.json) or whole assets tree.
    for dirpath, dirnames, filenames in os.walk(abs_root):
        dirnames.sort()
        if "manifest.json" in filenames and os.path.basename(dirpath).lower() == "models":
            cand = os.path.abspath(os.path.join(dirpath, "manifest.json"))
            if _is_model_manifest(cand):
                found[os.path.normcase(cand)] = cand

    return sorted(found.values())


def _is_nonzero_key(key: str) -> bool:
    """Return True for a non-empty texture key/hash that is not an all-zero sentinel."""
    return bool(key) and bool(key.strip("0"))


def index_ttpf_textures(assets_dir: str) -> dict[str, str]:
    """Index TTPF textures under assets_dir: uppercase 16-hex AssetKey -> absolute .dds path."""
    abs_root = os.path.abspath(assets_dir)
    index: dict[str, str] = {}
    if not os.path.isdir(abs_root):
        return index

    for dirpath, dirnames, filenames in os.walk(abs_root):
        dirnames.sort()
        if "manifest.json" not in filenames:
            continue
        manifest_path = os.path.join(dirpath, "manifest.json")
        data = _load_json_dict(manifest_path)
        if data is None:
            continue
        textures = data.get("Textures")
        if not isinstance(textures, list):
            continue
        for entry in textures:
            if not isinstance(entry, dict):
                continue
            raw_key = entry.get("AssetKey")
            raw_file = entry.get("File")
            if isinstance(raw_key, int) and not isinstance(raw_key, bool):
                key = f"{raw_key:016X}"
            elif isinstance(raw_key, str):
                key = raw_key.strip().upper()
            else:
                continue
            if not _is_nonzero_key(key) or not isinstance(raw_file, str):
                continue
            rel_file = raw_file.strip()
            if not rel_file:
                continue
            dds_path = os.path.abspath(os.path.join(dirpath, rel_file))
            if os.path.isfile(dds_path):
                index.setdefault(key, dds_path)

    return index


def _scan_embedded_dir(tex_dir: str) -> tuple[list[str], dict[int, str]]:
    """Return sorted .dds filenames in tex_dir and prefix-index -> absolute path map."""
    if not tex_dir or not os.path.isdir(tex_dir):
        return [], {}
    try:
        entries = sorted(os.listdir(tex_dir))
    except OSError:
        return [], {}
    dds_files: list[str] = []
    disk_by_index: dict[int, str] = {}
    for fname in entries:
        if not fname.lower().endswith(".dds"):
            continue
        full = os.path.abspath(os.path.join(tex_dir, fname))
        if not os.path.isfile(full):
            continue
        dds_files.append(fname)
        head, sep, _ = fname.partition("_")
        if sep and head.isdigit():
            disk_by_index.setdefault(int(head), full)
    return dds_files, disk_by_index


def _find_embedded_dds(tex_dir: str, dds_files: list[str], idx: int) -> str | None:
    prefix = f"{idx:03d}_"
    for fname in dds_files:
        if fname.startswith(prefix):
            return os.path.abspath(os.path.join(tex_dir, fname))
    return None


_PARAM_HASH_RE = re.compile(r"(?:0[xX])?([0-9A-Fa-f]{8})")


def _parse_param_hash(value: object) -> str:
    """Normalize a 32-bit shader parameter hash ("C8E8C282" or "0xc8e8c282") to uppercase hex."""
    if isinstance(value, str):
        match = _PARAM_HASH_RE.fullmatch(value.strip())
        if match:
            return match.group(1).upper()
    raise ValueError(f"base_color_param must be eight hex digits with optional 0x, got {value!r}")


def _ref_param_hash(ref: dict) -> str | None:
    """Return a texture reference's Param as uppercase 8-hex, or None if absent/malformed."""
    raw = ref.get("Param")
    if isinstance(raw, int) and not isinstance(raw, bool):
        return f"{raw:08X}" if 0 <= raw <= 0xFFFFFFFF else None
    if isinstance(raw, str):
        match = _PARAM_HASH_RE.fullmatch(raw.strip())
        return match.group(1).upper() if match else None
    return None


def resolve_model_textures(
    models_dir: str,
    item: dict,
    ttpf_index: dict[str, str],
    base_color_param: str | None = None,
) -> dict[int, str]:
    """Resolve shader_index -> existing absolute .dds path for a model manifest item.

    With base_color_param (normalized uppercase 8-hex), only texture references whose Param
    matches are considered and the empty-Materials fallback is disabled; unmatched shaders
    are left out so they render as clay.
    """
    result: dict[int, str] = {}
    file_name = item.get("File")
    stem = os.path.splitext(file_name)[0] if isinstance(file_name, str) and file_name else ""
    tex_dir = os.path.join(models_dir, f"{stem}_tex") if stem else ""
    dds_files, disk_by_index = _scan_embedded_dir(tex_dir)

    emb_by_index: dict[int, str] = {}
    emb_by_key: dict[str, str] = {}
    emb_by_hash: dict[str, str] = {}

    for emb in _list_field(item, "Embedded"):
        if not isinstance(emb, dict):
            continue
        raw_idx = emb.get("Index")
        if raw_idx is None or isinstance(raw_idx, bool):
            continue
        try:
            idx = int(raw_idx)
        except (TypeError, ValueError):
            continue
        dds_path = _find_embedded_dds(tex_dir, dds_files, idx) if dds_files else None
        if not dds_path:
            continue
        emb_by_index[idx] = dds_path
        raw_key = emb.get("Key")
        if isinstance(raw_key, str):
            norm_key = raw_key.strip().upper()
            if _is_nonzero_key(norm_key):
                emb_by_key.setdefault(norm_key, dds_path)
        raw_hash = emb.get("Hash")
        if isinstance(raw_hash, str):
            norm_hash = raw_hash.strip().upper()
            if _is_nonzero_key(norm_hash):
                emb_by_hash.setdefault(norm_hash, dds_path)

    materials = _list_field(item, "Materials")
    if materials:
        for mat in materials:
            if not isinstance(mat, dict):
                continue
            raw_sidx = mat.get("ShaderIndex")
            if raw_sidx is None or isinstance(raw_sidx, bool):
                continue
            try:
                sidx = int(raw_sidx)
            except (TypeError, ValueError):
                continue
            if sidx in result:
                continue

            chosen: str | None = None
            for ref in _list_field(mat, "Textures"):
                if not isinstance(ref, dict):
                    continue
                if base_color_param is not None and _ref_param_hash(ref) != base_color_param:
                    continue
                # 1. Embedded by TextureIndex
                tidx = ref.get("TextureIndex")
                if isinstance(tidx, int) and not isinstance(tidx, bool) and tidx >= 0:
                    cand = emb_by_index.get(tidx) or disk_by_index.get(tidx)
                    if cand and os.path.isfile(cand):
                        chosen = cand
                        break
                # 2. Embedded by Id64 -> Key
                raw_id64 = ref.get("Id64")
                id64 = raw_id64.strip().upper() if isinstance(raw_id64, str) else ""
                if _is_nonzero_key(id64) and id64 in emb_by_key:
                    cand = emb_by_key[id64]
                    if os.path.isfile(cand):
                        chosen = cand
                        break
                # 3. Embedded by Name -> Hash
                raw_name = ref.get("Name")
                name_hash = raw_name.strip().upper() if isinstance(raw_name, str) else ""
                if _is_nonzero_key(name_hash) and name_hash in emb_by_hash:
                    cand = emb_by_hash[name_hash]
                    if os.path.isfile(cand):
                        chosen = cand
                        break
                # 4. External TTPF by uppercase Id64
                if _is_nonzero_key(id64) and id64 in ttpf_index:
                    cand = ttpf_index[id64]
                    if isinstance(cand, str) and os.path.isfile(cand):
                        chosen = os.path.abspath(cand)
                        break

            if chosen is not None:
                result[sidx] = chosen
    elif dds_files and base_color_param is None:
        # Fallback when Materials is empty/missing and no parameter is selected: map embedded indices and 0.
        for idx, path in emb_by_index.items():
            result[idx] = path
        for idx, path in disk_by_index.items():
            result.setdefault(idx, path)
        result.setdefault(0, os.path.abspath(os.path.join(tex_dir, dds_files[0])))

    return result


def _translate_primitives(prims: list[dict], dx: float, dy: float, dz: float) -> None:
    """Translate primitive vertex positions in-place by (dx, dy, dz)."""
    if dx == 0.0 and dy == 0.0 and dz == 0.0:
        return
    for p in prims:
        pos = p.get("positions")
        if pos is None:
            continue
        if isinstance(pos, (bytes, bytearray, memoryview)):
            n_floats = len(pos) // 4
            if n_floats >= 3 and n_floats % 3 == 0:
                vals = array.array("f")
                vals.frombytes(pos)
                if sys.byteorder != "little":
                    vals.byteswap()
                for i in range(0, len(vals), 3):
                    vals[i] += dx
                    vals[i + 1] += dy
                    vals[i + 2] += dz
                if sys.byteorder != "little":
                    vals.byteswap()
                raw = vals.tobytes()
                p["positions"] = bytearray(raw) if isinstance(pos, bytearray) else raw
        elif isinstance(pos, array.array):
            for i in range(0, len(pos) - 2, 3):
                pos[i] += dx
                pos[i + 1] += dy
                pos[i + 2] += dz
        elif isinstance(pos, list):
            if not pos:
                continue
            first = pos[0]
            if isinstance(first, (int, float)):
                for i in range(0, len(pos) - 2, 3):
                    pos[i] += dx
                    pos[i + 1] += dy
                    pos[i + 2] += dz
            elif isinstance(first, tuple):
                p["positions"] = [(v[0] + dx, v[1] + dy, v[2] + dz) for v in pos]
            elif isinstance(first, list):
                for v in pos:
                    v[0] += dx
                    v[1] += dy
                    v[2] += dz
        elif isinstance(pos, tuple):
            if not pos:
                continue
            first = pos[0]
            if isinstance(first, (int, float)):
                out = list(pos)
                for i in range(0, len(out) - 2, 3):
                    out[i] += dx
                    out[i + 1] += dy
                    out[i + 2] += dz
                p["positions"] = tuple(out)
            else:
                p["positions"] = tuple((v[0] + dx, v[1] + dy, v[2] + dz) for v in pos)

        # Keep optional cached bounds consistent if the primitive carries them.
        if isinstance(p.get("min"), list) and len(p["min"]) == 3:
            p["min"] = [p["min"][0] + dx, p["min"][1] + dy, p["min"][2] + dz]
        if isinstance(p.get("max"), list) and len(p["max"]) == 3:
            p["max"] = [p["max"][0] + dx, p["max"][1] + dy, p["max"][2] + dz]
        if isinstance(p.get("bounds"), list) and len(p["bounds"]) == 6:
            b = p["bounds"]
            p["bounds"] = [
                b[0] + dx, b[1] + dy, b[2] + dz,
                b[3] + dx, b[4] + dy, b[5] + dz,
            ]


def _apply_grid_layout(models: list[dict], glb_mod) -> None:
    """Arrange models in non-overlapping XZ rows (4.0 m gap, min Y at 0.0)."""
    if not models:
        return
    cols = max(1, math.ceil(math.sqrt(len(models))))
    cur_x = 0.0
    cur_z = 0.0
    row_depth = 0.0
    col = 0

    for m in models:
        b = m["bounds"]
        sx = b[3] - b[0]
        sz = b[5] - b[2]
        dx = cur_x - b[0]
        dy = -b[1]
        dz = cur_z - b[2]
        _translate_primitives(m["prims"], dx, dy, dz)
        new_bounds = glb_mod.compute_bounds(m["prims"])
        if new_bounds is not None and len(new_bounds) == 6:
            m["bounds"] = new_bounds
        else:
            m["bounds"] = [
                b[0] + dx, b[1] + dy, b[2] + dz,
                b[3] + dx, b[4] + dy, b[5] + dz,
            ]

        row_depth = max(row_depth, sz)
        col += 1
        if col >= cols:
            col = 0
            cur_x = 0.0
            cur_z += row_depth + GRID_GAP
            row_depth = 0.0
        else:
            cur_x += sx + GRID_GAP


def _compute_spawn(chunks: list[dict]) -> list[float]:
    if not chunks:
        return [0.0, 20.0, 40.0]

    centers = [
        (0.5 * (c["bounds"][0] + c["bounds"][3]), 0.5 * (c["bounds"][2] + c["bounds"][5]))
        for c in chunks
    ]
    med_x = statistics.median(cx for cx, _ in centers)
    med_z = statistics.median(cz for _, cz in centers)

    best_idx = 0
    best_d2 = float("inf")
    for i, (cx, cz) in enumerate(centers):
        d2 = (cx - med_x) ** 2 + (cz - med_z) ** 2
        if d2 < best_d2:
            best_d2 = d2
            best_idx = i

    b = chunks[best_idx]["bounds"]
    cx = 0.5 * (b[0] + b[3])
    max_y = b[4]
    max_z = b[5]
    # The viewer loads chunks within 1400 m. Limiting the offsets to 300 m up and 720 m back
    # keeps the camera within 780 m of the chosen chunk's bounds, even for very large models.
    span = min(max(b[3] - b[0], b[4] - b[1], b[5] - b[2]), SPAWN_SPAN_CAP)
    return [
        float(cx),
        float(max_y + max(8.0, 0.25 * span)),
        float(max_z + max(15.0, 0.6 * span)),
    ]


def _write_json_atomic(path: str, data: dict) -> None:
    tmp_path = path + ".tmp"
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, allow_nan=False)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except OSError:
            pass
        raise


# Cell coordinates are written as decimal integers, without leading zeros.
_CHUNK_FILE_RE = re.compile(r"c_(?:0|-?[1-9][0-9]*)_(?:0|-?[1-9][0-9]*)\.glb(?:\.tmp)?")


def _prune_stale_chunks(chunks_dir: str, keep_filenames: set[str]) -> None:
    """Remove stale chunk files and partial writes left by earlier builds."""
    try:
        entries = os.listdir(chunks_dir)
    except OSError:
        return
    for name in entries:
        if name in keep_filenames:
            continue
        if not _CHUNK_FILE_RE.fullmatch(name):
            continue
        full = os.path.join(chunks_dir, name)
        if os.path.isfile(full):
            try:
                os.remove(full)
            except OSError:
                pass


# A build stages its chunks in <out>/<prefix><random>/ (the same volume as <out>/chunks, so
# publishing is a rename) and keeps the chunks it replaces in <staging>/previous/ until
# world.json is committed.
CHUNK_STAGING_PREFIX = ".chunks-staging-"
_STAGING_BACKUP = "previous"


def _remove_stale_staging(out_dir: str) -> None:
    """Remove staging directories left behind by builds that were killed."""
    try:
        entries = os.listdir(out_dir)
    except OSError:
        return
    for name in entries:
        path = os.path.join(out_dir, name)
        if name.startswith(CHUNK_STAGING_PREFIX) and os.path.isdir(path) and not os.path.islink(path):
            shutil.rmtree(path, ignore_errors=True)


def _verify_staged_chunk(path: str, expected_size: int, glb_mod) -> None:
    """Raise GlbError unless the staged file is the complete GLB that write_glb reported."""
    with open(path, "rb") as f:
        data = f.read()
    if len(data) != expected_size:
        raise glb_mod.GlbError(f"staged chunk {path} is {len(data)} bytes, expected {expected_size}")
    glb_mod.read_glb_summary(data)


def _publish_world(
    staging_dir: str,
    chunks_dir: str,
    chunk_names: list[str],
    world_path: str,
    world_doc: dict,
    glb_mod,
) -> None:
    """Move the verified staged chunks into chunks_dir, then commit world.json.

    Every live chunk that gets replaced is first moved into <staging>/previous/. If a move or
    the world.json write fails, the new chunks are taken out again and the previous ones put
    back, so the old world.json keeps matching its chunks. A previous chunk that cannot be put
    back stays in <staging>/previous/ and is named in the raised error.
    """
    backup_dir = os.path.join(staging_dir, _STAGING_BACKUP)
    os.makedirs(backup_dir, exist_ok=True)
    os.makedirs(chunks_dir, exist_ok=True)
    backed_up: list[str] = []
    placed: list[str] = []
    try:
        for name in chunk_names:
            live = os.path.join(chunks_dir, name)
            staged = os.path.join(staging_dir, name)
            os.chmod(staged, published_mode(live))
            if os.path.lexists(live):
                glb_mod._replace(live, os.path.join(backup_dir, name))
                backed_up.append(name)
            glb_mod._replace(staged, live)
            placed.append(name)
        _write_json_atomic(world_path, world_doc)
    except BaseException as exc:
        unrestored = []
        restore = set(backed_up)
        for name in reversed(placed):
            if name not in restore:
                try:
                    os.remove(os.path.join(chunks_dir, name))
                except OSError:
                    pass
        for name in reversed(backed_up):
            try:
                glb_mod._replace(os.path.join(backup_dir, name), os.path.join(chunks_dir, name))
            except OSError:
                unrestored.append(name)
        if unrestored:
            raise OSError(
                f"publishing chunks failed ({exc!r}) and {len(unrestored)} previous chunk(s) could not be "
                f"put back; they are in {backup_dir}: {', '.join(sorted(unrestored))}"
            ) from exc
        raise


def build_world(
    assets_dir: str,
    out_dir: str,
    *,
    chunk_size: float = 256.0,
    grid_layout: bool = False,
    up_axis: str = "z",
    max_tex_size: int = 512,
    textures: bool = True,
    limit: int | None = None,
    jobs: int = 1,
    base_color_param: str | None = None,
) -> dict:
    """Build world.json and chunk .glb files from rpf9dump asset exports.

    Unless limit=0 explicitly requests an empty world, raise ValueError before changing
    the output if no readable model manifests or no usable selected models are found.
    Individual unusable models are still skipped when other selected models can load.

    Chunks are written to a staging directory inside out_dir and verified before any live
    chunk is replaced. If writing, verifying or publishing fails, or the build is interrupted,
    the staging directory is removed and the previous world.json and chunks stay as they were.
    Only a hard kill during the final renames (one per chunk, then world.json) can leave new
    chunks next to the old world.json; the next build then removes the leftover staging.
    """
    if chunk_size <= 0 or not math.isfinite(chunk_size):
        raise ValueError(f"chunk_size must be positive, got {chunk_size!r}")
    if up_axis not in ("z", "y"):
        raise ValueError(f"up_axis must be 'z' or 'y', got {up_axis!r}")
    if limit is not None and limit < 0:
        raise ValueError(f"limit must be non-negative, got {limit!r}")
    if max_tex_size < 0:
        raise ValueError(f"max_tex_size must be non-negative, got {max_tex_size!r}")
    if base_color_param is not None:
        base_color_param = _parse_param_hash(base_color_param)
    workers = max(1, int(jobs))

    if not os.path.exists(assets_dir):
        raise FileNotFoundError(f"assets path does not exist: {assets_dir}")

    glb_mod = _get_glb_writer()
    dds_mod = _get_dds_png() if textures else None

    manifests = discover_model_manifests(assets_dir)
    if not manifests and limit != 0:
        raise ValueError(f"No readable model manifests found in {assets_dir}; output left unchanged")
    entries: list[tuple[str, dict]] = []
    for mpath in manifests:
        data = _load_json_dict(mpath)
        if data is None:
            continue
        items = data.get("Items")
        if not isinstance(items, list):
            continue
        models_dir = os.path.dirname(mpath)
        for item in items:
            if isinstance(item, dict):
                entries.append((models_dir, item))

    if limit is not None:
        entries = entries[:limit]

    ttpf_index = index_ttpf_textures(assets_dir) if textures else {}

    loaded_models: list[dict] = []
    for models_dir, item in entries:
        rel_file = item.get("File")
        if not isinstance(rel_file, str) or not rel_file.strip():
            continue
        obj_path = os.path.join(models_dir, rel_file)
        if not os.path.isfile(obj_path):
            continue
        try:
            prims = glb_mod.parse_obj(obj_path, is_path=True, up_axis=up_axis)
        except (OSError, glb_mod.GlbError, ValueError):
            continue
        if not prims:
            continue
        bounds = glb_mod.compute_bounds(prims)
        if bounds is None or len(bounds) != 6 or not all(math.isfinite(v) for v in bounds):
            continue

        tex_map = (
            resolve_model_textures(models_dir, item, ttpf_index, base_color_param)
            if textures
            else {}
        )
        mesh_shaders = _list_field(item, "MeshShaders")
        mat_to_dds: dict[str, str] = {}

        for p in prims:
            gidx = p.get("group_index", 0)
            if isinstance(gidx, int) and not isinstance(gidx, bool) and 0 <= gidx < len(mesh_shaders):
                sidx = mesh_shaders[gidx]
            else:
                sidx = gidx
            has_uv = bool(p.get("uvs"))
            if (
                textures
                and has_uv
                and isinstance(sidx, int)
                and not isinstance(sidx, bool)
                and sidx in tex_map
            ):
                dds_path = os.path.abspath(tex_map[sidx])
                mat_key = f"tex_{os.path.normcase(dds_path)}"
                p["material"] = mat_key
                mat_to_dds[mat_key] = dds_path
            else:
                p["material"] = ""

        loaded_models.append({
            "prims": prims,
            "bounds": bounds,
            "mat_to_dds": mat_to_dds,
        })

    if not loaded_models and limit != 0:
        raise ValueError(
            f"No usable models in {assets_dir} ({len(entries)} selected manifest items); "
            "check model File fields and OBJ geometry; output left unchanged"
        )

    if grid_layout:
        _apply_grid_layout(loaded_models, glb_mod)

    cells: dict[tuple[int, int], list[dict]] = {}
    for m in loaded_models:
        b = m["bounds"]
        cx = 0.5 * (b[0] + b[3])
        cz = 0.5 * (b[2] + b[5])
        cell_key = (math.floor(cx / chunk_size), math.floor(cz / chunk_size))
        cells.setdefault(cell_key, []).append(m)

    abs_out = os.path.abspath(out_dir)
    chunks_dir = os.path.join(abs_out, "chunks")
    created_dirs = [d for d in (abs_out, chunks_dir) if not os.path.isdir(d)]
    os.makedirs(abs_out, exist_ok=True)
    _remove_stale_staging(abs_out)
    staging_dir = tempfile.mkdtemp(prefix=CHUNK_STAGING_PREFIX, dir=abs_out)
    try:
        world_doc = _build_staged_world(
            staging_dir, chunks_dir, os.path.join(abs_out, "world.json"), cells, loaded_models,
            glb_mod=glb_mod, dds_mod=dds_mod, textures=textures, max_tex_size=max_tex_size,
            workers=workers, chunk_size=chunk_size, grid_layout=grid_layout,
        )
    except BaseException:
        backup_dir = os.path.join(staging_dir, _STAGING_BACKUP)
        if not (os.path.isdir(backup_dir) and os.listdir(backup_dir)):
            shutil.rmtree(staging_dir, ignore_errors=True)
        for d in reversed(created_dirs):
            try:
                os.rmdir(d)  # only if this build created it and left it empty
            except OSError:
                pass
        raise
    shutil.rmtree(staging_dir, ignore_errors=True)
    _prune_stale_chunks(chunks_dir, {f"{c['id']}.glb" for c in world_doc["chunks"]})
    return world_doc


def _build_staged_world(
    staging_dir: str,
    chunks_dir: str,
    world_path: str,
    cells: dict[tuple[int, int], list[dict]],
    loaded_models: list[dict],
    *,
    glb_mod,
    dds_mod,
    textures: bool,
    max_tex_size: int,
    workers: int,
    chunk_size: float,
    grid_layout: bool,
) -> dict:
    """Write every chunk into staging_dir, verify them all, then publish them with world.json."""
    # Thread-safe single-flight PNG decode cache keyed by normalized DDS path.
    tex_lock = threading.Lock()
    tex_cache: dict[str, bytes | None] = {}
    tex_inflight: dict[str, threading.Event] = {}

    def decode_dds_cached(dds_path: str) -> bytes | None:
        assert dds_mod is not None
        norm = os.path.normcase(os.path.abspath(dds_path))
        with tex_lock:
            if norm in tex_cache:
                return tex_cache[norm]
            ev = tex_inflight.get(norm)
            if ev is None:
                ev = threading.Event()
                tex_inflight[norm] = ev
                owner = True
            else:
                owner = False
        if not owner:
            ev.wait()
            with tex_lock:
                return tex_cache.get(norm)

        png_bytes: bytes | None = None
        try:
            png_bytes = dds_mod.dds_file_to_png_bytes(dds_path, max_size=max_tex_size)
        except (dds_mod.DdsError, OSError):
            png_bytes = None
        finally:
            with tex_lock:
                tex_cache[norm] = png_bytes
                tex_inflight.pop(norm, None)
            ev.set()
        return png_bytes

    def build_cell(cell_key: tuple[int, int], cell_models: list[dict]) -> tuple[dict, int, int, int]:
        ix, iz = cell_key
        cell_prims: list[dict] = []
        cell_dds: dict[str, str] = {}
        for m in cell_models:
            cell_prims.extend(m["prims"])
            cell_dds.update(m["mat_to_dds"])

        materials: dict[str, dict] = {}
        skipped_keys: set[str] = set()
        cell_skipped = 0

        if textures and cell_dds:
            for mat_key in sorted(cell_dds):
                dds_path = cell_dds[mat_key]
                png_bytes = decode_dds_cached(dds_path)
                if png_bytes is None:
                    cell_skipped += 1
                    skipped_keys.add(mat_key)
                else:
                    materials[mat_key] = {
                        "name": os.path.splitext(os.path.basename(dds_path))[0],
                        "png_bytes": png_bytes,
                        "base_color_factor": [1.0, 1.0, 1.0, 1.0],
                        "double_sided": True,
                    }

        if skipped_keys:
            for p in cell_prims:
                if p.get("material") in skipped_keys:
                    p["material"] = ""

        chunk_id = f"c_{ix}_{iz}"
        glb_path = os.path.join(staging_dir, f"{chunk_id}.glb")
        info = glb_mod.write_glb(glb_path, cell_prims, materials=materials if materials else None)
        chunk_rec = {
            "id": chunk_id,
            "url": f"world/chunks/{chunk_id}.glb",
            "bounds": info["bounds"],
            "models": len(cell_models),
            "triangles": info["triangles"],
        }
        return chunk_rec, int(info.get("textures", 0)), cell_skipped, int(info["bytes"])

    sorted_cells = sorted(cells.items(), key=lambda kv: kv[0])
    chunks: list[dict] = []
    total_triangles = 0
    textures_embedded = 0
    textures_skipped = 0

    if workers == 1 or len(sorted_cells) <= 1:
        cell_results = [build_cell(k, ms) for k, ms in sorted_cells]
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(build_cell, k, ms) for k, ms in sorted_cells]
            try:
                cell_results = [f.result() for f in futures]
            except BaseException:
                for f in futures:
                    f.cancel()  # stop cells that have not started; running ones finish before cleanup
                raise

    for chunk_rec, _, _, size in cell_results:
        _verify_staged_chunk(os.path.join(staging_dir, f"{chunk_rec['id']}.glb"), size, glb_mod)

    for chunk_rec, embedded_count, skipped_count, _ in cell_results:
        chunks.append(chunk_rec)
        total_triangles += int(chunk_rec["triangles"])
        textures_embedded += embedded_count
        textures_skipped += skipped_count

    spawn = _compute_spawn(chunks)
    world_doc = {
        "spawn": spawn,
        "chunk_size": float(chunk_size),
        "layout": "grid" if grid_layout else "spatial",
        "chunks": chunks,
        "stats": {
            "models": len(loaded_models),
            "chunks": len(chunks),
            "triangles": total_triangles,
            "textures_embedded": textures_embedded,
            "textures_skipped": textures_skipped,
        },
    }

    _publish_world(staging_dir, chunks_dir, [f"{c['id']}.glb" for c in chunks], world_path, world_doc, glb_mod)
    return world_doc


def _positive_int(text: str) -> int:
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"invalid int value: {text!r}") from None
    if value < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return value


def _non_negative_int(text: str) -> int:
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"invalid int value: {text!r}") from None
    if value < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return value


def _positive_float(text: str) -> float:
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"invalid float value: {text!r}") from None
    if value <= 0 or not math.isfinite(value):
        raise argparse.ArgumentTypeError("must be a positive finite number")
    return value


def _param_hash(text: str) -> str:
    try:
        return _parse_param_hash(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"invalid 32-bit hex value: {text!r}") from None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Build spatial world.json and .glb chunks from rpf9dump asset exports."
    )
    ap.add_argument(
        "assets_dir",
        help="Path to a models/ dir, an archive dir, or a full <work>\\assets tree.",
    )
    ap.add_argument(
        "--out",
        required=True,
        metavar="WORLD_DIR",
        help="Output world directory for world.json and chunks/*.glb.",
    )
    ap.add_argument(
        "--chunk-size",
        type=_positive_float,
        default=256.0,
        metavar="METRES",
        help="Spatial chunk edge length in metres (default: %(default)s).",
    )
    ap.add_argument(
        "--grid-layout",
        action="store_true",
        help="Arrange models in non-overlapping XZ rows instead of world-space positions.",
    )
    ap.add_argument(
        "--up",
        choices=("z", "y"),
        default="z",
        help="Source OBJ up axis to convert to glTF Y-up (default: %(default)s).",
    )
    ap.add_argument(
        "--max-tex-size",
        type=_non_negative_int,
        default=512,
        metavar="PX",
        help="Maximum texture width/height in pixels; 0 disables downscaling (default: %(default)s).",
    )
    ap.add_argument(
        "--no-textures",
        action="store_true",
        help="Skip texture resolution and DDS decoding (output clay geometry only).",
    )
    ap.add_argument(
        "--limit",
        type=_non_negative_int,
        default=None,
        metavar="N",
        help="Process at most N manifest items; 0 explicitly writes an empty world and removes old chunks.",
    )
    ap.add_argument(
        "--jobs",
        type=_positive_int,
        default=1,
        metavar="N",
        help="Number of worker threads for parallel chunk building (default: %(default)s).",
    )
    ap.add_argument(
        "--base-color-param",
        type=_param_hash,
        default=None,
        metavar="HEX",
        help="Use only texture references with this 32-bit shader parameter hash as base colour "
        "(e.g. C8E8C282 for character materials); materials without a resolvable match stay clay. "
        "Default: first resolvable reference.",
    )
    args = ap.parse_args(argv)

    try:
        result = build_world(
            args.assets_dir,
            args.out,
            chunk_size=args.chunk_size,
            grid_layout=args.grid_layout,
            up_axis=args.up,
            max_tex_size=args.max_tex_size,
            textures=not args.no_textures,
            limit=args.limit,
            jobs=args.jobs,
            base_color_param=args.base_color_param,
        )
    except (FileNotFoundError, ValueError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    st = result["stats"]
    print(
        f"world built: {st['models']} models -> {st['chunks']} chunks, "
        f"{st['triangles']} tris, {st['textures_embedded']} textures "
        f"({st['textures_skipped']} skipped) -> {os.path.abspath(args.out)}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
