"""Place the tile textures of the GridAtlasInfo map grids and check what the grid documents' zInflation box covers.

The four ``GridAtlasInfo`` documents in ``common.rpf`` (docs/gamedata.md, "Grid atlases") describe world-space
texture grids of 512-unit tiles. Their tile textures are in the texture dictionaries that ``make_game.py --stages
assets`` exports, but every tile carries the same texture name (the document's ``tileTextureNameSuffix``) and the
dictionary names are not in the data, so a tile's cell is not written anywhere. This tool recovers it:

- Each grid's global fallback (``globalFallbackTextureName``) is a mosaic of the whole grid at a few pixels per
  tile. Fallback image row 0 is the grid's north edge (largest y); cell = row * columns + column.
- ``content`` grids (water height R16, water fog BC6H) compare each tile, point-sampled at the fallback's
  resolution, with every fallback cell. Grids whose tiles share their dictionaries (the two fog grids) are matched
  jointly. A tile with detail whose runner-up distance is at least twice the best is an anchor.
- ``mask`` grids (shoreline, 8-bit BGRA) are mostly one background colour, which swamps a plain distance. The
  tool compares feature masks (pixels away from the fallback's most common colour) by intersection over union and
  aligns the tiles, in archive order, to cells in increasing text order with a dynamic programme.
- Archive order: the tile dictionaries are stored in the order of their cells sorted as text of the decimal cell
  index (``sorted(cells, key=str)``). The tool counts ascending steps of the anchors under that order and three
  other spellings, then places the tiles between two anchors when their number equals the free cells between them.

``zInflation`` is checked, not measured at runtime: the tool reports each grid's vertical box (origin z +-
zInflation), the Z bounds of the MHIU and PAMH heightmaps in the same carve, and a least-squares fit of the water
height fallback against MHIU terrain heights (values above ``FIT_MIN_VALUE``), which also gives the value-to-metre
scale and confirms the north-up row order.

    python tools/world_grid_atlas.py <work> --carve <common carve> [--json FILE]

``<work>`` is the builder's work folder (``<work>/assets/<archive>/textures/manifest.json``); ``--carve`` is the
``rpf9_carve.py`` output of ``common.rpf``. The tool prints one summary line per grid. ``--json`` also writes the
cell of every placed tile: that is data derived from your own copy of the game, so keep the file local.

Unlike most tools here this one imports numpy and Pillow (Pillow decodes BC6H); it is one of the texture-tool
exceptions in AGENTS.md. The placement, ordering and fitting logic is plain Python, and its tests run on the
standard library alone.
"""
import argparse
import json
import math
import os
import struct
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import report_paths  # noqa: E402

DX10_HEADER = 148
DXGI_R16_UNORM = 56
MASK_THRESHOLD = 32          # max channel distance from the background colour that still counts as background
MASK_BLOCK_SHARE = 0.25      # share of feature pixels in a tile block for the block to count as feature
DETAIL_STD = 0.5             # a texture with a smaller standard deviation is flat
ANCHOR_RATIO = 2.0           # runner-up distance / best distance for an anchor
ANCHOR_GAP = 0.5             # and runner-up minus best at least this
FIT_MIN_VALUE = 36000        # water-height values used for the terrain fit (clearly above sea level)
HEIGHT_GRID = "WaterSystemHeightMap"   # the other grids' placed cells are cross-checked against this one
METHODS = {"R16_UNORM": "content", "BC6H_UF16": "content", "B8G8R8A8_UNORM": "mask"}
ORDER_KEYS = {
    "text": lambda cell, cols: str(cell),
    "numeric": lambda cell, cols: cell,
    "row_col_text": lambda cell, cols: f"{cell // cols}_{cell % cols}",
    "col_row_text": lambda cell, cols: f"{cell % cols}_{cell // cols}",
}


# ---- plain Python: geometry, order, alignment, fit ------------------------------------------------------------

def geometry(doc: dict) -> dict:
    """Columns, rows and world box of a grid document; origin is the grid centre (docs/gamedata.md)."""
    tile = int(doc["tileSize"])
    width, height = int(doc["width"]), int(doc["height"])
    if tile <= 0 or width % tile or height % tile:
        raise ValueError(f"grid {width}x{height} is not a whole number of {tile}-unit tiles")
    ox, oy, oz = (float(v) for v in doc["origin"])
    z = float(doc.get("zInflation", 0.0))
    return {"columns": width // tile, "rows": height // tile, "tile": tile,
            "x_min": ox - width / 2, "x_max": ox + width / 2,
            "y_min": oy - height / 2, "y_max": oy + height / 2,
            "z_min": oz - z, "z_max": oz + z}


def cell_box(geom: dict, cell: int) -> tuple:
    """World x/y box of a cell; row 0 is the north edge (largest y)."""
    row, col = divmod(cell, geom["columns"])
    if not 0 <= row < geom["rows"]:
        raise ValueError(f"cell {cell} is outside the {geom['columns']}x{geom['rows']} grid")
    t = geom["tile"]
    x0 = geom["x_min"] + col * t
    y1 = geom["y_max"] - row * t
    return (x0, y1 - t, x0 + t, y1)


def text_order(cells: int) -> list:
    """Cells in the order the tile dictionaries are stored: sorted as text of the decimal index."""
    return sorted(range(cells), key=str)


def ascending_steps(cells_in_archive_order: list, columns: int) -> dict:
    """For each spelling in ORDER_KEYS: [ascending steps, steps] of the given cells in archive order."""
    out = {}
    steps = max(len(cells_in_archive_order) - 1, 0)
    for name, key in ORDER_KEYS.items():
        keys = [key(c, columns) for c in cells_in_archive_order]
        out[name] = [sum(1 for a, b in zip(keys, keys[1:]) if b > a), steps]
    return out


def fill_by_order(n_tiles: int, anchors: dict, order: list) -> tuple:
    """Place tiles between anchors (tile index -> cell) by archive order.

    ``order`` lists the cells in storage order. Anchors that break the order are dropped (the later one of each
    break). Between two kept anchors, k tiles fill the k cells between them only when there are exactly k free
    cells. Returns (placement dict tile -> cell, dropped anchors, tiles left unplaced).
    """
    pos = {c: i for i, c in enumerate(order)}
    kept, dropped, last = [], 0, -1
    for i in sorted(anchors):
        p = pos[anchors[i]]
        if p > last:
            kept.append((i, p))
            last = p
        else:
            dropped += 1
    placed = {i: order[p] for i, p in kept}
    left = 0
    bounds = [(-1, -1)] + kept + [(n_tiles, len(order))]
    for (i, a), (j, b) in zip(bounds, bounds[1:]):
        k = j - i - 1
        if k <= 0:
            continue
        if k == b - a - 1:
            for n in range(k):
                placed[i + 1 + n] = order[a + 1 + n]
        else:
            left += k
    return placed, dropped, left


def align(scores: list) -> list:
    """Order-constrained alignment: tile i (archive order) to slot path[i], strictly increasing, maximising the sum.

    ``scores[i][j]`` is tile i's score at slot j. Needs at least as many slots as tiles.
    """
    n = len(scores)
    if not n:
        return []
    m = len(scores[0])
    if m < n:
        raise ValueError(f"{n} tiles do not fit {m} slots in order")
    neg = float("-inf")
    best = list(scores[0])
    back = []
    for i in range(1, n):
        row = scores[i]
        cur = [neg] * m
        arg = [-1] * m
        run, run_at = neg, -1
        for j in range(1, m):
            if best[j - 1] > run:
                run, run_at = best[j - 1], j - 1
            if j >= i and run > neg:
                cur[j] = run + row[j]
                arg[j] = run_at
        back.append(arg)
        best = cur
    j = max(range(m), key=lambda k: best[k])
    path = [j]
    for arg in reversed(back):
        j = arg[j]
        path.append(j)
    return path[::-1]


def linear_fit(xs: list, ys: list) -> dict:
    """Least-squares y = slope * x + intercept, with Pearson r."""
    n = len(xs)
    if n < 2:
        raise ValueError("a fit needs at least two points")
    mx, my = math.fsum(xs) / n, math.fsum(ys) / n
    sxx = math.fsum((x - mx) ** 2 for x in xs)
    syy = math.fsum((y - my) ** 2 for y in ys)
    sxy = math.fsum((x - mx) * (y - my) for x, y in zip(xs, ys))
    if sxx == 0:
        raise ValueError("all x values are equal")
    slope = sxy / sxx
    r = sxy / math.sqrt(sxx * syy) if syy else 0.0
    return {"slope": slope, "intercept": my - slope * mx, "r": r, "n": n}


def inside(lo: float, hi: float, box_lo: float, box_hi: float) -> bool:
    return box_lo <= lo and hi <= box_hi


# ---- inputs -----------------------------------------------------------------------------------------------------

def load_manifests(work: str) -> list:
    """Every exported texture: dicts with archive, offset, dir, index, name, width, height, format, path."""
    assets = os.path.join(work, "assets")
    if not os.path.isdir(assets):
        raise ValueError(f"{work} has no assets folder (run make_game.py --stages assets)")
    out = []
    for archive in sorted(os.listdir(assets)):
        root = os.path.join(assets, archive, "textures")
        path = os.path.join(root, "manifest.json")
        if not os.path.isfile(path):
            continue
        with open(path, encoding="utf-8") as f:
            manifest = json.load(f)
        for item in manifest.get("Items", []):
            for t in item.get("Textures", []):
                out.append({"archive": archive, "offset": int(item["Offset"]), "dir": item["Dir"],
                            "index": int(t["Index"]), "name": t["Name"], "width": int(t["Width"]),
                            "height": int(t["Height"]), "format": t["Format"],
                            "path": os.path.join(root, item["Dir"], "%03d_%s.dds" % (t["Index"], t["Name"])),
                            "manifest": path})
    if not out:
        raise ValueError(f"no texture manifest under {assets}")
    return out


def load_documents(carve: str) -> dict:
    """GridAtlasInfo and TextureGridInfo documents of a carve, as plain values."""
    import gamedata_reader as gr
    reader = gr.Reader()
    docs = {"GridAtlasInfo": [], "TextureGridInfo": []}
    for doc in gr.documents([carve]):
        if doc.get("root_hint") not in docs:
            continue
        tag, _, value = reader.read(doc["data"])
        if tag in docs:
            docs[tag].append(value)
    if not docs["GridAtlasInfo"]:
        raise ValueError(f"no GridAtlasInfo document in {carve}")
    return docs


# ---- images (numpy / Pillow) ------------------------------------------------------------------------------------

def read_image(rec: dict):
    """A texture as a float array: (h, w) for R16, (h, w, c) otherwise."""
    import numpy as np
    with open(rec["path"], "rb") as f:
        data = f.read()
    fmt = rec["format"]
    if fmt == "R16_UNORM":
        if data[:4] != b"DDS " or data[84:88] != b"DX10" or struct.unpack_from("<I", data, 128)[0] != DXGI_R16_UNORM:
            raise ValueError(f"{rec['path']} is not a DX10 R16_UNORM DDS")
        count = rec["width"] * rec["height"]
        if len(data) < DX10_HEADER + 2 * count:
            raise ValueError(f"{rec['path']} is truncated")
        return np.frombuffer(data, "<u2", count, DX10_HEADER).reshape(rec["height"], rec["width"]).astype(float)
    if fmt.startswith("BC6H"):
        import io
        from PIL import Image
        with Image.open(io.BytesIO(data)) as im:
            return np.asarray(im.convert("RGB"), dtype=float)
    import dds_png
    w, h, rgba = dds_png.decode_dds(data)
    return np.frombuffer(rgba, np.uint8).reshape(h, w, 4).astype(float)


def split_cells(fallback, geom: dict):
    """Fallback mosaic -> array (cells, p, p[, c]) with cell = row * columns + column, row 0 at the top."""
    rows, cols = geom["rows"], geom["columns"]
    h, w = fallback.shape[:2]
    if w % cols or h != rows * (w // cols):
        raise ValueError(f"fallback {w}x{h} is not a {cols}x{rows} mosaic of square cells")
    p = w // cols
    rest = fallback.shape[2:]
    a = fallback.reshape(rows, p, cols, p, *rest).swapaxes(1, 2)
    return a.reshape(rows * cols, p, p, *rest)


def point_sample(tile, p: int):
    step = tile.shape[0] // p
    return tile[step // 2::step, step // 2::step][:p, :p]


def match_content(tiles: list, fallbacks: list, geom: dict) -> dict:
    """tiles: per dictionary, a list of images (one per grid); fallbacks: the matching mosaics."""
    import numpy as np
    cells = [split_cells(f, geom) for f in fallbacks]
    p = cells[0].shape[1]
    best, runner, detail = [], [], []
    for images in tiles:
        d = np.zeros(cells[0].shape[0])
        for img, c in zip(images, cells):
            s = point_sample(img, p)
            d += np.abs(c - s).reshape(len(c), -1).mean(1)
        o = np.argsort(d, kind="stable")[:2]
        best.append((int(o[0]), float(d[o[0]])))
        runner.append(float(d[o[1]]))
        detail.append(any(float(img.std()) > DETAIL_STD for img in images))
    anchors = {i: best[i][0] for i in range(len(tiles))
               if detail[i] and runner[i] >= ANCHOR_RATIO * max(best[i][1], 1e-9)
               and runner[i] - best[i][1] >= ANCHOR_GAP}
    return {"anchors": anchors, "detail": sum(detail), "best": best}


def match_mask(tiles: list, fallback, geom: dict) -> dict:
    """Feature-mask IoU of each tile against each cell, then order-constrained alignment to text order."""
    import numpy as np
    flat = fallback.reshape(-1, fallback.shape[-1])
    colours, counts = np.unique(flat, axis=0, return_counts=True)
    bg = colours[int(counts.argmax())]
    cells = split_cells(fallback, geom)
    p = cells.shape[1]
    cm = (np.abs(cells - bg).max(-1) > MASK_THRESHOLD).astype(np.int32)
    tm = []
    for img in tiles:
        b = img.shape[0] // p
        m = (np.abs(img - bg).max(-1) > MASK_THRESHOLD).reshape(p, b, p, b).mean((1, 3)) > MASK_BLOCK_SHARE
        tm.append(m)
    tm = np.array(tm, dtype=np.int32)
    inter = np.einsum("tij,cij->tc", tm, cm)
    union = tm.sum((1, 2))[:, None] + cm.sum((1, 2))[None, :] - inter
    iou = np.where(union > 0, inter / np.maximum(union, 1), 0.0)
    top = np.sort(iou, 1)[:, ::-1]
    clear = {i: int(iou[i].argmax()) for i in range(len(tiles)) if top[i, 0] - top[i, 1] > 0.2}
    order = text_order(len(cells))
    path = align(iou[:, order].tolist())
    placed = {i: order[j] for i, j in enumerate(path) if iou[i, order[j]] > 0}
    scores = [float(iou[i, c]) for i, c in placed.items()]
    return {"background": [int(v) for v in bg], "clear": clear, "placed": placed,
            "featureless_tiles": int((tm.sum((1, 2)) == 0).sum()),
            "feature_cells": int((cm.sum((1, 2)) > 0).sum()),
            "iou_median": sorted(scores)[len(scores) // 2] if scores else None}


def fallback_stats(fallback, geom: dict) -> dict:
    import numpy as np
    cells = split_cells(fallback, geom).reshape(geom["rows"] * geom["columns"], -1)
    const = cells.min(1) == cells.max(1)
    out = {"constant_cells": int(const.sum()), "value_min": float(fallback.min()), "value_max": float(fallback.max())}
    if fallback.ndim == 2:
        vals, counts = np.unique(fallback, return_counts=True)
        out["most_common_value"] = float(vals[counts.argmax()])
        out["most_common_share"] = round(float(counts.max()) / fallback.size, 4)
    return out


def water_fit(fallback, geom: dict, payloads: list) -> list:
    """Fit water-height fallback values against each MHIU terrain grid, north-up and (for contrast) south-up."""
    import numpy as np
    rows, cols = fallback.shape
    px = (geom["x_max"] - geom["x_min"]) / cols
    r, c = np.mgrid[0:rows, 0:cols]
    x = geom["x_min"] + (c + 0.5) * px
    out = []
    for pl in payloads:
        if pl.get("type") != "MHIU":
            continue
        b = pl["header"]["bounds"]
        w, h = pl["header"]["width"], pl["header"]["height"]
        z = np.frombuffer(pl["_grid"], np.uint8).reshape(h, w).astype(float)
        z = b["min"][2] + z / 255.0 * (b["max"][2] - b["min"][2])
        row = {"heightmap": "MHIU " + pl.get("offset_hex", "")}
        for label, y in (("north_up", geom["y_max"] - (r + 0.5) * px), ("south_up", geom["y_min"] + (r + 0.5) * px)):
            i = np.floor((x - b["min"][0]) / (b["max"][0] - b["min"][0]) * w).astype(int)
            j = np.floor((y - b["min"][1]) / (b["max"][1] - b["min"][1]) * h).astype(int)
            ok = (i >= 0) & (i < w) & (j >= 0) & (j < h) & (fallback > FIT_MIN_VALUE)
            if ok.sum() < 2:
                row[label] = None
                continue
            fit = linear_fit(fallback[ok].tolist(), z[j[ok], i[ok]].tolist())
            row[label] = {"n": fit["n"], "r": round(fit["r"], 4),
                          "units_per_metre": round(1 / fit["slope"], 3) if fit["slope"] else None,
                          "zero_metre_value": round(-fit["intercept"] / fit["slope"], 1) if fit["slope"] else None,
                          "_fit": fit}
        out.append(row)
    return out


# ---- driver ----------------------------------------------------------------------------------------------------

def run(work: str, carve: str) -> dict:
    import gamedata_maps as gm
    textures = load_manifests(work)
    by_name = {}
    for t in textures:
        by_name.setdefault(t["name"], []).append(t)
    docs = load_documents(carve)
    grids = sorted(docs["GridAtlasInfo"], key=lambda d: str(d.get("clientSystemName")))
    payloads = gm.read_payloads(carve)

    # group grids whose tiles live in the same dictionaries (matched jointly)
    keys = {}
    for doc in grids:
        recs = [t for t in by_name.get(doc["tileTextureNameSuffix"], []) if t["archive"] == _main_archive(
            by_name.get(doc["tileTextureNameSuffix"], []))]
        keys.setdefault(frozenset((t["archive"], t["offset"]) for t in recs), []).append(doc)

    summary = {"grids": [], "texture_grids": [], "heightmaps": [], "placements": {}}
    height_cells = None
    for docset in sorted(keys.values(), key=lambda ds: all(d.get("clientSystemName") != HEIGHT_GRID for d in ds)):
        is_height = any(d.get("clientSystemName") == HEIGHT_GRID for d in docset)
        geom = geometry(docset[0])
        for d in docset[1:]:
            if geometry(d) != geom:
                raise ValueError("grids that share tile dictionaries have different geometry")
        main = _main_archive(by_name.get(docset[0]["tileTextureNameSuffix"], []))
        per_doc = []
        for doc in docset:
            recs = [t for t in by_name.get(doc["tileTextureNameSuffix"], []) if t["archive"] == main]
            per_doc.append({(t["offset"], t["dir"]): t for t in recs})
        dicts = sorted(set.intersection(*(set(p) for p in per_doc)))
        if not dicts:
            raise ValueError(f"no tile dictionaries for {', '.join(d['tileTextureNameSuffix'] for d in docset)}")
        fmt = per_doc[0][dicts[0]]["format"]
        method = METHODS.get(fmt)
        fallbacks = []
        for doc in docset:
            fb = by_name.get(doc["globalFallbackTextureName"], [])
            if len(fb) != 1:
                raise ValueError(f"expected one {doc['globalFallbackTextureName']}, found {len(fb)}")
            fallbacks.append(read_image(fb[0]))
        result = {"method": method, "dictionaries": len(dicts), "tile_format": fmt}
        if method == "content":
            tiles = [[read_image(p[k]) for p in per_doc] for k in dicts]
            m = match_content(tiles, fallbacks, geom)
            anchors = m["anchors"]
            result["tiles_with_detail"] = m["detail"]
            placed, dropped, left = fill_by_order(len(dicts), anchors, text_order(geom["rows"] * geom["columns"]))
            result.update({"anchors": len(anchors), "anchor_order": ascending_steps(
                [anchors[i] for i in sorted(anchors)], geom["columns"]), "anchors_dropped": dropped,
                "placed_by_order": len(placed) - (len(anchors) - dropped), "unplaced": left})
        elif method == "mask":
            tiles = [read_image(per_doc[0][k]) for k in dicts]
            m = match_mask(tiles, fallbacks[0], geom)
            placed = m.pop("placed")
            clear = m.pop("clear")
            result.update(m)
            result.update({"anchors": len(clear), "anchor_order": ascending_steps(
                [clear[i] for i in sorted(clear)], geom["columns"]), "unplaced": len(dicts) - len(placed)})
        else:
            raise ValueError(f"no matching method for tile format {fmt}")
        result["placed"] = len(placed)
        result["distinct_cells"] = len(set(placed.values()))
        if is_height:
            height_cells = set(placed.values())
        elif height_cells is not None:
            result["in_water_height_cells"] = sum(1 for c in set(placed.values()) if c in height_cells)
            anchor_cells = list((anchors if method == "content" else clear).values())
            result["anchors_in_water_height_cells"] = sum(1 for c in anchor_cells if c in height_cells)
        for doc, fb in zip(docset, fallbacks):
            g = {"name": doc.get("clientSystemName"), "tile_name": doc["tileTextureNameSuffix"],
                 "tile_archives": _archive_counts(by_name.get(doc["tileTextureNameSuffix"], [])),
                 "fallback": doc["globalFallbackTextureName"], "fallback_size": list(fb.shape[1::-1]),
                 "fallback_stats": fallback_stats(fb, geom), "geometry": geom,
                 "z_inflation": float(doc.get("zInflation", 0.0)),
                 "sector_fallbacks": {o.get("sectorOverrideGlobalFallbackName"):
                                      _archive_counts(by_name.get(o.get("sectorOverrideGlobalFallbackName"), []))
                                      for o in doc.get("sectorOverrides") or []},
                 **result}
            if is_height and fmt == "R16_UNORM":
                fits = water_fit(fb, geom, payloads)
                g["water_fit"] = [{k: v for k, v in f.items()} for f in fits]
                g["water_metres"] = _water_metres(fb, fits)
                for f in g["water_fit"]:
                    for label in ("north_up", "south_up"):
                        if f.get(label):
                            f[label] = {k: v for k, v in f[label].items() if not k.startswith("_")}
            summary["grids"].append(g)
        summary["placements"][docset[0]["tileTextureNameSuffix"]] = [
            {"archive": main, "offset": dicts[i][0], "dir": dicts[i][1], "cell": c,
             "row": c // geom["columns"], "column": c % geom["columns"]} for i, c in sorted(placed.items())]

    for doc in docs["TextureGridInfo"]:
        geom = geometry(doc)
        summary["texture_grids"].append({
            "geometry": geom, "z_inflation": float(doc.get("zInflation", 0.0)),
            "fallbacks": {k: {"name": doc.get(k), "archives": _archive_counts(by_name.get(doc.get(k), []))}
                for k in ("lowDetailFallBack", "highDetailFallBack")}})

    for pl in payloads:
        b = pl["header"]["bounds"]
        summary["heightmaps"].append({"type": pl["type"], "offset": pl.get("offset_hex"),
                                      "z_min": b["min"][2], "z_max": b["max"][2]})
    zboxes = {(g["geometry"]["z_min"], g["geometry"]["z_max"]) for g in summary["grids"]}
    summary["z_check"] = [{"z_box": list(z), "heightmaps_inside": sum(
        inside(h["z_min"], h["z_max"], *z) for h in summary["heightmaps"]), "heightmaps": len(summary["heightmaps"])}
        for z in sorted(zboxes)]
    return summary


def _main_archive(recs: list):
    counts = _archive_counts(recs)
    return max(counts, key=lambda a: (counts[a], a)) if counts else None


def _archive_counts(recs: list) -> dict:
    out = {}
    for t in recs:
        out[t["archive"]] = out.get(t["archive"], 0) + 1
    return out


def _tile_format(doc: dict, by_name: dict):
    recs = by_name.get(doc["tileTextureNameSuffix"], [])
    return recs[0]["format"] if recs else None


def _water_metres(fallback, fits: list):
    """Fallback value range in metres under the best (highest r) north-up fit."""
    good = [f["north_up"]["_fit"] for f in fits if f.get("north_up")]
    if not good:
        return None
    fit = max(good, key=lambda f: f["r"])
    lo, hi = float(fallback.min()), float(fallback.max())
    return {"min": round(fit["slope"] * lo + fit["intercept"], 1),
            "max": round(fit["slope"] * hi + fit["intercept"], 1)}


def check_out_path(out: str, carve: str, textures: list) -> None:
    """Refuse a --json path inside the carve folder or equal to a manifest the tool reads."""
    for path in sorted({t["manifest"] for t in textures}):
        if report_paths.same_report_path(out, path):
            raise ValueError(f"--json {out} would overwrite the input {path}")
    try:
        common = os.path.commonpath([os.path.abspath(out), os.path.abspath(carve)])
    except ValueError:              # different drives on Windows
        return
    if report_paths.path_key(common) == report_paths.path_key(os.path.abspath(carve)):
        raise ValueError(f"--json {out} is inside the carve folder {carve}")


def report_lines(summary: dict) -> list:
    out = []
    for g in summary["grids"]:
        order = g.get("anchor_order", {})
        text = order.get("text", [0, 0])
        num = order.get("numeric", [0, 0])
        line = (f"{g['name']}: {g['dictionaries']} tile dictionaries of {g['tile_name']} ({g['tile_format']}), "
                f"{g['method']}: {g['anchors']} anchors (text order {text[0]}/{text[1]} ascending, numeric "
                f"{num[0]}/{num[1]}), {g['placed']} placed in {g['distinct_cells']} cells, {g['unplaced']} unplaced")
        if "in_water_height_cells" in g:
            line += (f"; cells with a water-height tile: {g['in_water_height_cells']} of placed, "
                     f"{g['anchors_in_water_height_cells']} of {g['anchors']} anchors")
        out.append(line)
        if g.get("water_metres"):
            fits = "; ".join(
                f"{f['heightmap']}: {f['north_up']['units_per_metre']} units/m, 0 m at {f['north_up']['zero_metre_value']}, "
                f"r {f['north_up']['r']} (south-up r {f['south_up']['r'] if f.get('south_up') else None})"
                for f in g["water_fit"] if f.get("north_up"))
            out.append(f"  water height vs terrain (values > {FIT_MIN_VALUE}): {fits}; fallback range "
                       f"{g['water_metres']['min']} to {g['water_metres']['max']} m")
    for z in summary["z_check"]:
        out.append(f"z box {z['z_box'][0]:g} to {z['z_box'][1]:g}: {z['heightmaps_inside']} of {z['heightmaps']} "
                   f"heightmap Z bounds inside")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("work", help="make_game.py work folder (with assets/<archive>/textures/manifest.json)")
    ap.add_argument("--carve", required=True, help="rpf9_carve.py output of common.rpf")
    ap.add_argument("--json", help="write the summary and tile placements here (game-derived: keep it local)")
    a = ap.parse_args(argv)
    try:
        if not os.path.isdir(a.carve):
            raise ValueError(f"--carve {a.carve} is not a folder")
        textures = load_manifests(a.work)
        if a.json:
            if os.path.isdir(a.json):
                raise ValueError(f"--json {a.json} is a folder")
            check_out_path(a.json, a.carve, textures)
        summary = run(a.work, a.carve)
    except (OSError, ValueError, KeyError) as e:
        print(f"world_grid_atlas: {e}", file=sys.stderr)
        return 2
    except ImportError as e:
        print(f"world_grid_atlas: needs numpy and Pillow ({e})", file=sys.stderr)
        return 2
    for line in report_lines(summary):
        print(line)
    if a.json:
        try:
            os.makedirs(os.path.dirname(os.path.abspath(a.json)), exist_ok=True)
            with open(a.json, "w", encoding="utf-8", newline="\n") as f:
                json.dump(summary, f, indent=1)
        except OSError as e:
            print(f"world_grid_atlas: cannot write {a.json}: {e}", file=sys.stderr)
            return 2
        print(f"wrote {a.json} (tile placements are game-derived: keep the file local)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
