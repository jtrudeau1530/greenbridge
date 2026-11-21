"""Index world metadata and map-placement resources in an RPF9 archive or a folder of carved files.

The table of contents of an RPF9 archive is encrypted, but entry payloads are plain zlib streams. A world
metadata resource is a stream whose first system page starts with 8 zero bytes, a ``0x5...`` pointer at +0x08,
and ``0DRP`` + the version byte ``0x7a`` at offset 0x10. Map-placement resources (the GTA V ``.ymap``
equivalent) are a subset of those: their schema contains the ``CMapData`` root (``d3593fa6``) and
``CEntityDef`` (``ce501483``) structs.

The tool only reads headers, the resource's own schema, and the scalar placement fields named in that schema.
It does not decrypt the archive table and does not extract models or textures.

Usage:
    python -I world_index.py <archive.rpf|folder> [--out FILE] [--csv FILE]
                             [--start N] [--limit BYTES] [--max-inflate BYTES]

For a 76 GB archive, bound the run with --start/--limit. Coverage and the scanned window are written to the
JSON summary.
"""
import argparse
import csv
import json
import mmap
import os
import re
import struct
import sys
import zlib

SIGNATURE = b"0DRP"
VERSION = 0x7A
ENTITY = 0xCE501483
MAPDATA = 0xD3593FA6
CAP = 64 << 20
CHUNK = 1 << 20
# Field-name hashes are case-sensitive joaat of the names below, confirmed by matching the schema member
# table of every sampled map file.
FIELD = {
    "archetypeName": 0xA023A02C,
    "position": 0x01166164,
    "rotation": 0x77DD4010,
    "scaleXY": 0x9CA32637,
    "scaleZ": 0x10FB7C42,
    "flags": 0x67D26872,
    "guid": 0x9A7B6E6D,
    "lodLevel": 0x6C8F1715,
    "parentIndex": 0xD89231BD,
    "parentInstance": 0xE74C3C23,
    "numChildren": 0xA687AC89,
    "lodDist": 0xBF74DFA7,
    "childLodDist": 0xCA974BCD,
    "priorityLevel": 0x2691F019,
    "name": 0xE6E17F14,
    "name64": 0x8BFB6095,
    "parent": 0xA99C4651,
    "streamingExtentsMin": 0xDD22821F,
    "streamingExtentsMax": 0xA22EA335,
    "entitiesExtentsMin": 0x1C75BCF1,
    "entitiesExtentsMax": 0x6D074437,
    "flags_map": 0x67D26872,
    "contentFlags": 0x6A675035,
}
LOD_NAMES = {0: "HD", 1: "LOD", 2: "SLOD1", 3: "SLOD2", 4: "SLOD3", 5: "ORPHAN_HD", 6: "SLOD4"}


def _u16(b, o):
    return struct.unpack_from("<H", b, o)[0] if 0 <= o and o + 2 <= len(b) else 0


def _u32(b, o):
    return struct.unpack_from("<I", b, o)[0] if 0 <= o and o + 4 <= len(b) else 0


def _u64(b, o):
    return struct.unpack_from("<Q", b, o)[0] if 0 <= o and o + 8 <= len(b) else 0


def _f32(b, o):
    return struct.unpack_from("<f", b, o)[0] if 0 <= o and o + 4 <= len(b) else 0.0


def _ptr(value, size):
    """Resolve a 0x5xxxxxxx pointer, the only segment a single carved system stream contains."""
    if value >> 28 != 5:
        return None
    off = value & 0x0FFFFFFF
    return off if off < size else None


def _members(b, table, count, size):
    if count > 256 or (count and table is None) or (table is not None and table + 16 * count > size):
        return None
    out = {}
    if not count:
        return out
    for k in range(count):
        o = table + 16 * k
        out[_u32(b, o)] = (_u32(b, o + 4), b[o + 8], b[o + 9], _u16(b, o + 10), _u32(b, o + 12))
    return out


def _value(b, base, member):
    if member is None:
        return None
    off, ty = member[0], member[1]
    p = base + off
    if ty == 0x10:
        return struct.unpack_from("<b", b, p)[0] if p < len(b) else None
    if ty == 0x11:
        return b[p] if p < len(b) else None
    if ty == 0x12:
        return struct.unpack_from("<h", b, p)[0] if p + 2 <= len(b) else None
    if ty == 0x13:
        return _u16(b, p)
    if ty == 0x14:
        return struct.unpack_from("<i", b, p)[0] if p + 4 <= len(b) else None
    if ty in (0x15, 0x4A):
        return _u32(b, p)
    if ty == 0x21:
        return _f32(b, p)
    if ty in (0x32, 0x33):
        return [_f32(b, p + 4 * i) for i in range(3)]
    if ty == 0x34:
        return [_f32(b, p + 4 * i) for i in range(4)]
    if ty in (0x08, 0x4F):
        return _u64(b, p)
    if ty == 0x60:
        return b[p] if p < len(b) else None
    return None


def _r4(values):
    return [round(v, 4) for v in values] if values else None


def parse_resource(b, collect=None):
    """Return a row dict for a 0x7a resource, or None if the header/schema does not validate.

    When ``collect`` is a list, one dict per CEntityDef is appended to it (local analysis only; the CLI does
    not use this, so the JSON stays an aggregate index).
    """
    if len(b) < 0x50 or b[0x10:0x14] != SIGNATURE or b[0x14] != VERSION:
        return None
    nst, nen, nbl = _u16(b, 0x48), _u16(b, 0x4A), _u16(b, 0x4C)
    if not (1 <= nst <= 4096 and nbl <= 4096 and nen <= 4096) or nbl == 0:
        return None
    st, it = _ptr(_u64(b, 0x28), len(b)), _ptr(_u64(b, 0x38), len(b))
    if st is None or it is None or st + 32 * nst > len(b) or it + 16 * nbl > len(b):
        return None

    structs = {}
    for i in range(nst):
        o = st + 32 * i
        name, size, count = _u32(b, o), _u32(b, o + 0x18), _u16(b, o + 0x1E)
        table = _ptr(_u64(b, o + 0x10), len(b))
        if size > (1 << 20) or count > 256:
            return None
        members = _members(b, table, count, len(b))
        if members is None:
            return None
        structs.setdefault(name, (size, members))

    blocks = []
    for i in range(nbl):
        o = it + 16 * i
        name, size, p = _u32(b, o), _u32(b, o + 4), _u64(b, o + 8)
        if size > (1 << 24):
            return None
        off = _ptr(p, len(b))
        blocks.append((name, size, off))
    if not any(name in structs and off is not None for name, _, off in blocks):
        return None

    entity_size = structs.get(ENTITY, (0, {}))[0]
    map_size = structs.get(MAPDATA, (0, {}))[0]
    row = {
        "signature": "0DRPz",
        "version": VERSION,
        "kind": "meta",
        "structs": nst,
        "enums": nen,
        "blocks": nbl,
        "root_objects": _u32(b, 0x20),
    }
    if not (entity_size and map_size):
        return row

    em = structs[ENTITY][1]
    mm = structs[MAPDATA][1]
    lod = {}
    archetypes = set()
    positions = []
    lod_dists = []
    for name, size, off in blocks:
        if name != ENTITY or off is None or entity_size == 0 or size % entity_size:
            continue
        for k in range(size // entity_size):
            o = off + k * entity_size
            a = _value(b, o, em.get(FIELD["archetypeName"]))
            if a is not None:
                archetypes.add(a)
            pos = _value(b, o, em.get(FIELD["position"]))
            if pos:
                positions.append(pos)
            level = _value(b, o, em.get(FIELD["lodLevel"]))
            if level in LOD_NAMES:
                lod[LOD_NAMES[level]] = lod.get(LOD_NAMES[level], 0) + 1
            dist = _value(b, o, em.get(FIELD["lodDist"]))
            if isinstance(dist, float) and dist >= 0:
                lod_dists.append(dist)
            if collect is not None:
                collect.append({
                    "archetype": a,
                    "position": pos,
                    "rotation": _value(b, o, em.get(FIELD["rotation"])),
                    "scale_xy": _value(b, o, em.get(FIELD["scaleXY"])),
                    "scale_z": _value(b, o, em.get(FIELD["scaleZ"])),
                    "flags": _value(b, o, em.get(FIELD["flags"])),
                    "guid": _value(b, o, em.get(FIELD["guid"])),
                    "lod_level": level,
                    "parent_index": _value(b, o, em.get(FIELD["parentIndex"])),
                    "parent_instance": _value(b, o, em.get(FIELD["parentInstance"])),
                    "num_children": _value(b, o, em.get(FIELD["numChildren"])),
                    "lod_dist": dist,
                    "child_lod_dist": _value(b, o, em.get(FIELD["childLodDist"])),
                })
    row.update({
        "kind": "map",
        "entity_size": entity_size,
        "entities": sum(size // entity_size for n, size, off in blocks
                        if n == ENTITY and off is not None and size % entity_size == 0),
        "archetypes": len(archetypes),
        "lod": lod,
        "lod_dist_min": min(lod_dists) if lod_dists else None,
        "lod_dist_max": max(lod_dists) if lod_dists else None,
    })
    if positions:
        row["position_min"] = _r4([min(p[i] for p in positions) for i in range(3)])
        row["position_max"] = _r4([max(p[i] for p in positions) for i in range(3)])

    for name, size, off in blocks:
        if name != MAPDATA or off is None or map_size == 0:
            continue
        row["map_name"] = _u32(b, off + mm.get(FIELD["name"], (8,))[0])
        row["parent"] = _u32(b, off + mm.get(FIELD["parent"], (0x68,))[0])
        if FIELD["name64"] in mm:
            row["map_name64"] = _u64(b, off + mm[FIELD["name64"]][0])
        for key, field in (("extent_min", "streamingExtentsMin"), ("extent_max", "streamingExtentsMax"),
                           ("entities_min", "entitiesExtentsMin"), ("entities_max", "entitiesExtentsMax")):
            v = _value(b, off, mm.get(FIELD[field]))
            if v:
                row[key] = _r4(v)
        row["map_flags"] = _value(b, off, mm.get(FIELD["flags_map"]))
        row["content_flags"] = _value(b, off, mm.get(FIELD["contentFlags"]))
        break
    return row


ZHDR = re.compile(rb"\x78[\x01\x5e\x9c\xda]")


def _needs_continuation(b):
    """True when a 0x7a header's instance table or a system block pointer lies beyond this stream."""
    if len(b) < 0x50 or b[0x10:0x14] != SIGNATURE or b[0x14] != VERSION:
        return False
    it = _ptr(_u64(b, 0x38), len(b))
    nbl = _u16(b, 0x4C)
    if it is None or it + 16 * nbl > len(b):
        return True
    for i in range(nbl):
        p = _u64(b, it + 16 * i + 8)
        if p >> 28 == 5 and (p & 0x0FFFFFFF) >= len(b):
            return True
    return False


def _inflate_stream(m, s, size, max_inflate, keep=False):
    """Inflate one zlib stream. Returns (data, used, eof, is_meta); data is kept for 0x7a or keep=True."""
    d = zlib.decompressobj()
    chunks = []
    kept = 0
    head = bytearray()
    is_meta = False
    p = s
    while not d.eof and p < size:
        piece = m[p:p + CHUNK]
        if not piece:
            break
        produced = d.decompress(piece)
        p += len(piece)
        if len(head) < 0x15:
            head.extend(produced[:0x15 - len(head)])
            if len(head) >= 0x15:
                is_meta = head[0x10:0x15] == b"0DRPz"
        if (is_meta or keep) and kept < max_inflate:
            chunks.append(produced)
            kept += len(produced)
        if d.unused_data or d.eof:
            break
    if not d.eof:
        return b"", 0, False, False
    return b"".join(chunks), p - s - len(d.unused_data), True, is_meta


def scan_rpf(path, start=None, limit=None, max_inflate=CAP):
    """Yield rows for every 0x7a resource in an RPF9 data area, plus scan statistics."""
    rows = []
    scanned = streams = 0
    with open(path, "rb") as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as m:
        size = len(m)
        data_start = _u32(m[:8], 4)
        pos = start if start is not None else data_start
        end = min(size, pos + limit) if limit else size
        while pos < end:
            found = ZHDR.search(m, pos, end)
            if not found:
                break
            s = found.start()
            if s % 16:
                pos = s + 1
                continue
            try:
                data, used, eof, is_meta = _inflate_stream(m, s, size, max_inflate)
            except zlib.error:
                pos = s + 1
                continue
            if not eof:
                pos = s + 1
                continue
            consumed = 1
            if is_meta:
                # Large resources put the system page in one stream and a small continuation in the next.
                # Append contiguous zlib streams until the resource's own schema resolves.
                for _ in range(2):
                    if parse_resource(data) is not None and not _needs_continuation(data):
                        break
                    nxt = (s + used + 15) & ~15
                    if nxt >= size:
                        break
                    mt = ZHDR.match(m, nxt, size)
                    if not mt:
                        break
                    more, mused, meof, _ = _inflate_stream(m, nxt, size, max_inflate, keep=True)
                    if not meof or not more:
                        break
                    data += more
                    used += mused
                    consumed += 1
                row = parse_resource(data)
                if row is not None:
                    row["offset"] = s
                    row["size"] = len(data)
                    rows.append(row)
            streams += consumed
            scanned += used
            pos = s + used
            if pos <= s:
                pos = s + 1
        return rows, {"path": os.path.abspath(path), "bytes": size,
                      "scan_start": start if start is not None else data_start,
                      "scan_end": end, "scanned_bytes": scanned, "streams": streams}


def scan_path(path, start=None, limit=None, max_inflate=CAP):
    """Scan an RPF9 archive, or every carved file in a directory."""
    if os.path.isdir(path):
        rows, stats = [], {"path": os.path.abspath(path), "files": 0, "scanned_bytes": 0}
        for root, _, files in os.walk(path):
            for name in sorted(files):
                p = os.path.join(root, name)
                if os.path.getsize(p) > max_inflate:
                    continue
                with open(p, "rb") as f:
                    head = f.read(4)
                if head == b"9FPR":
                    more, sub = scan_rpf(p, start, limit, max_inflate)
                    rows.extend(more)
                    stats["files"] += 1
                    stats["scanned_bytes"] += sub["scanned_bytes"]
                    continue
                with open(p, "rb") as f:
                    b = f.read(max_inflate + 1)
                row = parse_resource(b)
                stats["files"] += 1
                stats["scanned_bytes"] += len(b)
                if row is not None:
                    row["offset"] = 0
                    row["size"] = len(b)
                    row["file"] = os.path.relpath(p, path).replace("\\", "/")
                    rows.append(row)
        return rows, stats
    with open(path, "rb") as f:
        magic = f.read(4)
    if magic == b"9FPR":
        return scan_rpf(path, start, limit, max_inflate)
    with open(path, "rb") as f:
        b = f.read(max_inflate + 1)
    row = parse_resource(b)
    return ([dict(row, offset=0, size=len(b), file=os.path.basename(path))] if row else [],
            {"path": os.path.abspath(path), "files": 1, "scanned_bytes": len(b)})


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("path", help="RPF9 archive, one carved file, or a directory of carved files")
    ap.add_argument("--out", default=os.path.join("reports", "world_index.json"),
                    help="JSON output (default: reports/world_index.json)")
    ap.add_argument("--csv", help="optional CSV output")
    ap.add_argument("--start", type=int, help="first byte of an RPF9 data-area window")
    ap.add_argument("--limit", type=int, help="window size in bytes (default: to end of archive)")
    ap.add_argument("--max-inflate", type=int, default=CAP,
                    help="stop keeping bytes above this per resource, default 64 MiB")
    a = ap.parse_args(argv)
    rows, stats = scan_path(a.path, a.start, a.limit, a.max_inflate)
    summary = dict(stats)
    summary["resources"] = len(rows)
    summary["maps"] = sum(1 for r in rows if r["kind"] == "map")
    summary["meta"] = sum(1 for r in rows if r["kind"] == "meta")
    summary["entities"] = sum(r.get("entities") or 0 for r in rows)
    summary["versions"] = sorted({r["version"] for r in rows})
    doc = {"summary": summary, "resources": rows}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=1, sort_keys=True)
    if a.csv:
        cols = ["source", "offset", "size", "signature", "version", "kind", "structs", "enums", "blocks",
                "root_objects", "entities", "entity_size", "archetypes", "map_name", "map_name64", "parent",
                "map_flags",
                "content_flags", "extent_min", "extent_max", "entities_min", "entities_max", "position_min",
                "position_max", "lod", "lod_dist_min", "lod_dist_max"]
        os.makedirs(os.path.dirname(os.path.abspath(a.csv)), exist_ok=True)
        with open(a.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            for r in rows:
                r = dict(r)
                r["source"] = r.get("file") or r.get("path") or os.path.basename(a.path)
                for k, v in list(r.items()):
                    if isinstance(v, (list, dict)):
                        r[k] = json.dumps(v, sort_keys=True)
                w.writerow(r)
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
