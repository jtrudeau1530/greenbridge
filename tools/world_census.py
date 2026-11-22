"""Census every 0x7a map resource in an RPF9 archive, with measured byte coverage.

The archive table of contents is encrypted, but file payloads are zlib streams (and some graphics pages are
stored raw). This walks the data area in independent byte shards, inflates every stream, validates the 0x7a
header/schema, and joins the contiguous continuation streams of a split resource. Each shard writes its map
rows, accepted stream intervals and 0x6 page extents; the merge step counts maps/entities, finds duplicate
payloads, measures unread bytes, and audits the gaps for a 0x7a signature the walker did not read.

Usage:
    python -I world_census.py <archive.rpf> [--out DIR] [--shard-size BYTES] [--jobs N]
                              [--start N] [--limit BYTES] [--resume] [--merge-only]

Only aggregate metadata is written: resource offsets/sizes, map/entity counts, extents and content hashes.
No carved stream, model or texture is kept.
"""
from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import mmap
import os
import re
import struct
import sys
import time
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed

_TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
if _TOOLS_DIR not in sys.path:
    sys.path.insert(0, _TOOLS_DIR)

import world_index as WI

ZHDR = re.compile(rb"\x78[\x01\x5e\x9c\xda]")
SIG = b"0DRP"
VERSION = 0x7A
CAP = 64 << 20
CHUNK = 1 << 20
WARMUP = 8 << 20
JOIN_MAX = 64
BAND_GAP = 64 << 20
GAP_CHUNK = 4 << 20
FORMAT = 3


def inflate(m, s, size, max_inflate=CAP, keep=False):
    """Inflate one zlib stream.

    Returns (data, used, eof, is_meta, inflated, truncated). ``used`` is the stored compressed length;
    ``inflated`` counts the whole output even when only a 0x7a stream is kept.
    """
    d = zlib.decompressobj()
    chunks = []
    kept = 0
    inflated = 0
    carry = bytearray()
    decided = False
    is_meta = False
    truncated = False
    p = s
    while not d.eof and p < size:
        piece = m[p:p + CHUNK]
        if not piece:
            break
        produced = d.decompress(piece)
        p += len(piece)
        inflated += len(produced)
        if not decided:
            carry.extend(produced)
            if len(carry) >= 0x15:
                decided = True
                is_meta = bytes(carry[0x10:0x15]) == b"0DRPz"
                if is_meta or keep:
                    take = bytes(carry[:max(0, max_inflate - kept)])
                    chunks.append(take)
                    kept += len(take)
                    truncated = len(take) < len(carry)
                carry = None
        elif is_meta or keep:
            if kept < max_inflate:
                take = produced[:max_inflate - kept]
                chunks.append(take)
                kept += len(take)
                truncated = truncated or len(take) < len(produced)
            elif produced:
                truncated = True
        if d.unused_data or d.eof:
            break
    if not d.eof:
        return b"", 0, False, False, inflated, False
    used = p - s - len(d.unused_data)
    return b"".join(chunks), used, True, is_meta, inflated, truncated


def required_len(b):
    """Highest system offset (0x5 segment) referenced by a 0x7a resource's tables and blocks."""
    if len(b) < 0x50 or b[0x10:0x14] != SIG or b[0x14] != VERSION:
        return 0
    nst, nen, nbl = struct.unpack_from("<HHH", b, 0x48)
    if not (1 <= nst <= 4096 and nbl <= 4096 and nen <= 4096) or nbl == 0:
        return 0
    need = len(b)
    for ptr, count, width in ((struct.unpack_from("<Q", b, 0x28)[0], nst, 32),
                              (struct.unpack_from("<Q", b, 0x38)[0], nbl, 16)):
        if ptr >> 28 != 5:
            return 0
        need = max(need, (ptr & 0x0FFFFFFF) + count * width)
    st = struct.unpack_from("<Q", b, 0x28)[0] & 0x0FFFFFFF
    if st + 32 * nst > len(b):
        return need
    for i in range(nst):
        o = st + 32 * i
        count = struct.unpack_from("<H", b, o + 0x1E)[0]
        table = struct.unpack_from("<Q", b, o + 0x10)[0]
        if not count:
            continue
        if table >> 28 != 5:
            return 0
        need = max(need, (table & 0x0FFFFFFF) + 16 * count)
    it = struct.unpack_from("<Q", b, 0x38)[0] & 0x0FFFFFFF
    if it + 16 * nbl > len(b):
        return need
    for i in range(nbl):
        o = it + 16 * i
        size, ptr = struct.unpack_from("<I", b, o + 4)[0], struct.unpack_from("<Q", b, o + 8)[0]
        if size > (1 << 24):
            return 0
        if ptr >> 28 == 5:
            need = max(need, (ptr & 0x0FFFFFFF) + size)
    return need


def needs_more(b):
    """True when a 0x7a header points at system data beyond this stream (a split resource)."""
    return required_len(b) > len(b)


def assemble(m, s, size):
    """Inflate a 0x7a stream and its continuation pages.

    A continuation is normally the next contiguous zlib stream, but a page that does not compress is stored raw.
    Returns None for a stream that does not end inside the archive. Otherwise a dict with the joined bytes,
    stored length of the zlib parts, raw continuation length, spans and whether all system pointers resolve.
    """
    data, used, eof, is_meta, inflated, truncated = inflate(m, s, size)
    if not eof:
        return None
    out = {"data": data, "used": used, "raw_system": 0, "streams": 1, "pages": 1,
           "inflated": inflated, "is_meta": is_meta, "truncated": truncated,
           "spans": [(s, used)], "raw_spans": [], "joined": False, "end": s + used}
    if not is_meta:
        return out
    while out["pages"] < JOIN_MAX:
        row = WI.parse_resource(out["data"])
        if row is not None and not needs_more(out["data"]):
            break
        if out["truncated"] or out["pages"] >= JOIN_MAX:
            break
        nxt = (out["end"] + 15) & ~15
        if nxt >= size:
            break
        if ZHDR.match(m, nxt, size):
            more, mused, meof, mmeta, minfl, mtrunc = inflate(m, nxt, size, keep=True)
            if not meof:
                break
            if mmeta:
                # A continuation never starts a second 0x7a resource; this one is truncated.
                break
            out["data"] += more
            out["used"] += mused
            out["inflated"] += minfl
            out["streams"] += 1
            out["pages"] += 1
            out["joined"] = True
            out["truncated"] = out["truncated"] or mtrunc
            out["spans"].append((nxt, mused))
            out["end"] = nxt + mused
            continue
        need = required_len(out["data"])
        take = need - len(out["data"])
        if take <= 0 or take > (64 << 20) or nxt + take > size:
            break
        extra = m[nxt:nxt + take]
        if not extra or extra.count(0) == len(extra):
            break
        out["data"] += extra
        out["raw_system"] += take
        out["pages"] += 1
        out["joined"] = True
        out["raw_spans"].append((nxt, take))
        out["end"] = nxt + take
    out["complete"] = (not out["truncated"] and WI.parse_resource(out["data"]) is not None
                       and not needs_more(out["data"]))
    return out


def gfx_extent(data):
    """Return (graphics block count, highest 0x6 offset+size) for a parsed 0x7a resource."""
    it = WI._ptr(WI._u64(data, 0x38), len(data))
    count = need = 0
    for i in range(WI._u16(data, 0x4C) if it is not None else 0):
        o = it + 16 * i
        size, ptr = WI._u32(data, o + 4), WI._u64(data, o + 8)
        if ptr >> 28 == 6:
            count += 1
            need = max(need, (ptr & 0x0FFFFFFF) + size)
    return count, need


def has_map_root(data):
    """True when the resource's schema includes CMapData, even if it has no CEntityDef."""
    st = WI._ptr(WI._u64(data, 0x28), len(data))
    return st is not None and any(WI._u32(data, st + 32 * i) == WI.MAPDATA
                                 for i in range(WI._u16(data, 0x48)))


def inspect_map(data, row, res):
    """Add the exactness diagnostics and payload hash to a parsed map row."""
    out = dict(row)
    out.update({"offset": res["offset"], "size": len(data), "stored": res["used"],
                "raw_system": res["raw_system"], "inflated": res["inflated"],
                "streams": res["streams"], "sha1": hashlib.sha1(data).hexdigest(),
                "system_end": res["end"]})
    entity_size = out.get("entity_size") or 0
    it = WI._ptr(WI._u64(data, 0x38), len(data))
    blocks = WI._u16(data, 0x4C) if it is not None else 0
    counted = graphics = unresolved = duplicate = gfx = 0
    gfx_need = 0
    seen = set()
    for i in range(blocks):
        o = it + 16 * i
        name, size, ptr = WI._u32(data, o), WI._u32(data, o + 4), WI._u64(data, o + 8)
        hi = ptr >> 28
        off = ptr & 0x0FFFFFFF
        key = (name, off, size)
        if key in seen:
            duplicate += 1
        seen.add(key)
        if name == WI.ENTITY:
            if hi == 5 and entity_size and size % entity_size == 0 and off + size <= len(data):
                counted += 1
            elif hi == 6:
                graphics += 1
            else:
                unresolved += 1
        elif hi == 6:
            gfx += 1
            gfx_need = max(gfx_need, off + size)
    out.update({"entity_blocks_counted": counted, "entity_blocks_graphics": graphics,
                "entity_blocks_unresolved": unresolved, "duplicate_blocks": duplicate,
                "gfx_blocks": gfx, "gfx_need": gfx_need,
                "exact": graphics == 0 and unresolved == 0 and duplicate == 0})
    return out


def scan_shard(job):
    """Scan one [start, end) window and write its rows, intervals and stats to outdir."""
    archive, index, start, end, outdir, data_start = job
    t0 = time.time()
    stats = {"streams": 0, "stored": 0, "raw_system": 0, "inflated": 0, "rejected": 0, "meta": 0,
             "map_meta": 0, "meta_other": 0, "map_meta_bytes": 0,
             "invalid": 0, "incomplete": 0, "join_max": 0, "entity_gfx": 0,
             "entity_unresolved": 0, "adopted": 0}
    rows, intervals, raw_pages, raw_system = [], [], [], []
    with open(archive, "rb") as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as m:
        size = len(m)
        pos = max(data_start, start - WARMUP)
        search_end = min(size, end + 16)
        while pos < search_end:
            found = ZHDR.search(m, pos, search_end)
            if not found:
                break
            s = found.start()
            if s >= end:
                break
            if s % 16:
                pos = s + 1
                continue
            try:
                res = assemble(m, s, size)
            except (zlib.error, struct.error, IndexError):
                stats["rejected"] += 1
                pos = s + 1
                continue
            if res is None:
                stats["rejected"] += 1
                pos = s + 1
                continue
            for a, used in res["spans"]:
                intervals.append([a, a + used])
            stats["streams"] += res["streams"]
            stats["stored"] += res["used"]
            stats["inflated"] += res["inflated"]
            pos = res["end"]
            if pos <= s:
                pos = s + 1
            if not res["is_meta"]:
                continue
            res["offset"] = s
            if s < start:
                stats["adopted"] += 1
                continue
            raw_system.extend([a, a + used] for a, used in res["raw_spans"])
            stats["raw_system"] += res["raw_system"]
            if not res["complete"]:
                stats["incomplete"] += 1
                continue
            row = WI.parse_resource(res["data"])
            if row is None:
                stats["invalid"] += 1
                continue
            if row["kind"] != "map":
                stats["meta"] += 1
                if has_map_root(res["data"]):
                    stats["map_meta"] += 1
                    stats["map_meta_bytes"] += len(res["data"])
                else:
                    stats["meta_other"] += 1
                stats["join_max"] = max(stats["join_max"], res["pages"])
                _, need = gfx_extent(res["data"])
                if need:
                    raw_pages.append([(res["end"] + 15) & ~15, need])
                continue
            maprow = inspect_map(res["data"], row, res)
            stats["join_max"] = max(stats["join_max"], res["pages"])
            stats["entity_gfx"] += maprow["entity_blocks_graphics"]
            stats["entity_unresolved"] += maprow["entity_blocks_unresolved"]
            rows.append(maprow)
            if maprow["gfx_need"]:
                raw_pages.append([(res["end"] + 15) & ~15, maprow["gfx_need"]])
    rows.sort(key=lambda r: r["offset"])
    doc = {"index": index, "start": start, "end": end, "archive_size": size,
           "data_start": data_start, "elapsed": round(time.time() - t0, 3),
           "complete": True, "format": FORMAT, "stats": stats, "rows": rows,
           "intervals": intervals, "raw_pages": raw_pages, "raw_system": raw_system,
           "map_first": rows[0]["offset"] if rows else None,
           "map_last": rows[-1]["offset"] if rows else None}
    path = os.path.join(outdir, "shard_%03d_%d_%d.json" % (index, start, end))
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(doc, f, separators=(",", ":"))
    os.replace(tmp, path)
    return {"index": index, "start": start, "end": end, "elapsed": doc["elapsed"],
            "maps": len(rows), "entities": sum(r["entities"] for r in rows), **stats}


def _peek_meta(m, s, end):
    """True when a zlib stream at s inflates to a 0x7a header; used only to audit gaps."""
    d = zlib.decompressobj()
    out = bytearray()
    p = s
    while len(out) < 0x15 and p < end:
        piece = m[p:p + 256]
        if not piece:
            break
        try:
            out.extend(d.decompress(piece))
        except zlib.error:
            return False
        p += len(piece)
        if d.eof:
            break
    return len(out) >= 0x15 and out[0x10:0x15] == b"0DRPz"


def classify_gaps(archive, intervals, raw_pages, raw_system, scan_start, scan_end):
    """Merge accepted stream spans and account for every byte in the scanned range.

    ``raw_pages`` are 0x6 graphics extents and ``raw_system`` are uncompressed system-continuation spans;
    both are payload of an adjacent resource, not unexplained bytes.
    """
    spans = sorted((max(scan_start, a), min(scan_end, b)) for a, b in intervals if b > scan_start and a < scan_end)
    merged = []
    for a, b in spans:
        if a >= b:
            continue
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    segments = [(a, a + need, "raw_graphics") for a, need in raw_pages]
    segments += [(a, b, "raw_system") for a, b in raw_system if b > a]
    segments.sort()
    starts = [s[0] for s in segments]
    covered = sum(b - a for a, b in merged)
    result = {"covered": covered, "raw_graphics": 0, "raw_system": 0, "zero_padding": 0, "unread": 0,
              "gaps": 0, "unread_sample": [], "raw_0drp": 0,
              "raw_0drp_offsets": [], "gap_0drp": 0, "gap_0drp_other": 0,
              "raw_maps": 0, "raw_meta": 0, "missed_zlib_meta": 0,
              "missed_zlib_offsets": []}
    with open(archive, "rb") as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as m:
        cursor = scan_start
        bounds = merged + [[scan_end, scan_end]]
        for a, b in bounds:
            if cursor < a:
                result["gaps"] += 1
                pos = cursor
                idx = bisect.bisect_left(starts, cursor)
                while idx < len(segments) and segments[idx][0] < a:
                    s0, s1, kind = segments[idx]
                    start = max(pos, s0)
                    if start > pos:
                        _count_gap(m, pos, start, result)
                    take = max(0, min(s1, a) - start)
                    if take:
                        result[kind] += take
                        _audit_gap(m, start, start + take, result, allow_zlib=True)
                    pos = max(pos, start + take)
                    idx += 1
                if pos < a:
                    _count_gap(m, pos, a, result)
                    _audit_gap(m, pos, a, result, allow_zlib=True)
            cursor = max(cursor, b)
    result["accounted"] = (result["covered"] + result["raw_graphics"] + result["raw_system"]
                           + result["zero_padding"] + result["unread"])
    result["span"] = scan_end - scan_start
    result["coverage"] = result["covered"] / result["span"] if result["span"] else 0.0
    result["accounted_ratio"] = result["accounted"] / result["span"] if result["span"] else 0.0
    if len(result["unread_sample"]) > 200:
        result["unread_sample"] = result["unread_sample"][:200]
    return result


def _count_gap(m, a, b, result):
    if a >= b:
        return
    p = a
    while p < b:
        piece = m[p:min(b, p + GAP_CHUNK)]
        if not piece:
            break
        zero = piece.count(0)
        result["zero_padding"] += zero
        nz = len(piece) - zero
        if nz:
            result["unread"] += nz
            if len(result["unread_sample"]) < 200:
                i = next(k for k, c in enumerate(piece) if c)
                result["unread_sample"].append([p + i, nz])
        p += len(piece)


def _audit_gap(m, a, b, result, allow_zlib):
    """Search an unread gap for a 0x7a signature or an unaligned zlib stream that hides one."""
    if a >= b:
        return
    p = a
    while p < b:
        piece = m[p:min(b, p + GAP_CHUNK)]
        if not piece:
            break
        q = 0
        while True:
            i = piece.find(SIG, q)
            if i < 0:
                break
            at = p + i
            if at + 5 <= b and piece[i + 4] == VERSION:
                result["gap_0drp"] += 1
                start = at - 0x10
                if start >= 0:
                    head = m[start:start + 0x18]
                    if len(head) == 0x18 and head[:8] == b"\0" * 8 and (struct.unpack_from("<Q", head, 8)[0] >> 28) == 5:
                        row = WI.parse_resource(m[start:start + CAP])
                        if row is not None:
                            if row.get("kind") == "map":
                                result["raw_maps"] += 1
                            else:
                                result["raw_meta"] += 1
                            result["raw_0drp"] += 1
                            if len(result["raw_0drp_offsets"]) < 50:
                                result["raw_0drp_offsets"].append(at)
                        else:
                            result["gap_0drp_other"] += 1
                    else:
                        result["gap_0drp_other"] += 1
            q = i + 1
        if allow_zlib:
            q = 0
            while True:
                mt = ZHDR.search(piece, q)
                if not mt:
                    break
                at = p + mt.start()
                if at % 16:
                    if at + 5 <= b and _peek_meta(m, at, b):
                        result["missed_zlib_meta"] += 1
                        if len(result["missed_zlib_offsets"]) < 50:
                            result["missed_zlib_offsets"].append(at)
                q = mt.start() + 1
        p += len(piece)


def _expected_shards(start, end, shard_size):
    if shard_size < 1:  # pos would never advance, and the list would grow until memory ran out
        raise ValueError(f"shard_size must be at least 1, not {shard_size}")
    out = []
    index = 0
    pos = start
    while pos < end:
        stop = min(end, pos + shard_size)
        out.append({"index": index, "start": pos, "end": stop})
        index += 1
        pos = stop
    return out


def _shard_path(outdir, want):
    return os.path.join(outdir, "shard_%03d_%d_%d.json" % (want["index"], want["start"], want["end"]))


def _shard_ok(path, want):
    try:
        with open(path, encoding="utf-8") as f:
            doc = json.load(f)
    except (OSError, ValueError):
        return False
    return (doc.get("format") == FORMAT and doc.get("start") == want["start"]
            and doc.get("end") == want["end"] and doc.get("complete") is True)


def _load_shards(outdir, expected):
    docs = {}
    for want in expected:
        path = _shard_path(outdir, want)
        if _shard_ok(path, want):
            with open(path, encoding="utf-8") as f:
                docs[want["index"]] = json.load(f)
    return docs


def _duplicates(rows):
    by_hash = {}
    for r in rows:
        by_hash.setdefault(r["sha1"], []).append(r)
    groups = [v for v in by_hash.values() if len(v) > 1]
    extra = sum(len(v) - 1 for v in groups)
    extra_entities = sum(sum(x["entities"] for x in v[1:]) for v in groups)
    by_name = {}
    for r in rows:
        if r.get("map_name"):
            by_name.setdefault(r["map_name"], []).append(r)
    ngroups = [v for v in by_name.values() if len(v) > 1]
    return {"content_groups": len(groups), "content_extra_maps": extra,
            "content_extra_entities": extra_entities,
            "content_group_sample": [[v[0]["map_name"], len(v)] for v in groups[:20]],
            "name_groups": len(ngroups), "name_extra_maps": sum(len(v) - 1 for v in ngroups)}


def _bands(rows):
    bands = []
    for r in sorted(rows, key=lambda x: x["offset"]):
        if bands and r["offset"] - bands[-1]["last_end"] <= BAND_GAP:
            bands[-1]["last_end"] = max(bands[-1]["last_end"], r["offset"] + r["stored"])
            bands[-1]["maps"] += 1
            bands[-1]["entities"] += r["entities"]
        else:
            bands.append({"first": r["offset"], "last_end": r["offset"] + r["stored"],
                          "maps": 1, "entities": r["entities"]})
    return bands


def summarize(archive, outdir, start, end, shard_size, data_start):
    expected = _expected_shards(start, end, shard_size)
    docs = _load_shards(outdir, expected)
    missing = [w for w in expected if w["index"] not in docs]
    rows, intervals, raw_pages, raw_system = [], [], [], []
    totals = {"streams": 0, "stored": 0, "raw_system": 0, "inflated": 0, "rejected": 0, "meta": 0,
              "map_meta": 0, "meta_other": 0, "map_meta_bytes": 0,
              "invalid": 0, "incomplete": 0, "join_max": 0, "entity_gfx": 0,
              "entity_unresolved": 0, "adopted": 0}
    shards = []
    for want in expected:
        doc = docs.get(want["index"])
        if doc is None:
            shards.append({**want, "present": False})
            continue
        shards.append({"index": want["index"], "start": want["start"], "end": want["end"],
                       "present": True, "elapsed": doc.get("elapsed"),
                       "maps": len(doc.get("rows", [])),
                       "entities": sum(r["entities"] for r in doc.get("rows", [])),
                       "map_first": doc.get("map_first"), "map_last": doc.get("map_last"),
                       "stats": doc.get("stats", {})})
        rows.extend(doc.get("rows", []))
        intervals.extend(doc.get("intervals", []))
        raw_pages.extend(doc.get("raw_pages", []))
        raw_system.extend(doc.get("raw_system", []))
        for k in totals:
            if k == "join_max":
                totals[k] = max(totals[k], doc.get("stats", {}).get(k, 0))
            else:
                totals[k] += doc.get("stats", {}).get(k, 0)
    seen = set()
    unique = []
    duplicate_offsets = 0
    for r in sorted(rows, key=lambda x: x["offset"]):
        if r["offset"] in seen:
            duplicate_offsets += 1
            continue
        seen.add(r["offset"])
        unique.append(r)
    rows = unique
    gaps = classify_gaps(archive, intervals, raw_pages, raw_system, start, end)
    exact_maps = sum(1 for r in rows if r.get("exact"))
    exact_entities = sum(r["entities"] for r in rows if r.get("exact"))
    summary = {
        "archive": os.path.abspath(archive), "archive_size": os.path.getsize(archive),
        "data_start": data_start, "scan_start": start, "scan_end": end,
        "span": end - start, "shards_expected": len(expected), "shards_complete": len(docs),
        "complete": not missing, "missing_ranges": [[m["start"], m["end"]] for m in missing],
        "elapsed_shard_sum": round(sum(d.get("elapsed", 0) for d in docs.values()), 3),
        "maps": len(rows), "maps_exact": exact_maps,
        "entities": sum(r["entities"] for r in rows),
        "entities_exact": exact_entities,
        "duplicate_offsets": duplicate_offsets,
        "map_first": rows[0]["offset"] if rows else None,
        "map_last": rows[-1]["offset"] if rows else None,
        "totals": totals, "duplicates": _duplicates(rows),
        "bands": _bands(rows), "coverage": gaps, "shards": shards,
    }
    summary["census_exact"] = (summary["complete"] and summary["maps_exact"] == summary["maps"]
                               and totals["entity_gfx"] == 0 and totals["entity_unresolved"] == 0
                               and totals["incomplete"] == 0 and totals["invalid"] == 0
                               and gaps["raw_maps"] == 0 and gaps["missed_zlib_meta"] == 0)
    return summary


def positive_int(text):
    """--shard-size 0 or less never finished planning shards, and --jobs 0 made ProcessPoolExecutor raise."""
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"invalid int value: {text!r}") from None
    if value < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return value


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("archive")
    ap.add_argument("--out", default=os.path.join("work", "world", "census"))
    ap.add_argument("--shard-size", type=positive_int, default=2 << 30)
    ap.add_argument("--jobs", type=positive_int, default=max(1, min(12, (os.cpu_count() or 2) // 2)))
    ap.add_argument("--start", type=int)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--merge-only", action="store_true")
    a = ap.parse_args(argv)
    try:
        with open(a.archive, "rb") as f:
            head = f.read(8)
            size = os.fstat(f.fileno()).st_size
    except OSError as e:
        print(f"{a.archive}: {e.strerror or e}", file=sys.stderr)
        return 2
    if head[:4] != b"9FPR":
        print("not an RPF9 archive", file=sys.stderr)
        return 2
    data_start = struct.unpack_from("<I", head, 4)[0]
    start = a.start if a.start is not None else data_start
    # `is not None`: --limit 0 is an empty range (refused below), not "no limit".
    end = min(size, start + a.limit) if a.limit is not None else size
    if start < data_start or start >= end:
        print("bad scan range", file=sys.stderr)
        return 2
    os.makedirs(a.out, exist_ok=True)
    expected = _expected_shards(start, end, a.shard_size)
    t0 = time.time()
    if not a.merge_only:
        jobs = []
        for want in expected:
            path = _shard_path(a.out, want)
            if a.resume and _shard_ok(path, want):
                continue
            jobs.append((a.archive, want["index"], want["start"], want["end"], a.out, data_start))
        print("census: %d shards, %d to scan, %d jobs, %.2f GiB" %
              (len(expected), len(jobs), a.jobs, (end - start) / (1 << 30)), flush=True)
        failures = 0
        if a.jobs == 1:
            for done, job in enumerate(jobs, 1):
                try:
                    r = scan_shard(job)
                except Exception as e:
                    failures += 1
                    print("[%d/%d] shard %d FAILED: %r" % (done, len(jobs), job[1], e), flush=True)
                    continue
                print("[%d/%d] shard %03d %d..%d maps=%d entities=%d %.1fs" %
                      (done, len(jobs), r["index"], r["start"], r["end"], r["maps"],
                       r["entities"], r["elapsed"]), flush=True)
        else:
            with ProcessPoolExecutor(max_workers=a.jobs) as ex:
                futures = {ex.submit(scan_shard, job): job for job in jobs}
                for done, fut in enumerate(as_completed(futures), 1):
                    job = futures[fut]
                    try:
                        r = fut.result()
                    except Exception as e:  # keep completed shards; the rerun resumes them
                        failures += 1
                        print("[%d/%d] shard %d FAILED: %r" % (done, len(jobs), job[1], e), flush=True)
                        continue
                    print("[%d/%d] shard %03d %d..%d maps=%d entities=%d %.1fs" %
                          (done, len(jobs), r["index"], r["start"], r["end"], r["maps"],
                           r["entities"], r["elapsed"]), flush=True)
        if failures:
            print("%d shard(s) failed; writing the completed coverage" % failures, flush=True)
    summary = summarize(a.archive, a.out, start, end, a.shard_size, data_start)
    summary["elapsed"] = round(time.time() - t0, 3)
    with open(os.path.join(a.out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1, sort_keys=True)
    with open(os.path.join(a.out, "maps.json"), "w", encoding="utf-8") as f:
        json.dump({"summary": {k: summary[k] for k in
                               ("maps", "maps_exact", "entities", "entities_exact", "scan_start",
                                "scan_end", "complete", "census_exact")},
                   "maps": [r for d in _load_shards(a.out, expected).values() for r in d.get("rows", [])]},
                  f, indent=1, sort_keys=True)
    print(json.dumps({k: summary[k] for k in
                      ("maps", "maps_exact", "entities", "entities_exact", "complete", "census_exact",
                       "shards_complete", "shards_expected", "elapsed", "coverage", "duplicates")},
                     indent=1), flush=True)
    return 0 if summary["complete"] else 1


if __name__ == "__main__":
    sys.exit(main())
