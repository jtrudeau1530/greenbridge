"""Index the payloads of an RPF9 archive (or a directory of payloads) for BVH4 data.

The RPF9 table of contents in this package is encrypted, so file names cannot be recovered without keys; this tool
does not try. It walks the data area, decompresses every zlib stream that decodes cleanly to its end, and
classifies the inflated bytes from their headers. Nothing it writes contains game payload bytes: the rows hold
offsets, sizes, counts, a header tag and a bounding box.

The one signature it relies on was measured over the whole `*_bvh4` pack: a BVH stream has the 32-bit tag
0x81020006 at offset 0x100, a size in 64-byte units at 0x104, and pointers at 0x20/0x28 such that

    0x140 + 64 * size_in_64b_units == u32@0x20  and  u32@0x28 == u32@0x20 - 0x100

A stream is reported as `bvh4` only when all of that holds. A 208-byte instance record is reported separately: it
has the constant descriptor words listed in INSTANCE_WORDS and a 3x4 float matrix at 0x50.

Usage:
    python -I tools/bvh_index.py <archive.rpf> [--out index.json] [--csv index.csv]
                                 [--start N] [--limit BYTES] [--max-inflate BYTES]
    python -I tools/bvh_index.py <dir-with-rpf-or-bin> [--out index.json] [--csv index.csv]

Exit status is 0 even when nothing is found; the summary says how many streams were read.
"""
import argparse
import csv
import glob
import json
import mmap
import os
import re
import struct
import sys
import time
import zlib

ZHDR = re.compile(rb"\x78[\x01\x5e\x9c\xda]")
TAG = 0x81020006
HEAD = 0x140
TAG_OFF = 0x100
N_OFF = 0x104
END_OFF = 0x20
ROOT_OFF = 0x28
AABB_OFF = 0x110
INSTANCE_WORDS = ((0x08, 0x50000080), (0x10, 1), (0x18, 0x500000A0), (0x20, 0x60000000),
                  (0x28, 0x500000B0), (0x30, 0x500000C0), (0x38, 0x50000050))
FILL_BYTES = (0x00, 0x81, 0x94, 0x29)


def below_normal():
    if os.name == "nt":
        import ctypes
        k = ctypes.windll.kernel32
        k.SetPriorityClass(k.GetCurrentProcess(), 0x00004000)


def classify(head, size, tail=None):
    """Return a dict describing one payload.

    Only the first 0x140 bytes are inspected, so this works for a carved file or for bytes streaming out of a
    decompressor. `size` is the inflated size.
    """
    row = {"kind": "raw", "signature": "", "version": ""}
    body = head[3:40] if head[:3] == b"\xef\xbb\xbf" else head[:40]
    if body.startswith(b"<?xml"):
        row.update(kind="xml", signature="xml", fragment=False)
        rest = body
        close = rest.find(b"?>")
        if close >= 0:
            rest = rest[close + 2:]
        start = rest.find(b"<")
        end = rest.find(b">", start + 1)
        if 0 <= start < end:
            name = rest[start + 1:end].strip().split(b" ")[0].split(b"/")[0].lstrip(b"!-")
            row["root_element"] = name.decode("ascii", "replace")[:64]
        return row
    probe = head[:256]
    if size >= 64 and len(probe) >= 64:
        printable = sum(1 for c in probe if c in (9, 10, 13) or 32 <= c < 127)
        if printable >= len(probe) * 0.9 and b"<" in probe and b">" in probe:
            row.update(kind="xml", signature="xml", fragment=True)
            return row
    if size == 208 and all(struct.unpack_from("<Q", head, off)[0] == val for off, val in INSTANCE_WORDS):
        h = struct.unpack_from("<Q", head, 0x40)[0]
        w = struct.unpack_from("<I", head, 0xA4)[0]
        row.update(kind="instance208", signature="inst208:0x50000080", hash="%016x" % h,
                   matrix=[round(v, 4) for v in struct.unpack_from("<12f", head, 0x50)],
                   off_a0=struct.unpack_from("<I", head, 0xA0)[0], hi16=w >> 16, lo16=w & 0xFFFF)
        return row
    if 0x60 <= size <= 0x100000 and len(head) >= 0x30 and head[:8] == bytes(8) \
            and struct.unpack_from("<I", head, 8)[0] & 0xFF000000 == 0x50000000 \
            and struct.unpack_from("<I", head, 0xC)[0] == 0:
        row.update(kind="descriptor", signature="desc:0x5000xxxx",
                   desc_word="0x%08x" % struct.unpack_from("<I", head, 8)[0])
        return row
    if size >= HEAD and len(head) >= HEAD and struct.unpack_from("<I", head, TAG_OFF)[0] == TAG:
        n = struct.unpack_from("<I", head, N_OFF)[0]
        end = struct.unpack_from("<I", head, END_OFF)[0]
        root = struct.unpack_from("<I", head, ROOT_OFF)[0]
        if 0x140 + n * 64 == end and root == end - 0x100:
            common = dict(signature="bvh4:0x81020006", version="0x81020006",
                          node_count=n, node_bytes=n * 64, root_offset=root, declared_bytes=end,
                          bounds_min=[round(v, 4) for v in struct.unpack_from("<3f", head, AABB_OFF)],
                          bounds_max=[round(v, 4) for v in struct.unpack_from("<3f", head, AABB_OFF + 12)],
                          count_130=struct.unpack_from("<I", head, 0x130)[0],
                          count_134=struct.unpack_from("<I", head, 0x134)[0],
                          base_138=struct.unpack_from("<i", head, 0x138)[0])
            if end <= size:
                row.update(kind="bvh4", chunked=False, **common)
            elif size % 65536 == 0:
                # A payload larger than 8 MiB is split into 64 KiB-multiple RPF chunks; the header's node range
                # continues in the following chunks, so the declared end is past this chunk's inflated size.
                row.update(kind="bvh4-chunk", chunked=True, **common)
    return row


def inflate_stream(m, s, size, max_inflate):
    """Inflate the zlib stream at `s`. Returns (head, kept, total, used) or None if it does not decode fully.

    `kept` is the payload capped at max_inflate bytes; `total` is the real inflated size.
    """
    d = zlib.decompressobj()
    keep, kept, total, p = [], 0, 0, s

    def take(b):
        nonlocal kept
        if b and kept < max_inflate:
            piece = b[:max_inflate - kept]
            keep.append(piece)
            kept += len(piece)

    try:
        while not d.eof and p < size:
            chunk = m[p:p + (1 << 20)]
            if not chunk:
                break
            take(d.decompress(chunk))
            p += len(chunk)
            if d.unused_data:
                break
        take(d.flush())
    except zlib.error:
        return None
    total = kept if kept < max_inflate else _count_stream(m, s, size)
    if not d.eof or total < 64:
        return None
    used = p - s - len(d.unused_data)
    out = b"".join(keep)
    return out[:HEAD], out, total, used


def _count_stream(m, s, size):
    """Inflated length of a stream larger than the kept cap (second cheap pass, no output kept)."""
    d = zlib.decompressobj()
    total, p = 0, s
    while not d.eof and p < size:
        chunk = m[p:p + (1 << 20)]
        if not chunk:
            break
        total += len(d.decompress(chunk))
        p += len(chunk)
        if d.unused_data:
            break
    total += len(d.flush())
    return total


def scan_rpf(path, start=None, limit=None, max_inflate=1 << 28):
    rows = []
    t0 = time.time()
    with open(path, "rb") as fh:
        m = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            size = len(m)
            pos = start if start is not None else struct.unpack_from("<I", m, 4)[0]
            end = min(size, pos + limit) if limit else size
            span = end - pos
            last, zero, unread = pos, 0, 0
            while pos < end:
                mt = ZHDR.search(m, pos, end)
                if not mt:
                    break
                s = mt.start()
                got = inflate_stream(m, s, size, max_inflate)
                if got is None:
                    pos = s + 1
                    continue
                head, kept, total, used = got
                gap = m[last:s]
                if gap.count(0) == len(gap):
                    zero += len(gap)
                else:
                    unread += len(gap)
                row = classify(head, total, kept[-HEAD:] if len(kept) >= HEAD else kept)
                row.update(source=path, offset=s, stored=used, size=total)
                if row["kind"] in ("bvh4", "bvh4-chunk") and row["declared_bytes"] <= total:
                    body = kept[HEAD:HEAD + row["node_count"] * 64]
                    row["fill"] = {("%02x" % b): body.count(bytes([b])) for b in FILL_BYTES if body.count(bytes([b]))}
                    row["fill"]["total"] = len(body)
                rows.append(row)
                last = pos = s + used
            tailgap = m[last:end]
            if tailgap.count(0) == len(tailgap):
                zero += len(tailgap)
            else:
                unread += len(tailgap)
        finally:
            m.close()
    return rows, [summarize(rows, path, span, zero, unread, time.time() - t0)]


def summarize(rows, source, span, zero, unread, elapsed):
    kinds = {}
    for r in rows:
        kinds[r["kind"]] = kinds.get(r["kind"], 0) + 1
    stored = sum(r["stored"] for r in rows)
    inflated = sum(r["size"] for r in rows)
    bvh = [r for r in rows if r["kind"] in ("bvh4", "bvh4-chunk")]
    total_nodes = sum(r["node_count"] for r in bvh)
    return {"source": str(source), "streams": len(rows), "kinds": kinds, "stored": stored,
            "inflated": inflated, "span": span, "zero_padding": zero, "unread": unread,
            "readable_fraction": round(stored / span, 6) if span else None,
            "bvh4_streams": len(bvh), "bvh4_complete": sum(1 for r in bvh if r["kind"] == "bvh4"),
            "bvh4_chunked": sum(1 for r in bvh if r["kind"] == "bvh4-chunk"),
            "bvh4_nodes": total_nodes, "bvh4_node_bytes": total_nodes * 64,
            "elapsed_s": round(elapsed, 2)}


def scan_dir(path):
    rpfs = sorted(glob.glob(os.path.join(path, "**", "*.rpf"), recursive=True))
    rows, summaries = [], []
    if rpfs:
        for p in rpfs:
            r, s = scan_rpf(p)
            rows += r
            summaries.append(s)
        return rows, summaries
    for p in sorted(glob.glob(os.path.join(path, "**", "*.bin"), recursive=True)):
        size = os.path.getsize(p)
        with open(p, "rb") as f:
            head = f.read(HEAD)
            if size > HEAD:
                f.seek(-HEAD, os.SEEK_END)
                tail = f.read(HEAD)
            else:
                tail = head
        row = classify(head, size, tail)
        row.update(source=p, offset=0, stored=size, size=size)
        if row["kind"] in ("bvh4", "bvh4-chunk"):
            with open(p, "rb") as f:
                body = f.read(HEAD + row["node_count"] * 64)[HEAD:]
            row["fill"] = {("%02x" % b): body.count(bytes([b])) for b in FILL_BYTES}
            row["fill"]["total"] = len(body)
        rows.append(row)
    summary = summarize(rows, path, 0, 0, 0, 0.0)
    summary["span"] = None
    summary["readable_fraction"] = None
    return rows, [summary]


def write_outputs(rows, summaries, out, csv_path):
    if out:
        payload = summaries if len(summaries) > 1 else (summaries[0] if summaries else {})
        with open(out, "w", encoding="utf-8") as f:
            json.dump({"summary": payload, "streams": rows}, f, indent=0)
    if csv_path:
        cols = ["source", "offset", "stored", "size", "kind", "signature", "version", "node_count",
                "node_bytes", "root_offset", "declared_bytes", "chunked", "bounds_min", "bounds_max",
                "hash", "off_a0", "hi16", "lo16", "desc_word", "fragment"]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            for r in rows:
                r = dict(r)
                for k in ("bounds_min", "bounds_max", "matrix"):
                    if k in r:
                        r[k] = " ".join(str(v) for v in r[k])
                w.writerow(r)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("target", help="RPF9 archive or a directory of .rpf/.bin payloads")
    ap.add_argument("--out", help="write JSON (summary + per-stream rows)")
    ap.add_argument("--csv", help="write CSV (offset, size, signature, readable fields)")
    ap.add_argument("--start", type=int, help="data-area start (default: u32 at file offset 4)")
    ap.add_argument("--limit", type=int, help="stop after this many bytes of the data area")
    ap.add_argument("--max-inflate", type=int, default=1 << 28, help="keep at most this many bytes of one stream")
    a = ap.parse_args(argv)
    below_normal()
    if os.path.isdir(a.target):
        rows, summaries = scan_dir(a.target)
    else:
        rows, summaries = scan_rpf(a.target, a.start, a.limit, a.max_inflate)
    for s in summaries:
        print(json.dumps(s, indent=1))
    write_outputs(rows, summaries, a.out, a.csv)
    if not rows:
        print("no zlib streams found", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
