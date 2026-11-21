"""Index the story/minigame resource pages in an RPF9 data area or a folder of carved streams.

The RPF9 table of contents is encrypted, so this walks the data area the same way rpf9_carve.py
does: find every zlib stream that inflates cleanly to its end, then classify the inflated bytes.
It never decrypts, never writes game data, and does not decode resources.

The story-specific signature is `PSIN`: a section chain whose records are a four-byte tag
(`LSCH`, `LENM`, `LSNS`, `LGID`, `LRSC`, `CHKS`) followed by a big-endian u32 length that includes
the eight-byte record header. `0DRP` pages (magic at +16, version at +0x14, page count at +0x20)
and `RSC8` resource headers (version at +4) are reported with their measured fields. Anything else
is reported as `raw:<hex>` instead of being guessed.

Usage:
    python -I story_index.py <archive.rpf> <out.csv|out.json> [--start N] [--limit BYTES] [--quiet]
    python -I story_index.py <carved-dir> <out.csv|out.json> [--quiet]
"""
import argparse
import csv
import json
import mmap
import os
import re
import struct
import sys
import time
import zlib

# zlib stream start, or a Bink 2 video ('KB2' + revision letter). Kept in step with rpf9_carve.py.
ZHDR = re.compile(rb"\x78[\x01\x5e\x9c\xda]|KB2[a-z]")
PSIN_TAGS = (b"LSCH", b"LENM", b"LSNS", b"LGID", b"LRSC", b"CHKS")
COLUMNS = ["offset", "stored", "size", "signature", "version", "count", "detail"]


def _u32(data, at):
    return struct.unpack_from("<I", data, at)[0] if len(data) >= at + 4 else None


def _u64(data, at):
    return struct.unpack_from("<Q", data, at)[0] if len(data) >= at + 8 else None


def _be32(data, at):
    return int.from_bytes(data[at:at + 4], "big") if len(data) >= at + 4 else 0


def psin_sections(data):
    """Return a valid PSIN section chain as [(tag, offset, length), ...].

    The header's big-endian first-section offset at +4 is authoritative when it points at a known
    tag. Otherwise every LSCH occurrence is tried and the longest chain wins. A record is tag +
    big-endian u32 length, and the length counts the record header. The chain ends at CHKS or when
    the next record is not a known tag. Only measured tags are accepted.
    """
    def parse(at):
        chain = []
        pos = at
        while pos + 8 <= len(data):
            tag = data[pos:pos + 4]
            if tag not in PSIN_TAGS:
                break
            size = _be32(data, pos + 4)
            if size < 8 or pos + size > len(data):
                break
            chain.append((tag.decode("ascii"), pos, size))
            if tag == b"CHKS":
                break
            pos += size
        return chain

    first = _be32(data, 4)
    if 8 <= first < len(data) and data[first:first + 4] in PSIN_TAGS:
        chain = parse(first)
        if chain:
            return chain
    best = []
    start = 0
    while True:
        at = data.find(b"LSCH", start)
        if at < 0:
            break
        chain = parse(at)
        if len(chain) > len(best):
            best = chain
        start = at + 1
    return best


def _printable(data):
    if not data:
        return False
    sample = data[:512]
    good = sum(1 for b in sample if 9 <= b < 127 or b in (10, 13))
    return len(sample) >= 64 and good / len(sample) > 0.9


def _is_xml(data):
    if data.startswith(b"\xef\xbb\xbf"):
        data = data[3:]
    return data.lstrip()[:5] == b"<?xml"


def classify(data, offset, stored):
    """Return one record for an inflated stream. `offset` is its archive offset, `stored` its stored size."""
    record = {"offset": offset, "stored": stored, "size": len(data), "signature": "raw",
              "version": None, "count": None, "detail": ""}
    if data[:4] == b"PSIN":
        chain = psin_sections(data)
        record.update(signature="psin", count=len(chain),
                      detail="sections=" + ",".join(t for t, _, _ in chain))
    elif data[:4] == b"RBF0":
        schema = re.search(rb"[\x20-\x7e]{4,120}", data[8:160])
        record.update(signature="rbf0",
                      detail=(schema.group().decode("ascii", "replace") if schema else "rig metadata"))
    elif len(data) >= 0x30 and data[16:20] == b"0DRP":
        record.update(signature="page-directory", version=_u32(data, 0x14), count=_u64(data, 0x20),
                      detail="0DRP at +16")
    elif data[:4] == b"RSC8" or data[:4] == b"RSC7":
        version, declared, flags = _u32(data, 4), _u32(data, 8), _u32(data, 12)
        record.update(signature=data[:4].decode("ascii").lower(), version=version,
                      detail=f"declared_size={declared} flags={flags}")
    elif data[:4] == b"9FPR":
        record.update(signature="rpf9-nested", detail=f"data_off={_u32(data, 4)} toc={_u32(data, 8)}")
    elif data[:4] == b"MoVE":
        record.update(signature="move", version=_u32(data, 4), detail=f"size_hint={_u32(data, 0x14)}")
    elif _is_xml(data):
        record.update(signature="xml", detail="XML text")
    elif (len(data) >= 16 and _u32(data, 4) == 8 and _u32(data, 12) == 0
          and 0 < (_u32(data, 8) or 0) <= 4096):
        record.update(signature="typehash", version=8, count=_u32(data, 8),
                      detail=f"type=0x{_u32(data, 0):08x}")
    elif len(data) >= 8 and _u32(data, 0) in (2, 3) and data[4:8] == b"\x01\x01\x00\x00":
        record.update(signature="clip-stream", version=_u32(data, 0), detail="24-byte header")
    elif data[:3] == b"KB2" and len(data) >= 0x20:
        record.update(signature="bink2", version=chr(data[3]), count=_u32(data, 8),
                      detail=f"{_u32(data, 0x14)}x{_u32(data, 0x18)}")
    elif data[:8] == b"\x00" * 8:
        record.update(signature="zero-prefix", detail="binary page")
    elif _printable(data):
        record.update(signature="text", detail="printable text")
    else:
        record.update(signature="raw:" + data[:4].hex())
    return record


def _gap(gap):
    return (len(gap), 0) if gap.count(0) == len(gap) else (0, len(gap))


def scan_rpf(path, start=None, limit=None, progress=None):
    """Walk an RPF9 data area. Returns (records, summary). No tables of contents are read or decrypted."""
    rows = []
    t0 = time.time()
    with open(path, "rb") as fh:
        m = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
        size = len(m)
        first = struct.unpack_from("<I", m, 4)[0] if size >= 8 else 0
        pos = start if start is not None else first
        if not 0 <= pos <= size:
            raise ValueError("start offset is outside the archive")
        end = min(size, pos + limit) if limit else size
        span_start = pos
        last, zero, unread = pos, 0, 0
        while pos < end:
            match = ZHDR.search(m, pos, end)
            if not match:
                break
            s = match.start()
            if m[s:s + 3] == b"KB2":
                blen = (_u32(m, s + 4) or 0) + 8
                w, h = _u32(m, s + 0x14), _u32(m, s + 0x18)
                if not (64 <= blen <= size - s and w and h and 16 <= w <= 8192 and 16 <= h <= 8192):
                    pos = s + 1
                    continue
                z, u = _gap(m[last:s])
                zero += z
                unread += u
                head = m[s:s + min(blen, 0x20)]
                record = classify(head, s, blen)
                record["size"] = blen
                rows.append(record)
                last = pos = s + blen
                continue
            d = zlib.decompressobj()
            chunks, p = [], s
            try:
                while not d.eof and p < size:
                    block = m[p:p + (1 << 20)]
                    chunks.append(d.decompress(block))
                    p += len(block)   # advance by the actual slice, not the requested 1 MiB
                    if d.unused_data or d.eof:
                        break
                out = b"".join(chunks) + d.flush()
            except zlib.error:
                pos = s + 1
                continue
            if not d.eof or len(out) < 64:
                pos = s + 1
                continue
            used = p - s - len(d.unused_data)
            z, u = _gap(m[last:s])
            zero += z
            unread += u
            rows.append(classify(out, s, used))
            last = pos = s + used
            if progress and len(rows) % 2000 == 0:
                progress(len(rows), pos, end)
        z, u = _gap(m[last:end])
        zero += z
        unread += u
        m.close()
    summary = summarise(rows, end - span_start, zero, unread)
    summary["mode"] = "rpf9"
    summary["archive"] = path
    summary["elapsed_seconds"] = round(time.time() - t0, 3)
    return rows, summary


def scan_dir(path, progress=None):
    """Index already-carved files. An offset is taken from a leading hexadecimal file name when possible."""
    entries = []
    t0 = time.time()
    for name in sorted(os.listdir(path)):
        full = os.path.join(path, name)
        if not os.path.isfile(full):
            continue
        with open(full, "rb") as f:
            data = f.read()
        m = re.match(r"([0-9a-fA-F]{6,})", name)
        offset = int(m.group(1), 16) if m else len(entries)
        entries.append((offset, data))
    rows = [classify(data, offset, len(data)) for offset, data in entries]
    summary = summarise(rows, sum(len(data) for _, data in entries), 0, 0)
    summary["mode"] = "directory"
    summary["directory"] = path
    summary["elapsed_seconds"] = round(time.time() - t0, 3)
    if progress:
        progress(len(rows), len(rows), len(rows))
    return rows, summary


def summarise(rows, span, zero, unread):
    by_sig = {}
    for r in rows:
        a = by_sig.setdefault(r["signature"], {"files": 0, "stored": 0, "inflated": 0})
        a["files"] += 1
        a["stored"] += r["stored"]
        a["inflated"] += r["size"]
    stored = sum(r["stored"] for r in rows)
    return {
        "files": len(rows),
        "compressed_bytes": stored,
        "inflated_bytes": sum(r["size"] for r in rows),
        "bink2_files": sum(1 for r in rows if r["signature"] == "bink2"),
        "bink2_bytes": sum(r["stored"] for r in rows if r["signature"] == "bink2"),
        "zero_padding": zero,
        "unread_bytes": unread,
        "span": span,
        "readable_fraction": round(stored / span, 6) if span else 0.0,
        "signatures": by_sig,
    }


def write_output(path, rows, summary):
    """Write CSV or JSON from the file extension."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".json":
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"summary": summary, "files": rows}, f, indent=1)
        return
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(COLUMNS)
        for r in rows:
            w.writerow([f"0x{r['offset']:x}" if i == 0 else ("" if r[c] is None else r[c])
                        for i, c in enumerate(COLUMNS)])


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("source", help="an .rpf archive or a directory of carved .bin files")
    ap.add_argument("out", help="output path ending in .csv or .json")
    ap.add_argument("--start", type=lambda s: int(s, 0), help="first byte to scan (default: the RPF data start)")
    ap.add_argument("--limit", type=lambda s: int(s, 0), help="number of bytes to scan")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()
    if os.path.isdir(a.source):
        rows, summary = scan_dir(a.source)
    else:
        def progress(n, pos, end):
            if not a.quiet:
                print(f"[{pos / max(1, end):6.1%}] {n} streams", flush=True)
        rows, summary = scan_rpf(a.source, a.start, a.limit, progress)
    write_output(a.out, rows, summary)
    top = dict(sorted(summary["signatures"].items(), key=lambda kv: -kv[1]["files"])[:15])
    print(json.dumps({k: summary[k] for k in
                      ("files", "compressed_bytes", "inflated_bytes", "zero_padding", "unread_bytes",
                       "span", "readable_fraction", "bink2_files", "elapsed_seconds")}, indent=1))
    print(f"{len(summary['signatures'])} signature classes; top 15:")
    for name, counts in top.items():
        print(f"  {name:24} {counts['files']:6} files {counts['stored']:12} stored {counts['inflated']:12} inflated")
    return 0


if __name__ == "__main__":
    sys.exit(main())
