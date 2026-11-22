"""Index the stream signatures in the rest-of-world and script RPF9 archives.

The RPF9 table of contents is encrypted, so this walks the data area the same way
`rpf9_carve.py` does: find every zlib stream that inflates cleanly to its end, plus
validated Bink 2 streams, and classify what starts each stream. It reports the
measured signatures from `docs/restofworld.md` and writes one row per stream. It
does not decode resources, write payload bytes, or touch the table of contents.

Signatures are only accepted at a stream start, except `0DRP`, which is accepted at
+0 or +0x10 because the PSIN LRSC payload and a 0x7a metadata resource use both
placements. A four-byte magic inside an inflated page or XML text is not a stream
signature; that distinction matters because `ADAT` occurs inside `METADATA`, and
`AWC` occurs inside printable names.

Usage:
    python -I restofworld_index.py <archive.rpf> <out.csv|out.json> [--start N] [--limit BYTES]
    python -I restofworld_index.py <carved-dir> <out.csv|out.json>
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

# zlib stream start, or Bink 2 video ('KB2' + revision letter). Kept in step with rpf9_carve.py.
ZHDR = re.compile(rb"\x78[\x01\x5e\x9c\xda]|KB2[a-z]")
PSIN_TAGS = (b"LSCH", b"LENM", b"LSNS", b"LGID", b"LRSC", b"CHKS")
COLUMNS = ["offset", "stored", "size", "signature", "version", "count", "detail"]
MAX_INFLATE = 512 << 20


def _u32(data, at):
    return struct.unpack_from("<I", data, at)[0] if len(data) >= at + 4 else None


def _u64(data, at):
    return struct.unpack_from("<Q", data, at)[0] if len(data) >= at + 8 else None


def _be32(data, at):
    return int.from_bytes(data[at:at + 4], "big") if len(data) >= at + 4 else 0


def _printable(data):
    if not data:
        return False
    sample = data[:512]
    good = sum(1 for b in sample if 9 <= b < 127 or b in (10, 13))
    return len(sample) >= 64 and good / len(sample) > 0.9


def psin_sections(data):
    """Return a PSIN section chain as [(tag, offset, size), ...].

    The header's big-endian first-section offset at +4 is authoritative when it points at a
    known tag. A record is a four-byte tag and a big-endian u32 total size that includes the
    eight-byte record header. The chain stops at CHKS or at the first unknown/out-of-range
    record. Only the six measured tags are accepted.
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


def _is_xml(data):
    if data.startswith(b"\xef\xbb\xbf"):
        data = data[3:]
    return data.lstrip()[:5] == b"<?xml"


def classify(data, offset, stored):
    """Return one row for an inflated stream or a Bink 2 header."""
    row = {"offset": offset, "stored": stored, "size": len(data), "signature": "raw",
           "version": None, "count": None, "detail": ""}
    if data[:4] == b"PSIN":
        chain = psin_sections(data)
        complete = bool(chain) and chain[-1][0] == "CHKS" and chain[-1][1] + chain[-1][2] == len(data)
        row.update(signature="psin", count=len(chain),
                   detail=f"tags={','.join(t for t, _, _ in chain)};"
                          f"field8={_u32(data, 8)};complete={int(complete)}")
        return row
    if data[:4] == b"RBF0":
        found = re.search(rb"[\x20-\x7e]{4,120}", data[8:160])
        schema = found.group().decode("ascii", "replace") if found else "?"
        row.update(signature="rbf0", detail=f"schema={schema}")
        return row
    for at in (0x10, 0):
        if data[at:at + 4] == b"0DRP":
            count = _u64(data, at + 0x10) if at == 0x10 else _u32(data, at + 0x0c)
            row.update(signature="0drp", version=_u32(data, at + 4), count=count,
                       detail=f"0DRP@{at:#x}")
            return row
    if data[:4] in (b"RSC8", b"RSC7"):
        row.update(signature=data[:4].decode("ascii").lower(), version=_u32(data, 4),
                   detail=f"declared={_u32(data, 8)};flags={_u32(data, 12)}")
        return row
    if data[:4] == b"9FPR":
        row.update(signature="rpf9-nested",
                   detail=f"data_off={_u32(data, 4)};toc={_u32(data, 8)}")
        return row
    if data[:4] == b"AWC\x00":
        row.update(signature="awc", version=_u32(data, 4))
        return row
    if data[:4] == b"ADAT":
        row.update(signature="adat")
        return row
    if data[:4] == b"FSSM":
        row.update(signature="fssm", version=_u32(data, 4))
        return row
    if data[:4] == b"SCRP":
        row.update(signature="scrp", version=_u32(data, 4))
        return row
    if data[:4] == b"YSC\x00":
        row.update(signature="ysc")
        return row
    if data[:4] == b"RSCC":
        row.update(signature="rscc", version=_u32(data, 4))
        return row
    if data[:3] == b"KB2" and len(data) >= 0x20:
        row.update(signature="bink2", version=chr(data[3]), count=_u32(data, 8),
                   detail=f"{_u32(data, 0x14)}x{_u32(data, 0x18)}", size=stored)
        return row
    if data[:4] == b"\x42\x40\x00\x00":
        row.update(signature="respage", detail="0x4042")
        return row
    if data[:4] in (b"DDS ", b"DXT1", b"DXT3", b"DXT5", b"ATI2", b"BC5\x00", b"BC7\x00"):
        row.update(signature=data[:4].decode("ascii", "replace"))
        return row
    if _is_xml(data):
        row.update(signature="xml", detail="XML text")
        return row
    if len(data) >= 16 and _u32(data, 4) == 8 and _u32(data, 12) == 0 and 0 < (_u32(data, 8) or 0) <= 4096:
        row.update(signature="typehash", version=8, count=_u32(data, 8),
                   detail=f"type={_u32(data, 0):#010x}")
        return row
    if len(data) >= 8 and _u32(data, 0) in (2, 3) and data[4:8] == b"\x01\x01\x00\x00":
        row.update(signature="clip", version=_u32(data, 0))
        return row
    if data[:8] == b"\x00" * 8:
        pointer = _u64(data, 8)
        row.update(signature="zero-prefix",
                   detail=f"u64@8={pointer:#x}" if pointer is not None else "u64@8=?")
        return row
    if _printable(data):
        row.update(signature="text", detail="printable text")
        return row
    row.update(signature="raw:" + data[:4].hex())
    return row


def _gap(gap):
    return (len(gap), 0) if gap.count(0) == len(gap) else (0, len(gap))


def scan_rpf(path, start=None, limit=None, max_inflate=MAX_INFLATE):
    """Walk an RPF9 data area. Returns (rows, summary). No TOC is read or decrypted."""
    rows = []
    zero = unread = skipped = 0
    t0 = time.time()
    if os.path.getsize(path) == 0:
        raise ValueError("empty file")
    with open(path, "rb") as fh:
        with mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ) as m:
            size = len(m)
            first = struct.unpack_from("<I", m, 4)[0] if size >= 8 else 0
            pos = start if start is not None else first
            if not 0 <= pos <= size:
                raise ValueError("start offset is outside the archive")
            end = min(size, pos + limit) if limit else size
            span_start = pos
            last = pos
            while pos < end:
                found = ZHDR.search(m, pos, end)
                if not found:
                    break
                s = found.start()
                if m[s:s + 3] == b"KB2":
                    blen = (_u32(m, s + 4) or 0) + 8
                    w, h = _u32(m, s + 0x14), _u32(m, s + 0x18)
                    if not (64 <= blen <= end - s and w and h and 16 <= w <= 8192 and 16 <= h <= 8192):
                        pos = s + 1
                        continue
                    z, u = _gap(m[last:s])
                    zero += z
                    unread += u
                    rows.append(classify(m[s:s + 256], s, blen))
                    last = pos = s + blen
                    continue
                d = zlib.decompressobj()
                p, total, head, psin = s, 0, b"", bytearray()
                too_big = False
                try:
                    while not d.eof and p < end:   # inside the --limit window, as in story_index (#712)
                        block = m[p:min(p + (1 << 20), end)]
                        if not block:
                            break
                        p += len(block)
                        while block and not d.eof:
                            chunk = d.decompress(block, 1 << 20)
                            block = d.unconsumed_tail
                            if chunk:
                                total += len(chunk)
                                if total > max_inflate:
                                    too_big = True
                                    break
                                if len(head) < 256:
                                    head += chunk[:256 - len(head)]
                                if len(head) < 4 or head[:4] == b"PSIN":   # classify walks the whole PSIN chain
                                    psin += chunk
                            if not block:
                                break
                        if too_big:
                            break
                except zlib.error:
                    pos = s + 1
                    continue
                if too_big or not d.eof or total < 64:
                    skipped += 1
                    pos = s + 1
                    continue
                used = p - s - len(d.unused_data)
                z, u = _gap(m[last:s])
                zero += z
                unread += u
                row = classify(bytes(psin) if head[:4] == b"PSIN" else head, s, used)
                row["size"] = total
                rows.append(row)
                last = pos = s + used
            z, u = _gap(m[last:end])
            zero += z
            unread += u
    summary = summarise(rows, end - span_start, zero, unread)
    summary.update(mode="rpf9", archive=path, elapsed_seconds=round(time.time() - t0, 3),
                   skipped_candidates=skipped)
    return rows, summary


def scan_dir(path, max_inflate=MAX_INFLATE):
    """Index already-carved files. A leading hexadecimal file name supplies the offset."""
    rows = []
    skipped = 0
    t0 = time.time()
    for name in sorted(os.listdir(path)):
        full = os.path.join(path, name)
        if not os.path.isfile(full) or not (name.endswith(".bin") or name.endswith(".bk2")):
            continue
        if name.endswith(".bin") and os.path.getsize(full) > max_inflate:
            skipped += 1
            continue
        found = re.match(r"([0-9a-fA-F]{6,})", name)
        offset = int(found.group(1), 16) if found else len(rows)
        with open(full, "rb") as f:
            data = f.read()
        if len(data) >= 3 and data[:3] == b"KB2":
            row = classify(data, offset, len(data))
        else:
            row = classify(data, offset, len(data))
            row["size"] = len(data)
        rows.append(row)
    summary = summarise(rows, sum(r["stored"] for r in rows), 0, 0)
    summary.update(mode="directory", directory=path, elapsed_seconds=round(time.time() - t0, 3),
                   skipped_candidates=skipped)
    return rows, summary


def summarise(rows, span, zero, unread):
    by_sig = {}
    for row in rows:
        entry = by_sig.setdefault(row["signature"], {"files": 0, "stored": 0, "inflated": 0})
        entry["files"] += 1
        entry["stored"] += row["stored"]
        entry["inflated"] += row["size"]
    stored = sum(row["stored"] for row in rows)
    return {
        "files": len(rows),
        "compressed_bytes": stored,
        "inflated_bytes": sum(row["size"] for row in rows),
        "bink2_files": sum(1 for row in rows if row["signature"] == "bink2"),
        "bink2_bytes": sum(row["stored"] for row in rows if row["signature"] == "bink2"),
        "zero_padding": zero,
        "unread_bytes": unread,
        "span": span,
        "readable_fraction": round(stored / span, 6) if span else 0.0,
        "signatures": by_sig,
    }


def write_output(path, rows, summary):
    """Write CSV or JSON from the file extension."""
    if os.path.splitext(path)[1].lower() == ".json":
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            json.dump({"summary": summary, "files": rows}, f, indent=1)
        return
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(COLUMNS)
        for row in rows:
            writer.writerow([f"0x{row['offset']:x}" if i == 0 else ("" if row[c] is None else row[c])
                             for i, c in enumerate(COLUMNS)])


def offset(text):
    """--start: 0 or more, decimal or 0x hex."""
    try:
        value = int(text, 0)
    except ValueError:
        raise argparse.ArgumentTypeError("must be 0 or more") from None
    if value < 0:
        raise argparse.ArgumentTypeError("must be 0 or more")
    return value


def byte_count(text):
    """--limit/--max-inflate: at least 1; 0 used to mean no limit."""
    try:
        value = int(text, 0)
    except ValueError:
        raise argparse.ArgumentTypeError("must be at least 1") from None
    if value < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return value


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("source", help="an .rpf archive or a directory of carved .bin/.bk2 files")
    ap.add_argument("out", help="output path ending in .csv or .json")
    ap.add_argument("--start", type=offset, help="first byte to scan (default: RPF data start)")
    ap.add_argument("--limit", type=byte_count, help="number of bytes to scan")
    ap.add_argument("--max-inflate", type=byte_count, default=MAX_INFLATE,
                    help="skip a stream that inflates above this many bytes (default: 512 MiB)")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    if os.path.isdir(args.source) and (args.start is not None or args.limit is not None):
        ap.error("--start and --limit apply to a single archive, not a directory")
    if os.path.isdir(args.out):
        ap.error(f"{args.out} is a folder, not a .csv or .json file")
    folder = os.path.dirname(os.path.abspath(args.out))
    if not os.path.isdir(folder):
        ap.error(f"{folder} is not a folder, so {args.out} cannot be written")
    try:
        overwrites_source = os.path.samefile(args.source, args.out)
    except FileNotFoundError:
        overwrites_source = False
    except OSError as e:
        ap.error(f"cannot compare source and output paths: {e}")
    if overwrites_source:
        ap.error("output must not overwrite the source archive")
    try:
        if os.path.isdir(args.source):
            rows, summary = scan_dir(args.source, args.max_inflate)
        else:
            rows, summary = scan_rpf(args.source, args.start, args.limit, args.max_inflate)
    except ValueError as e:
        sys.exit(f"{args.source}: {e}")
    except OSError as e:
        sys.exit(f"{args.source}: {e.strerror or e}")
    write_output(args.out, rows, summary)
    if not args.quiet:
        print(json.dumps({k: summary[k] for k in
                          ("files", "compressed_bytes", "inflated_bytes", "zero_padding",
                           "unread_bytes", "span", "readable_fraction", "bink2_files",
                           "skipped_candidates", "elapsed_seconds")}, indent=1))
        for name, counts in sorted(summary["signatures"].items(), key=lambda kv: -kv[1]["files"]):
            print(f"  {name:24} {counts['files']:6} files {counts['stored']:12} stored "
                  f"{counts['inflated']:12} inflated")
    return 0


if __name__ == "__main__":
    try:  # a redirected stdout is strict ANSI code page under -I; write '?' for what it cannot encode
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    sys.exit(main())
