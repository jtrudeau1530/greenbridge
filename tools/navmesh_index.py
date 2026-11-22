"""Index the RPF9 resource page-directory streams in a navmesh archive or a carve folder.

The measured directory page carries the four bytes ``0DRP`` at offset 16, followed by version ``0x7A``
(122), a presumed page count at offset 32 and an opaque 64-bit key at offset 0x60. That marker is the RPF9
page-directory container, **not** a navmesh-specific magic (it also occurs in ``data.rpf`` and
``levels.rpf``), so this tool indexes containers. Run it on ``navmeshes.rpf`` to inventory its directory
pages.

The archive walk is the fixed one from ``rpf9_carve.py`` (advance by the bytes actually read, not by the
1 MiB chunk size). Tables of contents are encrypted and are not touched; only zlib streams are read.
No payload is written, and the output is metadata only.

Usage:
    python -I navmesh_index.py <archive.rpf|dir> [--out FILE] [--csv FILE]
                              [--start N] [--limit BYTES] [--pattern GLOB]

    --out     JSON path (default: <input name>.navmesh-index.json in the current directory)
    --csv     optional CSV path
    --start   first byte to scan, 0 or more (default: the archive's data start, u32 at offset 4)
    --limit   stop after this many bytes, at least 1 (default: to the end)
    --pattern glob for directory input (default: *.bin, the carve file names)
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

CHUNK = 1 << 20
HEAD = 0x80
MARKER = b"0DRP"
MARKER_OFFSET = 16
VERSION_OFFSET = MARKER_OFFSET + 4
PAGES_OFFSET = MARKER_OFFSET + 0x10
KEY_OFFSET = 0x60
FIELD68_OFFSET = 0x68
FIELDS = ["source", "offset", "stored", "size", "signature", "version", "pages", "key64", "u32_68"]

# zlib stream start. Bink is not expected in a navmesh archive but the carver recognizes it; here it is
# deliberately not matched, so a navmesh index only reports zlib resource pages.
ZHDR = re.compile(rb"\x78[\x01\x5e\x9c\xda]")


def below_normal():
    if os.name == "nt":
        import ctypes
        k = ctypes.windll.kernel32
        k.SetPriorityClass(k.GetCurrentProcess(), 0x00004000)


def u32(b, off):
    return struct.unpack_from("<I", b, off)[0] if len(b) >= off + 4 else None


def page_fields(head):
    """Return the measured page-directory fields, or None when the marker is absent at offset 16."""
    if len(head) < MARKER_OFFSET + 4 or head[MARKER_OFFSET:MARKER_OFFSET + 4] != MARKER:
        return None
    key = head[KEY_OFFSET:KEY_OFFSET + 8] if len(head) >= KEY_OFFSET + 8 else b""
    return {
        "signature": "0DRP@16",
        "version": u32(head, VERSION_OFFSET),
        "pages": u32(head, PAGES_OFFSET),
        "key64": key.hex() if key and key != bytes(8) else "",
        # Measured range includes 0xFFFFFFF9, so this is not a dense file index; expose the raw word.
        "u32_68": u32(head, FIELD68_OFFSET),
    }


def scan_archive(path, start=None, limit=None):
    """Walk zlib streams like the fixed carver. Returns (rows, summary). Rows are marker matches only."""
    with open(path, "rb") as fh:
        m = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
        size = len(m)
        if start is None:
            start = u32(m, 4) or 0
        if not 0 <= start < size:
            raise SystemExit(f"start {start} is outside {path} ({size} bytes)")
        end = min(size, start + limit) if limit else size
        rows, total, last, unread, zero = [], 0, start, 0, 0
        total_stored = total_inflated = 0
        pos = start
        t0 = time.time()
        while pos < end:
            mt = ZHDR.search(m, pos, end)
            if not mt:
                break
            s = mt.start()
            d = zlib.decompressobj()
            head, total_out, p = bytearray(), 0, s
            try:
                while not d.eof and p < size:
                    chunk = m[p:p + CHUNK]
                    if not chunk:
                        break
                    out = d.decompress(chunk)
                    if len(head) < HEAD:
                        head.extend(out[:HEAD - len(head)])
                    total_out += len(out)
                    p += len(chunk)
                    if d.unused_data or d.eof:
                        break
                tail = d.flush()
                if len(head) < HEAD:
                    head.extend(tail[:HEAD - len(head)])
                total_out += len(tail)
            except zlib.error:
                pos = s + 1
                continue
            if not d.eof or total_out < 64:
                pos = s + 1
                continue
            used = p - s - len(d.unused_data)
            gap = m[last:s]
            if gap.count(0) == len(gap):
                zero += len(gap)
            else:
                unread += len(gap)
            fields = page_fields(head)
            total += 1
            total_stored += used
            total_inflated += total_out
            if fields:
                rows.append({"source": path, "offset": s, "stored": used, "size": total_out, **fields})
            last = pos = s + used
        tail = m[last:end]
        if tail.count(0) == len(tail):
            zero += len(tail)
        else:
            unread += len(tail)
    span = end - start
    summary = {
        "input": path, "mode": "rpf", "start": start, "span": span, "streams": total,
        "matched": len(rows), "compressed_bytes": total_stored, "inflated_bytes": total_inflated,
        "matched_stored_bytes": sum(r["stored"] for r in rows),
        "matched_inflated_bytes": sum(r["size"] for r in rows),
        "zero_padding": zero, "unread_bytes": unread,
        "readable_fraction": round(total_stored / span, 6) if span else 0.0,
        "elapsed_s": round(time.time() - t0, 1),
    }
    return rows, summary


def scan_directory(root, pattern="*.bin"):
    """Read the first 128 bytes of each matching carved file. Rows are marker matches only."""
    rows, files, total_size = [], 0, 0
    t0 = time.time()
    for path in sorted(glob.glob(os.path.join(root, "**", pattern), recursive=True)):
        if not os.path.isfile(path):
            continue
        size = os.path.getsize(path)
        with open(path, "rb") as f:
            head = f.read(HEAD)
        files += 1
        total_size += size
        fields = page_fields(head)
        if fields:
            rows.append({"source": os.path.relpath(path, root), "offset": None, "stored": None,
                         "size": size, **fields})
    summary = {
        "input": root, "mode": "directory", "pattern": pattern, "start": None, "span": total_size,
        "streams": files, "matched": len(rows), "compressed_bytes": None,
        "inflated_bytes": total_size, "matched_inflated_bytes": sum(r["size"] for r in rows),
        "zero_padding": None, "unread_bytes": None,
        "readable_fraction": None, "elapsed_s": round(time.time() - t0, 1),
    }
    return rows, summary


def add_distribution(summary, rows):
    versions, pages = {}, {}
    for r in rows:
        versions[str(r["version"])] = versions.get(str(r["version"]), 0) + 1
        pages[str(r["pages"])] = pages.get(str(r["pages"]), 0) + 1
    summary["versions"] = dict(sorted(versions.items(), key=lambda kv: -kv[1]))
    summary["page_counts"] = dict(sorted(pages.items(), key=lambda kv: -kv[1]))
    summary["keys_present"] = sum(1 for r in rows if r["key64"])
    return summary


def write_outputs(rows, summary, out_path, csv_path):
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "pages": rows}, f, indent=1)
    written = []
    if csv_path:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        written.append(csv_path)
    return written


def byte_count(text):
    """--limit: 0 or less used to scan nothing (negative) or everything (0 is falsy) without a word."""
    try:
        value = int(text, 0)
    except ValueError:
        raise argparse.ArgumentTypeError(f"invalid int value: {text!r}") from None
    if value < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return value


def offset(text):
    """--start: a negative offset would slice the archive from its end."""
    try:
        value = int(text, 0)
    except ValueError:
        raise argparse.ArgumentTypeError(f"invalid int value: {text!r}") from None
    if value < 0:
        raise argparse.ArgumentTypeError("must be 0 or more")
    return value


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", help="RPF9 archive, or a directory of carved .bin files")
    ap.add_argument("--out", help="JSON output (default: <input name>.navmesh-index.json)")
    ap.add_argument("--csv", help="optional CSV output")
    ap.add_argument("--start", type=offset, help="first byte to scan in an archive")
    ap.add_argument("--limit", type=byte_count, help="maximum bytes to scan in an archive")
    ap.add_argument("--pattern", default="*.bin", help="directory input glob (default: *.bin)")
    a = ap.parse_args()
    below_normal()

    if os.path.isdir(a.input):
        rows, summary = scan_directory(a.input, a.pattern)
    else:
        try:
            with open(a.input, "rb") as f:
                magic = f.read(8)
        except OSError as e:
            raise SystemExit(f"{a.input}: {e.strerror or e}")
        if len(magic) < 8 or magic[1:4] != b"FPR" or magic[0] not in b"789":
            raise SystemExit(f"{a.input}: not an RPF7/8/9 archive (magic {magic[:4]!r})")
        rows, summary = scan_archive(a.input, a.start, a.limit)

    add_distribution(summary, rows)
    base = os.path.basename(os.path.normpath(a.input))
    out_path = a.out or os.path.splitext(base)[0] + ".navmesh-index.json"
    written = write_outputs(rows, summary, out_path, a.csv)
    print(json.dumps(summary, indent=1))
    print(f"wrote {out_path}" + (f", {', '.join(written)}" if written else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
