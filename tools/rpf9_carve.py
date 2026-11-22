"""Carve the zlib-compressed files out of an RPF9 archive without its (encrypted) table of contents.

The data area is mostly zlib streams laid end to end, separated by zero padding. This walks the archive, inflates
every stream that decodes cleanly to the end, and writes each one as <offset>.bin. It also records which byte
ranges it couldn't read, so coverage is measurable.

Usage:
    python -I rpf9_carve.py <archive.rpf> <out_dir> [--start N] [--limit BYTES] [--index-only]
"""
import argparse
import json
import mmap
import os
import re
import struct
import sys
import time
import zlib

# zlib stream start, or a Bink 2 video ('KB2' + revision letter, then u32 size-8).
# RPF9 magic as it sits on disk: the tag "RPF9" stored little-endian (docs/rpf.md, docs/bvh.md).
MAGIC = b"9FPR"
ZHDR = re.compile(rb"\x78[\x01\x5e\x9c\xda]|KB2[a-z]")


def below_normal():
    if os.name == "nt":
        import ctypes
        k = ctypes.windll.kernel32
        k.SetPriorityClass(k.GetCurrentProcess(), 0x00004000)


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
    """--limit: at least 1; 0 used to mean no limit and a negative value silently scanned less."""
    try:
        value = int(text, 0)
    except ValueError:
        raise argparse.ArgumentTypeError("must be at least 1") from None
    if value < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return value


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("archive")
    ap.add_argument("out")
    ap.add_argument("--start", type=offset)
    ap.add_argument("--limit", type=byte_count)
    ap.add_argument("--index-only", action="store_true")
    a = ap.parse_args()
    below_normal()
    try:
        size = os.path.getsize(a.archive)
    except OSError as e:
        sys.exit(f"{a.archive}: {e.strerror or e}")
    if size == 0:
        sys.exit(f"{a.archive}: empty file")
    if size < 8:
        sys.exit(f"{a.archive}: too short for an RPF9 header ({size} bytes)")
    with open(a.archive, "rb") as fh:
        if fh.read(4) != MAGIC:
            sys.exit(f"{a.archive}: not an RPF9 archive (expected the magic {MAGIC.decode()} at offset 0)")
    if a.start is not None and not 0 <= a.start < size:
        sys.exit(f"{a.archive}: --start {a.start} is outside the file (0..{size - 1})")
    if a.start is None:
        with open(a.archive, "rb") as fh:
            base = struct.unpack_from("<I", fh.read(8), 4)[0]
        if base > size:   # base == size is an empty data area, which is fine
            sys.exit(f"{a.archive}: header data offset {base} is past the end of the file ({size} bytes)")
    else:
        base = a.start
    os.makedirs(a.out, exist_ok=True)
    with open(a.archive, "rb") as fh:
        m = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
        pos = base
        end = min(size, pos + a.limit) if a.limit else size
        rows, last, unread, zero = [], pos, 0, 0
        t0 = time.time()
        while pos < end:
            mt = ZHDR.search(m, pos, end)
            if not mt:
                break
            s = mt.start()
            if m[s:s + 3] == b"KB2":
                if s + 0x1C > end:   # header cut off by the end of the file or of --limit's window: not a video
                    pos = s + 1
                    continue
                blen = struct.unpack_from("<I", m, s + 4)[0] + 8
                w, h = struct.unpack_from("<II", m, s + 0x14)
                if not (64 <= blen <= end - s and 16 <= w <= 8192 and 16 <= h <= 8192):
                    pos = s + 1
                    continue
                gap = m[last:s]
                if gap.count(0) == len(gap):
                    zero += len(gap)
                else:
                    unread += len(gap)
                rows.append({"off": s, "stored": blen, "size": blen, "type": "bink2", "w": w, "h": h,
                             "frames": struct.unpack_from("<I", m, s + 8)[0], "pre": "", "head": m[s:s + 16].hex()})
                if not a.index_only:
                    with open(os.path.join(a.out, f"{s:012x}.bk2"), "wb") as f:
                        f.write(m[s:s + blen])
                last = pos = s + blen
                continue
            d = zlib.decompressobj()
            try:
                chunks, p = [], s
                # Inflate only inside the scan window (all of the file without --limit): a stream that runs past
                # its end is left out, so readable + zero + unread add up to the span, as in story_index (#712).
                while not d.eof and p < end:
                    chunk = m[p:min(p + (1 << 20), end)]
                    if not chunk:
                        break
                    chunks.append(d.decompress(chunk))
                    p += len(chunk)   # the slice can be short at EOF; advancing by 1 MiB overran it
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
            gap = m[last:s]
            if gap.count(0) == len(gap):
                zero += len(gap)
            else:
                unread += len(gap)
            rows.append({"off": s, "stored": used, "size": len(out), "pre": m[max(0, s - 16):s].hex(),
                         "head": out[:16].hex()})
            if not a.index_only:
                with open(os.path.join(a.out, f"{s:012x}.bin"), "wb") as f:
                    f.write(out)
            last = pos = s + used
            if len(rows) % 2000 == 0:
                done = (pos - (a.start or 0)) / max(1, end - (a.start or 0))
                print(f"[{done:6.1%}] {len(rows)} files, {time.time() - t0:.0f}s", flush=True)
        tail = m[last:end]
        if tail.count(0) == len(tail):
            zero += len(tail)
        else:
            unread += len(tail)
    total = sum(r["stored"] for r in rows)
    bink = sum(r["stored"] for r in rows if r.get("type") == "bink2")
    span = end - base
    summary = {"archive": a.archive, "files": len(rows), "compressed_bytes": total,
               "inflated_bytes": sum(r["size"] for r in rows), "bink2_bytes": bink,
               "bink2_files": sum(1 for r in rows if r.get("type") == "bink2"),
               "zero_padding": zero, "unread_bytes": unread,
               "span": span}
    with open(os.path.join(a.out, "index.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump({"summary": summary, "files": rows}, f)
    print(json.dumps(summary, indent=1))
    span = max(1, span)   # an empty data area (or --start at the end) has nothing to divide by
    print(f"readable: {total / span:.1%}, zero padding {zero / span:.1%}, unread {unread / span:.1%}")
    return 0


if __name__ == "__main__":
    try:  # a redirected stdout is strict ANSI code page under -I; write '?' for what it cannot encode
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    sys.exit(main())
