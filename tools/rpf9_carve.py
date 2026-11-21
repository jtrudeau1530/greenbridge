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
ZHDR = re.compile(rb"\x78[\x01\x5e\x9c\xda]|KB2[a-z]")


def below_normal():
    if os.name == "nt":
        import ctypes
        k = ctypes.windll.kernel32
        k.SetPriorityClass(k.GetCurrentProcess(), 0x00004000)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("archive")
    ap.add_argument("out")
    ap.add_argument("--start", type=int)
    ap.add_argument("--limit", type=int)
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
    if a.start is not None and not 0 <= a.start < size:
        sys.exit(f"{a.archive}: --start {a.start} is outside the file (0..{size - 1})")
    os.makedirs(a.out, exist_ok=True)
    with open(a.archive, "rb") as fh:
        m = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
        pos = a.start if a.start is not None else struct.unpack_from("<I", m, 4)[0]
        end = min(size, pos + a.limit) if a.limit else size
        rows, last, unread, zero = [], pos, 0, 0
        t0 = time.time()
        while pos < end:
            mt = ZHDR.search(m, pos, end)
            if not mt:
                break
            s = mt.start()
            if m[s:s + 3] == b"KB2":
                if s + 0x1C > size:   # header cut off by the end of the file (a truncated copy): not a video
                    pos = s + 1
                    continue
                blen = struct.unpack_from("<I", m, s + 4)[0] + 8
                w, h = struct.unpack_from("<II", m, s + 0x14)
                if not (64 <= blen <= size - s and 16 <= w <= 8192 and 16 <= h <= 8192):
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
                while not d.eof and p < size:
                    chunk = m[p:p + (1 << 20)]
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
    span = end - (a.start if a.start is not None else struct.unpack_from("<I", open(a.archive, "rb").read(8), 4)[0])
    summary = {"archive": a.archive, "files": len(rows), "compressed_bytes": total,
               "inflated_bytes": sum(r["size"] for r in rows), "bink2_bytes": bink,
               "bink2_files": sum(1 for r in rows if r.get("type") == "bink2"),
               "zero_padding": zero, "unread_bytes": unread,
               "span": span}
    with open(os.path.join(a.out, "index.json"), "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "files": rows}, f)
    print(json.dumps(summary, indent=1))
    span = max(1, span)   # an empty data area (or --start at the end) has nothing to divide by
    print(f"readable: {total / span:.1%}, zero padding {zero / span:.1%}, unread {unread / span:.1%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
