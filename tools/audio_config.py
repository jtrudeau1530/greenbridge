"""Read the music/radio record table from the game's inflated audio-config blob.

The audio config is the zlib-compressed section at the start of `audio.rpf`, before the first AWC bank.
It is not part of the table of contents and it is not a script resource. The music records sit in a
few streams inside it and share a fixed 46-byte layout, keyed by a marker that is stable across every
record:

    record - 4   u32   duration in milliseconds
    record + 0   0c 01              marker
    record + 2   u16   config id
    record + 4   02 00 0c 00
    record + 8   u32   hash A        (unresolved)
    record + 12  u32   hash B        joaat(normalized name)
    record + 16  00 00 02 23
    record + 20  22 bytes            fixed flags/parameters

`hash B` is Jenkins one-at-a-time over the lowercase name with spaces and punctuation mapped to `_`.
The AWC stream id is `hash B & 0x1FFFFFFF`; the duration is the real song/ad length, while the AWC
manifest reports twice that value because of a stereo-sample counting artifact. That mapping was
checked against a package's AWC manifest: all 1,589 records resolve to a stream id and 1,584 of 1,589
durations equal `manifest duration / 2` within 0.01 s.

The config stores no per-track start offsets. It gives the station grouping as contiguous config-id
ranges and station-local content-id lists, but the mix order is not stored; a continuous radio mix
must be matched in the audio. This tool therefore outputs only the record table.

Usage:
    python -I audio_config.py <carve dir or blob> [--json] [--names FILE]
        [--min-ms N] [--max-ms N]

`--names` is an optional NUL-separated string pool (the package's debug/name stream); each string is
hashed with the same function and used to fill `name` for the matching hash B. The pool is a local
game file and is never copied or committed.

The input is the inflated output of `rpf9_carve.py`, either the per-stream `.bin` files or a single
concatenated config blob. Records are found by their marker and duration range, not by a hard-coded
stream offset, so an incomplete carve simply yields fewer records.
"""
import argparse
import json
import os
import struct
import sys

MARKER_1 = b"\x0c\x01"
MARKER_2 = b"\x02\x00\x0c\x00"
MARKER_3 = b"\x00\x00\x02\x23"
RECORD_BYTES = 46
MIN_MS = 20_000
MAX_MS = 2_000_000
STREAM_MASK = 0x1FFFFFFF


def joaat(text):
    """Jenkins one-at-a-time over the lowercased UTF-8 bytes, as used for the name hash."""
    h = 0
    for c in text.lower().encode("utf-8", "ignore"):
        h = (h + c) & 0xFFFFFFFF
        h = (h + (h << 10)) & 0xFFFFFFFF
        h ^= h >> 6
    h = (h + (h << 3)) & 0xFFFFFFFF
    h ^= h >> 11
    h = (h + (h << 15)) & 0xFFFFFFFF
    return h & 0xFFFFFFFF


def normalized_name(text):
    """Lowercase, and map every non-alphanumeric run to a single underscore."""
    out = []
    last = False
    for ch in text.lower():
        ok = ch.isalnum()
        if ok:
            out.append(ch)
        elif not last:
            out.append("_")
        last = not ok
    return "".join(out).strip("_")


def scan(blob, min_ms=MIN_MS, max_ms=MAX_MS):
    """Return the record table found in one inflated config blob."""
    out = []
    end = len(blob) - 20
    pos = 4
    while pos < end:
        if blob[pos:pos + 2] != MARKER_1:
            pos += 1
            continue
        if blob[pos + 4:pos + 8] != MARKER_2 or blob[pos + 16:pos + 20] != MARKER_3:
            pos += 1
            continue
        duration = struct.unpack_from("<I", blob, pos - 4)[0]
        if not (min_ms <= duration <= max_ms):
            pos += 1
            continue
        record = {
            "offset": pos,
            "id": struct.unpack_from("<H", blob, pos + 2)[0],
            "duration_ms": duration,
            "duration_s": round(duration / 1000.0, 3),
            "hash_a": struct.unpack_from("<I", blob, pos + 8)[0],
            "hash_b": struct.unpack_from("<I", blob, pos + 12)[0],
        }
        record["stream_id"] = record["hash_b"] & STREAM_MASK
        out.append(record)
        pos += RECORD_BYTES
    return out


def files_in(src):
    """Yield (name, bytes) for a blob or every `.bin` file in a directory, in name order."""
    if os.path.isfile(src):
        with open(src, "rb") as f:
            yield os.path.basename(src), f.read()
        return
    for name in sorted(os.listdir(src)):
        if not name.endswith(".bin"):
            continue
        path = os.path.join(src, name)
        if os.path.isfile(path):
            with open(path, "rb") as f:
                yield name, f.read()


def load_names(path):
    """Map joaat -> display string for a NUL-separated pool."""
    with open(path, "rb") as f:
        pool = f.read()
    names = {}
    for item in pool.split(b"\x00"):
        if not item:
            continue
        text = item.decode("utf-8", "replace")
        key = joaat(normalized_name(text))
        if key not in names:
            names[key] = text
    return names


def _file_at(starts, offset):
    """Return the stream name that contains a byte offset in a joined blob."""
    import bisect
    index = bisect.bisect_right(starts, (offset, "\uffff")) - 1
    return starts[max(index, 0)][1]


def collect(src, min_ms=MIN_MS, max_ms=MAX_MS, names=None, join=False):
    records = []
    parts = list(files_in(src))
    if join and len(parts) > 1:
        blob = b"".join(part for _, part in parts)
        starts = []
        offset = 0
        for name, part in parts:
            starts.append((offset, name))
            offset += len(part)
        for record in scan(blob, min_ms, max_ms):
            record["file"] = _file_at(starts, record["offset"])
            if names:
                record["name"] = names.get(record["hash_b"], "")
            records.append(record)
        return records
    for name, blob in parts:
        for record in scan(blob, min_ms, max_ms):
            record["file"] = name
            if names:
                record["name"] = names.get(record["hash_b"], "")
            records.append(record)
    return records


def summarize(records):
    ids = [r["id"] for r in records]
    counts = {}
    for i in ids:
        counts[i] = counts.get(i, 0) + 1
    return {
        "records": len(records),
        "distinct_ids": len(counts),
        "id_min": min(ids) if ids else None,
        "id_max": max(ids) if ids else None,
        "duration_s": round(sum(r["duration_ms"] for r in records) / 1000.0, 3),
        "named": sum(1 for r in records if r.get("name")),
        "duplicate_ids": sorted(i for i, n in counts.items() if n > 1),
    }


def main():
    ap = argparse.ArgumentParser(description="Read the audio-config music/radio record table.")
    ap.add_argument("src", help="inflated config blob or directory of carved .bin streams")
    ap.add_argument("--json", action="store_true", help="write records and summary as JSON")
    ap.add_argument("--names", help="optional NUL-separated name pool for hash B")
    ap.add_argument("--join", action="store_true",
                    help="concatenate the directory's .bin streams in name order before scanning "
                         "(records can straddle a zlib block boundary)")
    ap.add_argument("--min-ms", type=int, default=MIN_MS, help="minimum duration (default 20000)")
    ap.add_argument("--max-ms", type=int, default=MAX_MS, help="maximum duration (default 2000000)")
    a = ap.parse_args()

    names = load_names(a.names) if a.names else None
    records = collect(a.src, a.min_ms, a.max_ms, names, a.join)
    summary = summarize(records)

    if a.json:
        json.dump({"records": records, "summary": summary}, sys.stdout, indent=1)
        sys.stdout.write("\n")
        return 0

    for r in records:
        label = r.get("name") or f"0x{r['hash_b']:08X}"
        print(f"{r['file']}:{r['offset']:6d}  id=0x{r['id']:04X}  "
              f"dur={r['duration_s']:9.3f}s  stream=0x{r['stream_id']:08X}  {label}")
    print(f"{summary['records']} records, {summary['distinct_ids']} distinct ids, "
          f"ids 0x{summary['id_min']:04X}-0x{summary['id_max']:04X}, "
          f"{summary['duration_s']:.1f} s total"
          + (f", {summary['named']} named" if names else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
