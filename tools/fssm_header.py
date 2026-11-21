"""Read the pixel-shader input table from an FSSM blob.

FSSM is the precompiled RDNA2 program that the Xbox shader container carries.  For pixel shaders a
table of 8-byte input records starts at offset 0x138.  A record is present while its first byte is 6:

    byte 0      record kind (6 for an input)
    byte 1      semantic id
    byte 2      component mask (bit 0 = x, bit 3 = w)
    byte 3      running index (same semantic id, next index)
    bytes 4-5   unused here
    byte 6      parameter slot
    byte 7      flags; bit 4 (0x10) marks a 16-bit interpolant

The record's position in the table is the `v_interp attr<N>` number the code uses, not byte 6 - 1.

`directory_ok` checks the section-directory identity that most blobs follow:

    u32 at 0x120 (offset) + u32 at 0x12C (size) == u32 at 0x130 (next offset)

All facts above are from the format description supplied with this tool.  The optional stage lookup
reads the carve index (`<fssm dir>/../index.json`) only to report how many *pixel* shaders have a
table; the decoder itself needs no index.

On the local 26,503-blob corpus the directory identity held for 25,643 blobs (the 860 failures all
carry 0xDEADBEEF at 0x130), and 12,101 blobs were pixel shaders, 10,307 of them with a non-empty
table (55,791 records, 11,199 of them 16-bit).  The table count equals the number of distinct
`v_interp attr` numbers in 11,316/12,101 shaders (93.5%); 300 more declare exactly one unread input,
475 declare two or more, and the 10 shaders whose table starts with a non-6 record are the only ones
that undercount.  A 200-PS spot check agreed on 187/200.  Flag bit 4 matched the 16-bit interpolant
opcodes in 12,067/12,101 (99.7%).  `rdna2spirv.py` consumes it in the PS input path (pixel_interpolants:
every table position declared as a fragment input, 16-bit-flagged positions
rounded f32->f16->f32 on 32-bit reads).

Usage:
    python -I fssm_header.py <fssm dir or file> [--json] [--limit N]

Prints one line per blob and a corpus tally:
    N files, directory_ok X, PS with table Y, entries Z
"""
import argparse
import json
import os
import struct
import sys
from collections import Counter

TABLE_OFFSET = 0x138
RECORD_SIZE = 8
RECORD_KIND = 6
FLAG_FP16 = 0x10
DIRECTORY_OFFSET = 0x120
DIRECTORY_SIZE = 0x12C
DIRECTORY_NEXT = 0x130


def _u32(blob: bytes, offset: int):
    return struct.unpack_from("<I", blob, offset)[0]


def pixel_inputs(blob):
    """Return the pixel-input records as dicts, in table order.

    Keys: semantic, mask, index, slot, flags.  An empty list means the blob has no table (the byte
    at 0x138 is not 6) or the blob is too short to hold one.
    """
    out = []
    offset = TABLE_OFFSET
    while offset + RECORD_SIZE <= len(blob) and blob[offset] == RECORD_KIND:
        record = blob[offset:offset + RECORD_SIZE]
        out.append({
            "semantic": record[1],
            "mask": record[2],
            "index": record[3],
            "slot": record[6],
            "flags": record[7],
        })
        offset += RECORD_SIZE
    return out


def interpolants(blob):
    """Return (count, fp16_mask) where bit i of fp16_mask is set when table position i is 16-bit."""
    inputs = pixel_inputs(blob)
    mask = 0
    for position, entry in enumerate(inputs):
        if entry["flags"] & FLAG_FP16:
            mask |= 1 << position
    return len(inputs), mask


def directory_ok(blob):
    """True when offset + size == next offset in the section directory (0x120/0x12C/0x130)."""
    if len(blob) < DIRECTORY_NEXT + 4:
        return False
    return _u32(blob, DIRECTORY_OFFSET) + _u32(blob, DIRECTORY_SIZE) == _u32(blob, DIRECTORY_NEXT)


def stage_index(path):
    """Map shader stems to stages from the carve index, or {} when it is not next to the corpus.

    The index is the one `make_game.py` writes next to the carved `.dxbc` files.  Both the `fssm`
    directory (parent holds index.json) and a directory that contains index.json are accepted.
    """
    base = path if os.path.isdir(path) else os.path.dirname(path)
    for candidate in (os.path.join(base, "..", "index.json"), os.path.join(base, "index.json")):
        if os.path.isfile(candidate):
            try:
                with open(candidate, encoding="utf-8") as f:
                    rows = json.load(f)
                return {os.path.splitext(row["file"])[0]: row.get("stage") for row in rows}
            except (OSError, ValueError, KeyError, TypeError):
                return {}
    return {}


def files_of(path):
    if os.path.isfile(path):
        return [path]
    return [os.path.join(path, name) for name in sorted(os.listdir(path)) if name.endswith(".fssm")]


def summarize(path, stages):
    """Decode one blob and return the row used by the text and JSON output."""
    with open(path, "rb") as f:
        blob = f.read()
    stem = os.path.splitext(os.path.basename(path))[0]
    count, fp16 = interpolants(blob)
    return {
        "name": os.path.basename(path),
        "stage": stages.get(stem),
        "directory_ok": directory_ok(blob),
        "entries": count,
        "fp16_mask": fp16,
        "inputs": pixel_inputs(blob),
    }


def tally(rows, stages_known):
    if stages_known:
        ps = [row for row in rows if row["stage"] == "ps"]
        counts = Counter(row["entries"] for row in ps)
        scope = "ps"
    else:
        # Without a stage index, a non-empty table is the only PS signal the blob carries.
        ps = [row for row in rows if row["entries"]]
        counts = Counter(row["entries"] for row in rows)
        scope = "all"
    return {
        "files": len(rows),
        "directory_ok": sum(row["directory_ok"] for row in rows),
        "ps_with_table": sum(1 for row in ps if row["entries"]),
        "entries": sum(row["entries"] for row in ps),
        "entry_counts": dict(sorted(counts.items())),
        "entry_counts_scope": scope,
        "stage_source": "index.json" if stages_known else "table-only",
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description="Read the FSSM pixel-shader input table.")
    ap.add_argument("path", help="an .fssm file or a directory of them")
    ap.add_argument("--json", action="store_true", help="print one JSON document instead of lines")
    ap.add_argument("--limit", type=int, help="stop after this many blobs")
    args = ap.parse_args(argv)

    paths = files_of(args.path)
    if args.limit is not None:
        paths = paths[:args.limit]
    stages = stage_index(args.path)
    rows = [summarize(path, stages) for path in paths]
    summary = tally(rows, bool(stages))

    if args.json:
        json.dump({"files": rows, "tally": summary}, sys.stdout, indent=1)
        sys.stdout.write("\n")
        return 0

    for row in rows:
        stage = row["stage"] or "?"
        print(f"{row['name']}: stage={stage} directory_ok={row['directory_ok']} "
              f"entries={row['entries']} fp16=0x{row['fp16_mask']:x}")
    total = summary
    print(f"{total['files']} files, directory_ok {total['directory_ok']}, "
          f"PS with table {total['ps_with_table']}, entries {total['entries']}")
    label = "PS entry counts" if total["entry_counts_scope"] == "ps" else "entry counts"
    print(label + ": " + ", ".join(f"{k}:{v}" for k, v in total["entry_counts"].items()))
    if total["stage_source"] == "table-only":
        print("(no index.json found: counted every blob with a non-empty table as a PS)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
