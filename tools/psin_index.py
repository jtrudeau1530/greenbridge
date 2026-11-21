"""Index the PSIN metadata container used by the game's data and script archives.

PSIN is a small metadata container. It appears at the start of carved zlib streams in `data.rpf` and
`script.rpf`; it is not a script program and it is not a model or texture resource. Every measured
file uses the same header:

    +0   50 53 49 4e          "PSIN"
    +4   u32 big-endian       absolute offset of the first section (LSCH)
    +8   u32 little-endian    version (22 on every measured file)
    +12  u32 little-endian    byte count of the hash table that follows the header

The section table starts at the offset in `+4` and runs to the end of the file. Every section is:

    +0   4-byte ASCII tag
    +4   u32 big-endian       total size, including this 8-byte header

Sections are contiguous: the next section starts at `offset + size`, and the last one ends at EOF.
The measured tags are `LSCH`, `LENM`, `LSNS`, `LGID`, `LRSC` and `CHKS`. `LGID` is an empty section
(8 bytes). `LRSC` starts with a `0DRP` page directory header. `CHKS` carries the file size and a
checksum. `LSNS` is high-entropy and is not decoded by this tool.

On the package measured here the `script.rpf` carve contains 671 streams that start with `PSIN`:
670 are complete and one is the first 64 KiB of a larger record (its first-section offset is beyond
the carved block). A complete record in `data.rpf` is validated by the project's existing
`ArchiveReader` path, not by this tool. The tool deliberately indexes containers; it does not assign
meaning to the LSCH/LENM/CHKS payloads.

Usage:
    python -I psin_index.py <file-or-dir> [--json] [--limit N]

Text output prints one line per PSIN file and a summary; `--json` writes `{files, summary}`.
"""
import argparse
import json
import os
import struct
import sys

MAGIC = b"PSIN"
HEADER = 16
SECTION_HEADER = 8


def _u32_be(blob, off):
    return struct.unpack_from(">I", blob, off)[0]


def _u32_le(blob, off):
    return struct.unpack_from("<I", blob, off)[0]


def parse(blob):
    """Parse one PSIN container. Returns None when the blob is not PSIN."""
    if len(blob) < HEADER or blob[:4] != MAGIC:
        return None
    first = _u32_be(blob, 4)
    version = _u32_le(blob, 8)
    table_bytes = _u32_le(blob, 12)
    record = {
        "magic": "PSIN",
        "size": len(blob),
        "first_section": first,
        "version": version,
        "table_bytes": table_bytes,
        "sections": [],
        "rsc_version": None,
        "checksum_size": None,
        "checksum": "",
        "complete": False,
        "error": "",
    }
    if first < HEADER or first > len(blob):
        record["error"] = f"first section offset {first} is outside the file"
        return record
    offset = first
    while offset + SECTION_HEADER <= len(blob):
        tag = blob[offset:offset + 4]
        size = _u32_be(blob, offset + 4)
        if any(c < 0x20 or c > 0x7E for c in tag):
            record["error"] = f"non-text section tag at {offset}"
            break
        if size < SECTION_HEADER or offset + size > len(blob):
            record["error"] = f"section {tag.decode('latin-1')} at {offset} has size {size}"
            break
        payload_off = offset + SECTION_HEADER
        section = {
            "tag": tag.decode("latin-1"),
            "offset": offset,
            "size": size,
            "payload": size - SECTION_HEADER,
        }
        if tag == b"LRSC" and size - SECTION_HEADER >= 8 and blob[payload_off:payload_off + 4] == b"0DRP":
            section["rsc_magic"] = "0DRP"
            section["rsc_version"] = _u32_le(blob, payload_off + 4)
            if record["rsc_version"] is None:
                record["rsc_version"] = section["rsc_version"]
        elif tag == b"CHKS" and size - SECTION_HEADER >= 8:
            section["checksum_size"] = _u32_be(blob, payload_off)
            section["checksum"] = blob[payload_off + 4:payload_off + 8].hex()
            if record["checksum_size"] is None:
                record["checksum_size"] = section["checksum_size"]
                record["checksum"] = section["checksum"]
        record["sections"].append(section)
        offset += size
    if not record["error"] and offset != len(blob):
        record["error"] = f"section walk ended at {offset} of {len(blob)}"
    record["complete"] = not record["error"] and offset == len(blob)
    return record


def files_in(src):
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


def collect(src, limit=None):
    records = []
    for name, blob in files_in(src):
        record = parse(blob)
        if record is None:
            continue
        record["file"] = name
        records.append(record)
        if limit and len(records) >= limit:
            break
    return records


def summarize(records):
    tags = {}
    versions = {}
    for r in records:
        versions[r["version"]] = versions.get(r["version"], 0) + 1
        for s in r["sections"]:
            tags[s["tag"]] = tags.get(s["tag"], 0) + 1
    return {
        "files": len(records),
        "complete": sum(1 for r in records if r["complete"]),
        "incomplete": sum(1 for r in records if not r["complete"]),
        "versions": {str(k): v for k, v in sorted(versions.items())},
        "section_tags": dict(sorted(tags.items())),
        "section_bytes": sum(s["size"] for r in records for s in r["sections"]),
        "rsc_122": sum(1 for r in records if r["rsc_version"] == 122),
        "checksummed": sum(1 for r in records if r["checksum_size"] is not None),
    }


def main():
    ap = argparse.ArgumentParser(description="Index PSIN metadata containers.")
    ap.add_argument("src", help="PSIN file or directory of carved .bin streams")
    ap.add_argument("--json", action="store_true", help="write records and summary as JSON")
    ap.add_argument("--limit", type=int, help="stop after this many PSIN files")
    a = ap.parse_args()

    records = collect(a.src, a.limit)
    summary = summarize(records)
    if a.json:
        json.dump({"files": records, "summary": summary}, sys.stdout, indent=1)
        sys.stdout.write("\n")
        return 0
    for r in records:
        tags = " ".join(f"{s['tag']}:{s['size']}" for s in r["sections"]) or "-"
        state = "ok" if r["complete"] else f"INCOMPLETE ({r['error']})"
        extra = ""
        if r["rsc_version"] is not None:
            extra += f"  rsc={r['rsc_version']}"
        if r["checksum_size"] is not None:
            extra += f"  chks={r['checksum_size']}"
        print(f"{r['file']}  size={r['size']}  v={r['version']}  {state}  {tags}{extra}")
    print(f"{summary['files']} PSIN files, {summary['complete']} complete, "
          f"{summary['incomplete']} incomplete; tags {summary['section_tags']}; "
          f"rsc122 {summary['rsc_122']}, checksummed {summary['checksummed']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
