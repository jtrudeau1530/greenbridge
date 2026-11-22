"""Decode the MHIU and PAMH arrays in a carved RPF9 archive.

Both payloads start with a 44-byte header. MHIU carries six float32 world bounds followed by a
right-aligned square byte grid; the grid edge is recovered from the payload length and the
zero-filled prefix before it. PAMH carries a width/height pair, six bounds and a row table followed
by a width x height grid of little-endian u16 values.

    python -I gamedata_maps.py <carve_dir> [--json FILE]

<carve_dir> is an rpf9_carve.py output directory (index.json plus <offset>.bin files). Only
aggregate header fields, dimensions and cell counts are printed or written; the grid bytes are not.

The checksum field is reported but not checked: it is neither CRC32 nor Adler32 of the payload. The
MHIU prefix length is not stored in the header, so the reader accepts the largest square grid that
ends at the payload end and has an all-zero prefix. Both payloads in the measured archive satisfy
that rule. Standard library only.
"""
import argparse
import json
import math
import os
import struct
import sys

MAGIC = (b"MHIU", b"PAMH", b"SPS1Z")
HEADER = 44


def _u16(blob: bytes, offset: int) -> int:
    return struct.unpack_from("<H", blob, offset)[0]


def _u32(blob: bytes, offset: int) -> int:
    return struct.unpack_from("<I", blob, offset)[0]


def _bounds(blob: bytes, offset: int) -> dict:
    lo = list(struct.unpack_from("<3f", blob, offset))
    hi = list(struct.unpack_from("<3f", blob, offset + 12))
    return {"min": lo, "max": hi}


def _square_grid(data: bytes) -> tuple[int, int]:
    """Largest square byte grid ending at the payload end with a zero-filled prefix."""
    edge = math.isqrt(len(data))
    while edge >= 2:
        prefix = len(data) - edge * edge
        if not any(data[:prefix]):
            return edge, prefix
        edge -= 1
    raise ValueError("no square grid with a zero-filled prefix")


def parse_mhiu(blob: bytes) -> dict:
    if len(blob) < HEADER or blob[:4] != b"MHIU":
        raise ValueError("not an MHIU payload")
    size = _u32(blob, 0x24)
    if size != len(blob) - HEADER:
        raise ValueError(f"MHIU data size {size} does not match {len(blob) - HEADER} bytes")
    data = blob[HEADER:]
    edge, prefix = _square_grid(data)
    return {
        "type": "MHIU",
        "header": {
            "version": _u16(blob, 0x04),
            "field06": _u16(blob, 0x06),
            "field08": _u32(blob, 0x08),
            "bounds": _bounds(blob, 0x0C),
            "data_size": size,
            "checksum": _u32(blob, 0x28),
        },
        "grid": {
            "dtype": "u8",
            "width": edge,
            "height": edge,
            "cells": edge * edge,
            "prefix_bytes": prefix,
        },
        "_grid": data[prefix:],
    }


def parse_pamh(blob: bytes) -> dict:
    if len(blob) < HEADER or blob[:4] != b"PAMH":
        raise ValueError("not a PAMH payload")
    size = _u32(blob, 0x28)
    if size != len(blob) - HEADER:
        raise ValueError(f"PAMH data size {size} does not match {len(blob) - HEADER} bytes")
    width = _u16(blob, 0x0C)
    height = _u16(blob, 0x0E)
    if not width or not height:
        raise ValueError("PAMH has a zero grid dimension")
    row_bytes = height * 8
    expect = row_bytes + width * height * 2
    if size != expect:
        raise ValueError(f"PAMH data size {size} does not match {width}x{height} plus row table")
    data = blob[HEADER:]
    for i in range(height):
        reserved, row_width, row_offset = struct.unpack_from("<HHI", data, i * 8)
        if reserved != 0 or row_width != width or row_offset != i * width:
            raise ValueError(f"PAMH row {i} does not match the width/offset table")
    return {
        "type": "PAMH",
        "header": {
            "version": _u16(blob, 0x04),
            "field06": _u16(blob, 0x06),
            "checksum": _u32(blob, 0x08),
            "width": width,
            "height": height,
            "bounds": _bounds(blob, 0x10),
            "data_size": size,
        },
        "grid": {
            "dtype": "u16",
            "width": width,
            "height": height,
            "cells": width * height,
            "row_records": height,
            "prefix_bytes": row_bytes,
        },
        "_grid": data[row_bytes:],
    }


def parse_payload(blob: bytes) -> dict:
    if blob[:4] == b"MHIU":
        return parse_mhiu(blob)
    if blob[:4] == b"PAMH":
        return parse_pamh(blob)
    raise ValueError(f"unsupported payload magic {blob[:4]!r}")


def _read_row(directory: str, row: dict) -> bytes:
    path = os.path.join(directory, f'{row["off"]:012x}.bin')
    try:
        with open(path, "rb") as f:
            return f.read()
    except OSError:
        return bytes.fromhex(row.get("head", ""))


def _join(row: dict, data: bytes) -> dict:
    out = {
        "offset": row["off"],
        "offset_hex": f'{row["off"]:x}',
        "stored": row.get("stored", len(data)),
        "size": len(data),
        "chunks": 1,
    }
    out.update(parse_payload(data))
    return out


def read_payloads(directory: str) -> list[dict]:
    with open(os.path.join(directory, "index.json"), encoding="utf-8") as f:
        index = json.load(f)
    rows = sorted(index["files"], key=lambda r: r["off"])
    out = []
    i = 0
    while i < len(rows):
        data = _read_row(directory, rows[i])
        if data[:4] not in MAGIC:
            i += 1
            continue
        if len(data) < HEADER:
            raise ValueError(f"truncated {data[:4].decode('ascii')} header at {rows[i]['off']:#x}")
        target_size = HEADER + _u32(data, 0x24 if data[:4] == b"MHIU" else 0x28)
        size = len(data)
        parts = [data]
        j = i
        while size < target_size and rows[j].get("size") == 65536 and j + 1 < len(rows):
            nxt = _read_row(directory, rows[j + 1])
            if nxt[:4] in MAGIC or nxt.startswith((b"<?xml", b"\xef\xbb\xbf<")):
                break
            parts.append(nxt)
            size += len(nxt)
            j += 1
        payload = _join(rows[i], b"".join(parts))
        payload["stored"] = sum(rows[x].get("stored", 0) for x in range(i, j + 1))
        payload["chunks"] = j - i + 1
        out.append(payload)
        i = j + 1
    return out


def _public(payload: dict) -> dict:
    return {k: v for k, v in payload.items() if not k.startswith("_")}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("carve_dir", help="rpf9_carve.py output directory")
    ap.add_argument("--json", help="write the decoded header and grid summary to this JSON")
    a = ap.parse_args()

    try:
        payloads = read_payloads(a.carve_dir)
    except (OSError, ValueError, KeyError, struct.error) as e:
        print("error:", e, file=sys.stderr)
        return 1

    for p in payloads:
        g = p["grid"]
        if p["type"] == "MHIU":
            print(f'MHIU {p["offset_hex"]}: {p["size"]:,} B, {g["width"]}x{g["height"]} u8, '
                  f'{g["prefix_bytes"]:,} B zero prefix')
        else:
            print(f'PAMH {p["offset_hex"]}: {p["size"]:,} B, {g["width"]}x{g["height"]} u16, '
                  f'{g["row_records"]} row records')
    if not payloads:
        print("no MHIU or PAMH payloads")
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump({"payloads": [_public(p) for p in payloads]}, f, indent=1)
        print("wrote", a.json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
