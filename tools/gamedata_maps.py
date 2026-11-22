"""Decode the MHIU and PAMH (HMAP) heightmaps in a carved RPF9 archive.

Both payloads start with a 44-byte header and then use a row table plus packed
u8 heights. The layouts were measured on the Xbox Series 1.0.2.1 package and
match the published GTA V `heightmap.dat` (`HMAP`) layout:

`MHIU`
    0x00  "MHIU"
    0x04  u16 version
    0x06  u16 width          (2048 in both measured payloads)
    0x08  u16 height         (2048)
    0x0A  u16 pad
    0x0C  six f32 bounds     (min xyz, max xyz)
    0x24  u32 data size      (bytes after this 44-byte header)
    0x28  u32 unverified field
    0x2C  height x 8-byte row headers `{u16 start, u16 count, i32 offset}`
          then the packed u8 heights. A cell not covered by a row keeps the
          measured default 23. The source index is `offset + start + i`.
    The old "zero prefix + square grid" reading was an artifact: the prefix is
    the zeroed row headers of the map's empty (default) rows.

`PAMH` (`HMAP` read little-endian)
    0x00  "PAMH"             (the `HMAP` magic stored little-endian)
    0x04  u8 version major, u8 version minor
    0x06  u16 pad
    0x08  u32 compressed/flag word (GTA V writes 1; measured 0x1BBC8CED here)
    0x0C  u16 width
    0x0E  u16 height
    0x10  six f32 bounds     (min xyz, max xyz)
    0x28  u32 data size      (bytes after this 44-byte header)
    0x2C  height x 8-byte row headers
          then `width * height` u8 max rows and `width * height` u8 min rows.
    On the measured file the first array is a flat 110 (2.5 m) sea-surface
    layer and the second is the ground; min <= max in all 23,616 cells.

Both formats scale a u8 value into the header's Z box:

    z = min.z + value / 255 * (max.z - min.z)

    python -I gamedata_maps.py <carve_dir> [--json FILE]

<carve_dir> is an rpf9_carve.py output directory (index.json plus <offset>.bin
files). Only header fields, dimensions, row counts and value ranges are
printed or written; the height bytes are not. Standard library only.
"""
import argparse
import json
import os
import struct
import sys

MAGIC = (b"MHIU", b"PAMH", b"SPS1Z")
HEADER = 44
DEFAULT_MHIU_HEIGHT = 23


def _u16(blob: bytes, offset: int) -> int:
    return struct.unpack_from("<H", blob, offset)[0]


def _u32(blob: bytes, offset: int) -> int:
    return struct.unpack_from("<I", blob, offset)[0]


def _bounds(blob: bytes, offset: int) -> dict:
    lo = list(struct.unpack_from("<3f", blob, offset))
    hi = list(struct.unpack_from("<3f", blob, offset + 12))
    return {"min": lo, "max": hi}


def _z(value: int, bounds: dict) -> float:
    lo, hi = bounds["min"][2], bounds["max"][2]
    return lo + value / 255.0 * (hi - lo)


def _rows(blob: bytes, height: int) -> bytes:
    end = HEADER + height * 8
    if end > len(blob):
        raise ValueError("row table is truncated")
    return blob[HEADER:end]


def parse_mhiu(blob: bytes) -> dict:
    if len(blob) < HEADER or blob[:4] != b"MHIU":
        raise ValueError("not an MHIU payload")
    width = _u16(blob, 0x06)
    height = _u16(blob, 0x08)
    if not width or not height:
        raise ValueError("MHIU has a zero grid dimension")
    if width > 4096 or height > 4096:
        raise ValueError(f"MHIU grid {width}x{height} is out of range")
    bounds = _bounds(blob, 0x0C)
    size = _u32(blob, 0x24)
    if size != len(blob) - HEADER:
        raise ValueError(f"MHIU data size {size} does not match {len(blob) - HEADER} bytes")

    rows = _rows(blob, height)
    pixels = blob[HEADER + height * 8:]
    heights = bytearray([DEFAULT_MHIU_HEIGHT]) * (width * height)
    empty = filled_rows = filled_cells = 0
    spans = []
    for y in range(height):
        start, count = struct.unpack_from("<HH", rows, y * 8)
        offset = struct.unpack_from("<i", rows, y * 8 + 4)[0]
        if count == 0:
            empty += 1
            continue
        if start + count > width:
            raise ValueError(f"MHIU row {y} leaves the {width}-cell width")
        source = offset + start
        if source < 0 or source + count > len(pixels):
            raise ValueError(f"MHIU row {y} leaves the {len(pixels)}-byte pixel data")
        heights[y * width + start:y * width + start + count] = \
            pixels[source:source + count]
        filled_rows += 1
        filled_cells += count
        spans.append((source, source + count))

    spans.sort()
    for (_, end), (start, _) in zip(spans, spans[1:]):
        if start < end:
            raise ValueError("MHIU row data overlaps")

    return {
        "type": "MHIU",
        "header": {
            "version": _u16(blob, 0x04),
            "width": width,
            "height": height,
            "bounds": bounds,
            "data_size": size,
            "field28": _u32(blob, 0x28),
            "z_mapping": "min.z + value/255 * (max.z - min.z)",
        },
        "grid": {
            "dtype": "u8",
            "width": width,
            "height": height,
            "cells": width * height,
            "default": DEFAULT_MHIU_HEIGHT,
            "row_records": height,
            "filled_rows": filled_rows,
            "empty_rows": empty,
            "filled_cells": filled_cells,
            "data_bytes": len(pixels),
            "z_at_default": round(_z(DEFAULT_MHIU_HEIGHT, bounds), 2),
        },
        "_grid": bytes(heights),
    }


def parse_pamh(blob: bytes) -> dict:
    if len(blob) < HEADER or blob[:4] != b"PAMH":
        raise ValueError("not a PAMH payload")
    width = _u16(blob, 0x0C)
    height = _u16(blob, 0x0E)
    if not width or not height:
        raise ValueError("PAMH has a zero grid dimension")
    bounds = _bounds(blob, 0x10)
    size = _u32(blob, 0x28)
    if size != len(blob) - HEADER:
        raise ValueError(f"PAMH data size {size} does not match {len(blob) - HEADER} bytes")

    rows = _rows(blob, height)
    data = blob[HEADER + height * 8:]
    body = size - height * 8
    if body != width * height * 2:
        raise ValueError(
            f"PAMH data is {body} bytes; {width}x{height} needs "
            f"{width * height * 2} for max and min rows")
    half = width * height
    maximum = bytearray(half)
    minimum = bytearray(half)
    for y in range(height):
        start, count = struct.unpack_from("<HH", rows, y * 8)
        offset = struct.unpack_from("<i", rows, y * 8 + 4)[0]
        if start + count > width:
            raise ValueError(f"PAMH row {y} leaves the {width}-cell width")
        for i in range(count):
            x = start + i
            at = offset + x
            if at < 0 or at >= half:
                raise ValueError(f"PAMH row {y} leaves the {half}-cell row data")
            maximum[y * width + x] = data[at]
            minimum[y * width + x] = data[half + at]

    inversions = sum(1 for hi, lo in zip(maximum, minimum) if lo > hi)
    return {
        "type": "PAMH",
        "header": {
            "version_major": blob[0x04],
            "version_minor": blob[0x05],
            "flag": _u32(blob, 0x08),
            "width": width,
            "height": height,
            "bounds": bounds,
            "data_size": size,
            "z_mapping": "min.z + value/255 * (max.z - min.z)",
        },
        "grid": {
            "dtype": "u8 max + u8 min",
            "width": width,
            "height": height,
            "cells": width * height,
            "row_records": height,
            "max_min": min(maximum),
            "max_max": max(maximum),
            "min_min": min(minimum),
            "min_max": max(minimum),
            "min_above_max": inversions,
        },
        "_max": bytes(maximum),
        "_min": bytes(minimum),
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
            print(f'MHIU {p["offset_hex"]}: {p["size"]:,} B, '
                  f'{g["width"]}x{g["height"]} u8, {g["filled_rows"]} filled rows, '
                  f'{g["empty_rows"]} empty (default {g["default"]})')
        else:
            print(f'PAMH {p["offset_hex"]}: {p["size"]:,} B, '
                  f'{g["width"]}x{g["height"]} u8 max+min, {g["row_records"]} row records, '
                  f'{g["min_above_max"]} cells min>max')
    if not payloads:
        print("no MHIU or PAMH payloads")
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump({"payloads": [_public(p) for p in payloads]}, f, indent=1)
        print("wrote", a.json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
