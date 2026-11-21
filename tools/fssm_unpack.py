"""Unpack the native GPU program from each carved shader container.

Layout found in GTA VI's Xbox shaders (shader model 6.6 containers):

    DXBC
      DXIL part  = DxilProgramHeader (stub) + XDAT
        XDAT     = 'XDAT' u32 count, u32 header_size, then count x {u32 tag, u32 offset, u32 size}
                   (offsets are relative to the DXIL part)
          tag 0x12  16-byte hash
          tag 0x0d  'XBCH' u32 ?, u32 stored, u32 0, u32 unpacked, u32 0, then zlib -> 'FSSM' blob
          tag 0x07  DxilProgramHeader with 'NOBC' (no LLVM bitcode is shipped)

Some containers carry **two distinct tag-0x0D payloads** (195 of 26,503 measured). Both decompress to
valid FSSM blobs with the same code offset but different code sizes; the first is 64 or 128 bytes
longer. `xdat_entries_all` preserves their order, `xdat_entries` keeps the previous last-wins mapping
for compatibility, and `unpack()` defaults to the last payload (the one the existing corpus used) with
`payload="first"` available for the XDAT-order/SCARAB choice. The CLI exposes `--payload`.
      PRIV part  = Xbox private metadata

The FSSM blob is the precompiled RDNA2 program plus its metadata. It contains the s_endpgm (0xBF810000) that ends
the code.

Usage:
    python -I fssm_unpack.py <carved_shader_dir> [--payload first|last]
    (writes <dir>/fssm/<name>.fssm and fssm_index.json)
"""
import argparse
import json
import os
import struct
import sys
import zlib
from collections import Counter


def parts(d: bytes) -> dict:
    out = {}
    for k in range(struct.unpack_from("<I", d, 28)[0]):
        off = struct.unpack_from("<I", d, 32 + 4 * k)[0]
        out[d[off:off + 4]] = d[off + 8:off + 8 + struct.unpack_from("<I", d, off + 4)[0]]
    return out


def xdat_entries_all(dxil: bytes) -> list:
    """All XDAT entries in directory order, as (tag, payload) pairs."""
    x = dxil.find(b"XDAT")
    if x < 0:
        return []
    count, hdr = struct.unpack_from("<II", dxil, x + 4)
    out = []
    for k in range(count):
        tag, off, size = struct.unpack_from("<III", dxil, x + 12 + 12 * k)
        out.append((tag, dxil[off:off + size]))
    return out


def xdat_entries(dxil: bytes) -> dict:
    """Last payload per tag, kept for callers that want the old single-value view."""
    return {tag: payload for tag, payload in xdat_entries_all(dxil)}


def unpack(dxbc: bytes, payload: str = "last"):
    p = parts(dxbc)
    dxil = p.get(b"DXIL", b"")
    all_entries = xdat_entries_all(dxil)
    ents = {tag: value for tag, value in all_entries}
    blobs = [value for tag, value in all_entries if tag == 0x0D]
    if not blobs:
        return None, ents
    if payload == "first":
        blob = blobs[0]
    elif payload == "last":
        blob = blobs[-1]
    else:
        raise ValueError(f"payload must be 'first' or 'last', not {payload!r}")
    if not blob or blob[:4] != b"XBCH":
        return None, ents
    unpacked = struct.unpack_from("<I", blob, 16)[0]
    data = zlib.decompress(blob[24:])
    if len(data) != unpacked:
        raise ValueError(f"size mismatch {len(data)} != {unpacked}")
    return data, ents


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Unpack the FSSM program from each carved shader container.")
    ap.add_argument("src", help="carved shader directory")
    ap.add_argument("--payload", choices=("first", "last"), default="last",
                    help="tag-0x0D payload when a container has more than one (default: last)")
    a = ap.parse_args(argv)
    src = a.src
    out = os.path.join(src, "fssm")
    os.makedirs(out, exist_ok=True)
    rows, problems = [], Counter()
    for name in sorted(os.listdir(src)):
        if not name.endswith(".dxbc"):
            continue
        with open(os.path.join(src, name), "rb") as f:
            d = f.read()
        try:
            fssm, ents = unpack(d, payload=a.payload)
        except Exception as e:  # noqa: BLE001
            problems[type(e).__name__] += 1
            continue
        if fssm is None:
            problems["no XBCH"] += 1
            continue
        stem = name[:-5]
        with open(os.path.join(out, stem + ".fssm"), "wb") as f:
            f.write(fssm)
        ends = [i for i in range(0, len(fssm) - 3, 4) if fssm[i:i + 4] == b"\x00\x00\x81\xbf"]
        payloads = sum(1 for tag, _ in xdat_entries_all(parts(d).get(b"DXIL", b"")) if tag == 0x0D)
        rows.append({"name": stem, "size": len(fssm), "magic": fssm[:4].decode("latin-1"),
                     "version": struct.unpack_from("<H", fssm, 4)[0], "endpgm": ends,
                     "tags": sorted(hex(t) for t in ents), "xbch_payloads": payloads})
    with open(os.path.join(src, "fssm_index.json"), "w", encoding="utf-8") as f:
        json.dump(rows, f)
    print(f"{len(rows)} FSSM blobs, problems: {dict(problems)}")
    print("magic:", dict(Counter(r["magic"] for r in rows)))
    print("s_endpgm per blob:", dict(Counter(len(r["endpgm"]) for r in rows)))
    sizes = sorted(r["size"] for r in rows)
    print(f"size min {sizes[0]}, median {sizes[len(sizes) // 2]}, max {sizes[-1]}, total {sum(sizes) / 2**20:.0f} MiB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
