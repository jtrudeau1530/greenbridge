"""Unpack the native GPU program from each carved shader container.

Layout found in GTA VI's Xbox shaders (shader model 6.6 containers):

    DXBC
      DXIL part  = DxilProgramHeader (stub) + XDAT
        XDAT     = 'XDAT' u32 count, u32 header_size, then count x {u32 tag, u32 offset, u32 size}
                   (offsets are relative to the DXIL part)
          tag 0x12  16-byte hash
          tag 0x0d  'XBCH' u32 ?, u32 stored, u32 0, u32 unpacked, u32 0, then zlib -> 'FSSM' blob
          tag 0x07  DxilProgramHeader with 'NOBC' (no LLVM bitcode is shipped)
      PRIV part  = Xbox private metadata

The FSSM blob is the precompiled RDNA2 program plus its metadata. It contains the s_endpgm (0xBF810000) that ends
the code.

Usage:
    python -I fssm_unpack.py <carved_shader_dir>     (writes <dir>/fssm/<name>.fssm and fssm_index.json)
"""
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


def xdat_entries(dxil: bytes) -> dict:
    x = dxil.find(b"XDAT")
    if x < 0:
        return {}
    count, hdr = struct.unpack_from("<II", dxil, x + 4)
    ents = {}
    for k in range(count):
        tag, off, size = struct.unpack_from("<III", dxil, x + 12 + 12 * k)
        ents[tag] = dxil[off:off + size]
    return ents


def unpack(dxbc: bytes):
    p = parts(dxbc)
    ents = xdat_entries(p.get(b"DXIL", b""))
    blob = ents.get(0x0D)
    if not blob or blob[:4] != b"XBCH":
        return None, ents
    unpacked = struct.unpack_from("<I", blob, 16)[0]
    data = zlib.decompress(blob[24:])
    if len(data) != unpacked:
        raise ValueError(f"size mismatch {len(data)} != {unpacked}")
    return data, ents


def main() -> int:
    src = sys.argv[1]
    out = os.path.join(src, "fssm")
    os.makedirs(out, exist_ok=True)
    rows, problems = [], Counter()
    for name in sorted(os.listdir(src)):
        if not name.endswith(".dxbc"):
            continue
        with open(os.path.join(src, name), "rb") as f:
            d = f.read()
        try:
            fssm, ents = unpack(d)
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
        rows.append({"name": stem, "size": len(fssm), "magic": fssm[:4].decode("latin-1"),
                     "version": struct.unpack_from("<H", fssm, 4)[0], "endpgm": ends,
                     "tags": sorted(hex(t) for t in ents)})
    print(f"{len(rows)} FSSM blobs, problems: {dict(problems)}")
    if not rows:
        # No index: make_game.py takes fssm_index.json as proof this step finished, and an empty one is not.
        print(f"ERROR: no shader in {src} unpacked. Check that it holds the carved .dxbc files of a supported "
              "game version.", file=sys.stderr)
        return 1
    tmp = os.path.join(src, "fssm_index.json.tmp")   # written last and renamed in, so a stopped run leaves none
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(rows, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, os.path.join(src, "fssm_index.json"))
    print("magic:", dict(Counter(r["magic"] for r in rows)))
    print("s_endpgm per blob:", dict(Counter(len(r["endpgm"]) for r in rows)))
    sizes = sorted(r["size"] for r in rows)
    print(f"size min {sizes[0]}, median {sizes[len(sizes) // 2]}, max {sizes[-1]}, total {sum(sizes) / 2**20:.0f} MiB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
