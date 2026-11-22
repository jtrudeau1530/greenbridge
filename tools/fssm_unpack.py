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

# Defensive per-blob ceiling, not a maximum defined by the XBCH format.
# 64 MiB matches the repository's existing per-resource cap in tools/world_index.py.
MAX_FSSM_BYTES = 64 << 20


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
    stored = struct.unpack_from("<I", blob, 8)[0]
    unpacked = struct.unpack_from("<I", blob, 16)[0]
    if unpacked > MAX_FSSM_BYTES:
        raise ValueError(f"XBCH declared size {unpacked} exceeds limit {MAX_FSSM_BYTES}")
    compressed = blob[24:]
    if stored != len(compressed):
        raise ValueError(f"XBCH stored size {stored} does not match payload size {len(compressed)}")
    decompressor = zlib.decompressobj()
    # The extra byte detects a stream that expands past the declared size without inflating it all.
    data = decompressor.decompress(compressed, unpacked + 1)
    if len(data) > unpacked:
        raise ValueError(f"decompressed XBCH exceeds declared size {unpacked}")
    if not decompressor.eof:
        raise ValueError("truncated XBCH zlib stream")
    if decompressor.unused_data:
        raise ValueError("trailing data after XBCH zlib stream")
    if len(data) != unpacked:
        raise ValueError(f"size mismatch {len(data)} != {unpacked}")
    if len(data) < 6:
        raise ValueError("truncated FSSM magic or version")
    if data[:4] != b"FSSM":
        raise ValueError(f"invalid FSSM magic {data[:4]!r}")
    return data, ents


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Unpack the FSSM program from each carved shader container.")
    ap.add_argument("src", help="carved shader directory")
    ap.add_argument("--payload", choices=("first", "last"), default="last",
                    help="tag-0x0D payload when a container has more than one (default: last)")
    a = ap.parse_args(argv)
    src = a.src
    # Check before anything is created: makedirs below would otherwise make a mistyped source folder.
    if not os.path.isdir(src):
        what = "is not a folder" if os.path.exists(src) else "does not exist"
        print(f"ERROR: {src} {what}. Give the folder with the carved .dxbc files.", file=sys.stderr)
        return 2
    try:
        with os.scandir(src) as entries:
            shaders = sorted((entry for entry in entries
                              if entry.name.endswith(".dxbc") and entry.is_file()),
                             key=lambda entry: entry.name)
    except OSError as e:
        print(f"ERROR: cannot list {src}: {e.strerror or e}", file=sys.stderr)
        return 2
    out = os.path.join(src, "fssm")
    os.makedirs(out, exist_ok=True)
    rows, problems = [], Counter()
    for shader in shaders:
        with open(shader.path, "rb") as f:
            d = f.read()
        try:
            fssm, ents = unpack(d, payload=a.payload)
        except Exception as e:  # noqa: BLE001
            problems[type(e).__name__] += 1
            continue
        if fssm is None:
            problems["no XBCH"] += 1
            continue
        stem = shader.name[:-5]
        with open(os.path.join(out, stem + ".fssm"), "wb") as f:
            f.write(fssm)
        ends = [i for i in range(0, len(fssm) - 3, 4) if fssm[i:i + 4] == b"\x00\x00\x81\xbf"]
        payloads = sum(1 for tag, _ in xdat_entries_all(parts(d).get(b"DXIL", b"")) if tag == 0x0D)
        rows.append({"name": stem, "size": len(fssm), "magic": fssm[:4].decode("latin-1"),
                     "version": struct.unpack_from("<H", fssm, 4)[0], "endpgm": ends,
                     "tags": sorted(hex(t) for t in ents), "xbch_payloads": payloads})
    print(f"{len(rows)} FSSM blobs, problems: {dict(problems)}")
    if not rows:
        # No index: make_game.py takes fssm_index.json as proof this step finished, and an empty one is not.
        print(f"ERROR: no shader in {src} unpacked. Check that it holds the carved .dxbc files of a supported "
              "game version.", file=sys.stderr)
        return 1
    tmp = os.path.join(src, "fssm_index.json.tmp")   # written last and renamed in, so a stopped run leaves none
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
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
    try:  # a redirected stdout is strict ANSI code page under -I; write '?' for what it cannot encode
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    sys.exit(main())
