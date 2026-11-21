"""Carve shader containers out of an RPF9 archive without reading its (encrypted) table of contents.

Each shader in shaders.rpf is stored as:

    <name> 00  <u8 kind>  <u32 stored>  <u32 id>  <u32 stored>  <u32 size>  <raw deflate, `stored` bytes>

and inflates to a DXBC container (`size` bytes) holding DXIL plus an Xbox PRIV part.

Usage:
    python -I shader_carve.py <archive.rpf> <out_dir>

Writes <out_dir>/<name>.dxbc and <out_dir>/index.json (name, offset, stage, shader model, parts).
"""
import json
import mmap
import os
import re
import struct
import sys
import zlib

ENTRY = re.compile(rb"([\x21-\x7e]{3,200})\x00(.)(....)(....)\3(....)", re.DOTALL)
STAGES = {0: "ps", 1: "vs", 2: "gs", 3: "hs", 4: "ds", 5: "cs", 6: "lib", 13: "ms", 14: "as"}
RESERVED = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}


def parts(dxbc: bytes) -> dict:
    n = struct.unpack_from("<I", dxbc, 28)[0]
    out = {}
    for k in range(n):
        off = struct.unpack_from("<I", dxbc, 32 + 4 * k)[0]
        out[dxbc[off:off + 4].decode("latin-1")] = (off + 8, struct.unpack_from("<I", dxbc, off + 4)[0])
    return out


def shader_model(dxbc: bytes, p: dict):
    if "DXIL" not in p:
        return None, None
    off, _ = p["DXIL"]
    ver, kind = struct.unpack_from("<HH", dxbc, off)
    return STAGES.get(kind, str(kind)), f"{ver >> 4}.{ver & 0xF}"


def file_name(name: str, seen: dict, used: set) -> str:
    """A file name for `name` that no earlier shader has, as Windows compares them: without case, and with the
    device names (CON, NUL, COM1...) that open a device instead of a file. Same name twice means distinct
    variants; the second gets ~1, the third ~2."""
    base = re.sub(r'[<>:"/\\|?*]', "_", name)
    if base.split(".")[0].upper() in RESERVED:
        base = "_" + base
    key = base.lower()
    n = seen.get(key, 0)
    while True:
        fname = base + (f"~{n}" if n else "") + ".dxbc"
        if fname.lower() not in used:
            break
        n += 1
    seen[key] = n + 1
    used.add(fname.lower())
    return fname


def main() -> int:
    src, out = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    index, seen, used, bad = [], {}, set(), 0
    with open(src, "rb") as fh:
        m = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
        for mt in ENTRY.finditer(m):
            name = mt.group(1).decode("latin-1")
            stored = struct.unpack("<I", mt.group(3))[0]
            size = struct.unpack("<I", mt.group(5))[0]
            if not (16 < stored < 64 << 20 and 32 < size < 256 << 20):
                continue
            start = mt.end()
            try:
                data = zlib.decompressobj(-15).decompress(m[start:start + stored], size + 1)
            except zlib.error:
                bad += 1
                continue
            if data[:4] != b"DXBC" or len(data) != size:
                bad += 1
                continue
            p = parts(data)
            stage, sm = shader_model(data, p)
            fname = file_name(name, seen, used)
            with open(os.path.join(out, fname), "wb") as f:
                f.write(data)
            index.append({"name": name, "file": fname, "offset": mt.start(), "kind": mt.group(2)[0],
                          "stage": stage, "sm": sm, "size": size, "parts": {k: v[1] for k, v in p.items()}})
    # make_game.py takes index.json as proof the carve finished: write it last, under another name, then rename.
    tmp = os.path.join(out, "index.json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=1)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, os.path.join(out, "index.json"))
    from collections import Counter
    print(f"{len(index)} shaders carved, {bad} candidates rejected")
    print("stages:", dict(Counter(e["stage"] for e in index)))
    print("shader models:", dict(Counter(e["sm"] for e in index)))
    print("parts:", dict(Counter(k for e in index for k in e["parts"])))
    return 0


if __name__ == "__main__":
    sys.exit(main())
