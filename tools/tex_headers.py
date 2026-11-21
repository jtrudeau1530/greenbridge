"""Find texture headers in carved CPU-segment blobs and pair them with their GPU blob.

Texture descriptor layout (qwords, found in GTA VI's resource CPU segments):
  q[k-2]  packed u64 (unknown: size/flags)
  q[k-1]  0x980                     (descriptor type)
  q[k]    0x60000000 + offset       (GPU data pointer into the resource's GPU segment)
  q[k+4]  u16 width, u16 height, u16 depth/array, u8 ?, u8 DXGI format
  q[k+5]  tiling / flags (0x0001 = AMD SW_4K_S verified on an R16 texture)
The GPU segment is the carved blob that follows the CPU blob.

Usage:
    python -I tex_headers.py <carve dir> [--archives a,b] [--out textures_index.json]
"""
import argparse
import json
import os
import struct
import sys
from collections import Counter

DXGI = {
    2: "R32G32B32A32_FLOAT", 3: "R32G32B32A32_UINT", 4: "R32G32B32A32_SINT",
    6: "R32G32B32_FLOAT", 7: "R32G32B32_UINT", 8: "R32G32B32_SINT",
    10: "R16G16B16A16_FLOAT", 11: "R16G16B16A16_UNORM", 12: "R16G16B16A16_UINT",
    13: "R16G16B16A16_SNORM", 14: "R16G16B16A16_SINT",
    16: "R32G32_FLOAT", 17: "R32G32_UINT", 18: "R32G32_SINT",
    24: "R10G10B10A2_UNORM", 25: "R10G10B10A2_UINT", 26: "R11G11B10_FLOAT",
    28: "R8G8B8A8_UNORM", 29: "R8G8B8A8_UNORM_SRGB", 30: "R8G8B8A8_UINT",
    31: "R8G8B8A8_SNORM", 32: "R8G8B8A8_SINT",
    34: "R16G16_FLOAT", 35: "R16G16_UNORM", 36: "R16G16_UINT",
    37: "R16G16_SNORM", 38: "R16G16_SINT",
    41: "R32_FLOAT", 42: "R32_UINT", 43: "R32_SINT",
    49: "R8G8_UNORM", 50: "R8G8_UINT", 51: "R8G8_SNORM", 52: "R8G8_SINT",
    54: "R16_FLOAT", 56: "R16_UNORM", 57: "R16_UINT", 58: "R16_SNORM", 59: "R16_SINT",
    61: "R8_UNORM", 62: "R8_UINT", 63: "R8_SNORM", 64: "R8_SINT",
    71: "BC1_UNORM", 72: "BC1_UNORM_SRGB",
    74: "BC2_UNORM", 75: "BC2_UNORM_SRGB",
    77: "BC3_UNORM", 78: "BC3_UNORM_SRGB",
    80: "BC4_UNORM", 81: "BC4_SNORM",
    83: "BC5_UNORM", 84: "BC5_SNORM",
    87: "B8G8R8A8_UNORM", 91: "B8G8R8A8_UNORM_SRGB",
    95: "BC6H_UF16", 96: "BC6H_SF16",
    98: "BC7_UNORM", 99: "BC7_UNORM_SRGB",
}

GFX10_TO_DXGI = {
    1: 61,
    2: 63,
    5: 62,
    6: 64,
    7: 56,
    8: 58,
    11: 57,
    12: 59,
    13: 54,
    14: 49,
    15: 51,
    18: 50,
    19: 52,
    20: 42,
    21: 43,
    22: 41,
    23: 35,
    24: 37,
    27: 36,
    28: 38,
    29: 34,
    43: 26,
    44: 24,
    48: 25,
    56: 28,
    57: 31,
    60: 30,
    61: 32,
    62: 17,
    63: 18,
    64: 16,
    65: 11,
    66: 13,
    69: 12,
    70: 14,
    71: 10,
    72: 7,
    73: 8,
    74: 6,
    75: 3,
    76: 4,
    77: 2,
    130: 29,
    169: 71,
    170: 72,
    171: 74,
    172: 75,
    173: 77,
    174: 78,
    175: 80,
    176: 81,
    177: 83,
    178: 84,
    179: 95,
    180: 96,
    181: 98,
    182: 99,
}
DXGI_TO_GFX10 = {v: k for k, v in GFX10_TO_DXGI.items()}


def map_gfx10_format(gfx10_code):
    if gfx10_code not in GFX10_TO_DXGI:
        raise ValueError(f"unsupported or invalid GFX10 format: {gfx10_code}")
    dxgi_id = GFX10_TO_DXGI[gfx10_code]
    return dxgi_id, DXGI.get(dxgi_id, f"dxgi{dxgi_id}")


def dxgi_to_gfx10(dxgi_code):
    if dxgi_code not in DXGI_TO_GFX10:
        raise ValueError(f"unsupported DXGI format: {dxgi_code}")
    return DXGI_TO_GFX10[dxgi_code]


def calc_mipmap_count(w, h, depth=1):
    if w <= 0 or h <= 0 or depth <= 0:
        return 0
    return max(w, h, depth).bit_length()


def calc_mipmap_dimensions(w, h, depth=1, mips=0):
    if w <= 0 or h <= 0 or depth <= 0:
        return []
    max_mips = calc_mipmap_count(w, h, depth)
    count = max_mips if mips <= 0 else min(mips, max_mips)
    return [(max(1, w >> i), max(1, h >> i), max(1, depth >> i)) for i in range(count)]


FMT_INFO = {
    71: (8, 4), 72: (8, 4), 80: (8, 4), 81: (8, 4),
    74: (16, 4), 75: (16, 4), 77: (16, 4), 78: (16, 4), 83: (16, 4), 84: (16, 4),
    95: (16, 4), 96: (16, 4), 98: (16, 4), 99: (16, 4),
    61: (1, 1), 62: (1, 1), 63: (1, 1), 64: (1, 1),
    49: (2, 1), 50: (2, 1), 51: (2, 1), 52: (2, 1), 54: (2, 1), 56: (2, 1), 57: (2, 1), 58: (2, 1), 59: (2, 1),
    16: (8, 1), 17: (8, 1), 18: (8, 1),
    24: (4, 1), 25: (4, 1), 26: (4, 1), 28: (4, 1), 29: (4, 1), 30: (4, 1), 31: (4, 1), 32: (4, 1),
    34: (4, 1), 35: (4, 1), 36: (4, 1), 37: (4, 1), 38: (4, 1), 41: (4, 1), 42: (4, 1), 43: (4, 1),
    87: (4, 1), 91: (4, 1),
    6: (12, 1), 7: (12, 1), 8: (12, 1),
    10: (8, 1), 11: (8, 1), 12: (8, 1), 13: (8, 1), 14: (8, 1),
    2: (16, 1), 3: (16, 1), 4: (16, 1),
}


def calc_mip_level_bytes(w, h, depth=1, fmt=28, align=1):
    if fmt not in FMT_INFO:
        raise ValueError(f"unsupported format for byte calculation: {fmt}")
    bpe, blk = FMT_INFO[fmt]
    we = max(1, (w + blk - 1) // blk)
    he = max(1, (h + blk - 1) // blk)
    slice_bytes = we * he * bpe
    total = slice_bytes * max(1, depth)
    if align > 1:
        total = (total + align - 1) // align * align
    return total


DDSD_CAPS = 0x00000001
DDSD_HEIGHT = 0x00000002
DDSD_WIDTH = 0x00000004
DDSD_PITCH = 0x00000008
DDSD_PIXELFORMAT = 0x00001000
DDSD_MIPMAPCOUNT = 0x00020000
DDSD_LINEARSIZE = 0x00080000
DDSD_DEPTH = 0x00800000

DDSCAPS_COMPLEX = 0x00000008
DDSCAPS_MIPMAP = 0x00400000
DDSCAPS_TEXTURE = 0x00001000

DDSCAPS2_CUBEMAP = 0x00000200
DDSCAPS2_CUBEMAP_ALLFACES = 0x0000FE00
DDSCAPS2_VOLUME = 0x00200000

D3D10_RESOURCE_DIMENSION_TEXTURE2D = 3
D3D10_RESOURCE_DIMENSION_TEXTURE3D = 4
D3D10_RESOURCE_MISC_TEXTURECUBE = 0x4


def dds_header(w, h, depth=1, mips=1, fmt=28, is_cube=False, linear_size=0):
    hdr = bytearray(128)
    hdr[0:4] = b"DDS "
    flags = DDSD_CAPS | DDSD_HEIGHT | DDSD_WIDTH | DDSD_PIXELFORMAT
    if mips > 1:
        flags |= DDSD_MIPMAPCOUNT
    if depth > 1:
        flags |= DDSD_DEPTH
    if linear_size > 0:
        flags |= DDSD_LINEARSIZE
    else:
        flags |= 0x80000

    caps = DDSCAPS_TEXTURE
    if mips > 1:
        caps |= DDSCAPS_COMPLEX | DDSCAPS_MIPMAP
    if depth > 1 or is_cube:
        caps |= DDSCAPS_COMPLEX

    caps2 = 0
    if depth > 1:
        caps2 |= DDSCAPS2_VOLUME
    if is_cube:
        caps2 |= DDSCAPS2_CUBEMAP_ALLFACES

    struct.pack_into("<IIIIIII", hdr, 4, 124, flags, h, w, linear_size, depth if depth > 1 else 0, mips if mips > 1 else 1)
    struct.pack_into("<II4s", hdr, 76, 32, 4, b"DX10")
    struct.pack_into("<II", hdr, 108, caps, caps2)

    dim = D3D10_RESOURCE_DIMENSION_TEXTURE3D if depth > 1 else D3D10_RESOURCE_DIMENSION_TEXTURE2D
    misc = D3D10_RESOURCE_MISC_TEXTURECUBE if is_cube else 0
    array_size = 6 if is_cube else 1
    dxt10 = struct.pack("<IIIII", fmt, dim, misc, array_size, 0)
    return bytes(hdr) + dxt10


def dds_bytes(w, h, fmt, payload=b"", mips=1, depth=1, is_cube=False):
    return dds_header(w, h, depth=depth, mips=mips, fmt=fmt, is_cube=is_cube, linear_size=len(payload)) + payload


def parse_dds_header(data):
    if len(data) < 128:
        raise ValueError(f"DDS header truncated: expected at least 128 bytes, got {len(data)}")
    if data[:4] != b"DDS ":
        raise ValueError(f"invalid DDS magic: {data[:4]!r}")
    size, flags, h, w, pitch_linear, depth, mips = struct.unpack_from("<IIIIIII", data, 4)
    if size != 124:
        raise ValueError(f"invalid DDS header size: {size}")
    pf_size, pf_flags, fourcc = struct.unpack_from("<II4s", data, 76)
    if pf_size != 32:
        raise ValueError(f"invalid DDS pixel format size: {pf_size}")
    caps, caps2 = struct.unpack_from("<II", data, 108)

    info = {
        "magic": b"DDS ",
        "flags": flags,
        "width": w,
        "height": h,
        "depth": depth,
        "mips": mips,
        "pitch_linear": pitch_linear,
        "fourcc": fourcc,
        "caps": caps,
        "caps2": caps2,
    }

    if fourcc == b"DX10":
        if len(data) < 148:
            raise ValueError(f"truncated DX10 extension header: expected at least 148 bytes, got {len(data)}")
        dxgi_fmt, dim, misc, array_size, misc2 = struct.unpack_from("<IIIII", data, 128)
        info.update({
            "dxgi_format": dxgi_fmt,
            "resource_dimension": dim,
            "misc_flags": misc,
            "array_size": array_size,
            "misc_flags2": misc2,
        })
    return info


def scan_archive(carve, arch):
    # numpy is only needed for the descriptor scan; keeping it out of the module import lets the
    # standard-library tests run on a Python without numpy.
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("scan_archive needs numpy (pip install numpy)") from exc
    with open(os.path.join(carve, arch, "index.json"), encoding="utf-8") as f:
        files = sorted(json.load(f)["files"], key=lambda r: r["off"])
    out = []
    for i in range(len(files) - 1):
        v, g = files[i], files[i + 1]
        # CPU segment: starts with a zero qword but isn't zero-padded GPU data. The GPU segment is whatever
        # blob follows it (texture data often starts directly with BC blocks, not a zero header).
        if not (v["head"][:16] == "0" * 16 and not v["head"].startswith("0" * 32)):
            continue
        if g["head"].startswith("42400000") or g.get("type") == "bink2":
            continue
        # The GPU segment can span several consecutive blobs: everything up to the next CPU/header blob.
        chain = []
        for j in range(i + 1, min(len(files), i + 257)):
            r = files[j]
            if r["head"].startswith("42400000") or r.get("type") == "bink2" or (
                    r["head"][:16] == "0" * 16 and not r["head"].startswith("0" * 32) and chain):
                break
            chain.append([r["off"], r["size"]])
        with open(os.path.join(carve, arch, f"{v['off']:012x}.bin"), "rb") as f:
            V = f.read()
        q = np.frombuffer(V[: len(V) // 8 * 8], "<u8")
        # Any GPU pointer whose header (4 qwords later) decodes to sane dimensions and a known DXGI format.
        for k in np.nonzero(((q >> 28) == 6) & ((q >> 32) == 0))[0]:
            k = int(k)
            if k < 2 or k + 5 >= len(q):
                continue
            d = int(q[k + 4]).to_bytes(8, "little")
            w, h, depth = int.from_bytes(d[0:2], "little"), int.from_bytes(d[2:4], "little"), int.from_bytes(d[4:6], "little")
            fmt = d[7]
            if fmt not in DXGI or not (4 <= w <= 16384 and 4 <= h <= 16384 and 1 <= depth <= 2048):
                continue
            # Real texture data is 256-byte aligned (4 KB in every verified case); unaligned hits are other structs.
            if (int(q[k]) & 0xFF) != 0:
                continue
            out.append({"archive": arch, "cpu": v["off"], "gpu": g["off"], "gpu_size": g["size"], "gpu_chain": chain,
                        "data_off": int(q[k]) & 0x0FFFFFFF, "w": w, "h": h, "depth": depth, "b6": d[6], "fmt": fmt,
                        "fmt_name": DXGI.get(fmt, f"dxgi{fmt}"), "flags": int(q[k + 5]), "packed": int(q[k - 2]),
                        "dtype": int(q[k - 1]), "mips": (int(q[k + 5]) >> 16) & 0xFF, "q_index": k})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("carve")
    ap.add_argument("--archives")
    ap.add_argument("--out")
    a = ap.parse_args()
    archs = a.archives.split(",") if a.archives else sorted(
        d for d in os.listdir(a.carve) if os.path.isfile(os.path.join(a.carve, d, "index.json")))
    allt = []
    for arch in archs:
        t = scan_archive(a.carve, arch)
        print(f"{arch}: {len(t)} texture headers", flush=True)
        allt += t
    path = a.out or os.path.join(a.carve, "textures_index.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(allt, f)
    print(f"{len(allt)} textures -> {path}")
    print("formats:", Counter(t["fmt_name"] for t in allt).most_common(12))
    print("flags (low 16 bits):", Counter(hex(t["flags"] & 0xFFFF) for t in allt).most_common(10))
    print("byte6:", Counter(t["b6"] for t in allt).most_common(8))


if __name__ == "__main__":
    sys.exit(main())
