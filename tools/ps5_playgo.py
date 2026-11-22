"""Read file assignments from exported PlayGo plgx, FLT and FICM tables."""

import struct


MASK = (1 << 64) - 1
SEED0, SEED1, ROUND = 0x92ca8aab26a24f51, 0x09bbb761a41bc44d, 0x8000000080008081


def rol(value, shift):
    return ((value << shift) | (value >> (64 - shift))) & MASK


def path_hash(path):
    data = path.removeprefix("/").encode("ascii").upper()
    if len(data) > 1024:
        raise ValueError("candidate path too long")
    a, b, c = SEED0, rol(SEED0, 11), rol(SEED0, 23)
    offset = 0
    for _ in range(max(0, (len(data) - 1) // 8)):
        a ^= int.from_bytes(data[offset:offset + 8], "little")
        offset += 8
        x = rol(rol(c ^ b, 5) ^ a, 53)
        y = rol(rol(c ^ a, 17) ^ b, 11)
        z = rol(rol(b ^ a, 1) ^ c, 59)
        a, b, c = ((~y & z) ^ x ^ ROUND) & MASK, ((~z & x) ^ y) & MASK, (z ^ (~x & y)) & MASK
    a ^= int.from_bytes(data[offset:], "little") ^ SEED1
    x = rol(rol(c ^ b, 5) ^ a, 53)
    y = rol(rol(c ^ a, 17) ^ b, 11)
    z = rol(rol(b ^ a, 1) ^ c, 59)
    return ((~y & z) ^ x ^ ROUND) & MASK


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def u16(data, at):
    return struct.unpack_from("<H", data, at)[0]


def u32(data, at):
    return struct.unpack_from("<I", data, at)[0]


def text_at(data, offset):
    require(0 <= offset < len(data), "label offset")
    end = data.find(b"\x00", offset)
    require(end >= 0, "unterminated label")
    return data[offset:end].decode("utf-8")


def parse(chunk, hashes, ficm):
    require(len(chunk) >= 256 and chunk[:4] == b"plgx", "plgx header")
    require(u32(chunk, 16) == len(chunk) and u16(chunk, 4) == 0x1000, "plgx version/size")
    count, mcount = u16(chunk, 10), u16(chunk, 32)
    require(0 < count <= 255 and 0 < mcount <= 65535, "chunk counts")
    sections = []
    for at in range(192, 248, 8):
        start, size = u32(chunk, at), u32(chunk, at + 4)
        require(256 <= start <= len(chunk) and start + size <= len(chunk), "section bounds")
        sections.append(chunk[start:start + size])
    attrs, indices, labels, raw_ranges, *_ = sections
    require(len(attrs) == count * 32 and len(raw_ranges) == mcount * 16, "attribute lengths")
    require(len(indices) % 4 == 0, "mchunk index alignment")
    chunks = []
    for i in range(count):
        at = i * 32
        n, list_offset = u32(attrs, at + 4), u32(attrs, at + 24)
        require(list_offset % 4 == 0 and list_offset + n * 4 <= len(indices), "mchunk list bounds")
        owned = list(struct.unpack_from("<" + "I" * n, indices, list_offset))
        require(all(m < mcount for m in owned), "mchunk reference")
        chunks.append({"id": i, "label": text_at(labels, u32(attrs, at + 28)), "mchunks": owned})
    require(len(hashes) >= 56 and hashes[24:28] == b"\x7fFLT", "FLT header")
    require(u32(hashes, 0) == 1 and u32(hashes, 8) == 56, "FLT version/offset")
    n = u32(hashes, 36)
    require(n <= 1000000 and u32(hashes, 12) == n * 8 and len(hashes) == 56 + n * 8, "FLT sizes")
    require(hashes[40:56] == struct.pack("<QQ", SEED0, SEED1), "unsupported FLT seeds")
    values = list(struct.unpack_from("<" + "Q" * n, hashes, 56))
    require(all(a < b for a, b in zip(values, values[1:])), "FLT not strictly sorted")
    require(len(ficm) >= 16 and u32(ficm, 0) == 1 and u32(ficm, 8) == 16, "FICM header")
    require(u32(ficm, 12) == n * 2 and len(ficm) == 16 + n * 2, "FICM payload bytes/count")
    assignments = list(struct.unpack_from("<" + "H" * n, ficm, 16))
    require(all(i < count for i in assignments), "FICM chunk out of range")
    rows = [{"index": i, "path_hash": f"{h:016x}", "chunk": c} for i, (h, c) in enumerate(zip(values, assignments))]
    return {"chunks": chunks, "file_assignments": rows}
