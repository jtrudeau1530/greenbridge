"""Repackage the Opus streams of AWC audio banks into standard Ogg Opus files (lossless: no re-encoding).

    python -I awc_opus.py <bank.awc | dir of .awc> <out dir> [--limit N]

AWC layout (little-endian "ADAT"), as read by rpf9-explorer's OpusPreview.cpp: a stream table of u32
(id[28:0] | tag count[31:29]), then 8-byte tags (type[63:56], size[55:28], offset[27:0]). Tag 0x55 = data,
0x76 = format (u32 samples, u16 rate at +8, codec at +28, channels at +29). Codec 0x0B = Opus at 48 kHz: data is
2048-byte blocks of [u16 length][packet] ..., a zero length pads the rest of the block. Codec 0 = PCM16 (written as
.wav). Streamed music (tag 0x48, multi-channel interleaved blocks) is not handled here yet.

Output: <out>/<bank>/<stream id>.ogg (or .wav), plus a summary line per bank.
"""
import argparse
import os
import struct
import sys

# ---- Ogg ----
_CRC = []
for _i in range(256):
    _r = _i << 24
    for _ in range(8):
        _r = ((_r << 1) ^ 0x04C11DB7) if _r & 0x80000000 else (_r << 1)
    _CRC.append(_r & 0xFFFFFFFF)


def ogg_crc(data: bytes) -> int:
    c = 0
    for b in data:
        c = ((c << 8) & 0xFFFFFFFF) ^ _CRC[((c >> 24) ^ b) & 0xFF]
    return c


def ogg_page(serial: int, seq: int, granule: int, packets: list, flags: int) -> bytes:
    lacing = bytearray()
    for p in packets:
        n = len(p)
        while n >= 255:
            lacing.append(255)
            n -= 255
        lacing.append(n)
    hdr = struct.pack("<4sBBqIIIB", b"OggS", 0, flags, granule, serial, seq, 0, len(lacing)) + bytes(lacing)
    body = b"".join(packets)
    page = bytearray(hdr + body)
    struct.pack_into("<I", page, 22, ogg_crc(bytes(page)))
    return bytes(page)


def opus_samples(packet: bytes) -> int:
    """Samples at 48 kHz in one Opus packet (RFC 6716 TOC)."""
    toc = packet[0]
    config = toc >> 3
    if config < 12:
        frame = [480, 960, 1920, 2880][config & 3]          # SILK 10/20/40/60 ms
    elif config < 16:
        frame = [480, 960][config & 1]                        # hybrid 10/20 ms
    else:
        frame = [120, 240, 480, 960][config & 3]              # CELT 2.5/5/10/20 ms
    c = toc & 3
    if c == 0:
        frames = 1
    elif c in (1, 2):
        frames = 2
    else:
        frames = packet[1] & 0x3F if len(packet) > 1 else 0
    return frame * frames


def write_ogg_opus(path: str, packets: list, channels: int, serial: int, length: int = 0) -> int:
    """length: the stream's declared sample count at 48 kHz; the last page's granule position is trimmed to it
    (Ogg Opus end trimming), so decoders stop at the exact end instead of the padded last frame."""
    head = b"OpusHead" + struct.pack("<BBHIhB", 1, channels, 0, 48000, 0, 0)
    tags = b"OpusTags" + struct.pack("<I", 8) + b"leonida " + struct.pack("<I", 0)
    out = [ogg_page(serial, 0, 0, [head], 0x02), ogg_page(serial, 1, 0, [tags], 0x00)]
    seq, granule, pending, size = 2, 0, [], 0
    for i, p in enumerate(packets):
        pending.append(p)
        granule += opus_samples(p)
        size += len(p)
        last = i == len(packets) - 1
        if size > 4000 or len(pending) >= 50 or last:
            g = min(granule, length) if (last and 0 < length) else granule
            out.append(ogg_page(serial, seq, g, pending, 0x04 if last else 0x00))
            seq += 1
            pending, size = [], 0
    with open(path, "wb") as f:
        f.write(b"".join(out))
    return granule


def write_wav(path: str, pcm: bytes, rate: int, channels: int) -> None:
    hdr = b"RIFF" + struct.pack("<I", 36 + len(pcm)) + b"WAVEfmt " + struct.pack(
        "<IHHIIHH", 16, 1, channels, rate, rate * channels * 2, channels * 2, 16) + b"data" + struct.pack("<I", len(pcm))
    with open(path, "wb") as f:
        f.write(hdr + pcm)


# ---- AWC ----
def parse_awc(d: bytes):
    if d[:4] != b"ADAT":
        raise ValueError("not a little-endian AWC")
    flags, n, header_size = struct.unpack_from("<III", d, 4)
    p = 16 + (4 if flags & 0x400000 else 0)
    if flags & 0x10000:
        p += n * 2
    if flags & 0x200000:
        p += (n + 1) * 4
    info = p
    p += n * 4
    streams = []
    for i in range(n):
        v = struct.unpack_from("<I", d, info + 4 * i)[0]
        sid, ntags = v & 0x1FFFFFFF, v >> 29
        data = fmt = None
        for _ in range(ntags):
            t = struct.unpack_from("<Q", d, p)[0]
            p += 8
            typ, size, at = t >> 56, (t >> 28) & 0x0FFFFFFF, t & 0x0FFFFFFF
            if typ == 0x55:
                data = (at, size)
            elif typ == 0x76:
                fmt = (at, size)
        if data and fmt and fmt[1] >= 32:
            samples, = struct.unpack_from("<I", d, fmt[0])
            rate, = struct.unpack_from("<H", d, fmt[0] + 8)
            codec, channels = d[fmt[0] + 28], d[fmt[0] + 29]
            streams.append({"id": sid, "samples": samples, "rate": rate, "codec": codec, "channels": channels,
                            "data": data})
    return streams


def opus_packets(d: bytes, at: int, size: int):
    out = []
    for b in range(at, at + size, 2048):
        p = b
        end = b + 2048
        while p + 2 <= end:
            n, = struct.unpack_from("<H", d, p)
            if n == 0 or p + 2 + n > end:
                break
            out.append(bytes(d[p + 2:p + 2 + n]))
            p += 2 + n
    return out


def export_bank(path: str, out_dir: str):
    d = open(path, "rb").read()
    name = os.path.splitext(os.path.basename(path))[0]
    od = os.path.join(out_dir, name)
    os.makedirs(od, exist_ok=True)
    counts = {"opus": 0, "pcm": 0, "other": 0}
    for s in parse_awc(d):
        at, size = s["data"]
        if s["codec"] == 0x0B and size % 2048 == 0 and s["channels"] in (1, 2):
            pk = opus_packets(d, at, size)
            if pk:
                n48 = s["samples"] * 48000 // s["rate"] if s["rate"] else 0
                write_ogg_opus(os.path.join(od, f"{s['id']:08X}.ogg"), pk, s["channels"], s["id"], n48)
                counts["opus"] += 1
                continue
        if s["codec"] == 0 and s["channels"] in (1, 2):
            write_wav(os.path.join(od, f"{s['id']:08X}.wav"), d[at:at + size], s["rate"], s["channels"])
            counts["pcm"] += 1
            continue
        counts["other"] += 1
    return counts


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("out")
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    if os.path.isdir(a.src):
        banks = []
        for root, _, files in os.walk(a.src):
            banks += [os.path.join(root, f) for f in files if f.lower().endswith(".awc")]
        banks.sort()
    else:
        banks = [a.src]
    banks = banks[: a.limit] if a.limit else banks
    tot = {"opus": 0, "pcm": 0, "other": 0}
    for i, b in enumerate(banks):
        try:
            c = export_bank(b, a.out)
        except (ValueError, struct.error, IndexError) as e:
            print(f"  skip {os.path.basename(b)}: {e}")
            continue
        for k in tot:
            tot[k] += c[k]
        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{len(banks)} banks: {tot}", flush=True)
    print(f"{len(banks)} banks: {tot['opus']} Ogg Opus, {tot['pcm']} WAV, {tot['other']} not handled (music/XMA/other)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
