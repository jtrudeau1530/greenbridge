"""Repackage the Opus streams of AWC audio banks into standard Ogg Opus files (lossless: no re-encoding).

    python -I awc_opus.py <bank.awc | dir of .awc> <out dir> [--limit N] [--jobs N]

AWC layout (little-endian "ADAT"), as read by rpf9-explorer's OpusPreview.cpp: a stream table of u32
(id[28:0] | tag count[31:29]), then 8-byte tags (type[63:56], size[55:28], offset[27:0]). Tag 0x55 = data,
0x76 = format (u32 samples, u16 rate at +8, codec at +28, channels at +29). Codec 0x0B = Opus at 48 kHz: data is
2048-byte blocks of [u16 length][packet] ..., a zero length pads the rest of the block. Codec 0 = PCM16 (written as
.wav). Codec 2 = 32-bit float PCM (float .wav). Streamed music (tag 0x48, multi-channel interleaved blocks) is not handled here yet.

Streamed music (tag 0x48 descriptor: u32 blocks, u32 chunk size, u32 channels, then 16 bytes per channel: id,
samples, rate at +10, codec at +12) is split into one mono Ogg Opus per channel: each chunk starts with a
24-byte record per channel (subblock count at +4, repeated-prefix samples to skip at +8), the channel data are
its run of 2048-byte subblocks after the 2048-aligned header, and the first `skip` samples of a chunk repeat the
previous chunk's tail and are dropped (whole packets).

Output: <out>/<bank>/<stream id>.ogg (or .wav), plus a summary line per bank.

For 16-bit PCM copies of the Ogg Opus files, run tools/audio_wav.py on the output (make_game.py does it with --wav).
"""
import argparse
import os
import struct
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

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


def write_wav(path: str, pcm: bytes, rate: int, channels: int, float32: bool = False) -> None:
    bits = 32 if float32 else 16
    hdr = b"RIFF" + struct.pack("<I", 36 + len(pcm)) + b"WAVEfmt " + struct.pack(
        "<IHHIIHH", 16, 3 if float32 else 1, channels, rate, rate * channels * bits // 8, channels * bits // 8, bits) +         b"data" + struct.pack("<I", len(pcm))
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
    streams, music = [], []
    for i in range(n):
        v = struct.unpack_from("<I", d, info + 4 * i)[0]
        sid, ntags = v & 0x1FFFFFFF, v >> 29
        data = fmt = mus = None
        for _ in range(ntags):
            t = struct.unpack_from("<Q", d, p)[0]
            p += 8
            typ, size, at = t >> 56, (t >> 28) & 0x0FFFFFFF, t & 0x0FFFFFFF
            if typ == 0x55:
                data = (at, size)
            elif typ == 0x76:
                fmt = (at, size)
            elif typ == 0x48:
                mus = (at, size)
        if mus and data:
            music.append((mus, data))
        if data and fmt and fmt[1] >= 32:
            samples, = struct.unpack_from("<I", d, fmt[0])
            rate, = struct.unpack_from("<H", d, fmt[0] + 8)
            codec, channels = d[fmt[0] + 28], d[fmt[0] + 29]
            streams.append({"id": sid, "samples": samples, "rate": rate, "codec": codec, "channels": channels,
                            "data": data})
    return streams, music


def music_channels(d: bytes, mus, data, collect_packets: bool = True):
    """[(channel id, samples, rate, codec, [packets])] for one streamed-music descriptor.

    collect_packets=False keeps the packet lists empty for metadata-only callers (awc_manifest.py)."""
    (m_at, m_size), (d_at, d_size) = mus, data
    blocks, chunk, channels = struct.unpack_from("<III", d, m_at)
    if not (1 <= channels <= 32 and 1 <= blocks and chunk >= 2048 and m_size == 12 + 16 * channels):
        return []
    info = []
    for c in range(channels):
        base = m_at + 12 + 16 * c
        cid = struct.unpack_from("<I", d, base)[0] & 0x1FFFFFFF
        samples, = struct.unpack_from("<I", d, base + 4)
        rate, = struct.unpack_from("<H", d, base + 10)
        info.append([cid, samples, rate, d[base + 12], []])
    for b in range(blocks):
        at = d_at + b * chunk
        recs = [struct.unpack_from("<III", d, at + 24 * c + 4)[:2] for c in range(channels)]   # (count, skip)
        total = sum(r[0] for r in recs)
        header = (channels * 24 + total * 4 + 2047) & ~2047
        before = 0
        for c, (count, skip) in enumerate(recs):
            start = at + header + before * 2048
            before += count
            if info[c][3] != 0x0B or count == 0 or not collect_packets:
                continue
            for pk in opus_packets(d, start, count * 2048):
                n = opus_samples(pk)
                if skip >= n and skip > 0:
                    skip -= n
                    continue
                info[c][4].append(pk)
    return [tuple(x) for x in info]


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
    with open(path, "rb") as f:
        d = f.read()
    name = os.path.splitext(os.path.basename(path))[0]
    od = os.path.join(out_dir, name)
    os.makedirs(od, exist_ok=True)
    counts = {"opus": 0, "pcm": 0, "other": 0, "music": 0}
    streams, music = parse_awc(d)
    for mus, data in music:
        for cid, samples, rate, codec, pk in music_channels(d, mus, data):
            if pk:
                n48 = samples * 48000 // rate if rate else 0
                write_ogg_opus(os.path.join(od, f"{cid:08X}.ogg"), pk, 1, cid, n48)
                counts["music"] += 1
    for s in streams:
        at, size = s["data"]
        if s["codec"] == 0x0B and size % 2048 == 0 and s["channels"] in (1, 2):
            pk = opus_packets(d, at, size)
            if pk:
                n48 = s["samples"] * 48000 // s["rate"] if s["rate"] else 0
                write_ogg_opus(os.path.join(od, f"{s['id']:08X}.ogg"), pk, s["channels"], s["id"], n48)
                counts["opus"] += 1
                continue
        if s["codec"] == 2 and s["channels"] in (1, 2) and size == s["samples"] * s["channels"] * 4:
            # codec 2: 32-bit float PCM (4 bytes per sample; values can exceed +-1, e.g. impulse responses)
            write_wav(os.path.join(od, f"{s['id']:08X}.wav"), d[at:at + size], s["rate"], s["channels"], True)
            counts["pcm"] += 1
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
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) // 2),
                    help="banks to export at once (default: half the CPU count)")
    a = ap.parse_args()
    if os.path.isdir(a.src):
        banks = []
        for root, _, files in os.walk(a.src):
            banks += [os.path.join(root, f) for f in files if f.lower().endswith(".awc")]
        banks.sort()
    else:
        banks = [a.src]
    banks = banks[: a.limit] if a.limit else banks
    tot = {"opus": 0, "pcm": 0, "other": 0, "music": 0}
    jobs = max(1, a.jobs)

    def add(c):
        for k in tot:
            tot[k] += c[k]

    def skipped(b, e):
        print(f"  skip {os.path.basename(b)}: {e}")

    if jobs == 1:
        for i, b in enumerate(banks):
            try:
                add(export_bank(b, a.out))
            except (ValueError, struct.error, IndexError, OSError) as e:
                skipped(b, e)
                continue
            if (i + 1) % 200 == 0:
                print(f"  {i + 1}/{len(banks)} banks: {tot}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=jobs) as ex:
            futures = {ex.submit(export_bank, b, a.out): b for b in banks}
            for i, fut in enumerate(as_completed(futures), 1):
                b = futures[fut]
                try:
                    add(fut.result())
                except (ValueError, struct.error, IndexError, OSError) as e:
                    skipped(b, e)
                if i % 200 == 0:
                    print(f"  {i}/{len(banks)} banks: {tot}", flush=True)
    print(f"{len(banks)} banks: {tot['opus']} Ogg Opus, {tot['music']} music channels (Ogg Opus), {tot['pcm']} WAV, "
          f"{tot['other']} not handled (XMA/other)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
