# Audio

`audio.rpf` is one of the few archives that is not mostly zlib. It holds **AWC** banks, and each bank uses the
little-endian **`ADAT`** container. The build exports the banks whole, then repackages or decodes their streams:

- Opus (codec `0x0B`) becomes standard **Ogg Opus**, losslessly: the stored packets are rewrapped, not re-encoded.
- PCM (codec `0` and codec `2`) becomes **WAV**.
- XMA2 (codec `5`) becomes a RIFF XMA2 file, which **ffmpeg** decodes to WAV.

This page describes the container, what each tool writes, and the bank types that are easy to misread when listening.
The corpus counts and verification results are in [design.md](design.md#asset-status-no-exe-needed).

## The AWC (`ADAT`) container

All integers are little-endian.

```
offset  size  field
0x00    4     "ADAT"
0x04    u32   flags
0x08    u32   stream count (n)
0x0C    u32   header size
```

Optional tables follow the header, selected by `flags`. `tools/awc_opus.py` skips the ones seen in the package:

| Flag | Extra bytes before the stream table |
|---|---|
| `0x00400000` | 4 |
| `0x00010000` | `n * 2` |
| `0x00200000` | `(n + 1) * 4` |

Then comes the stream table, one `u32` per stream:

```
bits 31:29  tag count (0-7)
bits 28:0   stream id
```

Each stream's tags follow, 8 bytes each:

```
bits 63:56  tag type
bits 55:28  size
bits 27:0   offset (from the start of the bank)
```

The exporters use three tag types:

| Tag | Meaning |
|---|---|
| `0x55` | stream data |
| `0x76` | format |
| `0x48` | streamed-music descriptor |

### Format tag (`0x76`)

The fields the exporters read:

```
+0    u32  sample count
+8    u16  sample rate
+28   u8   codec
+29   u8   channel count
```

The exporters ignore the remaining bytes.

### Codecs

| Codec | Format | Export |
|---|---|---|
| `0` | PCM, 16-bit | `.wav`, 16-bit PCM |
| `2` | PCM, 32-bit float (4 bytes per sample; values can exceed ±1, e.g. impulse responses) | `.wav`, IEEE float |
| `0x0B` (11) | Opus, 48 kHz | `.ogg`, Ogg Opus, lossless repackage |
| `5` | XMA2 | `.xma.wav`, RIFF XMA2; ffmpeg decodes it to PCM WAV |

Opus data is stored in 2048-byte blocks. Each block is a sequence of `[u16 length][packet]`; a zero length pads
the rest of the block. The first 312 decoded samples are the encoder's lookahead (libopus at 48 kHz); the AWC does
not store it, so the exporter writes that value as `OpusHead` pre-skip and sets the last page's granule to
`pre-skip + declared samples`. Decoders then drop the priming and stop at the real end instead of the padded final
frame. `--pre-skip` overrides the 312 default for a stream encoded with a different lookahead.

The format tag's sample count is already in 48 kHz frames even when its rate field is 32 kHz, and a stereo count is
interleaved across both channels. The exporter therefore does not scale it by the rate field and halves it for
stereo; scaling a 32 kHz stream by 48000/32000 leaves up to a frame of padding untrimmed.

Codec `5` is not handled by `tools/awc_opus.py`. `rpf9dump xma` writes it as RIFF XMA2, and ffmpeg does the decode.

### Streamed music (`0x48`)

A `0x48` tag describes a streamed mix. Its descriptor is:

```
+0    u32  block count
+4    u32  chunk size
+8    u32  channel count
then 16 bytes per channel:
+0    u32  channel id (low 29 bits)
+4    u32  sample count
+10   u16  sample rate
+12   u8   codec
```

At the `0x55` data offset, each chunk starts with a 24-byte record per channel. The record's `+4` word is the
channel's subblock count and `+8` is a repeated-prefix sample count to skip. The channel data are its run of
2048-byte subblocks after the 2048-aligned header. The skipped samples repeat the previous chunk's tail, so the
exporter drops whole packets until the count is consumed. Each descriptor channel becomes one Ogg Opus file.

The descriptor's `channels` field is the number of channel groups, not necessarily the packet channel count.
An Opus packet's TOC stereo bit (`packet[0] & 0x04`) gives the real count, and the descriptor's `samples` field
counts interleaved samples across those packet channels. The exporter therefore writes a stereo Ogg with
`OpusHead channels=2` and `samples / 2` as the end granule when the packets are stereo; mono groups keep their
count unchanged. Reading `samples` as a frame count produces a file that reports twice its real duration.

## What the exporters write

`rpf9dump` puts output under `<out>/<archive basename>/`. For `audio.rpf`, that is `<out>/audio/`.

| Command | Output |
|---|---|
| `rpf9dump audio <audio.rpf> <out>` | `<out>/audio/awc/<path>.awc` (the original banks) and `<out>/audio/audio_manifest.json` |
| `rpf9dump xma <audio.rpf> <out>` | `<out>/audio/xma/<bank>/<stream id>.xma.wav` for every codec-5 stream |
| `python -I tools/awc_opus.py <bank or dir> <out> [--jobs N] [--pre-skip N]` | `<out>/<bank>/<stream id>.ogg`, or `.wav` for codec 0/2; `--jobs` processes that many banks at once |
| `python -I tools/audio_wav.py <ogg dir> --out <wav dir> [--jobs N]` | 16-bit PCM WAV mirroring the Ogg tree; `--jobs` ffmpeg processes at once |

When exporting a directory, `awc_opus.py` keeps the path to each bank below the input root. For example,
`a/music.awc` writes under `<out>/a/music/`. Passing that file on its own writes under `<out>/music/`.
To regenerate a flat export from an earlier version, run `awc_opus.py` with the original bank directory and a fresh output directory.

`audio_manifest.json` is the bank index: each bank's `Path`, `Offset`, `Size`, and its `Streams` with `Index`,
`Id` (hex), `Codec`, `Channels`, `SampleRate`, `DurationSeconds` and `IsStreamed`. For streamed music,
`Channels` is the Opus packet channel count and `DurationSeconds` is per channel: the tag `0x48` descriptor's
interleaved sample count is divided by that channel count.

`audio_wav.py` runs ffmpeg with `-c:a pcm_s16le` at the stream's own rate and channel count. For a single stream
it trims to the final Ogg Opus granule minus pre-skip. Chained streams have a separate granule timeline for each
link, so ffmpeg handles their end trimming. Existing WAVs are normally skipped on resume. If the destination
filesystem treats two output names as the same file, the converter keeps the input extensions and adds a counter
where needed. It leaves a previously shared WAV untouched. Before reusing a name ending in `.ogg.wav` or `.opus.wav`
(including counters such as `.ogg~2.wav`), it decodes the source again and compares the PCM, rate and channel
layout. These checks read the full streams serially during planning. If the audio differs (including decoding
differences between ffmpeg versions) or the existing WAV cannot be verified, it stays in place and the converter
tries the next counter. Plain existing WAVs are
trusted by name, so resuming into the same output folder assumes the same set of inputs.
With no `--out`, a directory writes to `<src>/wav` and a single file writes next to itself.

## How `make_game.py --wav` uses them

The `assets` stage runs these in order:

```
rpf9dump audio   -> <work>/assets/audio/awc + audio_manifest.json
awc_opus.py      -> <work>/assets/audio/ogg
audio_wav.py     -> <work>/assets/audio/wav        (only with --wav)
rpf9dump xma     -> <work>/assets/audio/xma/<bank>/<stream id>.xma.wav
ffmpeg           -> <work>/assets/audio/xma/<bank>/<stream id>.wav
```

`--wav` is opt-in because 16-bit PCM is several times larger than Ogg Opus. It only adds PCM copies of the Ogg Opus
output; the XMA2 decode is separate and runs independently when ffmpeg is on PATH. If ffmpeg is missing, the
`.xma.wav` files are kept and the build says so. The stage skips work that is already present, including existing
WAVs.

## Songs, station mixes and venue stems

A bank is an asset container, not a radio station. The same `ADAT` framing holds several different things:

| What you see | What it is |
|---|---|
| One stream, typically around five minutes | An individual song or music asset, not a station feed |
| A long continuous stream | A station-style mix, much longer than a single song |
| A bank with 3 or 8 channels | **Venue stems**: the music plus the audience/ambience channels for one venue |

The multi-channel banks are the easy mistake. A 3-channel bank is not three radio stations, and an 8-channel bank is
not an eight-station lineup: it is one venue's mix split into stems, so the game can balance the music against the
crowd. `awc_opus.py` writes one mono Ogg per channel and does not mix them. Listening to a single stem can sound like
a dry music bed or like pure audience noise; the in-game result is the mix.

Use the bank name, stream count, duration and channel count together before deciding what a bank is. Station
programming is assembled by the game from many assets, so a lone AWC is rarely a complete station.

The music/radio **config table** is a separate structure: 1,589 records with exact durations, AWC stream ids,
contiguous station grouping and no per-track start offsets. See [audio_config.md](audio_config.md) for the record
layout and the station-content references.

## Mission music: Home Sweet Home

The credited "Burn This Shit Down" is not a radio-station track. It is the song for the
**Home Sweet Home** mission (Chapter 1, mission 21/25): Lucia at Jason's house in Key Lento
during the house fire and the escape. The mission and its localized objective/fail strings are
present in `data.rpf`; the song audio is not.

Measured coverage on the Xbox Series 1.0.2.1 package:

| Container | Searched | Result |
|---|---|---|
| `story000.rpf` streamed-music banks | 753 banks / 2,642 channels | none |
| `story000.rpf` plain banks | 201 banks / 1,950 streams | none |
| `story000.rpf` Bink | 264 streams / 162 with audio / 165 tracks / 5,090 s | none |
| `movies.rpf` Bink | 229 tracks / 12,401 s | none |
| `audio.rpf` streamed cues under 30 s | 1,149 channels / 5.78 h | none |
| `audio.rpf` all banks | 7,162 banks, all streams | none |
| loop/chunked banks | 1,099 banks / 118,180 streams | none |
| `packs.rpf` inflated | 144,364 files / 9.74 GB | 0 AWC, 0 Bink2 |
| `data.rpf` inflated | 28,509 files / 1.41 GB | mission text, 0 song strings |
| `script.rpf`, `common.rpf`, `story000_script.rpf` | 1,058 / 955 / 2,633 files | none |
| `levels.rpf` raw | 94 `ADAT` magics, 2,820 `KB2` patterns | 0 valid AWC/Bink |

Fingerprint search: about 19,807 files / 19.9 M Chromaprint frames / 653 fingerprint-hours, with a
0.50-2.00 tempo sweep over the story-music and long-AWC targets. No candidate passes the vote
threshold. The positive controls keep their earlier results: Sensimilla 2,284 votes / 0.911 bits
/ 100 %; Don't Break My Heart 88 votes / 7.046 bits / 94.5 %.

A `data.rpf` hash table contains 12 occurrences of `joaat("burn")`, but it is not the song: the
config blob has none, no AWC stream ID matches its low 29 bits, and the table has no audio
reference.

Residual uncertainty: `levels.rpf` (76 GB) was not fully inflated, and a custom RSC8 audio
resource is the only mission-side container not decoded; the retail track may be a third master
or patch content not covered by the public references; sub-30 s stings were not exhaustively
fingerprinted. No game audio is stored in the repository; the searches ran on a local copy and
`radio-dump-v3` was not modified.

## Never commit or share extracted audio

Extracted banks, Ogg Opus, WAV, XMA2 and the manifest are game data. Keep them in the work folder, never in the
repository, an issue, a pull request or a file-sharing link. Format notes, counts and a few header bytes are fine;
the audio itself is not. See the ground rules in [design.md](design.md#ground-rules) and
[CONTRIBUTING.md](../CONTRIBUTING.md).

## What has been checked

- 300 random Opus streams decode with ffmpeg with no errors; their lengths match the declared sample counts.
- All 1,818 XMA2 streams decode after `rpf9dump xma`.
- A 4-channel streamed-music bank gives four 42.0 s mono channels with real levels and no decoder errors.
- A 1-channel streamed-music descriptor can carry stereo Opus packets; it exports as one stereo Ogg with the
  interleaved sample count halved, and a 3-channel descriptor with stereo packets exports three stereo Oggs.
- In the 3,008-bank / 3,116-channel streamed-music census, 2,501 of 2,958 one-channel descriptors carry stereo
  packets; the 50 multi-channel descriptors split into 20 all-stereo and 30 all-mono banks, with no channel
  mixing TOC stereo bits. 61 representative exports decode cleanly.
- All exported WAVs decode. Codec 2 values are stored as-is, so a float WAV can exceed ±1.

The tools export the streams and channels as stored. They do not reconstruct runtime mixing, looping, ducking or
music-segment selection; that belongs to the game's audio system.
