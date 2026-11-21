# Audio config music records

`audio.rpf` starts with a zlib-compressed **audio config**, then the raw AWC banks begin at archive
offset `0x129B000` (19,509,248). On the package measured here the config is 837 zlib streams and
54,143,493 inflated bytes. It is not a script resource and it is not part of the encrypted table of
contents. It holds the game's music/radio record table, the station record blocks and the mixer
configuration.

Carve the config without touching the AWC banks:

```powershell
python -I tools/rpf9_carve.py audio.rpf <out> --limit 0x129B000
python -I tools/audio_config.py <out> --join --json
```

`--join` concatenates the carved streams in name order before scanning. That matters because the
config is one logical blob split into zlib blocks, and a record can straddle a block boundary. On
the measured carve the per-file scan finds 1,588 records and the joined scan finds 1,589.

## Record layout

The music records share one 46-byte layout. A record's marker is at `+4`; the bytes before it carry
the duration:

| Offset | Size | Field |
|---:|---:|---|
| -4 | 4 | duration, milliseconds (u32) |
| 0 | 2 | `0c 01` marker |
| +2 | 2 | config id (u16) |
| +4 | 4 | `02 00 0c 00` |
| +8 | 4 | hash A (unresolved) |
| +12 | 4 | hash B: `joaat(normalized name)` |
| +16 | 4 | `00 00 02 23` |
| +20 | 22 | fixed flags/parameters (constant on this package) |

`hash B` is Jenkins one-at-a-time over the lowercased name, with runs of spaces and punctuation
collapsed to `_`. The AWC stream id is `hash B & 0x1FFFFFFF`. The duration is the real song/ad
length; the AWC manifest reports twice that value because of a stereo-sample counting artifact.

The records are not necessarily contiguous: they are 46 bytes apart inside a run, but omitted
records leave gaps. The tool finds records by the marker and the duration range, not by a fixed
stride or a hard-coded stream offset.

## What the table gives

On the measured package:

- **1,589 records**, 1,589 distinct ids, range `0xB422`–`0xBFC6`, 272,442.8 s total.
- **1,584 of 1,589** durations equal `AWC manifest duration / 2` within 0.01 s. The five exceptions
  are duplicate-title or placeholder hash collisions.
- **All 1,589** records map to an AWC stream id; the ids are the music streams exported by
  `tools/awc_opus.py`.
- The records are grouped by contiguous id ranges that follow station order. The same config also
  holds station records and station-local content-id lists; they are not part of this record table
  and this tool does not embed the package's ranges.

It does **not** give per-track start offsets. Id order is not mix order, and the schedule-like
structures in the config carry constant template fields, not cue times. A continuous radio mix must
be matched in the audio (`docs/music.md` if present, or the local radio-timing tools).

## Tool

```text
python -I tools/audio_config.py <carve dir or blob> [--json] [--names FILE] [--join]
                                 [--min-ms N] [--max-ms N]
```

- `scan(blob)` returns `offset`, `id`, `duration_ms`, `duration_s`, `hash_a`, `hash_b`,
  `stream_id`.
- `collect(src, names=..., join=...)` scans a blob or a directory; `names` maps hash B to a display
  string when a NUL-separated local name pool is supplied.
- `joaat` and `normalized_name` are the shared hash helpers.
- Text output prints one line per record and a summary; `--json` writes `{records, summary}`.

`--names` is optional. The package's debug name stream and the localization credits are local game
files; nothing from them is committed.

## Limits

- The table is record-level. It does not say which record is a song, a DJ line, an ad or a jingle;
  duration and the contiguous station ranges are the only built-in signals.
- Hash A is unresolved, and hash B resolves only through a supplied name pool or an AWC manifest.
- The first ~19.5 MiB bound is specific to this package. A different build can put the AWC start at
  another offset; use the first AWC bank (`ADAT`) as the boundary.
- No start offsets, no play order and no playlists are stored here. The config's station content
  lists reference records, but the mix order is assembled at runtime.
