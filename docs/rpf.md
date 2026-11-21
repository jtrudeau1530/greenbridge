# RPF9 archives and the carve tools

GTA VI ships its assets in RPF9 archives. The **table of contents** (names, sizes, offsets and flags) is
encrypted/packed; this project does not decode it and does not search for keys. The **data area** is mostly
zlib streams laid end to end, so it can be measured and carved without the TOC. Names in the recovery view
are offset-based (`recovered/...`), not the original paths.

## What is readable

- The header: magic `RPF9` (stored as `9FPR` little-endian), sizes/counts, and the data start at `u32 +4`.
  The TOC area immediately after the header is high-entropy and opaque.
- The data area: zlib streams, Bink 2 video (`KB2[a-z]`) in `movies.rpf`, and raw AWC banks in `audio.rpf`.
- From the existing survey in [docs/design.md](design.md): most archives are 93-99% plain zlib
  (`levels.rpf` 98.3%, about 1.1M files); `audio.rpf` is 99.9% raw AWC, not zlib; about half of
  `story000.rpf` and `shared_minigame000.rpf` is non-zlib. Unread bytes are recorded as a range, never
  guessed.

## Carving

```powershell
python -I tools/rpf9_carve.py <archive.rpf> <out_dir> [--start N] [--limit BYTES] [--index-only]
```

The walker starts at `--start` or the header's data offset, finds a zlib header (`78 01/5e/9c/da`) or a Bink 2
magic, inflates the stream, and continues after the bytes it actually consumed. `--index-only` records the
streams without writing the inflated files; `--limit` bounds a sample; `--start` starts at a known offset.

`<out_dir>/index.json` contains:

- `summary`: archive, files, compressed/inflated bytes, Bink 2 bytes/files, zero padding, unread bytes, span;
- `files`: `off`, `stored`, `size`, `pre` (bytes before the stream), `head` (first 16 inflated bytes), and for
  Bink 2 the `w`/`h`/`frames` fields.

The walker advances by `len(chunk)`, not by the 1 MiB read request. A short final slice used to make `p`
overshoot EOF, so `used` exceeded the real stream size and every trailing stream was skipped; the regression
test `tests/test_rpf9_carve.py` builds a two-stream archive to keep that fixed.

The carve is read-only and works on any `.rpf`, `.ttpf` or archive-like file. It does not decrypt the TOC, use
keys, or write anything outside the output directory.

## Recovery tools

`tools/rpf9dump` is a bulk exporter built on the Core library from
[breadbbbb/rpf9-explorer](https://github.com/breadbbbb/rpf9-explorer) (a `third_party` submodule; the
related-projects page records the licence boundary). It carves validated resources; it does
not decrypt the TOC either.

```powershell
dotnet build tools/rpf9dump/rpf9dump.csproj -c Release
rpf9dump ttpf    <pool.ttpf>  <out>
rpf9dump models <archive.rpf> <out>
rpf9dump textures <archive.rpf> <out>
rpf9dump audio  <audio.rpf>  <out>
rpf9dump xma    <audio.rpf>  <out>
```

`audio` writes the AWC banks plus `audio_manifest.json` (stream ids, codec, channels, rate, duration); `xma`
writes RIFF XMA2 WAVs that ffmpeg decodes. `models`/`textures` write OBJ/DDS plus a manifest. The tool's
`ArchiveReader` recovery names are offset-based, and its structured path validates XML/PSIN, models and
textures; the original RPF paths still require the missing executable/key.

`tools/manifest.py` is independent of RPF: it hashes a directory tree and verifies it later, with `--jobs` on
both paths. It is used to check exported assets without keeping a second copy.

## Limits

- The TOC is the hard boundary. File names, original paths and the exact file boundaries come from the TOC,
  so they are unavailable until the executable (or a documented key/schema) is part of the project. This
  repository does not search for or distribute keys.
- Some archives are not plain zlib (`audio.rpf`, parts of `story000.rpf`/`shared_minigame000.rpf`); the
  recovery tools handle the formats they know and report the rest as unread.
- Bink 2 video is carved as whole `.bk2` streams. FFmpeg builds generally expose only Bink 1; the package's
  `Bink2Scarlett.dll` decodes Bink 2 (see the movie notes).
- Extracted assets are for local use with your own copy. Never commit or share them.

## See also

- [docs/design.md](design.md) — the asset-stage table, coverage numbers and the recovery approach.
- [docs/story.md](story.md) — measured `story000`/`shared_minigame000` composition, nested RPF9 sub-archives and the `PSIN`/`RSC8` headers.
- `docs/audio.md` - AWC layout and the audio exporters, once the audio guide is merged.
- the related-projects page - the rpf9-explorer/PercyStar/SCARAB licence boundaries.
