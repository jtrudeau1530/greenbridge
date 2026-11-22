# `bink_play`: watch a Bink 2 movie through the package's decoder

`bink_play.cpp` maps the package's `Bink2Scarlett.dll` with `rt::Loader` (the path `bink_decode_test` checks) and plays
a Bink 2 movie in a window at the movie's frame rate. A movie inside an archive plays where it is: nothing is
extracted and nothing it decodes is written to disk.

The easy way is by name, through `tools/bink_play.py`:

```sh
set LEONIDA_GAME_DIR=C:\path\to\package
python tools/bink_play.py Stinger/RSG_STINGER.bk2     # or any unique part of the name: stinger
python tools/bink_play.py --list tv/                  # the names it knows
```

It names the streams of `movies.rpf` with `tools/bink_names.py` (from `common.rpf`'s mount lists) and starts
`bink_play movies.rpf@<offset>`. Naming takes about half a minute, so the table (names, offsets and frame sizes, no
movie data) is kept in `%LOCALAPPDATA%\Leonida\bink_names.csv` and rebuilt when the archive changes.

The executable itself:

```sh
bink_play <movie.bk2 | archive.rpf@OFFSET> [--game-dir DIR] [--frames N] [--headless] [--loop]
```

`--game-dir` defaults to `LEONIDA_GAME_DIR`, the movie to `LEONIDA_MOVIE`; a relative archive is taken from the game
folder. For `archive@OFFSET` it calls `BinkSetFileOffset(OFFSET)` and opens the archive with `BinkOpen`'s
`BINKFILEOFFSET` flag (0x20); without the flag the offset is ignored and `BinkOpen` reports "Not a Bink file".

| Key | Action |
|---|---|
| Space | pause / resume |
| Left / Right | one second back / forward (`BinkGoto`) |
| Home | back to the first frame |
| L | loop on / off |
| Esc, closing the window | quit |

The window title shows the frame number and the rate frames are being shown. The frame is scaled to the window with
its aspect ratio kept (GDI `StretchDIBits`, no GPU path).

`--headless` decodes without a window as fast as it can and prints the decode rate; `--frames N` stops after N
frames. CTest runs `bink_play --headless --frames 30` as `bink_play_headless`, which skips (exit 77) unless both
`LEONIDA_GAME_DIR` and `LEONIDA_MOVIE` are set.

## Measured on 1.0.2.1

`Stinger/RSG_STINGER.bk2` (3840x2160, 450 frames, 30 fps), played from inside `movies.rpf`: 90 frames decode
headless in about 1 s (81-118 fps over runs) on a Ryzen 7 5700X (the decoder is CPU-only), so the 4K stinger plays in
real time; the windowed run shows all 450 frames.

## Gaps

No audio: the movie's Bink Audio tracks are not decoded and no sound system is opened (as in `bink_decode_test`).
Playback uses the synchronous `BinkDoFrame` path only.
