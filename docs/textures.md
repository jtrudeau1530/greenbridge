# Textures: the BCPack gap

Measured on 2026-11-21 from one retail Xbox Series install (1.0.2.1). The corpus is all 57
RDS5 texture pools: 46 under `data/ttdb/**` and 11 under `pack/<name>/data/ttdb/**`, 57.94 GiB
of files. Nothing below reproduces game data; every number is an aggregate or a descriptor
field.

## Outcome

| | Textures |
|---|---:|
| catalog entries | 357,115 |
| exported with the complete source mip chain | 98,603 |
| exported with fewer mips | 210,292 |
| exported at least one mip | 308,895 |
| skipped | 48,220 |

**Bottom line:** the compact mip chain is decoded and validated below, and a clean-room
stdlib-only BCPack decoder (`tools/bcpack.py`, `tools/bcpack_tables.py`) now decodes the
rANS entropy layer for all six active modes (BC1, BC3, BC4, BC5, BC6H, BC7) and rebuilds
BC1, BC3, BC4 and BC5 blocks (95.0% of all BCPack chunks, 89.6% of BCPack bytes). BC6H and
BC7 block rebuilders are left for the follow-up PR; `rpf9dump`'s C# `TtpfReader` still uses
the baseline zlib/raw path until ported.

The compact layout rule moves 206,661 2D textures into complete exports and 1,218 more into
partial exports (`305,264` complete, `16,419` partial, `35,432` skipped under zlib/raw).
Resolving the 697 multi-slice descriptors (127 2D arrays — including the 19 two-slice arrays
formerly misread as DXGI 156/199/211/226 — 567 cubemaps, and 3 3D volumes; see
`docs/textures_arrays.md`) adds 690 more complete exports and 4 partials under zlib/raw,
bringing the zlib/raw totals to **305,954 complete**, **16,423 partial** and **34,738 skipped**
(33,507 2D reader BCPack, 1,223 compact all-BCPack, 3 array all-BCPack and 5 DXGI 11; **0**
remaining layout skips).

Adding BCPack BC1, BC3, BC4 and BC5 in `tools/ttpf_mips.py` unlocks **47,886** more complete
chains across the 57 pools (47,882 2D + 4 multi-slice arrays), bringing the projected split to
**353,840 complete** (99.08% of catalog entries), **2,518 partial** and **757 skipped** (5 DXGI
11 plus 752 remaining BC6H/BC7 BCPack skips; **0** layout skips and **0** skipped multi-slice
descriptors).

This is the same 308,895 / 48,220 baseline already recorded in `docs/design.md`. The recount
splits the 48,220 by cause: the old "(BCPack or unmapped)" label hides 14,713 skips whose
recorded limitation is a layout/format check, not the codec, and overstates unmapped, which
is zero. Together, the compact 2D and multi-slice layout rules resolve all 697 multi-slice
descriptors and 14,011 2D compact descriptors, leaving only BCPack and 5 DXGI 11 records.

## Skip breakdown

| Cause | Before | After (zlib/raw) |
|---|---:|---:|
| BCPack blocks every usable mip (`TtpfReader`: "All source mips require…") | 33,507 | 33,507 |
| compact mip mapping not in the validated profile table | 14,030 | 0 |
| compact layout resolved, but every mip is BCPack | — | 1,223 |
| former "compact element size unknown" (19 two-slice 2D arrays; see `docs/textures_arrays.md`) | — | 0 |
| 2D array / cubemap / volume descriptor (`dimension != 1` or `depth > 1`; see `docs/textures_arrays.md`) | 678 | 0 |
| array layout resolved, but every mip is BCPack | — | 3 |
| element format not in the parser table (DXGI 11, `R16G16B16A16_UNORM`) | 5 | 5 |
| no mapped allocation ("unmapped shard") | 0 | 0 |
| **total** | **48,220** | **34,738** |

The "before" compact row becomes the measured rows below it after the layout rules: records
the rule sizes but cannot decode under zlib/raw because every mip is BCPack. The 1,218 mixed
compact records and 4 mixed array records that become partial exports are not skips; they are
counted in the partial totals below.

The compact row is not one thing:

- 13,979 of the 14,030 carry the descriptor's auxiliary-mapping bits. 13,954 of those are
  mode 4 or 6; the other 25 are modes 5, 7, 36 and 38. The other 51 compact records are modes
  1, 32 or 34 without the aux flag.
- 11,570 of the 14,030 have no BCPack chunk anywhere. They fail because
  `ValidatedCompactProfile` has no entry for their
  `(mode, width, height, mips, nonTail, bytes)` tuple, not because of a codec.
- 19 mode-1 records (64 KiB) originally appeared as unknown DXGI formats 156 (7), 199 (6),
  211 (5) and 226 (1) at 256×256 with 7 mips (2) or 512×512 with 8 mips (17). With the
  corrected 7-bit format / 11-bit depth bitfield (`docs/textures_arrays.md`), all 19 are
  two-slice 2D arrays (`depth = 2`) with standard DXGI formats 28/71/83/98 (18 complete
  zlib/raw, 1 partial).

The five format-11 records are 4×4, 8×4 or 128×4 `R16G16B16A16_UNORM`; the parser's
element-size table has no 8-byte entry for format 11. The 678 records formerly skipped as
array/volume descriptors (plus the 19 two-slice arrays above, 697 total) now have validated
layouts in `tools/tex_headers.py` and `tools/ttpf_mips.py` (`docs/textures_arrays.md`).

## The BCPack footprint

The skip count is only the fully blocked textures. Counting mip 0 and the lower mips
independently of the parser's profile checks gives the full footprint:

| mip 0 | lower mips | Textures |
|---|---|---:|
| zlib/raw | zlib/raw | 305,264 |
| BCPack | at least one zlib/raw | 14,935 |
| zlib/raw | at least one BCPack | 9,999 |
| BCPack | every mip BCPack | 26,215 |
| no usable layout | — | 702 |
| **total** | | **357,115** |

- **41,150 textures have BCPack in mip 0** (26,215 + 14,935).
- **26,215 have no zlib/raw mip at all**, so they cannot export anything.
- 5,845 export from a lower mip because mip 0 is BCPack (`BaseMipLevel > 0`).
- 9,090 of the 14,935 are still skipped: 8,515 because no usable mip is entirely zlib/raw
  (the zlib block is mixed with BCPack or outside the usable range), 575 because the compact
  profile is not validated.
- 9,356 export the top mip but stop before at least one BCPack lower mip; 643 more are
  skipped for the compact profile before an export can happen.

Cross-tabulated against the skip table:

| Parser cause | all zlib | mip 0 zlib, lower BCPack | mip 0 BCPack, lower zlib | all BCPack | no layout | total |
|---|---:|---:|---:|---:|---:|---:|
| BCPack | 0 | 0 | 8,515 | 24,992 | 0 | 33,507 |
| compact profile | 11,570 | 643 | 575 | 1,223 | 19 | 14,030 |
| array/volume | 0 | 0 | 0 | 0 | 678 | 678 |
| format 11 | 0 | 0 | 0 | 0 | 5 | 5 |
| **total** | **11,570** | **643** | **9,090** | **26,215** | **702** | **48,220** |

The partial exports are a separate gap. Of the 210,292:

| Partial reason | Textures |
|---|---:|
| compact profile, standalone mips exported, packed tail missing | 196,257 |
| lower mips BCPack | 8,052 |
| catalog top-only record | 314 |
| other (lower-mip export, mixed ranges) | 5,669 |
| **total** | **210,292** |

The packed-tail gap affects more textures than the BCPack skip count (196,257 partial
exports vs 33,507 skips), but it loses only the small mips. It is also not a BCPack problem:
196,257 textures stop at the descriptor's `non_tail` field because `UsableMipLimit =
NonTailMipCount` for compact profiles, regardless of codec. The next section measures that
field against the layout and finds that it is not the tail boundary: most of the "packed
tail" is standalone mips the reader drops.

## The compact mip chain

The 196,257 partial exports and the 14,030 compact-profile skips are the same layout
problem, not a codec problem. `tools/tex_headers.py` now carries the layout rule as pure
functions, `tools/ttpf_mips.py` measures and decodes with it, and `tests/test_tex_headers.py`
covers the rule on synthetic chains. The counts below come from `scan` and `check` over the
same 57 pools.

### The descriptor mode is a flag plus a tile mode

`TtpfReader` reads the descriptor mode as one number (`b >> 23`) and selects a 64 KiB tile
only for mode 0. The observed values split into a low flag bit and a 4-bit Gfx10 tile mode at
bit 1:

| `(mode >> 1) & 0xF` | Gfx10 mode | Tile | Descriptors |
|---:|---|---:|---:|
| 0 | `SW_64KB_D` | 64 KiB | 39,283 |
| 1 | `SW_4KB_S` | 4 KiB | 70,497 |
| 2 | `SW_4KB_S` | 4 KiB | 195,957 |
| 3 | `SW_4KB_S` | 4 KiB | 50,811 |
| 9–12 | array/volume | — | 567 |

Modes 9–12 are exactly the 567 cubemap descriptors; the other 130 multi-slice records (127 2D
arrays and 3 volumes) use tile mode 0 or 1. Bit 0 of the field is independent: modes 4/5 share
tile mode 2, 6/7 share tile mode 3, and 36/38 share tile modes 2/3. The reader's `mode == 0`
test therefore gives a 4 KiB layout to modes 1 and 32, which are 64 KiB. Fifteen single-slice
mode 1/32 descriptors have a known element size (4 mode 1, 11 mode 32); all fifteen match a
64 KiB allocation span and one (`maps_4_7` #6034) decodes at 0.00–0.33 levels (see below). The
other 19 mode 1 records formerly counted as unknown-format 2D entries are two-slice 2D arrays
(`depth = 2`, DXGI 28/71/83/98; see `docs/textures_arrays.md`).

### The layout is the AMD mip tail

The compact profile is the Gfx10 mip chain: standalone mips are padded to whole blocks and
stored smallest-first; a mip whose element count fits half a tile width and a tile height,
among the last 8 (4 KiB) or 12 (64 KiB) mips, goes into one shared tile at offset 0 at the
AMD entry offset (`m = max_tail - 1 - k`; `entry = m > 6 ? 16 << m : m << 8`).

On all 356,413 single-slice 2D descriptors with a known element size — the 232,789
already-trusted compact exports, the 14,011 layout-capable records the profile table rejects,
and every non-compact mode — the computed size equals `next_allocation - base`. There are
**0 mismatches**. Of the remaining 702 descriptors, 697 (127 2D arrays, 567 cubemaps, and 3
volumes) are validated with 0 mismatches in `docs/textures_arrays.md`, leaving only the 5 DXGI
11 (`R16G16B16A16_UNORM`) records without a format entry.

`tools/tex_headers.py` exposes the rule without any container dependency:

- `gfx10_tile_mode(mode)` / `gfx10_tile_bytes(mode)` split the descriptor field into the
  flag bit and tile mode and pick 64 KiB only for tile mode 0.
- `parse_etpf_texture_record(record)` decodes the 16-byte ETPF record, including the 4-bit
  packed mip count plus the 3-bit `non_tail` field that make up the total mip count.
- `compact_surface_layout(width, height, mips, element_bytes, block, tile)` computes the
  allocation size and every mip's offset, region and in-tile origin. It rejects a mip count
  that cannot describe the surface, an element size outside 1/2/4/8/16 bytes, a block size
  other than 1 or 4, and any tile other than 4 or 64 KiB.
- `compact_layout_for_dxgi` applies `FMT_INFO` to that function; `compact_profile_key`
  returns the exact `(mode, width, height, mips, non_tail, bytes)` tuple; and
  `compact_span_matches` accepts an allocation span only when it is byte-exact.

`tools/ttpf_mips.py` uses those functions for `scan`, `layout`, `check` and `export`, so the
measurement and the proposed reader rule cannot drift apart.

The rule is only claimed for 2D descriptors with a 1/2/4/8/16-byte element. The 12-byte DXGI
formats 6/7/8 do not occur in the measured 2D set, so `compact_surface_layout` rejects them
rather than guessing a Gfx10 log pattern. Uncompressed tiling is also not validated: the
allocation size is computed, but no DDS is written for those formats.

### `non_tail` is not the layout boundary

The descriptor's 3-bit `non_tail` field (`b & 7`) is where the reader stops, and its note calls
the remaining mips "packed tail". The bytes do not follow that split:

- 177,111 of the 232,789 compact exports have `non_tail < first_tail`; 214,298 standalone
  mips lie beyond the limit, 204,108 of them in zlib/raw chunks.
- 196,257 partial exports stop exactly at `non_tail`; 158,272 of them stop **before** the
  computed tail boundary, and only 37,985 stop at it.
- Of the 1,072,068 missing mips, 1,065,138 are in zlib/raw chunks. That decodable part splits
  into 193,019 standalone mips and 872,119 packed-tail mips; the other 6,930 missing mips (in
  1,166 textures) touch BCPack.

`UsableMipLimit = NonTailMipCount` therefore drops decodable standalone mips, not only the
tile. The allocation spans and the decoded content both put the tail at the computed index.

### The 14,030 profile-table records

| | Textures |
|---|---:|
| same layout, new tuple (4 KiB) | 13,956 |
| same layout, modes 5/7/34/38 | 40 |
| same layout, 64 KiB (modes 1/32) | 15 |
| **2D compact layout resolved** | **14,011** |
| two-slice 2D arrays (formerly misread as DXGI 156/199/211/226; see `docs/textures_arrays.md`) | 19 |
| **total** | **14,030** |

The rejected 2D records contain 84 distinct six-tuples with a known element size; the 19
remaining records are two-slice 2D arrays handled by `array_surface_layout`. The ten largest
known-size 2D tuples are:

| `mode` | Width×height | Mips | `non_tail` | Element bytes | Records |
|---:|---|---:|---:|---:|---:|
| 4 | 512×512 | 3 | 3 | 16 | 8,823 |
| 4 | 256×256 | 2 | 1 | 16 | 1,831 |
| 4 | 256×256 | 3 | 1 | 8 | 1,744 |
| 4 | 512×256 | 3 | 1 | 8 | 344 |
| 4 | 256×128 | 3 | 1 | 8 | 319 |
| 4 | 1024×512 | 3 | 3 | 8 | 191 |
| 4 | 512×256 | 9 | 1 | 8 | 68 |
| 4 | 256×256 | 2 | 1 | 8 | 66 |
| 4 | 512×32 | 4 | 1 | 8 | 62 |
| 4 | 512×512 | 10 | 3 | 16 | 58 |

None of these needs a new table row: every one has zero allocation-span mismatches under the
computed rule. `scan --tuples` emits the complete list without texture bytes.

What the layout change does to the 14,030:

| | Textures |
|---|---:|
| complete zlib/raw chain (2D compact) | 11,570 |
| partial export: some readable contiguous mip range (2D compact) | 1,218 |
| still skipped: every mip is BCPack (2D compact) | 1,223 |
| two-slice 2D arrays (resolved in `docs/textures_arrays.md`: 18 full, 1 partial) | 19 |
| **total** | **14,030** |

- The 11,570 complete records hold all 35,727 mips (2.91 GiB decompressed).
- 469 of the 1,218 partial records have a readable tail tile while at least one larger mip is
  BCPack; the other 749 export some contiguous mip range.
- The 1,223 BCPack skips are compact records whose every mip is BCPack; they were counted in
  the 14,030 before and are not part of the 33,507.

The five DXGI 11 (`R16G16B16A16_UNORM`) records are a separate one-line gap: DXGI 11 is 8
bytes per element, but the vendored element table has no entry. Their layout then matches,
but uncompressed tiling is not validated, so they are not exported here.

### Sample validation

`tools/ttpf_mips.py check` detiles one chain, decodes BC1/BC3/BC4/BC5 and compares each mip
with a 2x2 box downsample of its parent at BC-block resolution (one block is 4x4 pixels). A
correct chain is 0–4 levels, as in the SixNG notes; the control is the first tail mip detiled
at origin (0,0).

| Pool # | Descriptor | Mips | non_tail / first_tail | Max error | Tail control |
|---|---|---:|---:|---:|---:|
| `interiors_0_3` #19 | 512×256 BC1 | 7 | 1 / 3 | 0.38 | 126.06 |
| `interiors_0_3` #10 | 512×512 BC1 | 8 | 3 / 3 | 0.46 | 57.94 |
| `graphics` #43 | 64×512 BC1 | 10 | 1 / 3 | 1.08 | 132.31 |
| `interiors_2_3` #468 | 512×512 BC1 | 8 | 3 / 3 | 0.60 | 113.83 |
| `maps_0_7` #3217 | 128×128 BC1 | 6 | 0 / 1 | 0.79 | 114.71 |
| `peds_1_3` #527 | 512×512 BC1 | 8 | 3 / 3 | 3.87 | 83.83 |
| `maps_1_7` #1500 | 128×256 BC4 | 6 | 1 / 2 | 0.34 | 72.08 |
| `decorations_0_3` #818 | 128×128 BC5 | 6 | 1 / 2 | 0.02 | 85.48 |
| `interiors_1_3` #507 | 256×256 BC3 | 7 | 1 / 2 | 1.21 | 51.84 |

`interiors_0_3` #19 is a partial export: the reader emits one mip, while all seven decode and
match. `graphics` #43 and `interiors_2_3` #468 are tuples outside the table; `maps_0_7` #3217
is mode 34 with `non_tail = 0`, which would leave zero standalone mips even if the mode gate
passed.

BC7 was checked on two samples with an external decoder (Pillow, not a dependency of the
tool): `terrain_4_7` #23 (512×512, 3 mips, 0.35–0.49) and `maps_4_7` #6034 (1024×1024, mode
32, 64 KiB, 0.00–0.33, control 4.44).

For `interiors_0_3` #19 the tool reports the byte layout: mip 0 (512×256, 128×64 blocks) at
24,576, mip 1 (256×128, 64×32) at 8,192, mip 2 (128×64, 32×16) at 4,096, then the 4 KiB tile
at 0 with mip 3 (64×32 px, 16×8 blocks) at origin (16,0), mip 4 (32×16 px, 8×4) at (8,8),
mip 5 (16×8 px, 4×2) at (0,12) and mip 6 (8×4 px, 2×1) at (0,8). The mip byte sizes 65,536 /
16,384 / 4,096 / 1,024 / 256 / 64 / 16 are exactly the BC1 block counts, and the exported DDS
is 87,524 bytes (148 + 87,376).

### What this unlocks

"Decompressed" is the block-linear BC payload size (`blocks × block bytes`), not the inflated
chunk size, which includes tile padding.

| | Textures | New mips | Decompressed |
|---|---:|---:|---:|
| partial exports that become complete | 195,091 | 1,065,138 | 1.69 GiB |
| compact skips that become complete | 11,570 | 35,727 | 2.91 GiB |
| **newly complete** | **206,661** | **1,100,865** | **4.59 GiB** |

The total export split before and after the 2D compact and multi-slice layout rules (zlib/raw
only; see `docs/textures_arrays.md` for the +690 complete / +4 partial multi-slice breakdown):

| | Before | After 2D compact | After + multi-slice | Change |
|---|---:|---:|---:|---:|
| complete chain | 98,603 | 305,264 | 305,954 | +207,351 |
| partial | 210,292 | 16,419 | 16,423 | −193,869 |
| skipped | 48,220 | 35,432 | 34,738 | −13,482 |
| **total** | **357,115** | **357,115** | **357,115** | |

The 34,738 remaining zlib/raw skips are exactly 33,507 2D reader BCPack + 1,223 compact
all-BCPack + 3 array all-BCPack + 5 DXGI 11 (**0** layout skips). The 16,423 partials are
1,166 compact records that still miss BCPack mips, 14,035 other partials, 1,218 compact
records converted from a skip to a partial export, and 4 partial 2D arrays. Nothing in this
section decodes BCPack.

## What BCPack is

BCPack is the Xbox Series hardware decompression codec for BCn texture data. The public GDK
enum exposes seven modes, one per BCn format (BC1–BC7); DirectStorage applies it after DEFLATE
and before the optional swizzle. It is not a replacement for BCn: it is an entropy-coded
preconditioner whose output is a standard BCn block stream. That is why a BCPack chunk has a
BCn codec id, not a separate texture format, and why the container can mix zlib and BCPack
chunks inside one texture.

## What the container says

The RDS5 index is readable without game code:

| Field | Value |
|---|---|
| magic | `RDS5` at 0x00 |
| virtual size | 56 bits at 0x08; top byte 0x16 |
| entry count | u32 at 0x04; `chunks = count − 2`, and the table's last entry is the sentinel |
| chunk table | 0x18, 8 bytes per entry, `(count − 1) * 8` bytes |
| ETPF index | `align(0x18 + (count − 1) * 8, 0x100)`; version 634 (`0x27A`) |
| texture records | a 16-byte ETPF header, then `area` bytes of records (at least 16 × textures) |
| asset keys | one u64 per record after the descriptor array; IDs are 32 bytes each |

Chunk entry `(u64 value)`:

| Field | Bits | Meaning |
|---|---:|---|
| physical offset | 0–25 | × 256; 256-byte aligned |
| codec | 26–29 | 0 raw, 1–7 BCPack BC1–BC7, 8 zlib |
| virtual offset | 30–51 | × 4096; 4 KiB aligned |
| padding | 52–63 | 12-bit byte count at the end of the physical range |

The parser reads 357,115 texture records and 1,026,833 chunks from the 57 pools with no
exceptions. The chunk table's offsets are 256-byte aligned (physical) and 4 KiB aligned
(virtual), and the virtual ranges cover every allocation the parser could lay out. The
12-bit padding on BCPack chunks is zero-filled (see the header section).

### Codec ids are the BCPack mode

The on-disk codec ids are `0x04, 0x0C, 0x10, 0x14, 0x18, 0x1C, 0x20`; shifting right by 26
normalizes them to 1, 3, 4, 5, 6, 7, 8. Across **1,026,833 chunks there is not one
format/codec mismatch**:

| codec | BCn mode | DXGI formats seen | Chunks | Packed MiB | Decoded MiB | Saving |
|---:|---|---|---:|---:|---:|---:|
| 0 | raw | 83, 98, 99 | 5 | 0.1 | 0.1 | 0.0% |
| 1 | BC1 | 71, 72 | 102,725 | 10,637.9 | 14,630.9 | 27.3% |
| 2 | BC2 | — | 0 | — | — | — |
| 3 | BC3 | 77, 78 | 35 | 7.4 | 10.1 | 26.3% |
| 4 | BC4 | 80 | 77 | 1.4 | 1.5 | 6.2% |
| 5 | BC5 | 83 | 333 | 60.6 | 74.6 | 18.8% |
| 6 | BC6H | 95, 96 | 268 | 67.8 | 79.1 | 14.2% |
| 7 | BC7 | 98, 99 | 5,167 | 1,097.3 | 1,221.5 | 10.2% |
| 8 | zlib (RFC 1950) | all | 918,223 | 47,179.3 | 128,529.3 | 63.3% |
| **total** | | | **1,026,833** | **59,051.8** | **144,547.1** | **59.1%** |

BCPack is 11.6 GiB of the 57.7 GiB chunk payload (20.1%), and BC1 is 10.4 GiB of that
(89.6%). The saving column is `1 − Σ packed / Σ decoded`; it includes tile padding, so it is
not the codec's saving on block data alone.

This mapping is the strongest evidence that the codec field is the GDK's
`DSTORAGE_BCPACK_MODE` (1 = BC1 … 7 = BC7), not a generic compression id. The field is per
chunk, so a texture can mix zlib and BCPack mips; 15,201 partial exports do.

### The descriptor `mode` is tiling, not codec

The ETPF descriptor's `mode` field (`b >> 23`) is the mip/tile family: mode 0 uses 64 KiB
blocks (`SW_64KB_D`), every other observed mode uses 4 KiB blocks (`SW_4KB_S`). It is
independent of the chunk codec. Formats 71/72 use 8-byte BC1 blocks, 74/75/77/78/83/84/95/96/
98/99 use 16-byte blocks, and 80/81 use 8-byte BC4 blocks. No BC2 (codec 2) texture exists in
this install.

### Character material texture parameters

Character drawables store material texture references as a 32-bit parameter hash immediately before
the 64-bit asset key. In the measured character corpus the texture content separates by parameter
rather than by array order:

| Parameter | Common formats | Measured role |
|---|---|---|
| `0xC8E8C282` | BC1/BC3/BC7 sRGB, some BC1/BC7 UNORM | sRGB albedo |
| `0xC6F46E79` | BC7 UNORM, BC1 UNORM | tangent-space normal map |
| `0x1E77BF4C` | BC5 UNORM, some BC1/BC7 | normal/detail normal |
| `0x739E3D0D` | BC7 UNORM, some BC1 | ORM/mask |
| `0x8ADB8480` | BC4 UNORM | mask |

The counts in the 246-record character corpus are 215 references each for `0xC6F46E79`,
`0x1E77BF4C` and `0x739E3D0D`, 164 for `0x8ADB8480`, and 64 for `0xC8E8C282`. Not every
material carries an albedo: the jacket, crop tee and some sports pieces have no `0xC8E8C282`
entry. Exporters must **not** fall back to the first texture in that case; a normal map painted
as a base colour is the usual symptom. The hashes are treated as measured corpus semantics, not
as a claim that they are globally unique across every shader.

## BCPack block structure and limits

A BCPack chunk is not a self-describing container. The container gives it a decoded (virtual)
size and a codec id; the stream itself has one 48-byte lane header and one interleaved rANS
byte stream. There is no magic, version, total length or format at offset 0.

| Layer | Bytes | Meaning |
|---|---:|---|
| lane header | 48 | 8 lanes × 6 rounds; byte `48r + l` is round `r`, lane `l` |
| lane 0 record | 6 or 7 | mode, length, 4-byte state; the 7th byte is at offset 48 for the 3-byte form |
| lane 1 record | 6 | constant lane on BC1/BC4: mode, `n−1`, 4-byte state |
| lanes 2–7 | 6+ | endpoint planes; a `0x10` mode is a 1-symbol uniform prefix |
| index lanes 8–11 | 6–7 each | 2×2 quadrant symbols, one per block; start at round 6 |
| entropy payload | rest | one shared stream; active lanes read one byte per round in lane order |
| padding | 0–255 | zero fill to the next 256-byte boundary |

The lane roles and the 6/7-byte records are the public research model (SixNG and bcpack-rs);
only lane 1 and the entropy budget are independently exact here.

The first byte of a lane (`hdr0`) is `[z:2][length form:1][flush:1][q:4]` in the public
patent/research model. The length form is the part that changes the record size:

| `hdr0` bit 5 | Header | First-fragment budget |
|---|---|---|
| 0 | `[hdr0][n−1]` | `n+1` (1–256 bytes) |
| 1 | `[hdr0][lo][hi]` | `(lo \| hi<<8) + 257` (257–65,792 bytes) |

The 4-byte rANS state follows the header, most-significant byte first. With the 3-byte form,
the state's last byte is read in round 6 after the 48-byte interleave; the observed BC1 lane-0
modes use both length forms. Later fragments are in-band: a lane that exhausts its budget with output
left returns to `hdr0` and reads a new 2/3-byte header. The public MDU tables use `M = 2^16`,
`L = 2^24`, 16 models (`q = 0..15`) and legal end states `L`, `L >> 8`, `L >> 16`.

The header is followed by a single byte stream, not per-lane streams: every lane that still
needs input reads one byte in lane order each round. Entropy on 300 sampled chunks per codec is
7.96–7.99 bits/byte. No general-purpose wrapper was found; zlib, gzip, bzip2 and LZMA reject
the first bytes, and raw-deflate attempts on random data are not a valid stream.

### Block sizes and counts

Every BCPack chunk decodes to an exact whole number of BCn blocks: 8 bytes for BC1/BC4 and 16
bytes for BC3/BC5/BC6H/BC7. The 108,605 BCPack chunks cover 17 block counts, from 256 to
524,288:

| Blocks | Chunks | Blocks | Chunks |
|---:|---:|---:|---:|
| 4,096 | 29,160 | 2,048 | 2,519 |
| 16,384 | 25,929 | 32,768 | 2,244 |
| 65,536 | 16,822 | 262,144 | 638 |
| 1,536 | 15,412 | 131,072 | 288 |
| 1,024 | 11,561 | 2,560 | 175 |
| 8,192 | 3,352 | 512 | 168 |
| 3,072 | 166 | 768 | 135 |
| 524,288 | 27 | 3,584 | 8 |
| 256 | 1 | | |

All 108,605 divisions `virtual_size / block_bytes` are exact.

### The one exact lane record

The lane-1 record is the part of the stream that is byte-exact from public facts. On BC1 and
BC4 it is `[0xde][n−1][state: 4 bytes big-endian]`; re-encoding `N` copies of the constant
symbol from `x = L` with `M = 2^16`, `K = 32` reproduces `n−1` and the 32-bit state exactly.
The full corpus:

| Codec | Chunks | Block divisor | Matches |
|---|---:|---:|---:|
| BC1 | 102,725 | 1 | 101,833 |
| BC1 | 102,725 | 2 | 268 |
| BC1 | 102,725 | 4 | 597 |
| BC1 | 102,725 | 8 | 27 |
| BC4 | 77 | 1 | 77 |

The record's block count is `N/k`; the `k > 1` cases are a subset of the 131,072-, 262,144-
and 524,288-block chunks, consistent with `k` equal streams. No BC1 or BC4 first record failed
the check. BC3 and BC5 use `0xdb` (`K = 2`), but the first lane-1 record matches the constant
model in only 7/35 and 39/333 chunks (BC5: 36 at `N`, 3 at `N/4`); BC6H and BC7 had no matching
constant lane 1 for `K` in {2, 4, 8, 16, 32}. The exact result is therefore a BC1/BC4 header
check, not a general BCPack header parser.

Padding is zero-filled: every padding run starts with zero bytes (checked `min(16, pad)` bytes
of all 108,229 padded BCPack chunks: 102,368 BC1, 35 BC3, 76 BC4, 332 BC5, 268 BC6H, 5,150
BC7). That is how the physical ranges reach the next 256-byte boundary.

Three skipped entries, with the fields the container actually exposes:

| Pool / index | Descriptor | Allocation | Chunks |
|---|---|---:|---|
| `graphics/ttdb_graphics.ttpf` #309 | 128×128 BC1_UNORM, 8 mips, mode 2 (4 KiB) | 12,288 B | 1 × codec 1: packed 7,648, pad 544 |
| `ui/ttdb_ui.ttpf` #0 | 512×512 BC1_UNORM, 1 mip, mode 0 (64 KiB) | 131,072 B | 1 × codec 1: packed 87,933, pad 899 |
| `terrain/ttdb_terrain_0_7.ttpf` #1537 | 512×1024 BC7_UNORM, 8 mips, mode 4 (4 KiB) | 700,416 B | 4 × codec 7; top chunk packed 499,734 of 524,288, pad 234 |

The first two bytes of those payloads are lane 0 and lane 1 modes: `0x1f 0xde`,
`0x35 0xde`, and `0x30 0x30`. The full-corpus counts are the useful part:

| Codec | Chunks | Byte 1 value | Count |
|---|---:|---|---:|
| 1 (BC1) | 102,725 | `0xde` | 102,725 / 102,725 |
| 3 (BC3) | 35 | `0xdb` | 35 / 35 |
| 4 (BC4) | 77 | `0xde` | 77 / 77 |
| 5 (BC5) | 333 | `0xdb` | 333 / 333 |
| 6 (BC6H) | 268 | `0x30`–`0x36` | 268 / 268 |
| 7 (BC7) | 5,167 | `0x30`–`0x33` | 4,978 / 5,167 |

The byte-1 value is constant within a codec (BC1/BC4 use `0xde`, BC3/BC5 use `0xdb`), and it
is the only fixed-offset value found. Byte 0 is often `0x30`–`0x37` on BC1, but the rest
includes `0x14`–`0x1f`, `0x5f` and `0xde`, so it is not a magic. BC6H uses only `0x30`–`0x36`;
BC7 has no constant byte 1 and its remaining 189 records use `0xdb`, `0x1c`–`0x1e` and
similar lane modes. The 256-byte dumps are not committed because they are game bytes; the
tables above and the container descriptors are the readable facts.

### Entropy framing and the pixel subset

The public MDU tables and a research lockstep lane schedule reproduce the entropy layer on the
200-chunk test set without guessing pixel semantics. The set is 200 textures whose mip 0 is
exactly one BC1 BCPack chunk and whose mip 1 is fully zlib/raw, in store order. All 200 chunks
passed the hard check:

- all 12 lanes finished with exactly `N` symbols;
- every lane ended at `L`, `L >> 8` or `L >> 16`;
- the decoder consumed exactly the stripped packed length, with no wrapper.

This is a byte-exact convergence check of the entropy layer; it does not identify the endpoint
or index semantics. A public research repository, [`bcpack-rs`](https://github.com/Ahmet-Dedeler/bcpack-rs)
(MIT OR Apache-2.0), publishes candidate BC1 endpoint and index rules. An independent
implementation of that rule, using the public MDU tables and its published codebook, on the
same 200 chunks gives (PSNR: each 2×2 mip0 quadrant against the mip1 pixel at that position):

| Subset | Chunks | Candidate BC1 vs mip 1 (PSNR) | Flat mip 1 |
|---|---:|---|---:|
| no pre-start back-reference | 40 | 30.92–45.69 dB, median 40.27 | 23.78 dB |
| exactly one pre-start reference | 44 | median 7.50 dB with a zero window | 21.81 dB |
| two or more | 116 | not decoded | — |

This is **mip consistency, not a byte-exact decode**. The 160 chunks with a pre-start reference
need the fixed 256-slot predictor window for a full decode; the research branches fit only
part of it and the full table is not published. The 40 clean chunks agree with mip 1 at 31–46
dB, which is strong evidence the candidate rule is close; this is not a held-out score, and
mip 1 is a downsampled image, not the source blocks. No plaintext BCPack/BC block pair was
found publicly. The repository's own `master` (commit `3534739`, read 2026-11-21) claims BC1
decodes at 36.6 dB median on held-out textures and says 621 of 3,276 pair textures are fully
known; the gap is the unpublished window and the unresolved codebook entries, not a
byte-exact validation. The code is not adopted here.

A decoder still needs:

1. the 256-slot pre-start predictor window (the largest blocker: 160/200 pair chunks);
2. a stable 256-entry index codebook (two fits agree on 248/256; 8 rare mirror pairs are open);
3. 3-colour block handling (the candidate table is fitted on 4-colour blocks);
4. record layouts and semantics for BC3, BC5, BC6H and BC7 (BC4 lane 1 is exact; the rest are
   not);
5. a reference plaintext pair or a published bitstream specification for a byte-exact check.

### The shipped binaries

The install folder contains 16 PE DLLs and `gameos.xvd`; the game executable is not present.
`tools/pe_inspect.py` and a plain ASCII/UTF-16 strings search find no BCPack/BCn/DStorage
import, export or exact string in any of the 16 DLLs. The only decompression strings are
Bink2's video path; `XboxBvhBuilder` contains `RANS` only inside `TRANSFORM` and a PDB path.
`gameos.xvd` has no printable `BCPack`, `DStorage` or `DirectStorage`; its `MDU`/`BCn` byte
sequences occur as isolated 3-byte sequences, not as printable identifiers. No shipped decoder
or dispatcher was found.

## Public sources and licences

| Source | Facts used | Licence / use |
|---|---|---|
| [DSTORAGE_BCPACK_MODE](https://learn.microsoft.com/en-us/gaming/gdk/docs/reference/system/dstorage/enums/dstorage_bcpack_mode) | Modes 1–7 are BC1, BC2, BC3, BC4, BC5, BC6H, BC7; Xbox Series only | [Microsoft Learn Terms of Use](https://learn.microsoft.com/en-us/legal/termsofuse): informational, personal/non-commercial use; linked and paraphrased, no text or code copied |
| [DirectStorage overview](https://learn.microsoft.com/en-us/gaming/gdk/docs/features/console/storage/directstorage/directstorage-overview) | BCPack is a custom entropy coder for BCn data; colour endpoints are separated from palette indices and coded with rANS; decompression order is DEFLATE, BCPack, swizzle | Same Microsoft Learn terms; linked and paraphrased |
| [US Patent 11,234,023](https://patents.google.com/patent/US11234023B2) | Public rANS architecture, banded exponential frequency model structure, and reference table check | Public patent disclosure; concepts only |
| [Khronos Data Format Specification (BPTC / S3TC / RGTC)](https://registry.khronos.org/DataFormat/specs/1.3/dataformat.1.3.html) | Standard BC1, BC3, BC4, BC5, BC6H and BC7 block layouts | Public specification |
| [SixNG, `formats/06-bcpack.md`](https://github.com/walkthestars/GTA6-Asset-Info/blob/main/formats/06-bcpack.md) | Independent container/header observations and the exact-end-state hard check | [GPL-3.0](https://github.com/walkthestars/GTA6-Asset-Info/blob/main/LICENSE); notes only, no code copied |
| [SixNG, `formats/05-ttpf.md`](https://github.com/walkthestars/GTA6-Asset-Info/blob/main/formats/05-ttpf.md) | Independent codec-id table and pool structure | GPL-3.0; notes only |
| [rpf9-explorer](https://github.com/breadbbbb/rpf9-explorer) | `TtpfReader` throws "An Xbox BCPack decompression backend is required."; no decoder | [MIT](https://github.com/breadbbbb/rpf9-explorer/blob/main/README.md); no code copied |
| [silicon-rans](https://github.com/microsoft/silicon-rans) | Exact MDU rANS tables (`mdu_minirans.v`) and legal end states used for the 200-chunk check | [MIT](https://github.com/microsoft/silicon-rans/blob/main/LICENSE); tables used as facts, no Verilog copied into this repository |
| [US11234023B2](https://patents.google.com/patent/US11234023B2/en) | Public MDU fragment-header, model and lane-schedule description | Public patent; linked and paraphrased, no code copied |
| [bcpack-rs](https://github.com/Ahmet-Dedeler/bcpack-rs) | Candidate BC1 entropy/endpoint/index work and the 40-chunk mip-consistency number | [MIT OR Apache-2.0](https://github.com/Ahmet-Dedeler/bcpack-rs/blob/master/LICENSE-MIT); research only, no code copied; its full predictor window is unpublished |
| [GitHub search `bcpack`](https://github.com/search?q=bcpack&type=repositories) | On 2026-11-21: SixNG (GPL-3.0 notes) and bcpack-rs (MIT/Apache research); no verified decoder | Search |
| SCARAB | No public GTA VI source found; the public SCARAB repositories are an unrelated GPL-3.0 Hollow Knight mod manager | Search on 2026-11-21; no facts used |

Microsoft's documentation gives the algorithm class and the mode enum, not the bitstream. The
public enum is documented for Xbox Series consoles; this note does not cover an Xbox One
variant. There is now a public *candidate* BC1 implementation (`bcpack-rs`), but it is not a
specification: it depends on an unpublished preset window and has no byte-exact plaintext
check. The SixNG notes still say 0% decoded, and the container and block-count facts above do
not change with the candidate.
| [rpf9-explorer](https://github.com/breadbbbb/rpf9-explorer) | The `TtpfReader` catalog checks and limitation strings measured here | [MIT](https://github.com/breadbbbb/rpf9-explorer/blob/main/README.md) |

## Clean-room BCPack bitstream specification and decoder

The five unknowns noted in the first measurement are resolved in `tools/bcpack.py` and
`tools/bcpack_tables.py` (pure Python standard library, no GDK/NDA code, no keys, no game
assets). The console-side hardware decompressor and GDK implementation are not linked,
copied or referenced; only plain retail chunk bytes were analysed against public BCn and
rANS specifications.

A clean-room decoder needs a published bitstream description or a byte-exact reference
implementation under a compatible licence. The public MDU tables and the candidate BC1 rules
cover the entropy layer, but no complete window/codebook or reference pair was found. The
public facts stop short of a block decoder:

1. **Predictor window.** The fixed 256-slot back-reference window is not published; 160 of the
   200 pair chunks above need it, and a zero window fails (median 7.50 dB).
2. **Index codebook.** Two independent fits agree on 248/256 entries, but 8 rare mirror pairs
   remain open, and 3-colour block handling is not settled.
3. **Record layout.** Lane 1 is exact; lane 0's 6/7-byte record and its in-band fragment
   headers are mapped, but varying plane and extension lanes still have no understood
   byte-length field.
4. **Lane semantics.** The BC1 candidate maps lane 0 to a back-reference and lanes 8–11 to
   2×2 quadrants; that is fitted, not byte-exact. BC3–BC7 pixel semantics are unstudied.
5. **Block order and timing.** The candidate uses Gfx10 stored order; mip consistency supports
   it, but no plaintext reference proves it. Exactly when a lane reads versus decodes within a
   round is still a hypothesis.
### 1. Entropy layer (all six active codecs)

Each BCPack chunk encodes `N = decoded_size / 8` synchronized lane elements across `nl`
interleaved 16-bit rANS lanes (`nl = 12` for BC1/BC3/BC4/BC5; `nl = 16` for BC6H/BC7).
16-byte block formats (BC3, BC5, BC6H, BC7) consume two lane elements per BC block.

Each lane reads one or more fragments until it has emitted `N` symbols:

```
hdr0 = z:2 | h3:1 | flush:1 | q:4
hdr1                               n = hdr1 + 1                (h3 = 0)
hdr1, hdr2                         n = hdr1 + 256*hdr2 + 257   (h3 = 1)
```

**What is missing, in one line:** the 256-slot pre-start predictor window, a stable 256-entry
index codebook, 3-colour handling, the varying/extension lane layouts, the BC3/5/6H/7
semantics, and a plaintext reference for a byte-exact check. The container, the entropy
framing and the BC1/BC4 constant lane are known; the pixels are not.
- Symbol width `w = default_width[lane] - z` (`1 <= w <= 10`).
- `q` (`0..15`) selects the banded exponential probability model for width `w`: band `0`
  covers symbol `0` and band `p >= 1` covers `[2^(p-1), 2^p)`, with total weight `65536`.
  `q = 0` is uniform; `q = 15` concentrates mass on symbol `0`. All 160 tables live in
  `tools/bcpack_tables.py`.
- rANS state: 32-bit unsigned integer with lower bound `L = 1 << 24`. Slot `0` is reserved
  as the end marker (symbol `0` owns cumulative range `[1, f0)`). While `state < L`, one
  byte is merged (`state = (state << 8) | byte`); while `state > L`, one symbol is decoded
  via `cf = state & 0xFFFF` and advanced via `f * (state >> 16) + cf - c`.
- `flush = 1`: terminates the fragment's rANS stream. Trailing zero bytes are omitted from
  the payload; once `n` bytes are consumed, the decoder shifts in up to 8 virtual zero
  bytes until `state >= L`, continues decoding, and requires the fragment to land on the
  exact home state `state == L`. The next fragment starts from `state = 0`.
- `flush = 0`: once `n` bytes are consumed and `state < L`, the next fragment header is
  read and decoding continues with the carried-over state under the new `(w, q)` model.
- After all lanes finish, any remaining bytes in the chunk must be zero.

### 2. Lane-array prefetch timing

Byte-to-lane interleaving is fixed by a deterministic per-cycle scheduler:

1. A lane holding a decoded symbol whose output slot is still full is output-stalled.
2. Non-stalled lanes in `PROC` advance their rANS state for any held symbol.
3. Lanes with an empty 1-byte input buffer request one byte (header or payload bytes while
   `in_rem > 0`). At most **8 bytes per cycle** are served in lane index order.
4. Any lane that requested a byte and received none freezes for the cycle (no state merge,
   no symbol release).
5. Served lanes (and lanes needing no byte) release any held symbol to their output slot,
   merge at most one byte when `state < L`, and decode the next symbol when `state > L`.
6. Once every active lane has a symbol in its output slot, one synchronized element is
   emitted and all slots clear.

### 3. Per-codec lane layouts and block rebuilders

| Codec | Format | Lanes | Default lane widths | Elements / block | Block rebuilder |
|---:|---|---:|---|---:|---|
| 1 | BC1 | 12 | `8 8 5 6 5 5 6 5 8 8 8 8` | 1 (8 B) | Implemented (`tools/bcpack.py`) |
| 3 | BC3 | 12 | `4 4 8 6 6 6 6 5 6 5 8 8` | 2 (16 B) | Implemented (`tools/bcpack.py`) |
| 4 | BC4 | 12 | `8 8 8 8 6 6 6 6 6 6 6 6` | 1 (8 B) | Implemented (`tools/bcpack.py`) |
| 5 | BC5 | 12 | `4 4 8 8 6 6 6 6 6 6 6 6` | 2 (16 B) | Implemented (`tools/bcpack.py`) |
| 6 | BC6H | 16 | `4 4 8 8 8 5 5 5 6 6 6 8 8 8 8 5` | 2 (16 B) | Entropy only (follow-up PR) |
| 7 | BC7 | 16 | `4 4 8 8 8 8 8 7 7 5 6 8 8 8 8 5` | 2 (16 B) | Entropy only (follow-up PR) |

For BC1, BC3, BC4 and BC5:

- **Lane 0** encodes a signed 8-bit backward distance `d` (zig-zag coded; split into two
  bit-interleaved 4-bit nibbles for 2-element formats BC3/BC5). `d = 0` predicts from the
  format's zero/midpoint default (`0` for RGB, `128` for 8-bit alpha/RG); `0 < d <= i`
  references previously decoded block `i - d`; `d > i` references entry `256 + i - d` of
  the format's fixed 256-entry built-in endpoint history.
- **Lane 1** is zero in all observed BC1/BC3/BC4/BC5 chunks (non-zero is rejected).
- **Endpoint lanes** store zig-zag deltas modulo each channel's bit depth (`32`/`64` for
  RGB565, `256` for 8-bit BC3 alpha / BC4 / BC5 endpoints).
- **Index lanes** store ranks into fixed codec lookup tables: 256-entry 2×2 quadrant tables
  for BC1 colour and BC3 colour, and 64-entry 3-bit index-pair tables for BC3 alpha, BC4
  and BC5.
- Blocks are decoded in the surface's native tiled memory order, so standard AMD Gfx10
  detiling (`tools/ttpf_mips.py`) applies unchanged after BCPack decompression.

### 4. Built-in codec history tables (strict vs. provisional)

The 256-entry built-in initial endpoint history tables and symbol-rank tables are fixed
codec constants shared by all DirectStorage BCPack streams, not game texture data. They
were recovered clean-room by algebraic and statistical fitting against independently
decodable blocks and mip-chain consistency on a retail copy, and stored without corpus
metadata in `tools/bcpack_tables.py`:

| Format | Confirmed history entries | Provisional history entries | Notes |
|---|---:|---:|---|
| BC1 | 248 | 7 | Entry 0 unreferenced; provisional entries (6, 9, 67, 68, 71, 74, 122) uncertain by ≤1 green step |
| BC3 | 0 | 35 | All 256 colour-quadrant and 64 alpha-pair rank entries populated; history entries provisional |
| BC4 | 0 | 55 | 64 pair entries populated; UI pool chunks reference additional unfitted history entries |
| BC5 | 3 | 211 | 64 pair entries populated; remaining history entries provisional |

`tools/bcpack.py` and `tools/ttpf_mips.py` run in **strict mode by default**: any chunk
referencing an unconfirmed or unknown built-in history entry raises
`BcpackProvisionalError`. Pass `--allow-provisional` to opt in.

### 5. Validation on the 57-pool corpus

- **Full-corpus projection (`tools/ttpf_mips.py scan` across all 57 pools, including multi-slice layouts in `docs/textures_arrays.md`):**
  - Baseline (2D compact + multi-slice layout rules, zlib/raw only): **305,954** complete, **16,423** partial, **34,738** skipped (33,507 2D BCPack + 1,223 compact BCPack + 3 array BCPack + 5 DXGI 11; 0 layout skips).
  - With BCPack BC1/BC3/BC4/BC5: **353,840** complete (**+47,886**, **99.08%** of catalog entries), **2,518** partial, **757** skipped (5 DXGI 11 + 752 remaining BC6H/BC7; 0 layout skips and 0 multi-slice skips).
- **Previously skipped examples above (`tools/ttpf_mips.py check`, strict mode):**
  - `graphics/ttdb_graphics.ttpf` #309 (128×128 BC1, 8 mips): **pass** (`max_error = 0.67`, `tail_origin_control = 102.3`).
  - `ui/ttdb_ui.ttpf` #0 (512×512 BC1, 1 mip): **pass**.
  - `terrain/ttdb_terrain_0_7.ttpf` #1537 (512×1024 BC7, 8 mips): blocked cleanly as unsupported codec 7 (deferred to the BC6H/BC7 follow-up PR).
- **Sampled cross-codec check:** 20/20 BC1 strict (`median_err = 0.68`), 7/7 BC3 provisional (`median_err = 1.44`, `max_err = 4.18`), 2/2 fitted BC4 provisional (`max_err = 0.43`; unfitted UI entries cleanly refused), 10/10 BC5 provisional (`median_err = 0.17`, `max_err = 0.85`), plus 5/5 BC6H and 10/10 BC7 chunks passing exact rANS entropy termination.

## Next step

1. **BC6H and BC7 block rebuilders: in review (#576; ASSET-38 in the tracker, ASSET-5 in the older list).**
   The rANS entropy layer already decodes both 16-lane formats; the BC6H/BC7 block rebuilders and history
   tables in #576 cover the remaining 5,435 BCPack chunks (752 skipped + remaining partial textures). They
   are not on `main` until #576 merges, so this step stays open until then. (Step 3 is separate: #576 does
   not add the BC6H/BC7 pixel decoders to `ttpf_mips.py check`.)
2. **Port the BCPack decoder to `TtpfReader` (`rpf9-explorer` / `tools/rpf9dump`; ASSET-44).**
   The compact layout rule and the multi-slice layouts are already upstream in `rpf9-explorer`;
   what is left is the BCPack decoder. For reference, the layout rule in C#: select `tile = 65536` when `((mode >> 1) & 0xF) == 0` and
   4096 otherwise, replace `ValidatedCompactProfile` with the allocation-span check, set
   `UsableMipLimit` to `FirstTail` (or `mips` when the tail tile is decodable), and wire
   BCPack behind `TtpfReader.DecodeChunk`.
3. **Decode BC6H/BC7 in `tools/ttpf_mips.py` `check`.** `tools/ttpf_mips.py` currently
   decodes BC1/3/4/5 pixels for the downsample check; adding BC6H/BC7 pixel decoders extends
   chain-error validation to those two formats.
4. **Add DXGI 11 (`R16G16B16A16_UNORM`, 5 records) and validate 2D uncompressed tiling (formats
   11, 28, 49, 56).** The 19 records formerly listed as DXGI 156/199/211/226 are resolved as
   two-slice 2D arrays (`docs/textures_arrays.md`).
5. **Update the `rpf9-explorer` submodule pin (BUILD-101).** The multi-slice layouts are done: all 697
   multi-slice descriptors (127 2D arrays, 567 cubemaps, and 3 3D volumes) are validated and exported
   in `tools/tex_headers.py` and `tools/ttpf_mips.py` (`docs/textures_arrays.md`, #520), and the C#
   `TtpfReader` parity landed in `breadbbbb/rpf9-explorer` PRs #5 and #6. `rpf9dump` only gets it
   once the pin moves.
5. **Port multi-slice layouts to C# `TtpfReader`.** All 697 multi-slice descriptors (127 2D
   arrays, 567 cubemaps, and 3 3D volumes) are now validated and exported in
   `tools/tex_headers.py` and `tools/ttpf_mips.py` (`docs/textures_arrays.md`); upstream C#
   `TtpfReader` parity has landed in `breadbbbb/rpf9-explorer` PRs #5 and #6 and is carried by
   the `917e9b9` submodule pin.
6. **Do not fake a BCPack decoder.** The current fallback (export the largest lower mip that
   is zlib/raw) is used by 5,845 textures and should stay. The public BC1 candidate is not
   adopted: it is mip-consistent on 40/200 chunks, not byte-exact, and its window is
   unpublished. If a complete window/codebook and a plaintext check appear, add the decoder
   behind `TtpfReader.DecodeChunk`; the container already carries the mode (codec id) and
   decoded size, so no index work is needed.
7. **If someone resumes the BCPack reverse engineering,** use the hard check the SixNG notes
   propose: every lane must consume exactly its byte budget and finish at the rANS lower
   bound, with a fixed symbol count. The entropy half now passes 200/200 chunks; the pixel
   half still needs reference blocks, and none are public.

## Reproduce

`tools/ttpf_mips.py` is the read-only measurement and decode tool (with `tools/bcpack.py`
enabled in strict confirmed-history mode by default):

```
python -I tools/ttpf_mips.py scan <game root> --json scan.json [--tuples]
python -I tools/ttpf_mips.py layout <pool.ttpf> <index>
python -I tools/ttpf_mips.py check <pool.ttpf> <index> [--allow-provisional]
python -I tools/ttpf_mips.py export <pool.ttpf> <index> <out.dds> [--allow-provisional]
python -I tests/test_bcpack.py
python -I tests/test_tex_headers.py
python -I tests/test_ttpf_mips.py
```

`scan` reproduces the 357,115 / 98,603 / 210,292 / 48,220 baseline split, reports both the
zlib/raw layout projection (`after_full` / `after_partial` / `after_skip`) and the BCPack
BC1/BC3/BC4/BC5 projection (`after_bcpack_full` / `after_bcpack_partial` /
`after_bcpack_skip`), and adds the layout and recovery counters; `--tuples` adds the 84
known-size rejected tuples. `check` reports the per-mip downsample error. `export` writes a
full BC chain when every source mip uses a supported codec. None of the tests need game
files: `tests/test_bcpack.py` builds synthetic rANS BCPack streams from scratch,
`tests/test_tex_headers.py` tests the layout rule on synthetic headers, and
`tests/test_ttpf_mips.py` builds synthetic zlib and BCPack RDS5 pools.

The BCPack counts above come from a read-only aggregation over the same `Pool` parser: it
mmaps each pool, reads the RDS5 header, chunk table and ETPF records, then reads the first 64
bytes of each BCPack chunk. The 200-chunk entropy check used the public MDU tables from
`silicon-rans` on the same mip-pair selection. The probes are not committed because they
duplicate `Pool` and add no decoder; they write no texture data.

`rpf9dump ttpf` writes one `manifest.json` per pool with a `Limitation` field for every
catalog record, and its summary line has the same `ok` / `skip` totals. The first split above
came from a read-only mirror of `TtpfReader`'s catalog checks: it mmapped each pool and read
only the 24-byte RDS5 header, the chunk table, the ETPF records and the chunk ranges, and it
wrote no texture data. `tools/ttpf_mips.py scan` is that mirror, now committed with the
compact-layout and BCPack work; it reproduces the baseline split and adds the recovery
columns.
