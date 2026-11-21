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

**Bottom line:** the gap is measurable and the codec mapping is certain, but BCPack itself is
not decodable from public facts. No BCPack decoder is added. The compact mip chain, which the
first measurement called the next tractable work, is now decoded and validated below; the
remaining gap is BCPack.

The compact layout rule moves 206,661 textures into complete exports and 1,218 more into
partial exports. After it, the totals are 305,264 complete, 16,419 partial and 35,432 skipped
(48,220 − 12,788). The complete count equals the 305,264 all-zlib/raw textures in the storage
table below, which is the consistency check: every texture with no BCPack mip now exports its
whole chain.

This is the same 308,895 / 48,220 total already recorded in `docs/design.md`. The recount
splits the 48,220 by cause: the old "(BCPack or unmapped)" label hides 14,713 skips whose
recorded limitation is a layout/format check, not the codec, and overstates unmapped, which
is zero. The compact layout rule resolves 12,788 of those 14,713; 1,925 remain layout or
format cases (1,223 compact all-BCPack, 19 compact unknown element size, 678 arrays/volumes
and 5 DXGI 11).

## Skip breakdown

| Cause | Before | After |
|---|---:|---:|
| BCPack blocks every usable mip (`TtpfReader`: "All source mips require…") | 33,507 | 33,507 |
| compact mip mapping not in the validated profile table | 14,030 | 0 |
| compact layout resolved, but every mip is BCPack | — | 1,223 |
| compact element size unknown (DXGI 156/199/211/226) | — | 19 |
| 2D array / volume descriptor (`dimension != 1` or `depth != 1`) | 678 | 678 |
| element format not in the parser table (DXGI 11, `R16G16B16A16_UNORM`) | 5 | 5 |
| no mapped allocation ("unmapped shard") | 0 | 0 |
| **total** | **48,220** | **35,432** |

The "before" compact row becomes two measured rows after the layout rule: records the rule
sizes but cannot decode because every mip is BCPack, and records whose element size is still
unknown. The 1,218 mixed compact records that become partial exports are not skips; they are
counted in the partial table below.

The compact row is not one thing:

- 13,979 of the 14,030 carry the descriptor's auxiliary-mapping bits. 13,954 of those are
  mode 4 or 6; the other 25 are modes 5, 7, 36 and 38. The other 51 compact records are modes
  1, 32 or 34 without the aux flag.
- 11,570 of the 14,030 have no BCPack chunk anywhere. They fail because
  `ValidatedCompactProfile` has no entry for their
  `(mode, width, height, mips, nonTail, bytes)` tuple, not because of a codec.
- 19 compact records have no element-size mapping. All 19 are mode 1 (64 KiB): DXGI 156 (7),
  199 (6), 211 (5) and 226 (1), at 256×256 with 7 mips (2) or 512×512 with 8 mips (17).
  They are included in the 14,030 and also fail the profile table.

The five format-11 records are 4×4, 8×4 or 128×4 `R16G16B16A16_UNORM`; the parser's
element-size table has no 8-byte entry for format 11. The 678 array/volume records are
non-2D descriptors (arrays or volumes), which the reader deliberately rejects until it has a
validated slice layout.

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

Modes 9–12 are exactly the 567 cube descriptors; the other 111 array/volume records use tile
mode 0 or 1. Bit 0 of the field is independent: modes 4/5 share tile mode 2, 6/7 share tile
mode 3, and 36/38 share tile modes 2/3. The reader's `mode == 0` test therefore gives a 4 KiB
layout to modes 1 and 32, which are 64 KiB. Fifteen mode 1/32 descriptors have a known element
size (4 mode 1, 11 mode 32); all fifteen match a 64 KiB allocation span and one (`maps_4_7`
#6034) decodes at 0.00–0.33 levels (see below). The other 19 mode 1 records are the
unknown-format ones in the table below.

### The layout is the AMD mip tail

The compact profile is the Gfx10 mip chain: standalone mips are padded to whole blocks and
stored smallest-first; a mip whose element count fits half a tile width and a tile height,
among the last 8 (4 KiB) or 12 (64 KiB) mips, goes into one shared tile at offset 0 at the
AMD entry offset (`m = max_tail - 1 - k`; `entry = m > 6 ? 16 << m : m << 8`).

On all 356,413 2D descriptors with a known element size — the 232,789 already-trusted compact
exports, the 14,011 layout-capable records the profile table rejects, and every non-compact
mode — the computed size equals `next_allocation - base`. There are **0 mismatches**. The
remaining 702 descriptors are 678 arrays/volumes and 24 formats without an element-size
mapping.

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
| **layout resolved** | **14,011** |
| format without an element-size mapping | 19 |
| **total** | **14,030** |

The rejected records contain 84 distinct six-tuples with a known element size, plus 4
unknown-size tuples shared by the 19 records above (88 total). The ten largest known-size
tuples are:

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
| complete zlib/raw chain | 11,570 |
| partial export: some readable contiguous mip range | 1,218 |
| still skipped: every mip is BCPack | 1,223 |
| still skipped: element size unknown | 19 |
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

The total export split before and after the layout rule:

| | Before | After | Change |
|---|---:|---:|---:|
| complete chain | 98,603 | 305,264 | +206,661 |
| partial | 210,292 | 16,419 | −193,873 |
| skipped | 48,220 | 35,432 | −12,788 |
| **total** | **357,115** | **357,115** | |

The 35,432 remaining skips are exactly 33,507 BCPack + 1,223 compact all-BCPack + 19 compact
unknown-size + 678 arrays/volumes + 5 DXGI 11. The 16,419 partials are 1,166 compact records
that still miss BCPack mips, 14,035 other partials, and the 1,218 compact records converted
from a skip to a partial export. Nothing here decodes BCPack.

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

## The chunk header

A BCPack chunk is not a self-describing container. A 256-byte dump of the first chunk of
each of the three entries below contains no magic, version, total length or format at offset
0. The only readable structure is the lane record area:

- The independent SixNG notes describe the first 48 bytes as eight lane records,
  column-interleaved: one byte per lane per round, six rounds. The byte-1 regularity below is
  consistent with that layout.
- The rest is entropy-coded lane data (rANS renormalization and prefetch in the SixNG
  description). Entropy on 300 sampled chunks per codec is 7.96–7.99 bits/byte. No
  general-purpose wrapper was found in the sampled chunks; zlib, gzip, bzip2 and LZMA reject
  the first bytes, and raw-deflate attempts on random data are not a valid stream.
- Padding is zero-filled: every padding run starts with zero bytes (checked `min(16, pad)`
  bytes of all 108,229 padded BCPack chunks: 102,368 BC1, 35 BC3, 76 BC4, 332 BC5, 268 BC6H,
  5,150 BC7). That is how the physical ranges reach the next 256-byte boundary.

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
table above and the container descriptors are the readable facts.

## Public sources and licences

| Source | Facts used | Licence / use |
|---|---|---|
| [DSTORAGE_BCPACK_MODE](https://learn.microsoft.com/en-us/gaming/gdk/docs/reference/system/dstorage/enums/dstorage_bcpack_mode) | Modes 1–7 are BC1, BC2, BC3, BC4, BC5, BC6H, BC7; Xbox Series only | [Microsoft Learn Terms of Use](https://learn.microsoft.com/en-us/legal/termsofuse): informational, personal/non-commercial use; linked and paraphrased, no text or code copied |
| [DirectStorage overview](https://learn.microsoft.com/en-us/gaming/gdk/docs/features/console/storage/directstorage/directstorage-overview) | BCPack is a custom entropy coder for BCn data; colour endpoints are separated from palette indices and coded with rANS; decompression order is DEFLATE, BCPack, swizzle | Same Microsoft Learn terms; linked and paraphrased |
| [SixNG, `formats/06-bcpack.md`](https://github.com/walkthestars/GTA6-Asset-Info/blob/main/formats/06-bcpack.md) | Independent notes: 0% decoded, no public specification or decoder, lane/header observations, open unknowns | [GPL-3.0](https://github.com/walkthestars/GTA6-Asset-Info/blob/main/LICENSE); notes only, no code copied |
| [SixNG, `formats/05-ttpf.md`](https://github.com/walkthestars/GTA6-Asset-Info/blob/main/formats/05-ttpf.md) | Independent codec-id table and pool structure | GPL-3.0; notes only |
| [rpf9-explorer](https://github.com/breadbbbb/rpf9-explorer) | The `TtpfReader` catalog checks and limitation strings measured here | [MIT](https://github.com/breadbbbb/rpf9-explorer/blob/main/README.md) |
| [GitHub repository search `bcpack`](https://github.com/search?q=bcpack&type=repositories) | No BCPack decoder; the only GTA VI result is the GPL-3.0 notes above | Search on 2026-11-21 |

Microsoft's documentation gives the algorithm class and the mode enum, not the bitstream.
The public enum is documented for Xbox Series consoles; this note does not cover an Xbox One
variant. No public bitstream specification and no open-source BCPack decoder were found: the
SixNG notes are documentation, and the only open code found for this format is the container
parser (`rpf9-explorer`, MIT). The SixNG notes reach the same conclusion from the data: the
rANS coder and stream layout are partly mapped, there is no decoder, and BC1 is the only mode
studied beyond its headers.

## Why there is no decoder in this change

A clean-room decoder needs a published bitstream description or a reference implementation
under a compatible licence. Neither was found publicly. The public facts stop well short of a
block decoder:

1. **Models.** The rANS frequency tables for the endpoint planes and the adaptive models for
   the index lanes are not published. Without them the symbol stream cannot be decoded.
2. **Record layout.** Varying plane and extension lanes have no understood byte-length field;
   the constant lanes do, but they are only the easy case.
3. **Lane semantics.** What lane 0 codes, how the four index lanes map to 2-bit indices, and
   how extension lanes are ordered are unknown.
4. **Block order and timing.** Whether blocks are linear or swizzled, and exactly when a lane
   reads versus decodes within a round, are not settled.
5. **Modes.** Public notes only studied BC1; BC4, BC5 and BC7 streams are unstudied beyond
   their headers.

The payload is **not** a documented BC7 or BC6H variant. BCPack decompresses *to* standard
BC1–BC7 blocks; the compressed stream is a separate entropy-coded representation. A decoder
built from synthetic blocks would test our guesses, not the format.

The console-side decompressor is hardware; the GDK implementation is not public source and
is not usable in a GPL-3.0 program. This change does not link it, copy it or work around it.
No DRM, key or protection work was done; only already-plain chunk bytes were read.

**What is missing, in one line:** the rANS models, the varying/extension lane record
layouts, lane-0/index-lane semantics, block order, and the BC4/5/6H/7 streams. The container
and the algorithm class are known; the bitstream is not.

## Next step

1. **Port the rule to `TtpfReader`.** The rule is implemented and tested in
   `tools/tex_headers.py`; in the reader, select `tile = 65536` when `((mode >> 1) & 0xF) == 0`
   and 4096 otherwise, replace `ValidatedCompactProfile` with the allocation-span check, and
   set `UsableMipLimit` to the computed `FirstTail` (or `mips` when the tail tile is
   zlib/raw). On this install that completes 206,661 textures without touching BCPack.
2. **Decode BC6H/BC7 in the validator.** `tools/ttpf_mips.py` decodes BC1/3/4/5; the BC7
   samples were checked with an external decoder. The layout is the same formula (16-byte
   elements), so this is validation breadth, not a new mapping.
3. **Identify DXGI 156/199/211/226 (19 records) and validate uncompressed tiling (formats
   11, 28, 49, 56).** These are the remaining non-BCPack blockers besides arrays.
4. **Arrays/volumes: 678 descriptors.** 567 use tile modes 9–12 and 111 are 2-slice; no
   validated slice layout exists yet.
5. **Do not fake a BCPack decoder.** The current fallback (export the largest lower mip that
   is zlib/raw) is used by 5,845 textures and should stay. If a public decoder or bitstream
   spec appears under a compatible licence, add it behind `TtpfReader.DecodeChunk`; the
   container already carries the mode (codec id) and decoded size, so no index work is
   needed.
6. **If someone resumes the BCPack reverse engineering,** use the hard check the SixNG notes
   propose: every lane must consume exactly its byte budget and finish at the rANS lower
   bound, with a fixed symbol count. Reference blocks are the missing input; none are public.

## Reproduce

`tools/ttpf_mips.py` is the read-only measurement and decode tool:

```
python -I tools/ttpf_mips.py scan <game root> --json scan.json [--tuples]
python -I tools/ttpf_mips.py layout <pool.ttpf> <index>
python -I tools/ttpf_mips.py check <pool.ttpf> <index>
python -I tools/ttpf_mips.py export <pool.ttpf> <index> <out.dds>
python -I tests/test_tex_headers.py
python -I tests/test_ttpf_mips.py
```

`scan` reproduces the 357,115 / 98,603 / 210,292 / 48,220 split of this note, reports the
projected `after_full` / `after_partial` / `after_skip` split, and adds the layout and
recovery counters; `--tuples` adds the 84 known-size rejected tuples. `check` reports the
per-mip downsample error. `export` writes a full BC chain only when every source mip is
zlib/raw. The tool writes no texture data. `tests/test_tex_headers.py` tests the layout rule
on synthetic headers and `tests/test_ttpf_mips.py` builds a synthetic RDS5 pool, so neither
test needs game files.

`rpf9dump ttpf` writes one `manifest.json` per pool with a `Limitation` field for every
catalog record, and its summary line has the same `ok` / `skip` totals. The first split above
came from a read-only mirror of `TtpfReader`'s catalog checks: it mmapped each pool and read
only the 24-byte RDS5 header, the chunk table, the ETPF records and the chunk ranges, and it
wrote no texture data. `tools/ttpf_mips.py scan` is that mirror, now committed with the
compact-layout work; it reproduces the split and adds the recovery columns.
