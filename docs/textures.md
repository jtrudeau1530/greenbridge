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
not decodable from public facts. No decoder is added; the next tractable work is the compact
packed tail, not BCPack.

This is the same 308,895 / 48,220 total already recorded in `docs/design.md`. The recount
splits the 48,220 by cause: the old "(BCPack or unmapped)" label hides 14,713 skips whose
recorded limitation is a layout/format check, not the codec, and overstates unmapped, which
is zero.

## Skip breakdown

| Cause | Textures |
|---|---:|
| BCPack blocks every usable mip (`TtpfReader`: "All source mips require…") | 33,507 |
| compact mip mapping not in the validated profile table | 14,030 |
| 2D array / volume descriptor (`dimension != 1` or `depth != 1`) | 678 |
| element format not in the parser table (DXGI 11, `R16G16B16A16_UNORM`) | 5 |
| no mapped allocation ("unmapped shard") | 0 |
| **total** | **48,220** |

The compact row is not one thing:

- 13,979 of the 14,030 carry the descriptor's auxiliary-mapping bits. 13,954 of those are
  mode 4 or 6; the other 25 are modes 5, 7, 36 and 38. The other 51 compact records are modes
  1, 32 or 34 without the aux flag.
- 11,570 of the 14,030 have no BCPack chunk anywhere. They fail because
  `ValidatedCompactProfile` has no entry for their
  `(mode, width, height, mips, nonTail, bytes)` tuple, not because of a codec.
- 19 compact records could not be mapped to a layout for the storage cross-tab. They are
  included in the 14,030 and also fail the profile table.

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
196,257 textures stop at the last standalone mip because `UsableMipLimit = NonTailMipCount`
for compact profiles, regardless of codec.

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

1. **Do not fake a BCPack decoder.** The current fallback (export the largest lower mip that
   is zlib/raw) is used by 5,845 textures and should stay.
2. **Fix the compact packed tail first.** 196,257 of the 210,292 partial exports stop at
   `NonTailMipCount`; the tail layout is documented (SixNG §5) and needs no BCPack decoder.
   This is the largest tractable texture gap.
3. **Then the small independent gaps:** 678 array/volume layouts and 5 DXGI 11 textures.
   Neither needs BCPack.
4. **If a public decoder or bitstream spec appears under a compatible licence,** add it behind
   `TtpfReader.DecodeChunk`; the container already carries the mode (codec id) and decoded
   size, so no index work is needed.
5. **If someone resumes the reverse engineering,** use the hard check the SixNG notes propose:
   every lane must consume exactly its byte budget and finish at the rANS lower bound, with a
   fixed symbol count. Reference blocks are the missing input; none are public.

## Reproduce

`rpf9dump ttpf` writes one `manifest.json` per pool with a `Limitation` field for every
catalog record, and its summary line has the same `ok` / `skip` totals. The split above comes
from a read-only mirror of `TtpfReader`'s catalog checks: it mmaps each pool and reads only
the 24-byte RDS5 header, the chunk table, the ETPF records and the chunk ranges. It writes no
texture data. The mirror is not committed because it duplicates `TtpfReader`; the counts are
the evidence.
