# TTPF arrays and volumes

Measured on 2026-11-21 from the same retail Xbox Series install (1.0.2.1) and the same 57
RDS5 pools as `docs/textures.md`. The scan read the 24-byte RDS5 header, the chunk table, the
ETPF records and the chunk ranges, then decompressed only enough zlib/raw chunks to count
zero-filled blocks. It wrote no texture data. Every number below is an aggregate, a descriptor
field or a byte offset; no texture, DDS or carved bytes are included.

This page is a measurement note, not an exporter. It adds no reader code and does not change
`tools/tex_headers.py` or `tools/ttpf_mips.py`.

## Outcome

| | Records |
|---|---:|
| TTPF descriptors scanned | 357,115 |
| array/volume descriptors (`dimension != 1` or `depth != 1`) | 678 |
| cube-shaped (`dimension = 2`, `depth = 3`, tile modes 9-12) | 567 |
| 2D arrays (`dimension = 1`, `depth > 1`, tile modes 0/1) | 108 |
| 3D volumes (`dimension = 3`, `depth = 4/16`, tile mode 1) | 3 |
| **validated allocation-and-padding layout** | **90** |
| **still without a validated slice layout** | **588** |

The 90 validated records are all `dimension = 2, depth = 3, mode = 18` (tile mode 9). Their
allocation is six copies of the 2D compact 4 KiB mip chain, and the zero-filled padding in a
16x16 sample lands on the AMD mip-tail offsets. The other 588 are measured below but have no
validated slice order, so the reader's `array/volume descriptor` skip remains correct for them.

The 111 tile-mode-0/1 records are not all two-slice: 99 have `depth = 2`, nine have
`depth = 4..32` and three are `dimension = 3`. The depth field alone therefore does not reduce
the non-cube group to a two-slice case.

## Descriptor fields that select these records

The 16-byte ETPF record is decoded by `parse_etpf_texture_record` in `tools/tex_headers.py`.
The fields that matter here:

| Field | Bits | Measured meaning |
|---|---|---|
| width | `a & 0x3FFF`, +1 | top-mip width |
| height | `(a >> 14) & 0x3FFF`, +1 | top-mip height |
| packed mips | `a >> 28` | packed-tail part of the mip count |
| non-tail | `b & 7` | mip-tail selector; the reader's `NonTailMipCount` for compact 2D |
| format | `(b >> 3) & 255` | DXGI id plus a high bit; see below |
| depth | `((b >> 11) & 1023) + 1` | slice/depth field |
| dimension | `(b >> 21) & 3` | 1 = 2D/array, 2 = cube-shaped, 3 = volume-shaped (observed) |
| mode | `b >> 23` | low flag bit plus the 4-bit Gfx10 tile mode at bit 1 |
| base | `(c & 0x1FFFFF) * 4096` | allocation offset, 4 KiB aligned |

`gfx10_tile_mode(mode) = (mode >> 1) & 0xF`. The observed tile modes split cleanly:

| `(mode >> 1) & 0xF` | Mode value | Shape | Records |
|---:|---:|---|---:|
| 0 | 1 | 2D array (`dimension = 1`, `depth 2..32`) | 108 |
| 1 | 2 | 3D volume (`dimension = 3`, `depth 4/16`) | 3 |
| 9 | 18 | cube-shaped | 90 |
| 10 | 20 | cube-shaped | 75 |
| 11 | 22 | cube-shaped | 122 |
| 12 | 24 | cube-shaped | 280 |

All 678 allocation spans are exact `next_base - base` values: **678 exact, 0 mismatch**, there
is no descriptor base inside any span, and every base is 4 KiB aligned (34 are also 64 KiB
aligned). The span is therefore the full allocation for the descriptor, not a truncated
prefix.

## The format high bit is a second-layer flag

Every array/volume record with `format >= 128` resolves through `format & 0x7F` to a known
DXGI format:

| Raw format | Effective DXGI | Element bytes (block) | Records |
|---:|---|---:|---:|
| 138 | 10 `R16G16B16A16_FLOAT` | 8 (1) | 2 |
| 157 | 29 `R8G8B8A8_UNORM_SRGB` | 4 (1) | 1 |
| 182 | 54 `R16_FLOAT` | 2 (1) | 1 |
| 199 | 71 `BC1_UNORM` | 8 (4) | 9 |
| 200 | 72 `BC1_UNORM_SRGB` | 8 (4) | 102 |
| 211 | 83 `BC5_UNORM` | 16 (4) | 4 |
| 215 | 87 `B8G8R8A8_UNORM` | 4 (1) | 1 |
| 223 | 95 `BC6H_UF16` | 16 (4) | 95 |
| 224 | 96 `BC6H_SF16` | 16 (4) | 1 |
| 226 | 98 `BC7_UNORM` | 16 (4) | 363 |

That is 579 of 678 records. The other 99 already use low formats 28 (`4 (1)`), 71 (`8 (4)`)
and 83 (`16 (4)`). So **all 678 have a known element size**; the old "unknown element size"
label does not apply to the array/volume set.

The bit is not part of the DXGI id. It behaves like a second-layer flag:

- 19 2D descriptors outside this set have a raw format of 156, 199, 211 or 226 and
  `dimension = 1, depth = 1`. All 19 spans are exactly **two** compact chains; the 18 with
  zlib/raw data have two populated compact blocks (the 19th is BCPack).
- The 9 `dimension = 1` arrays with `depth >= 4` all have `2 * depth` compact 64 KiB chains:
  depth 4 -> 8, 8 -> 16, 16 -> 32, 24 -> 48, 32 -> 64. All 9 are byte-exact.
- The 567 cube-shaped records also carry a high-bit format, and `2 * depth = 6` matches the
  six-chain layout of tile mode 9.

The high bit therefore explains the 19 "DXGI 156/199/211/226" compact records in
`docs/textures.md`: their element size is known and their allocation is two layers. It is not,
by itself, a complete slice rule for tile modes 10-12 or for the 3D volumes.

## Exact size fits

Candidate formulas are evaluated against the exact allocation span. "Compact 4K" and
"compact 64K" are `compact_surface_layout(width, height, mips, element_bytes, block, 4096 or
65536)` from `tools/tex_headers.py`.

| Candidate | Records tested | Exact | Mismatch |
|---|---:|---:|---:|
| 6 x compact 4K, tile mode 9 | 90 | 90 | 0 |
| 6 x compact 4K, tile mode 10 | 75 | 0 | 75 |
| N x compact 4K, tile mode 10, N = 8 or 9 | 75 | 75 | 0 |
| 6 x compact 4K, tile mode 11 | 122 | 0 | 122 |
| 11 x compact 4K, tile mode 11 | 122 | 122 | 0 |
| N x compact 4K, tile mode 12 | 280 | 0 | 280 |
| `depth` x compact 64K, 2D arrays | 108 | 0 | 108 |
| 3 x compact 64K, `depth = 2` arrays | 99 | 99 | 0 |
| `2 * depth` x compact 64K, `depth >= 4` arrays | 9 | 9 | 0 |
| compact 4K, 3D volumes | 3 | 0 | 3 |

The N in the tile-mode 10 row is not predicted by a descriptor field: 64x64 records use 9
chains and 128x128 records use 8. The 11-chain tile-mode 11 row is a size fit only; two of the
eleven chains are entirely zero (see below).

## Validated subset: tile mode 9 cubes (90 records)

Tile mode 9, `mode = 18`, is the only group with a complete allocation-and-padding layout.
It is **six slices**, each a full compact 4 KiB mip chain:

| Width x height | Mips | Raw -> DXGI | Records | Slice bytes | Span |
|---|---:|---|---:|---:|---:|
| 256 x 256 | 1 | 199 -> BC1_UNORM | 8 | 32,768 | 196,608 |
| 16 x 16 | 1 | 215 -> B8G8R8A8_UNORM | 1 | 4,096 | 24,576 |
| 128 x 128 | 1 | 223 -> BC6H_UF16 | 1 | 16,384 | 98,304 |
| 256 x 256 | 1 | 223 -> BC6H_UF16 | 6 | 65,536 | 393,216 |
| 128 x 128 | 1 | 224 -> BC6H_SF16 | 1 | 16,384 | 98,304 |
| 16 x 16 | 5 | 226 -> BC7_UNORM | 2 | 4,096 | 24,576 |
| 32 x 32 | 1 | 226 -> BC7_UNORM | 2 | 4,096 | 24,576 |
| 32 x 32 | 6 | 226 -> BC7_UNORM | 4 | 4,096 | 24,576 |
| 64 x 64 | 1 | 226 -> BC7_UNORM | 9 | 4,096 | 24,576 |
| 128 x 128 | 1 | 226 -> BC7_UNORM | 19 | 16,384 | 98,304 |
| 256 x 256 | 1 | 226 -> BC7_UNORM | 34 | 65,536 | 393,216 |
| 512 x 512 | 1 | 226 -> BC7_UNORM | 3 | 262,144 | 1,572,864 |

The slice offsets are `slice * slice_bytes` for slices 0..5. Within a slice, the mip offsets
are the ordinary compact 4 KiB layout:

- Mips = 1: one top mip at offset 0, `slice_bytes` = top-mip bytes.
- Mips = 5 or 6, 16x16/32x32: all mips fit in the one 4 KiB tail tile; the offsets are the
  AMD tail entries, not separate mip allocations.
- The multi-mip tile-mode-9 records (16x16 with 5 mips, 32x32 with 6 mips) have all mips in
  the one 4 KiB tail tile. The mips = 1 records are a single top mip and have no tail. No
  tile-mode-9 record needs a standalone mip larger than the slice itself.

Evidence:

- The six-slice size is byte-exact on 90/90 allocation spans.
- All 90 records have six populated 4 KiB/16 KiB/64 KiB/256 KiB slice blocks; none of the six
  is blank.
- In `graphics` #184 (16x16 BC7, 5 mips), each 4 KiB slice contains non-zero islands starting
  at **768, 1,024, 1,280, 1,536 and 2,048 bytes**. Those are exactly the `entry` offsets the
  compact rule produces for mips 4, 3, 2, 1 and 0 (the tail is at offset 0). The rest of the
  slice is zero padding. The same island pattern repeats in all six slices.

Limits of this validation:

- It validates allocation size, slice count, slice stride and the mip-tail offsets. It does
  **not** decode BC6H or BC7 pixels, so the face order and content are not checked.
- It does not add an exporter; the 90 records remain skipped by `TtpfReader` until a code
  change adopts the rule.
- Some of the 90 are BCPack-backed; those cannot export without a BCPack decoder regardless.

## What blocks the other 588

### Tile mode 10: 75 cube-shaped records

The allocation is always an integer number of compact 4 KiB chains, but the count is 8 or 9
and no descriptor field selects it:

| Width x height | Mips | Raw -> DXGI | Records | Chains | Populated | Span |
|---|---:|---|---:|---:|---:|---:|
| 64 x 64 | 5 or 7 | 226 -> BC7_UNORM | 50 | 9 | 9 | 73,728 |
| 128 x 128 | 8 | 200 -> BC1_UNORM | 25 | 8 | 8 | 98,304 |

Every chain is populated. A six-face model would need six chains, so three extra chains
(64x64) or two extra chains (128x128) are not accounted for by a face rule; `depth = 3` does
not select 8 or 9. No slice order is proven.

### Tile mode 11: 122 cube-shaped records

All 122 have the same span, 270,336 bytes = 11 compact 4 KiB chains of 24,576 bytes:

| Width x height | Mips | Raw -> DXGI | Records | Chains | Populated | Zero | Span |
|---|---:|---|---:|---:|---:|---:|---:|
| 128 x 128 | 6 or 8 | 223 -> BC6H_UF16 | 27 | 11 | 9 | 2 | 270,336 |
| 128 x 128 | 4, 6 or 8 | 226 -> BC7_UNORM | 95 | 11 | 9 | 2 | 270,336 |

The last two chains are entirely zero in every record. Nine populated chains is still not a
cube-face count, and neither `mips` (4/6/8), `depth` (3) nor `non_tail` (2) selects 9. No slice
order is proven.

### Tile mode 12: 280 cube-shaped records

The span is never an integer multiple of the compact 4 KiB chain, and the 512x512 group has a
hard size blocker:

| Width x height | Mips | Raw -> DXGI | Records | Compact 4K chain | Span | Span / chain |
|---|---:|---|---:|---:|---:|---:|
| 256 x 256 | 5, 6, 7 or 9 | 223/226 -> BC6H/BC7 | 142 | 90,112 | 860,160 | 9.545 |
| 512 x 512 | 10 | 200 -> BC1_UNORM_SRGB | 77 | 176,128 | 1,540,096 | 8.744 |
| 512 x 512 | 10 | 223/226 -> BC6H/BC7 | 61 | 352,256 | 2,998,272 | 8.512 |

For the 61 BC6H/BC7 512x512 records, six top mips alone would need
`6 * 1,048,576 = 6,291,456` bytes, which is more than twice the measured span. A single
512x512 BC1 top mip is 524,288 bytes, so six BC1 faces would need 3,145,728 bytes, also more
than the 1,540,096-byte span. These records therefore cannot be six complete cube faces of the
effective format at the descriptor's width and height. The 256x256 group fits six top mips
(393,216 of 860,160 bytes) but does not fit an integer number of compact chains; the remaining
466,944 bytes are not explained by the compact rule.

The same size bound rules out three or more faces for the 512x512 BC6H/BC7 group: three top
mips would already be 3,145,728 bytes, above the 2,998,272-byte span. Only one or two full
faces could fit, which is not a cube. The field that would make these records consistent is
not in the 16-byte descriptor as decoded today.

### 2D arrays: 108 records

All 108 spans are integer multiples of a compact 64 KiB chain:

| `depth` | Records | Chains | Span examples |
|---:|---:|---:|---|
| 2 | 99 | 3 | 1,179,648 (98 records), 589,824 (1) |
| 4 | 2 | 8 | 1,048,576 |
| 8 | 1 | 16 | 2,097,152 |
| 16 | 2 | 32 | 4,194,304; 12,582,912 |
| 24 | 2 | 48 | 6,291,456 |
| 32 | 2 | 64 | 4,194,304; 8,388,608 |

The size rule is complete on these 108 records (3 chains for `depth = 2`, `2 * depth` for
`depth >= 4`), but it is not a slice layout. In `peds_0_3` #299 (512x512 BC5, depth 2, 3 chains), the first 393,216-byte
chain contains the compact-like tail pattern (zero runs of 1,536, 1,984, 3,840, 7,168, 12,288
and 17,920 bytes) while the second and third chains have no zero run of 1 KiB or more and
376,079 / 369,866 non-zero bytes.
The three compact-sized units are therefore not three independent slices; the first unit looks
like a shared tail and the later units look dense. The 8-chain 256x256 sample #300 shows the
same split: the first four chains carry the compact-like pattern and the last four are dense.
Without a field that identifies the shared-tail group, the slice order is unresolved.

### 3D volumes: 3 records

All three have one mip, so only the top level is allocated:

| Width x height | Depth | Raw -> DXGI | Span | Top-volume bytes | Ratio |
|---|---:|---|---:|---:|---:|
| 256 x 32 | 4 | 182 -> R16_FLOAT | 262,144 | 65,536 | 4.0 |
| 32 x 32 | 16 | 138 -> R16G16B16A16_FLOAT | 262,144 | 131,072 | 2.0 |
| 32 x 64 | 16 | 138 -> R16G16B16A16_FLOAT | 524,288 | 262,144 | 2.0 |

The two R16G16B16A16 volumes are two top-volume bytes; the R16 volume is four. This is a size
relation only. No 3D tile block, mip-depth or slice order is validated, so the volume record
stays skipped.

## Limits

- No code is added. The measurement used a copy of the merged #183 parser/helpers for analysis
  only; `tools/tex_headers.py` and `tools/ttpf_mips.py` are unchanged.
- No texture, DDS, carved or decompressed payload is committed. The zero-run evidence above is
  reported as offsets and lengths only.
- The 90-record tile-mode-9 layout is allocation- and padding-validated, not pixel-decoded.
  BC6H and BC7 remain undecoded; BCPack still blocks any record whose mips use it.
- The other 588 records remain skipped by `TtpfReader`. The exact blocker by group is: 75
  tile-mode-10 (8/9-chain count not selected), 122 tile-mode-11 (9 populated chains / 2 zero,
  not selected), 280 tile-mode-12 (non-integral chain count; 512x512 is smaller than six top
  mips), 108 2D arrays (shared-tail group not identified) and 3 volumes (no 3D tile rule).
- The 19 2D high-bit records outside the 678 are not fixed by this note either: their element
  size is now known and their two-layer size is measured, but no second-layer exporter exists.

## Reproduce

The pool list and the 357,115 / 678 split come from the same scan as `docs/textures.md`:
`python -I tools/ttpf_mips.py scan <game root>` prints `entries` and `arrays`. The per-record
checks used a local copy of the merged #183 parser and `compact_surface_layout`; the formulas
are in the tables above. No analysis script, pool, header dump or texture payload is committed.

## Next checks

1. Decode BC7 with an external decoder and test the six tile-mode-9 faces for mip-downsample
   agreement; that would upgrade the 90-record subset from padding validation to pixel
   validation.
2. Find the field that selects the 8/9 chain count in tile mode 10 and the 9 populated chains
   in tile mode 11. `meta`, `extra` and the low three `b` bits are the only unassigned
   descriptor fields. `meta` differs between the two tile-mode-10 groups (0x146 vs 0x2A6) but
   also appears on other sizes, so it is a profile tag rather than a proven count selector.
3. For tile mode 12, look for a native dump or a shader-side resource view that gives the true
   slice count. The 512x512 span rules out six full faces, so a face-based decoder cannot be
   correct for that group.
4. For the 108 2D arrays, test the hypothesis that the first compact-sized group is a shared
   mip tail and the later dense groups are per-slice top mips. A decoded zlib/raw sample can
   confirm or reject the split without BCPack.
