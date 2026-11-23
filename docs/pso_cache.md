# PSO/shader cache

One cache blob was found in the package data, stored as a single resource in the data area of a top-level RPF9
archive. It is a chain of 4,205 adjacent zlib streams: 4,204 full 64 KiB streams and a 27,436-byte last one, with
31,437 bytes of zero padding between the streams. Concatenating the inflated bytes in archive order gives
**275,540,780 bytes** (0x106C6B2C); the first stream starts at archive offset 0x10579000 and the chain is the last
resource in the archive. This page describes that file; `tools/psocache_inspect.py` checks a carved copy of it.

The blob is little-endian. In the tables, *measured* means the field's value and the region arithmetic were checked
on this file. *Unproven* means the field is only located and described, not decoded.

## Header

The header is 0x80 bytes; the first 0x70 record of the next region starts exactly at 0x80.

| Offset | Size | Value measured | Status |
|---:|---:|---:|---|
| 0x00 | 4 | `50 53 00 09` | magic `PS\0\x09`, measured |
| 0x04 | 10 | NUL-terminated ASCII string | meaning unproven |
| 0x24 | 11 | NUL-terminated ASCII string | meaning unproven |
| 0x48 | 8 | 8 bytes | 8-byte value, meaning unproven |
| 0x50 | 4 | 25,471 | count of the 0x70-byte records, measured; the "graphics PSO" role is inferred from the record shape, unproven |
| 0x54 | 4 | 2,693 | count of the 0x18-byte records, measured; the "compute PSO" role is likewise inferred, unproven |
| 0x58 | 4 | 54,576 | meta-index entries, measured (54,576 × 8 ends where the next region starts) |
| 0x5c | 4 | 21,522 | shader-index entries, measured |
| 0x60 | 4 | 31,120,020 | meta blob size, measured (the meta index covers exactly this) |
| 0x64 | 4 | 240,894,464 | code blob size, measured (the last shader entry ends 24 bytes before it) |
| 0x68 | 0x18 | zero | unused in this file |

Two cross-checks tie the counts to the records. The u32 fields at +0x08, +0x0C and +0x50..+0x5C of the 0x70 records
are 0xFFFFFFFF or below 54,576 (50,942 values that are not 0xFFFFFFFF, maximum 54,441). All 10,772 fields at
+0x08..+0x14 of the 0x18 records are below 54,576 (maximum 54,575, the last meta-index entry). A wrong count or
stride breaks both.

## Regions

| Region | Offset | Size | End |
|---|---:|---:|---:|
| header | 0x0 | 0x80 | 0x80 |
| 0x70 records, 25,471 | 0x80 | 2,852,752 | 0x2B8810 |
| 0x18 records, 2,693 | 0x2B8810 | 64,632 | 0x2C8488 |
| meta index, 54,576 × 8 | 0x2C8488 | 436,608 | 0x332E08 |
| shader index, 21,522 × 8 | 0x332E08 | 172,176 | 0x35CE98 |
| meta blob | 0x35CE98 | 31,120,020 | 0x210A92C |
| code blob | 0x210A92C | 240,894,464 | 0x106C6B2C |

The regions are laid end to end and add up to the file size exactly:

```
0x80 + 25,471 * 0x70 + 2,693 * 0x18 + 54,576 * 8 + 21,522 * 8 + 31,120,020 + 240,894,464
   = 128 + 2,852,752 + 64,632 + 436,608 + 172,176 + 31,120,020 + 240,894,464
   = 275,540,780 = 0x106C6B2C
```

## Indexes

Both tables are arrays of `(u32 offset, u32 size)` pairs, 8 bytes per entry, relative to their blob.

- Meta index: 54,576 entries over the meta blob. The first offset is 0 and each entry starts where the previous one
  ends; the last ends at 31,120,020, so there is no gap, overlap or entry past the blob.
- Shader index: 21,522 entries over the code blob, offsets strictly ascending and all distinct. The largest
  `offset + size` is 0xE5BC1E8, 24 bytes before the declared code size 0xE5BC200.

A shader-index entry is not one program. Counting `s_endpgm` inside each entry's range gives 16,992 entries with one,
4,508 with two, 21 with three or more, and one with none. The code blob holds 26,092 `s_endpgm` (0xBF810000) and
1,070,769 `s_code_end` (0xBF9F0000), and 21,463 of the 21,522 entries end on an `s_code_end` dword (58 end on a zero
dword and one on another word). The run after the last `s_endpgm` is filler, so `s_code_end` is not a program
terminator and the `s_endpgm` count does not equal the shader count. A whole-file marker scan also finds two
`s_endpgm` words outside the code blob (one in the meta blob) and no `s_code_end`, so the marker count has to be
bounded to the code region.

## Meta blob

The blob spans [0x35CE98, 0x210A92C) and is 31,120,020 bytes. A meta-index entry starts with a short header and, in
20,318 of the 54,576 entries, continues with 32-byte chunks. The header length is the first multiple of 4 from 0 to
0x40 for which `(size - header) % 32 == 0` and every chunk's first dword is 0x41, 0x42 or 0x43. Measured headers:
0x10 on 17,634 entries, 0x14 on 2,684, 0x18 on 2,684 and 0x1c on 665. Another 3,349 entries fit that shape with zero
chunks. The remaining 30,909 entries do not parse as chunk lists at all; their layout is not decoded here.

| Chunk | Count | 32-byte body |
|---|---:|---|
| 0x41 | 402,328 | first dword 0x41, then seven dwords |
| 0x42 | 371,387 | first dword 0x42, then seven dwords |
| 0x43 | 20,318 | first dword 0x43, a 16-byte value, then 12 zero bytes (all 20,318) |

The second dword of a 0x41 chunk takes six measured values. An external layout note names the first four; that
mapping is not proven here.

| Value | Count | Meaning |
|---:|---:|---|
| 0x300 | 289,590 | named SRV by the external note, unproven |
| 0x302 | 46,186 | named CBV by the external note, unproven |
| 0x303 | 11,328 | named UAV by the external note, unproven |
| 0x002 | 13,897 | named sampler by the external note, unproven |
| 0x401 | 33,596 | no name |
| 0x100 | 7,731 | no name |

The third dword of a 0x42 chunk takes 26 distinct values shaped like 0x000SSIIRR (0x00070001, 0x000D0201 and so on),
which is consistent with space/index/register slots, but the fields are not decoded.

## What remains unknown

- The meaning of the two header strings, the 0x48 value and the six 0x41 second-dword values. The SRV/CBV/UAV/sampler
  names are the external note's, not this file's.
- The encoding of the 0x41 and 0x42 payload dwords and of the 8-byte values in the 0x70 records. The 0x43 16-byte
  value was not matched against any hash in the package.
- The 30,909 meta entries that are not chunk lists.
- Whether the code blob is raw RDNA2. The evidence is the two marker words and the `s_code_end` filler; no
  disassembly or translation was run on this blob.

## Checking a carved copy

`tools/psocache_inspect.py` reads the cache file only:

```
python -I tools/psocache_inspect.py <cache>
```

It checks the magic, reads the four counts and two sizes above, bounds all six regions against the file, checks that
the meta index is contiguous over the meta blob and that every shader entry stays inside the code blob, parses the
chunk lists and counts `s_endpgm` and `s_code_end` in the code blob. It prints offsets, sizes and counts, names every
field that does not fit instead of stopping, and exits 0 when the regions add up to the file size, 1 on a
disagreement and 2 when the file is not a cache. `--json` prints the same report as JSON. It does not decode a chunk,
disassemble the code or verify a hash.
