# BVH archives: measured composition

**Status: measured.** This is a read-only inventory of the `*_bvh4.rpf` packs that ship with the Xbox Series
package. No archive, payload, carved file or key is in the repository; the tables below are aggregate counts,
offsets and the few header bytes needed to describe the format. The packs were indexed with the fixed
`tools/rpf9_carve.py` from PR #90 (`--index-only`), whose decompressor advances by the bytes consumed instead of
the 1 MiB read size.

The groups measured were `core_bvh4.rpf` (8,297,921,520 bytes), `shared_minigame000_bvh4.rpf` (61,987,040 bytes)
and `story000_bvh4.rpf` (5,632 bytes).

## 1. Index result

`core_bvh4.rpf` was indexed in full, not bounded; the run took 843 s on the development machine.

| archive | file bytes | data-start (`u32@4`) | zlib streams | stored | inflated | zero padding | unread | readable | elapsed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `core_bvh4.rpf` | 8,297,921,520 | 65,024 | 565,596 | 7,685,366,227 | 129,929,580,001 | 577,512,352 | 34,977,917 | 92.6% | 843 s |
| `shared_minigame000_bvh4.rpf` | 61,987,040 | 1,264 | 2,560 | 59,142,787 | 970,044,012 | 2,672,073 | 170,916 | 95.4% | 3.3 s |
| `story000_bvh4.rpf` | 5,632 | 832 | 4 | 1,795 | 16,677 | 2,849 | 156 | 37.4% | 0.13 s |

"Readable" is stored bytes divided by the span from the data-start word to EOF. "Unread" is the non-zero gaps: the
encrypted table of contents and a small number of alignment bytes. The TOC is not decrypted here and no names were
recovered; the index is positional.

`core_bvh4.rpf` inflates to about 121 GiB, so the payload census below uses two levels of evidence:

* every one of the 565,596 chunks was classified from its first 0x140 inflated bytes. The carver had already
  proved that each stream inflates cleanly to its end, so the header read is sufficient;
* 270 BVH streams (522,595,968 bytes of 64-byte node area) were fully inflated for the fill census. All 2,560
  streams of `shared_minigame000_bvh4.rpf` (970 MB) were fully inflated.

## 2. Container layer (RPF9)

Verified:

* The file starts with the four bytes `39 46 50 52` (`9FPR`, the little-endian rendering of the RPF9 tag).
* `u32@4` is the offset the carver starts from; for `story000_bvh4.rpf` the first zlib stream starts exactly there.
  In the two larger packs a high-entropy table follows it before the first stream (`core_bvh4.rpf`: first stream
  at 69,648; `shared_minigame000_bvh4.rpf`: 16,400). The table is encrypted and stays opaque. No key or decryption
  work was done.
* The data area is zlib streams (`78 01/5e/9c/da`) separated by zero padding. There is no Bink 2 data in these
  three packs (`KB2*` count: 0).
* Large payloads are split into 8 MiB chunks. Ten files in `core_bvh4.rpf` start with a valid BVH header whose
  declared node range is larger than the 8 MiB chunk; the continuation chunk follows as the next stream and has no
  header of its own. A decoder must concatenate the chunks before parsing nodes. The index counts chunks, not
  files.

## 3. Payload families

| kind | streams (core) | share of streams | inflated bytes | share of inflated |
|---|---:|---:|---:|---:|
| `bvh4` complete | 271,627 | 48.02% | 129,769,988,608 | 99.877% |
| `bvh4-chunk` (first 8 MiB of a split file) | 10 | 0.002% | 83,886,080 | 0.065% |
| continuation chunk of a split file | 10 | 0.002% | 10,485,760 | 0.008% |
| `instance208` descriptor | 269,106 | 47.58% | 55,974,048 | 0.043% |
| other fixed-size descriptor | 24,832 | 4.39% | 8,770,816 | 0.007% |
| XML / content-manifest chunk | 11 | 0.002% | 474,689 | 0.000% |

`shared_minigame000_bvh4.rpf` has the same families: 1,237 `bvh4`, 1,233 `instance208`, 85 other descriptors and
5 XML chunks. `story000_bvh4.rpf` contains only 4 XML chunks (section 5).

### 3.1 BVH4 stream header

**Verified** on 271,627 complete streams and 10 chunked streams (271,637/271,637):

* 32-bit tag `0x81020006` at offset `0x100`.
* `u32@0x104` = size of the node area in 64-byte units.
* `u32@0x20` = `0x140 + 64 * size_in_64b_units` (end of the declared node area).
* `u32@0x28` = `u32@0x20 - 0x100` (a 256-byte tail table inside the node area).
* `0x140` is the first 64-byte node; the declared node area ends at `u32@0x20`.
* `0x110` holds six floats consistent with a 3D AABB (`min xyz`, `max xyz`). In `core_bvh4.rpf` all 271,627
  complete streams are ordered (`min <= max` on every axis); in `shared_minigame000_bvh4.rpf` 1,184 of 1,237 are
  ordered and 53 are not (sentinel or non-finite values).
* `0x108` is a hash-like 32-bit value; `0x10C` is a small varying word.
* `0x130`/`0x134`/`0x138` are an offset, an even byte size and a signed base. The pointed range is inside the
  stream and is often a near-monotonic `u16` sequence (274/400 sampled streams non-zero; 127/400 monotonic). This
  is consistent with a primitive/index remap table, not with a vertex buffer, but the exact role is **Open**.
* The 256-byte tail at `u32@0x28` contains `u32` tables, including groups of four references that begin with
  `0x04000000` in the sample. The node/table encoding was not decoded.

The name "BVH4" is the builder's own: `XboxBvh4Header`, `ConstructBVH4`, `RefitBVH4`, `PT_ConstructBVH4` and
`totalSizeIn64BNodes` appear in `XboxBvhBuilder.dll`. The measured field at `0x104` is a count of 64-byte units
whose product plus `0x140` lands exactly on the end pointer in 271,627/271,627 complete streams. That agrees with
the builder's naming but does not prove that `0x81020006` is the builder's header tag.

### 3.2 Instance descriptor (208 bytes)

There are exactly 269,106 in `core_bvh4.rpf`, and every one is immediately followed by a `bvh4` stream. The layout
is fixed and was identical in all 269,106 records (and in all 1,233 records of the shared pack):

| offset | contents |
|---|---|
| `0x00` | zero `u64` |
| `0x08` | `0x50000080`, zero high word |
| `0x10` | `1` |
| `0x18` | `0x500000A0` |
| `0x20` | `0x60000000` |
| `0x28` | `0x500000B0` |
| `0x30` | `0x500000C0` |
| `0x38` | `0x50000050` |
| `0x40` | 64-bit object key (235,533 distinct values in core) |
| `0x50` | 3x4 float transform (last column zero; identity in 178,867 of 269,106) |
| `0x88` | `0x00000101` |
| `0xA0` | 64-byte-aligned offset within the following chunk, equal to the declared node end plus 0, 64, 128 or 192 |
| `0xA4` | two `u16` counts, high and low; multiples of 256 and 64 in the sample |

The `0x5000xxxx` words are constant across records and point at offsets inside the same 208-byte template
(`0x50`, `0x80`, `0xa0`, `0xb0`, `0xc0`). They behave like serialized GPU virtual addresses built on a fixed
`0x50000000` base, but that reading is **speculative**. They are not per-object pointers: the same constants occur
in every record.

The 24,832 "other descriptor" streams are the same family at other sizes (160, 192, 256, 272, 368, ... bytes):
first eight bytes zero, a `0x5000xxxx` word at `0x08`, and zero at `0x0C`. Some carry a full 4x4 matrix instead of
the 3x4 transform; the larger ones (e.g. 272 and 368 bytes) contain several 8-byte keys followed by a table of
`0x5000xxxx` self-pointers. 2,528 of them are immediately followed by a BVH buffer. Their exact grouping semantics
are **Open**.

### 3.3 Fill and node occupancy

The declared node area in `core_bvh4.rpf` is 1,896,784,254 nodes (121,394,192,256 bytes), 93.4% of all inflated
bytes. A 270-stream sample covering 8,165,562 whole 64-byte nodes found:

| byte value | share of node area | whole-node count |
|---|---:|---:|
| `0x00` | 90.15% | 6,887,776 |
| `0x81` | 0.70% | 53,799 |
| `0x94` | 0.20% | 3,355 |
| `0x29` | 0.08% | 31 |
| other | 9.85% | — |

So about 90% of the declared node area is zero, and a further ~1% is one of three repeated fill values; the
populated content is roughly 12 GB. The same pattern holds in the shared pack (90.4% zero of 944,604,352 node
bytes). The fill bytes are not a container magic; they look like unpopulated/reserved node slots.

### 3.4 What is absent

Scans of the inflated payloads found no `RSC7`, `RSC8`, `BVH4`, `BVH`, `0DRP`, `PSIN`, `RBF0`, `DXBC`, `DXIL`,
`PK\x03\x04` or PNG signature. The only ASCII format markers in the data area are `<?xml` (11 chunks in core, 5 in
shared, 4 in story) and the strings inside those manifests, including the substring `bvh` in the per-tile
`*_bvh.rpf` entries. No DXBC/DXIL shader container starts any chunk in the header census.

## 4. Format family

**Verified by bytes:** the archives are prebuilt BVH4-style acceleration data, not shader payloads and not a
generic geometry container. The dominant bytes are 64-byte node units under a header whose declared node range
matches the 64-byte-unit count; the DLL that builds the runtime structures exposes `XboxBvh4Header`,
`ConstructBVH4`, `RefitBVH4`, `GetBVHCompactedSize` and the D3D12XBOX `DEHYDRATE`/`Rehydrate...` entry points.

**Plausible but not proven:** these are the offline-built, dehydrated inputs for `XboxBvhBuilder`'s rehydrate
path. The per-instance descriptor plus BVH buffer arrangement is consistent with a stream of build descriptors and
their acceleration structures. Whether the runtime rehydrates them or rebuilds from source geometry is **Open**;
`docs/d3d12_vulkan.md` §4.8/§7.2 already lists that call-path question.

**Not present as a whole:** source vertex/index meshes. The only geometry-like structure is the `u16` table at
header `0x130`, whose size is even and which is often monotonic; it is a candidate remap/primitive table. No
vertex-format header, index-format header or RSC resource container was found. If the runtime needs triangles, it
must get them from the level geometry or rebuild them; the BVH packs alone do not obviously carry them.

## 5. Cross-check of the small packs

Both small packs are ordinary RPF9 files with the same header, zlib streams and zero padding.

* `story000_bvh4.rpf` is **metadata-only**: 4 XML chunks (a `SSetupData`-shaped setup file, a content file and two
  changesets). It has no descriptors and no BVH streams. It registers per-tile `*_bvh.rpf` archives but contains
  none of their data.
* `shared_minigame000_bvh4.rpf` is **not empty**: 1,237 BVH streams and 1,233 instance records, plus 5 XML chunks.
  Its data is the same two-family layout as core. It is sparse in the same way: 90.4% of the declared node area is
  zero.
* Both use the same `0x81020006` tag, the same `0x140` node start and the same `0x5000xxxx` descriptor template.

`core_bvh4.rpf` ends with 11 XML chunks: one setup file, one content file and three copies of a changeset split
into three 64 KiB/26 KiB chunks. These list the per-tile BVH packs; they are the only readable text in the pack.

## 6. What a decoder or builder path needs

1. **Names are not available.** The RPF9 TOC is encrypted. Index by offset, or build a name map from the XML
   changesets and the tile pack convention; do not attempt decryption.
2. **Concatenate split files.** A `bvh4-chunk` (10 in core) is the first 8 MiB of a larger payload; append the
   following continuation chunks until the declared `u32@0x20` end is reached before reading nodes.
3. **Pair descriptor to buffer.** The 208-byte descriptor precedes its BVH chunk in stream order in
   269,106/269,106 cases. The other descriptor sizes account for 2,528 more BVH buffers; 3 more are preceded by
   other streams.
4. **Parse the header, then stop guessing.** `0x81020006`, `n64`, end, root, AABB and the `0x130` table can be read
   now. The 64-byte node encoding (child references, box encoding, leaf form) is still **Open** and must be
   recovered against `XboxBvh4Header`/`ConstructBVH4` or by validation, not assumed.
5. **Skip fill.** Ignore all-zero nodes and the `0x81`/`0x94`/`0x29` repeated blocks; they are not primitives.
6. **Do not feed these nodes to the AMD traversal.** The translator's `image_bvh_intersect_ray` path expects the
   AMD hardware node format (256-byte base units), while these payloads use the builder's 64-byte units. A
   converter or a rebuild is required; see `docs/d3d12_vulkan.md` §4.8.
7. **Result format.** Any decoder should emit world-space bounds plus primitive references, and validate counts
   against the header (`0x140 + 64*n == end`, ordered AABB) before use.

## 7. Tooling

`tools/bvh_index.py` (standard library only) scans an RPF9 archive or a directory of `.rpf`/`.bin` payloads and
writes JSON and/or CSV with offset, stored size, inflated size, kind, signature, node count, root offset, declared
bytes, bounds, descriptor key/transform and fill counts. It reads only headers and emits no payload bytes.

```powershell
python -I tools/bvh_index.py <archive.rpf> --out index.json --csv index.csv
python -I tools/bvh_index.py <carve-dir> --out index.json
python -I tests/test_bvh_index.py
```

The `bvh4` signature is reported only when the tag, the `0x140 + 64*n == end` formula and the `root == end - 0x100`
relation all hold; a larger payload whose declared end is past the chunk but which is a 64 KiB multiple is reported
as `bvh4-chunk`. The synthetic tests cover the tag, the formula rejection, the chunk case, the 208-byte template,
the other descriptor template, XML and XML fragments, the two-stream advance bug and CSV/JSON output.

Measured with the tool on `shared_minigame000_bvh4.rpf`: 2,560 streams, 59,142,787 stored, 970,044,012 inflated,
1,237 complete `bvh4`, 1,233 `instance208`, 85 descriptors, 5 XML. That matches the carver summary exactly.

## 8. No-sharing rule

Nothing from the packs may be committed: no `.rpf`, carved stream, decompressed buffer, XML text, tile list,
name hash, key or TOC dump. This document and the tool carry only aggregate counts, offsets, field layouts and a
handful of header bytes. The tests build their inputs by hand. Any future decoder must keep its fixtures synthetic
and write real output only to the user's private work folder.
