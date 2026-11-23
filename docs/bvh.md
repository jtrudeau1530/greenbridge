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

Counting the 10 chunked first chunks and their 10 continuation chunks with the complete buffers, 99.95% of all
inflated bytes belong to the BVH family. The archive is not a mixed asset pack.

### 3.1 BVH4 stream header

**Verified** on 271,627 complete streams and 10 chunked streams (271,637/271,637):

* The 64-byte `XboxBvh4Header` begins at payload offset `0x100`. Its `flags` word is `0x81020006` in these
  packs.
* `u32@0x20` = `0x140 + 64 * totalSizeIn64BNodes` (the declared end of the 64-byte node address space).
* `u32@0x28` = `u32@0x20 - 0x100`. These two words sit before the header and are not header fields.
* `0x110` holds six floats for `sceneBounds` (`min xyz`, `max xyz`). In `core_bvh4.rpf` all 271,627 complete
  streams are ordered (`min <= max` on every axis); in `shared_minigame000_bvh4.rpf` 1,184 of 1,237 are ordered
  and 53 are not (sentinel or non-finite values).

The field names, types and offsets match the public `XboxBvh4Header` used by the Xbox DXR traversal shaders. The
source is [`XDXRStandaloneTraverseShortStackCPU.hlsli`](https://github.com/GaijinEntertainment/DagorEngine/blob/9639942de5a8e5b01d91977f60ed82cda22d5f41/prog/gameLibs/bvh/shaders/XDXRStandaloneTraverseShortStackCPU.hlsli)
at commit `9639942de5a8e5b01d91977f60ed82cda22d5f41`: `XboxBvh4Header` at L2531 and the flag masks at L2551.

| header offset | field | type | sample value |
|---|---|---|---:|
| `0x100` | `flags` | `u32` | `0x81020006` |
| `0x104` | `totalSizeIn64BNodes` | `u32` | 9,943 |
| `0x108` | `crc32` | `u32` | `0xeab7667e` |
| `0x10c` | `numElements` | `u32` | 5,447 |
| `0x110` | `sceneBounds.min` | 3 x `f32` | `-47.416, -25.742, -5.624` |
| `0x11c` | `sceneBounds.max` | 3 x `f32` | `47.416, 25.742, 5.624` |
| `0x128` | `offsetToNodes` | `u32` | `0x59480` |
| `0x12c` | `offsetToPrimRemap` | `i32` | 0 |
| `0x130` | `rootNodePtr` | `u32` | `0x136ac` |
| `0x134` | `offsetToElements` | `u32` | `0x4280` |
| `0x138` | `offsetToParents` | `u32` | `0xffffab40` |
| `0x13c` | `compressionParams` | `u32` | `0x808c8a89` |

**Probe provenance.** The sample column is the 64-byte header at payload offset `0x100` of the complete `bvh4`
stream at archive offset `0x4050` (16,464) in `shared_minigame000_bvh4.rpf`, inflated size 655,360 bytes. The
pack is read-only and is not in the repository; only these header fields and the four root-record words quoted
below are reproduced. `tools/bvh_index.py` reports the same stream as `node_count=9943` and
`tail_offset=636416`.

The offsets are relative to the header at `0x100`, not to the start of the payload. In the sample,
`offsetToNodes` therefore addresses `0x59580` and `offsetToElements` addresses `0x4380`. The node and element
regions are not necessarily the simple `0x140` onward array that the first indexer assumed. `offsetToPrimRemap`
is source type `int32_t`, so the indexer keeps it signed. `offsetToParents` is source type `uint32_t`, so the
indexer keeps its raw value; it is a signed offset from the start of the structure (RT-151: rehydrated console BLASes
have it negative), and only the entries from the header end to `offsetToElements` are stored.

**Measured over every stream (ASSET-329):** 272,864 complete `bvh4` streams, 271,627 in `core_bvh4.rpf` and 1,237 in
`shared_minigame000_bvh4.rpf` (`story000_bvh4.rpf` has none), indexed with `tools/bvh_index.py`. Counts only.

| word | holds on |
|---|---|
| `offsetToPrimRemap` (`0x12c`) is 0 | 272,864 / 272,864 |
| `rootNodePtr` (`0x130`) is valid, and `(ptr >> 3) * 64` (64-byte units from `0x100`) lies in `[offsetToNodes, 0x40 + 64 * totalSizeIn64BNodes)` | 272,848 (types: `box16` 272,821, `box32` 27) |
| the other 16: `numElements` = 1, no node region, root is a `tri0` leaf in the element region | 16 (all in `core_bvh4.rpf`) |
| `offsetToElements` (`0x134`) and `offsetToNodes` are 128-byte aligned, and `offsetToNodes - offsetToElements` = `numElements * 64` rounded up to 128 | 272,864 |
| `offsetToParents` (`0x138`) is a multiple of 4 | 272,864 |
| `offsetToParents` is negative as `int32` | 237,347 |

So the "`u16` table at header `0x130`" of section 4 is not a table: it is these three header words read as `u16`s
(a root pointer, the element offset and the parent offset, the last usually negative; hence "often monotonic"). With `offsetToPrimRemap` 0 everywhere there is no
primitive remap table either, and the element region allocates 64 bytes per element, `numElements` of them, padded
to 128 bytes (a stride, not a record layout: section 4a describes the packed arrays inside it).
**Open:** where the parent array starts and how big it is. Neither `offsetToParents + 4 * unit` against the header end
or `offsetToElements`, nor a size of `4 *` the node or element count, holds on every stream.

The `flags` word has version `6` in its low 16 bits. It also sets `0x80000000` (bottom level), `0x00020000`
(offline) and `0x01000000` (dehydrated). It does not set `0x00010000` (compressed). The sample is therefore a
bottom-level, offline, dehydrated BVH, not a ready-to-traverse runtime tree.

The name "BVH4" is the builder's own: `XboxBvh4Header`, `ConstructBVH4`, `RefitBVH4`, `PT_ConstructBVH4` and
`totalSizeIn64BNodes` appear in `XboxBvhBuilder.dll`. The measured field at `0x104` is a count of 64-byte nodes
that does not include the 64-byte header: a structure is `64 * (totalSizeIn64BNodes + 1)` bytes, which lands exactly on
the end pointer in 271,627/271,627 complete streams (RT-151; the field was read as the whole size before). That agrees
with the builder's naming.

#### Node pointers and the runtime node records

A `rootNodePtr` or child pointer packs the node type in the low three bits and a 64-byte unit index in the high
29 bits:

```
type = ptr & 7
node_address = header + (ptr >> 3) * 64
```

The type values are `0..3` = `TRI0..TRI3`, `4` = `BOX16`, `5` = `BOX32`, `6` = `INSTANCE`, `7` =
`AABB_GEOMETRY`. `0xffffffff` is the invalid pointer; the indexer reports it as `root_node_invalid: true` with
the type name `invalid`, rather than naming it `AABB_GEOMETRY`. A `BOX32` node occupies 128 bytes, so it consumes
two 64-byte units; all other node records occupy 64 bytes.

The runtime records, which are what the traversal shaders read, are defined in the same file at L2285 (`Node16`),
L2302 (`Node32`) and L2342 (`Leaf`). The sample is dehydrated, so these source layouts are not instantiated by
it; the measured sample only confirms the node pointer and the header.

| record | size | layout |
|---|---:|---|
| `Node16` | 64 | four `u32` children at `+0`; four 12-byte half-float bounds at `+16` |
| `Node32` | 128 | four `u32` children at `+0`; four 24-byte float bounds at `+16`; four padding words at `+112` |
| `Leaf` | 64 | four `float3` vertices at `+0`; `GeometryIndexAndFlags` at `+48`; `BasePrimitiveIndex` at `+52`; `PrimitiveIndex2` at `+56`; `baryRemap` at `+60` |

The `Node16` bounds are six IEEE half values packed into three `u32` words. The shader expands them with
`XdxrDecompressBox16`; the first word holds `min.x`/`min.y`, the second `min.z`/`max.x`, and the third
`max.y`/`max.z`.

For the sample, `rootNodePtr = 0x136ac` decodes to index 9941, type `BOX16`, at payload address `0x9b640`.
That record begins `0x04000000, 0x13694, 0x1369c, 0x117b5`, so its first child is not a valid runtime
pointer. This is the dehydrated form, not a runtime `Node16`.

#### Packed dehydrated forms

The same file defines the packed records: `XboxBvh4NodePacked` at L2326, `XboxBvh4BVVDHeader` at L2334 and
`XboxBvh4LeafPacked` at L2376.

| record | size | layout |
|---|---:|---|
| `NodePacked` | 16 | four `u32` children |
| `LeafPacked` | 12 | `GeometryIndexAndFlags`, `BasePrimitiveIndex`, `PrimitiveIndex2AndBaryRemap` |
| `BVVDHeader` | 8 | `variantNodeIndex` plus a six-bit mask and two-bit child selector |

The layouts above are source definitions. The dehydrated stream is not a contiguous array of runtime `Node16`
records. Its element and box packing is measured in §4a: a CPU model that reads it reproduces the library's
rehydrated output byte for byte, and its structure holds on all 272,874 streams of the two packs. The
`BVVDHeader` table is not used by the samples there and stays **Open**. A decoder must not feed the dehydrated root
to the runtime traversal path; it has to rehydrate first.

### 3.2 Instance descriptor (208 bytes)

There are exactly 269,106 in `core_bvh4.rpf`, and every one is immediately followed by a `bvh4` stream. The layout
is fixed and was identical in all 269,106 records (and in all 1,233 records of the shared pack):

| offset | contents |
|---|---|
| `0x00` | zero `u64` |
| `0x08` | `0x50000080`, zero high word |
| `0x10` | `1`: the slot count N of §3.2a |
| `0x18` | `0x500000A0` |
| `0x20` | `0x60000000` |
| `0x28` | `0x500000B0` |
| `0x30` | `0x500000C0` |
| `0x38` | `0x50000050` |
| `0x40` | 64-bit object key (235,533 distinct values in core) |
| `0x50` | 3x4 float transform (last column zero; identity in 178,867 of 269,106): the geometry transform the BLAS was built with, §4a |
| `0x88` | `0x00000101`, the page word of the 32-byte block at `0x80` (§3.2a) |
| `0xA0` | the size of the following BLAS: its declared end rounded up to 256 bytes, so the declared node end plus 0, 64, 128 or 192 (269,106/269,106, §3.2a) |
| `0xA4` | high word of the entry at `0xA0`, undecoded; its low `u16` is a multiple of 64 (§3.2a) |

The `0x5000xxxx` words are constant across records and point at offsets inside the same 208-byte template
(`0x50`, `0x80`, `0xa0`, `0xb0`, `0xc0`). They behave like serialized virtual addresses built on a fixed
`0x50000000` base, and `0x60000000` at `0x20` like the base of a second (physical) segment. That reading is a
**hypothesis** (the RAGE resource convention); what is measured is that every such pointer resolves inside its own
stream, and §3.2a measures what the template does. They are not per-object pointers: the same constants occur in
every record.

### 3.2a Descriptor templates and grouping (ASSET-321)

**Status: measured** on all 24,832 "other descriptor" streams of `core_bvh4.rpf` and all 85 of
`shared_minigame000_bvh4.rpf`, with `tools/bvh_groups.py` (below). Counts, offsets and widths only.

All of them start with a zero `u64` and a `0x5000xxxx` word at `0x08`, as the 208-byte record does. That word
points at a 32-byte block inside the stream whose `u32` at `+8` (the "page word") is `0x101` or `0x1`; the other
28 bytes are zero. The streams fall into exactly two templates, and the page word separates them:

| template | page word | streams (core) | sizes (bytes) | count field | followed by |
|---|---|---:|---|---|---|
| instance, N = 1 (`instance208`) | `0x101` | 269,106 | 208 | N = 1 | `bvh4` 269,096, `bvh4-chunk` 10 |
| instance, N > 1 | `0x101` | 2,531 | 256 to 6,544 (9 sizes) | N = 2 to 89 | `bvh4` 2,531 |
| table | `0x1` | 22,301 | 160 to 10,096 | K = 1 to 227 | descriptor 22,183, `instance208` 118 |

2,531 + 22,301 = 24,832. N is 2 in 1,972 streams, 3 in 263, 6 in 249, 4 in 32, 5 in 8, 89 in 3, 13 in 2, and 7
and 25 in one each. Every stream with page word `0x101` is followed by a BVH stream (271,637/271,637) and no stream
with `0x1` is (0/22,301). Read with the hypothesis above, `0x101` is one virtual and one physical page and `0x1`
one virtual page only: the BVH buffer is the physical segment, and a table stream has none.

**Instance template.** The 208-byte record is its N = 1 case. The N > 1 streams are the census's "4x4 matrix"
streams: they hold N 3x4 matrices back to back. Pointers, not fixed offsets, place the regions after `0x50`; each
region is padded to 16 bytes, and matrices, entries, the two arrays and the map block tile the stream with no gap
or overlap in 2,531/2,531 N > 1 streams and 269,106/269,106 records. The order is matrices, map, entries, arrays in
269,106 + 1,972 streams and matrices, entries, arrays, map in 559.

| offset | width | contents | value range (core) |
|---|---|---|---|
| `0x00` | `u64` | zero | constant |
| `0x08` | `u64` | `0x5000xxxx` pointer to the 32-byte map block | page word `0x101` at `+8` |
| `0x10` | `u64` | N, the number of BLAS slots | 1 to 89 |
| `0x18` | `u64` | pointer to N `u64` entries | |
| `0x20` | `u64` | `0x60000000` | constant |
| `0x28`, `0x30` | `u64` each | pointers to two N x `u64` arrays | every element zero in all 271,637 streams |
| `0x38` | `u64` | pointer to the N matrices | `0x50000050` in all 271,637 |
| `0x40` | `u64` | key | 2,469 distinct in the 2,531 N > 1 streams |
| `0x48` | `u64` | zero | constant |
| `0x50` | N x 48 | N 3x4 `f32` matrices | |
| entry | `u64` | low `u32`: BLAS size in bytes, 0 for an empty slot; high `u32`: undecoded | low `u16` of the high word a multiple of 64, 256 to 38,976, in all 275,807 non-zero entries |

**Grouping rule (decoded).** The stream after an instance descriptor holds the BLAS of its non-zero entries back
to back. The first starts at offset 0, each next one where the previous non-zero entry's low `u32` ends, and each
non-zero low `u32` equals that BLAS's declared end (`u32@0x20` of its own wrapper, §3.1) rounded up to 256 bytes.
Each BLAS found there has the full §3.1 signature at its own offset (`0x81020006` at `+0x100`,
`0x140 + 64*n == u32@+0x20`, `u32@+0x28 == u32@+0x20 - 0x100`), and the bytes after the last one are zero.

* N > 1: 2,531/2,531 descriptors pass, which ties **6,701 BLAS** in 2,531 buffers to their descriptor. 13 of the
  descriptors have empty slots (19 entries whose low `u32` is 0; in 11 of the 13 an empty entry keeps a non-zero
  high word). With the empty entries skipped they pass like the rest. All 2,531 following streams were inflated whole for this check.
* N = 1: the low `u32` of the entry at `0xA0` equals the declared end of the following BLAS rounded up to 256 in
  269,106/269,106 records (the "declared node end plus 0, 64, 128 or 192" of the table in §3.2).

Before this the 2,531 were tied to their buffers only as "followed by". The census counted 2,528 plus 3 preceded
by other streams; this walk, with `bvh_index.classify`, finds all 2,531 followed by a `bvh4`, and all 271,637 BVH
streams preceded by an instance descriptor (269,106 + 2,531). The 3 are not reproduced. What is new is what such a
buffer holds: N BLAS, not one. 4,170 of the 6,701 BLAS sit past the first header of their stream, where an index
that reads only the first `0x140` bytes of each stream does not see them. The matrices pair with the slots by
position; that each is the geometry transform of the BLAS in its slot, as §4a measured for N = 1, is a
**hypothesis** for N > 1 that needs the source geometry to test.

**Table template** (22,301 streams in core, 81 in the shared pack). Two arrays, each a pointer followed by a
`u16 count, u16 capacity` pair:

| offset | width | contents | value range (core) |
|---|---|---|---|
| `0x00` | `u64` | zero | constant |
| `0x08` | `u64` | pointer to the 32-byte map block | page word `0x1` at `+8` |
| `0x10` | `u64` | zero | constant |
| `0x18` | `u64` | 1 | constant |
| `0x20` | `u64` | pointer to K `u32` values | |
| `0x28` | `u16`, `u16` | K, capacity | count = capacity, 1 to 227 |
| `0x30` | `u64` | pointer to K item pointers | |
| `0x38` | `u16`, `u16` | K, capacity | equal to `0x28` in 22,301/22,301 |
| item | 32 bytes | a `u64` at `+0x10`; the other 24 bytes zero | both halves of the `u64` non-zero |

In all 22,301 streams the K values are strictly ascending, the K pointers address items exactly 32 bytes apart in
pointer order, and the items, the map block, the pointer array and the value array (each padded to 16 bytes) tile
`0x40` to the end of the stream. A larger stream is a larger K: the census's "8-byte keys followed by a table of
self-pointers" are the items and the pointer array. There are 130,179 items; K is 4 in 11,953 streams, 1 in 3,888,
2 in 1,953 and 16 in 846. The 22,301 streams are all distinct, the 130,179 item `u64`s are distinct across the
pack, and each of their 64 bits is set in 50% of them.

**What does not tie a table stream to anything (tested over the whole pack).** For random 32-bit values the
expected number of matches between two sets is `|A| * |B| / 2^32`; a relation only counts if it beats that.

| candidate | observed | expected by chance |
|---|---:|---:|
| a table value (104,795 distinct) equals a BLAS `crc32` (199,615 distinct) | 7 | 4.9 |
| a table value equals a half of an instance key (475,168 distinct halves) | 6 | 11.6 |
| an item `u64` equals an instance key at `0x40` (238,000 distinct) | 0 | 0 |
| a half of an item `u64` (260,357 distinct) equals a BLAS `crc32` | 17 | 12.1 |
| a half of an item `u64` equals a half of an instance key | 30 | 28.8 |
| K equals the number of instance descriptors after or before the run | 0 of 119 runs | |

A BVH stream carries no key to match: in all 271,637 the first `0x100` bytes are zero except the wrapper words at
`0x20` and `0x28`. In archive order the table streams sit in 119 runs (15 of them 277 streams long, the longest
2,152); every run follows a `bvh4` stream and is followed by an `instance208` (118) or an N > 1 descriptor (1).
That is position, not a link.

**Open:** what the table streams map (a sorted `u32` lookup to a 64-bit value; both look like hashes, and names are
only in the encrypted TOC, so that is not tested here); the high `u32` of an instance entry; what the two zero
arrays of the instance template hold when they are not zero; whether the N matrices are the per-slot geometry
transforms. None of the 22,301 table streams can be tied to a BVH buffer with what the pack shows, and by the page
word they own none.

```powershell
python -I tools/bvh_groups.py <core_bvh4.rpf> --json <temp>/groups.json   # the template, page word and buffer counts
python -I tests/test_bvh_groups.py
```

The tool walks the archive with `bvh_index`'s own decompressor and classifier, parses each descriptor, inflates the
stream after each instance descriptor and checks it against the entries. Its rows and summary hold offsets, sizes,
counts and check results, never a key, value or payload byte. The chance table, the duplicate counts, the region
orders, the run lengths and the preamble count came from scratch scripts over the same parse and are not in the
tool.

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

**Verified (§4a):** these are the offline-built, dehydrated inputs for `XboxBvhBuilder`'s rehydrate path. Fed with
their source geometry, `RehydrateRaytracingAccelerationStructureX` returns `S_OK` on them, and an independent CPU
model of the packing gives the same bytes. Whether the game's runtime calls that path or rebuilds from source
geometry is still **Open**; `docs/d3d12_vulkan.md` §4.8/§7.2 lists that call-path question.

**Not present as a whole:** source vertex/index meshes. The `u16` "table" at header `0x130` once read as a remap table is the
`rootNodePtr`/`offsetToElements`/`offsetToParents` words, and `offsetToPrimRemap` is 0 on every stream (§3.1). No
vertex-format header, index-format header or RSC resource container was found. The leaves carry primitive
references only; the triangles come from the drawables in `levels.rpf`, in the frame of the instance descriptor's
transform (§4a).

## 4a. Rehydration and source geometry (measured)

**Status: measured on 10 BLAS** of the first nested pack in `core_bvh4.rpf` (one interior cell), on Windows 11 with
no GPU involved. The tools that did it live in a fork of another project (`americas-recomp`, GPL-2.0-only,
`wildheathen/americas-recomp`, branch `feat/bvh-rehydrate-tools`); none of their code is in this repository. Only
layouts, offsets and counts are reproduced here.

**The library runs on a PC.** `RehydrateRaytracingAccelerationStructureX` in `XboxBvhBuilder.dll` reads the
source, vertex, index, destination and scratch buffers as CPU pointers (console unified memory) and makes no device
calls. Mapping memory at each buffer's GPU virtual address (`VirtualAlloc`) and copying the data there is enough for
it to return `S_OK`. It decodes standard DXGI vertex and index formats only, and it needs one geometry descriptor per
geometry index the leaves use; fewer makes it read past the array.

**The packing (completes §3.1).** A CPU model that reads the dehydrated stream as below writes the same bytes as the
library on the 8 samples whose geometry is identified (0 differing bytes; 384 to 2,512,512 bytes per structure, 2 to
21,814 leaves).

* *Elements* start at `offsetToElements`, in chunks of 2,048 leaves at a 128 KiB stride. A chunk is five arrays,
  one entry per leaf: geometry index (3 bytes), flags (1 byte), base primitive (3 bytes), second primitive (3 bytes,
  `0xffffff` for none) and a vertex-order byte. Rehydration writes one 64-byte `Leaf` per entry from the geometry:
  the triangle's vertices in the slots the vertex-order byte selects, the fourth vertex for a paired second
  primitive, `GeometryIndexAndFlags = index << 8 | flags`, and the two primitive indices.
* *Boxes* start at `offsetToNodes` and are stored by level, bottom-up. The structure ends with the level table: the
  last `u32` is the level count, preceded by one (BOX32 end, BOX16 end) pair per level. A level holds BOX32 groups
  (8 KiB per 64 nodes), then BOX16 groups (4 KiB per 64). A group starts with one word per node, `child count << 24 |
  24-bit override mask`, then every child pointer, then the override values (`f32` in BOX32, half in BOX16), one per
  mask bit. Bit `6 * slot + c` covers component `c` of a slot (`min xyz`, then `max xyz`). Rehydration recomputes each
  slot's box from the rehydrated child and then writes the stored values over the masked components. BOX16 rounds
  min down and max up; a bound past the half range becomes `+-inf` (`0x7c00`/`0xfc00`), not `+-65504`.
* *Leaf frame.* Leaf vertices are `(R * v - t) * s`, from `compressionParams`: bit 31 enables a per-axis scale
  `2^(e - 127)` with `e` the bytes 0, 1 and 2; bit 30 enables a translation per axis, the centre of `sceneBounds`
  where the bounds do not straddle zero; bits 24-29 select one of 64 rotations, `Rz * Ry * Rx` in steps of pi/8
  about each axis (index 0, the identity, on every sample).

**The packing holds on every stream (structure only).** The byte-exact match above covers 8 samples.
`tools/bvh_packing_check.py` reads every stream with the same packing, without geometry (§7). It checks that the
bytes form exactly one tree:

* the level table is in order and ends before itself;
* every group fits in the bytes its nodes occupy;
* every node has 1 to 4 children and no override bit past them;
* a box pointer names a node of its own width on a lower level;
* a TRI0 pointer names a leaf slot `offsetToElements + 64 * k` with `k < numElements`;
* nothing points to the root, and every other node and every leaf has exactly one parent;
* every vertex-order byte is a permutation;
* every stored override lies inside `sceneBounds` in the leaf frame.

Result: **272,874 of 272,874** streams pass. That is 271,627 complete streams and 10 split ones (joined) in
`core_bvh4.rpf`, plus 1,237 in `shared_minigame000_bvh4.rpf`.

In `core_bvh4.rpf`:

| | count |
|---|---:|
| box nodes | 413,539,798 |
| leaves | 1,068,494,700 |
| stored override components | 17,082,560 |
| nodes with 2 / 3 / 4 children | 54,219,680 / 63,956,971 / 295,363,147 |
| nodes with 1 child | 0 |
| child pointers to BOX16 / BOX32 / TRI0 | 48,220,482 / 365,047,695 / 1,068,494,684 |
| child pointers of any other type | 0 |
| levels per structure | 1 to 19 (mode 7) |

Roots:

* 271,594 are BOX16 and 27 are BOX32. Each is the last node of a top level that holds only it.
* The other 16 structures are a single TRI0 leaf with no box levels.
* No root stores an override. Its box is therefore always recomputed from its children, and `sceneBounds` cannot be
  compared with stored values alone.

**The instance transform.** The 3x4 at descriptor `+0x50` is applied to the geometry, `x' = M * x`, before the BLAS
is built. The samples carry the identity or an axis-aligned 90-degree rotation. Without it, most leaves of the
rotated samples fall outside `sceneBounds` (175 of 181 and 207 of 273); with it, none do. The transposed matrix does
not fit.

**Where the triangles come from.** A leaf's geometry index is a position in the geometry list of a drawable's
primary-LOD model in `levels.rpf`, counting only the geometries whose shader the BVH keeps. On the 8 BLAS whose
drawable is a single resource, a geometry is left out exactly when its shader is one of two shaders: 48 geometries,
8 distinct shaders, no exception. The two name hashes stay out of this repository (§8). They are the shader-group
hashes (first word of the shader record; shader group at drawable `+0x10`, pointer array at group `+0x10`, `u16`
count at `+0x18`, shader index per geometry in the model's `u16` array at `+0x20`) that only skipped geometries
carry, so any identified BLAS that skips a geometry lets a user recover them locally. The positions in the samples
are the RAGE9 quantized-8 form: a 32-byte origin/axis/scale header at the vertex-format position offset minus 32,
then one contiguous stream of 8-byte records; the other vertex attributes follow in their own streams, so the bytes
per vertex of the whole block are not a position stride. The leaves address triangles in primitive order of the
meshlet index buffer.

**How a match is verified.** A byte-identical library result checks the packing, not the geometry: the model and
the library agree on any input. These checks use what the dehydrated stream recorded from the true geometry:

1. every rehydrated leaf lies inside `sceneBounds` mapped through the leaf frame, and the union of the leaves equals
   it (error 0 to 2e-7 of the extent for the right geometry; 5.7e-4 to 0.5 for a wrong decode);
2. every stored box override lies inside the range of its rehydrated slot and is no looser than it, within 4 float32
   ulps (the bake rounds some overrides one or two ulps away from the rehydrated value);
3. the internal box area, as a tie-break between candidates (3.6-7.4 % of a wrong decode's area);
4. paired leaves: the builder pairs two primitives that share an edge, so in the true index buffer the two triangles
   share two vertex indices. The true geometry scores 1.00, others 0.3-0.6 once an index has 30 or more pairs. This
   test needs no positions.

With these, 8 of the 10 BLAS each have exactly one matching drawable among all candidates of their triangle-count
signature (up to 467), and the library output equals the model on all 8. The other two are interior rooms that
aggregate props from several drawables in room-local coordinates: test 4 finds three of their pieces exactly, but
placing them needs the interior's entity list (`CMloArchetypeDef`, see #1981), which the BLAS does not store.

**Open:** what the two excluded shaders are; the placement of aggregated rooms; whether the runtime rehydrates
these packs at all (above).

## 5. Cross-check of the small packs

Both small packs are ordinary RPF9 files with the same header, zlib streams and zero padding.

* `story000_bvh4.rpf` is **metadata-only**: 4 XML chunks (a `SSetupData`-shaped setup file, a content file and two
  changesets). It has no descriptors and no BVH streams. It registers per-tile `*_bvh.rpf` archives but contains
  none of their data.
* `shared_minigame000_bvh4.rpf` is **not empty**: 1,237 BVH streams, 1,233 instance records, 85 other descriptors
  and 5 XML chunks. Its data is the same two-family layout as core. It is sparse in the same way: 90.4% of the
  declared node area is zero.
* Both use the same `0x81020006` tag, the same `0x140` node start and the same `0x5000xxxx` descriptor template.

`core_bvh4.rpf` ends with 11 XML chunks: one setup file, one content file and three copies of a changeset split
into three 64 KiB/26 KiB chunks. These list the per-tile BVH packs; they are the only readable text in the pack.

## 6. What a decoder or builder path needs

1. **Names are not available.** The RPF9 TOC is encrypted. Index by offset, or build a name map from the XML
   changesets and the tile pack convention; do not attempt decryption.
2. **Concatenate split files.** A `bvh4-chunk` (10 in core) is the first 8 MiB of a larger payload; append the
   following continuation chunks until the declared `u32@0x20` end is reached before reading nodes.
3. **Pair descriptor to buffer.** The 208-byte descriptor precedes its BVH chunk in stream order in
   269,106/269,106 cases. The 2,531 instance descriptors with N > 1 precede the other 2,531 BVH streams, each of
   which holds N BLAS back to back at the offsets the descriptor's entries give (§3.2a); read all of them, not only
   the header at `0x100`. The 22,301 table descriptors own no BVH buffer.
4. **Parse the header, then stop guessing.** The exact header table, node pointer encoding and runtime
   `Node16`/`Node32`/`Leaf` records are in section 3.1, and the dehydrated element and box packing in §4a; do not
   infer a runtime child array from the `0x140` slot space.
5. **Skip fill.** Ignore all-zero nodes and the `0x81`/`0x94`/`0x29` repeated blocks; they are not primitives.
6. **Do not feed these nodes to the AMD traversal.** The translator's `image_bvh_intersect_ray` path expects the
   AMD hardware node format (256-byte base units), while these payloads use the builder's 64-byte units. The
   sample is also dehydrated, so its root is not a runtime child array. A rehydrator, converter or rebuild is
   required; see `docs/d3d12_vulkan.md` §4.8. §4a has the rehydration inputs: the drawable geometry, the instance
   transform and the packing.
7. **Result format.** Any decoder should emit world-space bounds plus primitive references, and validate the
   wrapper (`0x140 + 64*n == end`, `u32@0x28 == end - 0x100`) and ordered AABB before use.

## 7. Tooling

`tools/bvh_index.py` (standard library only) scans an RPF9 archive or a directory of `.rpf`/`.bin` payloads and
writes JSON and/or CSV with offset, stored size, inflated size, kind, signature, node count, tail offset, the
`XboxBvh4Header` fields, root pointer/index/type/invalid flag, declared bytes, bounds, descriptor key/transform
and fill counts. It reads only headers and emits no payload bytes. The row's `version` is the low 16 bits of
`flags` as an integer; `flags` keeps the full 32-bit word.

```powershell
python -I tools/bvh_index.py <archive.rpf> --out <temp>/index.json --csv <temp>/index.csv
python -I tools/bvh_index.py <carve-dir> --out <temp>/index.json
python -I tests/test_bvh_index.py
```

`--start` must be 0 or more, and `--limit` and `--max-inflate` at least 1. A value outside that, or a `--start` past
the end of the archive, stops the tool with a one-line error and exit code 2. An empty archive, or one shorter
than its 8-byte header, has no streams. In directory mode an archive that cannot be read is reported on stderr as
`skipped` and the other archives are still indexed.

The `bvh4` signature is reported only when `flags == 0x81020006`, the `0x140 + 64*n == end` formula and the
`u32@0x28 == end - 0x100` wrapper relation all hold; a larger payload whose declared end is past the chunk but
which is a 64 KiB multiple is reported as `bvh4-chunk`. The synthetic tests cover the tag, the formula rejection,
the exact header fields and root pointer/type, the chunk case, the 208-byte template, the other descriptor
template, XML and XML fragments, the two-stream advance bug and CSV/JSON output.

Measured with the tool on `shared_minigame000_bvh4.rpf`: 2,560 streams, 59,142,787 stored, 970,044,012 inflated,
1,237 complete `bvh4`, 1,233 `instance208`, 85 descriptors, 5 XML. That matches the carver summary exactly.

`tools/bvh_groups.py` (standard library only) parses the descriptor streams into the two templates of §3.2a and
checks every instance descriptor against the BVH stream after it: N BLAS at the offsets its entries give. On the
shared pack it reports 1,233 + 4 instance descriptors (1,233 + 8 BLAS tied) and 81 table streams with 317 items.

`tools/bvh_rehydrate.py` (standard library only) turns a dehydrated BLAS back into the runtime layout on the CPU,
from float3 vertices, `uint16`/`uint32` or no indices and an optional 3x4 geometry transform, following the packing
in §4a. On the 8 measured samples whose geometry is identified, its output is byte-identical to what
`XboxBvhBuilder.dll`'s `RehydrateRaytracingAccelerationStructureX` writes (384 to 2,512,512 bytes per structure). It
is the reference for checking the builder's `Box16Rehydration`, `Box32Rehydration` and `LeafRehydration` programs on
a GPU ([embedded_shaders.md](embedded_shaders.md)). Its tests are synthetic.

The output keeps the console's size convention: `64 * (totalSizeIn64BNodes + 1)` bytes from the header, which is the
`0x140 + 64 * n` end of §3.1 with the header at `0x100`. `tools/xdxr_bvh.py validate` and the GPU builder use
`64 * totalSizeIn64BNodes` for the whole structure instead, so `validate` reports a rehydrated console structure as
not covering its buffer.

```powershell
python -I tools/bvh_rehydrate.py <dehydrated.bin> --geom <vertices.f32>[,<indices.u32>] --out <rehydrated.bin>
python -I tools/bvh_rehydrate.py <bvh4 stream> --header-offset 256 --geom ... --transform m00,...,m23 --out ...
python -I tests/test_bvh_rehydrate.py
```

`tools/bvh_geometry_check.py` says whether the geometry given to a rehydration is the geometry the BLAS was built
from, using what the dehydrated stream stores (§4a): leaves inside and equal to `sceneBounds`, stored box overrides
consistent with the rehydrated slots, and paired leaves sharing an edge in the index buffer, which needs no positions.
On the 8 samples the right geometry passes every check and a wrong decode fails (175 of 181 leaves outside, or 11 of
23 overrides inconsistent).

```powershell
python -I tools/bvh_geometry_check.py <dehydrated.bin> --geom <vertices.f32>[,<indices.u32>] [--json report.json]
python -I tests/test_bvh_geometry_check.py
```

`tools/bvh_packing_check.py` checks the §4a packing on every stream of a pack, without geometry. The checks are
listed in §4a and in the tool's docstring. It scans an `.rpf` the way `tools/bvh_index.py` does, in parallel windows
(a stream belongs to the window it starts in). A `bvh4-chunk` is joined with the streams after it up to its
declared end. The joined streams are counted as `continuation`, once each, even when one starts in the next window.
The tool also takes carved stream files (`--header-offset 256` for a whole payload).

The JSON report holds offsets and counts only: stream kinds, streams checked and passed, and failures per reason
with up to 20 example offsets. It also holds summed statistics: nodes, leaves, child pointers by type, child counts,
levels, overrides, and where the root sits.

Measured on Windows with 14 workers:

| pack | time | streams passed |
|---|---:|---:|
| `core_bvh4.rpf` | 235 s | 271,637 / 271,637 |
| `shared_minigame000_bvh4.rpf` | 3 s | 1,237 / 1,237 |

The exit status is 1 when a stream fails.

```powershell
python -I tools/bvh_packing_check.py <core_bvh4.rpf> --json <temp>/packing.json
python -I tools/bvh_packing_check.py <stream.bin> --header-offset 256
python -I tests/test_bvh_packing_check.py
```

`tools/bvh_match.py` finds the drawable geometry a dehydrated BLAS was built from, on top of `rpf9dump`'s model
export rather than a second decoder. It reads the leaf signature (triangles per geometry index, a leaf that holds two
primitives counting as two), keeps every (drawable, model) of a `rpf9dump models` manifest whose per-geometry triangle
counts contain that signature as an ordered subsequence, and checks the survivors with
`tools/bvh_geometry_check.py` from the model's exported OBJ and the 3x4 at `+0x50` of the instance descriptor. The
geometries of the two shaders the BVH leaves out are filtered with `--skip-shaders`: those name hashes are game data,
as §4a says, so the rule is off unless the user passes them. A geometry whose shader index the manifest cannot resolve
keeps its place and the checks decide. Measured on the ten BLAS of the sample pack, against the first 1 GiB window of
`levels.rpf` cut at 3 GB (exported as a whole slice): 1,228 models, 1 to 374 candidates from the signature per BLAS,
and exactly one consistent candidate each. A window holds one region's drawables, so a BLAS whose drawable is
elsewhere in the archive finds no candidate in it at all, and the tool says so instead of reporting a rejection.

`rpf9dump models` and `rpf9dump textures` take `--start`/`--length` to export a window of a file. A window need not be a
valid archive - a slice of `levels.rpf` cut at an offset has no `9FPR` header - because both modes read every resource
from its own compression blocks; a sub-range is never written to the index cache, which describes a whole file.

```powershell
dotnet build tools/rpf9dump/rpf9dump.csproj -c Release
dotnet tools/rpf9dump/bin/Release/net8.0/rpf9dump.dll models <levels.rpf> <out> --start 3221225472 --length 1073741824
python -I tools/bvh_match.py <dehydrated.bin> <out>/levels/models/manifest.json --descriptor <instance208.bin> [--skip-shaders H,H]
python -I tests/test_bvh_match.py
```

The sample set §4a was measured on can be rebuilt from any copy of the package, and nothing of it enters this
repository:

1. `tools/bvh_index.py` on `core_bvh4.rpf` lists each `bvh4` stream and its descriptor row; the 3x4 at `+0x50` is
   the transform the matching needs, and a carved stream is passed to the next tools with `--header-offset 256`
   (or `0` when the structure was carved from its own header);
2. export a window of `levels.rpf` that holds that region's drawables (`rpf9dump models`, `--start`/`--length`);
3. `tools/bvh_match.py` identifies the drawable's geometries against it, and `--write <dir>` stores exactly those
   geometries as the float3 and uint32 buffers the next step takes, with the transform recorded in `export.json`;
4. `tools/bvh_rehydrate.py` on the same stream with those buffers and `--transform` gives the runtime layout,
   byte-identical to what `XboxBvhBuilder.dll` writes on these samples (§4a).

With the two excluded shader hashes the run answers with one consistent candidate per sample; without them every
candidate is checked and the checks decide. Measured on the window above: 1,228 models in the export, 1 to 374
candidates from the signature per sample, and a rehydrated structure of `64 * (totalSizeIn64BNodes + 1)` bytes from
the header. Its root parent-table entry is `0xFFFFFFFF` in all eight identified samples (the entry at
`offsetToParents` read signed, plus `4 * ((root - offsetToElements) / 64)`), which is what the GPU builder writes
for a root and what `tools/xdxr_bvh.py validate` requires since #2625.

## 7a. Building runtime BVHs: `tools/xdxr_bvh.py`

The Vulkan layer has to write acceleration structures in this runtime layout, because the game's traversal shaders
read the nodes themselves (`docs/d3d12_vulkan.md` §4.8). `tools/xdxr_bvh.py` (standard library only) is the CPU
reference for that builder. It encodes a BLAS from triangles or a TLAS from instances, runs one traversal step
with the semantics of the hardware instruction, walks a whole BVH to the closest hit, and checks a buffer's
structure. Its fixtures are synthetic.

What it writes, and where each part comes from (line numbers in the public `.hlsli` quoted in §3.1):

| Part | Layout | Source |
|---|---|---|
| Header | §3.1 table; `flags = 0x80000006` (BLAS) or `0x00000006` (TLAS); `crc32` and `compressionParams` 0 | `XboxBvh4Header` L2531, flag masks L2551, `XDXR_VERSION_2403 = 6` |
| BOX32 / BOX16 | four children, unused slots `0xffffffff` with an empty box (min > max); BOX16 rounds min down and max up to half | `Node32` L2302, `Node16` L2285, `XdxrDecompressBox16` |
| Leaf | vertex slots 0–3, then `GeometryIndexAndFlags` (index << 8 \| flags), `BasePrimitiveIndex` (TRI0), `PrimitiveIndex2` (TRI1), one remap byte per triangle node type | `Leaf` L2342, `XDXR_Runtime_Callback_GetPrimitiveIndex` |
| TRI0 / TRI1 | TRI0 reads slots (0, 1, 2), TRI1 reads (1, 3, 2); TRI2/TRI3 read slot 4, which overlaps the metadata, and are not emitted | `Read3Verts`, `XdxrGetVertexIndicesForNodeType` |
| Barycentric remap | byte `8 * type` of leaf `+60`: bits [1:0] pick which edge function is the first barycentric (0 U, 1 V, 2 W), [3:2] the second; U weighs the first vertex read | `TestXboxBvh4Leaf` |
| INSTANCE | 128 bytes: object-to-world 3×4, ID \| mask << 24, contribution \| flags << 24, BLAS root pointer with bit 31 = identity rotation, BLAS address >> 8, world-to-object 3×4, instance index, parent | `Instance` L2436, `XDXR_INSTANCE_INTERNAL_FLAGS_*` |
| Parent table | one dword per 64-byte unit from `offsetToElements` on; the entry for unit `u` is at dword `u - offsetToElements/64 + offsetToParents/4` | `XboxBvh4Node_GetParentTableOffset` |

**ASSUMPTION:** the region order (header, parent table, elements, nodes), the median-split build and the greedy
pairing of triangles that share an edge into TRI0/TRI1 are this tool's choices. The console builder's own choices
are not public. A shader reaches every region through the header, so these choices don't change what it reads.

The traversal step (`intersect_node`) follows `TestXboxBvh4Node` and `TestXboxBvh4Leaf`. A box child is kept when
its entry distance is at most the ray extent, and the children are sorted with the network (0,2) (1,3) (0,1) (2,3)
(1,2). A triangle returns `(t_num, t_denom, i_num, j_num)`, or `(inf, 1, 0, 0)` on a miss. A box miss stays a miss
for an unbounded ray: the public code's `<=` against an infinite extent would keep it.

**Translator.** `image_bvh_intersect_ray` in `tools/rdna2spirv.py` used to follow RADV's software path on triangle
nodes: slots (0, 1, 2) for every type and (V, W) as the barycentrics, because RADV builds only one triangle per
node. It now reads the slots of the node type and applies the remap byte. AMD GPURT
(`CalcTriangleCompressionVertexIndices`, `SwizzleBarycentrics`) and the XDXR path agree on both. TRI1 nodes
therefore traverse to the right triangle (docs/design.md, Ray tracing row; `tests/semantic/test_image_bvh_tri.py`).
Its four triangle words are the ones native RDNA2 returns (docs/design.md, Ray tracing row; measured on an RX 6800
with the kernels `tools/bvh_tri_probe.py` builds, `tests/semantic/test_image_bvh_tri_hw.py`): k·(T, det, b_i, b_j)
with k = |d[kz]|³ on a hit, and on a miss copysign(inf, k·det) in word 0 with the other three kept, where the step
above returns (inf, 1, 0, 0). A ray exactly on an edge or a vertex follows RDNA2's tie rule (docs/design.md, Ray
tracing row): the zero edge from P to Q passes when ty = 0 ? sign(det)·tx > 0 : sign(d[kz])·sign(det)·ty > 0, with
(tx, ty) = Q′ − P′ sheared and the d[kz] < 0 swap undone. It was read off measured x/y-dominant rows and then predicted
the z-dominant rows before they ran; the step above and RADV break those ties differently. Measured on gfx1030 only,
with dyadic inputs.

**GPU builder.** `runtime/src/xdxr_build.comp` with `xdxr_build.h` writes the same layout on the GPU, straight into
the destination buffer:

1. one thread per triangle (TRI0 leaves, remap `0x09`) or per instance (128-byte records, the world-to-object
   inverse, the BLAS root and identity-rotation flag);
2. a 30-bit Morton key of each box centre, then a stable sort: up to 1,024 keys in one workgroup's shared memory
   (one dispatch), more with a radix sort of 8 passes of 4-bit digits over 256-element tiles (24 dispatches
   whatever the count, no padding to a power of two);
3. a binary radix tree over the sorted keys (Karras 2012; equal keys are told apart by their sorted position), one
   thread per inner node;
4. the inner nodes: those at an even depth of the binary tree, each taking the children of its children (two to
   four), numbered in sorted order by a scan, so the root is node 0; with the parent table. They are BOX32, or
   BOX16 (below), and TRI1 pairing (below) happens here;
5. the boxes, bottom-up in one dispatch: each element writes its box into its parent's slot and merges it into the
   parent's accumulator; the last child to arrive carries on with the parent's box. The invocations meet only
   through atomics ordered by `memoryBarrierBuffer`, so no plain load reads another workgroup's store of the same
   dispatch. An instance's `ParentNode` is written here;
6. the header.

The tree used to group four consecutive sorted elements per node, level by level. Those groups ignore where the
Morton prefix changes, so sibling boxes overlapped and a ray took about twice as many steps (found by DaniSBR1 on
D3D-410). Measured with `xdxr_bvh.py check-build`, which counts the traversal steps (one per node or triangle
tested, the work of one `image_bvh_intersect_ray`) over 1,000 random rays per case:

| Case | Steps per ray, four consecutive | Steps per ray, radix tree |
|---|---|---|
| `blas_three_sort_tiles` (5,000 random triangles) | 42.0 | 23.0 |
| `blas_box16_sort_tiles` (the same, BOX16) | 43.9 | 23.7 |
| `blas_paired_sort_tiles` (7,200-triangle grid, TRI1 pairing) | 37.6 | 19.5 |
| `blas_random` (1,000 random triangles) | 24.1 | 16.0 |
| `blas_grid_u16_stride16` (288-triangle height grid) | 11.0 | 8.0 |
| `blas_refit_in_place` (3,000 triangles, refitted after moving) | 74.8 | 55.5 |
| `tlas` (4 instances) | 26.3 | 18.1 |

Every ray finds the same hit as brute force in both. A CPU model of the choice (a scratch script, not kept) gave
35.1 steps per ray for a greedy surface-area collapse of the same radix tree, 35.8 for the largest-subtree one and
35.7 for the even-depth one this builder uses, against 76.8 for four consecutive elements (5,000 random triangles;
11.2, 11.4, 11.5 and 24.4 on a height field). The even-depth collapse is the only one of the three that needs no
pass from the top down.

The cost is in the build and in memory. Wall clock of `vk_xdxr_build_test <dir> <n>` (random triangles, best of 3):

| GPU | Triangles | Build before → after (ms) | Refit before → after (ms) |
|---|---|---|---|
| RTX 5090 | 1,000 | 0.17 → 0.17 | 0.10 → 0.09 |
| RTX 5090 | 1,000,000 | 5.50 → 5.99 | 4.12 → 4.28 |
| Radeon iGPU (gfx1036) | 1,000 | 0.17 → 0.23 | 0.09 → 0.10 |
| Radeon iGPU (gfx1036) | 1,000,000 | 133 → 180 | 41.8 → 62.2 |

On the iGPU about half of the refit's increase is the accumulator atomics (measured before BOX16 and pairing). The tree has more nodes: about one per
two elements, against one per three. A build writes 137,856 bytes for 1,000 triangles where it wrote 113,536, and
the layout leaves room for the most a tree can have.

The radix sort's scan writes the prefix sums of the digit counts to their own scratch array, not over the counts.
No invocation of the builder stores to a word and then uses a value it loaded from that word earlier. SPIRV-Cross
`6c09849f`, in MoltenVK 1.4.2, breaks that pattern for a pointer made by `OpBitcast` from a `uvec2`
(`GL_EXT_buffer_reference_uvec2`): it moves the load to where the value is used, after the store, and the load
returns the new value. MoltenVK `main` at `5189f8de` pins SPIRV-Cross `aa217aeb`, which has the same bug.
SPIRV-Cross [`38abe362`](https://github.com/KhronosGroup/SPIRV-Cross/commit/38abe362c57311a77808b7ec4d1d0d9ceb8d9370)
fixes it. With the old in-place scan, the radix path (more than 1,024 elements) lost and duplicated elements on
MoltenVK: `xdxr_build_check` found about 130 of 200 rays wrong in `blas_three_sort_tiles` and 210 of 300 in
`blas_refit_in_place`, which refits a radix-sorted tree. Measured on an Apple M3 Max, the separate arrays pass
10 runs out of 10 on Homebrew MoltenVK 1.4.2, on MoltenVK `5189f8de` built with SPIRV-Cross `57e4efcc`, on
KosmicKrisp and on llvmpipe. A pointer from `OpConvertUToPtr`, which the translator uses, kept its load before the
store in the same test.

**BOX16 nodes.** A bottom-level build with `MINIMIZE_MEMORY` writes BOX16 inner nodes: 64 bytes, not 128, so a BLAS
is about a fifth smaller (39,040 bytes against 48,768 for the 428-triangle BLAS of `d3d12_xdxr_rt_test`). The tree
is the same; only the node records and their offsets change, and
`GetRaytracingAccelerationStructurePrebuildInfo` gives the smaller size when it gets the same flag. Each child box
is rounded outward to halves (min toward -inf, max toward +inf) from the exact box, which the builder keeps in its
accumulators, so the rounding does not add up from level to level. The rounding is integer code, because `packHalf2x16`
rounds in an implementation-defined way (AMD's driver rounds toward zero). Beyond the half range a bound becomes
+-inf on its outward side and +-65504 on its inward side, so an empty slot (min +inf, max -inf) is still empty. A
bound smaller than the smallest normal half (2^-14) in magnitude becomes 0 or +-2^-14, whichever is on its outward
side, never a half denormal: a decoder that flushes those to zero would make the box smaller. `tools/xdxr_bvh.py`
`half_round` does the same, and `check-build` compares every BOX16 word bit for bit with `encode_box16` of the exact
box of the triangles under that child. Top-level builds keep BOX32 whatever the flags, because they are in world
space, where a half's step reaches 2 m at 2,048 m and 4 m at 4,096 m. An update must pass the same flags as its
build (D3D12 requires it), so a refit keeps the node type. **ASSUMPTION:** that `MINIMIZE_MEMORY` is the console's
switch for BOX16. The offline BLASes in the `*_bvh4.rpf` packs have BOX16 roots (§3.1), but `XboxBvhBuilder.dll`'s
choice between the two at run time is not traced. The one traced datum points at a finer rule: `docs/xdxr.md` §3
finds a different bottom-level path that needs `MINIMIZE_MEMORY` and a stream setting (type 0, `+0x08`) above 10.0,
so the flag alone may not be the switch. A bottom-level structure authored in world coordinates gets looser boxes
from BOX16 the same way a top-level one would: slower to trace, never wrong. A BOX16 build ends with one spare,
zeroed 64-byte unit after the last node it wrote, and the header's size counts it: the translator's
`image_bvh_intersect_ray` loads a box node's 112 bytes of BOX32 bounds before it checks the type, so the last
64-byte node would otherwise be read past the structure, or past a compacted copy of it.

**TRI1 pairing.** A bottom-level build without `ALLOW_UPDATE` pairs triangles into TRI0/TRI1 leaves, as the console
builder does in its offline BLASes (§4a, check 4 of "How a match is verified"). The pass that writes the nodes
pairs each triangle child of a node with a later triangle child of the same node (among its at most four children)
when four things hold. The two must share an edge in opposite directions, with the same two vertices bit for bit. They must come from
the same geometry, since a leaf holds one `GeometryIndexAndFlags`. And both must be active. The first triangle is
rotated so the shared edge is slots 1 -> 2, the sibling's third vertex goes in slot 3, its primitive index in
`PrimitiveIndex2`, and each node type's remap byte returns D3D12's barycentrics for its own vertex order. The node
still has one child per triangle, each with its own box: the TRI1 child points at the first triangle's leaf, and the
sibling's own leaf is left unused. The tree, the layout and the sizes are therefore the same as without pairing.
Nothing is saved yet: reclaiming the unused leaves would need a compaction that moves records and rewrites pointers,
and compaction here copies bytes. What pairing gives is the console's leaf format, with a pair's two triangles in
one 64-byte record. Because pairs never cross nodes, one thread owns each leaf it rewrites, and both pointers of a
pair have the same parent, so the parent table has one writer per entry. The pass records each triangle's pointer
(TRI0, or TRI1 for the sibling) in scratch, so the bottom-up box pass finds the sibling's slot through the first
triangle's leaf. On a sorted grid of 7,200 triangles, 2,208 of the 3,600 possible pairs form (2,625 with the tree of four
consecutive elements: a radix-tree node often has an inner node among its children, and only triangle children
pair). A refit rewrites every leaf as one triangle, so a structure that may be updated is
not paired. **ASSUMPTION:** that pairing only siblings is close enough. The console builder's pairing rule is not
traced; `tools/xdxr_bvh.py` `build_blas(pair=True)` pairs greedily over the whole mesh instead, and both traverse
the same way. `check-build` compares barycentrics as well as hits, and checks that a paired case has TRI1 nodes and
that cases that must not pair (other geometry, same direction) have none.

The node count depends on the keys. A kept node has two children only when both of its binary children are
elements, so with N elements there are at most (3N - 2) / 4 nodes. `XdxrLayout` computes the result and scratch
sizes for that many when the build is recorded, and the same numbers answer
`GetRaytracingAccelerationStructurePrebuildInfo`. The header's `totalSizeIn64BNodes` counts only the nodes the build
wrote.
A triangle with a NaN vertex gets an empty box (D3D12 treats it as inactive). So do an instance with mask 0, one with
no BLAS and one with a singular transform. An instance record holds its BLAS address in 256-byte units in 32 bits,
so BLAS addresses must be below 2^40, as console GPU addresses are. Where device addresses are wider (llvmpipe's are
host pointers), the builder takes a bias that is added to the addresses read from instance descs.

The Vulkan layer uses it for `BuildRaytracingAccelerationStructure` unless `LEONIDA_RT_AS=khr`
(`docs/d3d12_vulkan.md` §4.8); `d3d12_xdxr_rt_test <dir>` builds a BLAS and a TLAS through the D3D12 API and writes
them the same way.

`vk_xdxr_build_test <dir>` builds nine synthetic cases on the GPU and writes each structure with a description of its
scene. `xdxr_bvh.py check-build <dir>` validates them and traces rays against brute force. `ctest` runs both as
`xdxr_build` and `xdxr_build_check`.

**Updates** (D3D12 `PERFORM_UPDATE`) refit: the source structure is copied to the destination unless it is the
destination (as many bytes as its header gives, so a compacted source works too), the elements are rewritten at
their slots, and each node's boxes are recomputed bottom-up from the children it already points to, through the
parent table, by the same pass as a build. The tree is kept, so a refit is valid for any motion and skips the Morton pass and
the sort. It took 0.07 ms for 1,000 triangles and 9.8 ms for 1,000,000 on an RTX 2050, against 0.19 ms and 22 ms for
a build, with the tree of four consecutive elements (the radix tree's times are above). The tree traverses more
slowly as the geometry moves away from where it was built.

**Copies and postbuild info.** `CopyRaytracingAccelerationStructure` with `CLONE` or `COMPACT` copies the
structure's bytes. The records are position-independent: node pointers count from the structure's start, and a TLAS
names its BLASes by address, which a clone keeps. The size comes from the source header (`totalSizeIn64BNodes`), so
a copy can follow a build in the same command list, and it is bounded by the destination resource. A build's size
is the nodes it wrote, so `COMPACTED_SIZE` and `CURRENT_SIZE` (from
`EmitRaytracingAccelerationStructurePostbuildInfo` or the postbuild descs of a build) are both
`64 * (totalSizeIn64BNodes + 1)`, and compaction gives back the room the layout left for more nodes. Serialization, deserialization and the tools-visualization
mode are refused at replay as unsupported, not faked. In `LEONIDA_RT_AS=khr` mode the postbuild descs stay ignored,
as before.

Not done yet: serialization; a compaction that reclaims the leaves that pairing leaves unused. A surface-area
collapse of the radix tree, or treelet restructuring, would trim a few more steps.

```powershell
python -I tools/xdxr_bvh.py selftest --seed 1 --rays 400   # build BLASes, compare closest hits with brute force
python -I tools/xdxr_bvh.py validate <dumped BVH buffer>    # structural check of a runtime buffer
python -I tests/test_xdxr_bvh.py
```

## 7b. The first real pack BLAS on a GPU

Measured on the 1.0.2.1 package with the tools above; counts and offsets only.

* `core_bvh4.rpf` indexes to **271,637 BVH4 streams / 1,896,784,254 nodes (121.4 GB inflated)** in 1,034 s.
* The first BLAS (stream at `0x12060`, 39,257 nodes, 2,512,512 bytes rehydrated) matches **one** drawable among the
  258,037 models an export of `levels.rpf` produced: 12 geometries, 21,814 leaves, 0 leaves outside, 335/0
  overrides, area ratio 1.000 (`bvh_match.py`, `bvh_geometry_check.py`, `bvh_rehydrate.py` on the user's copy).
* The rehydrated header's `rootNodePtr` is the **dehydrated** root; the runtime root is the first node of the last
  entry of the level table at the end of the structure. The leaves are in the leaf frame (`(R*v - t)*s`): the header
  `sceneBounds` are the pre-frame AABB (±2.5), the leaf vertices are scaled (≈19,040), so test rays are built from a
  leaf's own vertices.
* Descriptor base: the descriptor's dword0 is the base in **256-byte units**; the game passes `ptr >> 8` there, and
  the `| 0x30` its shader ORs is only the temporary address it uses for the `rootNodePtr` load, not the descriptor
  base. Measured at base 0 and at base 0x1000 (structure copied to VA 0x1000, dword0 `0x10`): the game's RT shadow
  CS (`deferred_tiled` family, 32-bit form) on the RTX 2060 returns the CPU walker's children in the same order at
  both. Its second traversal site still needs the executable.
* The base and the address formula were also measured on the **native instruction**, with no emulation in the path:
  `tools/hw_oracle` ran `image_bvh_intersect_ray` and `image_bvh64_intersect_ray` on gfx1036 (Ryzen 9 9950X3D iGPU,
  driver 32.0.21030.2001) over hand-built box32 nodes, a decoy node at the buffer start and the real one at +0x1000.
  dword0 `(P >> 8) + 0x10` with node `0|5`, and dword0 `(P >> 8) + 0x08` with node `0x800>>3|5`, both traverse the
  node at `P + 0x1000`: address = base × 256 + (node & ~7) × 8, and the base and the node offset add. The control
  (base `P >> 8`, node `0|5`) reads the decoy. Both instruction widths agree. Not exercised: descriptor bits 39:32
  (the buffer fits in 32 bits), nodes other than box32, and a `0x1030`-style dword0, which the hardware would read as
  base 0x103000.

## 8. No-sharing rule

Nothing from the packs may be committed: no `.rpf`, carved stream, decompressed buffer, XML text, tile list,
name hash, key or TOC dump. This document and the tool carry only aggregate counts, offsets, field layouts and a
handful of header bytes. The tests build their inputs by hand. Any future decoder must keep its fixtures synthetic
and write real output only to the user's private work folder.
