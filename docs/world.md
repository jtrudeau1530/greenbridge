# World placement and streaming data

What is between the exported models/textures and a rendered world: the world metadata resources, the map
placement records, their transforms, bounds and LOD/streaming fields. Everything below was measured on the
Xbox Series 1.0.2.1 package. No game data is in this file; counts, aggregate bounds and format constants only.

## How much was read

`levels.rpf` is 76,155,065,056 bytes (70.92 GiB), so it was sampled in bounded windows rather than scanned
whole. `packs.rpf`, `data.rpf` and `common.rpf` are small enough to index completely. All indexes were made
with the corrected RPF9 carver from PR #90 (`--index-only`), which advances by the actual stream length. Every
stream is zlib; "unread" is non-zero bytes between streams that did not inflate, not a second storage format.

### First pass (four windows)

| Area | Bytes covered | Streams | Stored | Inflated | Zero | Unread | Coverage |
|---|---:|---:|---:|---:|---:|---:|---:|
| `packs.rpf` (whole archive) | 4,798,213,264 | 144,364 | 4.269 GiB | 9.067 GiB | 3.08% | 1.39% | 95.5% |
| `levels.rpf` 0–2 GiB | 2,147,483,648 | 3,185 | 1.994 GiB | 13.063 GiB | 0.28% | 0.02% | 99.7% |
| `levels.rpf` 20–21 GiB | 1,073,741,824 | 16,683 | 0.982 GiB | 2.359 GiB | 1.48% | 0.42% | 98.2% |
| `levels.rpf` 40–41 GiB | 1,073,741,824 | 14,314 | 0.985 GiB | 1.774 GiB | 1.17% | 0.35% | 98.5% |
| `levels.rpf` 70–70.8 GiB | 858,993,459 | 14,445 | 0.787 GiB | 1.923 GiB | 1.50% | 0.09% | 98.4% |
| `data.rpf` (whole archive) | 481,544,304 | 28,509 | 0.394 GiB | 1.307 GiB | 7.88% | 4.21% | 87.9% |
| `common.rpf` (whole archive) | 6,642,192 | 954 | 0.005 GiB | 0.047 GiB | 5.85% | 8.64% | 85.5% |

### Second pass (bounded map-band scan)

The second pass used `tools/world_index.py` on the gaps and on every 2 GiB around the two known map areas. The
requested 72–76 GiB window does not exist: the file ends at 70.92 GiB. The tail window is 70.8–70.92 GiB
(134,143,917 bytes). Coverage is scanned compressed bytes / window span; streams and `0x7a` resources are from
the same tool.

| Window (GiB) | Streams | Coverage | 0x7a resources | Map files | Entities |
|---|---:|---:|---:|---:|---:|
| 4–5 | 22,991 | 97.5% | 648 | 0 | 0 |
| 6–7 | 23,433 | 97.0% | 914 | 0 | 0 |
| 7–8 | 12,897 | 98.9% | 1,892 | 498 | 36,480 |
| 8–9 | 17,619 | 97.8% | 4,430 | 1,398 | 88,228 |
| 9–10 | 18,886 | 97.9% | 4,107 | 1,068 | 75,344 |
| 10–11 | 10,239 | 99.0% | 1,042 | 164 | 18,306 |
| 16–18 | 35,216 | 97.9% | 9,449 | 2,174 | 145,118 |
| 22–23 | 16,824 | 98.2% | 3,522 | 821 | 50,438 |
| 26–27 | 17,974 | 98.0% | 4,737 | 992 | 68,492 |
| 30–31 | 16,946 | 98.2% | 2,959 | 152 | 10,533 |
| 34–35 | 13,082 | 98.6% | 2,361 | 98 | 4,754 |
| 38–40 | 28,331 | 98.5% | 4,591 | 209 | 13,476 |
| 42–43 | 10,725 | 98.7% | 2,874 | 111 | 9,566 |
| 46–47 | 9,085 | 99.4% | 2,906 | 105 | 7,999 |
| 50–51 | 13,634 | 98.6% | 1,737 | 121 | 6,086 |
| 52–54 | 29,483 | 98.5% | 5,062 | 214 | 11,942 |
| 54–56 | 28,185 | 98.5% | 4,715 | 226 | 10,658 |
| 56–58 | 33,248 | 98.3% | 7,346 | 276 | 16,731 |
| 58–60 | 30,811 | 98.5% | 5,136 | 215 | 9,842 |
| 60–62 | 30,411 | 98.5% | 5,041 | 178 | 9,766 |
| 62–64 | 39,004 | 98.1% | 2,372 | 104 | 4,006 |
| 64–66 | 59,986 | 97.3% | 127 | 0 | 0 |
| 66–68 | 48,706 | 97.5% | 8,203 | 0 | 0 |
| 68–70 | 50,412 | 97.6% | 137 | 0 | 0 |
| 70.8–70.92 | 2,353 | 97.7% | 0 | 0 | 0 |
| **new windows** | **620,481** | **98.2%** | **86,308** | **9,124** | **597,765** |

The second pass scanned 37,026,725,270 bytes (34.48 GiB) in 35.125 GiB of windows. Adding the three first-pass
`world_index` windows gives 39,984,950,733 bytes (37.24 GiB, 52.5% of the archive); adding the carver-indexed
0–2 GiB window gives 42,126,097,955 bytes (39.23 GiB, 55.3%). This is still a sample, not a census.

The first four `levels.rpf` windows cover 5,153,960,755 bytes, 6.77% of the archive. The sampled stream split
is ~2.36 GiB in 20–21 GiB, 1.77 GiB in 40–41 GiB and 1.92 GiB in 70–70.8 GiB; the first two GiB are anomalous
(see the metadata family below). Positions and counts are sample measurements, not whole-map totals.

## Composition

Resource classes after joining contiguous graphics pages to their system page. "model" means a system page that
carries a `.#dr`, `.#ft` or `.#dd` name; it is an upper bound on model/fragment resources, not proof that
`rpf9dump` decodes geometry from it. "respage" is a standalone `42 40 00 00` graphics page. "system" is a
zero-head `0x5...` system page with no drawable name found in the stream. Counts are resources; bytes are
inflated.

| Area | model | respage | system | 0x7a meta (first page) | 0x7a map (first page) | PSIN | other/text |
|---|---:|---:|---:|---:|---:|---:|---:|
| `packs.rpf` | 27,224 / 5.350 GiB | 205 / 0.082 | 47,933 / 0.417 | 4,678 / 0.530 | 0 | 586 / 0.010 | 35,229 / 2.678 |
| `levels` 0–2 GiB | 0 | 0 | 0 | 2,855 / 13.054 | 0 | 8 / <0.001 | 322 / 0.009 |
| `levels` 20–21 GiB | 2,956 / 1.170 | 500 / 0.798 | 2,207 / 0.214 | 2,728 / 0.053 | 630 / 0.019 | 46 / <0.001 | 5,218 / 0.106 |
| `levels` 40–41 GiB | 1,172 / 0.442 | 782 / 0.672 | 1,984 / 0.305 | 3,282 / 0.026 | 96 / 0.002 | 32 / <0.001 | 6,014 / 0.328 |
| `levels` 70–70.8 GiB | 5,782 / 0.931 | 395 / 0.877 | 271 / 0.065 | 67 / 0.001 | 0 | 10 / <0.001 | 1,622 / 0.049 |
| `data.rpf` | 2,917 / 0.390 | 5 / 0.001 | 229 / 0.065 | 15,768 / 0.375 | 0 | 2,281 / 0.035 | 2,758 / 0.427 |
| `common.rpf` | — | — | — | — | — | — | 954 / 0.047 (257 XML docs) |

`data.rpf` also has 1,263 `RBF0` metadata resources (0.008 GiB) and 87 `GFX*` shader blobs (0.002 GiB).
`common.rpf` is XML config, not placement: 3,315 `<filename>` paths, 2,267 unique `.rpf` paths, including
519 `_metadata.rpf`, 117 `_merged.rpf`, 93 `_interiors.rpf`, 51 `_occl.rpf` and 2 `_occl_2.rpf` map-cell
archives. Those names explain the cell layout, but the outer `levels.rpf` does not expose them; its directory
table stays encrypted.

The 20–21 GiB map count needs a correction: a first system page of exactly 65,536 bytes is often followed by a
small continuation stream (usually 1–2 KiB) that the next stream scan alone would miss. `tools/world_index.py`
joins that continuation when the schema still has unresolved pointers. The join turns 630 first-page maps into
698 complete map files and 55,564 entities into 75,551. The other first-pass windows did not change. This is a
format rule, not a heuristic: the continuation is reached by system pointers in the first page, and the joined
schema then validates.

### Where the maps are

Map placement is not clustered in a few page ranges. It occupies one continuous band, bounded at the tested
window granularity:

- 0–2, 4–5 and 6–7 GiB: **0 maps**; 2–4 and 5–6 GiB were not sampled;
- 7–8 GiB: the band starts, 498 maps; the first map resource is at `0x1e0bed010` (7.51 GiB);
- 8–64 GiB: every sampled window has maps; density ranges from 96 to 1,398 maps per GiB (96 to 2,174 per
  sampled window);
- 62–64 GiB: the last sampled map is at `0xfb883d010` (62.88 GiB);
- 64–70.92 GiB: sampled 64–66, 66–68, 68–70, 70–70.8 and 70.8–70.92 GiB, **0 maps**.

That is a 57 GiB map band. The first pass's 794 files were not a 20/40 GiB cluster: 20–21 and 40–41 GiB are
inside the band, while its 0–2 and 70–70.8 GiB windows are outside it. The revised answer is that maps are spread
through roughly 80% of `levels.rpf`, with local density varying by more than 14×.

Within a window the maps are also packed in runs, not spread evenly. Median spacing between consecutive map
resource offsets is 8–24 KiB; runs are separated by hundreds of MiB (largest gaps 0.4–1.0 GiB). For example,
8–9 GiB has 1,398 maps over an 842 MiB span and 16–18 GiB has 2,174 over 1,331 MiB, while 58–60 GiB has 215
maps spread over 1,356 MiB in 30 runs.

## Signatures and strings

The class names are not stored as text: `CMapData` and `CInstance` had zero literal hits in `common.rpf` and
the sampled `levels.rpf` streams. The map roots are found by their struct hashes (`d3593fa6`, `ce501483`),
which are the case-sensitive joaat of `CMapData` and `CEntityDef`. Readable strings in the world streams are
model/material names (`.#dr`, `.#ft`) and, in `data.rpf`, UI/dialogue text; `ipl`, `stream`, `lod`, `cell`,
`zone` hits were substrings of those names or of unrelated text, not chunk tags. No repeated 4×4 or 3×4 matrix
table was found.

Byte offsets used: the bounded windows in the two tables above. The first map resource in the 20–21 GiB window
is at 21,588,586,512 (`0x506c7b010`); the first in the 40–41 GiB window is at 42,953,629,712
(`0xa003c6010`). The `0DRP` signature is at +0x10 of a map/meta stream; in a `PSIN` resource the same tag is in
the footer.

## The 0x7a resource

World metadata and map placement are **self-describing `0x7a` resources**, not `PSIN` and not `RBF0`. A carved
stream that starts a resource has 8 zero bytes, a `0x5...` page-map pointer at +0x08, then:

```
+0x10  "0DRP" + u8 0x7a          magic + version
+0x20  u32 root object count
+0x28  u64 -> struct table       32 bytes per struct
+0x30  u64 -> enum table
+0x38  u64 -> instance table     16 bytes per block
+0x48  u16 structs, u16 enums, u16 blocks, u16 reserved
```

Struct record (32 bytes): name hash, signature, flags, 0, `u64 -> member table`, size, 0, member count.
Member record (16 bytes): name hash, byte offset, type, subtype, reference, extra. Instance block (16 bytes):
struct hash, byte size, `u64` pointer. The pointers are `0x50000000 | file offset`; a `0x6...` pointer would
reach an appended graphics page. Fields are not named on disk. Their 32-bit hashes are the **case-sensitive
joaat** of the engine names; matching those hashes is how the fields below were identified.

Every 0x7a header is checked the same way: the struct/instance table pointers must lie inside the stream, every
struct's member table must lie inside the stream, and at least one block must have a known struct and a
resolvable pointer. A block whose struct hash is not in the table is kept opaque rather than guessed (some
resources use a page block there). Entity blocks must divide cleanly by the entity struct size before entities
are counted. A stream whose tables point past its end is treated as a split resource and its continuation is
joined, rather than rejected.

Two structs identify a map-placement file: root `d3593fa6` (`CMapData`, 0x460 bytes in the sample) and entity
`ce501483` (`CEntityDef`, 0x90 bytes). Both are the case-sensitive joaat of those class names, which confirms
both the names and the hash function. Files with no entity/root struct are counted as `meta`, not placement.

`0DRP` appears in one other place: at the end of a `PSIN` resource it is the page-directory chunk, followed by
the same `0x7a` u32 and a `CHKS` chunk. It is not a map file; it describes the pages of the model/texture
resource. `data.rpf` contains 15,768 such 0x7a/page-directory resources (most are page directories),
`packs.rpf` 4,678, and none of them has a `CMapData` or `CEntityDef` struct. The task's "0DRP page
directories" and a map file share the magic, not the payload type.

### The first two GiB are a different family

The 0–2 GiB window is 89.6% 0x7a resources (2,855 first pages; 2,824 validate completely), 13.05 GiB inflated,
and **none contain an entity block**. All 2,824 carry one identical 7-struct schema:

```
6c4d9553 71ca4525 ba51bd1e c2b80676 c69192ec e33df1c6 fb42c06a
```

Their blocks include 9,546 blocks of exactly 1,048,576 bytes, which at 16 bytes per slot is 65,536 slots: a
hash-table family, like the slot tables in the text containers. There are also many 16/576/1152/1728-byte
blocks. The header and schema are readable; the field semantics are not. This family is **not** placement and
should not be decoded as entity tables. It is the main reason the first two GiB look so different from the rest.

## Map placement records

Decoded from the entity schema; offsets below are the observed member offsets in the sampled files. `position`
is RAGE world space in metres, Z up. `rotation` is the stored quaternion and is the inverse of the entity's
world rotation (conjugate it for rendering). `scaleXY` applies to X/Y, `scaleZ` to Z.

| Field | Hash | Offset | Type | Meaning |
|---|---|---:|---|---|
| `archetypeName` | `a023a02c` | +0x44 | u32 | hash of the archetype/model name |
| `position` | `01166164` | +0x38 | 3×f32 | world position, metres |
| `rotation` | `77dd4010` | +0x10 | 4×f32 | quaternion x,y,z,w, stored inverse |
| `scaleXY` | `9ca32637` | +0x54 | f32 | XY scale |
| `scaleZ` | `10fb7c42` | +0x58 | f32 | Z scale |
| `flags` | `67d26872` | +0x48 | u32 | entity flags |
| entity id | `1b60404d` | +0x08 | u64 | per-entity id (not a name key; not unique, see world_names.md) |
| archetype key | `c8a7b90a` | +0x20 | u64 | per-archetype key: MurmurHash3 x64 128 low 64 of the name |
| `lodLevel` | `6c8f1715` | +0x7e | u8 | 0 HD, 1 LOD, 2 SLOD1, 3 SLOD2, 4 SLOD3, 5 ORPHAN_HD, 6 SLOD4 |
| `parentIndex` | `d89231bd` | +0x5c | i16 | parent entity index (signed; -1 = none) |
| `parentInstance` | `e74c3c23` | +0x5e | i16 | parent batch instance |
| `numChildren` | `a687ac89` | +0x68 | u32 | linked children |
| `lodDist` | `bf74dfa7` | +0x60 | f32 | LOD distance; -1 = use archetype default |
| `childLodDist` | `ca974bcd` | +0x64 | f32 | child LOD distance |
| `priorityLevel` | `2691f019` | +0x7c | u8 | priority |

The 2,716 entity schemas sampled for [world_names.md](world_names.md) all use the two 64-bit members above.
The earlier `guid` reading (`9a7b6e6d`, +0x20) was not a member of those schemas: +0x08 changes per placement,
while +0x20 is one value per archetype. [world_names.md](world_names.md) gives the measured formulas and the
name-resolution coverage.

The map root carries `name` (`e6e17f14`, +0x08, u32 hash), `parent` (`a99c4651`, +0x68, u32 hash),
`streamingExtentsMin` (`dd22821f`) / `streamingExtentsMax` (`a22ea335`), `entitiesExtentsMin` (`1c75bcf1`) /
`entitiesExtentsMax` (`6d074437`), `flags` and `contentFlags`. The extents are reached through the root's member
table, not fixed offsets.

### Name resolution

`archetypeName` and map `name`/`parent` are the lower-cased joaat of the model or map name. The per-archetype
+0x20 key and the map +0x10 key are MurmurHash3 x64 128 (seed 0, low 64); they confirm a 32-bit match instead of
assuming it. A local 79,738-name dictionary resolved 5,107/24,105 distinct archetype hashes (8,263/203,799
entities) in two 1 GiB windows, every one confirmed by the 64-bit key, and rejected one 32-bit-only candidate.
The same dictionary resolved 5/23,535 map names and 1/2,089 parent rows, all confirmed by the map key. This is a
name-source lower bound, not a formula limit: with the names `tools/rpf9_names.py` lists in `levels.rpf`, 99.85%
of the 1,670,435 entities a whole-archive `world_index.py` run reads resolve, all verified by the 64-bit key. See [world_names.md](world_names.md) and
`tools/world_names.py`.

### Measured placement summary

| Scope | Map files | Entities |
|---|---:|---:|
| 0–2 GiB (first pass) | 0 | 0 |
| 20–21 GiB (first pass) | 698 | 75,551 |
| 40–41 GiB (first pass) | 96 | 6,912 |
| 70–70.8 GiB (first pass) | 0 | 0 |
| **first-pass subtotal** | **794** | **82,463** |
| **second-pass new windows** | **9,124** | **597,765** |
| **measured total** | **9,918** | **680,228** |

The 20–21 GiB LOD split: HD 53,551; LOD 2,119; SLOD1 230; SLOD2 40; SLOD3 11; SLOD4 10; ORPHAN_HD 19,590.
`lodDist` is -1 for 74,375 entities (98.4%, meaning archetype default); the rest are 10–1200 m.
`parentIndex` is -1 for 19,874 entities (26.3%); the rest point at another entity. The 40–41 GiB window splits
HD 5,354; LOD 1,248; SLOD1 278; SLOD2 25; SLOD3 6; SLOD4 1.

Every sampled map file's three streaming extents are equal, i.e. a cube. Cube sizes range from 20 m to
42,744 m; common values in the 20–21 GiB window are 240, 300, 500, 600 and 2000 m. That per-file cube is the
cell/zone metadata a streamer needs. **No global cell table, zone table or 4×4/3×4 matrix array was found** in
the sampled streams. Placement is per-entity position + quaternion + scale, not a matrix; the repeated 16-byte
pattern in the 0–2 GiB family is a hash-table slot, not a matrix row.

### How many are there?

A full sharded census of the `levels.rpf` data area now gives a measured total: **23,535 map resources /
1,670,442 placement entities**. Every map row is exact against its declared entity size and every expected
shard completed. The earlier bounded passes measured 9,918 files and 680,228 entities in 55.3% of the archive;
that was a lower bound, and the full pass settles the extrapolation. See
[world_census.md](world_census.md) for the shard layout, the exactness condition and the 98.32% stream
coverage.

The full count is lower than SixNG's published 30,785 map files / 2,249,490 entities
(https://github.com/holdmysocks/SixNG), and the whole difference is now accounted for map by map (see
[world_census.md](world_census.md#the-sixng-difference-map-by-map)). The 23,528 maps both count agree
on every entity count. SixNG also counts 7,257 `0x7a` resources that have `CEntityDef` objects but no
`CMapData` struct: 961 interior archetype resources (`CMloArchetypeDef`, 540,665 entities in interior
space) and 6,296 `35abc248` entity groups (39,622 entities in group-local space). It misses 7 maps
(1,239 entities) whose entity data is in a continuation page. 7,257 − 7 = 7,250 maps and 580,287 −
1,239 = 579,048 entities. The 23,535 / 1,670,442 census figure is the count of world placement
lists; interiors and entity groups need the transform of whatever places them.

## What is readable, what is hash-only

Readable:
- the 0x7a header, schema, struct/member tables and instance blocks;
- map file count, per-file streaming/entity extents;
- every entity's position, rotation, scale, flags, per-entity id, per-archetype 64-bit key, archetype hash and LOD
  fields;
- in `common.rpf`, the full XML content-definition path list and cell names.

Hash-only or opaque:
- the RPF directory table and therefore original file names; resource names must be recovered from payloads;
- map file `name`/`parent` (32-bit hashes; `parent` is non-zero in 2,089 of the 23,535 census maps). A local
  dictionary resolves 5 names and 1 parent; all five names are confirmed by the map root's 64-bit key. The
  remaining map-cell names are not in the local model/XML exports;
- `archetypeName` values are 32-bit hashes, but nearly all of them resolve. With the `name.#dr`/`name.#ft` strings
  that `tools/rpf9_names.py` lists in `levels.rpf` as the dictionary, 1,667,978 of 1,670,435 entities (99.85%,
  227,422 of 228,681 distinct hashes) resolve, every one verified by the entity's 64-bit key; see
  [world_names.md](world_names.md#whole-archive-coverage). The earlier figures (5,107 of 24,105 hashes, 8,263 of
  203,799 entities = 4.05%, in two 1 GiB windows) used only names already exported locally;
- texture keys do not appear in placement records. They are 64-bit asset keys in model materials, so a runtime
  resolves placement → archetype → model → material asset key;
- the 0–2 GiB 7-struct family's field semantics and most of its payload;
- `PSIN` page-directory internals beyond the chunk tags (`MNEL`/`LSNS`, `LGID`/`LRSC`, `0DRP`/`0x7a`, `CHKS`);
- texture dictionaries: no `.#td` system page was found in the sampled `levels.rpf` windows. World textures in
  this package are the loose RDS5 pools under `data/ttdb` plus `textures.rpf`, so placement records do not
  carry texture names, only material/asset keys in the model resources.

## What a runtime needs

1. **RPF9 page reader.** Read the `9FPR` header; the data area starts at the u32 at +0x04. Streams start on a
   16-byte boundary and are zlib. Join a system page and its contiguous graphics pages; for 0x7a resources also
   join the small continuation page when the schema's pointers reach it.
2. **0x7a schema parser.** Validate the header and tables, then read instance blocks. Do not assume fixed
   offsets: field offsets come from the resource's member table. The map root/entity hashes above identify
   placement files.
3. **Transforms.** Apply `position`, conjugate the stored quaternion for world rotation, and use `scaleXY` /
   `scaleZ`. The source is RAGE Z-up; glTF-style Y-up is `(x, z, -y)` with the vector part rotated accordingly.
4. **Archetype keys.** Resolve the 32-bit `archetypeName` hash to an exported model. Confirm the match with the
   per-archetype 64-bit key at +0x20 (MurmurHash3 x64 low 64); do not use the +0x08 per-entity id as a name key.
   Hash-only records are normal; the runtime needs a hash→model table, with the bbox/archive-order fallback for
   names that were not recovered.
5. **Streaming cells.** Load map files by their streaming AABB (cube) and use `lodLevel`, `lodDist`,
   `childLodDist`, `parentIndex`, `parentInstance` and `numChildren` to switch HD/LOD/SLOD. Load the HD model
   first, then its parent/children; do not use one global LOD distance.
6. **Model/texture layer.** The existing manifest gives names/bounds; texture keys are 64-bit asset keys in
   model materials. Placement itself contains neither geometry nor texture names.

## Tool

`tools/world_index.py` scans an RPF9 archive or a directory of carved streams and writes JSON/CSV with
`offset, size, signature, version, kind, structs, blocks, entities, extents, position bounds, LOD counts`,
map name/parent hashes and the map root's 64-bit name key. It implements the header/schema validation above
and the continuation join. It does not decrypt anything and does not extract models/textures. Its JSON/CSV
output is aggregate (one row per resource). `--entities-out` adds a local-only JSON-lines file with one row per
complete entity (archetype hash and key, position, stored rotation, scales, LOD fields), the input of
`world_build.py --placements` below.

`tools/world_census.py` scans the whole `levels.rpf` data area in shards and counts every validated
`0x7a` map resource and placement entity. The measured full pass is 23,535 maps / 1,670,442 entities
with 98.32% stream coverage; the exactness condition and byte accounting are in
[world_census.md](world_census.md).

`tools/world_names.py` builds the hash→name dictionary from local model manifests, level-scan paths, name
lists and XML, then resolves `world_index`/`world_census` map rows and aggregate archetype counts. It
reports 32-bit-only candidates separately from matches confirmed by the 64-bit key. The measured dictionary
and coverage are in [world_names.md](world_names.md).

```powershell
python -I tools\world_index.py <levels.rpf> --start 21474836480 --limit 1073741824 `
    --out work\world\levels-20-21.json --csv work\world\levels-20-21.csv
python -I tests\test_world_index.py
python -I tests\test_world_names.py
```

For a 76 GB archive always bound the run with `--start/--limit`; the tool records the window in its summary.
A 0x7a resource split beyond the first continuation, or a map whose entity block lives in a graphics page, is
reported with fewer entities or as `meta` rather than guessed.

## Building a placed world

`tools/world_build.py --placements` builds the viewer's chunks from map placements instead of each model's own
coordinates. It covers points 3 and 4 of *What a runtime needs*; point 5 (HD/LOD switching) is not done.

1. `world_index.py <levels.rpf> --entities-out <work>\world\entities.jsonl` writes one row per complete entity.
2. rpf9dump exports the models (`models/manifest.json` + OBJ).
3. `world_build.py <assets> --out <world> --placements <entities.jsonl>`. Each selected entity is matched to an
   exported model by the lower-cased joaat of the manifest `Name` without its `.#dr`/`.#ft` suffix. When the
   entity row carries the +0x20 key, the model's Murmur3-64 must equal it, as in `world_names.py`; a 32-bit-only
   match is a `key_mismatch` and is not placed. The model is scaled by (`scaleXY`, `scaleXY`, `scaleZ`), rotated
   by the **conjugate** of the stored quaternion, translated to `position`, and converted from RAGE Z-up to
   glTF Y-up `(x, z, -y)`. Normals use the inverse-transpose; a negative scale product flips the winding.

`--lod-levels` picks the `lodLevel` values to place (default `0,5`: HD and ORPHAN_HD) and `--region
MINX,MINY,MAXX,MAXY` bounds the RAGE world X/Y. `world.json` then has `"layout": "placements"` and a
`stats.placements` block: `rows`, `malformed`, `other_lod_level`, `outside_region`, `selected`, `placed`,
`placed_unverified` (no key in the row), `key_mismatch`, `unresolved_archetype` (no exported model has that name),
`model_not_loadable` and `distinct_models`.

Measured on the 1.0.2.1 package:

| Step | Result |
|---|---|
| `--entities-out` over the whole data area (24 parallel `--start/--limit` windows) | 1,670,435 rows, about 1 min |
| `lodLevel` split | 0: 1,153,122 · 5: 396,871 · 1: 100,953 · 2: 16,828 · 3: 1,708 · 4: 601 · 6: 352 |
| one 700 m × 700 m downtown box, HD + ORPHAN_HD | 29,258 entities, 3,995 distinct archetypes |
| placed with models recovered by the pinned rpf9-explorer (917e9b9) | 10,729 (36.7%); 18,529 `unresolved_archetype`, 0 `key_mismatch` |
| same box, models recovered with breadbbbb/rpf9-explorer#8 and #10 applied | 29,229 of 29,258 entities have a model (3,992 of 3,995 archetypes) |
| build of the 10,729 placed instances (`--no-textures --jobs 8`, 256 m chunks) | 16 chunks, 29.3M triangles, 1.17 GB of GLB, median chunk 104 MB, max 164 MB; 5 min, 22 GB peak RSS |

The unresolved share in that box is a model-recovery gap, not a naming gap. All 1,800 archetypes that #8/#10 add
in that box are `.#ft` fragments (18,500 of the entities). The pinned `LooksLikeFragment` rejects their headers
(breadbbbb/rpf9-explorer#8).

Checks on the same box:
- The conjugate of every stored rotation equals SixNG's independently derived world rotation:
  |dot| ≥ 0.9999 on 29,124 of 29,125 entities matched by position and archetype, and the minimum is 0.99894.
  Scales are equal on all 29,125.
- Every placed instance's position lies inside its chunk's bounds (10,620 of 10,620 distinct
  position/archetype pairs).

**Not verified:** the result has not been looked at in a browser. Instances are baked into the chunk geometry,
not shared, so a dense box makes 100 MB+ chunks; a smaller `--chunk-size` keeps them small. Building a whole
city needs glTF instancing and LOD switching, which are next steps.

```powershell
python -I tools\world_index.py <levels.rpf> --out work\world\index.json --entities-out work\world\entities.jsonl
python -I tools\world_build.py work\assets --out work\world\placed --no-textures `
    --placements work\world\entities.jsonl --region=-900,500,-200,1200
python -I tests\test_world_place.py
```

The entity rows and the placed chunks come from your own copy of the game; keep them in the local work folder.

## Instanced props and grass

Map resources can carry two instance lists besides `CEntityDef`. All four struct names are the case-sensitive
joaat of the GTA V class names, and every member below is named by the resource's own member table.
`tools/instance_lists.py` reads them through that table (by name hash) and writes counts only.

| Hash | Struct | Size | Members (offset) |
|---|---|---:|---|
| `0xBA049770` | `rage__fwPropInstanceListDef` | 0x170 | `CEntityDef` members, `InstanceList` 0x90, `BatchAABB` 0xA0, `ScaleRange` 0x160, `InstanceLodDist` 0x168 |
| `0xEA6B3A96` | `rage__fwPropInstanceListDef__InstanceData` | 0x40 | `InstMat` 0x0 (3x4 floats, translation in the last column), `Tint` 0x34 |
| `0x7C475B5D` | `rage__fwGrassInstanceListDef` | 0x1A0 | `CEntityDef` members, `BatchAABB` 0x80, `ScaleRange` 0xA0, `LodFadeStartDist` 0xB8, `InstanceLodDist` 0xBC, `OrientToTerrain` 0xC0, `slodID` 0xB0, `IsSLODList` 0x150, `SLODArchetypeList` 0x158 (array), `InstanceList` 0x178 |
| `0xED86F522` | `rage__fwGrassInstanceListDef__InstanceData` | 0x10 | `Position` 0x0 (3 x u16 in `BatchAABB`), `NormalX` 0x6, `NormalY` 0x7, `Color` 0x8 (3 bytes), `Scale` 0xB, `ScaleZ` 0xC, `RotationIdx` 0xD |

Two more grass-list members are named by joaat from a brute force over common engine words: `slodID` (0xB0, member hash `9fe78fd1`) and `SLODArchetypeList` (0x158, `b451fc98`), so grass lists carry their own link to an SLOD archetype set next to `IsSLODList`. The other unnamed members stay as hashes in the schema.

An `InstanceList` is a u64 (low 12 bits: 1-based block index; bits 20+: offset in that block) and a u16 count.
Instance blocks are usually in the physical buffer, so the tool joins each resource's virtual and physical
streams from its block map (+0x08 -> +8 virtual count, +9 physical count).

Full run over `levels.rpf`, `story000.rpf` and `shared_minigame000.rpf` (stream occurrences, duplicates
included):

| Archive | Map resources | With lists | Prop lists | Prop instances | Grass lists | Grass instances |
|---|---:|---:|---:|---:|---:|---:|
| `levels.rpf` | 188,447 | 135,560 | 690,779 | 24,778,818 | 202,143 | 219,453,097 |
| `story000.rpf` | 48,084 | 2,683 | 4,833 | 1,652,980 | 5,504 | 5,641,657 |
| `shared_minigame000.rpf` | 787 | 472 | 3,317 | 51,795 | 592 | 437,849 |
| total | 237,318 | 138,715 | 698,929 | 26,483,593 | 208,239 | 225,532,603 |

Struct sizes and every member offset in the table match the schema in all 98,722 resources with prop lists
and all 57,650 with grass lists. 29,482 prop lists and 8,015 grass lists point to a block outside the
streams the tool joined; their instances are not counted.

Props (26,483,593 instances): every matrix is right-handed; 26,046,778 have orthonormal axes (within 0.01),
25,936,781 are uniformly scaled and 4,692,108 have unit scale; the translation is inside the list's
`BatchAABB` for 26,386,624. List `lodLevel`: LOD 424,012, HD 116,660, ORPHAN_HD 93,958, SLOD1 64,299. Every
list has an ordered `ScaleRange` and `0 < InstanceLodDist <= lodDist`.

Grass (225,532,603 instances): 225,379,032 `NormalX`/`NormalY` pairs (byte / 127.5 - 1) are inside the unit
disc; `RotationIdx` is 0 to 72; `ScaleZ` equals `Scale` in 52,021,564 and `Scale` is 0 in 47,679,734, so
`ScaleZ` is its own value, not ambient occlusion. `OrientToTerrain` is 1.0 in all 208,239 lists, `IsSLODList`
is set in 5,802, and `lodLevel` is ORPHAN_HD in 205,181 and SLOD1 in 3,058.

## No-sharing rule

Everything here is aggregate. Do not commit archives, carved streams, extracted models/textures, entity
dumps, or logs that quote them. `work/` and `reports/` are git-ignored; the tool output contains only resource
headers, counts and aggregate bounds, but it is still generated from the user's copy and stays local.
