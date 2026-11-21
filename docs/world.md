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
| `guid` | `9a7b6e6d` | +0x20 | u64 | placement GUID |
| 64-bit hash | — | +0x08 | u64 | per-archetype hash, function unknown |
| `lodLevel` | `6c8f1715` | +0x7e | u8 | 0 HD, 1 LOD, 2 SLOD1, 3 SLOD2, 4 SLOD3, 5 ORPHAN_HD, 6 SLOD4 |
| `parentIndex` | `d89231bd` | +0x5c | i16 | parent entity index (signed; -1 = none) |
| `parentInstance` | `e74c3c23` | +0x5e | i16 | parent batch instance |
| `numChildren` | `a687ac89` | +0x68 | u32 | linked children |
| `lodDist` | `bf74dfa7` | +0x60 | f32 | LOD distance; -1 = use archetype default |
| `childLodDist` | `ca974bcd` | +0x64 | f32 | child LOD distance |
| `priorityLevel` | `2691f019` | +0x7c | u8 | priority |

The map root carries `name` (`e6e17f14`, +0x08, u32 hash), `parent` (`a99c4651`, +0x68, u32 hash),
`streamingExtentsMin` (`dd22821f`) / `streamingExtentsMax` (`a22ea335`), `entitiesExtentsMin` (`1c75bcf1`) /
`entitiesExtentsMax` (`6d074437`), `flags` and `contentFlags`. The extents are reached through the root's member
table, not fixed offsets.

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

The measured 9,918 files and 680,228 entities are a **lower bound**, not a whole-archive total. The sample is
29 GiB of the 57 GiB map band (50.9%) and 39.23 GiB of the archive (55.3%, including the carver-indexed 0–2 GiB
window).

A single multiplier is weak because density is not uniform: sampled 1-GiB blocks range from 96 to 2,174 map
files and from 4,754 to 145,118 entities. Treating the 29 sampled 1-GiB units as exchangeable gives a
uniform-block estimate of **~26,600 map files / ~1.77 million entities**. A 20,000-resample block bootstrap
gives an approximate 95% interval of 16.0k–39.6k map files and 1.03M–2.64M entities. The selection was
purposive, not random, so those bounds are approximate; they are not a census.

SixNG's published full-archive measurement, 30,785 map files / 2,249,490 entities
(https://github.com/holdmysocks/SixNG), lies inside that interval. It is external evidence, not a measurement
made here. The observed 9,918 files are 32.2% of that count and 680,228 entities are 30.2%, even though 50.9% of
the map band was scanned. That gap is consistent with the unsampled parts being denser, with map files whose
entity blocks live in graphics pages, or with both; only a full band scan can settle it.

## What is readable, what is hash-only

Readable:
- the 0x7a header, schema, struct/member tables and instance blocks;
- map file count, per-file streaming/entity extents;
- every entity's position, rotation, scale, flags, GUID, 64-bit hash, archetype hash and LOD fields;
- in `common.rpf`, the full XML content-definition path list and cell names.

Hash-only or opaque:
- the RPF directory table and therefore original file names; resource names must be recovered from payloads;
- map file `name`/`parent` (32-bit hashes; `parent` is 0 in every sampled file);
- most `archetypeName` values. The repo's model manifest reports 155,598 models with names/bounds; this light
  check used 61,861 name hashes recovered from sampled drawables, fragments and `packs.rpf`, not the full
  manifest join. 5,538 of the 75,551 entities matched (7.3%). The remaining 92.7% are hash-only without a full
  name index and the bbox/archive-order linking step used by SixNG. This is a lower bound, not a claim that the
  names do not exist;
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
4. **Archetype keys.** Resolve the 32-bit `archetypeName` hash to an exported model. Keep the 64-bit +0x08 hash
   and GUID as secondary keys. Hash-only records are normal; the runtime needs a hash→model table, with the
   bbox/archive-order fallback for names that were not recovered.
5. **Streaming cells.** Load map files by their streaming AABB (cube) and use `lodLevel`, `lodDist`,
   `childLodDist`, `parentIndex`, `parentInstance` and `numChildren` to switch HD/LOD/SLOD. Load the HD model
   first, then its parent/children; do not use one global LOD distance.
6. **Model/texture layer.** The existing manifest gives names/bounds; texture keys are 64-bit asset keys in
   model materials. Placement itself contains neither geometry nor texture names.

## Tool

`tools/world_index.py` scans an RPF9 archive or a directory of carved streams and writes JSON/CSV with
`offset, size, signature, version, kind, structs, blocks, entities, extents, position bounds, LOD counts`.
It implements the header/schema validation above and the continuation join. It does not decrypt anything,
does not extract models/textures, and keeps the output aggregate (one row per resource, no entity dump).

```powershell
python -I tools\world_index.py <levels.rpf> --start 21474836480 --limit 1073741824 `
    --out work\world\levels-20-21.json --csv work\world\levels-20-21.csv
python -I tests\test_world_index.py
```

For a 76 GB archive always bound the run with `--start/--limit`; the tool records the window in its summary.
A 0x7a resource split beyond the first continuation, or a map whose entity block lives in a graphics page, is
reported with fewer entities or as `meta` rather than guessed.

## No-sharing rule

Everything here is aggregate. Do not commit archives, carved streams, extracted models/textures, entity
dumps, or logs that quote them. `work/` and `reports/` are git-ignored; the tool output contains only resource
headers, counts and aggregate bounds, but it is still generated from the user's copy and stays local.
