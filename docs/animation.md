# Animation archive (`anim.rpf`)

Measured state of the animation archive in the Xbox Series 1.0.2.1 package. Everything here was read from a
local copy; no archive, payload, name list or byte dump is committed. The tables are counts and short header
excerpts only.

The table of contents of an RPF9 is encrypted. This work does not read or decrypt it. It indexes the data area
the same way `tools/rpf9_carve.py` does: every zlib stream that inflates cleanly to its end is one **stream** in
the tables below. A logical RAGE resource can span several streams, so "stream" is not "file".

## Measured archive composition

```
archive             4,407,626,384 bytes
RPF9 magic          b"9FPR"
data start          0x4660 (18,016)
data span           4,407,608,368 bytes
```

The requested command (`python -I tools/rpf9_carve.py <package>/anim.rpf <work>/anim-index --index-only`) prints:

| Measure | rpf9_carve |
|---|---:|
| validated zlib streams | 130,449 |
| stored (compressed) bytes | 4,204,677,247 (3.92 GiB) |
| inflated bytes | 11,592,111,696 (10.80 GiB) |
| zero padding | 180,165,556 (4.09% of span) |
| unread bytes | 23,664,707 (0.54% of span) |
| readable | 95.40% of span |
| Bink 2 | 0 streams |
| elapsed | 141 s |

`rpf9_carve.py` overreads its last stream: it advances a 1 MiB read cursor by 1 MiB even when the final mmap
slice is shorter. That makes the last `PSIN` stream look 974,742 bytes stored instead of 1,766 and skips 21
streams at the tail. `tools/anim_index.py` advances by the actual slice length. Its corrected walk finds:

| Measure | corrected walk |
|---|---:|
| streams | 130,470 |
| stored bytes | 4,203,762,662 (3.915 GiB) |
| inflated bytes | 11,593,460,132 (10.80 GiB) |
| zero padding | 180,167,209 (4.09%) |
| unread bytes | 23,678,497 (0.54%) |
| readable | 95.38% |
| elapsed | ~140 s |

The 21 extra streams are the 1.35 MB `RBF0` rig metadata at the end of the data area (below). The class table
uses the corrected counts: the `PSIN` group is 213,747 stored in total, and the tail is its own row.

| Class (first bytes) | Streams | Stored | Inflated | Median | Max | Where |
|---|---:|---:|---:|---:|---:|---|
| `0DRP` at +16 (page directory) | 22,823 | 169,144,668 (161 MiB) | 3,883,044,624 (3.62 GiB) | 196,608 | 5,636,096 | first 1/8 of data area |
| starts with 8 zero bytes | 78,688 | 3,569,562,146 (3.32 GiB) | 6,918,674,880 (6.44 GiB) | 46,864 | 20,578,304 | throughout |
| other (no match below) | 28,646 | 464,618,867 (443 MiB) | 789,033,180 (752 MiB) | 23,984 | 65,536 | throughout |
| `RBF0` rig blob (1 header + 20 blocks) | 21 | 58,391 (57 KiB) | 1,348,436 (1.29 MiB) | 65,536 | 65,536 | tail |
| `pack:/` name table at 0 | 273 | 164,654 (161 KiB) | 711,536 (695 KiB) | 896 | 61,056 | throughout |
| `PSIN` at 0 | 18 | 213,747 (209 KiB) | 646,580 (631 KiB) | 35,768 | 65,536 | scattered |
| `SKEL_` bone list at 0 | 1 | 189 | 896 | 896 | 896 | early |

Only the first `RBF0` stream has the header; the other 20 are continuation blocks, so `anim_index.py` reports one
`rbf0` and 20 `raw:<hex>` records for that tail.

Two observations matter for a decoder:

- Every `0DRP` stream is in the first 1/8 of the data area (about 551 MB of span). The remaining 7/8 contains no
  `0DRP` header at +16.
- The 78,688 zero-prefixed streams are not opaque blobs. In a 200-stream sample, 78% have a 64-bit pointer into
  the `0x50000000` range at +8; the rest are mostly zero or other tables. Some pages carry a name table and
  high-entropy blocks later in the same stream.

## Container signatures actually observed

### `0DRP` page marker at +16

22,823 streams. The fixed part of the header is:

```
+0x00  8 zero bytes
+0x08  u64 pointer into 0x50xxxxxx
+0x10  "0DRP"                  (bytes 30 44 52 50)
+0x14  u32 122                 (same value in all 22,823)
+0x20  u64 page count          (observed 1..41, 32 distinct values)
+0x28  four u64 pointers       (0x50xxxxxx)
+0x48  four u16 values
```

Top page-count values: 1 (1,883), 2 (7,571), 3 (9,241), 4 (2,022), 19–26 (1,592). The largest sampled page
(5,636,096 bytes) is 94.6% zero bytes, with sparse pointers. That is a page directory/relocation table, not
keyframe data. The marker is reported as a page marker, not as a known GTA V resource magic.

### `MoVE` container

324 streams, all with the same header prefix:

```
+0x00  "MoVE"                  (bytes 4D 6F 56 45)
+0x04  u32 19                 (all 324)
+0x14  u32 size hint           (e.g. 852 for an 898-byte stream)
```

Stored 330,677 bytes, inflated 1,488,625 bytes. All 324 payloads are distinct. Entropy is 1.67–4.49 bits/byte, so
the payload is structured, not zlib-compressed. No readable name is stored at the start. The remaining fields are
not mapped; the name is the only evidence that it is motion-related.

### Name tables (`pack:/...`)

273 streams start directly with NUL-terminated ASCII paths; there is no container header before the names. They
hold 711,536 inflated bytes and 2,298 distinct paths:

| Suffix | Distinct paths |
|---|---:|
| `.clip` | 2,012 |
| `.anim` | 274 |
| `.motion` | 12 |

Paths are padded to 16-byte boundaries. The records include a 32-bit name hash; a second 32-bit value recurs as
`0x98e1750c` in some tables, but its meaning is not established. The name hash is **joaat32 of the path stem**:
strip `pack:/`, strip the final extension, lowercase. In one 161-name table all 161 hashes matched exactly;
across the 273 tables, 3,229 of 3,637 name occurrences (88.8%) and 248 of 273 tables matched every name. The
remaining 11.2% are not explained (duplicate names, missing records or a different record layout). A separate scan of the first 4 KiB of the 130,449 `rpf9_carve` streams found 7,661 distinct
`pack:/...` paths and 31,020
occurrences; 5,880 of those streams contain at least one path, bone name or animation word in that prefix. The
union of the two name sources is 8,020 distinct paths. Names also appear embedded at non-zero offsets in
zero-prefixed pages (for example at +0x497e0 and +0x4aa8 in one 327,680-byte page), not only in the 273
start-of-stream tables.

### Skeleton and bone names

A full-corpus prefix scan found 31 distinct `SKEL_*` tokens, 170 occurrences, in the first 4 KiB of the streams;
the count includes fragments. The four full-name streams give the 30-name union below. Two forms were seen in
carved samples:

- a plain NUL-terminated bone list (896-byte stream), and
- a front-coded table (2,064 bytes) whose entries are length-prefixed name fragments plus parent indices.

The `SKEL_<region>_<part>` shape is the conventional RAGE bone naming. The model manifest available here does
not expose bone names, so that match is not measured; the animation pages carry no model name next to the bones.

### Type-hash containers

1,371 streams share this 16-byte header shape:

```
+0x00  u32 type id
+0x04  u32 8
+0x08  u32 chunk count (1..4096)
+0x0c  u32 0
```

The most frequent type ids are `0x88560988` (721 streams), `0x51e5606c` (220), `0x51404a31` (133),
`0x8db79d92` (119) and `0x98a13c55` (7). Their payload is chunk-like, entropy 2.57–4.26 bits/byte. `PSIN` is a
separate 18-stream group (`PSIN` at offset 0, 0.21 MB stored / 0.65 MB inflated); `docs/design.md` already
describes `PSIN` as PSO metadata, and these bytes do not show that it is animation data.

### Packed clip streams

1,781 streams start with:

```
02 00 00 00 01 01 00 00 ...     or     03 00 00 00 01 01 00 00 ...
```

The leading u32 is 2 or 3; the next four bytes are constant. Stored 18,168,950 bytes, inflated 32,691,776 bytes.
1,767 distinct payloads (duplicates up to 10 copies). Entropy 4.03–7.79 bits/byte, so these are packed/quantized
bitstreams rather than zlib streams. The 24-byte header has five small fields after the type, but their meaning
is not established; no name is stored in the stream.

### Tail `RBF0` rig-set manager

The last 21 streams appear to be one logical `RBF0` blob split into 64 KiB blocks (20 × 65,536 plus 37,716
bytes); the first block ends mid-path (`...anim/rig`) and the next starts `/canonical/...`:

```
52 42 46 30 fb ff 01 00 00 00 15 00  "rage__crRigSetManager"
```

They hold 58,391 stored / 1,348,436 inflated bytes. The blob is a rig-set manager, not clip keyframes. Parsing
the `03 60` key strings, `fd ff` value strings and `ff ff 06 00`/`ff ff 07 00` field markers gives:

| Measure | Value |
|---|---:|
| Rig-set keys | 1,126 |
| Source → rig entries | 6,677 |
| Distinct rig paths | 3,669 |
| Distinct names (rig sets + rigs) | 4,795 |
| Empty rig sets | 56 (object 53, vehicle 3) |
| Rig-set categories | ped 664, vehicle 346, object 115, system 1 |
| Max entries / distinct rigs in one set | 45 / 41 |
| Entries with a group string | 6,677 (16 distinct groups) |

Source entries, with distinct rig paths in each source:

| Source | Entries | Distinct rigs |
|---|---:|---:|
| `generated@source` | 3,616 | 1,188 |
| `pipelined@source` | 2,135 | 1,931 |
| `canonical@source` | 888 | 534 |
| `canonical@legacy` | 38 | 16 |

Every distinct rig appears under exactly one source category. Every source tag is followed by exactly one rig
key before the next source (0 exceptions), and every entry is assigned to a rig-set by its offset.

The first rig-set block starts at 0x0045. The tag/length layout is visible directly:

```
0x0045  03 60 03 00 "key"
0x004c  1f 00 <31-byte rig-set path>
0x009a  fd ff 11 00 00 00 "canonical@source" 00
0x00b1  ff ff 06 00 03 00 "Rig" ...
0x00c0  fd ff 30 00 00 00 <48-byte rig path> 00
0x00f6  ff ff 07 00 05 00 "Group" ...
0x0107  fd ff 05 00 00 00 "pose" 00
0x0112  ff ff 08 10 08 00 "Priority" ...  a0 0f 00 00
```

- Rig-set key: `03 60` + u16 length + ASCII (no NUL). The first block is special and puts the schema name `key`
  before the first path; the following 1,125 blocks use `02 00 00 00 00 00 01 00 03 60` + u16 length + path.
- Entry: `fd ff` + u32 length including NUL + source, then `ff ff 06 00` (Rig), the rig key in the same
  `fd ff` form, `ff ff 07 00` (Group), and the group string in the same form. A `ff ff 08 10` field and a mostly
  zero u32 follow; the field semantics are not mapped.
- A non-empty rig-set ends with six `ff` bytes (1,070 occurrences). An empty rig-set has
  `04 00` + six zero bytes + four `ff` bytes and no entry (56 occurrences).
- The group names are the rig's channel/group field; the observed values are `deformation` (2,169),
  `deformation[2]` (1,702), `identity` (1,109), `pose` (938), `pose.ragdoll.post` (160), `facial` (152), and
  ten smaller values.

Mapping to the name tables:

- The 2,298 distinct `pack:/...` paths resolve to a stored joaat32 stem hash for **1,901** names (82.7%);
  **401** stay unresolved. By occurrence, 3,229 of 3,637 (88.8%) match, and 248 of 273 name tables are complete.
- The RBF0 names do **not** overlap the clip/anim/motion namespace: direct name overlap is 0 and stem-hash
  overlap with the 2,298 paths is 0.
- None of the 4,795 RBF0 names stores its joaat32 anywhere in the blob (0 matches); the rig-set and rig keys are
  name strings, not a hash table. A decoder must carry the names forward.

### Bone tables and motion matchers

Four small resources in this index carry full bone names. One is a standalone name list; three are motion-matcher
resources found in the prefix scan that embed a bone table and a `pack:/...motionmatcher` path:

| Offset | Inflated | Distinct names | Occurrences | joaat32 present | Kind |
|---|---:|---:|---:|---:|---|
| `0x101a32160` | 896 | 28 | 68 | 0 | standalone name list |
| `0x101c379a0` | 4,272 | 28 | 55 | 14 | motion matcher |
| `0x101c41d50` | 2,064 | 15 | 15 | 8 | motion matcher |
| `0x101ea66c0` | 2,848 | 30 | 50 | 19 | motion matcher |

The three motion matcher streams contain four distinct `pack:/...motionmatcher` paths. The union of all four
streams is 30 names (28 `SKEL_*` and 2 `PH_*`); 19 of the 30 have a joaat32 present in at least one stream; 11
do not. The standalone 896-byte stream has no hash table.

The 2,064-byte stream shows the embedded record layout most clearly: header `0f 00 e0 00 0c 00 16 00`, eight
0x30-byte records from 0x0528 to 0x0678, a u32 hash at +0x10, a parent-index array at 0x06d0, bind-pose floats
from 0x0100, the full name list at 0x0750, and a `pack:/...motionmatcher` path at 0x04a0:

```
0x0528  0c 00 01 00 00 00 00 00 00 00 00 00 19 de 33 81 0b 00 00 00
0x0558  0c 00 00 00 06 00 01 00 00 00 00 00 00 00 00 00 55 a9 3f 2b 03 00 00 00
```

All 8 of its +0x10 hashes match joaat32 of full names in its own 0x0750 list. The 2,848-byte stream has a
14-entry pointer table at 0x0000 (targets in the `0x5005xxxx` range), records at 0x02d0, and two motionmatcher
paths at 0x0210 and 0x0290; 19 of its 30 names hash-confirm. The 4,272-byte stream has 28 names and 14 hashes.

This is a measured **clip → bone-table binding for the four motionmatcher paths**: the path and the bone table
are in the same stream. It is not a rig → skeleton link for RBF0:

- RBF0 contains no `SKEL_` token and no lowercase `bone`/`skeleton` string; the only `Bones`/`Skel` text is
  inside two rig path names.
- 0 of the 3,669 rigs and 0 of the 1,126 rig sets match any of the 30 bone names or any of the 19 hashes present.
- The four motionmatcher paths are not in the 2,298 `.clip`/`.anim`/`.motion` name-table paths.

So RBF0 gives rig-set → source → rig, and the motion matcher resources show how a clip can carry its own bone
table. The link from an RBF0 rig to that table is still missing; it is the main unresolved binding.

### Magics not found

At stream starts across the 130,449 `rpf9_carve` streams, the counts of `YCD`, `YDD`, `YLD`, `YFD`, `RSC7`,
`RSC8`, `ADAT`, `AWC`, `RBF0`, `FSSM` and `KB2` are all zero. The corrected walk adds 21 tail streams whose first
starts with `RBF0`; the rest start inside the same blob. One three-byte `YDD` sequence occurs at +0xE1 inside a
high-entropy stream; it is not at a header boundary and is treated as coincidence, not as a YDD file.

## Readable, compressed and still opaque

- **Readable:** all 273 name tables (2,298 paths), the `0DRP` version/page count, 1,371 type-hash headers,
  1,781 clip-stream headers, the RBF0 mapping (1,126 rig sets / 6,677 entries / 3,669 rigs), one standalone
  bone-name list, and three motion matcher streams with embedded bone tables (30-name union; 19 bone hashes
  present). Some `MoVE` and zero-prefixed pages expose counts and float arrays directly.
- **Outer compression:** the only compression this work inverts is the RPF9 zlib framing. Inflating the 130,470
  streams accounts for the 11.59 GB above.
- **No nested zlib:** in eight sampled zero-prefixed pages, none of the `78 01/5E/9C/DA` byte pairs started a
  valid nested zlib stream. The outer zlib stream is the only compression layer observed.
- **No known codec at the sampled headers:** no zstd or LZ4 frame magic was found in the first 4 KiB of the
  130,449 indexed streams. gzip, bzip2 and lzma byte patterns occur, but none started a valid stream at the
  sampled positions.
  No Oodle signature is known from these bytes; none is guessed.
- **Still opaque:** high-entropy blocks inside `other` and zero-prefixed pages (4 KiB blocks up to 7.96
  bits/byte), the 1,781 packed clip streams, the fields of `MoVE` beyond the version and size hint, the 56 empty
  rig sets, the RBF0 numeric fields, and the rig→skeleton link (0 measured matches). Of the 30 bone names, 19
  have a confirmed stored hash and 11 do not; the four motionmatcher paths bind to embedded tables, but not to
  an RBF0 rig.

## Cross-check with the model and XML work

- `models-out/data/models/manifest.json` (the only model manifest available here) has 1,350 models: 1,341 with
  `HasBoneBindings`, 689 fragments, all with material entries. It exposes model names and 64-bit texture
  `Id64` keys, but no bone names or skeleton hashes.
- None of the 1,350 model names occurs in the 8,020 animation paths collected here, and none occurs in the
  `RBF0` rig metadata either. The sampled clip hash `0x25ed0554e58ba1fc` and the recurring `0x98e1750c` are
  absent from all manifest names and integer/string values. A real cross-check needs the full model export
  (155,598 models per `docs/design.md`) and the rig table.
- `common.rpf` is the readable metadata layer. It carved to 781 streams, 710 of them XML-ish. Strict keyword
  occurrences: `clip` 734, `anim` 1,131, `motion` 237, `gesture` 2, `bone` 428 (214 `<Bones>` open/close
  pairs), `ped_` 1,308, `blend` 40; `facial`, `ycd` and `skeleton` 0. The tags include `<AnimData>`,
  `<ClipSet>`, `<ClipSelectionRule>` and `<Bones>`, and a path to `platform:/anim/animscenes/gesture/gesture.rpf`.
  These are configuration references, not clip payloads.
- A 64 MiB sample of `data.rpf` (6,359 streams, 3 XML-ish) yielded no animation catalog. Its strict keyword
  hits are dialogue or model/decal names; the five `ycd#` hits are part of a model hash table, not a container
  header.

## Likely format family

The bytes support **new RAGE resource pages**, not the GTA V `YCD`/`RSC7` family:

- RPF9 container, `0DRP` page directories with `0x50xxxxxx` virtual pointers and page counts.
- `pack:/...` path tables with 16-byte-aligned names and 16-byte hash/pointer records.
- `SKEL_*` bone tables in plain and front-coded forms.
- `RBF0` rig-set metadata (`rage__crRigSetManager`): 1,126 rig sets, 6,677 source→rig entries and 3,669 distinct
  rigs, with source tags and group strings. The rig→skeleton key is not in RBF0 or the skeleton streams.
- `MoVE` (version 19) and the 2/3 clip streams as the animation payloads; neither is zlib inside the page.
- No `YCD`, `YDD`, `YLD`, `YFD`, `RSC7` or `RSC8` at any stream start.

`MoVE` is probably motion/motion-matching data and the 2/3 streams are probably packed clips, but only the bytes
above are measured. Treat those two labels as hypotheses until a decoder reads a known clip.

### What the bytes do not prove

- `0DRP` is a page marker with a version, page count and pointers. Its role is inferred from the page layout, not
  decoded, and the marker is not proven to be animation-specific.
- `MoVE` is named like motion data, but its fields beyond the version and size hint are unmapped.
- The 2/3 streams are packed bitstreams with a constant four-byte prefix. No clip has been decoded.
- The `RBF0` rig-set/rig/group entries are parsed. The numeric fields, the schema type table and the
  rig→skeleton reference are not; the measured skeleton match is 0.
- The three motion matcher streams embed bone tables, but whether those define a full skeleton or only the
  motion-matching bones is not established. 19 of 30 names have a matching joaat32 in at least one stream.

## What a future decoder needs

1. **Use the parsed RBF0 rig mapping.** It is 1,126 rig sets, 6,677 source→rig entries and 3,669 distinct rigs;
   56 rig sets are empty. The key/tag layout is above. The field after `ff ff 08 10` and the rest of the schema
   still need mapping.
2. **Resolve rig → skeleton.** None of the 3,669 rigs or 1,126 rig sets matches a stored skeleton name or one of
   the 19 bone hashes present. Three motion matcher streams bind four paths to embedded bone tables (30-name
   union; parent indices and bind data in the 2,064-byte example). Parse those records to learn the clip→bone
   binding, then find how an RBF0 rig points at the same table. The missing key is likely a separate rig asset
   or an unparsed numeric field, not these pages. Do not infer it from the category names.
3. **Group streams into resources.** The TOC is encrypted, so use the `0DRP` page header: count at +0x20,
   pointers at +0x08 and +0x28..+0x40, four u16 values at +0x48. The sampled layout places page headers among
   zero-prefixed pages; a logical resource can span several streams. Do not assume one stream is one file.
4. **Parse the name/hash tables.** Names are NUL-terminated and 16-byte aligned; the name hash is joaat32 of
   the stem (path without `pack:/` and without the final extension). The record stride varies (16 or 32 bytes in
   sampled tables), so do not assume one layout. The recurring `0x98e1750c` needs a meaning before it can be
   used as a type or source id.
5. **Join rigs to clips, then decode a known clip.** Pick one `.clip` name table entry, follow its pointer into
   the page, and use the rig entry for its rig identity. The rig→skeleton key still has to come from step 2.
   The 24-byte header and the packed bitstream are the blocking unknowns. `MoVE` (version 19, size hint at
   +0x14) is a second, separate format; map its fields only after a known payload decodes.
6. **Keep the compression boundary explicit.** Outer RPF9 zlib is understood. Do not add a zlib pass inside a
   page without evidence; the sampled pages had no valid nested stream.
7. **Re-run the model cross-check with the full export.** The current 1,350-model data.rpf manifest has bone
   bindings but no bone names or hashes. The full 155,598-model export plus the `RBF0` rig table is needed to
   match skeletons to models.

## Reproduction

```powershell
# Full stream index and coverage (writes index.json only)
python -I tools/rpf9_carve.py <package>\anim.rpf <work>\anim-index --index-only

# Signature index (CSV or JSON; --limit/--start for a bounded scan)
python -I tools/anim_index.py <package>\anim.rpf <work>\anim-index.csv
python -I tools/anim_index.py <work>\anim-carve <work>\anim-index.json
```

`tools/anim_index.py` reports `MoVE`, `0DRP`, `pack:/`, `SKEL_`, `RBF0`, `PSIN`, type-hash and 2/3 clip headers,
plus `raw:<hex>` for anything it does not recognise. It writes no payloads.

## No-sharing rule

Game files, carved streams, name lists, payload dumps and reports that quote them stay on the machine that owns
the package. Do not commit them, attach them to an issue or PR, or paste them into a chat. Short header excerpts
(a few bytes) and aggregate counts are allowed, as in `CONTRIBUTING.md`. The work folder is git-ignored; this
document contains no archive data.
