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
keyframe data. The marker is reported as a page marker, not as a known GTA V resource magic. Each `0DRP` stream is
the first system page of a resource; the page map at the `+0x08` pointer gives its page count (see
[Keyframe decoding](#keyframe-decoding)).

### `MoVE` container

324 streams, all with the same header prefix:

```
+0x00  "MoVE"                  (bytes 4D 6F 56 45)
+0x04  u32 19                 (all 324)
+0x14  u32 32-bit hash         (324 distinct values; not a size)
+0x18  u32 payload size        (184/324 equal len-46; the rest have 5/10/15/20 bytes of padding)
```

The header is **46 bytes**.

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
`0x98e1750c` in some tables. It is `joaat32("bonerotation")`, the track-type hash of rotation tracks (see
[Keyframe decoding](#keyframe-decoding)); these tables are second system pages of clip dictionaries. The name hash
is **joaat32 of the path stem**:
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
bitstreams rather than zlib streams. The 24-byte header has five small fields after the type; the measured
version split and the track side are below. No keyframe values are claimed. Each of these streams is the second
system page of a `0DRP` resource, the one right after its `0DRP` stream (see [Keyframe
decoding](#keyframe-decoding)).

### Partial clip decode

Two named clips were followed from their `pack:/...` name-table records to page-0 payloads. The name record
stores `joaat32(stem)`, a second hash, and two 64-bit `0x50xxxxxx` pointers; the pointer rule is page-0 offset
when it is below the page-0 size, otherwise a name-table page offset.

| Clip | Name hash | Page-0 stream | Clip record | Duration | Tracks |
|---|---|---:|---:|---:|---:|
| `breakout_quick_front` | `0xDC5BEF77` | `0x1D284010` | page0 `0x1F8A0` | 3.2333 s | 10 |
| `idle` | `0x71C21326` | `0x9F0E0010` | page0 `0xF580` | 6.0000 s | 8 |

The durations are `f32` at record `+0x78` (breakout) and `+0x68` (idle). Track arrays are `0x48`-byte records
with `type = 0xFFFF0001` and two hashes per record.

Bone masks:

- Breakout: a 63-entry body mask and a 38-entry finger mask.
- Idle: a 12-entry head mask and the same 38-entry finger mask.
- The 38-entry mask is byte-identical between the clips and **all 38 hashes resolve** to
  `SKEL_{L,R}_Finger{00..02,10..13,20..23,30..33,40..43}`.
- The 63-entry mask maps 25 body/`PH_` names; its other 38 entries are unresolved (likely facial or accessory
  bones, not assumed).

The 2,064-byte motionmatcher table also re-parses cleanly: 15 names, parent entries, and all 8 record hashes
resolve.

Header fields measured over the 1,781 streams:

- `+0x04` is always `0x00000101`.
- Version 3 (1,253 streams): `+0x0C = 0x0113` on 1,235, `+0x14 = 0x0207` on all 1,253.
- Version 2 (528 streams): `+0x0C = 0x0207` on all 528; `+0x14` is a 32-bit hash.
- `+0x08` is the leading packed-region length; the first 16-byte `(hash, 0, index, 0)` table follows a short
  sub-header (for example at `0x2C8`, `0x4C8`, `0x6C8` in one page).

**Still missing:** the page → packed-stream join. None of the 1,781 streams contains any of the 30 known bone
hashes or the nine tested clip-name hashes, as 32-bit or 64-bit values; only the shared track-type hash
`0x9A7B6E6D` crosses over. The track records and quantization-like blocks are visible, but the exact component
count/order/delta coding did not survive a unit-quaternion norm test. No keyframe value is claimed for these
`0DRP` pages. The clip dictionaries' keyframes are decoded in [Keyframe decoding](#keyframe-decoding).

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

The grammar-driven tree decoder (`tools/rbf0_tree.py`; `docs/rbf0.md` section 9) parses the same 1,348,436-byte
blob to its last byte and confirms the counts above: 9 descriptors, 28,962 elements, 20,031 text records, 1,126
string values, 6,677 integer values and maximum element depth 6. The 1,126 string values are the `key` attributes
on the rig-set `Item` elements. Each of the 6,677 entry `Item` elements contains `Dictionary`, `Rig` and `Group`
text children followed by a `Priority` integer. The record framing is consumed exactly; the numeric field
semantics remain unresolved. The `Dictionary` text names one of the four rig dictionaries and `Rig` is a key
there; see [Rig dictionaries](#rig-dictionaries).

Mapping to the name tables:

- The 2,298 distinct `pack:/...` paths resolve to a stored joaat32 stem hash for **1,901** names (82.7%);
  **401** stay unresolved. By occurrence, 3,229 of 3,637 (88.8%) match, and 248 of 273 name tables are complete.
- The RBF0 names do **not** overlap the clip/anim/motion namespace: direct name overlap is 0 and stem-hash
  overlap with the 2,298 paths is 0.
- The RBF0 table contains 4,795 distinct rig and rig-set names. An earlier version of the indexer reported
  zero stored joaat32 matches for every input. The count for this corpus needs to be measured again with the
  corrected `--rigs` report. A match indicates that the four hash bytes occur somewhere in the blob.

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
table. The RBF0 rig itself resolves into a rig dictionary ([Rig dictionaries](#rig-dictionaries)); what it does not
give is a whole skeleton for the clips.

### Magics not found

At stream starts across the 130,449 `rpf9_carve` streams, the counts of `YCD`, `YDD`, `YLD`, `YFD`, `RSC7`,
`RSC8`, `ADAT`, `AWC`, `RBF0`, `FSSM` and `KB2` are all zero. The corrected walk adds 21 tail streams whose first
starts with `RBF0`; the rest start inside the same blob. One three-byte `YDD` sequence occurs at +0xE1 inside a
high-entropy stream; it is not at a header boundary and is treated as coincidence, not as a YDD file.

## Keyframe decoding

`tools/anim_clips.py` decodes the clip dictionaries down to per-frame track values. It starts from the same stream
walk as `anim_index.py`, so the stream numbers below are the corrected walk's 130,470. Run on the 1.0.2.1 copy:

```powershell
python -I tools/anim_clips.py <package>\anim.rpf <work>\anim-clips.json --jobs 8
python -I tools/anim_clips.py <package>\anim.rpf --decode <dictionary offset> [--frames N]   # values to stdout
```

The JSON holds counts and one row per dictionary (offset, hash and counts, plus the name read from your copy).
It holds no values. `--decode` prints one dictionary with its decoded values and writes no file.

Provenance: channel types 0–8 follow the GTA V `crAnimation` channel set, as named by community tools such as
CodeWalker's YCD reader (static, quantized, indirect-quantized and cached-quaternion channels). No code was taken
from them. The resource framing, the GTA VI-only channel type 10, the type 3 and type 8 rules below and every
count in this section were measured on the local copy.

### Clip dictionaries are RAGE resources

A clip dictionary is a system-only RAGE resource. Its **root stream** is one of the `starts with 8 zero bytes`
streams above:

```
+0x00  8 zero bytes
+0x08  u64 page-map pointer (0x50xxxxxx)
+0x10  zero up to +0x38
```

At the page map, the byte at +8 is the number of system pages and the byte at +9 the number of graphics pages.
Every dictionary has 1 or 2 system pages and no graphics pages. Its pages are that many back-to-back streams,
each 16-byte aligned after the one before it. Join them, and a pointer `p` is the offset `p - 0x50000000` into
the joined bytes.

Measured: 61,510 root streams. 61,510 join into resources: 35,750 with one system page and 25,760 with two.
In 36 roots the page map says two pages, but the next listed stream does not start where the first one ends. Their
second page is right after the first stream, but `anim_index.py` does not list it: in 28 it is a zlib stream that
inflates to 16–48 bytes (`anim_index.py` lists streams of 64 bytes or more), and in 8 it is a 16-byte raw page,
like rpf9-explorer's raw system tail. `anim_clips.py` joins such a last page only when the joined pages then parse
as a clip dictionary; all 36 do, so no root is left unresolved. That recovery runs only after the page map itself
has been read from listed streams: a root whose page map is in such an unlisted last page stays unresolved ("page
map not in the next stream"). No anim.rpf root has that layout. In `story000.rpf` 5 roots do, with the page map in
a 32- or 48-byte zlib tail, and none of the 5 is a clip dictionary once that tail is joined. Of the joined resources, 61,505 are clip
dictionaries. 4 have `RIG` plus a version at +0x70 instead of a pack array (rig dictionaries, [decoded
below](#rig-dictionaries) by `tools/anim_rigs.py`), and
1 has some other value there. Together the dictionaries use 87,234 listed streams plus the 36 unlisted pages
(3,802,124,713 stored / 7,077,351,104 inflated bytes in the listed streams). By `anim_index.py` class those
streams are: zero-prefix 66,115, raw 20,845, name-table 273, skeleton-names 1.

The other classes fit the same framing:

- **The 273 `pack:/` name tables** at the start of a stream are the second system page of a clip dictionary. All
  273 sit inside dictionaries.
- **The `0DRP` (version 122) streams** are a different resource kind with the same page map. Their system pages are
  one stream in 2,892 resources and two in 19,931. In 19,909 of the two-page ones the second page is the next stream;
  in the other 22 it is not. Their second pages include all 1,781 "2/3 packed clip streams" and all 1,371 type-hash
  streams. So those classes are not stand-alone clip payloads: each one is the page right after a `0DRP` stream in
  the same resource. The `0DRP` root objects are not decoded here. Neither are the 324 `MoVE` streams, which are not
  pages of either kind.

### Objects

All offsets are page offsets. An `atString` is 16 bytes: `{char *ptr; u16 length; u16 capacity}`, or, when byte 15
has bit 7 set, up to 15 characters stored inline with `byte 15 & 0x7F` as the length (909 clip names and 3 dictionary
names). An array is `{T *ptr; u16 count; u16 capacity}`, and an `atMap` is `{Entry **buckets; u32 bucket count (11);
u32; u32 count}` with `Entry {u64 key; T *value; Entry *next}`. Hashes are `joaat32` of the lowercased name.

| Object | Field | Meaning |
|---|---|---|
| dictionary (page offset 0) | `+0x38` atString | dictionary name |
| | `+0x48` atMap | clips; key = `joaat32(dictionary name) << 32 \| clip hash` |
| | `+0x68` u32 | `joaat32(dictionary name)` (all 61,505 dictionaries) |
| | `+0x70` array, 16-byte items | `{pack *ptr; u32 hash; u32 0}`. With a pointer, a local pack and its `joaat32(name)`. With a null pointer, the hash is another clip dictionary's `joaat32(name)` (see `.motion` clips below) |
| animation pack | `+0x38` atString | pack name (`<dictionary>-N`) |
| | `+0x48` atMap | animations, key = `joaat32` of the animation name |
| | `+0x68` u32 | `joaat32(pack name)` |
| `.clip` (`+0x20` u32 < 16) | `+0x28` atString | `pack:/<name>.clip` |
| | `+0x68` pointer | list `{entry *ptr; u16 count; u16 capacity}`, entries 0x38 bytes: `+0` animation pointer, `+8` animation hash, `+0x28`/`+0x2C` start/end (s), `+0x30` rate |
| | `+0x78` f32 | duration (s) |
| `.motion` (`+0x20` is a pointer) | `+0x18` pointer | the dictionary root (page offset 0) |
| | `+0x28` u32, `+0x2C` u32 | the clip hash (the low half of its `atMap` key) and `joaat32` of the dictionary name |
| | `+0x30` atString | `pack:/<name>.motion` |
| | `+0x58` u32 | 1 or 2; selects where the parameter node is (see `+0x120`, `+0x138`); most likely the number of blend parameters (not verified) |
| | `+0x60` array, 16-byte items | children `{u64 0; u32 clip hash; u32 dictionary hash}`: the clips it blends |
| | `+0x70` array, 0x28-byte items | one item per child: `{f32 child duration; f32 duration in the blend; {ptr; u16 count} curve of (child time, blend time) f32 pairs; 16 zero bytes}` |
| | `+0x120` pointer (`+0x58` = 2) | a block whose `+0x18` points at the parameter node |
| | `+0x138` pointer (`+0x58` = 1) | the parameter node: `{u64 0; u32 type; u32 0; ...}` |
| parameter node, type 3 | `+0x10` array of u32, `+0x20` array of 8-byte items, `+0x30`/`+0x34` f32 | one sampled axis: its name hash(es), then `{u32 sample index; f32 value}` sorted by value, then min and max |
| `.motionmatcher` (`+0x20` is a pointer) | `+0x28` 16 bytes, `+0x38` atString | an id, then `pack:/<name>.motionmatcher`; the rest is not decoded |
| animation | `+0x24` u16 | frame count |
| | `+0x28` f32 | duration = (frames − 1) / fps |
| | `+0x50` array | sequence pointers |
| | `+0x60` array, 12-byte items | tracks `{u32 bone hash; u32 track type; u32 format}` |

A clip plays its animations at the same time, each over its own `[start, end]` window. In the samples, the
animations of one clip covered different bones and could have different frame rates.

**Track types** are `joaat32` of their names. This is the meaning of the recurring `0x98e1750c` noted above:

| Hash | Name | Tracks |
|---|---|---:|
| `0x98e1750c` | `bonerotation` | 7,673,334 |
| `0xf791a95f` | `bonetranslation` | 2,440,527 |
| `0x2deefadd` | `genericcontrol` | 1,454,457 |
| `0xf5c3a5b6` | `generictranslation` | 69,844 |
| `0x6ddef121` | `cameratranslation` | 69,645 |
| `0x364d3bdb` | `facialtranslation` | 63,394 |
| `0x7aaa9f60` | `bonescale` | 24,590 |
| `0xd9ad1a74` | `camerarotation` | 476 |
| 36 other hashes | not named | 882,732 |

A bone hash is `joaat32` of the lowercased bone name, for example `skel_pelvis`. In one checked rig, 64 of 66
track bones matched the skeleton's names; bone hash 0 is the whole-body (mover) track. **Formats:** 0 =
quaternion `(x, y, z, w)`, 2 = vec3, 4 = float (7,935,044, 2,907,303 and 1,824,430 tracks; format 3 is used by 12,222
tracks and is not mapped).

### Sequences

An animation's frames are split over one or more sequences. Consecutive sequences **share one frame**, so
`sum(sequence frames) = frames + sequences − 1` (true for 191,042 of 191,042 decoded animations). The repeated frame
decodes to the same pose on both sides: 814,011 of 814,880 quaternion boundaries are the same rotation (|q₁·q₂| >
0.9999, so q and −q count as equal), and 696,853 of 708,648 vec3/float boundaries agree within 1e-3. The cause of the
remaining 0.1% and 1.7% was not investigated.

A sequence is a 0x20-byte header and `length` bytes of data:

| Header | Type | Meaning |
|---|---|---|
| `+0x04` | u32 | data length (excluding the header) |
| `+0x0C` | u32 | frame-data offset in the data |
| `+0x16` | u16 | frames in this sequence |
| `+0x18` | u16 | bytes per frame |
| `+0x1C` | u16 | sum of the type-4 bit widths (true in 212,306 of 212,306 sequences) |
| `+0x1E` | u8 | 11 (the number of channel types) |

The data holds, in order:

1. **Channel definitions**, grouped by type 0 → 10. They end exactly at the frame-data offset in every decoded
   sequence.
2. **Frame data**: `frames × bytes per frame`. Each frame is an LSB-first bit field holding the type-3 values,
   then the type-4 values, then the type-5 indices, in channel order.
3. `u16 count[11]`, the number of channels of each type.
4. 11 lists of `u16 (track << 2 | component)`, one per type, each padded to a multiple of 4 entries. Item `i` of
   list `t` names the track and component of the `i`-th channel of type `t`.

### Channel types

| Type | Definition | Value | Channels |
|---|---|---|---:|
| 0 | 3 × f32 x, y, z | constant quaternion, w = +√(1 − x² − y² − z²) | 3,579,477 |
| 1 | 3 × f32 | constant vec3 | 2,533,788 |
| 2 | f32 | constant | 5,428,131 |
| 3 | none | 32 frame bits = an f32 per frame | 40 |
| 4 | u32 bits, f32 quantum, f32 offset | `bits` frame bits × quantum + offset | 7,600,074 |
| 5 | u32 frame bits, u32 value bits, u32 words, f32 quantum, f32 offset, then the table (`words` u32, value-bits entries) | frame bits index the table; entry × quantum + offset | 484,616 |
| 6 | — | not seen | 0 |
| 7 | none | rebuilds one quaternion component (below) | 4,942,792 |
| 8 | none | marks a quaternion with four stored components (below) | 227,655 |
| 9 | — | not seen | 0 |
| 10 | variable, see below | second-order delta coding (GTA VI) | 6,836,654 |

**Type 10** (new in this build, and the most common animated channel):

```
u32 word count (whole channel, header included) | u8 c1, c2, c3, 0 | f32 quantum | f32 offset | bits (LSB-first):
  nc = ceil(frames / 64) chunks:  nc × c1 bits   chunk payload positions (redundant, for random access)
                                  nc × c2 bits   chunk start values
  ceil((frames − 1) / 8) unary width codes: k zero bits then a one = bit width of the next 8 deltas
  ... padding ...
  last 8 × c3 u32 words: 8 lanes, interleaved (word w·8 + j belongs to lane j);
  lane j holds delta j of every 8-delta group, k bits each, low bits first.
decode, per chunk:  x = start; v = 0
                    for each of the next 63 frames: v += zigzag(delta); x += v
                    value = x × quantum + offset;   zigzag(u) = u / 2 for even u, −(u + 1) / 2 for odd u
```

Integrating a full chunk's 64th delta lands exactly on the next chunk's start value. That holds for
all 48,142,174 of 48,142,174 chunk boundaries. There is one width code per 8 *deltas*: reading `ceil(frames / 8)`
codes runs into the lanes.

**Quaternion tracks.** A track with a type-0 channel is a constant `(x, y, z, +√(1 − x² − y² − z²))`. Otherwise
its components come from scalar channels:

- **Type 7** appears on a track that stores three components (comp 0–2). The missing component is
  `+√(1 − sum of squares)`, placed at the type-7 channel's comp position, and the stored three fill the other
positions in order. Measured: 4,942,792 sequence tracks; the largest |x, y, z| overshoot before normalising is
0.0971.
- **Type 8** always sits at comp 0 on a track that stores all four components. Nothing is rebuilt; the four values
  are normalised. Measured: 227,655 sequence tracks; |q| of the four stored values is within 0.0190 of 1.

Static quaternions: 3,579,477, and 0 of them have |x, y, z| > 1. The (x, y, z, w) order and the rebuilt sign fit the
boundary check above. An earlier local comparison against one skeleton's bind rotations also fits (dot products
0.97–1.0); that comparison is not part of the tool.

### Coverage

191,042 animations in 52,923 local packs; 191,042 decode with every check above passing and 0 fail (0 skipped by the
time budget). 8,582 dictionaries have no local pack; dictionaries refer to 27,559 packs by hash only (18,595 distinct
hashes), and none of those hashes is the hash of one of the 52,923 local packs: they are dictionary hashes. In
anim.rpf, 16,693 of the 18,595 distinct hashes are the `joaat32(name)` of a clip dictionary here (21,014 of the
27,559 references). Clips: 312,263, of which 17,247 are
`.motion` and 180 are `.motionmatcher` (`+0x20` is a pointer in both; the name is at `+0x30` for `.motion` and at
`+0x38` for `.motionmatcher`, after a 16-byte id); `.motion` clips are described below. Sequences: 212,306. Before
the gap-page join, a full run with `--jobs 44` took 5 min 44 s on 44 cores; the slowest dictionary takes about 32 s.

### `.motion` and `.motionmatcher` clips

A `.motion` clip has no animation of its own; it lists other clips (its children), which it presumably blends. Its
fields are in the object table above. Measured on all 17,247 `.motion` clips:

- `+0x18` points at the dictionary root, `+0x28` is the clip's own hash (the low half of its `atMap` key) and `+0x2C`
  is the dictionary's `joaat32(name)`. `+0x58` is 1 on 8,932 and 2 on 8,315.
- They have 159,584 children `{clip hash, dictionary hash}`. For every child, the dictionary hash is one of the
  root's hash-only `+0x70` references (159,584 of 159,584), so those references are the dictionaries the
  `.motion` clips draw from.
- 114,925 children name a dictionary in anim.rpf, and 111,948 of those (97.4%) name a `.clip` of that
  dictionary. The other 2,977 name a hash that is not a clip there.
- The remaining 44,659 children name one of the 1,902 referenced dictionaries that are not in anim.rpf. A local
  search of every clip-dictionary-like root page in `levels`, `packs`, `story000`, `story000_script`,
  `story000_restofworld`, `core_restofworld`, `core_restofworld001`, `shared_minigame000`, `data`, `common` and
  `script` found none of those hashes, so this package does not resolve them.
- **Child times (`+0x70`).** One 0x28-byte item per child on all 17,247: the child's duration, its duration in the
  blend, and a piecewise-linear curve from child time to blend time. Of the 159,584 items, 158,991 have equal
  durations and the two points (0, 0) and (d, d), so the child plays at its own speed. 402 have a 5–82 point curve
  from (0, 0) to (duration, blend duration), increasing in both coordinates, so the child is time-warped. The other
  191 have two points, but 170 don't start at (0, 0) and 21 don't end at (d, d). For every child that names a
  `.clip` in anim.rpf, the first float is bit-for-bit that `.clip`'s duration (`+0x78`): 111,948 of 111,948.
- **Parameter node.** When `+0x58` is 1 the node is at `+0x138`; when it is 2, `+0x120` points at a block whose
  `+0x18` is the node. Every `.motion` has one, and it starts `{u64 0, u32 type, u32 0}`. By type: 3 on 10,171, 1 on
  3,557, 4 on 2,644, 5 on 482, 6 on 295, 8 on 98. Only type 3 is decoded here.
- **Type 3: one sampled axis.** It has one u32 name hash, then 1–16 `{u32 sample index, f32 value}` items and the
  axis min and max. On all 10,171 (29,123 samples), the items are sorted by value, min and max are the first and
  last values, and the sample indices are 0..n−1 in some order. They are not child indices: the sample count
  equals the child count on 4,589 of them. The name hashes are `joaat32` of a lower-case name. 88 distinct hashes
  occur, and a word-list search resolved 10 of them, covering 5,604 axes: `yawangle` 2,524, `rotationz` 1,691,
  `facingdirection` 638, `speed` 284, `surfaceincline` 271, `roll` 66, `verticalheight` 58, `pitch` 25,
  `pitchangle` 25, `height` 22. The ranges fit the names: `speed` is within [0, 1], `surfaceincline` within
  ±0.698 rad (40°), and `roll` within ±3.14. `anim_clips.py` reports every name as `motion_axis_name:<name or hash>`.
- Not decoded: parameter node types other than 3, how a type-3 sample index maps to the children, and the rest of
  the object from `+0x80`, including the `{ptr, count}` lists at `+0x90`.

The 180 `.motionmatcher` clips sit in 142 dictionaries that hold only `.motionmatcher` clips and no local pack.
The three motion matcher streams in [Bone tables and motion matchers](#bone-tables-and-motion-matchers)
(`0x101c379a0`, `0x101c41d50`, `0x101ea66c0`) are the second system pages of three of those dictionaries, with 2, 1 and 2
`.motionmatcher` clips. Their layout after the name is not decoded.

### Other archives

The same tool reads `story000.rpf` with no change: of 34,533 roots with the dictionary header, 34,518 join (7 with a
gap page) and 9,190 are clip dictionaries. The other 25,328 roots have some other value at `+0x70` and are not clip
dictionaries; 15 roots do not join. All 59,593 animations of those dictionaries decode with every check passing, and
all 54,352,969 type-10 chunk boundaries close. Its 14 `.motion` clips have 75 children, and all 75 name a `.clip` of a
dictionary in `story000.rpf`. None of its 9,188 distinct dictionary hashes is a dictionary in anim.rpf, and none is
one of the 1,902 hashes anim.rpf refers to without having them.

```
python -I tools/anim_clips.py <package>\pack\story000\story000.rpf <work>\story000-clips.json --jobs 20
```

## Rig dictionaries

`tools/anim_rigs.py` reads the four `RIG` roots above (ASSET-20). Run on the 1.0.2.1 copy:

```powershell
python -I tools/anim_index.py <package>\anim.rpf <work>\anim-index.json --rigs <work>\rigs.json
python -I tools/anim_rigs.py <package>\anim.rpf <work>\anim-rigs.json --rigs <work>\rigs.json
```

The output holds counts and one row per rig dictionary (offset, name, hash, entry count), no table items.

**Layout.** The root is the clip dictionary's: `+0x38` atString name, `+0x48` atMap, `+0x68` `joaat32(name)` (4 of
4), and at `+0x70` the bytes `RIG` and a version byte (2 in all four). Every map key has a zero high half (3,897 of
3,897). Every map value has a pointer at `+0x20` to a **`RIG` block** (3,897 of 3,897): `RIG`, a version byte (2),
a u32 length of what follows, then tables `{u32 kind hash; u32 size = 0xC + 8 × count; u32 count; u32 word}`, each
followed by `count` 8-byte items `{u32 hash; u32 word}`, with other bytes between the tables. The other bytes and
the item words are not mapped.

**The four dictionaries are the four RBF0 sources.** Their names are the four source strings of the rig-set
manager:

| Name | Entries | Keys that are `joaat32` of an RBF0 rig |
|---|---:|---:|
| `canonical@legacy` | 22 | 16 |
| `canonical@source` | 543 | 534 |
| `generated@source` | 1,256 | 1,188 |
| `pipelined@source` | 2,076 | 1,931 |

These are the distinct-rig counts per source in [the RBF0 table above](#tail-rbf0-rig-set-manager). The other 228
keys are not named by any RBF0 entry. **All 6,677 RBF0 entries resolve**: the entry's `Dictionary` text is the
rig dictionary's name, and `joaat32(lowercase Rig path)` is a key in it (6,677 of 6,677; 0 in another dictionary,
0 missing). So a rig set reaches its rigs' `RIG` blocks through rig-set → `{Dictionary, Rig}` → rig dictionary →
key → `+0x20`.

**Tables.** 6,606 tables are found in the 3,897 blocks. Four kind hashes hold all but 5 of them; the 5 are single
hits whose "kind" is an f32 near 1.0 (`0x3f7f....`, `0x3f76....`) or a small integer, so they are float data that
happens to fit the header rule. The kind names are not resolved:

| Kind | Tables | Items | Items that are a clip track's bone hash |
|---|---:|---:|---:|
| `0x74d00508` | 2,642 | 13,210 | 0 |
| `0x2316833f` | 2,437 | 118,853 | 91,559 |
| `0xb688b77e` | 1,427 | 72,528 | 39,880 |
| `0x5e25d3bf` | 95 | 4,416 | 0 |

"A clip track's bone hash" means the hash is the bone of some track in the 191,042 decoded animations.

**Rigs are not the clips' skeletons.** The clip side was measured with a local script over the same
`anim_clips.py` parse: for each of the 52,752 clip dictionaries with bone tracks, the union of its
`bonerotation`/`bonetranslation` bone hashes (mover track 0 left out).

- Against the 596 hashed names of the shared humanoid skeleton ([character_skeleton.md](character_skeleton.md)),
  44,359 dictionaries have at least 90% of their bones in it (16 have all of them).
- Against the rigs (the items of the two bone-bearing kinds above), no rig set covers the clips of a humanoid
  dictionary. Even the union over all the rigs of one rig set reaches 80% of a dictionary's bones for 11 of those
  44,359. Only 4 dictionaries, all of objects, are covered completely by a rig set.

So humanoid clips bind to the fragment skeleton, not to a `RIG` block. The rig tables look like per-rig bone or
channel subsets (the RBF0 groups are `deformation`, `pose`, `identity`, `facial` and others); that reading is not
measured. Still open in ASSET-20: the `0DRP` "2/3 packed clip streams" ([above](#packed-clip-streams)) and which
model or skeleton each non-humanoid clip dictionary drives.

## Readable, compressed and still opaque

- **Readable:** all 273 name tables (2,298 paths), the `0DRP` version/page count, 1,371 type-hash headers,
  1,781 clip-stream headers with their version split, two named clip records (durations and 0x48-byte track
  arrays), the 63/38 and 12/38 bone masks (all 38 finger hashes resolve), the RBF0 mapping (1,126 rig sets /
  6,677 entries / 3,669 rigs), one standalone bone-name list, and three motion matcher streams with embedded
  bone tables (30-name union; 19 bone hashes present). Some `MoVE` and zero-prefixed pages expose counts and
  float arrays directly.
- **Outer compression:** the only compression this work inverts is the RPF9 zlib framing. Inflating the 130,470
  streams accounts for the 11.59 GB above.
- **No nested zlib:** in eight sampled zero-prefixed pages, none of the `78 01/5E/9C/DA` byte pairs started a
  valid nested zlib stream. The outer zlib stream is the only compression layer observed.
- **No known codec at the sampled headers:** no zstd or LZ4 frame magic was found in the first 4 KiB of the
  130,449 indexed streams. gzip, bzip2 and lzma byte patterns occur, but none started a valid stream at the
  sampled positions.
  No Oodle signature is known from these bytes; none is guessed.
- **Still opaque:** high-entropy blocks inside `other` and zero-prefixed pages (4 KiB blocks up to 7.96
  bits/byte), the `0DRP` root objects and the packed component coding of their 1,781 clip-stream pages, the
  fields of `MoVE` beyond the version/hash/size words, the 56 empty rig sets, the RBF0 numeric fields, and
  the rig→skeleton link (0 measured matches). Of the 30 bone names, 19 have a confirmed stored hash and 11 do
  not; the four motionmatcher paths bind to embedded tables, but not to an RBF0 rig.

## Cross-check with the model and XML work

- `models-out/data/models/manifest.json` (the only model manifest available here) has 1,350 models: 1,341 with
  `HasBoneBindings`, 689 fragments, all with material entries. It exposes model names and 64-bit texture
  `Id64` keys, but no bone names or skeleton hashes.
- None of the 1,350 model names occurs in the 8,020 animation paths collected here, and none occurs in the
  `RBF0` rig metadata either. The sampled clip hash `0x25ed0554e58ba1fc` and the recurring `0x98e1750c` are
  absent from all manifest names and integer/string values. A real cross-check needs the full model export
  (350,251 models per `docs/design.md`) and the rig table.
  (382,781 models at `e293917` per `docs/design.md`) and the rig table.
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
- `MoVE` is named like motion data; the version, 32-bit hash and payload-size words are mapped, the remaining
  fields are not.
- The 2/3 streams expose header variants, track arrays and bone masks. They are the second system pages of `0DRP`
  resources, whose root objects and packed coding are not decoded. The keyframes of the clip dictionaries are
  decoded; see [Keyframe decoding](#keyframe-decoding).
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
   zero-prefixed pages; a logical resource can span several streams. Do not assume one stream is one file. The
   page map at the `+0x08` pointer gives the system page count, and that is how `tools/anim_clips.py` joins the
   clip dictionaries.
4. **Parse the name/hash tables.** Names are NUL-terminated and 16-byte aligned; the name hash is joaat32 of
   the stem (path without `pack:/` and without the final extension). The record stride varies (16 or 32 bytes in
   sampled tables), so do not assume one layout. The recurring `0x98e1750c` is `joaat32("bonerotation")`, a
   track type.
5. **Join rigs to clips, then decode a known clip.** Two named clips are already followed to page-0 payloads
   (durations and track arrays above). The missing link is the **page → packed stream**: none of the 1,781
   packed streams contains the 30 known bone hashes or the nine tested clip hashes. Map the track records and
   the 16-byte `(hash, 0, index, 0)` tables first, then test a component count/order/delta coding against a
   unit-quaternion norm. `MoVE` (version 19, header 46 bytes, hash at `+0x14`, payload size at `+0x18`) is a
   second, separate format; map its fields only after a known payload decodes.
6. **Keep the compression boundary explicit.** Outer RPF9 zlib is understood. Do not add a zlib pass inside a
   page without evidence; the sampled pages had no valid nested stream.
7. **Re-run the model cross-check with the full export.** The current 1,350-model data.rpf manifest has bone
   bindings but no bone names or hashes. The full 350,251-model export plus the `RBF0` rig table is needed to
   bindings but no bone names or hashes. The full 382,781-model export plus the `RBF0` rig table is needed to
   match skeletons to models.

## Reproduction

```powershell
# Full stream index and coverage (writes index.json only)
python -I tools/rpf9_carve.py <package>\anim.rpf <work>\anim-index --index-only

# Signature index (CSV or JSON; --limit/--start for a bounded scan)
python -I tools/anim_index.py <package>\anim.rpf <work>\anim-index.csv
python -I tools/anim_index.py <work>\anim-carve <work>\anim-index.json

# Keyframe decoding: counts and one row per clip dictionary; --decode prints one dictionary's values
python -I tools/anim_clips.py <package>\anim.rpf <work>\anim-clips.json --jobs 8
```

`tools/anim_index.py` reports `MoVE`, `0DRP`, `pack:/`, `SKEL_`, `RBF0`, `PSIN`, type-hash and 2/3 clip headers,
plus `raw:<hex>` for anything it does not recognise. It writes no payloads. `--start` must be 0 or more and
`--limit` at least 1 (decimal or `0x` hex); an out-of-range value stops with a one-line error and exit code 2
before an index is written.

## No-sharing rule

Game files, carved streams, name lists, payload dumps and reports that quote them stay on the machine that owns
the package. Do not commit them, attach them to an issue or PR, or paste them into a chat. Short header excerpts
(a few bytes) and aggregate counts are allowed, as in `CONTRIBUTING.md`. The work folder is git-ignored; this
document contains no archive data.
