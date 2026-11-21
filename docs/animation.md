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
- The 78,688 zero-prefixed streams are not opaque blobs. In a 200-stream sample, 78% start with eight zero bytes
  and a 64-bit pointer into the `0x50000000` range; the rest are mostly zero or other tables. Some pages carry a
  name table and high-entropy blocks later in the same stream.

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

Paths are padded to 16-byte boundaries. After the strings come 16-byte records containing a 32-bit hash and a
second 32-bit value; `0x98e1750c` recurs in the sampled pages, but its meaning is not established. A separate
scan of the first 4 KiB of the 130,449 `rpf9_carve` streams found 7,661 distinct `pack:/...` paths and 31,020
occurrences; 5,880 of those streams contain at least one path, bone name or animation word in that prefix. Names
also appear embedded at non-zero offsets in zero-prefixed pages (for example at +0x497e0 and +0x4aa8 in one
327,680-byte page), not only in the 273 start-of-stream tables.

### Skeleton and bone names

A full-corpus prefix scan found 31 distinct `SKEL_*` tokens, 170 occurrences, in the first 4 KiB of the streams.
Two forms were seen in carved samples:

- a plain NUL-terminated bone list (896-byte stream), and
- a front-coded table (2,064 bytes) whose entries are length-prefixed name fragments plus parent indices.

The `SKEL_<region>_<part>` naming shape matches the bone names used by the model side, but the animation pages
carry no model name next to them.

### Type-hash containers

1,355 streams share this 16-byte header shape:

```
+0x00  u32 type id
+0x04  u32 8
+0x08  u32 chunk count (1..4096)
+0x0c  u32 0
```

The most frequent type ids are `0x88560988` (721 streams), `0x51e5606c` (220), `0x51404a31` (133),
`0x8db79d92` (119) and `0x98a13c55` (7). Their payload is chunk-like, entropy 2.57–4.24 bits/byte. `PSIN` is a
separate 18-stream group (`PSIN` at offset 0, 0.21 MB stored / 0.65 MB inflated); `docs/design.md` already
describes `PSIN` as PSO metadata, and these bytes do not show that it is animation data.

### Packed clip streams

1,790 streams start with:

```
02 00 00 00 01 01 00 00 ...     or     03 00 00 00 01 01 00 00 ...
```

The leading u32 is 2 or 3; the next four bytes are constant. Stored 18,268,253 bytes, inflated 33,175,328 bytes.
1,186 distinct payloads (duplicates up to 10 copies). Entropy 4.3–6.5 bits/byte, so these are packed/quantized
bitstreams rather than zlib streams. The 24-byte header has five small fields after the type, but their meaning
is not established; no name is stored in the stream.

### Tail `RBF0` rig metadata

The last 21 streams are one logical `RBF0` blob split into 64 KiB blocks (20 × 65,536 plus 37,716 bytes):

```
52 42 46 30 fb ff 01 00 00 00 15 00  "rage__crRigSetManager"
```

They hold 58,391 stored / 1,348,436 inflated bytes and 20,224 printable runs (4,819 distinct). The readable
content is a rig-set manager, not clip keyframes:

| Lowercase token | Occurrences | Distinct |
|---|---:|---:|
| `anim/rigset/` paths | 1,126 | 1,126 |
| `anim/rig/` paths | 6,677 | 3,669 |
| `generated@source` | 3,616 | 1 |
| `pipelined@source` | 2,135 | 1 |
| `canonical@source` | 888 | 1 |
| `deformation` | 4,217 | 6 |
| `pose.` | 253 | 6 |
| `ped/` paths | 4,828 | 1,974 |
| `vehicle/` paths | 2,691 | 2,567 |
| `prop/` paths | 79 | 79 |
| `facial` | 164 | 5 |
| `motion` | 150 | 1 |
| `bone`, `SKEL_`, `clip` | 0 | 0 |

Counts are case-sensitive lowercase matches. Case-insensitively, `bone` is 4/2 and `skel` is 2/1, all inside two
rig path names; `clip` stays 0. `facial` and `motion` also appear capitalised in a few path names.

Schema field names include `rage__crRigSetManager`, `RigSet`, `Item`, `Entries`, `Dictionary`, `Priority`, `key`,
`canonical@legacy`, `canonical@source`, `generated@source`, `pipelined@source`, `deformation`, `facial`,
`identity` and `pose.ragdoll.post`. This is the likely rig-set → rig mapping table. It is the strongest lead for
joining animation paths to skeletons, and it was missing from the `rpf9_carve` index because of the tail overread.

### Magics not found

At stream starts across the 130,449 `rpf9_carve` streams, the counts of `YCD`, `YDD`, `YLD`, `YFD`, `RSC7`,
`RSC8`, `ADAT`, `AWC`, `RBF0`, `FSSM` and `KB2` are all zero. The corrected walk adds 21 tail streams whose first
starts with `RBF0`; the rest start inside the same blob. One three-byte `YDD` sequence occurs at +0xE1 inside a
high-entropy stream; it is not at a header boundary and is treated as coincidence, not as a YDD file.

## Readable, compressed and still opaque

- **Readable:** all 273 name tables (2,298 paths), 31 distinct `SKEL_*` tokens, the `0DRP` version/page count, 1,355
  type-hash headers, 1,790 clip-stream headers, and the 4,819 distinct strings in the `RBF0` rig metadata. Some
  `MoVE` and zero-prefixed pages expose counts and float arrays directly.
- **Outer compression:** the only compression this work inverts is the RPF9 zlib framing. Inflating the 130,470
  streams accounts for the 11.59 GB above.
- **No nested zlib:** in eight sampled zero-prefixed pages, none of the `78 01/5E/9C/DA` byte pairs started a
  valid nested zlib stream. The outer zlib stream is the only compression layer observed.
- **No known codec at the sampled headers:** no zstd or LZ4 frame magic was found in the first 4 KiB of the
  130,449 indexed streams. gzip, bzip2 and lzma byte patterns occur, but none started a valid stream at the
  sampled positions.
  No Oodle signature is known from these bytes; none is guessed.
- **Still opaque:** high-entropy blocks inside `other` and zero-prefixed pages (4 KiB blocks up to 7.96
  bits/byte), the 1,790 packed clip streams, and the fields of `MoVE` beyond the version and size hint.

## Cross-check with the model and XML work

- `models-out/data/models/manifest.json` (the only model manifest available here) has 1,350 models: 1,341 with
  `HasBoneBindings`, 689 fragments, all with material entries. It exposes model names and 64-bit texture
  `Id64` keys, but no bone names or skeleton hashes.
- None of the 1,350 model names occurs in the 8,020 animation paths collected here, and none occurs in the
  `RBF0` rig metadata either. The sampled clip hash `0x25ed0554e58ba1fc` and the recurring `0x98e1750c` are
  absent from all manifest integers. A real cross-check needs the full model export (155,598 models per
  `docs/design.md`) and the rig table.
- `common.rpf` is the readable metadata layer. It carved to 781 streams, 710 of them XML-ish. Strict keyword
  occurrences: `clip` 734, `anim` 1,131, `motion` 237, `gesture` 2, `bone` 428 (214 `<Bones>` open/close
  pairs), `ped_` 1,308, `blend` 40; `facial`, `ycd` and `skeleton` 0. The tags include `<AnimData>`,
  `<ClipSet>`, `<ClipSelectionRule>` and `<Bones>`, and a path to `platform:/anim/animscenes/gesture/gesture.rpf`.
  These are configuration references, not clip payloads.
- A 64 MiB sample of `data.rpf` (6,359 streams, 3 XML-ish) contains no animation catalog. Its strict keyword
  hits are dialogue or model/decal names; the five `ycd#` hits are part of a model hash table, not a container
  header.

## Likely format family

The bytes support **new RAGE resource pages**, not the GTA V `YCD`/`RSC7` family:

- RPF9 container, `0DRP` page directories with `0x50xxxxxx` virtual pointers and page counts.
- `pack:/...` path tables with 16-byte-aligned names and 16-byte hash/pointer records.
- `SKEL_*` bone tables in plain and front-coded forms.
- `RBF0` rig-set metadata (`rage__crRigSetManager`) mapping rig-set paths to canonical/generated/pipelined rigs.
- `MoVE` (version 19) and the 2/3 clip streams as the animation payloads; neither is zlib inside the page.
- No `YCD`, `YDD`, `YLD`, `YFD`, `RSC7` or `RSC8` at any stream start.

`MoVE` is probably motion/motion-matching data and the 2/3 streams are probably packed clips, but only the bytes
above are measured. Treat those two labels as hypotheses until a decoder reads a known clip.

### What the bytes do not prove

- `0DRP` is a page marker with a version, page count and pointers. Its role is inferred from the page layout, not
  decoded, and the marker is not proven to be animation-specific.
- `MoVE` is named like motion data, but its fields beyond the version and size hint are unmapped.
- The 2/3 streams are packed bitstreams with a constant four-byte prefix. No clip has been decoded.
- The `RBF0` rig-set manager is readable strings and schema field names. Its value encoding (indices, hashes,
  parent links) is not parsed yet.

## What a future decoder needs

1. **Parse the `RBF0` rig-set manager first.** It is 1.35 MB, has an explicit schema name
   (`rage__crRigSetManager`) and 4,819 readable rig/rig-set paths. Extract the rig-set → rig entries and the
   channel names (`deformation`, `facial`, `pose.*`). That is the missing skeleton/rig table; without it, clips
   have no bone order.
2. **Group streams into resources.** The TOC is encrypted, so use the `0DRP` page header: count at +0x20,
   pointers at +0x08 and +0x28..+0x40, four u16 values at +0x48. The sampled layout places page headers among
   zero-prefixed pages; a logical resource can span several streams. Do not assume one stream is one file.
3. **Parse the name/hash tables.** Names are NUL-terminated and 16-byte aligned; records are 16 bytes. The hash
   algorithm is not identified (joaat32 of the sampled paths did not match the observed values), so do not
   synthesise hashes. The recurring `0x98e1750c` needs a meaning before it can be used as a type or source id.
4. **Join rigs to clips, then decode a known clip.** Pick one `.clip` name table entry, follow its pointer into
   the page, and use the rig entry for its bone order. The 24-byte header and the packed bitstream are the
   blocking unknowns. `MoVE` (version 19, size hint at +0x14) is a second, separate format; map its fields only
   after a known payload decodes.
5. **Keep the compression boundary explicit.** Outer RPF9 zlib is understood. Do not add a zlib pass inside a
   page without evidence; the sampled pages had no valid nested stream.
6. **Re-run the model cross-check with the full export.** The current 1,350-model data.rpf manifest has bone
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
