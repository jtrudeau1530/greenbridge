# PSIN metadata containers

PSIN is a small metadata container used by the game's data and script archives. It appears at the
start of carved zlib streams in `data.rpf` and `script.rpf`; it is not a compiled script program and
it is not a model, texture or audio resource. The archive table of contents is still encrypted, so
the original names are unavailable; PSIN is identified by its own magic.

## Where it appears

- `script.rpf` carve: **671 streams** start with `PSIN`, all version 22. **670 are complete**; one is
  the first 64 KiB of a larger record (its first-section offset points past the carved block).
- `data.rpf` carve: **2,281 streams** start with `PSIN` (the same project's carve also reports 1,263
  `RBF0` and 3,132 resource pages). The project's existing `ArchiveReader`/`PsinMetadata` path
  validates **1,045 complete PSIN records** in `data.rpf` and **1,012** in `script.rpf` after joining
  adjacent 64 KiB blocks.
- `common.rpf` has no PSIN. `anim.rpf`, `navmeshes.rpf` and the BVH archives have no PSIN at stream
  starts in the measured samples.

## Header

Every measured file begins with the same 16-byte header:

| Offset | Size | Field |
|---:|---:|---|
| +0 | 4 | `50 53 49 4e` (`PSIN`) |
| +4 | 4 | big-endian absolute offset of the first section (`LSCH`) |
| +8 | 4 | little-endian version; **22** on every measured file |
| +12 | 4 | little-endian byte count of the hash table that follows the header |

The hash table sits between the header and the first section. This tool does not interpret it; it is
counted and its length is reported.

## Sections

Sections start at the offset in the header and run to the end of the file. Each section is:

| Offset | Size | Field |
|---:|---:|---|
| +0 | 4 | ASCII tag |
| +4 | 4 | big-endian **total** size, including this 8-byte header |

The next section starts at `offset + size`; the last section ends at EOF. A section with size 8 is
empty. The measured tag counts over the 670 complete `script.rpf` records are:

| Tag | Count | Measured payload |
|---|---:|---|
| `LSCH` | 670 | 464-544 bytes; the type schema (structs and members), decoded below |
| `LENM` | 670 | the schema's enums and their values, decoded below |
| `LSNS` | 670 | high-entropy body (8,608 bytes in the sampled record), not zlib/LZ4; not decoded |
| `LGID` | 670 | empty (8-byte section) |
| `LRSC` | 670 | starts with `0DRP` + version 122; resource page directory link |
| `CHKS` | 670 | file size and a 32-bit checksum |

The `LSCH` sizes on the measured set are 472-552 bytes including the 8-byte header. `LRSC` is the
link to the resource-page world: its payload begins `30 44 52 50` (`0DRP`), then a little-endian 122,
then the page fields used by the generic RPF pager. `CHKS` carries the file size in big-endian and a
32-bit hash; the tool reports the section, not a verified checksum.

## `LSCH` and `LENM`: the type schema

`LSCH` describes the structures of the container's data and `LENM` the enums those structures use. It is a
reflection schema of the kind RAGE's PSO format keeps in `PSCH`, with names reduced to 32-bit hashes. Neither
section carries strings.

### Corpus

Measured over **every PSIN stream in every archive** of package 1.0.2.1. Each `.rpf` except `audio.rpf` (raw
AWC, not zlib) was walked whole with the `rpf9_carve.py` stream walker; on `script.rpf` the PSIN-only walk gives
byte-for-byte the same 1,012 streams as `rpf9_carve.py`. Archives with no PSIN stream: `common`, `shaders`,
`textures`, `movies`, `appdata0`, `core_bvh4`, `core_licenses_data`, `core_restofworld001`,
`shared_minigame000_bvh4`, `story000_bvh4`.

| Archive | PSIN | complete | whole `LSCH` | `LSCH` decoded | whole `LENM` | `LENM` decoded | structs | members | enums | values |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `levels` | 6,982 | 6,650 | 6,680 | 6,680 | 6,671 | 6,671 | 51,414 | 296,370 | 13,913 | 178,110 |
| `story000` | 5,348 | 5,075 | 5,101 | 5,101 | 5,097 | 5,097 | 30,684 | 161,449 | 6,840 | 125,837 |
| `data` | 2,281 | 2,135 | 2,108 | 2,108 | 2,104 | 2,104 | 45,896 | 298,356 | 13,814 | 156,999 |
| `story000_script` | 2,309 | 2,118 | 2,160 | 2,160 | 2,156 | 2,156 | 54,683 | 465,992 | 8,587 | 83,740 |
| `script` | 1,012 | 1,009 | 1,009 | 1,009 | 1,009 | 1,009 | 3,879 | 25,763 | 1,009 | 6,054 |
| `packs` | 586 | 489 | 517 | 517 | 515 | 515 | 5,432 | 52,253 | 1,438 | 48,950 |
| `shared_minigame000` | 73 | 70 | 70 | 70 | 70 | 70 | 277 | 1,430 | 56 | 624 |
| `navmeshes` | 22 | 1 | 1 | 1 | 1 | 1 | 3 | 15 | 0 | 0 |
| `anim` | 18 | 10 | 10 | 10 | 10 | 10 | 43 | 200 | 5 | 20 |
| `core_restofworld` | 6 | 5 | 6 | 6 | 6 | 6 | 51 | 323 | 3 | 28 |
| `story000_restofworld` | 3 | 2 | 2 | 2 | 2 | 2 | 9 | 33 | 0 | 0 |
| **all** | **18,640** | **17,564** | **17,664** | **17,664** | **17,641** | **17,641** | **192,371** | **1,302,184** | **45,665** | **600,362** |

- 157 truncated streams still hold a whole `LSCH`; they are counted. 919 streams are cut off before their first
  section and hold neither.
- 57 complete streams (54 in `data`, 2 in `packs`, 1 in `anim`) use the PSO-style `PMAP`/`PSCH`/`PSIG`
  sections instead and have no `LSCH`.
- 6,421 `LENM` sections are empty (8 bytes): those schemas use no enum.
- `LSCH` sizes, header included: 472-552 bytes on `script`, 2,440-7,464 on `story000_script`, 56-17,496
  on `data`.
- `psin_index.py` reports no schema error on any of them (see the checks under *Tool*).
- Distinct name hashes: 3,175 structs, 10,943 members, 664 enums, 13,946 enum values.

### Shape of both sections

Each section stores **all headers first, then all elements**, in header order. The header count is not stored:
it is the one count *k* for which *k* headers plus the elements they announce fill the payload exactly. That
count exists and is unique on 17,664 of 17,664 `LSCH` and 17,641 of 17,641 `LENM` sections.

### `LSCH` struct header (32 bytes)

| Offset | Size | Field | Confidence |
|---:|---:|---|---|
| +0 | 4 | name hash (case-sensitive joaat) | high: see *Hashes* |
| +4 | 4 | layout signature | high: see below |
| +8 | 1 | flags: 0, 0x0c, 0x1c or 0x24 on 150,352 / 34,361 / 89 / 7,569 structs | field: high; meaning: unknown |
| +9 | 1 | log2 of the alignment, 0-4 | medium: the size is a multiple of 2^n on 192,371 of 192,371 |
| +10 | 2 | 0 on every struct | high |
| +12 | 4 | 0 on every struct | high |
| +16 | 8 | fill, `aa` × 8 on every struct | high |
| +24 | 4 | byte size of the struct | high: every member offset is below it |
| +28 | 2 | 0 on every struct | high |
| +30 | 2 | member count | high: the counts fill the section exactly |

The **layout signature** depends on the layout, not the name. A struct name hash that appears with two
signatures always has two layouts, and the other way round (3,175 of 3,175 names; 49 names have two layouts).
The 199 signatures that occur under more than one name give the same layout (size, offsets, types, sub-codes
and indices) under every name. Its hash function is not known.

### `LSCH` member (16 bytes)

The members follow the struct headers, each struct's members in a row.

| Offset | Size | Field | Confidence |
|---:|---:|---|---|
| +0 | 4 | name hash (case-sensitive joaat), or `0x100` for an array element descriptor | high |
| +4 | 4 | byte offset in the struct (0 for an element descriptor) | high: below the size on 1,302,184 of 1,302,184 |
| +8 | 1 | type code | field: high; per-type meaning: see below |
| +9 | 1 | sub-code: non-zero on every member of types 0x4d and 0x4f, 0 on all others | field: high; meaning: unknown |
| +10 | 2 | element index: for an array (type 0x5c), the position in this struct's member list of its `0x100` element descriptor | high: 49,375 of 49,375 arrays point at a descriptor |
| +12 | 4 | reference hash: a struct or enum name for the types below, else 0 | high for 0x05 and the enum types |

Type codes, with the measured references and the *gap*: the distance to the next member or to the end of the
struct, most common value first. The gap is an upper bound on the member's size, so every reading below is a
hypothesis until a struct is matched against its instance data.

| Type | Members | Reference | Gap | Reading |
|---:|---:|---|---|---|
| 0x05 | 192,944 | a struct name: 155,495 in the same file, 36,277 in another file, 1,172 not found | varies | embedded struct (high) |
| 0x40 | 2,311 | a struct name in another file (1,558) or not found (753) | 256, 128 | fixed block of structs (low) |
| 0x5c | 49,375 | none; element index instead | 16 | array; 16 bytes fits a pointer, count and capacity (medium) |
| 0x60 | 3,887 | an enum (3,866 same file, 21 other file) | 1 | 8-bit enum (medium) |
| 0x62 | 40,024 | an enum (39,758 / 253; 13 not found) | 4 | 32-bit enum (medium) |
| 0x63, 0x64 | 47, 85 | an enum | 1, 2 | 8/16-bit enum variants (low) |
| 0x65 | 16,217 | an enum (16,099 / 115; 1 not found) | 8, 4 | enum-indexed bit set (low) |
| 0x68 | 98 | an enum | 8 | unknown enum use (low) |
| 0x50, 0x53, 0x71, 0x73, 0x75-0x77, 0x54, 0x5a | 20,777, 367, 114,700, 111, 12,888, 4, 2, 1, 3 | a hash that is no struct or enum anywhere in the corpus | 0x71: 24; 0x75: 16 | unknown |
| 0x01, 0x11 | 93,811, 14,761 | - | 1 | 1-byte scalars (medium) |
| 0x13, 0x20 | 7,372, 7,356 | - | 2 | 2-byte scalars (medium) |
| 0x14, 0x15, 0x21, 0x4a | 35,088, 41,144, 110,948, 194,191 | - | 4 | 4-byte scalars (medium) |
| 0x07, 0x08, 0x30, 0x42 | 52,656, 2,232, 1,112, 1,084 | - | 8 | 8-byte values (medium) |
| 0x4d, 0x4f | 152,193, 15,076 | - (sub-code set) | 8, 4, 16 | unknown |
| 0x0d, 0x31, 0x33, 0x34, 0x52 | 582, 7,750, 50,665, 6,072, 50,156 | - | 16 | 16-byte values (medium) |
| 0x36 | 402 | - | 64 | 64-byte value (medium) |

The rest (0x0c, 0x10, 0x12, 0x16, 0x17, 0x32, 0x44, 0x4b) occur 3-1,551 times each. `psin_index.py --schema`
lists every member.

### `LENM` enum header (24 bytes) and value (16 bytes)

| Offset | Size | Field | Confidence |
|---:|---:|---|---|
| +0 | 4 | name hash; a member of the same file's `LSCH` refers to every enum (45,665 of 45,665) | high |
| +4 | 4 | layout signature: it changes exactly when the value list changes (664 of 664 names; 9 have two lists) | high |
| +8 | 8 | fill, `ee` × 8 on every enum | high |
| +16 | 4 | value count | high: the counts fill the section exactly |
| +20 | 4 | 0 on every enum | high |

The values follow the enum headers, each enum's values in a row:

| Offset | Size | Field | Confidence |
|---:|---:|---|---|
| +0 | 8 | signed value: 0, 1, 2, ... in 38,870 of 45,665 enums; 1,270 enums have a negative value | high |
| +8 | 8 | name hash (case-sensitive joaat) in the low half; the high half is 0 on every value | high; unique within each enum (45,665 of 45,665) |

### Hashes

Names are the **case-sensitive joaat** (`world_names.joaat(text, case_sensitive=True)`), as for the engine
member hashes in [world_names.md](world_names.md). Candidates: every identifier-like string in the inflated
`data.rpf` and `script.rpf` streams (3,386,171 strings), against the 32,313 distinct hash values above:

| Function | struct names | member names | enum names | enum value names | signatures |
|---|---:|---:|---:|---:|---:|
| joaat, case-sensitive | 26 / 3,175 | 733 / 10,943 | 0 / 664 | 1,023 / 13,946 | 1 / 3,750 |
| joaat of the lower-cased string only | 0 | 8 | 2 | 4 | 0 |
| control: CRC-32; CRC-32 xor a constant | | | | | 28; 30 over all |

1,782 name hits against a chance level of about 30 confirm the function. The low coverage only means that the
engine's type and field names are mostly not in the data files. The one signature hit is at chance level: the
signatures are not joaat of a name. No name or hash from the game is committed here.

### Links to the rest of the container

- Struct references (type 0x05) resolve in the same `LSCH` for 155,495 of 192,944 members and in another
  file's `LSCH` for 36,277 more. Enum references (types 0x60, 0x62-0x65, 0x68) resolve in the same file's
  `LENM` for 59,945 of 60,354 and in another file for 395 more.
- The region between the 16-byte header and `LSCH` is most likely the instance data these structs describe.
  This is a hypothesis: in the records looked at, it begins with struct name hashes from `LSCH`. The header's
  +12 field is no larger than that region on 18,578 of the 18,640 PSIN streams. Mapping the region onto the schema
  is not done here.
- `LSNS` (high-entropy) is not touched.

## What it does not give

- No plaintext script code, function names or string table. `LSNS` is high-entropy and is not
  decoded; the measured body does not decompress with zlib (15/-15/31) or raw LZ4.
- No station/track names, no model names, no file paths. The archive TOC is still the only source
  for original names.
- No names. `LSCH` and `LENM` give the schema as 32-bit name hashes only; resolving them needs a name
  list from elsewhere (see *Hashes*).

## Tool

```text
python -I tools/psin_index.py <PSIN file or carve directory> [--json] [--schema] [--limit N]
```

`--limit N` stops after N PSIN containers and must be at least 1 (0 or a negative value is refused with exit
code 2). Leave it out to index every container.

- `parse(blob)` returns the magic, version, table size, the ordered section list
  (`tag`, `offset`, `size`, `payload`) and a `complete` flag with the first structural error. When the
  `LRSC` payload starts with `0DRP`, the `0DRP` version is reported; `CHKS` contributes the file size
  and a 32-bit checksum. Text output and the summary include those fields.
- `decode_lsch(payload)` and `decode_lenm(payload)` return the structs (with members) and the enums (with
  values) laid out above, and raise `ValueError` when no header count fills the payload. `parse()` adds
  `structs`, `members`, `enums`, `enum_values` and `schema_error` (an undecodable section, a member past its
  struct's size, or an array whose element index is not an element descriptor) to every record; with
  `--schema` (or `parse(blob, schema=True)`) the decoded schema is included, hashes as hex.
- `collect(src)` scans a single file or every `.bin` in a carve directory; non-PSIN files are
  skipped.
- Text output prints one line per container and a tag summary; `--json` writes `{files, summary}`.
- The parser validates the first-section offset, each tag's printable range, each section size and
  the exact EOF; a 64 KiB split is reported as incomplete rather than guessed.

`tests/test_psin_index.py` covers complete records, an empty section, a bad first offset, a zero-size
section, truncated data, trailing bytes and the JSON CLI, and the `LSCH`/`LENM` layouts, empty sections,
a count that fits nothing and both schema checks. It uses synthetic headers and joaat hashes of invented names
only.

## Limits

- The 64 KiB split is not joined automatically. A caller that has the archive order can join adjacent
  blocks first; the tool then parses the complete container.
- `LSNS` and `CHKS` payloads are opaque to this tool; `LSCH` and `LENM` are decoded as above. The measured container layout
  is enough to index the files and measure coverage, not to run the metadata.
- No game data, names, hashes or payloads are committed. The carve directories are local.
