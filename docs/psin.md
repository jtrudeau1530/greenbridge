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

Measured over **every** PSIN in the `data.rpf` and `script.rpf` carves of package 1.0.2.1 (both archives carved
whole with `rpf9_carve.py`; 3,293 PSIN streams, 3,144 complete). On that package the `script.rpf` carve has
1,012 PSIN streams (1,009 complete) and `data.rpf` 2,281 (2,135 complete). 54 complete `data.rpf` records use
the PSO-style `PMAP`/`PSCH`/`PSIG` sections instead and have no `LSCH`.

| | `script.rpf` | `data.rpf` | all |
|---|---:|---:|---:|
| PSIN with a whole `LSCH` section | 1,009 | 2,108 | 3,117 |
| `LSCH` decoded, no schema error | 1,009 | 2,108 | **3,117** |
| structs / members | 3,879 / 25,763 | 45,896 / 298,356 | 49,775 / 324,119 |
| PSIN with a whole `LENM` section | 1,009 | 2,104 | 3,113 |
| `LENM` decoded | 1,009 | 2,104 | **3,113** |
| `LENM` with no enums (8-byte section) | 0 | 980 | 980 |
| enums / values | 1,009 / 6,054 | 13,814 / 156,999 | 14,823 / 163,053 |
| distinct struct / member / enum / value name hashes | | | 2,170 / 7,869 / 423 / 11,887 |

The `data.rpf` `LSCH` sections run from 56 to 17,496 bytes (header included); on `script.rpf` 472-552.
27 truncated `data.rpf` records still hold a whole `LSCH`; they are counted above.

### Shape of both sections

Each section stores **all headers first, then all elements**, in header order. The header count is not stored:
it is the one count *k* for which *k* headers plus the elements they announce fill the payload exactly. That
count exists and is unique on 3,117 of 3,117 `LSCH` and 3,113 of 3,113 `LENM` sections.

### `LSCH` struct header (32 bytes)

| Offset | Size | Field | Confidence |
|---:|---:|---|---|
| +0 | 4 | name hash (case-sensitive joaat) | high: see *Hashes* |
| +4 | 4 | layout signature | high: see below |
| +8 | 1 | flags: 0, 0x0c, 0x1c or 0x24 (45,689 / 3,978 / 9 / 99 structs) | field: high; meaning: unknown |
| +9 | 1 | log2 of the alignment: 0-4 | medium: the size is a multiple of 2^n on 49,775 of 49,775 |
| +10 | 2 | 0 on every struct | high |
| +12 | 4 | 0 on every struct | high |
| +16 | 8 | fill, `aa` × 8 on every struct | high |
| +24 | 4 | byte size of the struct | high: every member offset is below it |
| +28 | 2 | 0 on every struct | high |
| +30 | 2 | member count | high: fills the section exactly |

The **layout signature** is a property of the layout, not the name. Across the corpus a struct name hash that
appears with two different signatures always has two different layouts and vice versa (2,170 of 2,170 names; 16
names occur in two layouts). Structs that share a signature under different names (103 signatures) have
identical layouts (offsets, types and counts) in every case. Its hash function is not known: it matches no joaat
of a harvested string.

### `LSCH` member (16 bytes)

The members follow the struct headers, each struct's members in a row.

| Offset | Size | Field | Confidence |
|---:|---:|---|---|
| +0 | 4 | name hash (case-sensitive joaat), or `0x100` for an array element descriptor | high |
| +4 | 4 | byte offset in the struct (0 for an element descriptor) | high: below the size on 324,119 of 324,119 |
| +8 | 1 | type code | field: high; per-type meaning: see below |
| +9 | 1 | sub-code; non-zero only for types 0x4d and 0x4f | field: high; meaning: unknown |
| +10 | 2 | element index: for an array (type 0x5c), the index in this struct's member list of its `0x100` element descriptor | high: 7,742 of 7,742 arrays point at a descriptor |
| +12 | 4 | reference hash: a struct or enum name for the types listed below, else 0 | high for 5 and the enum types |

Type codes and what is measured about them. *Gap* is the distance to the next member or to the struct end, the
most common value first; it bounds the member's size from above.

| Type | Members | Reference | Gap (most common) | Reading |
|---:|---:|---|---|---|
| 0x05 | 48,692 | a struct name: 45,275 in the same file, 2,663 elsewhere in the corpus, 754 unresolved | varies | embedded struct (high) |
| 0x5c | 7,742 | none; element index instead | 16 | array, element type in the descriptor (high); 16 bytes fits a pointer + count + capacity (hypothesis) |
| 0x60 | 1,032 | an enum: 1,031 same file, 1 corpus | 1 | 8-bit enum (medium) |
| 0x62 | 14,814 | an enum: 14,663 same file, 137 corpus, 14 unresolved | 4 | 32-bit enum (medium) |
| 0x65 | 7,394 | an enum on 7,392: 7,376 same file, 14 corpus, 2 unresolved | 8 | enum-typed bit set (hypothesis) |
| 0x63, 0x64, 0x68 | 7, 2, 24 | an enum | | enum variants (low) |
| 0x50, 0x71, 0x73, 0x75-0x77, 0x40, 0x53 | 3,577, 15,288, 15, 307, ... | a hash that is no struct or enum in the corpus | 0x71: 24 | unknown |
| 0x01 | 31,532 | - | 1 | 1-byte scalar (medium) |
| 0x11 | 2,105 | - | 1 | 1-byte scalar (medium) |
| 0x13, 0x20 | 5,136, 7,356 | - | 2 | 2-byte scalars (medium) |
| 0x14, 0x15, 0x21, 0x4a | 4,520, 5,381, 41,310, 79,876 | - | 4 | 4-byte scalars (medium) |
| 0x07, 0x30 | 27,455, 790 | - | 8 | 8-byte scalars (medium) |
| 0x31, 0x33, 0x52 | 1,081, 933, 14,267 | - | 16 | 16-byte values (medium) |
| 0x36 | 402 | - | 64 | 64-byte value (medium) |

The other 14 type codes occur fewer than 1,100 times each; `psin_index.py --schema` lists them per file. The
gap is an upper bound and the readings are hypotheses until a struct is matched against its instance data.

### `LENM` enum header (24 bytes) and value (16 bytes)

| Offset | Size | Field | Confidence |
|---:|---:|---|---|
| +0 | 4 | name hash; every enum is referenced by a member of the same file's `LSCH` (14,823 of 14,823) | high |
| +4 | 4 | layout signature: changes exactly when the value list changes (423 of 423 names; 5 names have two lists) | high |
| +8 | 8 | fill, `ee` × 8 on every enum | high |
| +16 | 4 | value count | high: fills the section exactly |
| +20 | 4 | 0 on every enum | high |

The values follow the enum headers, each enum's values in a row:

| Offset | Size | Field | Confidence |
|---:|---:|---|---|
| +0 | 8 | signed value; 13,487 of 14,823 enums are 0, 1, 2, ...; 619 have a negative value | high |
| +8 | 8 | name hash (case-sensitive joaat) in the low half; the high half is 0 on every value | high; unique within each enum (14,823 of 14,823) |

### Hashes

Names are the **case-sensitive joaat** (`world_names.joaat(text, case_sensitive=True)`), as for the engine
member hashes in [world_names.md](world_names.md). Measured against every identifier-like string in the two
carves (3,386,171 candidates) over the hash sets of the table above (24,753 values, 24,681 distinct):

| Function | struct names | member names | enum names | enum value names | signatures |
|---|---:|---:|---:|---:|---:|
| joaat, case-sensitive | 26 / 2,170 | 506 / 7,869 | 0 / 423 | 708 / 11,887 | 0 / 2,404 |
| joaat of the lower-cased string, not also case-sensitive | 0 | 4 | 2 | 4 | 0 |
| control: CRC-32 / CRC-32 xor a constant | | | | | 21 / 24 over all |

1,240 case-sensitive hits against a chance level of about 22 confirm the function; the low coverage only says
that the engine's type and field names are not in the data files. The signatures do not resolve to any name.
No name or hash from the game is committed here.

### Linking to the rest of the container

- Member references to struct names resolve inside the same `LSCH` for 45,275 of 48,692 struct members and to
  another file's `LSCH` for 2,663 more; enum references (types 0x60, 0x62-0x65, 0x68) resolve in the same file's `LENM` for 23,100 of 23,271 and
  elsewhere in the corpus for 155 more.
- The region between the 16-byte header and `LSCH` most likely holds the instance data these structs describe
  (hypothesis: in the records looked at it begins with struct name hashes from `LSCH`). The header's +12 field is at most the size of that region on 3,117 of 3,117 files;
  mapping the region onto the schema is not done here.
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
