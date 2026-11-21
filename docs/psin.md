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
| `LSCH` | 670 | 464-544 bytes; a header/hash table, not decoded |
| `LENM` | 670 | name/hash metadata, not decoded |
| `LSNS` | 670 | high-entropy body (8,608 bytes in the sampled record), not zlib/LZ4; not decoded |
| `LGID` | 670 | empty (8-byte section) |
| `LRSC` | 670 | starts with `0DRP` + version 122; resource page directory link |
| `CHKS` | 670 | file size and a 32-bit checksum |

The `LSCH` sizes on the measured set are 472-552 bytes including the 8-byte header. `LRSC` is the
link to the resource-page world: its payload begins `30 44 52 50` (`0DRP`), then a little-endian 122,
then the page fields used by the generic RPF pager. `CHKS` carries the file size in big-endian and a
32-bit hash; the tool reports the section, not a verified checksum.

## What it does not give

- No plaintext script code, function names or string table. `LSNS` is high-entropy and is not
  decoded; the measured body does not decompress with zlib (15/-15/31) or raw LZ4.
- No station/track names, no model names, no file paths. The archive TOC is still the only source
  for original names.
- No semantics for the `LSCH`/`LENM` records. The tool indexes the container and the section sizes;
  it does not claim to decode the records.

## Tool

```text
python -I tools/psin_index.py <PSIN file or carve directory> [--json] [--limit N]
```

- `parse(blob)` returns the magic, version, table size, the ordered section list
  (`tag`, `offset`, `size`, `payload`) and a `complete` flag with the first structural error. When the
  `LRSC` payload starts with `0DRP`, the `0DRP` version is reported; `CHKS` contributes the file size
  and a 32-bit checksum. Text output and the summary include those fields.
- `collect(src)` scans a single file or every `.bin` in a carve directory; non-PSIN files are
  skipped.
- Text output prints one line per container and a tag summary; `--json` writes `{files, summary}`.
- The parser validates the first-section offset, each tag's printable range, each section size and
  the exact EOF; a 64 KiB split is reported as incomplete rather than guessed.

`tests/test_psin_index.py` covers complete records, an empty section, a bad first offset, a zero-size
section, truncated data, trailing bytes and the JSON CLI. It uses synthetic headers only.

## Limits

- The 64 KiB split is not joined automatically. A caller that has the archive order can join adjacent
  blocks first; the tool then parses the complete container.
- `LSCH`, `LENM`, `LSNS` and `CHKS` payloads are opaque to this tool. The measured container layout
  is enough to index the files and measure coverage, not to run the metadata.
- No game data, names, hashes or payloads are committed. The carve directories are local.
