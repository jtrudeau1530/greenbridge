# Navmesh archive (`navmeshes.rpf`)

**Status: measured container. No navmesh decoder.** The bytes identify an RPF9 page-directory container and its
page classes; they do not name the navmesh structures. Every number below is aggregate metadata from a read-only
scan of one Xbox Series 1.0.2.1 package. No payload, key, name or carved file is in this document or the
repository.

## Method

- The package archive is **6,707,512,496 bytes**; its RPF9 data area starts at `0x780`, leaving a span of
  6,707,510,576 bytes. (An earlier size estimate of 6,705,712,496 was 1,800,000 bytes low; the on-disk size
  above is what was measured.)
- Indexed in full with the fixed `tools/rpf9_carve.py` (`--index-only`, advance by bytes actually read, not by
  the 1 MiB chunk size). The encrypted table of contents was not touched.
- The same walk was repeated for a full-stream signature census, a full header-class census and a bounded
  `levels.rpf` cross-check. `data.rpf` was indexed in full.
- Index time: 162 s. Host: Windows, 28 logical CPUs, 34 GB RAM.

## Composition

| Metric | Value |
|---|---:|
| File bytes | 6,707,512,496 |
| Scanned span (file minus `0x780`) | 6,707,510,576 |
| zlib streams | 90,295 |
| Stored (compressed) bytes | 6,540,238,392 (97.5% of span) |
| Inflated bytes | 20,551,068,624 (3.14x stored) |
| Zero padding | 161,191,191 (2.4%) |
| Unread (non-zero, non-stream) | 6,080,993 (0.1%) |
| Bink 2 streams | 0 |

Streams fall into four measured classes. The class names describe the header bytes, not a game asset type.

| Class | Streams | Stored bytes | Inflated bytes | Median inflated |
|---|---:|---:|---:|---:|
| `0DRP` directory page (`0DRP` at +16) | 9,603 | 745,601,504 | 2,764,382,624 | 196,608 |
| typed data page (`u32 0x00010015` at +0x14) | 27,644 | 5,346,712,784 | 15,281,523,008 | 131,072 |
| `PSIN` page (`PSIN` at +0) | 22 | 628,344 | 1,413,376 | 65,536 |
| other (small pages/tables) | 53,026 | 447,295,760 | 2,503,749,616 | 2,656 |
| **total** | **90,295** | **6,540,238,392** | **20,551,068,624** | |

`0DRP` is concentrated: 9,604 of the 9,624 streams whose first MiB contains the bytes start in the first
768 MiB (20 elsewhere). A full-stream census finds 9,625; 9,603 have the header at +16. Data pages dominate the
remaining ~5.9 GiB. 27,220 streams inflate to exactly 131,072 bytes (128 KiB), and 53.6% of all stream sizes are
a multiple of 4 KiB, so pages are allocated in 128 KiB-class units.

## Directory page layout

Measured on all 9,603 `0DRP` pages for the counted fields, plus a 255-page sample for the rest. Offsets are from
the start of the inflated stream.

| Offset | Size | Observed | Meaning |
|---:|---:|---|---|
| +0x00 | 8 | zero | page prefix |
| +0x08 | 8 | pointer-like `0x0000000050xxxxxx` | inside the page after subtracting `0x50000000` (all samples) |
| +0x10 | 4 | `30 44 52 50` (`0DRP`) | directory magic |
| +0x14 | 4 | 122 (`0x7A`) in all 9,603 | version |
| +0x18 | 8 | zero | |
| +0x20 | 4 | 1-6; 3 in 7,650, 4 in 932, 2 in 779, 6 in 219, 5 in 22, 1 in 1 | presumed page count |
| +0x24 | 4 | zero | |
| +0x28 | 32 | four pointer-like qwords in the sample | same base; referenced offsets resolve inside the page |
| +0x48 | 16 | eight u16 values, varying | not decoded |
| +0x60 | 8 | opaque 64-bit key | 9,409 nonzero, 6,760 distinct nonzero values, 3,101 in duplicate groups, 194 zero |
| +0x68 | 4 | 0 to 4,294,967,289; 1,381 zeros | not a dense index; exposed raw as `u32_68` |
| +0x6c | 4 | zero | |
| +0x70 | varies | some pages repeat key / u32 / `0xffffffff` | not decoded |

The key is not a readable name and is not unique, so it is only a join key. A future name table must come from
the RPF9 TOC or the executable.

Typed data pages share a different prefix: eight zero bytes, a `0x50xxxxxx` pointer at +0x08, a small u32 at
+0x10 (9 in 18,791, 11 in 5,507, 3 in 3,149) and `0x00010015` at +0x14. Its low half is `0x15` (21); the word is a
version/flags candidate, not a magic. These pages are 128 KiB median and carry packed u16/float arrays.
The `AABB` and `NVMS` byte runs below occur inside those arrays.

## Signatures: what survives

Full-stream census; counts are streams containing the exact bytes at least once.

| Bytes | Position | Streams | Verdict |
|---|---|---:|---|
| `30 44 52 50` (`0DRP`) | +16 | 9,603 | **real header magic**, RPF9 directory page |
| `0DRP` | elsewhere in stream | 22 | coincidence inside data; not a header |
| `50 53 49 4e` (`PSIN`) | +0 | 22 | real header; fixed 64 KiB page; also in data/levels, not navmesh-specific |
| `41 41 42 42` (`AABB`) | payload | 86 | rejected: one hit/stream; sampled contexts are packed arrays, not headers |
| `4e 56 4d 53` (`NVMS`) | payload | 9 | rejected: one hit/stream inside u16 arrays; no fields |
| `50 44 52 30` (`PDR0`) | payload | 1 | rejected: one hit, no header fields |
| `RSC7`, `RSC8` | any | 0 | absent; any resource header would be in the encrypted TOC, which was not read |
| `NAVM`, `CNavMesh`, `NISP` | any | 0 | absent |
| `RBF0` | any | 0 | absent here; 1,263 streams in `data.rpf` (a named container there) |

The short `AABB`/`NVMS` counts are why this report does not name a navmesh magic. A four-byte run is not a
signature until its context and fields survive inspection; these did not.

## Readable versus opaque

- The only readable header token in a navmesh page is `0DRP`; `PSIN` pages carry their own four bytes.
- A first-1 MiB string scan found no `navmesh`, `portal`, `vehicle` or `region`. Short `nav`/`ped`/`poly` hits were
  not part of readable strings. ASCII runs in large pages are mostly packed float/u16 patterns.
- No file names, asset names or readable resource types are present. The 64-bit key and the `u32_68` word are
  opaque.
- 78,340 of 90,295 streams begin with at least eight zero bytes; the rest are mostly small tables, not text.

## Cross-check: `data.rpf` and `levels.rpf`

| Archive | Scope | Streams | `0DRP` at +16 | Version | `PSIN` at +0 | `RBF0` at +0 |
|---|---|---:|---:|---:|---:|---:|
| `navmeshes.rpf` | full | 90,295 | 9,603 | 122 | 22 | 0 |
| `data.rpf` | full | 28,509 | 15,768 | 122 | 2,281 | 1,263 |
| `levels.rpf` | first 2 GiB | 3,186 | 2,856 | 122 | 8 | 0 |

`0DRP` is therefore the generic RPF9 directory page, not a navmesh marker. It also changes shape by archive:
the +0x20 count is 1-3 in `data.rpf` (one outlier 89), 1-6 in `navmeshes.rpf` (3 or 4 in 8,582) and 3-6 in the
`levels.rpf` prefix. In `data.rpf`, 17,978 streams contain `0DRP` somewhere but only 15,768 have it at +16, so
the bytes also occur inside payloads. `levels.rpf` directory pages are spread across every 256 MiB window,
unlike the navmesh archive's first-768-MiB cluster.

This answers the cross-check only at container level: navmesh resources may be embedded in level pages, but the
bytes cannot prove it. No navmesh-specific payload marker was found, and `0DRP` alone cannot attribute a page to a
navmesh. Names and resource types from the TOC are required for that count.

## Format family

Supported by bytes:

- RPF9 archive with per-page zlib streams (magic bytes `39 46 50 52`, read as `RPF9`; data start word at +4).
- RAGE-style resource paging: a `0DRP` directory page (version 122) plus typed data pages; sampled pointers
  resolve inside the page after subtracting `0x50000000`.
- 128 KiB-class page allocation and a `0x00010015` word on the dominant data page flavour.

Not supported by bytes:

- A navmesh-specific magic. `NAVM`, `CNavMesh` and `NISP` are absent; `AABB`/`NVMS` are coincidences.
- Polygon, portal or AABB-table layouts. None was decoded, and the archive name is the only evidence that these
  pages are navmeshes.
- Readable names or a resource-type table.

## What a future decoder needs

1. **The RPF9 TOC key and reader.** It supplies names, resource types and exact page ownership. This repository's
   rule is to record that the TOC is encrypted and stop; no decryption or protection work is in scope.
2. **The executable's navmesh loader.** Its runtime types and field layouts (the `CNavMesh` name is a candidate,
   not a byte finding), the meaning of the +0x20 count and the +0x28 pointers, and where polygons, vertices,
   portals and any AABB tree begin. Bytes alone do not identify them.
3. **A join from the 64-bit key to names.** The 9,603 directory pages carry 9,409 nonzero keys but only 6,760
   distinct nonzero values; duplicates and 194 zero keys must be explained before the key can identify a tile.
4. **Data-page parsing.** Start with the `0x00010015` pages, 128 KiB units and the `0x50000000` base; validate
   every pointer against the inflated page before trusting it.
5. **Corpus-wide checks.** Re-run the tool on other versions/archives and compare class counts; a layout change
   will show up as a different +0x14 word or page-size mix.

## Tool

`tools/navmesh_index.py` is stdlib-only and performs the same fixed walk. It emits one row per `0DRP` directory
page with offset, stored/inflated size, signature, version, presumed page count, 64-bit key and raw `u32_68`.

```powershell
python -I tools/navmesh_index.py <package>\navmeshes.rpf --out navmesh-index.json --csv navmesh-index.csv
python -I tools/navmesh_index.py <carve-dir> --pattern *.bin --out carved-index.json
python -I tests/test_navmesh_index.py
```

Validation on the first 64 MiB of the real archive: 1,378 streams, 1,132 directory pages, all version 122,
1,077 keys present, about 2 s. A full run produced 9,603 rows (745,601,504 stored / 2,764,382,624 inflated) in
132 s, matching the independent census. The synthetic test builds its own RPF9 header and pages; it needs no
game files.

## No-sharing rule

The package is read-only. Do not commit, upload or share the archive, any carved page, a key, an extracted name,
or a report that quotes payload bytes. Only aggregate counts, offsets and four-byte magics belong in this
repository.
