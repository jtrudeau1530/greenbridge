# Story and minigame archives

This page measures the `story000` and `shared_minigame000` archive family in the Xbox 1.0.2.1
package. It is a container survey, not a resource decoder: the RPF9 table of contents is still
encrypted, so counts come from the data area and from fixed headers, never from names or paths.
Everything below is aggregate. No game payload, path or name is stored in this repository.

## Archives and index

Sizes are exact bytes from the package; `data offset` is the little-endian u32 at file offset 4.
A full index of `story000.rpf` was not completed (see the layout section); its headline row is the
first 1 GiB of the data area, and the full-file non-inflating signature scan is reported separately.

| Archive | File size | Data offset | Scope | Streams | Stored | Inflated | Zero padding | Unread | Readable | Elapsed |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| `story000.rpf` | 20,022,508,672 (18.65 GiB) | 0xA120 | first 1 GiB of data area | 55,284 | 1,002,413,457 | 3,963,061,696 | 67,544,007 | 3,803,929 | 93.4% | 44.0 s |
| `shared_minigame000.rpf` | 810,581,008 (773.03 MiB) | 0xFB0 | full | 5,073 | 443,771,951 | 928,713,043 | 4,723,814 | 362,081,227 | 54.7% | 24.5 s |
| `story000_script.rpf` | 78,023,728 (74.41 MiB) | 0x2B0 | full | 2,632 | 20,408,094 | 74,729,504 | 4,585,959 | 53,028,987 | 26.2% | 4.5 s |
| `story000_restofworld.rpf` | 198,354,992 (189.17 MiB) | 0xA40 | full | 631 | 188,163,116 | 255,343,231 | 862,330 | 9,326,922 | 94.9% | 1.0 s |
| `story000_bvh4.rpf` | 5,632 | 0x340 | full | 4 | 1,795 | 16,677 | 2,849 | 156 | 37.4% | 0.004 s |
| `shared_minigame000_bvh4.rpf` | 61,987,040 (59.12 MiB) | 0x4F0 | full | 2,560 | 59,142,787 | 970,044,012 | 2,672,073 | 170,916 | 95.4% | 2.65 s |

Readable is the carver's `stored / span`; Bink 2 streams count as readable. The first `story000`
row scans file offsets `0xA120`-`0x4000A11F`.

A full 20 GB walk was attempted. At 72% (about 14.4 billion bytes, 106,000 streams) the run was
stopped; one zlib candidate had expanded to 8.5 GB in memory. It is not reported as a full index.
The bounded windows below and the non-inflating signature scan cover the rest without that risk.

## `story000.rpf` layout

The archive is not uniform. A 64 MiB magic-count scan of the whole 20 GB file, plus the carver
samples below, separates four regions:

| Region | zlib carver result | Full-file signature result |
|---|---|---|
| GiB 0-4 | 93-100% readable in 1 GiB and 256 MiB samples; `0DRP` pages, `zero-prefix` pages, type-hash and clip headers | no RSC8 |
| GiB 5-13 | 0 streams in every sampled window: 5, 6, 8, 10 and 12 GiB | 225,281 `RSC8` headers, all version 8001, u32@8 sum 2,677,370,916 |
| GiB 14 | 2,547 streams / 268,670,234 stored / 472,245,792 inflated in a 256 MiB window; 780 `42 40 00 00` pages | no RSC8 |
| GiB 15-18 | Bink 2 only in the sampled windows; 42 streams at 16 GiB, 2 at 18 GiB | 264 valid Bink 2 streams, 3,327,650,728 bytes |

Window samples (each starts at the stated file offset and scans 256 MiB unless noted):

| Start | Streams | Stored | Inflated | Readable | Dominant class |
|---:|---:|---:|---:|---:|---|
| 0 (1 GiB) | 55,284 | 1,002,413,457 | 3,963,061,696 | 93.4% | `0DRP` 21,980; zero-prefix 22,681 |
| 2 GiB | 8,837 | 255,734,022 | 744,610,656 | 95.3% | `0DRP` 3,067 |
| 3 GiB | 476 | 268,264,094 | 341,274,256 | 99.9% | large zero-prefix pages |
| 4 GiB | 502 | 273,507,830 | 342,088,096 | 100% (crosses window) | large zero-prefix pages |
| 5 GiB | 0 | 0 | 0 | 0% | raw, no zlib |
| 6 GiB | 0 | 0 | 0 | 0% | raw, no zlib |
| 8 GiB | 0 | 0 | 0 | 0% | raw `RSC8` |
| 10 GiB (1 GiB) | 0 | 0 | 0 | 0% | raw `RSC8`; 25,685 headers in that GiB |
| 12 GiB | 0 | 0 | 0 | 0% | raw `RSC8` |
| 14 GiB | 2,547 | 268,670,234 | 472,245,792 | ~100% (crosses window) | `42 40 00 00` 780; zero-prefix 1,100 |
| 16 GiB | 42 | 277,746,976 | 277,746,976 | 100% | Bink 2 |
| 18 GiB | 2 | 208,322,272 | 208,322,272 | 77.6% | Bink 2 |

`RSC8` headers by 1 GiB bucket: 6 GiB 33,846; 7 GiB 41,391; 8 GiB 33,334; 9 GiB 29,687;
10 GiB 25,685; 11 GiB 25,983; 12 GiB 28,619; 13 GiB 6,736. The u32 at +8 sums to 2.49 GiB; the raw
region itself is about 7 GiB, so that field is not the total stored size of each resource. Every one
of the 225,281 headers has version 8001 (`0x1F41`); no `RSC7` header was found anywhere in the file.

Bink 2 by 1 GiB bucket: 15 GiB 78; 16 GiB 169; 17 GiB 10; 18 GiB 7. The 264 streams carry valid
`KB2[a-z]` headers and dimensions from 540x960 to 2560x1440. They account for 3.10 GiB of the
archive and are the only video class measured here.

## Readable payload classes

### `story000.rpf`, first 1 GiB

| Class | Streams | Stored | Inflated | Measured field |
|---|---:|---:|---:|---|
| `0DRP` page directory at +16 | 21,980 | 124,984,888 | 2,336,795,552 | version 122; page count u64 at +0x20 (top: 2 = 10,603, 3 = 5,996, 1 = 1,231, 4 = 646) |
| zero-prefix page | 22,681 | 737,279,203 | 1,234,168,960 | first 8 bytes zero |
| `80 80 80 80` buffer page | 3,594 | 43,479,838 | 203,792,288 | 16 bytes `0x80`, then zeros |
| type-hash header | 934 | 2,247,133 | 5,946,240 | u32 type, u32 8, u32 count, u32 0 |
| clip stream | 804 | 7,326,470 | 13,957,328 | u32 2 or 3, `01 01 00 00` |
| other raw starts | 5,291 (3,137 classes) | 87,095,925 | 168,401,328 | 3,036 classes occur once |

No `PSIN`, `RBF0`, `RSC7`, `RSC8`, `AWC`, `FSSM`, `SCRP`, `YSC` or valid `KB2` stream starts in
this 1 GiB. A prefix scan of all 55,284 inflated streams found no `RSC7`/`RSC8`, `DDS `, `DXT1`,
`DXT3`, `DXT5`, `ATI2`, `BC5`, `BC7` or `TXTR` magic in the first 4 KiB; the only `0DRP` matches
were the 21,980 page directories.

### `shared_minigame000.rpf`, full

| Class | Streams | Stored | Inflated | Measured field |
|---|---:|---:|---:|---|
| `42 40 00 00` resource page | 1,255 | 304,089,245 | 656,802,672 | 0x4042 marker; four u32s then float records |
| zero-prefix page | 1,782 | 104,870,151 | 188,050,712 | first 8 bytes zero |
| `0DRP` page directory at +16 | 787 | 6,251,535 | 19,538,880 | version 122; page counts 1 = 578, 2 = 192, 4 = 4, 6 = 3 |
| `PSIN` | 73 | 164,290 | 556,280 | 70 full six-section chains; 3 without a chain |
| `63 30 00 00` record page | 45 | 4,470,231 | 14,894,128 | float records, not decoded |
| XML text | 16 | 11,500 | 57,317 | UTF-8 BOM + `<?xml` |
| other text | 10 | 131,472 | 6,985,204 | printable NUL-terminated strings |
| raw `RSC8` header | 1,957 | - | - | version 8001; u32@8 sum 32,904,308 |

The 362 MB unread region is not empty: a 360 MB gap holds five page-aligned nested `RPF9`
headers and the 1,957 `RSC8` headers; the archive has 49 nested headers in total. The `RSC8`
fixed header is 0x30 bytes; a NUL-terminated name follows it before a pointer into the
`0x50xxxxxx` range. The name field was present in the sampled headers. No `RSC7`, `AWC`, `FSSM`,
`SCRP` or `YSC` container was validated.

### `story000_script.rpf`, full

`PSIN` dominates: 2,309 of 2,632 streams, 14,484,253 stored / 59,151,216 inflated. The 16-byte
header is:

| Offset | Field | Measurement |
|---:|---|---|
| +0 | `PSIN` | magic |
| +4 | big-endian first-section offset | points at `LSCH` in 2,173 of 2,309 script streams and 70 of 73 minigame streams |
| +8 | little-endian u32 | 261 on 1,771 script streams, but 77/265/264 and larger values elsewhere; not stable in minigame, so this page does not label it a version |
| +12 | little-endian u32 | 1,568 on 1,227 script streams; not interpreted here |

The section chain follows the measured layout:

```
+0  "PSIN"
... section records: 4-byte tag, big-endian u32 length (includes the 8-byte record header)
```

The chain is contiguous. In one 22,952-byte stream it runs from +0x2EE0 to +0x59A8:

| Section | Length | Next section |
|---|---:|---:|
| `LSCH` | 0x14C8 | +0x43A8 |
| `LENM` | 0x358 | +0x4700 |
| `LSNS` | 0x1234 | +0x5934 |
| `LGID` | 8 | +0x593C |
| `LRSC` | 0x58 | +0x5994 |
| `CHKS` | 0x14 | end |

Chain completeness: 2,118 streams have all six sections; 38 have `LSCH` and `LENM`; 4 have
`LSCH` only; 149 have no parseable chain. The other 323 streams are pages and raw starts.
No `SCRP` or `YSC` container appears, so the script archive's metadata is `PSIN`, not a named
script container.

The 53 MB unread region contains four nested `RPF9` headers (three in a 51.7 MB gap). Their data
is high-entropy and shows no `PSIN`, `RSC8`, `0DRP` or `SCRP`/`YSC` header. That payload is opaque
to this pass; it is not guessed to be bytecode.

### `story000_restofworld.rpf`, full

631 streams, 94.9% readable. Bink 2 is the bulk: 7 streams, 120,187,756 bytes, dimensions
720x1280 (5) and 1920x1080 (2). The rest is 101 `0DRP` pages (version 122), 400 zero-prefix pages,
3 `PSIN`, 13 XML, 3 text and 3 clip streams. Nine `RSC8` headers (version 8001) sit in an 8.96 MB
unread gap; the archive has 18 nested `RPF9` headers. This is the video/extra-content companion to
the story archive.

## `_bvh4` cross-check

| Archive | Streams | Inflated | Content | Overlap with main |
|---|---:|---:|---|---:|
| `story000_bvh4.rpf` | 4 | 16,677 | 4 XML streams (2 ids); no binary BVH payload | 0 of 4 SHA-256 |
| `shared_minigame000_bvh4.rpf` | 2,560 | 970,044,012 | 2,555 binary zero-prefix pages + 5 XML streams (3 ids) | 0 of 2,560 SHA-256 |

The XML streams identify themselves as environment-BVH change sets and carry 61 file-like entries
in `story000_bvh4` and 9 in `shared_minigame000_bvh4`. One filename string is common to both, but
the inflated stream sets have **no common SHA-256**. The binary pages start with zeroes, `0x50xxxxxx` pointers and identity matrices,
consistent with BVH node pages.
These are independent BVH archives, not empty indexes and not shared page sets. `story000_bvh4` is
metadata-only; `shared_minigame000_bvh4` has the payload.

The main-archive comparison scope was the first 1 GiB of `story000.rpf` and all of
`shared_minigame000.rpf`; zero overlaps at those scopes do not prove zero overlap with the
remaining 19 GB.

## Signatures actually observed

| Magic | Offset | Measured fields | Where |
|---|---|---|---|
| `9FPR` (RPF9 little-endian) | +0 | data offset u32 +4; next u32 +8; third u32 +12 | 442 headers in `story000`, 50 in minigame, 5 in script, 19 in restofworld, including the outer archive |
| zlib `78 01/5e/9c/da` | +0 | inflates cleanly to stream end | 93.4% of the first `story000` GiB; 54.7% of minigame |
| `PSIN` | +0 | section chain with big-endian lengths | 2,309 in script; 73 in minigame; 3 in restofworld |
| `0DRP` | +16 | version 122 at +0x14; page count u64 at +0x20 | 21,980 first `story000` GiB; 787 minigame; 101 restofworld |
| `RSC8` | +0 | version 8001 at +4; u32 at +8; flags u32 at +12; name at +0x30 | 225,281 in `story000`; 1,957 minigame; 9 restofworld |
| `RSC7` | +0 | - | 0 in every full-file scan |
| `KB2[a-z]` | +0 | size u32 +4; width +0x14; height +0x18 | 264 in `story000`; 7 in restofworld |
| `42 40 00 00` | +0 | 0x4042 marker; u32s and float records | 1,255 minigame; 780 in the 14 GiB `story000` window |
| `RBF0` | +0 | schema string | 0 at stream starts; 2 raw occurrences in 20 GB (not container starts) |
| `SCRP`, `YSC` | +0 | - | 0 validated |
| `AWC\0`, `FSSM` | +0 | - | 0 validated |
| XML | +0 | UTF-8 BOM + `<?xml` | 16 minigame; 13 restofworld; 4 + 5 in the bvh4 archives |

The `RSC8` version is the same in every validated header: `0x1F41` (8001). The `0DRP` version is
the same in every inflated page: `0x7A` (122). No format is inferred beyond these bytes.

## Readable versus opaque

Readable: zlib streams, Bink 2 streams, XML/text streams, `PSIN` section headers, `0DRP` page
headers and `RSC8` fixed headers. Opaque: encrypted RPF9 tables of contents, the body of each
`RSC8` resource, the nested high-entropy payloads in the script archive, and the raw region of
`story000` that has no zlib stream.

"Unread" in the index tables means bytes the zlib walk could not inflate. It is not a claim that
those bytes are empty or unimportant; in minigame and restofworld they contain `RSC8` headers and
nested RPF9 archives.

## Likely format family

The bytes support a **new RAGE resource family in an RPF9 container**:

- RPF9 data areas are zlib streams, with page-aligned embedded RPF9 sub-archives (437 of the 442
  `story000` headers are page-aligned).
- `0DRP` is a page header, not a GTA V `RSC7` header: magic at +16, version 122, page count at
  +0x20, and payload pages after it.
- `RSC8` is the resource header in the raw story region: version 8001, a u32 at +8, flags and a
  name string. No `RSC7` header was found in 20 GB.
- `PSIN` is a sectioned metadata container, not a script opcode stream. Repository notes already
  describe it as PSO metadata; this pass confirms the section names and the big-endian length
  chain, not the section semantics.
- Bink 2 video is present and separate from the resource pages.

No `RSC7`, `SCRP` or `YSC` container was observed. Models and textures are not claimed from a
magic here: prior model recovery already reports 3,303 models in `story000` and 783 in
`shared_minigame000` (`docs/design.md`), which is consistent with the `0DRP` page population, but
this page does not decode drawables or textures. No DDS, DXT1/3/5, ATI2, BC5, BC7 or TXTR magic
was found in the inflated sample.

## What a decoder needs

1. The RPF9 TOC key/schema. This project does not search for or distribute keys; without it,
   names, paths and exact entry boundaries remain unavailable.
2. A recursive RPF9 walker. The flat zlib carve does not group streams by the 441 additional RPF9
   headers in `story000`; a decoder needs their spans and TOCs to group resources.
3. An `RSC8` header parser and a type map for version 8001. The fixed 0x30-byte header and the
   following name are measured; the payload is not decoded.
4. A `PSIN` section walker. The tag/length chain is measured; `LSCH`, `LENM`, `LSNS`, `LGID`,
   `LRSC` and `CHKS` semantics are still open.
5. `0DRP` page grouping. The page count at +0x20 is the only measured grouping field; a decoder
   still needs the following stream boundaries.
6. Bink 2 decoding for the 264 story streams and 7 restofworld streams (see `docs/movies.md`).
7. The existing model exporter for drawables; texture identification needs a real container
   signature, which this pass did not find.

## No-sharing rule

Do not commit archives, carved streams, inflated payloads, asset names, path strings, hashes of
assets or name dumps. This page and the test contain aggregate counts and synthetic bytes only.
Keep carves in a local work directory, never in the repository or a PR. The tool writes offsets,
sizes, signatures and header fields; it does not write resource payloads or carved bytes.

## Reproduce

```powershell
python -I tools/story_index.py <archive.rpf> <out.csv|out.json> [--start N] [--limit BYTES]
python -I tools/rpf9_carve.py <archive.rpf> <out_dir> --index-only [--start N] [--limit BYTES]
python -I tests/test_story_index.py
```

`story_index.py` is stdlib-only. It classifies the same zlib/Bink walk as `rpf9_carve.py`, parses
the `PSIN` section chain, and reports `0DRP`/`RSC8`/`9FPR` headers when they start a stream or a
carved file. It does not recurse into the raw embedded archives.
