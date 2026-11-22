# Game data in `common.rpf`

This note measures the `common.rpf` archive from the Xbox Series 1.0.2.1 package. It records aggregate
counts, sizes and schema/root names only. No archive, carved file, XML value, entry name or per-file path is
committed. Everything below comes from a local copy and can be reproduced on any copy of the same version.

`common.rpf` is 6,642,192 bytes. Its RPF9 header is `RPF9` (`9FPR` little-endian) and the data area starts at
byte 38,272 (`0x9580`). The table of contents is encrypted/opaque; nothing here decrypts it or looks for keys.
The measurements cover the data area only.

## How to carve it

Use the carver with the short-final-slice fix (advance by the bytes actually read, not by the 1 MiB request):

```powershell
python -I tools/rpf9_carve.py common.rpf <work>\carve\common
```

Its `index.json` has `off`, `stored`, `size`, `pre` and `head` per stream. The `.bin` files are named by
offset and are **zlib streams, not files**. RPF9 splits a large file into 64 KiB inflated blocks, each with
its own zlib stream. A logical XML document is the concatenation of consecutive streams until its root element
closes; 178 of the 257 header-bearing documents are one stream, and the largest document spans 218.

`tools/gamedata_index.py` groups those blocks into payloads and writes an offset/size/type/root index:

```powershell
python -I tools/gamedata_index.py <work>\carve\common --csv common.csv --json common.json
```

The CSV/JSON contain offsets, sizes, the coarse type and the XML root element. They contain no element values
or entry names, but they are still local work products and must not be committed or attached to an issue.

### Coverage

| Measure | Value | Share of data span |
|---|---:|---:|
| Data span (file size minus header) | 6,603,920 B | 100.0% |
| zlib streams | 954 | - |
| Stored stream bytes | 5,647,255 B | 85.5% |
| Inflated bytes | 50,231,123 B | 8.9x the stored bytes |
| Zero padding | 386,138 B | 5.8% |
| Unread bytes | 570,527 B | 8.6% |

The earlier 781-stream / 39.5 MB / 87.3% figure came from the carver that advanced by the 1 MiB read request.
It counted skipped regions as part of the preceding stream, so its stored-byte total was too high and its
inflated total was too low. The fixed carver finds 173 more streams and 10.7 MB more data; its stored total is
lower because it measures each zlib stream's real end. Treat the old figures as superseded.

The 8.6% unread region is the non-zero gaps between streams: high-entropy bytes that do not decode as complete
zlib streams. Some contain a zlib header but fail the Adler check or are truncated. This is why 9 XML documents
survive only from their closing half and one 64 KiB XML block has no recoverable root. The carve does not guess
their content.

## Type split

Stream-level split, using the requested classes. `other` is 2 `MHIU` payloads (38 streams), 1 `PAMH` and 1
`SPS1Z`; there are no `RBF0`, `PSIN` or Bink 2 streams in this archive.

| Type | Streams | Stored B | Inflated B | Logical payloads |
|---|---:|---:|---:|---:|
| XML (`<?xml` / `<`) | 844 | 3,764,523 | 43,284,925 | 267 (266 docs + 1 fragment) |
| `RBF0` | 0 | 0 | 0 | 0 |
| `PSIN` | 0 | 0 | 0 | 0 |
| text / INI-like | 4 | 30,162 | 142,036 | 3 |
| binary | 66 | 1,396,087 | 4,312,136 | 2 |
| other (`MHIU`/`PAMH`/`SPS1Z`) | 40 | 456,483 | 2,492,026 | 4 |
| **total** | **954** | **5,647,255** | **50,231,123** | **276** |

By inflated bytes: 86.2% XML, 8.6% binary, 5.0% `other`, 0.3% text. XML is only 66.7% of stored bytes,
which is why a 6.6 MB archive inflates to 50.2 MB. XML grouping is exact because each document closes on its
root element. Non-XML grouping follows the 64 KiB block rule; none of the nine non-XML payloads here has a size
that is an exact multiple of 64 KiB, so their counts are unambiguous too.

The four non-XML structured payloads are the only readable binary metadata here: two `MHIU` blobs (about
1.19 MB and 1.21 MB inflated), one `PAMH` blob (48.6 KB) and one `SPS1Z` shader-parameter table (43.8 KB).
The two large opaque blobs (1.04 MB and 3.27 MB) have no recognised header and are left as binary. No model,
texture or audio signature is present in `common.rpf`; the two opaque blobs are not identified.

## MHIU and PAMH arrays

`tools/gamedata_maps.py` decodes both map payloads. It reads a carve directory, groups the 64 KiB
streams and prints or writes header fields and grid dimensions only; the grid bytes stay local.

```powershell
python -I tools/gamedata_maps.py <work>\carve\common --json maps.json
```

Both payloads start with a 44-byte header. The checksum field is not CRC32 or Adler32 of the payload;
the algorithm is still unknown.

| Header field | `MHIU` | `PAMH` |
|---|---|---|
| `magic` | 0x00 | 0x00 |
| `version` / second u16 | 0x04 / 0x06 | 0x04 / 0x06 |
| checksum | 0x28 | 0x08 |
| width, height (`u16` each) | not in the header | 0x0C, 0x0E |
| six `float32` bounds | 0x0C | 0x10 |
| data size | 0x24 | 0x28 |

`PAMH` stores its grid dimensions in the header. Its data is `height` records of eight bytes
`{u16 reserved = 0, u16 width, u32 row * width}`, followed by `width * height` little-endian `u16`
values. The measured payload is 144 x 164, with 164 row records and 23,616 cells.

`MHIU` does not store a grid edge. Its data is a zero-filled prefix followed by a square byte grid
that ends at the payload end; the reader takes the largest square with an all-zero prefix. The two
measured payloads resolve to 1090 x 1090 after a 1,027-byte prefix and 1100 x 1100 after a 432-byte
prefix. The role of that prefix is not decoded.

The grids are stored as raw normalized values. A mapping from the values to the world Z bounds is
possible but not confirmed, so the reader reports the arrays and dimensions without interpreting
them as heights.

## XML documents

257 documents start with an XML declaration and all 257 parse as well-formed XML with a standard parser.
Nine more complete documents are visible only from their closing root tag because their first block was in
the unread region; 2 of those 9 still parse from their first retained byte, and 7 are truncated at the start.
One 64 KiB XML block has no root at all. That gives 266 recoverable documents and 118 distinct root elements.

Size distribution of the 257 header-bearing documents: minimum 88 B, median 13,435 B, mean 165,603 B,
maximum 14,262,150 B. 59 are at most 1 KiB, 178 are at most 64 KiB, 3 exceed 1 MiB and the largest spans 218
streams. 41 start with a UTF-8 BOM; 255 declare UTF-8 (214 uppercase, 41 lowercase), 2 omit the encoding
attribute, 11 say `standalone="yes"`, and the 412 comments are hand-edit notes.

### Root elements

| Root element | Documents |
|---|---:|
| `timecycle_modifier_data` | 49 |
| `CVehiclePoints` | 32 |
| `timecycle_keyframe_data` | 27 |
| `CDataFileMgr__ContentChangeSetDef` | 23 |
| `CTuningFile` | 10 |
| `CDamageDataManager` | 5 |
| `GridAtlasInfo` | 4 |
| `CCoverProcessorTuning` | 3 |
| `CDataFileMgr__ContentDef` | 2 |
| `CVehiclePerformanceMaps` | 2 |
| `CVehicleBoneOffsetsArray` | 2 |
| 107 other root elements | 107 |
| **total** | **266** |

The repeated roots matter: a root element alone does not identify a file. There are 49 timecycle modifier
documents, 32 vehicle-point documents and so on. Original paths and names would have to come from the missing
table of contents.

### Functional categories

The 118 root elements group into these categories. Counts and bytes cover the 266 recoverable documents
(41.2 MiB), not the one unattributed block.

| Category | Files | Inflated B | Roots | Representative root elements |
|---|---:|---:|---:|---|
| time, weather, sky, water | 87 | 17,339,011 | 13 | `timecycle_modifier_data`, `timecycle_keyframe_data`, `CContentsOfWeatherXmlFile`, `ShoreWavePlacementData` |
| vehicles, tyres, handling data | 45 | 19,618,605 | 12 | `CVehiclePoints`, `CVehicleBoneOffsetsArray`, `CTyreDataSet`, `CVehiclePerformanceMaps` |
| peds, schedules, appearance | 29 | 2,029,321 | 29 | `CPedScheduleInfoManager`, `CPedVariationBodyTypeManager`, `CPairedPedDataBlock`, `CEnergyConfigInfos` |
| tuning, parameters, effects | 28 | 343,601 | 14 | `CTuningFile`, `CCoverProcessorTuning`, `GridAtlasInfo`, `PatchConstants` |
| mount / content manifests | 25 | 450,316 | 2 | `CDataFileMgr__ContentChangeSetDef`, `CDataFileMgr__ContentDef` |
| world, navigation, map | 19 | 1,660,133 | 19 | `GameShape__Data__LocationNameLoopkup`, `PathingStyleData`, `CTrainConfigs`, `CHazardNodes` |
| combat and AI behaviour | 13 | 809,074 | 13 | `CCombatInfoMgr`, `CFiringPatternInfoManager`, `CFleeDecisionMakerData`, `Combat__GoalData__Store` |
| animation, poses, weapons | 6 | 162,778 | 6 | `WeaponAnimations__FullbodyControllerData`, `WeaponAnimations__ConfigData`, `CAimUpperBodyDataBlock` |
| damage | 5 | 485,482 | 1 | `CDamageDataManager` |
| audio, video, speech | 4 | 13,741 | 4 | `LandingPageVideos`, `CTVPlaylistContainer`, `naSpeechEventTypeBatchRequestParams` |
| missions, scenarios, respawn | 3 | 57,327 | 3 | `CDynamicScenarioManager`, `CMissions`, `Respawn__MetaData` |
| conversation | 2 | 250,000 | 2 | `CConversationList`, `CConversationDictionary` |
| **total** | **266** | **43,219,389** | **118** | |

This archive has no root element for popcycle, weapon stats or handling. Weapon-related documents here cover
loadouts, combat references and animation configuration. Vehicle handling-style values appear under the
vehicle/tyre roots, and the remaining gameplay tables are not XML. Do not infer a full gameplay schema from
`common.rpf` alone.

## Schema style

Four patterns cover almost everything:

1. **Mount/content manifests.** `CDataFileMgr__ContentChangeSetDef` has `<id>`, optional registration/map
   fields, a `<fileList>` of `<Item>` with `<filename>` and `<fileType>`, plus flags and map phases.
   `CDataFileMgr__ContentDef` is the companion content description. There are 25 such documents and 3,315
   `<filename>` elements in total. The path tokens are `$(common)`, `$(target)` and `platform:/`; no XInclude
   is used.
2. **Typed object graphs.** `CCombatInfoMgr` and the conversations/damage roots use `<Item type="...">`.
   Fields are usually child elements, scalars use `value="..."`, and enums are bare text. Across the 259
   parseable documents there are 428,611 elements, 5,293 distinct tag names, 318 distinct `type` values
   (7,181 `type` attributes) and 33 distinct attribute names. The most common are `value` (84,585), `x`/`y`
   (20,825 each), `content` (13,449), `key` (9,180), `type` (7,181), `name` (4,688) and `ref` (924).
3. **Compact numeric tables.** `CVehicleBoneOffsetsArray` stores `<packedIndices content="short_array">`
   and `<palette content="float16_array">` as whitespace-separated numbers, with `numEntries` and
   `bitsPerIndex`. `content` values seen: `short_array` (10,309), `float16_array` (2,481),
   `float_array` (536), `matrix34` (46), `vector2_array` (40), `matrix33` (28), `vec3v_array` (9).
   `CTyreDataSet` uses nested groups of named floats. These tables are why the largest single document is
   14.26 MB and spans 218 streams.
4. **Timecycle arrays.** `timecycle_modifier_data` has a version root attribute, `<modifier>` records with
   name/description/apply-tag/index/user-flag attributes, and parameter elements whose text is a
   space-separated vector (`a b`, sometimes `a b c d`). Keyframe documents add `<cycle>`/`<region>`/`<axis>`.
   This is not the `name=value` attribute style of pattern 2.

A runtime consumer needs:

- a UTF-8 XML parser that tolerates a BOM, CRLF, tabs, comments and inconsistent whitespace around `=`;
- a root-element dispatch table (118 roots, 5,293 tag names), or an RTTI/class-metadata bridge if the
  executable's class names are available;
- a polymorphic `Item` type registry for `type=`;
- enum and symbolic-reference resolution: this archive uses symbolic `ref`/`key`/`name` strings
  (924 `ref`, 9,180 `key`); only 28 `key` values look like 8-digit hex hashes, so a name registry matters more
  than a joaat table here;
- whitespace-array and float16/packed-index decoders for the compact vehicle tables;
- a mount resolver for `$(common)`, `$(target)` and `platform:/` prefixes;
- streaming rather than whole-file parsing for the 14 MB table;
- strict handling of the 9 tail-only and 1 unattributed blocks: they are incomplete evidence, not documents
  to guess at.

## XML catalog

`tools/gamedata_xml.py` turns a carve into a per-file and per-root catalog. It is stdlib-only and rebuilds
multi-stream documents with `gamedata_index.py`, so it reads the same 267 XML payloads the index reports:

```powershell
python -I tools/gamedata_xml.py <work>\carve\common --json xml.json --csv xml.csv --check
```

The file rows carry offset/bytes/chunks, root, category, element count, maximum depth, attribute names, `$(name)`
and `platform:/` counts, include-like counts and the well-formed/fragment flags. The root rows add the
element-name and attribute histograms and the reference targets. `--check` separates a rootless fragment, a
tail-only document and a malformed document instead of treating all three as parse failures.

Measured over the 267 payloads (43,284,925 B):

| Measure | Value |
|---|---:|
| Well-formed | 259 (257 with a declaration, 2 tail-only) |
| Truncated at the start | 7 |
| Rootless fragment | 1 |
| Malformed | 0 |
| Named roots / catalogued roots | 118 (+1 unattributed) / 118 |
| Elements / distinct element names | 428,611 / 5,293 |
| Attribute names / `type` values / `content` kinds | 33 / 318 / 7 |
| Maximum depth | 17 |
| Comment starts (all / declaration-bearing) | 507 / 412 |
| `filename` / `parentPath` / `parentFile` / XInclude | 3,315 / 40 / 3 / 0 |
| `platform:/` references (levels / other) | 2,674 (2,346 / 328) |
| `$(common)` / `$(root)` / `$(target)` references | 588 / 28 / 23 |

Macro counts cover all element text; 587 of the 588 `$(common)` hits are in `<filename>` elements, which is the
filename-only bucket in the table above.

The root and element distribution is the one in the tables above: the five repeated roots hold 141 of the 266
documents, and the element histogram is dominated by the general node names `Item` (118,179), `item` (16,316)
and `Name` (12,850). The long tail is per-system parameter names; the manifest and timecycle names are a small
part of it. A category is a dispatch bucket, not a schema: for the 266 documents the catalog reproduces the
file and root counts in the functional table and assigns each root one category. No root has more than one
handler.

Reference handling is narrow. Every `$(...)` and `platform:/` string is element text, not an attribute; no
document uses XInclude. The macro alphabet is `$(common)`, `$(target)` and `$(root)`, and the platform targets
split into `platform:/levels/` and everything else. `filename` is the only include-like element with a large
population; `parentPath`/`parentFile` are the only relative references outside it. A resolver therefore needs
four prefix rules and a content-set/mount walk, not a general include processor.

A runtime dispatcher needs the 118-root table (or an RTTI class bridge), the 12 category buckets, the 318-value
`type` registry, the 33-name attribute whitelist, the `content=` array decoders and a streaming reader for the
14 MB document. The mount/reference half of that layer is measured separately in
[gamedata_graph.md](gamedata_graph.md): 25 manifest documents, 30 content-set ids, 12 resolved include edges
and 3,343 reachable target references. It also needs to mark the 9 tail-only documents and the one rootless
block as incomplete evidence: `gamedata_xml.py --check` is the reference for that split. The catalog is a local work product. Only
aggregate counts and schema/root/element/attribute names belong in git; the CSV/JSON, the element values, the
`type`/`content` values and per-file paths stay on the local machine. The tests are synthetic.

## Relationship to `data.rpf` and `script.rpf`

| Archive | Size | Streams | Inflated | Dominant payloads | XML | `RBF0` | `PSIN` |
|---|---:|---:|---:|---|---:|---:|---:|
| `common.rpf` | 6.6 MB | 954 | 50.2 MB | XML/config, text, `MHIU`, `PAMH`, `SPS1Z` | 844 streams | 0 | 0 |
| `data.rpf` | 481.5 MB | 28,509 | 1,403.3 MB | models/GPU pages, 1,263 `RBF0`, 2,281 `PSIN`, 3,132 resource pages | 0 | 1,263 | 2,281 |
| `script.rpf` | 4.4 MB | 714 | 5.0 MB | 671 `PSIN` plus binary | 0 | 0 | 671 |

There is **no XML, `RBF0` or `PSIN` duplication** between them: `common.rpf` contains no `RBF0`/`PSIN`, and
`data.rpf`/`script.rpf` have no XML documents at all — no stream starts as XML and no inflated stream contains
an XML declaration (the ten `<`-leading `data.rpf` streams are binary). This is a type-level and stream-start
check, not a byte hash of all 1.4 GB of inflated `data.rpf`; without the table of contents a duplicate stored
at a different block boundary would not be recognised.

The manifests in `common.rpf` reference the other archives by path token, not by archive handle. Counted over
the 3,315 `<filename>` elements:

| Reference bucket | Count |
|---|---:|
| `platform:/levels/` | 2,346 |
| `$(common)/` | 587 |
| other `platform:/` | 328 |
| relative/other | 31 |
| `$(target)/data/` | 23 |
| `$(target)/script/` | 0 |

So `common.rpf` is the XML/config layer that mounts levels content and references 23 `data.rpf` paths; none
of its `<filename>` entries use the `script` token. It may still reference script content through non-path
content names, which this count does not cover. The 587 `$(common)/` entries are mostly movie paths (537 end
in `.bk2`), not XML. Of the 57 `.xml` path entries, 50 are under `$(common)/` and 7 are relative/other; none
are under `data/` or `script/`. A further 40 `parentPath` attributes are relative references that cannot be
resolved without the table of contents. No XInclude (`xi:include`) elements occur.

## Readable vs binary

Readable: the 266 XML documents, 3 text/INI-like files, and the two `MHIU` blobs plus `PAMH`/`SPS1Z` headers.
The XML is ordinary text and can be consumed by any parser. `tools/gamedata_maps.py` decodes the `MHIU`
headers and square byte grids and the `PAMH` header, row table and u16 grid. `SPS1Z` is a NUL-separated
table of path/key strings with per-entry type/length markers; its structure is visible but its consumer is not
implemented.

Not fully readable: the 8.6% unread span, the two opaque binary blobs, and the missing first blocks of the 9
tail-only documents (their retained tails are readable but incomplete). The encrypted table of contents is
outside this work by rule; names and exact file boundaries stay unknown until a documented table reader exists.

## No-sharing rule

The archive, every carved `.bin`, `index.json` (its `head`/`pre` fields contain payload bytes), XML
excerpts, entry names, values, paths, screenshots and per-file reports stay on the local machine. Only
aggregate counts, sizes and schema/category names belong in git, issues or pull requests. The unit test uses
synthetic data; CI never reads a game folder. This is the same rule as `AGENTS.md` and `CONTRIBUTING.md`, and
it applies to this note as well.

## Next step

The most useful next step is a **schema bridge**, not more carving: check whether the 118 XML root tags match
RTTI class names in the executable, then build a streaming reader that dispatches on root and `type`. That
turns the measured XML into the runtime's config layer without hand-writing 5,293 tags. The `MHIU`/`PAMH`
arrays are decoded by `tools/gamedata_maps.py`; their checksum algorithm, the MHIU prefix and the value-to-Z
mapping remain open.
