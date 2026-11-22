# Model recovery

Measured state of `tools/rpf9dump models` on the Xbox Series 1.0.2.1 package, and the
upstream `rpf9-explorer` fix that closes it. The fix is merged upstream and the submodule is
pinned at `917e9b9`.

Before the fix, the exporter did not reject any vertex or index layout in the tested archives. It
dropped valid models in three validation paths, all in `third_party/rpf9-explorer` at commit
`48557ea`.

## Upstream status

[Leonida issue #29](https://github.com/ItsNotRexed/Leonida/issues/29) tracks this work. The decoder
fix in [rpf9-explorer PR #1](https://github.com/breadbbbb/rpf9-explorer/pull/1) is merged, and
`third_party/rpf9-explorer` is pinned at upstream `917e9b9` (was `48557ea`). The later upstream
commits add the `packs.rpf` drawable-dictionary roots (one drawable per entry), the TTPF
array/cubemap/volume layouts, and the external drawable-skeleton reader. This page keeps the
pre-pin measurements as the control and records the new pin's `packs.rpf` count below; it is not
a competing patch.

PR #1 reports 190,792 of 190,792 recognized candidates across the ten archives configured in
Leonida, including 169 previously rejected candidates (also verified after archive readback). A
later full-archive run at the same head independently reproduced the ten-archive total: 190,792
candidates, 190,792 recovered, read back and decoded, 0 decode or readback failures, no duplicate
model names, and 2.22 billion triangles (the pre-fix export was 155,598 models / 1.95 billion
triangles). Issue #29's run uses its own recognized-candidate set and covers primary-LOD geometry;
the pre-fix counts below use `rpf9dump`'s page-scan candidates and export every recovered
drawable/LOD variant. The two sets are different scopes, not contradictory.

### `packs.rpf` after the pin bump

`packs.rpf`'s `.#dr` pages are drawable-dictionary roots, not drawables, so the old candidate
scan never saw them. At the new pin the dictionary walk recovers **24,034 models** from that
archive (23,720 dictionary children plus the 314 fragments), against 314 before. A full
`rpf9dump models` export of the archive reports **`ok=24,034 fail=0 skip=0` in 263 s** (0
embedded textures); the OBJs and manifest are local. The 314 `48557ea`/`c3708fb` entries in the
table below are therefore the pre-pin control, not the current `packs.rpf` result. The other
archive rows were not re-counted at `917e9b9`.

## Independent validation

Scoped validation at PR #1 head `c3708fb` with .NET 8.0.428 on the real package:

| Scope | Candidates | Recovered | Rejects | Readback |
|---|---:|---:|---:|---:|
| `data.rpf` full | 2,313 | 2,313 | 0 | 2,313 / 2,313 |
| `common.rpf` | 0 | 0 | 0 | 0 / 0 |
| `levels.rpf` 0-8 GiB | 28,748 | 28,748 | 0 | 28,748 / 28,748 |
| `levels.rpf` 4-4.5 GiB | 4,010 | 4,010 | 0 | 4,010 / 4,010 |
| `levels.rpf` 12-16 GiB | 10,429 | 10,429 | 0 | 10,429 / 10,429 |

The `data.rpf` export reported `ok=2,313 fail=0 skip=0`, 2,313 unique OBJ names, and 0 duplicate
stems. All 963 pre-PR misses were recovered, and all 1,350 pre-PR OBJs were byte-identical to their
pre-PR versions. The PR smoke test passes; v1/v2 cache round-trip and two raw-segment entries were
verified.

Caveat: this scoped run did not scan the full `levels.rpf`; the full pass below covers it. The
`data.rpf` OBJ text is the only byte-identical file comparison; the full pass counted decoded
models, not OBJ bytes. No counterexample appeared in the tested scopes.

### Full-archive run

A full pass at the same head over `levels.rpf` (70.9 GiB) and the other model-bearing archives
reproduced the ten-archive total. The `48557ea` column is the pre-fix control:

| Archive | Candidates | `48557ea` | `c3708fb` |
|---|---:|---:|---:|
| `levels.rpf` (full, 70.9 GiB) | 183,801 | 149,811 | **183,801** |
| `pack/story000/story000.rpf` | 3,305 | 3,303 | **3,305** |
| `data.rpf` | 2,313 | 1,350 | **2,313** |
| `pack/shared_minigame000/shared_minigame000.rpf` | 971 | 783 | **971** |
| `packs.rpf` | 314 | 314 | 314 |
| `pack/core_restofworld/core_restofworld.rpf` | 88 | 37 | **88** |
| `common`, `textures`, `story000_restofworld`, `core_restofworld001` | 0 | 0 | 0 |
| **Total** | **190,792** | **155,598** | **190,792** |

The `packs.rpf` row is the pre-dictionary-root control: at `917e9b9` that archive recovers
24,034 models (see above), not 314. The other rows were measured before the dictionary-root
change.

At `c3708fb`, every candidate is recovered, read back and decoded: 0 decode failures, 0 readback
failures, no duplicate model names, and 2.22 billion triangles (1.95 billion before). The
169-candidate subset is a different scope and was not isolated here.

## How the exporter works

```
rpf9dump models <archive.rpf> <outdir>
```

`FindCandidateOffsets` scans 4 MiB blocks for a page-aligned 16-byte zero prefix followed by a zlib
stream at page offset +16. It inflates the first 176 bytes and keeps the offset when the result has
a drawable or fragment header (`RageModelDecoder.LooksLikeResource`). `RecoverAtOffsets` follows
the following zlib blocks, splits the inflated bytes into the system segment and the graphics
segment, probes the extents, and calls `RageModelDecoder.Decode`.

To locate a named drawable before decoding, `tools/rpf9_names.py` indexes the measured
`name.#dr`/`#ft`/`#dd` spellings in bounded zlib prefixes without the encrypted TOC; see
[resource_names.md](resource_names.md).

For each returned entry, `tools/rpf9dump/Program.cs` reads the segments, splits at
`ResourceSystemBytes`, decodes the model, writes one OBJ, and writes the manifest. The manifest is
`<outdir>/<archive>/models/manifest.json` with `src`, `Count`, and `Items[]`. Each item records the
offset, OBJ name, model name, mesh/vertex/triangle counts, fragment and bone-binding flags, bounds,
per-mesh shader indices, embedded textures, and material texture references.

The manifest has no failure, skip, or layout fields. `Program.cs` increments `fail` only when a
model entry returned by the carver fails `RageModelDecoder.Decode`; `skip` is not used in models
mode. A printed `fail=0 skip=0` therefore means "no decoded entry failed", not "nothing was
rejected".

## Pre-fix measured counts

Counts below are from submodule `48557ea` before the upstream fix:

| Archive | Candidates | Recovered | Not recovered |
|---|---:|---:|---:|
| `common.rpf` | 0 | 0 | 0 |
| `textures.rpf` | 0 | 0 | 0 |
| `data.rpf` | 2,313 | 1,350 | 963 |
| `levels.rpf`, 0-8 GiB sample | 28,748 | 26,184 | 2,564 |
| `levels.rpf`, 4-4.5 GiB sample | 4,010 | 3,821 | 189 |

`levels.rpf` was sampled, not scanned end to end. Its first 3.9 GiB held no model candidates.
All counts come from the same package and the same `48557ea` submodule.

The 2,313 candidates in `data.rpf` decode to 2,313 unique model names once the checks below are
relaxed, so the 963 missing entries are valid models, not duplicate or false candidates.

## Pre-fix rejection classes

Observed in `data.rpf` with the unmodified carver:

| Class | Count | Location |
|---|---:|---|
| Drawable bounds do not contain a geometry's bounds | 231 | `Core/RageModelDecoder.cs:377` |
| Graphics block has non-zero bytes after `RequiredGraphicsBytes` | 259 | `Core/Rpf9ModelRecovery.cs:140` |
| Fragment system block with trailing bytes; next block not requested | 388 | `Core/Rpf9ModelRecovery.cs:132-133` |
| A following block was found, but no split decoded | 67 | `Core/Rpf9ModelRecovery.cs:141-143` |
| No following block was found | 18 | `Core/Rpf9ModelRecovery.cs:112-113` |

In the `levels.rpf` 0-8 GiB sample the same diagnostic classes were 978 bounds failures, 190
graphics-trailing cases, 1,200 cases where a system block was not followed to the next block, and
196 cases where a block was found but no split decoded.

The last two classes are symptoms of the splitter. The first three are the actionable causes.

Cause isolation on `data.rpf`:

| Change | Recovered |
|---|---:|
| Unmodified carver | 1,350 |
| Relax system trailing-byte check only | 1,408 |
| Relax graphics trailing-byte check only | 1,609 |
| Relax drawable bounds check only | 1,581 |
| Truncate both spans to the probed extents | 1,969 |
| Truncate both spans and relax drawable bounds | 2,313 / 2,313 |

`RequiredSystemBytes` and `RequiredGraphicsBytes` are the ends of the last structures the probe
actually validated. Bytes after those ends are legal: system blocks can carry inline strings, and
a graphics zlib block can carry trailing data. Decoding the full spans and then requiring every
trailing byte to be zero rejects valid resources.

## Measured diagnosis and the upstream fix

This is our measured diagnosis, not a competing patch. The two hunks below record the minimum our
runs isolated; PR #1 implements this area more broadly, including cluster marker and palette
handling, packed-position bounds, and mixed raw/zlib block handling, and its exact code may differ.
The hunks are against `breadbbbb/rpf9-explorer` at `48557ea` and are not applied to this checkout.

```diff
--- a/Core/Rpf9ModelRecovery.cs
+++ b/Core/Rpf9ModelRecovery.cs
@@ -132,3 +132,2 @@
-            if (extent.RequiredSystemBytes > prefixLength ||
-                system.AsSpan(extent.RequiredSystemBytes).IndexOfAnyExcept((byte)0) >= 0) continue;
+            if (extent.RequiredSystemBytes > prefixLength) continue;
             int graphicsLength = total - prefixLength;
@@ -140,3 +139,4 @@
-            if (graphics.AsSpan(extent.RequiredGraphicsBytes).IndexOfAnyExcept((byte)0) >= 0) continue;
-            RageModel? decoded = RageModelDecoder.Decode(system, graphics);
+            RageModel? decoded = RageModelDecoder.Decode(
+                system.AsSpan(0, extent.RequiredSystemBytes),
+                graphics.AsSpan(0, extent.RequiredGraphicsBytes));
             if (decoded is null) continue;
```

```diff
--- a/Core/RageModelDecoder.cs
+++ b/Core/RageModelDecoder.cs
@@ -373,8 +373,6 @@
                     var mesh = ReadMesh(geometry, $"model_{mi}_geometry_{gi}", partMin, partMax, boneCount);
                     if (shaderMapping is int mapping)
                         mesh = mesh with { ShaderIndex = U16(sys, mapping + gi * 2) };
-                    float tolerance = Tolerance(min, max);
-                    Require(Contains(min, max, partMin, tolerance) && Contains(min, max, partMax, tolerance));
                     meshes.Add(mesh);
                 }
             }
```

The bounds check is safe to remove because `ReadMesh` still validates every referenced index and
every decoded vertex against the geometry's own part bounds (`Core/RageModelDecoder.cs:643-646`).
The deleted check compared each geometry with the drawable-level bounds, which is not a valid
envelope for a drawable with multiple LOD models.

The narrow two-hunk experiment in a temp copy recovered:

- `data.rpf`: 2,313 / 2,313 entries.
- `levels.rpf` 4-4.5 GiB: 4,010 / 4,010 entries.
- `levels.rpf` 0-8 GiB: 28,746 / 28,748 entries.

The two remaining 8 GiB cases are not tag rejections: one graphics extent is 48 bytes larger than
the assembled block and the next zlib stream starts 3,419 bytes past it; one model with 32
geometries throws inside `ReadMesh`. PR #1's broader patch resolves both; at `c3708fb` the same
sample has 0 rejects.

## Vertex and index tags

The pre-fix tag survey (`data.rpf` plus the 0-8 GiB `levels.rpf` sample) contains 198,286 geometry
records. No unhandled layout tag appears:

| Item | Observed |
|---|---|
| Position `(size, format)` | `(8, 16)` on 195,182 geometries; `(12, 6)` on 674; no other pair |
| Index buffer stride | 4 on every geometry |

`(12, 6)` and `(8, 16)` are the only position encodings accepted by
`Core/RageModelDecoder.cs:244` and `:548-571`. The index stride must be 4
(`Core/RageModelDecoder.cs:249` and `:576`). Other values would be rejected, but none occur in
`data.rpf` or the `levels.rpf` samples. `data.rpf` uses `(8, 16)` exclusively; the 674
`(12, 6)` geometries are spread over 256 models in the levels sample.

## Character skeletons

World drawables do not carry the shared 597-name humanoid skeleton. Character `.#ft` fragments
store that name/hash table, and each character `#dr` model points at a local bone-hash array that
maps into it. The measured name table, the following 597-record hierarchy/reference-TRS table,
the model mapping and the texture-parameter split are in
[character_skeleton.md](character_skeleton.md). No character bind matrices were recovered; that
page records the reference-pose caveat rather than inventing a bind pose.

## What to do next

1. **Done:** [rpf9-explorer PR #1](https://github.com/breadbbbb/rpf9-explorer/pull/1) is merged upstream.
2. **Done:** the pin is `917e9b9` (was `48557ea`).
3. Re-run `tools/rpf9dump models` over the full ten-archive set configured in Leonida at the new
   pin. `packs.rpf` is measured (24,034); the other archives still carry the `c3708fb` counts.
4. Replace the sample numbers in this page with the full counts, and keep the README and issue #29
   in sync.
