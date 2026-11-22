# Model recovery

Measured state of `tools/rpf9dump models` on the Xbox Series 1.0.2.1 package, and the
upstream `rpf9-explorer` fix that closes it. The fix is merged upstream and the submodule is
pinned at `e293917` (the 350,248-model counts below were measured at `917e9b9`, before
upstream `v1.2` PRs #8–#12 landed).

Before the fix, the exporter did not reject any vertex or index layout in the tested archives. It
dropped valid models in three validation paths, all in `third_party/rpf9-explorer` at commit
`48557ea`.

## Upstream status

[Leonida issue #29](https://github.com/ItsNotRexed/Leonida/issues/29) tracks this work. The decoder
fix in [rpf9-explorer PR #1](https://github.com/breadbbbb/rpf9-explorer/pull/1) is merged, and
`third_party/rpf9-explorer` is now pinned at upstream `e293917` (was `917e9b9`, originally
`48557ea`). Commits up to `917e9b9` added the `packs.rpf` drawable-dictionary roots (one drawable
per entry), the TTPF compact/array/cubemap/volume layouts, and the external drawable-skeleton
reader. Commits `917e9b9..e293917` add upstream `v1.2` model-discovery changes (PRs #8–#12:
optional fragment pointers, skipping texture dictionaries in drawable-dictionary scans, 16-bit
bone-palette indices, probe-count parity, and zero-radius drawables) plus BCPack BC1/BC3/BC4/BC5
decoding (PR #7). The 350,248-model counts below were measured at `917e9b9` and have not yet been
re-counted at `e293917`. This page keeps the pre-pin measurements as the control and records the
`917e9b9` counts below; it is not a competing patch.

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
model-bearing archives were re-counted at the new pin; the results are below.

### Other archives at `917e9b9`

A fresh scan/recover at the new pin (no cache; `models/3`) shows the same pattern in every
archive with drawable dictionaries: the `c3708fb` recovery is the non-child count, and the new
entries are dictionary children.

| Archive | Candidates | Models at `917e9b9` | Dictionary children | `c3708fb` |
|---|---:|---:|---:|---:|
| `data.rpf` | 2,402 | **2,581** | 268 | 2,313 |
| `pack/shared_minigame000/...` | 1,070 | **1,288** | 317 | 971 |
| `pack/core_restofworld/...` | 1,024 | **885** | 797 | 88 |
| `pack/story000/story000.rpf` | 4,195 | **7,460** | 4,156 | 3,304 |
| `packs.rpf` | 23,838 | **24,034** | 23,720 | 314 |
| `levels.rpf` | 206,981 | **314,000** | 130,225 | 183,801 |
| `common`, `textures`, `story000_restofworld`, `core_restofworld001` | 0 | 0 | 0 | 0 |
| **Total** | **239,510** | **350,248** | **159,483** | **190,792** |

The last column is the `c3708fb` recovery for comparison. `story000`'s standalone count is
3,304 vs 3,305; the full new-pin standalone total is 190,765, 27 below the `c3708fb` 190,792.
Those 27 are not merged duplicates: they are the 27 models with more than 256 geometries that the
probe rejects at `917e9b9` (26 in `levels.rpf`, 1 in `story000`; see
[rejected candidates](#rejected-candidates-at-917e9b9)). The new entries are **159,483
drawable-dictionary children**: one model per embedded drawable in a dictionary root, not new
pages. `core_restofworld` recovers 873 of its 1,024 candidates (885 models); the 151 rejects are
140 texture dictionaries and 11 drawable dictionaries whose drawable has radius 0.

`levels.rpf` is the sum of ten parallel recovery shards over 206,981 candidates: **314,000
models**, 130,225 dictionary children, 183,775 standalone. 1,037 candidates are rejected: the
1,009 in the last shard are texture dictionaries, not models, and 28 more in shards 1 to 7 are
models the probe caps reject (see [rejected candidates](#rejected-candidates-at-917e9b9)). The
`levels.rpf` full export was not re-run, so its triangle total is still the old measurement.

### Rejected candidates at `917e9b9`

Every candidate of each model-bearing archive was run through `RecoverAtOffsets` on its own and,
for each reject, through a copy of `TryResource` that records which check drops it (it agrees
with the real code on all 206,981 `levels.rpf` candidates). Measured on the 1.0.2.1 package:

| Archive | Rejects | Texture dictionaries | Geometry cap (`:351`) | List 4 cap (`:387`) | Radius 0 |
|---|---:|---:|---:|---:|---:|
| `levels.rpf` | 1,037 | 1,009 | 26 | 2 | 0 |
| `data.rpf` | 75 | 75 | 0 | 0 | 0 |
| `pack/core_restofworld/...` | 151 | 140 | 0 | 0 | 11 |
| `pack/shared_minigame000/...` | 18 | 18 | 0 | 0 | 0 |
| `pack/story000/story000.rpf` | 1 | 0 | 1 | 0 | 0 |
| `packs.rpf` | 0 | 0 | 0 | 0 | 0 |

Line numbers are in `third_party/rpf9-explorer/Core/RageModelDecoder.cs`.

- **Texture dictionaries (1,242).** `LooksLikeDrawableDictionary` (`:78`) checks only the
  dictionary header, which texture dictionaries share, so they become model candidates and fail
  at `:314` because their values are not drawables. `Rpf9TextureRecovery` recovers all 1,242.
  In `levels.rpf` they are the archive's last 0.12 GiB, each a 256- to 880-byte system block
  and a graphics block. None of them overlaps a recovered model. Upstream
  [rpf9-explorer#9](https://github.com/breadbbbb/rpf9-explorer/pull/9) stops the scan from
  proposing them.
- **Geometry cap (27).** The probe allows at most 256 geometries per model (`:351`), while
  `Decode` allows `MaxMeshes`, 1,024 (`:535`). These models have 263 to 377 geometries.
- **List 4 cap (2).** The probe caps renderer list 4 at 4,096 entries (`:387`) but never uses
  that count for sizing. These two dictionaries have 5,078 and 5,461.
- **Radius 0 (11).** Drawable dictionaries whose drawable has a zero radius and non-empty bounds,
  rejected by `LooksLikeDrawableHeader` (`:124`) through `:314`. Whether they decode otherwise
  has not been checked.

[rpf9-explorer#11](https://github.com/breadbbbb/rpf9-explorer/pull/11) raises both probe caps.
With it, the 29 capped candidates recover (41 models, 3.25 million triangles), and every
candidate already recovered decodes exactly as before in all six archives above.

One `levels.rpf` copy measured for this showed a 1,038th reject: a zlib block whose Adler-32
fails because of a single flipped bit in that copy (flipping it back restores the stored
checksum). It is local corruption, not part of the package, and is left out of the table.

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
24,034 models, and the other archives' new-pin counts are in the table above. The rows here are
the `c3708fb` measurements (190,792 models); the new-pin re-count is 350,248. Triangle totals
have not been re-run at the new pin.

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

## Vehicle ratings and performance estimates

`common.rpf` has one `CVehicleBaseRatingsAndEstimatesMap` XML document (see [gamedata.md](gamedata.md)): a
`Map` of 252 `Item key="<model>"` entries, each with four blocks. Counts and ranges only; no model names.

| Block | Fields |
|---|---|
| `estimatesRoad`, `estimatesOffroad` | `fTopSpeed`, `fCornerSpeed`, `fAccelerationTime`, `fBrakingTime`, `fStandingQuarterTime`, `fStandingQuarterSpeed`, `fAvgLatTraction` |
| `ratingsRoad`, `ratingsOffroad` | `fOverallRating`, `fSpeedRating`, `fHandlingRating`, `fAccelerationRating`, `fBrakesRating` |

- **Keys are vehicle model names.** The 252 keys are distinct plain names (no hex hashes), and all 252 are the
  base name of a `.yft` in the local archive catalog. 245 of them are also among the 319 models of the
  distant-light recipes; 74 of those 319 have no rating entry.
- **Ratings are not fractions.** All ten rating fields have a minimum of exactly 10 and medians of 142 to 585,
  with maxima up to 1,327 (`fAccelerationRating`). `fOverallRating` is not the mean of the other four: the two
  differ by a median of 38 and agree within 0.01 in only 10 of 504 cases.
- **Estimates look like SI units.** Road `fTopSpeed` has a median of 44.3 and a maximum of 77.4 (m/s would make
  that 160 and 279 km/h); `fStandingQuarterTime` has a median of 15.8, the right size for seconds over a
  quarter mile; `fAvgLatTraction` has a median of 1.42 on road and 0.92 off road (lateral g would fit). Offroad
  speeds are lower than road ones (median top speed 19.3).
- **Sentinels.** On each surface, 11 entries have every estimate 0 and 10 more have zero speeds with an
  infinite time (`inf`). `inf` occurs in 38 road and 99 offroad blocks overall, so a parser must accept `inf`
  in these fields (it reads as "never reaches the test speed or distance").

Not measured here (hypothesis): the rating scale and how `fOverallRating` weights its parts, and whether the
estimates are precomputed from handling data or measured by simulation.

## Character skeletons

World drawables do not carry the shared 597-name humanoid skeleton. Character `.#ft` fragments
store that name/hash table, and each character `#dr` model points at a local bone-hash array that
maps into it. The measured name table, the following 597-record hierarchy/reference-TRS table,
the model mapping and the texture-parameter split are in
[character_skeleton.md](character_skeleton.md). No character bind matrices were recovered; that
page records the reference-pose caveat rather than inventing a bind pose.

## Shader render buckets

`rpf9dump` writes two raw shader fields per material, `RenderBucket` (byte at shader +0x39) and
`RenderBucketMask` (u32 at +0x3C; `RageModelDecoder.cs:651`). Measured on a random sample of 3,000 `.ydr`/`.yft`
models (27,435 shaders):

- `RenderBucketMask == 0xFF00 | (1 << (RenderBucket & 0x7F))` in all 27,435. Bit 7 of `RenderBucket` is a
  separate flag (set in 438) that does not change the mask.
- `RenderBucket & 0x7F` takes the values 0-6, the GTA V drawable bucket list. Model-name tokens (counted per
  shader, from the model's name) line up with it:

| Bucket | GTA V meaning | Shaders | Name evidence (share of shaders whose model name matches) |
|---:|---|---:|---|
| 0 | opaque | 17,826 | baseline: water 0.6%, vegetation 0.5%, glass/window 0.2%, decal 0.1% |
| 1 | alpha | 725 | glass/window 7.6% (38x baseline) |
| 2 | decal | 7,282 | decal/graffiti/dirt 3.1% (31x) |
| 3 | cutout | 1,311 | vegetation 2.8% (5.6x) |
| 4 | no splash | 6 | one program (`fc34f08b`) |
| 5 | no water | 6 | the same program |
| 6 | water | 343 | water/river/pool/canal 80.2% |

Names are per model and a building holds glass, decal and opaque shaders alike, so the token shares are low
in absolute terms; the enrichment against bucket 0 is the signal. For export: bucket 1 needs blending, 2 is
a projected decal, 3 needs alpha testing, 6 is a water surface.

## Vehicle distant-light recipes

`common.rpf` has one `CVehicleDistantLightData` XML document (see [gamedata.md](gamedata.md)) that ties
vehicle models to far-away light recipes. Counts only; no model names.

| Path | Count | Content |
|---|---:|---|
| `distantLights`, `presetLights` | 1 each | empty |
| `recipeLights/Item` | 1,980 | `vehicleModelName`, `secondaryRecipes`, `lights` |
| `recipeLights/Item/secondaryRecipes/Item` | 2,383 | one token each; 0 to 3 per recipe (0: 312, 1: 1,029, 2: 563, 3: 76) |
| `recipeLights/Item/lights` | 1,980 | empty in all |

- **Every `vehicleModelName` is a real fragment model.** The 1,980 recipes name 319 distinct models, and all
  319 exist as `.yft` entries in the local archive catalog, each also with a `<name>_hi.yft` high-detail
  variant. None is a `.ydr`. A loader can therefore key the recipes by the vehicle's `.yft` base name.
- A model has 1 to 45 recipes (93 models have one); in 304 of the 319 models every recipe has a different set
  of secondary tokens, so a recipe is selected by its secondaries, not by the model alone.
- The 780 distinct secondary tokens are not catalog entries (2 coincide with an entry name) and not texture
  dictionary items, and are never another recipe's model name. 1,242 of the 2,383 contain the recipe's own
  model name.
- **No light data is stored**: `lights` is empty in all 1,980 recipes and both top-level lists are empty. In
  this build the document is an index from model and variant to recipe with the light positions absent,
  so distant vehicle lights cannot be reconstructed from it.

Not measured here (hypothesis): what the secondary tokens name (per-model parts or liveries would fit the
model-name substrings), and where the light positions are generated at runtime (for example from the model's
own light bones).

## What to do next

1. **Done:** [rpf9-explorer PR #1](https://github.com/breadbbbb/rpf9-explorer/pull/1) is merged upstream.
2. **Done:** the pin is `e293917` (was `917e9b9`, originally `48557ea`).
3. `tools/rpf9dump models` was re-run over the model-bearing archives at `917e9b9` (counts in the
   table above). A fresh re-count at `e293917` is still needed to measure the delta from upstream
   `v1.2` PRs #8–#12. Triangle totals were not re-run.
4. Keep the README and issue #29 in sync with the new counts.
