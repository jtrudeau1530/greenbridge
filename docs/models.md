# Model recovery

Pre-fix measured state of `tools/rpf9dump models` on the Xbox Series 1.0.2.1 package, and the
upstream `rpf9-explorer` fix that closes it. The fix is validated but not merged yet.

Before the fix, the exporter did not reject any vertex or index layout in the tested archives. It
dropped valid models in three validation paths, all in `third_party/rpf9-explorer` at commit
`48557ea`.

## Upstream status

[Leonida issue #29](https://github.com/ItsNotRexed/Leonida/issues/29) tracks this work. The decoder
fix in [rpf9-explorer PR #1](https://github.com/breadbbbb/rpf9-explorer/pull/1) is at head
`c3708fb` and has now been independently validated on the package in the scopes below. It is
awaiting upstream merge and a submodule pin bump. This page records the pre-PR gap, the independent
validation, and the scope caveat; it is not a competing patch.

PR #1 reports 190,792 of 190,792 recognized candidates across the ten archives configured in
Leonida, including 169 previously rejected candidates (also verified after archive readback). That
ten-archive total was not independently reproduced here. Issue #29's run uses its own
recognized-candidate set and covers primary-LOD geometry; the pre-fix counts below use
`rpf9dump`'s page-scan candidates and export every recovered drawable/LOD variant. The two sets are
different scopes, not contradictory.

## Independent validation

Validated at PR #1 head `c3708fb` with .NET 8.0.428 on the real package:

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

Caveat: the full 76 GB `levels.rpf` and the other seven archives were not scanned, so the exact
ten-archive 190,792/190,792 and 169/169 claims were not independently reproduced. No counterexample
appeared in the tested scopes. A scoped validation comment was posted on PR #1.

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

## What to do next

1. Merge [rpf9-explorer PR #1](https://github.com/breadbbbb/rpf9-explorer/pull/1).
2. Bump `third_party/rpf9-explorer` from `48557ea` to the merged commit.
3. Re-run `tools/rpf9dump models` over the full ten-archive set configured in Leonida.
4. Replace the sample numbers in this page with the full counts, and keep the README and issue #29
   in sync.
