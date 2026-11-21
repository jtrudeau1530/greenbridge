# Placement and archetype names

What a locally recovered name can resolve in the `0x7a` map resources of `levels.rpf`, and what it cannot.
Measured on the Xbox Series 1.0.2.1 package. This page contains counts, field hashes and formulas only; no
game names, entity rows or carved bytes.

## Result

Two keys are stored for most names, and they are independent enough to reject a same-hash collision:

| Field | Member hash | Offset | 32-bit key | 64-bit confirmation |
|---|---|---:|---|---|
| `archetypeName` | `a023a02c` | +0x44 | joaat, lower-cased | entity `c8a7b90a` at +0x20 |
| map `name` | `e6e17f14` | +0x08 | joaat, lower-cased | map `8bfb6095` at +0x10 |
| map `parent` | `a99c4651` | +0x68 | joaat, lower-cased | none in the root |
| entity id | `1b60404d` | +0x08 | none | per-entity, not a name key |

The 64-bit keys are not a second joaat. Both are the **low 64 bits of MurmurHash3 x64 128 with seed 0** over
the exact UTF-8 name bytes. In the two sampled windows every archetype name that matched the 32-bit hash also
matched this 64-bit key, and the one 32-bit-only candidate was rejected by it. The same formula reproduces the
map root's +0x10 key for all five map names that the local dictionary resolves.

The 32-bit field hashes themselves are the case-sensitive joaat of the engine member names (`joaat("CMapData")
= 0xd3593fa6`); the *values* are the lower-cased joaat of the model or map name.

## Dictionary

The measured dictionary was built only from names already recovered locally:

| Source | Input rows |
|---|---:|
| rpf9dump model manifests (`data.rpf`, `packs`, `core_restofworld`, `story000`) | 7,370 |
| rpf9dump levels scan paths (`0-8`, `4-4.5`, `12-16` GiB) | 43,187 |
| recovered name lists (levels and packs streams) | 16,844 |
| `common.rpf` XML `<filename>` strings | 6,448 |
| `common.rpf` XML `<name>`/`<Name>` strings | 15,378 |
| **merged exact names** | **79,738** |
| exact hash values / 32-bit collision groups (names) | 79,709 / 28 (57) |
| derived `_lod`/`_slod` spellings | 157,724 |
| derived hash values / collision groups (names) | 157,664 / 58 (118) |

Derived spellings are **not recovered strings**. They are LOD variants of recovered names whose 32-bit and
64-bit hashes both matched in the sample; the tool keeps them in a separate table and reports them separately.
The full 149,811-model `levels.rpf` export was not in the local artifact set used for this measurement, so the
dictionary is smaller than the project's model total.

## Archetype coverage

Two bounded 1 GiB windows were measured with entity aggregation only (counts per hash, no entity rows):

| Window | Maps | Entities | Distinct archetype hashes | Exact names resolved | Share | 32-bit-only rejected | With derived LOD |
|---|---:|---:|---:|---:|---:|---:|---:|
| 8–9 GiB | 1,398 | 88,228 | 12,400 | 1,942 hashes / 3,246 entities | 3.68% | 1 | 2,961 / 4,364 (4.95%) |
| 13–14 GiB | 1,318 | 115,571 | 11,705 | 3,165 hashes / 5,017 entities | 4.34% | 0 | 4,788 / 6,640 (5.75%) |
| **total** | **2,716** | **203,799** | **24,105** | **5,107 / 8,263** | **4.05%** | **1** | **7,749 / 11,004 (5.40%)** |

Every exact resolution above is confirmed by the +0x20 key: 5,107/5,107 hashes, 0 ambiguous. The one rejected
candidate had a valid lower-cased joaat but the wrong 64-bit key, which is exactly the same-hash case the
confirmation exists for. Derived `_lod`/`_slod` names are confirmed the same way; they add 2,741 entities.

This is a **dictionary-coverage lower bound**, not a claim that the other archetypes have no names. The
unresolved majority is dominated by models whose names are not in the local exports, and the same 32-bit
formula is proven by every resolved entry.

## Map-name coverage

The full `levels.rpf` census has 23,535 placement maps:

| Measure | Value |
|---|---:|
| Map rows / distinct `name` hashes | 23,535 / 23,535 |
| Names resolved by the dictionary | 5 (0.021%) |
| Of those, confirmed by the map +0x10 key | 5 / 5 |
| 32-bit-only candidates rejected | 0 |
| Rows with a non-zero `parent` | 2,089 |
| Distinct `parent` hashes | 2,053 |
| Parent hashes resolved | 1 (0.048%) |
| `parent` values that equal some map's `name` | 1,304 |
| Parent links whose parent name resolves | 1 |

The five map-name matches are not collisions: their map root's +0x10 value equals the Murmur3-64 of the same
name, and one of the maps also contains an entity whose archetype hash is the same name. The single resolved
parent equals a map name that resolves, so it is a real parent link. The remaining 23,530 names are unresolved
because those map-cell names are not in the local model/XML exports, not because a different hash formula was
observed. A runtime still needs a map-cell name source; this measurement only proves the formula.

## 64-bit fields

| Field | Window | Unique values / entities | Reading |
|---|---|---:|---|
| entity `1b60404d` (+0x08) | 13–14 GiB | 115,563 / 115,571 | per-entity id; 8 duplicate pairs |
| entity `1b60404d` (+0x08) | 8–9 GiB | 86,932 / 88,228 | per-entity id; duplicates follow duplicate payloads |
| entity `c8a7b90a` (+0x20) | 13–14 GiB | 11,705 / 11,705 | one key per archetype |
| entity `c8a7b90a` (+0x20) | 8–9 GiB | 12,400 / 12,400 | one key per archetype |

No name-only formula was found for +0x08: it changes per placement, so it cannot be a per-archetype key. The
earlier reading of +0x08 as a per-archetype hash and +0x20 as the GUID is reversed for this schema. The
`guid` member `9a7b6e6d` was not present in the entity schemas inspected in the 8–9, 13–14 and 20–21 GiB
windows.

## Tool

`tools/world_names.py` builds the dictionary and resolves map rows and aggregate archetype counts:

```powershell
python -I tools\world_names.py build --manifest <work>\assets --scan <work>\levels-scan.json `
    --names <work>\names.txt --xml <work>\common-concat.txt --out <work>\world\names.json

python -I tools\world_names.py resolve --dict <work>\world\names.json `
    --maps <work>\world\census\maps.json --archetypes <work>\world\archetypes.json `
    --out <work>\world\names-report.json

python -I tests\test_world_names.py
```

`build` accepts model manifests (a file or a tree containing `models/manifest.json`), rpf9dump scan JSON,
one-name-per-line text and `common.rpf` XML. `--derive-lod` adds the separate `_lod`/`_slod` table.

`resolve` accepts `world_index`/`world_census` map JSON (or a row list) and an aggregate archetype file:

```json
{"archetypes": {"a023a02c": 12}, "keys": {"a023a02c": "0011223344556677"}}
```

`keys` is optional. With it, a match is reported as verified only when the 32-bit candidate's Murmur3-64 equals
the stored key; candidates that match only the 32-bit hash are reported as `unconfirmed`, not guessed. Map rows
may carry `map_name64` to get the same confirmation. `tools/world_index.py` now emits that field, and a fresh
`world_census` run carries it through; the 23,535-row census JSON used above was produced before the field was
added, so the plain census run reported the five map-name matches as resolved but unverified. A five-row recheck
supplied the map root +0x10 keys and verified 5/5.

The dictionary and detailed reports contain game-derived names. Keep them in the local work folder; only
aggregate counts belong in git, issues or pull requests.
