# Character skeleton and skin format

Measured from the Xbox Series **1.0.2.1** package. This page covers the shared humanoid
skeleton stored in a character **fragment** (`.#ft`) and the model side that binds meshes to
it. It is separate from the world model exporter in [models.md](models.md): the world
drawables do not use this 597-name table.

No game data is reproduced here. Offsets, counts, field layouts and a few short name/hash
examples are enough to reproduce the measurements against a reader's own package.

The fragment carries the skeleton and a **25-vertex / 32-triangle bounds grid**, not a skin mesh. A
SixNG fragment-child walk finds no embedded drawables in any of the 2,865 `packs.rpf` character
fragments, while world fragments do have them; see
[character_models.md](character_models.md#what-is-not-decoded).

## The skeleton name table

Source resource: `ambient_femalebody_upper_nohead.#ft` in `packs.rpf`, stream offset
`0x45449F70` (stored 17,434 bytes, inflated 42,384 bytes).

| Range | Contents |
|---|---|
| `0x0008..0x7FF8` | 4,095-slot hash table, 8 bytes per slot: `u32 joaat32(name)`, `u32 boneIndex` |
| `0x8000` | page header; includes the `0x500280D0` pointer |
| `0x8088..0xA57E` | NUL-separated bone names (the first two entries repeat the fragment name) |
| `0xA57E` | `skel` marker |

The page contains **597 names and 596 hash entries**. Every matched hash equals
`joaat32(name)` over the lowercased ASCII name. The name table repeats the fragment name
(`ambient_femalebody_upper_nohead`), so 597 names share 596 hashes; the duplicate at index 596
is the entry without its own hash slot. Examples:

| Index | Name | `joaat32` |
|---:|---|---|
| 0 | `SKEL_ROOT` | `0x6686685C` |
| 1 | `SKEL_Pelvis` | `0x8133DE19` |
| 117 | `SKEL_L_Hand` | `0xA49BBD43` |
| 220 | `SKEL_Head` | `0x34C53E77` |
| 342 | `SKEL_R_Hand` | `0xB35ABA83` |

The table is a name/index dictionary, not a hierarchy. The hierarchy is in the next graphics
page.

## The skeleton instance table

The `.#ft` stream is followed by a 131,072-byte graphics page at `packs.rpf` offset
`0x4544F010`. Its `+0x08` pointer is `0x50027010`. The page-header pointer `0x500280D0` from
the `.#ft` page lands at `0x10C0` inside that graphics page. The page holds a **597-record
skeleton instance** starting at `0x150`, stride `0x50`:

| Record offset | Type | Meaning |
|---:|---|---|
| `+0x00` | `u32` | id/flags |
| `+0x04` | `u16` | next sibling |
| `+0x06` | `u16` | **parent index** (`0xFFFF` = root) |
| `+0x08` | `u32` | low 16 = bone index, high 16 = checksum |
| `+0x0C` | `u16` | bone index |
| `+0x0E` | `u16` | next/last-child link |
| `+0x10` | 4 × `f32` | reference rotation quaternion, `(x, y, z, w)` |
| `+0x20` | 3 × `f32` + pad | reference translation |
| `+0x30` | 3 × `f32` + pad | reference scale (all 1.0 in this page) |
| `+0x40` | `u32` + pad | pointer to another record |

Measured parent coverage:

- **580** non-root records have a valid measured parent.
- **16** records are roots: `SKEL_ROOT` (0) and the detached helper/attachment range
  572-595. `596` is the fragment-name record, not a deform bone.
- No cycles; every parent index is in range.

The parent field is enough to rebuild the real core hierarchy. Examples:

```
SKEL_ROOT(0)
  SKEL_Pelvis(1)
    SKEL_L_Thigh(2) -> SKEL_L_Calf(3) -> SKEL_L_Foot(4) -> SKEL_L_Toe0(7) ...
    SKEL_R_Thigh(38) -> SKEL_R_Calf(39) -> SKEL_R_Foot(40) -> SKEL_R_Toe0(43) ...
  SKEL_Spine_Root(106) -> SKEL_Spine0(107) ... SKEL_Spine6(113)
    SKEL_L_Clavicle(114) -> SKEL_L_UpperArm(115) -> SKEL_L_Forearm(116) -> SKEL_L_Hand(117) ...
    SKEL_Neck0(217) -> SKEL_Neck1(218) -> SKEL_Neck2(219) -> SKEL_Head(220)
    SKEL_R_Clavicle(339) -> SKEL_R_UpperArm(340) -> SKEL_R_Forearm(341) -> SKEL_R_Hand(342) ...
```

The reference TRS is **not the mesh bind pose**. Composing it places the spine branch on the
opposite side of the pelvis from the legs, so an exporter must not use it directly as a glTF
rest pose. Two honest options:

1. Keep identity joint transforms and identity inverse-bind matrices; the authored mesh stays
   in its bind pose.
2. Apply the measured reference TRS and derive inverse-bind matrices as `inverse(world)`. The
   mesh still renders in bind pose by construction, but the displayed skeleton is the measured
   reference pose, not a verified game bind pose. A test on the composed pose measured a
   maximum bind error of `2e-8`.

The true model bind/rest matrix array was not found in the model system or graphics pages.

## Model-side bone mapping

Each character drawable's `grmModel` header points at a **bone-hash array**:

| `grmModel` offset | Meaning |
|---:|---|
| `+0x08` | geometry pointer array |
| `+0x10` | geometry count (also duplicated at +0x12 and +0x36) |
| `+0x14` | local bone count |
| `+0x18` | bounds pointer |
| `+0x20` | **bone-hash array pointer** |
| `+0x28` | flags/hash word |

The array is `local bone count × u32`, sorted ascending. Each hash is matched against the
597-entry `joaat32` table to obtain the global bone index. Measured coverage:

| Part | Local bones | Resolved to the 597 table |
|---|---:|---:|
| head | 51 | 51 |
| body | 28 | 28 |
| hands | 198/199 | 199 |
| feet | 35 | 35 |
| undies | 44 / 18 | 43 / 18 |
| brabikini | 75 / 33 | 55 / 33 |
| jacket | 107 | 0 (hash-only) |
| crop tee | 137 | 80 |

Unresolved hashes are genuinely absent from the ambient table; they should be emitted as
`bone_0x...` fallbacks, never given invented names.

## Vertex skin streams (ASSET-12)

Measured with `tools/skin_streams.py` over every drawable resource that carves out of `packs.rpf`,
`data.rpf` and `levels.rpf` of the 1.0.2.1 package: the tool walks the same drawable → LOD →
`grmModel` → geometry chain as the readers above, joins each resource's pages (system pages first,
then graphics pages, one zlib stream per page) and reads the FVF semantic table, whose per-semantic
entries are a `u32` stream offset at `fvf + 4*i`, a `u8` element size at `fvf + 0xD0 + i` and a
`u8` format at `fvf + 0x104 + i` (vertex count = `u32` at `fvf + 0x138` >> 10). A size-0 slot is
a placeholder with no stream. The candidate scan only starts a resource at a 4096-aligned offset
with 16 preceding zero bytes — rpf9-explorer's recovery convention — and does not recurse into
nested archives, so the counts are a floor for the package.

| Layout | Weights (semantic 16/17) | Indices (semantic 20/21) | Meshes | Vertices |
|---|---|---|---:|---:|
| 4 influences, 8-bit | 16: `R8G8B8A8_UNORM` | 20: `R8G8B8A8_UINT` | 38,721 | 40,631,019 |
| 8 influences, 8-bit | 16, 17: `R8G8B8A8_UNORM` | 20, 21: `R8G8B8A8_UINT` | 23,860 | 47,611,935 |
| 4 influences, packed | 16: `R8G8B8A8_UNORM` | 20: `R32G32B32A32_UINT` | 18,247 | 39,468,907 |
| 8 influences, packed | 16, 17: `R8G8B8A8_UNORM` | 20, 21: `R32G32B32A32_UINT` | 16,111 | 9,512,230 |
| rigid, 8-bit | none | 20: `R8_UINT` | 5,364 | 3,606,677 |
| rigid, 16-bit | none | 20: `R16_UINT` | — | — |

Total: 102,303 skin meshes / 140,830,768 vertices. The meshes split by archive: `packs.rpf`
45,854 meshes / 60,142,591 vertices, `data.rpf` 3,222 / 3,854,484 (all 4-influence 8-bit),
`levels.rpf` 53,227 / 76,833,693. No 16-bit index form appears in the top-level streams of these
three archives; the 16-bit forms were measured (over nested-archive recovery) in the independent
measurement linked below. The format names are the DXGI equivalents of the measured format bytes —
28 for the weight slots, 30 for the 8-bit index slots, 3 for the packed index slots, 62 for the
rigid 8-bit index slot — anchored on the decoded bytes, not on a name table: the 1.0.2.1 FVF
format byte is not desktop DXGI numbering (desktop `R32G32B32A32_UINT` is 3 but desktop
`R8G8B8A8_UNORM` is 24, and here it is 28; `R16_UINT` is 56 in desktop DXGI and 57 here, as
breadbbbb/rpf9-explorer#10 reads it).

Rules the bytes support, validated per mesh against the model's own bone count (`grmModel + 0x14`)
on a deterministic sample (96 meshes per layout per archive; the census itself covers every mesh):

- **The index is into the model's own bone list.** Every component with a non-zero weight has an
  index below the local bone count — zero violations in every sampled mesh; a zero-weight
  component may carry any index, so the check masks by weight.
- **The weights are the UNORM bytes and sum to 255** over all present weight slots. Every sampled
  vertex of every layout in `data.rpf` and `levels.rpf`, and of the 8-influence and 8-bit
  4-influence layouts in `packs.rpf`, sums to exactly 255. The packed 4-influence meshes in
  `packs.rpf` round: of 50,648 sampled vertices, 31,884 sum to exactly 255, 11,604 to 247–254,
  and 7,160 fall below 247 (as low as 164).
- **Packed `R32G32B32A32_UINT`: the bone index is bits 0–11 of each u32.** Bits 9–11 were never
  set in the sampled meshes; bit 8 appears only where the model has more than 256 bones. Bits
  12–31 hold a separate, dense, unidentified field (smooth per-vertex values in the high bits —
  not a weight: the weights are already complete in 16/17).
- **Rigid meshes carry one index per vertex and no weight slot.**
- The vertex buffer head carries one native buffer descriptor word per used semantic; every skin
  slot's stream offset (bits 7–28 of the word) matched its descriptor. The word's size code
  (bits 3–6) equals the element bytes for the 4- and 8-byte slots, reads 1 for the 16-byte packed
  index slot and 0 for the 1-byte rigid slot, so only the offset is checked.

This corrects the earlier reading of this section (semantic 20 as packed `(u16 bone, u16 weight)`
pairs, semantic 21 as weights): semantic 21 is always index-typed, and the weights are the UNORM
bytes of semantics 16/17. The earlier "16 bytes that do not form `(bone, weight)` pairs" note was
the packed index layout above. [An independent measurement of the same layouts, over rpf9-explorer
recovery with the bone-palette pull requests applied, was posted in
#634](https://github.com/ItsNotRexed/Leonida/issues/634#issuecomment-7074504326); the counts there
are higher because that recovery also walks nested archives.

## Texture parameter split

The character material blocks carry several 32-bit parameter hashes before each 64-bit asset
key. Across the measured character materials, the texture content separates cleanly:

| Parameter | Common formats | Measured role |
|---|---|---|
| `0xC8E8C282` | BC1/BC3/BC7 sRGB, some BC1/BC7 UNORM | **sRGB albedo** when present |
| `0xC6F46E79` | BC7 UNORM, BC1 UNORM | tangent-space normal map |
| `0x1E77BF4C` | BC5 UNORM, some BC1/BC7 | normal/detail normal |
| `0x739E3D0D` | BC7 UNORM, some BC1 | ORM/mask |
| `0x8ADB8480` | BC4 UNORM | mask |

This is a measured corpus split, not a claim that the hashes are globally unique semantics.
The practical rule for exporters: use `0xC8E8C282` as albedo, and **do not fall back to the
first texture** when it is absent. A normal map painted as a base colour is the usual symptom
of that fallback.

To apply this rule, pass `--base-color-param C8E8C282` to `tools/world_build.py` (or
`base_color_param="C8E8C282"` to `build_world`): only references with that parameter hash are
used as base colour, and materials without a resolvable match stay untextured clay. Without the
option the exporter keeps its existing first-resolvable-reference behaviour, because these hash
roles are measured only on the character corpus.

## The Mandy search

Mandy is a KeyLento persistent/cutscene character schedule, not a unique drawable. `common.rpf`
contains the schedule entry `KeyLento/Perschars/Peds/Unique/Mandy` (pack
`PSP_KeyLento_perschars_peds`) in the `CPedScheduleInfoManager` XML. A read-only search of the
1.0.2.1 package found no `mandy`/`cs_mandy`/`ig_mandy`/`mp_mandy` drawable or fragment:

| Container | Search | Mandy drawable |
|---|---|---|
| `packs.rpf` | 144,364 streams / 9.74 GB; 26,585 decoded models at `e293917` (24,034 at `917e9b9`); 12,338 name records | none |
| `data.rpf` | 28,509 streams / 1.41 GB | none (localization/UI text only) |
| `common.rpf` | 954 streams / 50 MB | none (schedule and ped-appearance XML) |
| `levels.rpf` | 2,051 page-aligned nested archives, every zlib stream's first 4 MiB | none (an iFruit billboard and a house photo prop are the only name hits) |
| `story000.rpf` | 0-4 GiB and the 14 GiB region | none (animation clips `*_mandy.clip`, no drawable) |
| `story000_script.rpf`, `script.rpf`, `shared_minigame000.rpf`, rest-of-world, `audio.rpf`, `anim.rpf`, `movies.rpf`, `textures.rpf`/peds TTPF | full raw scans | none |

The name controls resolve `cs_kenton`, `u_f_o_rhonda`, `cs_monalisa` and `cs_lucia`; Mandy does
not. The delivered `mandy/` preview export is therefore a clearly labelled `cs_monalisa` proxy,
not a verified Mandy model.

Two further checks refine that negative. A missing `mandy` name is the common case, not an
anomaly: of the 13 `KeyLento/Perschars/Peds/Unique` entries, two resolve cleanly by name
(`kenton` -> `cs_kenton`, `rhonda` -> `u_f_o_rhonda`), one plausibly maps (`Bert` ->
`hc_cs_herbert_watson`), and the other ten, including `Mandy`, have no name hit. Also, the
package manifest lists 108 files and its `pack/` directory contains only `story000`,
`core_restofworld`, `core_restofworld001`, `shared_minigame000` and their `_bvh4`/`_licenses`
companions; there is no `psp_*` archive, so the `PSP_KeyLento_perschars_peds` pack named by the
schedule is not present in this build. Mandy is therefore not disproven: her appearance is either
a generic ped assembled at runtime or in a pack this package does not include.

Caveats: `levels.rpf` was prefix-bounded at 4 MiB per inflated stream, raw RSC8 bodies and
Bink/opaque regions were not decoded, and the encrypted RPF9 table of contents prevents a full
path enumeration (27 top-level entries remain). The evidence supports a shared ped-appearance
actor assembled from existing components, not a unique `mandy` resource.

## Reproduction

Steps 1-4 were measured with local tools that are not part of this repository; a reader can
reproduce them with a standard zlib/RPF9 reader:

1. Inflate the `.#ft` stream at `0x45449F70`; parse the 4,095-slot hash table and the
   NUL-separated names; verify `joaat32` lowercased.
2. Inflate the following 131,072-byte graphics page at `0x4544F010`; parse 597 records at
   `0x150 + 0x50*i`; read `u16 parent` at +0x06.
3. For a character `#dr` system page, read `grmModel+0x20`, then the `u32` bone hashes.
4. Map each hash through step 1; unresolved hashes are hash-only fallbacks.
5. Run `python -I tools/skin_streams.py <your copy's packs.rpf> [...] --json out.json` for the
   skin-stream census and validation; `python -I tests/test_skin_streams.py` runs the same walk
   and the same checks on synthetic resources built by the test.

No archive, model, texture or audio file is stored in the repository or in any test fixture.
