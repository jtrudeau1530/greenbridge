# Character skeleton and skin format

Measured from the Xbox Series **1.0.2.1** package. This page covers the shared humanoid
skeleton stored in a character **fragment** (`.#ft`) and the model side that binds meshes to
it. It is separate from the world model exporter in [models.md](models.md): the world
drawables do not use this 597-name table.

No game data is reproduced here. Offsets, counts, field layouts and a few short name/hash
examples are enough to reproduce the measurements against a reader's own package.

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

The vertex skin streams are in the FVF semantic table:

- Head/legacy head: semantic 20 is four packed `(u16 bone, u16 weight)` pairs; semantic 21 is
  a second index stream.
- Body/hands/feet/eyes/undies/jacket/tee: semantic 20 is four `u8` bone indices, semantic 21
  is four `u8` weights. A zero-weight row falls back to the first valid index.
- Brabikini: semantic 20 is 16 bytes that do not form `(bone, weight)` pairs and semantic 21
  has four `u32` indices with no weight bytes. Exported static; a rigid binding would be an
  invention.
- Eyecap: no semantic-20 weight stream and no valid palette. Exported static.

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

## Reproduction

The measurement tools were local and are not part of this repository. A reader can reproduce
the numbers with a standard zlib/RPF9 reader:

1. Inflate the `.#ft` stream at `0x45449F70`; parse the 4,095-slot hash table and the
   NUL-separated names; verify `joaat32` lowercased.
2. Inflate the following 131,072-byte graphics page at `0x4544F010`; parse 597 records at
   `0x150 + 0x50*i`; read `u16 parent` at +0x06.
3. For a character `#dr` system page, read `grmModel+0x20`, then the `u32` bone hashes.
4. Map each hash through step 1; unresolved hashes are hash-only fallbacks.

No archive, model, texture or audio file is stored in the repository or in any test fixture.
