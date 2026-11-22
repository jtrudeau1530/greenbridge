# GPU Strand Grooms (`smh`)

Hair, beards, brows, lashes, body hair, and animal fur in the Xbox Series build are not authored as static
triangle drawables. Each such item's `.#dr` drawable is a one-triangle stub, while the hair geometry is stored
separately as a **groom LOD resource** consumed by the `smh_*` compute and strand-rendering shaders.

This document specifies the measured binary layout of `smh` groom LOD resources, the shader-verified compute
pipeline (`smh_pose_0_CS_PreSkinSimGuides` and `smh_gen_1_CS_InterpolateStrands`), the exact Class 0 strand
scattering RNG, barycentric strand interpolation, and aggregate corpus measurements verified with the pure
standard-library decoder in `tools/smh_groom.py`. No game data or shader disassembly is included.

## Pipeline overview

Rendered strands are **not stored** in the archive. Each groom LOD stores a scalp triangle mesh whose vertices
are **guides** (polylines of rest-pose nodes, skin weights, arc lengths, root UVs, and per-guide attributes)
plus per-triangle strand density counts and optional region IDs. At runtime the `smh_*` GPU pipeline generates
and shades strands in stages:

1. **Guide pose & simulation (`smh_pose_*`, `smh_sim_*`)**: `smh_pose_0_CS_PreSkinSimGuides` reads guide rest
   positions (`+0x44`), bone indices (`+0x10`), and bone weights (`+0x14`) from the GPU asset buffer and skins
   guide curves into posed runtime buffers (21/21/22-bit packed positions).
2. **Strand scattering & interpolation (`smh_gen_*_CS_InterpolateStrands`)**: dispatches one thread per strand
   for each active strand class, picks a scalp triangle and uniform barycentric coordinates deterministically
   from the thread ID and asset seed, tests the triangle's region ID against a per-instance 256-bit style mask,
   blends the three corner guides' posed nodes, lengths, and UVs, and trims each strand to its randomized target
   length.
3. **Style modification & visibility (`smh_modify_*`, `smh_draw_*_CS_VisHair`, `VS_HairStrandGeometry`,
   `hair_resolve_*`)**: applies runtime clump, curl, frizz, and width parameters from per-instance style records
   and generates renderable strand geometry.

The strand class corresponds to the low five bits of the `smh_gen_N` shader variant: `smh_gen_1` generates
class 0 strands using `tri_count` byte 0 and header seed `+0x7c`; `smh_gen_2` reads byte 3 and `smh_gen_4`
reads bytes 1 and 3 using header seed `+0x80`. Higher variant bits (multiples of 32) select feature variants.

## Binary container and system page layout

On disk inside RPF9 archives (see [rpf.md](rpf.md)), each groom LOD is stored as a native RAGE resource: a
16-byte all-zero prefix on a 16-byte boundary followed by one zlib-compressed **system page** (464 to 672 bytes
inflated), padded to a 16-byte boundary, followed by one or more 16-byte-aligned zlib-compressed chunks that
concatenate to the **GPU asset buffer**.

System pointers use base `0x50000000` (`SYS_BASE`); GPU buffer pointers use base `0x60000000` (`GPU_BASE`).

| System offset | Type | Meaning |
|---|---|---|
| `+0x00` | `u8[8]` | Zero vtable placeholder before resource fixup |
| `+0x1c` | `u32` | Constant type ID `0x7d8f3db3` in every groom LOD resource |
| `+0x30` | `float3` | Bounding box minimum `(x, y, z)` in metres (skeleton-root space, Z-up) |
| `+0x40` | `float3` | Bounding box maximum `(x, y, z)` in metres (skeleton-root space, Z-up) |
| `+0x70` | `u64` | System pointer (`- 0x50000000`) to the GPU buffer descriptor (`+0x08`: `u32` dword count, `+0x0c`: `u16` stride = `4`, `+0x0e`: `u16` flags = `8`, `+0x18`: `u64` GPU pointer = `0x60000000`) |
| `+0x98` / `+0xa0` | `u32`, `u64` | Count and system pointer to `u32` skinning bone name hashes (`joaat32`, see [character_skeleton.md](character_skeleton.md)) |
| `+0xa8` / `+0xb0` | `u32`, `u64` | Region record count (`0..64`) and system pointer to 12-byte region records (`<HHHBBI`: triangles, guides, class 0 strands, pad, region ID, class 1 strands) |
| `+0xb8` | `u32[6]` | Six metadata hashes: slots `0..3` skinning bone hashes (`0x89ea2ed6` unused slot fill), slot 4 (`+0xc8`) style preset hash, slot 5 (`+0xcc`) part class hash |
| `+0xd0` | `u32, u32, u32, u16, u16` | Class 0 strand total, class 1 strand total, reserved word, triangle count, guide count |
| `+0xe0..` | packed | Style parameters (bytes and IEEE 754 `float16` values from `+0xe8`, including guide length and curl/braid fields) |

All rigs share a 1.0 m vertical root offset (groom `z + 1.0` equals standing height in metres).

## GPU asset buffer header (`0xA0` bytes)

The GPU asset buffer begins with 40 little-endian `u32` words (`0x00..0x9f`). Offset fields hold byte offsets
from the start of the GPU buffer (`0` means the optional array is absent). Element domains are **G** (per guide),
**N** (per node, stored guide-major as `node = guide * nodes_per_guide + k`), and **T** (per scalp triangle).

| Header offset | Name | Stride | Domain | Layout and shader usage |
|---|---|---:|---|---|
| `+0x00` | `attr_a` | 4 B | G | Three threshold bytes + `0xff` (`smh_gen`: compared against RNG) |
| `+0x04` | `attr_b` | 4 B | G | Three threshold bytes + pad (`smh_gen`) |
| `+0x08` | `uv` | 4 B | G | Two IEEE 754 `float16` values: root UV (`smh_gen`) |
| `+0x0c` | `attr_c` | 4 B | G | Four attribute bytes |
| `+0x10` | `bone_index` | 8 B | G | Eight `u8` skinning bone palette indices (`smh_pose`, `smh_gen`) |
| `+0x14` | `bone_weight` | 8 B | G | Eight `u8` skinning weights, normalized by their sum (`smh_pose`, `smh_gen`) |
| `+0x18` | `ext_18` | 12 B | G | Optional per-guide array (always paired with `ext_1c`) |
| `+0x1c` | `ext_1c` | 8 B | G | Optional per-guide array (always paired with `ext_18`) |
| `+0x20` | `tail` | var | — | Optional variable-size tail block (bounded by the next array offset or buffer end) |
| `+0x24` | `ext_24` | 4 B | G | Optional per-guide array (always paired with `direction`) |
| `+0x28` | `direction` | 4 B | G | Optional per-guide direction: three bytes decoded as `2 * b / 255 - 1` (`smh_gen`) |
| `+0x2c` | `sentinel` | — | — | Not an offset: `0xffffffff` in most LODs, `0x0001ffff` in 39 LODs |
| `+0x30` | `length` | 2 B | G | IEEE 754 `float16` guide arc length in metres (`smh_gen`) |
| `+0x34` | `node_4` | 4 B | N | Optional per-node array (always paired with `node_2`) |
| `+0x38` | `node_2` | 2 B | N | Optional per-node array (always paired with `node_4`) |
| `+0x3c` | `ext_3c` | — | — | Optional array offset (element layout not yet decoded; bounds `tail` when present) |
| `+0x40` | `triangle` | 6 B | T | Three `u16` guide vertex indices (`smh_gen`) |
| `+0x44` | `position` | 8 B | N | Four IEEE 754 `float16` values `(x, y, z, w)` rest position in metres (`smh_pose`, `smh_gen`) |
| `+0x48` | `tri_count` | 4 B | T | Four `u8` per-triangle strand counts by class; byte 0 = class 0 (`smh_gen_1`) |
| `+0x4c` | `tri_region` | 1 B | T | Optional `u8` region ID (`0` = none), tested against a 256-bit style mask (`smh_gen`) |
| `+0x50` | `flags` | — | — | `u32` feature/layout flags |
| `+0x54` | `nodes_packed` | — | — | `(nodes << 8) | low_byte`; `nodes = guides * nodes_per_guide` (`smh_sim`) |
| `+0x58` | `counts` | — | — | `guides | (triangles << 16)` (`smh_pose`, `smh_gen`) |
| `+0x5c` | `nodes_per_guide` | — | — | Low byte = nodes per guide (`smh_pose`) |
| `+0x60` | `length_random` | — | — | Two IEEE 754 `float16` values: `(amount, min_factor)` (`smh_gen`) |
| `+0x7c` | `seed_0` | — | — | `u32` RNG seed for strand class 0 (`smh_gen_1`) |
| `+0x80` | `seed_1` | — | — | `u32` RNG seed for strand class 1 (`smh_gen_2`, `smh_gen_4`) |
| `+0x90..0x9f` | `curve` | — | — | 16-byte monotonic lookup curve |

Stored guide lengths at `+0x30` match the polyline arc lengths integrated from `+0x44` to half-float rounding
precision (worst measured discrepancy 1.7 mm, median 0.37 mm across the corpus).

## LOD chain structure

A logical groom asset owns 1 to 4 LOD resources stored in descending guide-count order (guide count drops by
roughly `0.6x` per LOD while `nodes_per_guide`, `flags`, bounding box, and target strand totals remain stable).
Within an archive, LOD chains are usually contiguous, but some assets are stored **split**: LOD 0 appears, then
one or more other grooms appear, followed by LODs 1..3 of the outer asset. `tools/smh_groom.py` rejoins split
chains when `nodes_per_guide`, `flags`, bounding boxes (within 3 cm), and class 0/1 strand totals (within
`max(12, 5%)`) match and guide count strictly decreases.

## Class 0 RNG and barycentric strand interpolation

`tools/smh_groom.py` implements the exact 32-bit unsigned integer arithmetic of `smh_gen_1_CS_InterpolateStrands`
in pure Python standard library code:

### 1. Scalp triangle selection (4 attempts)

For each strand thread ID `t` in `0 .. count - 1`, all integer ops wrap modulo `2^32`:

1. Initialize `v3 = t * 0x19660D + 0xEF3EFB11`.
2. For `attempt` in `0, 1, 2, 3`:
   - Compute `s0 = attempt * 0x19660D + 0x86945BED` and `v11 = s0 * 0x153E39CE + v3`.
   - Mix `v5`, `v6`, `v7`, `v8`, `v10` using the shader's multiply-add and `x ^= x >> 16` sequence.
   - Candidate triangle index: `pick = (v8 >> 16) % triangles`.
   - Look up `rid = tri_region[pick]`. If `rid != 0` and bit `rid` is set in the 256-bit style mask, mark the
     strand culled for this attempt and continue to the next attempt.
   - Otherwise record `v20 = v5 >> 8` and test `(v6 & 0xFFFFFF) * 2^-24 < clamp_f32(tri_count[pick][0] * density_scale, 0, 1)`.
     If accepted, stop searching.
3. If all four attempts fail to accept, the final attempt's `tri`, `culled`, and last live `v20` remain.

`density_scale` is supplied per instance at runtime; offline export defaults to `1 / max(tri_count[:, 0])` so
the densest scalp triangle is always accepted.

### 2. Barycentric coordinates (`v_mad_u32_u24` cascade)

Using `mul24(a, b) = ((a & 0xFFFFFF) * (b & 0xFFFFFF)) & 0xFFFFFFFF` and constant `k = 0xC083C5`:

- `seed = (0x3C1 * (tri & 0xFFFF) + seed_0 * 0x8A30D + v20 * 31) & 0xFFFFFFFF`
- Six `mul24` + add steps with `0x3C6EF35F` and `x ^= x >> 12` folding yield two 24-bit integers `a` and `b`.
- `u1 = a * 2^-24`, `u2 = b * 2^-24`, `r = sqrt(u1)`.
- Uniform barycentric weights on the scalp triangle: `bary = (1 - r, (1 - u2) * r, u2 * r)`.

### 3. Rest-pose blending and length trimming

For each unculled strand on triangle `(g0, g1, g2)` with weights `(b0, b1, b2)`:

- Blend each node `k` of the three guides: `P[k] = b0 * pos[g0][k] + b1 * pos[g1][k] + b2 * pos[g2][k]`.
- Blend stored guide arc length: `L_guide = b0 * len[g0] + b1 * len[g1] + b2 * len[g2]`.
- Compute per-strand PCG hash of `seed`:
  `v = seed * 0x2C9277B5 + 0xAC564B05`, `w = (((v >> ((v >> 28) + 4)) ^ v) * 0x108EF2D9)`,
  `u_len = ((w >> 22) ^ (w & 0xFFFFFF)) * 2^-24`.
- Target length from header `+0x60` `(amount, low)`:
  `factor = low + u_len * (1 - low)`, `L_target = max(L_guide * (1 + (factor - 1) * amount), 0)`.
- Walk segments `k = 1 .. nodes_per_guide - 1`, clamping each segment to the remaining length (using a minimum
  segment length floor of `2^-23` m and zeroing steps below `1e-9` m). Trailing nodes past the trimmed tip repeat
  the tip coordinate.

## Measured corpus census

Aggregate counts measured with `python -I tools/smh_groom.py census` across the two package archives that hold
groom resources (other archives such as `data.rpf`, `story000`, and `shared_minigame000` hold none):

| Archive | Groom assets | LOD resources | Decode errors | GPU buffer bytes | Class 0 strands | Class 1 strands |
|---|---:|---:|---:|---:|---:|---:|
| `packs.rpf` | 5,741 | 20,010 | 0 | 902,231,288 | 77,123,314 | 35,359,612 |
| `core_restofworld.rpf` | 10 | 36 | 0 | 225,568 | 3,099 | 5,596 |
| **Total** | **5,751** | **20,046** | **0** | **902,456,856** | **77,126,413** | **35,365,208** |

Additional corpus-wide properties across all 20,046 decoded LODs and 5,751 LOD 0 assets:

- **Decode coverage**: 20,046 / 20,046 LOD resources decode cleanly (0 failures).
- **Geometry ranges**: 3 to 5,185 guides per LOD; 2 to 80 nodes per guide.
- **LOD depths**: 4,534 assets have 4 LODs, 341 have 3 LODs, 11 have 2 LODs, and 865 have 1 LOD.
- **Per-triangle strand sums**: summing `tri_count[:, 0]` equals system page `+0xd0` in 5,716 / 5,751 LOD 0
  assets; summing `tri_count[:, 1]` equals `+0xd4` in 3,158 / 5,751 LOD 0 assets (confirming class 1 placement
  uses an additional shader rule in `smh_gen_4`).
- **Optional arrays**: `node_4` / `node_2` appear together in 12,101 LODs; `ext_18` / `ext_1c` appear together
  in 10,684 LODs; `ext_24` / `direction` appear together in 5,718 LODs; `+0x3c` is non-zero in 10,756 LODs;
  header `+0x2c` is `0x0001ffff` in 39 LODs (`0xffffffff` in all other 20,007 LODs).
- **Heuristic bounding-box placement classes** (5,751 assets): 1,462 Facial hair, 1,371 Body hair, 1,119 Scalp
  hair, 1,076 Fur and other, 723 Brows and lashes.

## What is exact vs. modelled or open

| Component | Status | Notes |
|---|---|---|
| Binary layout (system page, GPU buffer header, guides, nodes, triangles, skin, lengths, UVs, counts, regions) | Exact | Verified across all 20,046 LODs (0 decode failures) against `smh_pose_0_CS_PreSkinSimGuides` and `smh_gen_1_CS_InterpolateStrands` |
| Class 0 root triangle selection & barycentric RNG | Exact | Bit-exact 32-bit unsigned integer port of `smh_gen_1_CS_InterpolateStrands`; `density_scale` defaults to `1 / max(tri_count)` |
| Rest-pose strand shape and length trimming | Modelled | Barycentric blend of three guides trimmed via `+0x60` PCG length variation; omits runtime style record (clump/curl/frizz/region mask), length texture, skinning, and simulation |
| Class 1 strand scattering | Approximate | Class 1 total is exact; `smh_gen_4` placement rule is not yet ported |
| Optional arrays `ext_18`, `ext_1c`, `ext_24`, `node_4`, `node_2`, `+0x3c`, `tail` | Undecoded | Spans are parsed and reported; per-field semantics remain open |
| Groom-to-item name linking | Partial / encrypted TOC | Groom resources store no asset name or name hash, and the nested RPF9 file table is encrypted (see [rpf.md](rpf.md) and [character_models.md](character_models.md)) |

## CLI usage (`tools/smh_groom.py`)

`tools/smh_groom.py` uses only the Python standard library (`python -I`) and reuses the binary glTF 2.0 container
primitives from `tools/glb_writer.py`. Synthetic unit tests live in `tests/test_smh_groom.py`.

```powershell
# Inspect a carved .smh file or an archive member at a byte offset
python -I tools/smh_groom.py inspect <file.smh> --json
python -I tools/smh_groom.py inspect <archive.rpf> --offset 0x1000

# Run an aggregate census over one or more archives, carved .smh files, or directories
python -I tools/smh_groom.py census <archive.rpf> --json

# Export a groom to glTF 2.0 binary (.glb) as 3-sided tubes, flat ribbons, LINES, or LINE_STRIP
python -I tools/smh_groom.py export-glb <file.smh> out_tubes.glb --geometry tubes --strands 6000
python -I tools/smh_groom.py export-glb <file.smh> out_ribbons.glb --geometry ribbons --strands 4000
python -I tools/smh_groom.py export-glb <file.smh> out_curves.glb --curves --strands 12000
python -I tools/smh_groom.py export-glb <file.smh> out_strips.glb --geometry line-strip --no-scalp --no-guides
```
