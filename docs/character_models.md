# Character models and ped parts

Measured from the pinned `e293917` export of `packs.rpf` (`tools/rpf9dump models`): **26,585
manifest rows**, **11,335 unique names**, **23,720 `.#dr` drawable rows** and **2,865 `.#ft`
fragment rows**. One row is one LOD or variant, so these are row counts, not character counts.

No game data is reproduced here.

## A character is a drawable plus a fragment

- A `.#dr` drawable carries the geometry, materials and texture references.
- A `.#ft` fragment carries the shared 597-name humanoid skeleton and the character's bounds. The
  measured skeleton is in [character_skeleton.md](character_skeleton.md).

The current exporter writes every fragment as a **25-vertex bounds cube** with `Embedded: null`
(2,865 of 2,865). Fragment child drawables are not walked, so a part that exists only as a
fragment child is not in the OBJ/GLB output even though its parent fragment is.

## Lucia records

22 unique names contain `lucia` (12 contain `cs_lucia`). They are parts, not one assembled model.
The decoded export used by the previews contains 13 parts:

| Part | Source drawable | Vertices | Triangles | Albedo key |
|---|---|---:|---:|---|
| head (hi) | `player_story000_00_head_fr1_lat_000_m.#dr` | 8,983 | 13,320 | none |
| head L0 | same | 4,638 | 4,940 | none |
| head L2 | same | 599 | 514 | none |
| eyes | `…_head_fr1_lat_000_m_eyes.#dr` | 434 | 816 | `0x429453B7A2EB9980` |
| eyecap | `…_head_fr1_lat_000_m_eyecap.#dr` | 214 | 304 | `0x5CE55BAA340248BD` |
| hands | `hand_fr1_000_nude_cull.#dr` | 10,440 | 13,534 | `0x4D89F2311DBBB7A8` |
| feet | `feet_fr1_000_full.#dr` | 2,309 | 4,082 | `0xA54D9F8B51CA6034` |
| body (legacy) | `cs_lucia_legacypedcs_fr1_000_fullbody_ws00_full.#dr` | 1,748 | 1,816 | `0x996FDBA2DB30CC12` |
| legacy head | `cs_lucia_legacyheadcs_fr1_001_regular_ws00_full.#dr` | 592 | 1,028 | `0x707771F769AA3542` |
| jacket (biker) | `cs_lucia_jacketbiker_fr1_000_regular_c_hl_long_full.#dr` | 9,818 | 10,837 | `0xF5073AA039BB3CED` |
| crop tee | `player_story000_00_tshirtcrop_fr1_001_slim_cc_un_long_full.#dr` | 4,932 | 6,756 | `0x46E8FA9436E9D43C` |
| bikini top | `brabikini_fr1_012_regular_ws00_full.#dr` | 6,547 | 5,697 | `0xDF3615ADFB1F4787` |
| undies | `undiesgstring_fr1_002_slim_hw_full.#dr` | 961 | 913 | `0xCFD339832A9C724A` |

Two caveats that matter for any render:

- **`body` is not a base body.** The only Lucia full-body record in `packs.rpf` is the
  `cs_lucia_legacypedcs` clothed legacy ped. The decoded part is 1,748 vertices / 1,816
  triangles, and the manifest lists 874- and 640-vertex rows of the same drawable. Do not present
  it as Lucia's hero body.
- **The head drawable has no texture references.** Its face cannot be textured from the material
  keys alone.

`player_story000_00` (Lucia's slot prefix) has 104 unique names: **63 drawables** plus **41
`char_bipedbounds` fragments**. The largest drawables:

| Drawable | Vertices | Triangles |
|---|---:|---:|
| `…_hoodiezip_fr1_001_regular_o_hd_long_cull_00.#dr` | 20,311 | 14,256 |
| `…_sweatshirtcrewneck_fr1_000_loose_cc_un_long_cull_00.#dr` | 19,217 | 13,600 |
| `…_hoodiezip_fr1_001_regular_h_hd_long_cull_00.#dr` | 19,141 | 14,106 |
| `…_shirtdress_fr1_000_slim_c_oc_un_long_cull_00.#dr` | 14,341 | 16,146 |
| `…_tshirtcrop_fr1_001_slim_cc_un_long_cull_00.#dr` | 12,254 | 15,437 |
| `…_jacketbiker_fr1_000_regular_o_hl_long_full.#dr` | 11,093 | 12,023 |

## Slot tokens

These are counts of unique names containing the token. They are a naming heuristic, not the game's
component table.

| Token | Unique names | Note |
|---|---:|---|
| `player_story` | 196 | Lucia (`000_00`) and Jason (`000_01`) |
| `head` | 3,741 | heads and head attachments across all characters |
| `hair` | 278 | includes accessories and non-human fur |
| `belt` | 44 | accessories |
| `pants` | 459 | `pantsjeans`, `pantscargo`, `pantsleggings`, `pantstrack`, … |
| `jeans` | 157 | `pantsjeans` / `shortsjeans` |
| `jacket` | 327 | `jacketbiker`, `jacketdenim`, `jacketblazer`, `jackettrack`, … |
| `uppr` | 17 | all `slod_uppr_*` super-LOD crowd parts |
| `lowr` | 21 | all `slod_lowr_*` super-LOD crowd parts |
| `jbib` | 0 | the GTA V slot code is not a name token here |
| `hands` | 8 | the singular `hand_` appears in 35 names |
| `feet` | 26 | bare feet plus creature variants |
| `bikini` | 45 | `brabikini_*` and related |
| `undies` | 102 | `undiesgstring`, `undiesbriefs`, `undiesboxers`, … |

## Textures

The character export requested 215 texture keys and matched **184**. Only `0xC8E8C282` is used as
sRGB albedo; `0xC6F46E79` / `0x1E77BF4C` are tangent-space normal maps, `0x739E3D0D` is ORM/mask
and `0x8ADB8480` is a BC4 mask. A part without an explicit albedo keeps a flat material colour;
it must not be painted with a normal or ORM map.

The hands and feet keys (`0x4D89F2311DBBB7A8`, `0xA54D9F8B51CA6034`) did not resolve in the
scanned TTPF pools. The head has no texture reference at all.

## What is not decoded

- **Fragment children.** All 2,865 fragments export as the 25-vertex bounds cube described above.
  A shared body that is only a child drawable would be invisible to the current export.
- **A ped-variation table.** `common.rpf` contains `CPedVariationBodyTypeManager`
  (`0x226000`) and `CPedComponentSetManager` (`0x228000`), but neither maps a component variation
  index to a drawable plus texture. The first is height and body-shape presets
  (`SCALE_PLAYER_STORY000_00` is 1.62 m); the second lists gameplay components. Slot selection
  from names is a heuristic, not the game's rule.
- **Lucia's hair.** Her three named hair drawables are 3-vertex / 1-triangle placeholders. Across
  all characters, 241 of 329 hair rows are 3-vertex placeholders, but 68 have more than 100
  vertices, so real hair geometry does exist for some characters. A static export of Lucia is bald
  by design, not by a missing decode.
- **`levels.rpf` names.** The archive has not been name-enumerated in this workspace; the negative
  for a Lucia body is scoped to the named `packs.rpf` records above.
- **Bind pose and full skinning.** The delivered GLBs use identity inverse-bind matrices and do
  not apply the 8-influence bone-palette skinning. Prior art places inverse bind matrices at the
  skeleton header `+0x08`; that is not yet reproduced here, so the reference TRS remains the only
  measured pose.

## Reproduce

```powershell
rpf9dump models <packs.rpf> <outdir>
python - <<'PY'
import json
items = json.load(open(r"<outdir>/packs/models/manifest.json", encoding="utf-8"))["Items"]
names = sorted({i["Name"] for i in items})
print(len(items), len(names))
print(sum(n.lower().endswith(".#ft") for n in names), "fragments")
print([n for n in names if "lucia" in n.lower()])
PY
```

The fragment and skinning fields are in the same manifest; the skeleton measurements are in
[character_skeleton.md](character_skeleton.md), and the world model export is in
[models.md](models.md).
