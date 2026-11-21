# Design notes

Groundwork for running the Xbox Series build of GTA VI natively on Windows. The work that does not need the
executable is in place; booting can start as soon as `game_scarlett_master.exe` is available.

## Ground rules

- Everything here runs from **your own copy** and stays on **your PC**. Game files, generated code, shader caches and
  reports that quote game data are never committed, uploaded or shared. `reports/` is git-ignored for this reason.
- Nothing in this folder decrypts, unpacks or works around Xbox package protection. The tools only read files that
  are already plain on disk.

## Why a loader before a recompiler

The Xbox Series CPU is x86-64, the same as a PC. Unlike Hookshot, where ARM64 code has to be
translated, most of the game's code can run as-is. The work is everything around it:

| Layer | Approach | Folder |
|---|---|---|
| CPU code | Map the PE image and run it natively. Patch only what can't run on desktop (privileged instructions, console-only intrinsics). | `runtime/src` |
| Imports (GameOS / GDK / Xbox DLLs) | Every imported `dll!function` goes to a shim. Unimplemented ones log once and return a failure, so the missing surface shows up at runtime. | `runtime/shims` |
| Graphics | Xbox D3D12 (`d3d12_xs`) to Vulkan. Shaders ship as compiled RDNA2 machine code; `tools\rdna2spirv.py` translates all 26,503 to SPIR-V. The D3D12 layer is still to do. | `tools\` (translator), later (D3D12) |
| Assets | `.rpf` archives (RPF9) read through the runtime's file layer. | later |

## Layout

```
tools/       Python analysis tools (no game data inside)
  pe_inspect.py   PE headers, sections, entropy, imports sorted by where they resolve, .pdata, TLS
  rpf_probe.py    RPF header survey of a game folder
  manifest.py     SHA-256 manifest of a game folder, plus a --verify mode
scripts/
  first_look.ps1  run once the exe arrives: the full report set for it
runtime/     C++ native loader (CMake + Ninja, MSVC)
  src/            PE mapper, relocations, import binding, TLS
  shims/          per-DLL shim tables
reports/     generated output (git-ignored)
docs/        notes
```

## Day one, when the exe is here

```powershell
.\scripts\first_look.ps1 -Exe "<game>\game_scarlett_master.exe"
```

This writes `reports/exe/` (JSON plus a readable summary). Check it in this order:

1. **Section entropy.** Sections near 8.0 bits/byte are still encrypted or compressed. If `.text` is one of them,
   nothing else can be done yet.
2. **Import table.** Each DLL is marked `bundled` (in the game folder), `system` (present on this Windows), or
   `shim` (needs a reimplementation). The `shim` list is the real to-do list.
3. **.pdata function count.** This gives function boundaries for patching and later analysis.

Then `runtime\build\gta6rt.exe --dry-run <exe>` maps the image and binds every import to the shim table without
running any game code. It reports what it couldn't bind.

## Reproduce the shader tests

Build the hand-written tests where LLVM supports AMDGPU and SPIRV-Tools is available. Run the generated shaders
with the Vulkan test tools on the target GPU. Use the same output folder in both commands.

```text
python -I runtime/test/shaders/build.py <out>
python -I runtime/test/shaders/check.py <out> [runtime-build-directory]
```

The wave-mask fixture checks every lane and the wave64 ballot fixture checks mask counts and first-set lanes
across both halves of a wave. Rebuild `vk_compute_test` before running `check.py`. It uses a full 64-wide compute
subgroup for these two checks, and the summary reports them as skipped if the device cannot provide one.
`cs_exec64` still checks per-lane EXEC writes on 32-wide subgroups. The flat-input fixture checks both integer
IDs and interpolated UVs. The corpus comparison is a separate test and uses shader files from the local build.

## Build the runtime

```powershell
.\scripts\build_runtime.ps1
```

It sets up MSVC through `vcvars64.bat` and builds at BelowNormal priority on half the cores, writing to `runtime\build\`.
If VS's "C++ CMake tools" component is installed, it uses CMake and Ninja. Otherwise it calls `cl` directly. That
component may not be installed, and the devkitPro MSYS `cmake` on PATH can't build MSVC projects. When you
add a source file, add it to `runtime\CMakeLists.txt` and to the file list in `build_runtime.ps1`.

## Asset status (no exe needed)

| Asset | State | Where |
|---|---|---|
| Archive data | Carved without the encrypted tables of contents. Most archives are 93–99% plain zlib (`levels.rpf` 98.3%, 1.1M files). Not zlib: `audio.rpf` (99.9%), about half of `story000.rpf` and `shared_minigame000.rpf`. | `<work>\carve\<archive>\`, `coverage.json` |
| Videos | 797 Bink 2 files (`KB2n`), cut out whole. | `*.bk2` in the carve folders |
| Shaders | 26,503 carved from `shaders.rpf`, native RDNA2 code unpacked (`FSSM`), disassembled, and translated to SPIR-V; all validate (see *Translator status*). | `<work>\shaders\` (`.dxbc`, `fssm\`, `disasm\`, `spirv\`) |
| Textures | **Solved.** Headers are in CPU segments: GPU pointer at q[k], u16 width/height/depth and DXGI format at q[k+4], mip count at byte 2 of q[k+5]. Data is AMD `SW_4K_S` tiled, largest mip first. The GPU segment is the blob run after the CPU blob. On 200 samples, 173 decode at least 1.5× smoother detiled than untiled (median 2.56×). Exports the top mip, slice 0, as DDS. **22,342 exported (1.23 GiB)**: every header whose GPU data follows its CPU blob. World and model textures are handled by the TXD and TTPF exporters below. GPU offsets must be 256-byte aligned, which drops the false headers in `navmeshes`/`anim`. | `tools\tex_headers.py`, `tools\tex_export.py`, `<work>\textures\` |
| Models | **Done: 155,598 models** (levels.rpf 149,811 with 1.9 billion triangles, story000 3,303, data 1,350, shared_minigame000 783, packs 314, core_restofworld 37; 0 failures). Exported with `tools\rpf9dump` (built on breadbbbb/rpf9-explorer's Core; see below). Resources are found at page offset +16 after 16 zero bytes, followed by a zlib system blob with a drawable/fragment header and then the following zlib blocks as graphics data. Positions are quantized: `bias + scale * ((q >> 8) & 0xFFFFFF)`. Output is OBJ (positions, UVs, indices) plus a manifest with real model names (e.g. `cvar_int_grg2b_toys_1`), bounds, per-mesh shader index and material texture references (parameter hash, name hash, 64-bit asset key). My earlier `0x1380`/`0x1601` descriptor notes stand, but the "data not located" blocker was wrong. | `<work>\assets\<archive>\models\` |
| Texture dictionaries | Same tool: TXD resources become DDS per texture. | `<work>\assets\<archive>\textures\` |
| World textures (TTPF) | **Done: 308,895 textures from all 57 pools, 0 failures, 48,220 entries skipped** (33,507 BCPack, 14,030 compact mapping, 678 arrays/volumes, 5 unsupported format, 0 unmapped). 46 pools are in `data\ttdb\**`, 11 in `pack\<name>\data\ttdb\**` (exported under `ttpf_pack\<pack>\`). `data\ttdb\**\*.ttpf` (57 pools, 58 GiB) are loose RDS5 texture pools, **not** inside RPFs. rpf9-explorer's `TtpfReader` reads the catalog and detiles them; checked visually (`ttdb_graphics`: scope reticles, clean lines). Names are generated (`texture_N_<assetkey>`); models reference textures by asset key. Some rows are skipped; the measured breakdown and the BCPack next step are in [textures.md](textures.md). | `<work>\assets\ttpf\<pool>\` + `manifest.json` |
| Animation | Indexed without the TOC: 130,470 zlib streams (4,203,762,662 stored / 11,593,460,132 inflated), 95.4% readable. New RAGE page directories (`0DRP`, version 122), `MoVE` (version 19), 273 `pack:/` name tables (2,298 paths), and a tail `RBF0` rig-set manager (`rage__crRigSetManager`: 1,126 rig sets, 6,677 source→rig entries, 3,669 distinct rigs; 56 empty). Bone data: one standalone 28-name table and 3 motion-matcher streams with embedded bone tables (30-name union, 19 joaat32 hashes present, 4 motionmatcher paths); no rig→skeleton link was found. No `YCD`/`RSC7` at any stream start. The `rpf9_carve` tail overread is measured in `docs/animation.md`. | `tools\anim_index.py`, `<work>\anim-index\` |
| Audio | `audio.rpf` holds raw (not zlib) **AWC** banks: Opus (codec 0x0B) and XMA2 streams. Exported as whole `.awc` files plus `audio_manifest.json` (stream ids, codec, channels, rate, duration). **Done: 7,162 banks, 27 GiB, 1,603,231 streams, about 1,324 hours** (codec 0x0B Opus 1,601,167; 5: 1,818; 2: 227; 0 PCM: 19). `tools/awc_opus.py` repackages every Opus stream **losslessly** into a standard Ogg Opus file (no re-encode; PCM streams become .wav). The AWC framing is 2048-byte blocks of [u16 length][packet], with a zero length padding the block (as rpf9-explorer's OpusPreview reads it). The final granule is trimmed to the declared sample count. Checked: 300 random streams decode with ffmpeg with no errors, lengths match the declared counts. XMA2 streams (codec 5, 1,818) are written as RIFF XMA2 files by `rpf9dump xma` (rpf9-explorer's `AwcRecovery.WriteXmaWave`) and decoded to PCM WAV by ffmpeg: all 1,818 decode. Streamed music (tag 0x48: chunks with a per-channel record table, each channel's 2048-byte subblocks after the aligned header, and a repeated-prefix skip per chunk) becomes one mono Ogg Opus per channel (3,008 banks contain music). Checked: a 4-channel music bank gives four 42.0 s channels with real levels and no decoder errors. Output: `<work>\assets\audio\ogg\<bank>\<stream id>.ogg`. **Done: 1,598,051 Opus streams + 3,116 music channels as Ogg Opus (22 GB), 19 PCM16 and 227 codec-2 streams (32-bit float PCM, 4 bytes per sample; values can exceed ±1) as WAV, 1,818 XMA2 as WAV. Every audio stream is converted; all WAVs decode.** `make_game.py --wav` also decodes the Ogg Opus output to 16-bit PCM WAV with `tools/audio_wav.py` (ffmpeg required; existing WAVs are skipped on resume). | `<work>\assets\audio\awc\` |
| World placement | **Measured in `levels.rpf`:** map files occupy a continuous band from ~7.5 to ~62.9 GiB; 9,918 map files / 680,228 entities in 29 GiB of the 57 GiB band (50.9%, lower bound; block estimate 16.0k-39.6k maps / 1.03M-2.64M entities). `0x7a` resources: `CMapData` root, 0x90-byte `CEntityDef` with position/quaternion/scale/archetype/flags/GUID/LOD fields; names and most archetypes are hash-only. | `tools\world_index.py`; [world.md](world.md) |
| Navmeshes | `navmeshes.rpf`: 90,295 zlib streams (6,540,238,392 stored / 20,551,068,624 inflated), 97.5% readable; 9,603 `0DRP` pages and 27,644 `0x00010015` pages, no `NAVM`/`CNavMesh`/`NVMS` magic. `0DRP` is generic RPF paging, not a navmesh container. | `tools\navmesh_index.py`; [navmesh.md](navmesh.md) |
| BVH | `core_bvh4.rpf`: 565,596 streams, 271,627 complete BVH4 streams + 10 eight-MiB split continuations, 269,106 instance records. Header `0x81020006` at +0x100, 64-byte node units, 208-byte instances with a 3x4 transform; prebuilt acceleration data, not source meshes or shaders. | `tools\bvh_index.py`; [bvh.md](bvh.md) |
| RBF0 | 1,263 `data.rpf` resources, 475,461 elements, 44,887 descriptors and 340,859 strings. Inline record stream with no section table; UI/Scaleform (`rage__fwui*`, screens, subtitles, settings). The `anim.rpf` tail uses the same format (`rage__crRigSetManager`). | `tools\rbf0_index.py`; [rbf0.md](rbf0.md) |
| Gamedata | `common.rpf`: 954 streams (5,647,255 stored / 50,231,123 inflated), 85.5% readable; 266 XML documents / 118 roots (time/weather, vehicles/handling, peds/schedules, tuning, manifests, world/navigation). No RBF0/PSIN. | `tools\gamedata_index.py`; [gamedata.md](gamedata.md) |
| PSIN | `script.rpf`: 671 streams start with `PSIN`, 670 complete; sections `LSCH`, `LENM`, `LSNS`, `LGID`, `LRSC`, `CHKS`. `data.rpf`: 2,281 streams, 1,045 complete after joining. `LSNS` is high-entropy; no plaintext scripts or names. | `tools/psin_index.py`; [psin.md](psin.md) |
| Audio config | `audio.rpf` first 19.5 MiB: 837 zlib streams / 54,143,493 inflated bytes. 1,589 music records, 1,589 distinct ids, exact milliseconds; `hash & 0x1fffffff` is the AWC stream id. 1,584/1,589 durations match the AWC manifest / 2; no per-track start offsets. | `tools/audio_config.py`; [audio_config.md](audio_config.md) |

### rpf9dump (bulk export)

`tools\rpf9dump` is a small CLI over the Core library from https://github.com/breadbbbb/rpf9-explorer (a friend's tool; source at `third_party\rpf9-explorer`, audited: no keys, no network, no custom build targets). It decrypts nothing. It carves validated resources, as `rpf9_carve.py` does, but with working model and texture decoders.

```
dotnet build tools\rpf9dump\rpf9dump.csproj -c Release -o <work>\tools\rpf9dump
rpf9dump models|textures <archive.rpf> <work>\assets
rpf9dump ttpf <pool.ttpf> <work>\assets\ttpf
rpf9dump audio <audio.rpf> <work>\assets
python -I make_game.py --stages assets   # everything, in order; skips what is done
```

Full-scan indexes are also saved to `%LOCALAPPDATA%\RpfPreview\Indexes`, so the rpf9-explorer GUI opens these archives with complete listings instead of its 16 MiB preview scan.

## Other prepared pieces

- `tools\shader_profile.py`: across the corpus there are 551 distinct RDNA mnemonics, and 204 of them cover 99% of
  all instructions (`<work>\shaders\profile.json`).
- `tools\shader_ir.py`: parses every disassembled instruction into opcode, typed operands (s/v registers and
  ranges, NSA register lists, constants, literals, export targets, interpolation attributes), modifiers and
  encoding bytes. On 1,000 shaders, 12 of 2.2M operands were left untyped before the last fix. This is the front end
  for an RDNA→SPIR-V translator.
- `tools\rtti_scan.py`: MSVC x64 RTTI scanner (class names, bases, vtables). Tested on the bundled DLLs; run it on
  the exe first.
- `tools\hashdict.py`: joaat hash→name dictionary (`<work>\hashes.json`, 1.9M entries, checked against
  `adder` = 0xB779A091). On day one, rerun with `--extra <exe>` so PSO field and type names resolve. Without the exe
  only about 1% do.
- Runtime shims: all of `xmem.dll` is shimmed with the measured argument shapes (`XMemAlloc`, `XMemFree`,
  `XMemVirtualQuery(process, address, out64)`, `XMemGetWorkingSetStatistics(out)`). The `d3d12_xs` root-signature
  serializer forwards to the PC `d3d12.dll` after clearing the Xbox flag byte; the deserializer forwards unchanged.
  `D3D12XboxCreateDevice` returns an Xbox-ABI proxy over the PC device: every PC vtable slot is forwarded with
  `this` adjusted, Xbox slot 62 is `CreateComputePipelineState`, and Xbox slot 76 maps resource creation to
  `CreateCommittedResource` with the Xbox `0x101`/`0x1` layout word replaced by the PC layout. Slots 9/10 are
  `ID3D12DescriptorHeap` methods, not device methods. The PC heap getters take a hidden return pointer (RDX),
  while Xbox expects the 8-byte handle in RAX, so slot 14 wraps the heap and slots 9/10 convert it. The heap holds
  a reference to the device proxy and its `GetDevice` returns that same proxy. Queues, fences, command lists and
  `PresentX` extensions remain unimplemented.
- PSO metadata is a new variant (`PSIN`, `LSCH` schema, `LENM`, `LSNS`, `LGID`, `LRSC`, `CHKS`, little-endian).
  CodeWalker's parser reads the container but not the new schema; dump it once the exe gives the field names.

## Shader translation (RDNA → SPIR-V)

`tools\rdna2spirv.py` (needs `spirv-as`/`spirv-val` on PATH: Vulkan SDK, or `spirv-tools` in WSL) translates parsed RDNA
code to SPIR-V assembly, assembles it with `spirv-as`, and validates it with `spirv-val` (Vulkan 1.2). It runs one
lane per invocation and keeps VGPRs and SGPRs as uint arrays with exact bit patterns. Version 0 treated EXEC as
all-ones and covered only branch-free vertex and pixel shaders with no memory loads: common float and integer ALU
ops and exports to `gl_Position`, params and MRTs. Unsupported instructions are reported by name, never guessed.
In that version, `VS_Passthrough` and `VS_BlurFilter` translated and validated. Coverage across the corpus
and the blocking instructions are in `<work>\shaders\spirv_run.log`; the former next steps, structured branches
(EXEC masks) and the descriptor/binding model for `s_load`, `s_buffer_load`, `buffer_*` and `image_*`, are now
implemented. Current coverage is in *Translator status* below.

Also handled: 64-bit scalar moves, per-invocation EXEC reads, whole-quad mode on EXEC, `m0`, SCC, vector and scalar
compares, `v_mul_hi_u32`, `v_cvt_pkrtz_f16_f32` (now exact round-toward-zero; the earlier round-to-nearest version
could overflow to inf), and pixel interpolation (`v_interp_p1/p2/mov` become fragment inputs at their attribute's
location). The float MODE register (`s_setreg`) is skipped, so Vulkan's default denormal and rounding behaviour applies.

### GPU tests (RTX 4060 Ti)

`runtime\test\vk_harness.cpp` (build: `runtime\build\test\build_harness.cmd`, MSVC + the Vulkan 1.4 headers from
Hookshot's CPM cache, loading `vulkan-1.dll` at runtime) draws one triangle with a given VS + FS into an RGBA32F
target and dumps the floats. Test shaders are in `<work>\gputest`.

- **`VS_Passthrough` (translated) is bit-exact.** All 64×64 pixels are written, and `param0.xy` = (u, 1 − v) with max
  error **0**, with zw = (0, 1) as exported. This covers the VertexIndex input, integer/float ALU, inline constants,
  and position/param exports.
- **400 arithmetic-only translated pixel shaders** (no buffers, descriptors or subgroup ops), driven by a generic
  full-screen VS feeding locations 0–15: the driver compiled every pipeline, every draw completed, and every output
  was finite. This is a stability check, not a value check (there's no reference for these).
- **Texture path is exact** (`runtime\test\vk_texture_test.cpp`): translated `PS_PassthroughCopy` gives max error
  **0.0** over all 64×64 pixels against the expected nearest-sampled 8×8 test texture. That covers push-constant
  user data, a heap-relative `s_load_dwordx8` of the T#, the hash-table lookup of the T# and of the shader's
  literal S#, bindless sampling, `v_interp`, and the f16 compressed export. **It also caught a translator bug**
  that validation can't: compressed exports read the wrong register pair (LLVM prints `v4, v4, v5, v5`), giving
  (r,g,r,g). That's fixed.
- After the export fix, 2 of the 400 smoke-test shaders (`PS_ImposterShadow` variants) gave non-finite red values.
  The cause was round-to-nearest `pkrtz` overflowing to inf, fixed with exact RTZ. One variant is now fully finite.
  The other still has 1 inf and 4 NaN in red, and RTZ can't create those from finite values, so they come from the
  shader's own maths on the synthetic test inputs (hardware would pass them through too). That's consistent with
  hardware, but unconfirmed without a reference.

### Translator status and runtime contract

Full corpus (verified pixel-shader inputs incl. front face, ACO-confirmed 16-bit semantics, `vcc_hi` as an SGPR,
code-embedded data via `s_getpc_b64`, global memory, exact packed-f16 modifiers): **All 26,503 shaders translate and pass `spirv-val`** (Vulkan 1.2 target, `--allow-offset-texture-operand`): 2,236 VS, 12,101 PS, 2,689 CS, 9,282 mesh, 195 geometry. `spirv/index.json` lists each one's requirements. Output:
`<work>\shaders\spirv\` (`.spvasm` + `.spv`), and the log is `<work>\shaders\spirv_run.log`.

How the generated SPIR-V expects the runtime to work:

| Piece | Contract |
|---|---|
| Execution model | One invocation = one lane. A wave = a subgroup, so cross-lane ops (`ds_swizzle`, `v_permlane*`, `v_readlane`/`v_writelane`, `v_mbcnt`) become subgroup shuffles. **Requires subgroup size == wave size** (wave32 maps directly onto NVIDIA GPUs; wave64 shaders need a 64-wide subgroup or emulation). |
| EXEC | A per-invocation `active` flag. VALU writes and exports are masked by it, scalar ops aren't, and `v_cmpx`, saveexec and EXEC writes update it. A pixel lane still inactive at the end is discarded. |
| Control flow | Reducible graphs use structured branches and natural loops. Irreducible graphs use the dispatch-loop form, which can also be forced with `--dispatch-loop` for comparisons. The control-flow unit test checks 18,228 generated graphs. AMD A/B checks compare the two forms for selected pixel shaders with defined test inputs. For shaders that never export colour, the comparison checks pixel coverage. |
| User data | **Push constants, 144 bytes:** dwords 0–31 = initial s0–s31 (the user SGPRs the console driver loads from the root signature: descriptor-table pointers, constants and so on), dwords 32–33 = the 64-bit **heap base**, dwords 34–35 = the **code base**: the Xbox VA where the runtime uploaded this shader's whole FSSM blob. The runtime has to reproduce the console driver's root-signature-to-user-SGPR mapping; that comes from the exe or driver behaviour. |
| Xbox GPU addresses | Shaders build Xbox GPU virtual addresses, and some force the high dword to 0 (descriptor tables in the low 4 GB). Every memory access goes to **heap base + Xbox VA**, so the runtime maps the console's GPU address window into one host buffer-device-address allocation. |
| Code-embedded data | 155 VS/PS had "undecodable words". They're **constant tables inside the code range** (float values like 0x3EF7…, plus zeros), read through `s_getpc_b64` + `s_add_u32`/`s_addc_u32` + `s_load` (147 of 155). The translator re-anchors each instruction to its true FSSM offset by matching encodings, translates only code reachable from entry (a branch into data is rejected), and turns `s_getpc_b64` into code base + next-instruction offset. **The runtime uploads each shader's FSSM blob into the heap and passes that VA as the code base.** Re-anchoring compares bytes exactly, except that VOP3P `op_sel_hi[2]` (bit 14) is ignored on two-source packed ops: the compiler stores 0 there, and LLVM prints its default of 1. Data words that decode with ignored fields set are printed re-encoded (e.g. `0xbe8d2242` as `s_rfe_b64` `0xbe802242`), so they can never match. Since the decoder consumed the same length, they are anchored in sequence and marked. A shader that can *reach* a marked word is rejected; none of the corpus does. All 155 now translate. |
| Global memory | `global_load/store_dword*`: address = (64-bit SGPR base + zero-extended 32-bit VGPR offset, or a 64-bit VGPR address) + signed 12-bit offset, translated through the heap base. Inactive lanes don't access memory, and loads merge their previous values with OpPhi. |
| Literal samplers | Some shaders build their S# from immediates (for example `0x92, 0xC00000, 0x14500000, 0`). The runtime registers those in the descriptor hash table too, by scanning the translated shaders' constant S# values, not only S#s the game binds. |
| Scalar/buffer memory | `s_load*`, `s_buffer_load*`, `buffer_load*` and `buffer_store*` use buffer device addresses (`PhysicalStorageBuffer64`). The base comes from the 64-bit SGPR pointer or the V# (dword0 \| dword1[15:0] << 32), and the stride is V# dword1[29:16]. **The runtime must put the game's GPU heap in buffer-device-addressable memory at the same addresses the game writes into descriptors** (or remap them). See *Buffer bounds checking* below for `num_records` checks. |
| Typed buffer loads | The format is V# word3[18:12] (GFX10_FORMAT, from Mesa's `gfx10-rsrc.json`, saved as `tools\gfx10_formats.json`). Supported: f32 ×1–4, u32 ×1/2/4, f16 ×1/2/4, unorm8 ×1/2/4, snorm8 ×4, uint8 ×4, unorm16 ×1/2/4, snorm16 ×2/4, 10-10-10-2 and 2-10-10-10 unorm. Others give 0. DST_SEL swizzle is applied. |
| Images and samplers | `image_sample_l` on 2D arrays used to read its LOD from the array layer; it now reads the LOD after all coordinates. `image_get_lod` and `image_gather4_lz` also work on 2D arrays. |
| Images and samplers (core) | Bindless. Set 0: binding 0 = 2D sampled images, 1 = samplers, 2 = descriptor hash table (storage buffer), 3 = 2D-array images (**cube textures are bound here as 6-layer 2D-array views**: the shader already selected the face with `v_cube*`), 4 = r32ui storage images, 5 = 1D, 6 = 3D. |
| Descriptor hash table | `uint mask; entries[mask+1] of 10 uints: key[0..7], index, pad`. The key is a T# or a padded S#. FNV-1a over the eight key words selects the first slot. The shared SPIR-V lookup probes up to eight slots and returns index 0 on a miss. The runtime supplies a fallback at index 0 and maps live descriptor keys to bindless indices. |
| Vertex fetch | `s_swappc_b64` at the start of a VS calls a driver-generated fetch program that isn't in the blob. It's replaced by Vulkan vertex inputs: each `ISG1` input-signature element (DXBC container) becomes an input at **location = its D3D input register**, typed by its component type, and is written to **v(4 + 4·register + component)**. That rule was checked over all 788 fetch-calling shaders: in 783 every read-before-write VGPR is predicted by it (715 exact). Reads of components the element doesn't declare (e.g. `.w` of an `xyz` position, 3 VS) are allowed: the fetch returns the format defaults (0, 0, 1), and Vulkan fills a `vec4` input from a narrower format the same way. Inputs are declared as `vec4`. **The runtime must bind vertex buffers by the game's D3D12 input layout, location = input register.** |
| PS system inputs | v0–v1 = barycentrics (consumed by `v_interp`, which becomes attribute inputs). **v2.. = the `SV_Position` components the shader reads, packed in order**. Verified on 2,531 of 3,000 PS: the VGPRs they read before writing equal exactly the position components their input signature reads. Filled from `FragCoord`: xy/z directly, and **w directly too**, because AMD's position-w register holds 1/w (Mesa lowers `load_frag_coord_w_rcp` straight to `frag_pos[3]`, `ac_nir_lower_intrinsics_to_args.c`), the same as Vulkan's `FragCoord.w`. **Front face** follows the used position components when the signature declares `SV_IsFrontFace`, as ±1.0 (Mesa: `load_front_face` = `front_face > 0`). Only enabled inputs are packed, in hardware order (Mesa `radv_shader_args.c`: barycentric pairs, POS_X/Y/Z/W, FRONT_FACE, ANCILLARY, SAMPLE_COVERAGE, POS_FIXED_PT). That explains the earlier "unverified" shaders: 102 read pos.xy + front face, and 76 read pos.x/y/w + front face. **Raw barycentric reads** (321 PS, e.g. `v_readfirstlane_b32 s41, v1`): AMD interpolates P = P0 + i(P1 − P0) + j(P2 − P0), so i/j are the weights of vertices 1/2, which is `BaryCoordKHR.y`/`.z` (`VK_KHR_fragment_shader_barycentric`). GPU-checked with `runtime/test/shaders/ps_bary.asm`: raw i/j equal the interpolated weights on every pixel. This assumes the pair is perspective-center, the only pair this layout models. A PS that reads any other unexplained register inside the system-input range is still rejected (none left in the corpus). Reads past the last hardware input are values undefined on some path, so zero stands in for them. All register arrays start at zero, so translated shaders are deterministic. |
| Inputs and outputs | VS: `v0` = `VertexIndex` (**unverified**: if the console runs vertex shaders as NGG, the input VGPR layout differs). `exp pos0` → `Position`, `exp paramN` → location N. PS: `v_interp_*` attribute N → input location N. `exp mrtN` → location N, `exp mrtz` x → `FragDepth`. |
| 16-bit results | From Mesa ACO (`aco_ir.cpp`: `instr_is_16bit`, `can_use_opsel`) for GFX10: **f16 VOP1/VOP2 ops** (`v_add/sub/mul/max/min_f16`, `v_fmac_f16`, `v_cvt_f16_f32`, f16 unary) and **opsel-capable VOP3 ops** (`v_fma_f16`, `_e64` forms of the 16-bit integer ops) are **partial writes that keep the high half**. The **VOP2 (`_e32`) 16-bit integer ops** (`v_lshrrev_b16`…) define the whole register, so the high half is modelled as zero. An op_sel destination bit writes the high half. The translator used to clear the high half on f16 ops; that's fixed. **op_sel on f16 VOP3 ops** (`v_fma_f16`, `v_med3_f16`, `v_add/sub/mul/max/min_f16`, f16 unary, `v_fmac_f16`): bit i (i < number of sources) reads the high half of source i, the last bit writes the high half of the destination; neg/abs apply to the selected half. Source: Mesa ACO `can_use_opsel` and the RDNA2 ISA VOP3 `OP_SEL` field. Checked by the structural tests in `tests/test_f16_op_sel.py` (which half is read, where the result goes); **not yet GPU-verified**. |
| POPS | Primitive-ordered pixel shading (10 PS, `src_pops_exiting_wave_id` wait loop, `s_sleep`, `MSG_ORDERED_PS_DONE`) becomes **fragment-shader interlock** (`PixelInterlockOrderedEXT`). It begins at the end of the entry block and ends at shader exit: a longer critical section with the same ordering. `src_pops_exiting_wave_id` reads 0, and **the runtime must put 0 (no collision, bit 31 clear) in the POPS collision SGPR**, so the hardware wait loop is skipped. Needs `VK_EXT_fragment_shader_interlock` (`fragmentShaderPixelInterlock`). |
| Texel offsets | `image_sample*_o`: the first address dword packs signed 6-bit offsets (x [5:0], y [13:8], z [21:16]), which become the SPIR-V `Offset` operand. A non-constant `Offset` outside gather needs **`VK_KHR_maintenance8`**, and is validated with `spirv-val --allow-offset-texture-operand`. Not yet GPU-tested. |
| Optional device features | Shaders may need `fragmentShaderBarycentric`, `fragmentShaderPixelInterlock` and `maintenance8`. All three are present on an RTX 4060 Ti, and `runtime/test/vk_harness.cpp` enables them when available. |
| Wave size | **Most of the game's shaders are wave64: 21,545 of 26,503.** An earlier marker scan (`s_mov_b64 exec, -1`, `exec_hi`, `s_and/or_saveexec_b64` and 64-bit masks) counted 9,930 of 12,101 PS, 8,666 of 9,282 mesh shaders, 1,115 of 2,236 VS and 923 of 2,689 CS. The translator maps a wave to a subgroup, and NVIDIA subgroups are 32 wide. What that means: (1) lane-independent code, which is most pixel-shader math, is exact; (2) wave-wide branch decisions and reductions are made per 32-lane half, which is exact unless scalar state computed for one half is later consumed by the other; (3) lane indices from `v_mbcnt_lo/hi` and wave IDs from the merged-wave SGPR must be 64-lane values. Mesh and compute shaders build thread IDs as wave_id·64 + mbcnt, so those need a mapping from `LocalInvocationIndex` (lane = index & 63, wave = index >> 6). Full fidelity needs either 64-wide subgroups (AMD GPUs, via `VK_EXT_subgroup_size_control` requiredSubgroupSize = 64) or emulating cross-half ops (ballot, readlane, shuffles across halves) through workgroup memory. |
| Mesh shaders | NGG wave64 programs → `VK_EXT_mesh_shader` (`MeshEXT`). **PSV0** (DXBC) gives NumThreads (128×1×1 in the corpus), MaxOutputVertices/Primitives (128/128) and the topology (triangles). System inputs, from Mesa's RADV setup for mesh as NGG on GFX10.3: v5 = workgroup index (the "vertex_id" slot; the game reads it with `v_readfirstlane_b32 s2, v5`), filled from WorkgroupId/NumWorkgroups; the merged-wave-info SGPR (s3 in this ABI) = wave ID << 24, from LocalInvocationIndex >> 6; `v_mbcnt` lanes = LocalInvocationIndex & 63; user data in s16.. as usual. 1,473 of 1,500 sampled mesh shaders read exactly v5, and the rest only add loop-carried registers. Exports: on NGG, thread t exports vertex t (`pos0`, `paramN`) and primitive t (`prim`). They're staged per invocation; an all-`off` export is the dummy a wave with no vertices must issue, and writes nothing. `s_sendmsg GS_ALLOC_REQ` (m0 = prims << 12 \| verts, sent by thread 0) stores the counts in a workgroup variable. At shader end, in uniform control flow: workgroup barrier, `OpSetMeshOutputsEXT(min(verts, max), min(prims, max))`, then each thread writes its staged vertex (Position + Location N) and primitive (`PrimitiveTriangleIndicesEXT` = bits [8:0], [18:10], [28:20]; `CullPrimitiveEXT` = bit 31). pos1+ (clip/cull distances) not modelled yet. GPU-checked with `runtime/test/shaders/ms_tri.asm` (`VKH_MESH=1 vk_harness`), built from the game's own idioms (thread ID from s3 and mbcnt across two waves, thread-0 ALLOC_REQ, per-thread exports): 0 error on every pixel. |
| Geometry shaders | All 195 are `$GS_VoxelizeGeneric` permutations: legacy (merged ES/GS) wave32 programs on LDS rings, with no `exp`. Inputs: vertex i sits at the dword offset in v0 lo / v0 hi / v1 lo; an LDS read at vertex base + 4·(4·reg + comp) is input-signature register reg, component comp (checked: SV_Position xyz in dwords 0–2 and TEXCOORD0.xy in 4–5, as ISG1 declares). Outputs: emitted vertex k is dwords [k·4·nreg, (k+1)·4·nreg) of the lane's GSVS area, dword = 4·reg + comp of the output signature (20 dwords = 5 OSG1 registers; every write matches a declared mask). Translation: synthetic offsets (vertex i at dword 64·i), LDS reads → `gl_in[i]` inputs, LDS writes → staged slots, `MSG_GS EMIT` → copy slot (emit count) to the outputs + `OpEmitVertex`, `CUT` → `OpEndPrimitive`, `GS_DONE` → nothing. The prologue's `s_bfm_b64`/`s_cselect` EXEC setup is fed s3 = 64 << 8. GPU-checked with `runtime/test/shaders/gs_tri.asm` (`VKH_GS=… vk_harness`): 0 error. |
| SCC from scalar ALU | `s_and/or/xor/andn2/orn2/nand/nor/xnor_b32` and `s_lshl/lshr_b32` set SCC = result ≠ 0, `s_add/sub_i32` and `s_addk_i32` set it to signed overflow, `s_min_u32`/`s_max_u32` to S0 < S1 / S0 > S1. **Before this fix these ops left SCC unchanged**, so a following `s_cbranch_scc*` read a stale flag. 32-bit scalar shifts now mask the amount to 5 bits. GPU-checked with `runtime/test/shaders/ps_scc.asm`. |
| Far branches | `s_getpc_b64; s_add_u32 K; s_addc_u32 0/−1; s_setpc_b64` on the same pair is a branch beyond `s_branch`'s ±32K-dword range. It's folded into one branch to (address after `s_getpc`) + K before CFG construction. |
| Buffer format stores | `buffer_store_format_*` and `_d16_*` encode every format the loads decode (the same FORMATS table: f32/u32, f16, unorm/snorm 8/16/10-10-10-2, uint 8/16). 1–2-byte elements are merged into their dword with atomic and/or so neighbours are kept. d16 data is unpacked first (f16 → float for float kinds, 16-bit integers for integer kinds). Before this, only four f32 formats, rgba16f and rgba8 unorm were stored, and everything else silently stored nothing. GPU-checked with `runtime/test/shaders/cs_store.asm` (rgba8 unorm, 10:10:10:2, two r16f halves of one dword, d16 rgba16f: exact, neighbours untouched). `buffer_load_format_d16_*` packs pairs the same way. |
| Other additions | `image_gather4_l(_o)`: no explicit-LOD gather in core Vulkan, so four `OpImageFetch` at mip round(LOD) in gather order, coordinates clamped to the edge (**approximation**: the sampler's wrap/mirror isn't reproduced at the border). `image_gather4_lz_o`: `OpImageGather … Offset`. `image_store` on 1D (bindings 17–19, `Image1D`) and with a16 coordinates. `ds_swizzle BITMASK_PERM` ("pippi" masks; GPU-checked). `v_add_f64`, `v_cvt_f32_f64` (`Float64`). `v_mul/mad_i32_i24`, `v_alignbyte_b32`, `v_xnor_b32`, `v_ffbh_i32`, `v_swap_b32`, `s_abs_i32`, `s_bfm_b32/b64`, 64-bit ordered and `v_cmpx` compares, 16-bit compares, 8/16-bit global loads, `neg(…)` constant operands (GPU-checked where listed in `runtime/test/shaders`). |
| Ray tracing | `image_bvh_intersect_ray` / `image_bvh64_intersect_ray` (64 CS: RT shadows) are one traversal step against AMD's hardware BVH node format. The traversal loop lives in the game's shader. The step is emulated over buffer reads following Mesa RADV's software path (`radv_nir_rt_common.c`, `intersect_ray_amd_software_box/_tri`; node layouts from `bvh_defines.h`). Node id = address >> 3 \| type (0–3 triangle, 4 box16, 5 box32). Box nodes give the four child ids, with misses and NaN boxes = `0xFFFFFFFF`, sorted by entry distance when descriptor bit 63 (box_sort_enable) is set. Triangle nodes use the watertight test with RADV's shared edge/vertex tie-breaks and give (t/\|det\|, sign(det), v/\|det\|, w/\|det\|), or (inf, 1, 0, 0) on a miss. The game's code divides by `.y` exactly as for the hardware's numerator/denominator. Address = descriptor base + ((node & ~7) << 3), base = descriptor bits [39:0] in 256-byte units (**assumption**: RADV only uses base 0). **The runtime must build the game's acceleration structures in this AMD node format** (RADV's GPU BVH builder produces it) wherever the game's D3D12 calls build them. Far branches with `s_sub_u32`/`s_subb_u32` (backward) are folded too. GPU-checked with `runtime/test/shaders/cs_bvh.asm`: sorted box children and the triangle result are exact. |
| Coverage | Every shader in the corpus translates. The remaining fidelity gap is wave64 cross-half code on 32-wide subgroups (see the wave-size and wave64 rows). |
| Flat pixel inputs | A pixel input location read only by `v_interp_mov_f32` is declared `Flat`. Locations read with `v_interp_p1/p2` remain interpolated. The `ps_flat` GPU test passes integer IDs through the first vertex alongside interpolated UVs. |
| Whole-wave masks | `tools/lane_masks.py` tracks per-lane mask bits through scalar registers, vector operations and saved EXEC spills. Whole-mask counts, first-set-lane queries and equality use subgroup operations. `cs_wavemask` checks all 64 lanes with a partially active EXEC mask. A physical 32-lane subgroup cannot directly exchange mask bits with the other half of a wave64, so `cs_wavemask` and the `cs_wave64_ballot` / `cs_wm_*` tests are exact only on 64-wide subgroups (AMD): on an RTX 4090 (subgroup 32) `s_bcnt1_i32_b64` of lanes 16–47 counts 16, one half (#101). `check.py` skips them where the GPU has no 64-wide compute subgroup. |
| Floating-point operations | Separate floating-point instructions carry `NoContraction` so the driver preserves their individual rounding. Instructions that explicitly fuse multiply-add still emit `Fma`. |
| Wave64 handling | Wave size is detected per shader from the code (64-bit EXEC writes, `exec_hi`, `v_mbcnt_hi`, `*_saveexec_b64`; mesh shaders are always wave64). The hardware lane is exact in compute and mesh shaders (LocalInvocationIndex mod 64 or 32) and is used by `v_mbcnt` and EXEC writes. In other stages it's the subgroup lane. Constant EXEC writes (`s_mov_b32 exec_lo/exec_hi, K`) set each lane from its own bit of K; before, any non-zero constant turned every lane on. A register written to `exec_hi` sets lanes 32–63 only. GPU-checked with `runtime/test/shaders/cs_exec64.asm`: lanes 16–47 enabled from two constants, exact on a 32-wide-subgroup GPU. Still approximate on 32-wide subgroups: cross-lane ops between the two halves of a wave64 (readlane/permlane/DPP/shuffles across lane 31/32, wave-wide reductions). The metadata index marks every wave64 shader so a runtime can request `requiredSubgroupSize = 64` where supported (AMD), which makes those exact too. |
| Metadata index | `spirv/index.json`: one entry per translated shader with its stage, `wave64` / `subgroup_size`, SPIR-V capabilities and extensions, the Vulkan **features** to enable (barycentrics, interlock, mesh, geometry, Float64, Int64, buffer device address, write-without-format, …), `needs_maintenance8`, the descriptor bindings it uses (set 0), push-constant size, `local_size`, `user_sgprs` (CS), `lds_bytes`, mesh `max_vertices`/`max_primitives`/`params`, geometry inputs/outputs, vertex-input locations and types, full `vertex_input_signature` (ISG1 register/mask/system-value/component-type, for PSO lookup), `wave64_emulated` (true for VS/PS/GS wave64: 64-wide subgroups can only be requested for compute/mesh, so graphics wave64 stays emulated 2x32 until cross-half ops go through LDS), `dynamic_format` (runtime V# FORMAT: codes outside FORMATS decode to 0) / `fixed_format_codes`, `src_sha256` and optional `fssm_sha256` (via `--fssm-dir`), PS system inputs, and input/output locations. A runtime can create every pipeline at load time from it without parsing SPIR-V. |
| Wave-uniform branches | VCC and EXEC are wave-wide registers, but one invocation holds only its own lane's bit. `s_cbranch_vccz/vccnz/execz/execnz` therefore reduce the condition across the subgroup (`OpGroupNonUniformAny`), so the whole wave takes the branch together, as on hardware. Before this, each lane decided alone. A lane could skip scalar instructions that the hardware wave executes (its SGPRs went stale for later code), and cross-lane ops or barriers inside the block saw a split wave. `v_cmp`/`v_cmp_class`/64-bit compares now write 0 for inactive lanes (hardware VOPC behaviour), so the reduction sees the true mask. GPU-checked with `runtime/test/shaders/ps_uniform_branch.asm`: lanes whose own VCC bit is 0 run the block's scalar add when their wave has any set bit. `s_cbranch_scc0/1` stay per lane; SCC is uniform as long as the SGPRs feeding it are. |
| Compute shaders | FSSM header word **0x138 = `COMPUTE_PGM_RSRC2`**: USER_SGPR [5:1], TGID_X/Y/Z_EN [9:7], TG_SIZE_EN [10], TIDIG_COMP_CNT [12:11], LDS_SIZE [23:15] in 512-byte units. Words **0x13C/0x140/0x144 = thread-group size**. This was checked against what all 2,689 CS read before writing: 2,281 match exactly, and the rest overshoot only through loop-carried registers in the straight-line check. 0x110, which looked similar, doesn't match. Layout: s0..user−1 = user data (push constants), then the enabled workgroup IDs (`WorkgroupId`), and v0/v1/v2 = thread ID x/y/z (`LocalInvocationId`, separate VGPRs on GFX10). Example: the 1,013 shaders with RSRC2 `0x998` have 12 user SGPRs, group ID x/y in s12/s13, and compute x = s12·32 + v0. Shaders with TG_SIZE_EN are rejected for now. GPU-checked with `runtime/test/shaders/cs_lds.asm` through `runtime/test/vk_compute_test.cpp`: 4 groups × 64 threads, LDS exchange between the two waves of a group behind `s_barrier`, store through a V# into the device-address heap, 0 mismatches. |
| LDS | A `Workgroup` uint array sized from LDS_SIZE, rounded up to a power of two that also covers every constant offset (15 of 486 shaders index past LDS_SIZE). Covers `ds_read/write_b32..b128`, `read2/write2` (+`st64`), `u16`/`i16`/`u8`/`i8` reads (also `d16`/`d16_hi`), `b16`/`b8` writes (atomic and/or, so neighbouring lanes don't race), the add/sub/min/max/and/or/xor atomics with and without return, `wrxchg`, `cmpst` (`OpAtomicCompareExchange` with DATA as comparator and DATA2 as value), and `ds_bpermute_b32` as a subgroup shuffle. Writes and atomics run only in active lanes. Indices wrap at the array size, where hardware would return 0 for out-of-range reads. `s_barrier` = `OpControlBarrier` Workgroup/Workgroup, AcquireRelease + WorkgroupMemory. |
| Typed image stores | `image_store` (2D, 2D array, 3D) writes the dmask components (missing ones = 0). The image's numeric type lives in the T# (word 1 FORMAT [28:20], 512 GFX10 formats), so it is decided at run time. A 512-bit table built from `tools/gfx10_formats.json` (30 UINT and 15 SINT formats) picks a **sint**, **uint** or **float** view (bindings 8 + 3·dim + class; dim 0 = 2D, 1 = 2D array, 2 = 3D; class 0 float, 1 uint, 2 sint). Views have Unknown format (`shaderStorageImageWriteWithoutFormat`). **The runtime binds every storage image into all three arrays of its dimension, or at least the one matching its format.** The earlier path wrote only component 0 as `r32ui`, which was wrong for every other format. GPU-checked with `runtime/test/shaders/cs_imgstore.asm` through `runtime/test/vk_store_test.cpp`: rgba8 unorm, r32 uint and r32 sint images, every texel correct. |
| Typed image loads | `image_load` on 1D/2D/2D-array/3D images uses the same T# FORMAT tables as stores: UINT/SINT formats are read with `OpImageRead` through the uint/sint storage views (`shaderStorageImageReadWithoutFormat`), everything else with `OpImageFetch` through the float sampled view as before. Results are the raw bits of the dmask components. **The runtime binds integer-format images into the uint/sint storage arrays of their dimension** (bindings 8–19), as for stores. Before this, every load went through the float view, which is undefined for integer formats. GPU-checked with `runtime/test/shaders/cs_imgload.asm` (r32 uint read back exactly). |
| SDWA | `_sdwa` forms: each source is narrowed to `src0_sel`/`src1_sel` (BYTE_n/WORD_n, zero-extended, or sign-extended under `sext()`) **before** neg/abs and the op. A VGPR result goes to `dst_sel`, with `dst_unused` setting the other bits: PAD = 0, SEXT = sign copies above the field, PRESERVE = old bits kept. VOPC has no `dst_sel`, and GFX10 `v_cmpx` has no written destination. **Until 2026-11-20 the selects were silently ignored.** That affected 3,826 of the 13,926 translated shaders: they passed `spirv-val` but computed wrong values. GPU-checked with `runtime/test/shaders/ps_sdwa.asm` and `ps_ops.asm` (also `v_pk_fma_f16`, `v_min3_u32`, `v_max3_i32`, `v_bfe_i32`): byte/word source selects, `sext()`, `UNUSED_PRESERVE` into WORD_1 and `UNUSED_SEXT` at BYTE_2 all match on every pixel. |
| Known approximations | `v_cvt_pkrtz_f16_f32` is now **exact round-toward-zero** (finite overflow gives ±65504, not inf). The GPU smoke test caught the old round-to-nearest version producing inf in `PS_ImposterShadow`. The other f16 packs (`v_cvt_f16_f32`, d16 results, mix ops) round to nearest, which matches the default mode. The MODE register (`s_setreg`) is ignored, so denormal and rounding behaviour is Vulkan's default. Whole-quad mode on non-EXEC masks is the identity. `v_mul_legacy_f32` now keeps the `0 × anything = 0` rule like `v_mad_legacy_f32` (GPU-checked with `runtime/test/shaders/ps_legacy_mul.asm` on AMD Radeon RX 9070 XT). |
| Buffer bounds checking | MUBUF `buffer_load*`, `buffer_store*`, and `buffer_atomic*` enforce `num_records` (V# word 2): raw buffers (`stride == 0`) check the starting byte offset (`offset < num_records`; per-dword tail bounds on multi-dword accesses aren't split yet), and structured buffers (`stride != 0`) check `index < num_records` (`idxen`) or `offset < num_records * stride`. Out-of-bounds stores/atomics skip memory access completely and out-of-bounds loads return 0. GPU-checked with `runtime/test/shaders/cs_bounds.asm` on AMD Radeon RX 9070 XT. |
| Wave64 mask ballot | `tools/lane_masks.py` tracks per-lane masks (including 32-bit `MASK_LO`/`MASK_HI` halves, `vcc_lo`/`vcc_hi`, and `s_bfm_b64` bitfields). `s_bcnt1_i32_b64`/`b32` and `s_ff1_i32_b64`/`b32` reconstruct the 64-bit wave mask via `OpGroupNonUniformBitwiseOr` (`GroupNonUniformArithmetic`), restricting 32-bit half ops and `s_cmp_eq/lg_u32` (`wave_equal`) to the known 32-bit half in wave64. Known remaining gaps: `s_cmp_eq/lg_u32` comparing opposite halves (lo vs hi) falls back to a whole-wave vote; `s_bcnt0`, `s_ff0`, and `s_bitcmp*` have no lane-mask-aware path yet. GPU-checked with `runtime/test/shaders/cs_wave64_ballot.asm` and `cs_wm_*.asm` on AMD Radeon RX 9070 XT (64-wide subgroup). |
| Mesh/compute input registers (historical measurement) | An early read-before-write analysis over all compute and mesh shaders found **compute** v0 (880), v0–v1 (614), v0–v2 (50), consistent with AMD's local thread ID x/y/z in v0/v1/v2, and scalar inputs centred on a pointer at s[4:5] and s6/s7. **Mesh:** every one of 2,969 reads exactly **v5**, plus s3 (always arithmetic), s[16:17] and s19–s23. The early pass also read FSSM 0x13C as the group width and compared the first SGPR multiply against it (82 match, 151 don't, 1,447 no simple multiply) and wrongly concluded that 0x13C was not the group width; the matching register varied by shader (s6/s10/s7/s9…). PSV0 gave the real thread-group sizes (32×1×1 for 1,253, 8×8×1 for 416, …). **Superseded:** 0x13C/0x140/0x144 are the thread-group size, `COMPUTE_PGM_RSRC2` at 0x138 gives the per-shader user-SGPR count and enabled workgroup IDs, and mesh takes NumThreads, limits and topology from PSV0 with v5 as the workgroup index. All 2,689 CS and 9,282 mesh shaders translate now (see the Compute shaders and Mesh shaders rows). |
| Formerly untranslated | Mesh (9,282), geometry (195) and compute (2,689) now translate; `s_swappc_b64` is replaced by Vulkan vertex inputs (see *Vertex fetch*) and the `tbuffer_load_format_*` forms use the instruction's format. A few rare ops are still reported by name rather than guessed. |

**Main blocker at the time (superseded): descriptor loads.** On a 3,000-shader sample, `s_load_dwordx4`/`x8` stop over half of them. Shaders
load texture, buffer and sampler descriptors through user-data SGPR pointers, and how the game's root signatures
map onto those SGPRs is set up by game code. Reading that mapping from the exe (root signature creation plus the
FSSM user-data metadata) before building the binding layer was the next step; guessing it would give silent
rendering bugs. The translator now handles those loads against the push-constant user-data contract (*User data*,
*Scalar/buffer memory* and the descriptor rows); the D3D12 layer still has to supply the root-signature mapping.
The other blockers listed then (mesh, the vertex-fetch call, geometry, compute and compressed exports) are closed
for the corpus: all 26,503 shaders translate (see *Translator status*). The remaining fidelity gap is wave64
cross-half code on 32-wide subgroups.

## Day one, in order

1. `scripts\first_look.ps1 -Exe ...` checks the exe is plain, lists the imports to shim, and does a dry run.
2. Find the RPF9 table-of-contents key and decryption in the exe, then add a table reader to `rpf9_carve.py`. Carved
   blobs then get real names and types, and the remaining unread data shows what it is.
3. Find the texture and mesh resource structures in the exe (their loaders give exact field layouts), which turns
   carved blobs into named textures and meshes.
4. Shader translation: already complete for all 26,503 shaders (see *Translator status*); the day-one work is
   validating them through the runtime.

## Disassembling shaders

The AMDGPU disassembler needs LLVM 21+ built with the AMDGPU target (`llvm-mc`; on Linux the `llvm-21` package
installs it at `/usr/lib/llvm-21/bin/llvm-mc`). The official Windows LLVM build has no AMDGPU target and ships no
`llvm-mc.exe`, so on Windows run it from WSL and give `LLVM_MC` the whole command:

```
set LLVM_MC=wsl -d Ubuntu -e /usr/lib/llvm-21/bin/llvm-mc
python -I make_game.py --game "D:\path\to\your\package" --stages shaders
```

`shader_disasm.py` disassembles one `s_endpgm` before starting, so a missing tool or AMDGPU target stops the run
with that reason instead of writing empty `.s` files. To disassemble by hand inside WSL:

```
LLVM_MC=/usr/lib/llvm-21/bin/llvm-mc python3 -I <repo>/tools/shader_disasm.py <work>/shaders/fssm
```

Code starts at the u32 at FSSM offset 0xC0 (the code-section offset) plus 0x28 bytes into that section, and ends at the
last `s_endpgm`. Use `--mcpu gfx1013`. On 300 random shaders it decoded 284 cleanly (134 invalid words in total),
against 201 clean and 6,560 invalid for gfx1030 and 283 clean for gfx1010–1012.

## Pixel-shader inputs

`tools/fssm_header.py` decodes the FSSM input table: 8-byte records from 0x138 while the first byte is 6.
Byte 1 is the semantic id, byte 2 the component mask, byte 3 the running index, byte 6 the parameter slot,
and byte 7 holds flags (bit 4 = 16-bit interpolant). The record's position is the `v_interp attr` number.
`interpolants()` returns the count and a fp16 bitmask by position; a translator can use both, but
`rdna2spirv.py` does not yet. Measured on the 26,503-blob corpus: 25,643 satisfy
`u32[0x120] + u32[0x12C] == u32[0x130]` (the 860 that do not store 0xDEADBEEF at 0x130); 12,101 are PS,
10,307 with a non-empty table (55,791 records, 11,199 of them 16-bit). The count equals the number of
distinct `v_interp attr` numbers in 11,316/12,101 PS (93.5%); 300 more declare one unread input, 475
declare two or more, and the 10 undercounts are one family whose first record at 0x138 has kind 7. A
200-PS spot check agreed on 187/200. Flag bit 4 matches the `*_f16` interpolation opcodes in
12,067/12,101 (99.7%).

## Baseline already recorded

- `reports/manifest.json`: SHA-256 of all 108 files in `Documents\GTA6` (208.5 GiB). Run
  `python -I tools\manifest.py verify <game>` to check a copy against it.
- `reports/rpf_probe.json`: all 22 archives are **RPF9**. Five have tables that look packed or encrypted (`levels`,
  `movies`, `core_bvh4`, `navmeshes`, `data`).
- `reports/pe/`: the 16 bundled DLLs are plain x86-64 PE with no encrypted sections. Through the loader their
  `xmem.dll` imports and two `d3d12_xs.dll` root-signature calls bind to shims. Every other import resolves on
  desktop Windows. The proxy covers the measured device slots and forwards the rest in PC vtable order; the Xbox
  queue/fence/command-list extensions and `PresentX` remain unknown.
