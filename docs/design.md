# Design notes

Groundwork for running the Xbox Series build of GTA VI natively on Windows. It's set up now so the work can start
as soon as `game_scarlett_master.exe` is available.

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
| Graphics | Xbox D3D12 (`d3d12_xs`) to Vulkan. Shaders ship as compiled RDNA2 machine code, so they need a translator from RDNA2 ISA to SPIR-V. **Largest single piece.** | later |
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
| Shaders | 26,503 carved from `shaders.rpf`, native RDNA2 code unpacked (`FSSM`), and disassembly working. | `<work>\shaders\` (`.dxbc`, `fssm\`, `disasm\`) |
| Textures | **Solved.** Headers are in CPU segments: GPU pointer at q[k], u16 width/height/depth and DXGI format at q[k+4], mip count at byte 2 of q[k+5]. Data is AMD `SW_4K_S` tiled, largest mip first. The GPU segment is the blob run after the CPU blob. On 200 samples, 173 decode at least 1.5× smoother detiled than untiled (median 2.56×). Exports the top mip, slice 0, as DDS. **22,342 exported (1.23 GiB)**: every header whose GPU data follows its CPU blob. World and model textures in `levels.rpf` have the same blocker as meshes. GPU offsets must be 256-byte aligned, which drops the false headers in `navmeshes`/`anim`. | `tools\tex_headers.py`, `tools\tex_export.py`, `<work>\textures\` |
| Models | **Exporting** with `tools\rpf9dump` (built on breadbbbb/rpf9-explorer's Core; see below). Resources are found at page offset +16 after 16 zero bytes, followed by a zlib system blob with a drawable/fragment header and then the following zlib blocks as graphics data. Positions are quantized: `bias + scale * ((q >> 8) & 0xFFFFFF)`. Output is OBJ (positions, UVs, indices) plus a manifest with real model names (e.g. `cvar_int_grg2b_toys_1`), bounds, per-mesh shader index and material texture references (parameter hash, name hash, 64-bit asset key). My earlier `0x1380`/`0x1601` descriptor notes stand, but the "data not located" blocker was wrong. | `<work>\assets\<archive>\models\` |
| Texture dictionaries | Same tool: TXD resources become DDS per texture. | `<work>\assets\<archive>\textures\` |
| World textures (TTPF) | `data\ttdb\**\*.ttpf` (57 pools, 58 GiB) are loose RDS5 texture pools, **not** inside RPFs. rpf9-explorer's `TtpfReader` reads the catalog and detiles them; checked visually (`ttdb_graphics`: scope reticles, clean lines). Names are generated (`texture_N_<assetkey>`); models reference textures by asset key. Some rows are skipped (BCPack or unmapped shards). | `<work>\assets\ttpf\<pool>\` + `manifest.json` |
| Audio | `audio.rpf` holds raw (not zlib) **AWC** banks: Opus (codec 0x0B) and XMA2 streams. Exported as whole `.awc` files plus `audio_manifest.json` (stream ids, codec, channels, rate, duration). **Done: 7,162 banks, 27 GiB, 1,603,231 streams, about 1,324 hours** (codec 0x0B Opus 1,601,167; 5: 1,818; 2: 227; 0 PCM: 19). Decoding to WAV is a later step. | `<work>\assets\audio\awc\` |

### rpf9dump (bulk export)

`tools\rpf9dump` is a small CLI over the Core library from https://github.com/breadbbbb/rpf9-explorer (a friend's tool; source at `third_party\rpf9-explorer`, audited: no keys, no network, no custom build targets). It decrypts nothing. It carves validated resources, as `rpf9_carve.py` does, but with working model and texture decoders.

```
dotnet build tools\rpf9dump\rpf9dump.csproj -c Release -o <work>\tools\rpf9dump
rpf9dump models|textures <archive.rpf> <work>\assets
rpf9dump ttpf <pool.ttpf> <work>\assets\ttpf
rpf9dump audio <audio.rpf> <work>\assets
bash scripts/dump_assets.sh      # everything, log in <work>\assets\dump.log
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
- Runtime shims: all of `xmem.dll` is shimmed (`XMemAlloc`, `XMemFree`, `XMemVirtualQuery`,
  `XMemGetWorkingSetStatistics`). The only imports the bundled DLLs still can't bind are two `d3d12_xs`
  root-signature calls.
- PSO metadata is a new variant (`PSIN`, `LSCH` schema, `LENM`, `LSNS`, `LGID`, `LRSC`, `CHKS`, little-endian).
  CodeWalker's parser reads the container but not the new schema; dump it once the exe gives the field names.

## Shader translation (RDNA → SPIR-V)

`tools\rdna2spirv.py` (needs `spirv-as`/`spirv-val` on PATH: Vulkan SDK, or `spirv-tools` in WSL) translates parsed RDNA
code to SPIR-V assembly, assembles it with `spirv-as`, and validates it with `spirv-val` (Vulkan 1.2). Version 0
runs one lane per invocation, keeps VGPRs and SGPRs as uint arrays with exact bit patterns, treats EXEC as
all-ones, and covers branch-free vertex and pixel shaders with no memory loads: common float and integer ALU ops
and exports to `gl_Position`, params and MRTs. Unsupported instructions are reported by name, never guessed.
`VS_Passthrough` and `VS_BlurFilter` translate and validate. Coverage across the corpus and the blocking
instructions are in `<work>\shaders\spirv_run.log`. The next steps are structured branches (EXEC masks) and
the descriptor/binding model for `s_load`, `s_buffer_load`, `buffer_*` and `image_*`.

Also handled now: 64-bit scalar moves, EXEC reads (all-ones), whole-quad mode on EXEC, `m0`, SCC, vector and scalar
compares, `v_mul_hi_u32`, `v_cvt_pkrtz_f16_f32` (rounds to nearest instead of toward zero, so it can be off by one
f16 ulp), and pixel interpolation (`v_interp_p1/p2/mov` become fragment inputs at their attribute's location). The
float MODE register (`s_setreg`) is skipped, so Vulkan's default denormal and rounding behaviour applies.

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
code-embedded data via `s_getpc_b64`, global memory, exact packed-f16 modifiers): **13,971 of 26,503 shaders
translated, all passing `spirv-val`** (Vulkan 1.2), with SDWA selects honoured. That's **97.4% of all vertex and pixel shaders** (13,971 of
14,337). The remaining 366 VS/PS: 321 PS with unexplained system inputs (rejected on purpose), 10 that use `src_pops_exiting_wave_id`,
10 whose embedded data can't be re-anchored, 8 `v_fma_f16` with op_sel, 4 `image_sample_o`, and a few rare
operands. GPU re-test on the final SPIR-V: `VS_Passthrough` exact, texture test within RTZ f16 rounding, 400/400
arithmetic-only PS finite and deterministic. 319 pixel shaders are deliberately rejected because they read unexplained system-input registers (raw
barycentrics and so on). GPU re-test against the final SPIR-V: `VS_Passthrough` exact, texture test within RTZ f16
rounding (1.3e-5), and **400/400** arithmetic-only PS finite and identical across two runs. Mesh (9,282), compute (2,689) and geometry (195) shaders aren't
attempted yet. Of the remaining ~580 VS/PS: 288 use 16-bit shifts (`v_lshrrev_b16`, and whether the destination's
high half is kept or cleared needs confirming), 155 contain words LLVM can't decode, and 62 write `vcc_hi`. Output:
`<work>\shaders\spirv\` (`.spvasm` + `.spv`), and the log is `<work>\shaders\spirv_run.log`.

How the generated SPIR-V expects the runtime to work:

| Piece | Contract |
|---|---|
| Execution model | One invocation = one lane. A wave = a subgroup, so cross-lane ops (`ds_swizzle`, `v_permlane*`, `v_readlane`/`v_writelane`, `v_mbcnt`) become subgroup shuffles. **Requires subgroup size == wave size** (wave32 maps directly onto NVIDIA GPUs; wave64 shaders need a 64-wide subgroup or emulation). |
| EXEC | A per-invocation `active` flag. VALU writes and exports are masked by it, scalar ops aren't, and `v_cmpx`, saveexec and EXEC writes update it. A pixel lane still inactive at the end is discarded. |
| Control flow | Any CFG becomes a dispatch loop (`pc` variable + `switch`), which is valid structured SPIR-V. |
| User data | **Push constants, 144 bytes:** dwords 0–31 = initial s0–s31 (the user SGPRs the console driver loads from the root signature: descriptor-table pointers, constants and so on), dwords 32–33 = the 64-bit **heap base**, dwords 34–35 = the **code base**: the Xbox VA where the runtime uploaded this shader's whole FSSM blob. The runtime has to reproduce the console driver's root-signature-to-user-SGPR mapping; that comes from the exe or driver behaviour. |
| Xbox GPU addresses | Shaders build Xbox GPU virtual addresses, and some force the high dword to 0 (descriptor tables in the low 4 GB). Every memory access goes to **heap base + Xbox VA**, so the runtime maps the console's GPU address window into one host buffer-device-address allocation. |
| Code-embedded data | 155 VS/PS had "undecodable words". They're **constant tables inside the code range** (float values like 0x3EF7…, plus zeros), read through `s_getpc_b64` + `s_add_u32`/`s_addc_u32` + `s_load` (147 of 155). The translator re-anchors each instruction to its true FSSM offset by matching encodings, translates only code reachable from entry (a branch into data is rejected), and turns `s_getpc_b64` into code base + next-instruction offset. **The runtime uploads each shader's FSSM blob into the heap and passes that VA as the code base.** 132 of 155 now translate. |
| Global memory | `global_load/store_dword*`: address = (64-bit SGPR base + zero-extended 32-bit VGPR offset, or a 64-bit VGPR address) + signed 12-bit offset, translated through the heap base. Inactive lanes don't access memory, and loads merge their previous values with OpPhi. |
| Literal samplers | Some shaders build their S# from immediates (for example `0x92, 0xC00000, 0x14500000, 0`). The runtime registers those in the descriptor hash table too, by scanning the translated shaders' constant S# values, not only S#s the game binds. |
| Scalar/buffer memory | `s_load*`, `s_buffer_load*`, `buffer_load*` and `buffer_store*` use buffer device addresses (`PhysicalStorageBuffer64`). The base comes from the 64-bit SGPR pointer or the V# (dword0 \| dword1[15:0] << 32), and the stride is V# dword1[29:16]. **The runtime must put the game's GPU heap in buffer-device-addressable memory at the same addresses the game writes into descriptors** (or remap them). `num_records` bounds checks aren't modelled. |
| Typed buffer loads | The format is V# word3[18:12] (GFX10_FORMAT, from Mesa's `gfx10-rsrc.json`, saved as `tools\gfx10_formats.json`). Supported: f32 ×1–4, u32 ×1/2/4, f16 ×1/2/4, unorm8 ×1/2/4, snorm8 ×4, uint8 ×4, unorm16 ×1/2/4, snorm16 ×2/4, 10-10-10-2 and 2-10-10-10 unorm. Others give 0. DST_SEL swizzle is applied. |
| Images and samplers | Bindless. Set 0: binding 0 = 2D sampled images, 1 = samplers, 2 = descriptor hash table (storage buffer), 3 = 2D-array images (**cube textures are bound here as 6-layer 2D-array views**: the shader already selected the face with `v_cube*`), 4 = r32ui storage images, 5 = 1D, 6 = 3D. |
| Descriptor hash table | `uint mask; entries[mask+1] of 10 uints: key[0..7], index, pad`. Key = the raw T# (8 dwords), or the S# (4 dwords) followed by four 0xFFFFFFFF. Hash = FNV-1a over the 8 key dwords, probed linearly for 8 slots. A miss gives index 0, so **the runtime keeps a fallback texture and sampler at index 0**. As the game binds descriptors, the runtime inserts each T#/S# with the bindless index of the Vulkan image view or sampler it created for it. |
| Vertex fetch | `s_swappc_b64` at the start of a VS calls a driver-generated fetch program that isn't in the blob. It's replaced by Vulkan vertex inputs: each `ISG1` input-signature element (DXBC container) becomes an input at **location = its D3D input register**, typed by its component type, and is written to **v(4 + 4·register + component)**. That rule was checked over all 788 fetch-calling shaders: in 783 every read-before-write VGPR is predicted by it (715 exact). The other 5 are rejected. **The runtime must bind vertex buffers by the game's D3D12 input layout, location = input register.** |
| PS system inputs | v0–v1 = barycentrics (consumed by `v_interp`, which becomes attribute inputs). **v2.. = the `SV_Position` components the shader reads, packed in order**. Verified on 2,531 of 3,000 PS: the VGPRs they read before writing equal exactly the position components their input signature reads. Filled from `FragCoord`: xy/z directly, and **w directly too**, because AMD's position-w register holds 1/w (Mesa lowers `load_frag_coord_w_rcp` straight to `frag_pos[3]`, `ac_nir_lower_intrinsics_to_args.c`), the same as Vulkan's `FragCoord.w`. **Front face** follows the used position components when the signature declares `SV_IsFrontFace`, as ±1.0 (Mesa: `load_front_face` = `front_face > 0`). Only enabled inputs are packed, in hardware order (Mesa `radv_shader_args.c`: barycentric pairs, POS_X/Y/Z/W, FRONT_FACE, ANCILLARY, SAMPLE_COVERAGE, POS_FIXED_PT). That explains the earlier "unverified" shaders: 102 read pos.xy + front face, and 76 read pos.x/y/w + front face. A PS that reads an unexplained register *inside* the system-input range (for example a raw barycentric) is **rejected** (about 21 of 3,000). Reads past the last hardware input are values undefined on some path, so zero stands in for them. All register arrays start at zero, so translated shaders are deterministic. |
| Inputs and outputs | VS: `v0` = `VertexIndex` (**unverified**: if the console runs vertex shaders as NGG, the input VGPR layout differs). `exp pos0` → `Position`, `exp paramN` → location N. PS: `v_interp_*` attribute N → input location N. `exp mrtN` → location N, `exp mrtz` x → `FragDepth`. |
| 16-bit results | From Mesa ACO (`aco_ir.cpp`: `instr_is_16bit`, `can_use_opsel`) for GFX10: **f16 VOP1/VOP2 ops** (`v_add/sub/mul/max/min_f16`, `v_fmac_f16`, `v_cvt_f16_f32`, f16 unary) and **opsel-capable VOP3 ops** (`v_fma_f16`, `_e64` forms of the 16-bit integer ops) are **partial writes that keep the high half**. The **VOP2 (`_e32`) 16-bit integer ops** (`v_lshrrev_b16`…) define the whole register, so the high half is modelled as zero. An op_sel destination bit writes the high half. The translator used to clear the high half on f16 ops; that's fixed. |
| SDWA | `_sdwa` forms: each source is narrowed to `src0_sel`/`src1_sel` (BYTE_n/WORD_n, zero-extended, or sign-extended under `sext()`) **before** neg/abs and the op. A VGPR result goes to `dst_sel`, with `dst_unused` setting the other bits: PAD = 0, SEXT = sign copies above the field, PRESERVE = old bits kept. VOPC has no `dst_sel`, and GFX10 `v_cmpx` has no written destination. **Until 2026-11-20 the selects were silently ignored.** That affected 3,826 of the 13,926 translated shaders: they passed `spirv-val` but computed wrong values. GPU-checked with `runtime/test/shaders/ps_sdwa.asm` and `ps_ops.asm` (also `v_pk_fma_f16`, `v_min3_u32`, `v_max3_i32`, `v_bfe_i32`): byte/word source selects, `sext()`, `UNUSED_PRESERVE` into WORD_1 and `UNUSED_SEXT` at BYTE_2 all match on every pixel. |
| Known approximations | `v_cvt_pkrtz_f16_f32` is now **exact round-toward-zero** (finite overflow gives ±65504, not inf). The GPU smoke test caught the old round-to-nearest version producing inf in `PS_ImposterShadow`. The other f16 packs (`v_cvt_f16_f32`, d16 results, mix ops) round to nearest, which matches the default mode. The MODE register (`s_setreg`) is ignored, so denormal and rounding behaviour is Vulkan's default. Whole-quad mode on non-EXEC masks is the identity. `v_mul_legacy_f32` is a plain multiply, while `v_mad_legacy_f32` keeps the 0×anything = 0 rule. |
| Mesh/compute input registers (measured, not yet usable) | Read-before-write analysis over all compute and mesh shaders. **Compute:** v0 (880), v0–v1 (614), v0–v2 (50), consistent with AMD's local thread ID x/y/z in v0/v1/v2. Scalar inputs centre on a pointer at s[4:5] and arithmetic registers s6/s7 (likely workgroup IDs). **Mesh:** every one of 2,969 reads exactly **v5**, plus s3 (always arithmetic), a pointer pair s[16:17], and s19–s23. Cross-check against FSSM header 0x13C as the group width: 82 match, 151 don't, 1,447 have no simple multiply. So **0x13C isn't the group width, and the workgroup-ID registers are unconfirmed**. The real thread-group sizes are in each container's `PSV0` part (runtime info v2, NumThreadsX/Y/Z: 32×1×1 for 1,253, 8×8×1 for 416, …). Against those, the first SGPR multiply matches NumThreadsX in only 82 shaders, with 150 others, so the first multiply isn't a reliable workgroup-ID marker. The matching register varies by shader (s6/s10/s7/s9…), which fits AMD's layout of workgroup IDs right after a **per-shader user-SGPR count**. That count comes from the console's register setup (exe/driver), so compute stays untranslated until it's known. Mesh and compute stay untranslated until the exe's dispatch setup, or a correctly decoded register block in the FSSM metadata, confirms them. |
| Not yet translated | Mesh (9,282), geometry and compute stages; `s_swappc_b64` (call to the vertex-fetch program); `tbuffer_*`; a few rare ops. Mesh, geometry and compute, plus the vertex-fetch call, need the per-stage hardware register setup (user-SGPR layout, system-value VGPRs). Get that from the FSSM metadata or the exe before translating them. |

**Main blocker (before descriptor support was added): descriptor loads.** On a 3,000-shader sample, `s_load_dwordx4`/`x8` stop over half of them. Shaders
load texture, buffer and sampler descriptors through user-data SGPR pointers, and how the game's root signatures
map onto those SGPRs is set up by game code. Read that mapping from the exe (root signature creation plus the
FSSM user-data metadata) before building the binding layer; guessing it would give silent rendering bugs. The
other blockers are mesh shaders (9,282, which need a mesh-stage entry model), the vertex-fetch subroutine call
(`s_swappc_b64`), geometry and compute stages, and compressed exports.

## Day one, in order

1. `scripts\first_look.ps1 -Exe ...` checks the exe is plain, lists the imports to shim, and does a dry run.
2. Find the RPF9 table-of-contents key and decryption in the exe, then add a table reader to `rpf9_carve.py`. Carved
   blobs then get real names and types, and the remaining unread data shows what it is.
3. Find the texture and mesh resource structures in the exe (their loaders give exact field layouts), which turns
   carved blobs into named textures and meshes.
4. Shader translation: RDNA2 disassembly to SPIR-V, starting with the most-used shaders.

## Disassembling shaders

The AMDGPU disassembler needs LLVM 21+ (`llvm-mc`; e.g. the `llvm-21` package in WSL, `/usr/lib/llvm-21/bin/llvm-mc`). The
official Windows LLVM build has no AMDGPU target. Run it inside WSL:

```
LLVM_MC=/usr/lib/llvm-21/bin/llvm-mc python3 -I <repo>/tools/shader_disasm.py <work>/shaders/fssm
```

Code starts at the u32 at FSSM offset 0xC0, plus 0x28, and ends at the last `s_endpgm`. Use `--mcpu gfx1013`. On
300 random shaders it decoded 284 cleanly (134 invalid words in total), against 201 clean and 6,560 invalid for
gfx1030 and 283 clean for gfx1010–1012.

## Baseline already recorded

- `reports/manifest.json`: SHA-256 of all 108 files in `Documents\GTA6` (208.5 GiB). Run
  `python -I tools\manifest.py verify <game>` to check a copy against it.
- `reports/rpf_probe.json`: all 22 archives are **RPF9**. Five have tables that look packed or encrypted (`levels`,
  `movies`, `core_bvh4`, `navmeshes`, `data`).
- `reports/pe/`: the 16 bundled DLLs are plain x86-64 PE with no encrypted sections. Through the loader they need
  only `xmem.dll` (`XMemAlloc` and `XMemFree` are shimmed; `XMemVirtualQuery` and `XMemGetWorkingSetStatistics` aren't
  yet) and two `d3d12_xs.dll` root-signature calls. Every other import resolves on desktop Windows.
