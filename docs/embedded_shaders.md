# Shader containers embedded in the package's DLLs

**Status: measured** on the 1.0.2.1 package. This note covers the shader programs that ship outside
`shaders.rpf`: the DXBC containers inside the package's DLLs, what each one carries, and what the measurement
changes for the ray-tracing and upscaling paths. No container, shader or payload is reproduced here; the tables
hold counts, offsets, chunk names and module identifiers.

## 1. Why they matter

`make_game.py`'s `shaders` stage carves `shaders.rpf` only (`make_game.py`:655). Two DLLs in the same package
carry compute programs of their own:

* `XboxBvhBuilder.dll`, Microsoft's XDXR acceleration-structure builder. Its constructor creates compute PSOs
  through device slot 62, and the builder is on the critical path for ray tracing.
* `amd_fidelityfx_upscaler_xbox.dll`, the FidelityFX 4.1.1 upscaler. Its programs are the `ACCC`/`FSPX` format
  covered in [fspx.md](fspx.md); this note only counts them.

`tools/pe_shader_containers.py` scans a binary for DXBC containers, validates the chunk table, reads the `ILDN`
module name and reports what each container carries. `--extract` writes the containers and an `index.json` the
translator reads; extracted shaders are game data and must stay in the user's work folder.

```sh
python -I tools/pe_shader_containers.py <dll> --json <work>/containers.json --extract <work>/embedded/dxbc
```

## 2. Inventory

The 16 DLLs and the package's root files were scanned. All the containers are in two DLLs:

| image | containers | bytes | min / median / max | chunks |
|---|---:|---:|---:|---|
| `XboxBvhBuilder.dll` | 52 | 674,404 | 4,072 / 7,454 / 74,508 | `SFI0 ISG1 OSG1 PSV0 RTS0 STAT ILDN HASH DXIL` |
| `amd_fidelityfx_upscaler_xbox.dll` | 29 | 289,656 | 1,764 / 5,176 / 42,616 | `SFI0 ISG1 OSG1 PSV0 RTS0 HASH ACCC` |
| other 14 DLLs | 0 | 0 | — | — |

The 29 `ACCC` containers hold `FSPX` payloads ([fspx.md](fspx.md)); the 52 `DXIL` containers are the subject of
this note. The FSR DLL has one rejected `DXBC` hit: the bytes are an immediate in its own magic check, and the
validator rejects it (the scan reports both counts).

**Every one of the 52 containers carries two representations of its program:**

* an LLVM bitcode blob in the `DXIL` chunk (`BC C0 DE`, 1,820 to 70,276 bytes), and
* an `XDAT` directory with tag `0x0d` whose payload starts with `XBCH`: the console program. The same chunk also
  holds tag `0x07` (a second `DXIL` program header) and tag `0x12` (a 16-byte hash). The `STAT` chunk holds a
  second, different bitcode blob, not the same one.

The bitcode is what a PC D3D12 compiler consumes; the runtime path is the `XBCH` record. The constructor passes
the **whole container** to the compute-PSO call: at `0x180021330` it loads `DAT_180025700` with size `0x1134`
(4,404 bytes, the size of container 0, `CopyAndCompactBVH`) and calls `FUN_18000c470`, whose vtable call at
`+0x1f0` (slot 62) receives that pointer, its size and two zero words. `FUN_18001d400` adds two more PSOs,
`XDXR PostBuildInfoQuery` (0x14a4 bytes) and `XDXR GetBVHCompactedSize`. `bvh_builder_test` measures **53** PSO
creations during construction over the **52** containers ([d3d12_vulkan.md](d3d12_vulkan.md) A.2), so one
container is created twice; which one is **Open**.

## 3. The 52 modules

The `ILDN` PDB path gives one module per container, all under
`X:\americas\src\dev\rage\3rdparty\Microsoft\XboxBvhBuilder\Build\Release\Output\XboxBvhBuilder\` (Microsoft's
component integrated into RAGE, not Rockstar shader code). Grouped by phase:

| phase | modules |
|---|---|
| sort (11) | `BitonicPreSortCS`, `BitonicPreSort64CS`, `BitonicInnerSortCS`, `BitonicInnerSort64CS`, `BitonicOuterSortCS`, `BitonicOuterSort64CS`, `RadixCount`, `RadixAlloc`, `RadixReset`, `CalculateMortonCodes`, `CalculateMortonCodes64` |
| prefix (7) | `PrefixSum64`, `PrefixSum128`, `PrefixSum256`, `PrefixSum512`, `PrefixSum1024`, `PrefixSum2048`, `PrefixSum4096` |
| kdop (3) | `BuildKDOP`, `ResetKDOP`, `BuildApexPointMap` |
| build (7) | `ConstructBVH4`, `BuildBVHSplits`, `BuildBVHSplits64`, `SplitInstances`, `FindTreelets`, `RearrangeBVHs`, `RearrangeTriangles` |
| refit (2) | `RefitBVH4`, `CopyForBVHUpdate` |
| load (6) | `LoadTriangles`, `ReloadTriangles`, `LoadInstances`, `LoadProceduralGeometry`, `ReloadProcedurals`, `ResetToIndexBuffer` |
| path tracing (6) | `PT_ConstructBVH4`, `PT_BeginPass`, `PT_LoadInstances`, `PT_WriteInstances`, `PT_WritePartitions`, `PT_WriteOfflinePartition` |
| rehydration (3) | `Box16Rehydration`, `Box32Rehydration`, `LeafRehydration` |
| headers/util (7) | `DecompressHeader`, `InitHeaders`, `InitPartitionedHeaders`, `GetBVHCompactedSize`, `CopyCounters`, `ResetCounterValues`, `CopyAndCompactBVH` |

## 4. They are console FSSM, and the existing translator already takes them

The `XBCH` record has the same shape as the game's own shaders: `XBCH` + stored/unpacked sizes + a zlib stream
that expands to `FSSM` ([fssm_crosscheck.md](fssm_crosscheck.md)). On all 52 containers:

| step | result |
|---|---|
| `tools/fssm_unpack.py` | **52/52** decompress to a valid `FSSM` blob (716 B to 101,320 B, median 2,200 B) |
| `tools/shader_disasm.py --mcpu gfx1013` | **52/52** disassemble; 42 with no invalid word, 10 with invalid words in the code |
| `tools/rdna2spirv.py` | **42** translate and pass `spirv-val` |

The 10 untranslated programs stop at four rules:

| module(s) | blocker |
|---|---|
| `BuildKDOP`, `LoadInstances`, `ReloadProcedurals`, `LoadProceduralGeometry`, `LoadTriangles` | `global_atomic_fmax` |
| `PT_LoadInstances` | `global_atomic_fmin` |
| `Box16Rehydration`, `Box32Rehydration`, `ReloadTriangles` | `buffer_load_ushort` |
| `LeafRehydration` | `tbuffer_load_format_x GFX10_FORMAT_8_UINT` |

Reproduction, after `--extract` above:

```sh
python -I tools/fssm_unpack.py <work>/embedded/dxbc
python -I tools/shader_disasm.py <work>/embedded/dxbc/fssm --out <work>/embedded/dxbc/disasm --mcpu gfx1013
python -I tools/rdna2spirv.py <work>/embedded/dxbc/disasm --out <work>/embedded/spirv
```

## 5. What this changes

* **The RT builder does not need a DXIL path for the console code.** Its programs are RDNA2 `FSSM`, and 41 of
  52 already pass `spirv-val` with the translator the project uses for the game's 26,503 shaders. The bitcode in
  the same container is a second representation (the one the PC-compiler experiment measured); it is not what
  the console runtime executes.
* **The four missing rules are small and have real fixtures.** `global_atomic_fmax`/`fmin` cover 6 programs,
  `buffer_load_ushort` 3 and `tbuffer_load_format_x` with format 8 one. They are ordinary translator rules, not
  a new compiler backend.
* **The rehydration programs are the ones closest to the shipped data.** `Box16Rehydration`,
  `Box32Rehydration` and `LeafRehydration` are exactly the programs that would turn the dehydrated BVH packs
  into runtime trees ([bvh.md](bvh.md) §3.1). Two of the four rules stand between them and translation.
* **What is still open.** None of the 52 has been executed on a GPU; translation and `spirv-val` do not prove the
  output. Which of them the game actually creates is a call-site question (RT-5). The module-to-pipeline order
  above comes from `ILDN`, not from a runtime trace. The `STAT` bitcode and the `0x07 NOBC` record are not
  explained.

## 6. Tool and tests

`tools/pe_shader_containers.py` is standard library only and reads the file, never runs it. It accepts a
container only when the chunk table is sorted, 4-byte aligned, entirely inside the declared total and its last
chunk ends exactly at that total; that is what rejects the FSR DLL's `DXBC` immediate. The synthetic tests in
`tests/test_pe_shader_containers.py` build containers by hand (no game data) and cover the accepted layout, a
bare magic, a truncated chunk, an `ACCC`/`FSPX` container, the `ILDN` name and the extraction index.
