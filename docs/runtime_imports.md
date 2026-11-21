# Runtime: measured imports of the shipped modules

`gta6rt --dry-run` maps an x86-64 image, applies base relocations, binds every import through the shim
registry or the host, and prints a report. It does **not** call `DllMain`, TLS callbacks or the entry point, so
this page measures mapping and binding only; it does not claim the modules work.

The game executable is still missing from the package available here. Every other PE that the package ships is
present, and all of them are Xbox Series builds. Running the dry-run over those images is the closest thing to a
loader test that does not need the executable.

## Reproduce

Build the runtime, then point it at the package:

```powershell
powershell -File scripts/build_runtime.ps1

$game = "D:\path\to\your\package"
runtime\build\gta6rt.exe --dry-run "$game\XboxBvhBuilder.dll" --game-dir $game
```

Each run prints one line per mapped module:

```
XboxBvhBuilder.dll  base=0000000180000000 size=909312 entry=+0x227cc  shim=2 host=55 module=0 unimplemented=0
    shim          d3d12_xs.dll!D3D12SerializeVersionedRootSignature
    shim          xmem.dll!XMemVirtualQuery
1 module(s) mapped, 0 unimplemented import(s)
```

`shim` counts imports served by the clean-room tables, `host` counts imports served by the desktop DLLs,
`module` counts imports served by another bundled image, and `unimplemented` counts imports with no binding.

## Measured result

All 16 DLLs in the package map at their preferred base `0x180000000` with no relocation, and every import binds:

| Image | Size of image | Entry RVA | Shim | Host | Module | Unimplemented |
|---|---:|---:|---:|---:|---:|---:|
| `Bink2Scarlett.dll` | 278,528 | `0x2ac1c` | 2 | 58 | 0 | 0 |
| `XboxBvhBuilder.dll` | 909,312 | `0x227cc` | 2 | 55 | 0 | 0 |
| `amd_fidelityfx_upscaler_xbox.dll` | 802,816 | `0x26780` | 4 | 82 | 0 | 0 |
| `amd_fidelityfx_loader_xbox.dll` | 32,768 | `0x18d0` | 0 | 22 | 0 | 0 |
| `libHttpClient.GDK.dll` | 999,424 | `0x11f0` | 0 | 198 | 0 | 0 |
| `XCurl.dll` | 188,416 | `0x17b0` | 0 | 140 | 0 | 0 |
| `concrt140.dll` | 303,104 | `0x29160` | 0 | 103 | 0 | 0 |
| `msvcp140.dll` | 577,536 | `0x525a0` | 0 | 187 | 0 | 0 |
| `msvcp140_1.dll` | 36,864 | `0x15c0` | 0 | 33 | 0 | 0 |
| `msvcp140_2.dll` | 192,512 | `0x1efa0` | 0 | 96 | 0 | 0 |
| `msvcp140_atomic_wait.dll` | 81,920 | `0x3c80` | 0 | 64 | 0 | 0 |
| `msvcp140_codecvt_ids.dll` | 28,672 | `0x1340` | 0 | 24 | 0 | 0 |
| `vccorlib140.dll` | 344,064 | `0x27e50` | 0 | 114 | 0 | 0 |
| `vcomp140.dll` | 86,016 | `0x6580` | 0 | 78 | 0 | 0 |
| `vcruntime140.dll` | 110,592 | `0xf890` | 0 | 41 | 0 | 0 |
| `vcruntime140_1.dll` | 49,152 | `0x3e90` | 0 | 34 | 0 | 0 |

The six unique Xbox-specific imports (eight consumer/import pairs) are exactly the set recorded in
[`d3d12_vulkan.md` §7](d3d12_vulkan.md#7-prior-art-and-measured-xbox-abi):

| Consumer | Import | Binding |
|---|---|---|
| `Bink2Scarlett.dll` | `xmem.dll!XMemAlloc` | shim |
| `Bink2Scarlett.dll` | `xmem.dll!XMemFree` | shim |
| `XboxBvhBuilder.dll` | `xmem.dll!XMemVirtualQuery` | shim |
| `XboxBvhBuilder.dll` | `d3d12_xs.dll!D3D12SerializeVersionedRootSignature` | shim |
| `amd_fidelityfx_upscaler_xbox.dll` | `xmem.dll!XMemAlloc` | shim |
| `amd_fidelityfx_upscaler_xbox.dll` | `xmem.dll!XMemFree` | shim |
| `amd_fidelityfx_upscaler_xbox.dll` | `xmem.dll!XMemGetWorkingSetStatistics` | shim |
| `amd_fidelityfx_upscaler_xbox.dll` | `d3d12_xs.dll!D3D12CreateRootSignatureDeserializer` | shim |

The video path (`Bink2Scarlett.dll`), the BVH builder and the FidelityFX upscaler therefore all go through the
two shim DLLs that already exist. `libHttpClient.GDK.dll` and `XCurl.dll` import only desktop APIs.

## What this proves and what it does not

- It proves the loader parses real Xbox Series PE files, maps their sections, applies no unnecessary relocation,
  walks their import tables, and classifies every thunk.
- The loader also registers each image's unwind table. `pe_loader_test` checks, for every DLL present in
  `LEONIDA_GAME_DIR`, that the whole exception directory is registered, that `RtlLookupFunctionEntry` returns
  each entry for its first byte, and that the entry point is covered (directly, or through the `jmp rel32` thunks
  an incrementally linked entry point goes through: a thunk touches no stack, so it has no entry of its own).
  Measured on all 16 DLLs: every table passes the loader's checks and every entry resolves to itself.

  | Image | Unwind entries | Entry point |
  |---|---:|---|
  | `Bink2Scarlett.dll` | 527 | covered |
  | `XboxBvhBuilder.dll` | 449 | covered |
  | `amd_fidelityfx_upscaler_xbox.dll` | 477 | covered |
  | `amd_fidelityfx_loader_xbox.dll` | 46 | covered |
  | `libHttpClient.GDK.dll` | 2,452 | `jmp rel32` thunk to a covered function |
  | `XCurl.dll` | 446 | covered |
  | `concrt140.dll` | 1,121 | covered |
  | `msvcp140.dll` | 1,228 | covered |
  | `msvcp140_1.dll` | 45 | covered |
  | `msvcp140_2.dll` | 469 | covered |
  | `msvcp140_atomic_wait.dll` | 86 | covered |
  | `msvcp140_codecvt_ids.dll` | 32 | covered |
  | `vccorlib140.dll` | 697 | covered |
  | `vcomp140.dll` | 195 | covered |
  | `vcruntime140.dll` | 236 | covered |
  | `vcruntime140_1.dll` | 79 | covered |
- It does **not** prove any imported function behaves correctly when called. That needs the executable, or a
  harness that calls a known entry point.
- The executable's own imports are unknown. `MicrosoftGame.Config` names `game_scarlett_master.exe`; the package
  here does not contain it, so its import table cannot be measured yet.
- The CRT images are present in the package, but `pe_loader.cpp` prefers the host copies for the `vcruntime`/
  `msvcp`/`concrt`/`vccorlib`/`vcomp` families. The table above maps the packaged images only to test the
  parser; a real run binds the host copies.
- No game data is stored in this repository. The counts are output of a local run against the user's package.
