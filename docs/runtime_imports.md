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
| `Bink2Scarlett.dll` | 278,528 | `0x2ac1c` | 3 | 57 | 0 | 0 |
| `XboxBvhBuilder.dll` | 909,312 | `0x227cc` | 3 | 54 | 0 | 0 |
| `amd_fidelityfx_upscaler_xbox.dll` | 802,816 | `0x26780` | 5 | 81 | 0 | 0 |
| `amd_fidelityfx_loader_xbox.dll` | 32,768 | `0x18d0` | 4 | 18 | 0 | 0 |
| `libHttpClient.GDK.dll` | 999,424 | `0x11f0` | 10 | 188 | 0 | 0 |
| `XCurl.dll` | 188,416 | `0x17b0` | 8 | 132 | 0 | 0 |
| `concrt140.dll` | 303,104 | `0x29160` | 1 | 102 | 0 | 0 |
| `msvcp140.dll` | 577,536 | `0x525a0` | 9 | 178 | 0 | 0 |
| `msvcp140_1.dll` | 36,864 | `0x15c0` | 1 | 32 | 0 | 0 |
| `msvcp140_2.dll` | 192,512 | `0x1efa0` | 1 | 95 | 0 | 0 |
| `msvcp140_atomic_wait.dll` | 81,920 | `0x3c80` | 3 | 61 | 0 | 0 |
| `msvcp140_codecvt_ids.dll` | 28,672 | `0x1340` | 1 | 23 | 0 | 0 |
| `vccorlib140.dll` | 344,064 | `0x27e50` | 6 | 108 | 0 | 0 |
| `vcomp140.dll` | 86,016 | `0x6580` | 7 | 71 | 0 | 0 |
| `vcruntime140.dll` | 110,592 | `0xf890` | 3 | 38 | 0 | 0 |
| `vcruntime140_1.dll` | 49,152 | `0x3e90` | 1 | 33 | 0 | 0 |

Since the library-loader shims (`runtime/shims/library_loader.cpp`), a module's `LoadLibrary*`, `GetModuleHandle*`, `GetProcAddress` and `FreeLibrary` imports count as shim, through `kernel32`, `kernelbase` or the `api-ms-win-core-libraryloader-*` set it names. That is why every image above has a few more shim and fewer host imports than before; the Xbox-specific imports below are unchanged.

The process shim (`runtime/shims/process.cpp`) serves `ExitProcess` and `GetCommandLineA/W` the same way: through `kernel32`, `kernelbase`, and `api-ms-win-core-processthreads-l1-1-0` and `api-ms-win-core-processenvironment-l1-1-0`, the sets the API-set umbrella libraries (`OneCore_apiset.lib`, `mincore.lib`) bind them to. `vcomp140.dll` imports `ExitProcess` from the first of these. The UCRT's `_get_narrow_winmain_command_line` and `_get_wide_winmain_command_line` are served too, so a `WinMain` executable gets the game's command line as `lpCmdLine`. The table above is `gta6rt --dry-run` on each DLL at the commit that added this paragraph.

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

## Console paths and the write overlay

A title's file calls name the console's drives, so `GAME:\data\x.sps` is a file inside the installed game folder.
[ivall/xcl](https://github.com/ivall/xcl) (GPL-2.0-only; facts only, no code) documents for Xbox Series GDK titles that
`GAME:` maps to the game folder and that writes go to a writable overlay instead of the package, which stays read-only.
`gta6rt` therefore shims `CreateFileA/W`, `FindFirstFileA/W`/`FindNextFileA/W`/`FindClose`, `GetFileAttributesA/W`,
`GetFileAttributesExA/W` and `DeleteFileA/W` (`runtime/shims/file_system.cpp`), under `kernel32.dll`, `kernelbase.dll`
and the `api-ms-win-core-file-*` sets a GDK title links against.

- A `GAME:` path (any case, `\` or `/`) maps under `--game-dir`. A path without the prefix is a host path and is
  passed to the host unchanged.
- An open that can write, and `DeleteFile`, use the overlay: `--overlay DIR`, default `<game-dir>\overlay`. A
  write-open of a package file copies it into the overlay first, so `OPEN_EXISTING` keeps its bytes; a delete of a
  package file leaves an empty marker under `<overlay>\.gta6rt-deleted`, which reads and listings treat as gone. The
  package is never written.
- Reads prefer the overlay copy and then the package. `FindFirstFile` merges the two folders, the overlay winning on a
  name, and hides deleted names and the marker folder.
- A path that climbs above `GAME:` with `..`, or names another drive, fails with `ERROR_INVALID_NAME`, so a title
  cannot reach outside the package and the overlay.

`runtime/test/file_system_test.cpp` (ctest `file_system`) covers this over folders it makes in `%TEMP%`. Not mapped
yet: `CreateDirectory*`/`RemoveDirectory*`, `MoveFile*`, `CopyFile*` and `SetFileAttributes*`; a title that makes a
folder through them gets the host's error for a name like `GAME:\...`.

## When an unimplemented import is called

The table above has no unimplemented imports, but the game executable will. A call to one used to return 0
silently, which for an `HRESULT` function is `S_OK` with its out parameters never written; the caller then carries on
with stale pointers and fails somewhere unrelated. `LEONIDA_UNIMPLEMENTED` chooses what the thunk does instead
(details in `runtime/README.md`):

- `log` (default): print the name once, return 0. Unchanged.
- `fail`: return `E_NOTIMPL` for imports known to return `HRESULT`, 0 otherwise.
- `stop`: print the import, the caller as `module+RVA` and rcx/rdx/r8/r9, then break (debugger) or exit with code 3.

The `--dry-run` report ends with `unimplemented by return policy: hresult=N other=N`, so the first run against the
executable shows how many of its missing imports `fail` would turn into errors.

The `HRESULT` list is by DLL and name: `D3D12Create*` and `D3D12Serialize*` (`d3d12.dll`, `d3d12_xs.dll`),
`D3D12GetDebugInterface`, `D3D12EnableExperimentalFeatures`, `CreateDXGIFactory*`, `DXGIGetDebugInterface1`,
`DXGIDeclareAdapterRemovalSupport` (`dxgi.dll`), and in `xgameruntime.dll` `XGameRuntimeInitialize`, `XGameEventWrite`
and the families `XAsync*`, `XTaskQueue*`, `XUser*`, `XStore*`, `XGameSave*`, `XPackage*`, `XGameUi*` except the 33
functions their public reference pages show returning `void`, `bool` or `int` (for example `XTaskQueueDispatch`,
`XUserCloseHandle`, `XStoreIsLicenseValid`). The mixed `XSystem*`, `XGameStreaming*`, `XThread*` and `XDisplay*`
families are not listed. The sources are the Microsoft Learn pages for the D3D12/DXGI functions and the GDK reference
(`gaming/gdk/docs/reference/system/<family>/functions/<name>`); `D3D12XboxCreateDevice` is matched by name from the
measurement in `d3d12_vulkan.md`, because its page could not be found.

## Run-time loaded modules

Not every console module is reached through an import table. `libHttpClient.GDK.dll` and `XCurl.dll` both use the
GDK runtime, but `dumpbin /imports` shows **no import of `xgameruntime.dll`** in either. Both carry the wide string
`xgameruntime.dll` and the ASCII names `InitializeApiImpl`, `InitializeApiImplEx`, `InitializeApiImplEx2`,
`QueryApiImpl` and `UninitializeApiImpl`, next to a statically linked `XGameRuntimeInitialize`. So `xgameruntime.lib`'s
thunks `LoadLibraryW` the runtime and `GetProcAddress` those five names at run time. xodus-gaming/xgameruntime-docs
(GPL-3.0) documents the same mechanism for PC GDK titles: `InitializeApiImplEx2` is what
`XGameRuntimeInitialize[WithOptions]` calls, and `QueryApiImpl` hands out the runtime's COM classes, whose methods
mostly correspond one to one to the `xgameruntime.lib` functions. The executable is very likely linked the same way.

Bound to the host, those calls would load the PC's Gaming Services runtime, or fail, and never reach the offline
shims. `runtime/shims/library_loader.cpp` therefore answers `LoadLibrary*` and `GetModuleHandle*` for a module the
shim registry owns (`rt::shims::owns_dll`: `xgameruntime.dll`, `d3d12_xs.dll`, `xmem.dll`, `gameinput.dll`) with a
pseudo-handle, and `GetProcAddress` on it resolves through the registry. A name the registry doesn't serve is `NULL`
with `ERROR_PROC_NOT_FOUND` and is logged once. Every other module goes to the host unchanged.

`runtime/shims/xgameruntime.cpp` serves the five entry points. The initializers succeed offline and log the GDK
version and mode once. `QueryApiImpl` logs each class and interface id it is asked for and returns `0x80070032`, the
documented error for a class the runtime doesn't provide. The class method layouts are not known yet, so the first
boot with the executable lists the interfaces to implement next. `xgameruntime_test` drives this path through the
api-set binding `XCurl.dll` uses.

Which classes the package's GDK modules reference, from a byte search for the class and interface GUIDs that
xgameruntime-docs lists (16-byte in-memory form, every `.dll` of the package; counts only, nothing else read):

| Module | Runtime classes referenced (highest interface id present) |
|---|---|
| `XCurl.dll` | `XThreadingImpl`, `XErrorImpl`, `XNetworkingImpl` (`IXNetworkingImpl2`) |
| `libHttpClient.GDK.dll` | 24 of the 24 documented classes, each GUID once, the pattern of a linked id table rather than of use. Highest interfaces: `IXSystemImpl5`, `IXUserImpl5` (not 6), `IXGameImpl3`, `IXStoreImpl6`, `IXPackageImpl3` (not 4), `IXGameUiImpl4`, `IXGameSaveImpl3`, `IXGameStreamingImpl3`, `IXPersistentLocalStorageImpl3`, `IXUserDeviceImpl2`, `IXNetworkingImpl2` |
| every other `.dll` | none |

So `XCurl.dll` needs threading, error reporting and networking from the runtime, and the interface versions in
`libHttpClient.GDK.dll` date the GDK the package was built with: newer than the first `IXUserImpl5`, older than
`IXUserImpl6`. The executable's own set is still unmeasured. `QueryApiImpl`'s log gives it on the first boot.

What this does not cover: a title that `LoadLibrary`s one of the package's bundled DLLs (`Bink2Scarlett.dll`,
`XboxBvhBuilder.dll`, the FidelityFX DLLs) still gets the host loader, which loads the Xbox build without the
registry's bindings for its own console imports (`xmem.dll`, `d3d12_xs.dll`).

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
