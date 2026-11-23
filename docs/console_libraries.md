# Console libraries: what the console build can import, and how the loader answers

Status: catalogue, sourced. This page lists the console-only or console-flavoured DLLs an Xbox
Series title's imports can name, what is measured about each one's export surface, and what
`gta6rt` does with each today. It is the module-level counterpart to
[runtime_imports.md](runtime_imports.md) (what the 16 shipped DLLs import) and to
[d3d12_vulkan.md §7](d3d12_vulkan.md#7-prior-art-and-measured-xbox-abi) (the measured ABI inside
the main one, `d3d12_xs.dll`).

Basis tags:

- **This repo** — verifiable in this tree: a `SHIM_TABLE` in `runtime/shims/*.cpp`, the
  `classify()` list in `runtime/shims/registry.cpp`, the binding order in
  `runtime/src/pe_loader.cpp`, or a measurement recorded in another page here.
- **Sibling-layer measurement** — a fact folded in from `d3d12_vulkan.md` §7, measured elsewhere
  by running the console DLLs or a retail title behind a compatibility layer. Paraphrased, not
  re-measured here; no external repository is named on this page.

Scope limit: the game executable is not in the package, so its own import table is unmeasured.
The list below is the surface a console title *can* name — from sibling-layer inventories of the
console DLLs and from one retail title's run — not a claim about what this title does.

## 1. How an import binds today

`Loader::resolve` in `runtime/src/pe_loader.cpp` binds each import in a fixed order (**this
repo**):

1. **Shim registry** — `rt::shims::find(dll, name)`. Ordinal imports look up `"#N"`, so a shim
   serves an ordinal import only if its table carries the `"#N"` spelling.
2. **Bundled module** — a DLL of the same name in `--game-dir`, mapped by the same loader (the
   `module` count of the dry run; the CRT families are deliberately excluded and always take the
   host copy).
3. **Host** — `LoadLibraryEx` on `System32` plus `GetProcAddress`, by name or ordinal.
4. **Unimplemented thunk** — a generated stub whose behaviour `LEONIDA_UNIMPLEMENTED` sets
   (`log`/`fail`/`stop`; see
   [runtime_imports.md](runtime_imports.md#when-an-unimplemented-import-is-called)). Under `fail`,
   `classify()` in `runtime/shims/registry.cpp` decides by DLL and name which imports return
   `E_NOTIMPL`.

A module the registry serves at all and that is not a host DLL is **owned** (`owns_dll`): a
title's `LoadLibrary`/`GetModuleHandle` on it gets a pseudo-handle and its `GetProcAddress`
resolves through the registry by name or `"#N"`, with unknown names returning
`ERROR_PROC_NOT_FOUND`
([runtime_imports.md](runtime_imports.md#run-time-loaded-modules)). Owned today:
`d3d12_xs.dll`, `xgameruntime.dll`, `xmem.dll` and `gameinput.dll` — the modules with a
`SHIM_TABLE` entry.

The registry also holds tables under `kernel32.dll`, `kernelbase.dll`, the `api-ms-*` sets,
`msvcrt.dll` and `ucrtbase*.dll` (file system, library loader, process entry/exit). Those are
host DLLs a few functions are replaced on, not owned console modules, so they are not in the
list below.

## 2. The modules

| Module | What it is | Leonida today | Status |
|---|---|---|---|
| `d3d12_xs.dll` | Console D3D12 runtime front end | 4 names registered | partial shim |
| `xgameruntime.dll` | GDK runtime (task queues, users, store, packaging) | 39 names registered; 2 impl classes | partial shim |
| `xmem.dll` | Console tracked allocator / virtual-memory API | all 4 measured imports implemented | shimmed |
| `gameinput.dll` | GameInput v0, controller input | `GameInputCreate` registered (SDL3 build) | shimmed |
| `xaudio2_9.dll` | Console XAudio2 2.9 | no table; host binds it by ordinal | classified only |
| `dstorage_xs.dll` | Console DirectStorage | no table; `DStorageGetFactory` classified | classified only |
| `xg_xs.dll` | Texture layout helpers | nothing | unhandled |
| `AcpHal.dll` | Audio co-processor HAL | nothing | unhandled |
| `chad`, `xapu`, `xcrdapi`, `*legacy` | DSP, APU and card-API modules; names only | nothing | unhandled |

### `d3d12_xs.dll`

The console D3D12 runtime's front end: a thin DLL that loads the user-mode driver, which owns the
device and command objects (**sibling-layer measurement**;
[d3d12_xs_surface.md](d3d12_xs_surface.md) records the reported 116-export surface — 23 functions
plus 93 GUID data exports — and which of the 83 standard GUIDs match the PC SDK). Beyond the
standard `D3D12*` names its exports include `D3D12XboxCreateDevice` — the device entry point, call
shape `(adapter, params, riid, out)` with a 24-word params block that is only logged —
`D3D12XboxCreateSyntheticDevice` and `D3D12XboxSetProcessDebugFlags` (**sibling-layer
measurement**).

Leonida registers four names (**this repo**, `runtime/shims/d3d12_xs.cpp`):
`D3D12SerializeVersionedRootSignature`, `D3D12SerializeRootSignature` and
`D3D12CreateRootSignatureDeserializer`, each forwarded to the host `d3d12.dll` after the Xbox
high flag byte is cleared on a copy; and `D3D12XboxCreateDevice`, which returns an Xbox-ABI proxy
over the PC `ID3D12Device` — or over the Vulkan-backed device with `LEONIDA_D3D12=vulkan` — with
the measured slot overrides (device slot 62 as compute-PSO creation, slot 76 as texture creation,
descriptor-heap start handles returned in RAX). The other 19 function exports, including
`D3D12XboxCreateSyntheticDevice` and `D3D12XboxSetProcessDebugFlags`, are not registered.
`classify()` marks the `D3D12Create*`/`D3D12Serialize*` families, the two debug entries and both
`D3D12XboxCreate*Device` names as HRESULT; `D3D12XboxSetProcessDebugFlags` is deliberately left
out because its return was not measured (**this repo**, `registry.cpp`).

The COM surface behind the device — queue, command list, fence, `PresentX` — is vtable work, not
an import problem; it is tracked in [d3d12_vulkan.md](d3d12_vulkan.md),
[d3d12_command_objects.md](d3d12_command_objects.md) and
[d3d12_xs_console_abi.md](d3d12_xs_console_abi.md). The presentation core
(`runtime/shims/xbox_present.h`) is wired to the measured `PresentX` queue slots (30, or 32 on the
newer layout) in `d3d12_xs_device.cpp`; the plane and parameter struct layouts are still provisional.

### `xgameruntime.dll`

The GDK runtime. Not in the package; GDK titles reach it either by import or by `LoadLibraryW` +
`GetProcAddress` of `InitializeApiImpl`, `InitializeApiImplEx`, `InitializeApiImplEx2`,
`QueryApiImpl` and `UninitializeApiImpl`, which is how the package's `libHttpClient.GDK.dll` and
`XCurl.dll` do it (**this repo**,
[runtime_imports.md](runtime_imports.md#run-time-loaded-modules)).

Leonida registers 38 names plus `GameInputCreate` (**this repo**,
`runtime/shims/xgameruntime.cpp`): the five loader entry points,
`XGameRuntimeInitialize`/`Uninitialize`, five `XSystem*` queries, three `XPackage*` queries, the
15-function `XTaskQueue*` set and the 8-function `XAsync*` set, both on the vendored
`third_party/xtaskqueue`. `QueryApiImpl` serves `XThreadingImpl` and `XErrorImpl` and answers
every other class with the documented "not provided" error so the first boot lists what is
missing. `classify()` marks the `XAsync*`/`XTaskQueue*`/`XUser*`/`XStore*`/`XGameSave*`/
`XPackage*`/`XGameUi*` families as HRESULT except the members the public pages document as
`void`/`bool`/`int` (**this repo**, `registry.cpp`).

### `xmem.dll`

The console's tracked allocator. The only console module whose whole measured import surface is
implemented (**this repo**, `runtime/shims/xmem.cpp`): `XMemAlloc`/`XMemFree` on the 64-byte
aligned CRT heap, `XMemVirtualQuery(process, address, out64)` filling the
`MEMORY_BASIC_INFORMATION` prefix and zeroing the rest of the measured 64-byte output, and
`XMemGetWorkingSetStatistics(kind, out)` writing the process working set at +8 of the measured
output. All four names are exactly the set the package's DLLs import
([runtime_imports.md](runtime_imports.md)).

A sibling layer additionally measured a title's physical-page calls on this module: 64 KiB pages,
with runs of consecutive pages mapping contiguously inside one unified region and scattered or
explicitly addressed mappings refused (**sibling-layer measurement**). Nothing in the package
imports them and no such names are registered; see section 3.

### `gameinput.dll`

GameInput v0. The console build's ordinal 1 is `GameInputCreate` (**sibling-layer
measurement**).

Leonida registers `GameInputCreate` by name under both `gameinput.dll` and `xgameruntime.dll`:
the PC API lives in the former, the GDK links it through the latter, and which the executable
imports is unknown until it exists (**this repo**, `runtime/shims/gameinput.cpp`; the shim's API
surface is [gameinput.md](gameinput.md)). The registration is a name only — no `"#1"` entry — so
an ordinal-1 import would not hit the table (section 3). The shim is optional and built only with
SDL3; without it the import falls through to the host DLL or the unimplemented thunk as before.

### `xaudio2_9.dll`

Console XAudio2 2.9. Ordinal 1 is `XAudio2Create` (**sibling-layer measurement**). Voices accept
the console formats — XMA/XMA2 and an Opus format tag `0x704F` — and a `CO_E_NOTINITIALIZED`
from the system XAudio2 is retried after `CoInitializeEx` on the calling thread; forwarding to
the system XAudio2 2.9 works with those two quirks (**sibling-layer measurement**).

Leonida has no table for it (**this repo**). The package's `Bink2Scarlett.dll` imports
`XAudio2_9.dll` ordinal 1 and the dry run binds it to the host copy by ordinal — 57 host, 0
unimplemented ([movies.md](movies.md), [runtime_imports.md](runtime_imports.md)) — so creation by
ordinal resolves today, but the console voice formats and the retry quirk are not reproduced: a
voice created with XMA/XMA2 or `0x704F` would reach the host unremapped. `classify()` marks
`xaudio2_9.dll`/`xaudio2_9redist.dll` `XAudio2Create` as HRESULT — by name; the ordinal lookup
classifies as Other (**this repo**, `registry.cpp`).

### `dstorage_xs.dll`

Console DirectStorage; named in the sibling export inventory with no export list measured
(**sibling-layer measurement**). Leonida has no table; `DStorageGetFactory` is classified HRESULT
under both `dstorage.dll` and `dstorage_xs.dll` (**this repo**, `registry.cpp`).

### `xg_xs.dll`

Texture layout helpers. The sibling export inventory names `XGCreateTexture2DComputer` and
`XGComputeOptimalSwizzleMode` (**sibling-layer measurement**); prior-art text folded into
[d3d12_vulkan.md](d3d12_vulkan.md#35-root-signature-to-user-sgprs) also names
`XGComputeTexture2DLayout` as the module's untiling reference. Leonida has no table and no
classification (`xg_xs.dll!XGCreateTexture2DComputer` is pinned as Other in
`unimplemented_policy_test`; **this repo**). Nothing in the package imports it.

### `AcpHal.dll`

The audio co-processor HAL. The sibling export inventory names `ApuAlloc`, `ApuFree`,
`ApuMapVirtualAddress`, `ApuIsVirtualAddressValid`, `AcpHalCreate`,
`AcpHalAllocateShapeContexts` and `AcpHalReleaseShapeContexts` (**sibling-layer measurement**).
Leonida has nothing for it (`acphal.dll!AcpHalCreate` is pinned as Other in
`unimplemented_policy_test`; **this repo**).

### `chad`, `xapu`, `xcrdapi`, `*legacy` variants

A retail title booted to its menu and lobby through console-library stand-ins, and its
console-only library set added `chad` (`XDsp*` exports), `xapu` (`XApu*` exports), `xcrdapi` and
`*legacy` DLL variants to the modules above (**sibling-layer measurement**). These are names
only: no export list, call shape or argument is measured, and Leonida has nothing for them.

### Not console modules

`d3d12.dll`, `dxgi.dll` and the `api-ms-*` sets are ordinary host bindings, not console modules.
The package's own middleware (`Bink2Scarlett.dll`, `XboxBvhBuilder.dll`, the FidelityFX pair,
`libHttpClient.GDK.dll`, `XCurl.dll`, the CRT set) binds as `module`, with the CRT families
deliberately host-bound; their imports are catalogued in
[runtime_imports.md](runtime_imports.md), not here. `xgs12_10.dll`, the user-mode driver a real
`d3d12_xs.dll` loads (recorded in [d3d12_vulkan.md](d3d12_vulkan.md) from prior-art text), is not
an import target a title names; Leonida replaces the whole stack rather than providing it.

## 3. Gaps

What a real title is known to call that Leonida does not yet cover (**sibling-layer
measurement** unless noted):

- **The whole `xg_xs`, `AcpHal`, `dstorage_xs`, `chad`, `xapu`, `xcrdapi` and `*legacy`
  surface.** No table, no classification. A retail run reached its menu and lobby with stand-ins
  for them; for this title, whether any is imported at all is unknown until the executable exists.
- **The `xaudio2_9` quirks.** Creation by ordinal already binds to the host (**this repo**), but
  the XMA/XMA2 and Opus `0x704F` voice formats and the `CoInitializeEx` retry would need a shim
  that wraps or remaps — the host sees console format tags it does not know.
- **The unregistered `d3d12_xs` functions.** 4 of the 23 function exports are registered; the
  rest — `D3D12XboxCreateSyntheticDevice`, `D3D12XboxSetProcessDebugFlags`, the `DXGIX*`
  present/frame-notification functions, `D3D12UploadCustomMicrocodeX`-type console entries —
  would hit the unimplemented thunk. Two are at least classified (`D3D12XboxCreate*Device` →
  HRESULT under `fail`).
- **`xmem` physical-page calls.** A title's physical-page API uses 64 KiB pages and contiguous
  mappings inside one unified region; the shim covers only the four allocator/query names the
  package measured.
- **Ordinal-only registrations.** The shim tables serve names; an ordinal import is looked up as
  `"#N"`. `gameinput.dll` and `xaudio2_9.dll` ordinal 1 are the measured console entry points,
  and neither has a `"#1"` entry — a static ordinal import therefore skips the shim and binds to
  the host DLL where one exists (as `xaudio2_9.dll!#1` does today; **this repo**), and a
  `GetProcAddress` for ordinal 1 on an owned module returns `ERROR_PROC_NOT_FOUND`.
- **Run-time loading of bundled middleware.** A title that `LoadLibrary`s a package DLL such as
  `Bink2Scarlett.dll` gets the host loader, which maps the Xbox build without the registry's
  bindings for its own console imports — recorded as an open item in
  [runtime_imports.md](runtime_imports.md#run-time-loaded-modules) (**this repo**).
