# gta6rt

`gta6rt` is the native Windows runtime for the Xbox Series build. It is deliberately small: the game
executable is still missing from the package used here, so this directory contains the machinery that
can be built and tested without it — the PE loader, the measured import shims, and the GPU/Vulkan test
tools.

## Build

The normal path is the repository build script:

```powershell
powershell -File scripts/build_runtime.ps1
```

It uses CMake + Ninja from Visual Studio when available, otherwise `cl.exe`, and writes
`runtime\build\gta6rt.exe` plus `gta6rt_tls_host.dll` (the loader's TLS-index helper; it must stay next to
the executable). The direct CMake path works too:

```powershell
cmake -S runtime -B runtime/build
cmake --build runtime/build --config RelWithDebInfo
```

Requirements: Windows x64, Visual Studio 2022 with the C++ desktop workload. The GPU tools need the
Vulkan SDK; the runtime and CPU tests do not. The loader is x64-only and the PE mapper rejects any
other machine.

## Dry run

```powershell
gta6rt --dry-run <image.exe|image.dll> [--game-dir <package>]
```

The dry run maps the image and the bundled modules it imports, applies base relocations, binds every
import through the shim registry, a host DLL, or another bundled module, parses exports, installs static
TLS for the calling thread, applies section protections, and
prints a per-module report. It does **not** call `DllMain`, TLS callbacks, or the entry point, so no
code from the mapped images runs. TLS reserves its module index through the runtime's own
`gta6rt_tls_host.dll`, and host imports load system DLLs, as they did before.

`gta6rt --init <image>` runs the same mapping and then `Loader::initialise()`: each module's TLS
callbacks and each DLL's `DllMain(DLL_PROCESS_ATTACH)`, dependencies first. An EXE's entry point is not
run.

The report line is:

```
<name>  base=<addr> size=<image size> entry=+<rva>  shim=N host=N module=N unimplemented=N
```

Shim and unimplemented imports are listed by name. A non-zero `unimplemented` count is the useful
signal: it names the next shim that is missing. The measured result for the 16 DLLs shipped in the
package is recorded in the runtime-imports documentation.

## PE loader

`src/pe_loader.cpp` maps x86-64 images at their preferred base, or elsewhere (with relocations) when
that base is taken, copies headers and sections, applies `IMAGE_REL_BASED_DIR64` relocations, binds
imports by name or ordinal, parses the export table, sets section protections, and registers the
image's x64 unwind table (`.pdata`) with `RtlAddFunctionTable`, so exceptions raised in or passing
through a mapped image are dispatched and stack walks can step over its frames. The table is validated
first (entries in bounds, sorted and non-overlapping) and removed when the loader is destroyed. Modules
are deduped by lower-case file name, so an import graph that reaches the same DLL twice maps it once.

`load()` also installs static TLS for the calling thread. `__declspec(thread)` reads `TEB->ThreadLocalStoragePointer[_tls_index]`, which is a
different index space from the `TlsAlloc` slots, and only the OS loader can allocate a module index.
The runtime therefore ships a minimal `gta6rt_tls_host.dll`; the loader reuses the index the OS assigns
it, writes that index into the image's `AddressOfIndex`, and copies the image's TLS template into the
slot. This reserves one index, so **one manually mapped TLS module can be live at a time**; a second
one is rejected with an error instead of sharing the slot. Images without a TLS directory (or with an
all-zero one) load normally.

Current limits, by design:

- `load()` runs no image code. Static TLS is data, not code. The TLS index helper is the runtime's own
  `gta6rt_tls_host.dll`; host imports still load system DLLs as before.
- `--dry-run` never runs image code. `initialise()` runs TLS callbacks and a DLL's `DllMain`;
  `run_tls_callbacks()` is the opt-in seam for `DLL_THREAD_ATTACH`/`DETACH` and for callers that do not
  use `initialise()`. An EXE's entry point is the caller's to run.
- The loader does not deliver thread attach/detach events. Call `install_tls_for_thread()` before mapped
  code runs on a thread the runtime created and `uninstall_tls_for_thread()` before it exits; destroy the
  `Loader` on the thread that called `load()`.
- The TOC of the game's RPF archives is encrypted and is not read here. The loader works on the PE
  files and on any images the caller passes it.

## Shims

`shims/registry.cpp` holds the static tables. The current shims are:

- `xmem.dll` — `XMemAlloc`, `XMemFree`, `XMemGetWorkingSetStatistics`, and the measured three-argument
  `XMemVirtualQuery(process, address, out)`. Allocations are 64-byte aligned; the working-set and
  virtual-query outputs fill the observed 64-byte structures and return `FALSE`/`0` on a null or bad
  address.
- `d3d12_xs.dll` — `D3D12SerializeVersionedRootSignature` and
  `D3D12CreateRootSignatureDeserializer`, with the console root-signature high byte (`0x03000000`)
  cleared on a copy before forwarding to the PC `d3d12.dll`. `D3D12XboxCreateDevice` wraps a device in
  the measured-slot Xbox-ABI proxy: the host PC `ID3D12Device` by default, or the Vulkan-backed device
  (`shims/d3d12_vulkan_device.cpp`) when `LEONIDA_D3D12=vulkan` is set and the runtime was built with the
  Vulkan SDK and Python 3. The Vulkan device covers command submission, descriptor heaps, render-target and
  depth views, committed resources and copies (uploads, readbacks, BC textures); see `docs/d3d12_vulkan.md`
  section 3.2 for what is missing.

The two serializer functions and `D3D12XboxCreateDevice` are the only `d3d12_xs` imports used by the
shipped DLLs (measured; see `docs/design.md` and `docs/d3d12_vulkan.md`). The game executable's own
imports are unknown until it is available.

## Tests

```powershell
ctest --test-dir runtime/build --output-on-failure
```

On Windows the build configures the CPU tests that do not need a GPU or game files, including the
root-signature forwarding test (`root_sig_test`) and the clear-command test. The Vulkan tools are built
when the Vulkan SDK is found:

- `vk_harness` — hand-written RDNA2 shader programs on a real Vulkan device.
- `vk_texture_test`, `vk_compute_test`, `vk_store_test` — texture, LDS/compute, and buffer-store paths.
- `vk_command_replay_test`, `d3d12_com_test`, `d3d12_vulkan_device_test` — the D3D12 command path on
  Vulkan: replay, the COM command objects, and the Vulkan device behind the Xbox-ABI proxy. Set
  `LEONIDA_VK_VALIDATION=1` to run them under the Khronos validation layer.
- `vk_pipeline_check <spirv dir> [cache.bin] [--jobs N] [--start I] [--count N] [--max-bytes N]` -
  creates a shader module and, for compute shaders, a pipeline for every entry in the translator's
  `index.json`; reports `MODULE FAIL` / `PIPELINE FAIL` per shader and a summary. `--max-bytes`
  names and skips modules larger than the limit, which is how the corpus run works around the
  driver's compile time on multi-megabyte ray-tracing shaders. The measured partial pass is in
  [docs/pipeline_check.md](../docs/pipeline_check.md).

The PE-loader and Bink-decode tests that need the retail package are optional: when
`LEONIDA_GAME_DIR` (and, for Bink, `LEONIDA_MOVIE`) is unset they print `SKIP` and exit 0. They read
the files in place and never copy or commit game data.

## Current limits

`gta6rt --run <exe> [-- args]` runs an executable's entry point inside the runtime (`shims/process.cpp` gives it its
own command line and turns its `exit` into the end of the run); CTest's `run_tiny_game` does that with a stand-in game.
The game does not boot yet. The missing executable blocks the import-driven work: the real device
entry point, command-list/queue/fence extensions, `PresentX`, and most of the `xgameruntime` surface (an offline
shim covers `XGameRuntimeInitialize`, device/console queries, `XTaskQueue` and `XAsync`; see `xgameruntime_test`). The
D3D12-to-Vulkan design and the measured console ABI are in
[docs/d3d12_vulkan.md](../docs/d3d12_vulkan.md); the shader translator and its GPU checks are described
in [docs/design.md](../docs/design.md), and the differential-validation design is in the shader-validation
work.

No game data, keys, or extracted assets belong in this directory or in the repository.

### Typed UINT clear regression

`clear_color_test` checks numeric UINT conversion and unchanged float clears without a GPU.
`d3d12_uint_clear_test` compares 21 full/rectangle readbacks with Windows WARP across the three
currently supported typed UINT render-target formats. Fixtures are synthetic; no shaders or game data are needed.
The GPU test runs explicitly, while the CPU test is registered in CTest.

```powershell
ctest --test-dir runtime/build -C RelWithDebInfo -R '^clear_color$' --output-on-failure
$env:LEONIDA_VK_VALIDATION = '1'
$env:VK_VALIDATION_VALIDATE_SYNC = '1'
.\runtime\build\RelWithDebInfo\d3d12_uint_clear_test.exe
```

### RGBA8 typeless RTV clear regression (phase 1)

`typeless_rtv_test` checks typed-view recording and replay without a GPU. It is registered in CTest.
`d3d12_typeless_rtv_test` compares synthetic full and rectangular clears with Windows WARP through explicit
UNORM/SRGB/UINT RTVs on a one-mip, one-layer, single-sample 2D RGBA8 typeless resource. It checks descriptor
snapshots, alternating view formats, SRGB with UAV resource usage, and invalid view rejection.
Other typeless families, typed-resource casts, SINT/SNORM, mip/array expansion, MSAA and 3D views are not added.

```powershell
ctest --test-dir runtime/build -C RelWithDebInfo -R '^typeless_rtv$' --output-on-failure
$env:LEONIDA_VK_VALIDATION = '1'
$env:VK_VALIDATION_VALIDATE_SYNC = '1'
.\runtime\build\RelWithDebInfo\d3d12_typeless_rtv_test.exe
```
