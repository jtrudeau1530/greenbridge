# Import routing: shim, bundled, host or unimplemented

How `gta6rt` binds an import, per DLL. The code paths are `Loader::resolve` for the import table
(`runtime/src/pe_loader.cpp`), `Loader::resolve_forwarder` for a forwarded export, and the `LoadLibrary*` /
`GetProcAddress` shims for run-time loads (`runtime/shims/library_loader.cpp`). Measured against the 16 shipped
modules in [runtime_imports.md](runtime_imports.md): every import binds, 0 unimplemented.

## The bind order

The lookup key is the DLL name normalised the way `LoadLibrary` resolves a file name (`dll_key`,
`pe_loader.cpp`): lower-case, a trailing `.` dropped, and `.dll` added when the name has no extension —
so a descriptor naming `GameInput` or `gameinput.dll.` keys the same as `gameinput.dll`, matching the
host `LoadLibraryExA` it can fall through to.

`resolve(dll, name)` tries, in order:

1. **Shim** — `rt::shims::find("dll!name")`, the clean-room tables registered by `SHIM_TABLE` in
   `runtime/shims/*.cpp`. The registry is searched first even for host DLLs, which is how `kernel32!CreateFileW`
   or `ExitProcess` get redirected without losing the rest of `kernel32`.
2. **Bundled** — `<game-dir>/<dll>` mapped by the same loader, unless the name is in `prefer_host_copy`
   (the `vcruntime140`/`msvcp140`/`concrt140`/`vccorlib140`/`vcomp140` family: the host CRT is always used, so a
   mapped image's `RtlPcToFileHeader` can be redirected, `pe_loader.cpp`). A bundled export that is itself a
   forwarder string is followed by `resolve_forwarder` (8 deep, then a loop is declared).
3. **Host** — `LoadLibraryExA(dll, LOAD_LIBRARY_SEARCH_SYSTEM32)` + `GetProcAddress`. System32 only, so the game
   folder is never in the search. API-set names (`api-ms-*`, `ext-ms-*`) resolve here through the host's API-set
   schema; the measured run binds every api-ms import the package asks for.
4. **Unimplemented** — `shims::make_unimplemented_thunk`: a 32-byte stub per import that calls `rt_unimplemented`
   with `"dll!name"` and the argument registers; `LEONIDA_UNIMPLEMENTED` = log (default) / fail / stop picks the
   answer (`runtime/shims/registry.cpp`, policy documented in `runtime/README.md` and
   [runtime_imports.md](runtime_imports.md#when-an-unimplemented-import-is-called)).

Step 3 is skipped when `shims::owns_dll(dll)` is true. `xgameruntime.dll` and `gameinput.dll` are real PC names
(Gaming Services installs `xgameruntime.dll` in System32); without the skip, an import the registry doesn't serve
would bind to the live service runtime while a run-time `GetProcAddress` for the same name returns NULL, and the
process would be online-capable. `resolve_forwarder` applies the same order, so a forwarder string naming a
shimmed module resolves to the shim.

## `owns_dll`: whole-module shims

`owns_dll` (registry.cpp) is false for `api-ms-*`/`ext-ms-*` prefixes and for the eight host DLLs that the tables
partially override — `kernel32`, `kernelbase`, `msvcrt`, `ucrtbase`, `ucrtbased`, `vulkan-1`, `ntdll`, `user32` —
and otherwise true exactly when the registry holds at least one `dll!` entry. Owned today: `d3d12_xs.dll`,
`xgameruntime.dll`, `gameinput.dll`, `xmem.dll`.

A title's `LoadLibrary*`/`GetModuleHandle*` for an owned module returns a pseudo-handle
(`library_loader.cpp`); `GetProcAddress` on it resolves through the registry only — an unserved name is NULL with
`ERROR_PROC_NOT_FOUND`, never the host. `FreeLibrary` on a pseudo-handle is a no-op. Every other module goes to
the real `LoadLibrary*`/`GetProcAddress`.

## Per-DLL outcome

| DLL | Registry entries | Static import | Run-time load |
|---|---|---|---|
| `d3d12_xs.dll` | 4: `D3D12XboxCreateDevice`, `D3D12Serialize{,Versioned}RootSignature`, `D3D12CreateRootSignatureDeserializer` | those → shim; any other → unimplemented (no host namesake) | owned → pseudo-handle |
| `xgameruntime.dll` | ~40: `*ApiImpl*` entry points, `XGameRuntime*`, `XSystem*`, `XPackage*`, `XTaskQueue*`, `XAsync*`, `GameInputCreate` | registered → shim; unregistered → unimplemented | owned → pseudo-handle (the path `xgameruntime.lib` thunks use) |
| `gameinput.dll` | `GameInputCreate` | registered → shim; unregistered → unimplemented | owned → pseudo-handle |
| `xmem.dll` | `XMemAlloc`, `XMemFree`, `XMemVirtualQuery`, `XMemGetWorkingSetStatistics` | registered → shim; unregistered → unimplemented | owned → pseudo-handle |
| `d3d12.dll`, `dxgi.dll` | none | host (the real PC DLLs; the console surface rides on them through the `d3d12_xs` device proxy and the DXGI COM wrappers) | real host DLL |
| `xaudio2_9.dll`, `xaudio2_9redist.dll` | none | host if installed, else unimplemented | real host DLL |
| `dstorage.dll`, `dstorage_xs.dll` | none | `dstorage.dll` host if the DirectStorage redist is installed, else unimplemented; `dstorage_xs.dll` unimplemented | real host DLL / fails |
| `xg_xs.dll`, `acphal.dll`, `chad*`/`xapu*`/`xcrdapi*` (legacy Xbox names) | none | unimplemented | fails (host can't load them) |
| `kernel32`, `kernelbase` | file (`CreateFile*`, `FindFirstFile*`, `GetFileAttributes*`, `DeleteFile*`), loader (`LoadLibrary*`, `GetModuleHandle*`, `GetProcAddress`, `FreeLibrary`), process (`ExitProcess`, `GetCommandLine*`) | registered → shim; everything else → host | real host DLL (not owned) |
| `msvcrt`, `ucrtbase`, `ucrtbased` | exit/`abort`/argv/command-line overrides | registered → shim; everything else → host | real host DLL |
| `ntdll`, `user32`, `vulkan-1` | none | host | real host DLL |
| `api-ms-*`, `ext-ms-*` | only the sets a GDK title links: `core-file-l1-1-0/l1-2-0/l2-1-0`, `core-libraryloader-l1-1-0/l1-1-1/l1-2-0/l1-2-1/l1-2-2`, `core-processthreads-l1-1-0`, `core-processenvironment-l1-1-0`, `crt-runtime-l1-1-0` | registered names → shim; all other names and sets → host API-set schema (this is what covers every UCRT `api-ms-win-crt-*` variant without a table entry) | real host resolution |
| `vcruntime140*`, `msvcp140*`, `concrt140`, `vccorlib140`, `vcomp140` | none | host, even when a bundled copy exists (`prefer_host_copy`) | real host DLL |
| the package's own DLLs (`Bink2Scarlett.dll`, `XCurl.dll`, …) | none | bundled module, mapped by the loader | host `LoadLibrary` — see the caveat in [runtime_imports.md](runtime_imports.md#run-time-loaded-modules) |
| anything else | none | unimplemented | host or failure |

## Edge cases

- **`d3d12.dll`/`dxgi.dll` can never hit the unimplemented thunk** for a name the PC DLL exports: no registry
  entries exist for them, and they are real System32 DLLs. A name they don't export ends unimplemented, which is
  the wanted answer.
- **Ordinals**: registry keys are `dll!#N` in `Entry.name`, but no shipped table registers an ordinal; an
  ordinal-only import from a shimmed or console-only DLL lands unimplemented. Host DLLs bind ordinals through
  `GetProcAddress` normally.
- **No console DLL matches the host list.** The `owns_dll` host list holds only names that also exist on the PC
  and have partial table overrides. A console-only name (`d3d12_xs`, `xmem`, `xg_xs`, …) can't be owned by
  accident and can't be excluded from the tables by accident.
- **Owned ≠ bundled-blocked.** A bundled copy of an owned name would still map through the bundled step before
  the unimplemented fallback; nothing in the package ships one.
