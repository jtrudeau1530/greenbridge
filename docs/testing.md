# Running the GPU tests

The GPU tests are small hand-written RDNA2 programs in `runtime/test/shaders`. They are translated to SPIR-V and run
on your GPU, and the results are compared with what the RDNA2 hardware gives. They need no game files, and every
result helps: post it in [#166](https://github.com/ItsNotRexed/Leonida/issues/166), one comment per GPU and driver.
If your GPU is already covered there, testing an open pull request from the review queue in
[#156](https://github.com/ItsNotRexed/Leonida/issues/156) helps more.

This page collects the setup steps and the problems testers have run into so far.

## What you need

- **Python 3.11+.** `scripts/run_gpu_tests.py` needs at least 3.10.
- **`llvm-mc` with the AMDGPU target** (LLVM 21+). The official LLVM Windows release has no AMDGPU target. On
  Windows, use one of:
  - `scripts\build_llvm_mc.ps1`, which builds a native `llvm-mc.exe` (needs git and the Visual Studio C++ tools);
  - conda-forge's `llvm-tools` package, whose Windows build includes the AMDGPU target;
  - `llvm-21` in WSL, with `LLVM_MC` set to the whole command:
    `LLVM_MC="wsl -d Ubuntu -e /usr/lib/llvm-21/bin/llvm-mc"`.

  On Linux and macOS, the distribution's or Homebrew's `llvm-mc` on `PATH` is enough.
- **SPIRV-Tools** (`spirv-as`, `spirv-val`): part of the Vulkan SDK, or a distribution package.
- **A C++20 compiler, CMake and the Vulkan headers** to build the test programs:
  - Windows: Visual Studio 2022 with "Desktop development with C++" and the Vulkan SDK.
  - Linux: CMake 3.24+, GCC or Clang, and the Vulkan headers and loader. The headers should be recent enough
    for `VK_KHR_maintenance8`; older ones still build, but skip what needs it.
  - macOS: `brew install llvm molten-vk vulkan-headers vulkan-loader spirv-tools vulkan-tools`. The test programs
    load MoltenVK directly; set `LEONIDA_VULKAN_LOADER` to use the Vulkan loader instead.

## Build the test programs

Windows, from a "Developer PowerShell for VS 2022" or a plain shell:

```powershell
scripts\build_runtime.ps1            # CMake + Ninja into runtime\build
```

If you configure with CMake's Visual Studio generator instead, the programs end up in `runtime\build\Release` (or
`RelWithDebInfo`); pass that folder to `--build` below.

Linux and macOS:

```sh
cmake -S runtime -B runtime/build/linux
cmake --build runtime/build/linux --parallel
```

On macOS, add `-DVulkan_INCLUDE_DIR=/opt/homebrew/include` to the first command.

## Run them

The whole suite, assembled, translated and run in one go:

```sh
python -I scripts/run_gpu_tests.py --build runtime/build/linux     # Windows: the default runtime\build
```

`--filter <text>` runs only matching tests, and `--work <dir>` keeps the intermediate files.

The two-step form, which `CONTRIBUTING.md` describes:

```sh
python -I runtime/test/shaders/build.py <out dir>
python -I runtime/test/shaders/check.py <out dir> <folder with the test programs>
```

## GPUs with 32-wide subgroups (NVIDIA, Intel, Apple)

The game's shaders are mostly wave64. A GPU that can't run 64-wide compute subgroups skips the wave64 tests
(`cs_wavemask`, `cs_wave64_ballot`, `cs_wm_*`, `cs_w64x_*`, `ms_w64x_ballot`); that's expected, not a failure.

To test the wave64 emulation that makes them exact on these GPUs, set `LEONIDA_W64_EMULATE=1`, **for both
`build.py` and `check.py`**: `build.py` translates with emulation only when it's set. If you set it only for
`check.py`, the shaders are checked against the emulated expectations without being emulated, and every wave64 test
fails.

## Choosing the GPU

With more than one GPU, pick one through the Vulkan loader, for example
`VK_LOADER_DRIVERS_SELECT=*amd*` or `VK_LOADER_DRIVERS_SELECT=nv-vk*` (or `VK_DRIVER_FILES=<path to the ICD json>`).
`vulkaninfo --summary` shows which devices the loader then sees.

## Known problems

| What you see | Cause | What to do |
|---|---|---|
| Windows refuses to start freshly built test programs ("An Application Control policy has blocked this file"), and the runner stops with a traceback | Smart App Control on Windows 11 | Turn Smart App Control off for the run |
| Validation errors without a VUID, or messages from a layer you didn't ask for | An implicit Vulkan layer from other software (Rockstar Social Club, ReShade, MangoHud, an overlay) | Set `VK_LOADER_LAYERS_DISABLE=~implicit~` |
| Every test crashes inside `libnvidia-eglcore.so` | NVIDIA's 535 server driver (Ubuntu 24.04's default) | Use driver 615 or newer |
| `build.py`: 0 translated, reason `.text` | An old `build.py` with LLVM 19 | Update to current `main` |
| `vk_harness` doesn't compile: `VK_KHR_maintenance8` names unknown | Vulkan headers older than the SDK the code was written against | Update to current `main` (the code now builds without it), or use current Khronos Vulkan-Headers; conda-forge's package is too old |
| `ps_bary` and `ps_uniform_branch` fail by about 0.9 | The GPU runs pixel shaders wave64 without fragment barycentrics (GCN3 and RDNA1 on RADV), or an old AMD Windows driver (22.40) | Current `main` skips them on such devices; on Windows, update the driver |
| `gs_tri` and `ms_tri` fail on a Mac | MoltenVK has no geometry stage and no `VK_EXT_mesh_shader` | Current `main` reports them as skipped |
| `run_gpu_tests.py` fails on a `dict \| None` annotation | Python older than 3.10 | Use Python 3.11+ |

## Posting your result

Use the template at the top of [#166](https://github.com/ItsNotRexed/Leonida/issues/166): GPU, driver, subgroup
range, OS, the commit you built, the tool versions and the counts each script printed. Post counts and pass/fail
lines only, never shaders or logs from the game. A failure that isn't in the table above gets its own issue, linked
from your entry.
