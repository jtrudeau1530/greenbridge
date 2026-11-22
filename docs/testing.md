# Running the GPU tests

The GPU tests are small hand-written RDNA2 programs in `runtime/test/shaders`. They are translated to SPIR-V and run
on your GPU, and the results are compared with what the RDNA2 hardware gives. They need no game files, and every
result helps: post it in [#166](https://github.com/ItsNotRexed/Leonida/issues/166), one comment per GPU and driver.
If your GPU is already covered there, testing an open pull request from the review queue in the [coordination
issue][claims] helps more.

This page collects the setup steps and the problems testers have run into so far.

[claims]: https://github.com/ItsNotRexed/Leonida/issues?q=is%3Aissue%20is%3Aopen%20in%3Atitle%20%22Coordinating%20who%20works%20on%20what%22

## What you need

- **Python 3.11+.** `scripts/run_gpu_tests.py` needs at least 3.10.
- **`llvm-mc` with the AMDGPU target** (LLVM 21+). The official LLVM Windows release has no AMDGPU target. On
  Windows, use one of:
  - `scripts\build_llvm_mc.ps1`, which builds a native `llvm-mc.exe` (needs git and the Visual Studio C++ tools);
  - conda-forge's `llvm-tools` package, whose Windows build includes the AMDGPU target;
  - `llvm-21` in WSL, with `LLVM_MC` set to the whole command:
    `LLVM_MC="wsl -d Ubuntu -e /usr/lib/llvm-21/bin/llvm-mc"`.

  On Linux and macOS, the distribution's or Homebrew's `llvm-mc` on `PATH` is enough, if it is 21 or newer.

  An older `llvm-mc` is not enough for everything. Ubuntu 24.04's `llvm-18` (18.1.3) runs `run_gpu_tests.py`,
  `valu_fuzz.py --family mem` and `lds_fuzz.py`, but rejects `op_sel` on the 16-bit integer VOP3 forms that
  `valu_fuzz.py --family f16` generates (`v_ashrrev_i16_e64 ... op_sel:[0,0,0]`: "not a valid operand"), so no f16
  program assembles. 23.1.6 assembles them.

  On Windows, ROCm 7.2's bundled `llvm-mc.exe` reports LLVM 21.0.0git. It assembled the TEST-71 memory and
  LDS programs but rejected the 16-bit shift used by the f16 generator. To check an assembler directly in
  PowerShell, set `$env:LLVM_MC` to the path of its executable and run:

  ```powershell
  'v_lshlrev_b16_e64 v34, v27, v27 op_sel:[0,1,1]' | & $env:LLVM_MC -triple=amdgcn--amdpal -mcpu=gfx1013 -show-encoding
  ```

  If it reports `not a valid operand`, point `LLVM_MC` at a build that accepts this instruction. The Windows
  conda-forge `llvm-tools` 23.1.6 package assembled the same f16 programs.
- **SPIRV-Tools** (`spirv-as`, `spirv-val`): part of the Vulkan SDK, or a distribution package.
- **A C++20 compiler, CMake and the Vulkan headers** to build the test programs:
  - Windows: Visual Studio 2022 with "Desktop development with C++", the Windows SDK 10.0.26100 or newer, and the
    Vulkan SDK.
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

The test programs and `scripts/run_gpu_tests.py` should come from the same commit. Some checks tolerate an older
binary (`unpinned_subgroup_skip`), but mixing them can report false failures in the compute tests: 28 `cs_*`
failures on both an RTX 5070 Ti Laptop and a Radeon 610M with the `426023a` script and `48ad23f` binaries, where
either matching pair is clean. If you cannot build locally, the `runtime-windows` artifact of the Runtime workflow
belongs to the commit that built it; check that commit out as well.

### Without a compiler: a CI-built runtime

If you cannot build the test programs locally, the Runtime workflow publishes them as the `runtime-windows`
artifact of its own commit. It holds Windows executables, so use them on Windows or under Wine as in the
section below; GitHub expires the artifact after 7 days. Open the run for that commit under Actions ->
Runtime, download and unzip `runtime-windows`, and pass the unzipped folder to `--build`:

```sh
python -I scripts/run_gpu_tests.py --build <unzipped runtime-windows>
```

`build.py` and `check.py` take the same folder as their second argument:

```sh
python -I runtime/test/shaders/build.py <out dir>
python -I runtime/test/shaders/check.py <out dir> <unzipped runtime-windows>
```

The artifact belongs to the commit that built it, so run the scripts from that commit as well (see the note above).

## Run them

The whole suite, assembled, translated and run in one go:

```sh
python -I scripts/run_gpu_tests.py --build runtime/build/linux     # Windows: the default runtime\build
```

`--filter <text>` runs only matching tests, and `--work <dir>` keeps the intermediate files.

A GPU test program that runs longer than 120 seconds is killed and counted as a failure, so a hung driver does
not stall the suite. `LEONIDA_GPU_TEST_TIMEOUT` sets the limit in seconds for both `check.py` and
`run_gpu_tests.py`; `0` turns the limit off.
**Rebuild the test programs after pulling.** The programs carry the V# and push-constant layout the translator expects. An executable from an older commit still loads and runs every shader, so a stale `vk_compute_test` does not fail loudly: when a newer translator starts enforcing a descriptor rule the old program does not set up (the raw V# in `pc[2..3]`, for one), every compute test reports the untouched `0xEEEEEEEE` fill pattern and the run looks like a translator regression. If all the compute tests in a run return that pattern, rebuild first (`cmake --build runtime/build/linux`, or `cmake --build runtime\build` on Windows) and run again before reading anything into the words.

The two-step form, which `CONTRIBUTING.md` describes:

```sh
python -I runtime/test/shaders/build.py <out dir>
python -I runtime/test/shaders/check.py <out dir> <folder with the test programs>
```

For D3D12/runtime tests, `LEONIDA_VK_VALIDATION=1` enables `VK_LAYER_KHRONOS_validation`. When running with validation, set `DISABLE_RTSS_LAYER=1` and `DISABLE_VULKAN_OBS_CAPTURE=1` if MSI Afterburner (RivaTuner) or OBS is installed: their overlay layers can add unsupported swapchain usage, and in an elevated (administrator) terminal the loader ignores `VK_LOADER_LAYERS_DISABLE=~implicit~` (see "Known problems").

## The Windows runtime on Linux, under Wine or Proton

The Linux build above makes only the Vulkan programs. The loader (`gta6rt`), its shims and their tests are Windows
programs; `runtime/cmake/clang-cl-xwin.cmake` cross-compiles them on Linux with clang-cl and lld-link (LLVM 19+, but not 22.1.7 or 22.1.8 (see Known problems): the
MSVC STL that xwin lays out rejects older Clang; versioned names go in with `-DCMAKE_CXX_COMPILER=clang-cl-19` etc.)
against an MSVC CRT and Windows SDK that [xwin](https://github.com/Jake-Shadle/xwin) downloads from Microsoft
(running it means accepting Microsoft's licence), and `ctest` runs them through Wine:

```sh
xwin --accept-license --sdk-version 10.0.26100 splat --use-winsysroot-style --preserve-ms-arch-notation \
     --output ~/.local/share/xwin-sdk
cmake -S runtime -B runtime/build/windows --toolchain "$PWD/runtime/cmake/clang-cl-xwin.cmake" \
      -DXWIN_DIR="$HOME/.local/share/xwin-sdk"
cmake --build runtime/build/windows --parallel
WINEPREFIX="$HOME/.local/share/leonida-wine" ctest --test-dir runtime/build/windows
```

The build type defaults to `RelWithDebInfo`: xwin leaves out the debug CRT, so `Debug` needs `--include-debug-libs`.
`make_game.py`'s `runtime` stage makes this same build on Linux when the tools and the SDK are there (`--runtime-build
runtime/build/windows` takes a folder made by hand like a Windows build). CMake packages are
looked up only inside the SDK folder, so a Windows SDL3 for the GameInput shim is passed as `-DSDL3_DIR=<its
cmake folder>` (`CMAKE_PREFIX_PATH` is not searched here).

For Proton, start any program once with `proton run` (with `STEAM_COMPAT_DATA_PATH` and
`STEAM_COMPAT_CLIENT_INSTALL_PATH` set), so that it puts vkd3d-proton and DXVK into the prefix, then configure with
`-DCMAKE_CROSSCOMPILING_EMULATOR="<Proton>/files/bin/wine"` and run `ctest` with
`WINEPREFIX=$STEAM_COMPAT_DATA_PATH/pfx WINEDLLOVERRIDES="d3d12=n;d3d12core=n;dxgi=n"`. `proton run` itself doesn't
pass the program's output on, so the tests can't run through it.

The tests that compare with the host's D3D12 (`root_sig_test`, `root_sig`, `device_proxy`, `d3d12_vulkan_device`)
check against Wine's vkd3d or vkd3d-proton here, not Microsoft's runtime, so some of them fail for that reason.

## GPUs with 32-wide subgroups (NVIDIA, Intel, Apple)

The game's shaders are mostly wave64. A GPU that can't run 64-wide compute subgroups skips the wave64 tests
(`cs_wavemask`, `cs_wave64_ballot`, `cs_wm_*`, `cs_w64x_*`, `ms_w64x_ballot`); that's expected, not a failure.

To test the wave64 emulation that makes them exact on these GPUs, set `LEONIDA_W64_EMULATE=1`, **for both
`build.py` and `check.py`**: `build.py` translates with emulation only when it's set. If you set it only for
`check.py`, the shaders are checked against the emulated expectations without being emulated, and every wave64 test
fails.

## GPUs without mesh shaders (NVIDIA Pascal and older)

Without `VK_EXT_mesh_shader`, the `ms_*` tests are skipped; that's expected, not a failure. To test the path that
runs them anyway, set `LEONIDA_MESH_COMPUTE=1` for `run_gpu_tests.py`, together with `LEONIDA_W64_EMULATE=1` on a
32-wide GPU (every mesh shader is wave64). The mesh shaders are then translated with `--mesh-compute`, dispatched as
compute and drawn by a generated vertex shader (`docs/design.md`, "Mesh shaders as compute"), so the device needs
`bufferDeviceAddress` and `shaderInt64` instead.

## Without a GPU: Mesa llvmpipe

Mesa's llvmpipe is a software Vulkan driver (`mesa-vulkan-drivers` on Debian and Ubuntu). CI runs the suite on it
for every pull request (`.github/workflows/runtime.yml`), and you can do the same on Linux without a GPU:

```sh
sudo apt-get install -y libvulkan-dev mesa-vulkan-drivers vulkan-tools spirv-tools llvm-18
cmake -S runtime -B runtime/build/linux && cmake --build runtime/build/linux --parallel
python -I scripts/run_gpu_tests.py --build runtime/build/linux --llvm-mc /usr/lib/llvm-18/bin/llvm-mc
LEONIDA_W64_EMULATE=1 python -I scripts/run_gpu_tests.py --build runtime/build/linux \
    --llvm-mc /usr/lib/llvm-18/bin/llvm-mc --filter cs_
```

With Mesa 25.2 (llvmpipe, LLVM 20) the first run passes 52 of 88 and skips 36, the second passes 37 of 39 and
skips 2. The skips are expected: llvmpipe has 8-wide subgroups (so every wave64 test skips without emulation, and
the 32-lane tests `ps_bitmask_perm`, `cs_wqm_w32` and `cs_w32_lane_wrap` skip in both runs), no
`VK_KHR_fragment_shader_barycentric` (`ps_bary`, `ps_uniform_branch`, `ps_scc_mask32`) and no
`VK_KHR_maintenance8` (`ps_texoff`). With
`LEONIDA_W64_EMULATE=1` the mesh tests abort inside llvmpipe's compiler ("LLVM ERROR: Cannot emit physreg copy
instruction"), which is why the second run is limited to compute. llvmpipe is not a substitute for a result from a
real GPU in [#166](https://github.com/ItsNotRexed/Leonida/issues/166): it catches regressions, not hardware
differences.

## GPUs without fragment barycentrics (NVIDIA Volta, Pascal and older)

`ps_bary`, `ps_uniform_branch` and `ps_scc_mask32` read raw barycentrics, which need
`VK_KHR_fragment_shader_barycentric`; so do `ps_interp_f16_high` and `ms_f16_pack` (16-bit interpolants are
`PerVertexKHR` inputs, and `--bary-fallback` cannot express them, so they skip under it). NVIDIA has it from Turing on; Volta (V100, Titan V) and older lack it, as do
llvmpipe and some other drivers. Where the device lacks it they're skipped; that's expected, not a failure. To
test the fallback that runs them anyway, set `LEONIDA_BARY_FALLBACK=1`, **for both `build.py` and `check.py`** (or
for `run_gpu_tests.py`). The shaders are then translated with `--bary-fallback` and drawn through a pass-through
geometry shader that supplies the barycentrics (`docs/design.md`, "Barycentric fallback"), so the device needs
geometry shaders instead.

## Macs: 16-bit interpolation without PerVertexKHR

`ps_interp_f16_high` (and `ms_f16_pack`) normally read 16-bit interpolants through `PerVertexKHR` inputs, which
MoltenVK can't convert to Metal ("PerVertexKHR decoration is not supported in MSL"). `run_gpu_tests.py` asks
`tools/vk_subgroup_info.py` whether the device can take them and, on MoltenVK, translates with
`--f16-interp-fallback` by itself: the vertex or mesh shader then also exports the decoded f16 halves
(`docs/design.md`, "16-bit interpolation without `PerVertexKHR`"). `LEONIDA_F16_INTERP_FALLBACK=1` or `0` forces it
on or off; `build.py` (for `check.py`) translates with the flag only when that variable is `1`. `ms_f16_pack` also runs
on a Mac with `LEONIDA_MESH_COMPUTE=1 LEONIDA_W64_EMULATE=1`.

## GPUs without VK_KHR_maintenance8 (NVIDIA Pascal and older)

`ps_texoff` and `ps_texoff_implicit` sample with a run-time texel offset, which needs `VK_KHR_maintenance8`;
without it they're skipped, which is expected, not a failure. To test the fallback, set `LEONIDA_TEXOFF_FALLBACK=1`
**for both `build.py` and `check.py`** (or for `run_gpu_tests.py`): the offset is then folded into the coordinate
(`docs/design.md`, "Texel offsets"), and the +16/-32 sample is judged on every device.

## Fused multiply-add: GLSL Fma or OpFmaKHR

`cs_fma_fused` checks that `v_fma_f32`, `v_fmac_f32`, `v_fmaak_f32`, `v_fmamk_f32` (and one `v_fma_f16`) round once, on
operands taken from user SGPRs so the driver cannot fold them. It passes with the default translation (`--fma glsl`,
GLSL.std.450 `Fma`) and with `LEONIDA_FMA=khr` **for `build.py`** (or for `run_gpu_tests.py`), which emits `OpFmaKHR`.
That needs `VK_KHR_shader_fma` with `shaderFmaFloat32`; the test programs enable it when the device has it and exit
77 (a SKIP) for a module that needs it on a device without it. The Vulkan specification allows a driver to split GLSL
`Fma` into two roundings, which is why `khr` exists. Neither the AMD (26.11.2) nor the NVIDIA (617.42) driver we tried
does, so `cs_fma_fused` passing with `glsl` is not evidence that every driver is safe.

## Hardware oracle (optional, AMD)

`valu_fuzz.py`, `lds_fuzz.py`, `f32_fuzz.py` and `image_fuzz.py` take `--oracle hw`: the same programs run as native
compute kernels on an AMD GPU (`tools/hw_oracle/`; setup in [shader_validation.md](shader_validation.md), "Hardware
oracle"). It needs AMD clang with the amdgcn target (`LEONIDA_AMD_CLANG`), the AMD driver's HIP runtime and a host C
compiler on Windows, or the libdrm launcher on Linux. Without them a run prints `hardware oracle unavailable: ...` and
exits 77, so CI skips it. `image_fuzz.py --oracle hw` loads a linear texture through a T# built for the GPU's family
(GFX11 on gfx11xx, GFX10 on gfx103x; `tests/test_image_fuzz_hw.py` has the exact words) and compares every texel with
its CPU model; add `--negative-control` to check that wrong models are caught. A gfx1150 (RDNA3.5) is not RDNA2: a
match supports the model and does not prove RDNA2. Run these one at a time: a second job on the same GPU can hang it.

## Choosing the GPU

With more than one GPU, pick one through the Vulkan loader, for example
`VK_LOADER_DRIVERS_SELECT=*amd*` or `VK_LOADER_DRIVERS_SELECT=nv-vk*` (or `VK_DRIVER_FILES=<path to the ICD json>`).
`vulkaninfo --summary` shows which devices the loader then sees.

Two GPUs of one vendor share one driver, so the loader cannot tell them apart (an RTX 5070 Ti next to an RTX 3060,
say; without a choice the D3D12 tests take the last discrete GPU). `LEONIDA_VK_DEVICE` names the GPU by part of its
name, without case: `LEONIDA_VK_DEVICE=5070` or `LEONIDA_VK_DEVICE=radeon`. The D3D12 tests (`vk_test_device.h`)
and the layer's standalone device read it, and a value that matches no device fails the run instead of testing
another GPU. For the host side of the comparison tests, `LEONIDA_HOST_ADAPTER` picks the D3D12 adapter the same way.
The GPU tools of `runtime/test/shaders` (`vk_compute_test`, `vk_harness`, ...) read it the same way through
`vk_device_pick.h`, and fail with a message naming the filter when it matches no device.

## Known problems

| What you see | Cause | What to do |
|---|---|---|
| Windows refuses to start freshly built test programs ("An Application Control policy has blocked this file"), and the runner stops with a traceback | Smart App Control on Windows 11 | Turn Smart App Control off for the run |
| Validation errors without a VUID, or messages from a layer you didn't ask for | An implicit Vulkan layer from other software (Rockstar Social Club, ReShade, MangoHud, an overlay) | Set `VK_LOADER_LAYERS_DISABLE=~implicit~` |
| Validation errors that only your machine shows (e.g. `xbox_present`'s sRGB swapchain `STORAGE` message), with MSI Afterburner/RivaTuner or OBS installed | An overlay layer that hooks the swapchain: RivaTuner's `VK_LAYER_RTSS` adds `STORAGE` usage to it. On some setups it stays loaded even with `VK_LOADER_LAYERS_DISABLE=~implicit~` | Also set the layer's own switch: `DISABLE_RTSS_LAYER=1`, `DISABLE_VULKAN_OBS_CAPTURE=1` (OBS). Any other layer's switch is the `disable_environment` entry in its manifest (the JSON listed under `HKLM\SOFTWARE\Khronos\Vulkan\ImplicitLayers` or `HKCU\...`). `VK_LOADER_DEBUG=layer` prints the layers that were actually inserted |
| Validation errors without a VUID, or messages from a layer you didn't ask for | An implicit Vulkan layer from other software (Rockstar Social Club, ReShade, MangoHud, an overlay) | Set `VK_LOADER_LAYERS_DISABLE=~implicit~`. Note that third-party overlays (RivaTuner `VK_LAYER_RTSS` from MSI Afterburner, OBS `VK_LAYER_OBS_HOOK`) stay loaded under `~implicit~` and can add `VK_IMAGE_USAGE_STORAGE_BIT` to swapchains; set `DISABLE_RTSS_LAYER=1` and `DISABLE_VULKAN_OBS_CAPTURE=1` to disable them |
| False validation failures in `xbox_present` or present tests (e.g. sRGB swapchain `STORAGE` messages) with MSI Afterburner or OBS running | Overlay Vulkan layers (RivaTuner `VK_LAYER_RTSS`, OBS `VK_LAYER_OBS_HOOK`) remain active under `VK_LOADER_LAYERS_DISABLE=~implicit~` and inject `VK_IMAGE_USAGE_STORAGE_BIT` into swapchains | Set `DISABLE_RTSS_LAYER=1` and `DISABLE_VULKAN_OBS_CAPTURE=1` for validation runs |
| Validation errors that only your machine shows (e.g. `xbox_present`'s sRGB swapchain `STORAGE` message), with MSI Afterburner/RivaTuner or OBS installed | An overlay layer that hooks the swapchain: while RivaTuner Statistics Server is running, its `VK_LAYER_RTSS` adds `STORAGE` usage to it. In an elevated (administrator) terminal the loader ignores `VK_LOADER_LAYERS_DISABLE` (`VK_LOADER_DEBUG=all` says so), so `~implicit~` leaves the layer in | Run the tests from a normal terminal, close the overlay application, or set the layer's own switch, which works elevated too: `DISABLE_RTSS_LAYER=1`, `DISABLE_VULKAN_OBS_CAPTURE=1` (OBS). Any other layer's switch is the `disable_environment` entry in its manifest (the JSON listed under `HKLM\SOFTWARE\Khronos\Vulkan\ImplicitLayers` or `HKCU\...`). `VK_LOADER_DEBUG=layer` prints the layers that were actually inserted |
| `vk_sync_validation` fails with `LEONIDA_VK_VALIDATION=1` ("the write-after-write hazard was not reported") | Synchronization validation is off although the tests ask for it: an older layer, or a layer setting such as `VK_KHRONOS_VALIDATION_VALIDATE_SYNC=false` or a `vk_layer_settings.txt` that overrides it. Every other GPU test then reports "0 validation errors" without having been checked | Remove the override or update the Vulkan SDK, then rerun `ctest -R vk_sync_validation` until it passes |
| Validation errors that only your machine shows (e.g. `xbox_present`'s sRGB swapchain `STORAGE` message), with MSI Afterburner/RivaTuner or OBS installed | An overlay layer that hooks the swapchain: while RivaTuner Statistics Server is running, its `VK_LAYER_RTSS` adds `STORAGE` usage to it. In an elevated (administrator) terminal the loader ignores `VK_LOADER_LAYERS_DISABLE` (`VK_LOADER_DEBUG=all` says so), so `~implicit~` leaves the layer in | Run the tests from a normal terminal, close the overlay application, or set the layer's own switch, which works elevated too: `DISABLE_RTSS_LAYER=1`, `DISABLE_VULKAN_OBS_CAPTURE=1` (OBS). Any other layer's switch is the `disable_environment` entry in its manifest (the JSON listed under `HKLM\SOFTWARE\Khronos\Vulkan\ImplicitLayers` or `HKCU\...`). `VK_LOADER_DEBUG=layer` prints the layers that were actually inserted |
| Every test crashes inside `libnvidia-eglcore.so` | NVIDIA's 535 server driver (Ubuntu 24.04's default) | Use driver 615 or newer |
| `build.py`: 0 translated, reason `.text` | An old `build.py` with LLVM 19 | Update to current `main` |
| `vk_harness` doesn't compile: `VK_KHR_maintenance8` names unknown | Vulkan headers older than the SDK the code was written against | Update to current `main` (the code now builds without it), or use current Khronos Vulkan-Headers; conda-forge's package is too old |
| `ps_bary` and `ps_uniform_branch` fail by about 0.9 | The GPU runs pixel shaders wave64 without fragment barycentrics (GCN3 and RDNA1 on RADV), or an old AMD Windows driver (22.40) | Current `main` skips them on such devices; on Windows, update the driver |
| `ps_interp_f16_high` fails on a Mac: `PerVertexKHR decoration is not supported in MSL`, then `vkCreateGraphicsPipelines` -3 | MoltenVK can't convert `PerVertexKHR` inputs to Metal | Current `main` translates it with `--f16-interp-fallback` on MoltenVK (see "Macs: 16-bit interpolation") |
| `ps_vcmpx_forms` fails on a Mac: word 1 reads `0xf05e8e16`, want `0xf05e8a12` | MoltenVK's fast-math makes a compare of a NaN with itself true (SHADER-234) | Current `main`'s test programs turn it off (`MVK_CONFIG_FAST_MATH_ENABLED=0`, unless you set the variable yourself); rebuild them |
| On a Mac, a translated shader's GPU time is 1.5 to 16 times what it would be without the precise-arithmetic overhead | MoltenVK's SPIRV-Cross turns every `NoContraction` float add, subtract and multiply into a call to an `[[clang::optnone]]` helper that Metal does not inline. MoltenVK 1.4.2 and `main` up to at least `5189f8de` are affected (SHADER-234, [design notes](design.md#translator-status-and-runtime-contract)) | Expected with these MoltenVK versions, and the output is still exact. Before you time shaders on a Mac, build MoltenVK with SPIRV-Cross `98cbb7c0` or newer in `ExternalRevisions/SPIRV-Cross_repo_revision` (KhronosGroup/MoltenVK#2917 moves it to `57e4efcc`), and point `LEONIDA_VULKAN_LOADER` at its `libMoltenVK.dylib` |
| `gs_tri` and `ms_tri` fail on a Mac | MoltenVK has no geometry stage and no `VK_EXT_mesh_shader` | Current `main` reports them as skipped |
| On a Mac, `valu_fuzz.py --family f16` reads `+0` where the model wants `-0`, from a multiply whose exact product is -0 (`v_mul_f16`, `v_pk_mul_f16`) or from an FMA whose product and addend are both -0 (`v_fma_f16`, `v_fmac_f16`, `v_pk_fma_f16`) | MoltenVK computes a `NoContraction` multiply as `fma(l, r, +0)`, which drops the sign. MoltenVK 1.4.2 and `main` up to at least `5189f8de` are affected (SHADER-182, [design notes](design.md#translator-status-and-runtime-contract)) | Expected with these MoltenVK versions. To run the rest of the fuzzer clean, add `--skip-op` for each of the five ops. To see them match, install Homebrew's `mesa` and use its KosmicKrisp driver: `LEONIDA_VULKAN_LOADER=/opt/homebrew/opt/vulkan-loader/lib/libvulkan.1.dylib VK_DRIVER_FILES=/opt/homebrew/share/vulkan/icd.d/kosmickrisp_mesa_icd.aarch64.json python3 -I tools/valu_fuzz.py --family f16 --build runtime/build/linux`. Or build MoltenVK with SPIRV-Cross `300681bb` or newer in `ExternalRevisions/SPIRV-Cross_repo_revision`, and point `LEONIDA_VULKAN_LOADER` at its `libMoltenVK.dylib` |
| CMake stops with "needs the Windows SDK 10.0.26100 or newer", or MSVC reports `'Desc_1_2': is not a member of 'D3D12_VERSIONED_ROOT_SIGNATURE_DESC'` | A Windows SDK older than 10.0.26100 (10.0.22621 is the common one), which has no version 1.2 root signatures | `winget install Microsoft.WindowsSDK.10.0.26100`, or add it in the Visual Studio Installer, then delete `runtime/build` and configure again |
| `ctest` lists `fsr_dispatch` and `bvh_builder` as failed (`Not Run`) after clang-cl hits an internal compiler error in `fsr_dispatch_test.cpp` and `bvh_builder_test.cpp` | clang-cl 22.1.8 on Fedora (Linux cross-build) and Windows, plus 22.1.7 on Windows | Use clang-cl 23.1.6 (both files compile on Windows) |
| `run_gpu_tests.py` fails on a `dict \| None` annotation | Python older than 3.10 | Use Python 3.11+ |
| `vulkaninfo` lists `Microsoft Direct3D12 (<gpu>)` next to the real GPU and the tests run on it | The Windows "OpenCL™, OpenGL® and Vulkan® Compatibility Pack" installs Mesa's Dozen, a Vulkan-on-D3D12 layer that reports `DISCRETE_GPU` like the native card | Current `main` prefers a native ICD over Microsoft's vendor (`0x1414`) in every harness; set `LEONIDA_VK_DEVICE=<part of the real GPU name>` to select explicitly. See #915. |
| A GPU test that uses `image_load` (or lighting built from real shaders) reads zeros from its image; @Silliana saw every texel except pixel (0,0) come back zero in their own layer (#603, not reproduced here) | In this runtime the shader turns the T# words into an index through the descriptor hash table (`runtime/src/descriptor_table.h`: a miss gives index 0), and index 0 is a 1×1 image of zeros (`runtime/src/vulkan_bindless.h`), so a hand-built or zeroed T# that was never registered reads zeros even at (0,0). The translator does not compare coordinates with the T# size, and `cs_imgload.asm` only uses its T# words as a hash key | Register the descriptor through the runtime (for D3D12, create it with `CreateShaderResourceView`) so its T# hits the table, with the image's real width and height (stored as size − 1, see `runtime/src/gfx10_descriptors.h`) |

## Posting your result

Use the template at the top of [#166](https://github.com/ItsNotRexed/Leonida/issues/166): GPU, driver, subgroup
range, OS, the commit you built, the tool versions and the counts each script printed. Post counts and pass/fail
lines only, never shaders or logs from the game. A failure that isn't in the table above gets its own issue, linked
from your entry.
