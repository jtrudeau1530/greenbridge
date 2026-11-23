# GPU test results

This page keeps a local record of GPU test runs. The project's public tracker is
[#166](https://github.com/ItsNotRexed/Leonida/issues/166) (one comment per GPU and driver);
post counts and pass/fail lines there, never game shaders, logs or private paths.

| GPU | Driver, OS | Commit | build.py | check.py | run_gpu_tests.py | Failed | Entry |
|---|---|---|---|---|---|---|---|
| AMD Radeon RX 9070 GRE (RDNA4, GFX1201), subgroup 32..64 | RADV, Mesa 26.3.1-arch3.5, Vulkan 1.4.363, CachyOS | `05961f0` | 264/264 translated and valid | 203/206 pass, 3 skipped | 218/235 passed, 16 skipped (native and emulated) | `ms_wave_info_s3`, flaky (see below) | [local note](gpu-test-results/RX9070GRE.txt) |
| AMD Radeon RX 7900 GRE (RDNA3, gfx1100), subgroup 32..64 | AMD proprietary 26.11.2 (LLPC), Vulkan 1.4.359, Windows 11 x64 | `5d74c27` | 247/247 translated and valid | run via run_gpu_tests.py | 225/241 passed, 16 skipped | none | [#166](https://github.com/ItsNotRexed/Leonida/issues/166#issuecomment-7115041436) |

## AMD Radeon RX 9070 GRE — 2026-11-23

```
GPU:        AMD Radeon RX 9070 GRE (RDNA4 / GFX1201, Navi 48, PCI 1002:7550), subgroup 32..64
Driver:     RADV (Mesa 26.3.1-arch3.5), Vulkan 1.4.363
Subgroup:   32..64 (minSubgroupSize 32, maxSubgroupSize 64; default 64)
OS:         CachyOS, kernel 7.3.5-2-cachyos
Commit:     05961f0eeb24486ce25ddc7dfb453ca87db2be5f (main, 2026-11-23)
Tools:      llvm-mc 23.1.5 (native), SPIRV-Tools v2026.4, CMake 4.4.5 + Ninja 1.13.2, GCC 16.2.1, Python 3.14.9
build.py:   264 shaders: 264 translated, 264 pass spirv-val (6 mesh fixtures not built: ms_clip, ms_f16_pack, ms_tri, ms_v5_workgroup, ms_w64x_ballot, ms_wave_info_s3)
check.py:   203 of 206 pass, 3 skipped (the three mesh fixtures built only by run_gpu_tests.py)
run_gpu_tests.py: 218/235 passed, 16 skipped, 1 failed (native and LEONIDA_W64_EMULATE=1 alike)
Failed:     ms_wave_info_s3: at (0,0) got (-1,-1,-1,-1), want (0.0078,0.0078,0.5,1.0). Flaky on this RADV GPU: 1 pass and 2 fails in three isolated --filter reruns; same signature as the known round-44 mesh regression (#6172, fixed by #5741/#5855). Not a new failure.
Skipped:    16 skips are the documented "run by d3d12_*_test / w64_graphics.py" cases.
```

Setup: `cmake -S runtime -B runtime/build/linux -G Ninja -DCMAKE_BUILD_TYPE=Release -DLEONIDA_VULKAN_HEADERS_FETCH=ON`;
implicit Vulkan layers disabled (`VK_LOADER_LAYERS_DISABLE='~implicit~'`) because MangoHud/Steam layers are installed.
Inside `run_gpu_tests.py`, 241/241 shaders translated and passed `spirv-val` in both the native and emulated runs.
Every pixel test other than the flaky mesh fixture was exact (4096/4096).

AI-assisted (Hermes Agent): built `main` at `05961f0`, ran `build.py`, `check.py` and
`run_gpu_tests.py` (native and emulated) on this GPU, and verified the isolated reruns of the one failure.

## AMD Radeon RX 7900 GRE — 2026-11-23

```
GPU:        AMD Radeon RX 7900 GRE (RDNA3 / gfx1100, PCI 1002:744C), subgroup 32..64
Driver:     AMD proprietary 26.11.2 (LLPC), Vulkan 1.4.359
Subgroup:   32..64 (minSubgroupSize 32, maxSubgroupSize 64; default 64)
OS:         Windows 11 x64
Commit:     5d74c2723 (main, 2026-11-23)
Tools:      llvm-mc 23.1.5 (MSYS2 UCRT64), Vulkan SDK 1.4.363 / SPIRV-Tools v2026.4, MSVC 14.51 (VS 2026 Community), Python 3.12.1
build.py:   247 shaders: 247 translated, 247 pass spirv-val
            wave64 VS/PS/GS on 32-wide subgroups: tier A: 1, tier C: 1
check.py:   not run separately (run_gpu_tests.py below)
run_gpu_tests.py: 225/241 passed, 16 skipped
Failed:     none
Skipped:    16 skips are the documented "run by d3d12_*_test / w64_graphics.py" cases.
```

Setup: `scripts\build_runtime.ps1` from a plain PowerShell. Without Ninja on `PATH` it took the cl.exe fallback, which
builds every Vulkan program the GPU tests use. `llvm-mc` came from the MSYS2 package `mingw-w64-ucrt-x86_64-llvm`
(23.1.5, AMDGPU target included) with `C:\msys64\ucrt64\bin` on `PATH`; on an old MSYS2 install, run `pacman -Syu`
until it reports nothing to do before installing it. Loader warnings from installed Medal/OBS Vulkan layers did not
affect the run. This is the first RDNA3 entry: wave64 and mesh shader tests ran natively (all `ms_*` pass).

AI-assisted (Claude): helped set up the tools and write this entry; the build and test run were done on this GPU.
