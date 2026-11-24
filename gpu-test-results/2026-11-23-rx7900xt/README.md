# RX 7900 XT — recovered validation, 2026-11-23

GPU: AMD Radeon RX 7900 XT (RDNA3 / Navi 31). Historical environment: Arch Linux, RADV Mesa 26.2.3, Wine 11.18, clang-cl 23.1.6 + xwin, RelWithDebInfo. Individual early runs may use a different compiler version; see their historical reports.

These are reconstructions from saved results, not new test runs. Raw game output is deliberately absent. No non-black game image or playable state was established. See [corrections and current evidence](../../docs/validation-recovery.md).

Available records: 60 run directories, 94 test-suite logs, 11 early loader records, 70 historical reports. Missing fields mean not recovered, not a pass.

## Runtime comparisons

| Run | Init / run exit | Traced calls | Failed trace outcomes | Successful present milestone |
|---|---|---:|---:|---:|
| [c2-4348-db8a4a1e4-20261123-114411-440707](runs/c2-4348-db8a4a1e4-20261123-114411-440707.json) | 0 / 3 | 0 | 0 | 0 |
| [c2-4348-db8a4a1e4-20261123-114615-474349](runs/c2-4348-db8a4a1e4-20261123-114615-474349.json) | 0 / 3 | 0 | 0 | 0 |
| [c2a-7216-ea2a0bf26-20261123-113841-403478](runs/c2a-7216-ea2a0bf26-20261123-113841-403478.json) | 0 / 5 | 0 | 0 | 0 |
| [c2a-7216-ea2a0bf26-20261123-113913-406294](runs/c2a-7216-ea2a0bf26-20261123-113913-406294.json) | 0 / 5 | 0 | 0 | 0 |
| [c3a-7216-3553b0b5c-20261123-114053-424973](runs/c3a-7216-3553b0b5c-20261123-114053-424973.json) | 0 / 124 | 0 | 0 | 0 |
| [c3a-7216-3553b0b5c-20261123-114252-434214](runs/c3a-7216-3553b0b5c-20261123-114252-434214.json) | 0 / 5 | 0 | 0 | 0 |
| [c3a-7216-3553b0b5c-20261123-114319-436727](runs/c3a-7216-3553b0b5c-20261123-114319-436727.json) | 0 / 3 | 0 | 0 | 0 |
| [c3a-7216-3553b0b5c-20261123-114434-444988](runs/c3a-7216-3553b0b5c-20261123-114434-444988.json) | 0 / 5 | 0 | 0 | 0 |
| [c3a-7216-3553b0b5c-20261123-114552-470448](runs/c3a-7216-3553b0b5c-20261123-114552-470448.json) | 0 / 5 | 0 | 0 | 0 |
| [c3a-7216-3553b0b5c-20261123-114624-477075](runs/c3a-7216-3553b0b5c-20261123-114624-477075.json) | 0 / 3 | 0 | 0 | 0 |
| [c3a-7216-3553b0b5c-20261123-114721-483136](runs/c3a-7216-3553b0b5c-20261123-114721-483136.json) | 0 / 3 | 0 | 0 | 0 |
| [c3b-diag-58950fce1-20261123-115337-526837](runs/c3b-diag-58950fce1-20261123-115337-526837.json) | 0 / 5 | 0 | 0 | 0 |
| [c3b-diag-a6b6ac35c-20261123-115543-548906](runs/c3b-diag-a6b6ac35c-20261123-115543-548906.json) | 0 / 5 | 0 | 0 | 0 |
| [c3c-diag-aacf08f51-20261123-120102-588580](runs/c3c-diag-aacf08f51-20261123-120102-588580.json) | 0 / 5 | 0 | 0 | 0 |
| [c3d-diag-aca2c62ed-20261123-120528-622011](runs/c3d-diag-aca2c62ed-20261123-120528-622011.json) | 0 / 3 | 0 | 0 | 0 |
| [c3e-diag-03034f8bc-20261123-121031-658925](runs/c3e-diag-03034f8bc-20261123-121031-658925.json) | 0 / 3 | 0 | 0 | 0 |
| [c3e-diag-03034f8bc-20261123-121258-671758](runs/c3e-diag-03034f8bc-20261123-121258-671758.json) | 0 / 3 | 1640 | 0 | 0 |
| [c3f-diag-6f2ebe266-20261123-121740-706115](runs/c3f-diag-6f2ebe266-20261123-121740-706115.json) | 0 / 3 | 1637 | 0 | 0 |
| [c3g-diag-13c446509-20261123-122236-741622](runs/c3g-diag-13c446509-20261123-122236-741622.json) | 0 / 3 | 1637 | 0 | 0 |
| [c3h-bvh-90830ffbe-20261123-130547-1036670](runs/c3h-bvh-90830ffbe-20261123-130547-1036670.json) | 0 / 3 | 1656 | 0 | 0 |
| [c3h-diag-90830ffbe-20261123-123001-789182](runs/c3h-diag-90830ffbe-20261123-123001-789182.json) | 0 / 99 | 1640 | 0 | 0 |
| [c3h-diag-90830ffbe-20261123-125529-978410](runs/c3h-diag-90830ffbe-20261123-125529-978410.json) | 0 / 99 | 1640 | 0 | 0 |
| [c3i-7163-084c8c60e-20261123-141313-1345364](runs/c3i-7163-084c8c60e-20261123-141313-1345364.json) | 0 / 3 | 1656 | 0 | 0 |
| [c3j-7330-efc347830-20261123-141620-1362786](runs/c3j-7330-efc347830-20261123-141620-1362786.json) | 0 / 5 | 1656 | 0 | 0 |
| [c3k-7115-e22994464-20261123-142312-1408991](runs/c3k-7115-e22994464-20261123-142312-1408991.json) | 0 / 3 | 1658 | 1 | 0 |
| [c3l-7323-a30e78ee7-20261123-143518-1475178](runs/c3l-7323-a30e78ee7-20261123-143518-1475178.json) | 0 / 3 | 1660 | 0 | 0 |
| [c3m-6891-23060e087-20261123-144658-1538463](runs/c3m-6891-23060e087-20261123-144658-1538463.json) | 0 / 5 | 2261 | 0 | 0 |
| [c3n-7115v2-8f756ebe0-20261123-145449-1587607](runs/c3n-7115v2-8f756ebe0-20261123-145449-1587607.json) | 0 / 5 | 2261 | 0 | 0 |
| [c3o-7201-8517-20261123-150512-1647735](runs/c3o-7201-8517-20261123-150512-1647735.json) | 0 / 3 | 2334 | 7 | 0 |
| [c3o-7201-8517-20261123-153246-1814164](runs/c3o-7201-8517-20261123-153246-1814164.json) | 0 / 3 | 2334 | 7 | 0 |
| [c3p-7529-20261123-153319-1818623](runs/c3p-7529-20261123-153319-1818623.json) | 0 / 3 | 2334 | 7 | 0 |
| [c3p-7529-20261123-155353-1975025](runs/c3p-7529-20261123-155353-1975025.json) | 0 / 3 | 2334 | 7 | 0 |
| [c3q-7746-20261123-155424-1979539](runs/c3q-7746-20261123-155424-1979539.json) | 0 / 3 | 2334 | 1 | 0 |
| [c3q-7746-20261123-164851-2307166](runs/c3q-7746-20261123-164851-2307166.json) | 0 / 3 | 2334 | 1 | 0 |
| [c3r-7880-20261123-164923-2311787](runs/c3r-7880-20261123-164923-2311787.json) | 0 / 3 | 2334 | 1 | 0 |
| [c3s-2490reset-20261123-165632-2358055](runs/c3s-2490reset-20261123-165632-2358055.json) | 0 / 3 | 2336 | 0 | 0 |
| [c5972cap-20261123-181447-2844731](runs/c5972cap-20261123-181447-2844731.json) | 0 / 124 | 56878 | 0 | 64 |
| [c5972cap-20261123-181723-2858811](runs/c5972cap-20261123-181723-2858811.json) | 0 / 124 | 57095 | 0 | 64 |
| [c5972cap-20261123-181823-2865319](runs/c5972cap-20261123-181823-2865319.json) | 0 / 124 | 55981 | 0 | 32 |
| [c5972cap-20261123-181947-2873720](runs/c5972cap-20261123-181947-2873720.json) | 0 / 124 | 130018 | 0 | 2048 |
| [c5972cap-20261123-182242-2888275](runs/c5972cap-20261123-182242-2888275.json) | 0 / 124 | 124657 | 0 | 1024 |
| [c5972diag-20261123-183407-2953114](runs/c5972diag-20261123-183407-2953114.json) | 0 / 124 | 67739 | 0 | 256 |
| [c5972diag2-20261123-184115-3000868](runs/c5972diag2-20261123-184115-3000868.json) | 0 / 124 | 67184 | 0 | 256 |
| [c5972diag2-20261123-190117-3120149](runs/c5972diag2-20261123-190117-3120149.json) | 0 / 124 | 74290 | 0 | 512 |
| [c5972layout-20261123-193550-3338242](runs/c5972layout-20261123-193550-3338242.json) | 0 / 124 | 66949 | 0 | 256 |
| [c5972pull-20261123-195800-3468504](runs/c5972pull-20261123-195800-3468504.json) | 0 / 124 | 64442 | 0 | 256 |
| [c5972pullmirror-20261123-201026-3541023](runs/c5972pullmirror-20261123-201026-3541023.json) | 0 / 124 | 64706 | 0 | 256 |
| [c5972rect-20261123-184930-3058665](runs/c5972rect-20261123-184930-3058665.json) | 0 / 124 | 67140 | 0 | 256 |
| [c5972rect-20261123-185056-3067472](runs/c5972rect-20261123-185056-3067472.json) | 0 / 124 | 68050 | 0 | 256 |
| [c5972rect-20261123-185757-3101049](runs/c5972rect-20261123-185757-3101049.json) | 0 / 124 | 74932 | 0 | 512 |
| [c5972rect-20261123-185833-3105904](runs/c5972rect-20261123-185833-3105904.json) | 0 / 124 | 72889 | 0 | 512 |
| [c5972slot0-20261123-194505-3395715](runs/c5972slot0-20261123-194505-3395715.json) | 0 / 124 | 65429 | 0 | 256 |
| [c5972vsin-20261123-190831-3168435](runs/c5972vsin-20261123-190831-3168435.json) | 0 / 124 | 66039 | 0 | 256 |
| [c5972vsin2-20261123-191445-3212197](runs/c5972vsin2-20261123-191445-3212197.json) | 0 / 124 | 67102 | 0 | 256 |
| [c5972vsin3-20261123-192313-3267333](runs/c5972vsin3-20261123-192313-3267333.json) | 0 / 124 | 67102 | 0 | 256 |
| [f4fe-fu-20261123-163827-2246976](runs/f4fe-fu-20261123-163827-2246976.json) | 0 / 3 | 53758 | 0 | 1 |
| [f4fe-fu-20261123-172400-2583922](runs/f4fe-fu-20261123-172400-2583922.json) | 0 / 3 | 53757 | 0 | 1 |
| [f4fe-fu-20261123-173135-2625721](runs/f4fe-fu-20261123-173135-2625721.json) | 0 / 3 | 53754 | 0 | 1 |
| [f5972-fu-20261123-173559-2663804](runs/f5972-fu-20261123-173559-2663804.json) | 0 / 124 | 67999 | 0 | 256 |
| [ladder-ffc6f6da0-20261123-134259-1213959](runs/ladder-ffc6f6da0-20261123-134259-1213959.json) | 0 / 3 | 1656 | 0 | 0 |

## Synthetic suites

| Saved suite | Passed | Failed | Skipped |
|---|---:|---:|---:|
| [claude--logs--c3h-bvh-90830ffbe--build](suites/claude--logs--c3h-bvh-90830ffbe--build.json) | 0 | 0 | 0 |
| [claude--logs--c3h-bvh-90830ffbe--corpus](suites/claude--logs--c3h-bvh-90830ffbe--corpus.json) | 1 | 0 | 0 |
| [claude--logs--c3h-bvh-90830ffbe--ctest](suites/claude--logs--c3h-bvh-90830ffbe--ctest.json) | 39 | 4 | 0 |
| [claude--logs--c3h-bvh-control12-codex-20261123-135632--ctest](suites/claude--logs--c3h-bvh-control12-codex-20261123-135632--ctest.json) | 2 | 10 | 0 |
| [claude--logs--c3i-7163-084c8c60e--ctest](suites/claude--logs--c3i-7163-084c8c60e--ctest.json) | 10 | 0 | 0 |
| [claude--logs--c3j-7330-efc347830--ctest](suites/claude--logs--c3j-7330-efc347830--ctest.json) | 11 | 1 | 0 |
| [claude--logs--c3k-7115-e22994464--build](suites/claude--logs--c3k-7115-e22994464--build.json) | 0 | 0 | 0 |
| [claude--logs--c3k-7115-e22994464--ctest](suites/claude--logs--c3k-7115-e22994464--ctest.json) | 40 | 5 | 0 |
| [claude--logs--c3l-7323-a30e78ee7--build](suites/claude--logs--c3l-7323-a30e78ee7--build.json) | 0 | 0 | 0 |
| [claude--logs--c3l-7323-a30e78ee7--ctest](suites/claude--logs--c3l-7323-a30e78ee7--ctest.json) | 41 | 5 | 0 |
| [claude--logs--c3m-6891-23060e087--build](suites/claude--logs--c3m-6891-23060e087--build.json) | 0 | 0 | 0 |
| [claude--logs--c3m-6891-23060e087--ctest](suites/claude--logs--c3m-6891-23060e087--ctest.json) | 46 | 5 | 0 |
| [claude--logs--c3n-7115v2-8f756ebe0--build](suites/claude--logs--c3n-7115v2-8f756ebe0--build.json) | 0 | 0 | 0 |
| [claude--logs--c3n-7115v2-8f756ebe0--ctest](suites/claude--logs--c3n-7115v2-8f756ebe0--ctest.json) | 46 | 5 | 0 |
| [claude--logs--c3o-7201-8517--build](suites/claude--logs--c3o-7201-8517--build.json) | 0 | 0 | 0 |
| [claude--logs--c3o-7201-8517--ctest](suites/claude--logs--c3o-7201-8517--ctest.json) | 46 | 5 | 0 |
| [claude--logs--c3p-7529--build](suites/claude--logs--c3p-7529--build.json) | 0 | 0 | 0 |
| [claude--logs--c3p-7529--ctest](suites/claude--logs--c3p-7529--ctest.json) | 45 | 5 | 1 |
| [claude--logs--c3p-7529--library_shaders](suites/claude--logs--c3p-7529--library_shaders.json) | 0 | 0 | 0 |
| [claude--logs--c3p-7529--pytests](suites/claude--logs--c3p-7529--pytests.json) | 0 | 0 | 0 |
| [claude--logs--c3q-7746--build](suites/claude--logs--c3q-7746--build.json) | 0 | 0 | 0 |
| [claude--logs--c3q-7746--ctest](suites/claude--logs--c3q-7746--ctest.json) | 45 | 5 | 1 |
| [claude--logs--c3q-7746--exe_shaders](suites/claude--logs--c3q-7746--exe_shaders.json) | 0 | 0 | 0 |
| [claude--logs--c3q-7746--pytests](suites/claude--logs--c3q-7746--pytests.json) | 0 | 0 | 0 |
| [claude--logs--c3r-7880--build](suites/claude--logs--c3r-7880--build.json) | 0 | 0 | 0 |
| [claude--logs--c3r-7880--ctest](suites/claude--logs--c3r-7880--ctest.json) | 45 | 5 | 1 |
| [claude--logs--c3s-2490reset--build](suites/claude--logs--c3s-2490reset--build.json) | 0 | 0 | 0 |
| [claude--logs--c3s-2490reset--ctest](suites/claude--logs--c3s-2490reset--ctest.json) | 45 | 5 | 1 |
| [claude--logs--c5972cap--build](suites/claude--logs--c5972cap--build.json) | 0 | 0 | 0 |
| [claude--logs--c5972cap--ctest](suites/claude--logs--c5972cap--ctest.json) | 64 | 5 | 1 |
| [claude--logs--c5972diag--build](suites/claude--logs--c5972diag--build.json) | 0 | 0 | 0 |
| [claude--logs--c5972diag--ctest](suites/claude--logs--c5972diag--ctest.json) | 64 | 5 | 1 |
| [claude--logs--c5972diag2--build](suites/claude--logs--c5972diag2--build.json) | 0 | 0 | 0 |
| [claude--logs--c5972diag2--ctest](suites/claude--logs--c5972diag2--ctest.json) | 64 | 5 | 1 |
| [claude--logs--c5972layout--build](suites/claude--logs--c5972layout--build.json) | 0 | 0 | 0 |
| [claude--logs--c5972layout--ctest](suites/claude--logs--c5972layout--ctest.json) | 64 | 5 | 1 |
| [claude--logs--c5972pull--build](suites/claude--logs--c5972pull--build.json) | 0 | 0 | 0 |
| [claude--logs--c5972pull--ctest](suites/claude--logs--c5972pull--ctest.json) | 67 | 5 | 1 |
| [claude--logs--c5972pullmirror--ctest](suites/claude--logs--c5972pullmirror--ctest.json) | 67 | 5 | 1 |
| [claude--logs--c5972rect--build](suites/claude--logs--c5972rect--build.json) | 0 | 0 | 0 |
| [claude--logs--c5972rect--ctest](suites/claude--logs--c5972rect--ctest.json) | 64 | 5 | 1 |
| [claude--logs--c5972slot0--build](suites/claude--logs--c5972slot0--build.json) | 0 | 0 | 0 |
| [claude--logs--c5972slot0--ctest](suites/claude--logs--c5972slot0--ctest.json) | 64 | 5 | 1 |
| [claude--logs--c5972vsin--build](suites/claude--logs--c5972vsin--build.json) | 0 | 0 | 0 |
| [claude--logs--c5972vsin--ctest](suites/claude--logs--c5972vsin--ctest.json) | 64 | 5 | 1 |
| [claude--logs--c5972vsin2--build](suites/claude--logs--c5972vsin2--build.json) | 0 | 0 | 0 |
| [claude--logs--c5972vsin2--ctest](suites/claude--logs--c5972vsin2--ctest.json) | 64 | 5 | 1 |
| [claude--logs--c5972vsin3--build](suites/claude--logs--c5972vsin3--build.json) | 0 | 0 | 0 |
| [claude--logs--c5972vsin3--ctest](suites/claude--logs--c5972vsin3--ctest.json) | 64 | 5 | 1 |
| [claude--logs--claude-boot-v2-c1-6582-e86f4b4ba--build](suites/claude--logs--claude-boot-v2-c1-6582-e86f4b4ba--build.json) | 0 | 0 | 0 |
| [claude--logs--claude-boot-v2-c1-6582-e86f4b4ba--ctest](suites/claude--logs--claude-boot-v2-c1-6582-e86f4b4ba--ctest.json) | 25 | 2 | 0 |
| [claude--logs--claude-boot-v2-c2-4348-db8a4a1e4--build](suites/claude--logs--claude-boot-v2-c2-4348-db8a4a1e4--build.json) | 0 | 0 | 0 |
| [claude--logs--claude-boot-v2-c2-4348-db8a4a1e4--ctest](suites/claude--logs--claude-boot-v2-c2-4348-db8a4a1e4--ctest.json) | 26 | 3 | 0 |
| [claude--logs--claude-boot-v2-c2a-7216-ea2a0bf26--build](suites/claude--logs--claude-boot-v2-c2a-7216-ea2a0bf26--build.json) | 0 | 0 | 0 |
| [claude--logs--claude-boot-v2-c2a-7216-ea2a0bf26--ctest](suites/claude--logs--claude-boot-v2-c2a-7216-ea2a0bf26--ctest.json) | 27 | 4 | 0 |
| [claude--logs--claude-boot-v2-c3-devabi-ca5344c9a--build](suites/claude--logs--claude-boot-v2-c3-devabi-ca5344c9a--build.json) | 0 | 0 | 0 |
| [claude--logs--claude-boot-v2-c3-devabi-ca5344c9a--ctest](suites/claude--logs--claude-boot-v2-c3-devabi-ca5344c9a--ctest.json) | 38 | 5 | 0 |
| [claude--logs--claude-boot-v2-c3b-diag-58950fce1--build](suites/claude--logs--claude-boot-v2-c3b-diag-58950fce1--build.json) | 0 | 0 | 0 |
| [claude--logs--claude-boot-v2-c3b-diag-58950fce1--ctest](suites/claude--logs--claude-boot-v2-c3b-diag-58950fce1--ctest.json) | 1 | 0 | 0 |
| [claude--logs--claude-boot-v2-c3b-diag-a6b6ac35c--build](suites/claude--logs--claude-boot-v2-c3b-diag-a6b6ac35c--build.json) | 0 | 0 | 0 |
| [claude--logs--claude-boot-v2-c3b-diag-a6b6ac35c--ctest](suites/claude--logs--claude-boot-v2-c3b-diag-a6b6ac35c--ctest.json) | 1 | 0 | 0 |
| [claude--logs--claude-boot-v2-c3c-diag-aacf08f51--build](suites/claude--logs--claude-boot-v2-c3c-diag-aacf08f51--build.json) | 0 | 0 | 0 |
| [claude--logs--claude-boot-v2-c3c-diag-aacf08f51--ctest](suites/claude--logs--claude-boot-v2-c3c-diag-aacf08f51--ctest.json) | 35 | 4 | 0 |
| [claude--logs--claude-boot-v2-c3d-diag-aca2c62ed--build](suites/claude--logs--claude-boot-v2-c3d-diag-aca2c62ed--build.json) | 0 | 0 | 0 |
| [claude--logs--claude-boot-v2-c3d-diag-aca2c62ed--ctest](suites/claude--logs--claude-boot-v2-c3d-diag-aca2c62ed--ctest.json) | 36 | 4 | 0 |
| [claude--logs--claude-boot-v2-c3e-diag-03034f8bc--build](suites/claude--logs--claude-boot-v2-c3e-diag-03034f8bc--build.json) | 0 | 0 | 0 |
| [claude--logs--claude-boot-v2-c3e-diag-03034f8bc--ctest](suites/claude--logs--claude-boot-v2-c3e-diag-03034f8bc--ctest.json) | 36 | 4 | 0 |
| [claude--logs--claude-boot-v2-c3f-diag-6f2ebe266--build](suites/claude--logs--claude-boot-v2-c3f-diag-6f2ebe266--build.json) | 0 | 0 | 0 |
| [claude--logs--claude-boot-v2-c3f-diag-6f2ebe266--ctest](suites/claude--logs--claude-boot-v2-c3f-diag-6f2ebe266--ctest.json) | 36 | 4 | 0 |
| [claude--logs--claude-boot-v2-c3g-diag-13c446509--build](suites/claude--logs--claude-boot-v2-c3g-diag-13c446509--build.json) | 0 | 0 | 0 |
| [claude--logs--claude-boot-v2-c3g-diag-13c446509--ctest](suites/claude--logs--claude-boot-v2-c3g-diag-13c446509--ctest.json) | 36 | 4 | 0 |
| [claude--logs--claude-boot-v2-c3h-diag-90830ffbe--build](suites/claude--logs--claude-boot-v2-c3h-diag-90830ffbe--build.json) | 0 | 0 | 0 |
| [claude--logs--claude-boot-v2-c3h-diag-90830ffbe--ctest](suites/claude--logs--claude-boot-v2-c3h-diag-90830ffbe--ctest.json) | 36 | 4 | 0 |
| [claude--logs--claude-tls-ab-1d15a1a16--build-full](suites/claude--logs--claude-tls-ab-1d15a1a16--build-full.json) | 0 | 0 | 0 |
| [claude--logs--claude-tls-ab-1d15a1a16--ctest-prelim-with-widening](suites/claude--logs--claude-tls-ab-1d15a1a16--ctest-prelim-with-widening.json) | 16 | 3 | 0 |
| [claude--logs--claude-tls-ab-1d15a1a16--ctest](suites/claude--logs--claude-tls-ab-1d15a1a16--ctest.json) | 17 | 2 | 0 |
| [claude--logs--claude-tls-ab-guards-e7839b706--build](suites/claude--logs--claude-tls-ab-guards-e7839b706--build.json) | 0 | 0 | 0 |
| [claude--logs--claude-tls-ab-guards-e7839b706--ctest](suites/claude--logs--claude-tls-ab-guards-e7839b706--ctest.json) | 27 | 2 | 0 |
| [claude--logs--f4fe-fu--build](suites/claude--logs--f4fe-fu--build.json) | 0 | 0 | 0 |
| [claude--logs--f4fe-fu--ctest-dstorage-alone](suites/claude--logs--f4fe-fu--ctest-dstorage-alone.json) | 0 | 1 | 0 |
| [claude--logs--f4fe-fu--ctest-rest-headless](suites/claude--logs--f4fe-fu--ctest-rest-headless.json) | 190 | 14 | 29 |
| [claude--logs--f4fe-fu--ctest](suites/claude--logs--f4fe-fu--ctest.json) | 55 | 5 | 1 |
| [claude--logs--f5972-fu--build](suites/claude--logs--f5972-fu--build.json) | 0 | 0 | 0 |
| [claude--logs--f5972-fu--ctest](suites/claude--logs--f5972-fu--ctest.json) | 63 | 5 | 1 |
| [claude--logs--ladder-ffc6f6da0--build](suites/claude--logs--ladder-ffc6f6da0--build.json) | 0 | 0 | 0 |
| [claude--logs--ladder-ffc6f6da0--ctest](suites/claude--logs--ladder-ffc6f6da0--ctest.json) | 222 | 18 | 30 |
| [claude--logs--shaders-20261123-123338](suites/claude--logs--shaders-20261123-123338.json) | 0 | 0 | 0 |
| [claude--work-gap-w32-8448--tx-8448-negctl-tests](suites/claude--work-gap-w32-8448--tx-8448-negctl-tests.json) | 0 | 0 | 0 |
| [claude--work-gap-w32-8448--tx-8448-tests](suites/claude--work-gap-w32-8448--tx-8448-tests.json) | 0 | 0 | 0 |
| [claude--work-gapfix5--tx-test_translate](suites/claude--work-gapfix5--tx-test_translate.json) | 0 | 0 | 0 |
| [claude--work-gapfix5--tx-tests](suites/claude--work-gapfix5--tx-tests.json) | 0 | 0 | 0 |
| [logs--development-launcher-tests](suites/logs--development-launcher-tests.json) | 5 | 0 | 0 |
| [logs--tests-main-8575d36](suites/logs--tests-main-8575d36.json) | 5 | 0 | 0 |
| [logs--tests-pr6919-36936f3](suites/logs--tests-pr6919-36936f3.json) | 5 | 0 | 0 |

## Historical reports

- [report-4313.txt](reports/report-4313.txt)
- [report-6582.txt](reports/report-6582.txt)
- [report-6739-runtime-bvh-validation.txt](reports/report-6739-runtime-bvh-validation.txt)
- [report-6891-c3m-result.txt](reports/report-6891-c3m-result.txt)
- [report-6919.txt](reports/report-6919.txt)
- [report-7052-c3e-result.txt](reports/report-7052-c3e-result.txt)
- [report-7066-c3d-result.txt](reports/report-7066-c3d-result.txt)
- [report-7075-c3c-result.txt](reports/report-7075-c3c-result.txt)
- [report-7115-c3k-result.txt](reports/report-7115-c3k-result.txt)
- [report-7115-c3n-recheck.txt](reports/report-7115-c3n-recheck.txt)
- [report-7115-host-regression-ack.txt](reports/report-7115-host-regression-ack.txt)
- [report-7154-bvh-source-answer.txt](reports/report-7154-bvh-source-answer.txt)
- [report-7154-c1c2.txt](reports/report-7154-c1c2.txt)
- [report-7154-c2a.txt](reports/report-7154-c2a.txt)
- [report-7154-navi31-startup-answer.txt](reports/report-7154-navi31-startup-answer.txt)
- [report-7154-tls.txt](reports/report-7154-tls.txt)
- [report-7154.txt](reports/report-7154.txt)
- [report-7216.txt](reports/report-7216.txt)
- [report-7235-c3f-result.txt](reports/report-7235-c3f-result.txt)
- [report-7235-c3g-diagnostic.txt](reports/report-7235-c3g-diagnostic.txt)
- [report-7240-c3h-diagnostic.txt](reports/report-7240-c3h-diagnostic.txt)
- [report-7323-c3l-result.txt](reports/report-7323-c3l-result.txt)
- [report-7330-wine-comparison.txt](reports/report-7330-wine-comparison.txt)
- [report-7492-cache-review.txt](reports/report-7492-cache-review.txt)
- [report-7492-guard-resolved.txt](reports/report-7492-guard-resolved.txt)
- [report-7496-c3b-status.txt](reports/report-7496-c3b-status.txt)
- [report-7529-c3p-result.txt](reports/report-7529-c3p-result.txt)
- [report-7584-packed-span-reviewed.txt](reports/report-7584-packed-span-reviewed.txt)
- [report-7584-span-publication-review.txt](reports/report-7584-span-publication-review.txt)
- [report-7584-split-handler-scope.txt](reports/report-7584-split-handler-scope.txt)
- [report-7746-c3q-result.txt](reports/report-7746-c3q-result.txt)
- [report-7905-wine-baseline.txt](reports/report-7905-wine-baseline.txt)
- [report-7913-shader-status.txt](reports/report-7913-shader-status.txt)
- [report-8344-existing-queue-route.txt](reports/report-8344-existing-queue-route.txt)
- [report-8517-c3o-result.txt](reports/report-8517-c3o-result.txt)
- [report-8517-linux-coverage-question.txt](reports/report-8517-linux-coverage-question.txt)
- [report-8851-7323-review-handoff.txt](reports/report-8851-7323-review-handoff.txt)
- [report-9403-fspx-format-review.txt](reports/report-9403-fspx-format-review.txt)
- [report-9403-scope-reviewed.txt](reports/report-9403-scope-reviewed.txt)
- [report-cache-coordination.txt](reports/report-cache-coordination.txt)
- [report-coordination-continuation.txt](reports/report-coordination-continuation.txt)
- [report-coordination.txt](reports/report-coordination.txt)
- [report-forge-source-review.txt](reports/report-forge-source-review.txt)
- [report-frontier-handoff-evidence.txt](reports/report-frontier-handoff-evidence.txt)
- [report-frontier-submit-correction.txt](reports/report-frontier-submit-correction.txt)
- [report-slot108-existing-followup.txt](reports/report-slot108-existing-followup.txt)
- [report-7584-cef56-synthetic-result.txt](reports/report-7584-cef56-synthetic-result.txt)
- [report-7880-reset-isolation.txt](reports/report-7880-reset-isolation.txt)
- [report-8448-wave32-baseline.txt](reports/report-8448-wave32-baseline.txt)
- [report-8526-five-gap-validation.txt](reports/report-8526-five-gap-validation.txt)
- [report-8526-runtime-gapfix.txt](reports/report-8526-runtime-gapfix.txt)
- [report-8526-runtime-gapfix7.txt](reports/report-8526-runtime-gapfix7.txt)
- [report-8851-a50-7021-import.txt](reports/report-8851-a50-7021-import.txt)
- [report-8851-c5972cap.txt](reports/report-8851-c5972cap.txt)
- [report-8851-c5972diag.txt](reports/report-8851-c5972diag.txt)
- [report-8851-c5972diag2.txt](reports/report-8851-c5972diag2.txt)
- [report-8851-c5972rect.txt](reports/report-8851-c5972rect.txt)
- [report-8851-composite-layout.txt](reports/report-8851-composite-layout.txt)
- [report-8851-composite-table.txt](reports/report-8851-composite-table.txt)
- [report-8851-f5972-runtime.txt](reports/report-8851-f5972-runtime.txt)
- [report-8851-post-refresh-watches.txt](reports/report-8851-post-refresh-watches.txt)
- [report-8851-probe-pair.txt](reports/report-8851-probe-pair.txt)
- [report-8851-record-inputs.txt](reports/report-8851-record-inputs.txt)
- [report-8851-tagged-emission.txt](reports/report-8851-tagged-emission.txt)
- [report-9403-captured-corpus-count.txt](reports/report-9403-captured-corpus-count.txt)
- [report-9403-corpus-followup.txt](reports/report-9403-corpus-followup.txt)
- [report-9677-shader-scope-slot41.txt](reports/report-9677-shader-scope-slot41.txt)
- [report-9677-test1674-reset-status.txt](reports/report-9677-test1674-reset-status.txt)
- [report-f4fe-fu-result.txt](reports/report-f4fe-fu-result.txt)
- [report-frontier-7584-synthetic.txt](reports/report-frontier-7584-synthetic.txt)

## Additional recovered evidence

- [Pull/mirror comparison and newly failing WARP assertion](reports/pullmirror-comparison.txt).
- [Historical community test references](community-evidence-index.json): 1,827 cached public comment references, with author, date and count fragments where present. These fragments are incomplete context, not independently verified test results; raw comment bodies and attachments are not republished.
