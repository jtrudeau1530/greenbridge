# GPU pipeline corpus check

`runtime/test/vk_pipeline_check.cpp` creates a `VkShaderModule` for every entry in the translator's
`index.json` and, for compute shaders, a real compute pipeline with the runtime's descriptor and push
constant contract. It is a driver-level check: it proves the modules and pipelines are accepted by
the Vulkan implementation, not that the shader results are correct. Value-level checks are in
`docs/shader_validation.md`.

The checker exits nonzero if a module fails to load, a pipeline fails to compile, or the requested final
cache save fails. It still finishes checking the other entries and attempts the final save so the next run
can reuse successful pipelines. Entries reported as skipped do not count as failures.

## Corpus

The translated corpus is 26,503 entries: 2,236 vertex, 12,101 pixel, 2,689 compute, 9,282 mesh and
195 geometry shaders. A run was split into 12 contiguous index shards of about 2,209 entries each.
The shards were run with a per-shard pipeline cache; each saves a cache of roughly 190-203 MB.

| Shard | Modules | CS pipelines | Result | Cache | Wall |
|---|---:|---:|---|---:|---:|
| s2 | 2,207 | 208 | pass | 196,869,933 B | 2h21m35s |
| s3 | 2,209 | 200 | pass | 190,022,273 B | 1h50m54s |
| s4 | 2,210 | 213 | pass | 198,941,511 B | 2h01m55s |
| s5 | 2,209 | 248 | pass | 201,997,254 B | 2h01m09s |
| s6 | 2,209 | 213 | pass | 190,939,966 B | 2h00m05s |
| s7 | 2,209 | 229 | pass | 202,013,032 B | 1h59m27s |
| s8 | 2,210 | 227 | pass | 199,121,775 B | 2h00m11s |
| s9 | 2,210 | 240 | pass | 198,407,685 B | 2h00m27s |
| s10 | 2,210 | 218 | pass | 202,890,354 B | 2h01m58s |
| s11 | 2,209 | 221 | pass | 200,378,101 B | 2h01m42s |
| **total (10/12)** | **22,092** | **2,217** | **0 module / 0 pipeline failures** | **1,981,581,884 B** | |

Stage split over the ten shards: 2,217 compute, 163 geometry, 7,767 mesh, 10,088 pixel and 1,857
vertex modules. The ten shards cover **22,092 of 26,503 entries (83.4%)** and **2,217 of 2,689
compute pipelines (82.4%)**.

Two shards did not complete:

- **s0** was stopped after 1 h 57 m at 81% of its bytes, with no read progress for 27 minutes on a
  3.61 MB compute shader.
- **s1** was stopped at the same time at 83% of its bytes on a 2.52 MB pixel shader.

They account for the missing 4,411 modules (472 compute, 32 geometry, 1,515 mesh, 2,013 pixel, 379
vertex) and 472 compute pipelines. No summary was printed for either shard. This is a driver
compile-time limit, not a module or pipeline rejection: no `MODULE FAIL` or `PIPELINE FAIL` line
exists in any completed shard.

A later full-corpus build on different hardware (RTX 4060 Ti, driver 595) completed **26,503 of 26,503**
modules and pipelines with **0 failures** and a 3.81 GB cache; see the "Full-corpus pipeline build" row in
[`docs/design.md`](design.md). The two stalled shards above therefore record a hardware/driver-specific
compile-time stall, not a project-wide limit.

The recurring slow family at every shard tail was a set of 6.7-7.2 MB ray-tracing resolve compute
shaders. One observed single-module stall exceeded 27 minutes. The tool has no per-shader timer, so
these are lower bounds, not exact compile times.

## Bounded guaranteed-finish sample

A separate bounded run covers eight 30-entry bins plus 12 large compute shaders: **252/252 modules
accepted** and **172/172 compute pipelines accepted**, 0 failures. Stage split: 172 compute, 20 each
geometry/mesh/pixel/vertex. Wave split: 97 wave64 / 75 wave32. The nine processes finished in about
5 minutes; their caches total 42,098,592 bytes. This is a sanity check on the same device, descriptor
layout and subgroup path, not a substitute for the 26,503-entry corpus.

A validation-layer run of the same 252 shaders (temporary instrumented copy, not committed) produced
**zero shader/module/pipeline messages**. The 9 errors and 36 warnings were loader noise from
overlay layers (a missing Epic JSON, three non-conforming layer names, one duplicate OBS hook). The
22,092 full-shard modules were not re-run through the validation layer; their result is driver
acceptance only.

## Skipping huge modules

`vk_pipeline_check` now accepts `--max-bytes N`:

```powershell
vk_pipeline_check <spirv dir> [cache.bin] --max-bytes 2000000 --count 100
```

Every `.spv` larger than `N` is reported as `BIG SKIP <name> (<bytes>, limit <N>)`, counted by stage,
and excluded from the module/pipeline totals; the summary prints how many entries were checked. On a
30-entry smoke test with `--max-bytes 20000`, 24 modules were skipped and the remaining 6 pixel
modules passed.

The report above therefore states the limit and the skip count; a later run on an RTX 4060 Ti / 595
completed the full 26,503-entry set without them (`docs/design.md`). `--max-bytes` remains a visibility
tool, not a pass: a skipped module has not been validated.

## Application pipeline-cache feedback

With `--cache-feedback`, the checker groups successful pipeline creations by translated shader stage
and reports three counts. A **hit** means the driver set both the valid-feedback bit and the
application-cache-hit bit. **No hit reported** means feedback was valid and the hit bit was clear.
**Unavailable** means the driver returned no valid feedback, or feedback was unsupported.
The three counts sum to the successful pipelines for the stage. Skips and failures are counted separately.

A driver can also use its own caches. A clear hit bit does not establish a fresh compile, and a
reported hit may reuse data populated earlier in the same process. The seed-file byte count alone
cannot identify which pipelines benefited.

To compare runs, start with a new cache path and keep the device, index and flags identical, including
`--stats` if used. For example, from a scratch directory that contains no `sample.cache`, run the same
command twice, replacing `path/to/spirv` with the local translated-shader directory.

```sh
vk_pipeline_check path/to/spirv sample.cache --cache-feedback
vk_pipeline_check path/to/spirv sample.cache --cache-feedback
```

On Windows with an RTX 5090 (driver 621.14), a synthetic three-compute, one-vertex and one-pixel
sample reported zero application-cache hits on its first run and five on the seeded run. On Mesa
llvmpipe the same five pipelines succeeded both times and neither run reported a hit.

## Device

The reported runs used an NVIDIA GeForce RTX 5070 Ti (discrete, subgroup size 32 only), driver
621.24 / Windows 32.0.16.2124, Vulkan 1.4.357, loader 1.4.363, with 28 logical CPUs and 34 GB RAM.
The executable used the Vulkan SDK at `C:\VulkanSDK\1.4.363.0`; no source change was needed to build
or run it.

## What this does and does not prove

It proves, for the ten completed shards, that every translated module is accepted and every compute
pipeline is created with the runtime's resource contract. It does not prove:

- that any shader computes the right values (see `docs/shader_validation.md`);
- that the two incomplete shards are valid or invalid;
- that the graphics pipelines (vertex/pixel/geometry/mesh) execute or draw the right pixels: the tool
  creates them with `vkCreateGraphicsPipelines` (dynamic rendering) but never draws with them;
- that the game's descriptors, root signatures and push constants match the contract at run time
  (the missing executable still owns that).

No shader data, names or caches are committed. The caches live in the work directory and are inputs
to a later run, not part of the repository.
