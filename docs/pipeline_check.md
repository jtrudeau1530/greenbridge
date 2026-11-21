# GPU pipeline corpus check

`runtime/test/vk_pipeline_check.cpp` creates a `VkShaderModule` for every entry in the translator's
`index.json` and, for compute shaders, a real compute pipeline with the runtime's descriptor and push
constant contract. It is a driver-level check: it proves the modules and pipelines are accepted by
the Vulkan implementation, not that the shader results are correct. Value-level checks are in
`docs/shader_validation.md`.

## Corpus

The translated corpus is 26,503 entries: 2,236 vertex, 12,101 pixel, 2,689 compute, 9,282 mesh and
195 geometry shaders. A run was split into 12 contiguous index shards of about 2,209 entries each.
The shards were run with the per-shard pipeline cache enabled; each shard saves a cache of roughly
190-203 MB.

| Shard | Modules | CS pipelines | Result | Cache |
|---|---:|---:|---|---:|
| s3 | 2,209 | 200 | pass | 190,022,273 B |
| s4 | 2,210 | 213 | pass | 198,941,511 B |
| s5 | 2,209 | 248 | pass | 201,997,254 B |
| s6 | 2,209 | 213 | pass | 190,939,966 B |
| s7 | 2,209 | 229 | pass | 202,013,032 B |
| s8 | 2,210 | 227 | pass | 199,121,775 B |
| s9 | 2,210 | 240 | pass | 198,407,685 B |
| s10 | 2,210 | 218 | pass | 202,890,354 B |
| s11 | 2,209 | 221 | pass | 200,378,101 B |
| **total (9/12)** | **19,885** | **2,009** | **0 module / 0 pipeline failures** | |

Stage split over the nine shards: 2,009 compute, 147 geometry, 7,010 mesh, 9,053 pixel and 1,666
vertex modules. The nine shards cover 19,885 of 26,503 entries (75.0%).

Three shards did not complete:

- **s0** was stopped after 27 minutes with no read progress on a 3.61 MB compute shader.
- **s1** was stopped at its boundary on a 2.52 MB pixel shader, also with no read progress.
- **s2** was still pinned on 6.7-7.2 MB compute modules when the run was stopped; it produced no
  summary.

This is a driver-compilation limit, not a module or pipeline rejection. The tool's per-shader
timeout path, the named in-flight shader and the saved cache make the boundary measurable; the
missing shards are **not** evidence that those shaders are invalid.

## Bounded slices

Eight independent 30-entry slices of the index were run without the huge modules: 240 modules and
160 compute pipelines, all accepted, 0 failures. They exercise the same device, descriptor layout,
subgroup-size path and pipeline creation as the shards. They are a sanity check on the path, not a
substitute for the 26,503-entry corpus.

## Skipping huge modules

`vk_pipeline_check` now accepts `--max-bytes N`:

```powershell
vk_pipeline_check <spirv dir> [cache.bin] --max-bytes 2000000 --count 100
```

Every `.spv` larger than `N` is reported as `BIG SKIP <name> (<bytes>, limit <N>)`, counted by stage,
and excluded from the module/pipeline totals; the summary prints how many entries were checked. On a
30-entry smoke test with `--max-bytes 20000`, 24 modules were skipped and the remaining 6 pixel
modules passed.

A full corpus report should therefore state the limit and the skip count. A run with a 2 MB limit
can finish the 26,503-entry set while naming the handful of huge modules that the driver cannot
complete in a useful time. The option is a visibility tool, not a pass: a skipped module has not
been validated.

## What this does and does not prove

It proves, for the nine completed shards, that every translated module is accepted and every
compute pipeline is created with the runtime's resource contract. It does not prove:

- that any shader computes the right values (see `docs/shader_validation.md`);
- that the three incomplete shards are valid or invalid;
- that the graphics pipelines (vertex/pixel/geometry/mesh) link and execute, since the tool creates
  modules for those stages but only pipelines for compute;
- that the game's descriptors, root signatures and push constants match the contract at run time
  (the missing executable still owns that).

No shader data, names or caches are committed. The caches live in the work directory and are inputs
to a later run, not part of the repository.
