# Documentation

These notes record what has been measured in the local Xbox Series package and what is still open.
The headline counts are the measured numbers at the time of the note; the linked page has the
method, scope and caveats. No game data is included.

## Start here

| Document | What it covers |
|---|---|
| [design.md](design.md) | Overall project design, asset status, translator contract and open blockers. |
| [related_projects.md](related_projects.md) | Prior art we may read and the licence boundaries on it. |
| [rpf.md](rpf.md) | RPF9 archive layout, the carve tools and the unreadable encrypted TOC. |

## Archives and assets

| Document | Measured content |
|---|---|
| [models.md](models.md) | 155,598 models exported to OBJ with bounds, shader index and material keys. |
| [textures.md](textures.md) | 308,895 TTPF textures exported; 48,220 skips, mostly BCPack. |
| [animation.md](animation.md) | 130,470 `anim.rpf` streams, 95.4% readable; 1,126 rig sets / 6,677 source-to-rig entries / 3,669 rigs. |
| [navmesh.md](navmesh.md) | 90,295 `navmeshes.rpf` streams, 97.5% readable; `0DRP` pages, no navmesh-specific magic. |
| [bvh.md](bvh.md) | 565,596 `core_bvh4.rpf` streams; 271,627 BVH4 streams and 269,106 instances, prebuilt acceleration data. |
| [rbf0.md](rbf0.md) | 1,263 RBF0 resources, 475,461 elements; UI/Scaleform definitions, not a sectioned container. |
| [gamedata.md](gamedata.md) | `common.rpf`: 954 streams, 266 XML documents / 118 roots; 0 RBF0/PSIN. |
| [world.md](world.md) | `levels.rpf` placement: 794 map files / 82,463 entities in the sampled windows; `0x7a` resource layout. |
| [psin.md](psin.md) | PSIN metadata container: header, section walk, 671 streams in the `script.rpf` carve. |
| [movies.md](movies.md) | Bink 2 streams and audio tracks; decode through the real loader and `Bink2Scarlett.dll`. |

## Audio

| Document | Measured content |
|---|---|
| [audio.md](audio.md) | AWC banks: Opus, PCM and XMA2; lossless Ogg Opus repack and WAV paths. |
| [audio_config.md](audio_config.md) | The music/radio record table: 1,589 records, exact durations, AWC stream ids; no start offsets. |

## Runtime, GPU and validation

| Document | Measured content |
|---|---|
| [d3d12_vulkan.md](d3d12_vulkan.md) | Xbox D3D12 to Vulkan plan and status; device proxy, queues, command lists, barriers, copies. |
| [d3d12_xs_surface.md](d3d12_xs_surface.md) | The 116-slot `d3d12_xs.dll` export surface; 83/83 standard GUIDs match the PC SDK. |
| [runtime_imports.md](runtime_imports.md) | The 16 shipped modules' measured imports and shim/host/unimplemented split. |
| [pipeline_check.md](pipeline_check.md) | Driver-level corpus check: 10/12 shards, 22,092 modules, 2,217 compute pipelines, 0 failures. |
| [shader_validation.md](shader_validation.md) | Differential CPU/GPU validation design and the current CPU interpreter prototype. |
| [fssm_crosscheck.md](fssm_crosscheck.md) | FSSM/RDNA2 facts cross-checked against SCARAB; code starts at `code_off + 0x28`. |
