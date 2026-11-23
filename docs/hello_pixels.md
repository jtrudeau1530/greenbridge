# Hello pixels: a package-free draw through vk_harness

**Status: measured** on Mesa llvmpipe (LLVM 20.1.2, Vulkan 1.3.275, Linux). This is a synthetic complement to
[first_light.md](first_light.md): where that page draws a real game model with real translated shaders, this one
draws a quad built entirely in the repository — hand-written SPIR-V assembly, a generated mesh — so anyone with
a Vulkan device can check the pipeline without the game package.

## What it draws

`tools/hello_pixels.py --run --check` assembles two tiny shaders (no GLSL compiler needed, `spirv-as` only):

- **VS** (`vs.asm`): `Location 0` float3 position already in NDC, `Location 1` float2 uv; stores the position
  into `gl_Position` and forwards uv to the interpolating varying. This is the same shape as the first_light VS.
- **PS** (`ps.asm`): `rgb = (u, v, (u+v)/2)` dimmed to 20% on every other cell of a 16×16 checker, computed with
  core instructions only (`OpFMod` cell coordinate, `OpFOrdLessThan`/`OpSelect` parity). `OriginUpperLeft`.

The mesh is a generated fullscreen indexed quad (`u32` counts, `5f` vertices, `u32` indices — the
`VKH_DRAW_MESH` format with `VKH_VERTEX_LAYOUT=0:3f,1:2f`), drawn into the RGBA32F attachment, read back, and
written as `image.raw`, `image.ppm` and a zlib-only `image.png`.

## What --check verifies

Same style as the first_light oracle, so the image has to be *right*, not just non-clear:

- the buffer is exactly `w*h*16` bytes of RGBA32F;
- coverage is 100% — a quad that misses the image fails, and so does an all-clear image;
- seven probe pixels match `expected_pixel`, the closed form of the same pattern (u, v, (u+v)/2 with cell
  parity) computed at pixel centres, within 0.04. A flipped axis, a swapped u/v or a mirrored checker moves
  probe values out of tolerance.

`tests/test_hello_pixels.py` runs the oracle on synthetic buffers without a GPU or vk_harness: the accepted
image, a truncated buffer, a wrong probe pixel and a coverage loss away from every probe.

## Why it matters

first_light needs `shaders.rpf`, `data.rpf` and an OBJ export. This needs nothing but the built harness and
`spirv-as`, so it is the fastest way to tell a fresh machine "your Vulkan path prints pixels": vertex fetch →
interpolation → fragment → readback, exercised end to end. It also serves as the smallest reproducer for
VKH_DRAW_MESH bugs — the file format and layout string are spelled out in one generated mesh.

## Reproduce it

```sh
cmake -S runtime -B runtime/build -G Ninja
cmake --build runtime/build --target vk_harness
python -I tools/hello_pixels.py --work <dir> --size 512 --run --check
# <dir>/image.png: a u/v gradient dimmed on alternating 16x16 cells
```

On Windows the same command works against `runtime/build/windows/vk_harness.exe`; on a machine without
`spirv-as`, drop the `vs.spv`/`ps.spv` that `--emit-only` produced on a machine that has one into the `--work`
directory; the tool uses the prebuilt pair when `spirv-as` is absent (fresh assembly always wins when it is
present).
