# Differential shader validation

This note describes a value-level check for the RDNA2 → SPIR-V translator; it is not a claim that
the corpus is correct. The CPU differential oracle in `tools/shader_diff.py` is implemented and
tested (`tests/test_shader_diff.py`): it compares the ISA model in `tools/cpu_ref.py` against
translated SPIR-V run in `tests/semantic/spirv_mini_interp.py` over seeded straight-line programs.
Run it with `python -I tools/shader_diff.py --programs 200 --seed 7` (`--self-test` checks the
harness itself). The GPU-side runners are described below: `tools/valu_fuzz.py` (16-bit and memory ops), `tools/lds_fuzz.py`, `tools/crosslane_fuzz.py` and `tools/cf_diff.py`.

## Why `spirv-val` is not enough

`spirv-val` checks that a module is structurally valid SPIR-V for a target environment. It does not
check that the module computes what the RDNA2 program computes. Those are different questions, and
this repository already has cases where only the second question had a useful answer:

- All 26,503 shaders translate and pass `spirv-val` (Vulkan 1.2), yet the SDWA source/destination
  selects were silently ignored until 2026-11-20. That passed validation for 3,826 of the 13,926
  translated shaders and produced wrong values (`docs/design.md`, "SDWA").
- The compressed-export bug was found by a GPU test, not by validation: LLVM prints `v4, v4, v5, v5`
  for a compressed pair, and the translated shader exported `(r, g, r, g)`.
- `v_cvt_pkrtz_f16_f32` with round-to-nearest overflowed a finite value to infinity; the GPU smoke
  test found it, `spirv-val` could not.
- The other project's execution harness found a read-after-store coherence bug: a shader that stored
  a texel and loaded it back got stale data because the read and write used different bindings. That
  is a well-formed, validator-clean pair of instructions with the wrong observable result.

A validator answers "is this a legal module?" A differential test answers "does this module produce
the same bits as the original program for the same inputs?" Only the second one can find a wrong
instruction, a wrong operand order, a missing modifier, or a wrong rounding mode.

## What "differential" means here

Two independent computations start from the same architectural state:

1. **Reference.** A CPU interpreter executes the original RDNA2 instructions and records the
   registers (and, later, memory) the program writes.
2. **System under test.** The translator turns the same listing into SPIR-V; the existing Vulkan test
   tools dispatch it on a real GPU and read back the same registers, buffers or images.

The comparison is bit-for-bit on every value the test declares observable. A mismatch is reported
with the first differing lane/register, both bit patterns, the random seed, and (in the future
runner) the RDNA2 instruction that last wrote the value. The translator is the path under test; the
CPU model and the SPIR-V interpreter are the two independent oracles in the shipped CPU runner. The
translator → `spirv-as`/`spirv-val` → Vulkan tools path is what the future GPU oracle will run, not
a reference to be trusted. A hand-written SPIR-V twin of a test program is a useful third control: it
checks the harness and driver without involving the translator, so a harness failure is not blamed on
translation.

The CPU model is deliberately not a second translator. It interprets the listing's mnemonics and
operands; it never consumes SPIR-V. That independence is what makes a disagreement meaningful.

## The two sides

### CPU interpreter (`tools/cpu_ref.py`, prototype)

- Input is a `.s` listing in the format produced by `tools/shader_disasm.py` and by
  `runtime/test/shaders/build.py`: a header line, then tab-indented instructions with an optional
  `; encoding: [...]` comment per line. `tools/shader_ir.py` parses the listing and treats line 1 as
  the header, so a raw `llvm-mc` dump has to be given the same header `build.py` writes.
- State is an SGPR file (32-bit values), a VGPR file (one 32-bit value per lane), VCC, EXEC, SCC and
  a program counter. Float registers keep raw bits; only float instructions reinterpret them.
  Uninitialized registers read as zero, matching the translator's deterministic register arrays.
- One lane or 64 lanes. Scalar instructions execute once; vector instructions execute for lanes whose
  EXEC bit is set, and inactive lanes keep their old value. EXEC and VCC are 64-bit masks even in
  one-lane mode, so scalar reads of `exec_lo`/`exec_hi` stay meaningful.
- The implemented subset is intentionally small: `s_mov_b32`, `v_mov_b32`, `v_add_f32`, `v_sub_f32`,
  `v_mul_f32`, `v_fma_f32`, `v_cndmask_b32`, `v_and_b32`, `v_or_b32`, `v_xor_b32`,
  `v_lshlrev_b32`, `v_lshrrev_b32`, `v_ashrrev_i32`, `s_add_u32`, `s_lshl_b32`, `s_lshr_b32`,
  `s_endpgm`. Anything else raises `Unsupported` with the mnemonic and PC. It never guesses an
  instruction's meaning.
- f32 add/sub/mul produce correctly rounded binary32 results. FMA is evaluated exactly with rational
  arithmetic and rounded once to binary32, so it cannot inherit a double-rounding error from a host
  `float`.
- NaNs are canonicalized to `0x7fc00000`; the comparison layer should treat all NaNs as equal by
  default because Vulkan does not promise a NaN payload.
- Denormals are preserved. A GPU that flushes f32 denormals will disagree on the denormal cases, and
  the harness must report that as an environment mismatch, not a translator bug.

### GPU side (existing tools, future driver)

The repository already has the pieces: `runtime/test/shaders/build.py` assembles `.asm` with
`llvm-mc`, `tools/rdna2spirv.py` translates the resulting `.s`, and the `runtime/test/vk_*_test.cpp`
programs dispatch SPIR-V on Vulkan and read buffers/images back. A GPU-side differential runner would:

1. assemble the `.asm` to a `.s` listing with `llvm-mc -show-encoding`;
2. run the CPU model on that listing with the test's inputs;
3. translate the same listing, `spirv-as` + `spirv-val`, then dispatch;
4. read back the declared outputs and compare raw bits;
5. print the first mismatch and the seed; exit nonzero on any mismatch.

Keeping one listing for both sides matters: if the CPU model and the translator parse different text,
the test measures the parser, not the translation.

## Building test programs

Hand-written RDNA2 assembly is the source of truth. Each test is an `.asm` file whose header comment
states the inputs, the observable outputs and the expected bits, matching the existing convention in
`runtime/test/shaders/*.asm`. The CPU expected values are computed by the interpreter and pinned in
the test table; the GPU must produce the same bits. Pinning the values catches a regression in the
CPU model as well as in the translator.

`llvm-mc` is already a prerequisite for the shader pipeline. The test build is the same one
`build.py` uses:

```
llvm-mc -triple=amdgcn--amdpal -mcpu=gfx1013 -show-encoding test.asm
```

A minimal case, with the expected value in the header comment like the existing GPU samples:

```
; f32_add_nan  hand-written test; expected v2 = 0x7fc00000
v_mov_b32 v0, 0x7f800000      ; +inf
v_mov_b32 v1, 0xff800000      ; -inf
v_add_f32 v2, v0, v1
s_endpgm
```

The prototype can check the CPU side of that case directly:

```
python -I tools/cpu_ref.py <listing.s> --parse-only
python -I tools/cpu_ref.py <listing.s> --json
python -I tests/test_cpu_ref.py
```

The scalar ALU has its own differential fuzzer (FUZZ-4), `tools/salu_diff.py`: a separate reference for 87
SOP1/SOP2/SOPC/SOPK opcodes with SCC kept explicit, seeded programs, the translated SPIR-V run in
`tests/semantic/spirv_mini_interp.py`, and failing programs shrunk to the instructions that matter.
64-bit mask ops are left out by default (the translator models them as wave masks), and `s_bfe_*` control
words stay inside offset + width <= 32 unless `--bfe-any`. Until SHADER-46 (#349) lands, a default run on
main reports `s_ff0_i32_b32` with a constant source; its docstring lists the known divergences.

```
python -I tools/salu_diff.py --programs 2000 --seed 1
python -I tests/test_salu_diff.py
```

### Hardware oracle (optional)

`--oracle hw` also runs the same programs natively on an AMD GPU and compares the hardware's three SGPRs and SCC
with the reference (`--oracle translator,hw` does both; `hw` alone leaves the translator out and then also
runs the 64-bit mask ops, which the hardware reads as plain integers). It tests the reference model, not the
translator. No tool is needed unless you pass `--oracle hw`; when one is missing the run prints
`hardware oracle unavailable: <what>` and exits 77, so CI is unaffected.

```
python -I tools/salu_diff.py --oracle hw --programs 5000 --seed 1
python -I tools/hw_oracle/runner.py        # checks the setup: prints the device, or what is missing (exit 77)
```

Setup on Windows, about 800 MB and no admin rights:

- **Driver.** The AMD Windows driver ships the HIP runtime as `amdhip64_7.dll` in System32; nothing else is
  loaded from the SDK. `LEONIDA_HIP_DLL` overrides the path.
- **Compiler.** AMD clang with the amdgcn backend and `ld.lld`. The stock LLVM from winget has no amdgcn
  target. The ROCm wheel has one:
  `py -m venv .venv` then `.venv\Scripts\python -m pip install --index-url https://stable.repo.amd.com/rocm/whl-next/ "rocm-sdk-core==10.0.0"`;
  `clang.exe` ends up in `.venv\Lib\site-packages\_rocm_sdk_core\lib\llvm\bin`. A `.venv` in the working
  directory or repository is found by itself; otherwise set `LEONIDA_AMD_CLANG` to that `clang.exe`.
- **Linux/ROCm is not supported yet:** `hw_oracle.c` is Windows-only (it loads the driver's DLL).
- **Host C compiler.** MinGW gcc (on PATH, or `LEONIDA_HOST_CC`) builds `tools/hw_oracle/hw_oracle.c` once into the
  temp folder. By hand: `gcc -O1 tools/hw_oracle/hw_oracle.c -o hw_oracle.exe`.
- `LEONIDA_HW_ARCH` overrides the detected architecture (`amdgpu-arch`, else `gfx1150`). GFX10 spellings
  (`s_load_dwordx2`, `global_store_dwordx4`) are used for `gfx10xx`, so the same kernels assemble for an RDNA2 card
  (checked by assembling for gfx1030 with clang and gfx1013 with `llvm-mc`; not run on one).

How it works: each program becomes one kernel (`tools/hw_oracle/asm.py`: load the output pointer, run the program,
store `v[8:11]`), 500 kernels per code object. The code object needs an `.amdgpu_metadata` block; without it the HIP
runtime fails the load with error 209 (`hipErrorNoBinaryForGpu`), which looks like a wrong GPU. The launcher starts
one workgroup of one thread per kernel and pre-fills the buffer with `0xDEAD0000`, so a kernel that did not write
shows up as a mismatch. A program the assembler rejects is counted as a skip, not a pass.

What it can and cannot say:

- It says what **that GPU** does. A Radeon 880M is gfx1150 (RDNA3.5), not RDNA2: a match supports the reference, and
  a difference is first checked against the RDNA2 ISA text before it is called a reference bug. Only an RDNA2
  card settles RDNA2.
- Skipped on gfx11, and listed in the run's skip counts: `s_ff0_i32_b32` and `s_ff0_i32_b64` (GFX11 has no
  find-first-zero), and programs with two different 32-bit literals in one instruction (not encodable). Renamed
  opcodes (`s_andn2_*` to `s_and_not1_*`, `s_ff1_*` to `s_ctz_*`, `s_flbit_i32_b32` to `s_clz_i32_u32`, ...) are
  accepted by LLVM under their old names with the same behaviour and are run.
- Only the scalar unit is observed, on one lane with EXEC = 1; nothing about wave64 or the other stages.

Result of the first runs (Radeon 880M, driver 26.11.2, AMD clang 23.0.0git): about 40,000 seeded programs
(1 to 10 instructions each, all opcodes in the table, seeds 1 to 3, 11, 21, 31, 41, 100, 200, with and without
`--bfe-any` and `--inline64`) gave 0 mismatches apart from one finding. `s_bfe_i32` with `offset + width > 32`:
the pseudocode taken literally (the old reference) sign-extends from bit `width - 1` of a zero-filled value, but
the hardware stops the field at bit 31 (`width = min(width, 32 - offset)`) and sign-extends from there. 4,000 random
control words, including bits 23 to 31 of the control word (ignored), fit that rule exactly. The translator already
did this (0 of 600 differ), so the reference was the one that was wrong, and it now follows the measured rule. The
rule rests on that gfx1150 measurement (4000 control words). LLVM's `constantFoldBFE` does the same for the VOP3
`v_bfe` nodes, but with a 5-bit width, so it is only analogous; RDNA2 is unmeasured.

Two assembler facts matter when writing edge cases:

- An integer token in a float-operand position uses AMD's inline-constant encoding for values 0..64
  and -16..-1. `v_add_f32 v0, v1, 2` therefore means `v1 + 2.0`, not `v1 + bits(2)`. To feed a
  specific bit pattern, put it in a VGPR with `v_mov_b32` first.
- A literal outside the inline range is emitted as a 32-bit literal and printed as hex (for example
  `0x100`), so denormal patterns above the inline range can be written directly. The smallest
  denormal (`0x00000001`) cannot: load it with `v_mov_b32`.

The first test matrix should cover:

| Case | Program shape | What it pins |
|---|---|---|
| Inline constants | `v_mov_b32` with `1.0`, `-1`, `0x3f800000` | bit patterns, not just numeric equality |
| NaN | `inf + -inf`, `0 * inf`, NaN input through add/mul | canonical quiet NaN and propagation policy |
| Signed zero | `-0 + +0`, `0 - 0`, `-1 * 0` | sign of zero results |
| Denormals | smallest denormal + itself, `* 0.5`, `* 1.5`, max denormal + min denormal | gradual underflow and rounding |
| FMA | exact midpoint case (see below) | fused, single rounding |
| 16-bit | packed `0x3c003c00`, `v_and_b32 0xffff`, `v_lshrrev_b32 16` | packed halves survive moves and masks |
| `cndmask` | VCC lane mask 0b1010, both source values distinct | select direction and per-lane masking |
| Shifts | amount 0, 31, 32, 33; negative arithmetic shift | 5-bit amount mask and sign extension |
| EXEC low half | write only `exec_lo`, then a vector op | the high half of a wave64 mask is untouched |
| EXEC both halves | `exec_lo = 0xffff0000`, `exec_hi = 0` | inactive lanes unchanged |

The FMA case is the one a naive host implementation gets wrong. A product of two binary32 values can
land exactly on a binary32 midpoint, and a tiny third addend decides the rounding direction. Evaluating
`a * b + c` in host double and then rounding to binary32 can lose that addend. The prototype uses exact
rational arithmetic for FMA; a test pins the result so the implementation cannot regress to the
double-rounded answer.

## Harness design

`tools/shader_diff.py` owns the test table, the seed and the comparison, leaving
`cpu_ref.py` a pure interpreter (it is never modified by the runner). Per case:

- **Inputs.** Initial SGPRs/VGPRs (per lane), VCC, EXEC, and later buffer/image contents. Inputs are
  generated from a named seed (SplitMix64 or `random.Random(seed)`), so a failure is reproducible
  from the seed alone. Hand-pinned cases should also accept explicit values.
- **Outputs.** A declared list of observable registers, buffer ranges or image texels. Compare raw
  bits. For NaN, compare "both NaN" by default and record the payloads; for finite values, require
  exact equality.
- **First mismatch.** Report case name, seed, lane, register/buffer offset, expected bits, actual
  bits, and the RDNA2 instruction at the PC that last wrote the value. Stop at the first mismatch;
  the full diff is noise once one lane diverges.
- **Environment checks.** Query the device for f32 denormal preservation, f16 support and subgroup
  size before running cases that need them. A missing feature is a skip with a reason, never a pass.
- **Harness canaries.** Hand-written SPIR-V twins for a few cases, plus an intentionally wrong
  expected value, must fail. A harness that cannot fail is not a harness.
- **No silent skips.** Count `pass`, `mismatch`, `skip(reason)` and `harness-error`. A run whose
  mismatch count is zero but whose skip count is 90% is not evidence of broad correctness.

The existing `runtime/test` tools already expose the hard parts (device selection, push constants,
descriptor setup, readback). The differential runner should call them rather than grow a second
Vulkan stack.

### LDS and `ds_*` on the GPU (`tools/lds_fuzz.py`, FUZZ-6)

LDS ops are workgroup-wide, so `tools/lds_fuzz.py` runs on the GPU rather than one lane in the
interpreter. Each seeded program is one 64-thread group (wave64, or two wave32s) with 4 KiB of LDS
that starts from a known pattern. It runs 4 to 12 random `ds_*` instructions, each followed by
`s_waitcnt` and `s_barrier`, then stores all 16 result VGPRs of every thread and the whole LDS.
`vk_compute_test` reads them back with `VKC_OUT_WORDS` / `VKC_DUMP`.

- **Instructions.** Every read width (`b32` to `b128`, `u8`/`i8`/`u16`/`i16`, `u16_d16[_hi]`, `u8_d16[_hi]`, `i8_d16[_hi]`), the
  byte and short writes (also `d16_hi`), `read2`/`write2` with and without `st64`, the add, sub,
  min, max, and, or and xor atomics with and without return, `wrxchg`, `cmpst`, `ds_swizzle_b32`
  and `ds_bpermute_b32`.
- **Reference.** A model in the same file, written from the ISA and not from the translator.
- **Determinism.** Programs stay deterministic. Writes, returning atomics, `wrxchg` and `cmpst` use
  addresses no other lane touches in the same op. Only order-independent atomics without return
  share addresses. Every access is in bounds, because the translator wraps out-of-range indices
  where hardware returns 0.
- **Swizzle.** Offsets are the two forms the LLVM `swizzle()` macros encode, checked with
  `llvm-mc 23 -mcpu=gfx1013`: bit 15 = quad permute; otherwise and = bits 4:0, or = 9:5,
  xor = 14:10, within each 32-lane group.
- **EXEC.** Random EXEC masks are used on every op except `ds_swizzle`.
- **Reports.** A mismatch names the thread, the register or LDS word, and the op that last wrote it.
  `--no-exec KIND` drops one op kind's masks without changing the rest of the program, to separate
  a known finding from new ones.

First results on an RX 6700 XT (RADV, Mesa 26.3.1), native wave64/wave32 and with
`--wave64-emulate` at subgroup 32:

- **One finding, already fixed by #253.** With wave32 `ds_bpermute` under EXEC, a source lane that
  is off returned its value instead of 0: 12 of 240 programs (seeds 1 to 3) failed, each time
  under a masked wave32 `ds_bpermute`, and with #253's translator all 480 programs (seeds 1 to 3,
  native and emulated) match.
- **Everything else matches.** With `--no-exec bpermute`, 1,040 programs match (seeds 1 to 8).
- **Not covered.** The fuzzer does not generate `ds_permute_b32` yet (FUZZ-10); the translator supports it since
  SHADER-90. The 8-bit d16 reads (`ds_read_u8_d16[_hi]`, `ds_read_i8_d16[_hi]`) are translated since SHADER-85
  (`cs_w32_lds_u8_d16`) and the fuzzer generates them.
### Memory instructions (`tools/mem_diff.py`, FUZZ-5)

`cpu_ref.py` has no memory model, so memory instructions get their own runner. `tools/mem_diff.py`
carries a small reference model written from the RDNA2 addressing rules (V# base and stride, MUBUF
`soffset + offset + voffset + vindex * stride`, SMEM pointer/V# plus offsets, signed 12-bit `global_*`
offsets, 32-bit buffer atomics) and compares it with the translation run in `spirv_mini_interp`. Both
the exported words and the whole final memory are compared, so stores and atomics are checked too.

```powershell
python -I tools/mem_diff.py --programs 3000 --seed 1 [--family buffer_load] [--max-ops 4]
python -I tools/mem_diff.py --self-test
python -I tests/test_mem_diff.py
```

Families: `s_load`, `s_buffer_load`, `buffer_load`, `buffer_store`, `buffer_atomic`, `global`
(dword loads/stores, `ubyte`/`sbyte`/`ushort`/`sshort`). Every access is generated in bounds and
dword aligned (sub-dword loads at their natural alignment), with no format in word3: out-of-bounds
results, `OOB_SELECT`, swizzling and the `s_buffer_load` alignment truncation depend on descriptor
fields the translator does not model yet, so they need a GPU-confirmed rule before they are fuzzed.
`spirv_mini_interp` gained `OpSConvert` and the 32-bit atomics on PhysicalStorageBuffer memory for
this. Images have their own fuzzer (`tools/image_fuzz.py`, below).

The `buffer_format` family covers `buffer_load_format_*` on every format the translator decodes, with a
random DST_SEL of 0/1/X/Y/Z/W, and `tbuffer_load_format_*` on the five formats the 1.0.2.1 corpus uses
(`16_16_SNORM`, `8_8_8_8_UNORM`, `8_8_8_8_UINT`, `2_10_10_10_UNORM`, `16_16_16_16_UINT`, about 53,000
instructions) with an identity swizzle. Its reference decodes each element from the GFX10 format *name*
(`tools/gfx10_formats.json`; channel widths listed from the top bit), not from the translator's own
table: UNORM/SNORM correctly rounded (`x/(2ⁿ−1)`), missing channels *c* mod *n*, DST_SEL 1 = integer 1
for UINT/SINT and 1.0f otherwise (SHADER-86, RX 6700). Elements are naturally aligned. Left out: whether
`tbuffer` applies the V#'s DST_SEL (unmeasured).

Over 5,000 programs the reference and translator match bit for bit; breaking the `16_16_SNORM` kind, the
`8_8_8_8_UINT` widths or the DST_SEL order in the translator makes 33, 37 and 191 of 300 programs fail.

A mutation check (inst offset dropped, V# stride narrowed, the SMEM pointer's high word dropped) makes
76, 45 and 42 of 150 programs fail, so the runner does catch address errors.

### Images on the GPU (`tools/image_fuzz.py`)

`tools/image_fuzz.py` starts on the image path, where the T# format, the view type and the binding layout
meet the driver. Each case assembles a pixel shader that reads one texel with `image_load` (a nonzero `dmask`, the
T# in `s[8:15]`), runs it through `runtime/test/vk_image_fuzz.cpp` against a synthetic texture of a chosen
format, and compares every RGBA32F pixel with an oracle that models the RDNA2 texel fetch for that format: a
fetch is a memory read, so there is no filter, border or LOD to model.

```sh
cmake --build runtime/build --target vk_image_fuzz
python -I tools/image_fuzz.py --programs 8 --seed 1 [--show]
python -I tools/image_fuzz.py --self-test
python -m pytest tests/test_image_fuzz.py
```

Covered: 2D `image_load`, `R8G8B8A8_UNORM`, `R16G16B16A16_SFLOAT`, `R32G32B32A32_SFLOAT`, `R32_SFLOAT`,
and `R8G8B8A8_UINT`/`SINT`, all as sampled images, as the runtime binds a read-only texture (integer formats are
read through the translator's uint/sint aliases of the sampled array). The shader converts the loaded 8-bit integer values to float for
readback; every such value is exactly representable, including signed negatives. Wider integers are not covered.
Texture sizes divide the 64x64 draw (8/16/32/64), so every texel is read at least once. The nonzero masks cycle from
`0xf` through `0x1`–`0xe`; 90 programs cover all 15 masks for each of the six default formats. Selected channels
are packed into consecutive destination registers, while unwritten registers keep initialized sentinel values.
The packing model follows the image VGPR usage description in AMD's
[RDNA 1.0 ISA guide](https://www.amd.com/content/dam/amd/en/documents/radeon-tech-docs/instruction-set-architectures/rdna-shader-instruction-set-architecture.pdf);
the Vulkan test compares translated shaders on the host GPU, not native RDNA2 execution.

Devices: the harness needs `fragmentStoresAndAtomics`, and no without-format storage-image feature: a translated
`image_load` fetches every format from the sampled arrays, UINT/SINT ones through integer aliases (SHADER-151, the
"Typed image loads" row of `docs/design.md`). Until then it declared the uint/sint storage path even when the case
read a float format, so an Intel UHD 770 (Xe-LP), which lacks `shaderStorageImageReadWithoutFormat`, skipped every
case. Not covered yet: wider UINT/SINT formats, mips, 1D/2D-array/3D/cube,
`image_sample`/`gather` with filters and borders, and texel offsets with `VK_KHR_maintenance8`.

### 16-bit and memory ops on the GPU (`tools/valu_fuzz.py`, FUZZ-2, FUZZ-5)

`tools/valu_fuzz.py --family f16|mem` uses the same runner shape as `lds_fuzz.py`. It generates seeded
64-thread compute programs, translates them, runs them with `vk_compute_test`'s `VKC_DUMP`, and compares every
thread's 16 result VGPRs (plus the scratch memory for `mem`) bit for bit with a model of the instructions written in
the same file. The model doesn't import the translator. `tests/test_valu_fuzz.py` checks the models on the CPU:
the f16/f32 rounding against Python's IEEE packers, signed zeros, the partial-write rules and out-of-range
buffer accesses. If llvm-mc is present, it also checks that every family's
programs assemble for gfx1013.

- **`f16`**: f16 add/sub/mul/min/max/fma/fmac/med3, `v_cvt_f16_f32`/`v_cvt_f32_f16`/`v_cvt_pkrtz_f16_f32`,
  `v_pk_*` (f16 and shifts) with `op_sel`/`op_sel_hi`, 16-bit integer ops and `v_pack_b32_f16`. It only uses
  encodings `llvm-mc -mcpu=gfx1013` accepts: on GFX10 the VOP3 forms of the f16 VOP2 ops take no `op_sel`, and the
  16-bit integer ops exist only as `op_sel` VOP3. Which form keeps the other half follows the 16-bit row of
  `docs/design.md`. Results whose exact value is NaN, infinite or denormal, denormal inputs, and min/max of +0
  against -0 are marked uncertain and not compared.
- **`mem`**: raw (`offen` + `soffset` + offset) and structured (`idxen`) `buffer_load/store_dword` to `dwordx4`
  through V#s built in SGPRs, with a random `num_records` so some threads are out of range (loads read 0, stores
  are dropped). Each thread uses only its own 64-byte slot. Accesses are wholly in or out of range, because
  multi-dword accesses straddling `num_records` are the known gap in "Buffer bounds checking". Byte/short buffer
  loads and scalar loads aren't generated: the translator rejects the former, and SHADER-41 covers the latter.

`--skip-op OP` (an op name) sets a known finding aside without changing the rest of the program.
`--show SEED:INDEX:WAVE` prints a program and its model.

### Wave and cross-lane ops on the GPU (`tools/crosslane_fuzz.py`, FUZZ-7)

Cross-lane ops read other lanes, so like FUZZ-6 this runs on the GPU and not in the one-lane interpreter.
Each seeded program is one compute group of a wave32 or wave64 program. Group sizes cover one full wave, two or
three waves, and a partial last wave (20, 50, 40 or 100 threads). The program runs 24 to 28 random ops, then
every thread stores its six data VGPRs and eight data SGPRs, so each half of a wave reports its own copy of
every scalar. `vk_compute_test` reads them back with `VKC_OUT_WORDS` / `VKC_DUMP`.

- **Instructions.** `v_readlane_b32` and `v_writelane_b32` (constant and SGPR lane selects; in wave32 also
  selects of 32 and up), `v_readfirstlane_b32`, `v_cmp` into VCC or an SGPR mask, `v_cmpx`, `s_and_saveexec`
  and its restore, nested up to three deep, constant EXEC halves, `s_bcnt1`/`s_bcnt0`/`s_ff1` of a mask or
  EXEC, `v_mbcnt_lo/hi` on a mask, EXEC or plain data, `v_cndmask` on a mask, `s_and/or/xor/andn2` of masks,
  `v_permlane16_b32` / `v_permlanex16_b32`, and forward `s_cbranch_execz/vccz/vccnz` over one to four of these.
  With `--dpp`: a DPP16 or DPP8 src0 on `v_mov`/`v_not`/`v_add_nc_u32`/`v_xor`, with every DPP16 control,
  `row_mask`, `bank_mask`, `bound_ctrl` and `fi`. Masks read as plain data (`v_mov_b32 v, s_mask`) belong to
  #293 and are left out with `--no-maskdata`.
- **Reference.** A 64-lane model in the same file, written from the ISA (rules in its docstring). Its unit
  test, `tests/test_crosslane_fuzz.py`, needs no GPU.
- **Defined behaviour only.** Lane selects, permlane and DPP never read a lane missing from a partial wave.
  EXEC constants only enable lanes that exist. Every mask register is zero-initialised, as compiled code does,
  so a branch that skips the `v_cmp` writing a mask leaves a defined mask after the join.
- **Hypotheses are reported apart.** `v_permlane16/x16` with op_sel FI = 0 reading a lane that is off in
  EXEC is modelled as the DPP "invalid source" case (vD kept). That is now measured, with the same result on both:
  @aslan2142 on RDNA2 and @AlxcBTW on RDNA3.5 (#374). FI = 1 returns the source lane's value, FI = 0 leaves the
  destination unwritten and FI = 0 with BOUND_CTRL gives 0; #580 (SHADER-74) makes the translator do that. The
  fuzzer's docstring (`tools/crosslane_fuzz.py`, since #750) already says it was measured, but `op_permlane` still
  tags the case `hyp` because the translator models it only with #580, so a program that hits it and mismatches
  is filed as `hyp_only` instead of `mismatch` (one that matches still counts as `match`). The model (vD kept)
  already matches the measurement; dropping the tag once #580 lands is a follow-up.
  `--permlane-fetch-inactive` models FI = 1 instead.
- **Modes.** Wave32 runs natively. Wave64 runs with `--wave64-emulate`; `--w64-native` adds the plain build,
  which on a 32-wide subgroup is the documented cross-half approximation and is expected to mismatch.
- **Reports.** The thread, wave, lane and register of the first differences. `--minimize` drops ops while the
  mismatch persists, and `--keep DIR` saves the programs. `--translator PATH` tests another checkout's
  translator (for example an open pull request).

First results on an RTX 4090 (subgroup 32, driver 621.14):

- **SHADER-52 (#407).** In wave32, `v_readlane`/`v_writelane` masked the lane select with 63, not 31. 23 of
  100 wave32 programs failed, each one minimized to a `v_writelane` with a lane select of 32 or more.
- **Two DPP bugs in #238** (SHADER-23, posted there). Non-mov DPP ops wrote `op(old vD, src1)` in lanes that
  should be left unwritten (invalid source without `bound_ctrl`, or a lane off in `row_mask`/`bank_mask`), and
  DPP16 ignored `fi:1`. Each was confirmed with a hand-written program.
- **Everything else matches.** On `main`, 1,000 programs (`--seed 8 --ops 28 --no-maskdata --no-w32-hi-lanes`, 500 wave32 native and 500 wave64 emulated) give 0 mismatches. 297 of them are `hyp_only` (permlane with FI = 0 and an inactive source), and with `--permlane-fetch-inactive` all 1,000 match. Native wave64 on the 32-wide card mismatches as documented: `s_bcnt1` of EXEC counts one half, and a `v_readlane` from the other half wraps.
- **Closed question (#374):** what hardware does for permlane with FI = 0 was measured on RDNA2 and RDNA3.5 (see above).

## Coverage plan and cost

Start with straight-line scalar/vector ALU because it is deterministic, needs no descriptors and
covers the operations most shaders use. Then add, in order:

1. **EXEC/VCC and `cndmask`.** Per-lane masking and scalar state are where a one-lane-per-invocation
   translator differs most from hardware.
2. **16-bit arithmetic and partial writes.** `v_add/sub/mul/fma_f16`, `op_sel`, destination-half
   preservation, packed ops. The translator's contract calls out which forms keep the high half;
   those rules need value tests.
3. **Branches and loops.** Modelled control flow, then irreducible graphs and the dispatch-loop form.
   The existing control-flow unit tests check graph structure; differential tests check the values.
   `tools/cf_diff.py` (FUZZ-8) does this for whole waves: seeded programs built from the shapes
   compilers emit (saveexec if/else, `v_cmpx`, VCC- and SCC-voted branches, uniform and per-lane
   loops, two-entry cycles), run by `tools/cf_ref.py` (cpu_ref plus branches, compares and saveexec)
   and by `tests/semantic/spirv_wave_interp.py` (every invocation of a subgroup, with structured
   reconvergence), compared lane by lane in both the structured and the dispatch-loop form.
4. **LDS, global memory, buffers and images.** These need a memory model and descriptor setup; the
   existing `vk_compute_test`/`vk_store_test`/`vk_texture_test` programs show the shape.
5. **Wave operations.** Ballot, shuffles, DPP, `readlane`/`writelane`, `permlane`, `mbcnt`, and the
   wave64 split on 32-wide subgroups. These are the largest semantic gap and need a 64-lane model
   plus a subgroup-size-aware host. `tools/crosslane_fuzz.py` (FUZZ-7, above) is that model, run on
   the GPU.
6. **Pixel interpolation and exports.** Barycentrics, per-vertex inputs, compressed exports, MRT and
   depth. The existing GPU tests already pin several of these; a CPU model makes them differential.

Cost is dominated by the GPU dispatch and pipeline creation, not by the Python interpreter. A small
case is one `llvm-mc` run, one translation, one `spirv-as` + `spirv-val`, one pipeline and one
dispatch: seconds per case on a desktop GPU. The interpreter is milliseconds for the programs in this
subset. A suite of a few hundred cases is practical locally; running every corpus shader is not the
goal, because many need descriptors and data the harness cannot invent. The useful target is a
hand-picked suite that covers each instruction family and each known semantic trap, plus seeded
fuzzing over operand values and lane masks within that suite.

## What this cannot model

The prototype and the first harness stage deliberately do not cover:

- **Descriptors.** T#/V#, S# and `s_load`/`image_*`/`buffer_*` need realistic descriptor memory and
  binding setup. A wrong descriptor is indistinguishable from a wrong translation.
- **Interpolation.** `v_interp_p1/p2/mov` depend on rasterization state, barycentric conventions and
  the pixel center. The CPU model would need the same triangle and sample positions as the GPU.
- **Wave operations.** Ballot, shuffles, DPP, permlane and cross-half wave64 behaviour need a full
  64-lane model and a GPU whose subgroup size matches the wave. On a 32-wide subgroup host, a wave64
  program is not the same execution. The one-lane interpreter still can't run them; the GPU fuzzer
  `tools/crosslane_fuzz.py` covers them instead, with wave64 checked through `--wave64-emulate`.
- **Memory and LDS.** Address spaces, out-of-bounds rules, atomics, ordering and barriers. These are
  modelled later, with the same "unsupported, never guessed" rule.
- **Transcendentals and approximation.** `v_rcp_f32`, `v_rsq_f32`, `v_exp_f32`, `v_log_f32` and the
  trig instructions have implementation-defined accuracy. The other project handled a 5–14 ulp
  reciprocal difference by computing an envelope, not by demanding a bit match. A differential test
  should either use exact instructions or compare within a documented tolerance/envelope.
- **The MODE register and rounding modes.** The translator skips `s_setreg`; the CPU model assumes
  round-to-nearest and gradual underflow. Tests that depend on flush-to-zero or round-toward-zero are
  out of scope until the harness sets and checks those modes.
- **NaN payloads and signed-zero corner cases in min/max/clamp.** Vulkan leaves some of this
  implementation-defined. Compare NaNs by class, and pin only the cases the ISA defines.
- **Wave-size-dependent operand forms.** `v_cndmask_b32` takes a 32-bit SGPR mask in wave32 and an
  `s[lo:hi]` pair in wave64. The model follows the form printed in the listing; a listing assembled
  for the wrong wave size is a test bug, not a translator result.
- **Scheduling and hazards.** The model is sequential. It does not test wait counts, dependency
  stalls, or whether two instructions can overlap. Those are hardware timing properties, not
  arithmetic results.

Growing the model means adding one instruction family at a time, with a hand-written test and a
pinned expected value, and removing the `Unsupported` path for that family. A family that cannot be
modelled faithfully stays a reported skip; it is never approximated silently.

## Prior art

The technique is described in another project's write-ups; no code from them is copied.

- [Xbox Series shaders](https://github.com/realdody/americas-recomp/blob/xbox-series/docs/xbox/XBOX_SHADERS.md),
  sections "Execution harness" and "A CPU reference for small real shaders".
- [tools/xbox README](https://github.com/realdody/americas-recomp/blob/xbox-series/tools/xbox/README.md),
  which describes `exec` and the check against a Python reference.

Their reported numbers, as a scale reference only: 183 of 872 compute shaders of 3,000 bytes or less
gave byte-identical CPU/GPU results, with none differing and 687 skipped; for shaders up to 1,500
bytes the later run reported 242 matches, 1 mismatch (reciprocal precision), 2 harness failures and
435 skips; the pixel-shader run reported 157 bit-identical, 0 differing and 581 skips. The numbers
are context for the scale of the approach, not a result of this change.

The other project is GPL-2.0-only and this one is GPL-3.0-or-later. The design above is an
independent implementation from the ISA and from LLVM's disassembly; the links are for the technique
and the reported results, not for code.

## Deliberate limits of this change

This change adds the design, the CPU prototype and its unit tests. It does not modify
`tools/rdna2spirv.py`, does not add a GPU runner, and does not use game data. The prototype is useful
now because it pins the semantics of the first instruction families and gives the future harness a
reference that fails loudly on anything it does not understand.
