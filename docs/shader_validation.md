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

- All 26,503 shaders translate and pass `spirv-val` (Vulkan 1.2) at `43d4703`, after `dcd8ecf` fixed
  the ID-bound regression (#881); the intervening `dedd75c` run was 26,501, and `793b06f` before
  #223's LLVM 23 source-field fix was 25,032. Yet the SDWA source/destination selects were silently
  ignored until 2026-11-20: validation passed for 3,826 of the 13,926 shaders translated at the time
  and produced wrong values (`docs/design.md`, "SDWA").
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
words stay inside offset + width <= 32 unless `--bfe-any`. Its docstring lists the known divergences.

`--branches` puts SCC into control flow, one lane: after an SCC producer comes either a chain of 1..3 consumers
(`s_cselect_b32`, `s_addc_u32`, `s_subb_u32`) or `s_cbranch_scc0`/`scc1` forward over the next 1..3 scalar
instructions (no loops or nesting), and half the instructions are SCC producers. The listing carries
`; encoding:` placeholders of each instruction's size, which places the branch targets. It is a flag so that the
default seeds keep their programs; translator oracle only.

```
python -I tools/salu_diff.py --programs 2000 --seed 1
python -I tools/salu_diff.py --branches --programs 2000 --seed 1
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
  loaded from the SDK. `LEONIDA_HIP_DLL` overrides the path. Drivers from 2024 ship `amdhip64_6.dll` instead and
  refuse code object v6 (LLVM 23's default) with error 209; the oracle then assembles as v5 (seen with driver
  32.0.11020.1002 on gfx1103).
- **true16.** LLVM 23 wants `v8.l`, not `v8`, for 16-bit operands on GFX11. The oracle asks the assembler once and
  uses the spelling it takes, for the ops its tools run: the two f16/f32 conversions and `f16_trans_sweep.py`'s
  seven. Another f16 op needs adding to `asm.true16_spelling`.
- **Compiler.** AMD clang with the amdgcn backend and `ld.lld`. The stock LLVM from winget has no amdgcn
  target. The ROCm wheel has one:
  `py -m venv .venv` then `.venv\Scripts\python -m pip install --index-url https://stable.repo.amd.com/rocm/whl-next/ "rocm-sdk-core==10.0.0"`;
  `clang.exe` ends up in `.venv\Lib\site-packages\_rocm_sdk_core\lib\llvm\bin`. A `.venv` in the working
  directory or repository is found by itself; otherwise set `LEONIDA_AMD_CLANG` to that `clang.exe`.
- **Linux** has a launcher of its own, without ROCm: see the next list. `hw_oracle.c` is the Windows one (it loads
  the driver's DLL).
- **Host C compiler.** `tools/hw_oracle/hw_oracle.c` is built once into the temp folder by MinGW gcc (on PATH or in
  `C:\msys64`), else by MSVC: a `cl` on PATH (a Developer shell), else the Visual Studio that `vswhere` finds, through
  its `vcvars64.bat`. The Visual Studio the runtime needs is therefore enough. `LEONIDA_HOST_CC` overrides the choice
  (gcc or `cl.exe`). By hand: `gcc -O1 tools/hw_oracle/hw_oracle.c -o hw_oracle.exe` or `cl /O1 tools\hw_oracle\hw_oracle.c`.
- `LEONIDA_HW_ARCH` overrides the detected architecture (`amdgpu-arch`, else `gfx1150`). GFX10 spellings
  (`s_load_dwordx2`, `global_store_dwordx4`) are used for `gfx10xx`, so the same kernels assemble for an RDNA2 card
  (checked by assembling for gfx1030 with clang and gfx1013 with `llvm-mc`, and run on gfx1036 on Linux).
  `gfx12xx` (RDNA4) uses the GFX11 spellings. An architecture outside gfx10, gfx11 and gfx12 is `unavailable`
  (exit 77), like a missing tool.

Setup on Linux, without ROCm. `tools/hw_oracle/hw_oracle_drm.c` submits each kernel as a raw compute dispatch
through the kernel driver (`libdrm_amdgpu`), so nothing sits between the instructions and the GPU:

- **GPU.** GFX10 only, which is RDNA1 and RDNA2 (compute engine version 10); the launcher answers anything else
  with exit 77. Name the render node, for example `LEONIDA_HW_DRM_NODE=/dev/dri/renderD129`
  (`ls -l /dev/dri/by-path` shows which node is which PCI device). There is no default device, so a raw command
  stream never goes to a GPU by accident.
- **Not the GPU you are looking at.** A raw stream bypasses the Vulkan driver, and a kernel that hangs makes the
  kernel driver reset that GPU. A node with a display connected is refused unless
  `LEONIDA_HW_DRM_ALLOW_DISPLAY=1`. An integrated GPU next to a discrete card is the convenient case.
- **Tools.** `clang` with the amdgcn target (the distribution's LLVM has it; `LEONIDA_AMD_CLANG` overrides), `gcc`
  and the libdrm headers. The launcher is built once into the temp folder. By hand:
  `gcc -O1 tools/hw_oracle/hw_oracle_drm.c -o hw_oracle_drm $(pkg-config --cflags --libs libdrm_amdgpu)`.
- **Access.** Your user must be able to open the render node (the `render` or `video` group on most
  distributions).
- **The command stream** is libdrm's own compute dispatch test for GFX10 (`amdgpu_test_dispatch_memset`), with what
  belongs to the kernel put in: its entry point, its group size, and what its kernel descriptor says.
  - `COMPUTE_PGM_RSRC1`, `COMPUTE_PGM_RSRC2` and the wave size are read out of the descriptor in the assembled
    object, so the `.amdhsa_*` directives take effect as they do under the HIP runtime: register counts, float
    rounding and denormal modes, IEEE mode, DX10 clamp. A test checks that an f32 rounding mode set there changes
    a sum on the hardware.
  - A descriptor that asks for something the launcher does not provide (LDS, scratch memory, more than the
    kernarg pointer in the user SGPRs) is refused instead of being run differently.
  - Every submission gets command memory of its own. While one buffer was rewritten in place, the second kernel of
    a process ran the first one's commands, every time; `tests/test_hw_oracle.py` runs 300 distinct kernels to
    catch that.

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
- Skipped on gfx12 as well: the twelve SOPK compares `s_cmpk_{eq,lg,gt,ge,lt,le}_{i32,u32}`, which AMD clang
  assembles for gfx1150 and rejects for gfx1201. They are about a third of the programs there (seed 1: 870 of 2000
  skipped in all, 1130 run).
- Only the scalar unit is observed, on one lane with EXEC = 1; nothing about wave64 or the other stages.

Result of the first runs (Radeon 880M, driver 26.11.2, AMD clang 23.0.0git): about 40,000 seeded programs
(1 to 10 instructions each, all opcodes in the table, seeds 1 to 3, 11, 21, 31, 41, 100, 200, with and without
`--bfe-any` and `--inline64`) gave 0 mismatches apart from one finding. `s_bfe_i32` with `offset + width > 32`:
the pseudocode taken literally (the old reference) sign-extends from bit `width - 1` of a zero-filled value, but
the hardware stops the field at bit 31 (`width = min(width, 32 - offset)`) and sign-extends from there. 4,000 random
control words, including bits 23 to 31 of the control word (ignored), fit that rule exactly. The translator already
did this (0 of 600 differ), so the reference was the one that was wrong, and it now follows the measured rule. The
rule rests on that gfx1150 measurement (4000 control words). LLVM's `constantFoldBFE` does the same for the VOP3
`v_bfe` nodes, but with a 5-bit width, so it is only analogous.

Result on RDNA2 (the integrated GPU of a Ryzen 9 7950X3D, Raphael, gfx1036; the Linux launcher, Linux 7.3.5,
libdrm 2.4.134, clang 23.1.5 from the distribution): 70,000 seeded programs, 0 mismatches.

| Run | Programs | Match | Mismatch | Skipped (two literals) |
|---|---|---|---|---|
| seeds 1 to 8 | 40,000 | 39,439 | 0 | 561 |
| seeds 1 to 4, `--bfe-any` | 20,000 | 19,727 | 0 | 273 |
| seeds 1 and 2, `--inline64` | 10,000 | 9,855 | 0 | 145 |

So the reference, the `s_bfe_i32` rule above included, holds on RDNA2 as measured, and GFX10's `s_ff0_i32_b32` and
`s_ff0_i32_b64`, which gfx1150 cannot run, are covered. As a control, a reference whose `s_min_i32` sets SCC for
equal operands too (`<=` for `<`) gave 2 mismatches in 5,000 programs, minimized to that instruction.

RDNA4 (Radeon RX 9070, gfx1201, driver 26.11.2, `rocm-sdk-core` 10.0.0 clang): 10,000 programs (seeds 1, 2, 3 and
500, 2000 each, and seed 500 with `--bfe-any`) gave 0 mismatches, so gfx1201 follows the same `s_bfe` rule. With
`--oracle translator,hw`, 200 of 200 programs matched (seed 1).

#### RDNA2 integer rows (`tools/int_hw_probe.py`, SHADER-137)

`tools/int_hw_probe.py` runs 87 hand-written cases across 19 integer groups (both wave32 and wave64) natively on
an AMD GPU through `tools/hw_oracle` and compares every output word against `tools/rdna2spirv.py` +
`tests/semantic/spirv_mini_interp.py` (with scalar ops run twice, SCC preset to 0 and 1, so SCC written vs kept is
observed directly).

Measured on an **AMD Radeon 680M (gfx1035, RDNA2)**, Windows 11 (`10.0.26300`), AMD driver `26.5.1`
(`32.0.21043.7012`, `amdhip64_6.dll`), conda-forge `clang 22.1.8` + `lld 23.1.6` (`--oracle both --wave both`):
**87 cases: 82 bit-exact matches, 5 known live translator divergences, 0 unexpected mismatches, 0 wave32/wave64
differences** (plus `tools/salu_diff.py --oracle hw,translator --bfe-any --programs 500 --seed 1`: 500/500 match).

- **Matched RDNA2 silicon in both wave32 and wave64 (82/87):** `s_absdiff_i32`, `s_mul_hi_i32/u32`, `s_lshl1..4_add_u32`
  (64-bit shift-out and add carry into SCC), `s_sext_i32_i8/i16`, `s_flbit_i32`, `s_cmov_b32`, `s_cmovk_i32`,
  `s_pack_ll/lh/hh_b32_b16`, `s_bfe_u32/i32` past bit 31, `s_brev_b64`, `s_bfe_u64/i64`, `s_bitset0/1_b64`,
  `s_flbit_i32_b64/i64`, `s_quadmask_b32/b64`, `s_bitreplicate_b64_b32`, all nine 64-bit scalar bitwise ops on scalar
  data, `s_movrels_b32/b64`, `v_add/sub/subrev_nc_u32` / `v_add/sub_nc_i32` / `v_add/sub_nc_u16` with `clamp` and
  `op_sel`, `v_mad_u64_u32` / `v_mad_i64_i32` (`sdst` bit 64 sign of exact 65-bit sum and inactive-EXEC masking),
  `v_sad_u8`/`hi_u8`/`u16`/`u32`, `v_msad_u8`, `v_lerp_u8`, `v_qsad_pk_u16_u8`, `v_mqsad_pk_u16_u8`, `v_mqsad_u32_u8`,
  `v_mul_hi_u32_u24`/`i32_i24`, `v_lshrrev_b64`/`v_ashrrev_i64`, `v_mad_u16`/`i16` (`clamp`, `op_sel`, untouched half
  kept), `v_cvt_f16_u16/i16` and `v_cvt_u16/i16_f16` (`_e64` keeps high half; NaN → 0, ±inf saturates; note GFX10
  does not accept `op_sel` on these opcodes), `v_cvt_i32_f32` (control: all ±qNaN/±sNaN → 0), and
  `v_cvt_pknorm_i16/u16_f32` (±qNaN → 0, ±inf saturation, RNE ties at ±0.5).
- **Live translator divergences confirmed on gfx1035 (5/87, pinned in `KNOWN_DIVERGENCES`):**
  1. **`v_add_co_u32`, `v_sub_co_u32`, `v_subrev_co_u32`, `v_add_co_ci_u32`, `v_sub_co_ci_u32`, `v_subrev_co_ci_u32`
     with `clamp`:** hardware saturates `D` (`0xffffffff` on carry, `0x00000000` on borrow) while still reporting the
     real carry/borrow (`1`) in `sdst`; the translator currently ignores `clamp` on these ops.
  2. **`v_cvt_rpi_i32_f32` precision:** hardware computes exact `floor(x + 0.5)` without binary32 rounding of `x + 0.5`
     (`0x3effffff` → `0`, `0x4b000001` → `0x00800001`, `0x4b7fffff` → `0x00ffffff`), whereas the translator rounds
     `x + 0.5` in f32 first (`1`, `0x00800002`, `0x01000000`).
  3. **`v_cvt_flr_i32_f32` / `v_cvt_rpi_i32_f32` on NaN:** hardware saturates by the NaN's sign bit (`0x7fc00000` →
     `0x7fffffff`, `0xffc00000` → `0x80000000`), whereas `v_cvt_i32_f32` gives `0` for every NaN; the translator gives
     `0` for both `flr` and `rpi`.

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
- **Subgroup size.** Wave32 programs, and wave64 ones built with `--emulate`, ask `vk_compute_test` for
  32-lane subgroups (`VKC_SUBGROUP_SIZE=32`). Where the device can't pin that size, `vk_compute_test`
  prints `compute subgroup size: 32 not available` and runs at the driver's choice, and the wave32
  program is a skip, not a mismatch: its words come from another wave size. Mesa llvmpipe (8 lanes)
  is such a device. Emulated wave64 uses no subgroup operations, so it is still compared there.
  `valu_fuzz.py` and `crosslane_fuzz.py` follow the same rule.

First results on an RX 6700 XT (RADV, Mesa 26.3.1), native wave64/wave32 and with
`--wave64-emulate` at subgroup 32:

- **One finding, already fixed by #253.** With wave32 `ds_bpermute` under EXEC, a source lane that
  is off returned its value instead of 0: 12 of 240 programs (seeds 1 to 3) failed, each time
  under a masked wave32 `ds_bpermute`, and with #253's translator all 480 programs (seeds 1 to 3,
  native and emulated) match.
- **Everything else matches.** With `--no-exec bpermute`, 1,040 programs match (seeds 1 to 8).
- **Not in the default mix.** `ds_permute_b32` is generated with `--permute` (below). The 8-bit d16 reads
  (`ds_read_u8_d16[_hi]`, `ds_read_i8_d16[_hi]`) are translated since SHADER-85 (`cs_w32_lds_u8_d16`) and the
  fuzzer generates them.

#### Hardware oracle for the LDS programs (`--oracle hw`)

`--oracle hw` runs each program natively on an AMD GPU through `tools/hw_oracle/` (setup: "Hardware oracle" under
FUZZ-4) and compares every stored word, 16 result registers per thread and all 1,024 LDS dwords, with the model;
`--oracle translator,hw` does both and prints what the translator did with each program the hardware disagrees
with. It tests the **model**, not the translator. Without the tools it prints `hardware oracle unavailable: ...` and
exits 77.

```
python -I tools/lds_fuzz.py --oracle hw --programs 2000 --seed 1
python -I tools/lds_fuzz.py --oracle hw --extended --programs 2000 --seed 1 --negative-control
python -I tools/lds_fuzz.py --oracle hw --programs 50 --seed 1 --wave 32 --only 1,18,20,24 --minimize
```

- **What runs.** The program's real group size (`--threads 32|64|128|256|mix`: one partial wave, one wave, or several
  waves joined by `s_barrier`), wave32 and native wave64 (`.amdhsa_wavefront_size32 0`), its EXEC masks (the entry
  EXEC is saved in `s10` and restored, so a partial wave keeps its missing lanes off), and the LDS size in the
  kernel descriptor and metadata (`.amdhsa_group_segment_fixed_size`; without it the group gets no LDS). Results go
  out with `global_store_b32` through the output pointer.
- **GFX11 spellings** (`hw_oracle/asm.py`, checked by assembling every mnemonic with AMD clang for gfx1030 and
  gfx1150): `ds_read_*` is `ds_load_*`, `ds_write_*` is `ds_store_*`, `ds_read2[st64]_b32` is
  `ds_load_2addr[_stride64]_b32`, `ds_write2[st64]_b32` is `ds_store_2addr[_stride64]_b32`, `ds_wrxchg_rtn_b32` is
  `ds_storexchg_rtn_b32`. The assembler still takes the old names on gfx1150 except `ds_cmpst_rtn_b32`.
  **`ds_cmpstore_rtn_b32` takes the value to store before the compare value**, the reverse of GFX10's
  `ds_cmpst_rtn_b32 vdst, vaddr, cmp, store`; llvm-mc encodes both spellings of the same text identically, so only
  the operand order says which. Unswapped, every program with a `cmpst` failed on gfx1150 (each minimized to that one
  op); swapped, none does. This is measured on GFX11 only: on gfx12xx the programs with a `cmpst` are skipped with
  that reason. The model keeps the GFX10 order. Nothing else the generator emits is missing on GFX11, so no other
  program is skipped for its semantics; an assembler rejection is counted as `rejected`, never as a pass.
- **Wider op mix**, opt-in so that a seed keeps its programs: `--permute` (`ds_permute_b32`), `--swizzle-exec`
  (`ds_swizzle_b32` under EXEC masks), `--threads`, `--extended` (all three). `ds_permute_b32` and a masked
  `ds_swizzle_b32` carry a hypothesis (the model's reading of the ISA text); on the hardware a failing one is a
  mismatch like any other and counted in `hyp_only`, on the translator it is reported apart as `HYPOTHESIS MISMATCH`.
  The swizzle hypothesis was falsified at first ("an active lane reads its source lane's register whether or not
  that lane is on"): gfx1150 returns 0 when the source lane is off in EXEC, as `ds_bpermute` does. The model now
  says that.
- **Negative controls.** `--perturb NAME` judges the hardware against a deliberately wrong model (mismatches must
  appear); `--negative-control` re-judges the words of the run with all 15 (`max_i32` unsigned, `min_u32` signed,
  swapped `cmpst`, zero-extended `i8`/`i16`, d16 reads clearing the other half, `b16_d16_hi` shift, swapped
  `write2`, `st64` unit, bpermute reading off lanes or crossing the wave64 halves, reversed quad permute, swapped
  swizzle or/xor, permute writing no zero) and exits 1 if one goes unnoticed. `--minimize` drops ops while a
  program still differs, `--keep DIR` writes the native kernel of each mismatch, `--only I,J` runs chosen indices.

Results on a Radeon 880M (gfx1150, RDNA3.5, driver 26.11.2, AMD clang 23.0.0git), `--oracle hw`:

| Programs | Count (wave32 + wave64) | Mismatches |
| --- | --- | --- |
| default mix, seeds 1, 2, 3, 11, 21, 31, 41, 100, 200, 2,000 each | 36,000 | 0 |
| `--extended`, seeds 1, 2, 3, 11, 21, 31, 2,000 each (permute, masked swizzle, groups of 32 to 256) | 24,000 | 0 |
| seed 1, `--only` the twelve wave32 programs SHADER-105 names (1 18 20 23 24 26 29 30 32 33 34 38) | 12 | 0 |

Negative control (1,000 extended programs): each of the 15 wrong models mismatches on 126 to 547 programs, none goes
unnoticed. gfx1150 is RDNA3.5,
not RDNA2: the match supports the model, and RDNA2 itself rests on the ISA text and the translator results on RDNA2
cards.

With `--oracle translator,hw` on the same Radeon 880M (Vulkan, subgroup 64), seed 1, 100 programs per run: the
default mix matches the model on the translator, native and emulated (200/200 each). With `--extended` the
translator differs from the hardware in 22 of 177 programs (native and emulated alike), all in a `ds_swizzle_b32`
under an EXEC mask whose source lane is off: the translator hands the active lane the source lane's register, the
hardware gives 0 (the swizzle hypothesis above). So the translator differs from gfx1150 for a masked
`ds_swizzle_b32` with an off source lane; RDNA2 is unverified. Not fixed here. Partial-wave programs (`--threads` 32
in wave64) are skipped for the translator, which cannot launch them. NVIDIA was not run.

**Other backends.** The Linux libdrm launcher of #1265 (`tools/hw_oracle/drm.py`) does not run these programs yet:
it refuses a kernel with LDS and caps the output at 4096 words, and `lds_fuzz` turns that refusal into exit 77
("unavailable"). RDNA2 results need LDS and a larger word count in that backend first. The numbers above are gfx1150
only.

**Overlap with #789.** #789 adds `ds_permute_b32` to the default op mix of `lds_fuzz.py`; here it is opt-in
(`--permute`) so that existing seeds keep their programs. Whichever lands second reconciles the two.

**SHADER-105** (Intel `lds_fuzz` mismatches in DS atomics `max_i32`/`max_u32`, `ds_cmpst`, small reads, and
registers that kept their initial value): the model matches gfx1150 for every one of those families, under EXEC
masks, in wave32 and wave64, and in multi-wave groups. The Intel mismatches went away with #820's constant-EXEC fix
(50/50 native and 100/100 emulated for seeds 1 to 3). `ds_bpermute` (SHADER-31, #253) matches the model on gfx1150
too, including "an off source lane reads 0" in both wave sizes.

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
instructions). Generated MTBUF cases use a bound 32_UINT V# and take their data format from the
instruction. The reference decodes elements from the GFX10 format *name* in `tools/gfx10_formats.json`
(channel widths listed from the top bit). UNORM/SNORM values use correctly rounded division.
MUBUF repeats missing channels (*c* mod *n*) and DST_SEL 1 gives integer 1 for UINT/SINT and 1.0f
otherwise (SHADER-86, RX 6700). MTBUF loads return instruction-format channels in order, zero-fill
missing channels and return zero for an INVALID V# format (native gfx1036, wave32 and wave64).
Elements are naturally aligned.

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

#### Hardware oracle for the image fetches (`--oracle hw`)

`--oracle hw` runs the same fetches natively on an AMD GPU through `tools/hw_oracle/` (setup: "Hardware oracle" under
FUZZ-4) and compares every result register of every texel, bit for bit, with the CPU model above;
`--oracle translator,hw` does both on the same textures and says, for each case the hardware disagrees with, what the
translator did. It tests the **model**: whether `image_load` on real hardware returns what the oracle says. Without the
tools it prints `hardware oracle unavailable: ...` and exits 77.

```
python -I tools/image_fuzz.py --oracle hw --programs 2400 --seed 1
python -I tools/image_fuzz.py --oracle hw --programs 180 --seed 1 --negative-control
python -I tools/image_fuzz.py --oracle translator,hw --programs 90 --seed 1
python -m pytest tests/test_image_fuzz_hw.py
```

- **What runs.** One raw compute kernel per case, wave32 and native wave64 alternating. The texture is linear
  (`SW_MODE` 0) device memory the kernel carries behind its own code, 256-byte aligned and found with `s_getpc_b64`, so
  the launchers need no input buffer; rows are padded to a 256-byte pitch and the padding is a pattern that is wrong to
  read (`0xCD`). The kernel builds the T# in `s[8:15]` (the base address from the PC, the other words are literals),
  runs `image_load v[8:..], v[2:3], s[8:15] dmask:.. dim:SQ_RSRC_IMG_2D unorm` for one texel per lane (1,024 lanes
  and a short loop for 64x64) and stores `v[8:11]` for each texel, so the sentinels of the registers `dmask` does not
  write are compared too. A texture that was not 256-byte aligned would make every word `0xBAADF00D`; the run reports
  that as an error instead of reading from a wrong base. The same texels, sizes (8 to 64 per side) and 15 masks as the
  translator run; a seed names the same textures for both.
- **The T# per family** (`hw_oracle/tsharp.py`, exact words in `tests/test_image_fuzz_hw.py`). Facts from Mesa's AMD
  register database and `ac_descriptors.c` and LLVM's `SIDefines.h` (MIT / Apache-2.0; own code, see
  [related_projects.md](related_projects.md)). GFX10 and GFX11 differ here, and GFX12 is a third layout that is not
  built (`--oracle hw` on gfx12xx exits 77 with that reason):

  | | GFX10 (gfx10xx, RDNA1/2) | GFX11 (gfx11xx, RDNA3 and RDNA3.5) |
  | --- | --- | --- |
  | WORD1 `FORMAT` | `[28:20]`, 9 bits | `[27:20]`, 8 bits |
  | `R8G8B8A8_UNORM`, `_UINT`, `_SINT` | 56, 60, 61 | 42, 46, 47 |
  | `R16G16B16A16_SFLOAT`, `R32G32B32A32_SFLOAT`, `R32_SFLOAT` | 71, 77, 22 | 57, 63, 22 |
  | WORD2 `RESOURCE_LEVEL` (bit 31) | 1 ("must be 1") | not there (0) |
  | `MAX_MIP` | WORD5 `[7:4]` on GFX10.3 | WORD1 `[19:16]` |
  | linear pitch | WORD4: `DEPTH` `[12:0]` and `PITCH_MSB` bit 13, GFX10.3 only | same |

  The pitch is always written explicitly (the row size rounded up to 256 bytes); GFX10.1 (gfx1010) has no such field
  and the builder leaves it zero. DST_SEL is X, Y, Z, W, except `R32_SFLOAT`, which reads (X, 0, 0, 1) as the model
  says. `WORD5` carries Mesa's `PERF_MOD` 4, the rest is zero.
- **Negative control.** `--negative-control` runs the cases and then re-judges the same hardware words with seven
  deliberately wrong models (UNORM8 `/256`, red and blue swapped, SINT8 read unsigned, `R32_SFLOAT` alpha 0, dmask
  components not packed, unwritten registers read as 0, texel coordinates transposed) and runs the kernels again with
  the T# `FORMAT` of the next format in the table. It exits 1 unless every one of them mismatches.

Results on a Radeon 880M (gfx1150, RDNA3.5, driver 26.11.2, AMD clang 23.0.0git):

| Run | Cases | Mismatches |
| --- | --- | --- |
| `--oracle hw`, seeds 1, 2, 3, 2,400 cases each (wave32 and wave64, all six formats, all 15 masks, 8 to 64 per side) | 7,200 | 0 |
| `--oracle translator,hw`, seed 1 (Vulkan on the same GPU, subgroup 64) | 12 | 0 (translator and hardware) |

Negative control (`--oracle hw --programs 180 --seed 1 --negative-control`): the unperturbed model mismatches on 0 of
180 cases; the seven wrong models are caught by 16 to 168 of them (`unorm_256` 30, `swap_rb` 144, `sint_unsigned` 30,
`alpha_zero` 16, `dmask_in_place` 132, `no_sentinels` 168, `transpose` 166), and a kernel built with the next format's
`FORMAT` code is caught on every format (24 to 30 of 30 cases each). So a mismatch would have shown.

Finding: none. The CPU model and the hardware agree word for word on all 7,200 cases, including NaN, infinity, signed
zero and denormal inputs (NaN payloads are compared as "any NaN", as in the translator run), the sentinel registers
`dmask` leaves alone, and the `R32_SFLOAT` (r, 0, 0, 1) fill. The translator run on 12 of the same cases agrees too; a
longer `translator,hw` run was not done (the Vulkan side takes about half a minute per case on this machine).

**RDNA3.5 is not RDNA2.** The gfx1150 T# and image unit are GFX11's; a match shows that the model is right for the
texel fetch of these formats on that hardware, which is not the console's. The GFX10 words are built and assemble for
gfx1030 (`tests/test_image_fuzz_hw.py`) but have not run on a GFX10 GPU. The Linux libdrm backend of #1265
(`tools/hw_oracle/drm.py`) was not tried: it copies only `.text` (the texture is in `.text`, so that works in
principle, but it needs the loaded code to be 256-byte aligned, which the kernel checks and reports) and the LDS
section above records a 4096-word output cap there, while a 64x64 case stores 16,384 words. Until that is measured
on a GFX10 card, treat RDNA2 as unverified.

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
- **`mem`**: `buffer_load/store_dword` to `dwordx4` with `offen`, `idxen`, `idxen offen` and offset-only addressing
  (offset-only loads use an SGPR `soffset` that is the slot of the wave's first thread), through V#s built in SGPRs.
  Each run draws a random OOB_SELECT, stride and `num_records` for them (any dword-aligned byte count, so accesses end
  mid-slot and straddle the end), and the model applies the per-mode rule of "Buffer bounds checking" in design.md per
  dword (modes 1 and 2 check once). `s_load_dword/x2/x4` and `s_buffer_load_dword/x2/x4` read what the stores wrote
  from the slot of the wave's first thread (a wave-uniform `soffset`, so no other wave races with the store); the
  `s_buffer_load` V# has its own stride and `num_records` and the model keeps design.md's bound
  (`num_records` bytes, or `num_records * stride`) and ignores OOB_SELECT. Byte/short buffer loads are still not
  generated (the translator rejects them).

`--skip-op OP` (an op name) sets a known finding aside without changing the rest of the program.
`--show SEED:INDEX:WAVE` prints a program and its model.

#### Hardware oracle (`--oracle hw`, `translator,hw`)

`--oracle hw` runs the same programs natively on the AMD GPU through `tools/hw_oracle/` (setup: "Hardware oracle"
above; exit 77 without it) and compares the 16 result VGPRs per thread (and the scratch memory for `mem`) with the
model, in wave32 and in native wave64, 64 threads, so it tests the **model** and not the translator. With
`translator,hw` both run on the same programs. `--negative-control` runs deliberately wrong models (six for `f16`,
eight for `mem`) and fails unless each one is caught.

```
python -I tools/valu_fuzz.py --family mem --oracle hw --programs 2000 --seed 1
python -I tools/valu_fuzz.py --family f16 --oracle hw --negative-control --programs 300
```

*What differs on GFX11 (gfx1150) and how it is handled.* The programs are the same ones the translator gets, so
every form has to mean the same on gfx1013 and gfx1150. Checked against Mesa ACO (`aco_ir.cpp`: `instr_is_16bit`,
`can_use_opsel`, `get_gfx11_true16_mask`), the `gfx11-rsrc.json` register description and by running them:

| Form | GFX10 (gfx1013) | GFX11 (gfx1150) | Handling |
|---|---|---|---|
| f16 VOP2 `_e32`/`_e64` (add, sub, mul, min, max, fmac), `v_cvt_f16_f32`, `v_cvt_f32_f16`, `v_cvt_pkrtz_f16_f32` | low half written, high half kept | same (ACO treats both generations alike; true16 bit 7 of a VGPR selects the half only for v128 and up, which the programs never use) | run |
| `v_fma_f16`, `v_med3_f16`, `v_add_nc_u16`, `v_sub_nc_u16` VOP3 with `op_sel` | `op_sel` source bits, destination bit = high half, other half kept | same | run |
| `v_mul_lo_u16`, `v_lshlrev/lshrrev/ashrrev_b16`, `v_max/min_u16/i16` VOP3 with `op_sel` | same fields | same fields and meaning, but LLVM's gfx11 assembler has no `op_sel` operand for these, and GFX11 renumbered the three shifts (`0x314..0x316` to `0x338..0x33a`) | run: with `op_sel` nonzero the instruction words are written out (`GFX11_VOP3_OPCODE`, checked against llvm-mc in the tests); all-zero `op_sel` uses the plain mnemonic |
| `v_pk_add/mul/max/min/fma_f16`, `v_pk_lshlrev/lshrrev_b16`, `v_pk_ashrrev_i16` with `op_sel`/`op_sel_hi` | per-half selection | same | run |
| `v_pack_b32_f16` | halves moved under IEEE mode 0; under IEEE mode 1 a **signaling NaN** half is quieted (`0x7c96` becomes `0x7e96`; measured on gfx1036, see the RDNA2 results below) | same, gfx1150 quiets it too (measured under IEEE mode 1 only) | run; a signaling-NaN half is "uncertain" in the model (the kernels run with IEEE mode 1) |
| f16 transcendentals (`v_exp/log/rcp/rsq/sqrt/sin/cos_f16`) | write the low half | same | not generated: covered by FUZZ-29 (#1310), which sweeps all 65,536 inputs on the hardware oracle |
| 16-bit integer VOP2 `_e32` forms (`v_add_u16_e32`, ...) | exist, define the whole register (design.md) | **do not exist** (only VOP3, with `op_sel`) | not generated, nothing to compare |
| `v_pk_fmac_f16`, `v_mad_f16`/`v_mac_f16`, `v_subrev_f16` | various | removed or not in this fuzzer | not generated |
| f16 VOP1/VOP2 with VGPR operands v128 and up | whole VGPR | bit 7 selects the high half of v(n-128) (true16) | not generated (the programs use v20..v35, `v0..v127` is what LLVM's fake16 encoding allows too) |
| V# word 1 / word 3 | STRIDE `[29:16]`, SWIZZLE_ENABLE bit 31; FORMAT 7 bits `[18:12]`, RESOURCE_LEVEL bit 24, OOB_SELECT `[29:28]` | STRIDE `[29:16]`, SWIZZLE_ENABLE `[31:30]`; FORMAT 6 bits `[17:12]`, no RESOURCE_LEVEL, OOB_SELECT `[29:28]`, TYPE `[31:30]` = 0 | the hardware kernels build the GFX11 word 3 (`word3()`), the translator programs the GFX10 one; same logical semantics (`--gfx10-word3` runs the old word on gfx1150: it still works, bit 24 is ignored) |
| `buffer_load_dword`, `s_load_dword`, ... | `dword` names | `b32` names | the hardware kernels use the family's spelling (LLVM accepts both) |

*Memory ordering the kernel needs on GFX11 (the model has none).* Measured with `--hw-sync` on 600 to 1,500
programs each: with no wait before an `s_load`/`s_buffer_load` that follows a store, 18 of 300 wave32 programs read stale
data; `s_waitcnt_vscnt null, 0` alone leaves 7 of 300, because an earlier scalar load can have left the line in the
scalar cache, which is not coherent with vector stores; `s_dcache_inv` alone leaves 69 of 1,500 (the store had not
left the pipe); both together give 0 mismatches over 20,000 programs, and adding `buffer_gl0_inv` and `buffer_gl1_inv`
changes nothing (a vector load after a store of the same thread needs neither). The default (`--hw-sync scalar`) is
vscnt wait plus `s_dcache_inv`; vector accesses only need `s_waitcnt vmcnt(0)`.

*Results (Radeon 880M, gfx1150, driver 26.11.2, AMD clang 23.0.0git).* Hardware, 5 seeds (101 to 105) x 2,000 programs x wave32 and native wave64 per family: **`mem` 20,000 of 20,000 match, `f16` 20,000 of 20,000 match, 0 mismatches, 0 assembler rejections** (plus earlier development runs of a further 15,000 `mem` and 12,000 `f16` programs; the one mismatch they produced, the `v_pack_b32_f16` NaN quieting above, is now modelled as uncertain). Negative controls (`--negative-control --programs 500`, 1,000 programs per row, unperturbed model: 0 mismatches): `f16` hi half zeroed 994, `op_sel` destination ignored 863, packed halves swapped 864, `v_min` as `v_max` 424, `v_pack_b32_f16` halves swapped 254, `op_sel` of the encoded 16-bit integer ops ignored 862; `mem` OOB_SELECT ignored 506, mode 0 without the stride check 54, index unchecked 364, soffset left out of the check 69, whole-access instead of per-dword 45, out-of-range loads keeping the old value 590, `s_buffer_load` unbounded 209, its stride ignored 81. Every perturbation is caught. Not run here: `translator,hw` for `mem` (see below); RDNA2 is in the next paragraph.

*Results on RDNA2 (gfx1036 integrated GPU, AMD Windows driver 32.0.21044.64, HIP backend, AMD clang 23.0.0git, `main` bb2e1842).*
- **Hardware, same protocol** (seeds 101 to 105 x 2,000 programs x wave32 and native wave64 per family): **`f16` 20,000 of 20,000 and `mem` 20,000 of 20,000 match, 0 mismatches, 0 assembler rejections.** On a gfx1030, ROCm 6.2's clang 19 rejected 3,768 of 4,000 `f16` programs (#1365); clang 23 rejects none.
- **Negative controls**, in the order above: `f16` 992, 865, 826, 430, 280 and 879 caught; `mem` 514, 57, 354, 60, 37, 580, 214 and 89 caught. The unperturbed model gives 0 mismatches.
- **`translator,hw` at seed 3** (100 programs, the same GPU through the AMD Vulkan driver): `mem` 200 of 200 match, `f16` 199 of 200. The one difference, `s3_22_w64`, is a driver mismatch of the #1506 kind. The hardware and `tests/semantic/spirv_mini_interp` both give the model's value for it, and the same emulated SPIR-V matches on NVIDIA.
- **`v_pack_b32_f16` with signaling NaNs**, measured with hand-written raw kernels in both wave sizes and both IEEE modes. The inputs were five signaling-NaN halves, quiet NaNs, ±0, +inf and a denormal, in either half, with `op_sel` and with `neg`.
  - IEEE mode 1, the assembler's default: a signaling-NaN half is quieted. Bit 9 is set and the sign and payload are kept, as on gfx1150.
  - IEEE mode 0: the half is moved unchanged.
  - `neg` flips the sign before any quieting (`0x7c01` gives `0xfe01` under mode 1 and `0xfc01` under mode 0).
  - Every other input is moved unchanged.
  - The translator moves the bits, which is exact under IEEE mode 0. Which mode the game's shaders run in is not recorded here.

*Findings.* The OOB_SELECT per-mode rules, the per-dword straddle rule and the `s_buffer_load` bound of design.md hold
on gfx1150 (they were measured on RDNA2 and, for the vector part, on this GPU before; the `s_buffer_load` bound and the
`idxen offen` and offset-only soffset forms are new on gfx1150). The one difference is `v_pack_b32_f16` quieting
signaling NaNs (only signaling NaN halves are not judged; a quiet NaN is compared). The default translator path of `mem` ran on the Radeon 880M (`VK_LOADER_DRIVERS_SELECT=*amd*`, `vk_compute_test` built with g++ as in the lab script): `--family mem --programs 200 --seed 1` gave 400 of 400 programs (native wave64 and wave32) matching the model, 0 not translated, 0 harness errors, so the translator agrees with the model on the OOB_SELECT, straddle, `s_load` and `s_buffer_load` rules there. On the RTX 5070 `vk_compute_test` loses the device (-4) for this and for the unchanged generator of `main`, so use the AMD device.

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
- **Hardware oracle (optional).** `--oracle hw` (or `translator,hw`) also runs every program natively on the AMD
  GPU through `tools/hw_oracle/` (setup: see the FUZZ-4 section above) and compares all stored words with the model;
  without the tools it prints `hardware oracle unavailable: ...` and exits 77. It tests the **model**, in wave32
  and in native wave64 (`.amdhsa_wavefront_size32 0`, wave64 code objects are assembled with
  `-mwavefrontsize64`), with the program's real group size (partial waves, two or more waves) and its EXEC setup,
  one thread per lane, each storing its 14 words with `global_store_b32`. Branch placeholders become labels. After
  every instruction the native kernel has `s_nop 7`, because the model has no hazards and the real unit wants wait
  states after some VALU writes of SGPRs, VCC and EXEC; `--no-hw-guard` leaves them out (a 4,000-program run
  without them matched too, so the generator does not reach a hazard on this GPU). With `translator,hw`, a program
  where the hardware differs from the model also prints what the translator did with it.
  `hyp_confirmed` counts programs with a hypothesis op that matched on the hardware, split into `hyp_permlane`
  (FI = 0 source lane off in EXEC keeps vD; the claim comes from the oracle measurements in #374 on RDNA2 and
  RDNA3.5, and #580 models it in the translator) and `hyp_dpp` (DPP8 source lane off in EXEC with fi = 0 reads 0;
  the claim is #238's, and nothing measured it before this oracle). On the hardware path a hypothesis that fails
  is a falsified claim: it counts as a mismatch (exit 1) and in `hyp_only`, and `--minimize` marks the
  hypothesis ops that remain. A startup probe checks that a wave64 kernel runs as wave64 (lane 32 reads 32) and a
  wave32 kernel as wave32, so a runtime that ignored the wave size would fail loudly. `--keep` writes the kernel
  body that ran.
  GFX11 differences looked at, none of which changes a result for what the generator emits: the assembler takes
  the same `row_share`/`row_xmask`/`dpp8`/`fi`/`bound_ctrl:1` and `v_permlane16/x16` spellings for gfx1030 and
  gfx1150; `row_newbcast`, `row_bcast` and `wave_shl` exist on neither (not generated); `v_cmpx_*_e32` writes EXEC
  only on both (the generator never uses the `_e64` form, whose GFX10 SDST the assembler no longer takes either);
  `v_mbcnt_*` is VOP3-only on both; `s_ff1_i32_*` is the GFX10 name of GFX11's `s_ctz_i32_*` (an accepted alias);
  `v_readlane_b32` with a lane select of 32 and up in wave32 uses select bits [4:0] on gfx1150 too. Nothing
  needed a skip, so `hw_skip_reason` returns none and the only skips are assembler rejections (none seen). That
  was gfx1150.
  On RDNA2 (gfx1036, through the Linux launcher of the FUZZ-4 section): 24,000 programs (seeds 1 to 6) match in
  both wave sizes, 12,000 each, 6,586 of them `hyp_permlane`; with `--dpp` (seeds 1 and 2) 8,000 more match,
  972 of them `hyp_dpp`. As a control, `--permlane-fetch-inactive` there mismatches in 239 of 1,000 programs. A
  GFX10 assembler has no `global_store_b32`, so `asm.raw_kernel_text` puts the GFX10 name (`global_store_dword`)
  back for `gfx10xx`; nothing else the generator emits needed a change.
  First results (Radeon 880M, driver 26.11.2): 24,000 programs on the hardware, half wave32 and half native
  wave64, seeds 11 to 16 (4,000 programs each), with and without `--dpp`, 12 to 60 ops, with and without
  `--no-maskdata` and `--no-hw-guard`: match 24,000, mismatch 0, hyp_only 0, rejected 0. 7,652 of them contained
  a permlane with an inactive source lane or a DPP8 with an inactive source lane (`hyp_confirmed`: 3,567 in
  wave32, 4,085 in wave64) and every one matched, so both hypotheses hold on gfx1150 in both wave sizes. RDNA2
  itself is not measured by this run.
  Against the translator (`translator,hw`, 200 programs, seed 7): every translator mismatch of this build had
  masks read as plain data (#293); with `--no-maskdata` it has 0 mismatches and only the known permlane
  `hyp_only` cases (#580). The hardware agrees with the model on both, so those are translator gaps, not model
  errors.
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

### 32-bit float VALU on the GPU (`tools/f32_fuzz.py`, FUZZ-3)

`--oracle hw` (or `translator,hw`) runs the same seeded VOP1/VOP2/VOP3 f32 programs natively on the AMD GPU and judges
the hardware's `v8` and VCC against `F32Model` bit for bit (NaN matches any NaN; an f16 NaN in a `v_cvt_f16_f32` /
`v_cvt_pkrtz_f16_f32` half too). Setup is the one in "Hardware oracle" above; without it the run prints
`hardware oracle unavailable: ...` and exits 77. It tests the **model**, on wave32 with one lane; with `translator,hw`
a hardware mismatch also prints what the translated SPIR-V did.

```
python -I tools/f32_fuzz.py --oracle hw --programs 5000 --seed 1            # table + informational hardware tables
python -I tools/f32_fuzz.py --oracle hw --negative-control                   # wrong models must be caught
python -I tools/f32_fuzz.py --oracle hw --perturb mul_rtz --programs 500     # one wrong model, mismatches must appear
```

- **Float MODE.** An assembled kernel's descriptor starts with IEEE 1, DX10_CLAMP 1, f32 denormals flushed
  (`.amdhsa_float_denorm_mode_32 0`), f16/f64 denormals kept (3), round to nearest (MODE `0x3c0`, read back with
  `s_getreg_b32`). The primary mode, `flush`, is the state inferred for the game's shaders (IEEE off, f32 denormals
  flushed, DX10_CLAMP on: MODE `0x1c0`). Three more are run for what the model does not judge: `keep` (f32 denormals
  kept, `0x1f0`), `no_dx10` (`0x0c0`), `ieee` (`0x3c0`). The run reads MODE back in a kernel for each and fails if a
  directive did not take. Rounding is always to nearest even. On gfx1170+ (GFX11.7, GFX12) IEEE and DX10_CLAMP are
  fixed and AMD clang rejects `.amdhsa_ieee_mode` / `.amdhsa_dx10_clamp` ("directive unsupported on gfx1170+"), so
  `asm.mode_skip_reason` reports a kernel that sets either as not runnable there and the runner never assembles it;
  the MODE probe then fails naming that reason instead of a whole chunk failing to assemble (FUZZ-43).
- **GFX11 against GFX10, handled explicitly** (`tools/hw_oracle/asm.py`, checked by assembling every opcode for gfx1013,
  gfx1010, gfx1030, gfx1100, gfx1150 and gfx1200):
  - *Skipped, never substituted:* `v_mad_f32`, `v_mac_f32`, `v_madak_f32` and `v_mad_legacy_f32` exist on GFX10.1 only
    (gfx1010/1011/1013, the console's family); gfx1030 and every GFX11 part reject them. The fused replacements
    (`v_fma_f32`, `v_fmac_f32`, `v_fmaak_f32`, `v_fma_dx9_zero_f32`) round once, so they are different instructions. The
    unfused mad semantics (and SHADER-118's flush) therefore stay unmeasured: no RDNA2 desktop part and no GFX11 part has
    them. Also not generated: `v_madmk_f32` (GFX10.1 only, like the mad opcodes), `v_min_legacy_f32` (in no GFX10 or GFX11
    part: llvm-mc rejects it for gfx1010, gfx1013, gfx1030 and gfx1150), `v_fma_legacy_f32`/`v_fmac_legacy_f32` (new in
    GFX10.3, not in gfx1013).
  - *Renamed, run:* `v_mul_legacy_f32` is `v_mul_dx9_zero_f32` in GFX11 (the DX9 rule `0 * x = 0`, inf and NaN included,
    held in every program) and `v_cvt_pkrtz_f16_f32` is `v_cvt_pk_rtz_f16_f32`; the assembler takes the old spelling.
  - *Same, with a syntax change:* GFX11's assembler wants `clamp` before `mul:N`/`div:2`; the native kernels print them in
    that order (the same bits). Opcode numbers differ between the families (`v_cmp_class_f32_e32` is `7d10` on gfx1013 and
    `7cfc` on gfx1150); only mnemonics are shared. omod and clamp exist in VOP3 on both; no op_sel is used for f32.
    The transcendental unit is separate on GFX11, so a `s_nop 7` follows the instruction in the native kernel
    (`--no-hw-guard` leaves it out).
  - *Not covered:* wave64 (the f32 ALU does not depend on it), `v_fma_mix*`, f16 and f64 arithmetic.
- **Programs.** The seeded random ones, plus `--probes N` (default 12) per op and kind: denormal operand, denormal
  result, NaN payloads, clamp of NaN/-0, invalid operations (inf - inf, 0 * inf, ...), and `cancel` (`z = -round(x * y)`,
  which separates fused from unfused). `v_min/max/min3/max3/med3_f32` are in the table for the hardware only, with the
  RDNA2 rules measured in #634.
- **Judged / not judged.** A program is *match* or *mismatch* where the model has an answer. Where it declines (denormal
  operand or result, clamp of NaN or -0, cube of NaN) the hardware still runs and is recorded in the informational tables,
  also under the other three modes. A *known deviation* is hardware behaviour already analysed and written down as a model
  alternative (`KNOWN_DEVIATIONS`); the hardware words must equal the alternative exactly, anything else is a mismatch.

Results (Radeon 880M, gfx1150, driver 26.11.2, AMD clang 23.0.0git), seeds 1 to 6, 5,000 random programs + 24 probes per
op and kind each, 46,967 programs in all: **30,577 match, 0 mismatch, 57 known deviation**, 5,417 skipped for GFX11,
5,903 denormal, 5,013 other not judged (clamp of NaN or -0, cube of NaN). "Match" accepts any NaN for any NaN: payload, sign and quiet bit are in the informational tables, not judged. Seed 1 by family:

| Family | Programs | Match | Mismatch | Known dev | GFX11 skip | Denormal | Other |
|---|---|---|---|---|---|---|---|
| add / sub / mul | 921 | 631 | 0 | 0 | 0 | 176 | 114 |
| DX9 legacy mul / mad | 453 | 133 | 0 | 0 | 239 | 51 | 30 |
| mad / mac / madak | 698 | 0 | 0 | 0 | 698 | 0 | 0 |
| fma / fmac / fmaak | 700 | 509 | 0 | 0 | 0 | 132 | 59 |
| int to f32 convert | 708 | 708 | 0 | 0 | 0 | 0 | 0 |
| f16 convert | 478 | 383 | 0 | 0 | 0 | 90 | 5 |
| `v_cvt_pk_u8_f32` | 183 | 130 | 0 | 0 | 0 | 53 | 0 |
| floor / ceil / trunc / rndne | 795 | 582 | 0 | 5 | 0 | 96 | 112 |
| rcp / rsq / sqrt / exp / log | 956 | 704 | 0 | 0 | 0 | 119 | 133 |
| cube | 786 | 435 | 0 | 1 | 0 | 104 | 246 |
| `v_cmp_class_f32` | 171 | 147 | 0 | 0 | 0 | 24 | 0 |
| min / max / med3 (hardware only) | 989 | 732 | 0 | 1 | 0 | 128 | 128 |

Negative control: `--negative-control` re-judges the same hardware words with a deliberately wrong model for six ops
(`v_add_f32` low bit flipped, `v_mul_f32` rounding toward zero, `v_fma_f32` unfused, `v_mul_legacy_f32` as IEEE multiply,
`v_rndne_f32` as floor, `v_min_f32` NaN rule changed); every one produced mismatches (1 to 112 per run; the `v_fma_f32` one only after the `cancel` probes were added, a plain random sweep missed it in seed 2;
the legacy multiply is the weakest, 1 to 4 in seeds 1 to 6, because `0 * inf` is rare in random programs; the `zero_inf` probes (a zero against inf, NaN or a normal number, DX9 multiply) were added afterwards and catch it at small sizes). The `cancel` probes (seeds 2, 7, 8) give 72 of 72 fused-FMA matches. Seeds 1 to 6 ran without them.

On RDNA2 (gfx1036, through the Linux launcher of the FUZZ-4 section, the same seeds 1 to 6): 38,858 programs,
**28,671 match, 0 mismatch, 56 known deviation**, 4,224 skipped (the four mad opcodes are GFX10.1-only), 3,097
denormal, 2,810 other not judged. The four MODE settings read back in the kernels as `0x1c0`, `0x1f0`, `0xc0` and
`0x3c0`, so the float mode of the kernel descriptor reaches the hardware through that launcher as well.
`--negative-control` catches all six wrong models there (3 to 25 mismatches each).

The same seeds through the Windows HIP launcher (`hw_oracle.c`) on the integrated GPU of a Ryzen 7 9800X3D (gfx1036;
Windows 10 22H2, AMD driver 32.0.13036.4, `amdhip64_6.dll`, AMD clang 23.0.0git from rocm-sdk-core 10.0.0, main at
41f44826) give the same counts in every column: 38,858 programs, **28,671 match, 0 mismatch, 56 known deviation**
(all `omod_neg_zero`), 4,224 skipped, 3,097 denormal, 2,810 other. `--negative-control` again catches all six (3 to
25 mismatches). Two launchers on two machines and two operating systems agree on the hardware words.

`v_rcp_iflag_f32` (SHADER-136; added to the rcp / rsq / sqrt / exp / log family after the run above, so it is not in
its counts) on the same gfx1036 through the Windows launcher, `--ops v_rcp_iflag_f32 --programs 1000 --probes 24`,
seeds 1 to 6: 6,552 programs, **5,971 match, 0 mismatch, 35 known deviation** (`omod_neg_zero`), 279 denormal, 267
other. It shares `v_rcp_f32`'s model, and the clamp and omod bits behave as on the other ops (summed over the six
seeds, per MODE setting flush / keep / no_dx10 / ieee): clamp makes a -0 result +0 in every mode (98 each) and a NaN
result +0 except with DX10_CLAMP off, where the NaN passes (169); omod is applied with IEEE off, including with f32
denormals kept (461 each; like the other transcendental ops), and ignored with IEEE on (461); a -0 result becomes +0
under omod (35 each, none with IEEE on); a signalling NaN passes unchanged with IEEE off and is quieted with IEEE on
(59); denormal operands read as zero and denormal results flush in every mode.

What the hardware answered where the model does not judge (gfx1150; the RDNA2 run above agrees where it is cited):

- **No model error found.** The model was right wherever it had an answer, in 30,577 programs; the min/max/med3
  rules measured on RDNA2 in #634 (one NaN gives the other operand, both NaN give src0, -0 < +0, med3 with a NaN is min3)
  also hold on gfx1150 for quiet and signalling NaNs with IEEE off.
- **On gfx1150 output modifiers turn a -0 result into +0** (`mul:2`, `mul:4`, `div:2`; 57 programs, every seed; a flushed
  denormal result, which is a -0, too). RDNA2 does the same: 56 programs in the same seeds on gfx1036. LLVM notes that omod mishandles signed zeros
  (`SIFoldOperands.cpp`) and D30209 describes -0 being forced to +0 before the modifier, which points the same way. The
  model and the translator both keep -0, so on that evidence the translator likely differs from the hardware
  (`translator,hw`: all 6 deviations of 2,000 programs are "matches the model, so it shares the model's answer"). Not
  fixed in this change; the finding is posted in the coordination issue (see #1006). A deviation is only accepted where
  the omod was active (not with IEEE on, not for ALU ops with f32 denormals kept).
- **omod only acts with MODE.IEEE = 0**, and for the ALU ops only with f32 denormals flushed: with `keep` the omod of
  add/mul/fma/min/max/cvt/rounding ops is ignored; the transcendental ops and the cube ops still apply it. LLVM says the
  same for both conditions (`SIFoldOperands.cpp`: "omod is ignored by hardware if IEEE bit is enabled"; "If output
  denormals are enabled, omod is ignored"). This supports
  the corpus inference (omod on 102,365 float instructions, so IEEE off and f32 denormals flushed in the game's shaders).
- **f32 denormals (SHADER-35's MODE question).** With `denorm_mode_32 = 0` a denormal operand is read as a zero of its sign
  and a denormal result is written as a zero of its sign, for add/sub/mul/fma/fmac/fmaak, min/max/med3, rounding ops and
  DX9 mul; with `3` they are exact. `v_rcp/rsq/sqrt/exp/log_f32` flush in both modes (as #634 measured on RDNA2).
  `v_cmp_class_f32` classifies a denormal as a denormal in both. `v_cubesc/cubetc` pass a denormal operand through.
- **clamp.** With DX10_CLAMP 1, clamp of a NaN gives +0; with 0 the NaN passes through. clamp of -0 gives +0 in every mode.
  The model still declines both.
- **NaN payloads.** With IEEE off a signalling NaN passes through add/mul/fma/rounding/transcendental ops unchanged
  (payload and sign); with IEEE on it is quieted (bit 22 set). With several NaN operands the first (lowest source) wins
  in add/mul/fma/fmaak/fmac; for min/max the rule above applies. `v_sub_f32` returns the NaN of src1 with its sign flipped
  (it computes src0 + -src1). Invalid operations (inf - inf, 0 * inf, sqrt/rsq/log of a negative) give `0xffc00000` in every
  mode and op, never `0x7fc00000`. `v_cvt_f16_f32` and `v_cvt_pkrtz_f16_f32` turn a NaN into an f16 NaN with the top 10
  payload bits and the quiet bit set (not the canonical `0x7e00` the model writes; payload not judged, quietness is).
  `v_cvt_f32_f16` keeps the f16 payload shifted left by 13 and does not quiet it (the model agrees).


### f16 transcendentals on the hardware (`tools/f16_trans_sweep.py`, FUZZ-29)

The `16-bit results` row of `docs/design.md` listed `v_exp_f16`/`v_log_f16` and the NaN and denormal operands as
unverified. f16 has only 65,536 bit patterns, so instead of sampling, this tool runs every one of them through
`v_exp_f16`, `v_log_f16`, `v_rcp_f16`, `v_rsq_f16`, `v_sqrt_f16`, `v_sin_f16` and `v_cos_f16` on the hardware
oracle (setup above; exit 77 without it). Each op is 8 raw kernels of 1,024 threads. Each thread computes 8 inputs
and stores them packed two to a dword, which is 4,096 words per kernel, the launcher's limit. The op runs in wave32
and wave64 and under five MODEs, and MODE is read back with `s_getreg_b32` first:

| Mode | IEEE | f16 denormals | f16 rounding |
|---|---|---|---|
| `keep16` | 0 | kept (3) | nearest |
| `flush16` | 0 | flushed (0) | nearest |
| `ieee_keep16` | 1 | kept | nearest |
| `ieee_flush16` | 1 | flushed | nearest |
| `keep16_rtz` | 0 | kept | toward zero |

All modes have DX10_CLAMP 1 and f32 denormals flushed.

```
python -I tools/f16_trans_sweep.py                                  # all ops and modes, about 5 minutes
python -I tools/f16_trans_sweep.py --ops v_exp_f16 --modes keep16 --json out.json
python -I tools/f16_trans_sweep.py --self-test                      # the reference, no GPU
```

The reference is the exact result correctly rounded to f16, computed with `decimal` at 60 digits: for sin/cos that is
sin/cos(2πx), reduced in turns. That reference is the translator's rule (widen, the GLSL.std.450 function in f32,
round to nearest even) with an exact f32 function, the best a driver can do. Inputs where the translated SPIR-V is
undefined are tabulated, not judged. For each mode the tool counts:
- bit-exact results, 1 ulp and more than 1 ulp;
- denormal operands that give the hardware's own word for a zero of the same sign, and denormal results written as zero;
- NaN results, with signalling NaNs kept, quieted or changed.

It also counts the inputs on which each mode differs from `keep16`, and lists the word for 23 special inputs. For
sin/cos a second table compares the hardware with Sin/Cos of the f32 product 6.2831855 · x, by |x| band.

Result on an RX 6700 XT (gfx1031, Navi 22; the Linux launcher from #1265, Linux 7.3.5, amdgpu, clang 23.1.5).
wave32 and wave64 gave the same word for every input of every op in every mode. There was no GPU fault, timeout or
reset in the kernel log.

| Op (`keep16`) | Inputs that are not NaN | Bit-exact | 1 ulp | Undefined in SPIR-V |
|---|---|---|---|---|
| `v_exp_f16` | 63,490 | 63,490 | 0 | 0 |
| `v_log_f16` | 63,490 | 31,744 | 0 | 31,746 (≤ 0) |
| `v_rcp_f16` | 63,490 | 63,488 | 0 | 2 (±0) |
| `v_rsq_f16` | 63,490 | 31,744 | 0 | 31,746 (≤ 0) |
| `v_sqrt_f16` | 63,490 | 31,746 | 0 | 31,744 (< 0, -0 excluded) |
| `v_sin_f16` | 63,490 | 63,474 | 14 | 2 (±inf) |
| `v_cos_f16` | 63,490 | 63,484 | 4 | 2 (±inf) |

- **Correctly rounded.** The five non-trigonometric ops never differ from the correctly rounded result. sin/cos differ
  by 1 ulp on 14 and 4 inputs, all with |x| ≤ 1. For |x| > 1 every result is bit-exact, up to 65,504 turns: the f16
  ops have no ±256 domain limit.
- **The f32 product loses the turn.** Against Sin/Cos(f32 6.2831855 · x), which is what the translator computes before
  #594 is fixed, 16,366 sin and 2,036 cos inputs are more than 1 ulp off, up to 9,944 ulp. Nearly all of them have
  |x| > 1 (sin: 1,010 inputs at 1 < |x| ≤ 256 and 15,354 above 256). Reducing in turns first, as #1283 and #616
  propose, is what the hardware does; for f16 the reduction must not stop at ±256.
- **f16 denormals follow `denorm_mode_16_64`, for operands and results alike.** With 0, a denormal operand gives the
  word of a zero of its sign (`v_log_f16` -inf, `v_rcp_f16` and `v_rsq_f16` ±inf, `v_sqrt_f16` and `v_sin_f16` ±0).
  Denormal results become ±0 (831 for `v_exp_f16`, 4,094 for `v_rcp_f16`). With 3, both are kept. The f32
  transcendentals behave differently: they flush in both modes.
- **The f16 field of FP_ROUND is ignored.** Under round toward zero every op gives the same word as under round to
  nearest, on all 65,536 inputs.
- **IEEE only changes signalling NaNs.** With IEEE 0, all 1,022 f16 signalling NaNs pass through unchanged. With IEEE 1
  they are quieted (bit 9 set) with payload and sign kept. Quiet NaNs pass through unchanged in every mode. This is the
  f32 behaviour of the previous section.
- **Undefined in SPIR-V, defined on the hardware.** log(±0) = -inf; rcp(±0) = ±inf; rsq(+0) = +inf; rsq(-0) = -inf;
  sqrt(-0) = -0. log, rsq and sqrt of a negative (-inf included), and sin/cos(±inf), give `0xfe00`, the negative quiet
  NaN (`0xffc00000` in f32).
- **Controls.** `keep16_rtz` also judges against a round-toward-zero reference, so it shows what a wrong rounding
  model produces: about half the results 1 ulp off. The kernel layout is checked without a GPU: each of the 65,536
  inputs is computed once and stored in its own half (`tests/test_f16_trans_sweep.py`). On a GPU the test checks
  exp2 of integers and wave32 = wave64.

### `v_fma_mix*` f16 rounding in nearest mode (`tools/fma_mix_probe.py`)

`docs/design.md` leaves one f16 question open. In round-to-nearest mode, does `v_fma_mixlo_f16` / `v_fma_mixhi_f16`
round the exact x · y + z once, straight to f16? Or does it round to f32 first and then narrow, which is what the
translator does (f32 `Fma`, then `f16_res`)? The RX 6800 word behind the toward-zero rule can't answer this. For
1 · 2 - 2^-30 both readings give 0x4000 in nearest mode, and toward zero two truncations equal one.

The two readings only differ when the f32 rounding lands exactly on an f16 tie, or moves a value off one. The probe has
seven such inputs, each one f16 ulp apart under the two rules, plus one control:
- a tie just above and just below, for both signs;
- the subnormal tie at 2^-25;
- the overflow tie at 65520 (inf against 65504);
- a tie made by the product rather than by z;
- the same with f16 operands (`op_sel_hi:[1,1,1]`).

Every side reports which rule its words follow:

```
python -I tools/fma_mix_probe.py                       # the translator, on the CPU interpreter
python -I tools/fma_mix_probe.py --oracle hw           # also the native instruction on an AMD GPU (hardware oracle)
python -I tools/fma_mix_probe.py --oracle vulkan --build runtime/build   # also the translated shader on this GPU
```

The `hw` side is the measurement. It runs one wave32 lane per input, under the MODE inferred for the game: IEEE 0,
DX10_CLAMP 1, f32 denormals flushed, f16 denormals kept, nearest. The `vulkan` side only shows that a driver runs the
translator's sequence as written. Its operands come in user SGPRs, so the driver can't fold them. Exit 0 means every
side follows the same rule. Exit 1 means they differ, or a side follows neither rule. Exit 77 means a requested side
is unavailable.

`tests/test_fma_mix_probe.py` checks the expected words against a derivation written out by hand. It also checks that
the translator follows the single rounding rule, and that the compute shader gives the pixel shader's words on the
interpreter. Hardware measurement on AMD RDNA2 confirmed that the hardware rounds once, straight to f16, so the translator
uses `Gen.fma_mix_odd` in nearest mode as well. Round-to-odd at f32 then narrows exactly, giving the single-rounded word on
all seven discriminating inputs.

Results:

| GPU | Driver | `translator` | `vulkan` | `hw` |
|---|---|---|---|---|
| NVIDIA GeForce RTX 5060 (Blackwell), Windows 11 | 621.24 | double | double, 8 of 8 words | not AMD |
| AMD Radeon iGPU (RDNA2, gfx1036), Windows 11, HIP backend | 32.0.21044.64 | double | not run | **single**, 7 of 7 ties; control `0x4100` |

RDNA2 rounds once. Every tie input gives the single-rounded word (A `0x3c01`, B `0x3c01`, C `0xbc01`, D `0x0001`,
E `0x7bff`, F `0x3c01`, G `0x3c01`), so the translator is one f16 ulp off on exactly these inputs (SHADER-135; the
translator is not changed here). On the RTX 5060, a translator copy changed to use
`fma_mix_odd` in nearest mode gave the single-rounded word for all seven inputs on the `vulkan` side as well.
| NVIDIA GeForce RTX 5060 (Blackwell), Windows 11 | 621.24 | single | single, 8 of 8 words | not AMD |
| AMD Radeon gfx1036 (RDNA2), Linux / Windows | | single | single, 8 of 8 words | single, 8 of 8 words |

Measurement on AMD RDNA2 (gfx1036) confirmed single rounding across all 7 discriminating cases (finding 7079901079).
Using `fma_mix_odd` produces the exact hardware single-rounded words in both nearest-even and toward-zero modes.

### f16 output modifiers, clamp and `v_fma_mix*` operands on RDNA2 (SHADER-135)

*Measured on an RDNA2 iGPU (gfx1036), AMD Windows driver 32.0.21044.64, HIP backend, AMD clang 23.0.0git, `main`
a0acd33d.* Raw kernels written by hand, one lane each. The MODE comes from the kernel descriptor (`ieee`,
`dx10_clamp`, `denorm_mode_32`, `denorm_mode_16_64`). Every program writes one word, starting from a destination of
`0x5a5a5a5a`, so a half the op must not write shows up. 1,385 programs assembled, each run in wave32 and wave64, and
the two wave sizes agree on every word. `tools/fma_mix_probe.py --oracle hw` (the section above) and
`tools/f16_trans_sweep.py --ops v_exp_f16,v_log_f16` ran on the same GPU.

- **An omod on an f16 result acts only with IEEE 0 and f16 denormals flushed (`denorm_mode_16_64 = 0`).** In that
  MODE `mul:2`, `mul:4` and `div:2` scale the results of `v_add_f16`, `v_mul_f16`, `v_fma_f16`, `v_fmac_f16_e64`,
  `v_cvt_f16_f32_e64`, `v_exp_f16_e64` and `v_rcp_f16_e64` (`v_fmac_f16_e64` 0.75 · 1 + -0 `mul:4` gives `0x4200`).
  With `denorm_mode_16_64 = 3` or IEEE 1 they leave the result as it is (`0x3a00`), and `denorm_mode_32` makes no
  difference. The f32 control (`v_add_f32_e64 mul:2`) follows `denorm_mode_32` instead, as on gfx1150 above. Unlike
  the f32 transcendentals there, `v_exp_f16` and `v_rcp_f16` obey the same rule as the ALU ops. When the omod acts, a
  -0 result becomes +0 (`v_fmac_f16` -0 · 1 + -0 `mul:2` gives `0x0000`), and it comes before the clamp
  (`v_cvt_f16_f32_e64` 0.75 `clamp mul:2` gives `0x3c00`, not `0x3e00`).
- **clamp** on `v_cvt_f16_f32_e64`, `v_fmac_f16_e64`, `v_fma_mixlo_f16` and `v_fma_mix_f32` clamps to [0, 1] in every
  MODE: -0.5 and -0 give +0, 3.0 and an overflow (1e5, 65504 · 2) give 1.0. A NaN gives +0 with DX10_CLAMP 1. With
  DX10_CLAMP 0 it stays the NaN the op gives without clamp. This is SHADER-186's f32 rule, here for the f16 results and
  the mix ops.
- **Constants in `v_fma_mix*` operands.** A float inline constant is the f32 constant where `op_sel_hi` makes the
  operand f32, and the f16 constant with a zero high half where it makes it f16. `1.0`, `-1.0`, `0.5`, `2.0` and
  `-4.0` give their value either way. `0.15915494` gives `0x3e22f983` as f32 and `0x3118` as f16 (`0x3e230000` once
  widened). With `op_sel_hi = 1` and `op_sel = 1`, every float constant reads +0.0. An integer inline constant is its
  32-bit integer. As an f32 operand, `1` and `64` are f32 denormals (+0 with `denorm_mode_32 = 0`) and `-1` is the NaN
  `0xffffffff`. As an f16 operand, `op_sel` picks a half: `1` gives `0x0001` low and +0 high, and `-1` gives `0xffff`
  in both. `op_sel` is ignored for an f32 operand. For gfx10, llvm-mc rejects a 32-bit literal in these ops ("invalid
  operand"). It encodes a literal equal to an f16 inline constant, such as `0x3c00`, as that constant (`0xf2`).
- **Mix-op denormals follow the mode of their own width.** An f16 operand (`op_sel_hi = 1`) and an f16 result
  (`mixlo`/`mixhi`) become a zero of their sign with `denorm_mode_16_64 = 0`, and are kept with 3, whatever
  `denorm_mode_32` says. An f32 denormal is flushed with `denorm_mode_32 = 0` and kept with 3.
- **Mix-op NaNs.** An f16 NaN operand widens with its payload (`0x7c01` → `0x7f802000`, `0xfe01` → `0xffc02000`).
  `v_fma_mix_f32` keeps a signalling NaN, f16 or f32, with IEEE 0, and quiets it with IEEE 1, payload kept
  (`0x7f800001` → `0x7fc00001`). `v_fma_mixlo_f16` quiets every NaN in every MODE and keeps the top payload bits
  (`0x7c01` → `0x7e01`, `0x7d55` → `0x7f55`, f32 `0x7fa00000` → `0x7f00`, f32 `0x7f800001` → `0x7e00`).
- **`v_exp_f16` and `v_log_f16` give gfx1031's counts exactly**, on all 65,536 inputs in the five MODEs of the f16
  transcendentals section: 0 ulp on every input that is not a NaN, with f16 denormals kept. With them flushed, 831
  `v_exp_f16` results and 1,023 `v_log_f16` operands become zero. Signalling NaNs are quieted only with IEEE 1, and
  wave32 = wave64.

**The translator against these words.** Each program was translated as a pixel shader and run on
`tests/semantic/spirv_mini_interp.py`, then compared with the hardware under the MODE the translator assumes for the
game (IEEE 0, DX10_CLAMP 1, f32 denormals flushed, f16 denormals kept). It gives the same word for the float
constants in every `op_sel`/`op_sel_hi` combination, for the integer constants as f16 halves, for clamp on values that
are not NaN, and for the f16 denormals. It differs in five ways, none fixed here:

1. **f16 omod.** The translator applies `mul:2`/`mul:4`/`div:2` to every f16 result. The hardware ignores them in this
   MODE (24 words: `v_add/mul/fma/fmac/exp/rcp_f16`). `v_cvt_f16_f32` drops its omod and clamp today, so its omod
   matches by accident; #718 would make it apply. The two assumptions can't both hold for the game. Either its shaders
   run with f16 denormals flushed, and keeping them is wrong, or every f16 omod in them does nothing. A corpus count
   of omod on f16 instructions would decide, since a compiler that emits one expects it to act (LLVM folds an omod only
   when "output denormals" are off, the `SIFoldOperands.cpp` comment quoted above). That count needs the game files.
2. **clamp of a NaN.** The interpreter refuses `FClamp` with a NaN operand, which GLSL.std.450 leaves undefined. The
   hardware gives +0 with DX10_CLAMP 1, for `v_fmac_f16`, `v_fma_mixlo_f16` and `v_fma_mix_f32` alike (SHADER-186).
3. **Mix-op rounding.** The hardware rounds once (the section above).
4. **f32 denormal operands of the mix ops.** They are not flushed with `denorm_mode_32 = 0`: `0x00000001`, `0x807fffff`
   and the integer constants `1` and `64` used as f32 give the denormal, where the hardware gives a zero of its sign
   (compare SHADER-118 for the mad ops).
5. **NaN payloads.** Where the hardware keeps the payload and, with IEEE 0, the signalling bit, the translator writes
   the canonical quiet NaN (`0x7fc00000`, `0x7e00`). For example, `v_fmac_f16` keeps `0x7c01`. This is the "not
   modelled yet" note of the f16 transcendentals row in `docs/design.md`, here for these ops too.

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
