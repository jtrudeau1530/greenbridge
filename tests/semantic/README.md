# Translator regression tests

Standard library only; no GPU, no LLVM, no SPIRV-Tools, no game files.

    python -m unittest discover -s tests/semantic -v

What they check (`tools/rdna2spirv.py`, pixel shaders plus one compute shader):

- `test_ps_semantics.py` translates `ps_sdwa`, `ps_ops`, `ps_ops2`, `ps_ops3`, `ps_ops4` and `ps_legacy_mul` and
  *executes* the SPIR-V with `spirv_mini_interp.py`, then compares every `mrt0` word with the expected value. Each
  shader exports four 32-bit words, so six shaders give 24 asserted values. The expected words are written down by
  hand, with the ISA rules in the module docstring, and are cross-checked against plain-Python reference calculations.
- `test_value_semantics.py` does the same for the recent value-level fixes: scalar b16 shifts, packed b16 shifts
  with `op_sel`, `v_add_nc_u16`, `v_sin_f16`/`v_cos_f16`, `s_buffer_load_dwordx2` into VCC and flat
  `v_interp_mov` input next to an interpolated one. `ps_ops5`, `ps_ops6`, `ps_vcc_x2` and `ps_flat` give 16 more
  asserted values, and `ps_fma_literal` 4 more: which operand the literal K multiplies in
  `v_fmamk/fmaak/madmk/madak_f32`. `cs_bounds` runs as compute and asserts the four dwords in its buffer after in-bounds
  load/store and out-of-bounds load/store attempts (the V# has num_records 16). `ps_ops4` and `ps_legacy_mul`
  are also asserted there, duplicating the matching `test_ps_semantics.py` cases. As above, constants are
  written by hand and recomputed by plain-Python references checked against each other. `cs_fmt_1010102` runs
  as compute too and asserts 16 dwords: 10:10:10:2 words read through formats 50 and 44 (from the V# and from
  tbuffer's own format) and stored in both. `ps_scc_minmax` asserts SCC after `s_min`/`s_max` `_i32`/`_u32`:
  0 on a tie, and 1 on a pair (-1 against 1) that only the right signedness orders that way.
- `spirv_wave_interp.py` runs every invocation of one subgroup at once through structured control flow
  (split at a selection, rejoin at its merge, loops until no invocation is left) with the subgroup
  operations the translator emits. It is the SPIR-V side of `tools/cf_diff.py` (FUZZ-8);
  `test_spirv_wave_interp.py` pins its convergence rules on hand-written modules.
- `test_interp.py` tests the interpreter against the SPIR-V / GLSL.std.450 definitions. The interpreter raises
  `UnsupportedConstruct` for anything it does not model instead of approximating it. It can seed Input variables,
  PushConstant words and a flat physical-address memory map for the tests above.
- `test_wave64_cross_half.py` translates the GPU tests `cs_w64x_readlane`, `cs_w64x_writelane` and
  `cs_w64x_bpermute` (no `--wave64-emulate`) and runs all 64 lanes in lockstep with
  `spirv_mini_interp.run_workgroup`. On one 64-wide subgroup the 24 words must equal the GPU check's values
  (`runtime/test/shaders/check.py`). On two 32-wide subgroups (NVIDIA without emulation) `ds_bpermute` stays
  exact, `v_writelane` writes stay exact, and a `v_readlane` from the other half is a shuffle from lane >= 32,
  undefined in SPIR-V (`UndefinedResult`); a "lane mod 32" model of issue #94 gives the words a 32-wide GPU
  shows (all 24 matched on an RTX 4070 SUPER, driver 610.88, without `--wave64-emulate`). Two SPIR-V sabotages (readlane lane & 31, bpermute without its half base) must change the
  64-wide result.
- `test_wave64_subgroup_ops.py` runs eight more wave64 GPU tests the same way (`cs_w64x_ballot`,
  `_readfirstlane`, `_load_ballot_load`, `_partial`, `cs_wave64_ballot`, `cs_wm_bfm64`, `cs_wm_half_cmp`,
  `cs_wavemask`): on one 64-wide subgroup every word equals the GPU check; on two 32-wide subgroups the
  words pinned there are the ones an RTX 4070 SUPER (driver 610.88) wrote without `--wave64-emulate`. The
  cs_w64x tests with branches need llvm-mc encodings and are not covered.
- `test_alu_int_misc.py` translates one-instruction synthetic listings for `v_bfm_b32`, `v_and_or_b32`,
  `v_add_lshl_u32`, `v_lshl_or_b32`, `v_mad_i32_i24` and `v_mul_lo_u32` and checks the mrt0 word against
  hand-computed ISA values (masked shift counts, wrapping adds/products, 24-bit sign extension).
- `test_negative_controls.py` sabotages the translated SPIR-V text and requires the result to change.
- `test_mutations.py` sabotages a temporary copy of `tools/` (13 regressions across SDWA, min3/max3, `v_bfe_i32`,
  `v_pk_fma_f16`, `v_fma_f16 op_sel`, `v_subrev_co_ci_u32`, `v_cmpx_eq_u64`, `v_mul_i32_i24`, `v_ffbh_i32`,
  `v_pk_lshlrev_b16`, `v_sin_f16` and `v_mul_legacy_f32`) and requires `test_ps_semantics.py` to fail for the
  matching shader.
- `test_fixtures.py` checks `tests/semantic/fixtures/disasm/*.s` equals what `regen_fixtures.py` makes from
  `runtime/test/shaders/*.asm` (`python tests/semantic/regen_fixtures.py` rewrites them).

Limits: single lane (or lanes in lockstep, one instruction per lane per round, that exchange values only through
`OpGroupNonUniformShuffle`, `Ballot`, `Any`, `All` and `BitwiseOr` Reduce), forward-only structured control flow (the fragment-exit discard test and the compute
dynamic-bounds merge; no backward branches, loops, calls or switches), no texture access. PushConstant words,
Input variables and a flat physical-address `{address: word}` map can be supplied per run. A shift by at least the bit width is an undefined value: arithmetic on it stays
undefined and any other use (store, branch or select condition, shuffle, output) raises. Values that pass
through a float op lose NaN payloads (Python floats), so expected output words avoid NaN bit patterns. `Fma` is
computed in double precision then rounded to binary32 (exact for the f16-derived operands used). The `.asm`
sources are written for `llvm-mc -mcpu=gfx1013`; these tests do not assemble them, so assembler syntax is not
checked here.
To add a shader: put the `.asm` in `runtime/test/shaders/`, add it to `SHADERS` in `regen_fixtures.py`, regenerate,
and add its expected words (with derivation) to the matching test file.
