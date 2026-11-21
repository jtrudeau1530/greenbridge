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
  asserted values; `cs_bounds` runs as compute and asserts the four dwords in its buffer after in-bounds
  load/store and out-of-bounds load/store attempts (the V# has num_records 16). `ps_ops4` and `ps_legacy_mul`
  are also asserted there, duplicating the matching `test_ps_semantics.py` cases. As above, constants are
  written by hand and recomputed by plain-Python references checked against each other.
- `test_interp.py` tests the interpreter against the SPIR-V / GLSL.std.450 definitions. The interpreter raises
  `UnsupportedConstruct` for anything it does not model instead of approximating it. It can seed Input variables,
  PushConstant words and a flat physical-address memory map for the tests above.
- `test_negative_controls.py` sabotages the translated SPIR-V text and requires the result to change.
- `test_mutations.py` sabotages a temporary copy of `tools/` (13 regressions across SDWA, min3/max3, `v_bfe_i32`,
  `v_pk_fma_f16`, `v_fma_f16 op_sel`, `v_subrev_co_ci_u32`, `v_cmpx_eq_u64`, `v_mul_i32_i24`, `v_ffbh_i32`,
  `v_pk_lshlrev_b16`, `v_sin_f16` and `v_mul_legacy_f32`) and requires `test_ps_semantics.py` to fail for the
  matching shader.
- `test_fixtures.py` checks `tests/semantic/fixtures/disasm/*.s` equals what `regen_fixtures.py` makes from
  `runtime/test/shaders/*.asm` (`python tests/semantic/regen_fixtures.py` rewrites them).

Limits: single lane, forward-only structured control flow (the fragment-exit discard test and the compute
dynamic-bounds merge; no backward branches, loops, calls or switches), no texture access. PushConstant words,
Input variables and a flat physical-address `{address: word}` map can be supplied per run. Values that pass
through a float op lose NaN payloads (Python floats), so expected output words avoid NaN bit patterns. `Fma` is
computed in double precision then rounded to binary32 (exact for the f16-derived operands used). The `.asm`
sources are written for `llvm-mc -mcpu=gfx1013`; these tests do not assemble them, so assembler syntax is not
checked here.
To add a shader: put the `.asm` in `runtime/test/shaders/`, add it to `SHADERS` in `regen_fixtures.py`, regenerate,
and add its expected words (with derivation) to the matching test file.
