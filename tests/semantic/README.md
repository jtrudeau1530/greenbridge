# Translator regression tests

Standard library only; no GPU, no LLVM, no SPIRV-Tools, no game files.

    python -m unittest discover -s tests/semantic -v

What they check (`tools/rdna2spirv.py`, branch-free pixel shaders only):

- `test_ps_semantics.py` translates `ps_sdwa` and `ps_ops` and *executes* the SPIR-V with
  `spirv_mini_interp.py`, then compares every `mrt0` word with the expected value. Each shader exports four 32-bit
  words, so two shaders give eight asserted values. The expected words are written down by hand, with the ISA rules
  in the module docstring, and are cross-checked against plain-Python reference calculations.
- `test_interp.py` tests the interpreter against the SPIR-V / GLSL.std.450 definitions. The interpreter raises
  `UnsupportedConstruct` for anything it does not model instead of approximating it.
- `test_negative_controls.py` sabotages the translated SPIR-V text and requires the result to change.
- `test_mutations.py` sabotages a temporary copy of `tools/` (five regressions: SDWA `sext`, SDWA field offset,
  min3/max3, `v_bfe_i32`, `v_pk_fma_f16`) and requires `test_ps_semantics.py` to fail for the matching shader.
- `test_fixtures.py` checks `tests/semantic/fixtures/disasm/*.s` equals what `regen_fixtures.py` makes from
  `runtime/test/shaders/*.asm` (`python tests/semantic/regen_fixtures.py` rewrites them).

Limits: single lane, no control flow, no memory or texture access; `Fma` is computed in double precision then
rounded to binary32 (exact for the f16-derived operands used). The `.asm` sources are written for
`llvm-mc -mcpu=gfx1013`; these tests do not assemble them, so assembler syntax is not checked here.
To add a shader: put the `.asm` in `runtime/test/shaders/`, add it to `SHADERS` in `regen_fixtures.py`, regenerate,
and add its expected words (with derivation) to `test_ps_semantics.py`.
