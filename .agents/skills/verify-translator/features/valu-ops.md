# VALU, SDWA, and Legacy Ops

## Sub-features
- SDWA source extraction and sign extension
- Integer min3/max3, `v_bfe_i32`, `v_mul_i32_i24`, and `v_ffbh_i32`
- Packed f16 FMA and shifts with `op_sel`
- `v_mul_legacy_f32` zero-times-anything guard and `v_sin`/`v_cos` scaling

## How to get to it (user POV)
`make_game.py` invokes `tools/rdna2spirv.py` on disassembled RDNA2 shaders. Translated modules must pass `spirv-val` and produce exact RDNA2 export words.

## Driving it with verify_translator.py
```sh
python -I .agents/skills/verify-translator/scripts/verify_translator.py drive --feature valu-ops --evidence <dir>
```
Observable end state: `unittest_rc == 0` across all 24 `mrt0` words in `tests/semantic/test_ps_semantics.py` and `spirv_val_ok == true` when SPIRV-Tools is installed.

## Gotchas
- Run `tests/semantic/test_mutations.py` after editing core VALU emission in `tools/rdna2spirv.py`.
- Without `llvm-mc`, `.asm` assembly syntax is not checked locally; Linux CI runs LLVM 21.
