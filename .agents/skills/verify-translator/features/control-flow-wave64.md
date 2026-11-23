# Control Flow, EXEC Masks, and Wave64 Emulation

## Sub-features
- `s_and/or/andn2_saveexec_b64` and `s_andn2_wrexec_b64` across 64 lanes plus SCC
- Structured CFG vs dispatch-loop fallback (`MAX_STRUCTURED_DEPTH`)
- `--wave64-emulate` round splitting and memory ordering

## How to get to it (user POV)
Wave64 shaders running on 32-wide NVIDIA or 8-wide llvmpipe subgroups must preserve lane masks and cross-half shuffles identically to native 64-wide AMD hardware.

## Driving it with verify_translator.py
```sh
python -I .agents/skills/verify-translator/scripts/verify_translator.py drive --feature control-flow-wave64 --evidence <dir>
```
Observable end state: `tests/semantic/test_saveexec_b64.py` passes and `cs_w64x_saveexec_b64` passes `spirv-val`.

## Gotchas
- Changes to `tools/cfg_structure.py` or `tools/wave64_halves.py` also require `tests/test_cfg_structure.py` and `tests/semantic/test_w64x_scalar_load_order.py`.
