---
name: verify-translator
description: "Verify tools/rdna2spirv.py end to end against hand-written RDNA2 listings, SPIRV-Tools (spirv-as/spirv-val), the CPU SPIR-V interpreter, and optional corpus slices. Use after editing tools/rdna2spirv.py, tools/shader_ir.py, tools/cfg_structure.py, tools/wave64_halves.py, or tests/semantic/."
---

# Verify Translator

Project-local verification skill for Leonida's RDNA2 to SPIR-V shader translator (`tools/rdna2spirv.py`). Runs without game data on any OS and writes machine-readable proof artifacts.

## Launch

This is a short-lived CLI translator, not a daemon. Each drive creates a private temporary directory for intermediate `.spvasm` and `.spv` files and writes only its proof artifact to `--evidence`.

```sh
python -I .agents/skills/verify-translator/scripts/verify_translator.py doctor --evidence <dir>
```

## Doctor

Run `doctor` first on a fresh checkout or whenever results look off. It confirms `tools/rdna2spirv.py` and `tests/semantic/support.py` exist, reports `spirv-as`, `spirv-val`, and `llvm-mc` availability, and prints `ready_for_cpu_spirv`.

## Drive

Drive the real translator path (`tools/rdna2spirv.py`, `spirv-as`, `spirv-val`, and `tests/semantic/spirv_mini_interp.py`) for the mapped feature your change touches:

```sh
python -I .agents/skills/verify-translator/scripts/verify_translator.py drive --feature valu-ops --evidence <dir>
python -I .agents/skills/verify-translator/scripts/verify_translator.py drive --feature memory-bounds --evidence <dir>
python -I .agents/skills/verify-translator/scripts/verify_translator.py drive --feature control-flow-wave64 --evidence <dir>
```

Feature map: [features/README.md](features/README.md).

## Evidence

Each run writes `<evidence>/<feature>_{pass,fail}.json` with the unittest return code, `spirv-val` verdict, timing, and output tail.

Proof standards:
- Never claim a translator change works from `compileall` alone.
- Run `python -I scripts/check_semantic_behavior.py` on any new or edited semantic test so a fake Python-only oracle cannot pass.
- When a local `<work>/shaders/dxbc/disasm` folder exists outside the repo, also check a 200-shader slice (`features/corpus-regression.md`) without quoting game disassembly or SPIR-V in logs or PRs.

## Cleanup

Temporary build directories are removed automatically on exit. Proof JSON files under `--evidence` survive cleanup.

## Helpers

- `.agents/skills/verify-translator/scripts/verify_translator.py`: executable stdlib-only (`python -I`) harness supporting `doctor` and `drive`.
