# Corpus Slice Regression

## Sub-features
- 200-shader quick slice (`--limit` 200)
- Full 8-shard run (`--shard` and `--merge`)

## How to get to it (user POV)
When a user has a local `<work>/shaders/dxbc/disasm` folder built from their own copy, a translator change must never reduce translated or `spirv-val`-valid counts.

## Driving it with tools/rdna2spirv.py
```sh
python -I tools/rdna2spirv.py <work>/shaders/dxbc/disasm --out <scratch_before> --limit 200
python -I tools/rdna2spirv.py <work>/shaders/dxbc/disasm --out <scratch_after> --limit 200
```
Observable end state: `after.translated >= before.translated` and `after.spirv_val_valid >= before.spirv_val_valid`.

## Gotchas
- Requires a local `<work>` folder outside the repository. Never copy or quote shader disassembly or SPIR-V into the repo, logs, or pull requests.
- If no local `<work>` folder exists, report this journey as skipped rather than weakening the check.
