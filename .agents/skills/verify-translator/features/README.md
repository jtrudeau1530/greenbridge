# Translator Feature Map

| Feature | File | Primary harness target |
|---|---|---|
| VALU, SDWA, f16, and legacy ops | [valu-ops.md](valu-ops.md) | `test_ps_semantics` + `ps_ops` through `spirv-val` |
| Buffer, SMEM, and global bounds | [memory-bounds.md](memory-bounds.md) | `test_buffer_oob_select` + `cs_buf_tail` through `spirv-val` |
| Control flow, EXEC, and Wave64 | [control-flow-wave64.md](control-flow-wave64.md) | `test_saveexec_b64` + `cs_w64x_saveexec_b64` through `spirv-val` |
| Corpus slice regression | [corpus-regression.md](corpus-regression.md) | `tools/rdna2spirv.py` `--limit` 200 slice |
