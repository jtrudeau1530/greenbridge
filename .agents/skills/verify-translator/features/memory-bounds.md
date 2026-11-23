# Memory Bounds (MUBUF, SMEM, Global)

## Sub-features
- MUBUF `OOB_SELECT` modes 0..3 for literal and runtime V# descriptors
- Raw buffer per-dword tail bounds (`num_records`)
- SMEM dword truncation and signed 21-bit immediate offsets
- Global memory signed 12-bit byte offset arithmetic

## How to get to it (user POV)
Translated shaders reading or writing GPU buffers must return 0 on out-of-bounds loads and drop out-of-bounds stores instead of faulting the Vulkan device.

## Driving it with verify_translator.py
```sh
python -I .agents/skills/verify-translator/scripts/verify_translator.py drive --feature memory-bounds --evidence <dir>
```
Observable end state: `tests/semantic/test_buffer_oob_select.py` passes all 33 cases and `cs_buf_tail` passes `spirv-val`.

## Gotchas
- Global 64-bit pointer arithmetic is covered by `tests/semantic/test_global_mem_offset.py` because `tests/semantic/spirv_mini_interp.py` rejects 64-bit integer pointers.
