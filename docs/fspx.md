# FSPX: the FidelityFX DLL's native compute programs

The package's FSR upscaler, `amd_fidelityfx_upscaler_xbox.dll`, carries 29 compute shaders as `DXBC` containers with
`SFI0`, `ISG1`, `OSG1`, `PSV0`, `RTS0`, `HASH` and `ACCC` parts and no `DXIL` ([d3d12_command_objects.md](d3d12_command_objects.md)
§3.5a). The `ACCC` part is an `FSPX` payload: a small directory around a precompiled RDNA2 program. It is not the
`XBCH`/`FSSM` format of the game's own shaders ([design.md](design.md)), but it holds the same kind of program, and
`tools/fspx_unpack.py` lays it out so `tools/rdna2spirv.py` translates it like a game compute shader.

Everything below was measured on the 1.0.2.1 package's DLL (SHA-256 `faff02f2…a75fa`). "Read from the DLL" means
Ghidra 12.2 output (through REA) for the named function; "measured" means checked on all 29 programs unless a count is
given; anything else is marked as inferred.

## Layout

```
u32  'FSPX'
u32  version          1 in all 29; the DLL does not read it
u32  count            4 in all 29
u32  total size       == the ACCC part size in all 29
count x { u32 kind, u32 offset, u32 size, u32 0 }    offset from the start of the payload
```

| Kind | Size | What it is |
|---|---|---|
| 0 | 28 bytes in all 29 | Dispatch state, see below. Required by the DLL. |
| 1 | 20 bytes in all 29 | Unknown. Optional for the DLL. |
| 6 | 768 to 41,536 bytes | The RDNA2 program, at offset 0x100 in all 29. Required by the DLL. |
| 7 | 152 to 696 bytes | Resource and hash words, see below. Optional for the DLL. |

### How the DLL reads it

The DLL's reader (RVA 0x1a20, read from the DLL) clamps the total to the buffer, then returns:

- 1 for a null or shorter-than-16-byte buffer, 2 when the magic is not `FSPX`,
- 4 when `16 + 16 * count` does not fit the total,
- 5 when kind 0 or kind 6 is missing or empty, else 0.

It skips an entry whose range starts inside the directory or ends past the total, and keeps the last entry of each
kind 0, 1, 6 and 7; other kinds are ignored. `tools/fspx_unpack.py`'s `parse_fspx` follows the same rules and codes.

The pipeline function (RVA 0x4aa0, read from the DLL) accepts either a bare `FSPX` payload or a `DXBC` container whose
`ACCC` part is one. It copies kind 6 into an `XMemAlloc` block (size rounded up to 256 bytes, attributes
`0xa8004000`) and calls device slot 67 with:

| Offset | Field |
|---|---|
| +0x00 | root signature |
| +0x08, +0x10 | kind 0 pointer, size |
| +0x18, +0x20 | kind 1 pointer, size |
| +0x28, +0x30 | the copied kind 6, its size |
| +0x38, +0x40 | kind 7 pointer, size |

The DLL's own messages name that call `DeserializeComputePipelineStateX` and the path "ACO". When the call fails,
the message says it falls back to "NewBE", and the DLL sets the environment variable `FFX_FSR4_XBOX_ACO_ACTIVE` to
`0`. Its next pipeline call is slot 62 (measured in #864), so "NewBE" is inferred to be that path, the one a PC
device refuses (§3.5a).

## Kind 0: dispatch state

Seven u32 words.

| Word | Meaning | How far it is checked |
|---|---|---|
| 0, 1 | Unknown | The same two values in all 29. |
| 2 | 0 | All 29. |
| 3 | `NumThreadsX` (bits 15..0), `NumThreadsY` (31..16) | Measured: equal to the container's `PSV0` thread-group size in all 29. |
| 4 | `NumThreadsZ` (15..0); bit 20 set = wave32 | Z measured against `PSV0` in all 29. Wave size inferred: bit 20 is clear in exactly the programs whose code has a wave64 marker (`rdna2spirv`'s `WAVE64_RE`) in 28 of 29; the 29th has no marker either way. |
| 5 | `COMPUTE_PGM_RSRC1` | Inferred. `FLOAT_MODE` 0xC0, `DX10_CLAMP` and `MEM_ORDERED` in all 29. With a VGPR granule of 8 for wave32 and 4 for wave64, the highest VGPR the code uses is inside the allocation in all 21 programs that decode with no invalid word, at its last register in several. |
| 6 | `COMPUTE_PGM_RSRC2` | Inferred. `LDS_SIZE` covers the largest `ds_*` offset in all 29. `USER_SGPR` plus the enabled `TGID` SGPRs cover the SGPRs the code reads before writing them in 20 of those 21 (8 or 14 user SGPRs here). In the 21st, a wave64 program, the only SGPRs past them are the carry-out destination of a `v_add_co_ci_u32_e64`, which the check counted as a read. |

`rdna2spirv.py` reads words 3, 4 (low half) and 6 from `fspx/<name>.json` where it reads the FSSM header for a game
compute shader.

## Kind 6: the program

- **Not compressed.** Each program holds exactly one `s_endpgm`, followed by padding (mostly `s_nop 0`).
- **A 36-byte prologue** comes first. It is three 12-byte VOP3 words with opcode 0x180 (bits 31..16 = `0xD580`), plus
  literals.
  - They carry the container's shader hash. Measured: the low halves of words 0 and 3 and the literals in words 2 and
    5 equal the 16 bytes after kind 7's `0x43` word, which are the digest in the container's `HASH` part, in all 29.
  - `llvm-mc` rejects them. Their effect on the GPU was not checked; that they are no-ops is inferred.
  - `tools/fspx_unpack.py` disassembles from the end of the prologue.
- **Target.** The programs use `v_dot4_i32_i8` (1,458 times) and `v_dot4c_i32_i8` (25,813 times), which fits the
  provider's name (`FSR4_Int8`). `llvm-mc` (LLVM 23.1.5) decodes them for `gfx1011`, `gfx1012`, `gfx1030`, `gfx1031`,
  `gfx1033`, `gfx1034` and `gfx1036`, but not for `gfx1010`, `gfx1013` (the target `shader_disasm.py` uses for the
  game) or `gfx1100`. Measured on one 10,432-byte program, prologue included: 13 invalid words with the first group,
  1,276 with `gfx1010` and `gfx1013`, 1,290 with `gfx1100`.
  - `fspx_unpack.py --disasm` defaults to `gfx1030`.
- **What llvm-mc still rejects.** With `gfx1030`, 21 of the 29 decode with no invalid word. The other 89 words, in 8
  programs, are all VOP3P opcode 0x16 with `op_sel` 2 and `op_sel_hi` 5. That is the opcode of the 1,458
  `v_dot4_i32_i8` that `llvm-mc` does decode, all of which have `op_sel` 0.
  - Measured on RDNA2 hardware (gfx1036, through `tools/hw_oracle`, the instruction given as two raw words): with
    `op_sel` 2 and `op_sel_hi` 5 the result is the dot product with the two 16-bit halves of the second source
    exchanged, in all 5 operand sets (4 of them give another word under the plain rule). So `op_sel` selects halves
    here as it does for the packed 16-bit ops. The words are in `tests/semantic/test_alu_dot4_i8.py`.

## Kind 7: records

- **Format.** It starts with the word `0x403` in all 29, and the words `0x41`, `0x42` and `0x43` appear in all 29.
  That they tag records is inferred from the layout around them.
- **Records `0x41` and `0x42`.** The word 3 after each `0x41` or `0x42` is a register number: the 121 such numbers
  in the 29 programs are all among the register numbers the container's `RTS0` declares (measured; the check ignores
  the register class, so it is weaker than it looks). That the records list the resources the program uses is
  inferred: in one program, checked by hand, a declared SRV does not appear.
- **Record `0x43`.** The 16 bytes after it are the container's shader hash (its `HASH` part), the value embedded in
  the program's prologue.
- **Not read by the translator,** which does not read the game's FSSM binding table either. The programs load their
  descriptors with `s_load` and `s_buffer_load`, as the game's shaders do; that this is all they need is inferred.

## Translation status

With `tools/fspx_unpack.py --disasm` and `tools/rdna2spirv.py` on the 29 programs:

- **17 translate and pass `spirv-val`.** 13 of them are the 16 programs that use no dot4 instruction, minus 3 that
  stop at `v_pk_add_u16`, `image_atomic_add` (2D) and `v_pk_fmac_f16`. The other 4 use dot4 (design.md, "Signed-byte
  dot products").
- **9 more programs that use dot4 do not translate yet.** The first thing each stops at:
  - 6 at an `s_buffer_load_dwordx8` or `_dwordx16` whose destination range ends in `vcc_lo, vcc_hi`;
  - 2 at a reachable word `llvm-mc` does not decode (the VOP3P words above);
  - 1 at an `src_scc` operand.

## Unknown

- Kind 0 words 0 and 1.
- Kind 1.
- The exact layout of the kind-7 records.
- What the prologue does on the GPU.
- Whether the game itself ships FSPX programs. Its shaders measured so far are `FSSM`.
