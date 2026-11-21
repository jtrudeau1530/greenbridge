; s_ff0_i32_b32 / s_ff1_i32_b32 on 32-bit constants: the index of the lowest zero / set bit, -1 if there is none.
; A constant is a value here, not a lane mask (FUZZ-4 in #156; found by tools/salu_diff.py).
; 3 -> lowest zero bit 2; 0x8c7dfb89 -> 1; -1 has no zero bit -> -1; 0x100 -> lowest set bit 8.
; Expected mrt0 bits: 2, 1, 0xffffffff, 8
s_ff0_i32_b32 s0, 3
s_ff0_i32_b32 s1, 0x8c7dfb89
s_ff0_i32_b32 s2, -1
s_ff1_i32_b32 s3, 0x100
v_mov_b32 v0, s0
v_mov_b32 v1, s1
v_mov_b32 v2, s2
v_mov_b32 v3, s3
exp mrt0 v0, v1, v2, v3 done vm
s_endpgm
