; VOP3 output modifiers on v_fmac_f16_e64 (docs/design.md "v_fmac_f16, v_interp_p2/mov_f32 output modifiers"):
; D.f16 = S0 * S1 + D, then mul:2 / mul:4 / div:2, then clamp to [0, 1], before narrowing to f16; the destination's
; high half (0xAAAA) is kept.
;   x: 1.0 + 1.0 * 1.0 = 2.0, clamp: 1.0 (0x3c00)
;   y: 0.5 + 0.5 * 0.5 = 0.75, mul:2: 1.5 (0x3e00)
;   z: -1.0 + 0.5 * 0.5 = -0.75, clamp: 0.0 (0x0000)
;   w: 2.0 + 1.0 * 2.0 = 4.0, div:2: 2.0 (0x4000)
; Expected mrt0 bits: 0xaaaa3c00, 0xaaaa3e00, 0xaaaa0000, 0xaaaa4000
v_mov_b32_e32 v2, 0xaaaa3c00
v_fmac_f16_e64 v2, 1.0, 1.0 clamp
v_mov_b32_e32 v3, 0xaaaa3800
v_fmac_f16_e64 v3, 0.5, 0.5 mul:2
v_mov_b32_e32 v4, 0xaaaabc00
v_fmac_f16_e64 v4, 0.5, 0.5 clamp
v_mov_b32_e32 v5, 0xaaaa4000
v_fmac_f16_e64 v5, 1.0, 2.0 div:2
exp mrt0 v2, v3, v4, v5 done vm
s_endpgm
