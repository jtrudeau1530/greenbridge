; Sample the synthetic 8x8 gradient in vk_texture_test with image_sample_d_cl ... a16: the four derivatives stay f32,
; then the two f16 coordinates and the f16 min-LOD clamp (0.0, before an f16 pad of 1.0) are packed. The texture has
; one mip, so the clamp changes nothing. The expected RGBA at each pixel is
; (32*floor(x/8)/255, 32*floor(y/8)/255, 128/255, 1) over the 64x64 framebuffer.
v_interp_p1_f32_e32 v2, v0, attr0.x
v_interp_p2_f32_e32 v2, v1, attr0.x
v_interp_p1_f32_e32 v3, v0, attr0.y
v_interp_p2_f32_e32 v3, v1, attr0.y
s_load_dwordx8 s[8:15], s[0:1], 0x0
s_waitcnt lgkmcnt(0)
s_mov_b32 s16, 0x92
s_mov_b32 s17, 0xc00000
s_mov_b32 s18, 0x14500000
s_mov_b32 s19, 0
v_mov_b32_e32 v8, 0x3c800000
v_mov_b32_e32 v9, 0
v_mov_b32_e32 v10, 0
v_mov_b32_e32 v11, 0x3c800000
v_cvt_pkrtz_f16_f32 v12, v2, v3
v_mov_b32_e32 v13, 0x3c000000
image_sample_d_cl v[4:7], v[8:13], s[8:15], s[16:19] dmask:0xf dim:SQ_RSRC_IMG_2D a16
s_waitcnt vmcnt(0)
exp mrt0 v4, v5, v6, v7 done vm
s_endpgm
