# What game shaders expect from their inputs

Some game shaders give a black, speckled or hung result when one input value is wrong, and that looks like a translator
or layer bug when it isn't. This page lists those inputs per shader, so a wrong picture or a device-lost has a
documented first check. Each section says what was reported, by whom, and what was measured in the shader to confirm
it.

Measured on the local package's disassembly (`work/shaders/dxbc/disasm`) at `16282101`. Offsets, bit positions and
counts only; no shader text.

## Studio-shot lighting (`PS_STUDIOSHOT_PBR_MATERIAL_STANDARD_LIGHTING`)

Reported by @Silliana in [#1892](https://github.com/ItsNotRexed/Leonida/issues/1892). The corpus has one program
with this name. It shades with up to eight lights, read from a buffer of 512-byte light records. Colour and
intensity come from a separate table of light profiles.

### First check for a device-lost or a dark frame

1. **The dword at record +0x4C.** Its top byte is a loop count, so it must hold an integer and never a float (see
   [Light-shaping planes](#light-shaping-planes)). #1892 reports a GPU hang when a float sits here. That report gives
   the dword as +0x40; the count is measured at +0x4C (+0x40 is a different field, described below).
2. **The profile intensity.** A light whose profile intensity is 0 or less is skipped entirely. So an all-black light
   doesn't mean the light maths is broken.
3. **The shadow-strength byte at record +0x73.** At 0 the shadow map is still sampled, but the result does nothing.

### Light loop

- The loop always runs eight times (light 0 to 7); the light count doesn't come from a constant. An unused slot is
  switched off through its profile: point it at a profile with intensity 0.
- Light *i*'s record starts at byte *i* × 512 (0x200) of the light buffer.
- All 61 image samples and all 27 cube-coordinate sequences in the program are inside this loop. That matches #1892:
  the shader has no ambient term, and its 2D and cube samples are per-light cookies and shadows. Indirect light comes
  from the separate probe system (`indirect_light_probes` / `CS_ApplyProbeIrradiance`), not from this shader.

### Light record (512 bytes)

The fields below are the ones confirmed. The rest of the record is read but not decoded here.

| Offset | Type | Use |
|---|---|---|
| +0x10 | 3 × f32 | Light position; the shader subtracts the world position of the pixel from it. |
| +0x20 | u32 | Bits 0–8: profile index (9 bits) into the profile table. Bits 9–17 and 18–21 are two more fields, not decoded. |
| +0x28 | u32 | Flags. Bit 24 enables the light-shaping planes; bit 16 takes part in the shadow-map gate; bits 17, 20, 21, 26, 27 and 30 are tested, not decoded. |
| +0x40 | u32 | Low 24 bits non-zero turns on one per-light factor; at zero that factor is 1.0, so the light still shades. The top byte isn't read on the paths checked. |
| +0x4C | u32 | Top byte: number of light-shaping planes (0–255). See below. |
| +0x5F | u8 | Read as byte / 255; at 0 one block of the light's shading is skipped. |
| +0x73 | u8 | Shadow strength, read as byte / 255 right before the shadow-map block. |
| +0xB0 | planes | Light-shaping planes, 32 bytes apart. |

#### Light-shaping planes

When flag bit 24 at +0x28 is set and the dword at +0x4C is at least `0x01000000`, the shader loops over *n* planes,
where *n* is that dword's top byte. Plane *k* is four f32 (normal xyz, then w) at record +0xB0 + 0x20 × *k*. For each
plane it computes `saturate(dot(normal, pixel − light) + w)`, squares it and multiplies it into a per-light factor that starts at 1.0.

Only 11 planes fit in the record (the last at +0x1F0). A larger count reads on into the next light's record, and past the
end of the buffer after the last light. A float in this dword has its exponent in the top byte: 1.0f (`0x3F800000`)
gives 63 planes. The loop itself ends after at most 255 iterations. If a device-lost still shows up there, check this
dword and the buffer's bounds first.

#### Correction to #1892

#1892 puts two checks on one dword at +0x40: a top-byte loop count, and low 24 bits that must be non-zero or the light
is skipped. The shader splits these across two dwords:

- The count is the top byte of the dword at **+0x4C**, gated by flag bit 24 at +0x28. On that path the low 24 bits
  aren't tested.
- The low-24-bit test is on the dword at **+0x40**, and at zero it disables one factor, not the whole light.

The light is skipped only through a profile intensity of 0 or less.

### Profile table

- It is a separate buffer from the light records, made of 16-byte entries.
- The light record's profile index (bits 0–8 of +0x20) selects the entry.
- The entry's fourth dword (+12) is the intensity, as f32. When it's 0 or less the light is skipped before the rest of
  its record is read. #1892 gives the first three dwords as the RGB colour; the shader loads the whole entry later in the
  light's shading.
