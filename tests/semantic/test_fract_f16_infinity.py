"""Exact f16 Fract infinity word measured on gfx1036 (docs/design.md)."""
from collections.abc import Callable
from pathlib import Path

from conftest import Shader, requires_tools
from support import mrt0_words


@requires_tools
def test_fract_f16_infinity(make_shader: Callable[[str, str], Shader]) -> None:
    fixture = Path(__file__).with_name("fixtures") / "ps_fract_f16_infinity.asm"
    shader = make_shader(fixture.read_text(encoding="ascii"), fixture.stem)
    spvasm = shader.translate()
    shader.validate(spvasm)
    for source, clamped in ((0x3E00, 0x3800), (0x7C00, 0), (0xFC00, 0)):
        words = mrt0_words(spvasm, push_constants=[source] + [0] * 31)
        assert words == [0x1357FE00, 0x2468FE00, 0x3F800000, 0x12340000 | clamped]
