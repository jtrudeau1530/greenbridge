"""Binary min/max with quiet NaNs, using the operand priority measured on gfx1030."""
from collections.abc import Callable
from pathlib import Path

from conftest import Shader, requires_tools
from support import mrt0_words


@requires_tools
def test_minmax_f32_payload(make_shader: Callable[[str, str], Shader]) -> None:
    fixture = Path(__file__).with_name("fixtures") / "ps_minmax_f32_payload.asm"
    shader = make_shader(fixture.read_text(encoding="ascii"), fixture.stem)
    spvasm = shader.translate()
    shader.validate(spvasm)

    a, b, half = 0x7FE00080, 0xFFD10040, 0x3F000000
    for lhs, rhs, expected in (
        (a, b, (a, a, b, b)),
        (a, half, (half, half, half, half)),
        (half, b, (half, half, half, half)),
        (0, 0x80000000, (0x80000000, 0, 0x80000000, 0)),
    ):
        words = mrt0_words(spvasm, push_constants=[lhs, rhs] + [0] * 30)
        assert words == [word ^ 0x40C00000 for word in expected]
