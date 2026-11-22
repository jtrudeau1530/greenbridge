"""Format checks for the texture mode probe, without addrlib or game files."""
import pytest

from pathlib import Path
import sys

pytest.importorskip("numpy")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import tex_mode_probe


@pytest.mark.parametrize("fmt", ["BC4", "BC5", "BC6H", "BC7"])
def test_probe_refuses_formats_without_a_color_scorer(fmt, monkeypatch, capsys, tmp_path):
    index = tmp_path / "index.json"
    index.write_text("[]", encoding="utf-8")
    monkeypatch.setattr(tex_mode_probe, "load", lambda: ({}, {}))
    monkeypatch.setattr(sys, "argv", ["tex_mode_probe.py", str(index), str(tmp_path), "--fmt", fmt])
    with pytest.raises(SystemExit) as exc:
        tex_mode_probe.main()
    assert exc.value.code == 2
    assert fmt in capsys.readouterr().err
