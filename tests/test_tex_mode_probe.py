"""Format checks for the texture mode probe, without addrlib or game files.

    python -I -m pytest tests/test_tex_mode_probe.py
"""
import pytest

import json
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


def test_format_prefix_does_not_select_an_unsupported_index_row(monkeypatch, capsys, tmp_path):
    index = tmp_path / "index.json"
    rows = [{"fmt_name": "BC10_UNORM", "w": 256, "h": 256, "archive": "synthetic", "gpu": 16}]
    index.write_text(json.dumps(rows), encoding="utf-8")
    monkeypatch.setattr(tex_mode_probe, "load", lambda: ({}, {}))
    monkeypatch.setattr(sys, "argv", ["tex_mode_probe.py", str(index), str(tmp_path), "--fmt", "BC1"])
    tex_mode_probe.main()
    assert capsys.readouterr().out == "winners: []\n"


@pytest.mark.parametrize("value", ["0", "-1", "x"])
def test_probe_refuses_a_count_below_one(value, monkeypatch, capsys, tmp_path):
    # --n 0 probed nothing and printed "winners: []" as if it had run; -1 silently dropped the last texture.
    index = tmp_path / "index.json"
    index.write_text("[]", encoding="utf-8")
    monkeypatch.setattr(tex_mode_probe, "load", lambda: pytest.fail("the probe must stop before loading addrlib"))
    monkeypatch.setattr(sys, "argv", ["tex_mode_probe.py", str(index), str(tmp_path), "--n", value])
    with pytest.raises(SystemExit) as exc:
        tex_mode_probe.main()
    assert exc.value.code == 2
    assert "--n" in capsys.readouterr().err
