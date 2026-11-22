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
    assert capsys.readouterr().out == "winners: []\nprobed 0 of 0 textures; 0 skipped\n"


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


@pytest.mark.parametrize("data_size, candidate_count, parts", [(32760, 0, 1), (32768, 1, 1), (40960, 2, 1),
                                                                (32768, 1, 2), (40960, 2, 3)])
def test_probe_reports_available_candidates(data_size, candidate_count, parts, monkeypatch, capsys, tmp_path):
    # A synthetic 4K layout covers 64x64 BC1 blocks (32768 bytes). With two mips,
    # the "last" placement needs another 8192 bytes; a top-mip-only buffer fits just "first".
    # parts > 1 splits the GPU blob over that many carved files listed in gpu_chain, as the carver does for
    # most large textures; the probe must read them all, as tex_export.py does.
    eq = [(0, 0)] * 3 + [(1 << i, 0) for i in range(5)] + [(0, 1 << i) for i in range(4)]
    entry = (0, 0, 0, 0, 0, "synthetic 8 bpe")
    monkeypatch.setattr(tex_mode_probe, "load", lambda: ({}, {"GFX10_SW_4K_S_PATINFO": [entry]}))
    monkeypatch.setattr(tex_mode_probe, "equation", lambda nib, e, bb: eq)
    carve = tmp_path / "synthetic"
    carve.mkdir()
    cuts = [data_size * i // parts for i in range(parts + 1)]
    chain = [[16 + 0x100000 * i, cuts[i + 1] - cuts[i]] for i in range(parts)]
    for off, size in chain:
        (carve / f"{off:012x}.bin").write_bytes(bytes(size))
    row = {"fmt_name": "BC1_UNORM", "w": 256, "h": 256, "archive": "synthetic", "gpu": 16,
           "flags": 2 << 16, "data_off": 0, "gpu_size": data_size}
    if parts > 1:
        row["gpu_chain"] = chain
    index = tmp_path / "index.json"
    index.write_text(json.dumps([row]), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["tex_mode_probe.py", str(index), str(tmp_path)])

    tex_mode_probe.main()

    out = capsys.readouterr().out
    if candidate_count == 0:
        assert out == ("256x256 BC1_UNORM gpu=0x10  skipped: no candidate fits in the GPU blob\n"
                       "winners: []\nprobed 0 of 1 textures; 1 skipped\n")
    else:
        second = "n/a" if candidate_count == 1 else "0.0000 4K_S synthetic 8 bpe [last]"
        assert out == (
            f"256x256 BC1_UNORM mips=2 gpu={data_size}  best 0.0000 4K_S synthetic 8 bpe [first]"
            f"  2nd {second}  linear 0.0000\n"
            "winners: [(('4K_S', 'first'), 1)]\n"
            "probed 1 of 1 textures; 0 skipped\n"
        )
