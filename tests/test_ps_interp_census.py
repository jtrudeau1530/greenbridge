"""SHADER-181: PSV0 and FSSM interpolation modes and their agreement with the code.

Every container, blob and listing here is built by hand; no game data is used.

    python -I -m pytest tests/test_ps_interp_census.py
"""
import json
import os
import struct
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS = os.path.join(ROOT, "tools")
TOOL = os.path.join(TOOLS, "ps_interp_census.py")
sys.path.insert(0, TOOLS)

import ps_interp_census as census  # noqa: E402


def psv0(stage, inputs, info_size=52):
    """A PSV0 part: inputs are (name, index, kind, rows, start_row, mode)."""
    info = bytearray(info_size)
    if info_size >= 36:
        info[24] = stage
        info[28] = len(inputs)
    strings = bytearray(b"\0")
    offsets = {}
    for name, *_ in inputs:
        if name not in offsets:
            offsets[name] = len(strings)
            strings += name.encode() + b"\0"
    while len(strings) % 4:
        strings += b"\0"
    indexes = [index for _, index, *_ in inputs]
    out = struct.pack("<I", info_size) + bytes(info) + struct.pack("<I", 0)
    out += struct.pack("<I", len(strings)) + bytes(strings)
    out += struct.pack("<I", len(indexes)) + struct.pack(f"<{len(indexes)}I", *indexes)
    if inputs:
        out += struct.pack("<I", 16)
        for k, (name, _index, kind, rows, start_row, mode) in enumerate(inputs):
            out += struct.pack("<IIBBBBBBBB", offsets[name], k, rows, start_row, 0x4, kind, 3, mode, 0, 0)
    return out


def dxbc(chunks):
    """A DXBC container holding the given {fourcc: payload} parts."""
    header = 32 + 4 * len(chunks)
    body, offsets = b"", []
    for fourcc, payload in chunks.items():
        offsets.append(header + len(body))
        body += fourcc + struct.pack("<I", len(payload)) + payload
    return (b"DXBC" + bytes(24) + struct.pack("<I", len(chunks))
            + struct.pack(f"<{len(chunks)}I", *offsets) + body)


def fssm(flags):
    data = bytearray(0x140 + 8 * len(flags))
    data[0:4] = b"FSSM"
    for i, f in enumerate(flags):
        struct.pack_into("<8B", data, 0x138 + 8 * i, 6, 4, 0xF, i, 0, 0, i + 1, f)
    return bytes(data)


def write(root, name, psv, table=None, listing=None):
    root.mkdir(exist_ok=True)
    (root / f"{name}.dxbc").write_bytes(dxbc({b"ISG1": b"", b"PSV0": psv}))
    if table is not None:
        (root / "fssm").mkdir(exist_ok=True)
        (root / "fssm" / f"{name}.fssm").write_bytes(table)
    if listing is not None:
        (root / "disasm").mkdir(exist_ok=True)
        (root / "disasm" / f"{name}.s").write_text(listing)


LISTING = (
    "; ps  code 0x0..0x10 (header), gfx1013, 0 invalid words\n"
    "\tv_interp_mov_f32_e32 v2, p0, attr0.x                    ; encoding: [0x00,0x00,0x08,0xc8]\n"
    "\tv_interp_p1_f32_e32 v3, v0, attr1.y                     ; encoding: [0x00,0x05,0x0c,0xc8]\n"
    "\tv_interp_p2_f32_e32 v3, v1, attr1.y                     ; encoding: [0x01,0x05,0x0d,0xc8]\n"
    "\tv_interp_p1ll_f16 v4, v0, attr2.x high                  ; encoding: [0x00,0x00,0x42,0xd7]\n"
    "\ts_endpgm\n"
)


def test_psv_input_modes_and_kinds():
    stage, inputs = census.psv_inputs(psv0(0, [("SV_Position", 0, 3, 1, 0, 4), ("TEXCOORD", 0, 0, 1, 1, 2),
                                               ("TEXCOORD", 1, 0, 2, 2, 1), ("COLOR", 0, 0, 1, 4, 5)]))
    assert stage == 0
    assert [(e["name"], e["index"], e["kind"], e["rows"], e["start_row"], e["mode"]) for e in inputs] == [
        ("SV_Position", 0, 3, 1, 0, 4), ("TEXCOORD", 0, 0, 1, 1, 2), ("TEXCOORD", 1, 0, 2, 2, 1),
        ("COLOR", 0, 0, 1, 4, 5)]


def test_psv_version0_has_no_stage():
    assert census.psv_inputs(psv0(0, [], info_size=24)) == (None, [])


def test_truncated_psv_raises_value_error():
    with pytest.raises(ValueError):
        census.psv_inputs(psv0(0, [("TEXCOORD", 0, 0, 1, 0, 2)])[:-8])


def test_code_reads_by_position():
    assert census.code_reads(LISTING) == {0: {"mov"}, 1: {"interp"}, 2: {"interp"}}


def test_census_counts_and_agreement(tmp_path):
    root = tmp_path / "dxbc"
    inputs = [("SV_Position", 0, 3, 1, 0, 4), ("TEXCOORD", 0, 0, 1, 1, 1), ("TEXCOORD", 1, 0, 1, 2, 2),
              ("TEXCOORD", 2, 0, 1, 3, 2), ("TEXCOORD", 3, 0, 1, 4, 2)]
    write(root, "ok", psv0(0, inputs), fssm([0x01, 0x02, 0x1A, 0x02]), LISTING)
    # Position 0 is linear but read by v_interp_mov: the code disagrees with the table.
    write(root, "bad", psv0(0, inputs), fssm([0x02, 0x02, 0x1A]), LISTING)
    write(root, "vs", psv0(1, [("POSITION", 0, 0, 1, 0, 0)]))
    t, disagree = census.census(str(root))
    assert t["containers"] == 3 and t["pixel_shaders"] == 2
    assert t["psv_attribute_modes"] == {"constant": 2, "linear": 6}
    assert t["psv_position_modes"] == {"noperspective": 2}
    assert t["psv_shader_mode_sets"] == {"constant+linear": 2}
    assert t["fssm_modes"] == {"flat": 1, "linear": 6}
    assert t["code"] == {"flat read by mov": 1, "linear read by interp": 4, "linear unread": 1,
                         "linear read by mov": 1}
    assert t["code_shaders"] == {"agree": 1, "disagree": 1}
    assert disagree == ["bad"]


def test_missing_sidecars_are_counted(tmp_path):
    root = tmp_path / "dxbc"
    write(root, "nofssm", psv0(0, [("TEXCOORD", 0, 0, 1, 0, 2)]))
    write(root, "nodisasm", psv0(0, [("TEXCOORD", 0, 0, 1, 0, 2)]), fssm([0x02]))
    t, _ = census.census(str(root))
    assert t["fssm_missing"] == 1 and t["disasm_missing"] == 1 and t["code_shaders"] == {}


def test_unreadable_container_is_counted_not_fatal(tmp_path):
    root = tmp_path / "dxbc"
    root.mkdir()
    (root / "junk.dxbc").write_bytes(dxbc({b"PSV0": b"\x40\x00\x00\x00"}))
    t, _ = census.census(str(root))
    assert t["unreadable"] == 1 and t["pixel_shaders"] == 0


def test_cli_json_and_text(tmp_path):
    root = tmp_path / "dxbc"
    write(root, "ok", psv0(0, [("TEXCOORD", 0, 0, 1, 0, 1), ("TEXCOORD", 1, 0, 1, 1, 2), ("TEXCOORD", 2, 0, 1, 2, 2)]),
          fssm([0x01, 0x02, 0x02]), LISTING)
    p = subprocess.run([sys.executable, "-I", TOOL, str(root), "--json"], capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    doc = json.loads(p.stdout)
    assert doc["code_shaders"] == {"agree": 1} and "disagreements" not in doc
    p = subprocess.run([sys.executable, "-I", TOOL, str(root)], capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    assert "1 containers, 1 pixel shaders (0 unreadable)" in p.stdout
    assert "code_shaders: agree 1" in p.stdout
    assert "disagrees:" not in p.stdout  # no shader names without --list-disagreements


def test_cli_rejects_a_missing_folder_and_negative_limit(tmp_path):
    p = subprocess.run([sys.executable, "-I", TOOL, str(tmp_path / "nope")], capture_output=True, text=True)
    assert p.returncode == 2 and "is not a folder" in p.stderr
    p = subprocess.run([sys.executable, "-I", TOOL, str(tmp_path), "--limit", "-1"], capture_output=True, text=True)
    assert p.returncode == 2 and "--limit" in p.stderr
