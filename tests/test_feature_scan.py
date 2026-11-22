"""feature_scan: stage census, SFI0 flags, output streams, MMA/GDS opcodes and code-base counts on a synthetic work folder.

    python -I tests/test_feature_scan.py

Every input is built here: tiny DXBC containers with only the parts the scanner reads, three hand-written .s
listings and a two-entry spirv/index.json. No game data.
"""
import json
import os
import struct
import subprocess
import sys
import tempfile

TOOLS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools")
sys.path.insert(0, TOOLS)
import feature_scan as F  # noqa: E402

failures = 0


def check(cond, what):
    global failures
    print(("PASS " if cond else "FAIL ") + what)
    failures += not cond


def signature(streams):
    """ISG1/OSG1 payload: count, first-element offset, 32-byte elements with the stream index first."""
    body = struct.pack("<II", len(streams), 8)
    for i, s in enumerate(streams):
        body += struct.pack("<6I", s, 0, 0, 0, 3, i) + bytes([0xF, 0xF, 0, 0]) + struct.pack("<I", 0)
    return body


def short_signature():
    """An OSG1 part whose declared five elements do not fit in the eight payload bytes present."""
    return struct.pack("<II", 5, 8) + bytes(8)


def dxbc(parts):
    """A DXBC container: 32-byte header (part count last), part offsets, then fourcc + size + payload per part."""
    head = 32 + 4 * len(parts)
    offsets, blob = [], b""
    for fourcc, payload in parts:
        offsets.append(head + len(blob))
        blob += fourcc.encode() + struct.pack("<I", len(payload)) + payload
    total = head + len(blob)
    header = b"DXBC" + bytes(16) + struct.pack("<II", 1, total) + struct.pack("<I", len(parts))
    return header + b"".join(struct.pack("<I", o) for o in offsets) + blob


def build(root):
    sh = os.path.join(root, "shaders")
    dx = os.path.join(sh, "dxbc")
    os.makedirs(os.path.join(dx, "disasm"))
    os.makedirs(os.path.join(sh, "spirv"))
    shaders = [
        # name, stage, SFI0 mask, output streams, listing
        ("a_ps", "ps", 0x1000 | 0x80000, [0], "\tv_mov_b32 v0, v1\n\ts_endpgm\n"),
        ("b_gs", "gs", 0, [0, 1], "\tds_ordered_count v0, v1 offset:772 gds\n\ts_endpgm\n"),
        ("c_cs", "cs", 0x8000000, [], "\tv_mfma_f32_32x32x8f16 a[0:15], v[0:1], v[2:3], a[0:15]\n\ts_endpgm\n"),
        ("d_vs", "vs", 0, [0], None),
        # s_getpc for data (the address is kept and offset), and s_getpc for a far branch (setpc follows)
        ("e_vs", "vs", 0, [0], "\ts_getpc_b64 s[4:5]\n\ts_add_u32 s4, s4, 0x40\n\ts_load_dword s6, s[4:5], 0x0\n\ts_endpgm\n"),
        ("f_ps", "ps", 0, [0], "\ts_getpc_b64 s[4:5]\n\ts_add_u32 s4, s4, 0x40\n\ts_addc_u32 s5, s5, 0\n\ts_setpc_b64 s[4:5]\n"),
    ]
    index = []
    for name, stage, mask, streams, listing in shaders:
        parts = [("SFI0", struct.pack("<Q", mask)), ("OSG1", signature(streams))]
        with open(os.path.join(dx, name + ".dxbc"), "wb") as f:
            f.write(dxbc(parts))
        index.append({"name": name, "file": name + ".dxbc", "stage": stage})
        if listing:
            with open(os.path.join(dx, "disasm", name + ".s"), "w", encoding="utf-8") as f:
                f.write(f"; {name}\n" + listing)
    with open(os.path.join(dx, "index.json"), "w", encoding="utf-8") as f:
        json.dump(index, f)
    with open(os.path.join(sh, "spirv", "index.json"), "w", encoding="utf-8") as f:
        json.dump({"a_ps": {"stage": "ps", "uses_code_base": True}, "d_vs": {"stage": "vs"}}, f)
    return sh, index


def test_feature_scan():
    """Runs under pytest (CI's translator-tests job) and as a plain script."""
    with tempfile.TemporaryDirectory() as tmp:
        sh, index = build(tmp)
        d = F.scan_dxbc(os.path.join(sh, "dxbc"), index)
        check(d["stages"]["ps"] == 2 and d["stages"]["gs"] == 1 and d["stages"]["hs"] == 0 and d["stages"]["as"] == 0,
              "stage census counts each stage and lists the absent ones as 0")
        check(d["sfi0"].get("ROVS") == {"ps": 1} and d["sfi0"].get("SHADING_RATE") == {"ps": 1},
              "SFI0 bits 0x1000 and 0x80000 decode to ROVS and SHADING_RATE")
        check(d["sfi0"].get("WAVE_MMA") == {"cs": 1}, "SFI0 bit 0x8000000 decodes to WAVE_MMA")
        check(d["output_on_stream_other_than_0"] == {"gs": 1} and d["gs_output_stream_count"] == {2: 1},
              "an OSG1 element on stream 1 is counted, stream 0 alone is not")

        stage_of = {e["name"]: e["stage"] for e in index}
        s = F.scan_disasm(os.path.join(sh, "dxbc", "disasm"), stage_of)
        check(s["files"] == 5, "five listings scanned")
        check(s["getpc_for_data_shaders"] == {"vs": 1},
              "s_getpc kept for data counts as a code base, s_getpc into s_setpc (far branch) does not")
        check(s["wave_mma_shaders"] == {"cs": 1} and "v_mfma_f32_32x32x8f16" in s["wave_mma_ops"], "v_mfma is found")
        check(s["ordered_or_gws_shaders"] == {"gs": 1} and s["gds_shaders"] == {"gs": 1},
              "ds_ordered_count with gds is found")
        check("ps" not in s["wave_mma_shaders"] and "ps" not in s["gds_shaders"], "a plain listing has neither")

        sp = F.scan_spirv(os.path.join(sh, "spirv", "index.json"))
        check(sp["uses_code_base"] == {"ps": 1} and sp["translated"] == {"ps": 1, "vs": 1}, "uses_code_base per stage")

        r = subprocess.run([sys.executable, "-I", os.path.join(TOOLS, "feature_scan.py"), sh, "--out",
                            os.path.join(tmp, "out.json")], capture_output=True, text=True)
        check(r.returncode == 0 and "SFI0 WAVE_MMA: 1" in r.stdout and "hs 0" in r.stdout, "the CLI prints the summary")
        check(os.path.isfile(os.path.join(tmp, "out.json")), "--out writes the JSON")
        check("a_ps" not in r.stdout and "b_gs" not in r.stdout, "the summary prints counts, not shader names")
    assert failures == 0, f"{failures} feature_scan check(s) failed"


def test_short_signature():
    """A malformed signature is ignored instead of aborting the count-only scan with struct.error."""
    check(F.streams(short_signature()) == set(),
          "a declared element table longer than the part yields no streams")
    check(F.streams(struct.pack("<II", 1, 0xFFFF)) == set(),
          "an element offset past the part yields no streams")
    check(F.streams(struct.pack("<II", 1, 0) + bytes(32)) == set(),
          "an element offset inside the count/offset header is ignored")
    check(F.streams(signature([0])) == {0} and F.streams(signature([0, 1])) == {0, 1},
          "well-formed signatures still report every element's stream")

    with tempfile.TemporaryDirectory() as tmp:
        sh, _ = build(tmp)
        dx = os.path.join(sh, "dxbc")
        with open(os.path.join(dx, "bad.dxbc"), "wb") as f:
            f.write(dxbc([("SFI0", struct.pack("<Q", 0)), ("OSG1", short_signature())]))
        with open(os.path.join(dx, "index.json"), encoding="utf-8") as f:
            index = json.load(f)
        index.append({"name": "bad", "file": "bad.dxbc", "stage": "ps"})
        with open(os.path.join(dx, "index.json"), "w", encoding="utf-8") as f:
            json.dump(index, f)

        d = F.scan_dxbc(dx, index)
        check(d["stages"]["ps"] == 3 and d["unreadable"] == 0,
              "scan_dxbc still counts a shader whose OSG1 is truncated")
        check(d["output_on_stream_other_than_0"] == {"gs": 1},
              "the truncated signature adds no bogus stream and keeps the real one")
        r = subprocess.run([sys.executable, "-I", os.path.join(TOOLS, "feature_scan.py"), sh],
                           capture_output=True, text=True)
        check(r.returncode == 0 and "SFI0 WAVE_MMA: 1" in r.stdout,
              "the CLI still prints the summary with a truncated OSG1 present")
    assert failures == 0, f"{failures} short-signature check(s) failed"


def test_part_payload_outside_container_is_unreadable():
    # Allow the flags to fit physically, while the declared part extends past EOF.
    data = bytearray(dxbc([("SFI0", struct.pack("<Q", 0x8000000))]))
    offset, = struct.unpack_from("<I", data, 32)
    struct.pack_into("<I", data, offset + 4, 16)
    with tempfile.TemporaryDirectory() as tmp:
        with open(os.path.join(tmp, "bad.dxbc"), "wb") as f:
            f.write(data)
        with open(os.path.join(tmp, "valid.dxbc"), "wb") as f:
            f.write(dxbc([("SFI0", struct.pack("<Q", 0x8000000))]))
        rows = [{"file": "bad.dxbc", "stage": "ps"}, {"file": "valid.dxbc", "stage": "gs"}]
        result = F.scan_dxbc(tmp, rows)
        check(result["unreadable"] == 1, "a DXBC part extending past EOF counts as unreadable")
        check(result["sfi0"] == {"WAVE_MMA": {"gs": 1}}, "only the valid file contributes feature flags")
    assert failures == 0, f"{failures} DXBC bounds check(s) failed"


def test_bad_shaders_folder_is_one_line():
    tool = os.path.join(TOOLS, "feature_scan.py")
    with tempfile.TemporaryDirectory() as tmp:
        as_file = os.path.join(tmp, "shaders.txt")
        with open(as_file, "w", encoding="utf-8") as f:
            f.write("not a folder")
        no_index = os.path.join(tmp, "no_index")
        os.makedirs(os.path.join(no_index, "dxbc"))
        broken = os.path.join(tmp, "broken")
        os.makedirs(os.path.join(broken, "dxbc"))
        with open(os.path.join(broken, "dxbc", "index.json"), "w", encoding="utf-8") as f:
            f.write("[{")
        not_list = os.path.join(tmp, "not_list")
        os.makedirs(os.path.join(not_list, "dxbc"))
        with open(os.path.join(not_list, "dxbc", "index.json"), "w", encoding="utf-8") as f:
            json.dump({"files": []}, f)
        out = os.path.join(tmp, "out.json")
        for src, expect in ((os.path.join(tmp, "missing"), "does not exist"), (as_file, "is not a folder"),
                            (no_index, "Run the shaders stage"), (broken, "cannot read"),
                            (not_list, "is not a shader list")):
            r = subprocess.run([sys.executable, "-I", tool, src, "--out", out], capture_output=True, text=True)
            check(r.returncode == 2 and expect in r.stderr and "Traceback" not in r.stderr and not r.stdout,
                  f"{os.path.basename(src)}: exit 2 with one line ({expect!r}), got {r.returncode}: "
                  f"{r.stderr.strip().splitlines()[-1:] if r.stderr.strip() else ''}")
        check(not os.path.exists(out), "no --out is written for a bad shaders folder")
    assert failures == 0, f"{failures} bad-input check(s) failed"


if __name__ == "__main__":
    try:
        test_feature_scan()
        test_short_signature()
        test_part_payload_outside_container_is_unreadable()
        test_bad_shaders_folder_is_one_line()
    except AssertionError:
        pass
    print(f"test_feature_scan: {'OK' if not failures else 'FAILED'} ({failures} failure{'s' if failures != 1 else ''})")
    sys.exit(1 if failures else 0)
