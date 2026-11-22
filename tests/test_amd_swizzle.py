"""amd_swizzle.load: parses addrlib's pattern tables without evaluating them as Python.

    python -I tests/test_amd_swizzle.py
"""
import pytest  # noqa: E402 (keeps the CI bash loop from running this file)

import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))

try:
    import amd_swizzle as A  # noqa: E402
except ModuleNotFoundError as e:
    if e.name != "numpy":
        raise
    import pytest  # noqa: E811
    pytest.skip("amd_swizzle needs numpy", allow_module_level=True)


def table(name, rows):
    body = "\n".join("    {" + ", ".join(r) + ", }, // " + str(i) for i, r in enumerate(rows))
    return f"const UINT_64 GFX10_SW_PATTERN_{name}[][{len(rows[0])}] =\n{{\n{body}\n}};\n"


def header(nibble01_row):
    return (table("NIBBLE01", [nibble01_row]) + table("NIBBLE2", [["0"] * 4]) +
            table("NIBBLE3", [["0"] * 4]) + table("NIBBLE4", [["0"] * 4]) +
            "const ADDR_SW_PATINFO GFX10_SW_64K_D_X_PATINFO[] =\n{\n"
            "    {   0,    0,    0,    0,    0, } , // 1 pipes 1 bpe\n};\n")


def load(text):
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "gfx10SwizzlePattern.h")
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        return A.load(path)


def test_symbols_and_xor():
    nibbles, tables = load(header(["X0", "0", "Y1", "X3 ^ Y3", "Z0", "S2 ^ Y4 ^ X5", "7", "Y0"]))
    assert nibbles["NIBBLE01"] == [[1, 0, 1 << 17, (1 << 3) ^ (1 << 19), 1 << 32,
                                    (1 << 50) ^ (1 << 20) ^ (1 << 5), 7, 1 << 16]]
    assert nibbles["NIBBLE2"] == [[0, 0, 0, 0]]
    assert tables["GFX10_SW_64K_D_X_PATINFO"] == [(0, 0, 0, 0, 0, "1 pipes 1 bpe")]


def test_rejects_code():
    for cell in ("__import__('os').getcwd()", "X0 | Y0", "X16", "-1"):
        try:
            load(header([cell] + ["0"] * 7))
        except ValueError:
            continue
        raise AssertionError(f"accepted {cell!r}")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)


def test_64k_block_extent_comes_from_the_element_count():
    # XOR modes have no x/y bit that appears alone, so the old solo-bit guess returned 1x1 or too small.
    for bpe, bw, bh in ((1, 256, 256), (2, 256, 128), (4, 128, 128), (8, 128, 64), (16, 64, 64)):
        _, meta = A.offsets([(1, 0), (0, 1)], bpe, 1, 1)
        assert meta[:2] == (bw, bh), (bpe, meta)
        assert bw * bh * bpe == 65536


def cli(tmp_path, *args):
    import subprocess
    tool = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools", "amd_swizzle.py")
    return subprocess.run([sys.executable, "-I", tool, str(tmp_path / "blob.bin"), *args],
                          capture_output=True, text=True, timeout=120, check=False)


def test_out_cannot_overwrite_the_input(tmp_path):
    # Before: --out naming the blob rendered the preview over the sample it had just detiled.
    blob = tmp_path / "blob.bin"
    blob.write_bytes(bytes(range(256)) * 64)
    before = blob.read_bytes()
    p = cli(tmp_path, "--width", "256", "--height", "256", "--out", str(blob))
    assert p.returncode == 1, p.stderr
    assert "overwrite the input" in p.stderr
    assert "Traceback" not in p.stderr
    assert blob.read_bytes() == before


def test_out_directory_is_a_message_not_a_traceback(tmp_path):
    (tmp_path / "blob.bin").write_bytes(bytes(range(256)) * 64)
    p = cli(tmp_path, "--width", "256", "--height", "256", "--out", str(tmp_path))
    assert p.returncode == 1, p.stderr
    assert "is a directory" in p.stderr
    assert "Traceback" not in p.stderr


def test_uncompressed_feature_uses_raw_rgb_bytes():
    import numpy as np
    raw = np.arange(8, dtype=np.uint8).reshape(-1, 4)
    got = A.feature(raw, 4)
    assert got.shape == (2, 3)
    assert got.tolist() == [[0, 1, 2], [4, 5, 6]]
