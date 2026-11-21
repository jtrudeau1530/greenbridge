"""amd_swizzle.load: parses addrlib's pattern tables without evaluating them as Python.

    python -I tests/test_amd_swizzle.py
"""
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))

try:
    import amd_swizzle as A  # noqa: E402
except ModuleNotFoundError as e:
    if e.name != "numpy":
        raise
    raise unittest.SkipTest("amd_swizzle needs numpy")


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
