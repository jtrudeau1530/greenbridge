"""Exercise the swizzle-search CLI with a synthetic blob."""
import pytest

from pathlib import Path
import subprocess
import sys

pytest.importorskip("numpy")
TOOL = Path(__file__).resolve().parents[1] / "tools" / "swizzle_search.py"


def test_preview_with_no_fitting_layout_reports_the_problem(tmp_path):
    blob = tmp_path / "tiny.bin"
    blob.write_bytes(bytes(32))
    preview = tmp_path / "preview.png"
    result = subprocess.run([sys.executable, "-I", str(TOOL), str(blob), "--out", str(preview)],
                            capture_output=True, text=True)
    assert result.returncode == 1
    assert "no layout" in result.stderr.lower()
    assert "Traceback" not in result.stderr
    assert not preview.exists()
