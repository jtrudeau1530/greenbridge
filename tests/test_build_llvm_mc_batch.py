"""Exercise the native LLVM builder's batch writer without compiling LLVM."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_llvm_mc.ps1"


@unittest.skipUnless(os.name == "nt", "requires Windows PowerShell and cmd.exe")
class BuildBatchTest(unittest.TestCase):
    def test_non_ascii_paths_survive_the_batch_writer(self):
        text = SCRIPT.read_text(encoding="utf-8-sig")
        start = text.index('$bat = Join-Path $Temp "_build.cmd"')
        writer = text[start:text.index("$p = Start-Process", start)]

        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory) / "LLVM 東京"
            work.mkdir()
            (work / "vcvars64.bat").write_bytes(b"@exit /b 0\r\n")
            driver = Path(directory) / "write-build.ps1"
            driver.write_text(
                "param([string]$Temp)\n"
                "$ErrorActionPreference = 'Stop'\n"
                "$installer = $Temp\n"
                "$vcvars = Join-Path $Temp 'vcvars64.bat'\n"
                "$body = 'cd /d \"' + $Temp + '\" || exit /b 1'\n"
                + writer,
                encoding="utf-8", newline="\n",
            )
            subprocess.run(
                ["powershell.exe", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
                 "-File", str(driver), "-Temp", str(work)],
                check=True, timeout=30,
            )
            batch = (work / "_build.cmd").read_bytes()
            self.assertTrue(batch.startswith(b"@echo off\r\n"))
            self.assertIn(str(work).encode("utf-8"), batch)
            subprocess.run(["cmd.exe", "/d", "/c", "_build.cmd"], cwd=work, check=True, timeout=30)


if __name__ == "__main__":
    unittest.main()
