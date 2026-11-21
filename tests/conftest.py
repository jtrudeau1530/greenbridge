"""Shared fixtures for the translator tests.

Nothing here touches game data: shaders are tiny hand-written assembly programs, assembled with llvm-mc into the
same ``; encoding:`` listing format that tools/shader_disasm.py produces, next to a synthetic empty DXBC container
(rdna2spirv.translate() looks for ``<disasm dir>/../<name>.dxbc`` to read the input signature).
"""
import glob
import os
import shutil
import struct
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import rdna2spirv  # noqa: E402


def _has_amdgpu(mc):
    """True if this llvm-mc was built with the AMDGPU target (the official Windows release archives are not)."""
    try:
        r = subprocess.run([mc, "--version"], capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return False
    return "amdgcn" in r.stdout


def _find_llvm_mc():
    candidates = []
    env = os.environ.get("LLVM_MC")
    if env:
        candidates.append(shutil.which(env))      # None when the path doesn't exist
    candidates.append(shutil.which("llvm-mc"))
    # Debian/Ubuntu install it versioned, out of PATH.
    candidates += sorted(glob.glob("/usr/lib/llvm-*/bin/llvm-mc"), reverse=True)
    # Default location of the official Windows installer, which doesn't add LLVM to PATH unless asked to.
    candidates += glob.glob(r"C:\Program Files\LLVM\bin\llvm-mc.exe")
    for c in candidates:
        if c and _has_amdgpu(c):
            return c
    return None


LLVM_MC = _find_llvm_mc()
HAVE_SPIRV = shutil.which("spirv-as") and shutil.which("spirv-val")

requires_tools = pytest.mark.skipif(
    not (LLVM_MC and HAVE_SPIRV),
    reason=("needs an llvm-mc built with the amdgcn target, and spirv-as/spirv-val on PATH; "
            f"LLVM_MC={os.environ.get('LLVM_MC')!r}, found llvm-mc: {LLVM_MC!r}, "
            f"spirv-as/val: {bool(HAVE_SPIRV)}"),
)

# 32-byte DXBC header: magic, digest, version, size, chunk count = 0. Gives an empty input signature.
EMPTY_DXBC = b"DXBC" + bytes(16) + struct.pack("<III", 1, 32, 0)

# Compute shaders take their system registers from the FSSM header (rdna2spirv.translate reads it from
# <disasm dir>/../fssm/<name>.fssm): word 0x138 is COMPUTE_PGM_RSRC2, words 0x13C/0x140/0x144 the thread-group size.
# 0x8088 = 4 user SGPRs, TGID_X, 512 B of LDS; group 64x1x1 (the header the cs_* samples in runtime/test/shaders use).
DEFAULT_CS_HEADER = (0x8088, (64, 1, 1))


class Shader:
    def __init__(self, workdir, name, asm, cs_header=DEFAULT_CS_HEADER):
        self.name = name
        disasm = workdir / "disasm"
        disasm.mkdir(exist_ok=True)
        (workdir / f"{name}.dxbc").write_bytes(EMPTY_DXBC)
        if name.startswith("cs_"):
            rsrc2, (lx, ly, lz) = cs_header
            (workdir / "fssm").mkdir(exist_ok=True)
            (workdir / "fssm" / f"{name}.fssm").write_bytes(bytes(0x138) + struct.pack("<4I", rsrc2, lx, ly, lz))
        r = subprocess.run([LLVM_MC, "-triple=amdgcn--amdpal", "-mcpu=gfx1013", "-show-encoding"],
                           input=asm, capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        self.path = disasm / f"{name}.s"
        self.path.write_text(r.stdout)
        self.workdir = workdir

    def translate(self):
        return rdna2spirv.translate(str(self.path))

    def validate(self, spvasm):
        sa = self.workdir / f"{self.name}.spvasm"
        sv = self.workdir / f"{self.name}.spv"
        sa.write_text(spvasm)
        r = subprocess.run(["spirv-as", str(sa), "-o", str(sv), "--target-env", "vulkan1.2"],
                           capture_output=True, text=True)
        assert r.returncode == 0, "spirv-as: " + r.stderr
        r = subprocess.run(["spirv-val", "--target-env", "vulkan1.2", str(sv)], capture_output=True, text=True)
        assert r.returncode == 0, "spirv-val: " + r.stderr


@pytest.fixture
def make_shader(tmp_path):
    """make_shader(asm, name="ps_test") -> Shader. The name's prefix ("ps_", "vs_", "cs_") selects the stage.
    For cs_ shaders, cs_header=(RSRC2, (x, y, z)) sets the synthetic FSSM header."""
    def make(asm, name="ps_test", **kw):
        return Shader(tmp_path, name, asm, **kw)
    return make
