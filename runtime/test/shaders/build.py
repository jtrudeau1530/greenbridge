"""Build the hand-written test shaders in this folder into SPIR-V.

    python3 -I runtime/test/shaders/build.py <out dir> [name ...]

Each .asm is RDNA2 assembly. It is assembled with llvm-mc (LLVM 21+ with the AMDGPU target; set LLVM_MC to its
path or to a whole command, as for the shaders stage). Listings, synthetic FSSM headers and DXBC containers are
staged in a temporary directory. After validation, the build writes <out>/spirv/<name>.spv and the translator
metadata.

Compute shaders can override the default header with leading "; group: X,Y,Z" and "; rsrc2: 0x..." comments.
The stage comes from the name's prefix. Mesh and geometry shaders (ms_, gs_) also need a DXBC container with
their signatures, which this doesn't make.
The comment at the top of each .asm says which test program to run on the result and what it should print.
"""
import json
import os
import re
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS = os.path.join(HERE, "..", "..", "..", "tools")
TRANSLATOR = os.path.join(TOOLS, "rdna2spirv.py")
sys.path.insert(0, TOOLS)
import mc_command  # noqa: E402

MC = mc_command.mc_command()
RSRC2, GROUP = 0x8088, (64, 1, 1)
sys.path.insert(0, HERE)  # this script is also run with python -I
from compute_fixture import compute_header  # noqa: E402


def asm_mattr(path):
    for ln in Path(path).read_text(encoding="utf-8").splitlines():
        s = ln.strip()
        if not s:
            continue
        if not s.startswith(";"):
            break
        m = re.match(r";\s*mattr:\s*(\S+)", s)
        if m:
            return m.group(1)
    return None


CLIP_CASES = {"vs_clip": 1, "vs_clip_param4": 4, "vs_clip_undeclared": 1}


def shader_source(name):
    return "vs_clip" if name in CLIP_CASES else name


def clip_container():
    # One OSG1 element: stream 0, SV_ClipDistance0.xy, float, logical register 2.
    name = b"SV_ClipDistance\0"
    payload = struct.pack("<8I2B6x", 1, 8, 0, 40, 0, 2, 3, 2, 3, 0) + name
    return (b"DXBC" + bytes(16) + struct.pack("<4I", 1, 44 + len(payload), 1, 36)
            + b"OSG1" + struct.pack("<I", len(payload)) + payload)


def translate_tests(work, names):
    disasm, fssm, spirv = (work / d for d in ("disasm", "fssm", "spirv"))
    disasm.mkdir()
    fssm.mkdir()
    for n in names:
        asm_path = os.path.join(HERE, shader_source(n) + ".asm")
        raw = Path(asm_path).read_text(encoding="utf-8")
        mattr = asm_mattr(asm_path)
        cmd = MC + ["-triple=amdgcn--amdpal", "-mcpu=gfx1013"]
        if n in CLIP_CASES:
            cmd.append(f"-defsym=PARAM_SLOT={CLIP_CASES[n]}")
        if mattr:
            cmd.append(f"-mattr={mattr}")
        cmd.append("-show-encoding")
        # Feed the text on stdin: an LLVM_MC command that runs in WSL cannot open a Windows path argument.
        r = subprocess.run(cmd, input=raw, capture_output=True, text=True)
        if r.returncode or "error:" in r.stderr:
            sys.exit(f"{n}.asm did not assemble:\n{r.stderr}")
        # rdna2spirv.py rejects assembler directives such as the "\t.text" llvm-mc prints first.
        code = "".join(ln for ln in r.stdout.splitlines(keepends=True) if not ln.lstrip().startswith("."))
        (disasm / (n + ".s")).write_text(f"; {n}  hand-written test\n{code}", encoding="utf-8")
        if n.startswith("cs_"):
            (fssm / (n + ".fssm")).write_bytes(compute_header(asm_path))
        elif n in CLIP_CASES and n != "vs_clip_undeclared":
            (work / (n + ".dxbc")).write_bytes(clip_container())
        else:
            (work / (n + ".dxbc")).write_bytes(bytes(32))  # chunk count 0
    cmd = [sys.executable, "-I", TRANSLATOR, disasm, "--out", spirv]
    if os.environ.get("LEONIDA_W64_EMULATE") == "1":   # wave64 on 32-wide subgroups (compute shaders only)
        cmd.append("--wave64-emulate")
    return subprocess.run(cmd).returncode


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    out = Path(sys.argv[1])
    available = sorted({p.stem for p in Path(HERE).glob("*.asm")} | CLIP_CASES.keys())
    names = sys.argv[2:] or available
    unknown = sorted(set(names) - set(available))
    if unknown:
        sys.exit("Unknown test shaders: " + ", ".join(unknown))
    skipped = [n for n in names if n.startswith(("ms_", "gs_"))]
    if skipped:
        print("not built (need a DXBC container with signatures):", ", ".join(skipped))
    names = [n for n in names if n not in skipped]
    if not names:
        return 0
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="test-shaders-", dir=out) as tmp:
        work = Path(tmp)
        status = translate_tests(work, names)
        if status:
            return status
        built = work / "spirv"
        index = built / "index.json"
        validated = json.loads(index.read_text(encoding="utf-8")) if index.exists() else {}
        missing = [n for n in names if n not in validated or not (built / (n + ".spv")).is_file()]
        if missing:
            sys.exit("Translation did not produce validated SPIR-V for: " + ", ".join(missing))
        spirv = out / "spirv"
        spirv.mkdir(exist_ok=True)
        published_index = spirv / index.name
        combined = json.loads(published_index.read_text(encoding="utf-8")) if published_index.exists() else {}
        combined.update(validated)
        index.write_text(json.dumps(combined, indent=2) + "\n", encoding="utf-8")
        for n in validated:
            for suffix in (".spv", ".spvasm"):
                artifact = built / (n + suffix)
                artifact.replace(spirv / artifact.name)
        index.replace(published_index)
        sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "tools"))
        import spirv_stubs
        spirv_stubs.generate_for_index(str(spirv), combined)
    return 0


if __name__ == "__main__":
    sys.exit(main())
