"""Build the hand-written test shaders in this folder into SPIR-V.

    python3 -I runtime/test/shaders/build.py <out dir> [name ...]

Each .asm is RDNA2 assembly. It is assembled with llvm-mc (LLVM 21+ with the AMDGPU target; set LLVM_MC to its
path, as for the shaders stage). Listings, synthetic FSSM headers and DXBC containers are staged in a temporary
directory. After validation, the build writes <out>/spirv/<name>.spv and the translator metadata.

Compute shaders can override the default header with leading "; group: X,Y,Z" and "; rsrc2: 0x..." comments.
The stage comes from the name's prefix. Mesh and geometry shaders (ms_, gs_) also need a DXBC container with
their signatures, which this doesn't make.
The comment at the top of each .asm says which test program to run on the result and what it should print.
"""
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
TRANSLATOR = os.path.join(HERE, "..", "..", "..", "tools", "rdna2spirv.py")
LLVM_MC = os.environ.get("LLVM_MC", "llvm-mc")
sys.path.insert(0, HERE)  # this script is also run with python -I
from compute_fixture import compute_header


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


def translate_tests(work, names):
    disasm, fssm, spirv = (work / d for d in ("disasm", "fssm", "spirv"))
    disasm.mkdir()
    fssm.mkdir()
    for n in names:
        asm_path = os.path.join(HERE, n + ".asm")
        mattr = asm_mattr(asm_path)
        cmd = [LLVM_MC, "-triple=amdgcn--amdpal", "-mcpu=gfx1013"]
        if mattr:
            cmd.append(f"-mattr={mattr}")
        cmd += ["-show-encoding", asm_path]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode or "error:" in r.stderr:
            sys.exit(f"{n}.asm did not assemble:\n{r.stderr}")
        # rdna2spirv.py rejects assembler directives such as the "\t.text" llvm-mc prints first.
        code = "".join(ln for ln in r.stdout.splitlines(keepends=True) if not ln.lstrip().startswith("."))
        (disasm / (n + ".s")).write_text(f"; {n}  hand-written test\n{code}", encoding="utf-8")
        if n.startswith("cs_"):
            (fssm / (n + ".fssm")).write_bytes(compute_header(asm_path))
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
    available = sorted(p.stem for p in Path(HERE).glob("*.asm"))
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
