"""Find the tools, build the launcher, assemble kernels and run them on the GPU. Standard library only.

Tools are located through environment variables first, then by looking in the usual places:
    LEONIDA_AMD_CLANG   AMD clang (its folder must also hold ld.lld); default: amdclang / clang on PATH that lists
                        the amdgcn target, %HIP_PATH% / %ROCM_PATH%, a .venv holding the `rocm-sdk-core` wheel
    LEONIDA_HIP_DLL     the HIP runtime (amdhip64_7.dll); default: the one in System32 (it ships with the driver)
    LEONIDA_HOST_CC     a C compiler for hw_oracle.c (MinGW gcc); default: gcc on PATH, C:\\msys64\\ucrt64\\bin
    LEONIDA_HW_ARCH     the GPU architecture to assemble for; default: what amdgpu-arch reports, else gfx1150
Anything missing raises Unavailable, which callers turn into exit 77 ("skipped").
"""
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile

try:
    from . import asm
except ImportError:   # run as a script
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import asm

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
SENTINEL = 0xDEAD0000


class Unavailable(Exception):
    """A required tool or the GPU is not there."""


def _exe(name):
    return name + (".exe" if os.name == "nt" else "")


def _has_amdgcn(clang):
    try:
        out = subprocess.run([clang, "--print-targets"], capture_output=True, text=True, timeout=60).stdout
    except (OSError, subprocess.SubprocessError):
        return False
    return "amdgcn" in out


def find_clang():
    env = os.environ.get("LEONIDA_AMD_CLANG")
    if env:
        if not os.path.isfile(env):
            raise Unavailable(f"LEONIDA_AMD_CLANG={env} does not exist")
        return env
    cands = [shutil.which(n) for n in ("amdclang", "clang")]
    for var in ("HIP_PATH", "ROCM_PATH"):
        root = os.environ.get(var)
        if root:
            cands += [os.path.join(root, "lib", "llvm", "bin", _exe("clang")), os.path.join(root, "bin", _exe("clang"))]
    tail = os.path.join("Lib", "site-packages", "_rocm_sdk_core", "lib", "llvm", "bin", _exe("clang"))
    cands.append(os.path.join(REPO, ".venv", tail))   # only the repo root's venv, never a glob under the cwd
    for c in cands:
        if c and os.path.isfile(c) and _has_amdgcn(c):
            return c
    raise Unavailable("no AMD clang (set LEONIDA_AMD_CLANG; `pip install rocm-sdk-core` from "
                      "https://stable.repo.amd.com/rocm/whl-next/ provides one)")


def find_hip_dll():
    env = os.environ.get("LEONIDA_HIP_DLL")
    if env:
        if not os.path.isfile(env):
            raise Unavailable(f"LEONIDA_HIP_DLL={env} does not exist")
        return env
    if os.name != "nt":
        raise Unavailable("the launcher targets the Windows AMD driver (amdhip64_7.dll)")
    for n in ("amdhip64_7.dll", "amdhip64_6.dll"):   # the launcher loads whatever path this returns
        p = os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32", n)
        if os.path.isfile(p):
            return p
    raise Unavailable("no amdhip64_7.dll in System32 (the AMD Windows driver provides it)")


def find_cc():
    env = os.environ.get("LEONIDA_HOST_CC")
    if env:
        if not os.path.isfile(env):
            raise Unavailable(f"LEONIDA_HOST_CC={env} does not exist")
        return env
    for c in (shutil.which("gcc"), r"C:\msys64\ucrt64\bin\gcc.exe", r"C:\msys64\mingw64\bin\gcc.exe"):
        if c and os.path.isfile(c):
            return c
    raise Unavailable("no MinGW gcc for hw_oracle.c (set LEONIDA_HOST_CC)")


def detect_arch(clang):
    env = os.environ.get("LEONIDA_HW_ARCH")
    if env:
        return env
    tool = os.path.join(os.path.dirname(clang), _exe("amdgpu-arch"))
    if os.path.isfile(tool):
        try:
            out = subprocess.run([tool], capture_output=True, text=True, timeout=60).stdout.split()
            if out:
                return out[0]
        except (OSError, subprocess.SubprocessError):
            pass
    return "gfx1150"


def _find_lld(clang):
    p = os.path.join(os.path.dirname(clang), _exe("ld.lld"))
    if os.path.isfile(p):
        return p
    found = shutil.which("ld.lld")
    if found:
        return found
    raise Unavailable("no ld.lld next to the AMD clang")


class Oracle:
    """Assembles batches of scalar programs for one architecture and runs them on device 0."""

    def __init__(self, arch=None, workdir=None, chunk=500):
        if not os.path.isfile(os.path.join(HERE, "hw_oracle.c")):
            raise Unavailable("hw_oracle.c is missing")
        self.clang = find_clang()
        self.lld = _find_lld(self.clang)
        self.dll = find_hip_dll()
        self.arch = arch or detect_arch(self.clang)
        asm.family(self.arch)
        self.chunk = chunk
        self.workdir = workdir or tempfile.mkdtemp(prefix="leonida-hw-oracle-")
        self.exe = self._build_launcher()
        self.device = self._probe()

    def _build_launcher(self):
        src = os.path.join(HERE, "hw_oracle.c")
        with open(src, "rb") as f:
            tag = hashlib.sha1(f.read()).hexdigest()[:10]
        exe = os.path.join(tempfile.gettempdir(), f"leonida_hw_oracle_{tag}{'.exe' if os.name == 'nt' else ''}")
        if not os.path.isfile(exe):
            tmp = exe + f".{os.getpid()}"
            r = subprocess.run([find_cc(), "-O1", src, "-o", tmp], capture_output=True, text=True)
            if r.returncode:
                raise Unavailable("hw_oracle.c did not build: " + r.stderr.strip()[:300])
            os.replace(tmp, exe)
        return exe

    def _probe(self):
        """Run an empty kernel list: confirms the DLL loads and a device exists."""
        text, names, _ = asm.module_text([["v_mov_b32 v8, 0", "v_mov_b32 v9, 0", "v_mov_b32 v10, 0", "v_mov_b32 v11, 0"]],
                                         self.arch, prefix="probe")
        co, err = self._assemble(text)
        if co is None:
            raise Unavailable("probe kernel did not assemble: " + err[:300])
        rc, out, err = self._launch(co, names)
        if rc == 77:
            raise Unavailable(err.strip() or "no AMD device")
        if rc:
            raise Unavailable(f"probe launch failed (exit {rc}): {err.strip()[:300]}")
        head = out.splitlines()[0] if out else ""
        if "arch=" in head and self.arch not in head:
            raise Unavailable(f"device reports {head!r} but assembling for {self.arch} (set LEONIDA_HW_ARCH)")
        return head.lstrip("# ")

    def _assemble(self, text):
        s = os.path.join(self.workdir, "k.s")
        o = os.path.join(self.workdir, "k.o")
        co = os.path.join(self.workdir, "k.co")
        with open(s, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
        r = subprocess.run([self.clang, f"--target=amdgcn-amd-amdhsa", f"-mcpu={self.arch}", "-c", s, "-o", o],
                           capture_output=True, text=True)
        if r.returncode:
            return None, r.stderr
        r = subprocess.run([self.lld, "-shared", o, "-o", co], capture_output=True, text=True)
        if r.returncode:
            return None, r.stderr
        return co, ""

    def _launch(self, co, names, words=4):
        r = subprocess.run([self.exe, co, str(words), self.dll], input="\n".join(names) + "\n", capture_output=True, text=True)
        return r.returncode, r.stdout, r.stderr

    def run(self, programs, words=4):
        """Run each program (a list of lines). Returns one entry per program: a list of `words` ints, or a str
        explaining why it could not run (assembler rejection)."""
        results = [None] * len(programs)
        for lo in range(0, len(programs), self.chunk):
            idx = list(range(lo, min(lo + self.chunk, len(programs))))
            self._run_chunk(programs, idx, results, words)
        return results

    def _run_chunk(self, programs, idx, results, words):
        live = list(idx)
        while live:
            text, names, starts = asm.module_text([programs[i] for i in live], self.arch)
            co, err = self._assemble(text)
            if co is not None:
                break
            bad = asm.failing_kernels(err, starts)
            if not bad:
                raise RuntimeError("assembler failed with no kernel to blame:\n" + err[:600])
            first = err.strip().splitlines()[0] if err.strip() else "?"
            for k in bad:
                results[live[k]] = "assembler rejected: " + first.split("error:", 1)[-1].strip()
            live = [i for k, i in enumerate(live) if k not in bad]
        else:
            # every program in the chunk was rejected: that is a broken harness (a bad prologue, the wrong
            # architecture), not a thousand skips
            raise RuntimeError("the assembler rejected every program in a chunk: " +
                               (str(results[idx[0]]) if idx else "empty chunk"))
        rc, out, err = self._launch(co, names, words)
        if rc:
            raise RuntimeError(f"hw_oracle exited {rc}: {err.strip()[:300]}")
        got = {}
        for line in out.splitlines():
            if line.startswith("#") or not line.strip():
                continue
            name, *w = line.split()
            got[name] = [int(x, 16) for x in w]
        for n, i in zip(names, live):
            results[i] = got.get(n, "kernel produced no output")

    def close(self):
        shutil.rmtree(self.workdir, ignore_errors=True)


def open_oracle(**kw):
    """An Oracle, or raise Unavailable."""
    return Oracle(**kw)


if __name__ == "__main__":
    try:
        o = Oracle()
    except Unavailable as e:
        print(f"hardware oracle unavailable: {e}")
        sys.exit(77)
    print(f"hardware oracle ready: {o.device}; clang {o.clang}")
    o.close()
