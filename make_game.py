r"""Build the PC port from your own copy of the game.

    Build.bat                                                   # double-click: finds your copy and builds
    python -I make_game.py                                      # same, from a terminal
    python -I make_game.py --find                               # list the copies found on this PC
    python -I make_game.py --game "D:\path\to\your\package" --out "D:\Games\Leonida"
    python -I make_game.py --game "..." --check                 # only check the copy, build nothing
    python -I make_game.py --stages shaders,assets              # only some stages

The game folder is your own Xbox Series package, version 1.0.2.1, with its files already readable on disk
(see game/xbox-1.0.2.1.json for the file list). It is found automatically in Downloads, Desktop, Documents,
Games, your home folder and the top of each drive (two folder levels down). This script never decrypts or
works around anything: it reads plain files and writes everything it makes under --out. The only thing it
downloads is the third_party/rpf9-explorer submodule, when it is missing. Never share what it makes.

Stages, in order (each skips work that is already done, so a stopped build continues where it left off):
    check     file names and sizes against game/<version>.json (add --full-check for SHA-256, slow)
    shaders   carve shaders.rpf, unpack the native RDNA2 code, disassemble, translate to SPIR-V
    assets    models (OBJ), texture dictionaries and texture pools (DDS), audio banks (AWC; --wav adds PCM WAV)
    runtime   build the loader (needs the game's executable to run, not to build)
    pipelines create every translated shader on your GPU's driver (and every compute pipeline), report anything
              the driver rejects, and save the pipeline cache next to the port so the first boot doesn't compile
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
TOOLS = os.path.join(ROOT, "tools")
VERSIONS = os.path.join(ROOT, "game")
STAGES = ["check", "shaders", "assets", "runtime", "pipelines"]
# Archives scanned for models and texture dictionaries, small to large.
MODEL_ARCHIVES = [
    "common.rpf", "textures.rpf", "data.rpf",
    "pack/story000_restofworld/story000_restofworld.rpf", "pack/shared_minigame000/shared_minigame000.rpf",
    "pack/core_restofworld/core_restofworld.rpf", "pack/core_restofworld001/core_restofworld001.rpf",
    "packs.rpf", "pack/story000/story000.rpf", "levels.rpf",
]


def log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')}  {msg}", flush=True)


def fail(msg: str) -> "None":
    print(f"\nERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def below_normal() -> None:
    """Keep the PC usable while a long build runs."""
    if os.name == "nt":
        import ctypes
        k32 = ctypes.windll.kernel32
        k32.SetPriorityClass(k32.GetCurrentProcess(), 0x00004000)


def run(args: list, **kw) -> None:
    log("  " + " ".join(os.path.basename(str(a)) if i == 0 else str(a) for i, a in enumerate(args)))
    r = subprocess.run([str(a) for a in args], **kw)
    if r.returncode != 0:
        fail(f"{os.path.basename(str(args[0]))} exited with {r.returncode}")


def py(tool: str, *args) -> None:
    run([sys.executable, "-I", os.path.join(TOOLS, tool), *args])


def tool_available(cmd: str) -> bool:
    """A PATH name, a path, or a whole command line such as 'wsl -e /usr/lib/llvm-21/bin/llvm-mc'."""
    cmd = cmd.strip().strip('"')
    if os.path.isfile(cmd):
        return True
    head = cmd.split()[0].strip('"') if cmd.split() else ""
    return bool(head and (shutil.which(head) or os.path.isfile(head)))


def need(exe: str, hint: str) -> str:
    if not tool_available(exe):
        fail(f"'{exe}' was not found on PATH. {hint}")
    return exe


# ---------------------------------------------------------------- check
def versions() -> list:
    out = []
    for name in sorted(os.listdir(VERSIONS)):
        if name.endswith(".json"):
            with open(os.path.join(VERSIONS, name), encoding="utf-8") as f:
                out.append(json.load(f))
    return out


# Small files checked by size to recognise a version without reading anything large.
PROBES = ("MicrosoftGame.Config", "Catalog000.bin", "appdata0.rpf")


def match_version(game: str):
    """The supported version this folder is, or None. Only names and sizes are read."""
    for v in versions():
        probes = [p for p in PROBES if p in v["files"]]
        if probes and all(os.path.isfile(os.path.join(game, p)) and
                          os.path.getsize(os.path.join(game, p)) == v["files"][p]["size"] for p in probes):
            return v
    return None


def load_version(game: str) -> dict:
    v = match_version(game)
    if v is None:
        fail("This doesn't look like a supported version of the game. Point --game at the folder that holds "
             "MicrosoftGame.Config and the .rpf archives. Supported: "
             + ", ".join(f"{x['title']} {x['version']}" for x in versions()))
    return v


def find_dumps() -> list:
    """Folders on this PC that hold a supported version: Downloads, Desktop, Documents, the home folder and the
    top of each drive, up to two folder levels down. Only folder names and a few file sizes are read."""
    home = os.path.expanduser("~")
    roots = [os.path.join(home, d) for d in ("Downloads", "Desktop", "Documents", "Games")] + [home]
    if os.name == "nt":
        import string
        roots += [f"{d}:\\" for d in string.ascii_uppercase if os.path.isdir(f"{d}:\\")]
    else:
        roots += ["/mnt", "/media"]
    skip = {"windows", "program files", "program files (x86)", "programdata", "appdata", "$recycle.bin",
            "system volume information", "node_modules", ".git"}
    seen, found = set(), []

    def visit(d: str, depth: int) -> None:
        try:
            real = os.path.realpath(d)
        except OSError:
            return
        if real.lower() in seen:
            return
        seen.add(real.lower())
        if os.path.isfile(os.path.join(d, "MicrosoftGame.Config")):
            v = match_version(d)
            if v:
                found.append((d, v))
            return
        if depth == 0:
            return
        try:
            with os.scandir(d) as it:
                subs = [e.path for e in it if e.is_dir(follow_symlinks=False)
                        and e.name.lower() not in skip and not e.name.startswith((".", "$"))]
        except OSError:
            return
        for s in subs:
            visit(s, depth - 1)

    for r in roots:
        if os.path.isdir(r):
            visit(r, 2)
    return found


def choose_game() -> str:
    """No --game given: find the dump, or ask for it."""
    log("looking for your game files...")
    found = find_dumps()
    interactive = sys.stdin is not None and sys.stdin.isatty()
    if len(found) == 1:
        log(f"found {found[0][1]['title']} {found[0][1]['version']} in {found[0][0]}")
        return found[0][0]
    if found:
        for i, (d, v) in enumerate(found, 1):
            print(f"  {i}. {d}  ({v['title']} {v['version']})")
        if not interactive:
            fail("More than one copy found: choose one with --game.")
        while True:
            pick = input("Which one? ").strip()
            if pick.isdigit() and 1 <= int(pick) <= len(found):
                return found[int(pick) - 1][0]
    if not interactive:
        fail("Your game files weren't found. Give the folder with --game.")
    print("Your game files weren't found automatically. Paste the folder that holds MicrosoftGame.Config "
          "and the .rpf archives.")
    while True:
        d = input("Game folder: ").strip().strip('"')
        if d and match_version(d):
            return d
        print("  That folder isn't a supported version of the game. Try again.")


def default_out() -> str:
    return os.path.join(os.path.expanduser("~"), "Documents", "Leonida")


# ---------------------------------------------------------------- preflight
MIN_FREE_GB = {"shaders": 30, "assets": 300}


def preflight(stages: list, work: str, wav: bool = False) -> None:
    """Find every missing prerequisite before a long build starts, and list them all at once."""
    missing = []
    if wav and "assets" in stages and not shutil.which("ffmpeg"):
        missing.append("ffmpeg (needed for --wav): https://ffmpeg.org/download.html, then reopen the terminal")
    if sys.version_info < (3, 11):
        missing.append("Python 3.11 or newer: https://www.python.org/downloads/")
    if "shaders" in stages:
        llvm = os.environ.get("LLVM_MC", "llvm-mc")
        if not tool_available(llvm):
            missing.append(
                "llvm-mc (LLVM 21+ with the AMDGPU target). The official Windows LLVM build has no AMDGPU "
                "target: install llvm-21 in WSL and set LLVM_MC to the whole command, e.g. "
                "LLVM_MC=\"wsl -d Ubuntu -e /usr/lib/llvm-21/bin/llvm-mc\" (docs/design.md)")
        for t in ("spirv-as", "spirv-val"):
            if not shutil.which(t):
                missing.append(f"{t}: install the Vulkan SDK, https://vulkan.lunarg.com/ (then reopen the terminal)")
    if "assets" in stages:
        if not shutil.which("dotnet"):
            missing.append(".NET 8 SDK: https://dotnet.microsoft.com/download/dotnet/8.0")
        ensure_submodule()
    if "runtime" in stages and os.name == "nt":
        vswhere = os.path.join(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"),
                               "Microsoft Visual Studio", "Installer", "vswhere.exe")
        if not os.path.isfile(vswhere):
            missing.append("Visual Studio 2022 with 'Desktop development with C++'")
    if missing:
        fail("Install these first, then run the build again:\n  - " + "\n  - ".join(dict.fromkeys(missing)))
    need_gb = sum(MIN_FREE_GB.get(s, 0) for s in stages)
    os.makedirs(work, exist_ok=True)
    free_gb = shutil.disk_usage(work).free / 2**30
    if free_gb < need_gb:
        fail(f"{free_gb:.0f} GB free on the drive of {work}, about {need_gb} GB is needed. "
             "Free some space or choose another drive with --work.")


def ensure_submodule() -> None:
    core = os.path.join(ROOT, "third_party", "rpf9-explorer", "Core")
    if os.path.isdir(core) and os.listdir(core):
        return
    git = shutil.which("git")
    if git and subprocess.run([git, "-C", ROOT, "rev-parse", "--resolve-git-dir", ".git"],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0:
        log("fetching third_party/rpf9-explorer")
        r = subprocess.run([git, "-C", ROOT, "submodule", "update", "--init", "--recursive"])
        if r.returncode == 0 and os.path.isdir(core):
            return
    fail("third_party/rpf9-explorer is missing. Download the project with: "
         "git clone --recursive <this repository's URL>")


def check(game: str, full: bool) -> dict:
    if not os.path.isdir(game):
        fail(f"Game folder not found: {game}")
    v = load_version(game)
    log(f"check: {v['title']} {v['version']}, {len(v['files'])} files")
    bad = 0
    for rel, info in v["files"].items():
        p = os.path.join(game, rel)
        if not os.path.isfile(p):
            print(f"  missing  {rel}"); bad += 1
        elif os.path.getsize(p) != info["size"]:
            print(f"  wrong size  {rel}"); bad += 1
    if bad:
        fail(f"{bad} file(s) missing or different. Use a complete, unmodified copy of version {v['version']}.")
    if full:
        py("manifest.py", "verify", game, "--manifest", os.path.join(VERSIONS, f"xbox-{v['version']}.json"))
    exe = os.path.join(game, v["expected_exe"])
    log(f"check: all files present{' and hashes match' if full else ''}. "
        + ("Executable found." if os.path.isfile(exe) else
           f"{v['expected_exe']} is not in the folder: the runtime stage can be built but not run."))
    return v


# ---------------------------------------------------------------- shaders
def shaders(game: str, work: str, jobs: int) -> None:
    src = os.path.join(game, "shaders.rpf")
    sh = os.path.join(work, "shaders")
    dxbc = os.path.join(sh, "dxbc")
    spirv = os.path.join(sh, "spirv")
    # Each stage counts as finished only when the file it writes last is there, never because its folder exists
    # or holds many files: a run stopped part-way leaves both. Re-running a stage drops the later stages' proof.
    carved = os.path.join(dxbc, "index.json")             # shader_carve.py, renamed into place at the end
    unpacked = os.path.join(dxbc, "fssm_index.json")      # fssm_unpack.py, the same
    disassembled = os.path.join(dxbc, "disasm.done")      # written here once shader_disasm.py exits with 0
    translated = os.path.join(spirv, "index.json")        # rdna2spirv.py --merge, renamed into place
    if not os.path.isfile(carved):
        log("shaders: carving shaders.rpf")
        drop(unpacked, disassembled, translated)
        py("shader_carve.py", src, dxbc)
    if not os.path.isfile(unpacked):
        log("shaders: unpacking native code")
        drop(disassembled, translated)
        py("fssm_unpack.py", dxbc)
    # rdna2spirv.py finds index.json, the .dxbc files and fssm/ in the parent of the .s files' folder, so the
    # disassembly must sit inside the carve folder: next to it, every stage reads as "?" and is rejected.
    disasm = os.path.join(dxbc, "disasm")
    old = os.path.join(sh, "disasm")
    if os.path.isdir(old) and not os.path.isdir(disasm):
        log("shaders: moving disasm into the carve folder and translating again")
        os.rename(old, disasm)
        shutil.rmtree(spirv, ignore_errors=True)
    if not os.path.isfile(disassembled):
        need(os.environ.get("LLVM_MC", "llvm-mc"),
             "Install LLVM 21 or newer with the AMDGPU target, or set LLVM_MC to llvm-mc's full path.")
        log("shaders: disassembling (gfx1013)")
        drop(translated)
        py("shader_disasm.py", os.path.join(dxbc, "fssm"), "--out", disasm, "--mcpu", "gfx1013", "--jobs", jobs)
        with open(disassembled, "w", encoding="utf-8") as f:
            f.write("shader_disasm.py finished\n")
    if not os.path.isfile(translated):
        need("spirv-as", "Install the Vulkan SDK (it includes spirv-as and spirv-val) and reopen the terminal.")
        need("spirv-val", "Install the Vulkan SDK and reopen the terminal.")
        log(f"shaders: translating RDNA2 to SPIR-V on {jobs} processes (the longest step)")
        translate(disasm, spirv, jobs)
    log("shaders: done")


def drop(*paths: str) -> None:
    """Remove the files that mark later stages finished, so they run again on this stage's new output."""
    for p in paths:
        if os.path.isfile(p):
            os.remove(p)


def translate(disasm: str, spirv: str, jobs: int) -> None:
    """rdna2spirv.py runs on one core: start one --shard per job side by side, then merge their index parts."""
    os.makedirs(spirv, exist_ok=True)
    for f in os.listdir(spirv):  # parts from an earlier run with another --jobs would be merged in too
        if f.startswith("index.part") and f.endswith(".json"):
            os.remove(os.path.join(spirv, f))
    tool = os.path.join(TOOLS, "rdna2spirv.py")
    log(f"  rdna2spirv.py {disasm} --out {spirv} --shard 0..{jobs - 1}/{jobs}")
    # --no-asm: keep only the .spv modules; the .spvasm text would add ~35 GB for the corpus
    procs = [subprocess.Popen([sys.executable, "-I", tool, disasm, "--out", spirv, "--shard", f"{i}/{jobs}", "--no-asm"])
             for i in range(jobs)]
    bad = [f"shard {i} exited with {r}" for i, r in enumerate(p.wait() for p in procs) if r != 0]
    if bad:
        fail("rdna2spirv.py: " + ", ".join(bad))
    py("rdna2spirv.py", disasm, "--out", spirv, "--merge")


# ---------------------------------------------------------------- assets
def rpf9dump(work: str) -> str:
    exe = os.path.join(work, "tools", "rpf9dump", "rpf9dump.exe" if os.name == "nt" else "rpf9dump")
    if not os.path.isfile(exe):
        core = os.path.join(ROOT, "third_party", "rpf9-explorer", "Core")
        if not os.path.isdir(core):
            fail("third_party/rpf9-explorer is empty. Run: git submodule update --init")
        need("dotnet", "Install the .NET 8 SDK.")
        log("assets: building rpf9dump")
        run(["dotnet", "build", os.path.join(TOOLS, "rpf9dump", "rpf9dump.csproj"), "-c", "Release",
             "-o", os.path.dirname(exe), "--nologo", "-v", "q"])
    return exe


def assets(game: str, work: str, wav: bool, jobs: int) -> None:
    exe = rpf9dump(work)
    out = os.path.join(work, "assets")
    for rel in MODEL_ARCHIVES:
        src = os.path.join(game, rel)
        name = os.path.splitext(os.path.basename(rel))[0]
        for mode in ("models", "textures"):
            if os.path.isfile(os.path.join(out, name, mode, "manifest.json")):
                continue
            log(f"assets: {mode} from {rel}")
            run([exe, mode, src, out])
    # Texture pools: data/ttdb/** and each pack's own pack/<name>/data/ttdb/** (pool names repeat across packs,
    # so pack pools go under ttpf_pack/<pack>/).
    pools = []
    for root, _, files in os.walk(game):
        pools += [os.path.join(root, f) for f in files if f.lower().endswith(".ttpf")]
    for p in sorted(pools, key=os.path.getsize):
        name = os.path.splitext(os.path.basename(p))[0]
        parts = os.path.relpath(p, game).replace("\\", "/").split("/")
        dest = os.path.join(out, "ttpf_pack", parts[1]) if parts[0] == "pack" else os.path.join(out, "ttpf")
        if os.path.isfile(os.path.join(dest, name, "manifest.json")):
            continue
        log(f"assets: texture pool {os.path.relpath(p, game)}")
        run([exe, "ttpf", p, dest])
    if not os.path.isfile(os.path.join(out, "audio", "audio_manifest.json")):
        log("assets: audio banks")
        run([exe, "audio", os.path.join(game, "audio.rpf"), out])
    if not os.path.isdir(os.path.join(out, "audio", "ogg")):
        # Lossless: the banks' Opus packets repackaged as standard Ogg Opus files (PCM streams as .wav)
        log("assets: audio streams -> Ogg Opus")
        py("awc_opus.py", os.path.join(out, "audio", "awc"), os.path.join(out, "audio", "ogg"))
    if wav:
        ogg = os.path.join(out, "audio", "ogg")
        if not os.path.isdir(ogg):
            fail("assets: no Ogg Opus output to decode. The audio bank step must finish first (run the assets stage).")
        # Opt-in: lossless Ogg Opus -> 16-bit PCM WAV, several times larger. Existing WAVs are skipped.
        log("assets: audio Ogg Opus -> PCM WAV")
        py("audio_wav.py", ogg, "--out", os.path.join(out, "audio", "wav"), "--jobs", str(jobs))
    xma_dir = os.path.join(out, "audio", "xma")
    if not os.path.isdir(xma_dir):
        log("assets: XMA2 streams")
        run([exe, "xma", os.path.join(game, "audio.rpf"), out])
        ff = shutil.which("ffmpeg")
        if ff:
            log("assets: XMA2 -> PCM WAV (ffmpeg)")
            for root, _, files in os.walk(xma_dir):
                for f in files:
                    if f.endswith(".xma.wav"):
                        src = os.path.join(root, f)
                        subprocess.run([ff, "-v", "error", "-y", "-i", src, "-c:a", "pcm_s16le", src[:-8] + ".wav"])
        else:
            log("assets: ffmpeg not on PATH: XMA2 streams kept as .xma.wav (ffmpeg decodes them)")
    log("assets: done")


# ---------------------------------------------------------------- runtime
def runtime(game: str, out: str, v: dict) -> None:
    if os.name != "nt":
        fail("The runtime builds on Windows only for now.")
    log("runtime: building")
    run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
         os.path.join(ROOT, "scripts", "build_runtime.ps1")])
    built = os.path.join(ROOT, "runtime", "build", "gta6rt.exe")
    os.makedirs(out, exist_ok=True)
    shutil.copy2(built, os.path.join(out, "gta6rt.exe"))
    with open(os.path.join(out, "game_path.txt"), "w", encoding="utf-8") as f:
        f.write(os.path.abspath(game))
    if not os.path.isfile(os.path.join(game, v["expected_exe"])):
        log(f"runtime: built, but {v['expected_exe']} is missing from your copy, so there is nothing to run yet.")
    else:
        log(f"runtime: done. Start {os.path.join(out, 'gta6rt.exe')}")


def pipelines(work: str, out: str) -> None:
    exe = os.path.join(ROOT, "runtime", "build", "vk_pipeline_check.exe")
    spirv = os.path.join(work, "shaders", "spirv")
    if not os.path.isfile(exe):
        fail("vk_pipeline_check.exe wasn't built: run the runtime stage with the Vulkan SDK installed.")
    if not os.path.isfile(os.path.join(spirv, "index.json")):
        fail("No translated shaders yet: run the shaders stage first.")
    log("pipelines: creating every shader module and compute pipeline on this GPU")
    os.makedirs(out, exist_ok=True)
    run([exe, spirv, os.path.join(out, "pipeline_cache.bin")])
    log("pipelines: done")


def main() -> int:
    ap = argparse.ArgumentParser(description="Build the PC port from your own copy of the game.")
    ap.add_argument("--game", help="your package folder (found automatically if left out)")
    ap.add_argument("--out", help="folder for the finished port (default Documents/Leonida)")
    ap.add_argument("--work", help="folder for intermediate files (hundreds of GB; default <out>/work)")
    ap.add_argument("--stages", default=",".join(STAGES), help="comma list from: " + ",".join(STAGES))
    ap.add_argument("--check", action="store_true", help="only check the game files")
    ap.add_argument("--full-check", action="store_true", help="also compare SHA-256 of every file (slow)")
    ap.add_argument("--wav", action="store_true",
                    help="also decode the Ogg Opus audio to 16-bit PCM WAV (much larger; needs ffmpeg)")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument("--find", action="store_true", help="only list the game copies found on this PC")
    a = ap.parse_args()
    below_normal()
    if a.find:
        found = find_dumps()
        for d, v in found:
            print(f"{d}  ({v['title']} {v['version']})")
        print(f"{len(found)} found")
        return 0
    game = os.path.abspath(a.game or choose_game())
    v = check(game, a.full_check)
    if a.check:
        return 0
    out = os.path.abspath(a.out or default_out())
    work = os.path.abspath(a.work or os.path.join(out, "work"))
    for p in (out, work):
        try:
            inside = os.path.commonpath([p, game]) == game
        except ValueError:          # different drives
            inside = False
        if inside:
            fail("--out and --work must be outside the game folder.")
    stages = [s.strip() for s in a.stages.split(",") if s.strip()]
    for s in stages:
        if s not in STAGES:
            fail(f"unknown stage '{s}'")
    preflight(stages, work, a.wav)
    log(f"building into {out} (work files in {work})")
    t0 = time.time()
    if "shaders" in stages:
        shaders(game, work, a.jobs)
    if "assets" in stages:
        assets(game, work, a.wav, a.jobs)
    if "runtime" in stages:
        runtime(game, out, v)
    if "pipelines" in stages:
        pipelines(work, out)
    log(f"finished in {(time.time() - t0) / 60:.0f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
