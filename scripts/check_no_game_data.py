"""Fail when a tracked file starts like game data, whatever its name.

    python3 -I scripts/check_no_game_data.py            # every file `git ls-files` lists
    python3 -I scripts/check_no_game_data.py PATH...    # only these files

The extension check in .github/workflows/no-game-data.yml misses a renamed file: a DDS saved as .bin, an executable
as .txt. This reads the first bytes of each file and matches the magic numbers of the package's formats and the
files the build extracts from it: RPF9 archives, TTPF texture pools, RAGE resources, XBCH/FSSM/DXBC shader
containers, SPIR-V, DDS, AWC audio, Ogg, RIFF WAVE, Bink, glTF binaries, PE executables and XVD images. The magics are
the ones docs/ measures (docs/gamedata.md and the shader and audio notes). Text formats (disassembly, .obj, .spvasm)
have no magic and stay with the extension check.

Prints one line per match, `<path>: <format>`, and exits 1 if there is any. Standard library only, no game files.
"""
import os
import subprocess
import sys

HEAD = 0x208  # enough for the XVD magic at 0x200

# (format, offset, magic). Matched at the start of the file only, so the same bytes inside text don't count.
MAGICS = (
    ("RPF9 archive", 0, b"9FPR"),
    ("RPF9 archive", 0, b"RPF9"),
    ("TTPF texture pool", 0, b"TTPF"),
    ("RAGE resource", 0, b"RSC7"),
    ("RAGE resource", 0, b"RSC8"),
    ("RAGE resource", 0, b"RSCC"),
    ("RAGE page directory", 0, b"0DRP"),
    ("RBF0 tree", 0, b"RBF0"),
    ("XBCH shader container", 0, b"XBCH"),
    ("FSSM shader", 0, b"FSSM"),
    ("DXBC container", 0, b"DXBC"),
    ("SPIR-V module", 0, b"\x03\x02\x23\x07"),
    ("SPIR-V module", 0, b"\x07\x23\x02\x03"),
    ("DDS texture", 0, b"DDS "),
    ("AWC audio bank", 0, b"ADAT"),
    ("AWC audio bank", 0, b"TADA"),
    ("Ogg stream", 0, b"OggS"),
    ("Bink 2 video", 0, b"KB2"),
    ("Bink video", 0, b"BIK"),
    ("glTF binary", 0, b"glTF"),
    ("XVD image", 0x200, b"msft-xvd"),
)


def is_pe(head):
    """MZ header whose e_lfanew points at a PE signature inside the bytes read."""
    if not head.startswith(b"MZ") or len(head) < 0x40:
        return False
    offset = int.from_bytes(head[0x3C:0x40], "little")
    return head[offset:offset + 4] == b"PE\0\0"


def classify(head):
    """The format `head` (the first bytes of a file) starts like, or None."""
    for name, offset, magic in MAGICS:
        if head[offset:offset + len(magic)] == magic:
            return name
    if head.startswith(b"RIFF") and head[8:12] == b"WAVE":
        return "RIFF WAVE audio"
    if is_pe(head):
        return "PE executable"
    return None


def read_head(path):
    with open(path, "rb") as f:
        head = f.read(HEAD)
        if head.startswith(b"MZ") and len(head) >= 0x40:
            offset = int.from_bytes(head[0x3C:0x40], "little")
            if HEAD <= offset < 1 << 20:  # the PE signature sits past the first bytes read
                f.seek(offset)
                head += b"\0" * (offset - len(head)) + f.read(4)
        return head


def tracked_files(root):
    out = subprocess.run(["git", "ls-files", "-z"], cwd=root, check=True, capture_output=True).stdout
    return [os.path.join(root, p) for p in out.decode("utf-8", "surrogateescape").split("\0") if p]


def check(paths):
    """[(path, format)] for every regular file in `paths` that starts like game data."""
    found = []
    for path in paths:
        if os.path.islink(path) or not os.path.isfile(path):
            continue  # submodules, symlinks, files deleted in the working tree
        name = classify(read_head(path))
        if name:
            found.append((path, name))
    return found


def display(path):
    try:
        return os.path.relpath(path)
    except ValueError:  # Windows: another drive than the working directory
        return path


def main(argv):
    paths = argv or tracked_files(os.getcwd())
    found = check(paths)
    for path, name in found:
        print(f"{display(path)}: {name}")
    if found:
        print(f"{len(found)} file(s) start like game data. Remove them; see CONTRIBUTING.md.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    try:  # a redirected stdout is strict ANSI code page under -I; write '?' for what it cannot encode
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    sys.exit(main(sys.argv[1:]))
