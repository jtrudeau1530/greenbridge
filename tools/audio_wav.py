"""Decode Ogg Opus audio to 16-bit PCM WAV with ffmpeg.

    python -I audio_wav.py <src dir | .ogg/.opus file> [--out DIR] [--jobs N] [--ffmpeg PATH]

Every .ogg/.opus file under the source is decoded to <out>/<relative path>.wav (the name keeps its directory,
the extension becomes .wav). Existing .wav files are skipped, so a stopped run continues where it left off, and
.wav inputs are never decoded. With no --out, a source directory writes to <src>/wav and a single file writes
next to itself.

The output is pcm_s16le at the stream's own sample rate and channel count (Opus is always 48 kHz). It is trimmed
to the Ogg Opus end granule (minus the header's pre-skip) with atrim, because ffmpeg otherwise can keep up to one
extra partial frame when a stream's last page lies beyond the declared end. ffmpeg must be on PATH or given with
--ffmpeg; without it the tool exits with an explanation instead of a traceback.

For PCM copies of the AWC banks' Opus streams, first repackage the banks with awc_opus.py, then run this tool on
its output (make_game.py does both, the second only with --wav).
"""
import argparse
import concurrent.futures
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import time

EXTS = (".ogg", ".opus")


def error(msg: str) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)


def find_ffmpeg(explicit: str):
    """The ffmpeg executable to run, or None (with the reason already printed)."""
    if explicit:
        candidate = os.path.expanduser(explicit)
        if os.path.dirname(candidate):
            if os.path.isfile(candidate):
                return os.path.abspath(candidate)
        else:
            found = shutil.which(candidate)
            if found:
                return found
        error(f"ffmpeg not found: {explicit}. Give --ffmpeg the path to ffmpeg.exe, or install ffmpeg and put it "
              "on PATH: https://ffmpeg.org/download.html")
        return None
    found = shutil.which("ffmpeg")
    if not found:
        error("ffmpeg was not found on PATH (it decodes the Ogg Opus files to WAV). Install it from "
              "https://ffmpeg.org/download.html and reopen the terminal, or pass --ffmpeg PATH.")
    return found


def looks_like_opus(path: str) -> bool:
    """True if the first Ogg page carries an OpusHead packet (so .ogg files that are Vorbis are left alone)."""
    try:
        with open(path, "rb") as f:
            return b"OpusHead" in f.read(512)
    except OSError:
        return True         # let ffmpeg report the real problem


def ogg_output_samples(path: str) -> int:
    """Samples per channel the Ogg Opus stream declares (last page granule minus pre-skip), or 0 if unknown."""
    try:
        size = os.path.getsize(path)
        with open(path, "rb") as f:
            first = f.read(65536)
            head = first.find(b"OpusHead")
            if head < 0 or head + 12 > len(first):
                return 0
            pre_skip = struct.unpack_from("<H", first, head + 10)[0]
            start = max(0, size - 131072)
            f.seek(start)
            tail = f.read()
        # The last page ends at EOF; walk back over possible "OggS" bytes inside the payload until one fits.
        end = len(tail)
        while True:
            pos = tail.rfind(b"OggS", 0, end)
            if pos < 0:
                return 0
            end = pos
            if pos + 27 > len(tail) or tail[pos + 4] != 0:
                continue
            nseg = tail[pos + 26]
            if pos + 27 + nseg > len(tail):
                continue
            body = sum(tail[pos + 27:pos + 27 + nseg])
            if start + pos + 27 + nseg + body != size:
                continue
            granule, = struct.unpack_from("<q", tail, pos + 6)
            return granule - pre_skip if granule > pre_skip else 0
    except (OSError, struct.error):
        return 0


def collect(src: str):
    """([(input, relative output stem)], number of .ogg/.opus files that are not Ogg Opus)."""
    files, not_opus = [], 0
    for root, dirs, names in os.walk(src):
        dirs.sort()
        for name in sorted(names):
            if not name.lower().endswith(EXTS):
                continue
            path = os.path.join(root, name)
            rel = os.path.relpath(path, src)
            if looks_like_opus(path):
                files.append((path, os.path.splitext(rel)[0]))
            else:
                not_opus += 1
    return files, not_opus


def decode(ffmpeg: str, src: str, dst: str):
    """Decode under the destination directory and rename the finished WAV into place."""
    try:
        out_dir = os.path.dirname(dst)
        os.makedirs(out_dir, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=out_dir, ignore_cleanup_errors=True) as tmp_dir:
            tmp = os.path.join(tmp_dir, "decoded.wav")
            cmd = [ffmpeg, "-v", "error", "-y", "-i", src]
            samples = ogg_output_samples(src)
            if samples:
                cmd += ["-af", f"atrim=end_sample={samples}"]
            cmd += ["-c:a", "pcm_s16le", tmp]
            r = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
            if r.returncode != 0:
                return False, (r.stderr.strip() or f"ffmpeg exited with {r.returncode}")
            os.replace(tmp, dst)
        return True, ""
    except OSError as e:
        return False, str(e)


def main() -> int:
    ap = argparse.ArgumentParser(description="Decode Ogg Opus audio to 16-bit PCM WAV with ffmpeg.")
    ap.add_argument("src", help="a directory to search recursively, or one .ogg/.opus file")
    ap.add_argument("--out", help="output directory (default: <src>/wav for a directory, next to a single file)")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) // 2),
                    help="ffmpeg processes to run at once (default: half the CPU count)")
    ap.add_argument("--ffmpeg", help="ffmpeg executable (default: ffmpeg on PATH)")
    a = ap.parse_args()

    src = os.path.abspath(a.src)
    if os.path.isfile(src):
        if not src.lower().endswith(EXTS):
            error(f"{a.src} is not an .ogg/.opus file (already-WAV files are not decoded).")
            return 2
        if not looks_like_opus(src):
            error(f"{a.src} is not Ogg Opus.")
            return 2
        items = [(src, os.path.splitext(os.path.basename(src))[0])]
        not_opus = 0
        default_out = os.path.dirname(src)
    elif os.path.isdir(src):
        items, not_opus = collect(src)
        default_out = os.path.join(src, "wav")
    else:
        error(f"source not found: {a.src}")
        return 2

    out = os.path.abspath(a.out or default_out)
    print(f"{len(items)} Ogg Opus file(s) under {src} -> {out}")
    if not items:
        print(f"Nothing to do: no Ogg Opus files found"
              + (f" ({not_opus} .ogg/.opus file(s) are not Ogg Opus)" if not_opus else "") + ".")
        return 0

    ffmpeg = find_ffmpeg(a.ffmpeg)
    if not ffmpeg:
        return 2

    jobs = max(1, a.jobs)
    t0 = time.time()
    written = skipped = 0
    todo = []
    for path, rel in items:
        dst = os.path.join(out, rel + ".wav")
        if os.path.exists(dst):
            skipped += 1
        else:
            todo.append((path, rel, dst))

    failures = []
    done = 0
    workers = min(jobs, len(todo)) if todo else 1
    if todo:
        print(f"Decoding {len(todo)} file(s) with {workers} job(s)...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(decode, ffmpeg, path, dst): (path, rel)
                   for path, rel, dst in todo}
        for future in concurrent.futures.as_completed(futures):
            path, rel = futures[future]
            try:
                ok, message = future.result()
            except Exception as e:                  # never lose the summary to an unexpected worker error
                ok, message = False, f"{type(e).__name__}: {e}"
            done += 1
            if ok:
                written += 1
            else:
                failures.append((rel, message))
            if done % 200 == 0:
                print(f"  {done}/{len(todo)} decoded", flush=True)

    seconds = time.time() - t0
    summary = f"{len(items)} Ogg Opus file(s): {written} WAV written, {skipped} already present"
    if failures:
        summary += f", {len(failures)} failed"
    print(summary + f" ({seconds:.1f}s)")
    if not_opus:
        print(f"  {not_opus} .ogg/.opus file(s) skipped: not Ogg Opus")
    for rel, message in failures[:10]:
        first = next((line.strip() for line in message.splitlines() if line.strip()), "unknown error")
        print(f"  FAILED {rel}: {first}")
    if len(failures) > 10:
        print(f"  ... and {len(failures) - 10} more failures")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
