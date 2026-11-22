"""Read shader names from a completed carve or unpack index."""

import json
import ntpath


def shader_stems(path: str, field: str, suffix: str = "") -> set[str]:
    with open(path, encoding="utf-8") as f:
        rows = json.load(f)
    if not isinstance(rows, list):
        raise ValueError(f"{path}: expected an array of shader entries")
    stems = set()
    for row in rows:
        value = row.get(field) if isinstance(row, dict) else None
        if not isinstance(value, str) or not value or value in (".", "..") or ntpath.basename(value) != value:
            raise ValueError(f"{path}: expected a shader basename in {field!r}")
        if suffix and (not value.endswith(suffix) or len(value) == len(suffix)):
            raise ValueError(f"{path}: expected a {suffix} filename in {field!r}")
        stems.add(value.removesuffix(suffix))
    return stems
