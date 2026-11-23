"""Generate and validate the translator contract table in docs/design.md from docs/contracts/*.json.

    python -I tools/build_design_doc.py           # regenerate docs/design.md from docs/contracts/*.json
    python -I tools/build_design_doc.py --check   # verify schema, consistency, and freshness (CI)

Why one file per topic? In round-42 alone, 74 pull requests edited the 376 KB contract table in
docs/design.md (15 adding new rows, 51 editing existing ones), producing merge conflicts and
spliced/duplicated rows every round. Storing each contract topic in its own JSON file under
docs/contracts/ eliminates shared-file add/add conflicts while keeping docs/design.md rendered
in place for readers. Standard library only; no game data.
"""
import argparse
import hashlib
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DESIGN_REL = os.path.join("docs", "design.md")
CONTRACTS_REL = os.path.join("docs", "contracts")
VALID_STATUSES = ("gpu_verified", "cpu_only", "spec_only")
TABLE_HEADER = "| Piece | Contract |"


def parse_design_table(text):
    """Split docs/design.md into (prefix_lines, [(piece, contract)], suffix_lines)."""
    lines = text.splitlines()
    start = None
    for idx, line in enumerate(lines):
        if line.strip() == TABLE_HEADER:
            start = idx
            break
    if start is None or start + 1 >= len(lines) or not lines[start + 1].startswith("|"):
        raise ValueError(f"{DESIGN_REL}: missing {TABLE_HEADER!r} table header")
    end = start + 2
    while end < len(lines) and lines[end].startswith("|"):
        end += 1
    rows = []
    for lineno, line in enumerate(lines[start + 2 : end], start + 3):
        m = re.match(r"^\|([^|]+)\|(.*)\|$", line)
        if not m:
            raise ValueError(f"{DESIGN_REL}:{lineno}: malformed contract table row")
        rows.append((m.group(1).strip(), m.group(2).strip()))
    return lines[: start + 2], rows, lines[end:]


def slug_for(piece, used_slugs):
    base = re.sub(r"[^a-z0-9]+", "_", piece.lower()).strip("_")[:56] or "topic"
    if base not in used_slugs:
        used_slugs.add(base)
        return base
    suffix = hashlib.sha1(piece.encode("utf-8")).hexdigest()[:6]
    candidate = f"{base}_{suffix}"
    used_slugs.add(candidate)
    return candidate


def load_contracts(contracts_dir):
    """Load and validate all docs/contracts/*.json files. Returns list of dicts sorted by (rank, piece)."""
    if not os.path.isdir(contracts_dir):
        raise ValueError(f"missing contracts directory: {contracts_dir}")
    entries = []
    seen_pieces = {}
    errors = []
    for name in sorted(os.listdir(contracts_dir)):
        if not name.endswith(".json"):
            continue
        path = os.path.join(contracts_dir, name)
        try:
            with open(path, encoding="utf-8") as f:
                obj = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"{name}: invalid JSON ({exc})")
            continue
        if not isinstance(obj, dict):
            errors.append(f"{name}: root must be a JSON object")
            continue
        piece = obj.get("piece")
        status = obj.get("status")
        sentences = obj.get("sentences")
        rank = obj.get("rank", 1000)
        if not isinstance(piece, str) or not piece.strip():
            errors.append(f"{name}: 'piece' must be a non-empty string")
            continue
        if "|" in piece:
            errors.append(f"{name}: 'piece' must not contain '|'")
        if status not in VALID_STATUSES:
            errors.append(f"{name}: 'status' must be one of {VALID_STATUSES}, got {status!r}")
        if (
            not isinstance(sentences, list)
            or not sentences
            or not all(isinstance(s, str) and s.strip() for s in sentences)
        ):
            errors.append(f"{name}: 'sentences' must be a non-empty list of non-empty strings")
            continue
        if not isinstance(rank, int) or isinstance(rank, bool):
            errors.append(f"{name}: 'rank' must be an integer")
            continue

        key = piece.strip()
        if key in seen_pieces:
            errors.append(f"{name}: duplicate piece {key!r} (already in {seen_pieces[key]})")
        seen_pieces[key] = name

        joined = " ".join(s.strip() for s in sentences)
        if status == "gpu_verified" and "**Not GPU-verified.**" in joined:
            errors.append(f"{name}: status is 'gpu_verified' but text still says '**Not GPU-verified.**'")
        if "||" in joined:
            for m in re.finditer(r"`[^`]+`", joined):
                if len(m.group(0)) > 160:
                    errors.append(f"{name}: suspicious spliced code span around '||' (len={len(m.group(0))})")
                    break

        entries.append({"file": name, "piece": key, "status": status, "rank": rank, "sentences": [s.strip() for s in sentences]})

    if errors:
        raise ValueError("contract validation failed:\n  " + "\n  ".join(errors))
    if not entries:
        raise ValueError(f"no contract *.json files found in {contracts_dir}")
    entries.sort(key=lambda e: (e["rank"], e["piece"].lower(), e["file"]))
    return entries


def render_design(design_text, entries):
    prefix, _, suffix = parse_design_table(design_text)
    rendered_rows = [f"| {e['piece']} | {' '.join(e['sentences'])} |" for e in entries]
    return "\n".join(prefix + rendered_rows + suffix) + "\n"


def validate_and_render(root):
    design_path = os.path.join(root, DESIGN_REL)
    contracts_dir = os.path.join(root, CONTRACTS_REL)
    with open(design_path, encoding="utf-8") as f:
        current = f.read()
    entries = load_contracts(contracts_dir)
    expected = render_design(current, entries)
    return current, expected, entries


def main(argv=None):
    ap = argparse.ArgumentParser(description="Build or check docs/design.md contract table from docs/contracts/*.json.")
    ap.add_argument("--root", default=ROOT, help="Repository root")
    ap.add_argument("--check", action="store_true", help="Exit non-zero if docs/design.md differs from docs/contracts/*.json")
    args = ap.parse_args(argv)

    root = os.path.abspath(args.root)
    try:
        current, expected, entries = validate_and_render(root)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    if args.check:
        if current != expected:
            print(
                "docs/design.md contract table is out of date with docs/contracts/*.json.\n"
                "Run: python -I tools/build_design_doc.py",
                file=sys.stderr,
            )
            return 1
        print(f"build_design_doc: OK ({len(entries)} contracts in sync)")
        return 0

    design_path = os.path.join(root, DESIGN_REL)
    with open(design_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(expected)
    print(f"build_design_doc: wrote {len(entries)} contract rows to {DESIGN_REL}")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(errors="replace")
    sys.exit(main())
