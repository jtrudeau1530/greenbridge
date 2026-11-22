"""docs/i18n/README.md: one row per language, each pointing at a quickstart that exists.

    python -I tests/test_i18n_index.py

Reads only that one file; no game data. A synthetic table with a duplicate language code must be
reported, so the check cannot pass by parsing nothing.
"""
import os
import re
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INDEX = os.path.join(ROOT, "docs", "i18n", "README.md")
DOCS_README = os.path.join(ROOT, "docs", "README.md")

ROW = re.compile(r"^\|\s*(?P<language>[^|]+?)\s*\|\s*(?P<file>[^|]+?)\s*\|\s*"
                 r"(?P<commit>[^|]+?)\s*\|\s*(?P<maintainer>[^|]+?)\s*\|$")
LINK = re.compile(r"\[[^\]]*\]\(([^)]+)\)")
CODE = re.compile(r"`([A-Za-z][A-Za-z0-9-]*)`")
COMMIT = re.compile(r"^[0-9a-f]{7,40}$")


def parse(text):
    """Rows of the language table. Raises ValueError for a row the table format does not parse."""
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("|"):
            continue
        if set(line.replace("|", "").replace(" ", "")) <= set("-:"):
            continue                       # the header separator
        match = ROW.match(line)
        if not match:
            raise ValueError(f"malformed table row: {line}")
        if match.group("language").strip() == "Language":
            continue                       # the header
        link = LINK.search(match.group("file"))
        code = CODE.search(match.group("language"))
        if not link or not code:
            raise ValueError(f"row without a file link or a language code: {line}")
        rows.append({"code": code.group(1), "file": link.group(1),
                     "commit": match.group("commit").strip().strip("`"),
                     "maintainer": match.group("maintainer").strip().strip("`")})
    return rows


def duplicate_codes(rows):
    seen = set()
    duplicates = []
    for row in rows:
        if row["code"] in seen:
            duplicates.append(row["code"])
        seen.add(row["code"])
    return duplicates


class IndexTest(unittest.TestCase):
    def read_rows(self):
        with open(INDEX, encoding="utf-8") as f:
            return parse(f.read())

    def test_one_row_per_language(self):
        rows = self.read_rows()
        self.assertGreaterEqual(len(rows), 3)
        self.assertEqual(duplicate_codes(rows), [])

    def test_every_row_points_at_an_existing_quickstart(self):
        base = os.path.dirname(INDEX)
        for row in self.read_rows():
            with self.subTest(language=row["code"]):
                path = os.path.normpath(os.path.join(base, row["file"]))
                self.assertTrue(os.path.isfile(path), path)
                self.assertTrue(row["maintainer"].startswith("@"), row)
                self.assertRegex(row["commit"], COMMIT.pattern, row)

    def test_docs_readme_names_every_language_once(self):
        codes = [row["code"] for row in self.read_rows()]
        with open(DOCS_README, encoding="utf-8") as f:
            lines = [line for line in f.read().splitlines()
                     if line.startswith("Community quickstarts (stable onboarding text only")]
        self.assertEqual(len(lines), 1, f"docs/README.md has {len(lines)} 'Other languages' lines")
        line = lines[0]
        for code in codes:
            self.assertEqual(line.count(f"`{code}`"), 1, f"{code} is not named exactly once: {line}")
        listed = set(re.findall(r"`([A-Za-z]{2,3}(?:-[A-Za-z]+)?)`", line))
        self.assertEqual(listed - set(codes), set(),
                         f"docs/README.md names languages without an i18n row: {listed - set(codes)}")

    def test_a_duplicate_language_is_reported(self):
        table = ("| Language | File | Synced commit | Maintainer |\n"
                 "|---|---|---|---|\n"
                 "| 简体中文 (`zh-CN`) | [`zh-CN/quickstart.md`](zh-CN/quickstart.md) | `e7584c5` | `@a` |\n"
                 "| 简体中文 (`zh-CN`) | [`zh-CN/quickstart.md`](zh-CN/quickstart.md) | `0d61c357` | `@a` |\n")
        rows = parse(table)
        self.assertEqual(duplicate_codes(rows), ["zh-CN"])

    def test_a_malformed_row_is_rejected(self):
        table = ("| Language | File | Synced commit | Maintainer |\n"
                 "|---|---|---|---|\n"
                 "| 简体中文 | `zh-CN/quickstart.md` | `e7584c5` | `@a` |\n")
        with self.assertRaises(ValueError):
            parse(table)


if __name__ == "__main__":
    unittest.main()
