"""Each table in docs/design.md names a topic once.

    python -I tests/test_docs_design_rows.py

The translator contract table is where every new rule is recorded (CONTRIBUTING.md, AGENTS.md). When two branches
edit the same row and a merge keeps both sides, the table ends up with two rows for one topic, each holding facts the
other lacks, and a reader who stops at the first one misses the rest. On 2026-11-22 twenty topics appeared two to six
times. This test fails on a repeated first cell within one table. No game files or build tools.
"""
import os
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DESIGN = os.path.join(ROOT, "docs", "design.md")


def repeated_topics(lines):
    """[(topic, [1-based line numbers])] for first cells that appear more than once in one table."""
    found = []
    table, in_fence = [], False

    def close():
        seen = {}
        for n in table[2:]:                         # skip the header and the |---| rule
            cells = lines[n - 1].split("|")
            if len(cells) > 2 and cells[1].strip():
                seen.setdefault(cells[1].strip(), []).append(n)
        found.extend((k, v) for k, v in seen.items() if len(v) > 1)

    for n, line in enumerate(lines, 1):
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
        if not in_fence and line.startswith("|"):
            table.append(n)
            continue
        if table:
            close()
            table = []
    if table:
        close()
    return found


class DesignTableRows(unittest.TestCase):
    def test_checker_finds_a_repeated_topic_within_one_table_only(self):
        lines = ["| Topic | Rule |", "|---|---|", "| EXEC | old |", "| LDS | x |", "| EXEC | new |", "",
                 "| Topic | Rule |", "|---|---|", "| LDS | other table |", "```", "| EXEC | in a code block |",
                 "| EXEC | in a code block |", "```"]
        self.assertEqual(repeated_topics(lines), [("EXEC", [3, 5])])

    def test_design_md_names_each_topic_once_per_table(self):
        with open(DESIGN, encoding="utf-8") as f:
            lines = f.read().split("\n")
        repeated = repeated_topics(lines)
        self.assertEqual(repeated, [], "rows to merge into one (topic, lines): " +
                         "; ".join(f"{k[:60]!r} at {v}" for k, v in repeated))


if __name__ == "__main__":
    unittest.main()
