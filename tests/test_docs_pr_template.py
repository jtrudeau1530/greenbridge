"""The pull request template prompts for what AGENTS.md asks a pull request to say.

    python -I tests/test_docs_pr_template.py

AGENTS.md asks for the claimed item, a `Depends on: #N` line for every pull request the new one builds on (the
coordination issue builds its merge chains from those lines), what was tested and on which GPU and driver, and
whether the work was AI-assisted. A rule with no prompt in `.github/pull_request_template.md` is the one authors
leave out. This reads the two tracked files only; no game files.
"""
import os
import re
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATE = os.path.join(".github", "pull_request_template.md")

# (what AGENTS.md says, what the template has to contain). The left side is checked too, so a rule that is
# reworded or dropped in AGENTS.md fails here instead of leaving a prompt nobody asks for.
RULES = (
    (r"claimed in the coordination issue", r"coordination issue"),
    (r"`Depends on: #N`", r"^Depends on:"),
    (r"on which GPU and driver", r"GPU and driver"),
    (r"say that it was AI-assisted", r"^AI-assisted:"),
)


def read(rel):
    with open(os.path.join(ROOT, rel), encoding="utf-8") as f:
        return f.read()


def unprompted(agents, template):
    """The rules of RULES that `agents` states and `template` has no prompt for."""
    return [told for told, prompt in RULES
            if re.search(told, agents, re.I) and not re.search(prompt, template, re.M)]


class PullRequestTemplateTest(unittest.TestCase):
    def test_agents_md_still_states_each_rule(self):
        agents = read("AGENTS.md")
        for told, _ in RULES:
            with self.subTest(rule=told):
                self.assertRegex(agents, re.compile(told, re.I))

    def test_template_prompts_for_each_rule(self):
        self.assertEqual(unprompted(read("AGENTS.md"), read(TEMPLATE)), [])

    def test_a_template_without_a_prompt_is_reported(self):
        agents = "Put `Depends on: #N` in it and say that it was AI-assisted, and on which GPU and driver."
        self.assertEqual(unprompted(agents, "Tested: GPU and driver\n"),
                         ["`Depends on: #N`", "say that it was AI-assisted"])
        # A mention inside another sentence is not a prompt: the line has to start with it.
        self.assertEqual(unprompted(agents, "GPU and driver; no Depends on: here\nAI-assisted: no\n"),
                         ["`Depends on: #N`"])


if __name__ == "__main__":
    unittest.main()
