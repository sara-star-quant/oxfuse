"""Tests for scripts/dco_check.py (standard library only).

Run with `python3 -I scripts/test_dco_check.py`."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import dco_check as d  # noqa: E402

PATTERNS = ["maint@example.com", "*+bot[[]bot]@users.noreply.github.com"]


def log(*commits):
    return "".join(f"{sha}{d.SEP}{parents}{d.SEP}{email}{d.SEP}{body}{d.END}\n" for sha, parents, email, body in commits)


class Dco(unittest.TestCase):
    def test_an_outside_commit_needs_its_authors_signoff(self):
        commits = d.parse(log(("a" * 40, "p", "dev@example.org", "add x\n\nSigned-off-by: Dev <dev@example.org>\n")))
        self.assertEqual(d.missing(commits, PATTERNS), [])
        commits = d.parse(log(("b" * 40, "p", "dev@example.org", "add x\n")))
        self.assertEqual([c["sha"] for c in d.missing(commits, PATTERNS)], ["b" * 40])

    def test_a_signoff_by_someone_else_does_not_count(self):
        commits = d.parse(log(("c" * 40, "p", "dev@example.org", "add x\n\nSigned-off-by: Other <other@example.org>\n")))
        self.assertEqual(len(d.missing(commits, PATTERNS)), 1)

    def test_the_address_match_ignores_case(self):
        commits = d.parse(log(("d" * 40, "p", "Dev@Example.org", "x\n\nsigned-off-by: Dev <dev@example.org>\n")))
        self.assertEqual(d.missing(commits, PATTERNS), [])

    def test_exempt_authors_and_bots_need_nothing(self):
        commits = d.parse(log(
            ("e" * 40, "p", "maint@example.com", "x\n"),
            ("f" * 40, "p", "49699333+bot[bot]@users.noreply.github.com", "x\n"),
        ))
        self.assertEqual(d.missing(commits, PATTERNS), [])

    def test_merge_commits_are_skipped(self):
        commits = d.parse(log(("1" * 40, "p1 p2", "dev@example.org", "Merge branch\n")))
        self.assertEqual(d.missing(commits, PATTERNS), [])

    def test_the_exempt_file_parses_and_comments_are_dropped(self):
        self.assertEqual(d.patterns_from("# c\n\na@b.c\n  x*@y.z \n"), ["a@b.c", "x*@y.z"])

    def test_the_shipped_patterns_exempt_the_maintainer_and_dependabot_only(self):
        patterns = d.patterns_from((d.ROOT / ".github" / "dco-exempt.txt").read_text())
        self.assertTrue(d.exempt("pzverkov@users.noreply.github.com", patterns))
        self.assertTrue(d.exempt("10436770+pzverkov@users.noreply.github.com", patterns))
        self.assertTrue(d.exempt("49699333+dependabot[bot]@users.noreply.github.com", patterns))
        self.assertFalse(d.exempt("someone@users.noreply.github.com", patterns))
        self.assertFalse(d.exempt("evil+dependabot@users.noreply.github.com", patterns))


if __name__ == "__main__":
    unittest.main()
