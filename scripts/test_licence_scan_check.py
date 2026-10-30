"""Tests for scripts/licence_scan_check.py (standard library only).

Run with `python3 -I scripts/test_licence_scan_check.py`."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import licence_scan_check as c  # noqa: E402


def file(path, expression="", holders=()):
    return {"type": "file", "path": f"oxfuse/{path}", "detected_license_expression_spdx": expression, "holders": [{"holder": h} for h in holders]}


class Scancode(unittest.TestCase):
    def test_our_licence_and_our_holder_pass(self):
        report = {"files": [file("LICENSE", "Apache-2.0", ["SARA STAR QUANT LLC"]), file("src/a.rs")]}
        self.assertEqual(c.scancode_findings(report), [])

    def test_the_holder_may_end_with_a_dot_or_omit_the_suffix(self):
        for h in ["SARA STAR QUANT LLC.", "SARA STAR QUANT"]:
            self.assertEqual(c.scancode_findings({"files": [file("NOTICE", "", [h])]}), [])

    def test_another_licence_in_an_ordinary_file_fails(self):
        report = {"files": [file("crates/core/src/x.rs", "MIT")]}
        self.assertEqual(c.scancode_findings(report), [("crates/core/src/x.rs", "licence MIT")])

    def test_files_that_only_talk_about_licences_may_name_them(self):
        report = {"files": [file("deny.toml", "MIT OR Apache-2.0"), file("docs/licensing.md", "GPL-3.0-only AND MIT")]}
        self.assertEqual(c.scancode_findings(report), [])

    def test_the_mention_list_is_per_file_not_per_directory(self):
        report = {"files": [file("docs/other.md", "GPL-3.0-only")]}
        self.assertEqual(len(c.scancode_findings(report)), 1)

    def test_a_foreign_copyright_holder_fails_even_in_a_mention_file(self):
        report = {"files": [file("docs/licensing.md", "Apache-2.0", ["Somebody Else"])]}
        self.assertEqual(c.scancode_findings(report), [("docs/licensing.md", "copyright holder Somebody Else")])

    def test_directories_are_ignored(self):
        self.assertEqual(c.scancode_findings({"files": [{"type": "directory", "path": "oxfuse/x", "detected_license_expression_spdx": "GPL-2.0-only"}]}), [])

    def test_relative_drops_the_scanned_directory_name(self):
        self.assertEqual(c.relative("oxfuse/a/b.rs"), "a/b.rs")
        self.assertEqual(c.relative("b.rs"), "b.rs")


class Snippets(unittest.TestCase):
    def test_a_judged_file_passes_and_a_new_match_fails(self):
        report = {
            "crates/core/src/types.rs": [{"id": "snippet", "matched": "16%", "component": "flow", "licenses": [{"name": "MIT"}]}],
            "crates/new.rs": [{"id": "snippet", "matched": "80%", "component": "thing", "licenses": [{"name": "GPL-3.0"}]}],
            "crates/clean.rs": [{"id": "none"}],
        }
        self.assertEqual(c.snippet_findings(report), [("crates/new.rs", "80%", "thing", ["GPL-3.0"])])

    def test_every_judged_file_has_a_reason(self):
        self.assertTrue(all(len(reason) > 10 for reason in c.ASSESSED.values()))
        self.assertTrue(all(len(reason) > 10 for reason in c.MENTIONS.values()))
        for path in list(c.ASSESSED) + list(c.MENTIONS):
            self.assertTrue((c.__file__ and Path(c.__file__).resolve().parent.parent / path).exists(), path)


if __name__ == "__main__":
    unittest.main()
