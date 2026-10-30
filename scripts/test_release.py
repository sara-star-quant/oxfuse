"""Tests for scripts/release.py (standard library only).

Run with `python3 -I scripts/test_release.py`."""
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import release as r  # noqa: E402

CARGO = """[workspace]
members = ["crates/core"]

[workspace.package]
version = "0.5.1"
edition = "2024"

[workspace.dependencies]
oxfuse-core = { path = "crates/core", version = "0.5.1" }
oxfuse-runtime = { path = "crates/runtime", version = "0.5.1" }
clap = { version = "4.6.7", features = ["derive"] }
oxfuse-wgpu = { path = "crates/wgpu", version = "0.5.1" }
oxfuse-onnx = { path = "crates/onnx", version = "0.5.1" }
"""

LOCK = """[[package]]
name = "oxfuse"
version = "0.5.1"
dependencies = []

[[package]]
name = "serde"
version = "0.5.1"
source = "registry+https://github.com/rust-lang/crates.io-index"

[[package]]
name = "oxfuse-core"
version = "0.5.1"

[[package]]
name = "oxfuse-onnx"
version = "0.5.1"

[[package]]
name = "oxfuse-runtime"
version = "0.5.1"

[[package]]
name = "oxfuse-wgpu"
version = "0.5.1"
"""

LOG = """# Changelog

## Unreleased

- New thing.

## 0.5.1 - 2026-10-30

- Old thing.
"""


class Versions(unittest.TestCase):
    def test_reads_the_workspace_version_and_the_four_path_dependencies(self):
        self.assertEqual(r.workspace_version(CARGO), "0.5.1")
        deps = r.path_dependency_versions(CARGO)
        self.assertEqual([c for c, _ in deps], ["oxfuse-core", "oxfuse-runtime", "oxfuse-wgpu", "oxfuse-onnx"])

    def test_set_version_changes_those_lines_and_nothing_else(self):
        out = r.set_version(CARGO, "0.6.0")
        self.assertEqual(r.workspace_version(out), "0.6.0")
        self.assertTrue(all(v == "0.6.0" for _, v in r.path_dependency_versions(out)))
        self.assertIn('clap = { version = "4.6.7"', out)
        self.assertIn('edition = "2024"', out)
        self.assertEqual(out.count("0.6.0"), 5)
        self.assertEqual(r.set_version(out, "0.6.0"), out)

    def test_lock_versions_only_counts_workspace_crates_without_a_source(self):
        self.assertEqual(sorted(r.lock_versions(LOCK)), r.WORKSPACE_CRATES)
        self.assertNotIn("serde", r.lock_versions(LOCK))


class Changelog(unittest.TestCase):
    def test_roll_moves_unreleased_under_a_dated_heading(self):
        out = r.roll_changelog(LOG, "0.5.2", "2026-11-02")
        self.assertIn("## Unreleased\n\n## 0.5.2 - 2026-11-02\n\n- New thing.", out)
        self.assertIn("## 0.5.1 - 2026-10-30", out)

    def test_roll_refuses_an_empty_or_missing_unreleased_and_a_repeat(self):
        with self.assertRaises(SystemExit):
            r.roll_changelog("## Unreleased\n\n## 0.5.1 - 2026-10-30\n\n- x\n", "0.5.2", "2026-11-02")
        with self.assertRaises(SystemExit):
            r.roll_changelog("# Changelog\n", "0.5.2", "2026-11-02")
        with self.assertRaises(SystemExit):
            r.roll_changelog(LOG, "0.5.1", "2026-11-02")

    def test_release_check_wants_the_version_right_under_unreleased(self):
        rolled = r.roll_changelog(LOG, "0.5.2", "2026-11-02")
        self.assertEqual(r.changelog_problems(rolled, "0.5.2", True), [])
        self.assertEqual(r.changelog_problems(LOG, "0.5.2", False), [])
        self.assertEqual(len(r.changelog_problems(LOG, "0.5.2", True)), 1)
        self.assertEqual(len(r.changelog_problems("# Changelog\n", "0.5.2", False)), 1)


class Comparison(unittest.TestCase):
    ROW = "| Oxfuse 0.5.1 | Latest release | 0.5.1, 2026-10-30. | <https://crates.io/api/v1/crates/oxfuse/0.5.1> | 2026-10-30 |"

    def test_the_row_follows_the_version_and_date(self):
        out = r.set_comparison_row("a\n" + self.ROW + "\nb\n", "0.5.2", "2026-11-02")
        self.assertIn("| Oxfuse 0.5.2 | Latest release | 0.5.2, 2026-11-02. | <https://crates.io/api/v1/crates/oxfuse/0.5.2> | 2026-11-02 |", out)
        self.assertTrue(out.startswith("a\n") and out.endswith("\nb\n"))

    def test_a_missing_row_is_an_error(self):
        with self.assertRaises(SystemExit):
            r.set_comparison_row("nothing", "0.5.2", "2026-11-02")


class Pins(unittest.TestCase):
    def wf(self, tool, sha="a" * 40, ver="v1.0"):
        return f"      - uses: actions/checkout@{sha} # {ver}\n      - uses: x/y@{'b' * 40} # v2\n        with:\n          tool: {tool}\n"

    def test_one_version_everywhere_is_fine(self):
        texts = {"a.yml": self.wf("just@1.59.0,cargo-fuzz@0.13.2"), "b.yml": self.wf("just@1.59.0")}
        self.assertEqual(r.pin_problems(texts), [])

    def test_a_tool_with_two_versions_is_reported_with_both_files(self):
        texts = {"a.yml": self.wf("just@1.59.0"), "b.yml": self.wf("just@1.60.0")}
        problems = r.pin_problems(texts)
        self.assertEqual(len(problems), 1)
        self.assertIn("`just`", problems[0])
        self.assertIn("a.yml", problems[0])
        self.assertIn("b.yml", problems[0])

    def test_an_action_pinned_to_two_commits_is_reported(self):
        texts = {"a.yml": self.wf("just@1.59.0"), "b.yml": self.wf("just@1.59.0", sha="c" * 40, ver="v1.1")}
        problems = r.pin_problems(texts)
        self.assertEqual(len(problems), 1)
        self.assertIn("actions/checkout", problems[0])

    def test_kani_versions_are_compared(self):
        texts = {"a.yml": 'kani-version: "0.68.0"\n', "b.yml": 'kani-version: "0.69.0"\n'}
        self.assertEqual(len(r.pin_problems(texts)), 1)


class Whole(unittest.TestCase):
    def tree(self, cargo=CARGO, lock=LOCK, log=LOG):
        d = Path(tempfile.mkdtemp())
        (d / "Cargo.toml").write_text(cargo)
        (d / "Cargo.lock").write_text(lock)
        (d / "CHANGELOG.md").write_text(log)
        (d / ".github" / "workflows").mkdir(parents=True)
        return d

    def test_a_consistent_tree_passes(self):
        version, problems = r.check(self.tree(), False)
        self.assertEqual((version, problems), ("0.5.1", []))

    def test_a_drifted_path_dependency_and_lock_are_reported(self):
        cargo = CARGO.replace('crates/runtime", version = "0.5.1"', 'crates/runtime", version = "0.5.0"')
        lock = LOCK.replace('name = "oxfuse-core"\nversion = "0.5.1"', 'name = "oxfuse-core"\nversion = "0.5.0"')
        _, problems = r.check(self.tree(cargo=cargo, lock=lock), False)
        text = "\n".join(problems)
        self.assertIn("oxfuse-runtime path dependency says 0.5.0", text)
        self.assertIn("Cargo.lock has oxfuse-core 0.5.0", text)
        lock = LOCK.replace('name = "oxfuse-onnx"\nversion = "0.5.1"\n\n', "")
        _, problems = r.check(self.tree(lock=lock), False)
        self.assertIn("Cargo.lock has oxfuse-onnx None", "\n".join(problems))

    def test_a_missing_path_dependency_is_reported(self):
        cargo = CARGO.replace('oxfuse-wgpu = { path = "crates/wgpu", version = "0.5.1" }\n', "")
        _, problems = r.check(self.tree(cargo=cargo), False)
        self.assertIn("expected the four path dependencies of the root Cargo.toml", problems)


if __name__ == "__main__":
    unittest.main()
