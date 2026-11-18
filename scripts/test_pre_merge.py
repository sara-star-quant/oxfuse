"""Tests for scripts/pre_merge.py (standard library only).

Run with `python3 -I scripts/test_pre_merge.py`."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import pre_merge as p  # noqa: E402


def check(name, bucket, workflow="ci"):
    return {"name": name, "bucket": bucket, "workflow": workflow}


GATE = check("required", "pass")


class Problems(unittest.TestCase):
    def test_all_green_and_skipped_pass(self):
        self.assertEqual(p.problems([GATE, check("test", "pass"), check("docs", "skipping")]), [])

    def test_a_red_check_outside_the_gate_blocks(self):
        got = p.problems([GATE, check("diff (1)", "fail", "mutants")])
        self.assertEqual(got, [("mutants/diff (1)", "fail")])

    def test_a_running_or_cancelled_check_blocks(self):
        got = p.problems([GATE, check("kani", "pending", "kani"), check("semver", "cancel", "semver")])
        self.assertEqual([x[0] for x in got], ["kani/kani", "semver/semver"])

    def test_a_check_can_be_accepted_by_name_or_by_workflow_and_name(self):
        red = check("diff (1)", "fail", "mutants")
        self.assertEqual(p.problems([GATE, red], ["diff (1)"]), [])
        self.assertEqual(p.problems([GATE, red], ["mutants/diff (1)"]), [])
        self.assertEqual(len(p.problems([GATE, red], ["diff (2)"])), 1)

    def test_the_gate_cannot_be_accepted_away(self):
        self.assertEqual(len(p.problems([check("required", "fail")], ["required"])), 1)

    def test_a_missing_gate_blocks(self):
        self.assertEqual(p.problems([check("test", "pass")]), [("required", "the gate has not reported")])

    def test_no_checks_at_all_blocks(self):
        self.assertEqual(len(p.problems([])), 1)

    def test_a_pending_gate_blocks(self):
        self.assertEqual(p.problems([check("required", "pending")]), [("ci/required", "the gate is pending")])


class Cli(unittest.TestCase):
    def test_arguments_are_checked(self):
        for bad in (["x"], ["x", "abc"], ["x", "1", "--accept"], ["x", "1", "oops"]):
            self.assertEqual(p.main(bad), 2)


if __name__ == "__main__":
    unittest.main()
