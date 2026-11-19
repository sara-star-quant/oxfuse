"""Tests for scripts/pre_merge.py (standard library only).

Run with `python3 -I scripts/test_pre_merge.py`."""
import contextlib
import io
import sys
import unittest
from pathlib import Path
from unittest import mock

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
        for bad in (["x"], ["x", "abc"], ["x", "1", "--accept"], ["x", "1", "--sha"], ["x", "1", "oops"]):
            self.assertEqual(p.main(bad), 2)

    def test_flags_are_parsed_in_any_order(self):
        self.assertEqual(p.parse(["12"]), ("12", False, None, []))
        self.assertEqual(
            p.parse(["12", "--accept", "a", "--wait", "--sha", "abc", "--accept", "b"]),
            ("12", True, "abc", ["a", "b"]),
        )


class Perf(unittest.TestCase):
    def test_only_files_under_a_performance_path_count(self):
        files = [
            "crates/runtime/src/cpu/matmul.rs",
            "crates/runtime/src/math.rs",
            "crates/runtime/src/analyze.rs",
            "crates/core/src/shape.rs",
            "docs/benchmarks.md",
            "crates/wgpu/src/shaders/min.wgsl",
        ]
        self.assertEqual(
            p.perf_files(files),
            ["crates/runtime/src/cpu/matmul.rs", "crates/runtime/src/math.rs", "crates/wgpu/src/shaders/min.wgsl"],
        )

    def test_every_kernel_file_the_perf_notes_name_is_covered(self):
        for path in (
            "crates/runtime/src/cpu/vecmath.rs",
            "crates/runtime/src/kernels.rs",
            "crates/runtime/src/math.rs",
            "crates/runtime/src/plan.rs",
            "crates/runtime/benches/kernels.rs",
            "crates/wgpu/src/shaders.rs",
            "crates/wgpu/src/shaders/sub.wgsl",
        ):
            self.assertEqual(p.perf_files([path]), [path], path)

    def test_a_similar_name_outside_the_listed_paths_does_not_count(self):
        self.assertEqual(p.perf_files(["crates/runtime/src/cpu_notes.md", "crates/runtime/src/mathx.rs"]), [])


class Wait(unittest.TestCase):
    def fetcher(self, rounds):
        it = iter(rounds)
        return lambda: next(it)

    def test_returns_at_once_when_nothing_is_pending(self):
        slept = []
        got = p.wait_for(self.fetcher([[check("t", "pass")]]), sleep=slept.append)
        self.assertEqual(got, [check("t", "pass")])
        self.assertEqual(slept, [])

    def test_polls_until_the_pending_check_finishes(self):
        slept = []
        rounds = [[check("t", "pending")], [check("t", "pending")], [check("t", "fail")]]
        got = p.wait_for(self.fetcher(rounds), sleep=slept.append, poll=7)
        self.assertEqual(got, [check("t", "fail")])
        self.assertEqual(slept, [7, 7])

    def test_gives_up_at_the_limit_and_returns_what_it_has(self):
        clock = iter([0, 5, 11, 20])
        got = p.wait_for(
            lambda: [check("t", "pending")], sleep=lambda s: None, now=lambda: next(clock), limit=10
        )
        self.assertEqual(got, [check("t", "pending")])


class Main(unittest.TestCase):
    """The whole command with the `gh` calls replaced."""

    def run_main(self, args, head="aaa", files=("docs/x.md",), checks=(GATE,)):
        with mock.patch.object(p, "fetch_checks", lambda pr: list(checks)), mock.patch.object(
            p, "fetch_head", lambda pr: head
        ), mock.patch.object(p, "fetch_files", lambda pr: list(files)), contextlib.redirect_stdout(io.StringIO()) as out:
            code = p.main(["x", "7", *args])
        return code, out.getvalue()

    def test_a_clean_pull_request_passes(self):
        self.assertEqual(self.run_main([])[0], 0)

    def test_the_wrong_head_blocks(self):
        code, out = self.run_main(["--sha", "bbb"])
        self.assertEqual(code, 1)
        self.assertIn("BLOCKED head", out)
        self.assertEqual(self.run_main(["--sha", "aaa"])[0], 0)

    def test_a_kernel_change_blocks_until_the_ab_result_is_accepted(self):
        files = ["crates/runtime/src/cpu/matmul.rs"]
        code, out = self.run_main([], files=files)
        self.assertEqual(code, 1)
        self.assertIn("BLOCKED perf-ab", out)
        self.assertEqual(self.run_main(["--accept", "perf-ab"], files=files)[0], 0)

    def test_a_red_check_still_blocks_with_the_other_options_given(self):
        checks = [GATE, check("diff (0)", "fail", "mutants")]
        self.assertEqual(self.run_main(["--sha", "aaa", "--accept", "perf-ab"], checks=checks)[0], 1)


if __name__ == "__main__":
    unittest.main()
