"""Run: python3 -I scripts/test_bench_trend.py"""

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import bench_trend as t


class Trend(unittest.TestCase):
    def test_collect_reads_the_median_of_every_benchmark(self):
        with tempfile.TemporaryDirectory() as d:
            for full_id, median in (("mlp/cpu/1", 5000.0), ("mlp/ndarray/8", 2.5e6)):
                new = Path(d, *full_id.split("/"), "new")
                new.mkdir(parents=True)
                (new / "benchmark.json").write_text(json.dumps({"full_id": full_id}))
                (new / "estimates.json").write_text(
                    json.dumps({"median": {"point_estimate": median}, "mean": {"point_estimate": 1.0}})
                )
            self.assertEqual(t.collect(Path(d)), {"mlp/cpu/1": 5000.0, "mlp/ndarray/8": 2.5e6})

    def test_units(self):
        self.assertEqual(t.fmt(999), "999 ns")
        self.assertEqual(t.fmt(1000), "1.00 us")
        self.assertEqual(t.fmt(2.5e6), "2.50 ms")
        self.assertEqual(t.fmt(3e9), "3.00 s")

    def test_without_a_previous_run_there_is_no_change_column(self):
        out = t.compare({"a/b/1": 1000.0})
        self.assertIn("| a/b/1 | 1.00 us | - | - |", out)

    def test_change_is_signed_and_flagged_at_the_threshold(self):
        old = {"a": 100.0, "b": 100.0, "c": 100.0, "d": 100.0}
        new = {"a": 129.0, "b": 130.0, "c": 70.0, "d": 100.0}
        out = t.compare(new, old)
        self.assertIn("+29.0% |", out)
        self.assertIn("+30.0% !", out)
        self.assertIn("-30.0% !", out)
        self.assertIn("+0.0% |", out)

    def test_new_and_vanished_benchmarks_are_listed(self):
        out = t.compare({"new": 10.0}, {"gone": 20.0})
        self.assertIn("| new | 10 ns | - | - |", out)
        self.assertIn("| gone | - | 20 ns | gone |", out)

    def test_a_zero_previous_value_does_not_divide(self):
        self.assertIn("| a | 10 ns | - | - |", t.compare({"a": 10.0}, {"a": 0.0}))


if __name__ == "__main__":
    unittest.main()
