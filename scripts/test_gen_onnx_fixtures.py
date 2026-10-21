"""Run: .venv-onnx/bin/python -I scripts/test_gen_onnx_fixtures.py"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import gen_onnx_fixtures as g


class Oracles(unittest.TestCase):
    def fixture(self, name):
        return next(f for f in g.fixtures() if f.name == name)

    def test_oracles_disagree_fails_generation(self):
        fx = self.fixture("linear")

        def broken(model, feeds):
            return [out + 1.0 for out in g.run_reference(model, feeds)]

        with self.assertRaises(g.Disagreement):
            g.build(fx, ref_run=broken)
        with self.assertRaises(g.Disagreement):
            g.build(fx, ort_run=broken)

    def test_off_by_one_ulp_fails_an_exact_fixture(self):
        fx = self.fixture("linear")

        def nudged(model, feeds):
            return [np_next(out) for out in g.run_reference(model, feeds)]

        with self.assertRaises(g.Disagreement):
            g.build(fx, ort_run=nudged)

    def test_every_exact_fixture_agrees(self):
        for fx in g.fixtures():
            g.build(fx)


def np_next(a):
    import numpy as np
    return np.nextafter(np.asarray(a, dtype=np.float32), np.float32(np.inf))


if __name__ == "__main__":
    unittest.main()
