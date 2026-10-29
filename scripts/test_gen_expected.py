"""Tests for the comparison of committed expected files (standard library only).

Run with `python3 -I scripts/test_gen_expected.py`."""
import sys
import types
import unittest
from pathlib import Path

# gen_expected imports NumPy at the top; the comparison needs none of it.
sys.modules.setdefault("numpy", types.ModuleType("numpy"))
sys.path.insert(0, str(Path(__file__).parent))
from gen_expected import same_doc  # noqa: E402

EXACT = {"abs": 0.0, "rel": 0.0}


def doc(values, bound=None, shape=None):
    out = {"name": "y", "shape": shape or [len(values)], "values": values}
    if bound is not None:
        out["bound"] = bound
    return {"tolerance": EXACT, "outputs": [out]}


class SameDoc(unittest.TestCase):
    def test_exact_outputs_must_match_bit_for_bit(self):
        self.assertTrue(same_doc(doc([1.0, 2.0]), doc([1.0, 2.0])))
        self.assertFalse(same_doc(doc([1.0, 2.0]), doc([1.0, 2.0000000000000004])))

    def test_a_bounded_output_may_differ_inside_its_bound(self):
        a, b = doc([1.0, 2.0], [0.01, 0.01]), doc([1.0000000000000002, 2.0], [0.01, 0.01])
        self.assertTrue(same_doc(a, b))

    def test_a_bounded_output_outside_the_bound_differs(self):
        self.assertFalse(same_doc(doc([1.0], [0.5]), doc([1.75], [0.5])))
        self.assertTrue(same_doc(doc([1.0], [0.5]), doc([1.5], [0.5])))

    def test_the_bound_may_differ_in_the_last_bits_only(self):
        self.assertTrue(same_doc(doc([1.0], [0.01]), doc([1.0], [0.01 * (1 + 1e-12)])))
        self.assertFalse(same_doc(doc([1.0], [0.01]), doc([1.0], [0.0101])))

    def test_a_missing_or_extra_bound_differs(self):
        self.assertFalse(same_doc(doc([1.0], [0.01]), doc([1.0])))
        self.assertFalse(same_doc(doc([1.0]), doc([1.0], [0.01])))

    def test_shape_count_and_names_must_match(self):
        self.assertFalse(same_doc(doc([1.0, 2.0], shape=[2]), doc([1.0, 2.0], shape=[1, 2])))
        self.assertFalse(same_doc(doc([1.0]), doc([1.0, 2.0])))
        self.assertFalse(same_doc(doc([1.0], [0.01]), doc([1.0, 2.0], [0.01, 0.01])))
        other = doc([1.0])
        other["outputs"][0]["name"] = "z"
        self.assertFalse(same_doc(doc([1.0]), other))


if __name__ == "__main__":
    unittest.main()
