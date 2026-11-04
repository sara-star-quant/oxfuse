import unittest

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import coverage_badge as c  # noqa: E402


class Badge(unittest.TestCase):
    def test_percent_reads_the_line_total(self):
        report = {"data": [{"totals": {"lines": {"percent": 87.456}, "functions": {"percent": 1.0}}}]}
        self.assertEqual(c.percent(report), 87.456)

    def test_message_has_one_decimal(self):
        self.assertEqual(c.badge(87.456)["message"], "87.5%")
        self.assertEqual(c.badge(100.0)["message"], "100.0%")

    def test_colour_steps_at_each_boundary(self):
        for p, want in [(90, "brightgreen"), (89.9, "green"), (80, "green"), (79.9, "yellowgreen"),
                        (70, "yellowgreen"), (69.9, "yellow"), (60, "yellow"), (59.9, "orange")]:
            self.assertEqual(c.colour(p), want, p)

    def test_schema_is_the_shields_endpoint_one(self):
        b = c.badge(50)
        self.assertEqual((b["schemaVersion"], b["label"]), (1, "coverage"))


if __name__ == "__main__":
    unittest.main()
