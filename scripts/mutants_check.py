#!/usr/bin/env python3
"""Compare the missed mutants of a cargo-mutants run with docs/known-survivors.txt.

A missed mutant that is not on the list fails the check. With --full, a listed
mutant that this run caught also fails, so the list only ever shrinks. Without
--full (a pull request run covers only the changed code) stale entries are
ignored. Line and column numbers are dropped from the names, so moving code
does not break the list.

Usage:
  python3 -I scripts/mutants_check.py [--full] [--only PREFIX]... [--known FILE] [MUTANTS_OUT_DIR]
--only keeps the mutants whose file starts with PREFIX, on both sides, for a run
that covers one package.
  python3 -I scripts/mutants_check.py --self-test
Stdlib only.
"""

import re
import sys
import unittest
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
KNOWN = ROOT / "docs" / "known-survivors.txt"


def normalize(line):
    """`path.rs:12:5: replace a with b in f` -> `path.rs: replace a with b in f`."""
    return re.sub(r"^([^:\s]+):\d+(?::\d+)?(?:-\d+:\d+)?: ", r"\1: ", line.strip())


def read_names(path, only=()):
    if not path.exists():
        return Counter()
    names = [normalize(l) for l in path.read_text().splitlines() if l.strip() and not l.startswith("#")]
    return Counter(n for n in names if not only or n.startswith(tuple(only)))


def compare(missed, known, full):
    """Returns (new, stale): names missed beyond the list, and listed names not missed."""
    new = missed - known
    stale = known - missed if full else Counter()
    return new, stale


def main(argv):
    full = "--full" in argv
    known_path = KNOWN
    args = [a for a in argv if a != "--full"]
    if "--known" in args:
        i = args.index("--known")
        known_path = Path(args[i + 1])
        del args[i : i + 2]
    only = []
    while "--only" in args:
        i = args.index("--only")
        only.append(args[i + 1])
        del args[i : i + 2]
    out = Path(args[0]) if args else ROOT / "mutants.out"
    missed = read_names(out / "missed.txt", only)
    new, stale = compare(missed, read_names(known_path, only), full)
    for name, count in sorted(new.items()):
        print(f"NEW missed mutant x{count}: {name}")
    for name, count in sorted(stale.items()):
        print(f"now caught, remove from {known_path.name} x{count}: {name}")
    print(f"{sum(missed.values())} missed, {sum(new.values())} new, {sum(stale.values())} stale")
    return 1 if new or stale else 0


class Tests(unittest.TestCase):
    def test_normalize_drops_positions(self):
        self.assertEqual(
            normalize("crates/a/src/b.rs:12:5: replace + with - in f"),
            "crates/a/src/b.rs: replace + with - in f",
        )
        self.assertEqual(
            normalize("crates/a/src/b.rs:12:5-12:9: replace + with - in f"),
            "crates/a/src/b.rs: replace + with - in f",
        )

    def test_a_new_miss_fails_and_a_listed_one_does_not(self):
        known = Counter({"a.rs: x": 1})
        self.assertEqual(compare(Counter({"a.rs: x": 1}), known, True), (Counter(), Counter()))
        new, _ = compare(Counter({"a.rs: x": 1, "b.rs: y": 1}), known, False)
        self.assertEqual(new, Counter({"b.rs: y": 1}))

    def test_a_second_miss_of_the_same_name_is_new(self):
        new, _ = compare(Counter({"a.rs: x": 2}), Counter({"a.rs: x": 1}), False)
        self.assertEqual(new, Counter({"a.rs: x": 1}))

    def test_only_filters_both_sides(self):
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "k.txt"
            f.write_text("crates/a/x.rs: m1\ncrates/b/y.rs: m2\n")
            self.assertEqual(read_names(f, ["crates/a/"]), Counter({"crates/a/x.rs: m1": 1}))
            self.assertEqual(len(read_names(f)), 2)

    def test_stale_entries_count_only_for_a_full_run(self):
        known = Counter({"a.rs: x": 1})
        self.assertEqual(compare(Counter(), known, False)[1], Counter())
        self.assertEqual(compare(Counter(), known, True)[1], Counter({"a.rs: x": 1}))


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        unittest.main(argv=[sys.argv[0]])
    sys.exit(main(sys.argv[1:]))
