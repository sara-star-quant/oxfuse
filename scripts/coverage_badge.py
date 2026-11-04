"""Turn `cargo llvm-cov --json --summary-only` output into a shields.io endpoint file.

  python3 -I scripts/coverage_badge.py cov.json badge.json

Line coverage of the whole workspace, test code excluded by llvm-cov. Standard library only.
"""
import json
import sys


def percent(report):
    return report["data"][0]["totals"]["lines"]["percent"]


def colour(p):
    return "brightgreen" if p >= 90 else "green" if p >= 80 else "yellowgreen" if p >= 70 else "yellow" if p >= 60 else "orange"


def badge(p):
    return {"schemaVersion": 1, "label": "coverage", "message": f"{p:.1f}%", "color": colour(p)}


if __name__ == "__main__":
    with open(sys.argv[1]) as f:
        p = percent(json.load(f))
    with open(sys.argv[2], "w") as f:
        json.dump(badge(p), f)
    print(f"line coverage {p:.1f}%")
