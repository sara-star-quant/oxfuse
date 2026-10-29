"""Weekly benchmark trend: collect criterion medians, compare with the last run.

Usage:
  python3 -I scripts/bench_trend.py collect OUT.json
  python3 -I scripts/bench_trend.py compare NEW.json [OLD.json]

Hosted runners are noisy (10 to 30 percent), so nothing here fails a build; a
change of FLAG percent or more is marked so a person can look.
"""

import json
import sys
from pathlib import Path

CRITERION = Path("target/criterion")
FLAG = 30.0


def collect(criterion=CRITERION):
    medians = {}
    for bench in sorted(criterion.rglob("new/benchmark.json")):
        meta = json.loads(bench.read_text())
        estimates = json.loads(bench.with_name("estimates.json").read_text())
        medians[meta["full_id"]] = estimates["median"]["point_estimate"]
    return medians


def fmt(ns):
    for unit, scale in (("s", 1e9), ("ms", 1e6), ("us", 1e3)):
        if ns >= scale:
            return f"{ns / scale:.2f} {unit}"
    return f"{ns:.0f} ns"


def compare(new, old=None):
    lines = ["| benchmark | now | last run | change |", "|---|---|---|---|"]
    for name in sorted(new):
        before = (old or {}).get(name)
        if not before:
            lines.append(f"| {name} | {fmt(new[name])} | - | - |")
            continue
        change = (new[name] - before) / before * 100
        mark = " !" if abs(change) >= FLAG else ""
        lines.append(f"| {name} | {fmt(new[name])} | {fmt(before)} | {change:+.1f}%{mark} |")
    for name in sorted(set(old or {}) - set(new)):
        lines.append(f"| {name} | - | {fmt(old[name])} | gone |")
    return "\n".join(lines)


def main(argv):
    if len(argv) == 3 and argv[1] == "collect":
        medians = collect()
        if not medians:
            sys.exit("no results under target/criterion")
        Path(argv[2]).write_text(json.dumps(medians, indent=1, sort_keys=True) + "\n")
    elif len(argv) in (3, 4) and argv[1] == "compare":
        new = json.loads(Path(argv[2]).read_text())
        old = json.loads(Path(argv[3]).read_text()) if len(argv) == 4 else None
        print(compare(new, old))
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main(sys.argv)
