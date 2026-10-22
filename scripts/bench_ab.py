#!/usr/bin/env python3
"""Compare a benchmark filter between a base commit and the working tree.

Timings on a laptop move by 5 to 25 percent between runs, so a single before
and after run cannot support a claim. This builds the base commit in a git
worktree, then runs the same criterion filter on the base and on the working
tree alternately, several rounds, and reports per benchmark the median of the
per-run medians, the spread, and a verdict that is "faster" or "slower" only
when the difference is larger than both 3 percent and the spread of either side.

Usage:
  python3 -I scripts/bench_ab.py [--base REF] [--package P] [--rounds N] FILTER
  python3 -I scripts/bench_ab.py --self-test
Stdlib only. The base build lives in target/bench-ab-base (kept between runs). The base
runs the working tree's benchmark files, so a case added with the change is
measured on the old code as well.
"""

import json
import os
import shutil
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASE_TREE = ROOT / "target" / "bench-ab-tree"
BASE_TARGET = ROOT / "target" / "bench-ab-base"
MIN_EFFECT = 0.03
# The criterion bench target of each package; the library harness would reject --noplot.
BENCH_TARGETS = {"oxfuse-runtime": "kernels", "oxfuse-wgpu": "gpu"}
BENCH_DIRS = {"oxfuse-runtime": "crates/runtime/benches", "oxfuse-wgpu": "crates/wgpu/benches"}


def read_medians(criterion_dir):
    """{benchmark id: median ns per iteration} from criterion's `new` results."""
    out = {}
    for bench in Path(criterion_dir).glob("**/new/benchmark.json"):
        estimates = bench.parent / "estimates.json"
        if not estimates.exists():
            continue
        full_id = json.loads(bench.read_text())["full_id"]
        out[full_id] = json.loads(estimates.read_text())["median"]["point_estimate"]
    return out


def summarize(base_runs, head_runs):
    """Rows (id, base median, head median, delta, spread, verdict) from lists of
    per-run {id: ns} dicts. Spread is (max - min) / median of the noisier side."""
    rows = []
    for name in sorted(set().union(*base_runs) & set().union(*head_runs)):
        base = [r[name] for r in base_runs if name in r]
        head = [r[name] for r in head_runs if name in r]
        base_m, head_m = statistics.median(base), statistics.median(head)
        spread = max((max(v) - min(v)) / statistics.median(v) for v in (base, head))
        delta = (head_m - base_m) / base_m
        if abs(delta) <= max(MIN_EFFECT, spread):
            verdict = "noise"
        else:
            verdict = "faster" if delta < 0 else "slower"
        rows.append((name, base_m, head_m, delta, spread, verdict))
    return rows


def fmt_ns(ns):
    for unit, scale in (("s", 1e9), ("ms", 1e6), ("us", 1e3)):
        if ns >= scale:
            return f"{ns / scale:.2f} {unit}"
    return f"{ns:.0f} ns"


def render(rows, rounds):
    lines = [
        f"{'benchmark':44} {'base':>10} {'head':>10} {'change':>8} {'spread':>7}  verdict",
    ]
    for name, base, head, delta, spread, verdict in rows:
        lines.append(
            f"{name:44} {fmt_ns(base):>10} {fmt_ns(head):>10} {delta:>+8.1%} {spread:>7.1%}  {verdict}"
        )
    lines.append(f"({rounds} interleaved rounds; median of per-run medians)")
    return "\n".join(lines)


def run(cmd, cwd, env=None):
    subprocess.run(cmd, cwd=cwd, env={**os.environ, **(env or {})}, check=True)


def bench(tree, target, package, filter_, features):
    env = {"CARGO_TARGET_DIR": str(target)}
    # Criterion keeps results of earlier runs; only this run's benchmarks may count.
    shutil.rmtree(Path(target) / "criterion", ignore_errors=True)
    cmd = ["cargo", "bench", "-p", package, "--bench", BENCH_TARGETS[package], "--locked"]
    if features:
        cmd += ["--features", features]
    cmd += ["--", filter_, "--noplot"]
    run(cmd, tree, env)
    return read_medians(Path(target) / "criterion")


def main(argv):
    if "--self-test" in argv:
        return self_test()
    args = list(argv)
    opts = {"--base": "origin/main", "--package": "oxfuse-runtime", "--rounds": "5", "--features": ""}
    while args and args[0] in opts:
        opts[args[0]] = args[1]
        del args[0:2]
    if len(args) != 1:
        print(__doc__)
        return 2
    filter_ = args[0]
    if BASE_TREE.exists():
        run(["git", "worktree", "remove", "--force", str(BASE_TREE)], ROOT)
    run(["git", "worktree", "add", "--detach", str(BASE_TREE), opts["--base"]], ROOT)
    # Both sides run the benchmarks of the working tree, so a case added with the
    # change is measured on the base code too.
    bench_dir = BENCH_DIRS[opts["--package"]]
    shutil.rmtree(BASE_TREE / bench_dir)
    shutil.copytree(ROOT / bench_dir, BASE_TREE / bench_dir)
    head_target = ROOT / "target" / "bench-ab-head"
    rounds = int(opts["--rounds"])
    base_runs, head_runs = [], []
    try:
        for i in range(rounds):
            print(f"round {i + 1}/{rounds}", flush=True)
            # Alternate which side goes first so drift does not favour one.
            order = ("base", "head") if i % 2 == 0 else ("head", "base")
            for side in order:
                tree, target = (BASE_TREE, BASE_TARGET) if side == "base" else (ROOT, head_target)
                runs = base_runs if side == "base" else head_runs
                runs.append(bench(tree, target, opts["--package"], filter_, opts["--features"]))
    finally:
        run(["git", "worktree", "remove", "--force", str(BASE_TREE)], ROOT)
    print(render(summarize(base_runs, head_runs), rounds))
    return 0


def self_test():
    quiet = [{"a": 100.0}, {"a": 101.0}, {"a": 99.0}]
    slower = [{"a": 130.0}, {"a": 131.0}, {"a": 129.0}]
    rows = summarize(quiet, slower)
    assert rows[0][5] == "slower" and abs(rows[0][3] - 0.30) < 1e-9, rows
    assert summarize(slower, quiet)[0][5] == "faster"
    assert summarize(quiet, [{"a": 102.0}, {"a": 101.0}, {"a": 103.0}])[0][5] == "noise"
    # A 10 percent change inside a 25 percent spread is noise.
    noisy = [{"a": 80.0}, {"a": 100.0}, {"a": 105.0}]
    shifted = [{"a": 90.0}, {"a": 110.0}, {"a": 115.0}]
    assert summarize(noisy, shifted)[0][5] == "noise"
    # Only benchmarks present on both sides are compared.
    only = summarize([{"a": 1.0, "b": 2.0}], [{"a": 1.0}])
    assert [r[0] for r in only] == ["a"]
    assert fmt_ns(1500) == "1.50 us" and fmt_ns(2.5e6) == "2.50 ms" and fmt_ns(12) == "12 ns"
    assert "slower" in render(rows, 3)
    print("self-test ok")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
