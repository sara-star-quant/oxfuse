#!/usr/bin/env python3
"""Turn criterion's output into docs/benchmarks.md.

Reads target/criterion/**/new/{benchmark,estimates}.json (median time per
iteration), prints a markdown report with the machine it ran on, and with
--write saves it to docs/benchmarks.md. Stdlib only.

Usage:
  python3 -I scripts/bench_report.py [--write]
"""

import json
import os
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CRITERION = ROOT / "target" / "criterion"

HEADER = """# Benchmarks

Medians from `just bench` (criterion), written by `just bench-report`. These
numbers describe the machine below and nothing else; they are not claims about
other hardware. Backends are compiled once and only `run` is timed, so times
include the executor's own work (allocation, and copies of non-moved outputs).
Nothing here gates CI: timing is noisy.

Reproduce: `just bench && just bench-report`. For the threaded rows, also run
`cargo bench -p oxfuse-runtime --locked --features parallel` before reporting.

GPU rows (`gpu-matmul`, `gpu-add`) come from `just bench-gpu`, which also times the
fast CPU backend in the same run, on one thread and on all threads. `gpu-cold` is
compile plus the first run (a driver may cache compiled shaders across samples, so
it is a lower bound on a first run in a new process). `gpu-resident` keeps inputs
on the device and times `run_device` until the device is idle, including the copy
of the outputs. `gpu-batched` is the resident run with 16 submits per wait, timed
per run, so the gap to `gpu-resident` is the cost of one submit-and-wait.
`gpu-with-transfers` is a host `run`: upload, run, download. The
last column divides the best CPU time by the resident GPU time, so a value below
1 means the GPU is slower. GPU results are not bit-identical to the CPU rows (see
`semantics.md`).

## Optimization log

Each fast kernel returns exactly the reference bits. The figures in this log
were measured at the commit that made each change, so they can differ a little
from the tables below. A change stays only if a
benchmark shows a gain; anything that does not is recorded here and dropped.

- MatMul: cache-tiled loops (j-tile, k-tile, row, k, j) with the ascending-k
  sum order kept. Single-row inner loop: 10.2 ms at 512. Four rows sharing
  each loaded row of B: 8.6 ms at 512. Kept the four-row version.
- MatMul, register block: a 4 x 16 block of C is held in registers for a whole
  k-tile instead of being loaded and stored at every step. The adds into each
  element are the same, in the same order, so the bits are unchanged. 8.70 ms
  to 5.8 ms at 512 (run-to-run spread 0.03 ms), about 0.49 of
  ndarray's speed (5.77 ms against 2.81 ms in the table below). Kept. Tried and dropped, all slower at 512: 4 x 8 (7.4 ms), 4 x 12
  (7.2 ms), 4 x 24 (7.0 ms), 4 x 32 (6.2 ms), 6 x 8 and 8 x 8 (7.3 ms). Packing
  the B tile into a contiguous buffer was not tried: B rows are already read
  sequentially, and the measured limit was traffic on C, not on B.
- Transpose: 32 x 32 tiles for rank 2, strided odometer for other ranks.
  2.59 ms to 0.73 ms at 1024 x 1024. Kept.
- Executor: inputs and constants are borrowed instead of copied, and a
  computed output buffer moves out instead of being copied. Add at 2^20
  elements 426 us to 117 us, Relu 230 us to 66 us, 64 x 64 matmul 19.0 us to
  18.2 us. Kept. These gains come from the executor, not from the kernels.
- Threads (feature `parallel`): matmul rows split across rayon tasks of 16
  rows, same bits at any thread count. Threshold 2^21 multiply-adds; 64 x 64
  (2^18) stays serial, 128 x 128 (2^21) already wins: 146 us to 88 us. At 256
  the time drops from 1.12 ms to 0.31 ms and at 512 from 8.66 ms to 1.6 ms on
  this 8-core machine. Kept, off by default.
- GPU MatMul, register block: each thread of a 16 x 16 workgroup holds a 4 x 4
  block of the result in sixteen scalar variables, so a workgroup covers a
  64 x 64 block and loads one tile of A and one of B per step of 16 in k. Each
  output still sums in ascending k from one accumulator. At 1024 x 1024 in a
  throwaway single-dispatch harness outside the repo (so no repo command
  reproduces these particular figures; the table below is the reproducible
  measurement): one output per thread 16.8 ms (128 GFLOP/s), 2 x 2 per thread
  7.8 ms, 4 x 4 per thread 3.84 ms (559 GFLOP/s). Kept 4 x 4. Tried and
  dropped: 8 x 8 threads with 4 x 4 blocks 4.06 ms, with 8 x 8 blocks 4.79 ms,
  8 x 8 threads with one output 19.3 ms, and the same 4 x 4 block held in an
  array indexed inside a loop, about 55 ms (the device spills it to memory,
  three times slower than no blocking at all). Results stay inside the
  documented bound of the float64 result but are not bit-identical to the
  reference, because the device fuses multiply-add.
- Add, Relu, Sigmoid, Mul, Reshape: the reference loops already vectorize
  (cpu and reference measured within noise at the baseline), so these still
  call the reference kernels. Sigmoid also keeps std's exp, so it is
  identical by construction.
"""


def commit():
    """The git tree id of crates/ at HEAD, marked when tracked files differ from
    it. A tree id survives squashing and amending, so a table stays attached to
    the code it measured (check with `git rev-parse HEAD:crates`)."""
    tree = sh('git', '-C', str(ROOT), 'rev-parse', '--short', 'HEAD:crates')
    dirty = sh('git', '-C', str(ROOT), 'status', '--porcelain', '--untracked-files=no',
               '--', 'crates', 'Cargo.toml', 'Cargo.lock', 'scripts')
    return f"crates tree {tree} (plus uncommitted changes)" if dirty.strip() else f"crates tree {tree}"


def wgpu_version():
    """The wgpu version the GPU benchmarks built against, from Cargo.lock."""
    lock = (ROOT / "Cargo.lock").read_text()
    marker = 'name = "wgpu"\nversion = "'
    i = lock.find(marker)
    return lock[i + len(marker):].split('"', 1)[0] if i >= 0 else "unknown"


def sh(*cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def environment():
    if platform.system() == "Darwin":
        cpu = sh("sysctl", "-n", "machdep.cpu.brand_string")
        cores = sh("sysctl", "-n", "hw.logicalcpu")
        mem = sh("sysctl", "-n", "hw.memsize")
        memory = f"{int(mem) // 2**30} GiB" if mem.isdigit() else "unknown"
    else:
        info = Path("/proc/cpuinfo").read_text() if Path("/proc/cpuinfo").exists() else ""
        names = [l.split(":", 1)[1].strip() for l in info.splitlines() if l.startswith("model name")]
        cpu = names[0] if names else "unknown"
        cores = str(len(names)) if names else "unknown"
        memory = "unknown"
    adapter = CRITERION / "gpu-adapter.txt"
    gpu = adapter.read_text().strip() if adapter.exists() else "none recorded"
    return "\n".join(
        [
            f"- CPU: {cpu} ({cores} logical cores)",
            f"- GPU adapter: {gpu}, wgpu {wgpu_version()}",
            f"- Memory: {memory}",
            f"- OS: {platform.platform()}",
            f"- Rust: {sh('rustc', '--version')}",
            f"- RUSTFLAGS: `{os.environ.get('RUSTFLAGS', '')}`",
            f"- Code: {commit()}",
        ]
    )


def collect():
    results = {}
    for bench in CRITERION.rglob("new/benchmark.json"):
        meta = json.loads(bench.read_text())
        estimates = json.loads(bench.with_name("estimates.json").read_text())
        median_ns = estimates["median"]["point_estimate"]
        group, variant, size = meta["full_id"].split("/")
        results.setdefault(group, {}).setdefault(int(size), {})[variant] = median_ns
    return results


def fmt_time(ns):
    return f"{ns / 1e3:.1f} us" if ns < 1e6 else f"{ns / 1e6:.2f} ms"


GPU_ORDER = ["gpu-cold", "gpu-resident", "gpu-batched", "gpu-with-transfers", "cpu", "cpu-threads"]


def gpu_table(group, by_size):
    lines = [
        f"### {group}",
        "",
        "| size | " + " | ".join(GPU_ORDER) + " | best cpu / resident gpu |",
        "|---|" + "---|" * (len(GPU_ORDER) + 1),
    ]
    for size in sorted(by_size):
        row = by_size[size]
        cells = []
        for v in GPU_ORDER:
            ns = row.get(v)
            if ns is None:
                cells.append("-")
            elif group == "gpu-matmul":
                cells.append(f"{fmt_time(ns)} ({2 * size**3 / ns:.1f} GFLOP/s)")
            else:
                cells.append(fmt_time(ns))
        cpu = [row[v] for v in ("cpu", "cpu-threads") if v in row]
        ratio = f"{min(cpu) / row['gpu-resident']:.2f}x" if cpu and "gpu-resident" in row else "-"
        lines.append(f"| {size} | " + " | ".join(cells) + f" | {ratio} |")
    return "\n".join(lines)


def table(group, by_size):
    if group.startswith("gpu-"):
        return gpu_table(group, by_size)
    variants = sorted({v for row in by_size.values() for v in row}, key=lambda v: (v != "reference", v))
    lines = [
        f"### {group}",
        "",
        "| size | " + " | ".join(variants) + " | best cpu vs reference |",
        "|---|" + "---|" * (len(variants) + 1),
    ]
    for size in sorted(by_size):
        row = by_size[size]
        cells = []
        for v in variants:
            ns = row.get(v)
            if ns is None:
                cells.append("-")
            elif group == "matmul":
                cells.append(f"{fmt_time(ns)} ({2 * size**3 / ns:.1f} GFLOP/s)")
            else:
                cells.append(fmt_time(ns))
        fast = [row[v] for v in row if v.startswith("cpu")]
        speedup = f"{row['reference'] / min(fast):.2f}x" if "reference" in row and fast else "-"
        lines.append(f"| {size} | " + " | ".join(cells) + f" | {speedup} |")
    return "\n".join(lines)


def main():
    if not CRITERION.exists():
        sys.exit("no target/criterion: run `just bench` first")
    results = collect()
    if not results:
        sys.exit("no benchmark results found under target/criterion")
    parts = [HEADER, "## Environment", "", environment(), "", "## Results", ""]
    for group in sorted(results):
        parts += [table(group, results[group]), ""]
    report = "\n".join(parts)
    if "--write" in sys.argv:
        (ROOT / "docs" / "benchmarks.md").write_text(report + "\n")
        print("wrote docs/benchmarks.md")
    else:
        print(report)


if __name__ == "__main__":
    main()
