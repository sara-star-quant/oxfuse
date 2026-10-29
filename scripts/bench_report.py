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
compile plus the first run on a fresh device per sample (a device keeps the
pipelines it compiled; a driver may still cache compiled shaders across samples,
so it is a lower bound on a first run in a new process). `gpu-resident` keeps inputs
on the device and times `run_device` until the device is idle, including the copy
of the outputs. `gpu-batched` is the resident run with 16 submits per wait, timed
per run, so the gap to `gpu-resident` is the cost of one submit-and-wait.
`gpu-with-transfers` is a host `run`: upload, run, download. The
last column divides the best CPU time by the resident GPU time, so a value below
1 means the GPU is slower. GPU `MatMul` matches the CPU bit for bit on the two
adapters it was measured on; the other GPU ops stay inside the bounds in
`semantics.md`. `matmul-skinny` and `matmul-head` time a small batch (the id is
the number of rows) against a 1024 x 1024 weight and a 784 x 10 weight; the
ratio to `ndarray` uses the single-threaded `cpu` column.
`mlp` times a 784-256-10 classifier (two matrix products around a `Relu`, no bias) at
batch 1, 8 and 32; the id is the batch, and `ndarray` runs the same three steps.

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
  documented bound of the float64 result. They were not bit-identical to the
  reference of that time because the device fuses multiply-add. The reference
  fuses now, and the Metal and T4 results are bit-identical to it.
- Code placement: after the in-place kernels were added to the module that held
  the matmul, the single-threaded 512 matmul went from 5.9 ms to 6.4 ms
  (+9 percent, spread 3 percent; bisected to that commit with `just bench-ab`).
  Forcing or forbidding inlining of `matmul_rows` and padding the module did
  nothing. Moving only the elementwise kernels out fixed the single-threaded
  case but cost the threaded 64, 128 and 256 matmuls 5 to 11 percent. With the
  matmul, the transpose and the elementwise kernels each in their own module,
  every matmul size is within noise of 0.3.0, single-threaded and threaded (512:
  5.82 ms on both). The cause is how rustc partitions and lays out code, not
  any change to the loops, so the module split pins the layout.
- Add, Relu, Mul, Reshape: the reference loops already vectorize (cpu and
  reference measured within noise at the baseline), so these still call the
  reference kernels.
- MatMul, fused multiply-add: the definition changed to one fused multiply-add
  per step, so the result differs from 0.3.0 in the last bits and a GPU that
  fuses can now match it exactly. With the same loops, 512 went from 6.02 ms
  to 3.99 ms on the M1 Pro (a 6 x 16 block instead of 4 x 16). In a scratch
  harness on a quiet machine, without the edge handling, the block sizes gave
  (GFLOP/s) 2 x 16 51, 3 x 16 72, 4 x 8 51, 4 x 12 63, 4 x 16 71, 4 x 24 93,
  6 x 8 66, 6 x 16 97, 8 x 8 73, 8 x 12 88. Kept 6 x 16. The kernel in the
  repo measures about 71 GFLOP/s in the table below, so the 97 is an upper
  bound it does not reach yet. The `gemm` crate was tried as a drop-in: 98
  GFLOP/s on one thread, but it splits k above about 1024 and the bits then
  differ from the fused sum (up to 249428 ulp at k = 6000), so it is not used.
- MatMul, explicit SIMD vectors (`fearless_simd`, level picked at run time):
  the same speed as the autovectorized loop on the M1 Pro (512: 3.86 ms and
  3.85 ms) and 10 percent faster at 64. It is there for x86: on a Xeon at
  2.2 GHz with AVX2, in one session, the 512 matmul took 26.8 ms before the
  fused definition, 447 ms after it (a fused multiply-add compiled without the
  FMA flag is a call into libm), and 9.2 ms with the vectors. The reference
  kernels run their scalar loops inside the same feature-enabled wrapper:
  the reference 128 x 128 matmul took 3.66 ms before, 8.96 ms after the fused
  definition and 3.61 ms with the wrapper.
- MatMul, tail columns in place: the columns that do not fill a vector were
  copied into a buffer on every call. They are read in place now, and only the
  last rows of B, where a full vector would run past the end, are copied.
  One row against a 784 x 10 weight: 5.5 us to 1.3 us on the M1 Pro and 6 us
  to 2 us on the Xeon; the 1024 x 1024 weight is unchanged within noise.
- Sigmoid: an `exp` of our own (only IEEE operations and fused multiply-adds,
  so the result does not depend on the platform's math library) on SIMD vectors. 2^20 elements on
  the M1 Pro: 1.76 ms to 0.82 ms, the reference 1.94 ms to 1.02 ms. On the
  Xeon: 5.2 ms to 1.85 ms, the reference 17.1 ms to 14.7 ms. The first x86
  attempt was five times slower: a generic helper that took the vector
  function as a parameter was not compiled with the AVX2 features, and writing
  the loop out in each function fixed it.
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


# Multiply-adds-times-two of one run, from the size in the id, for the groups that report GFLOP/s.
FLOPS = {
    "matmul": lambda s: 2 * s**3,
    "gpu-matmul": lambda s: 2 * s**3,
    "matmul-skinny": lambda rows: 2 * rows * 1024 * 1024,
    "matmul-head": lambda rows: 2 * rows * 784 * 10,
    "mlp": lambda batch: 2 * batch * (784 * 256 + 256 * 10),
}


def table(group, by_size):
    if group in ("gpu-add", "gpu-matmul"):
        return gpu_table(group, by_size)
    variants = sorted({v for row in by_size.values() for v in row}, key=lambda v: (v != "reference", v))
    has_reference = any("reference" in row for row in by_size.values())
    has_ndarray = "ndarray" in variants
    heads = ["size"] + variants
    if has_reference:
        heads.append("best cpu vs reference")
    if has_ndarray:
        heads.append("cpu vs ndarray")
    lines = [f"### {group}", "", "| " + " | ".join(heads) + " |", "|" + "---|" * len(heads)]
    for size in sorted(by_size):
        row = by_size[size]
        cells = []
        for v in variants:
            ns = row.get(v)
            if ns is None:
                cells.append("-")
            elif group in FLOPS:
                cells.append(f"{fmt_time(ns)} ({FLOPS[group](size) / ns:.1f} GFLOP/s)")
            else:
                cells.append(fmt_time(ns))
        fast = [row[v] for v in row if v.startswith("cpu")]
        if has_reference:
            cells.append(f"{row['reference'] / min(fast):.2f}x" if "reference" in row and fast else "-")
        if has_ndarray:
            cells.append(f"{row['ndarray'] / row['cpu']:.2f}x" if "ndarray" in row and "cpu" in row else "-")
        lines.append(f"| {size} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


OTHER_MACHINES = """## Other machines

A Tesla T4 (NVIDIA, Vulkan, driver 580) on a free Kaggle notebook with a 4 vCPU
Xeon at 2.2 GHz (AVX2), run on 2026-10-22 at commit 4c633ff with the flow in
[adapters.md](adapters.md). Same method and columns as above; the CPU columns
are that notebook's CPU. The T4's `MatMul` matched the reference bit for bit.

### gpu-matmul (Tesla T4)

| size | gpu-cold | gpu-resident | gpu-batched | gpu-with-transfers | cpu | cpu-threads | best cpu / resident gpu |
|---|---|---|---|---|---|---|---|
| 128 | 30.75 ms (0.1 GFLOP/s) | 91.4 us (45.9 GFLOP/s) | 48.1 us (87.3 GFLOP/s) | 210.8 us (19.9 GFLOP/s) | 139.8 us (30.0 GFLOP/s) | 93.1 us (45.1 GFLOP/s) | 1.02x |
| 512 | 31.95 ms (8.4 GFLOP/s) | 313.0 us (857.6 GFLOP/s) | 249.7 us (1074.8 GFLOP/s) | 1.29 ms (208.2 GFLOP/s) | 9.30 ms (28.9 GFLOP/s) | 4.87 ms (55.1 GFLOP/s) | 15.56x |

### gpu-add (Tesla T4)

| size | gpu-cold | gpu-resident | gpu-batched | gpu-with-transfers | cpu | cpu-threads | best cpu / resident gpu |
|---|---|---|---|---|---|---|---|
| 1048576 | 32.14 ms | 136.9 us | 94.5 us | 3.31 ms | 502.2 us | 411.1 us | 3.00x |
"""


def main():
    if not CRITERION.exists():
        sys.exit("no target/criterion: run `just bench` first")
    results = collect()
    if not results:
        sys.exit("no benchmark results found under target/criterion")
    parts = [HEADER, "## Environment", "", environment(), "", "## Results", ""]
    for group in sorted(results):
        parts += [table(group, results[group]), ""]
    parts += [OTHER_MACHINES]
    report = "\n".join(parts)
    if "--write" in sys.argv:
        (ROOT / "docs" / "benchmarks.md").write_text(report + "\n")
        print("wrote docs/benchmarks.md")
    else:
        print(report)


if __name__ == "__main__":
    main()
