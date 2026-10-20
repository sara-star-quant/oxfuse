# Benchmarks

Medians from `just bench` (criterion), written by `just bench-report`. These
numbers describe the machine below and nothing else; they are not claims about
other hardware. Backends are compiled once and only `run` is timed, so times
include the executor's own work (allocation, and copies of non-moved outputs).
Nothing here gates CI: timing is noisy.

Reproduce: `just bench && just bench-report`. For the threaded rows, also run
`cargo bench -p oxfuse-runtime --locked --features parallel` before reporting.

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
- Add, Relu, Sigmoid, Mul, Reshape: the reference loops already vectorize
  (cpu and reference measured within noise at the baseline), so these still
  call the reference kernels. Sigmoid also keeps std's exp, so it is
  identical by construction.

## Environment

- CPU: Apple M1 Pro (8 logical cores)
- Memory: 16 GiB
- OS: macOS-26.7.1-arm64-arm-64bit-Mach-O
- Rust: rustc 1.99.1 (58cf8f9c3 2026-10-13)
- RUSTFLAGS: ``
- Commit: d94b081

## Results

### add

| size | reference | cpu | cpu-threads | best cpu vs reference |
|---|---|---|---|---|
| 1048576 | 446.1 us | 116.1 us | 115.8 us | 3.85x |

### matmul

| size | reference | cpu | cpu-threads | ndarray | best cpu vs reference |
|---|---|---|---|---|---|
| 64 | 168.6 us (3.1 GFLOP/s) | 12.0 us (43.6 GFLOP/s) | 11.8 us (44.5 GFLOP/s) | 6.4 us (81.9 GFLOP/s) | 14.30x |
| 128 | 1.59 ms (2.6 GFLOP/s) | 92.4 us (45.4 GFLOP/s) | 69.0 us (60.8 GFLOP/s) | 47.7 us (87.9 GFLOP/s) | 23.04x |
| 256 | 14.70 ms (2.3 GFLOP/s) | 724.2 us (46.3 GFLOP/s) | 252.6 us (132.8 GFLOP/s) | 360.6 us (93.0 GFLOP/s) | 58.18x |
| 512 | 121.02 ms (2.2 GFLOP/s) | 5.77 ms (46.6 GFLOP/s) | 1.18 ms (228.1 GFLOP/s) | 2.81 ms (95.4 GFLOP/s) | 102.84x |

### pipeline

| size | cpu-optimized | cpu-threads-optimized | cpu-threads-unoptimized | cpu-unoptimized | best cpu vs reference |
|---|---|---|---|---|---|
| 256 | 187.4 us | 129.1 us | 202.7 us | 247.6 us | - |

### relu

| size | reference | cpu | cpu-threads | best cpu vs reference |
|---|---|---|---|---|
| 1048576 | 231.2 us | 68.1 us | 88.3 us | 3.39x |

### sigmoid

| size | reference | cpu | cpu-threads | best cpu vs reference |
|---|---|---|---|---|
| 1048576 | 1.89 ms | 1.74 ms | 1.74 ms | 1.08x |

### transpose

| size | reference | cpu | cpu-threads | best cpu vs reference |
|---|---|---|---|---|
| 1024 | 2.57 ms | 566.6 us | 566.3 us | 4.54x |

