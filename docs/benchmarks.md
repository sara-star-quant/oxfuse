# Benchmarks

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
of the outputs. `gpu-with-transfers` is a host `run`: upload, run, download. The
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

## Environment

- CPU: Apple M1 Pro (8 logical cores)
- GPU adapter: Apple M1 Pro (Metal), wgpu 30.0.1
- Memory: 16 GiB
- OS: macOS-26.7.1-arm64-arm-64bit-Mach-O
- Rust: rustc 1.99.1 (58cf8f9c3 2026-10-13)
- RUSTFLAGS: ``
- Code: crates tree c0f23a8

## Results

### add

| size | reference | cpu | cpu-threads | best cpu vs reference |
|---|---|---|---|---|
| 1048576 | 444.7 us | 113.1 us | 112.7 us | 3.94x |

### gpu-add

| size | gpu-cold | gpu-resident | gpu-with-transfers | cpu | cpu-threads | best cpu / resident gpu |
|---|---|---|---|---|---|---|
| 1048576 | 2.61 ms | 389.7 us | 3.10 ms | 123.0 us | 120.8 us | 0.31x |
| 16777216 | 33.05 ms | 3.49 ms | 39.49 ms | 2.60 ms | 2.60 ms | 0.74x |

### gpu-matmul

| size | gpu-cold | gpu-resident | gpu-with-transfers | cpu | cpu-threads | best cpu / resident gpu |
|---|---|---|---|---|---|---|
| 128 | 963.4 us (4.4 GFLOP/s) | 348.7 us (12.0 GFLOP/s) | 368.1 us (11.4 GFLOP/s) | 94.4 us (44.4 GFLOP/s) | 68.8 us (61.0 GFLOP/s) | 0.20x |
| 256 | 1.09 ms (30.9 GFLOP/s) | 398.9 us (84.1 GFLOP/s) | 609.9 us (55.0 GFLOP/s) | 738.5 us (45.4 GFLOP/s) | 249.9 us (134.2 GFLOP/s) | 0.63x |
| 512 | 1.71 ms (156.8 GFLOP/s) | 715.8 us (375.0 GFLOP/s) | 1.29 ms (208.1 GFLOP/s) | 5.93 ms (45.3 GFLOP/s) | 1.17 ms (229.9 GFLOP/s) | 1.63x |
| 1024 | 6.28 ms (342.0 GFLOP/s) | 3.50 ms (613.0 GFLOP/s) | 6.16 ms (348.3 GFLOP/s) | 47.01 ms (45.7 GFLOP/s) | 8.04 ms (267.1 GFLOP/s) | 2.30x |
| 2048 | 34.20 ms (502.3 GFLOP/s) | 25.05 ms (685.8 GFLOP/s) | 34.74 ms (494.5 GFLOP/s) | 400.65 ms (42.9 GFLOP/s) | 72.43 ms (237.2 GFLOP/s) | 2.89x |

### matmul

| size | reference | cpu | cpu-threads | ndarray | best cpu vs reference |
|---|---|---|---|---|---|
| 64 | 167.7 us (3.1 GFLOP/s) | 12.0 us (43.6 GFLOP/s) | 12.1 us (43.5 GFLOP/s) | 6.4 us (82.0 GFLOP/s) | 13.93x |
| 128 | 1.60 ms (2.6 GFLOP/s) | 92.5 us (45.3 GFLOP/s) | 68.6 us (61.2 GFLOP/s) | 48.1 us (87.3 GFLOP/s) | 23.35x |
| 256 | 14.69 ms (2.3 GFLOP/s) | 724.8 us (46.3 GFLOP/s) | 249.2 us (134.6 GFLOP/s) | 360.4 us (93.1 GFLOP/s) | 58.95x |
| 512 | 121.13 ms (2.2 GFLOP/s) | 5.77 ms (46.6 GFLOP/s) | 1.17 ms (230.3 GFLOP/s) | 2.81 ms (95.4 GFLOP/s) | 103.94x |

### pipeline

| size | cpu-optimized | cpu-threads-optimized | cpu-threads-unoptimized | cpu-unoptimized | best cpu vs reference |
|---|---|---|---|---|---|
| 256 | 186.4 us | 126.7 us | 202.2 us | 249.2 us | - |

### relu

| size | reference | cpu | cpu-threads | best cpu vs reference |
|---|---|---|---|---|
| 1048576 | 231.0 us | 68.7 us | 67.0 us | 3.45x |

### sigmoid

| size | reference | cpu | cpu-threads | best cpu vs reference |
|---|---|---|---|---|
| 1048576 | 1.89 ms | 1.74 ms | 1.74 ms | 1.08x |

### transpose

| size | reference | cpu | cpu-threads | best cpu vs reference |
|---|---|---|---|---|
| 1024 | 2.58 ms | 568.8 us | 575.5 us | 4.53x |

