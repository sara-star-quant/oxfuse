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

## Environment

- CPU: Apple M1 Pro (8 logical cores)
- GPU adapter: Apple M1 Pro (Metal), wgpu 30.0.1
- Memory: 16 GiB
- OS: macOS-26.7.1-arm64-arm-64bit-Mach-O
- Rust: rustc 1.99.1 (58cf8f9c3 2026-10-13)
- RUSTFLAGS: ``
- Code: crates tree 71912ef (plus uncommitted changes)

## Results

### add

| size | reference | cpu | cpu-threads | best cpu vs reference |
|---|---|---|---|---|
| 1048576 | 438.0 us | 114.8 us | 121.8 us | 3.82x |

### gpu-add

| size | gpu-cold | gpu-resident | gpu-batched | gpu-with-transfers | cpu | cpu-threads | best cpu / resident gpu |
|---|---|---|---|---|---|---|---|
| 1048576 | 3.00 ms | 357.8 us | 197.2 us | 2.69 ms | 113.0 us | 114.8 us | 0.32x |
| 16777216 | 27.92 ms | 3.70 ms | 2.28 ms | 37.35 ms | 2.61 ms | 2.62 ms | 0.71x |

### gpu-many

| size | loop-of-16-runs | run-many-16 |
|---|---|---|
| 128 | 5.42 ms | 3.38 ms |
| 512 | 20.89 ms | 17.54 ms |

### gpu-matmul

| size | gpu-cold | gpu-resident | gpu-batched | gpu-with-transfers | cpu | cpu-threads | best cpu / resident gpu |
|---|---|---|---|---|---|---|---|
| 128 | 1.48 ms (2.8 GFLOP/s) | 260.5 us (16.1 GFLOP/s) | 76.3 us (55.0 GFLOP/s) | 335.1 us (12.5 GFLOP/s) | 65.4 us (64.1 GFLOP/s) | 53.4 us (78.6 GFLOP/s) | 0.20x |
| 256 | 2.04 ms (16.5 GFLOP/s) | 325.8 us (103.0 GFLOP/s) | 135.9 us (246.9 GFLOP/s) | 589.0 us (57.0 GFLOP/s) | 504.8 us (66.5 GFLOP/s) | 167.3 us (200.5 GFLOP/s) | 0.51x |
| 512 | 2.73 ms (98.4 GFLOP/s) | 759.9 us (353.3 GFLOP/s) | 482.6 us (556.2 GFLOP/s) | 1.31 ms (205.2 GFLOP/s) | 4.00 ms (67.1 GFLOP/s) | 808.9 us (331.9 GFLOP/s) | 1.06x |
| 1024 | 6.48 ms (331.2 GFLOP/s) | 3.75 ms (572.3 GFLOP/s) | 3.26 ms (657.8 GFLOP/s) | 5.79 ms (370.9 GFLOP/s) | 32.18 ms (66.7 GFLOP/s) | 5.69 ms (377.5 GFLOP/s) | 1.52x |
| 2048 | 33.48 ms (513.1 GFLOP/s) | 26.15 ms (656.9 GFLOP/s) | 25.05 ms (685.9 GFLOP/s) | 34.29 ms (501.0 GFLOP/s) | 291.90 ms (58.9 GFLOP/s) | 53.85 ms (319.0 GFLOP/s) | 2.06x |

### gpu-outputs

| size | gpu-with-transfers |
|---|---|
| 8 | 466.8 us |

### matmul

| size | reference | cpu | cpu-threads | ndarray | best cpu vs reference | cpu vs ndarray |
|---|---|---|---|---|---|---|
| 64 | 217.2 us (2.4 GFLOP/s) | 8.1 us (64.6 GFLOP/s) | 8.4 us (62.2 GFLOP/s) | 6.8 us (77.6 GFLOP/s) | 26.77x | 0.83x |
| 128 | 2.17 ms (1.9 GFLOP/s) | 62.4 us (67.3 GFLOP/s) | 53.7 us (78.2 GFLOP/s) | 50.7 us (82.7 GFLOP/s) | 40.44x | 0.81x |
| 256 | 20.23 ms (1.7 GFLOP/s) | 473.0 us (70.9 GFLOP/s) | 166.6 us (201.4 GFLOP/s) | 377.9 us (88.8 GFLOP/s) | 121.41x | 0.80x |
| 512 | 170.38 ms (1.6 GFLOP/s) | 3.79 ms (70.9 GFLOP/s) | 812.6 us (330.4 GFLOP/s) | 2.95 ms (91.1 GFLOP/s) | 209.68x | 0.78x |

### matmul-head

| size | cpu | cpu-threads | ndarray | cpu vs ndarray |
|---|---|---|---|---|
| 1 | 1.3 us (12.5 GFLOP/s) | 1.3 us (11.6 GFLOP/s) | 4.5 us (3.5 GFLOP/s) | 3.60x |
| 2 | 1.3 us (23.7 GFLOP/s) | 1.4 us (22.4 GFLOP/s) | 4.4 us (7.2 GFLOP/s) | 3.31x |
| 3 | 1.5 us (30.5 GFLOP/s) | 1.6 us (28.6 GFLOP/s) | 4.5 us (10.4 GFLOP/s) | 2.95x |
| 5 | 2.2 us (36.1 GFLOP/s) | 2.3 us (33.9 GFLOP/s) | 4.6 us (17.2 GFLOP/s) | 2.10x |
| 8 | 3.6 us (34.9 GFLOP/s) | 3.8 us (32.6 GFLOP/s) | 4.7 us (26.9 GFLOP/s) | 1.30x |
| 32 | 12.8 us (39.3 GFLOP/s) | 13.6 us (36.9 GFLOP/s) | 14.7 us (34.2 GFLOP/s) | 1.15x |

### matmul-skinny

| size | cpu | cpu-threads | ndarray | cpu vs ndarray |
|---|---|---|---|---|
| 1 | 80.4 us (26.1 GFLOP/s) | 84.5 us (24.8 GFLOP/s) | 333.5 us (6.3 GFLOP/s) | 4.15x |
| 2 | 85.3 us (49.2 GFLOP/s) | 90.2 us (46.5 GFLOP/s) | 334.4 us (12.5 GFLOP/s) | 3.92x |
| 3 | 96.5 us (65.2 GFLOP/s) | 100.8 us (62.4 GFLOP/s) | 334.9 us (18.8 GFLOP/s) | 3.47x |
| 5 | 158.2 us (66.3 GFLOP/s) | 166.1 us (63.1 GFLOP/s) | 336.1 us (31.2 GFLOP/s) | 2.12x |
| 8 | 264.8 us (63.4 GFLOP/s) | 266.2 us (63.0 GFLOP/s) | 333.3 us (50.3 GFLOP/s) | 1.26x |
| 32 | 977.4 us (68.7 GFLOP/s) | 547.8 us (122.5 GFLOP/s) | 881.4 us (76.1 GFLOP/s) | 0.90x |

### mlp

| size | reference | cpu | cpu-threads | ndarray | best cpu vs reference | cpu vs ndarray |
|---|---|---|---|---|---|---|
| 1 | 260.9 us (1.6 GFLOP/s) | 23.3 us (17.5 GFLOP/s) | 23.1 us (17.6 GFLOP/s) | 57.5 us (7.1 GFLOP/s) | 11.30x | 2.47x |
| 8 | 2.09 ms (1.6 GFLOP/s) | 61.2 us (53.1 GFLOP/s) | 61.2 us (53.2 GFLOP/s) | 57.9 us (56.2 GFLOP/s) | 34.12x | 0.95x |
| 32 | 8.36 ms (1.6 GFLOP/s) | 208.3 us (62.4 GFLOP/s) | 141.4 us (92.0 GFLOP/s) | 168.4 us (77.2 GFLOP/s) | 59.09x | 0.81x |

### pipeline

| size | cpu-optimized | cpu-threads-optimized | cpu-threads-unoptimized | cpu-unoptimized |
|---|---|---|---|---|
| 256 | 125.0 us | 87.1 us | 165.5 us | 165.0 us |

### relu

| size | reference | cpu | cpu-threads | best cpu vs reference |
|---|---|---|---|---|
| 1048576 | 230.2 us | 65.6 us | 67.0 us | 3.51x |

### sigmoid

| size | reference | cpu | cpu-threads | best cpu vs reference |
|---|---|---|---|---|
| 1048576 | 1.02 ms | 823.0 us | 236.7 us | 4.30x |

### transpose

| size | reference | cpu | cpu-threads | best cpu vs reference |
|---|---|---|---|---|
| 1024 | 2.70 ms | 447.8 us | 287.3 us | 9.40x |

## Other machines

A Tesla T4 (NVIDIA, Vulkan, driver 580) on a free Kaggle notebook with a 4 vCPU
Xeon at 2.2 GHz (AVX2), run on 2026-10-22 at commit 4c633ff with the flow in
[adapters.md](adapters.md). Same method and columns as above; the CPU columns
are that notebook's CPU. The T4's `MatMul` matched the reference bit for bit.

### mlp (Xeon, AVX2)

The same classifier on the free Kaggle notebook's 2.2 GHz Xeon (AVX2), run on
2026-10-29 at commit 7b675de by the flow in the `oxfuse-x86-ab` notes: five
interleaved rounds, medians. `cpu vs ndarray` is the single-threaded `cpu`
column against `ndarray`; the notebook's timer resolves 1 us.

| size | reference | cpu | ndarray | best cpu vs reference | cpu vs ndarray |
|---|---|---|---|---|---|
| 1 | 330 us | 33 us | 106 us | 10.00x | 3.21x |
| 8 | 2.63 ms | 133 us | 109 us | 19.78x | 0.82x |
| 32 | 10.45 ms | 502 us | 305 us | 20.81x | 0.61x |

### gpu-matmul (Tesla T4)

| size | gpu-cold | gpu-resident | gpu-batched | gpu-with-transfers | cpu | cpu-threads | best cpu / resident gpu |
|---|---|---|---|---|---|---|---|
| 128 | 30.75 ms (0.1 GFLOP/s) | 91.4 us (45.9 GFLOP/s) | 48.1 us (87.3 GFLOP/s) | 210.8 us (19.9 GFLOP/s) | 139.8 us (30.0 GFLOP/s) | 93.1 us (45.1 GFLOP/s) | 1.02x |
| 512 | 31.95 ms (8.4 GFLOP/s) | 313.0 us (857.6 GFLOP/s) | 249.7 us (1074.8 GFLOP/s) | 1.29 ms (208.2 GFLOP/s) | 9.30 ms (28.9 GFLOP/s) | 4.87 ms (55.1 GFLOP/s) | 15.56x |

### gpu-add (Tesla T4)

| size | gpu-cold | gpu-resident | gpu-batched | gpu-with-transfers | cpu | cpu-threads | best cpu / resident gpu |
|---|---|---|---|---|---|---|---|
| 1048576 | 32.14 ms | 136.9 us | 94.5 us | 3.31 ms | 502.2 us | 411.1 us | 3.00x |

