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
- Code: crates tree 0061bc2 (plus uncommitted changes)

## Results

### add

| size | reference | cpu | cpu-threads | best cpu vs reference |
|---|---|---|---|---|
| 1048576 | 442.9 us | 111.4 us | 113.4 us | 3.98x |

### gpu-add

| size | gpu-cold | gpu-resident | gpu-batched | gpu-with-transfers | cpu | cpu-threads | best cpu / resident gpu |
|---|---|---|---|---|---|---|---|
| 1048576 | 2.94 ms | 395.2 us | 199.8 us | 2.58 ms | 105.9 us | 112.0 us | 0.27x |
| 16777216 | 27.99 ms | 3.70 ms | 2.31 ms | 36.25 ms | 2.60 ms | 2.60 ms | 0.70x |

### gpu-many

| size | loop-of-16-runs | run-many-16 |
|---|---|---|
| 128 | 6.22 ms | 3.13 ms |
| 512 | 20.64 ms | 17.20 ms |

### gpu-matmul

| size | gpu-cold | gpu-resident | gpu-batched | gpu-with-transfers | cpu | cpu-threads | best cpu / resident gpu |
|---|---|---|---|---|---|---|---|
| 128 | 1.44 ms (2.9 GFLOP/s) | 302.6 us (13.9 GFLOP/s) | 80.7 us (52.0 GFLOP/s) | 383.9 us (10.9 GFLOP/s) | 62.3 us (67.3 GFLOP/s) | 48.5 us (86.5 GFLOP/s) | 0.16x |
| 256 | 1.99 ms (16.8 GFLOP/s) | 363.3 us (92.4 GFLOP/s) | 140.2 us (239.4 GFLOP/s) | 634.2 us (52.9 GFLOP/s) | 482.0 us (69.6 GFLOP/s) | 160.0 us (209.7 GFLOP/s) | 0.44x |
| 512 | 2.58 ms (103.9 GFLOP/s) | 774.3 us (346.7 GFLOP/s) | 484.7 us (553.8 GFLOP/s) | 1.29 ms (208.4 GFLOP/s) | 3.83 ms (70.1 GFLOP/s) | 792.3 us (338.8 GFLOP/s) | 1.02x |
| 1024 | 6.39 ms (336.1 GFLOP/s) | 3.76 ms (571.1 GFLOP/s) | 3.26 ms (658.1 GFLOP/s) | 5.68 ms (378.0 GFLOP/s) | 30.65 ms (70.1 GFLOP/s) | 5.36 ms (400.8 GFLOP/s) | 1.43x |
| 2048 | 33.20 ms (517.5 GFLOP/s) | 26.21 ms (655.4 GFLOP/s) | 25.11 ms (684.1 GFLOP/s) | 34.05 ms (504.5 GFLOP/s) | 276.51 ms (62.1 GFLOP/s) | 52.35 ms (328.2 GFLOP/s) | 2.00x |

### gpu-outputs

| size | gpu-with-transfers |
|---|---|
| 8 | 479.9 us |

### matmul

| size | reference | cpu | cpu-threads | ndarray | best cpu vs reference | cpu vs ndarray |
|---|---|---|---|---|---|---|
| 64 | 202.9 us (2.6 GFLOP/s) | 7.9 us (66.2 GFLOP/s) | 7.9 us (66.0 GFLOP/s) | 6.5 us (81.2 GFLOP/s) | 25.61x | 0.81x |
| 128 | 2.04 ms (2.1 GFLOP/s) | 61.1 us (68.7 GFLOP/s) | 48.6 us (86.2 GFLOP/s) | 48.1 us (87.3 GFLOP/s) | 41.85x | 0.79x |
| 256 | 18.90 ms (1.8 GFLOP/s) | 471.1 us (71.2 GFLOP/s) | 160.6 us (209.0 GFLOP/s) | 360.4 us (93.1 GFLOP/s) | 117.71x | 0.76x |
| 512 | 158.91 ms (1.7 GFLOP/s) | 3.76 ms (71.4 GFLOP/s) | 788.4 us (340.5 GFLOP/s) | 2.81 ms (95.4 GFLOP/s) | 201.56x | 0.75x |

### matmul-head

| size | cpu | cpu-threads | ndarray | cpu vs ndarray |
|---|---|---|---|---|
| 1 | 1.3 us (12.5 GFLOP/s) | 1.3 us (12.3 GFLOP/s) | 4.3 us (3.7 GFLOP/s) | 3.41x |
| 2 | 1.3 us (24.0 GFLOP/s) | 1.3 us (23.5 GFLOP/s) | 4.1 us (7.6 GFLOP/s) | 3.16x |
| 3 | 1.5 us (30.9 GFLOP/s) | 1.6 us (30.0 GFLOP/s) | 4.3 us (11.0 GFLOP/s) | 2.81x |
| 5 | 2.2 us (36.2 GFLOP/s) | 2.2 us (35.5 GFLOP/s) | 4.3 us (18.2 GFLOP/s) | 1.99x |
| 8 | 3.6 us (34.9 GFLOP/s) | 3.7 us (34.3 GFLOP/s) | 4.4 us (28.5 GFLOP/s) | 1.22x |
| 32 | 12.8 us (39.4 GFLOP/s) | 13.0 us (38.6 GFLOP/s) | 13.8 us (36.3 GFLOP/s) | 1.09x |

### matmul-skinny

| size | cpu | cpu-threads | ndarray | cpu vs ndarray |
|---|---|---|---|---|
| 1 | 78.7 us (26.7 GFLOP/s) | 80.0 us (26.2 GFLOP/s) | 306.9 us (6.8 GFLOP/s) | 3.90x |
| 2 | 85.1 us (49.3 GFLOP/s) | 84.7 us (49.5 GFLOP/s) | 308.0 us (13.6 GFLOP/s) | 3.62x |
| 3 | 96.1 us (65.5 GFLOP/s) | 96.2 us (65.4 GFLOP/s) | 308.4 us (20.4 GFLOP/s) | 3.21x |
| 5 | 158.5 us (66.2 GFLOP/s) | 158.6 us (66.1 GFLOP/s) | 329.2 us (31.9 GFLOP/s) | 2.08x |
| 8 | 265.3 us (63.2 GFLOP/s) | 265.3 us (63.2 GFLOP/s) | 309.2 us (54.3 GFLOP/s) | 1.17x |
| 32 | 982.1 us (68.3 GFLOP/s) | 526.9 us (127.4 GFLOP/s) | 829.0 us (80.9 GFLOP/s) | 0.84x |

### pipeline

| size | cpu-optimized | cpu-threads-optimized | cpu-threads-unoptimized | cpu-unoptimized |
|---|---|---|---|---|
| 256 | 125.1 us | 84.0 us | 167.3 us | 167.2 us |

### relu

| size | reference | cpu | cpu-threads | best cpu vs reference |
|---|---|---|---|---|
| 1048576 | 234.8 us | 66.4 us | 64.9 us | 3.62x |

### sigmoid

| size | reference | cpu | cpu-threads | best cpu vs reference |
|---|---|---|---|---|
| 1048576 | 1.02 ms | 822.1 us | 229.6 us | 4.44x |

### transpose

| size | reference | cpu | cpu-threads | best cpu vs reference |
|---|---|---|---|---|
| 1024 | 2.56 ms | 447.6 us | 278.1 us | 9.22x |

## Other machines

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

