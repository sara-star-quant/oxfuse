# Architecture

Oxfuse is three crates, an optional fourth for GPUs, and a one-way pipeline.

```
JSON file
   |  oxfuse (crates/cli): loader, GraphFile types, commands
   v
Graph  --- PassManager (fold, layout, cse, dce) --->  Graph      oxfuse-core
   |
   |  Plan::build                                         oxfuse-runtime
   v
Plan --- Backend::compile ---> Executable::run(&[Tensor]) -> Vec<Tensor>
          ReferenceBackend | CpuBackend | WgpuBackend (optional)
```

## Crates and their direction

| Crate | Holds | Depends on |
|---|---|---|
| `oxfuse-core` | the IR (`Graph`, `Op`, `TensorType`), shape inference, the builder, the verifier, `Limits`, the error type, the optimization passes | nothing in the workspace |
| `oxfuse-runtime` | `Tensor`, `Plan`, the `Backend` and `Executable` traits, `ReferenceBackend`, `CpuBackend`, `ReferenceEvaluator` | `oxfuse-core` |
| `oxfuse-wgpu` (directory `crates/wgpu`) | the wgpu backend: generated WGSL kernels, a CPU model of their grids, device tensors, the self-test report. The CLI builds it only with `--features wgpu` | `oxfuse-core`, `oxfuse-runtime` |
| `oxfuse` (directory `crates/cli`, library `oxfuse_cli`) | the JSON format, the loader, the commands and the `oxfuse` binary | both |

Dependencies point one way: core knows nothing about files or backends. That is why constant folding takes a `ConstEvaluator` trait object from the caller. The CLI hands it `ReferenceEvaluator`, so folding runs the same reference kernels the reference backend runs, without core depending on runtime.

## What happens to a graph

1. **Load.** The loader checks the file size, parses the JSON (unknown fields are errors), checks names, orders the nodes topologically, and drives the `GraphBuilder`. The builder infers every type with the shape rules and enforces the `Limits`, so a `Graph` that exists is already well typed. Errors carry a stable code and, where a place in the file is known, a JSON path.
2. **Optimize.** `PassManager` runs `fold`, `layout`, `cse`, `dce` once, in that order. It verifies the graph and its limits before the first pass and after each one, and rejects a pass that changes the graph's interface. Passes rebuild the graph instead of editing it. See [optimization-passes.md](optimization-passes.md).
3. **Plan.** `Plan::build` turns the graph into topologically ordered steps over numbered buffer slots, with constants and input bindings resolved up front. No name or hash lookup happens at run time.
4. **Execute.** A backend compiles the plan and runs it on host tensors.

## Backends

- **`ReferenceBackend`** uses naive loops that are meant to be obviously right. It is never optimized, and its kernels are the oracle for everything else.
- **`CpuBackend`** has its own executor and its own kernels (`crates/runtime/src/cpu/`). Its contract is stronger than a tolerance: for every op it returns the same bits as the reference, including NaN, infinity and signed zero, at any thread count. The fast matmul reorders loops for cache and SIMD but keeps each output element's sum in ascending `k` order from `+0.0`, with no fused multiply-add and no zero skipping. It is the default backend; constant folding still uses the reference kernels. Ops where a custom kernel gained nothing still call the reference kernel; [benchmarks.md](benchmarks.md) records which.

- **`WgpuBackend`** (optional feature, crate `oxfuse-wgpu`) runs the plan as compute dispatches on a device. Shapes are static, so compile fixes every buffer, pipeline and dispatch size. It keeps intermediate tensors on the device, offers an explicit `upload`, `run_device` and `download`, and copies results into fresh buffers so they stay valid after the next run. Its numerical contract has two tiers: [semantics.md](semantics.md). Measured devices: [adapters.md](adapters.md).

The CPU executor borrows inputs and constants and moves computed output buffers out instead of copying. The optional `parallel` feature splits matmul by output rows only, which is why the thread count cannot change a result.

## How the checks fit together

| Check | What it compares |
|---|---|
| NumPy oracle (`scripts/gen_expected.py`) | the reference backend against float64 NumPy, on hand-computed examples and a seeded random corpus |
| Equivalence tests | optimized against unoptimized graphs, and fast against reference backend, bit for bit, on examples, the corpus and random graphs |
| Fuzz targets (`fuzz/`) | the loader against arbitrary text, and the backends against each other on whatever loads |
| GPU tests (`crates/wgpu/tests`) | the device against the reference and the float64 oracle, a CPU model of every kernel grid, exhaustive and stratified sweeps, limits and concurrency |
| Mutation testing | whether the tests notice deliberate breakage (`just mutants`) |

The oracle is independent code in another language, so a bug shared by both backends would still show up as a disagreement with NumPy.

## Known limits

- The `Backend` trait assumes host tensors. The wgpu executable adds its device API beside it instead of changing the trait; see decision 0012.
- Because the fast backend is bit-identical to the reference by design, no test on outputs can show that a fast kernel was actually dispatched. Only the benchmarks show it. See "Known test limit" in `CONTRIBUTING.md`.
- Oxfuse is CPU by default. The wgpu backend is optional and is measured, not promised, on the devices in [adapters.md](adapters.md).

## Memory bounds and when to revisit them

`Limits` caps every tensor at 2^28 elements and all values of a graph together at 2^29 elements (2 GiB of `f32`), and the loader refuses a file over 64 MiB. Every allocation a backend makes is a function of those caps, so a graph that loads has a known worst case and the backends do not guard individual allocations. Allocating beyond what the machine has still aborts the process on the CPU backends (`vec![0.0; n]`), and the buffer reuse in the fast backend lowers the peak but does not change this. On the GPU, an out-of-memory error from the device is captured per call and returned as `Error::Backend`.

This is a deliberate trade: a fallible allocation path (`try_reserve_exact` on every output, and a new `Error` variant, which is a breaking change to a public enum) adds code and a test matrix to guard a case the caps make unlikely. Revisit it when any of these becomes true:

- The caps are raised so that the worst case passes 4 GiB, or a caller sets `Limits` that large.
- Oxfuse is run on a laptop or a small board where 2 GiB is a large share of free memory.
- It is run on an integrated GPU. The GPU shares system memory, so the host copy, the device buffers and the staging buffers of `run()` all count against the same pool, and one tensor can cost three times its size at once. Keeping tensors resident with `run_device` avoids the copies.
- A device reports a `max_buffer_size` or storage binding limit smaller than a tensor the caps allow (the checks in `check_tensor` already return an error for that case).

The plan for that day: compute the peak from the execution plan before running (the liveness the buffer reuse already uses), compare it with a configurable budget, and fail with a named error before allocating anything; then switch the large allocations to `try_reserve_exact`.
