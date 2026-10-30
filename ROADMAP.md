# Roadmap

A stage is marked done only when it is implemented and tested.

| Stage | Deliverable | Status |
|---|---|---|
| 1 | JSON graph, typed IR, verifier, reference backend, CLI | done |
| 2 | Seven ops, invalid-graph corpus, NumPy oracle with a seeded corpus | done |
| 3 | Exact passes (fold, CSE, DCE), pass manager, IR dump, property tests | done |
| 4 | Fast CPU backend (blocked matmul, optional threads), benchmarks against `matrixmultiply` | done |
| 5 | Located diagnostics, fuzzing, architecture and semantics docs, register-blocked matmul, fast backend as default | done |
| 6 | Optional wgpu GPU backend, checked against the CPU reference and the oracle, with a two-tier numerical contract and a self-test for other devices | done |
| 7 | ONNX import for a small operator subset, validated against an independent runtime | done |
| 8 | Speed and trust: buffer reuse, SIMD matmul with run time dispatch, GPU host path and device-loss handling, fused-multiply-add `MatMul` and an `exp` of our own so the result no longer depends on the platform's math library, mutation testing in CI, Kani proofs | done, shipped in 0.4.0 |
| 9 | Distribution and observability: a golden corpus for other runtimes, `oxfuse run --manifest` and `oxfuse compare`, work and message limits for hostile input, an `mlp` benchmark with a weekly trend, prebuilt binaries with checksums and attestations for the five tested platforms | done on `main`, ships in 0.5.0 |
| 10 | Small-batch speed: one and two rows as wide blocks, a few rows split by columns over threads (a profile ruled out a memory plan and fusion), same bits | done, shipped in 0.5.1 |
| 11 | Certified error bounds: `oxfuse analyze` gives each output element an interval that contains the exact real result, for the inputs given | done on `main` |

## Considered, no commitment

Next, in order: numpy broadcasting and the elementwise operators, then error bounds over a range of inputs, and a transformer-lite set (`Softmax`, `LayerNormalization`, `Gelu`, reductions with a fixed order). Later, if there is demand: convolution and pooling, f16 and int8, a WASM build, Python bindings. Each operator ships only with its evidence: an oracle test, SIMD and reference kernels that agree bit for bit, a GPU bound, mutation testing and a row in the claims ledger.

## Not planned

Broadcasting outside `Add` and `Mul`, batched `MatMul`, operator fusion, Python bindings, a C interface with a C++ example, a RISC-V vector backend checked under emulation, a formally verified hardware block. Each needs a design and evidence first. No dates.
