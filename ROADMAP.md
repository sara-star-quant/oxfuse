# Roadmap

A stage is marked done only when it is implemented and tested.

| Stage | Deliverable | Status |
|---|---|---|
| 1 | JSON graph, typed IR, verifier, reference backend, CLI | done |
| 2 | Seven ops, invalid-graph corpus, NumPy oracle with a seeded corpus | done |
| 3 | Exact passes (fold, CSE, DCE), pass manager, IR dump, property tests | done |
| 4 | Fast CPU backend (blocked matmul, optional threads), benchmarks against `matrixmultiply` | done |
| 5 | Located diagnostics, fuzzing, architecture and semantics docs, decision records, register-blocked matmul, fast backend as default | done |
| 6 | Optional wgpu GPU backend, checked against the CPU reference and the oracle, with a two-tier numerical contract and a self-test for other devices | done |
| 7 | ONNX import for a small operator subset, validated against an independent runtime | next, not started |

## Not planned

Broadcasting beyond scalars, batched `MatMul`, operator fusion, Python bindings, a C interface with a C++ example, buffer reuse, a RISC-V vector backend checked under emulation, a formally verified hardware block. Each needs a design and evidence first. No dates.
