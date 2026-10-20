# Claims ledger

Every claim this project makes about itself, with the evidence behind it. A claim without a row here does not belong in the README. `cargo test -p oxfuse --test claims` checks that every evidence path exists.

| Claim | Evidence | How to check |
|---|---|---|
| The optimization passes never change a result, including NaN, infinity and signed zero | `crates/runtime/tests/optimization_equivalence.rs`, `crates/cli/tests/optimization.rs` | `cargo test --workspace` |
| The reference backend agrees with an independent NumPy float64 oracle on every example and corpus graph | `scripts/gen_expected.py`, `tests/corpus`, `crates/cli/tests/examples.rs` | `just oracle-check && cargo test -p oxfuse --test examples` |
| `MatMul` stays within the documented rounding bound | `crates/runtime/src/kernels.rs`, `tests/corpus` | `cargo test -p oxfuse --test examples` |
| The fast CPU backend returns exactly the reference backend's bits, on every shipped graph and on random graphs | `crates/runtime/tests/backend_contract.rs`, `crates/runtime/tests/cpu_equivalence.rs`, `crates/cli/tests/backends.rs` | `cargo test --workspace` |
| `oxfuse run` defaults to the fast backend, and the output is the reference's bytes | `crates/cli/tests/backends.rs`, `docs/decisions/0011-default-backend.md` | `cargo test -p oxfuse --test backends` |
| The threaded matmul returns the same bits at any thread count | `crates/runtime/src/cpu/kernels.rs` | `just test-parallel` |
| `Sigmoid` on the fast backend needs no separate accuracy bound because it runs the reference code | `crates/runtime/src/cpu/kernels.rs` | read the dispatch in `apply` |
| Shapes are inferred for every op, and every IR invariant is re-checked after each pass | `crates/core/src/shape.rs`, `crates/core/src/verify.rs`, `crates/core/src/passes/mod.rs` | `cargo test -p oxfuse-core` |
| Malformed graphs are rejected with a stable error code | `crates/core/src/error.rs`, `tests/invalid` | `cargo test -p oxfuse --test invalid_graphs` |
| The workspace builds and passes its tests on Rust 1.88, the declared minimum | `Cargo.toml`, `justfile`, `.github/workflows/ci.yml` | `just msrv` |
| Untrusted input files are bounded before anything is allocated | `crates/core/src/limits.rs`, `tests/invalid` | `cargo test -p oxfuse --test invalid_graphs` |
| Graph errors name the place in the file as a JSON path (`nodes[1].inputs[0]`); errors about the graph as a whole have none | `crates/cli/src/loader.rs`, `crates/cli/tests/invalid_graphs.rs` | `cargo test -p oxfuse --lib loader && cargo test -p oxfuse --test invalid_graphs` |
| The loader does not panic on arbitrary text, and graphs give the same bits on both backends with and without optimization, within the recorded fuzz runs: 3.07 million generated valid graphs compared in 20 minutes (with infinity and NaN inputs and sizes that reach the fast matmul's register block), 41.8 million raw-byte executions of the differential target in 10 minutes (about 0.2% of raw-byte executions load a graph, so that count says little about the backends), and 87.2 million loader executions in 20 minutes; no crash. Apple M1 Pro, `nightly-2026-05-22`, 2026-10-20, seeded from the shipped graphs | `fuzz/fuzz_targets/loader.rs`, `fuzz/fuzz_targets/differential.rs`, `fuzz/fuzz_targets/generated.rs` | `just fuzz generated 60` |
| The tests catch deliberate breakage; the surviving mutants are explained | `CONTRIBUTING.md` | `just mutants` |
| Measured speed of the fast backend on the recorded machine | `docs/benchmarks.md` | `just bench && just bench-report` |
| Oxfuse is CPU only; no accelerator or vendor support is claimed | `crates/cli/tests/repo_hygiene.rs` | `cargo test -p oxfuse --test repo_hygiene` |
