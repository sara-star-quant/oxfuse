# 0011: The fast CPU backend is the default

Date: 2026-10-20. Status: accepted.

## Context

`oxfuse run` used the reference backend by default and offered `--backend cpu` as an opt-in. The plan for the fast backend said the default would flip only after the claims ledger and fuzzing existed. Both exist now, and the fast backend returns the reference's bits for every op.

## Decision

`oxfuse run` uses the fast CPU backend unless `--backend reference` is given. `BackendKind::default()` is `Cpu`.

The evidence: `crates/cli/tests/backends.rs` compares the printed bytes of both backends on every shipped graph, `crates/runtime/tests/cpu_equivalence.rs` does the same on random graphs, and the `generated` fuzz target compared 3.07 million valid graphs, with infinity and NaN inputs, over 20 minutes without a divergence (see `docs/claims.md`). An earlier run of the raw-byte differential target counted 73.8 million executions, but only about 0.2% of those loaded a graph, so it is not cited as evidence.

## Consequences

- No output changes, because the backends are bit-identical. Only speed changes, and only on the machine in `docs/benchmarks.md`.
- The reference backend stays the oracle. Constant folding always uses `ReferenceEvaluator`, so the optimizer never depends on the fast kernels. `--backend reference` runs the whole graph on it.
- Reverting is one line in `commands.rs` and one in `main.rs`.
- The `parallel` feature is still off by default, so the default fast backend is single-threaded.
- Alternatives considered: keeping the reference as the default (safer by habit, but slower for no output difference once the equivalence evidence exists).
