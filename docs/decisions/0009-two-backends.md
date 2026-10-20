# 0009: two backends behind one trait

Date: 2026-10-20. Status: accepted.

## Context

A fast backend needs a correctness oracle that shares none of its tricks. If the oracle were optimized too, a bug in a shared kernel would hide.

## Decision

`Backend` has two implementations. `ReferenceBackend` uses naive loops and is never optimized. `CpuBackend` has its own executor and kernels and must return the reference's bits. A kernel in the fast backend delegates to the reference kernel until a benchmark shows a custom one is faster.

## Consequences

- The reference stays obviously right, and the fast backend is tested against it with exact comparisons, not tolerances.
- Both are checked against the NumPy float64 oracle on shipped graphs, and fuzzed against each other.
- The fast backend is bit-identical to the reference, so the default backend can change without changing any output.
- The trait assumes host tensors: see 0001.
