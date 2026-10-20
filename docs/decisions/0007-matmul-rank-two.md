# 0007: `MatMul` accepts rank 2 and its rule is written for batches

Date: 2026-10-20. Status: accepted.

## Context

ONNX-style graphs use batched matrix multiplication. Version one needs the plain matrix product and a kernel that is easy to prove correct.

## Decision

The shape rule is written for rank 2 and above, and the function restricts it to rank 2. The kernels handle `[M, K] x [K, N]` only.

## Consequences

- Batching later widens one check and adds a loop over the leading dimensions; the kernel contract (ascending-`k` sums from `+0.0`) does not change.
- A rank 3 operand is rejected today with a message that names the rank.
- Alternatives considered: batched kernels now (nothing exercises them).
