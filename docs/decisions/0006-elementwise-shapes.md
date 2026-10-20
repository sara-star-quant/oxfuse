# 0006: elementwise ops take equal shapes or one scalar

Date: 2026-10-20. Status: accepted.

## Context

Full NumPy broadcasting touches every elementwise kernel, the fast backend and the oracle.

## Decision

`Add` and `Mul` accept two operands of equal shape, or one rank-0 operand and one of any shape. Everything else is rejected with a message that says broadcasting is not supported yet.

## Consequences

- It is a strict subset of NumPy broadcasting, so every accepted graph means the same thing there and the oracle needs no special case.
- Widening is one function in `crates/core/src/shape.rs` plus kernels and tests. The rejection message names the limitation.
- Alternatives considered: general broadcasting now (more kernel and test surface than any current use needs).
