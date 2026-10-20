# 0003: `Op` is an enum with exhaustive matches

Date: 2026-10-20. Status: accepted.

## Context

Ops need shape rules, kernels, a loader mapping and documentation. A trait-object or registry design lets a new op skip one of those silently.

## Decision

`Op` is a plain enum and every consumer matches on it exhaustively, with no wildcard arm.

## Consequences

- Adding a variant breaks the build at every place that needs an arm, which is the checklist in `docs/adding-an-op.md`.
- Dispatch is a match, with no virtual call.
- Ops are not pluggable from outside the crate. That is acceptable: the op set is small and every op needs reference, fast and oracle implementations anyway.
- Alternatives considered: a trait per op (flexible, but nothing lists what a new op forgot).
