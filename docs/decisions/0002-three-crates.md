# 0002: three crates, with the optimizer inside core

Date: 2026-10-20. Status: accepted.

## Context

Oxfuse needs the IR and passes, the backends, and the command-line front end. Splitting further (an optimizer crate, a format crate, one crate per backend) adds manifests and cross-crate plumbing without adding a boundary that anything depends on.

## Decision

Three crates: `oxfuse-core` (IR, shape inference, verifier, limits, errors, passes), `oxfuse-runtime` (tensors, plan, backends) and `oxfuse` in `crates/cli` (JSON format, loader, commands, binary). The optimizer is a module of core.

## Consequences

- The dependency direction is enforced by the compiler: core cannot reach runtime or the CLI. Constant folding therefore takes a `ConstEvaluator` from the caller instead of calling a backend.
- A backend can be reused without the CLI, and the CLI can change its file format without touching the IR.
- Alternatives considered: five crates (more ceremony, no extra boundary) and one crate (the JSON format would leak into the IR).
