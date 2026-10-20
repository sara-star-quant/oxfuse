# 0004: passes rebuild the graph

Date: 2026-10-20. Status: accepted.

## Context

A pass could edit a graph in place or return a new one. In-place edits leave stale value ids and make before and after states hard to compare.

## Decision

A pass is `fn run(&self, graph: &Graph) -> Result<Graph, Error>`. It reads one graph and builds another through the builder.

## Consequences

- Before and after dumps are two graphs that both exist, and equivalence tests run the same inputs through both.
- No pass can leave a dangling id, because ids are assigned by the builder. The pass manager verifies each result and rejects a pass that changes the interface.
- Each pass costs a rebuild. Pass time has not been measured, so nothing here claims it is small.
- Alternatives considered: in-place mutation with a worklist (faster, harder to verify).
