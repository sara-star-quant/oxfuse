# 0013: ONNX import is a separate crate behind a CLI feature

Date: 2026-10-21. Status: accepted.

## Context

Oxfuse only ran graphs written for it. A model produced elsewhere is the most direct test of whether the compiler is useful, and an independent runtime is the strongest oracle available for it.

## Decision

- A new crate `oxfuse-onnx` decodes a model and lowers a float32 subset into the existing IR. The CLI reaches it through the off-by-default feature `onnx`, the same pattern as the GPU backend, so the default install gains no protobuf dependency.
- The decoder is `prost` derive structs for the part of `onnx.proto` that is read. Unknown fields are skipped by the wire rules. No `protoc` runs at build time.
- Opsets 13 through 21 only. Older opsets carry attribute forms and broadcast rules that would each need tests for no gain; a model can be re-exported.
- Broadcasting stays out of the IR. A constant operand is expanded at import, which covers the bias of a linear layer; a non-constant operand that needs broadcasting is an error. General broadcasting touches every kernel and is a later decision (0006).
- Every unsupported op, attribute, dtype or domain is an error that names the node. Nothing is approximated.
- Fixtures come from a generator that evaluates each model with two independent implementations and a float64 copy, and refuses to write on any disagreement.

## Consequences

- The IR, the passes and the three backends are unchanged.
- Expanding a constant costs memory proportional to the output; the expansion is checked against the limits first.
- Python (3.14 at the time of writing, because onnxruntime has no wheel for newer versions) is needed only to regenerate or check fixtures, never to build or test.
- Alternatives considered: a decoder generated from `onnx.proto` with `protoc` (a build dependency for every user), accepting old opsets (more variants than evidence), and adding NumPy broadcasting to the IR now (more kernel and test surface than any import needs).
