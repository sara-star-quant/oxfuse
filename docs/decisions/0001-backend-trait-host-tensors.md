# 0001: The backend trait assumes host tensors

Date: 2026-10-20. Status: accepted.

## Context

`Backend::compile` takes a verified graph and returns an `Executable` whose `run` takes `&[Tensor]` and returns `Vec<Tensor>`. `Tensor` owns a contiguous `Vec<f32>` in host memory. Both the reference and the CPU backend fit this shape.

## Decision

Keep the trait this small. It is enough for any backend that computes on host memory, and a second implementation (the CPU backend) has validated it.

## Consequences

- A backend with device memory (a GPU or another accelerator) would have to copy inputs in and outputs out on every `run`, which is the wrong cost model for graphs that stay on the device. Supporting one means revising the trait: a device tensor type, explicit transfers, and compile-time placement.
- That revision belongs to the accelerator stage in `ROADMAP.md`, to be designed against a concrete target. Nothing in the current code assumes it will happen.
- Alternatives considered: making `Tensor` an enum of host and device storage now (speculative, nothing to test against) and a second trait for device backends (premature).
