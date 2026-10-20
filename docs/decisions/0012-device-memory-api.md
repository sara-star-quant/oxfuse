# 0012: a device memory API on the wgpu executable

Date: 2026-10-20. Status: accepted.

## Context

Decision 0001 kept the backend trait to host tensors and said an accelerator backend would have to revise it against a concrete target. The wgpu backend is that target.

## Decision

`Backend` and `Executable` stay unchanged, so the CLI and every shared test work with the wgpu backend as they do with the CPU ones. `WgpuExecutable` adds an inherent device API: `upload` (host tensor to `DeviceTensor`), `run_device` (device tensors in and out, intermediates never leave the device) and `download`. Host `run` is upload, `run_device`, download.

- No `DeviceBackend` trait yet. With one device backend a trait has no second implementation to validate it and no caller that is generic over it. Promote the methods to a trait when a second device backend exists.
- Shapes are static, so `compile` creates every buffer, pipeline, parameter block and dispatch size once. `run_device` only binds, encodes and submits.
- Intermediate buffers belong to the executable and are reused, so concurrent runs on one executable are serialized by a mutex. Returned tensors are copied into fresh buffers, so a result stays valid after the next run.
- Reshape shares its input's buffer; no kernel runs.
- A device validation error or device loss is returned as `Error::Backend`, never a panic, and waiting for the device times out after 30 seconds.
- No adapter is an error, never a fallback to the CPU.

## Consequences

- Every `run_device` copies its outputs once. A graph that keeps results on the device across many runs pays that copy per run; avoiding it needs output buffers owned by the caller, which no current use needs.
- Tensor size is bounded by the device's storage-buffer limit and by 32-bit shader indices (2^30 elements); `compile` rejects larger tensors with an error that names the limit.
- Alternatives considered: a `DeviceBackend` trait now (nothing to generalize over), and an enum of host and device storage in `Tensor` (a change to every backend for the benefit of one).
