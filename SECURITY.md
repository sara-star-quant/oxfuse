# Security

Oxfuse reads graph files, ONNX models and input files, so the parsing and resource-limit code is the part that matters.

## What is in scope

- Crashes, panics, hangs or unbounded memory use caused by a crafted graph file, ONNX model or inputs file. Files are untrusted: sizes, value counts, tensor ranks and element counts are bounded before anything is allocated (see the limits in `crates/core/src/limits.rs`).
- Any case where an optimization pass changes the result of a graph.
- Incorrect results from an operation compared with its documented semantics.

The code of every oxfuse crate is `forbid(unsafe_code)`, makes no network calls and loads no native code from a model. Its dependencies are another matter: `wgpu` and `fearless_simd` (the SIMD kernels) use `unsafe` inside, and their advisories count as ours.

## What is not

This is an experimental project and not meant for safety-critical or production use. A graph that is merely slow or heavy within the documented limits is not a vulnerability; a crafted file that hangs the loader or gets past the limits is.

## Reporting

Please use GitHub's private vulnerability reporting on this repository ("Security" tab, "Report a vulnerability") instead of a public issue. Reports are handled on a best-effort basis; there is no service-level agreement. Fixes go into the latest release and `main`; older releases are not patched.

Report at <https://github.com/sara-star-quant/oxfuse/security/advisories/new>. Past advisories and the list of supported versions live at <https://github.com/sara-star-quant/oxfuse/security>. Dependency advisories are checked in CI with `cargo deny` (see [CONTRIBUTING.md](CONTRIBUTING.md)). A first reply usually comes within a week.
