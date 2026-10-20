# Security

Oxfuse reads graph and input files, so the parsing and resource-limit code is the part that matters.

## What is in scope

- Crashes, panics, hangs or unbounded memory use caused by a crafted graph file or inputs file. Files are untrusted: sizes, value counts, tensor ranks and element counts are bounded before anything is allocated (see the limits in `crates/core/src/limits.rs`).
- Any case where an optimization pass changes the result of a graph.
- Incorrect results from an operation compared with its documented semantics.

The code is `forbid(unsafe_code)`, makes no network calls and loads no native code from a model.

## What is not

This is an experimental project and not meant for safety-critical or production use. Performance is not a security property here.

## Reporting

Please use GitHub's private vulnerability reporting on this repository ("Security" tab, "Report a vulnerability") instead of a public issue. Reports are handled on a best-effort basis; there is no service-level agreement. Only the latest commit on `main` is supported.
