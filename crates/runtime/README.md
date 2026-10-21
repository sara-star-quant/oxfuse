# oxfuse-runtime

Tensors, execution plans and the CPU backends for [oxfuse](https://github.com/sara-star-quant/oxfuse) graphs: a reference backend, and a fast backend that returns the same bits. The optional `parallel` feature threads the fast matmul without changing results.

An example that builds a graph, runs the passes and compares both backends is `examples/optimize_and_run.rs`:

    cargo run --release -p oxfuse-runtime --example optimize_and_run

Experimental. The optional GPU backend is the `oxfuse-wgpu` crate. Measurements are in the repository's `docs/benchmarks.md`.

Licensed under Apache-2.0.
