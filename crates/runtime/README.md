# oxfuse-runtime

Part of [oxfuse](https://github.com/sara-star-quant/oxfuse), a small tensor graph compiler in Rust. Start with the [repository README](https://github.com/sara-star-quant/oxfuse#readme) if you landed here from a search.

Tensors, execution plans and the CPU backends for [oxfuse](https://github.com/sara-star-quant/oxfuse) graphs: a reference backend, and a fast backend that returns the same bits on the platforms it is tested on. The fast matmul and `Sigmoid` run on SIMD vectors picked at run time (AVX2 or AVX-512 on x86, NEON on ARM) through `fearless_simd`, which sets the minimum Rust to 1.89. The optional `parallel` feature threads the fast matmul without changing results.

An example that builds a graph, runs the passes and compares both backends is `examples/optimize_and_run.rs`:

```sh
cargo run --release -p oxfuse-runtime --example optimize_and_run
```

Experimental. The optional GPU backend is the `oxfuse-wgpu` crate. Measurements are in the repository's `docs/benchmarks.md`.

Licensed under Apache-2.0.
