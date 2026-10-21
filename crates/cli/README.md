# oxfuse

Command-line tool for [oxfuse](https://github.com/sara-star-quant/oxfuse), a small tensor graph compiler in Rust.

    oxfuse check graph.json
    oxfuse run graph.json --inputs inputs.json [--backend reference|cpu|wgpu]
    oxfuse inspect graph.json

Experimental. The `wgpu` backend and `oxfuse gpu-selftest` need the `wgpu` feature. See the repository for the graph format, examples and benchmarks.

Licensed under Apache-2.0.
