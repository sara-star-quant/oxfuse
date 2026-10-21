# oxfuse

Command-line tool for [oxfuse](https://github.com/sara-star-quant/oxfuse), a small tensor graph compiler in Rust.

```sh
cargo install oxfuse                  # add --features onnx,wgpu for ONNX import and the GPU backend
oxfuse check graph.json
oxfuse run graph.json --inputs inputs.json [--backend reference|cpu|wgpu]
oxfuse run model.onnx --inputs inputs.json    # needs --features onnx
oxfuse inspect graph.json
oxfuse gpu-selftest                   # needs --features wgpu
```

Experimental. The `wgpu` backend and `oxfuse gpu-selftest` need the `wgpu` feature. See the repository for the graph format, examples and benchmarks.

Licensed under Apache-2.0.
