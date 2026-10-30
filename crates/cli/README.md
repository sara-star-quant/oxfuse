# oxfuse

Command-line tool for [oxfuse](https://github.com/sara-star-quant/oxfuse), a small tensor graph compiler in Rust.

```sh
cargo install oxfuse                  # add --features onnx,wgpu for ONNX import and the GPU backend
oxfuse check graph.json
oxfuse run graph.json --inputs inputs.json [--backend reference|cpu|wgpu] [--manifest run.json] [--max-macs N]
oxfuse run model.onnx --inputs inputs.json    # needs --features onnx
oxfuse inspect graph.json
oxfuse compare a.json b.json          # bit differences between two outputs files
oxfuse gpu-selftest                   # needs --features wgpu
```

What it is for: exact, repeatable float32 results (the same bits on Linux, Windows and macOS CI), golden outputs and a corpus for checking other runtimes, run records and `compare`, and a GPU numerics probe. What it is not for: large models, training or production inference. See [use cases](https://github.com/sara-star-quant/oxfuse/blob/main/docs/use-cases.md).

Experimental. The `wgpu` backend and `oxfuse gpu-selftest` need the `wgpu` feature. See the repository for the graph format, examples and benchmarks.

Licensed under Apache-2.0.
