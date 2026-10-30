# oxfuse-onnx

Part of [oxfuse](https://github.com/sara-star-quant/oxfuse), a small tensor graph compiler in Rust. Start with the [repository README](https://github.com/sara-star-quant/oxfuse#readme) if you landed here from a search.

ONNX import for [oxfuse](https://github.com/sara-star-quant/oxfuse): reads a float32 model made of a small operator subset (`MatMul`, `Gemm`, `Add`, `Mul`, `Relu`, `Sigmoid`, `Reshape`, `Flatten`, `Transpose`, `Identity`, `Dropout`, `Constant`; opsets 13 to 21) and lowers it into the oxfuse IR. Anything else fails with an error that names the node and the reason.

**For AI work:** bring a small float32 ONNX model in and check another runtime's results against Oxfuse's exact ones. Not for large models, training or production inference ([use cases](https://github.com/sara-star-quant/oxfuse/blob/main/docs/use-cases.md)).

```rust,ignore
// cargo add oxfuse-onnx oxfuse-runtime
let bytes = std::fs::read("model.onnx")?;
let graph = oxfuse_onnx::import(&bytes, &oxfuse_onnx::ImportOptions::default())?;
// then compile `graph` with a backend from `oxfuse-runtime`, as in its README
```

From a shell, `cargo install oxfuse --features onnx` and `oxfuse run model.onnx --inputs inputs.json` do the same.

The command-line tool builds it with `--features onnx` (`oxfuse run model.onnx`). The subset, limits and error codes are in `docs/onnx.md` in the repository; fixtures are checked against onnxruntime and the `onnx` reference evaluator.

Experimental. Licensed under Apache-2.0.
