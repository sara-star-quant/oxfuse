# oxfuse-onnx

Part of [oxfuse](https://github.com/sara-star-quant/oxfuse), a small tensor graph compiler in Rust. Start with the [repository README](https://github.com/sara-star-quant/oxfuse#readme) if you landed here from a search.

ONNX import for [oxfuse](https://github.com/sara-star-quant/oxfuse): reads a float32 model made of a small operator subset (`MatMul`, `Gemm`, `Add`, `Mul`, `Relu`, `Sigmoid`, `Reshape`, `Flatten`, `Transpose`, `Identity`, `Dropout`, `Constant`; opsets 13 to 21) and lowers it into the oxfuse IR. Anything else fails with an error that names the node and the reason.

```rust,ignore
let graph = oxfuse_onnx::import(&bytes, &oxfuse_onnx::ImportOptions::default())?;
```

The command-line tool builds it with `--features onnx` (`oxfuse run model.onnx`). The subset, limits and error codes are in `docs/onnx.md` in the repository; fixtures are checked against onnxruntime and the `onnx` reference evaluator.

Experimental. Licensed under Apache-2.0.
