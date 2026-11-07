# oxfuse-onnx

Part of [oxfuse](https://github.com/sara-star-quant/oxfuse), a small tensor graph compiler in Rust. Start with the [repository README](https://github.com/sara-star-quant/oxfuse#readme) if you landed here from a search.

ONNX import for [oxfuse](https://github.com/sara-star-quant/oxfuse): reads a float32 model made of a small operator subset (`MatMul`, `Gemm`, `Add`, `Mul`, `Sub`, `Div`, `Neg`, `Relu`, `Sigmoid`, `Reshape`, `Flatten`, `Transpose`, `Identity`, `Dropout`, `Constant`; opsets 13 to 21) and lowers it into the oxfuse IR. Anything else fails with an error that names the node and the reason.

**For AI work:** bring a small float32 ONNX model in and check another runtime's results against Oxfuse's exact ones. Not for large models, training or production inference ([use cases](https://github.com/sara-star-quant/oxfuse/blob/main/docs/use-cases.md)).

```rust
// cargo add oxfuse-onnx   (depends only on oxfuse-core; add oxfuse-runtime to run the graph)
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use oxfuse_onnx::proto::*;
    use prost::Message;

    // A one-node model, Relu on a [1, 2] input, built in memory.
    let dim = |v| Dimension { value: Some(dimension::Value::DimValue(v)) };
    let model = ModelProto {
        ir_version: Some(9),
        opset_import: vec![OperatorSetIdProto { domain: String::new(), version: Some(13) }],
        graph: Some(GraphProto {
            node: vec![NodeProto {
                op_type: "Relu".into(),
                input: vec!["x".into()],
                output: vec!["y".into()],
                ..NodeProto::default()
            }],
            input: vec![ValueInfoProto {
                name: "x".into(),
                r#type: Some(TypeProto {
                    tensor_type: Some(TypeTensor {
                        elem_type: data_type::FLOAT,
                        shape: Some(TensorShapeProto { dim: vec![dim(1), dim(2)] }),
                    }),
                }),
            }],
            output: vec![ValueInfoProto { name: "y".into(), r#type: None }],
            ..GraphProto::default()
        }),
    };

    // Import needs only `oxfuse-core` types; no backend is involved.
    let graph = oxfuse_onnx::import(&model.encode_to_vec(), &oxfuse_onnx::ImportOptions::default())?;
    assert_eq!(graph.inputs().len(), 1);
    assert_eq!(graph.outputs().len(), 1);
    Ok(())
}
```

From a shell, `cargo install oxfuse --features onnx` and `oxfuse run model.onnx --inputs inputs.json` do the same.

The command-line tool builds it with `--features onnx` (`oxfuse run model.onnx`). The subset, limits and error codes are in `docs/onnx.md` in the repository; fixtures are checked against onnxruntime and the `onnx` reference evaluator.

Experimental. Licensed under Apache-2.0.
