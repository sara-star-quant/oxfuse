# oxfuse

Command-line tool for [oxfuse](https://github.com/sara-star-quant/oxfuse), a small tensor graph compiler in Rust.

```sh
cargo install oxfuse                  # add --features onnx,wgpu for ONNX import and the GPU backend
oxfuse check graph.json
oxfuse run graph.json --inputs inputs.json [--backend reference|cpu|wgpu] [--manifest run.json] [--max-macs N]
oxfuse run model.onnx --inputs inputs.json    # needs --features onnx
oxfuse inspect graph.json
oxfuse compare a.json b.json          # bit differences between two outputs files
oxfuse analyze graph.json --inputs inputs.json   # each output with a bound on its distance from the exact result
oxfuse gpu-selftest                   # needs --features wgpu
```

**For AI work:** exact, repeatable float32 results, golden outputs and a corpus for checking other runtimes, run records with `compare`, and a GPU numerics probe. Not for large models, training or production inference ([use cases](https://github.com/sara-star-quant/oxfuse/blob/main/docs/use-cases.md)).

Try it with no clone (this is a complete run; the graph is `relu(x @ w)`):

```sh
cargo install oxfuse
cat > graph.json <<'EOF'
{"format_version":1,"inputs":[{"name":"x","dtype":"f32","shape":[2,3]}],"constants":[{"name":"w","dtype":"f32","shape":[3,2],"values":[0.5,-1.0,1.5,2.0,-0.5,1.0]}],"nodes":[{"name":"m","op":"MatMul","inputs":["x","w"]},{"name":"y","op":"Relu","inputs":["m"]}],"outputs":["y"]}
EOF
echo '{"x":{"shape":[2,3],"values":[1.0,-2.0,3.0,4.0,5.0,-6.0]}}' > inputs.json
oxfuse check graph.json
oxfuse run graph.json --inputs inputs.json --backend reference --manifest ref.json > ref.out
oxfuse run graph.json --inputs inputs.json --backend cpu --manifest cpu.json > cpu.out
oxfuse compare ref.out cpu.out --manifest-a ref.json --manifest-b cpu.json   # verdict: identical
```

`cpu.out` holds `{"outputs":[{"name":"y","shape":[2,2],"values":[0.0,0.0,12.5,0.0]}]}`.

Experimental. The `wgpu` backend and `oxfuse gpu-selftest` need the `wgpu` feature. See the repository for the graph format, examples and benchmarks.

Licensed under Apache-2.0.
