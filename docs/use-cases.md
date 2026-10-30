# What Oxfuse is for

Oxfuse runs small float32 graphs and gives you the same bits on every platform it is tested on, with the evidence to back that up. That is a narrow promise, and it decides where Oxfuse helps AI work and where it does not. It is experimental software. It is not a production inference engine, and nothing here claims otherwise.

Every statement below points at a command you can run or a row in [claims.md](claims.md).

## Research

- **Reproducibility studies.** The reference and the fast CPU backend return the same bits on Linux and Windows (x86_64, arm64) and macOS (arm64), at any SIMD level and thread count. If two runs differ, the difference is not Oxfuse's arithmetic. `oxfuse run --manifest run.json` records the version, backend, SIMD level and SHA-256 of the model, inputs and each output; `oxfuse compare a.json b.json` reports the bit differences and whether the contract permits them ([manifest.md](manifest.md)).
- **An oracle for another runtime.** The [golden corpus](golden-corpus.md) is a set of graphs with the exact output bits Oxfuse returns. Another project can run it in its CI to check its own `MatMul`, `Sigmoid` and layout ops against one fixed definition, and `compare` says whether a difference is a bug or an allowed variation.
- **Numerical behaviour of a GPU.** `oxfuse gpu-selftest` reports FMA use, subnormal flushing, NaN behaviour and ulp error on any wgpu adapter. Results so far are in [adapters.md](adapters.md); a report for a card we have not seen is welcome.
- **Studying summation order.** `MatMul` is one fused multiply-add per step in ascending `k`, and `Sigmoid` uses an `exp` written for this project and hash-checked over all 2^32 inputs ([semantics.md](semantics.md)). Order and rounding are defined, not left to a library, so you can reason about the error of a graph exactly. A tool that bounds the error of a whole graph is on the [roadmap](../ROADMAP.md) and does not exist yet.
- **Teaching and review.** The IR, the verifier and the four exact passes are small enough to read in an afternoon ([architecture.md](architecture.md), [ir.md](ir.md)).

## Production, in a narrow sense

- **Regression and audit tests.** Keep golden outputs for a small model in the repository and fail a build when any bit moves: run `--backend reference` once, then `cmp` against `--backend cpu` or against the output of a new release.
- **A check on other stacks.** Run the same ONNX model on Oxfuse and on another runtime, then `compare` the outputs. Oxfuse reads float32 models built from a small operator subset ([onnx.md](onnx.md), 14 fixtures checked against onnxruntime and the `onnx` reference evaluator).
- **Small exact models where the bits matter more than the speed.** On the recorded machine the fast backend is about as fast as `ndarray` for small products and ahead of it at batch 1 ([benchmarks.md](benchmarks.md)). That is not a claim about your hardware, and large models need a real inference engine.

## What it is not

Not an inference server, not for training, not for large models, and not for speed-critical CPU work. The operator set is seven float32 ops plus a small ONNX subset, with no broadcasting beyond a scalar operand and no batched `MatMul`. For those jobs use ONNX Runtime (through `ort`), `tract`, `candle` or `burn`; [comparison.md](comparison.md) quotes what each says about itself.

## What would have to change

Broadcasting, batched `MatMul`, more operators, and a stable API (the crates are at 0.5, and a minor version can break). The order and the evidence for each step are in the [roadmap](../ROADMAP.md); there are no dates.

## Try it

```sh
cargo install oxfuse
git clone https://github.com/sara-star-quant/oxfuse && cd oxfuse
oxfuse run examples/tiny_mlp.json --inputs examples/tiny_mlp.inputs.json --backend reference --manifest ref.json > ref_out.json
oxfuse run examples/tiny_mlp.json --inputs examples/tiny_mlp.inputs.json --backend cpu --manifest cpu.json > cpu_out.json
oxfuse compare ref_out.json cpu_out.json
```

The last command prints `verdict: identical`.
