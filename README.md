# Oxfuse

[![ci](https://github.com/sara-star-quant/oxfuse/actions/workflows/ci.yml/badge.svg)](https://github.com/sara-star-quant/oxfuse/actions/workflows/ci.yml)
[![crates.io](https://img.shields.io/crates/v/oxfuse.svg)](https://crates.io/crates/oxfuse)
[![docs.rs](https://img.shields.io/docsrs/oxfuse-core)](https://docs.rs/oxfuse-core)
[![MSRV](https://img.shields.io/crates/msrv/oxfuse)](https://crates.io/crates/oxfuse)
[![license](https://img.shields.io/crates/l/oxfuse.svg)](LICENSE)

A tensor-graph compiler in Rust. Define a tensor graph once (JSON, or an ONNX model), type-check it, optimize it without changing results, and run it on the reference CPU backend, the fast CPU backend or, optionally, a GPU through wgpu (Vulkan, Metal, DX12). Experimental, not for production use.

## Who it is for

Oxfuse is a small tensor graph compiler whose claims come with evidence. It fits when you want:

- **Exact, reproducible results** for small float32 graphs and ONNX models: the CPU backends return the same bits on every machine and thread count.
- **An oracle for other runtimes**: golden outputs for tests, or a check that another stack's result is right.
- **A GPU numerical probe**: `oxfuse gpu-selftest` reports FMA use, subnormal flushing, NaN behavior and ulp error on any wgpu adapter.
- **A readable reference** for how a typed IR (intermediate representation: the graph as typed values and operations), a verifier and exact optimization passes fit together.

Practical examples:

```sh
# Golden outputs for a test suite: the fast backend must match the reference byte for byte.
oxfuse run examples/tiny_mlp.json --inputs examples/tiny_mlp.inputs.json --backend reference > golden.json
oxfuse run examples/tiny_mlp.json --inputs examples/tiny_mlp.inputs.json --backend cpu > out.json
cmp golden.json out.json

# Check a small ONNX model against another runtime: compare its numbers with Oxfuse's exact ones.
oxfuse run model.onnx --inputs inputs.json --backend reference     # needs --features onnx

# Decide whether a GPU can be trusted for your numerics before you rely on it.
oxfuse gpu-selftest                                                 # needs --features wgpu

# See what each optimization pass changes in a graph.
oxfuse inspect examples/tiny_mlp.json
```

It is not for production inference, large models, training, or speed-critical CPU work. For those, use ONNX Runtime (through `ort`), `tract`, `candle` or `burn`. The operator set is seven float32 ops plus a small ONNX subset, with no broadcasting beyond a scalar operand.

## What it does

- **Compiles** a JSON graph to the IR, which the passes rewrite, with shape inference and a verifier that re-checks every invariant after each pass.
- **Optimizes** with four exact passes: constant folding, layout elision (transpose and reshape round trips), common-subexpression elimination, dead-code elimination. Output bits never change.
- **Runs** on two CPU backends and an optional GPU backend (`--features wgpu`). The fast CPU backend returns the same bits as the reference backend and is the default.
- **Imports** float32 ONNX models made of a small operator subset (`--features onnx`), checked against two independent ONNX implementations ([docs/onnx.md](docs/onnx.md)).
- **Rejects** malformed graphs with a stable error code and the JSON path of the offender.

## Evidence

| Property | How it is established |
|---|---|
| Correct against an independent implementation | NumPy float64 oracle, seeded random corpus plus all examples |
| Fast backend equals reference, bit for bit | Contract tests, random-graph tests, differential fuzzing |
| Optimizations are exact, including NaN, infinity, signed zero | Equivalence tests with and without passes |
| Tests detect real defects | cargo-mutants: 1326 of 1337 viable mutants stopped, the other 11 explained one by one ([docs/mutation-testing.md](docs/mutation-testing.md)) |
| Hostile input is handled | Loader fuzzing, bounded allocation, 3.07M generated graphs compared across backends with no divergence |
| GPU backend: exact where IEEE allows, bounded elsewhere | Relu bit-exact on all 2^32 inputs; MatMul inside the float64 bound; Sigmoid inside a measured ulp bound; a CPU model of every kernel grid; Kani bounded model checking of the dispatch fold and the matmul, transpose and permute index arithmetic (bounded sizes, not a proof of the whole backend) |
| ONNX models run correctly | 14 fixtures checked against onnxruntime and the `onnx` reference evaluator, bit-exact except one `Sigmoid` model; importer fuzzing ([docs/onnx.md](docs/onnx.md)) |
| Builds on Rust 1.88 or newer | MSRV job in CI |

Every claim maps to a row and a reproduction command in [docs/claims.md](docs/claims.md). Measured speeds, with machine and commands, are in [docs/benchmarks.md](docs/benchmarks.md). On a 512x512 matmul the single-threaded fast backend runs at 0.49x of `ndarray`'s speed on the recorded machine.

## Scope

| Op | Inputs | Attributes | Rule |
|---|---|---|---|
| `Relu` | 1 | none | `x > 0 ? x : 0`, NaN propagates |
| `Sigmoid` | 1 | none | `1 / (1 + exp(-x))` in `f32` |
| `Add` | 2 | none | equal shapes, or one scalar operand |
| `Mul` | 2 | none | equal shapes, or one scalar operand |
| `MatMul` | 2 | none | rank 2: `[M, K] x [K, N]` |
| `Reshape` | 1 | `shape` | element count must match |
| `Transpose` | 1 | `perm` | output dim `i` is input dim `perm[i]` |

`f32` only, row-major. CPU by default; an optional GPU backend (`--features wgpu`, `--backend wgpu`) is measured on the devices in [docs/adapters.md](docs/adapters.md), with no support claimed for any device.

Not supported: broadcasting beyond a scalar operand, batched `MatMul`, `-1` in `Reshape`, NaN or infinity in JSON files, a GPU in the default build. No device or vendor support is claimed or implied.

## GPUs

The GPU backend runs on wgpu. It is measured on an Apple M1 Pro (Metal), a Tesla T4 (NVIDIA, Vulkan, a free notebook) and on software adapters, see [docs/adapters.md](docs/adapters.md). AMD and Intel Arc cards should work through Vulkan or DX12 but are untested. If you try your card, run `oxfuse gpu-selftest` and share the report in an issue.

## Install

```sh
cargo install oxfuse                    # CPU backends
cargo install oxfuse --features wgpu    # adds the GPU backend
cargo install oxfuse --features onnx    # adds ONNX model import
```

As a library, depend on `oxfuse-core` and `oxfuse-runtime`, plus `oxfuse-wgpu` for the GPU backend and `oxfuse-onnx` for ONNX import.

## A first run

A graph is named inputs, constants and nodes. This one is `relu(x @ w + b)` (`examples/tiny_mlp.json`):

```json
{
  "format_version": 1,
  "inputs": [{ "name": "x", "dtype": "f32", "shape": [2, 3] }],
  "constants": [
    { "name": "w", "dtype": "f32", "shape": [3, 2], "values": [1.0, 0.0, 0.0, 1.0, 1.0, 1.0] },
    { "name": "b", "dtype": "f32", "shape": [2, 2], "values": [-5.0, 0.0, 2.0, -4.0] }
  ],
  "nodes": [
    { "name": "projection", "op": "MatMul", "inputs": ["x", "w"] },
    { "name": "biased", "op": "Add", "inputs": ["projection", "b"] },
    { "name": "out", "op": "Relu", "inputs": ["biased"] }
  ],
  "outputs": ["out"]
}
```

    $ oxfuse run examples/tiny_mlp.json --inputs examples/tiny_mlp.inputs.json
    {"outputs":[{"name":"out","shape":[2,2],"values":[0.0,1.0,1.0,0.0]}]}

    $ cargo run -q -p oxfuse -- inspect examples/tiny_mlp.json    # typed IR and pass effects
    $ cargo run -q -p oxfuse -- run  examples/relu.json --inputs examples/relu.inputs.json --backend reference --no-opt

As a library, there is a runnable example for each backend family:

    $ cargo run --release -p oxfuse-runtime --example optimize_and_run   # passes, then reference and fast CPU backends
    $ cargo run --release -p oxfuse-wgpu --example matmul                # GPU, with the bound check and device-resident run

Errors carry a code and a location: ``error[E017]: node `y`: MatMul: inner dimensions differ: [2, 3] x [4, 5] (at nodes[0])``.

Full gate: `just ci`. Oracle check: `just oracle-check` (needs `numpy`, see `scripts/requirements.txt`).

## Documentation

[Architecture](docs/architecture.md) | [IR](docs/ir.md) | [Numerical semantics](docs/semantics.md) | [ONNX import](docs/onnx.md) | [Optimization passes](docs/optimization-passes.md) | [Adding an op](docs/adding-an-op.md) | [Roadmap](ROADMAP.md) | [Contributing](CONTRIBUTING.md) | [Security](SECURITY.md)

Apache-2.0. See [LICENSE](LICENSE).
