# Oxfuse

[![ci](https://github.com/sara-star-quant/oxfuse/actions/workflows/ci.yml/badge.svg)](https://github.com/sara-star-quant/oxfuse/actions/workflows/ci.yml)
[![crates.io](https://img.shields.io/crates/v/oxfuse.svg)](https://crates.io/crates/oxfuse)
[![docs.rs](https://img.shields.io/docsrs/oxfuse-core)](https://docs.rs/oxfuse-core)
[![MSRV](https://img.shields.io/crates/msrv/oxfuse)](https://crates.io/crates/oxfuse)
[![license](https://img.shields.io/crates/l/oxfuse.svg)](LICENSE)

A tensor-graph compiler in Rust whose CPU results are the same bits on every platform it is tested on (Linux and Windows on arm64 and x86_64, macOS on arm64). Define a small float32 graph once (JSON, or an ONNX model), type-check it, optimize it without changing results, and run it on a reference CPU backend, a fast SIMD CPU backend or, optionally, a GPU through wgpu (Vulkan, Metal, DX12). Every claim has a test and a row in [docs/claims.md](docs/claims.md). Experimental, not for production use.

## Quick start

```sh
cargo install oxfuse                  # CPU backends; add --features wgpu and/or onnx
git clone https://github.com/sara-star-quant/oxfuse && cd oxfuse
oxfuse run examples/tiny_mlp.json --inputs examples/tiny_mlp.inputs.json --backend cpu
```

No Rust toolchain? Each [release](https://github.com/sara-star-quant/oxfuse/releases) has prebuilt binaries (ONNX and GPU on) for the five platforms below, with `SHA256SUMS`. Check one with `shasum -c SHA256SUMS` and `gh attestation verify FILE --repo sara-star-quant/oxfuse`.

As a library, depend on `oxfuse-core` and `oxfuse-runtime`, plus `oxfuse-wgpu` for the GPU backend and `oxfuse-onnx` for ONNX import. [A first run](#a-first-run) below walks through the graph format.

## Platforms

Every pull request runs the full test suite, the threaded tests and the sweep that hashes all 2^32 `exp` and `Sigmoid` inputs on five platforms, and the hash must come out the same on all of them:

| OS | Architecture | CI runner |
|---|---|---|
| Linux (Ubuntu) | x86_64 | `ubuntu-latest` |
| Linux (Ubuntu) | arm64 | `ubuntu-24.04-arm` |
| macOS | arm64 (Apple silicon) | `macos-latest` |
| Windows | x86_64 | `windows-latest` |
| Windows | arm64 | `windows-11-arm` |

It needs Rust 1.89 or newer. Not tested: macOS on Intel, 32-bit targets, big-endian targets, WebAssembly. The optional GPU backend is measured on the adapters in [docs/adapters.md](docs/adapters.md), not on every CI platform.

## The contract

- **Same bits on the tested platforms.** `MatMul` is one fused multiply-add per step in ascending `k`, and `Sigmoid` uses an `exp` written for this project. The reference and the fast CPU backend agree bit for bit on x86 and ARM, at any SIMD level and thread count. A NaN's sign and payload are the one thing IEEE 754 leaves to the hardware.
- **Exact optimizer.** The four passes never change an output bit, infinity and signed zero included.
- **A GPU is measured, not trusted.** `MatMul` matches the reference bit for bit on the two devices that fuse the same way (Apple M1 Pro, Tesla T4); every other op has a documented bound, and `oxfuse gpu-selftest` measures any adapter.
- **Evidence over adjectives.** Mutation testing, an exhaustive hash over all 2^32 `Sigmoid` inputs on five platforms, differential fuzzing and a NumPy oracle back these, see [Evidence](#evidence). The precise rules are in [docs/semantics.md](docs/semantics.md).

## Who it is for

Oxfuse is a small tensor graph compiler whose claims come with evidence. It fits when you want:

- **Exact, reproducible results** for small float32 graphs and ONNX models: the same bits on the tested machines, platforms and thread counts.
- **An oracle for other runtimes**: golden outputs for tests, or a check that another stack's result is right.
- **A GPU numerical probe**: `oxfuse gpu-selftest` reports FMA use, subnormal flushing, NaN behavior and ulp error on any wgpu adapter.
- **A readable reference** for how a typed IR (intermediate representation: the graph as typed values and operations), a verifier and exact optimization passes fit together.

Practical examples:

```sh
# Golden outputs for a test suite: the fast backend must match the reference byte for byte.
oxfuse run examples/tiny_mlp.json --inputs examples/tiny_mlp.inputs.json --backend reference > golden.json
oxfuse run examples/tiny_mlp.json --inputs examples/tiny_mlp.inputs.json --backend cpu > out.json
cmp golden.json out.json
oxfuse run examples/tiny_mlp.json --inputs examples/tiny_mlp.inputs.json --manifest run.json > out.json   # also writes a record of the run
oxfuse compare golden.json out.json                                  # bit differences and what they mean (see docs/manifest.md)

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
| `MatMul` and `Sigmoid` return the same bits on the tested platforms | One fused multiply-add per matmul step, an `exp` of our own, and a hash of all 2^32 `exp` and `Sigmoid` outputs that macOS, Linux and Windows CI each reproduce |
| Fast backend equals reference, bit for bit (all NaNs count as one value) | Contract tests, random-graph tests, differential fuzzing on arm64 and x86_64 |
| Optimizations are exact, including infinity, signed zero and where NaNs appear | Equivalence tests with and without passes |
| Tests detect real defects | cargo-mutants: all 1161 viable mutants of the CPU-side crates stopped, no survivor, reproduced by parallel CI shards; the GPU crate is mutation-tested locally on a real adapter ([docs/mutation-testing.md](docs/mutation-testing.md)) |
| Hostile input is handled | Loader fuzzing, bounded allocation, 3.07M generated graphs compared across backends with no divergence |
| GPU backend: exact where IEEE allows, bounded elsewhere | Relu bit-exact on all 2^32 inputs; MatMul bit-identical to the reference on an Apple M1 Pro and a Tesla T4 (inside the float64 bound on any adapter); Sigmoid inside a measured ulp bound; a CPU model of every kernel grid; Kani bounded model checking of the dispatch fold and the matmul, transpose and permute index arithmetic (bounded sizes, not a proof of the whole backend) |
| ONNX models run correctly | 14 fixtures checked against onnxruntime and the `onnx` reference evaluator, bit-exact except one `Sigmoid` model; importer fuzzing ([docs/onnx.md](docs/onnx.md)) |
| Builds on Rust 1.89 or newer | MSRV job in CI |

Every claim maps to a row and a reproduction command in [docs/claims.md](docs/claims.md). Measured speeds, with machine and commands, are in [docs/benchmarks.md](docs/benchmarks.md). On a 512x512 matmul the single-threaded fast backend runs at 0.80x of `ndarray`'s speed (3.7x with threads), and one row against a 784x10 weight runs at 3.3x `ndarray`'s speed. A 784-256-10 classifier runs at 3.7x `ndarray`'s speed at batch 1, 1.03x at batch 8 and 0.79x at batch 32 (1.1x with threads), all on the recorded machine.

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

[Architecture](docs/architecture.md) | [IR](docs/ir.md) | [Numerical semantics](docs/semantics.md) | [Numeric contract matrix](docs/contract.json) | [Provenance](docs/provenance.md) | [Comparison](docs/comparison.md) | [Run record](docs/manifest.md) | [Golden corpus](docs/golden-corpus.md) | [Use cases](docs/use-cases.md) | [ONNX import](docs/onnx.md) | [Optimization passes](docs/optimization-passes.md) | [Adding an op](docs/adding-an-op.md) | [Roadmap](ROADMAP.md) | [Contributing](CONTRIBUTING.md) | [Security](SECURITY.md)

Copyright 2026 SARA STAR QUANT LLC. Apache-2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
