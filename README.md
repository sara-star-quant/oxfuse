# Oxfuse

**A tensor-graph compiler in Rust whose float32 results are defined to the bit.**

[![ci](https://github.com/sara-star-quant/oxfuse/actions/workflows/ci.yml/badge.svg)](https://github.com/sara-star-quant/oxfuse/actions/workflows/ci.yml)
[![coverage](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/sara-star-quant/oxfuse/badges/coverage.json)](https://github.com/sara-star-quant/oxfuse/actions/workflows/coverage.yml)
[![kani](https://github.com/sara-star-quant/oxfuse/actions/workflows/kani.yml/badge.svg)](https://github.com/sara-star-quant/oxfuse/actions/workflows/kani.yml)
[![mutants](https://github.com/sara-star-quant/oxfuse/actions/workflows/mutants.yml/badge.svg)](docs/mutation-testing.md)
[![crates.io](https://img.shields.io/crates/v/oxfuse.svg)](https://crates.io/crates/oxfuse)
[![docs.rs](https://img.shields.io/docsrs/oxfuse-core)](https://docs.rs/oxfuse-core)
[![MSRV](https://img.shields.io/crates/msrv/oxfuse)](https://crates.io/crates/oxfuse)
[![license](https://img.shields.io/crates/l/oxfuse.svg)](LICENSE)

Define a small float32 graph once (JSON, or an ONNX model), type-check it, optimize it without changing a result bit, and run it on a reference CPU backend, a fast SIMD CPU backend or, optionally, a GPU through wgpu. The CPU results are the same bits on every platform it is tested on (Linux and Windows on arm64 and x86_64, macOS on arm64). Every claim has a test and a row in [docs/claims.md](docs/claims.md).

> **Status: experimental.** Seven float32 operators plus a small ONNX subset. Not for production inference, large models or training. [Who it is for, and who it is not](docs/use-cases.md).

## A first run

```sh
cargo install oxfuse                  # CPU backends; add --features wgpu and/or onnx
git clone https://github.com/sara-star-quant/oxfuse && cd oxfuse
oxfuse run examples/tiny_mlp.json --inputs examples/tiny_mlp.inputs.json --backend cpu
```

No Rust toolchain? Each [release](https://github.com/sara-star-quant/oxfuse/releases) has prebuilt binaries (ONNX and GPU on) for the five platforms below, with `SHA256SUMS`. Check one with `shasum -c SHA256SUMS` and `gh attestation verify FILE --repo sara-star-quant/oxfuse`. As a library, depend on `oxfuse-core` and `oxfuse-runtime`, plus `oxfuse-wgpu` for the GPU backend and `oxfuse-onnx` for ONNX import.

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

Errors carry a code and a location: ``error[E017]: node `y`: MatMul: inner dimensions differ: [2, 3] x [4, 5] (at nodes[0])``. Library examples: `cargo run --release -p oxfuse-runtime --example optimize_and_run` (passes, then both CPU backends) and `cargo run --release -p oxfuse-wgpu --example matmul` (GPU, with the bound check).

## What you can do with it

| You want | Command |
|---|---|
| Golden outputs for a test suite: the fast backend must match the reference byte for byte | `oxfuse run g.json --inputs i.json --backend reference > golden.json`, then `--backend cpu > out.json`, then `cmp` |
| A record of a run, and a verdict on a difference | `oxfuse run ... --manifest run.json`, then `oxfuse compare golden.json out.json` ([docs/manifest.md](docs/manifest.md)) |
| A bound on how far each output is from the exact result | `oxfuse analyze g.json --inputs i.json` ([docs/analysis.md](docs/analysis.md)) |
| A small ONNX model checked against another runtime | `oxfuse run model.onnx --inputs i.json --backend reference` (needs `--features onnx`) |
| To know whether a GPU can be trusted for your numerics | `oxfuse gpu-selftest` (needs `--features wgpu`) |
| What each optimization pass changes | `oxfuse inspect g.json` |
| To check your own runtime against fixed outputs | the [golden corpus](docs/golden-corpus.md) attached to each release |

## The contract

- **Same bits on the tested platforms.** `MatMul` is one fused multiply-add per step in ascending `k`, and `Sigmoid` uses an `exp` written for this project. The reference and the fast CPU backend agree bit for bit on x86 and ARM, at any SIMD level and thread count. A NaN's sign and payload are the one thing IEEE 754 leaves to the hardware.
- **Exact optimizer.** Four passes (constant folding, layout elision, common-subexpression elimination, dead-code elimination) never change an output bit, infinity and signed zero included.
- **A GPU is measured, not trusted.** `MatMul` matches the reference bit for bit on the devices that fuse the same way and were measured (Apple silicon through Metal, and a Tesla T4; the chips are listed in [docs/adapters.md](docs/adapters.md)); every other op has a documented bound, and `oxfuse gpu-selftest` measures any adapter.
- **Malformed input is rejected, not guessed.** Every error has a stable code and the JSON path of the offender; sizes and element counts are bounded before anything is allocated.

The precise rules are in [docs/semantics.md](docs/semantics.md).

## Evidence

| Property | How it is established |
|---|---|
| Correct against an independent implementation | NumPy float64 oracle, seeded random corpus plus all examples |
| `MatMul` and `Sigmoid` return the same bits on the tested platforms | One fused multiply-add per matmul step, an `exp` of our own, and a hash of all 2^32 `exp` and `Sigmoid` outputs that macOS, Linux and Windows CI each reproduce |
| Fast backend equals reference, bit for bit (all NaNs count as one value) | Contract tests, random-graph tests, differential fuzzing on arm64 and x86_64 |
| Optimizations are exact, including infinity, signed zero and where NaNs appear | Equivalence tests with and without passes |
| Tests detect real defects | cargo-mutants: all 1161 viable mutants of the CPU-side crates stopped, no survivor, reproduced by parallel CI shards; the GPU crate is mutation-tested locally on a real adapter ([docs/mutation-testing.md](docs/mutation-testing.md)) |
| Line coverage | cargo-llvm-cov in CI, library code only; it says what the tests run, mutation testing says how well they check it |
| Hostile input is handled | Loader fuzzing, bounded allocation, 3.07M generated graphs compared across backends with no divergence |
| GPU backend: exact where IEEE allows, bounded elsewhere | Relu bit-exact on all 2^32 inputs; MatMul bit-identical to the reference on the Apple silicon GPUs and the Tesla T4 measured so far (inside the float64 bound on any adapter); Sigmoid inside a measured ulp bound; a CPU model of every kernel grid; Kani bounded model checking of the dispatch fold and the matmul, transpose, permute and broadcast index arithmetic (bounded sizes, not a proof of the whole backend) |
| ONNX models run correctly | 14 fixtures checked against onnxruntime and the `onnx` reference evaluator, bit-exact except one `Sigmoid` model; importer fuzzing ([docs/onnx.md](docs/onnx.md)) |
| Builds on Rust 1.89 or newer | MSRV job in CI |

Every claim maps to a row and a reproduction command in [docs/claims.md](docs/claims.md). Measured speeds, with machine and commands, are in [docs/benchmarks.md](docs/benchmarks.md). On a 512x512 matmul the single-threaded fast backend runs at 0.77x of `ndarray`'s speed (3.6x with threads), and one row against a 784x10 weight runs at 3.4x `ndarray`'s speed. A 784-256-10 classifier with a bias after each product runs at 3.9x `ndarray`'s speed at batch 1, 0.97x at batch 8 and 0.70x at batch 32 (0.85x with threads), all on the recorded machine.

## Scope

| Op | Inputs | Attributes | Rule |
|---|---|---|---|
| `Relu` | 1 | none | `x > 0 ? x : 0`, NaN propagates |
| `Sigmoid` | 1 | none | `1 / (1 + exp(-x))` in `f32` |
| `Add` | 2 | none | NumPy broadcasting: trailing dimensions are aligned, and a dimension of 1 (or a missing one) stretches |
| `Mul` | 2 | none | as `Add` |
| `Sub` | 2 | none | as `Add`; first minus second |
| `Div` | 2 | none | as `Add`; first divided by second |
| `Neg` | 1 | none | same shape; flips the sign |
| `MatMul` | 2 | none | rank 2: `[M, K] x [K, N]` |
| `Reshape` | 1 | `shape` | element count must match |
| `Transpose` | 1 | `perm` | output dim `i` is input dim `perm[i]` |

`f32` only, row-major. CPU by default; an optional GPU backend (`--features wgpu`, `--backend wgpu`) is measured on the devices in [docs/adapters.md](docs/adapters.md), with no support claimed for any device. Not supported: broadcasting in `MatMul`, batched `MatMul`, `-1` in `Reshape`, NaN or infinity in JSON files, a GPU in the default build. The plan for more operators is in [ROADMAP.md](ROADMAP.md).

For production inference, large models, training or speed-critical CPU work, use ONNX Runtime (through `ort`), `tract`, `candle` or `burn`; [docs/comparison.md](docs/comparison.md) quotes what each says about itself.

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

## GPUs

The GPU backend runs on wgpu (Vulkan, Metal, DX12). It is measured on Apple silicon (Metal; the M1 Pro and the macOS CI runner's device, other Apple chips are untested), a Tesla T4 (NVIDIA, Vulkan, a free notebook) and on software adapters, see [docs/adapters.md](docs/adapters.md). AMD and Intel Arc cards should work through Vulkan or DX12 but are untested. If you try your card, run `oxfuse gpu-selftest` and share the report in an issue.

## Documentation

[Architecture](docs/architecture.md) | [IR](docs/ir.md) | [Numerical semantics](docs/semantics.md) | [Numeric contract matrix](docs/contract.json) | [Claims and how to reproduce them](docs/claims.md) | [Benchmarks](docs/benchmarks.md) | [Provenance](docs/provenance.md) | [Comparison](docs/comparison.md) | [Run record](docs/manifest.md) | [Golden corpus](docs/golden-corpus.md) | [Error bounds](docs/analysis.md) | [Licensing](docs/licensing.md) | [Use cases](docs/use-cases.md) | [ONNX import](docs/onnx.md) | [Optimization passes](docs/optimization-passes.md) | [Mutation testing](docs/mutation-testing.md) | [Audit guide](docs/audit.md) | [Adding an op](docs/adding-an-op.md) | [Roadmap](ROADMAP.md) | [Changelog](CHANGELOG.md)

## Security, contributing, citing

Report a vulnerability through the private channel in [SECURITY.md](SECURITY.md). Contributions are welcome under a sign-off (DCO) rule: see [CONTRIBUTING.md](CONTRIBUTING.md) for the gate (`just ci`), the oracle check and the release steps. To cite Oxfuse, use "Cite this repository" on GitHub ([CITATION.cff](CITATION.cff)).

Copyright 2026 SARA STAR QUANT LLC. Apache-2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
