# Oxfuse

[![ci](https://github.com/sara-star-quant/oxfuse/actions/workflows/ci.yml/badge.svg)](https://github.com/sara-star-quant/oxfuse/actions/workflows/ci.yml)

A tensor-graph compiler in Rust. It takes a JSON graph, type-checks it, optimizes it without changing results, and runs it on the CPU. Experimental, not for production use.

## What it does

- **Compiles** a JSON graph to a typed IR with shape inference and a verifier that re-checks every invariant after each pass.
- **Optimizes** with three exact passes: constant folding, common-subexpression elimination, dead-code elimination. Output bits never change.
- **Runs** on two CPU backends. The fast backend returns the same bits as the reference backend. It is the default.
- **Rejects** malformed graphs with a stable error code and the JSON path of the offender.

## Evidence

| Property | How it is established |
|---|---|
| Correct against an independent implementation | NumPy float64 oracle, seeded random corpus plus all examples |
| Fast backend equals reference, bit for bit | Contract tests, random-graph tests, differential fuzzing |
| Optimizations are exact, including NaN, infinity, signed zero | Equivalence tests with and without passes |
| Tests detect real defects | cargo-mutants: 268 of 280 caught, survivors documented |
| Hostile input is handled | Loader fuzzing, bounded allocation, 3.07M generated graphs compared across backends with no divergence |
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

`f32` only, row-major, CPU only.

Not supported: broadcasting beyond a scalar operand, batched `MatMul`, `-1` in `Reshape`, NaN or infinity in JSON files, any accelerator. No vendor support is claimed or implied.

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

Errors carry a code and a location: ``error[E017]: node `y`: MatMul: inner dimensions differ: [2, 3] x [4, 5] (at nodes[0])``.

Full gate: `just ci`. Oracle check: `just oracle-check` (needs `numpy`, see `scripts/requirements.txt`).

## Documentation

[Architecture](docs/architecture.md) | [IR](docs/ir.md) | [Numerical semantics](docs/semantics.md) | [Optimization passes](docs/optimization-passes.md) | [Adding an op](docs/adding-an-op.md) | [Design decisions](docs/decisions) | [Roadmap](ROADMAP.md) | [Contributing](CONTRIBUTING.md) | [Security](SECURITY.md)

Apache-2.0. See [LICENSE](LICENSE).
