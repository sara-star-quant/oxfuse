# Golden corpus

A set of graphs with the exact bits Oxfuse returns for them, for checking another runtime's CI against ours. It ships with each release as `oxfuse-corpus-VERSION.tar.gz` (built by `just golden`).

Each case is a folder under `cases/`:

| File | Content |
|---|---|
| `graph.json` | the graph, in the format of [ir.md](ir.md) |
| `inputs.json` | the inputs (absent when the graph has none) |
| `expected.outputs.json` | what `oxfuse run` prints, so `oxfuse compare` reads it |
| `expected.json` | the same values as f32 bit patterns (`"0x3f800000"`), the operators the graph uses with their contract level, and the NumPy float64 tolerance the case was checked with (`null` for generated cases) |

`index.json` lists the cases. Hex bit patterns are the reference; decimals are for people.

How to read a result: the level of an operator in `expected.json` says what to expect. For `exact` operators (`Relu`, `Add`, `Mul`, `Reshape`, `Transpose`) any runtime should return the same bits. `MatMul` (`fused-defined`) and `Sigmoid` (`own-math`) are defined by Oxfuse: one fused multiply-add per step in ascending k, and our own `exp`. A runtime that sums in another order or uses its library `exp` will differ by a few ulp; that is not a bug in either, and [semantics.md](semantics.md) says how much to expect.

Check the corpus against Oxfuse itself:

```sh
oxfuse run cases/example_tiny_mlp/graph.json --inputs cases/example_tiny_mlp/inputs.json > out.json
oxfuse compare cases/example_tiny_mlp/expected.outputs.json out.json
```

The corpus is built from the repository's examples, its seeded random graphs and a few generated sweeps (`Relu` on special values, `Sigmoid` across its range, `MatMul` at row counts 1 to 70). The archive is reproducible: the same commit gives the same bytes.
