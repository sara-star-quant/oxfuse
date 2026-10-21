# The IR

A `Graph` is a list of values in definition order. A value is an input, a constant, or the result of a node. Because values are stored in definition order, a node's inputs always have smaller ids than the node itself, so the list is already a topological order.

## Values and types

- A **value** has a `TensorType` (`f32` and a shape), a source and an optional name. Ids are `%0`, `%1`, ... in definition order.
- Sources: `input` (supplied at run time, in declared order), `const` (data stored in the graph), or a node `Op(%a, %b)`.
- A shape is a list of dimensions. Every dimension is at least 1. Rank 0 (a scalar) is allowed. Rank is at most 8.
- The graph also stores its **outputs** (value ids) and the output names, in declared order.

## Ops

| Op | Inputs | Attributes | Result shape |
|---|---|---|---|
| `Relu` | 1 | none | same as the input |
| `Sigmoid` | 1 | none | same as the input |
| `Add` | 2 | none | the common shape; if one operand is rank 0, the other operand's shape |
| `Mul` | 2 | none | as `Add` |
| `MatMul` | 2 | none | `[M, N]` from `[M, K]` and `[K, N]`; rank 2 only |
| `Reshape` | 1 | `shape` | the `shape` attribute; the element count must match |
| `Transpose` | 1 | `perm` | `[input[perm[0]], input[perm[1]], ...]`; `perm` must be a permutation of `0..rank` |

The shape rules live in one function per op in `crates/core/src/shape.rs`. `MatMul` and elementwise rules are written for the general case and restricted in the function, so widening them later is a one-function change.

## Invariants the verifier re-checks

The builder cannot produce an invalid graph. The verifier exists to catch a bug in a pass that rebuilds one. `Graph::verify` checks:

- the output names match the outputs, and every output id is in range;
- every dimension is at least 1 and no element count overflows;
- every node uses only values defined before it;
- every stored node type equals the type inferred from its inputs;
- the input list matches the values whose source is an input;
- constants hold exactly as many values as their shape needs;
- names are unique.

`Graph::verify_limits` re-checks the resource limits (value count, elements per tensor, total elements, rank), so a pass cannot grow a graph past the limits it started under.

## Text dump

`oxfuse inspect FILE` and `oxfuse run FILE --dump-ir` print a deterministic text form. The same graph always prints the same text, and snapshot tests pin it. For `examples/tiny_mlp.json`:

```
graph {
  %0 x = input : f32[2, 3]
  %1 w = const : f32[3, 2] = [1.0, 0.0, 0.0, 1.0, 1.0, 1.0]
  %2 b = const : f32[2, 2] = [-5.0, 0.0, 2.0, -4.0]
  %3 projection = MatMul(%0, %1) : f32[2, 2]
  %4 biased = Add(%3, %2) : f32[2, 2]
  %5 out = Relu(%4) : f32[2, 2]
  output out = %5
}
```

## The JSON file

`format_version` is 1 today. The top-level fields are `format_version`, `inputs` (optional), `constants` (optional), `nodes` and `outputs`. Unknown fields are errors. Inputs and constants carry `name`, `dtype` (`"f32"`) and `shape`; constants also carry `values`. Nodes carry `name`, `op`, `inputs` (names) and optional `attrs` (`shape`, `perm`). Nodes may appear in any order. JSON cannot express NaN or infinity, so values from files are finite. By default a graph can be 64 MiB of file, 100,000 values, 2^28 elements per tensor and 2^29 elements in total, with tensors up to rank 8.

These are defaults, not part of the format. They live in `Limits`, so a library user can pass higher ones without any format change, and the command-line tool could get flags for them later.

The format has its own number, separate from the crate version. Files written for a version keep loading: when the format changes, the new version gets a new number and later releases read both. The releases are listed on [GitHub](https://github.com/sara-star-quant/oxfuse/releases).

| `format_version` | Since | Read by |
|---|---|---|
| 1 | 0.1.0 | every release |
