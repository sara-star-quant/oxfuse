# Certified error bounds: `oxfuse analyze`

`oxfuse analyze` runs the reference kernels on your inputs and prints, for every output element, the value and a bound: the exact real-number result of the same graph on the same float32 inputs lies within `value +- bound`. It is the first tool here that says how far a result can be from the truth, not only whether two runs agree.

## Input boxes

Give a range instead of a point and `analyze` encloses every output element for every input in the range. An input is either `{"shape", "values"}` (a point) or `{"shape", "lo", "hi"}` (a box); the two can sit side by side, and a point is a box with equal ends. The ends must be finite numbers, and `lo` may not be above `hi`:

```sh
oxfuse analyze examples/tiny_mlp.json --inputs boxes.json
```

`boxes.json`:

```json
{"x":{"shape":[2,3],"lo":[0.5,-2.5,2.5,-0.5,3.5,-1.5],"hi":[1.5,-1.5,3.5,0.5,4.5,-0.5]}}
```

```json
{"outputs":[{"name":"out","shape":[2,2],"lo":[0.0,0.0,0.0,0.0],"hi":[3.576278935213398e-7,2.000000298023231,2.000000178813941,4.76837195506619e-7]}]}
```

For every input in the box, the exact real-number result and the float32 result of the reference backend (what `run --backend reference` prints) lie in `[lo, hi]` of each output element. There is no `values` or `bounds` in this form, because there is no single value. A `null` end is unbounded: a divisor whose range reaches zero, or a result that can pass the largest float32. `run` does not accept boxes.

How it is built: each element is a center plus coefficients on one shared noise symbol per input element plus an independent error (an affine form), so a value used twice is not counted twice and the width grows slowly with depth. `Add`, `Sub`, `Neg`, `Reshape`, `Transpose`, a product or quotient with a constant, and a `MatMul` with a constant operand are exact on the symbols, with the float32 rounding of every result (and of every fused step of a `MatMul`) and the float64 rounding of the analysis added to the error. `Relu` is exact when its sign is decided and the standard chord relaxation when not. `Min`, `Max`, `Sigmoid`, a product of two varying values and a `MatMul` of two varying matrices use plain interval arithmetic for that node. The answer is intersected with a plain interval analysis of the whole graph, so it is never wider than that. The analysis tracks at most 256 symbols (`oxfuse_runtime::analyze_box_with` takes another number): a wider input gives its widest elements a symbol each and treats the rest as independent errors, which is looser and still sound.

How wide: the median over output elements of the enclosure width divided by the range the reference backend reaches on 400 sampled points of the box, for random 16-wide ReLU networks of the given depth (five networks per cell). The sampled range is narrower than the true range, so these ratios overstate the width.

| Box | Method | 1 | 2 | 3 | 4 | 5 | 6 (layers) |
|---|---|---|---|---|---|---|---|
| +-1 percent | plain intervals | 1.96 | 4.2 | 10.7 | 32 | 86 | 287 |
| +-1 percent | affine forms | 1.96 | 1.8 | 1.9 | 1.9 | 1.9 | 2.5 |
| +-10 percent | plain intervals | 1.96 | 4.7 | 12.5 | 29 | 81 | 244 |
| +-10 percent | affine forms | 1.96 | 2.2 | 3.0 | 6.1 | 17 | 75 |

Reproduce with `cargo test -p oxfuse-runtime --lib the_width_over_the_sampled_range_by_depth -- --nocapture`. A large box costs width at each `Relu` whose sign it cannot decide, so the enclosure of a deep network over a wide box can be many times the true range: it is a sound outer bound, not a tight one.

## What it assumes

- **IEEE 754 binary32 round to nearest**, with a correctly rounded fused multiply-add (Rust's `mul_add` is, in hardware or in software).
- **Gradual underflow.** A thread that flushes subnormals to zero, or reads them as zero, breaks the rounding model; `analyze` checks this at start and refuses to run.
- **float32 inputs.** Decimal numbers in an inputs file round to float32 when read, and the interval covers those float32 values.

## What it does not do

- **A point bound covers the inputs you give.** For a range of inputs give a box (below); the enclosure it returns is an interval for every input in the range, and it is looser than a point bound.
- **The bound is pessimistic.** It adds up worst cases. On the 50 graphs of the NumPy oracle corpus and the examples it was 3.7 times the real error at the median, 20.8 times at the 90th percentile and 1311 times at the worst element (419 elements where the real error is not zero; the others sit on exact results).
- **Nothing is certified once a value is not finite.** An element that is, or came from, an infinity or NaN has a `null` bound. A non-finite output value is an error, as in `run`.
- **Only the reference arithmetic.** The fast CPU backend returns the same bits, so the bound holds for it too. A GPU that does not fuse the same way has its own measured bound ([semantics.md](semantics.md)), and this tool does not cover it.
- **The graph is analyzed as written**, without the optimization passes. For an ONNX model that is the imported graph: the importer does no arithmetic of its own (a `Gemm` `beta` scaling is a `Mul` node), so the interval covers the model's real-number result.

## Example

```sh
oxfuse analyze examples/tiny_mlp.json --inputs examples/tiny_mlp.inputs.json
```

```json
{"outputs":[{"name":"out","shape":[2,2],"values":[0.0,1.0,1.0,0.0],"bounds":[4.172325134277348e-7,2.384185791015627e-7,1.192092895507813e-7,4.172325134277348e-7]}]}
```

The `values` are what `oxfuse run --backend reference` prints, bit for bit.

## How the bound is built

Each element carries its computed value and a bound `e` on its distance from the exact value. Inputs and constants start at `e = 0`. Each operator adds what it can move the result by:

| Operator | Bound after the operator |
|---|---|
| `Add` | `e_a + e_b` plus half an ulp of the result: round to nearest is within half an ulp (IEEE 754) |
| `Mul` | `\|a\| e_b + \|b\| e_a + e_a e_b` plus half an ulp of the result |
| `Sub` | `e_a + e_b` plus half an ulp of the result, as `Add` |
| `Div` | `(e_a \|b\| + \|a\| e_b) / (\|b\| (\|b\| - e_b))` plus half an ulp of the result; infinite when `e_b >= \|b\|`, because the true divisor may then be zero |
| `Neg` | unchanged: a sign flip does not round |
| `Min`, `Max` | the larger of the two input bounds: the result is one of the operands, so nothing rounds, and the operation moves by no more than the larger input error |
| `MatMul` | the fixed order of the kernel replayed step by step: one fused multiply-add per `k`, so each step adds `\|a\| e_b + \|b\| e_a + e_a e_b` and half an ulp of the running sum |
| `Relu` | unchanged: `max(0, x)` never moves a value further than its input moved |
| `Sigmoid` | `e / 4` (the slope of the sigmoid is at most 1/4) plus a relative `2^-22 (1 + 2^-20)` of the result for the own `exp` (within 1 ulp for every input, checked over all 2^32) and the addition and division that follow |
| `Reshape`, `Transpose` | each bound moves with its element |

Below the normal range the spacing of floats is fixed, so half an ulp there is `2^-150`. The bound itself is computed in float64 and rounded up after every operation, so rounding in the analysis can only make it larger. The method is the standard first-order rounding analysis of Higham, *Accuracy and Stability of Numerical Algorithms* ([provenance.md](provenance.md)). Replaying the `MatMul` order is why the summation order is part of the contract.

## How it is checked

- The NumPy float64 oracle result of every shipped graph lies inside its interval (`crates/cli/tests/analyze.rs`).
- The exact result of 300 random graphs over every operator lies inside, and the values equal the reference backend's bits (`crates/runtime/src/analyze.rs` tests).
- The relative bound of the own `sigmoid` holds on a dense sample of bit patterns, and its constants are pinned by a test.
- Mutation testing of the module has no missed mutant ([mutation-testing.md](mutation-testing.md)).
