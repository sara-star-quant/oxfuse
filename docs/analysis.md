# Certified error bounds: `oxfuse analyze`

`oxfuse analyze` runs the reference kernels on your inputs and prints, for every output element, the value and a bound: the exact real-number result of the same graph on the same float32 inputs lies within `value +- bound`. It is the first tool here that says how far a result can be from the truth, not only whether two runs agree.

## What it does not do

- **It is for the inputs you give.** It does not bound the error over a range of inputs.
- **The bound is pessimistic.** It adds up worst cases. On the 50 graphs of the NumPy oracle corpus and the examples it was 3.7 times the real error at the median, 20.8 times at the 90th percentile and 1311 times at the worst element (419 elements where the real error is not zero; the others sit on exact results).
- **Nothing is certified once a value is not finite.** An element that is, or came from, an infinity or NaN has a `null` bound. A non-finite output value is an error, as in `run`.
- **Only the reference arithmetic.** The fast CPU backend returns the same bits, so the bound holds for it too. A GPU that does not fuse the same way has its own measured bound ([semantics.md](semantics.md)), and this tool does not cover it.
- **The graph is analyzed as written**, without the optimization passes.

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
