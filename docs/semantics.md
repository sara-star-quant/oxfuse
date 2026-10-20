# Numerical semantics

What each op computes, bit for bit where that is promised, and what is deliberately not specified. All values are `f32`, stored contiguous and row-major.

## Ops

| Op | Definition |
|---|---|
| `Relu` | `x > 0 ? x : 0`. NaN stays NaN. `-0.0` becomes `+0.0`. |
| `Sigmoid` | `1 / (1 + exp(-x))` in `f32` with `std`'s `exp`. Large negative `x` makes `exp` overflow to infinity and the result 0, without NaN. NaN stays NaN. Tested to stay within an absolute error of 4 times the `f32` epsilon of the float64 formula on `[-20, 20]`. |
| `Add`, `Mul` | IEEE `f32` addition and multiplication per element. The operands have equal shapes, or one is rank 0 and is applied to every element of the other. No other broadcasting. |
| `MatMul` | For `[M, K] x [K, N]`, each output element is the sum over `k` from 0 to K-1 in ascending order, starting from `+0.0`, one `f32` multiply and one `f32` add per step. No fused multiply-add, no reassociation, and no skipping of a zero operand (`0 * NaN` and `0 * inf` stay NaN). |
| `Reshape` | The same elements in the same order under a new shape. The element count must match and `-1` is not supported. |
| `Transpose` | A copy in which output dimension `i` is input dimension `perm[i]`. |

## What the backends promise

- `Relu`, `Add`, `Mul`, `Reshape` and `Transpose` are exact, so every backend gives the same bits.
- `CpuBackend` returns the reference backend's bits for every op, including `MatMul` and `Sigmoid`, for any input including NaN, infinity and signed zero. This is tested by comparing bit patterns, not by a tolerance.
- The threaded matmul splits the output by rows and never splits the `k` sum, so the thread count cannot change a result.
- The optimization passes are exact: an optimized graph gives the same bits as the unoptimized one on the reference backend, errors included.

## What is compared with NumPy

The reference backend is checked against an independent NumPy float64 oracle (`scripts/gen_expected.py`). Exact ops must match on exactly representable inputs. `MatMul` on random inputs is held to the bound `K * eps * sum(|a_ik * b_kj|)` against the float64 result, where `eps` is the `f32` machine epsilon, and to exact equality on small-integer inputs. The oracle and the Rust code apply the same input rule: a JSON number is read as float64 and rounded to `f32` with round-to-nearest-even.

## Values and errors

- JSON files hold finite numbers only. NaN and infinity reach the backends through Rust tests, not through files.
- A run that produces a non-finite output through the CLI fails with an error instead of printing `null`, because JSON cannot represent it.
- Shape values never come from data. Every shape is known when the graph is built.

## Not specified

- The bits of `Sigmoid` across different `std` versions or platforms: `exp` comes from the platform's math library.
- NaN payload and sign bits beyond what the hardware's `f32` operations produce.
- Any op, dtype or broadcasting rule not listed here. They are rejected at load time.
