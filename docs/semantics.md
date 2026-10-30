# Numerical semantics

What each op computes, bit for bit where that is promised, and what is deliberately not specified. All values are `f32`, stored contiguous and row-major.

## Ops

| Op | Definition |
|---|---|
| `Relu` | `x > 0 ? x : 0`. NaN stays NaN. `-0.0` becomes `+0.0`. |
| `Sigmoid` | `1 / (1 + exp(-x))` in `f32`, with an `exp` written for this project (`crates/runtime/src/math.rs`) that uses only IEEE operations and fused multiply-adds, so the result does not depend on the platform's math library. That `exp` is within 1 ulp of the correctly rounded result and equal to it for all but 6,412,479 of the 2^32 `f32` bit patterns (0.15 percent), checked against CORE-MATH over every one of them. Large negative `x` makes `exp` overflow to infinity and the result 0, without NaN. NaN stays NaN. Tested to stay within a relative error of 4 times the `f32` epsilon of the float64 formula on `[-20, 20]`. |
| `Add`, `Mul` | IEEE `f32` addition and multiplication per element. The operands broadcast like NumPy: trailing dimensions are aligned and a dimension of 1 (or a missing one) stretches to the other operand's size, so a scalar, a row `[n]` against `[m, n]` or a column `[m, 1]` against `[m, n]` all work. Each output element is the operation on one element of each operand; no sum is involved, so broadcasting never changes a result. The GPU backend runs equal shapes and a scalar operand; any other broadcast makes `compile` return `Error::Backend` (it names the shapes) until its kernels exist. |
| `MatMul` | For `[M, K] x [K, N]`, each output element is the sum over `k` from 0 to K-1 in ascending order, starting from `+0.0`, one fused multiply-add per step (`acc = fma(a, b, acc)`, a single `f32` rounding, defined by IEEE 754, so it does not depend on the platform). No reassociation, and no skipping of a zero operand (`0 * NaN` and `0 * inf` stay NaN). |
| `Reshape` | The same elements in the same order under a new shape. The element count must match and `-1` is not supported. |
| `Transpose` | A copy in which output dimension `i` is input dimension `perm[i]`. |

## What the backends promise

- `Relu`, `Add`, `Mul`, `Reshape` and `Transpose` are exact, so every backend gives the same bits.
- `CpuBackend` returns the reference backend's bits for every op, including `MatMul` and `Sigmoid`, for any input including infinity and signed zero, and returns a NaN exactly where the reference does. This is tested by comparing bit patterns, not by a tolerance, with all NaNs counted as one value: the sign and payload of a NaN made from two NaN operands or from an invalid operation (`inf - inf`, `0 * inf`) are left to the hardware and to the operand order the compiler picks, and x86 and ARM differ (see "Not promised" below). The first fuzz run on x86 found this, and a replay on the published 0.3.0 reproduced it.
- The threaded matmul splits the output by rows (by columns for 16 rows or fewer) and never splits the `k` sum, so the thread count cannot change a result.
- A row's result does not depend on the batch size: row `i` of an `m`-row run equals the run of that row alone, bit for bit (tested on the reference and the fast CPU backend for row counts from 1 to 13 and 31 to 70, and on the GPU where the adapter fuses). The register block, the wider blocks for one and two rows, the tail columns and the thread split only change which elements share a pass; each element's sum is the same.
- The optimization passes are exact: an optimized graph gives the same bits as the unoptimized one on the reference backend, errors included.

## The GPU backend

`WgpuBackend` (feature `wgpu`) runs the same ops on a device. The shading language does not pin down floating-point behavior the way IEEE 754 does, and the device flushes subnormal numbers to zero, so the contract is stated in tiers.

**Promised.** These hold on any adapter, for the inputs named:

| Op | Contract |
|---|---|
| `Relu` | Bit-exact for every input, including NaN, infinity, signed zero and subnormals. It is computed on the bit pattern, so no floating-point rule applies. |
| `Reshape`, `Transpose` | Bit-exact for every input. They only move bits. |
| `Add`, `Mul` | Bit-exact when both operands hold only zero and normal numbers and the true result is zero or normal. |
| `MatMul` | When every input is zero or a normal number, within `K * eps * sum(|a_ik * b_kj|) + 2 * K * f32::MIN_POSITIVE` of the float64 result. The first term is the CPU bound. The second covers products and partial sums that the device flushes to zero, which lose less than one smallest-normal number each. Bit-identical to the reference on devices that fuse multiply-add the same way (tested on the Apple M1 Pro and the Tesla T4 in [adapters.md](adapters.md)); on any other adapter `gpu-selftest` reports how far off it is. |

**Measured, not promised.** These are measurements of one adapter (the first row of [adapters.md](adapters.md)); other adapters may differ and `oxfuse gpu-selftest` measures them:

| Op | Measurement |
|---|---|
| `Sigmoid` | Within 65 ulp of the reference for every finite input whose result is a normal number, over all 2^32 inputs. A result below the smallest normal number may be flushed to zero. The tests enforce this figure only on that adapter. |

**Outside both.** Subnormal inputs to `Add`, `Mul` or `MatMul`, and any NaN or infinity reaching them, are not promised: the device may flush a subnormal operand or a subnormal result to zero, and the language lets a compiler assume NaN and infinity never occur. `adapters.md` records what each measured device does with them, for example that `infinity * subnormal` is NaN on the first one.

The CLI rejects non-finite input and output, but JSON can express subnormal numbers, and `--backend wgpu` prints whatever the device computed, flushed zeros included. Use `--backend cpu` when subnormal values matter.

## What is compared with NumPy

The reference backend is checked against an independent NumPy float64 oracle (`scripts/gen_expected.py`). Exact ops must match on exactly representable inputs. `MatMul` on random inputs is held to the bound `K * eps * sum(|a_ik * b_kj|)` against the float64 result, where `eps` is the `f32` machine epsilon, and to exact equality on small-integer inputs. The oracle and the Rust code apply the same input rule: a JSON number is read as float64 and rounded to `f32` with round-to-nearest-even.

## Values and errors

- JSON files hold finite numbers only. NaN and infinity reach the backends through Rust tests, not through files.
- A run that produces a non-finite output through the CLI fails with an error instead of printing `null`, because JSON cannot represent it.
- Shape values never come from data. Every shape is known when the graph is built.

## Not specified

- NaN payload and sign bits beyond what the hardware's `f32` operations produce.
- The floating-point environment: the contract assumes the default one. A thread that sets flush-to-zero or denormals-are-zero (x86 `MXCSR`, AArch64 `FPCR.FZ`) flushes subnormal products, partial sums and `exp` results on that thread only, so a threaded run could then differ from a serial one.
- The `parallel` feature on 32-bit targets: it is built and tested on 64-bit platforms only (the row-band sizes multiply by the row length, which can overflow a 32-bit `usize` for the largest shapes the limits allow).
- Any op, dtype or broadcasting rule not listed here (`MatMul` does not broadcast). They are rejected at load time.
