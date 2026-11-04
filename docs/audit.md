# Audit guide

For someone who has to decide whether to trust this code. About 12,000 lines of library source in five crates, 47 distinct crates in the default build (131 with every feature), no `unsafe` of its own.

## What to read, in this order

| Order | File | Why it matters | Lines |
|---|---|---|---|
| 1 | [semantics.md](semantics.md), [contract.json](contract.json) | What is promised, per operator and backend | |
| 2 | `crates/runtime/src/math.rs`, `crates/runtime/src/cpu/vecmath.rs` | The own `exp` and `Sigmoid`; the one place a result could drift between platforms | 510 |
| 3 | `crates/runtime/src/kernels.rs`, `crates/runtime/src/cpu/matmul.rs` | The reference and the fast kernels; the fused multiply-add order | 900 |
| 4 | `crates/core/src/shape.rs`, `crates/core/src/verify.rs` | Shape inference, broadcasting, and the checks that make a `Graph` well typed | 870 |
| 5 | `crates/core/src/passes/` | The four optimization passes; each must keep every output bit | 1,500 |
| 6 | `crates/cli/src/loader.rs`, `crates/onnx/src/decode.rs` | The two doors for untrusted input (JSON and ONNX) and the limits in `crates/core/src/limits.rs` | 1,100 |
| 7 | `crates/runtime/src/analyze.rs` | The error bounds of `oxfuse analyze`; a wrong bound is the worst failure here | 640 |
| 8 | `crates/wgpu/src/` | The optional GPU backend: generated WGSL, a CPU model of each grid, the self-test | 4,100 |

Skip the tests on a first pass; read them to see what a claim is checked against.

## Trust boundaries

- **Inputs are untrusted**: a graph file, an ONNX model, an inputs file. Size, rank, element count and work (`max_macs`) are bounded before anything is allocated ([architecture.md](architecture.md), [onnx.md](onnx.md)).
- **No `unsafe`**: `unsafe_code = "forbid"` in the workspace lints. The dependencies that use `unsafe` are `wgpu` and `fearless_simd` (the SIMD kernels); their advisories count as ours ([SECURITY.md](../SECURITY.md)).
- **No network, no native code loaded from a model**, no build script of our own.
- **Dependencies**: `cargo deny` (advisories, licences, sources) on every pull request, a ScanCode licence scan on release branches ([licensing.md](licensing.md)), and Dependabot.

## Check it yourself

| To see | Run | Time |
|---|---|---|
| Everything builds, lints clean, tests pass | `just ci` | minutes |
| The same bits on this platform | `just exp-sweep` (hashes all 2^32 `exp` and `Sigmoid` inputs) | minutes |
| The answers agree with an independent implementation | `just oracle-check` (NumPy float64) | seconds |
| The tests catch real defects | `just mutants` ([mutation-testing.md](mutation-testing.md)) | hours |
| The index arithmetic is right at the bounds tested | `just kani` | minutes |
| A release is what CI built | `gh attestation verify FILE --repo sara-star-quant/oxfuse` and `SHA256SUMS` on the release page | seconds |
| Provenance of every function | [provenance.md](provenance.md): each function names the source it follows | reading |

Every claim has a row with its test and command in [claims.md](claims.md).

## What the tests do not show

- Line coverage (about 98 percent of library lines) says what the tests run, not how well they check it. Mutation testing is the measure of that: all viable mutants of the CPU-side crates are stopped, the GPU crate is mutated locally on a real adapter.
- The GPU claims are about the adapters in [adapters.md](adapters.md) only.
- Formal verification is bounded model checking of index arithmetic (Kani), not a proof of the numerics.
- No third-party security audit and no legal review of the licence analysis have been done; [licensing.md](licensing.md) lists what is open.
