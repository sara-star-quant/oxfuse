# Changelog

All notable changes. The format follows Keep a Changelog; versions follow Semantic Versioning, and the crates move together on one version.

## Unreleased

- Breaking: `Limits` is `#[non_exhaustive]`, so a later limit is not a breaking change. Build one from `Limits::default()` and the `with_*` methods (`Limits::default().with_max_values(10)`); a struct literal no longer compiles outside `oxfuse-core`. New limit `max_macs` (no cap by default): `GraphBuilder::finish` refuses a graph whose `estimated_macs` is above it with `LimitExceeded { what: "multiply-adds" }`, so a library user or an ONNX import is protected before any pass runs. `oxfuse run --max-macs N` now sets it and still reports `E109`.
- Licensing: a sign-off (DCO) rule for outside commits, a ScanCode licence scan on release branches and a local snippet scan (`just licence-snippets`) for releases. `docs/licensing.md` records the licence review (what ships, under what terms, the options compared, why Apache-2.0 stays). `NOTICE` now credits the ONNX project for the file format description the importer follows, and the binary's `THIRD_PARTY_LICENSES.txt` says its own crates are listed under their own licence.
- `oxfuse analyze graph.json --inputs inputs.json` prints every output with a bound: the exact real-number result lies within `value +- bound`, computed by replaying the fixed operator order with the standard rounding model. `oxfuse_runtime::analyze` is the library entry point. For the inputs given only, and pessimistic (3.7 times the real error at the median on the oracle corpus). See `docs/analysis.md`.
## 0.5.1 - 2026-10-30

- With the `parallel` feature, a few rows (4 to 16) of a large product are split by columns over up to four threads, where before they ran on one. With a 1024 by 1024 weight, 5 rows take 0.6x and 8 rows 0.5x of the time on an M1 Pro and on x86 (4 vCPU Xeon). Each sum stays whole in one task, so the bits do not change with the thread count.
- The fast CPU matmul runs one and two rows as wider blocks (four and two strips of 16 columns), which hides the latency of the fused multiply-add. The bits do not change. A 784-256-10 classifier at batch 1 is 1.5x faster on an M1 Pro and on x86 (Xeon 2.2 GHz), batch 8 is 8 percent faster, batch 32 is unchanged.- Docs: `docs/use-cases.md` says what Oxfuse is for in research and in narrow production uses, and what it is not for; the crate READMEs link it. Each crate README now says what the crate is for in AI work and carries an example that runs from crates.io alone (the `oxfuse` README is a complete run without a clone).
- CI: `cargo semver-checks` runs on release branches, the NumPy oracle check runs on every change (bounded outputs are compared inside their bound, since a BLAS differs in the last bits between machines), and the pull request mutants job runs as four shards and is cancelled by a new push.

## 0.5.0 - 2026-10-29

- Breaking: `Error`, `OnnxError` and `CliError` are `#[non_exhaustive]`, so a `match` on them needs a `_` arm and later variants are not breaking. `OnnxError::TooManyMessages` and the new `CliError` variants (`E109` to `E113`) arrive with this release, and `RunOptions` gained fields.
- Prebuilt `oxfuse` binaries (ONNX and GPU features on) for Linux and Windows on x86_64 and arm64 and macOS on arm64, attached to each release with `SHA256SUMS` and a build attestation (`gh attestation verify`). Each archive carries `LICENSE`, `NOTICE` and `THIRD_PARTY.txt`.
- Benchmarks: an `mlp` group (a 784-256-10 classifier at batch 1, 8 and 32) with an `ndarray` control, the report regenerated from one run, and a weekly non-gating `bench-trend` workflow on one x86_64 and one arm64 runner.
- A golden corpus for other runtimes' CI: graphs with the exact bits Oxfuse returns, built by `just golden`. See `docs/golden-corpus.md`.
- `oxfuse compare A B` compares two outputs files bit by bit and gives a verdict (identical, permitted variation, contract violation, unresolved) using the run records from `--manifest`. See `docs/manifest.md`.
- `oxfuse run --manifest FILE` writes a JSON record of the run: version, backend, platform, SIMD level, and SHA-256 of the model, inputs and each output. See `docs/manifest.md`.
- `oxfuse run --max-macs N` refuses a graph that needs more than `N` multiply-adds (`E109`) before any pass runs, and `Graph::estimated_macs` gives the number.
- The ONNX importer counts messages before decoding and refuses more than ten per `max_values` (`E314`). A 16 MiB file of empty nodes took 1.2 GB to decode and now fails at 21 MB.

## 0.4.1 - 2026-10-27

- GPU backend: a panic in one thread no longer poisons the device state for every later call (locks recover instead of panicking); a failed shader compile is dropped from the shared pipeline cache so a retry compiles again; when a device error and a backend error happen in one call, the device error is reported first and the backend error follows as context (typed errors such as a wrong input count are never replaced).
- Tests and docs: a row's result is proven independent of the batch size (row counts from 1 to 13 and 31 to 70, across the register block, the k-tile, the tail columns and the thread split) on the reference and fast CPU backends and on fusing GPU adapters; `docs/semantics.md` states it.
- `docs/contract.json` states each operator's numeric contract on the reference, CPU and GPU backends (exact, fused-defined, own-math, bounded or measured). A test fails when an operator has no row, an evidence path is missing or a named test does not exist.
- `docs/provenance.md` records the source and licence of every algorithm and of the dependencies; a test fails when a function of the math and kernel files has no row.
- `docs/comparison.md`: what ONNX Runtime, tract, rten, candle, Burn and the RepDL paper say about themselves (scope, licence, devices, training, reproducibility, ONNX operators, releases), each cell quoted from the page at a named version with the link and the date read. No speed column, no ranking. `just links` checks the links before a release.
- CI and `cargo deny`: unknown registries and git sources are denied, and the mutants workflow no longer interpolates the base branch name into a shell command.

## 0.4.0 - 2026-10-23

- Testing and evidence: CI runs the test suite and a hash over all 2^32 `exp` and `Sigmoid` inputs on five platforms (Linux, Windows on x86_64 and arm64, macOS on arm64) and the hash must match; weekly mutation testing runs in nine parallel jobs with a known-survivors ratchet, and all 973 viable mutants of the CPU-side crates are stopped (1062 in total, 0 missed); Kani proofs cover the transpose composition and band placement.
- Licence: `LICENSE` names the copyright owner (Copyright 2026 SARA STAR QUANT LLC) and every crate ships `LICENSE` and a `NOTICE` file.
- The fast CPU matmul reads the last columns of B in place instead of copying them first, which makes narrow outputs such as a 784 by 10 classifier head 3 to 4 times faster at one row. `just bench` now also times small batches against a hidden-layer weight and the classifier head.
- The reference matmul runs with hardware fused multiply-add on x86 too (it was 2.4 times slower there than in 0.3.0 after the fused definition). The bits do not change.
- Breaking: `Sigmoid` uses an `exp` written for this project instead of the platform's, so it no longer depends on the platform's math library: the same bits on the tested platforms, within 1 ulp of the correctly rounded result (6,412,479 of 2^32 inputs differ from it, all by 1 ulp). The bits differ from 0.3.0 in the last place for some inputs. The fast CPU `Sigmoid` runs on SIMD vectors and takes about half the time (1.76 ms to 0.84 ms for 2^20 elements on an M1 Pro).
- The fast CPU matmul runs on explicit SIMD vectors (`fearless_simd`), and x86 picks AVX2 or AVX-512 at run time with no build flags. Without it the fused multiply-add would call into libm on x86 and run about 40 times slower. Minimum Rust is now 1.89.
- Breaking: `MatMul` now does one fused multiply-add per step (`acc = fma(a, b, acc)`, ascending k, from `+0.0`), so results can differ from 0.3.0 in the last bits. The reference and the CPU backends agree bit for bit on the tested platforms, and the GPU backend matches them on the adapters that fuse (Apple M1 Pro and Tesla T4 measured). The fast CPU matmul is about 1.5 times faster at 512 by 512.
- `Executable::run_many` runs a batch of inputs; the GPU backend submits the batch at once and waits once. The default implementation calls `run` per element.
- GPU backend: `run` records the kernels and the output copies in one command buffer and waits once for all outputs.
- `just bench-ab` compares a benchmark filter between a base commit and the working tree over interleaved rounds.
- GPU backend: a lost device is reported as `Error::Backend("device lost ...")` by every call, and a later call no longer panics inside wgpu; validation and out-of-memory errors are attributed to the thread and call that caused them; uploads write straight into the mapped buffer; `Context::with_timeout` replaces the fixed 30 second wait.
- The weekly fuzz run also compares generated graphs on a software Vulkan device (`gpu_generated`).
- GPU backend: executables on one device share compiled pipelines, so compiling many graphs no longer recompiles the same shaders.
- Crate metadata: `artificial-intelligence` category and the `onnx`, `inference` and `wgpu` keywords where they apply, shorter descriptions, `docs.rs` builds with all features for `oxfuse` and `oxfuse-wgpu`.
- Crate-level docs with a runnable quick start for `oxfuse-core` and `oxfuse-runtime`; sub-crate READMEs point to the root README.
- README: who the project is for and who it is not for.
- GPU backend: `Context::with_timeout` clamps the timeout to one year, so `Duration::MAX` no longer panics; a device loss during `wait_idle` is reported as a loss and not as a poll error.
- Docs: the `Sigmoid` accuracy claim names the exact count (6,412,479 inputs differ from the correctly rounded result), and `docs/semantics.md` states the default floating-point environment and that `parallel` is tested on 64-bit only.

## 0.3.0 - 2026-10-21

- ONNX import for a small float32 operator subset (`oxfuse-onnx`, `oxfuse --features onnx`), opsets 13 to 21, every unsupported feature a named error.
- Simpler GPU host code, measured NVIDIA wording, a security policy.

## 0.2.1 - 2026-10-21

- Crate descriptions, keywords and READMEs updated for the GPU backend. No code changes.

## 0.2.0 - 2026-10-21

- wgpu backend (`oxfuse-wgpu`, `--backend wgpu`, `oxfuse gpu-selftest`).
- Breaking: `Error::Backend` (E021) added to `oxfuse-core`.

## 0.1.0 - 2026-10-20

- First release: seven ops, exact optimization passes (constant folding, common-subexpression elimination, dead-code elimination), reference and fast CPU backends, JSON graph format, NumPy oracle.
