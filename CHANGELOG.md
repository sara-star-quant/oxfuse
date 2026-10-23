# Changelog

All notable changes. The format follows Keep a Changelog; versions follow Semantic Versioning, and the crates move together on one version.

## Unreleased

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
