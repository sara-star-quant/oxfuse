# Changelog

All notable changes. The format follows Keep a Changelog; versions follow Semantic Versioning, and the crates move together on one version.

## Unreleased

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
