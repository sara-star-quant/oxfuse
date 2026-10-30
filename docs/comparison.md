# Comparison

What other projects say about themselves, taken from their own pages. It is not a benchmark and not a ranking, and there is no speed column. When a page says nothing on a topic, the cell says so; that does not mean the project lacks the feature. Spot a mistake? Open an issue and we will fix the cell.

Each cell links the page at the version we read, not the main branch. Our own cells link the v0.4.0 docs; the current claims are in [claims.md](claims.md) and the [contract matrix](contract.json).

| Project | Topic | What the page says | Source | Read on |
|---|---|---|---|---|
| Oxfuse 0.4.0 | Scope | README: "A tensor-graph compiler in Rust whose CPU results are the same bits on every platform it is tested on" (the platforms are listed there). It states it is not for production inference, large models, training or speed-critical CPU work. | <https://github.com/sara-star-quant/oxfuse/blob/v0.4.0/README.md> | 2026-10-27 |
| ONNX Runtime v1.31.1 | Scope | README: "ONNX Runtime is a cross-platform inference and training machine-learning accelerator". | <https://github.com/microsoft/onnxruntime/blob/v1.31.1/README.md> | 2026-10-27 |
| tract v0.23.9 | Scope | README: "Sonos' neural-network inference engine." It loads ONNX and NNEF models. | <https://github.com/sonos/tract/blob/v0.23.9/README.md> | 2026-10-27 |
| rten v0.27.0 | Scope | README: RTen "is a machine learning runtime" and supports models in ONNX format. | <https://github.com/robertknight/rten/blob/v0.27.0/README.md> | 2026-10-27 |
| candle 0.11.0 | Scope | README: "Candle is a minimalist ML framework for Rust with a focus on performance (including GPU support) and ease of use." | <https://github.com/huggingface/candle/blob/0.11.0/README.md> | 2026-10-27 |
| Burn v0.22.1 | Scope | README: "Burn is both a tensor library and a deep learning framework", "optimized for" numerical computing, training and inference. | <https://github.com/tracel-ai/burn/blob/v0.22.1/README.md> | 2026-10-27 |
| Oxfuse 0.4.0 | Licence | Apache-2.0 (single `LICENSE` file). | <https://github.com/sara-star-quant/oxfuse/blob/v0.4.0/LICENSE> | 2026-10-27 |
| ONNX Runtime v1.31.1 | Licence | README: "This project is licensed under the MIT License". | <https://github.com/microsoft/onnxruntime/blob/v1.31.1/README.md> | 2026-10-27 |
| tract v0.23.9 | Licence | README: "All original work is licensed under either of" the Apache License 2.0 or the MIT license. | <https://github.com/sonos/tract/blob/v0.23.9/README.md> | 2026-10-27 |
| rten v0.27.0 | Licence | The repository at the tag holds `LICENSE-APACHE.txt` and `LICENSE-MIT.txt`. | <https://github.com/robertknight/rten/tree/v0.27.0> | 2026-10-27 |
| candle 0.11.0 | Licence | README badges link `LICENSE-MIT` and `LICENSE-APACHE`. | <https://github.com/huggingface/candle/blob/0.11.0/README.md> | 2026-10-27 |
| Burn v0.22.1 | Licence | README: "Burn is distributed under the terms of both the MIT license and the Apache License (Version 2.0)." | <https://github.com/tracel-ai/burn/blob/v0.22.1/README.md> | 2026-10-27 |
| Oxfuse 0.4.0 | Devices | CPU by default. An optional wgpu backend is measured on the adapters listed in `docs/adapters.md`; no device is claimed as supported. | <https://github.com/sara-star-quant/oxfuse/blob/v0.4.0/docs/adapters.md> | 2026-10-27 |
| ONNX Runtime v1.31.1 | Devices | Not stated in the README linked. | <https://github.com/microsoft/onnxruntime/blob/v1.31.1/README.md> | 2026-10-27 |
| tract v0.23.9 | Devices | README says it runs on embedded CPUs, on GPUs, in the browser through WebAssembly and on workstations, and has a table of backends. | <https://github.com/sonos/tract/blob/v0.23.9/README.md> | 2026-10-27 |
| rten v0.27.0 | Devices | README: "RTen currently supports CPU inference only." SIMD for AVX2, AVX-512, Arm Neon and WebAssembly. | <https://github.com/robertknight/rten/blob/v0.27.0/README.md> | 2026-10-27 |
| candle 0.11.0 | Devices | README states a focus on performance "(including GPU support)" and lists backends under Features. | <https://github.com/huggingface/candle/blob/0.11.0/README.md> | 2026-10-27 |
| Burn v0.22.1 | Devices | README has a "Supported Backends" section with GPU and CPU backend tables. | <https://github.com/tracel-ai/burn/blob/v0.22.1/README.md> | 2026-10-27 |
| Oxfuse 0.4.0 | Training | Inference only, no training. | <https://github.com/sara-star-quant/oxfuse/blob/v0.4.0/README.md> | 2026-10-27 |
| ONNX Runtime v1.31.1 | Training | README: inference and training. | <https://github.com/microsoft/onnxruntime/blob/v1.31.1/README.md> | 2026-10-27 |
| tract v0.23.9 | Training | Described as an inference engine; training not mentioned in the README. | <https://github.com/sonos/tract/blob/v0.23.9/README.md> | 2026-10-27 |
| rten v0.27.0 | Training | README describes ML inference. | <https://github.com/robertknight/rten/blob/v0.27.0/README.md> | 2026-10-27 |
| candle 0.11.0 | Training | README Features list: "Model training." | <https://github.com/huggingface/candle/blob/0.11.0/README.md> | 2026-10-27 |
| Burn v0.22.1 | Training | README: "training and inference". | <https://github.com/tracel-ai/burn/blob/v0.22.1/README.md> | 2026-10-27 |
| Oxfuse 0.4.0 | Reproducibility | The reference and fast CPU backends return the same bits on five tested platforms; each operator has a stated contract per backend. `MatMul` is a fused multiply-add per step in ascending k, and `Sigmoid` uses an `exp` pinned by a hash over all 2^32 inputs. | <https://github.com/sara-star-quant/oxfuse/blob/v0.4.0/docs/semantics.md> | 2026-10-27 |
| ONNX Runtime v1.31.1 | Reproducibility | A session option `session.use_deterministic_compute`, documented in the header as "Enable or disable deterministic compute for a session." Nothing about results across hardware in that header. | <https://github.com/microsoft/onnxruntime/blob/v1.31.1/include/onnxruntime/core/session/onnxruntime_session_options_config_keys.h#L926> | 2026-10-27 |
| tract v0.23.9 | Reproducibility | Not stated in the README linked. | <https://github.com/sonos/tract/blob/v0.23.9/README.md> | 2026-10-27 |
| rten v0.27.0 | Reproducibility | Not stated in the README linked. | <https://github.com/robertknight/rten/blob/v0.27.0/README.md> | 2026-10-27 |
| candle 0.11.0 | Reproducibility | Not stated in the README linked. | <https://github.com/huggingface/candle/blob/0.11.0/README.md> | 2026-10-27 |
| Burn v0.22.1 | Reproducibility | Not stated in the README linked. | <https://github.com/tracel-ai/burn/blob/v0.22.1/README.md> | 2026-10-27 |
| RepDL (paper) | Reproducibility | Title: "RepDL: Bit-level Reproducible Deep Learning Training and Inference" (arXiv preprint, 10 October 2025). A library for PyTorch, not a runtime for ONNX graphs. | <https://arxiv.org/abs/2510.09180> | 2026-10-27 |
| Oxfuse 0.4.0 | ONNX operators | Seven operators and an importer for a float32 subset, opsets 13 to 21, every other feature a named error. | <https://github.com/sara-star-quant/oxfuse/blob/v0.4.0/docs/onnx.md> | 2026-10-27 |
| ONNX Runtime v1.31.1 | ONNX operators | Not stated in the README linked. | <https://github.com/microsoft/onnxruntime/blob/v1.31.1/README.md> | 2026-10-27 |
| tract v0.23.9 | ONNX operators | Not stated in the README linked. | <https://github.com/sonos/tract/blob/v0.23.9/README.md> | 2026-10-27 |
| rten v0.27.0 | ONNX operators | README: "RTen supports most standard ONNX operators", with a tracking issue for details, and some operators from the ONNX Runtime contrib set. | <https://github.com/robertknight/rten/blob/v0.27.0/README.md> | 2026-10-27 |
| candle 0.11.0 | ONNX operators | Not stated in the README linked. | <https://github.com/huggingface/candle/blob/0.11.0/README.md> | 2026-10-27 |
| Burn v0.22.1 | ONNX operators | README lists an ONNX import project (`burn-onnx`) that turns ONNX models into Rust code. | <https://github.com/tracel-ai/burn/blob/v0.22.1/README.md> | 2026-10-27 |
| Oxfuse 0.5.1 | Latest release | 0.5.1, 2026-10-30. | <https://crates.io/api/v1/crates/oxfuse/0.5.1> | 2026-10-30 |
| ONNX Runtime | Latest release | v1.31.1, 2026-10-26. | <https://github.com/microsoft/onnxruntime/releases/tag/v1.31.1> | 2026-10-27 |
| tract | Latest release | v0.23.9, 2026-10-15. | <https://github.com/sonos/tract/releases/tag/v0.23.9> | 2026-10-27 |
| rten | Latest release | 0.27.0, 2026-10-02. | <https://crates.io/api/v1/crates/rten/0.27.0> | 2026-10-27 |
| candle | Latest release | candle-core 0.11.0, 2026-06-26. | <https://crates.io/api/v1/crates/candle-core/0.11.0> | 2026-10-27 |
| Burn | Latest release | v0.22.1, 2026-10-20. | <https://github.com/tracel-ai/burn/releases/tag/v0.22.1> | 2026-10-27 |
