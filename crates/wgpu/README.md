# oxfuse-wgpu

Part of [oxfuse](https://github.com/sara-star-quant/oxfuse), a small tensor graph compiler in Rust. Start with the [repository README](https://github.com/sara-star-quant/oxfuse#readme) if you landed here from a search.

An optional GPU backend for [oxfuse](https://github.com/sara-star-quant/oxfuse) tensor graphs, written as WGSL compute kernels on wgpu. Tensors stay on the device between ops. The index arithmetic of the kernels is checked with Kani bounded model checking, and every kernel has a CPU model that is tested against it. Results are exact where the shading language allows and bounded elsewhere; the contract is in `docs/semantics.md` and the devices it was measured on are in `docs/adapters.md`, both in the repository.

The command-line tool builds it with `--features wgpu` (`oxfuse run --backend wgpu`, `oxfuse gpu-selftest`). `gpu-selftest` measures any adapter against the reference backend.

A complete example, with the bound check and the device-resident path, is `examples/matmul.rs`:

```sh
cargo run --release -p oxfuse-wgpu --example matmul
```

The GPU backend runs on wgpu. It is measured on an Apple M1 Pro (Metal), a Tesla T4 (NVIDIA, Vulkan) and on software adapters. AMD and Intel Arc cards should work through Vulkan or DX12 but are untested. If you try your card, run `oxfuse gpu-selftest` and share the report.

Experimental. No device or vendor support is claimed.

Licensed under Apache-2.0.
