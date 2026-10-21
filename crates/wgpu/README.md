# oxfuse-wgpu

An optional GPU backend for [oxfuse](https://github.com/sara-star-quant/oxfuse) tensor graphs, written as WGSL compute kernels on wgpu. Tensors stay on the device between ops. Results are exact where the shading language allows and bounded elsewhere; the contract is in `docs/semantics.md` and the devices it was measured on are in `docs/adapters.md`, both in the repository.

The command-line tool builds it with `--features wgpu` (`oxfuse run --backend wgpu`, `oxfuse gpu-selftest`). `gpu-selftest` measures any adapter against the reference backend.

Experimental. No device or vendor support is claimed.

Licensed under Apache-2.0.
