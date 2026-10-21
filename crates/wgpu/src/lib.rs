//! A wgpu compute backend for oxfuse graphs (Vulkan, Metal, DX12).
//!
//! Part of [oxfuse](https://github.com/sara-star-quant/oxfuse). Results are
//! exact where the shading language allows and inside a documented bound
//! elsewhere; `oxfuse gpu-selftest` measures any adapter against the
//! reference backend. See `examples/matmul.rs` for a complete run.

mod bounds;
mod config;
mod context;
mod exec;
pub mod model;
mod proofs;
mod selftest;
pub mod shaders;

pub use bounds::{
    SIGMOID_ULP_BOUND, gpu_matmul_bound, matmul_bound, sigmoid_matches, sigmoid_ulp_bound_for,
    ulp_distance,
};
pub use config::{
    DeviceCaps, KERNEL_STORAGE_BUFFERS, KernelConfig, MAX_TENSOR_ELEMENTS, check_tensor,
    element_index, grid_fold,
};
pub use context::{Context, LimitsProfile};
pub use exec::{DeviceTensor, WgpuBackend, WgpuExecutable};
pub use selftest::{SelfTestReport, run_selftest};
pub use wgpu::AdapterInfo;
