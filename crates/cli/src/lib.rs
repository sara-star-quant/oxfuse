//! Command-line front end: JSON graph format, loader and commands. This crate
//! is the `oxfuse` binary; most users want `cargo install oxfuse` and the
//! [repository README](https://github.com/sara-star-quant/oxfuse).

mod commands;
mod error;
mod format;
mod loader;
mod manifest;

#[cfg(feature = "wgpu")]
pub use commands::gpu_selftest;
pub use commands::{
    BackendKind, Dims, RunFailure, RunOptions, RunOutput, check, check_with, dims_from_args,
    inspect, run, run_with,
};
pub use error::CliError;
pub use format::{NamedTensorFile, OutputsFile, TensorFile};
pub use loader::load_graph;
