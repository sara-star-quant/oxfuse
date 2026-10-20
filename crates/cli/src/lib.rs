//! Command-line front end: JSON graph format, loader and commands.

mod commands;
mod error;
mod format;
mod loader;

pub use commands::{BackendKind, RunFailure, RunOptions, RunOutput, check, inspect, run, run_with};
pub use error::CliError;
pub use format::{NamedTensorFile, OutputsFile, TensorFile};
pub use loader::load_graph;
