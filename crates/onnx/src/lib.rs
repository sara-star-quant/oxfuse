//! ONNX import for oxfuse: decode a model, check its structure, lower the
//! supported float32 subset into the oxfuse IR. Anything outside the subset
//! is an error that names the node and the reason, never an approximation.
//!
//! Part of [oxfuse](https://github.com/sara-star-quant/oxfuse). The imported
//! graph runs on any oxfuse backend:
//!
//! ```no_run
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let bytes = std::fs::read("model.onnx")?;
//! let graph = oxfuse_onnx::import(&bytes, &oxfuse_onnx::ImportOptions::default())?;
//! # let _ = graph;
//! # Ok(())
//! # }
//! ```

mod decode;
mod error;
mod lower;
pub mod model;
pub mod proto;
mod tensor;

use oxfuse_core::Limits;
use std::collections::BTreeMap;

pub use decode::decode_model;
pub use error::OnnxError;
pub use tensor::{Dense, read_tensor};

/// Knobs for an import.
#[derive(Debug, Clone, Default)]
pub struct ImportOptions {
    pub limits: Limits,
    /// Values for named dimensions (`--dim NAME=VALUE`).
    pub dims: BTreeMap<String, usize>,
}

/// Import a serialized ONNX model into an oxfuse graph.
pub fn import(bytes: &[u8], opts: &ImportOptions) -> Result<oxfuse_core::Graph, OnnxError> {
    let model = decode_model(bytes, &opts.limits)?;
    lower::lower(model::check_model(model, opts)?, opts)
}
