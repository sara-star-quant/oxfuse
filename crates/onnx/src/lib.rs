//! ONNX import for oxfuse: decode a model, check its structure, lower the
//! supported float32 subset into the oxfuse IR. Anything outside the subset
//! is an error that names the node and the reason, never an approximation.
//!
//! Part of [oxfuse](https://github.com/sara-star-quant/oxfuse). Usable on its
//! own: it depends only on `oxfuse-core`, and the imported graph can be
//! inspected, optimized or handed to any oxfuse backend.
//!
//! ```
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! use oxfuse_onnx::proto::*;
//! use prost::Message;
//!
//! // A one-node model, Relu on a [1, 2] input, built in memory.
//! let dim = |v| Dimension { value: Some(dimension::Value::DimValue(v)) };
//! let model = ModelProto {
//!     ir_version: Some(9),
//!     opset_import: vec![OperatorSetIdProto { domain: String::new(), version: Some(13) }],
//!     graph: Some(GraphProto {
//!         node: vec![NodeProto {
//!             op_type: "Relu".into(),
//!             input: vec!["x".into()],
//!             output: vec!["y".into()],
//!             ..NodeProto::default()
//!         }],
//!         input: vec![ValueInfoProto {
//!             name: "x".into(),
//!             r#type: Some(TypeProto {
//!                 tensor_type: Some(TypeTensor {
//!                     elem_type: data_type::FLOAT,
//!                     shape: Some(TensorShapeProto { dim: vec![dim(1), dim(2)] }),
//!                 }),
//!             }),
//!         }],
//!         output: vec![ValueInfoProto { name: "y".into(), r#type: None }],
//!         ..GraphProto::default()
//!     }),
//! };
//!
//! // Import needs only `oxfuse-core` types; no backend is involved.
//! let graph = oxfuse_onnx::import(&model.encode_to_vec(), &oxfuse_onnx::ImportOptions::default())?;
//! assert_eq!(graph.inputs().len(), 1);
//! assert_eq!(graph.outputs().len(), 1);
//! # Ok(())
//! # }
//! ```

/// Compiles and runs the README examples as doctests.
#[cfg(doctest)]
#[doc = include_str!("../README.md")]
struct ReadmeDoctests;

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
