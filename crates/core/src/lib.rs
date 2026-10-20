//! Typed tensor-graph IR: types, shape inference, builder and verifier.

mod builder;
mod dump;
mod error;
mod graph;
mod limits;
mod op;
pub mod passes;
pub mod shape;
mod types;
mod verify;

pub use builder::GraphBuilder;
pub use error::Error;
pub use graph::{Graph, Source, Value};
pub use limits::Limits;
pub use op::Op;
pub use types::{DType, TensorType, ValueId};
