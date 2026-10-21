//! Typed tensor-graph IR: types, shape inference, builder and verifier, plus
//! the exact optimization passes (constant folding, common-subexpression
//! elimination, dead-code elimination).
//!
//! Part of [oxfuse](https://github.com/sara-star-quant/oxfuse), a small tensor
//! graph compiler whose CPU backends return identical bits. Start here to
//! build and check a graph; `oxfuse-runtime` runs it.
//!
//! ```
//! use oxfuse_core::{GraphBuilder, Limits, Op};
//!
//! # fn main() -> Result<(), oxfuse_core::Error> {
//! let mut b = GraphBuilder::new(Limits::default());
//! let x = b.input("x", vec![2, 3])?;
//! let w = b.constant("w", vec![3, 2], vec![0.5, -1.0, 1.5, 2.0, -0.5, 1.0])?;
//! let m = b.node("m", Op::MatMul, &[x, w])?;
//! let y = b.node("y", Op::Relu, &[m])?;
//! // `finish` runs the verifier: shapes and types are re-checked here.
//! let graph = b.finish(&[y])?;
//! assert_eq!(graph.outputs().len(), 1);
//! # Ok(())
//! # }
//! ```
//!
//! Not for: general ML work. The op set is seven float32 ops.

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
