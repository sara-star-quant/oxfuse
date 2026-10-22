//! Tensors, execution plans and the CPU backends: a plain reference backend
//! and a fast backend that returns exactly the same bits (the `parallel`
//! feature adds threads and still does not change a bit).
//!
//! Part of [oxfuse](https://github.com/sara-star-quant/oxfuse). Build a graph
//! with `oxfuse-core`, then compile and run it:
//!
//! ```
//! use oxfuse_core::{GraphBuilder, Limits, Op};
//! use oxfuse_runtime::{Backend, CpuBackend, Tensor};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let mut b = GraphBuilder::new(Limits::default());
//! let x = b.input("x", vec![2])?;
//! let y = b.node("y", Op::Relu, &[x])?;
//! let graph = b.finish(&[y])?;
//!
//! let input = Tensor::new(vec![2], vec![-1.0, 2.0])?;
//! let out = CpuBackend.compile(&graph)?.run(&[input])?;
//! assert_eq!(out[0].data(), &[0.0, 2.0]);
//! # Ok(())
//! # }
//! ```

mod backend;
mod cpu;
mod evaluator;
mod kernels;
mod math;
mod plan;
mod reference;
mod tensor;

pub use backend::{Backend, Executable};
pub use cpu::CpuBackend;
pub use evaluator::ReferenceEvaluator;
pub use math::sigmoid;
pub use plan::{InputBinding, Plan, Step};
pub use reference::ReferenceBackend;
pub use tensor::Tensor;
