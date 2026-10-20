//! Tensors, execution plans and the CPU backends.

mod backend;
mod cpu;
mod evaluator;
mod kernels;
mod plan;
mod reference;
mod tensor;

pub use backend::{Backend, Executable};
pub use cpu::CpuBackend;
pub use evaluator::ReferenceEvaluator;
pub use plan::{InputBinding, Plan, Step};
pub use reference::ReferenceBackend;
pub use tensor::Tensor;
