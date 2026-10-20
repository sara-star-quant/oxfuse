//! The portable backend interface. Kept minimal on purpose: it assumes host
//! tensors, so a device-memory backend would need to revise it.

use crate::Tensor;
use oxfuse_core::{Error, Graph};

pub trait Backend {
    fn compile(&self, graph: &Graph) -> Result<Box<dyn Executable>, Error>;
}

pub trait Executable {
    /// `inputs` follow the graph's declared input order; the result follows
    /// the graph's output order.
    fn run(&self, inputs: &[Tensor]) -> Result<Vec<Tensor>, Error>;
}
