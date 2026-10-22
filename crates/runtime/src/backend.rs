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

    /// Runs the graph once per element of `batches` and returns the results in
    /// the same order; every result equals what `run` returns for that element.
    /// The default runs them one after another. The GPU backend overrides it to
    /// submit the whole batch at once and wait once, which hides the per-run
    /// wait; its memory use grows with the batch.
    fn run_many(&self, batches: &[Vec<Tensor>]) -> Result<Vec<Vec<Tensor>>, Error> {
        batches.iter().map(|inputs| self.run(inputs)).collect()
    }
}
