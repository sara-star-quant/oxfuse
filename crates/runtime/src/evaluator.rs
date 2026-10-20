//! Constant evaluation with the reference kernels, so a folded constant is
//! bit-identical to the value the run-time path computes.

use crate::kernels;
use oxfuse_core::passes::ConstEvaluator;
use oxfuse_core::{Error, Op, TensorType};

pub struct ReferenceEvaluator;

impl ConstEvaluator for ReferenceEvaluator {
    fn eval(&self, op: &Op, inputs: &[(&TensorType, &[f32])]) -> Result<Vec<f32>, Error> {
        let data: Vec<&[f32]> = inputs.iter().map(|(_, data)| *data).collect();
        let shapes: Vec<&[usize]> = inputs.iter().map(|(ty, _)| ty.shape.as_slice()).collect();
        Ok(kernels::apply(op, &data, &shapes))
    }
}
