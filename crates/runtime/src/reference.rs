//! The oracle backend: naive, obviously correct, never optimized.

use crate::{Backend, Executable, Plan, Tensor, kernels};
use oxfuse_core::{Error, Graph};
use std::sync::Arc;

pub struct ReferenceBackend;

impl Backend for ReferenceBackend {
    fn compile(&self, graph: &Graph) -> Result<Box<dyn Executable>, Error> {
        Ok(Box::new(ReferenceExecutable {
            plan: Plan::build(graph),
        }))
    }
}

struct ReferenceExecutable {
    plan: Plan,
}

enum Slot {
    Empty,
    Shared(Arc<[f32]>),
    Owned(Vec<f32>),
}

impl Slot {
    fn data(&self) -> &[f32] {
        match self {
            Slot::Empty => &[],
            Slot::Shared(data) => data,
            Slot::Owned(data) => data,
        }
    }
}

impl Executable for ReferenceExecutable {
    fn run(&self, inputs: &[Tensor]) -> Result<Vec<Tensor>, Error> {
        let plan = &self.plan;
        if inputs.len() != plan.inputs.len() {
            return Err(Error::InputCount {
                expected: plan.inputs.len(),
                found: inputs.len(),
            });
        }
        let mut slots: Vec<Slot> = (0..plan.slot_shapes.len()).map(|_| Slot::Empty).collect();
        for (binding, tensor) in plan.inputs.iter().zip(inputs) {
            if tensor.shape() != binding.shape.as_slice() {
                return Err(Error::InputShape {
                    name: binding.name.clone(),
                    expected: binding.shape.clone(),
                    found: tensor.shape().to_vec(),
                });
            }
            slots[binding.slot] = Slot::Owned(tensor.data().to_vec());
        }
        for (slot, data) in &plan.constants {
            slots[*slot] = Slot::Shared(Arc::clone(data));
        }
        for step in &plan.steps {
            let out = {
                let inputs: Vec<&[f32]> = step.inputs.iter().map(|&s| slots[s].data()).collect();
                let shapes: Vec<&[usize]> = step
                    .inputs
                    .iter()
                    .map(|&s| plan.slot_shapes[s].as_slice())
                    .collect();
                kernels::apply(&step.op, &inputs, &shapes)
            };
            slots[step.output] = Slot::Owned(out);
        }
        plan.outputs
            .iter()
            .map(|&slot| Tensor::new(plan.slot_shapes[slot].clone(), slots[slot].data().to_vec()))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxfuse_core::{GraphBuilder, Limits, Op};

    fn relu_graph(shape: Vec<usize>) -> Graph {
        let mut b = GraphBuilder::new(Limits::default());
        let x = b.input("x", shape).unwrap();
        let y = b.node("y", Op::Relu, &[x]).unwrap();
        b.finish(&[y]).unwrap()
    }

    fn run(graph: &Graph, inputs: &[Tensor]) -> Result<Vec<Tensor>, Error> {
        ReferenceBackend.compile(graph)?.run(inputs)
    }

    #[test]
    fn relu_clamps_negatives() {
        let input = Tensor::new(vec![2, 3], vec![-1.5, 0.0, 2.5, -3.0, 3.0, -0.5]).unwrap();
        let out = run(&relu_graph(vec![2, 3]), &[input]).unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].shape(), &[2, 3]);
        assert_eq!(out[0].data(), &[0.0, 0.0, 2.5, 0.0, 3.0, 0.0]);
    }

    #[test]
    fn relu_of_negative_zero_is_positive_zero() {
        let input = Tensor::new(vec![1], vec![-0.0]).unwrap();
        let out = run(&relu_graph(vec![1]), &[input]).unwrap();
        assert_eq!(out[0].data()[0].to_bits(), 0.0f32.to_bits());
    }

    #[test]
    fn relu_propagates_nan() {
        let input = Tensor::new(vec![2], vec![f32::NAN, f32::NEG_INFINITY]).unwrap();
        let out = run(&relu_graph(vec![2]), &[input]).unwrap();
        assert!(out[0].data()[0].is_nan());
        assert_eq!(out[0].data()[1], 0.0);
    }

    #[test]
    fn relu_passes_positive_infinity() {
        let input = Tensor::new(vec![1], vec![f32::INFINITY]).unwrap();
        let out = run(&relu_graph(vec![1]), &[input]).unwrap();
        assert_eq!(out[0].data()[0], f32::INFINITY);
    }

    #[test]
    fn scalar_input_works() {
        let input = Tensor::new(vec![], vec![-4.0]).unwrap();
        let out = run(&relu_graph(vec![]), &[input]).unwrap();
        assert_eq!(out[0].shape(), &[] as &[usize]);
        assert_eq!(out[0].data(), &[0.0]);
    }

    #[test]
    fn constant_only_graph_runs_with_no_inputs() {
        let mut b = GraphBuilder::new(Limits::default());
        let c = b.constant("c", vec![2], vec![-1.0, 2.0]).unwrap();
        let r = b.node("r", Op::Relu, &[c]).unwrap();
        let g = b.finish(&[r]).unwrap();
        let out = run(&g, &[]).unwrap();
        assert_eq!(out[0].data(), &[0.0, 2.0]);
    }

    #[test]
    fn output_can_be_an_input() {
        let mut b = GraphBuilder::new(Limits::default());
        let x = b.input("x", vec![2]).unwrap();
        let g = b.finish(&[x]).unwrap();
        let input = Tensor::new(vec![2], vec![-1.0, 1.0]).unwrap();
        let out = run(&g, std::slice::from_ref(&input)).unwrap();
        assert_eq!(out[0], input);
    }

    #[test]
    fn same_value_can_be_output_twice() {
        let mut b = GraphBuilder::new(Limits::default());
        let x = b.input("x", vec![1]).unwrap();
        let y = b.node("y", Op::Relu, &[x]).unwrap();
        let g = b.finish(&[y, y]).unwrap();
        let out = run(&g, &[Tensor::new(vec![1], vec![-2.0]).unwrap()]).unwrap();
        assert_eq!(out.len(), 2);
        assert_eq!(out[0], out[1]);
    }

    #[test]
    fn wrong_input_count_is_rejected() {
        let err = run(&relu_graph(vec![1]), &[]).unwrap_err();
        assert_eq!(
            err,
            Error::InputCount {
                expected: 1,
                found: 0
            }
        );
    }

    #[test]
    fn wrong_input_shape_is_rejected() {
        let input = Tensor::new(vec![3], vec![1.0, 2.0, 3.0]).unwrap();
        let err = run(&relu_graph(vec![2]), &[input]).unwrap_err();
        assert_eq!(
            err,
            Error::InputShape {
                name: "x".into(),
                expected: vec![2],
                found: vec![3]
            }
        );
    }

    #[test]
    fn executable_is_reusable() {
        let exe = ReferenceBackend.compile(&relu_graph(vec![1])).unwrap();
        let a = exe
            .run(&[Tensor::new(vec![1], vec![-1.0]).unwrap()])
            .unwrap();
        let b = exe
            .run(&[Tensor::new(vec![1], vec![5.0]).unwrap()])
            .unwrap();
        assert_eq!(a[0].data(), &[0.0]);
        assert_eq!(b[0].data(), &[5.0]);
    }

    #[test]
    fn add_then_scale_by_a_scalar_constant() {
        let mut b = GraphBuilder::new(Limits::default());
        let a = b.input("a", vec![2, 2]).unwrap();
        let c = b.input("b", vec![2, 2]).unwrap();
        let two = b.constant("two", vec![], vec![2.0]).unwrap();
        let sum = b.node("sum", Op::Add, &[a, c]).unwrap();
        let scaled = b.node("scaled", Op::Mul, &[sum, two]).unwrap();
        let g = b.finish(&[sum, scaled]).unwrap();
        let a = Tensor::new(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let c = Tensor::new(vec![2, 2], vec![10.0, 20.0, 30.0, 40.0]).unwrap();
        let out = run(&g, &[a, c]).unwrap();
        assert_eq!(out[0].data(), &[11.0, 22.0, 33.0, 44.0]);
        assert_eq!(out[1].data(), &[22.0, 44.0, 66.0, 88.0]);
    }

    #[test]
    fn matmul_graph_runs() {
        let mut b = GraphBuilder::new(Limits::default());
        let x = b.input("x", vec![2, 2]).unwrap();
        let w = b
            .constant("w", vec![2, 2], vec![5.0, 6.0, 7.0, 8.0])
            .unwrap();
        let y = b.node("y", Op::MatMul, &[x, w]).unwrap();
        let g = b.finish(&[y]).unwrap();
        let input = Tensor::new(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let out = run(&g, &[input]).unwrap();
        assert_eq!(out[0].data(), &[19.0, 22.0, 43.0, 50.0]);
    }
}
