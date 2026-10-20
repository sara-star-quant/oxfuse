//! Constant folding: a node whose inputs are all constants is evaluated now
//! and replaced by a constant of the same name. The evaluator is injected so
//! core never depends on a backend; with the reference kernels the folded
//! value is bit-identical to what the run-time path would compute.

use super::rebuild::Rebuilder;
use super::{Pass, PassOutput};
use crate::shape::checked_num_elements;
use crate::{Error, Graph, Limits, Op, Source, TensorType, ValueId};
use std::sync::Arc;

pub trait ConstEvaluator {
    /// Evaluates `op` over constant inputs. Called only for nodes of a
    /// verified graph, so the input types satisfy the op's shape rule.
    fn eval(&self, op: &Op, inputs: &[(&TensorType, &[f32])]) -> Result<Vec<f32>, Error>;
}

pub struct ConstantFolding {
    evaluator: Arc<dyn ConstEvaluator>,
    limits: Limits,
    max_elements: usize,
}

impl ConstantFolding {
    /// Nodes with a larger output or input stay as nodes: folding them would
    /// spend compile-time memory and time that the run-time path would spend
    /// anyway (and `inspect` never runs the graph at all).
    pub const DEFAULT_MAX_ELEMENTS: usize = 1 << 20;

    pub fn new(evaluator: Arc<dyn ConstEvaluator>, limits: Limits) -> Self {
        Self {
            evaluator,
            limits,
            max_elements: Self::DEFAULT_MAX_ELEMENTS,
        }
    }

    pub fn with_max_elements(mut self, max_elements: usize) -> Self {
        self.max_elements = max_elements;
        self
    }

    /// The cap applies to the output and to every input. Bounding the inputs
    /// bounds the work too: a MatMul whose operands and result each have at
    /// most `n` elements costs at most `n^1.5` multiply-adds.
    fn within_cap(&self, shape: &[usize]) -> bool {
        checked_num_elements(shape).is_some_and(|n| n <= self.max_elements)
    }
}

impl Pass for ConstantFolding {
    fn name(&self) -> &'static str {
        "fold"
    }

    fn run(&self, graph: &Graph) -> Result<PassOutput, Error> {
        let mut rebuild = Rebuilder::new(graph, &self.limits);
        let mut constants: Vec<Option<Arc<[f32]>>> = vec![None; graph.values().len()];
        let mut folded = 0;
        for (index, value) in graph.values().iter().enumerate() {
            let old = ValueId(index as u32);
            match &value.source {
                Source::Constant(data) => {
                    constants[index] = Some(Arc::clone(data));
                    rebuild.copy(old)?;
                }
                Source::Node { op, inputs }
                    if inputs.iter().all(|id| constants[id.index()].is_some())
                        && self.within_cap(value.ty.shape.as_slice())
                        && inputs
                            .iter()
                            .all(|id| self.within_cap(&graph.value(*id).ty.shape)) =>
                {
                    let args: Vec<(&TensorType, &[f32])> = inputs
                        .iter()
                        .map(|id| {
                            let data = constants[id.index()].as_deref();
                            (
                                &graph.value(*id).ty,
                                data.expect("all inputs were checked to be constants"),
                            )
                        })
                        .collect();
                    let data: Arc<[f32]> = self.evaluator.eval(op, &args)?.into();
                    rebuild.replace_with_constant(old, Arc::clone(&data))?;
                    constants[index] = Some(data);
                    folded += 1;
                }
                _ => {
                    rebuild.copy(old)?;
                }
            }
        }
        Ok(PassOutput {
            graph: rebuild.finish()?,
            changed: folded,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{GraphBuilder, Op, Source, ValueId};

    /// Just enough kernels for the tests; the runtime has the real ones.
    struct Eval;
    impl ConstEvaluator for Eval {
        fn eval(&self, op: &Op, inputs: &[(&TensorType, &[f32])]) -> Result<Vec<f32>, Error> {
            match op {
                Op::Relu => Ok(inputs[0]
                    .1
                    .iter()
                    .map(|&x| if x > 0.0 { x } else { 0.0 })
                    .collect()),
                Op::Add => Ok(inputs[0]
                    .1
                    .iter()
                    .zip(inputs[1].1)
                    .map(|(a, b)| a + b)
                    .collect()),
                _ => Err(Error::Verify(
                    "test evaluator does not support this op".into(),
                )),
            }
        }
    }

    fn fold() -> ConstantFolding {
        ConstantFolding::new(std::sync::Arc::new(Eval), Limits::default())
    }

    fn constant_data(graph: &Graph, name: &str) -> Vec<f32> {
        let id = (0..graph.values().len())
            .map(|i| ValueId(i as u32))
            .find(|&id| graph.label(id) == name)
            .unwrap();
        match &graph.value(id).source {
            Source::Constant(data) => data.to_vec(),
            other => panic!("`{name}` is not a constant: {other:?}"),
        }
    }

    fn builder() -> GraphBuilder {
        GraphBuilder::new(Limits::default())
    }

    #[test]
    fn folds_a_constant_chain_and_leaves_input_dependent_nodes() {
        let mut b = builder();
        let x = b.input("x", vec![2]).unwrap();
        let a = b.constant("a", vec![2], vec![1.0, -2.0]).unwrap();
        let c = b.constant("b", vec![2], vec![3.0, 4.0]).unwrap();
        let s = b.node("s", Op::Add, &[a, c]).unwrap();
        let r = b.node("r", Op::Relu, &[s]).unwrap();
        let y = b.node("y", Op::Add, &[x, r]).unwrap();
        let g = b.finish(&[y]).unwrap();
        let out = fold().run(&g).unwrap();
        assert_eq!(out.changed, 2);
        assert_eq!(constant_data(&out.graph, "s"), [4.0, 2.0]);
        assert_eq!(constant_data(&out.graph, "r"), [4.0, 2.0]);
        assert!(matches!(
            out.graph.value(out.graph.outputs()[0]).source,
            Source::Node { op: Op::Add, .. }
        ));
        assert_eq!(out.graph.values().len(), g.values().len());
    }

    #[test]
    fn folded_output_keeps_its_name() {
        let mut b = builder();
        let a = b.constant("a", vec![1], vec![-5.0]).unwrap();
        let r = b.node("r", Op::Relu, &[a]).unwrap();
        let g = b.finish_named(vec![("result".to_owned(), r)]).unwrap();
        let out = fold().run(&g).unwrap();
        assert_eq!(out.graph.output_names(), ["result"]);
        assert_eq!(constant_data(&out.graph, "r"), [0.0]);
    }

    #[test]
    fn nothing_to_fold_rebuilds_the_same_graph() {
        let mut b = builder();
        let x = b.input("x", vec![2]).unwrap();
        let y = b.node("y", Op::Relu, &[x]).unwrap();
        let g = b.finish(&[y]).unwrap();
        let out = fold().run(&g).unwrap();
        assert_eq!(out.changed, 0);
        assert_eq!(out.graph, g);
    }

    #[test]
    fn non_finite_results_fold_like_runtime() {
        let mut b = builder();
        let pos = b.constant("pos", vec![1], vec![f32::INFINITY]).unwrap();
        let neg = b.constant("neg", vec![1], vec![f32::NEG_INFINITY]).unwrap();
        let sum = b.node("sum", Op::Add, &[pos, neg]).unwrap();
        let g = b.finish(&[sum]).unwrap();
        let out = fold().run(&g).unwrap();
        assert_eq!(out.changed, 1);
        assert!(constant_data(&out.graph, "sum")[0].is_nan());
    }

    #[test]
    fn large_results_are_left_for_run_time() {
        let mut b = builder();
        let a = b
            .constant("a", vec![4], vec![1.0, -1.0, 2.0, -2.0])
            .unwrap();
        let r = b.node("r", Op::Relu, &[a]).unwrap();
        let g = b.finish(&[r]).unwrap();
        let small = fold().with_max_elements(3);
        let out = small.run(&g).unwrap();
        assert_eq!(out.changed, 0);
        assert_eq!(out.graph, g);
        let exact = fold().with_max_elements(4);
        assert_eq!(exact.run(&g).unwrap().changed, 1);
    }

    #[test]
    fn evaluator_errors_propagate() {
        let mut b = builder();
        let a = b.constant("a", vec![1], vec![1.0]).unwrap();
        let s = b.node("s", Op::Sigmoid, &[a]).unwrap();
        let g = b.finish(&[s]).unwrap();
        let err = fold().run(&g).err().unwrap();
        assert!(matches!(err, Error::Verify(_)));
    }

    #[test]
    fn wrong_length_results_are_rejected() {
        struct Short;
        impl ConstEvaluator for Short {
            fn eval(&self, _: &Op, _: &[(&TensorType, &[f32])]) -> Result<Vec<f32>, Error> {
                Ok(vec![0.0])
            }
        }
        let mut b = builder();
        let a = b.constant("a", vec![2], vec![1.0, 2.0]).unwrap();
        let r = b.node("r", Op::Relu, &[a]).unwrap();
        let g = b.finish(&[r]).unwrap();
        let folding = ConstantFolding::new(std::sync::Arc::new(Short), Limits::default());
        let err = folding.run(&g).err().unwrap();
        assert!(matches!(
            err,
            Error::DataLength {
                expected: 2,
                found: 1,
                ..
            }
        ));
    }

    #[test]
    fn large_inputs_are_not_folded_even_when_the_output_is_small() {
        // [1,4] x [4,1] has a one-element output but costs work proportional
        // to the inner dimension. The test evaluator rejects MatMul, so any
        // attempt to fold it would surface as an error.
        let mut b = builder();
        let a = b.constant("a", vec![1, 4], vec![1.0; 4]).unwrap();
        let c = b.constant("c", vec![4, 1], vec![1.0; 4]).unwrap();
        let p = b.node("p", Op::MatMul, &[a, c]).unwrap();
        let g = b.finish(&[p]).unwrap();
        let capped = fold().with_max_elements(3);
        let out = capped.run(&g).unwrap();
        assert_eq!(out.changed, 0);
        assert_eq!(out.graph, g);
    }
}
