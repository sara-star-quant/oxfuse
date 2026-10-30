//! Builds a graph, inferring each node's type as it is added, so a graph
//! with an ill-typed node cannot be constructed.

use crate::shape::{check_shape, checked_num_elements, infer};
use crate::{Error, Graph, Limits, Op, Source, TensorType, Value, ValueId};
use std::collections::HashMap;
use std::sync::Arc;

pub struct GraphBuilder {
    limits: Limits,
    values: Vec<Value>,
    inputs: Vec<ValueId>,
    names: HashMap<String, ValueId>,
    total_elements: usize,
}

impl GraphBuilder {
    pub fn new(limits: Limits) -> Self {
        Self {
            limits,
            values: Vec::new(),
            inputs: Vec::new(),
            names: HashMap::new(),
            total_elements: 0,
        }
    }

    pub fn lookup(&self, name: &str) -> Option<ValueId> {
        self.names.get(name).copied()
    }

    pub fn input(&mut self, name: &str, shape: Vec<usize>) -> Result<ValueId, Error> {
        check_shape(name, &shape, &self.limits)?;
        let id = self.push(name, TensorType::f32(shape), Source::Input)?;
        self.inputs.push(id);
        Ok(id)
    }

    pub fn constant(
        &mut self,
        name: &str,
        shape: Vec<usize>,
        data: Vec<f32>,
    ) -> Result<ValueId, Error> {
        self.constant_shared(name, shape, data.into())
    }

    /// Like `constant`, but shares the buffer, so rebuilding a graph never
    /// copies weights.
    pub fn constant_shared(
        &mut self,
        name: &str,
        shape: Vec<usize>,
        data: Arc<[f32]>,
    ) -> Result<ValueId, Error> {
        let expected = check_shape(name, &shape, &self.limits)?;
        if data.len() != expected {
            return Err(Error::DataLength {
                name: name.to_owned(),
                shape,
                expected,
                found: data.len(),
            });
        }
        self.push(name, TensorType::f32(shape), Source::Constant(data))
    }

    pub fn node(&mut self, name: &str, op: Op, inputs: &[ValueId]) -> Result<ValueId, Error> {
        let mut types = Vec::with_capacity(inputs.len());
        for id in inputs {
            let value = self.values.get(id.index()).ok_or_else(|| {
                Error::Verify(format!("node `{name}` uses undefined value %{}", id.0))
            })?;
            types.push(&value.ty);
        }
        let ty = infer(name, &op, &types)?;
        check_shape(name, &ty.shape, &self.limits)?;
        self.push(
            name,
            ty,
            Source::Node {
                op,
                inputs: inputs.to_vec(),
            },
        )
    }

    pub fn finish(self, outputs: &[ValueId]) -> Result<Graph, Error> {
        let names = outputs
            .iter()
            .map(|id| match self.values.get(id.index()) {
                Some(Value {
                    name: Some(name), ..
                }) => (name.clone(), *id),
                _ => (format!("%{}", id.0), *id),
            })
            .collect();
        self.finish_named(names)
    }

    pub fn finish_named(self, outputs: Vec<(String, ValueId)>) -> Result<Graph, Error> {
        let (output_names, outputs): (Vec<String>, Vec<ValueId>) = outputs.into_iter().unzip();
        let graph = Graph {
            values: self.values,
            inputs: self.inputs,
            outputs,
            output_names,
        };
        graph.verify()?;
        let macs = graph.estimated_macs();
        if macs > self.limits.max_macs {
            return Err(Error::LimitExceeded {
                what: "multiply-adds",
                found: macs,
                limit: self.limits.max_macs,
            });
        }
        Ok(graph)
    }

    fn push(&mut self, name: &str, ty: TensorType, source: Source) -> Result<ValueId, Error> {
        // Ids are u32, so the cap can never exceed u32::MAX whatever the limits say.
        let cap = self.limits.max_values.min(u32::MAX as usize);
        if self.values.len() >= cap {
            return Err(Error::LimitExceeded {
                what: "values",
                found: self.values.len() + 1,
                limit: cap,
            });
        }
        let total = self
            .total_elements
            .saturating_add(checked_num_elements(&ty.shape).unwrap_or(usize::MAX));
        if total > self.limits.max_total_elements {
            return Err(Error::TensorLimit {
                name: name.to_owned(),
                what: "total tensor elements",
                found: total,
                limit: self.limits.max_total_elements,
            });
        }
        if self.names.contains_key(name) {
            return Err(Error::DuplicateName {
                name: name.to_owned(),
            });
        }
        let id =
            ValueId(u32::try_from(self.values.len()).expect("the value cap keeps ids within u32"));
        self.total_elements = total;
        self.names.insert(name.to_owned(), id);
        self.values.push(Value {
            ty,
            source,
            name: Some(name.to_owned()),
        });
        Ok(id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn builder() -> GraphBuilder {
        GraphBuilder::new(Limits::default())
    }

    #[test]
    fn builds_input_relu_graph() {
        let mut b = builder();
        let x = b.input("x", vec![2, 3]).unwrap();
        let y = b.node("y", Op::Relu, &[x]).unwrap();
        let g = b.finish(&[y]).unwrap();
        assert_eq!(g.values().len(), 2);
        assert_eq!(g.inputs(), &[x]);
        assert_eq!(g.outputs(), &[y]);
        assert_eq!(g.value(y).ty, TensorType::f32(vec![2, 3]));
        assert_eq!(g.label(y), "y");
    }

    #[test]
    fn lookup_finds_defined_names_only() {
        let mut b = builder();
        let x = b.input("x", vec![1]).unwrap();
        assert_eq!(b.lookup("x"), Some(x));
        assert_eq!(b.lookup("nope"), None);
    }

    #[test]
    fn scalar_input_is_allowed() {
        assert!(builder().input("s", vec![]).is_ok());
    }

    #[test]
    fn constant_length_mismatch_is_rejected() {
        let err = builder()
            .constant("c", vec![2, 3], vec![0.0; 5])
            .unwrap_err();
        assert_eq!(
            err,
            Error::DataLength {
                name: "c".into(),
                shape: vec![2, 3],
                expected: 6,
                found: 5
            }
        );
    }

    #[test]
    fn duplicate_name_across_kinds_is_rejected() {
        let mut b = builder();
        b.input("x", vec![1]).unwrap();
        let err = b.constant("x", vec![1], vec![0.0]).unwrap_err();
        assert_eq!(err, Error::DuplicateName { name: "x".into() });
    }

    #[test]
    fn wrong_arity_is_rejected() {
        let mut b = builder();
        let x = b.input("x", vec![1]).unwrap();
        let err = b.node("y", Op::Relu, &[x, x]).unwrap_err();
        assert!(matches!(
            err,
            Error::Arity {
                expected: 1,
                found: 2,
                ..
            }
        ));
    }

    #[test]
    fn undefined_input_value_is_rejected_not_a_panic() {
        let err = builder().node("y", Op::Relu, &[ValueId(9)]).unwrap_err();
        assert!(matches!(err, Error::Verify(_)));
    }

    #[test]
    fn graph_without_outputs_is_rejected() {
        let mut b = builder();
        b.input("x", vec![1]).unwrap();
        assert_eq!(b.finish(&[]).unwrap_err(), Error::NoOutputs);
    }

    #[test]
    fn value_count_limit_is_enforced() {
        let limits = Limits {
            max_values: 2,
            ..Limits::default()
        };
        let mut b = GraphBuilder::new(limits);
        b.input("a", vec![1]).unwrap();
        b.input("b", vec![1]).unwrap();
        let err = b.input("c", vec![1]).unwrap_err();
        assert!(matches!(err, Error::LimitExceeded { what: "values", .. }));
    }

    #[test]
    fn huge_shape_is_rejected_before_allocation() {
        let mut b = builder();
        let overflow = b.input("a", vec![usize::MAX, 2]).unwrap_err();
        assert!(matches!(overflow, Error::InvalidShape { .. }));
        let too_big = b.input("b", vec![1 << 20, 1 << 20]).unwrap_err();
        assert!(matches!(
            too_big,
            Error::TensorLimit {
                what: "tensor elements",
                ..
            }
        ));
        let too_deep = b.input("c", vec![1; 9]).unwrap_err();
        assert!(matches!(
            too_deep,
            Error::TensorLimit {
                what: "tensor rank",
                ..
            }
        ));
    }

    #[test]
    fn zero_dimension_is_rejected() {
        let err = builder().input("x", vec![3, 0]).unwrap_err();
        assert!(matches!(err, Error::InvalidShape { .. }));
    }

    #[test]
    fn scalar_operand_on_either_side_keeps_the_tensor_shape() {
        let mut b = builder();
        let t = b.input("t", vec![2, 3]).unwrap();
        let s = b.input("s", vec![]).unwrap();
        let left = b.node("l", Op::Add, &[s, t]).unwrap();
        let right = b.node("r", Op::Mul, &[t, s]).unwrap();
        let g = b.finish(&[left, right]).unwrap();
        assert_eq!(g.value(left).ty, TensorType::f32(vec![2, 3]));
        assert_eq!(g.value(right).ty, TensorType::f32(vec![2, 3]));
    }

    #[test]
    fn mismatched_elementwise_shapes_are_rejected() {
        let mut b = builder();
        let x = b.input("x", vec![2, 3]).unwrap();
        let z = b.input("z", vec![3, 2]).unwrap();
        let err = b.node("y", Op::Add, &[x, z]).unwrap_err();
        assert!(matches!(err, Error::ShapeMismatch { .. }));
    }

    #[test]
    fn matmul_output_over_the_element_limit_is_rejected() {
        let mut b = builder();
        let a = b.input("a", vec![1 << 15, 1]).unwrap();
        let c = b.input("c", vec![1, 1 << 15]).unwrap();
        let err = b.node("y", Op::MatMul, &[a, c]).unwrap_err();
        assert!(matches!(
            err,
            Error::TensorLimit {
                what: "tensor elements",
                ..
            }
        ));
    }

    #[test]
    fn reshape_beyond_the_rank_limit_is_rejected() {
        let mut b = builder();
        let x = b.input("x", vec![512]).unwrap();
        let op = Op::Reshape { shape: vec![2; 9] };
        let err = b.node("y", op, &[x]).unwrap_err();
        assert!(matches!(
            err,
            Error::TensorLimit {
                what: "tensor rank",
                ..
            }
        ));
    }

    #[test]
    fn total_allocation_across_values_is_bounded() {
        let mut b = builder();
        let a = b.input("a", vec![1 << 14, 1]).unwrap();
        let c = b.input("c", vec![1, 1 << 14]).unwrap();
        b.node("y1", Op::MatMul, &[a, c]).unwrap();
        let err = b.node("y2", Op::MatMul, &[a, c]).unwrap_err();
        assert_eq!(
            err,
            Error::TensorLimit {
                name: "y2".into(),
                what: "total tensor elements",
                found: 2 * (1 << 14) + 2 * (1 << 28),
                limit: 1 << 29,
            }
        );
    }

    #[test]
    fn many_small_values_still_fit_the_total_budget() {
        let mut b = builder();
        let mut last = b.input("x0", vec![64]).unwrap();
        for i in 1..200 {
            last = b.node(&format!("n{i}"), Op::Relu, &[last]).unwrap();
        }
        assert!(b.finish(&[last]).is_ok());
    }

    #[test]
    fn finish_names_outputs_after_their_values() {
        let mut b = builder();
        let x = b.input("x", vec![1]).unwrap();
        let y = b.node("y", Op::Relu, &[x]).unwrap();
        let g = b.finish(&[y, x]).unwrap();
        assert_eq!(g.output_names(), ["y", "x"]);
    }

    #[test]
    fn finish_named_keeps_the_given_names_even_for_duplicates() {
        let mut b = builder();
        let x = b.input("x", vec![1]).unwrap();
        let y = b.node("y", Op::Relu, &[x]).unwrap();
        let g = b
            .finish_named(vec![("first".to_owned(), y), ("second".to_owned(), y)])
            .unwrap();
        assert_eq!(g.output_names(), ["first", "second"]);
        assert_eq!(g.outputs(), &[y, y]);
    }

    #[test]
    fn constant_shared_does_not_copy_the_data() {
        let mut b = builder();
        let data: std::sync::Arc<[f32]> = std::sync::Arc::from(vec![1.0, 2.0]);
        let c = b
            .constant_shared("c", vec![2], std::sync::Arc::clone(&data))
            .unwrap();
        let g = b.finish(&[c]).unwrap();
        match &g.value(c).source {
            Source::Constant(stored) => assert!(std::sync::Arc::ptr_eq(stored, &data)),
            other => panic!("expected a constant, got {other:?}"),
        }
    }

    #[test]
    fn value_limit_error_reports_the_exact_numbers() {
        let limits = Limits {
            max_values: 2,
            ..Limits::default()
        };
        let mut b = GraphBuilder::new(limits);
        b.input("a", vec![1]).unwrap();
        b.input("b", vec![1]).unwrap();
        assert_eq!(
            b.input("c", vec![1]).unwrap_err(),
            Error::LimitExceeded {
                what: "values",
                found: 3,
                limit: 2
            }
        );
    }

    #[test]
    fn total_element_budget_accepts_exactly_the_limit_and_rejects_one_more() {
        let limits = Limits {
            max_total_elements: 10,
            ..Limits::default()
        };
        let mut b = GraphBuilder::new(limits);
        b.input("a", vec![4]).unwrap();
        b.input("b", vec![6]).unwrap();
        assert_eq!(
            b.input("c", vec![1]).unwrap_err(),
            Error::TensorLimit {
                name: "c".into(),
                what: "total tensor elements",
                found: 11,
                limit: 10
            }
        );
    }

    /// x[2,3] @ w[3,4] costs 24, and Relu of the 8 outputs 8 more.
    fn matmul_relu(limits: Limits) -> Result<crate::Graph, Error> {
        let mut b = GraphBuilder::new(limits);
        let x = b.input("x", vec![2, 3]).unwrap();
        let w = b.input("w", vec![3, 4]).unwrap();
        let m = b.node("m", Op::MatMul, &[x, w]).unwrap();
        let y = b.node("y", Op::Relu, &[m]).unwrap();
        b.finish(&[y])
    }

    #[test]
    fn the_work_limit_accepts_exactly_the_estimate_and_rejects_one_less() {
        let graph = matmul_relu(Limits::default()).unwrap();
        assert_eq!(graph.estimated_macs(), 32);
        assert!(matmul_relu(Limits::default().with_max_macs(32)).is_ok());
        assert_eq!(
            matmul_relu(Limits::default().with_max_macs(31)).unwrap_err(),
            Error::LimitExceeded {
                what: "multiply-adds",
                found: 32,
                limit: 31
            }
        );
    }

    #[test]
    fn there_is_no_work_limit_by_default() {
        let mut b = GraphBuilder::new(Limits::default());
        let x = b.input("x", vec![1000, 1000]).unwrap();
        let w = b.input("w", vec![1000, 1000]).unwrap();
        let m = b.node("m", Op::MatMul, &[x, w]).unwrap();
        assert_eq!(b.finish(&[m]).unwrap().estimated_macs(), 1_000_000_000);
    }
}
