//! The IR: an arena of values in definition order. Every op has one output.

use crate::{Op, TensorType, ValueId};
use std::sync::Arc;

#[derive(Debug, Clone)]
pub enum Source {
    Input,
    /// Shared so rebuilding passes and execution plans never copy weights.
    Constant(Arc<[f32]>),
    Node {
        op: Op,
        inputs: Vec<ValueId>,
    },
}

impl PartialEq for Source {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Input, Self::Input) => true,
            (Self::Constant(a), Self::Constant(b)) => {
                Arc::ptr_eq(a, b)
                    || (a.len() == b.len()
                        && a.iter()
                            .zip(b.iter())
                            .all(|(x, y)| x.to_bits() == y.to_bits()))
            }
            (
                Self::Node {
                    op: op_a,
                    inputs: in_a,
                },
                Self::Node {
                    op: op_b,
                    inputs: in_b,
                },
            ) => op_a == op_b && in_a == in_b,
            _ => false,
        }
    }
}

impl Eq for Source {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Value {
    pub ty: TensorType,
    pub source: Source,
    pub name: Option<String>,
}

/// A graph is only constructed through `GraphBuilder::finish`, which verifies it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Graph {
    pub(crate) values: Vec<Value>,
    pub(crate) inputs: Vec<ValueId>,
    pub(crate) outputs: Vec<ValueId>,
    /// Parallel to `outputs`. The names users see; passes never change them.
    pub(crate) output_names: Vec<String>,
}

impl Graph {
    pub fn values(&self) -> &[Value] {
        &self.values
    }

    pub fn value(&self, id: ValueId) -> &Value {
        &self.values[id.index()]
    }

    /// Graph inputs in declared order.
    pub fn inputs(&self) -> &[ValueId] {
        &self.inputs
    }

    pub fn outputs(&self) -> &[ValueId] {
        &self.outputs
    }

    pub fn output_names(&self) -> &[String] {
        &self.output_names
    }

    /// The value's name, or `%<id>` when it has none.
    pub fn label(&self, id: ValueId) -> String {
        match &self.value(id).name {
            Some(name) => name.clone(),
            None => format!("%{}", id.0),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{GraphBuilder, Limits, Op};

    fn graph_with_constant(value: f32) -> crate::Graph {
        let mut b = GraphBuilder::new(Limits::default());
        let c = b.constant("c", vec![1], vec![value]).unwrap();
        b.finish(&[c]).unwrap()
    }

    #[test]
    fn a_graph_with_a_nan_constant_equals_its_clone() {
        let g = graph_with_constant(f32::NAN);
        assert_eq!(g, g.clone());
    }

    #[test]
    fn constants_compare_by_bits_so_signed_zeros_differ() {
        assert_ne!(graph_with_constant(0.0), graph_with_constant(-0.0));
        assert_eq!(graph_with_constant(-0.0), graph_with_constant(-0.0));
    }

    #[test]
    fn nodes_with_the_same_op_but_different_inputs_are_not_equal() {
        let build = |swap: bool| {
            let mut b = GraphBuilder::new(Limits::default());
            let x = b.input("x", vec![2]).unwrap();
            let z = b.input("z", vec![2]).unwrap();
            let inputs = if swap { [z, x] } else { [x, z] };
            let y = b.node("y", Op::Add, &inputs).unwrap();
            b.finish(&[y]).unwrap()
        };
        assert_eq!(build(false), build(false));
        assert_ne!(build(false), build(true));
    }

    #[test]
    fn nodes_with_the_same_inputs_but_different_ops_are_not_equal() {
        let build = |op: Op| {
            let mut b = GraphBuilder::new(Limits::default());
            let x = b.input("x", vec![2]).unwrap();
            let y = b.node("y", op, &[x]).unwrap();
            b.finish(&[y]).unwrap()
        };
        assert_ne!(build(Op::Relu), build(Op::Sigmoid));
    }
}
