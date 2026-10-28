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

    /// Multiply-adds a run costs, counting one per output element for every
    /// node but `MatMul`, which costs `m * k * n`. Inputs and constants cost
    /// nothing. A bound for refusing graphs, not a timing; it saturates
    /// instead of wrapping.
    pub fn estimated_macs(&self) -> usize {
        let elements = |id: ValueId| -> usize {
            self.value(id)
                .ty
                .shape
                .iter()
                .fold(1usize, |n, &d| n.saturating_mul(d))
        };
        self.values
            .iter()
            .enumerate()
            .map(|(i, v)| match &v.source {
                Source::Node {
                    op: Op::MatMul,
                    inputs,
                } => elements(inputs[0]).saturating_mul(v.ty.shape[1]),
                Source::Node { .. } => elements(ValueId(i as u32)),
                _ => 0,
            })
            .fold(0usize, usize::saturating_add)
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
    use crate::{GraphBuilder, Limits, Op, ValueId};

    fn graph_with_constant(value: f32) -> crate::Graph {
        let mut b = GraphBuilder::new(Limits::default());
        let c = b.constant("c", vec![1], vec![value]).unwrap();
        b.finish(&[c]).unwrap()
    }

    fn macs_of(build: impl FnOnce(&mut GraphBuilder) -> ValueId) -> usize {
        let mut b = GraphBuilder::new(Limits::default());
        let out = build(&mut b);
        b.finish(&[out]).unwrap().estimated_macs()
    }

    #[test]
    fn matmul_costs_m_times_k_times_n() {
        let macs = macs_of(|b| {
            let a = b.input("a", vec![3, 5]).unwrap();
            let w = b.input("w", vec![5, 7]).unwrap();
            b.node("y", Op::MatMul, &[a, w]).unwrap()
        });
        assert_eq!(macs, 3 * 5 * 7);
    }

    #[test]
    fn a_vector_matmul_and_unit_dimensions_are_counted() {
        let macs = macs_of(|b| {
            let a = b.input("a", vec![1, 4]).unwrap();
            let w = b.input("w", vec![4, 1]).unwrap();
            b.node("y", Op::MatMul, &[a, w]).unwrap()
        });
        assert_eq!(macs, 4);
    }

    #[test]
    fn every_other_node_costs_its_output_elements() {
        for op in [
            Op::Relu,
            Op::Sigmoid,
            Op::Reshape { shape: vec![6] },
            Op::Transpose { perm: vec![1, 0] },
        ] {
            let macs = macs_of(|b| {
                let x = b.input("x", vec![2, 3]).unwrap();
                b.node("y", op.clone(), &[x]).unwrap()
            });
            assert_eq!(macs, 6, "{op:?}");
        }
        for op in [Op::Add, Op::Mul] {
            let macs = macs_of(|b| {
                let x = b.input("x", vec![2, 3]).unwrap();
                let z = b.input("z", vec![2, 3]).unwrap();
                b.node("y", op.clone(), &[x, z]).unwrap()
            });
            assert_eq!(macs, 6, "{op:?}");
        }
    }

    #[test]
    fn inputs_and_constants_cost_nothing_and_nodes_add_up() {
        let macs = macs_of(|b| {
            let x = b.input("x", vec![2, 3]).unwrap();
            let c = b.constant("c", vec![3, 4], vec![0.5; 12]).unwrap();
            let m = b.node("m", Op::MatMul, &[x, c]).unwrap();
            b.node("r", Op::Relu, &[m]).unwrap()
        });
        assert_eq!(macs, 2 * 3 * 4 + 2 * 4);
    }

    #[test]
    fn a_value_used_twice_is_counted_once_per_node() {
        let macs = macs_of(|b| {
            let x = b.input("x", vec![4]).unwrap();
            let r = b.node("r", Op::Relu, &[x]).unwrap();
            b.node("y", Op::Add, &[r, r]).unwrap()
        });
        assert_eq!(macs, 4 + 4);
    }

    #[test]
    fn a_graph_without_nodes_costs_nothing() {
        let macs = macs_of(|b| b.input("x", vec![9]).unwrap());
        assert_eq!(macs, 0);
    }

    #[test]
    fn the_estimate_saturates_instead_of_wrapping() {
        // Limits lifted so the builder accepts shapes whose product overflows.
        let limits = Limits {
            max_tensor_elements: usize::MAX,
            max_total_elements: usize::MAX,
            ..Limits::default()
        };
        let mut b = GraphBuilder::new(limits);
        let a = b.input("a", vec![2, 1 << 62]).unwrap();
        let w = b.input("w", vec![1 << 62, 2]).unwrap();
        let y = b.node("y", Op::MatMul, &[a, w]).unwrap();
        assert_eq!(b.finish(&[y]).unwrap().estimated_macs(), usize::MAX);
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
