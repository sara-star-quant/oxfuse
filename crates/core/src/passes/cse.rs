//! Common-subexpression elimination: a node with the same op and the same
//! (remapped) inputs as an earlier node is aliased to it. Keys use the new
//! ids, so merging is transitive. Constants and inputs are never merged, and
//! `Add(a, b)` is not `Add(b, a)`; both are valid extensions, left out until
//! a measured case needs them.

use super::rebuild::Rebuilder;
use super::{Pass, PassOutput};
use crate::{Error, Graph, Limits, Op, Source, ValueId};
use std::collections::HashMap;

pub struct CommonSubexpressionElimination {
    limits: Limits,
}

impl CommonSubexpressionElimination {
    pub fn new(limits: Limits) -> Self {
        Self { limits }
    }
}

impl Pass for CommonSubexpressionElimination {
    fn name(&self) -> &'static str {
        "cse"
    }

    fn run(&self, graph: &Graph) -> Result<PassOutput, Error> {
        let mut rebuild = Rebuilder::new(graph, &self.limits);
        let mut seen: HashMap<(Op, Vec<ValueId>), ValueId> = HashMap::new();
        let mut merged = 0;
        for (index, value) in graph.values().iter().enumerate() {
            let old = ValueId(index as u32);
            let Source::Node { op, inputs } = &value.source else {
                rebuild.copy(old)?;
                continue;
            };
            let key = (
                op.clone(),
                inputs
                    .iter()
                    .map(|&id| rebuild.mapped(id))
                    .collect::<Vec<_>>(),
            );
            if let Some(&existing) = seen.get(&key) {
                rebuild.alias(old, existing);
                merged += 1;
            } else {
                let new = rebuild.copy(old)?;
                seen.insert(key, new);
            }
        }
        Ok(PassOutput {
            graph: rebuild.finish()?,
            changed: merged,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{GraphBuilder, Op};

    fn builder() -> GraphBuilder {
        GraphBuilder::new(Limits::default())
    }

    fn run(graph: &Graph) -> PassOutput {
        CommonSubexpressionElimination::new(Limits::default())
            .run(graph)
            .unwrap()
    }

    #[test]
    fn merges_identical_nodes() {
        let mut b = builder();
        let x = b.input("x", vec![2]).unwrap();
        let p = b.node("p", Op::Relu, &[x]).unwrap();
        let q = b.node("q", Op::Relu, &[x]).unwrap();
        let y = b.node("y", Op::Add, &[p, q]).unwrap();
        let g = b.finish(&[y]).unwrap();
        let out = run(&g);
        assert_eq!(out.changed, 1);
        assert_eq!(out.graph.values().len(), 3);
        let add_inputs = match &out.graph.value(out.graph.outputs()[0]).source {
            crate::Source::Node { inputs, .. } => inputs.clone(),
            other => panic!("{other:?}"),
        };
        assert_eq!(add_inputs[0], add_inputs[1]);
    }

    #[test]
    fn merged_outputs_keep_their_names() {
        let mut b = builder();
        let x = b.input("x", vec![2]).unwrap();
        let p = b.node("p", Op::Relu, &[x]).unwrap();
        let q = b.node("q", Op::Relu, &[x]).unwrap();
        let g = b.finish(&[p, q]).unwrap();
        let out = run(&g);
        assert_eq!(out.changed, 1);
        assert_eq!(out.graph.output_names(), ["p", "q"]);
        assert_eq!(out.graph.outputs()[0], out.graph.outputs()[1]);
    }

    #[test]
    fn merging_is_transitive() {
        let mut b = builder();
        let x = b.input("x", vec![2]).unwrap();
        let c = b.constant("c", vec![2], vec![1.0, 2.0]).unwrap();
        let a1 = b.node("a1", Op::Relu, &[x]).unwrap();
        let a2 = b.node("a2", Op::Relu, &[x]).unwrap();
        let b1 = b.node("b1", Op::Add, &[a1, c]).unwrap();
        let b2 = b.node("b2", Op::Add, &[a2, c]).unwrap();
        let g = b.finish(&[b1, b2]).unwrap();
        let out = run(&g);
        assert_eq!(out.changed, 2);
        assert_eq!(out.graph.values().len(), 4);
    }

    #[test]
    fn different_attributes_are_not_merged() {
        let mut b = builder();
        let x = b.input("x", vec![6]).unwrap();
        let r1 = b
            .node("r1", Op::Reshape { shape: vec![2, 3] }, &[x])
            .unwrap();
        let r2 = b
            .node("r2", Op::Reshape { shape: vec![3, 2] }, &[x])
            .unwrap();
        let r3 = b
            .node("r3", Op::Reshape { shape: vec![2, 3] }, &[x])
            .unwrap();
        let g = b.finish(&[r1, r2, r3]).unwrap();
        let out = run(&g);
        assert_eq!(out.changed, 1);
        assert_eq!(out.graph.outputs()[0], out.graph.outputs()[2]);
        assert_ne!(out.graph.outputs()[0], out.graph.outputs()[1]);
    }

    #[test]
    fn operand_order_matters() {
        let mut b = builder();
        let x = b.input("x", vec![2]).unwrap();
        let z = b.input("z", vec![2]).unwrap();
        let s1 = b.node("s1", Op::Add, &[x, z]).unwrap();
        let s2 = b.node("s2", Op::Add, &[z, x]).unwrap();
        let g = b.finish(&[s1, s2]).unwrap();
        let out = run(&g);
        assert_eq!(out.changed, 0);
        assert_eq!(out.graph, g);
    }

    #[test]
    fn constants_and_inputs_are_never_merged() {
        let mut b = builder();
        let x1 = b.input("x1", vec![2]).unwrap();
        let x2 = b.input("x2", vec![2]).unwrap();
        let c1 = b.constant("c1", vec![2], vec![1.0, 2.0]).unwrap();
        let c2 = b.constant("c2", vec![2], vec![1.0, 2.0]).unwrap();
        let g = b.finish(&[x1, x2, c1, c2]).unwrap();
        let out = run(&g);
        assert_eq!(out.changed, 0);
        assert_eq!(out.graph, g);
    }

    #[test]
    fn nothing_to_merge_rebuilds_the_same_graph() {
        let mut b = builder();
        let x = b.input("x", vec![2]).unwrap();
        let p = b.node("p", Op::Relu, &[x]).unwrap();
        let q = b.node("q", Op::Sigmoid, &[p]).unwrap();
        let g = b.finish(&[q]).unwrap();
        let out = run(&g);
        assert_eq!(out.changed, 0);
        assert_eq!(out.graph, g);
    }
}
