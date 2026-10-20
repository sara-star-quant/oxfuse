//! Dead-code elimination: keep what the outputs transitively use, plus every
//! graph input. Inputs are never removed because they are the interface.

use super::rebuild::Rebuilder;
use super::{Pass, PassOutput};
use crate::{Error, Graph, Limits, Source};

pub struct DeadCodeElimination {
    limits: Limits,
}

impl DeadCodeElimination {
    pub fn new(limits: Limits) -> Self {
        Self { limits }
    }
}

impl Pass for DeadCodeElimination {
    fn name(&self) -> &'static str {
        "dce"
    }

    fn run(&self, graph: &Graph) -> Result<PassOutput, Error> {
        let count = graph.values().len();
        let mut live = vec![false; count];
        for id in graph.outputs().iter().chain(graph.inputs()) {
            live[id.index()] = true;
        }
        // Inputs of a node have smaller ids, so one backward sweep suffices.
        for index in (0..count).rev() {
            if !live[index] {
                continue;
            }
            if let Source::Node { inputs, .. } = &graph.values()[index].source {
                for input in inputs {
                    live[input.index()] = true;
                }
            }
        }
        let mut rebuild = Rebuilder::new(graph, &self.limits);
        let mut removed = 0;
        for (index, &is_live) in live.iter().enumerate() {
            if is_live {
                rebuild.copy(crate::ValueId(index as u32))?;
            } else {
                removed += 1;
            }
        }
        Ok(PassOutput {
            graph: rebuild.finish()?,
            changed: removed,
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
        DeadCodeElimination::new(Limits::default())
            .run(graph)
            .unwrap()
    }

    #[test]
    fn removes_dead_nodes_and_unused_constants() {
        let mut b = builder();
        let x = b.input("x", vec![2]).unwrap();
        let c = b.constant("c", vec![2], vec![1.0, 2.0]).unwrap();
        let live = b.node("live", Op::Relu, &[x]).unwrap();
        let dead = b.node("dead", Op::Mul, &[x, c]).unwrap();
        b.node("deader", Op::Relu, &[dead]).unwrap();
        let g = b.finish(&[live]).unwrap();
        let out = run(&g);
        assert_eq!(out.changed, 3);
        let names: Vec<String> = (0..out.graph.values().len())
            .map(|i| out.graph.label(crate::ValueId(i as u32)))
            .collect();
        assert_eq!(names, ["x", "live"]);
        assert_eq!(out.graph.output_names(), ["live"]);
    }

    #[test]
    fn unused_inputs_survive() {
        let mut b = builder();
        let x = b.input("x", vec![2]).unwrap();
        let z = b.input("z", vec![3]).unwrap();
        let y = b.node("y", Op::Relu, &[x]).unwrap();
        let g = b.finish(&[y]).unwrap();
        let out = run(&g);
        assert_eq!(out.changed, 0);
        assert_eq!(out.graph.inputs().len(), 2);
        assert_eq!(out.graph.label(z), "z");
        assert_eq!(out.graph, g);
    }

    #[test]
    fn outputs_that_are_inputs_or_constants_survive() {
        let mut b = builder();
        let x = b.input("x", vec![1]).unwrap();
        let c = b.constant("c", vec![1], vec![5.0]).unwrap();
        let g = b.finish(&[x, c, x]).unwrap();
        let out = run(&g);
        assert_eq!(out.changed, 0);
        assert_eq!(out.graph, g);
    }

    #[test]
    fn a_graph_with_no_dead_code_is_rebuilt_identically() {
        let mut b = builder();
        let x = b.input("x", vec![2, 2]).unwrap();
        let w = b
            .constant("w", vec![2, 2], vec![1.0, 2.0, 3.0, 4.0])
            .unwrap();
        let p = b.node("p", Op::MatMul, &[x, w]).unwrap();
        let r = b.node("r", Op::Reshape { shape: vec![4] }, &[p]).unwrap();
        let g = b.finish(&[r]).unwrap();
        let out = run(&g);
        assert_eq!(out.changed, 0);
        assert_eq!(out.graph, g);
    }

    #[test]
    fn a_node_used_by_live_and_dead_code_stays() {
        let mut b = builder();
        let x = b.input("x", vec![2]).unwrap();
        let shared = b.node("shared", Op::Relu, &[x]).unwrap();
        let live = b.node("live", Op::Sigmoid, &[shared]).unwrap();
        b.node("dead", Op::Sigmoid, &[shared]).unwrap();
        let g = b.finish(&[live]).unwrap();
        let out = run(&g);
        assert_eq!(out.changed, 1);
        assert_eq!(out.graph.values().len(), 3);
    }

    #[test]
    fn constants_are_shared_not_copied() {
        let mut b = builder();
        let c = b.constant("c", vec![2], vec![1.0, 2.0]).unwrap();
        let g = b.finish(&[c]).unwrap();
        let out = run(&g);
        match (&g.value(c).source, &out.graph.value(c).source) {
            (crate::Source::Constant(a), crate::Source::Constant(b)) => {
                assert!(std::sync::Arc::ptr_eq(a, b))
            }
            _ => panic!("expected constants"),
        }
    }
}
