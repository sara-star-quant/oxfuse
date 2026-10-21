//! Layout elision: a pair of `Transpose` nodes becomes one `Transpose` (or
//! nothing, when the permutations cancel), a pair of `Reshape` nodes becomes
//! one `Reshape`, and a `Transpose` with the identity permutation or a
//! `Reshape` to the shape it already has is aliased to its input. Both ops
//! only move bits, so the result is bit-identical and the intermediate value,
//! if nothing else reads it, is left for dead-code elimination.

use super::rebuild::Rebuilder;
use super::{Pass, PassOutput};
use crate::{Error, Graph, Limits, Op, Source, ValueId};
use std::collections::HashMap;

pub struct LayoutElision {
    limits: Limits,
}

impl LayoutElision {
    pub fn new(limits: Limits) -> Self {
        Self { limits }
    }
}

/// A `Transpose` or `Reshape` node of the new graph: the op, the value it
/// reads, and that value's shape.
struct Link {
    op: Op,
    source: ValueId,
    source_shape: Vec<usize>,
}

impl Pass for LayoutElision {
    fn name(&self) -> &'static str {
        "layout"
    }

    fn run(&self, graph: &Graph) -> Result<PassOutput, Error> {
        let mut rebuild = Rebuilder::new(graph, &self.limits);
        let mut links: HashMap<ValueId, Link> = HashMap::new();
        let mut changed = 0;
        for (index, value) in graph.values().iter().enumerate() {
            let old = ValueId(index as u32);
            let Source::Node { op, inputs } = &value.source else {
                rebuild.copy(old)?;
                continue;
            };
            if !matches!(op, Op::Transpose { .. } | Op::Reshape { .. }) {
                rebuild.copy(old)?;
                continue;
            }
            let input = rebuild.mapped(inputs[0]);
            // Look through one earlier layout node of the same kind.
            let (source, source_shape, op) = match (links.get(&input), op) {
                (
                    Some(Link {
                        op: Op::Transpose { perm: inner },
                        source,
                        source_shape,
                    }),
                    Op::Transpose { perm: outer },
                ) => {
                    let composed = outer.iter().map(|&i| inner[i]).collect();
                    (
                        *source,
                        source_shape.clone(),
                        Op::Transpose { perm: composed },
                    )
                }
                (
                    Some(Link {
                        op: Op::Reshape { .. },
                        source,
                        source_shape,
                    }),
                    Op::Reshape { .. },
                ) => (*source, source_shape.clone(), op.clone()),
                _ => (input, graph.value(inputs[0]).ty.shape.clone(), op.clone()),
            };
            let identity = match &op {
                Op::Transpose { perm } => perm.iter().enumerate().all(|(i, &p)| i == p),
                Op::Reshape { shape } => *shape == source_shape,
                _ => unreachable!("only layout ops reach here"),
            };
            if identity {
                rebuild.alias(old, source);
                changed += 1;
                continue;
            }
            let new = if source == input {
                rebuild.copy(old)?
            } else {
                changed += 1;
                rebuild.replace_with_node(old, op.clone(), &[source])?
            };
            links.insert(
                new,
                Link {
                    op,
                    source,
                    source_shape,
                },
            );
        }
        Ok(PassOutput {
            graph: rebuild.finish()?,
            changed,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::GraphBuilder;

    fn run(graph: &Graph) -> PassOutput {
        LayoutElision::new(Limits::default()).run(graph).unwrap()
    }

    fn op_of(graph: &Graph, id: ValueId) -> &Op {
        match &graph.value(id).source {
            Source::Node { op, .. } => op,
            other => panic!("{other:?}"),
        }
    }

    fn input_of(graph: &Graph, id: ValueId) -> ValueId {
        match &graph.value(id).source {
            Source::Node { inputs, .. } => inputs[0],
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn two_transposes_that_cancel_become_their_input() {
        let mut b = GraphBuilder::new(Limits::default());
        let x = b.input("x", vec![2, 3]).unwrap();
        let t = b
            .node("t", Op::Transpose { perm: vec![1, 0] }, &[x])
            .unwrap();
        let u = b
            .node("u", Op::Transpose { perm: vec![1, 0] }, &[t])
            .unwrap();
        let g = b.finish(&[u]).unwrap();
        let out = run(&g);
        assert_eq!(out.changed, 1);
        let output = out.graph.outputs()[0];
        assert!(matches!(out.graph.value(output).source, Source::Input));
        assert_eq!(out.graph.output_names(), ["u"]);
    }

    #[test]
    fn two_transposes_compose_in_the_right_order() {
        // inner [1, 2, 0], outer [1, 0, 2]: output dimension i is input
        // dimension inner[outer[i]], so the composition is [2, 1, 0]. Swapping
        // the two permutations would give [1, 0, 2], and a shape of [3, 2, 4].
        let mut b = GraphBuilder::new(Limits::default());
        let x = b.input("x", vec![2, 3, 4]).unwrap();
        let t = b
            .node(
                "t",
                Op::Transpose {
                    perm: vec![1, 2, 0],
                },
                &[x],
            )
            .unwrap();
        let u = b
            .node(
                "u",
                Op::Transpose {
                    perm: vec![1, 0, 2],
                },
                &[t],
            )
            .unwrap();
        let g = b.finish(&[u]).unwrap();
        let out = run(&g);
        let output = out.graph.outputs()[0];
        assert_eq!(
            op_of(&out.graph, output),
            &Op::Transpose {
                perm: vec![2, 1, 0]
            }
        );
        assert_eq!(out.graph.value(output).ty.shape, vec![4, 3, 2]);
        assert!(matches!(
            out.graph.value(input_of(&out.graph, output)).source,
            Source::Input
        ));
    }

    #[test]
    fn inverse_permutations_cancel() {
        let mut b = GraphBuilder::new(Limits::default());
        let x = b.input("x", vec![2, 3, 4]).unwrap();
        let t = b
            .node(
                "t",
                Op::Transpose {
                    perm: vec![1, 2, 0],
                },
                &[x],
            )
            .unwrap();
        let u = b
            .node(
                "u",
                Op::Transpose {
                    perm: vec![2, 0, 1],
                },
                &[t],
            )
            .unwrap();
        let g = b.finish(&[u]).unwrap();
        let out = run(&g);
        assert!(matches!(
            out.graph.value(out.graph.outputs()[0]).source,
            Source::Input
        ));
    }

    #[test]
    fn a_chain_of_three_collapses_to_one() {
        let mut b = GraphBuilder::new(Limits::default());
        let x = b.input("x", vec![2, 3, 4]).unwrap();
        let t = b
            .node(
                "t",
                Op::Transpose {
                    perm: vec![1, 0, 2],
                },
                &[x],
            )
            .unwrap();
        let u = b
            .node(
                "u",
                Op::Transpose {
                    perm: vec![0, 2, 1],
                },
                &[t],
            )
            .unwrap();
        let v = b
            .node(
                "v",
                Op::Transpose {
                    perm: vec![2, 1, 0],
                },
                &[u],
            )
            .unwrap();
        let g = b.finish(&[v]).unwrap();
        let out = run(&g);
        let output = out.graph.outputs()[0];
        assert!(matches!(op_of(&out.graph, output), Op::Transpose { .. }));
        assert!(matches!(
            out.graph.value(input_of(&out.graph, output)).source,
            Source::Input
        ));
    }

    #[test]
    fn an_identity_transpose_is_removed() {
        let mut b = GraphBuilder::new(Limits::default());
        let x = b.input("x", vec![2, 3]).unwrap();
        let t = b
            .node("t", Op::Transpose { perm: vec![0, 1] }, &[x])
            .unwrap();
        let g = b.finish(&[t]).unwrap();
        let out = run(&g);
        assert_eq!(out.changed, 1);
        assert!(matches!(
            out.graph.value(out.graph.outputs()[0]).source,
            Source::Input
        ));
    }

    #[test]
    fn two_reshapes_become_one() {
        let mut b = GraphBuilder::new(Limits::default());
        let x = b.input("x", vec![6]).unwrap();
        let r = b
            .node("r", Op::Reshape { shape: vec![2, 3] }, &[x])
            .unwrap();
        let s = b
            .node("s", Op::Reshape { shape: vec![3, 2] }, &[r])
            .unwrap();
        let g = b.finish(&[s]).unwrap();
        let out = run(&g);
        assert_eq!(out.changed, 1);
        let output = out.graph.outputs()[0];
        assert_eq!(
            op_of(&out.graph, output),
            &Op::Reshape { shape: vec![3, 2] }
        );
        assert!(matches!(
            out.graph.value(input_of(&out.graph, output)).source,
            Source::Input
        ));
    }

    #[test]
    fn a_reshape_back_to_the_original_shape_is_removed() {
        let mut b = GraphBuilder::new(Limits::default());
        let x = b.input("x", vec![2, 3]).unwrap();
        let r = b.node("r", Op::Reshape { shape: vec![6] }, &[x]).unwrap();
        let s = b
            .node("s", Op::Reshape { shape: vec![2, 3] }, &[r])
            .unwrap();
        let g = b.finish(&[s]).unwrap();
        let out = run(&g);
        assert!(matches!(
            out.graph.value(out.graph.outputs()[0]).source,
            Source::Input
        ));
    }

    #[test]
    fn a_reshape_to_its_own_shape_is_removed() {
        let mut b = GraphBuilder::new(Limits::default());
        let x = b.input("x", vec![2, 3]).unwrap();
        let r = b
            .node("r", Op::Reshape { shape: vec![2, 3] }, &[x])
            .unwrap();
        let g = b.finish(&[r]).unwrap();
        let out = run(&g);
        assert_eq!(out.changed, 1);
    }

    #[test]
    fn a_transpose_and_a_reshape_are_not_combined() {
        let mut b = GraphBuilder::new(Limits::default());
        let x = b.input("x", vec![2, 3]).unwrap();
        let t = b
            .node("t", Op::Transpose { perm: vec![1, 0] }, &[x])
            .unwrap();
        let r = b.node("r", Op::Reshape { shape: vec![6] }, &[t]).unwrap();
        let g = b.finish(&[r]).unwrap();
        let out = run(&g);
        assert_eq!(out.changed, 0);
        assert_eq!(out.graph.values().len(), 3);
    }

    #[test]
    fn an_intermediate_with_another_reader_stays() {
        let mut b = GraphBuilder::new(Limits::default());
        let x = b.input("x", vec![2, 3]).unwrap();
        let t = b
            .node("t", Op::Transpose { perm: vec![1, 0] }, &[x])
            .unwrap();
        let u = b
            .node("u", Op::Transpose { perm: vec![1, 0] }, &[t])
            .unwrap();
        let g = b.finish(&[t, u]).unwrap();
        let out = run(&g);
        assert_eq!(out.graph.output_names(), ["t", "u"]);
        assert!(matches!(
            op_of(&out.graph, out.graph.outputs()[0]),
            Op::Transpose { .. }
        ));
        assert!(matches!(
            out.graph.value(out.graph.outputs()[1]).source,
            Source::Input
        ));
    }

    #[test]
    fn other_ops_are_left_alone() {
        let mut b = GraphBuilder::new(Limits::default());
        let x = b.input("x", vec![2]).unwrap();
        let y = b.node("y", Op::Relu, &[x]).unwrap();
        let g = b.finish(&[y]).unwrap();
        let out = run(&g);
        assert_eq!(out.changed, 0);
        assert_eq!(out.graph, g);
    }
}
