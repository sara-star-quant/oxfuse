//! The layout pass only moves bits: optimized graphs must return the
//! unoptimized graph's exact bits on the reference backend.

mod common;

use common::bits;
use oxfuse_core::passes::default_pipeline;
use oxfuse_core::{Graph, GraphBuilder, Limits, Op};
use oxfuse_runtime::{Backend, CpuBackend, ReferenceBackend, ReferenceEvaluator, Tensor};
use std::sync::Arc;

fn run_both(graph: &Graph, input: &Tensor) {
    let optimized = default_pipeline(Limits::default(), Arc::new(ReferenceEvaluator))
        .run(graph)
        .unwrap()
        .graph;
    assert!(
        optimized.values().len() < graph.values().len(),
        "nothing was elided"
    );
    let inputs = std::slice::from_ref(input);
    let want = ReferenceBackend
        .compile(graph)
        .unwrap()
        .run(inputs)
        .unwrap();
    let reference = ReferenceBackend
        .compile(&optimized)
        .unwrap()
        .run(inputs)
        .unwrap();
    let cpu = CpuBackend.compile(&optimized).unwrap().run(inputs).unwrap();
    assert_eq!(bits(&want), bits(&reference));
    assert_eq!(bits(&want), bits(&cpu));
}

/// Distinct values, so any wrong permutation changes the bits.
fn tensor(shape: &[usize]) -> Tensor {
    let len = shape.iter().product::<usize>();
    Tensor::new(
        shape.to_vec(),
        (0..len).map(|i| i as f32 * 0.5 - 7.0).collect(),
    )
    .unwrap()
}

fn transposes(shape: &[usize], perms: &[&[usize]]) -> Graph {
    let mut b = GraphBuilder::new(Limits::default());
    let mut value = b.input("x", shape.to_vec()).unwrap();
    for (i, perm) in perms.iter().enumerate() {
        value = b
            .node(
                &format!("t{i}"),
                Op::Transpose {
                    perm: perm.to_vec(),
                },
                &[value],
            )
            .unwrap();
    }
    b.finish(&[value]).unwrap()
}

#[test]
fn every_pair_of_rank_three_permutations() {
    const PERMS: [[usize; 3]; 6] = [
        [0, 1, 2],
        [0, 2, 1],
        [1, 0, 2],
        [1, 2, 0],
        [2, 0, 1],
        [2, 1, 0],
    ];
    let x = tensor(&[2, 3, 4]);
    for first in &PERMS {
        for second in &PERMS {
            let graph = transposes(&[2, 3, 4], &[first, second]);
            run_both(&graph, &x);
        }
    }
}

#[test]
fn a_chain_of_four_transposes() {
    let graph = transposes(
        &[2, 3, 4],
        &[&[1, 2, 0], &[0, 2, 1], &[2, 1, 0], &[1, 0, 2]],
    );
    run_both(&graph, &tensor(&[2, 3, 4]));
}

#[test]
fn reshape_chains_and_round_trips() {
    let mut b = GraphBuilder::new(Limits::default());
    let x = b.input("x", vec![2, 3, 4]).unwrap();
    let r1 = b
        .node("r1", Op::Reshape { shape: vec![6, 4] }, &[x])
        .unwrap();
    let r2 = b
        .node("r2", Op::Reshape { shape: vec![24] }, &[r1])
        .unwrap();
    let r3 = b
        .node(
            "r3",
            Op::Reshape {
                shape: vec![2, 3, 4],
            },
            &[r2],
        )
        .unwrap();
    let r4 = b
        .node("r4", Op::Reshape { shape: vec![4, 6] }, &[r3])
        .unwrap();
    let graph = b.finish(&[r4]).unwrap();
    run_both(&graph, &tensor(&[2, 3, 4]));
}

#[test]
fn a_transpose_pair_feeding_a_matmul() {
    let mut b = GraphBuilder::new(Limits::default());
    let x = b.input("x", vec![3, 5]).unwrap();
    let w = b
        .constant("w", vec![5, 2], (0..10).map(|i| i as f32 - 4.5).collect())
        .unwrap();
    let t = b
        .node("t", Op::Transpose { perm: vec![1, 0] }, &[x])
        .unwrap();
    let u = b
        .node("u", Op::Transpose { perm: vec![1, 0] }, &[t])
        .unwrap();
    let m = b.node("m", Op::MatMul, &[u, w]).unwrap();
    let graph = b.finish(&[m, t]).unwrap();
    // `t` is also an output, so only the second transpose goes away.
    let optimized = default_pipeline(Limits::default(), Arc::new(ReferenceEvaluator))
        .run(&graph)
        .unwrap()
        .graph;
    let input = tensor(&[3, 5]);
    let inputs = std::slice::from_ref(&input);
    let want = ReferenceBackend
        .compile(&graph)
        .unwrap()
        .run(inputs)
        .unwrap();
    let got = CpuBackend.compile(&optimized).unwrap().run(inputs).unwrap();
    assert_eq!(bits(&want), bits(&got));
    assert_eq!(optimized.values().len(), graph.values().len() - 1);
}
