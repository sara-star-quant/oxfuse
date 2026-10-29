//! The graph of the `mlp` benchmark, shared with the test that pins its output
//! (`tests/mlp_bench_graph.rs`), so the benchmark cannot silently time a
//! different graph than the one the test checked.

use oxfuse_core::{Graph, GraphBuilder, Limits, Op};

/// A small classifier: 784 inputs, 256 hidden units with `Relu`, 10 outputs.
/// No bias terms: the IR has no broadcasting beyond scalars.
pub const INPUT: usize = 784;
pub const HIDDEN: usize = 256;
pub const CLASSES: usize = 10;

/// Deterministic values in [-1, 1).
pub fn data(len: usize, seed: u32) -> Vec<f32> {
    let mut state = seed.wrapping_mul(2_654_435_761).wrapping_add(1);
    (0..len)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (state >> 8) as f32 / (1u32 << 23) as f32 - 1.0
        })
        .collect()
}

pub const W1_SEED: u32 = 11;
pub const W2_SEED: u32 = 12;
pub const X_SEED: u32 = 13;

pub fn mlp_graph(batch: usize) -> Graph {
    let mut b = GraphBuilder::new(Limits::default());
    let x = b.input("x", vec![batch, INPUT]).unwrap();
    let w1 = b
        .constant("w1", vec![INPUT, HIDDEN], data(INPUT * HIDDEN, W1_SEED))
        .unwrap();
    let w2 = b
        .constant("w2", vec![HIDDEN, CLASSES], data(HIDDEN * CLASSES, W2_SEED))
        .unwrap();
    let h = b.node("h", Op::MatMul, &[x, w1]).unwrap();
    let a = b.node("a", Op::Relu, &[h]).unwrap();
    let y = b.node("y", Op::MatMul, &[a, w2]).unwrap();
    b.finish(&[y]).unwrap()
}
