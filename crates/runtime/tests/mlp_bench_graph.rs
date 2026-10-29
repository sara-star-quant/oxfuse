//! The `mlp` benchmark times this graph; its output is pinned here so the
//! benchmark cannot drift to another graph or another result.

#[path = "../benches/mlp_graph.rs"]
mod mlp_graph;

use mlp_graph::{CLASSES, INPUT, X_SEED, data, mlp_graph};
use oxfuse_core::{Op, Source};
use oxfuse_runtime::{Backend, CpuBackend, ReferenceBackend, Tensor};

/// FNV-1a over the output bit patterns.
fn fnv(values: &[f32]) -> u64 {
    values.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, v| {
        v.to_bits()
            .to_le_bytes()
            .iter()
            .fold(h, |h, &b| (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3))
    })
}

fn run(backend: &dyn Backend, batch: usize) -> Vec<f32> {
    let graph = mlp_graph(batch);
    let x = Tensor::new(vec![batch, INPUT], data(batch * INPUT, X_SEED)).unwrap();
    let out = backend.compile(&graph).unwrap().run(&[x]).unwrap();
    assert_eq!(out[0].shape(), [batch, CLASSES]);
    out[0].data().to_vec()
}

#[test]
fn the_graph_is_two_matmuls_around_a_relu() {
    let graph = mlp_graph(8);
    let ops: Vec<&Op> = graph
        .values()
        .iter()
        .filter_map(|v| match &v.source {
            Source::Node { op, .. } => Some(op),
            _ => None,
        })
        .collect();
    assert_eq!(ops, [&Op::MatMul, &Op::Relu, &Op::MatMul]);
}

#[test]
fn the_output_hashes_are_pinned_and_both_backends_agree() {
    let mut got = Vec::new();
    for batch in [1usize, 8, 32] {
        let reference = run(&ReferenceBackend, batch);
        let cpu = run(&CpuBackend, batch);
        assert!(
            reference
                .iter()
                .zip(&cpu)
                .all(|(a, b)| a.to_bits() == b.to_bits()),
            "batch {batch}: reference and cpu differ"
        );
        got.push(fnv(&reference));
    }
    assert_eq!(
        got,
        [
            10_851_165_696_699_687_133u64,
            9_452_000_439_711_095_477,
            3_782_252_911_996_144_250
        ],
        "{got:?}"
    );
}

#[test]
fn the_output_is_the_float64_network_to_within_rounding() {
    let batch = 8;
    let x = data(batch * INPUT, X_SEED);
    let w1 = data(INPUT * mlp_graph::HIDDEN, mlp_graph::W1_SEED);
    let w2 = data(mlp_graph::HIDDEN * CLASSES, mlp_graph::W2_SEED);
    let got = run(&ReferenceBackend, batch);
    for r in 0..batch {
        let hidden: Vec<f64> = (0..mlp_graph::HIDDEN)
            .map(|j| {
                let s: f64 = (0..INPUT)
                    .map(|k| f64::from(x[r * INPUT + k]) * f64::from(w1[k * mlp_graph::HIDDEN + j]))
                    .sum();
                s.max(0.0)
            })
            .collect();
        for c in 0..CLASSES {
            let want: f64 = (0..mlp_graph::HIDDEN)
                .map(|j| hidden[j] * f64::from(w2[j * CLASSES + c]))
                .sum();
            let diff = (f64::from(got[r * CLASSES + c]) - want).abs();
            assert!(
                diff <= 1e-3 * want.abs().max(1.0),
                "row {r} class {c}: {diff}"
            );
        }
    }
}

#[test]
fn a_row_of_a_batch_equals_the_batch_of_one() {
    let one = run(&CpuBackend, 1);
    let eight = run(&CpuBackend, 8);
    // The input generator is a stream, so row 0 of the batch is the batch of one.
    assert!(
        one.iter()
            .zip(&eight[..CLASSES])
            .all(|(a, b)| a.to_bits() == b.to_bits())
    );
}
