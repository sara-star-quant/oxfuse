//! A row's result does not depend on how many rows ride along: row `i` of an
//! `m`-row run equals the run of that row alone, bit for bit. There is no
//! split-K and the register block, the tail columns and the thread split only
//! regroup rows, never the sum inside one element.

mod common;

use common::comparable_bits;
use oxfuse_core::{GraphBuilder, Limits, Op};
use oxfuse_runtime::{Backend, CpuBackend, ReferenceBackend, Tensor};

/// Rows around the 6-row register block and the 16-row thread task. With the
/// sizes below only 64 and 70 reach the thread threshold, and 70 leaves a short
/// last task (70 = 4 * 16 + 6).
const ROWS: [usize; 18] = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 31, 32, 33, 64, 70,
];
/// `K` crosses the 256-deep k-tile; `N` and `P` leave tail columns of 6 and 10.
const K: usize = 300;
const N: usize = 118;
const P: usize = 10;

/// A deterministic value in [-2, 2).
fn value(seed: usize) -> f32 {
    let h = (seed as u32).wrapping_mul(2_654_435_761) >> 8;
    (h % 4096) as f32 / 1024.0 - 2.0
}

/// `m` rows of `K` values. Every fifth row carries one special value, so the
/// other rows stay finite and the comparison is not all NaN.
fn rows(m: usize, salt: usize) -> Vec<f32> {
    const SPECIAL: [f32; 5] = [f32::INFINITY, f32::NEG_INFINITY, -0.0, 0.0, f32::NAN];
    let mut x: Vec<f32> = (0..m * K).map(|i| value(i + salt)).collect();
    for r in (2..m).step_by(5) {
        x[r * K + (r * 13) % K] = SPECIAL[(r / 5) % SPECIAL.len()];
    }
    x
}

/// x[m, K] -> MatMul W1[K, N] -> Relu -> MatMul W2[N, P] -> Sigmoid.
fn mlp(backend: &dyn Backend, m: usize) -> Box<dyn oxfuse_runtime::Executable> {
    let mut b = GraphBuilder::new(Limits::default());
    let x = b.input("x", vec![m, K]).unwrap();
    let w1 = b.constant("w1", vec![K, N], finite(K * N, 1)).unwrap();
    let h = b.node("h", Op::MatMul, &[x, w1]).unwrap();
    let r = b.node("r", Op::Relu, &[h]).unwrap();
    let w2 = b.constant("w2", vec![N, P], finite(N * P, 2)).unwrap();
    let o = b.node("o", Op::MatMul, &[r, w2]).unwrap();
    let y = b.node("y", Op::Sigmoid, &[o]).unwrap();
    backend.compile(&b.finish(&[y]).unwrap()).unwrap()
}

/// Weights stay finite so a special in `x` is the only source of one.
fn finite(len: usize, salt: usize) -> Vec<f32> {
    (0..len)
        .map(|i| (((i + salt * 7919) as u32).wrapping_mul(40_503) % 2000) as f32 / 1000.0 - 1.0)
        .collect()
}

fn bits_of(t: &Tensor) -> Vec<u32> {
    t.data().iter().map(|&v| comparable_bits(v)).collect()
}

fn check(backend: &dyn Backend, name: &str) {
    let single = mlp(backend, 1);
    for m in ROWS {
        let x = rows(m, m);
        let full = mlp(backend, m)
            .run(&[Tensor::new(vec![m, K], x.clone()).unwrap()])
            .unwrap();
        let full = bits_of(&full[0]);
        assert!(
            full.iter().any(|&b| !f32::from_bits(b).is_nan()),
            "{name}: all NaN"
        );
        for r in 0..m {
            let row = Tensor::new(vec![1, K], x[r * K..(r + 1) * K].to_vec()).unwrap();
            let alone = bits_of(&single.run(&[row]).unwrap()[0]);
            assert_eq!(
                &full[r * P..(r + 1) * P],
                alone.as_slice(),
                "{name}: row {r} of {m} differs from the row alone"
            );
        }
    }
}

#[test]
fn the_reference_backend_is_batch_invariant() {
    check(&ReferenceBackend, "reference");
}

#[test]
fn the_cpu_backend_is_batch_invariant() {
    check(&CpuBackend, "cpu");
}
