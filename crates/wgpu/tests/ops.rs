//! Transpose and MatMul on a real adapter against the reference and the f64 oracle.

mod common;

use common::{available, bits, data, device, measurements_dir, reference, tensor};
use oxfuse_core::Op;
use oxfuse_runtime::Tensor;
use oxfuse_wgpu::{matmul_bound, ulp_distance};

// Around the 16-thread tile and the 64-wide block of the default configuration.
const EDGES: [usize; 8] = [1, 15, 16, 17, 33, 63, 64, 65];

#[test]
fn transpose_is_bit_exact_against_the_reference() {
    if !available() {
        return;
    }
    for &rows in &EDGES {
        for &cols in &EDGES {
            let t = tensor(&[rows, cols], (rows * 100 + cols) as u64, false);
            let op = Op::Transpose { perm: vec![1, 0] };
            assert_eq!(
                bits(&device(op.clone(), std::slice::from_ref(&t))),
                bits(&reference(op, &[t])),
                "{rows}x{cols}"
            );
        }
    }
}

#[test]
fn permute_is_bit_exact_for_higher_ranks() {
    if !available() {
        return;
    }
    for perm in [vec![2, 0, 1], vec![1, 2, 0], vec![0, 2, 1], vec![0, 1, 2]] {
        let t = tensor(&[3, 5, 17], 3, false);
        let op = Op::Transpose { perm: perm.clone() };
        assert_eq!(
            bits(&device(op.clone(), std::slice::from_ref(&t))),
            bits(&reference(op, &[t])),
            "perm {perm:?}"
        );
    }
    // The identity on rank two must not be taken for the swap.
    let t = tensor(&[3, 5], 6, false);
    let op = Op::Transpose { perm: vec![0, 1] };
    assert_eq!(
        bits(&device(op.clone(), std::slice::from_ref(&t))),
        bits(t.data())
    );
    let t = tensor(&[2, 1, 2, 1, 2, 1, 2, 3], 4, false);
    let op = Op::Transpose {
        perm: vec![7, 6, 5, 4, 3, 2, 1, 0],
    };
    assert_eq!(
        bits(&device(op.clone(), std::slice::from_ref(&t))),
        bits(&reference(op, &[t]))
    );
}

/// f64 accumulation of the same f32 inputs: the oracle the documented bound is stated against.
fn oracle(a: &Tensor, b: &Tensor, m: usize, k: usize, n: usize) -> Vec<(f64, f64)> {
    let mut out = Vec::with_capacity(m * n);
    for i in 0..m {
        let row: Vec<f32> = (0..k).map(|x| a.data()[i * k + x]).collect();
        for j in 0..n {
            let col: Vec<f32> = (0..k).map(|x| b.data()[x * n + j]).collect();
            let exact: f64 = row
                .iter()
                .zip(&col)
                .map(|(x, y)| f64::from(*x) * f64::from(*y))
                .sum();
            out.push((exact, matmul_bound(&row, &col)));
        }
    }
    out
}

fn matmul_cases() -> Vec<(usize, usize, usize)> {
    let mut cases = Vec::new();
    for &m in &EDGES {
        for &k in &EDGES {
            for &n in &EDGES {
                cases.push((m, k, n));
            }
        }
    }
    cases.push((64, 64, 64));
    cases.push((1, 257, 1));
    cases
}

#[test]
fn matmul_stays_inside_the_documented_bound_of_the_f64_oracle() {
    if !available() {
        return;
    }
    for (m, k, n) in matmul_cases() {
        let a = tensor(&[m, k], (m * 7 + k) as u64, true);
        let b = tensor(&[k, n], (n * 13 + k) as u64, true);
        let got = device(Op::MatMul, &[a.clone(), b.clone()]);
        for (idx, (exact, bound)) in oracle(&a, &b, m, k, n).into_iter().enumerate() {
            let err = (f64::from(got[idx]) - exact).abs();
            assert!(
                err <= bound,
                "{m}x{k}x{n} element {idx}: error {err:e} exceeds bound {bound:e}"
            );
        }
    }
}

#[test]
fn matmul_ulp_distance_from_the_reference_is_recorded() {
    if !available() {
        return;
    }
    let mut worst = 0u32;
    let mut worst_case = (0, 0, 0);
    for (m, k, n) in matmul_cases() {
        let a = tensor(&[m, k], (m * 7 + k) as u64, true);
        let b = tensor(&[k, n], (n * 13 + k) as u64, true);
        let got = device(Op::MatMul, &[a.clone(), b.clone()]);
        let want = reference(Op::MatMul, &[a, b]);
        for (g, w) in got.iter().zip(&want) {
            let d = ulp_distance(*g, *w).expect("finite inputs give finite sums");
            if d > worst {
                worst = d;
                worst_case = (m, k, n);
            }
        }
    }
    let line =
        format!("matmul max ulp distance from the reference: {worst} (case {worst_case:?})\n");
    std::fs::write(measurements_dir().join("matmul-ulp.txt"), &line).unwrap();
    eprintln!("{line}");
}

#[test]
fn data_helper_is_deterministic_and_finite_when_asked() {
    assert_eq!(bits(&data(50, 1, true)), bits(&data(50, 1, true)));
    assert!(
        data(500, 2, true)
            .iter()
            .all(|v| v.is_finite() && (*v == 0.0 || v.is_normal()))
    );
}
