//! Gemm lowering: transposes, scaling, and the optional bias.

mod common;

use common::*;
use oxfuse_onnx::{OnnxError, proto::*};

/// Small integer values, so every product and sum is exact in float32.
fn fill(n: usize, seed: usize) -> Vec<f32> {
    (0..n)
        .map(|i| ((i * 7 + seed * 3) % 5) as f32 - 2.0)
        .collect()
}

struct Case {
    trans_a: bool,
    trans_b: bool,
    alpha: f32,
    beta: f32,
    /// Shape of C and whether it is a constant (initializer).
    c: Option<(Vec<i64>, bool)>,
}

impl Default for Case {
    fn default() -> Self {
        Case {
            trans_a: false,
            trans_b: false,
            alpha: 1.0,
            beta: 1.0,
            c: None,
        }
    }
}

const M: usize = 2;
const K: usize = 3;
const N: usize = 4;

fn build(case: &Case) -> ModelProto {
    let a_shape: Vec<i64> = if case.trans_a {
        vec![K as i64, M as i64]
    } else {
        vec![M as i64, K as i64]
    };
    let b_shape: Vec<i64> = if case.trans_b {
        vec![N as i64, K as i64]
    } else {
        vec![K as i64, N as i64]
    };
    let mut inputs = vec![input("a", &a_shape), input("b", &b_shape)];
    let mut inits = vec![];
    let mut args = vec!["a", "b"];
    if let Some((shape, constant)) = &case.c {
        let count: i64 = shape.iter().product();
        if *constant {
            inits.push(init("c", shape, &fill(count as usize, 9)));
        } else {
            inputs.push(input("c", shape));
        }
        args.push("c");
    }
    let mut n = node("Gemm", &args, &["y"]);
    n = with_int(n, "transA", case.trans_a as i64);
    n = with_int(n, "transB", case.trans_b as i64);
    n = with_float(n, "alpha", case.alpha);
    n = with_float(n, "beta", case.beta);
    model(13, graph(inputs, inits, vec![n], &["y"]))
}

/// Expected output from the Gemm definition, in float64.
fn expected(case: &Case) -> Vec<f32> {
    let a = fill(M * K, 1);
    let b = fill(K * N, 2);
    let at = |i: usize, k: usize| if case.trans_a { a[k * M + i] } else { a[i * K + k] } as f64;
    let bt = |k: usize, j: usize| if case.trans_b { b[j * K + k] } else { b[k * N + j] } as f64;
    let c = case.c.as_ref().map(|(shape, _)| {
        let count: i64 = shape.iter().product();
        (shape.clone(), fill(count as usize, 9))
    });
    let mut out = vec![];
    for i in 0..M {
        for j in 0..N {
            let mut acc = 0.0f64;
            for k in 0..K {
                acc += at(i, k) * bt(k, j);
            }
            let mut v = case.alpha as f64 * acc;
            if let Some((shape, data)) = &c {
                let idx = match shape.as_slice() {
                    [] | [1] | [1, 1] => 0,
                    [n] if *n as usize == N => j,
                    [1, n] if *n as usize == N => j,
                    [m, 1] if *m as usize == M => i,
                    [_, _] => i * N + j,
                    other => panic!("unexpected C shape {other:?}"),
                };
                v += case.beta as f64 * data[idx] as f64;
            }
            out.push(v as f32);
        }
    }
    out
}

fn check(case: Case) -> Vec<&'static str> {
    let g = load_ok(&build(&case));
    let a = fill(M * K, 1);
    let b = fill(K * N, 2);
    let a_shape: &[usize] = if case.trans_a { &[K, M] } else { &[M, K] };
    let b_shape: &[usize] = if case.trans_b { &[N, K] } else { &[K, N] };
    let c_data;
    let mut feeds: Vec<(&[usize], &[f32])> = vec![(a_shape, &a), (b_shape, &b)];
    let c_shape: Vec<usize>;
    if let Some((shape, false)) = &case.c {
        let count: i64 = shape.iter().product();
        c_data = fill(count as usize, 9);
        c_shape = shape.iter().map(|d| *d as usize).collect();
        feeds.push((&c_shape, &c_data));
    }
    assert_eq!(run(&g, &feeds), vec![expected(&case)]);
    assert_eq!(
        g.label(g.outputs()[0]),
        "y",
        "the last node carries the ONNX name"
    );
    assert_eq!(shape_of_output(&g, 0), vec![M, N]);
    ops(&g)
}

#[test]
fn gemm_plain_emits_matmul_add_only() {
    let c = Case {
        c: Some((vec![N as i64], true)),
        ..Case::default()
    };
    assert_eq!(check(c), vec!["MatMul", "Add"]);
}

#[test]
fn gemm_without_c_is_a_single_matmul() {
    assert_eq!(check(Case::default()), vec!["MatMul"]);
}

#[test]
fn gemm_trans_a_b_all_four() {
    for (ta, tb) in [(false, false), (true, false), (false, true), (true, true)] {
        let ops = check(Case {
            trans_a: ta,
            trans_b: tb,
            c: Some((vec![M as i64, N as i64], true)),
            ..Case::default()
        });
        let transposes = ops.iter().filter(|o| **o == "Transpose").count();
        assert_eq!(transposes, ta as usize + tb as usize, "{ta} {tb}");
    }
}

#[test]
fn gemm_alpha_values_1_0_neg_other() {
    for alpha in [1.0, 0.0, -1.0, 2.5] {
        let ops = check(Case {
            alpha,
            c: Some((vec![N as i64], true)),
            ..Case::default()
        });
        // alpha is never dropped: only exactly 1.0 skips the multiply.
        assert_eq!(ops.contains(&"Mul"), alpha != 1.0, "alpha {alpha}: {ops:?}");
    }
}

#[test]
fn gemm_beta_values_1_0_neg_other() {
    for beta in [1.0, 0.0, -1.0, 2.5] {
        for constant in [true, false] {
            let ops = check(Case {
                beta,
                c: Some((vec![M as i64, N as i64], constant)),
                ..Case::default()
            });
            // A constant C is scaled at import; a runtime C gets a Mul; beta 0 drops C.
            let muls = ops.iter().filter(|o| **o == "Mul").count();
            assert_eq!(
                muls,
                usize::from(beta != 1.0 && beta != 0.0 && !constant),
                "beta {beta} const {constant}"
            );
        }
    }
}

#[test]
fn gemm_alpha_and_beta_together_with_transposes() {
    check(Case {
        trans_a: true,
        trans_b: true,
        alpha: -2.0,
        beta: 0.5,
        c: Some((vec![1, N as i64], true)),
    });
}

#[test]
fn gemm_c_shapes() {
    for shape in [
        vec![],
        vec![1],
        vec![N as i64],
        vec![1, N as i64],
        vec![M as i64, 1],
        vec![M as i64, N as i64],
        vec![1, 1],
    ] {
        check(Case {
            c: Some((shape.clone(), true)),
            beta: 2.0,
            ..Case::default()
        });
    }
}

#[test]
fn gemm_runtime_c_same_shape_or_scalar() {
    for shape in [vec![M as i64, N as i64], vec![]] {
        check(Case {
            c: Some((shape, false)),
            ..Case::default()
        });
    }
}

#[test]
fn gemm_c_wrong_shape_rejected() {
    for shape in [
        vec![3],
        vec![M as i64, 3],
        vec![3, M as i64, N as i64],
        vec![N as i64 + 1],
    ] {
        let m = build(&Case {
            c: Some((shape.clone(), true)),
            ..Case::default()
        });
        match load(&m) {
            Err(OnnxError::Shape { reason, .. }) => {
                assert!(reason.contains("broadcast"), "{shape:?}: {reason}")
            }
            other => panic!("{shape:?}: {other:?}"),
        }
    }
}

#[test]
fn gemm_non_constant_c_broadcast_rejected() {
    let m = build(&Case {
        c: Some((vec![N as i64], false)),
        ..Case::default()
    });
    match load(&m) {
        Err(OnnxError::Shape { reason, .. }) => {
            assert!(reason.contains("not a constant"), "{reason}")
        }
        other => panic!("{other:?}"),
    }
}

#[test]
fn gemm_empty_optional_input_name() {
    let n = node("Gemm", &["a", "b", ""], &["y"]);
    let m = model(
        13,
        graph(
            vec![
                input("a", &[M as i64, K as i64]),
                input("b", &[K as i64, N as i64]),
            ],
            vec![],
            vec![n],
            &["y"],
        ),
    );
    assert_eq!(ops(&load_ok(&m)), vec!["MatMul"]);
}

#[test]
fn gemm_rank_other_than_two_rejected() {
    let m = model(
        13,
        graph(
            vec![input("a", &[2, 2, 2]), input("b", &[2, 2])],
            vec![],
            vec![node("Gemm", &["a", "b"], &["y"])],
            &["y"],
        ),
    );
    assert!(matches!(load(&m), Err(OnnxError::Shape { .. })));
}

#[test]
fn gemm_unknown_attribute_rejected() {
    let n = with_int(node("Gemm", &["a", "b"], &["y"]), "broadcast", 1);
    let m = model(
        13,
        graph(
            vec![
                input("a", &[M as i64, K as i64]),
                input("b", &[K as i64, N as i64]),
            ],
            vec![],
            vec![n],
            &["y"],
        ),
    );
    assert!(
        matches!(load(&m), Err(OnnxError::UnsupportedAttr { attr, .. }) if attr == "broadcast")
    );
}

#[test]
fn gemm_trans_flag_other_than_0_or_1_rejected() {
    let n = with_int(node("Gemm", &["a", "b"], &["y"]), "transA", 2);
    let m = model(
        13,
        graph(
            vec![
                input("a", &[M as i64, K as i64]),
                input("b", &[K as i64, N as i64]),
            ],
            vec![],
            vec![n],
            &["y"],
        ),
    );
    assert!(matches!(load(&m), Err(OnnxError::Shape { .. })));
}

#[test]
fn gemm_input_count_is_checked() {
    let ab = || vec![input("a", &[2, 2]), input("b", &[2, 2])];
    for args in [&["a"][..], &["a", "b", "a", "b"], &["", "b"], &["a", ""]] {
        let m = model(
            13,
            graph(ab(), vec![], vec![node("Gemm", args, &["y"])], &["y"]),
        );
        match load(&m) {
            Err(OnnxError::Shape { reason, .. }) => {
                assert!(reason.contains("2 or 3 inputs"), "{args:?}: {reason}")
            }
            other => panic!("{args:?}: {:?}", other.map(|_| ())),
        }
    }
}

fn gemm_with_c(beta: f32, c: Vec<f32>, runtime_c: bool) -> ModelProto {
    let mut n = node("Gemm", &["a", "b", "c"], &["y"]);
    n = with_float(n, "beta", beta);
    let mut inputs = vec![input("a", &[1, 1]), input("b", &[1, 1])];
    let mut inits = vec![];
    if runtime_c {
        inputs.push(input("c", &[1, 1]));
    } else {
        inits.push(init("c", &[1, 1], &c));
    }
    model(13, graph(inputs, inits, vec![n], &["y"]))
}

#[test]
fn gemm_beta_zero_ignores_a_non_finite_c() {
    // ONNX adds beta * C only when beta is not zero, so inf or NaN in C
    // cannot reach the result (onnxruntime and the reference evaluator agree).
    for beta in [0.0, -0.0] {
        for special in [f32::INFINITY, f32::NAN] {
            let g = load_ok(&gemm_with_c(beta, vec![special], false));
            assert_eq!(ops(&g), vec!["MatMul"], "beta {beta}");
            assert_eq!(g.label(g.outputs()[0]), "y");
            assert_eq!(
                run(&g, &[(&[1, 1], &[2.0]), (&[1, 1], &[3.0])]),
                vec![vec![6.0]]
            );
            let g = load_ok(&gemm_with_c(beta, vec![], true));
            assert_eq!(
                run(
                    &g,
                    &[(&[1, 1], &[2.0]), (&[1, 1], &[3.0]), (&[1, 1], &[special])]
                ),
                vec![vec![6.0]]
            );
        }
    }
}

#[test]
fn gemm_beta_zero_still_checks_the_shape_of_c() {
    let mut m = gemm_with_c(0.0, vec![1.0], false);
    m.graph.as_mut().unwrap().initializer[0] = init("c", &[3], &[1.0; 3]);
    match load(&m) {
        Err(OnnxError::Shape { reason, .. }) => assert!(reason.contains("broadcast"), "{reason}"),
        other => panic!("{:?}", other.map(|_| ())),
    }
}
