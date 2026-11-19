//! `analyze` certifies the imported graph, so nothing the importer computes may round
//! outside the graph: a constant scaled at import would be treated as exact.

mod common;

use common::*;
use oxfuse_core::{GraphBuilder, Limits, Op};
use oxfuse_runtime::{Tensor, analyze};

const C: [f32; 4] = [0.3, 0.7, 1.1, 1.9];
const BETA: f32 = 0.1;

fn inputs() -> [Tensor; 2] {
    [
        Tensor::new(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap(),
        Tensor::new(vec![2, 2], vec![1.0, 0.0, 0.0, 1.0]).unwrap(),
    ]
}

/// `Gemm(a, b, C)` with `beta = 0.1` and a constant C: 0.1 * 0.3 does not round to
/// the real product, so the imported graph has to keep that multiplication in the
/// graph, where `analyze` counts its rounding.
#[test]
fn a_scaled_constant_c_is_bounded_like_an_explicit_multiplication() {
    let mut gemm = node("Gemm", &["a", "b", "c"], &["y"]);
    gemm = with_float(gemm, "beta", BETA);
    let imported = load_ok(&model(
        13,
        graph(
            vec![input("a", &[2, 2]), input("b", &[2, 2])],
            vec![init("c", &[2, 2], &C)],
            vec![gemm],
            &["y"],
        ),
    ));

    let mut g = GraphBuilder::new(Limits::default());
    let a = g.input("a", vec![2, 2]).unwrap();
    let b = g.input("b", vec![2, 2]).unwrap();
    let c = g.constant("c", vec![2, 2], C.to_vec()).unwrap();
    let k = g.constant("k", vec![], vec![BETA]).unwrap();
    let mm = g.node("mm", Op::MatMul, &[a, b]).unwrap();
    let bc = g.node("bc", Op::Mul, &[k, c]).unwrap();
    let y = g.node("y", Op::Add, &[mm, bc]).unwrap();
    let explicit = g.finish(&[y]).unwrap();

    let got = analyze(&imported, &inputs()).unwrap().remove(0);
    let want = analyze(&explicit, &inputs()).unwrap().remove(0);
    assert_eq!(got.values, want.values, "the same bits");
    assert_eq!(got.bounds, want.bounds, "the same bound");
}
