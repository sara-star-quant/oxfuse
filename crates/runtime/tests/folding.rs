//! Folding with the real kernels must give the bits the run-time path gives.

use oxfuse_core::passes::{ConstantFolding, Pass};
use oxfuse_core::{GraphBuilder, Limits, Op};
use oxfuse_runtime::{Backend, ReferenceBackend, ReferenceEvaluator, Tensor};
use std::sync::Arc;

#[test]
fn folded_matmul_and_transpose_match_runtime_bits() {
    let mut b = GraphBuilder::new(Limits::default());
    let x = b.input("x", vec![2, 2]).unwrap();
    let a = b
        .constant("a", vec![2, 3], vec![0.1, -0.7, 1.3, 2.9, -3.1, 0.5])
        .unwrap();
    let w = b
        .constant("w", vec![3, 2], vec![1.1, 0.3, -0.9, 2.2, 0.8, -1.4])
        .unwrap();
    let product = b.node("product", Op::MatMul, &[a, w]).unwrap();
    let flipped = b
        .node("flipped", Op::Transpose { perm: vec![1, 0] }, &[product])
        .unwrap();
    let sig = b.node("sig", Op::Sigmoid, &[flipped]).unwrap();
    let y = b.node("y", Op::Add, &[x, sig]).unwrap();
    let graph = b.finish(&[y]).unwrap();

    let folding = ConstantFolding::new(Arc::new(ReferenceEvaluator), Limits::default());
    let folded = folding.run(&graph).unwrap();
    assert_eq!(folded.changed, 3);

    let input = Tensor::new(vec![2, 2], vec![0.25, -1.5, 3.0, 0.125]).unwrap();
    let plain = ReferenceBackend
        .compile(&graph)
        .unwrap()
        .run(std::slice::from_ref(&input))
        .unwrap();
    let fast = ReferenceBackend
        .compile(&folded.graph)
        .unwrap()
        .run(std::slice::from_ref(&input))
        .unwrap();
    let bits = |t: &[Tensor]| t[0].data().iter().map(|v| v.to_bits()).collect::<Vec<_>>();
    assert_eq!(bits(&plain), bits(&fast));
}
