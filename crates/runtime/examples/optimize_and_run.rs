//! Builds `relu(x @ (w + w) + b)`, optimizes it, and runs it on both CPU
//! backends. The passes fold `w + w` into one constant, and the fast backend
//! returns the same bits as the reference one.
//! Run with `cargo run --release -p oxfuse-runtime --example optimize_and_run`.

use std::sync::Arc;

use oxfuse_core::passes::default_pipeline;
use oxfuse_core::{GraphBuilder, Limits, Op};
use oxfuse_runtime::{Backend, CpuBackend, ReferenceBackend, ReferenceEvaluator, Tensor};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let limits = Limits::default();
    let mut b = GraphBuilder::new(limits.clone());
    let x = b.input("x", vec![2, 3])?;
    let w = b.constant("w", vec![3, 2], vec![0.5, -1.0, 1.5, 2.0, -0.5, 1.0])?;
    let bias = b.constant("b", vec![2, 2], vec![0.25, -0.25, 0.5, -0.5])?;
    let w2 = b.node("w2", Op::Add, &[w, w])?;
    let m = b.node("m", Op::MatMul, &[x, w2])?;
    let s = b.node("s", Op::Add, &[m, bias])?;
    let y = b.node("y", Op::Relu, &[s])?;
    let graph = b.finish(&[y])?;

    let optimized = default_pipeline(limits, Arc::new(ReferenceEvaluator)).run(&graph)?;
    for r in &optimized.reports {
        println!(
            "{:<6} values {} -> {}, changed {}",
            r.pass, r.values_before, r.values_after, r.changed
        );
    }

    let input = Tensor::new(vec![2, 3], vec![1.0, -2.0, 3.0, 4.0, 5.0, -6.0])?;
    let want = ReferenceBackend
        .compile(&optimized.graph)?
        .run(std::slice::from_ref(&input))?;
    let got = CpuBackend
        .compile(&optimized.graph)?
        .run(std::slice::from_ref(&input))?;
    println!("y = {:?}", got[0].data());

    let bits = |t: &Tensor| t.data().iter().map(|v| v.to_bits()).collect::<Vec<_>>();
    assert_eq!(
        bits(&want[0]),
        bits(&got[0]),
        "the backends must agree bit for bit"
    );
    println!("reference and cpu backends agree bit for bit");
    Ok(())
}
