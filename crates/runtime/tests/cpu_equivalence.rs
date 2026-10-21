//! Random valid graphs: the fast backend must return the reference backend's
//! exact bits, with and without the optimization pipeline.

mod common;

use common::{bits, build, case};
use oxfuse_core::Limits;
use oxfuse_core::passes::default_pipeline;
use oxfuse_runtime::{Backend, CpuBackend, ReferenceBackend, ReferenceEvaluator};
use proptest::prelude::*;
use std::sync::Arc;

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    #[test]
    fn the_cpu_backend_returns_the_reference_bits(case in case()) {
        let (graph, inputs) = build(&case);
        let reference = ReferenceBackend.compile(&graph).unwrap().run(&inputs).unwrap();
        let cpu = CpuBackend.compile(&graph).unwrap().run(&inputs).unwrap();
        prop_assert_eq!(bits(&reference), bits(&cpu));
    }

    #[test]
    fn the_cpu_backend_returns_the_reference_bits_on_optimized_graphs(case in case()) {
        let (graph, inputs) = build(&case);
        let optimized = default_pipeline(Limits::default(), Arc::new(ReferenceEvaluator))
            .run(&graph)
            .unwrap()
            .graph;
        let reference = ReferenceBackend.compile(&graph).unwrap().run(&inputs).unwrap();
        let cpu = CpuBackend.compile(&optimized).unwrap().run(&inputs).unwrap();
        prop_assert_eq!(bits(&reference), bits(&cpu));
    }
}

/// Above the size where Sigmoid is split across threads (with the `parallel`
/// feature), including a length that leaves a short last chunk.
#[test]
fn a_large_sigmoid_returns_the_reference_bits() {
    use oxfuse_core::{GraphBuilder, Op};
    use oxfuse_runtime::Tensor;
    const SPECIAL: [f32; 8] = [
        f32::NAN,
        f32::INFINITY,
        f32::NEG_INFINITY,
        -0.0,
        0.0,
        -90.0,
        88.5,
        f32::MIN_POSITIVE / 2.0,
    ];
    for len in [65_535usize, 65_536, 200_003] {
        let data: Vec<f32> = (0..len)
            .map(|i| match i % 11 {
                0 => SPECIAL[(i / 11) % SPECIAL.len()],
                _ => (i as f32 * 0.37).sin() * 20.0,
            })
            .collect();
        let mut b = GraphBuilder::new(Limits::default());
        let x = b.input("x", vec![len]).unwrap();
        let y = b.node("y", Op::Sigmoid, &[x]).unwrap();
        let graph = b.finish(&[y]).unwrap();
        let inputs = [Tensor::new(vec![len], data).unwrap()];
        let reference = ReferenceBackend
            .compile(&graph)
            .unwrap()
            .run(&inputs)
            .unwrap();
        let cpu = CpuBackend.compile(&graph).unwrap().run(&inputs).unwrap();
        assert_eq!(bits(&reference), bits(&cpu), "length {len}");
    }
}

/// Buffer reuse must never change a result. Each graph shares operands,
/// reuses a buffer in a chain, lists an intermediate as an output, or feeds
/// a scalar into an in-place op, in both operand orders.
mod reuse {
    use super::*;
    use oxfuse_core::{Graph, GraphBuilder, Op, ValueId};
    use oxfuse_runtime::Tensor;

    const N: usize = 6;
    const VALUES: [f32; N] = [f32::NAN, -0.0, 3.5, f32::INFINITY, -2.25, 0.0];

    fn vector(seed: usize) -> Tensor {
        let data = (0..N).map(|i| VALUES[(i + seed) % N]).collect();
        Tensor::new(vec![N], data).unwrap()
    }

    fn check(build: impl FnOnce(&mut GraphBuilder, ValueId, ValueId, ValueId) -> Vec<ValueId>) {
        let mut b = GraphBuilder::new(Limits::default());
        let x = b.input("x", vec![N]).unwrap();
        let s = b.input("s", vec![]).unwrap();
        let c = b.constant("c", vec![N], VALUES.to_vec()).unwrap();
        let outputs = build(&mut b, x, s, c);
        let graph: Graph = b.finish(&outputs).unwrap();
        for (seed, scalar) in VALUES.into_iter().enumerate() {
            let inputs = [vector(seed), Tensor::new(vec![], vec![scalar]).unwrap()];
            let reference = ReferenceBackend
                .compile(&graph)
                .unwrap()
                .run(&inputs)
                .unwrap();
            let cpu = CpuBackend.compile(&graph).unwrap().run(&inputs).unwrap();
            assert_eq!(bits(&reference), bits(&cpu), "seed {seed}");
        }
    }

    #[test]
    fn a_value_added_to_itself() {
        check(|b, x, _, _| {
            let y = b.node("y", Op::Relu, &[x]).unwrap();
            vec![b.node("z", Op::Add, &[y, y]).unwrap()]
        });
    }

    #[test]
    fn a_chain_reuses_one_buffer() {
        check(|b, x, _, _| {
            let a = b.node("a", Op::Sigmoid, &[x]).unwrap();
            let r = b.node("r", Op::Relu, &[a]).unwrap();
            let m = b.node("m", Op::Mul, &[r, r]).unwrap();
            vec![
                b.node("n", Op::Reshape { shape: vec![2, 3] }, &[m])
                    .unwrap(),
            ]
        });
    }

    #[test]
    fn an_intermediate_that_is_also_an_output() {
        check(|b, x, _, c| {
            let y = b.node("y", Op::Relu, &[x]).unwrap();
            let z = b.node("z", Op::Add, &[y, c]).unwrap();
            vec![y, z]
        });
    }

    #[test]
    fn an_output_listed_twice_and_used_later() {
        check(|b, x, _, c| {
            let y = b.node("y", Op::Sigmoid, &[x]).unwrap();
            let z = b.node("z", Op::Mul, &[c, y]).unwrap();
            vec![y, z, y]
        });
    }

    #[test]
    fn a_scalar_in_either_operand_position() {
        check(|b, x, s, c| {
            let y = b.node("y", Op::Relu, &[x]).unwrap();
            let l = b.node("l", Op::Add, &[s, y]).unwrap();
            let r = b.node("r", Op::Mul, &[l, s]).unwrap();
            let k = b.node("k", Op::Add, &[c, r]).unwrap();
            vec![b.node("o", Op::Mul, &[s, k]).unwrap()]
        });
    }

    #[test]
    fn inputs_and_constants_are_never_overwritten() {
        check(|b, x, s, c| {
            let a = b.node("a", Op::Add, &[x, c]).unwrap();
            let r = b
                .node("r", Op::Reshape { shape: vec![3, 2] }, &[c])
                .unwrap();
            let t = b.node("t", Op::Reshape { shape: vec![N] }, &[r]).unwrap();
            let m = b.node("m", Op::Mul, &[a, t]).unwrap();
            vec![b.node("o", Op::Add, &[m, s]).unwrap(), x, c]
        });
    }

    #[test]
    fn a_reshape_output_alongside_its_source() {
        check(|b, x, _, _| {
            let y = b.node("y", Op::Relu, &[x]).unwrap();
            let r = b
                .node("r", Op::Reshape { shape: vec![3, 2] }, &[y])
                .unwrap();
            vec![r, y]
        });
    }
}
