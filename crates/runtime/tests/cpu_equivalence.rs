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
