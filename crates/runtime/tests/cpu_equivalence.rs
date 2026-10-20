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
