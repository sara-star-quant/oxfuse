//! Random valid graphs through the default pipeline. A random program is
//! interpreted step by step; a step the shape rules reject is skipped, so
//! every generated graph is valid by construction.

use oxfuse_core::passes::default_pipeline;
use oxfuse_core::{Graph, Limits, TensorType};
use oxfuse_runtime::{Backend, ReferenceBackend, ReferenceEvaluator};
use proptest::prelude::*;
use std::sync::Arc;

mod common;
use common::*;

fn interface(graph: &Graph) -> (Vec<(String, TensorType)>, Vec<String>, Vec<TensorType>) {
    (
        graph
            .inputs()
            .iter()
            .map(|&id| (graph.label(id), graph.value(id).ty.clone()))
            .collect(),
        graph.output_names().to_vec(),
        graph
            .outputs()
            .iter()
            .map(|&id| graph.value(id).ty.clone())
            .collect(),
    )
}

fn pipeline() -> oxfuse_core::passes::PassManager {
    default_pipeline(Limits::default(), Arc::new(ReferenceEvaluator))
}

/// The properties below prove nothing on graphs where the passes have no work
/// to do. This pins how often the generator gives each pass real work.
#[test]
fn the_generator_gives_every_pass_real_work() {
    use proptest::strategy::{Strategy, ValueTree};
    use proptest::test_runner::TestRunner;
    let mut runner = TestRunner::deterministic();
    const SAMPLES: usize = 400;
    let (mut folded, mut live_merges, mut removed, mut named_twice) = (0, 0, 0, 0);
    for _ in 0..SAMPLES {
        let case = case().new_tree(&mut runner).unwrap().current();
        let (graph, _) = build(&case);
        let full = pipeline().run(&graph).unwrap();
        let without_cse = pipeline().without("cse").run(&graph).unwrap();
        folded += usize::from(full.reports[0].changed > 0);
        removed += usize::from(full.reports[2].changed > 0);
        // A merge that survives dead-code elimination changed the final graph.
        live_merges += usize::from(full.graph.values().len() < without_cse.graph.values().len());
        let outputs = full.graph.outputs();
        named_twice += usize::from(
            outputs
                .iter()
                .enumerate()
                .any(|(i, id)| outputs[..i].contains(id)),
        );
    }
    assert!(
        folded >= SAMPLES / 4,
        "fold had work in only {folded}/{SAMPLES} graphs"
    );
    assert!(
        removed >= SAMPLES / 4,
        "dce had work in only {removed}/{SAMPLES} graphs"
    );
    assert!(
        live_merges >= SAMPLES / 10,
        "cse merges reached the result in only {live_merges}/{SAMPLES} graphs"
    );
    assert!(
        named_twice >= SAMPLES / 10,
        "one value was an output twice in only {named_twice}/{SAMPLES} graphs"
    );
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    #[test]
    fn optimized_graphs_compute_the_same_bits(case in case()) {
        let (graph, inputs) = build(&case);
        let optimized = pipeline().run(&graph).unwrap();
        let plain = ReferenceBackend.compile(&graph).unwrap().run(&inputs).unwrap();
        let fast = ReferenceBackend.compile(&optimized.graph).unwrap().run(&inputs).unwrap();
        prop_assert_eq!(bits(&plain), bits(&fast));
    }

    #[test]
    fn optimization_preserves_the_graph_interface(case in case()) {
        let (graph, _) = build(&case);
        let optimized = pipeline().run(&graph).unwrap();
        prop_assert_eq!(interface(&graph), interface(&optimized.graph));
    }

    #[test]
    fn the_pipeline_is_idempotent(case in case()) {
        let (graph, _) = build(&case);
        let once = pipeline().run(&graph).unwrap();
        let twice = pipeline().run(&once.graph).unwrap();
        prop_assert!(twice.reports.iter().all(|r| r.changed == 0), "{:?}", twice.reports);
        prop_assert_eq!(once.graph, twice.graph);
    }
}
