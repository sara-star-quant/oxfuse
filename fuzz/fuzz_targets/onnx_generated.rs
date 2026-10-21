//! Generated valid ONNX models: the importer must accept every one, and the
//! fast backend must match the reference on the result.

#![no_main]

use arbitrary::Unstructured;
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let mut u = Unstructured::new(data);
    let Ok(bytes) = oxfuse_fuzz::onnx::generate(&mut u) else {
        return;
    };
    let graph = oxfuse_fuzz::onnx::import_small(&bytes)
        .unwrap_or_else(|e| panic!("a generated model failed to import: {e}"));
    let seed = data.iter().fold(0usize, |acc, &b| {
        acc.wrapping_mul(31).wrapping_add(usize::from(b))
    });
    oxfuse_fuzz::compare_graph(&graph, seed);
});
