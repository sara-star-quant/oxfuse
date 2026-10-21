//! Arbitrary bytes into the ONNX importer: no panic, no hang, memory bounded
//! by the limits. A model that imports is also run on both CPU backends.

#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    if let Ok(graph) = oxfuse_fuzz::onnx::import_small(data)
        && !oxfuse_fuzz::onnx::has_exotic_nan(&graph)
    {
        let seed = data.iter().fold(0usize, |acc, &b| {
            acc.wrapping_mul(31).wrapping_add(usize::from(b))
        });
        oxfuse_fuzz::compare_graph(&graph, seed);
    }
});
