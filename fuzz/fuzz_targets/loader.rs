//! Arbitrary text into the graph loader: it must reject or accept, never
//! panic or hang, and anything it accepts stays inside the limits.

#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let Ok(text) = std::str::from_utf8(data) else {
        return;
    };
    if let Ok(graph) = oxfuse_cli::load_graph(text, &oxfuse_fuzz::small()) {
        graph
            .verify_limits(&oxfuse_fuzz::small())
            .expect("a loaded graph stays inside the limits");
    }
});
