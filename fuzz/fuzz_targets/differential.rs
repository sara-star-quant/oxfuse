//! Any graph that loads must give the same bits on the reference backend,
//! on the fast backend, and with and without the optimization passes. Raw
//! bytes rarely form a valid graph; `generated` covers that side.

#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let Ok(text) = std::str::from_utf8(data) else {
        return;
    };
    let seed = data
        .iter()
        .fold(0usize, |acc, &b| acc.wrapping_mul(31).wrapping_add(usize::from(b)));
    oxfuse_fuzz::compare(text, seed);
});
