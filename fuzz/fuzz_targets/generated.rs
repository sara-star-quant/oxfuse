//! The same comparison as `differential`, on graphs generated to be valid, so
//! nearly every execution reaches it.

#![no_main]

use arbitrary::Unstructured;
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let mut u = Unstructured::new(data);
    let Ok(text) = oxfuse_fuzz::generate(&mut u) else {
        return;
    };
    let seed = data
        .iter()
        .fold(0usize, |acc, &b| acc.wrapping_mul(31).wrapping_add(usize::from(b)));
    assert!(
        oxfuse_fuzz::compare(&text, seed),
        "a generated graph failed to load ({:?}): {}",
        oxfuse_fuzz::load_error(&text),
        &text[..text.len().min(300)]
    );
});
