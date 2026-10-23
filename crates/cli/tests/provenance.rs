//! `docs/provenance.md` names a source for every function that computes a
//! result. A function without a row, or a row naming a function that is gone,
//! fails the build.

use std::collections::BTreeSet;
use std::fs;
use std::path::PathBuf;

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// The files whose functions compute or move results.
const FILES: [&str; 7] = [
    "crates/runtime/src/math.rs",
    "crates/runtime/src/kernels.rs",
    "crates/runtime/src/cpu/vecmath.rs",
    "crates/runtime/src/cpu/matmul.rs",
    "crates/runtime/src/cpu/kernels.rs",
    "crates/runtime/src/cpu/elementwise.rs",
    "crates/wgpu/src/shaders.rs",
];

/// Names of the functions before the test module. `main` is the WGSL entry
/// point inside the shader strings, not a Rust function.
fn functions(path: &str) -> BTreeSet<String> {
    let source = fs::read_to_string(root().join(path)).unwrap();
    let code = source.split("\nmod tests {").next().unwrap();
    code.lines()
        .filter_map(|line| {
            let line = line.trim_start();
            let line = line.strip_prefix("pub").map_or(line, |rest| {
                rest.trim_start_matches(|c| c != ' ').trim_start()
            });
            let line = line.strip_prefix("const ").unwrap_or(line);
            let name = line.strip_prefix("fn ")?;
            let end = name.find(|c: char| !(c.is_alphanumeric() || c == '_'))?;
            Some(name[..end].to_owned())
        })
        .filter(|name| name != "main")
        .collect()
}

/// `path::name` items in backticks in the ledger's tables.
fn listed() -> BTreeSet<(String, String)> {
    let ledger = fs::read_to_string(root().join("docs/provenance.md")).expect("docs/provenance.md");
    ledger
        .lines()
        .filter(|l| l.starts_with("| "))
        .flat_map(|l| l.split('`').skip(1).step_by(2))
        .filter_map(|item| item.rsplit_once("::"))
        .map(|(file, name)| (file.to_owned(), name.to_owned()))
        .collect()
}

#[test]
fn every_function_has_a_row_and_every_row_names_a_real_function() {
    let listed = listed();
    let mut found = BTreeSet::new();
    for file in FILES {
        let names = functions(file);
        assert!(
            !names.is_empty(),
            "no functions found in {file}: the scan is broken"
        );
        for name in names {
            found.insert((file.to_owned(), name));
        }
    }
    let missing: Vec<_> = found.difference(&listed).collect();
    let stale: Vec<_> = listed.difference(&found).collect();
    assert!(
        missing.is_empty(),
        "functions with no row in docs/provenance.md: {missing:?}"
    );
    assert!(
        stale.is_empty(),
        "rows naming functions that do not exist: {stale:?}"
    );
}

#[test]
fn every_algorithm_row_names_a_source_and_a_relationship() {
    let ledger = fs::read_to_string(root().join("docs/provenance.md")).unwrap();
    let section = ledger
        .split("## Algorithms")
        .nth(1)
        .unwrap()
        .split("\n## ")
        .next()
        .unwrap();
    let rows: Vec<Vec<&str>> = section
        .lines()
        .filter(|l| l.starts_with("| ") && !l.starts_with("| Algorithm"))
        .map(|l| l.trim_matches('|').split('|').map(str::trim).collect())
        .collect();
    assert!(rows.len() >= 6, "the ledger lost rows");
    for cells in rows {
        assert_eq!(cells.len(), 6, "{cells:?}");
        assert!(cells.iter().all(|c| !c.is_empty()), "{cells:?}");
        assert!(
            cells[3].starts_with("derived") || cells[3].starts_with("adapted"),
            "relationship must be `derived` or `adapted`: {cells:?}"
        );
    }
}

#[test]
fn the_scan_finds_the_functions_it_should() {
    let names = functions("crates/runtime/src/math.rs");
    assert_eq!(
        names,
        ["exp", "pow2", "sigmoid"]
            .map(String::from)
            .into_iter()
            .collect()
    );
}
