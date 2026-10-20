//! The claims ledger is only worth anything if every piece of evidence it
//! names exists. Evidence cells hold repo paths in backticks.

use std::fs;
use std::path::PathBuf;

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

#[test]
fn claims_evidence_exists() {
    let ledger = fs::read_to_string(root().join("docs/claims.md")).expect("docs/claims.md");
    let mut rows = 0;
    for line in ledger
        .lines()
        .filter(|l| l.starts_with("| ") && !l.starts_with("| Claim") && !l.starts_with("|---"))
    {
        rows += 1;
        let cells: Vec<&str> = line.trim_matches('|').split('|').map(str::trim).collect();
        assert_eq!(
            cells.len(),
            3,
            "a ledger row needs claim, evidence and how to check: {line}"
        );
        let paths: Vec<&str> = cells[1].split('`').skip(1).step_by(2).collect();
        assert!(!paths.is_empty(), "no evidence path in: {line}");
        for path in paths {
            assert!(
                root().join(path).exists(),
                "evidence `{path}` does not exist: {line}"
            );
        }
    }
    assert!(
        rows >= 8,
        "the ledger should cover the headline claims, found {rows}"
    );
}

#[test]
fn performance_numbers_live_only_in_the_benchmark_report() {
    // README may point at the report but must not repeat numbers that can drift.
    let readme = fs::read_to_string(root().join("README.md")).unwrap();
    assert!(
        readme.contains("docs/benchmarks.md"),
        "README should link the benchmark report"
    );
    for forbidden in ["GFLOP", "x faster", "times faster"] {
        assert!(
            !readme.contains(forbidden),
            "README repeats a measured number (`{forbidden}`)"
        );
    }
}
