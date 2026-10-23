//! `docs/contract.json` states, per operator and backend, how the result
//! relates to the reference. It cannot rot: every operator needs a row for
//! every backend, every level is defined, and every evidence path and named
//! test exists.

use oxfuse_core::Op;
use serde_json::Value;
use std::collections::BTreeSet;
use std::fs;
use std::path::PathBuf;

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn matrix() -> Value {
    let text = fs::read_to_string(root().join("docs/contract.json")).expect("docs/contract.json");
    serde_json::from_str(&text).expect("valid JSON")
}

fn string<'a>(v: &'a Value, key: &str) -> &'a str {
    v[key].as_str().unwrap_or_else(|| panic!("`{key}` in {v}"))
}

#[test]
fn every_op_has_exactly_one_row_per_backend() {
    let m = matrix();
    let backends: Vec<&str> = m["backends"]
        .as_array()
        .unwrap()
        .iter()
        .map(|b| b.as_str().unwrap())
        .collect();
    let mut seen = BTreeSet::new();
    for row in m["rows"].as_array().unwrap() {
        let key = (
            string(row, "op").to_owned(),
            string(row, "backend").to_owned(),
        );
        assert!(seen.insert(key.clone()), "duplicate row {key:?}");
    }
    let mut want = BTreeSet::new();
    for op in Op::ALL_NAMES {
        for backend in &backends {
            want.insert((op.to_string(), backend.to_string()));
        }
    }
    assert_eq!(seen, want, "rows must be Op::ALL_NAMES x backends");
}

#[test]
fn every_level_is_defined_and_every_definition_is_used() {
    let m = matrix();
    let defined: BTreeSet<&str> = m["levels"]
        .as_object()
        .unwrap()
        .keys()
        .map(String::as_str)
        .collect();
    let used: BTreeSet<&str> = m["rows"]
        .as_array()
        .unwrap()
        .iter()
        .map(|r| string(r, "level"))
        .collect();
    assert_eq!(defined, used);
}

#[test]
fn every_evidence_path_and_named_test_exists() {
    let m = matrix();
    for row in m["rows"].as_array().unwrap() {
        let name = format!("{} on {}", string(row, "op"), string(row, "backend"));
        let evidence = row["evidence"].as_array().unwrap();
        assert!(!evidence.is_empty(), "{name}: no evidence");
        for path in evidence {
            let path = path.as_str().unwrap();
            assert!(
                root().join(path).exists(),
                "{name}: `{path}` does not exist"
            );
        }
        let tests = row["tests"].as_array().unwrap();
        assert!(!tests.is_empty(), "{name}: no test named");
        for t in tests {
            let (file, test) = (string(t, "file"), string(t, "test"));
            let source = fs::read_to_string(root().join(file))
                .unwrap_or_else(|_| panic!("{name}: `{file}` does not exist"));
            // A test is a `fn`, or a `check!(name)` line of the contract macro.
            let found = source.contains(&format!("fn {test}("))
                || source.contains(&format!("check!({test})"));
            assert!(found, "{name}: `{test}` is not in `{file}`");
        }
    }
}

#[test]
fn the_claims_ledger_points_at_the_matrix() {
    let ledger = fs::read_to_string(root().join("docs/claims.md")).unwrap();
    assert!(
        ledger.contains("docs/contract.json"),
        "docs/claims.md must cite docs/contract.json"
    );
}
