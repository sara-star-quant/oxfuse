//! The docs must keep up with the code: every op is described, and every doc
//! is reachable from the README.

use oxfuse_core::Op;
use std::fs;
use std::path::PathBuf;

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn read(path: &str) -> String {
    fs::read_to_string(root().join(path)).unwrap_or_else(|e| panic!("cannot read {path}: {e}"))
}

#[test]
fn every_op_is_documented() {
    for op in Op::ALL_NAMES {
        for doc in ["docs/ir.md", "docs/semantics.md"] {
            assert!(
                read(doc).contains(&format!("`{op}`")),
                "{doc} does not describe `{op}`"
            );
        }
    }
}

#[test]
fn readme_links_every_doc() {
    let readme = read("README.md");
    let mut missing = Vec::new();
    for entry in fs::read_dir(root().join("docs")).unwrap() {
        let path = entry.unwrap().path();
        if path.extension().is_some_and(|e| e == "md") {
            let name = path.file_name().unwrap().to_string_lossy().into_owned();
            if !readme.contains(&format!("docs/{name}")) {
                missing.push(name);
            }
        }
    }
    assert!(missing.is_empty(), "README does not link: {missing:?}");
}

fn records() -> Vec<(String, String)> {
    let mut found: Vec<(String, String)> = fs::read_dir(root().join("docs/decisions"))
        .unwrap()
        .map(|entry| {
            let path = entry.unwrap().path();
            let name = path.file_name().unwrap().to_string_lossy().into_owned();
            let text = fs::read_to_string(&path).unwrap();
            (name, text)
        })
        .collect();
    found.sort();
    found
}

#[test]
fn adrs_are_numbered_dated_and_have_a_status() {
    for (index, (name, text)) in records().iter().enumerate() {
        let number = format!("{:04}", index + 1);
        assert!(
            name.starts_with(&format!("{number}-")) && name.ends_with(".md"),
            "{name} should be numbered {number} (records are consecutive)"
        );
        assert!(
            text.starts_with(&format!("# {number}: ")),
            "{name} must start with `# {number}: title`"
        );
        let meta = text.lines().nth(2).unwrap_or_default();
        let (date, status) = meta.split_once(". Status: ").unwrap_or_else(|| {
            panic!("{name}: line 3 must read `Date: YYYY-MM-DD. Status: <status>.`")
        });
        let date = date.strip_prefix("Date: ").unwrap_or("");
        let digits = date.bytes().filter(u8::is_ascii_digit).count();
        assert!(
            date.len() == 10
                && digits == 8
                && date.as_bytes()[4] == b'-'
                && date.as_bytes()[7] == b'-',
            "{name}: bad date `{date}`"
        );
        let status = status.trim_end_matches('.');
        assert!(
            ["accepted", "superseded", "rejected"]
                .iter()
                .any(|s| status.starts_with(s)),
            "{name}: bad status `{status}`"
        );
    }
}

#[test]
fn there_is_one_record_per_spec_decision() {
    let records = records();
    assert!(
        records.len() >= 10,
        "expected records 0001 to 0011, found {}",
        records.len()
    );
    let licence = &records[9];
    assert!(
        licence.0.contains("licence"),
        "0010 should be the licence record: {}",
        licence.0
    );
    assert!(
        licence
            .1
            .lines()
            .nth(2)
            .unwrap()
            .contains("Status: superseded"),
        "0010 records a reversed decision and must say so"
    );
}

#[test]
fn readme_points_at_the_claims_ledger() {
    assert!(read("README.md").contains("docs/claims.md"));
}
