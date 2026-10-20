//! Guards the checklist in `docs/adding-an-op.md`: every op must ship
//! with an example graph and a row in the README's supported-ops table.

use oxfuse_core::Op;
use std::collections::BTreeSet;
use std::fs;
use std::path::PathBuf;

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

#[test]
fn every_op_has_an_example() {
    let mut used = BTreeSet::new();
    for entry in fs::read_dir(root().join("examples")).unwrap() {
        let path = entry.unwrap().path();
        let name = path.file_name().unwrap().to_str().unwrap();
        if !name.ends_with(".json")
            || name.ends_with(".inputs.json")
            || name.ends_with(".expected.json")
        {
            continue;
        }
        let graph: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(&path).unwrap()).unwrap();
        for node in graph["nodes"].as_array().unwrap() {
            used.insert(node["op"].as_str().unwrap().to_owned());
        }
    }
    for op in Op::ALL_NAMES {
        assert!(
            used.contains(op),
            "no example uses {op}; add one to examples/ (see docs/adding-an-op.md)"
        );
    }
}

#[test]
fn readme_lists_every_op() {
    let readme = fs::read_to_string(root().join("README.md")).unwrap();
    for op in Op::ALL_NAMES {
        assert!(
            readme.contains(&format!("| `{op}` |")),
            "README.md supported-ops table has no row for {op}"
        );
    }
}
