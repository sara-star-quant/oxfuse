//! Every file in `tests/invalid/` must be rejected with a stable diagnostic.
//! Review snapshot changes with `git diff`; accept intended ones by running
//! the tests with `INSTA_UPDATE=always`.

use std::fs;
use std::path::PathBuf;
use std::process::Command;

#[test]
fn invalid_graphs_produce_stable_diagnostics() {
    insta::glob!("../../../tests/invalid", "*.json", |path| {
        let output = Command::new(env!("CARGO_BIN_EXE_oxfuse"))
            .arg("check")
            .arg(path)
            .output()
            .unwrap();
        assert_eq!(output.status.code(), Some(1), "{}", path.display());
        insta::assert_snapshot!(String::from_utf8_lossy(&output.stderr).trim_end());
    });
}

fn count_files(dir: PathBuf, extension: &str) -> usize {
    fs::read_dir(dir)
        .unwrap()
        .filter(|e| {
            e.as_ref()
                .unwrap()
                .path()
                .extension()
                .is_some_and(|x| x == extension)
        })
        .count()
}

/// A deleted graph would leave its snapshot behind and the glob test above
/// would still pass; a new graph without a reviewed snapshot fails here too.
#[test]
fn every_invalid_graph_has_exactly_one_snapshot() {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let graphs = count_files(manifest.join("../../tests/invalid"), "json");
    let snapshots = fs::read_dir(manifest.join("tests/snapshots"))
        .unwrap()
        .filter(|e| {
            let name = e.as_ref().unwrap().file_name();
            let name = name.to_str().unwrap();
            name.starts_with("invalid_graphs__") && name.ends_with(".snap")
        })
        .count();
    assert_eq!(
        graphs, snapshots,
        "tests/invalid files and snapshots differ"
    );
}
