//! The passes are exact, so optimizing must never change an output byte.
//! Snapshots pin what each pass does to a small graph.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn is_graph(name: &str) -> bool {
    name.ends_with(".json") && !name.ends_with(".inputs.json") && !name.ends_with(".expected.json")
}

fn stdout_of(args: &[&std::ffi::OsStr]) -> Vec<u8> {
    let output = Command::new(env!("CARGO_BIN_EXE_oxfuse"))
        .args(args)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    output.stdout
}

fn check_dir(dir: &Path) -> usize {
    let mut checked = 0;
    for entry in fs::read_dir(dir).unwrap() {
        let path = entry.unwrap().path();
        let name = path.file_name().unwrap().to_str().unwrap().to_owned();
        if !is_graph(&name) {
            continue;
        }
        let inputs = dir.join(format!("{}.inputs.json", name.trim_end_matches(".json")));
        let mut optimized: Vec<&std::ffi::OsStr> = vec!["run".as_ref(), path.as_os_str()];
        let mut plain: Vec<&std::ffi::OsStr> =
            vec!["run".as_ref(), path.as_os_str(), "--no-opt".as_ref()];
        if inputs.exists() {
            for args in [&mut optimized, &mut plain] {
                args.push("--inputs".as_ref());
                args.push(inputs.as_os_str());
            }
        }
        assert_eq!(
            stdout_of(&optimized),
            stdout_of(&plain),
            "{name}: optimizing changed the output"
        );
        checked += 1;
    }
    checked
}

#[test]
fn optimizing_never_changes_the_output() {
    let root = root();
    assert!(check_dir(&root.join("examples")) >= 8);
    assert!(check_dir(&root.join("tests/corpus")) >= 50);
    assert!(check_dir(&root.join("tests/passes")) >= 3);
}

/// A run that fails must fail the same way with and without optimization:
/// an overflowing output is an error in both modes, and an overflowing dead
/// node is no error in either.
#[test]
fn optimizing_does_not_change_errors() {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR"));
    let overflow = |outputs: &str| {
        format!(
            r#"{{"format_version":1,
            "constants":[{{"name":"a","dtype":"f32","shape":[],"values":[3e38]}},
                         {{"name":"b","dtype":"f32","shape":[],"values":[10]}},
                         {{"name":"ok","dtype":"f32","shape":[],"values":[1]}}],
            "nodes":[{{"name":"boom","op":"Mul","inputs":["a","b"]}}],
            "outputs":[{outputs}]}}"#
        )
    };
    let failing = dir.join("overflow_output.json");
    fs::write(&failing, overflow(r#""boom""#)).unwrap();
    let passing = dir.join("overflow_dead.json");
    fs::write(&passing, overflow(r#""ok""#)).unwrap();
    for (graph, should_fail) in [(&failing, true), (&passing, false)] {
        let run = |extra: &[&str]| {
            Command::new(env!("CARGO_BIN_EXE_oxfuse"))
                .arg("run")
                .arg(graph)
                .args(extra)
                .output()
                .unwrap()
        };
        let (optimized, plain) = (run(&[]), run(&["--no-opt"]));
        assert_eq!(
            optimized.status.success(),
            !should_fail,
            "{}",
            graph.display()
        );
        assert_eq!(optimized.status.code(), plain.status.code());
        assert_eq!(optimized.stdout, plain.stdout);
        assert_eq!(optimized.stderr, plain.stderr);
    }
}

#[test]
fn pass_stages_are_stable() {
    insta::glob!("../../../tests/passes", "*.json", |path| {
        if !is_graph(path.file_name().unwrap().to_str().unwrap()) {
            return;
        }
        let text = String::from_utf8(stdout_of(&["inspect".as_ref(), path.as_os_str()])).unwrap();
        insta::assert_snapshot!(text.trim_end());
    });
}

/// A removed pass graph would leave its snapshot behind and still pass.
#[test]
fn every_pass_graph_has_exactly_one_snapshot() {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let graphs = fs::read_dir(manifest.join("../../tests/passes"))
        .unwrap()
        .filter(|e| is_graph(e.as_ref().unwrap().file_name().to_str().unwrap()))
        .count();
    let snapshots = fs::read_dir(manifest.join("tests/snapshots"))
        .unwrap()
        .filter(|e| {
            let name = e.as_ref().unwrap().file_name();
            let name = name.to_str().unwrap();
            name.starts_with("optimization__") && name.ends_with(".snap")
        })
        .count();
    assert_eq!(graphs, snapshots);
}
