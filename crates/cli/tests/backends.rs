//! The two backends must print the same bytes for every graph we ship,
//! including the corpus graphs whose matmul sums round.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn is_graph(name: &str) -> bool {
    name.ends_with(".json") && !name.ends_with(".inputs.json") && !name.ends_with(".expected.json")
}

fn run(graph: &Path, inputs: &Path, backend: &str) -> Vec<u8> {
    let mut command = Command::new(env!("CARGO_BIN_EXE_oxfuse"));
    command.arg("run").arg(graph).args(["--backend", backend]);
    if inputs.exists() {
        command.arg("--inputs").arg(inputs);
    }
    let output = command.output().unwrap();
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
        assert_eq!(
            run(&path, &inputs, "cpu"),
            run(&path, &inputs, "reference"),
            "{name}: the backends printed different bytes"
        );
        checked += 1;
    }
    checked
}

#[test]
fn the_backends_print_the_same_bytes_for_every_shipped_graph() {
    let root = root();
    assert!(check_dir(&root.join("examples")) >= 8);
    assert!(check_dir(&root.join("tests/corpus")) >= 50);
    assert!(check_dir(&root.join("tests/passes")) >= 3);
}

#[test]
fn an_unknown_backend_is_a_usage_error() {
    let graph = root().join("examples/relu.json");
    let output = Command::new(env!("CARGO_BIN_EXE_oxfuse"))
        .arg("run")
        .arg(graph)
        .args(["--backend", "gpu"])
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&output.stderr).contains("reference"));
}

#[test]
fn the_fast_backend_is_the_default() {
    assert_eq!(
        oxfuse_cli::BackendKind::default(),
        oxfuse_cli::BackendKind::Cpu
    );
    let help = Command::new(env!("CARGO_BIN_EXE_oxfuse"))
        .args(["run", "--help"])
        .output()
        .unwrap();
    let help = String::from_utf8_lossy(&help.stdout);
    assert!(help.contains("[default: cpu]"), "{help}");
}

#[test]
fn running_without_the_flag_prints_what_the_reference_prints() {
    let root = root();
    for dir in ["examples", "tests/passes"] {
        let dir = root.join(dir);
        for entry in fs::read_dir(&dir).unwrap() {
            let name = entry.unwrap().file_name().into_string().unwrap();
            if !is_graph(&name) {
                continue;
            }
            let graph = dir.join(&name);
            let inputs = dir.join(format!("{}.inputs.json", name.trim_end_matches(".json")));
            let mut command = Command::new(env!("CARGO_BIN_EXE_oxfuse"));
            command.arg("run").arg(&graph);
            if inputs.exists() {
                command.arg("--inputs").arg(&inputs);
            }
            let default = command.output().unwrap();
            assert!(default.status.success());
            assert_eq!(default.stdout, run(&graph, &inputs, "reference"), "{name}");
        }
    }
}
