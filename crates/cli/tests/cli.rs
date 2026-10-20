//! Exit codes and diagnostics of the real binary.

use std::fs;
use std::path::PathBuf;
use std::process::{Command, Output};

fn oxfuse(args: &[&std::ffi::OsStr]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_oxfuse"))
        .args(args)
        .output()
        .unwrap()
}

fn example(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../examples")
        .join(name)
}

fn temp_file(name: &str, contents: &str) -> PathBuf {
    let path = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(name);
    fs::write(&path, contents).unwrap();
    path
}

fn stderr(output: &Output) -> String {
    String::from_utf8_lossy(&output.stderr).into_owned()
}

#[test]
fn check_prints_a_summary() {
    let graph = example("relu.json");
    let output = oxfuse(&["check".as_ref(), graph.as_os_str()]);
    assert!(output.status.success());
    assert_eq!(
        String::from_utf8(output.stdout).unwrap(),
        "ok: 1 input(s), 1 node(s), 1 output(s)\n"
    );
}

#[test]
fn missing_file_exits_nonzero_with_a_code() {
    let output = oxfuse(&["check".as_ref(), "/no/such/file.json".as_ref()]);
    assert_eq!(output.status.code(), Some(1));
    assert!(
        stderr(&output).starts_with("error[E101]:"),
        "{}",
        stderr(&output)
    );
}

#[test]
fn cyclic_graph_is_rejected_with_the_cycle_code() {
    let graph = temp_file(
        "cycle.json",
        r#"{"format_version":1,"nodes":[
            {"name":"a","op":"Relu","inputs":["b"]},
            {"name":"b","op":"Relu","inputs":["a"]}],"outputs":["a"]}"#,
    );
    let output = oxfuse(&["check".as_ref(), graph.as_os_str()]);
    assert_eq!(output.status.code(), Some(1));
    assert!(
        stderr(&output).starts_with("error[E005]:"),
        "{}",
        stderr(&output)
    );
}

#[test]
fn run_without_required_inputs_is_rejected() {
    let graph = example("relu.json");
    let output = oxfuse(&["run".as_ref(), graph.as_os_str()]);
    assert_eq!(output.status.code(), Some(1));
    assert!(
        stderr(&output).starts_with("error[E103]:"),
        "{}",
        stderr(&output)
    );
}

#[test]
fn input_of_wrong_shape_is_rejected() {
    let graph = example("relu.json");
    let inputs = temp_file(
        "wrong_shape.inputs.json",
        r#"{"x":{"shape":[3],"values":[1,2,3]}}"#,
    );
    let output = oxfuse(&[
        "run".as_ref(),
        graph.as_os_str(),
        "--inputs".as_ref(),
        inputs.as_os_str(),
    ]);
    assert_eq!(output.status.code(), Some(1));
    assert!(
        stderr(&output).starts_with("error[E014]:"),
        "{}",
        stderr(&output)
    );
}

/// `oxfuse run FILE | head -c 1` closes stdout early. That is normal
/// shell use, not an error, and must not panic.
#[test]
fn closing_stdout_early_does_not_panic() {
    use std::process::Stdio;
    let values = vec!["0.5"; 20_000].join(",");
    let graph = temp_file(
        "big_output.json",
        &format!(
            r#"{{"format_version":1,
            "constants":[{{"name":"c","dtype":"f32","shape":[20000],"values":[{values}]}}],
            "nodes":[{{"name":"r","op":"Relu","inputs":["c"]}}],
            "outputs":["r"]}}"#
        ),
    );
    let mut child = Command::new(env!("CARGO_BIN_EXE_oxfuse"))
        .arg("run")
        .arg(&graph)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    drop(child.stdout.take());
    let output = child.wait_with_output().unwrap();
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(!stderr.contains("panicked"), "{stderr}");
    assert_eq!(output.status.code(), Some(0), "{stderr}");
}

#[test]
fn inspect_prints_the_ir_of_every_stage() {
    let graph = example("tiny_mlp.json");
    let output = oxfuse(&["inspect".as_ref(), graph.as_os_str()]);
    assert!(output.status.success(), "{}", stderr(&output));
    let text = String::from_utf8(output.stdout).unwrap();
    assert!(text.starts_with("== input ==\ngraph {\n"), "{text}");
    assert!(text.contains("== after dce"), "{text}");
}

#[test]
fn unknown_pass_exits_nonzero_with_its_code() {
    let graph = example("relu.json");
    let output = oxfuse(&[
        "inspect".as_ref(),
        graph.as_os_str(),
        "--disable-pass".as_ref(),
        "nope".as_ref(),
    ]);
    assert_eq!(output.status.code(), Some(1));
    assert!(
        stderr(&output).starts_with("error[E106]:"),
        "{}",
        stderr(&output)
    );
}

#[test]
fn dump_ir_is_printed_even_when_the_run_fails() {
    let graph = temp_file(
        "overflow_dump.json",
        r#"{"format_version":1,
        "constants":[{"name":"a","dtype":"f32","shape":[],"values":[3e38]},
                     {"name":"b","dtype":"f32","shape":[],"values":[10]}],
        "nodes":[{"name":"boom","op":"Mul","inputs":["a","b"]}],
        "outputs":["boom"]}"#,
    );
    let output = oxfuse(&["run".as_ref(), graph.as_os_str(), "--dump-ir".as_ref()]);
    assert_eq!(output.status.code(), Some(1));
    let text = stderr(&output);
    assert!(text.starts_with("== input ==\ngraph {\n"), "{text}");
    assert!(text.contains("after dce"), "{text}");
    assert!(text.contains("error[E012]:"), "{text}");
    assert!(
        text.find("== input ==").unwrap() < text.find("error[E012]:").unwrap(),
        "the IR must come before the error line: {text}"
    );
}
