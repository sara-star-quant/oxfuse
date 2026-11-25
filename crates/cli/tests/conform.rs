//! `oxfuse conform`: another runtime's outputs against the enclosure of the graph.

use serde_json::Value;
use std::ffi::OsString;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("oxfuse-conform-{}-{name}", std::process::id()));
    fs::create_dir_all(&dir).unwrap();
    dir
}

fn oxfuse(args: Vec<OsString>) -> Output {
    Command::new(env!("CARGO_BIN_EXE_oxfuse"))
        .args(args)
        .output()
        .unwrap()
}

fn text(bytes: &[u8]) -> String {
    String::from_utf8_lossy(bytes).into_owned()
}

/// The outputs of `run` on one backend, written to `out`.
fn run_to(graph: &Path, inputs: Option<&Path>, backend: &str, out: &Path) {
    let mut args: Vec<OsString> = vec!["run".into(), graph.into()];
    if let Some(inputs) = inputs {
        args.extend(["--inputs".into(), inputs.into()]);
    }
    args.extend(["--backend".into(), backend.into()]);
    let output = oxfuse(args);
    assert!(
        output.status.success(),
        "run {backend}: {}",
        text(&output.stderr)
    );
    fs::write(out, output.stdout).unwrap();
}

fn conform(outputs: &Path, graph: &Path, inputs: Option<&Path>) -> Output {
    let mut args: Vec<OsString> = vec![
        "conform".into(),
        outputs.into(),
        "--graph".into(),
        graph.into(),
    ];
    if let Some(inputs) = inputs {
        args.extend(["--inputs".into(), inputs.into()]);
    }
    oxfuse(args)
}

fn tiny() -> (PathBuf, PathBuf) {
    let r = root();
    (
        r.join("examples/tiny_mlp.json"),
        r.join("examples/tiny_mlp.inputs.json"),
    )
}

/// Every graph of a directory with its inputs file, if it has one.
fn graphs_in(dir: &Path) -> Vec<(PathBuf, Option<PathBuf>)> {
    let mut found = Vec::new();
    for entry in fs::read_dir(dir).unwrap() {
        let path = entry.unwrap().path();
        let name = path.file_name().unwrap().to_str().unwrap().to_owned();
        if !name.ends_with(".json")
            || name.ends_with(".inputs.json")
            || name.ends_with(".expected.json")
        {
            continue;
        }
        let inputs = dir.join(format!("{}.inputs.json", name.trim_end_matches(".json")));
        found.push((path, inputs.exists().then_some(inputs)));
    }
    found.sort();
    found
}

#[test]
fn the_outputs_of_both_cpu_backends_are_inside_for_every_shipped_graph() {
    let dir = scratch("shipped");
    let r = root();
    let mut checked = 0;
    for folder in ["examples", "tests/corpus"] {
        for (graph, inputs) in graphs_in(&r.join(folder)) {
            for backend in ["reference", "cpu"] {
                let out = dir.join("out.json");
                run_to(&graph, inputs.as_deref(), backend, &out);
                let verdict = conform(&out, &graph, inputs.as_deref());
                assert!(
                    verdict.status.success(),
                    "{} on {backend}: {}{}",
                    graph.display(),
                    text(&verdict.stdout),
                    text(&verdict.stderr)
                );
                assert!(text(&verdict.stdout).ends_with("verdict: inside\n"));
                checked += 1;
            }
        }
    }
    assert!(checked >= 100, "only {checked} runs were checked");
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn a_box_around_the_inputs_contains_the_outputs_of_the_point_inside_it() {
    let dir = scratch("box");
    let (graph, point) = tiny();
    let out = dir.join("out.json");
    run_to(&graph, Some(&point), "reference", &out);
    let boxed = dir.join("box.json");
    fs::write(
        &boxed,
        r#"{"x":{"shape":[2,3],"lo":[0.5,-2.5,2.5,-0.5,3.5,-1.5],"hi":[1.5,-1.5,3.5,0.5,4.5,-0.5]}}"#,
    )
    .unwrap();
    let verdict = conform(&out, &graph, Some(&boxed));
    assert!(verdict.status.success(), "{}", text(&verdict.stderr));
    let _ = fs::remove_dir_all(&dir);
}

/// The enclosure of every output element for a point input, read from `analyze` on the box whose
/// ends are the point (a point is a box with equal ends).
fn enclosure_of_the_point(dir: &Path) -> (Vec<f64>, Vec<f64>) {
    let (graph, point) = tiny();
    let inputs: Value = serde_json::from_str(&fs::read_to_string(point).unwrap()).unwrap();
    let x = &inputs["x"];
    let boxed = dir.join("point-as-box.json");
    fs::write(
        &boxed,
        serde_json::json!({"x": {"shape": x["shape"], "lo": x["values"], "hi": x["values"]}})
            .to_string(),
    )
    .unwrap();
    let output = oxfuse(vec![
        "analyze".into(),
        graph.into(),
        "--inputs".into(),
        boxed.into(),
    ]);
    let got: Value = serde_json::from_slice(&output.stdout).unwrap();
    let ends = |v: &Value| {
        v.as_array()
            .unwrap()
            .iter()
            .map(|e| e.as_f64().unwrap())
            .collect()
    };
    (
        ends(&got["outputs"][0]["lo"]),
        ends(&got["outputs"][0]["hi"]),
    )
}

/// The largest float32 that is not above `x`.
fn float32_at_or_below(x: f64) -> f32 {
    let f = x as f32;
    if f64::from(f) <= x {
        f
    } else if f > 0.0 {
        f32::from_bits(f.to_bits() - 1)
    } else {
        f32::from_bits(f.to_bits() + 1)
    }
}

#[test]
fn one_float32_step_past_the_enclosure_is_outside_and_the_last_value_in_it_is_inside() {
    let dir = scratch("boundary");
    let (graph, point) = tiny();
    let (lo, hi) = enclosure_of_the_point(&dir);
    // Element 1 of the output is 1.0: its upper end is just above 1.
    let inside = float32_at_or_below(hi[1]);
    let outside = f32::from_bits(inside.to_bits() + 1);
    assert!(f64::from(inside) <= hi[1] && f64::from(outside) > hi[1] && f64::from(inside) >= lo[1]);
    let outputs = |v: f32| {
        format!(r#"{{"outputs":[{{"name":"out","shape":[2,2],"values":[0.0,{v:?},1.0,0.0]}}]}}"#)
    };
    let file = dir.join("out.json");
    fs::write(&file, outputs(inside)).unwrap();
    let ok = conform(&file, &graph, Some(&point));
    assert!(
        ok.status.success(),
        "{}{}",
        text(&ok.stdout),
        text(&ok.stderr)
    );
    fs::write(&file, outputs(outside)).unwrap();
    let bad = conform(&file, &graph, Some(&point));
    assert_eq!(bad.status.code(), Some(1));
    let report = text(&bad.stdout);
    assert!(
        report.contains("out: 3 of 4 inside, 1 outside, 0 unresolved; first outside at index 1:"),
        "{report}"
    );
    assert!(report.ends_with("verdict: outside\n"));
    assert!(text(&bad.stderr).contains("error[E114]"));
    let _ = fs::remove_dir_all(&dir);
}

fn fails_with(outputs: &str, needles: &[&str]) {
    static CALLS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let call = CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let dir = scratch(&format!("errors-{call}"));
    let (graph, point) = tiny();
    let file = dir.join("out.json");
    fs::write(&file, outputs).unwrap();
    let output = conform(&file, &graph, Some(&point));
    let stderr = text(&output.stderr);
    assert_eq!(output.status.code(), Some(1), "{stderr}");
    assert!(
        !text(&output.stdout).contains("verdict: inside"),
        "a mismatch is never a pass"
    );
    for needle in needles {
        assert!(stderr.contains(needle), "stderr lacks `{needle}`: {stderr}");
    }
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn a_wrong_shape_a_wrong_missing_or_extra_output_and_a_bad_file_are_named_errors() {
    let good = r#"{"name":"out","shape":[2,2],"values":[0.0,1.0,1.0,0.0]}"#;
    fails_with(
        r#"{"outputs":[{"name":"out","shape":[4],"values":[0.0,1.0,1.0,0.0]}]}"#,
        &["error[E115]", "`out`", "[4]", "[2, 2]"],
    );
    fails_with(
        r#"{"outputs":[{"name":"result","shape":[2,2],"values":[0.0,1.0,1.0,0.0]}]}"#,
        &["error[E115]", "`result`", "not an output of the graph"],
    );
    fails_with(r#"{"outputs":[]}"#, &["error[E115]", "`out`", "missing"]);
    fails_with(
        &format!(r#"{{"outputs":[{good},{{"name":"extra","shape":[1],"values":[1.0]}}]}}"#),
        &["error[E115]", "`extra`"],
    );
    fails_with(
        &format!(r#"{{"outputs":[{good},{good}]}}"#),
        &["error[E111]", "`out`", "twice"],
    );
    fails_with(
        r#"{"outputs":[{"name":"out","shape":[2,2],"values":[0.0,1.0]}]}"#,
        &["error[E113]", "`out`"],
    );
    fails_with("not json", &["error[E113]"]);
    // NaN and null are not numbers: the file is rejected, never judged.
    fails_with(
        r#"{"outputs":[{"name":"out","shape":[2,2],"values":[0.0,NaN,1.0,0.0]}]}"#,
        &["error[E113]"],
    );
    fails_with(
        r#"{"outputs":[{"name":"out","shape":[2,2],"values":[0.0,null,1.0,0.0]}]}"#,
        &["error[E113]"],
    );
}

#[test]
fn an_unbounded_enclosure_is_unresolved_never_inside() {
    let dir = scratch("unbounded");
    let graph = dir.join("g.json");
    fs::write(
        &graph,
        r#"{"format_version":1,"inputs":[{"name":"x","dtype":"f32","shape":[2]},{"name":"y","dtype":"f32","shape":[2]}],"nodes":[{"name":"z","op":"Div","inputs":["x","y"]}],"outputs":["z"]}"#,
    )
    .unwrap();
    let inputs = dir.join("i.json");
    fs::write(
        &inputs,
        r#"{"x":{"shape":[2],"values":[1.0,1.0]},"y":{"shape":[2],"lo":[-1.0,2.0],"hi":[1.0,4.0]}}"#,
    )
    .unwrap();
    let out = dir.join("out.json");
    fs::write(
        &out,
        r#"{"outputs":[{"name":"z","shape":[2],"values":[3.0,0.4]}]}"#,
    )
    .unwrap();
    let verdict = conform(&out, &graph, Some(&inputs));
    assert_eq!(verdict.status.code(), Some(1));
    let report = text(&verdict.stdout);
    assert!(
        report.contains("z: 1 of 2 inside, 0 outside, 1 unresolved"),
        "{report}"
    );
    assert!(report.ends_with("verdict: unresolved\n"));
    // A value past a finite end is outside even next to an unresolved element.
    fs::write(
        &out,
        r#"{"outputs":[{"name":"z","shape":[2],"values":[3.0,9.0]}]}"#,
    )
    .unwrap();
    let verdict = conform(&out, &graph, Some(&inputs));
    assert!(text(&verdict.stdout).ends_with("verdict: outside\n"));
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn the_documented_examples_are_what_the_command_prints() {
    let dir = scratch("docs");
    let (graph, point) = tiny();
    let docs = fs::read_to_string(root().join("docs/conform.md")).unwrap();
    let out = dir.join("out.json");
    run_to(&graph, Some(&point), "reference", &out);
    let inside = text(&conform(&out, &graph, Some(&point)).stdout);
    assert!(
        docs.contains(inside.trim_end()),
        "docs/conform.md must show the inside report"
    );
    let nudged = r#"{"outputs":[{"name":"out","shape":[2,2],"values":[0.0,1.0,1.5,0.0]}]}"#;
    assert!(
        docs.contains(nudged),
        "docs/conform.md must show the nudged outputs"
    );
    fs::write(&out, nudged).unwrap();
    let outside = text(&conform(&out, &graph, Some(&point)).stdout);
    assert!(
        docs.contains(outside.trim_end()),
        "docs/conform.md must show the outside report"
    );
    let _ = fs::remove_dir_all(&dir);
}
