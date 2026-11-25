//! `oxfuse analyze`: the exact result, as the NumPy float64 oracle computes it,
//! must lie inside the certified interval of every graph we ship.

use serde_json::Value;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn analyze(graph: &Path, inputs: &Path) -> Command {
    let mut command = Command::new(env!("CARGO_BIN_EXE_oxfuse"));
    command.arg("analyze").arg(graph);
    if inputs.exists() {
        command.arg("--inputs").arg(inputs);
    }
    command
}

fn floats(v: &Value) -> Vec<f64> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap())
        .collect()
}

/// (graphs checked, widest bound over actual error seen).
fn check_dir(dir: &Path) -> (usize, f64) {
    let (mut checked, mut widest) = (0, 0.0f64);
    for entry in fs::read_dir(dir).unwrap() {
        let path = entry.unwrap().path();
        let name = path.file_name().unwrap().to_str().unwrap().to_owned();
        if !name.ends_with(".json")
            || name.ends_with(".inputs.json")
            || name.ends_with(".expected.json")
        {
            continue;
        }
        let stem = name.trim_end_matches(".json");
        let expected = dir.join(format!("{stem}.expected.json"));
        if !expected.exists() {
            continue;
        }
        let output = analyze(&path, &dir.join(format!("{stem}.inputs.json")))
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{name}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        let got: Value = serde_json::from_slice(&output.stdout).unwrap();
        let want: Value = serde_json::from_str(&fs::read_to_string(&expected).unwrap()).unwrap();
        for (g, w) in got["outputs"]
            .as_array()
            .unwrap()
            .iter()
            .zip(want["outputs"].as_array().unwrap())
        {
            // The values print as the shortest decimal of an f32; read them back as f32.
            let (values, bounds, oracle) = (
                floats(&g["values"])
                    .into_iter()
                    .map(|v| f64::from(v as f32))
                    .collect::<Vec<_>>(),
                g["bounds"].as_array().unwrap(),
                floats(&w["values"]),
            );
            assert_eq!(values.len(), oracle.len(), "{name}");
            for (i, ((v, b), x)) in values.iter().zip(bounds).zip(&oracle).enumerate() {
                let b = b
                    .as_f64()
                    .unwrap_or_else(|| panic!("{name}[{i}]: no certificate"));
                let off = (x - v).abs();
                assert!(
                    off <= b * (1.0 + 1e-12) + 1e-300,
                    "{name}[{i}]: oracle {x} vs value {v}: off by {off}, bound {b}"
                );
                if off > 0.0 {
                    widest = widest.max(b / off);
                }
            }
        }
        checked += 1;
    }
    (checked, widest)
}

#[test]
fn the_oracle_result_is_inside_the_interval_for_every_shipped_graph() {
    let root = root();
    let (examples, _) = check_dir(&root.join("examples"));
    let (corpus, widest) = check_dir(&root.join("tests/corpus"));
    assert!(examples >= 8, "{examples}");
    assert!(corpus >= 50, "{corpus}");
    assert!(
        widest > 1.0,
        "a bound is never below the real error: {widest}"
    );
}

#[test]
fn the_output_names_the_values_and_one_bound_per_value() {
    let root = root();
    let output = analyze(
        &root.join("examples/tiny_mlp.json"),
        &root.join("examples/tiny_mlp.inputs.json"),
    )
    .output()
    .unwrap();
    assert!(output.status.success());
    let got: Value = serde_json::from_slice(&output.stdout).unwrap();
    let out = &got["outputs"][0];
    assert_eq!(out["name"], "out");
    assert_eq!(out["shape"], serde_json::json!([2, 2]));
    assert_eq!(out["values"].as_array().unwrap().len(), 4);
    assert_eq!(out["bounds"].as_array().unwrap().len(), 4);
    assert!(
        out["bounds"]
            .as_array()
            .unwrap()
            .iter()
            .all(|b| b.as_f64().unwrap() > 0.0)
    );
}

#[test]
fn the_values_equal_what_run_prints() {
    let root = root();
    let (graph, inputs) = (
        root.join("examples/tiny_mlp.json"),
        root.join("examples/tiny_mlp.inputs.json"),
    );
    let analyzed: Value =
        serde_json::from_slice(&analyze(&graph, &inputs).output().unwrap().stdout).unwrap();
    let ran = Command::new(env!("CARGO_BIN_EXE_oxfuse"))
        .arg("run")
        .arg(&graph)
        .arg("--inputs")
        .arg(&inputs)
        .args(["--backend", "reference"])
        .output()
        .unwrap();
    let ran: Value = serde_json::from_slice(&ran.stdout).unwrap();
    assert_eq!(
        analyzed["outputs"][0]["values"],
        ran["outputs"][0]["values"]
    );
}

#[test]
fn an_element_that_passed_through_infinity_has_a_null_bound() {
    let dir = std::env::temp_dir().join(format!("oxfuse-analyze-{}", std::process::id()));
    fs::create_dir_all(&dir).unwrap();
    let graph = dir.join("g.json");
    fs::write(
        &graph,
        r#"{"format_version":1,"inputs":[{"name":"x","dtype":"f32","shape":[2]}],"constants":[{"name":"c","dtype":"f32","shape":[],"values":[3.0e38]}],"nodes":[{"name":"m","op":"Mul","inputs":["x","c"]},{"name":"y","op":"Relu","inputs":["m"]}],"outputs":["y"]}"#,
    )
    .unwrap();
    let inputs = dir.join("i.json");
    fs::write(&inputs, r#"{"x":{"shape":[2],"values":[-10.0,0.5]}}"#).unwrap();
    // -inf goes through Relu to 0: the value is finite but nothing certifies it.
    let output = analyze(&graph, &inputs).output().unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let got: Value = serde_json::from_slice(&output.stdout).unwrap();
    let out = &got["outputs"][0];
    assert_eq!(out["values"][0].as_f64(), Some(0.0));
    assert!(out["bounds"][0].is_null());
    assert!(out["bounds"][1].as_f64().unwrap() > 0.0);
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn a_non_finite_output_value_is_an_error_like_in_run() {
    let dir = std::env::temp_dir().join(format!("oxfuse-analyze-nf-{}", std::process::id()));
    fs::create_dir_all(&dir).unwrap();
    let graph = dir.join("g.json");
    fs::write(
        &graph,
        r#"{"format_version":1,"inputs":[{"name":"x","dtype":"f32","shape":[1]}],"constants":[{"name":"c","dtype":"f32","shape":[],"values":[3.0e38]}],"nodes":[{"name":"y","op":"Mul","inputs":["x","c"]}],"outputs":["y"]}"#,
    )
    .unwrap();
    let inputs = dir.join("i.json");
    fs::write(&inputs, r#"{"x":{"shape":[1],"values":[10.0]}}"#).unwrap();
    let output = analyze(&graph, &inputs).output().unwrap();
    assert_eq!(output.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&output.stderr).contains("error[E"));
    let _ = fs::remove_dir_all(&dir);
}

// ---- input boxes ----

const BOX_INPUTS: &str =
    r#"{"x":{"shape":[2,3],"lo":[0.5,-2.5,2.5,-0.5,3.5,-1.5],"hi":[1.5,-1.5,3.5,0.5,4.5,-0.5]}}"#;

fn scratch(name: &str) -> PathBuf {
    let dir =
        std::env::temp_dir().join(format!("oxfuse-analyze-box-{}-{name}", std::process::id()));
    fs::create_dir_all(&dir).unwrap();
    dir
}

fn write_inputs(dir: &Path, text: &str) -> PathBuf {
    let path = dir.join("inputs.json");
    fs::write(&path, text).unwrap();
    path
}

fn tiny_graph() -> PathBuf {
    root().join("examples/tiny_mlp.json")
}

fn stdout_json(output: &std::process::Output) -> Value {
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    serde_json::from_slice(&output.stdout).unwrap()
}

fn ends(v: &Value) -> Vec<Option<f64>> {
    v.as_array().unwrap().iter().map(Value::as_f64).collect()
}

#[test]
fn a_box_input_gives_lo_and_hi_and_every_corner_of_the_box_is_inside() {
    let dir = scratch("corners");
    let inputs = write_inputs(&dir, BOX_INPUTS);
    let got = stdout_json(&analyze(&tiny_graph(), &inputs).output().unwrap());
    let out = &got["outputs"][0];
    assert_eq!(out["name"], "out");
    assert_eq!(out["shape"], serde_json::json!([2, 2]));
    assert!(out.get("values").is_none() && out.get("bounds").is_none());
    let (lo, hi) = (ends(&out["lo"]), ends(&out["hi"]));
    assert_eq!((lo.len(), hi.len()), (4, 4));
    let (xl, xh) = (
        [0.5, -2.5, 2.5, -0.5, 3.5, -1.5],
        [1.5, -1.5, 3.5, 0.5, 4.5, -0.5],
    );
    for corner in 0..64u32 {
        let x: Vec<f64> = (0..6)
            .map(|i| if corner >> i & 1 == 0 { xl[i] } else { xh[i] })
            .collect();
        let point = dir.join("corner.json");
        fs::write(
            &point,
            format!(r#"{{"x":{{"shape":[2,3],"values":{x:?}}}}}"#),
        )
        .unwrap();
        let ran = Command::new(env!("CARGO_BIN_EXE_oxfuse"))
            .arg("run")
            .arg(tiny_graph())
            .arg("--inputs")
            .arg(&point)
            .args(["--backend", "reference"])
            .output()
            .unwrap();
        let ran: Value = serde_json::from_slice(&ran.stdout).unwrap();
        for (i, v) in ran["outputs"][0]["values"]
            .as_array()
            .unwrap()
            .iter()
            .enumerate()
        {
            let v = v.as_f64().unwrap();
            assert!(
                lo[i].unwrap() <= v && v <= hi[i].unwrap(),
                "corner {corner} element {i}: {v} outside [{:?}, {:?}]",
                lo[i],
                hi[i]
            );
        }
    }
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn a_file_of_points_prints_exactly_what_it_printed_before_and_the_docs_show_it() {
    let root = root();
    let output = analyze(&tiny_graph(), &root.join("examples/tiny_mlp.inputs.json"))
        .output()
        .unwrap();
    let text = String::from_utf8(output.stdout).unwrap();
    let docs = fs::read_to_string(root.join("docs/analysis.md")).unwrap();
    assert!(text.starts_with(
        r#"{"outputs":[{"name":"out","shape":[2,2],"values":[0.0,1.0,1.0,0.0],"bounds":["#
    ));
    assert!(
        docs.contains(text.trim_end()),
        "docs/analysis.md must show the point output"
    );
}

#[test]
fn the_documented_box_example_is_what_the_command_prints() {
    let dir = scratch("docs");
    let inputs = write_inputs(&dir, BOX_INPUTS);
    let output = analyze(&tiny_graph(), &inputs).output().unwrap();
    let text = String::from_utf8(output.stdout).unwrap();
    let docs = fs::read_to_string(root().join("docs/analysis.md")).unwrap();
    assert!(
        docs.contains(BOX_INPUTS),
        "docs/analysis.md must show the box inputs"
    );
    assert!(
        docs.contains(text.trim_end()),
        "docs/analysis.md must show the box output"
    );
    let _ = fs::remove_dir_all(&dir);
}

const TWO_INPUT_GRAPH: &str = r#"{"format_version":1,"inputs":[{"name":"x","dtype":"f32","shape":[2]},{"name":"y","dtype":"f32","shape":[2]}],"nodes":[{"name":"z","op":"Sub","inputs":["x","y"]}],"outputs":["z"]}"#;

#[test]
fn a_point_and_a_box_can_be_mixed_and_each_is_read_from_its_own_input() {
    let dir = scratch("mixed");
    let graph = dir.join("g.json");
    fs::write(&graph, TWO_INPUT_GRAPH).unwrap();
    let inputs = write_inputs(
        &dir,
        r#"{"x":{"shape":[2],"values":[10.0,20.0]},"y":{"shape":[2],"lo":[1.0,2.0],"hi":[3.0,6.0]}}"#,
    );
    let got = stdout_json(&analyze(&graph, &inputs).output().unwrap());
    let (lo, hi) = (
        ends(&got["outputs"][0]["lo"]),
        ends(&got["outputs"][0]["hi"]),
    );
    // x - y with x a point: [10 - 3, 10 - 1] and [20 - 6, 20 - 2].
    for (i, (l, h)) in [(7.0, 9.0), (14.0, 18.0)].into_iter().enumerate() {
        assert!(
            lo[i].unwrap() <= l && l - lo[i].unwrap() < 1e-5,
            "lo[{i}] {:?}",
            lo[i]
        );
        assert!(
            hi[i].unwrap() >= h && hi[i].unwrap() - h < 1e-5,
            "hi[{i}] {:?}",
            hi[i]
        );
    }
    let _ = fs::remove_dir_all(&dir);
}

fn fails_with(graph: &Path, inputs: &str, needles: &[&str]) {
    // One directory per call: the tests run in parallel and must not share `inputs.json`.
    static CALLS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let call = CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let dir = scratch(&format!("errors-{call}"));
    let path = write_inputs(&dir, inputs);
    let output = analyze(graph, &path).output().unwrap();
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert_eq!(output.status.code(), Some(1), "{stderr}");
    for needle in needles {
        assert!(stderr.contains(needle), "stderr lacks `{needle}`: {stderr}");
    }
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn a_malformed_box_is_an_error_that_names_what_is_wrong() {
    let g = tiny_graph();
    let lo = "[0,0,0,0,0,0]";
    // Both forms at once, one end only, neither.
    fails_with(
        &g,
        &format!(r#"{{"x":{{"shape":[2,3],"values":{lo},"lo":{lo},"hi":{lo}}}}}"#),
        &["not both"],
    );
    fails_with(
        &g,
        &format!(r#"{{"x":{{"shape":[2,3],"lo":{lo}}}}}"#),
        &["needs `hi`"],
    );
    fails_with(
        &g,
        &format!(r#"{{"x":{{"shape":[2,3],"hi":{lo}}}}}"#),
        &["needs `lo`"],
    );
    fails_with(&g, r#"{"x":{"shape":[2,3]}}"#, &["needs `values`"]);
    // An inverted box names the input and the element; so does a NaN-free length mismatch.
    fails_with(
        &g,
        r#"{"x":{"shape":[2,3],"lo":[0,0,0,0,2,0],"hi":[1,1,1,1,1,1]}}"#,
        &["error[E022]", "`x`", "element 4"],
    );
    fails_with(
        &g,
        r#"{"x":{"shape":[2,3],"lo":[0,0,0],"hi":[1,1,1,1,1,1]}}"#,
        &["`x`"],
    );
    fails_with(
        &g,
        r#"{"x":{"shape":[2,3],"lo":[0,0,0,0,0,0],"hi":[1,1,1,1,1,1]},"extra":{"shape":[1],"values":[1]}}"#,
        &["extra"],
    );
}

#[test]
fn run_does_not_take_a_box() {
    let dir = scratch("run");
    let inputs = write_inputs(&dir, BOX_INPUTS);
    let output = Command::new(env!("CARGO_BIN_EXE_oxfuse"))
        .arg("run")
        .arg(tiny_graph())
        .arg("--inputs")
        .arg(&inputs)
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&output.stderr).contains("unknown field"));
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn a_divisor_box_that_reaches_zero_gives_null_ends_not_an_error() {
    let dir = scratch("div");
    let graph = dir.join("g.json");
    fs::write(
        &graph,
        r#"{"format_version":1,"inputs":[{"name":"x","dtype":"f32","shape":[2]},{"name":"y","dtype":"f32","shape":[2]}],"nodes":[{"name":"z","op":"Div","inputs":["x","y"]}],"outputs":["z"]}"#,
    )
    .unwrap();
    let inputs = write_inputs(
        &dir,
        r#"{"x":{"shape":[2],"values":[1.0,1.0]},"y":{"shape":[2],"lo":[-1.0,2.0],"hi":[1.0,4.0]}}"#,
    );
    let got = stdout_json(&analyze(&graph, &inputs).output().unwrap());
    let out = &got["outputs"][0];
    assert!(out["lo"][0].is_null() && out["hi"][0].is_null(), "{out}");
    assert!(out["lo"][1].as_f64().unwrap() <= 0.25 && out["hi"][1].as_f64().unwrap() >= 0.5);
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn an_inputs_file_that_is_not_an_object_says_what_a_box_looks_like() {
    fails_with(
        &tiny_graph(),
        "[1, 2]",
        &[
            "expected an object mapping input names to",
            "\"lo\"",
            "\"hi\"",
        ],
    );
}
