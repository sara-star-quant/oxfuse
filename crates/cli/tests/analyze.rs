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
