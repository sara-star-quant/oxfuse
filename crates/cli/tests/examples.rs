//! Runs every graph in a directory through the real binary and compares to
//! its expected file. The expected file carries its own tolerance: zero means
//! exact equality, which is what the exact ops require. Signed zeros compare
//! equal here because NumPy does not define the sign of a zero matmul sum;
//! the end-to-end test at the bottom pins the sign for Relu.

use oxfuse_cli::OutputsFile;
use serde::Deserialize;
use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

#[derive(Deserialize)]
struct Expected {
    tolerance: Tolerance,
    outputs: Vec<ExpectedTensor>,
}

#[derive(Deserialize)]
struct Tolerance {
    abs: f64,
    rel: f64,
}

#[derive(Deserialize)]
struct ExpectedTensor {
    name: String,
    shape: Vec<usize>,
    values: Vec<f64>,
    /// Overrides the file-level tolerance for this output.
    #[serde(default)]
    tolerance: Option<Tolerance>,
    /// Per-element absolute error bound, for outputs that legitimately round
    /// (MatMul sums). Replaces the tolerance when present.
    #[serde(default)]
    bound: Option<Vec<f64>>,
}

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn run_graph(graph: &Path, inputs: Option<&Path>) -> Output {
    let mut command = Command::new(env!("CARGO_BIN_EXE_oxfuse"));
    command.arg("run").arg(graph);
    if let Some(inputs) = inputs {
        command.arg("--inputs").arg(inputs);
    }
    command.output().unwrap()
}

fn within(got: f64, want: f64, tolerance: &Tolerance) -> bool {
    got == want || (got - want).abs() <= tolerance.abs + tolerance.rel * want.abs()
}

/// How one element is judged. A bound wins; otherwise zero tolerance on a
/// graph without MatMul means bit equality, so a wrong zero sign is caught;
/// otherwise the tolerance applies.
fn element_ok(
    got: f32,
    want: f64,
    tolerance: &Tolerance,
    bound: Option<f64>,
    bitwise: bool,
) -> bool {
    if let Some(bound) = bound {
        (f64::from(got) - want).abs() <= bound
    } else if bitwise && tolerance.abs == 0.0 && tolerance.rel == 0.0 {
        got.to_bits() == (want as f32).to_bits()
    } else {
        within(f64::from(got), want, tolerance)
    }
}

/// Runs every `NAME.json` in `dir` (skipping `.inputs.json` and
/// `.expected.json`) and returns how many graphs were checked.
fn check_dir(dir: &Path) -> usize {
    let mut checked = 0;
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
        let graph: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(&path).unwrap()).unwrap();
        // NumPy does not define the sign of a zero matmul sum, so graphs with
        // MatMul are compared by value; everything else is compared by bits.
        let bitwise = !graph["nodes"]
            .as_array()
            .unwrap()
            .iter()
            .any(|n| n["op"] == "MatMul");
        let inputs = dir.join(format!("{stem}.inputs.json"));
        let output = run_graph(&path, inputs.exists().then_some(inputs.as_path()));
        assert!(
            output.status.success(),
            "{name}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        let got: OutputsFile = serde_json::from_slice(&output.stdout).unwrap();
        let expected = fs::read_to_string(dir.join(format!("{stem}.expected.json"))).unwrap();
        let want: Expected = serde_json::from_str(&expected).unwrap();
        assert_eq!(
            got.outputs.len(),
            want.outputs.len(),
            "{name}: output count"
        );
        for (g, w) in got.outputs.iter().zip(&want.outputs) {
            assert_eq!(g.name, w.name, "{name}: output name");
            assert_eq!(g.shape, w.shape, "{name}: shape of {}", g.name);
            assert_eq!(
                g.values.len(),
                w.values.len(),
                "{name}: length of {}",
                g.name
            );
            let tolerance = w.tolerance.as_ref().unwrap_or(&want.tolerance);
            for (i, (&gv, &wv)) in g.values.iter().zip(&w.values).enumerate() {
                let bound = w.bound.as_ref().map(|b| b[i]);
                assert!(
                    element_ok(gv, wv, tolerance, bound, bitwise),
                    "{name}: {}[{i}] is {gv}, want {wv} (bound {bound:?})",
                    g.name
                );
            }
        }
        checked += 1;
    }
    checked
}

#[test]
fn every_example_matches_its_expected_output() {
    let checked = check_dir(&root().join("examples"));
    assert!(
        checked >= 2,
        "expected at least two examples, found {checked}"
    );
}

#[test]
fn every_corpus_graph_matches_the_numpy_oracle() {
    let checked = check_dir(&root().join("tests/corpus"));
    assert!(
        checked >= 50,
        "expected at least 50 corpus graphs, found {checked}"
    );
}

#[test]
fn corpus_graphs_have_no_dead_nodes() {
    let dir = root().join("tests/corpus");
    for entry in fs::read_dir(&dir).unwrap() {
        let path = entry.unwrap().path();
        let name = path.file_name().unwrap().to_str().unwrap().to_owned();
        if !name.ends_with(".json")
            || name.ends_with(".inputs.json")
            || name.ends_with(".expected.json")
        {
            continue;
        }
        let graph: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(&path).unwrap()).unwrap();
        let nodes = graph["nodes"].as_array().unwrap();
        let mut live: BTreeSet<String> = graph["outputs"]
            .as_array()
            .unwrap()
            .iter()
            .map(|o| o.as_str().unwrap().to_owned())
            .collect();
        let mut changed = true;
        while changed {
            changed = false;
            for node in nodes {
                if live.contains(node["name"].as_str().unwrap()) {
                    for input in node["inputs"].as_array().unwrap() {
                        changed |= live.insert(input.as_str().unwrap().to_owned());
                    }
                }
            }
        }
        for node in nodes {
            let node_name = node["name"].as_str().unwrap();
            assert!(
                live.contains(node_name),
                "{name}: node {node_name} is dead, so the oracle never checks it"
            );
        }
    }
}

#[test]
fn bitwise_mode_distinguishes_signed_zeros() {
    let exact = Tolerance { abs: 0.0, rel: 0.0 };
    assert!(element_ok(0.0, 0.0, &exact, None, true));
    assert!(!element_ok(-0.0, 0.0, &exact, None, true));
    assert!(!element_ok(0.0, -0.0, &exact, None, true));
    assert!(element_ok(-0.0, 0.0, &exact, None, false));
}

#[test]
fn a_bound_replaces_the_tolerance() {
    let exact = Tolerance { abs: 0.0, rel: 0.0 };
    assert!(element_ok(1.0, 1.25, &exact, Some(0.25), true));
    assert!(!element_ok(1.0, 1.25, &exact, Some(0.2), true));
}

#[test]
fn tolerance_zero_means_exact() {
    let exact = Tolerance { abs: 0.0, rel: 0.0 };
    assert!(within(1.5, 1.5, &exact));
    assert!(within(-0.0, 0.0, &exact));
    assert!(!within(1.5, 1.5000001, &exact));
}

#[test]
fn tolerance_allows_absolute_and_relative_error() {
    let loose = Tolerance {
        abs: 1e-6,
        rel: 1e-3,
    };
    assert!(within(1000.5, 1000.0, &loose));
    assert!(within(1e-7, 0.0, &loose));
    assert!(!within(1002.0, 1000.0, &loose));
}

#[test]
fn relu_example_prints_negative_zero_as_positive_zero() {
    let examples = root().join("examples");
    let output = run_graph(
        &examples.join("relu.json"),
        Some(&examples.join("relu.inputs.json")),
    );
    assert!(output.status.success());
    assert_eq!(
        String::from_utf8(output.stdout).unwrap(),
        "{\"outputs\":[{\"name\":\"y\",\"shape\":[2,3],\"values\":[0.0,0.0,2.5,0.0,3.0,0.0]}]}\n"
    );
}
