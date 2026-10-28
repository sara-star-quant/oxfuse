//! `oxfuse compare A B`: two output files, one verdict.

use serde_json::{Value, json};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn example(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../examples")
        .join(name)
}

fn scratch(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(name)
}

fn oxfuse(args: &[&Path]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_oxfuse"))
        .args(args)
        .output()
        .unwrap()
}

fn text(bytes: &[u8]) -> String {
    String::from_utf8_lossy(bytes).into_owned()
}

/// Runs an example on a backend; returns the outputs file and its manifest.
fn run(graph: &str, inputs: Option<&str>, backend: &str, tag: &str) -> (PathBuf, PathBuf) {
    let out = scratch(&format!("{tag}.out.json"));
    let manifest = scratch(&format!("{tag}.manifest.json"));
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_oxfuse"));
    cmd.arg("run").arg(example(graph));
    if let Some(inputs) = inputs {
        cmd.arg("--inputs").arg(example(inputs));
    }
    let output = cmd
        .args(["--backend", backend, "--manifest"])
        .arg(&manifest)
        .output()
        .unwrap();
    assert!(output.status.success(), "{}", text(&output.stderr));
    fs::write(&out, output.stdout).unwrap();
    (out, manifest)
}

fn outputs_file(tag: &str, outputs: Value) -> PathBuf {
    let path = scratch(&format!("{tag}.json"));
    fs::write(&path, json!({ "outputs": outputs }).to_string()).unwrap();
    path
}

fn one(name: &str, shape: &[usize], values: &[f64]) -> Value {
    json!({ "name": name, "shape": shape, "values": values })
}

fn compare(a: &Path, b: &Path, extra: &[&Path]) -> Output {
    let mut args: Vec<&Path> = vec![Path::new("compare"), a, b];
    args.extend(extra);
    oxfuse(&args)
}

#[test]
fn equal_files_are_identical_and_exit_zero() {
    let f = outputs_file("eq", json!([one("y", &[2], &[1.0, 2.0])]));
    let out = compare(&f, &f, &[]);
    assert!(out.status.success(), "{}", text(&out.stderr));
    let stdout = text(&out.stdout);
    assert!(stdout.contains("y: identical"), "{stdout}");
    assert!(stdout.contains("verdict: identical"), "{stdout}");
}

#[test]
fn reference_and_cpu_are_identical_on_every_example_that_runs() {
    for name in [
        "matmul",
        "relu",
        "sigmoid",
        "elementwise",
        "reshape",
        "transpose",
        "tiny_mlp",
    ] {
        let inputs = format!("{name}.inputs.json");
        let (a, _) = run(
            &format!("{name}.json"),
            Some(&inputs),
            "reference",
            &format!("{name}_r"),
        );
        let (b, _) = run(
            &format!("{name}.json"),
            Some(&inputs),
            "cpu",
            &format!("{name}_c"),
        );
        let out = compare(&a, &b, &[]);
        assert!(out.status.success(), "{name}: {}", text(&out.stdout));
        assert!(text(&out.stdout).contains("verdict: identical"), "{name}");
    }
}

#[test]
fn one_flipped_low_bit_reports_the_count_the_ulp_and_the_index() {
    let a = outputs_file("lo_a", json!([one("y", &[4], &[1.0, 2.0, 3.0, 4.0])]));
    // 3.0 + 1 ulp in f32 is 3.0000002.
    let b = outputs_file(
        "lo_b",
        json!([one("y", &[4], &[1.0, 2.0, 3.000_000_2, 4.0])]),
    );
    let out = compare(&a, &b, &[]);
    assert!(!out.status.success());
    let stdout = text(&out.stdout);
    assert!(
        stdout.contains("y: 1 of 4 values differ, max 1 ulp, first at index 2"),
        "{stdout}"
    );
    assert!(stdout.contains("verdict: unresolved"), "{stdout}");
    assert!(text(&out.stderr).contains("error[E110]"));
}

#[test]
fn signed_zero_is_zero_ulp_but_still_a_difference() {
    let a = outputs_file("z_a", json!([one("y", &[1], &[0.0])]));
    let b = outputs_file("z_b", json!([one("y", &[1], &[-0.0])]));
    let out = compare(&a, &b, &[]);
    assert!(!out.status.success());
    assert!(
        text(&out.stdout).contains("y: 1 of 1 values differ, max 0 ulp, first at index 0"),
        "{}",
        text(&out.stdout)
    );
}

#[test]
fn two_oxfuse_runs_that_differ_are_a_contract_violation() {
    let (a, ma) = run(
        "sigmoid.json",
        Some("sigmoid.inputs.json"),
        "reference",
        "v_a",
    );
    let (_, mb) = run("sigmoid.json", Some("sigmoid.inputs.json"), "cpu", "v_b");
    // Same manifest hashes, so make the second file differ by one ulp and
    // give it a manifest that describes it.
    let mut doc: Value = serde_json::from_str(&fs::read_to_string(&a).unwrap()).unwrap();
    let first = doc["outputs"][0]["values"][0].as_f64().unwrap() as f32;
    doc["outputs"][0]["values"][0] = json!(f32::from_bits(first.to_bits() + 1));
    let b = scratch("v_b.out.json");
    fs::write(&b, doc.to_string()).unwrap();
    fix_hashes(&mb, &b);
    let out = compare(
        &a,
        &b,
        &[
            Path::new("--manifest-a"),
            &ma,
            Path::new("--manifest-b"),
            &mb,
        ],
    );
    assert!(!out.status.success());
    assert!(
        text(&out.stdout).contains("verdict: contract violation"),
        "{}",
        text(&out.stdout)
    );
}

/// Rewrites a manifest's output hashes to describe `outputs`.
fn fix_hashes(manifest: &Path, outputs: &Path) {
    use sha2::{Digest, Sha256};
    let mut m: Value = serde_json::from_str(&fs::read_to_string(manifest).unwrap()).unwrap();
    let o: Value = serde_json::from_str(&fs::read_to_string(outputs).unwrap()).unwrap();
    for (i, t) in o["outputs"].as_array().unwrap().iter().enumerate() {
        let bytes: Vec<u8> = t["values"]
            .as_array()
            .unwrap()
            .iter()
            .flat_map(|v| (v.as_f64().unwrap() as f32).to_bits().to_le_bytes())
            .collect();
        let hex: String = Sha256::digest(bytes)
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect();
        m["outputs"][i]["sha256"] = json!(hex);
    }
    fs::write(manifest, m.to_string()).unwrap();
}

/// Two outputs that differ by one ulp, each with a manifest that matches it.
fn pair_with_manifests(
    tag: &str,
    edit_a: impl Fn(&mut Value),
    edit_b: impl Fn(&mut Value),
) -> Output {
    let (a, ma) = run(
        "sigmoid.json",
        Some("sigmoid.inputs.json"),
        "reference",
        &format!("{tag}_a"),
    );
    let (_, mb) = run(
        "sigmoid.json",
        Some("sigmoid.inputs.json"),
        "reference",
        &format!("{tag}_b"),
    );
    let mut doc: Value = serde_json::from_str(&fs::read_to_string(&a).unwrap()).unwrap();
    let first = doc["outputs"][0]["values"][0].as_f64().unwrap() as f32;
    doc["outputs"][0]["values"][0] = json!(f32::from_bits(first.to_bits() + 1));
    let b = scratch(&format!("{tag}_b.out.json"));
    fs::write(&b, doc.to_string()).unwrap();
    fix_hashes(&mb, &b);
    for (path, edit) in [(&ma, &edit_a as &dyn Fn(&mut Value)), (&mb, &edit_b)] {
        let mut m: Value = serde_json::from_str(&fs::read_to_string(path).unwrap()).unwrap();
        edit(&mut m);
        fs::write(path, m.to_string()).unwrap();
    }
    compare(
        &a,
        &b,
        &[
            Path::new("--manifest-a"),
            &ma,
            Path::new("--manifest-b"),
            &mb,
        ],
    )
}

#[test]
fn a_gpu_run_against_a_cpu_run_is_a_permitted_variation() {
    let out = pair_with_manifests("gpu", |_| {}, |m| m["backend"] = json!("wgpu"));
    assert!(out.status.success(), "{}", text(&out.stderr));
    assert!(
        text(&out.stdout).contains("verdict: permitted variation"),
        "{}",
        text(&out.stdout)
    );
}

#[test]
fn different_oxfuse_versions_are_unresolved_not_a_violation() {
    let out = pair_with_manifests("ver", |_| {}, |m| m["oxfuse_version"] = json!("0.0.1"));
    assert!(!out.status.success());
    assert!(
        text(&out.stdout).contains("verdict: unresolved"),
        "{}",
        text(&out.stdout)
    );
}

#[test]
fn only_one_manifest_is_unresolved() {
    let (a, ma) = run(
        "sigmoid.json",
        Some("sigmoid.inputs.json"),
        "reference",
        "one_a",
    );
    let mut doc: Value = serde_json::from_str(&fs::read_to_string(&a).unwrap()).unwrap();
    let first = doc["outputs"][0]["values"][0].as_f64().unwrap() as f32;
    doc["outputs"][0]["values"][0] = json!(f32::from_bits(first.to_bits() + 1));
    let b = scratch("one_b.out.json");
    fs::write(&b, doc.to_string()).unwrap();
    let out = compare(&a, &b, &[Path::new("--manifest-a"), &ma]);
    assert!(text(&out.stdout).contains("verdict: unresolved"));
}

#[test]
fn a_manifest_that_does_not_describe_its_file_is_an_error() {
    let (a, ma) = run("relu.json", Some("relu.inputs.json"), "cpu", "mm_a");
    let (b, _) = run("sigmoid.json", Some("sigmoid.inputs.json"), "cpu", "mm_b");
    // ma describes relu's output, not sigmoid's.
    let out = compare(
        &a,
        &b,
        &[
            Path::new("--manifest-a"),
            &a,
            Path::new("--manifest-b"),
            &ma,
        ],
    );
    assert!(
        text(&out.stderr).contains("error[E112]"),
        "{}",
        text(&out.stderr)
    );
    let out = compare(&b, &a, &[Path::new("--manifest-a"), &ma]);
    assert!(
        text(&out.stderr).contains("error[E112]"),
        "{}",
        text(&out.stderr)
    );
}

#[test]
fn output_name_and_shape_mismatches_are_named_errors() {
    let y = outputs_file("n_y", json!([one("y", &[2], &[1.0, 2.0])]));
    let z = outputs_file("n_z", json!([one("z", &[2], &[1.0, 2.0])]));
    let both = outputs_file(
        "n_both",
        json!([one("y", &[2], &[1.0, 2.0]), one("z", &[1], &[1.0])]),
    );
    let wide = outputs_file("n_wide", json!([one("y", &[1, 2], &[1.0, 2.0])]));
    for (a, b, needle) in [
        (&y, &z, "`y`"),
        (&y, &both, "`z`"),
        (&both, &y, "`z`"),
        (&y, &wide, "shape"),
    ] {
        let out = compare(a, b, &[]);
        assert!(!out.status.success());
        let err = text(&out.stderr);
        assert!(err.contains("error[E111]") && err.contains(needle), "{err}");
    }
}

#[test]
fn a_repeated_output_name_is_an_error() {
    let dup = outputs_file(
        "dup",
        json!([one("y", &[1], &[1.0]), one("y", &[1], &[1.0])]),
    );
    let out = compare(&dup, &dup, &[]);
    assert!(text(&out.stderr).contains("error[E111]"));
}

#[test]
fn unreadable_and_malformed_files_are_named_errors() {
    let ok = outputs_file("ok", json!([one("y", &[1], &[1.0])]));
    let missing = scratch("does_not_exist.json");
    assert!(text(&compare(&ok, &missing, &[]).stderr).contains("error[E101]"));
    let bad = scratch("bad.json");
    fs::write(&bad, "{ not json").unwrap();
    assert!(text(&compare(&bad, &ok, &[]).stderr).contains("error[E113]"));
    let extra = scratch("extra.json");
    fs::write(&extra, r#"{ "outputs": [], "surprise": 1 }"#).unwrap();
    assert!(text(&compare(&extra, &ok, &[]).stderr).contains("error[E113]"));
}

#[test]
fn a_value_count_that_disagrees_with_the_shape_is_an_error() {
    let ok = outputs_file("sh_ok", json!([one("y", &[2], &[1.0, 2.0])]));
    let short = outputs_file("sh_short", json!([one("y", &[2], &[1.0])]));
    assert!(text(&compare(&ok, &short, &[]).stderr).contains("error[E113]"));
}
