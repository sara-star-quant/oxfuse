//! `oxfuse run --manifest`: a record of what ran, to diff and to pin.

use serde_json::Value;
use sha2::{Digest, Sha256};
use std::fs;
use std::path::PathBuf;
use std::process::{Command, Output};

fn example(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../examples")
        .join(name)
}

fn scratch(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(name)
}

fn run(graph: &str, inputs: Option<&str>, backend: &str, manifest: &PathBuf) -> Output {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_oxfuse"));
    cmd.arg("run").arg(example(graph));
    if let Some(inputs) = inputs {
        cmd.arg("--inputs").arg(example(inputs));
    }
    cmd.args(["--backend", backend])
        .arg("--manifest")
        .arg(manifest)
        .output()
        .unwrap()
}

fn manifest(graph: &str, inputs: Option<&str>, backend: &str, file: &str) -> Value {
    let path = scratch(file);
    let out = run(graph, inputs, backend, &path);
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    serde_json::from_str(&fs::read_to_string(&path).unwrap()).unwrap()
}

fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

#[test]
fn the_manifest_has_exactly_the_documented_keys() {
    let m = manifest(
        "matmul.json",
        Some("matmul.inputs.json"),
        "cpu",
        "keys.json",
    );
    let mut keys: Vec<&str> = m.as_object().unwrap().keys().map(String::as_str).collect();
    keys.sort_unstable();
    assert_eq!(
        keys,
        [
            "arch",
            "backend",
            "disabled_passes",
            "features",
            "inputs_sha256",
            "manifest_version",
            "model_sha256",
            "optimize",
            "os",
            "outputs",
            "oxfuse_version",
            "simd",
            "unix_time",
        ]
    );
    assert_eq!(m["manifest_version"], 1);
    let simd = m["simd"].as_str().unwrap();
    assert!(!simd.is_empty() && simd.chars().all(|c| c.is_ascii_alphanumeric() || c == '_'));
    assert_eq!(m["oxfuse_version"], env!("CARGO_PKG_VERSION"));
    assert_eq!(m["backend"], "cpu");
    assert_eq!(m["optimize"], true);
    assert_eq!(m["disabled_passes"], serde_json::json!([]));
    assert_eq!(m["os"], std::env::consts::OS);
    assert_eq!(m["arch"], std::env::consts::ARCH);
    assert!(m["unix_time"].as_u64().unwrap() > 1_700_000_000);
    let mut features: Vec<&str> = m["features"]
        .as_object()
        .unwrap()
        .keys()
        .map(String::as_str)
        .collect();
    features.sort_unstable();
    assert_eq!(features, ["onnx", "parallel", "wgpu"]);
    assert!(
        m["features"]
            .as_object()
            .unwrap()
            .values()
            .all(Value::is_boolean)
    );
}

#[test]
fn model_and_inputs_are_hashed_by_their_file_bytes() {
    let m = manifest(
        "matmul.json",
        Some("matmul.inputs.json"),
        "reference",
        "files.json",
    );
    let model = fs::read(example("matmul.json")).unwrap();
    let inputs = fs::read(example("matmul.inputs.json")).unwrap();
    assert_eq!(m["model_sha256"], sha256_hex(&model));
    assert_eq!(m["inputs_sha256"], sha256_hex(&inputs));
}

#[test]
fn a_graph_without_inputs_has_a_null_inputs_hash() {
    let m = manifest("const_relu.json", None, "cpu", "noinputs.json");
    assert!(m["inputs_sha256"].is_null());
}

#[test]
fn an_output_hash_is_over_the_little_endian_bit_patterns() {
    // relu over [-1, 2] is [0, 2]; the digest was computed outside Rust
    // from struct.pack("<ff", 0.0, 2.0).
    let inputs = scratch("relu_two.inputs.json");
    fs::write(
        &inputs,
        r#"{ "x": { "shape": [2], "values": [-1.0, 2.0] } }"#,
    )
    .unwrap();
    let graph = scratch("relu_two.json");
    fs::write(
        &graph,
        r#"{ "format_version": 1, "inputs": [{ "name": "x", "dtype": "f32", "shape": [2] }],
             "nodes": [{ "name": "y", "op": "Relu", "inputs": ["x"] }], "outputs": ["y"] }"#,
    )
    .unwrap();
    let path = scratch("relu_two.manifest.json");
    let out = Command::new(env!("CARGO_BIN_EXE_oxfuse"))
        .arg("run")
        .arg(&graph)
        .arg("--inputs")
        .arg(&inputs)
        .arg("--manifest")
        .arg(&path)
        .output()
        .unwrap();
    assert!(out.status.success());
    let m: Value = serde_json::from_str(&fs::read_to_string(&path).unwrap()).unwrap();
    let outputs = m["outputs"].as_array().unwrap();
    assert_eq!(outputs.len(), 1);
    assert_eq!(outputs[0]["name"], "y");
    assert_eq!(outputs[0]["shape"], serde_json::json!([2]));
    assert_eq!(
        outputs[0]["sha256"],
        "3f710ac088db33363087de2b9a657541fe5447821debaa9fe5cbd538eb1a5f29"
    );
}

#[test]
fn the_hash_follows_every_bit_including_the_sign_of_zero() {
    // Hash what the run printed, parsed back: -0.0 and 0.0 differ in bits.
    let m = manifest("relu.json", Some("relu.inputs.json"), "cpu", "zero.json");
    let out = Command::new(env!("CARGO_BIN_EXE_oxfuse"))
        .args(["run"])
        .arg(example("relu.json"))
        .arg("--inputs")
        .arg(example("relu.inputs.json"))
        .output()
        .unwrap();
    let printed: Value = serde_json::from_slice(&out.stdout).unwrap();
    let values = printed["outputs"][0]["values"].as_array().unwrap();
    let bytes: Vec<u8> = values
        .iter()
        .flat_map(|v| (v.as_f64().unwrap() as f32).to_le_bytes())
        .collect();
    assert_eq!(m["outputs"][0]["sha256"], sha256_hex(&bytes));
    let flipped: Vec<u8> = bytes
        .chunks(4)
        .enumerate()
        .flat_map(|(i, c)| {
            let mut c = c.to_vec();
            if i == 0 {
                c[3] ^= 0x80;
            }
            c
        })
        .collect();
    assert_ne!(sha256_hex(&flipped), sha256_hex(&bytes));
}

#[test]
fn reference_and_cpu_agree_on_output_hashes_and_differ_in_the_backend_field() {
    let a = manifest(
        "tiny_mlp.json",
        Some("tiny_mlp.inputs.json"),
        "reference",
        "ref.json",
    );
    let b = manifest(
        "tiny_mlp.json",
        Some("tiny_mlp.inputs.json"),
        "cpu",
        "cpu.json",
    );
    assert_eq!(a["outputs"], b["outputs"]);
    assert_eq!(a["model_sha256"], b["model_sha256"]);
    assert_ne!(a["backend"], b["backend"]);
}

#[test]
fn two_runs_differ_only_in_the_time() {
    let mut a = manifest(
        "tiny_mlp.json",
        Some("tiny_mlp.inputs.json"),
        "cpu",
        "twice_a.json",
    );
    let mut b = manifest(
        "tiny_mlp.json",
        Some("tiny_mlp.inputs.json"),
        "cpu",
        "twice_b.json",
    );
    a.as_object_mut().unwrap().remove("unix_time");
    b.as_object_mut().unwrap().remove("unix_time");
    assert_eq!(a, b);
}

#[test]
fn passes_and_optimization_are_recorded() {
    let path = scratch("passes.json");
    let out = Command::new(env!("CARGO_BIN_EXE_oxfuse"))
        .arg("run")
        .arg(example("tiny_mlp.json"))
        .arg("--inputs")
        .arg(example("tiny_mlp.inputs.json"))
        .args(["--no-opt", "--disable-pass", "cse", "--manifest"])
        .arg(&path)
        .output()
        .unwrap();
    assert!(out.status.success());
    let m: Value = serde_json::from_str(&fs::read_to_string(&path).unwrap()).unwrap();
    assert_eq!(m["optimize"], false);
    assert_eq!(m["disabled_passes"], serde_json::json!(["cse"]));
}

#[test]
fn an_unwritable_path_is_an_error_after_the_outputs_are_printed() {
    let bad = scratch("no_such_dir").join("m.json");
    let out = run("relu.json", Some("relu.inputs.json"), "cpu", &bad);
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("error[E101]"), "{stderr}");
    assert!(stderr.contains("m.json"), "{stderr}");
    // The results are not lost.
    assert!(String::from_utf8_lossy(&out.stdout).contains("\"outputs\""));
}

#[test]
fn no_home_directory_or_user_name_leaks_into_the_file() {
    let m = manifest(
        "matmul.json",
        Some("matmul.inputs.json"),
        "cpu",
        "leak.json",
    );
    let text = m.to_string();
    for var in ["HOME", "USER", "USERNAME"] {
        if let Ok(v) = std::env::var(var)
            && v.len() > 2
        {
            assert!(!text.contains(&v), "{var} value leaked");
        }
    }
}

#[test]
fn a_failed_run_writes_no_manifest() {
    let path = scratch("failed.json");
    let _ = fs::remove_file(&path);
    let out = Command::new(env!("CARGO_BIN_EXE_oxfuse"))
        .arg("run")
        .arg(example("matmul.json"))
        .arg("--manifest")
        .arg(&path)
        .output()
        .unwrap();
    assert!(!out.status.success(), "matmul needs its inputs file");
    assert!(!path.exists());
}
