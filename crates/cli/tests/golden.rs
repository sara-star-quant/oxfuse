//! The golden corpus: graphs plus the exact bits Oxfuse returns, built from the
//! repository's examples and some generated sweeps. `just golden` writes it.

use serde_json::{Value, json};
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// Relative path to file contents.
type Files = BTreeMap<String, Vec<u8>>;

struct Case {
    name: String,
    graph: String,
    inputs: Option<String>,
    oracle: Value,
}

fn read(path: &Path) -> String {
    fs::read_to_string(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

fn from_files(dir: &str, stem: &str, name: &str) -> Case {
    let dir = root().join(dir);
    let inputs = dir.join(format!("{stem}.inputs.json"));
    let expected = dir.join(format!("{stem}.expected.json"));
    let oracle = if expected.exists() {
        serde_json::from_str::<Value>(&read(&expected)).unwrap()["tolerance"].clone()
    } else {
        Value::Null
    };
    Case {
        name: name.to_owned(),
        graph: read(&dir.join(format!("{stem}.json"))),
        inputs: inputs.exists().then(|| read(&inputs)),
        oracle,
    }
}

/// A small deterministic generator so the sweeps never change.
fn lcg(seed: u64) -> impl FnMut() -> f32 {
    let mut state = seed;
    move || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 40) as f32 / (1u64 << 24) as f32) * 4.0 - 2.0
    }
}

fn graph_text(
    inputs: &[(&str, Vec<usize>)],
    constants: Value,
    nodes: Value,
    outputs: &[&str],
) -> String {
    let inputs: Vec<Value> = inputs
        .iter()
        .map(|(n, s)| json!({ "name": n, "dtype": "f32", "shape": s }))
        .collect();
    serde_json::to_string_pretty(&json!({
        "format_version": 1,
        "inputs": inputs,
        "constants": constants,
        "nodes": nodes,
        "outputs": outputs,
    }))
    .unwrap()
        + "\n"
}

fn inputs_text(name: &str, shape: &[usize], values: &[f32]) -> String {
    serde_json::to_string_pretty(&json!({ name: { "shape": shape, "values": values } })).unwrap()
        + "\n"
}

fn generated() -> Vec<Case> {
    let mut cases = Vec::new();
    // Relu on the values where implementations differ: zeros of both signs,
    // the smallest subnormal, the largest finite numbers.
    let specials = [
        0.0,
        -0.0,
        f32::from_bits(1),
        -f32::from_bits(1),
        f32::MIN_POSITIVE,
        -f32::MIN_POSITIVE,
        f32::MAX,
        f32::MIN,
        1.0,
        -1.0,
    ];
    cases.push(Case {
        name: "relu_specials".into(),
        graph: graph_text(
            &[("x", vec![specials.len()])],
            json!([]),
            json!([{ "name": "y", "op": "Relu", "inputs": ["x"] }]),
            &["y"],
        ),
        inputs: Some(inputs_text("x", &[specials.len()], &specials)),
        oracle: Value::Null,
    });
    // Sigmoid across the range where it is neither 0 nor 1, plus the tails.
    let n = 4096;
    let sweep: Vec<f32> = (0..n)
        .map(|i| -20.0 + 40.0 * i as f32 / (n - 1) as f32)
        .collect();
    cases.push(Case {
        name: "sigmoid_sweep".into(),
        graph: graph_text(
            &[("x", vec![n])],
            json!([]),
            json!([{ "name": "y", "op": "Sigmoid", "inputs": ["x"] }]),
            &["y"],
        ),
        inputs: Some(inputs_text("x", &[n], &sweep)),
        oracle: Value::Null,
    });
    // MatMul at row counts around the register block and the thread split.
    let (k, cols) = (300, 118);
    let mut next = lcg(7);
    let weights: Vec<f32> = (0..k * cols).map(|_| next()).collect();
    for rows in [1, 7, 13, 33, 64, 70] {
        let mut next = lcg(rows as u64);
        let a: Vec<f32> = (0..rows * k).map(|_| next()).collect();
        cases.push(Case {
            name: format!("matmul_rows_{rows}"),
            graph: graph_text(
                &[("x", vec![rows, k])],
                json!([{ "name": "w", "dtype": "f32", "shape": [k, cols], "values": weights }]),
                json!([{ "name": "y", "op": "MatMul", "inputs": ["x", "w"] }]),
                &["y"],
            ),
            inputs: Some(inputs_text("x", &[rows, k], &a)),
            oracle: Value::Null,
        });
    }
    cases
}

fn all_cases() -> Vec<Case> {
    let mut cases = Vec::new();
    for stem in [
        "const_relu",
        "elementwise",
        "matmul",
        "relu",
        "reshape",
        "sigmoid",
        "tiny_mlp",
        "transpose",
    ] {
        cases.push(from_files("examples", stem, &format!("example_{stem}")));
    }
    for stem in [
        "cse_duplicate",
        "dce_dead",
        "fold_chain",
        "layout_round_trip",
    ] {
        cases.push(from_files("tests/passes", stem, &format!("pass_{stem}")));
    }
    let mut graphs: Vec<String> = fs::read_dir(root().join("tests/corpus"))
        .unwrap()
        .filter_map(|e| {
            let name = e.unwrap().file_name().into_string().unwrap();
            let stem = name.strip_suffix(".json")?.to_owned();
            (!stem.contains('.')).then_some(stem)
        })
        .collect();
    graphs.sort();
    for stem in graphs {
        cases.push(from_files("tests/corpus", &stem, &format!("random_{stem}")));
    }
    cases.extend(generated());
    cases.sort_by(|a, b| a.name.cmp(&b.name));
    cases
}

fn scratch(name: &str) -> PathBuf {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(name);
    fs::create_dir_all(&dir).unwrap();
    dir
}

/// Runs `oxfuse run` on a backend; returns stdout.
fn run(backend: &str, graph: &Path, inputs: Option<&Path>) -> String {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_oxfuse"));
    cmd.arg("run").arg(graph).args(["--backend", backend]);
    if let Some(inputs) = inputs {
        cmd.arg("--inputs").arg(inputs);
    }
    let out = cmd.output().unwrap();
    assert!(
        out.status.success(),
        "{}: {}",
        graph.display(),
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8(out.stdout).unwrap()
}

fn contract_levels() -> BTreeMap<String, String> {
    let doc: Value = serde_json::from_str(&read(&root().join("docs/contract.json"))).unwrap();
    doc["rows"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|r| r["backend"] == "reference")
        .map(|r| {
            (
                r["op"].as_str().unwrap().to_owned(),
                r["level"].as_str().unwrap().to_owned(),
            )
        })
        .collect()
}

fn ops_of(graph: &str) -> Vec<String> {
    let doc: Value = serde_json::from_str(graph).unwrap();
    let mut ops: Vec<String> = doc["nodes"]
        .as_array()
        .unwrap()
        .iter()
        .map(|n| n["op"].as_str().unwrap().to_owned())
        .collect();
    ops.sort();
    ops.dedup();
    ops
}

fn hex_bits(values: &Value) -> Vec<String> {
    values
        .as_array()
        .unwrap()
        .iter()
        .map(|v| format!("0x{:08x}", (v.as_f64().unwrap() as f32).to_bits()))
        .collect()
}

/// Builds the whole corpus in memory, checking every case on both CPU backends.
fn build(tag: &str) -> Files {
    let work = scratch(tag);
    let levels = contract_levels();
    let mut files = Files::new();
    let mut index = Vec::new();
    for case in all_cases() {
        let graph_path = work.join(format!("{}.graph.json", case.name));
        fs::write(&graph_path, &case.graph).unwrap();
        let inputs_path = case.inputs.as_ref().map(|text| {
            let p = work.join(format!("{}.inputs.json", case.name));
            fs::write(&p, text).unwrap();
            p
        });
        let reference = run("reference", &graph_path, inputs_path.as_deref());
        let cpu = run("cpu", &graph_path, inputs_path.as_deref());
        assert_eq!(reference, cpu, "{}: reference and cpu differ", case.name);
        let parsed: Value = serde_json::from_str(&reference).unwrap();
        let bits: Vec<Value> = parsed["outputs"]
            .as_array()
            .unwrap()
            .iter()
            .map(|o| json!({ "name": o["name"], "shape": o["shape"], "bits": hex_bits(&o["values"]) }))
            .collect();
        let ops = ops_of(&case.graph);
        let contract: Vec<Value> = ops
            .iter()
            .map(|op| json!({ "op": op, "level": levels[op] }))
            .collect();
        let expected = json!({
            "case": case.name,
            "ops": ops,
            "contract": contract,
            "oracle_tolerance": case.oracle,
            "outputs": bits,
        });
        let dir = format!("cases/{}", case.name);
        files.insert(format!("{dir}/graph.json"), case.graph.into_bytes());
        if let Some(inputs) = case.inputs {
            files.insert(format!("{dir}/inputs.json"), inputs.into_bytes());
        }
        files.insert(
            format!("{dir}/expected.outputs.json"),
            reference.into_bytes(),
        );
        files.insert(
            format!("{dir}/expected.json"),
            (serde_json::to_string_pretty(&expected).unwrap() + "\n").into_bytes(),
        );
        index.push(case.name);
    }
    let index = json!({ "oxfuse_version": env!("CARGO_PKG_VERSION"), "cases": index });
    files.insert(
        "index.json".into(),
        (serde_json::to_string_pretty(&index).unwrap() + "\n").into_bytes(),
    );
    files.insert(
        "README.md".into(),
        read(&root().join("docs/golden-corpus.md")).into_bytes(),
    );
    files
}

/// `GOLDEN_OUT=dir cargo test -p oxfuse --test golden -- --ignored write_corpus`
#[test]
#[ignore = "writes files; run by `just golden`"]
fn write_corpus() {
    let out = PathBuf::from(std::env::var("GOLDEN_OUT").expect("set GOLDEN_OUT"));
    let _ = fs::remove_dir_all(&out);
    for (path, bytes) in build("golden_write") {
        let full = out.join(path);
        fs::create_dir_all(full.parent().unwrap()).unwrap();
        fs::write(full, bytes).unwrap();
    }
}

#[test]
fn building_twice_gives_the_same_bytes() {
    assert!(build("golden_a") == build("golden_b"));
}

#[test]
fn every_expected_value_is_hex_and_agrees_with_the_decimal_file() {
    let files = build("golden_hex");
    let mut checked = 0;
    for (path, bytes) in &files {
        if !path.ends_with("/expected.json") {
            continue;
        }
        let expected: Value = serde_json::from_slice(bytes).unwrap();
        let outputs_path = path.replace("expected.json", "expected.outputs.json");
        let decimal: Value = serde_json::from_slice(&files[&outputs_path]).unwrap();
        for (hex, dec) in expected["outputs"]
            .as_array()
            .unwrap()
            .iter()
            .zip(decimal["outputs"].as_array().unwrap())
        {
            let bits = hex["bits"].as_array().unwrap();
            let count: usize = hex["shape"]
                .as_array()
                .unwrap()
                .iter()
                .map(|d| d.as_u64().unwrap() as usize)
                .product();
            assert_eq!(bits.len(), count, "{path}");
            for b in bits {
                let b = b.as_str().unwrap();
                assert!(
                    b.len() == 10
                        && b.starts_with("0x")
                        && b[2..].chars().all(|c| matches!(c, '0'..='9' | 'a'..='f')),
                    "{path}: {b}"
                );
            }
            let again: Vec<Value> = hex_bits(&dec["values"])
                .into_iter()
                .map(Value::from)
                .collect();
            assert_eq!(bits, &again, "{path}");
            checked += bits.len();
        }
    }
    assert!(checked > 10_000, "only {checked} values");
}

#[test]
fn every_operator_is_covered_and_every_case_names_its_contract_level() {
    let files = build("golden_ops");
    let mut seen = std::collections::BTreeSet::new();
    for (path, bytes) in &files {
        if !path.ends_with("/expected.json") {
            continue;
        }
        let expected: Value = serde_json::from_slice(bytes).unwrap();
        let ops = expected["ops"].as_array().unwrap();
        let contract = expected["contract"].as_array().unwrap();
        assert_eq!(ops.len(), contract.len(), "{path}");
        for c in contract {
            assert!(!c["level"].as_str().unwrap().is_empty(), "{path}");
            seen.insert(c["op"].as_str().unwrap().to_owned());
        }
    }
    for name in oxfuse_core::Op::ALL_NAMES {
        assert!(seen.contains(name), "no golden case uses {name}");
    }
}

#[test]
fn the_cases_are_many_unique_and_indexed() {
    let files = build("golden_index");
    let index: Value = serde_json::from_slice(&files["index.json"]).unwrap();
    let names: Vec<&str> = index["cases"]
        .as_array()
        .unwrap()
        .iter()
        .map(|n| n.as_str().unwrap())
        .collect();
    assert!(names.len() >= 70, "{}", names.len());
    let mut sorted = names.clone();
    sorted.sort_unstable();
    sorted.dedup();
    assert_eq!(names, sorted);
    assert_eq!(index["oxfuse_version"], env!("CARGO_PKG_VERSION"));
    for name in names {
        assert!(files.contains_key(&format!("cases/{name}/graph.json")));
        assert!(files.contains_key(&format!("cases/{name}/expected.json")));
    }
}

/// The README's commands, run against a written corpus.
#[test]
fn the_readme_commands_work() {
    let dir = scratch("golden_readme");
    let _ = fs::remove_dir_all(&dir);
    for (path, bytes) in build("golden_readme_build") {
        let full = dir.join(path);
        fs::create_dir_all(full.parent().unwrap()).unwrap();
        fs::write(full, bytes).unwrap();
    }
    let readme = read(&dir.join("README.md"));
    let block = readme
        .split("```sh\n")
        .nth(1)
        .and_then(|rest| rest.split("```").next())
        .expect("a sh block");
    let mut commands = 0;
    for line in block.lines().filter(|l| l.starts_with("oxfuse ")) {
        let (command, redirect) = match line.split_once(" > ") {
            Some((c, f)) => (c, Some(f)),
            None => (line, None),
        };
        let args: Vec<&str> = command.split_whitespace().skip(1).collect();
        let out = Command::new(env!("CARGO_BIN_EXE_oxfuse"))
            .args(&args)
            .current_dir(&dir)
            .output()
            .unwrap();
        assert!(
            out.status.success(),
            "{line}: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        if let Some(file) = redirect {
            fs::write(dir.join(file), out.stdout).unwrap();
        } else {
            assert!(String::from_utf8_lossy(&out.stdout).contains("verdict: identical"));
        }
        commands += 1;
    }
    assert_eq!(commands, 2);
}
