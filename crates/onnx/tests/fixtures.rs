//! Every committed fixture, run on the reference and fast CPU backends and
//! compared with the float64 values the generator wrote (see
//! `scripts/gen_onnx_fixtures.py`, which checks them against onnxruntime).

mod common;

use common::*;
use oxfuse_core::Graph;
use oxfuse_onnx::{ImportOptions, import};
use oxfuse_runtime::{Backend, CpuBackend, ReferenceBackend, Tensor};
use serde_json::Value;
use std::path::PathBuf;

fn dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

struct Fixture {
    name: String,
    graph: Graph,
    inputs: Vec<Tensor>,
    outputs: Vec<(String, Vec<usize>, Vec<f64>)>,
    abs: f64,
    rel: f64,
}

fn numbers(v: &Value) -> Vec<f64> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap())
        .collect()
}

fn shape(v: &Value) -> Vec<usize> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_u64().unwrap() as usize)
        .collect()
}

fn load_fixtures() -> Vec<Fixture> {
    let mut found = Vec::new();
    for entry in std::fs::read_dir(dir()).unwrap() {
        let path = entry.unwrap().path();
        let file = path.file_name().unwrap().to_string_lossy().into_owned();
        let Some(name) = file.strip_suffix(".expected.json") else {
            continue;
        };
        let expected: Value =
            serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        let mut opts = ImportOptions::default();
        for (k, v) in expected["dims"].as_object().unwrap() {
            opts.dims.insert(k.clone(), v.as_u64().unwrap() as usize);
        }
        let bytes = std::fs::read(dir().join(format!("{name}.onnx"))).unwrap();
        let graph = import(&bytes, &opts).unwrap_or_else(|e| panic!("{name}: {e}"));
        // Feed in the graph's own input order, looked up by name.
        let feeds = expected["inputs"].as_array().unwrap();
        let inputs = graph
            .inputs()
            .iter()
            .map(|id| {
                let label = graph.label(*id);
                let f = feeds
                    .iter()
                    .find(|f| f["name"] == label.as_str())
                    .unwrap_or_else(|| panic!("{name}: no data for input `{label}`"));
                let data: Vec<f32> = numbers(&f["data"]).iter().map(|v| *v as f32).collect();
                Tensor::new(shape(&f["shape"]), data).unwrap()
            })
            .collect();
        let outputs = expected["outputs"]
            .as_array()
            .unwrap()
            .iter()
            .map(|o| {
                (
                    o["name"].as_str().unwrap().to_owned(),
                    shape(&o["shape"]),
                    numbers(&o["data"]),
                )
            })
            .collect();
        found.push(Fixture {
            name: name.to_owned(),
            graph,
            inputs,
            outputs,
            abs: expected["tolerance"]["abs"].as_f64().unwrap(),
            rel: expected["tolerance"]["rel"].as_f64().unwrap(),
        });
    }
    found.sort_by(|a, b| a.name.cmp(&b.name));
    found
}

fn run(backend: &dyn Backend, f: &Fixture) -> Vec<Tensor> {
    backend
        .compile(&f.graph)
        .unwrap_or_else(|e| panic!("{}: {e}", f.name))
        .run(&f.inputs)
        .unwrap_or_else(|e| panic!("{}: {e}", f.name))
}

fn assert_matches(f: &Fixture, got: &[Tensor], backend: &str) {
    assert_eq!(got.len(), f.outputs.len(), "{} on {backend}", f.name);
    assert_eq!(f.graph.output_names().len(), f.outputs.len());
    for (i, (name, shape, want)) in f.outputs.iter().enumerate() {
        assert_eq!(&f.graph.output_names()[i], name, "{}: output order", f.name);
        assert_eq!(
            got[i].shape(),
            shape.as_slice(),
            "{}/{name} on {backend}",
            f.name
        );
        for (j, (g, w)) in got[i].data().iter().zip(want).enumerate() {
            let g = f64::from(*g);
            let w32 = f64::from(*w as f32);
            let ok = if f.abs == 0.0 && f.rel == 0.0 {
                g == w32
            } else {
                (g - w).abs() <= f.abs + f.rel * w.abs()
            };
            assert!(ok, "{}/{name}[{j}] on {backend}: got {g}, want {w}", f.name);
        }
    }
}

#[test]
fn every_fixture_matches_the_float64_values_on_the_reference_backend() {
    let all = load_fixtures();
    assert!(all.len() >= 10, "fixtures are missing");
    for f in &all {
        assert_matches(f, &run(&ReferenceBackend, f), "reference");
    }
}

#[test]
fn every_fixture_matches_on_the_cpu_backend_and_is_bit_equal_to_reference() {
    for f in &load_fixtures() {
        let cpu = run(&CpuBackend, f);
        assert_matches(f, &cpu, "cpu");
        let reference = run(&ReferenceBackend, f);
        for (a, b) in cpu.iter().zip(&reference) {
            let a: Vec<u32> = a.data().iter().map(|v| v.to_bits()).collect();
            let b: Vec<u32> = b.data().iter().map(|v| v.to_bits()).collect();
            assert_eq!(a, b, "{}: cpu differs from reference", f.name);
        }
    }
}

#[test]
fn fixture_set_covers_every_supported_op() {
    let mut seen = std::collections::BTreeSet::new();
    for entry in std::fs::read_dir(dir()).unwrap() {
        let path = entry.unwrap().path();
        if path.extension().is_some_and(|e| e == "onnx") {
            let m = oxfuse_onnx::decode_model(&std::fs::read(&path).unwrap(), &Default::default())
                .unwrap();
            for n in m.graph.unwrap().node {
                seen.insert(n.op_type);
            }
        }
    }
    let want = [
        "Add",
        "Constant",
        "Dropout",
        "Flatten",
        "Gemm",
        "Identity",
        "MatMul",
        "Mul",
        "Relu",
        "Reshape",
        "Sigmoid",
        "Transpose",
    ];
    let got: Vec<&str> = seen.iter().map(String::as_str).collect();
    assert_eq!(got, want);
}

#[test]
fn non_finite_initializer_runs_without_panic() {
    let m = model(
        13,
        graph(
            vec![input("x", &[3])],
            vec![init(
                "w",
                &[3],
                &[f32::NAN, f32::INFINITY, f32::NEG_INFINITY],
            )],
            vec![node("Add", &["x", "w"], &["y"])],
            &["y"],
        ),
    );
    let g = load_ok(&m);
    let x = Tensor::new(vec![3], vec![1.0, 2.0, 3.0]).unwrap();
    for backend in [&ReferenceBackend as &dyn Backend, &CpuBackend] {
        let out = backend
            .compile(&g)
            .unwrap()
            .run(std::slice::from_ref(&x))
            .unwrap();
        let d = out[0].data();
        assert!(d[0].is_nan() && d[1] == f32::INFINITY && d[2] == f32::NEG_INFINITY);
    }
}
