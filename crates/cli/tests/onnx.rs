//! ONNX models through the command line. The importer is an optional
//! feature: the default build must say so clearly, and the feature build
//! must run every committed fixture on every backend.

use std::path::PathBuf;
use std::process::{Command, Output};

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn oxfuse(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_oxfuse"))
        .args(args)
        .current_dir(root())
        .output()
        .unwrap()
}

fn text(bytes: &[u8]) -> String {
    String::from_utf8_lossy(bytes).into_owned()
}

const FIXTURES: &str = "crates/onnx/tests/fixtures";

#[cfg(not(feature = "onnx"))]
#[test]
fn an_onnx_file_without_the_feature_is_a_clear_error() {
    for file in ["linear.onnx", "linear.ONNX"] {
        let path = format!("{FIXTURES}/{file}");
        for args in [["check", &path], ["inspect", &path]] {
            let out = oxfuse(&args);
            assert!(!out.status.success());
            let err = text(&out.stderr);
            assert!(
                err.contains("error[E107]") && err.contains("--features onnx"),
                "{err}"
            );
        }
    }
}

#[cfg(feature = "onnx")]
mod enabled {
    use super::*;
    use serde_json::{Value, json};

    fn names() -> Vec<String> {
        let mut found: Vec<String> = std::fs::read_dir(root().join(FIXTURES))
            .unwrap()
            .filter_map(|e| {
                let f = e.unwrap().file_name().to_string_lossy().into_owned();
                f.strip_suffix(".expected.json").map(str::to_owned)
            })
            .collect();
        found.sort();
        assert!(found.len() >= 10);
        found
    }

    fn expected(name: &str) -> Value {
        serde_json::from_str(
            &std::fs::read_to_string(root().join(format!("{FIXTURES}/{name}.expected.json")))
                .unwrap(),
        )
        .unwrap()
    }

    /// A file of its own for every call: tests of one process run in parallel,
    /// and two tests of the same fixture used to write one path at once, so
    /// one of them could read the file half written.
    fn scratch(file: &str, contents: &str) -> String {
        use std::sync::atomic::{AtomicUsize, Ordering};
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let dir = std::env::temp_dir().join(format!("oxfuse-onnx-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join(format!("{}-{file}", NEXT.fetch_add(1, Ordering::Relaxed)));
        std::fs::write(&path, contents).unwrap();
        path.to_string_lossy().into_owned()
    }

    /// `--dim` flags for a fixture's named dimensions.
    fn dim_flags(e: &Value) -> Vec<String> {
        e["dims"]
            .as_object()
            .unwrap()
            .iter()
            .flat_map(|(k, v)| ["--dim".to_owned(), format!("{k}={v}")])
            .collect()
    }

    fn inputs_file(name: &str, e: &Value) -> String {
        let mut map = serde_json::Map::new();
        for i in e["inputs"].as_array().unwrap() {
            map.insert(
                i["name"].as_str().unwrap().to_owned(),
                json!({"shape": i["shape"], "values": i["data"]}),
            );
        }
        scratch(
            &format!("{name}.inputs.json"),
            &Value::Object(map).to_string(),
        )
    }

    fn run_fixture(name: &str, backend: Option<&str>) -> Option<Value> {
        let e = expected(name);
        let model = format!("{FIXTURES}/{name}.onnx");
        let inputs = inputs_file(name, &e);
        let mut args = vec!["run".to_owned(), model, "--inputs".into(), inputs];
        args.extend(dim_flags(&e));
        if let Some(b) = backend {
            args.extend(["--backend".to_owned(), b.to_owned()]);
        }
        let args: Vec<&str> = args.iter().map(String::as_str).collect();
        let out = oxfuse(&args);
        if !out.status.success() && text(&out.stderr).contains("no adapter") {
            return None;
        }
        assert!(out.status.success(), "{name}: {}", text(&out.stderr));
        Some(serde_json::from_str(&text(&out.stdout)).unwrap())
    }

    fn assert_fixture_output(name: &str, got: &Value, backend: &str) {
        let e = expected(name);
        let (abs, rel) = (
            e["tolerance"]["abs"].as_f64().unwrap(),
            e["tolerance"]["rel"].as_f64().unwrap(),
        );
        let want = e["outputs"].as_array().unwrap();
        let got = got["outputs"].as_array().unwrap();
        assert_eq!(got.len(), want.len(), "{name} on {backend}");
        for (g, w) in got.iter().zip(want) {
            assert_eq!(g["name"], w["name"], "{name} on {backend}");
            assert_eq!(g["shape"], w["shape"], "{name} on {backend}");
            let (gv, wv) = (
                g["values"].as_array().unwrap(),
                w["data"].as_array().unwrap(),
            );
            for (x, y) in gv.iter().zip(wv) {
                let (x, y) = (x.as_f64().unwrap(), y.as_f64().unwrap());
                let ok = if abs == 0.0 && rel == 0.0 {
                    x == f64::from(y as f32)
                } else {
                    // Sigmoid on a GPU is looser than the CPU's; see docs/semantics.md.
                    let scale = if backend == "wgpu" { 100.0 } else { 1.0 };
                    (x - y).abs() <= scale * (abs + rel * y.abs())
                };
                assert!(ok, "{name} on {backend}: got {x}, want {y}");
            }
        }
    }

    #[test]
    fn run_onnx_model_matches_the_fixture_on_reference_and_cpu() {
        for name in names() {
            for backend in [None, Some("reference"), Some("cpu")] {
                let got = run_fixture(&name, backend).unwrap();
                assert_fixture_output(&name, &got, backend.unwrap_or("default"));
            }
        }
    }

    #[test]
    fn reference_and_cpu_print_the_same_bytes_for_onnx_models() {
        for name in names() {
            let a = run_fixture(&name, Some("reference")).unwrap();
            let b = run_fixture(&name, Some("cpu")).unwrap();
            assert_eq!(a, b, "{name}");
        }
    }

    #[cfg(feature = "wgpu")]
    #[test]
    fn run_onnx_model_matches_the_fixture_on_wgpu() {
        for name in names() {
            // MatMul on the device fuses multiply-add, but dyadic fixtures are exact anyway.
            match run_fixture(&name, Some("wgpu")) {
                Some(got) => assert_fixture_output(&name, &got, "wgpu"),
                None => return,
            }
        }
    }

    #[test]
    fn check_and_inspect_accept_onnx_models() {
        let model = format!("{FIXTURES}/linear.onnx");
        let out = oxfuse(&["check", &model]);
        assert!(out.status.success(), "{}", text(&out.stderr));
        assert!(
            text(&out.stdout).starts_with("ok: 1 input(s)"),
            "{}",
            text(&out.stdout)
        );
        let out = oxfuse(&["inspect", &model]);
        assert!(out.status.success(), "{}", text(&out.stderr));
        assert!(
            text(&out.stdout).contains("MatMul"),
            "{}",
            text(&out.stdout)
        );
    }

    #[test]
    fn dim_flag_binds_a_named_dimension() {
        let model = format!("{FIXTURES}/named_dimension.onnx");
        let without = oxfuse(&["check", &model]);
        assert!(!without.status.success());
        let err = text(&without.stderr);
        assert!(
            err.contains("error[E312]") && err.contains("--dim N=VALUE"),
            "{err}"
        );
        let with = oxfuse(&["check", &model, "--dim", "N=3"]);
        assert!(with.status.success(), "{}", text(&with.stderr));
        let zero = oxfuse(&["check", &model, "--dim", "N=0"]);
        assert!(!zero.status.success());
    }

    #[test]
    fn dim_flag_bad_syntax() {
        let model = format!("{FIXTURES}/named_dimension.onnx");
        for bad in ["N", "N=", "=1", "N=x", "N=-1", "N=1=2"] {
            let out = oxfuse(&["check", &model, "--dim", bad]);
            assert!(!out.status.success(), "{bad}");
            assert!(
                text(&out.stderr).contains("--dim"),
                "{bad}: {}",
                text(&out.stderr)
            );
        }
    }

    #[test]
    fn dim_flag_repeated_name_is_an_error() {
        let model = format!("{FIXTURES}/named_dimension.onnx");
        let out = oxfuse(&["check", &model, "--dim", "N=3", "--dim", "N=4"]);
        assert!(!out.status.success());
        let err = text(&out.stderr);
        assert!(err.contains("error[E108]") && err.contains("`N`"), "{err}");
    }

    #[test]
    fn error_text_names_the_node_and_the_reason() {
        // Same length, so the encoded lengths stay valid: MatMul -> Einsum.
        let mut model = std::fs::read(root().join(FIXTURES).join("linear.onnx")).unwrap();
        let at = model.windows(6).position(|w| w == b"MatMul").unwrap();
        model[at..at + 6].copy_from_slice(b"Einsum");
        let path = scratch("einsum.onnx", "");
        std::fs::write(&path, model).unwrap();
        let out = oxfuse(&["check", &path]);
        assert!(!out.status.success());
        let err = text(&out.stderr);
        assert!(
            err.contains("error[E305]") && err.contains("Einsum"),
            "{err}"
        );
    }

    #[test]
    fn a_json_graph_still_loads_next_to_an_onnx_model() {
        let out = oxfuse(&["check", "examples/relu.json"]);
        assert!(out.status.success(), "{}", text(&out.stderr));
    }

    #[test]
    fn garbage_with_an_onnx_extension_is_a_decode_error_not_a_panic() {
        let path = scratch("junk.onnx", "this is not protobuf");
        let out = oxfuse(&["check", &path]);
        assert!(!out.status.success());
        assert!(
            text(&out.stderr).contains("error[E300]"),
            "{}",
            text(&out.stderr)
        );
    }
}
