//! The `check` and `run` commands. Both return the text to print, so they
//! are testable without spawning the binary.

use crate::format::{InputsFile, NamedTensorFile, OutputsFile};
use crate::{CliError, load_graph};
use oxfuse_core::passes::{PassManager, default_pipeline};
use oxfuse_core::{Error, Graph, Limits, Source};
use oxfuse_runtime::{Backend, CpuBackend, ReferenceBackend, ReferenceEvaluator, Tensor};
use std::collections::BTreeMap;
use std::fs;
use std::io::Read;
use std::path::Path;
use std::sync::Arc;

/// Values for the named dimensions of an ONNX model (`--dim NAME=VALUE`).
pub type Dims = BTreeMap<String, usize>;

/// Collect `--dim` arguments; naming a dimension twice is a mistake to
/// report, not a value to pick silently.
pub fn dims_from_args(args: &[(String, usize)]) -> Result<Dims, CliError> {
    let mut dims = Dims::new();
    for (name, value) in args {
        if dims.insert(name.clone(), *value).is_some() {
            return Err(CliError::DuplicateDim { name: name.clone() });
        }
    }
    Ok(dims)
}

/// Non-exhaustive: the `Wgpu` variant exists only when the `wgpu` feature is on,
/// so downstream code must not match this enum exhaustively.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum BackendKind {
    /// The naive oracle. Constant folding always uses it.
    Reference,
    /// The fast backend, bit-identical to the reference. See decision 0011.
    #[default]
    Cpu,
    /// A GPU through wgpu, built with the `wgpu` feature. Not bit-identical to
    /// the reference on `MatMul` and `Sigmoid`: see docs/semantics.md.
    #[cfg(feature = "wgpu")]
    Wgpu,
}

#[derive(Debug, Clone)]
pub struct RunOptions {
    pub backend: BackendKind,
    pub optimize: bool,
    pub disabled: Vec<String>,
    pub dump_ir: bool,
    /// Named dimensions of an ONNX model; ignored for JSON graphs.
    pub dims: Dims,
}

impl Default for RunOptions {
    fn default() -> Self {
        Self {
            backend: BackendKind::default(),
            optimize: true,
            disabled: Vec::new(),
            dump_ir: false,
            dims: Dims::new(),
        }
    }
}

#[derive(Debug)]
pub struct RunOutput {
    pub stdout: String,
    pub stderr: String,
}

fn pipeline(options: &RunOptions, limits: &Limits) -> Result<PassManager, CliError> {
    let mut manager =
        default_pipeline(limits.clone(), Arc::new(ReferenceEvaluator)).with_dumps(options.dump_ir);
    let known = manager.pass_names();
    // Validate even with --no-opt, so a typo is never silently ignored.
    for name in &options.disabled {
        if !known.contains(&name.as_str()) {
            return Err(CliError::UnknownPass {
                name: name.clone(),
                known: known.join(", "),
            });
        }
        manager = manager.without(name);
    }
    if !options.optimize {
        return Ok(PassManager::new(limits.clone()).with_dumps(options.dump_ir));
    }
    Ok(manager)
}

pub fn inspect(path: &Path, options: &RunOptions) -> Result<String, CliError> {
    let limits = Limits::default();
    let graph = load_path(path, &limits, &options.dims)?;
    let manager = pipeline(
        &RunOptions {
            dump_ir: true,
            ..options.clone()
        },
        &limits,
    )?;
    Ok(manager.run(&graph)?.dump_text())
}

pub fn check(path: &Path) -> Result<String, CliError> {
    check_with(path, &Dims::new())
}

pub fn check_with(path: &Path, dims: &Dims) -> Result<String, CliError> {
    let limits = Limits::default();
    let graph = load_path(path, &limits, dims)?;
    let nodes = graph
        .values()
        .iter()
        .filter(|v| matches!(v.source, Source::Node { .. }))
        .count();
    Ok(format!(
        "ok: {} input(s), {} node(s), {} output(s)\n",
        graph.inputs().len(),
        nodes,
        graph.outputs().len()
    ))
}

pub fn run(graph_path: &Path, inputs_path: Option<&Path>) -> Result<String, CliError> {
    Ok(run_with(graph_path, inputs_path, &RunOptions::default())?.stdout)
}

/// A failed run: the error, plus whatever IR the user asked to see before it
/// happened. `--dump-ir` matters most exactly when a run fails.
#[derive(Debug)]
pub struct RunFailure {
    pub error: CliError,
    pub stderr: String,
}

impl RunFailure {
    fn bare(error: impl Into<CliError>) -> Self {
        Self {
            error: error.into(),
            stderr: String::new(),
        }
    }
}

impl From<RunFailure> for CliError {
    fn from(failure: RunFailure) -> Self {
        failure.error
    }
}

pub fn run_with(
    graph_path: &Path,
    inputs_path: Option<&Path>,
    options: &RunOptions,
) -> Result<RunOutput, RunFailure> {
    let limits = Limits::default();
    let graph = load_path(graph_path, &limits, &options.dims).map_err(RunFailure::bare)?;
    let manager = pipeline(options, &limits).map_err(RunFailure::bare)?;
    let optimized = manager.run(&graph).map_err(|error| RunFailure {
        error: error.into(),
        // A pass failed: the input stage is the part we can still show.
        stderr: if options.dump_ir {
            format!("== input ==\n{graph}\n")
        } else {
            String::new()
        },
    })?;
    let stderr = optimized.dump_text();
    match execute(optimized.graph, inputs_path, &limits, options.backend) {
        Ok(stdout) => Ok(RunOutput { stdout, stderr }),
        Err(error) => Err(RunFailure { error, stderr }),
    }
}

fn execute(
    graph: Graph,
    inputs_path: Option<&Path>,
    limits: &Limits,
    backend: BackendKind,
) -> Result<String, CliError> {
    let tensors = bind_inputs(&graph, inputs_path, limits)?;
    #[cfg(feature = "wgpu")]
    let wgpu;
    let backend: &dyn Backend = match backend {
        BackendKind::Reference => &ReferenceBackend,
        BackendKind::Cpu => &CpuBackend,
        #[cfg(feature = "wgpu")]
        BackendKind::Wgpu => {
            wgpu = oxfuse_wgpu::WgpuBackend::new()?;
            &wgpu
        }
    };
    let outputs = backend.compile(&graph)?.run(&tensors)?;
    let named = graph.output_names().iter().cloned().zip(outputs).collect();
    render_outputs(named)
}

/// A graph from a file: ONNX when the extension says so (and the build
/// can read it), the JSON format otherwise.
fn load_path(path: &Path, limits: &Limits, dims: &Dims) -> Result<Graph, CliError> {
    let is_onnx = path
        .extension()
        .is_some_and(|e| e.eq_ignore_ascii_case("onnx"));
    if !is_onnx {
        return load_graph(&read_limited(path, limits)?, limits);
    }
    #[cfg(feature = "onnx")]
    {
        let options = oxfuse_onnx::ImportOptions {
            limits: limits.clone(),
            dims: dims.clone(),
        };
        Ok(oxfuse_onnx::import(
            &read_bytes_limited(path, limits)?,
            &options,
        )?)
    }
    #[cfg(not(feature = "onnx"))]
    {
        let _ = (limits, dims);
        Err(CliError::OnnxDisabled {
            path: path.display().to_string(),
        })
    }
}

/// Checks the size from metadata first so an oversized regular file is never
/// read, then reads at most one byte past the limit so pipes, FIFOs and
/// devices (which report length 0) cannot bypass it.
fn read_limited(path: &Path, limits: &Limits) -> Result<String, CliError> {
    let bytes = read_bytes_limited(path, limits)?;
    String::from_utf8(bytes).map_err(|_| CliError::Io {
        path: path.display().to_string(),
        source: std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "stream did not contain valid UTF-8",
        ),
    })
}

#[cfg_attr(not(feature = "onnx"), allow(dead_code))]
fn read_bytes_limited(path: &Path, limits: &Limits) -> Result<Vec<u8>, CliError> {
    let io = |source| CliError::Io {
        path: path.display().to_string(),
        source,
    };
    let limit = limits.max_file_bytes;
    let oversized = |found: usize| {
        CliError::Core(Error::LimitExceeded {
            what: "file bytes",
            found,
            limit,
        })
    };
    let len = fs::metadata(path).map_err(io)?.len();
    if len > limit as u64 {
        return Err(oversized(usize::try_from(len).unwrap_or(usize::MAX)));
    }
    let mut bytes = Vec::new();
    fs::File::open(path)
        .map_err(io)?
        .take(limit as u64 + 1)
        .read_to_end(&mut bytes)
        .map_err(io)?;
    if bytes.len() > limit {
        return Err(oversized(bytes.len()));
    }
    Ok(bytes)
}

fn bind_inputs(
    graph: &Graph,
    path: Option<&Path>,
    limits: &Limits,
) -> Result<Vec<Tensor>, CliError> {
    let ids = graph.inputs();
    let Some(path) = path else {
        if ids.is_empty() {
            return Ok(Vec::new());
        }
        return Err(CliError::MissingInputs { count: ids.len() });
    };
    let text = read_limited(path, limits)?;
    let InputsFile(mut file) =
        serde_json::from_str(&text).map_err(|source| CliError::InputsJson {
            path: path.display().to_string(),
            source,
        })?;
    let mut tensors = Vec::with_capacity(ids.len());
    for &id in ids {
        let name = graph.label(id);
        let entry = file
            .remove(&name)
            .ok_or_else(|| CliError::MissingInput { name: name.clone() })?;
        ensure_finite(&name, &entry.values)?;
        tensors.push(Tensor::named(&name, entry.shape, entry.values)?);
    }
    if let Some(name) = file.into_keys().next() {
        return Err(CliError::UnknownInput { name });
    }
    Ok(tensors)
}

fn ensure_finite(name: &str, values: &[f32]) -> Result<(), Error> {
    if values.iter().all(|v| v.is_finite()) {
        Ok(())
    } else {
        Err(Error::NonFinite {
            name: name.to_owned(),
        })
    }
}

/// JSON cannot represent NaN or infinity and serde_json would write `null`,
/// so a non-finite value is an error instead of silent data loss.
fn render_outputs(outputs: Vec<(String, Tensor)>) -> Result<String, CliError> {
    let mut rendered = Vec::with_capacity(outputs.len());
    for (name, tensor) in outputs {
        ensure_finite(&name, tensor.data())?;
        rendered.push(NamedTensorFile {
            name,
            shape: tensor.shape().to_vec(),
            values: tensor.data().to_vec(),
        });
    }
    let mut text = serde_json::to_string(&OutputsFile { outputs: rendered })
        .expect("serializing finite tensors cannot fail");
    text.push('\n');
    Ok(text)
}

/// Measures the first adapter against the reference and renders the report.
#[cfg(feature = "wgpu")]
pub fn gpu_selftest(downlevel: bool) -> Result<String, CliError> {
    let profile = if downlevel {
        oxfuse_wgpu::LimitsProfile::Downlevel
    } else {
        oxfuse_wgpu::LimitsProfile::Adapter
    };
    let context = Arc::new(oxfuse_wgpu::Context::new(profile)?);
    Ok(oxfuse_wgpu::run_selftest(context)?.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    const GRAPH: &str = r#"{"format_version":1,
        "inputs":[{"name":"x","dtype":"f32","shape":[2]}],
        "nodes":[{"name":"y","op":"Relu","inputs":["x"]}],
        "outputs":["y"]}"#;

    fn test_dir() -> std::path::PathBuf {
        let dir =
            std::env::temp_dir().join(format!("oxfuse-commands-tests-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn temp(name: &str, contents: &str) -> std::path::PathBuf {
        let path = test_dir().join(name);
        fs::write(&path, contents).unwrap();
        path
    }

    #[test]
    fn check_counts_inputs_nodes_and_outputs() {
        let graph = temp("check.json", GRAPH);
        assert_eq!(
            check(&graph).unwrap(),
            "ok: 1 input(s), 1 node(s), 1 output(s)\n"
        );
    }

    #[test]
    fn run_executes_relu() {
        let graph = temp("run.json", GRAPH);
        let inputs = temp(
            "run.inputs.json",
            r#"{"x":{"shape":[2],"values":[-1.0,2.0]}}"#,
        );
        let out = run(&graph, Some(&inputs)).unwrap();
        assert_eq!(
            out,
            "{\"outputs\":[{\"name\":\"y\",\"shape\":[2],\"values\":[0.0,2.0]}]}\n"
        );
    }

    #[test]
    fn run_without_inputs_file_is_rejected_when_graph_has_inputs() {
        let graph = temp("noinputs.json", GRAPH);
        assert!(matches!(
            run(&graph, None),
            Err(CliError::MissingInputs { count: 1 })
        ));
    }

    #[test]
    fn missing_input_entry_is_rejected() {
        let graph = temp("missing.json", GRAPH);
        let inputs = temp("missing.inputs.json", "{}");
        assert!(matches!(
            run(&graph, Some(&inputs)),
            Err(CliError::MissingInput { name }) if name == "x"
        ));
    }

    #[test]
    fn extra_input_entry_is_rejected() {
        let graph = temp("extra.json", GRAPH);
        let inputs = temp(
            "extra.inputs.json",
            r#"{"x":{"shape":[2],"values":[1,2]},"z":{"shape":[1],"values":[1]}}"#,
        );
        assert!(matches!(
            run(&graph, Some(&inputs)),
            Err(CliError::UnknownInput { name }) if name == "z"
        ));
    }

    #[test]
    fn input_with_wrong_data_length_names_the_input() {
        let graph = temp("len.json", GRAPH);
        let inputs = temp("len.inputs.json", r#"{"x":{"shape":[2],"values":[1]}}"#);
        assert!(matches!(
            run(&graph, Some(&inputs)),
            Err(CliError::Core(Error::DataLength { name, .. })) if name == "x"
        ));
    }

    #[test]
    fn input_of_wrong_shape_is_rejected() {
        let graph = temp("shape.json", GRAPH);
        let inputs = temp(
            "shape.inputs.json",
            r#"{"x":{"shape":[3],"values":[1,2,3]}}"#,
        );
        assert!(matches!(
            run(&graph, Some(&inputs)),
            Err(CliError::Core(Error::InputShape { .. }))
        ));
    }

    #[test]
    fn missing_file_is_an_io_error() {
        let err = check(Path::new("/definitely/not/here.json")).unwrap_err();
        assert!(matches!(err, CliError::Io { .. }));
    }

    #[test]
    fn oversized_file_is_rejected_before_reading() {
        let path = temp("big.json", GRAPH);
        let limits = Limits {
            max_file_bytes: 4,
            ..Limits::default()
        };
        assert!(matches!(
            read_limited(&path, &limits),
            Err(CliError::Core(Error::LimitExceeded {
                what: "file bytes",
                ..
            }))
        ));
    }

    #[cfg(unix)]
    #[test]
    fn non_regular_file_cannot_bypass_the_size_limit() {
        use std::io::Write;
        let fifo = test_dir().join("limit.fifo");
        let _ = fs::remove_file(&fifo);
        assert!(
            std::process::Command::new("mkfifo")
                .arg(&fifo)
                .status()
                .unwrap()
                .success()
        );
        let writer_path = fifo.clone();
        let writer = std::thread::spawn(move || {
            if let Ok(mut file) = fs::OpenOptions::new().write(true).open(writer_path) {
                let _ = file.write_all(&vec![b' '; 1 << 20]);
            }
        });
        let limits = Limits {
            max_file_bytes: 64,
            ..Limits::default()
        };
        let result = read_limited(&fifo, &limits);
        writer.join().unwrap();
        assert!(matches!(
            result,
            Err(CliError::Core(Error::LimitExceeded {
                what: "file bytes",
                ..
            }))
        ));
    }

    #[test]
    fn non_finite_output_is_an_error() {
        let nan = Tensor::new(vec![1], vec![f32::NAN]).unwrap();
        assert!(matches!(
            render_outputs(vec![("y".to_owned(), nan)]),
            Err(CliError::Core(Error::NonFinite { name })) if name == "y"
        ));
        let inf = Tensor::new(vec![1], vec![f32::INFINITY]).unwrap();
        assert!(render_outputs(vec![("y".to_owned(), inf)]).is_err());
    }

    #[test]
    fn non_finite_input_is_an_error() {
        // `1e39` overflows f32 when parsed from the inputs file.
        let graph = temp("inf.json", GRAPH);
        let inputs = temp(
            "inf.inputs.json",
            r#"{"x":{"shape":[2],"values":[1e39,1]}}"#,
        );
        let result = run(&graph, Some(&inputs));
        assert!(matches!(
            result,
            Err(CliError::Core(Error::NonFinite { .. }))
        ));
    }

    #[test]
    fn arithmetic_overflow_to_infinity_is_an_error_not_null() {
        let graph = temp(
            "overflow.json",
            r#"{"format_version":1,
            "constants":[{"name":"a","dtype":"f32","shape":[],"values":[3e38]},
                         {"name":"b","dtype":"f32","shape":[],"values":[10]}],
            "nodes":[{"name":"y","op":"Mul","inputs":["a","b"]}],
            "outputs":["y"]}"#,
        );
        assert!(matches!(
            run(&graph, None),
            Err(CliError::Core(Error::NonFinite { name })) if name == "y"
        ));
    }

    #[test]
    fn duplicate_input_names_in_the_inputs_file_are_rejected() {
        let graph = temp("dup.json", GRAPH);
        let inputs = temp(
            "dup.inputs.json",
            r#"{"x":{"shape":[2],"values":[1,2]},"x":{"shape":[2],"values":[3,4]}}"#,
        );
        let err = run(&graph, Some(&inputs)).unwrap_err();
        assert_eq!(err.code(), "E102");
        assert!(err.to_string().contains("duplicate input `x`"), "{err}");
    }

    const FOLDABLE: &str = r#"{"format_version":1,
        "inputs":[{"name":"x","dtype":"f32","shape":[2]}],
        "constants":[{"name":"c","dtype":"f32","shape":[2],"values":[-1,2]}],
        "nodes":[{"name":"r","op":"Relu","inputs":["c"]},
                 {"name":"y","op":"Add","inputs":["x","r"]}],
        "outputs":["y"]}"#;

    #[test]
    fn optimizing_does_not_change_the_output() {
        let graph = temp("fold.json", FOLDABLE);
        let inputs = temp(
            "fold.inputs.json",
            r#"{"x":{"shape":[2],"values":[0.5,0.25]}}"#,
        );
        let optimized = run_with(&graph, Some(&inputs), &RunOptions::default()).unwrap();
        let plain = run_with(
            &graph,
            Some(&inputs),
            &RunOptions {
                optimize: false,
                ..RunOptions::default()
            },
        )
        .unwrap();
        assert_eq!(optimized.stdout, plain.stdout);
        assert_eq!(optimized.stderr, "");
    }

    #[test]
    fn dump_ir_goes_to_stderr_with_one_stage_per_pass() {
        let graph = temp("dump.json", FOLDABLE);
        let inputs = temp(
            "dump.inputs.json",
            r#"{"x":{"shape":[2],"values":[0.5,0.25]}}"#,
        );
        let options = RunOptions {
            dump_ir: true,
            ..RunOptions::default()
        };
        let out = run_with(&graph, Some(&inputs), &options).unwrap();
        assert!(
            out.stderr.starts_with("== input ==\ngraph {\n"),
            "{}",
            out.stderr
        );
        for stage in [
            "after fold (changed 1",
            "after cse (changed 0",
            "after dce (changed 1",
        ] {
            assert!(
                out.stderr.contains(stage),
                "missing `{stage}` in {}",
                out.stderr
            );
        }
        assert!(out.stdout.starts_with("{\"outputs\""));
    }

    #[test]
    fn no_opt_dump_shows_only_the_input() {
        let graph = temp("noopt.json", FOLDABLE);
        let inputs = temp(
            "noopt.inputs.json",
            r#"{"x":{"shape":[2],"values":[0.5,0.25]}}"#,
        );
        let options = RunOptions {
            optimize: false,
            dump_ir: true,
            ..RunOptions::default()
        };
        let out = run_with(&graph, Some(&inputs), &options).unwrap();
        assert!(out.stderr.starts_with("== input =="));
        assert!(!out.stderr.contains("after "), "{}", out.stderr);
    }

    #[test]
    fn disabling_a_pass_removes_its_stage() {
        let graph = temp("disable.json", FOLDABLE);
        let options = RunOptions {
            disabled: vec!["cse".to_owned()],
            ..RunOptions::default()
        };
        let text = inspect(&graph, &options).unwrap();
        assert!(text.contains("after fold") && text.contains("after dce"));
        assert!(!text.contains("after cse"), "{text}");
    }

    #[test]
    fn an_unknown_pass_name_is_rejected_with_the_known_names() {
        let graph = temp("unknown_pass.json", FOLDABLE);
        let options = RunOptions {
            disabled: vec!["nope".to_owned()],
            ..RunOptions::default()
        };
        let err = inspect(&graph, &options).unwrap_err();
        assert_eq!(err.code(), "E106");
        assert_eq!(
            err.to_string(),
            "unknown pass `nope` (known: fold, cse, dce)"
        );
    }

    #[test]
    fn an_unknown_pass_is_rejected_even_with_no_opt() {
        let graph = temp("noopt_typo.json", FOLDABLE);
        let options = RunOptions {
            optimize: false,
            disabled: vec!["fnord".to_owned()],
            ..RunOptions::default()
        };
        assert_eq!(inspect(&graph, &options).unwrap_err().code(), "E106");
    }

    #[test]
    fn inspect_without_optimization_prints_only_the_input_graph() {
        let graph = temp("inspect.json", FOLDABLE);
        let options = RunOptions {
            optimize: false,
            ..RunOptions::default()
        };
        let text = inspect(&graph, &options).unwrap();
        assert!(text.starts_with("== input ==\ngraph {\n"), "{text}");
        assert!(!text.contains("after "), "{text}");
    }

    const OVERFLOWING: &str = r#"{"format_version":1,
        "constants":[{"name":"a","dtype":"f32","shape":[],"values":[3e38]},
                     {"name":"b","dtype":"f32","shape":[],"values":[10]}],
        "nodes":[{"name":"boom","op":"Mul","inputs":["a","b"]}],
        "outputs":["boom"]}"#;

    #[test]
    fn dump_ir_survives_a_failed_run() {
        let graph = temp("fail_dump.json", OVERFLOWING);
        let options = RunOptions {
            dump_ir: true,
            ..RunOptions::default()
        };
        let failure = run_with(&graph, None, &options).unwrap_err();
        assert_eq!(failure.error.code(), "E012");
        assert!(
            failure.stderr.starts_with("== input ==\ngraph {\n"),
            "{}",
            failure.stderr
        );
        assert!(failure.stderr.contains("after fold"), "{}", failure.stderr);
    }

    #[test]
    fn dump_ir_survives_missing_inputs() {
        let graph = temp("missing_inputs_dump.json", FOLDABLE);
        let options = RunOptions {
            dump_ir: true,
            ..RunOptions::default()
        };
        let failure = run_with(&graph, None, &options).unwrap_err();
        assert_eq!(failure.error.code(), "E103");
        assert!(failure.stderr.contains("after dce"), "{}", failure.stderr);
    }

    #[test]
    fn a_failed_run_without_dump_ir_has_no_stderr_text() {
        let graph = temp("fail_quiet.json", OVERFLOWING);
        let failure = run_with(&graph, None, &RunOptions::default()).unwrap_err();
        assert_eq!(failure.stderr, "");
    }

    #[test]
    fn a_regular_file_at_the_limit_reads_and_a_larger_one_reports_its_real_size() {
        let at_limit = Limits {
            max_file_bytes: 64,
            ..Limits::default()
        };
        let exactly = temp("exactly64.json", &" ".repeat(64));
        assert_eq!(read_limited(&exactly, &at_limit).unwrap().len(), 64);
        let large = temp("size200.json", &" ".repeat(200));
        assert!(matches!(
            read_limited(&large, &at_limit),
            Err(CliError::Core(Error::LimitExceeded {
                what: "file bytes",
                found: 200,
                limit: 64
            }))
        ));
    }

    #[cfg(unix)]
    #[test]
    fn a_pipe_carrying_exactly_the_limit_is_accepted() {
        use std::io::Write;
        let fifo = test_dir().join("exact.fifo");
        let _ = fs::remove_file(&fifo);
        assert!(
            std::process::Command::new("mkfifo")
                .arg(&fifo)
                .status()
                .unwrap()
                .success()
        );
        let writer_path = fifo.clone();
        let writer = std::thread::spawn(move || {
            let mut file = fs::OpenOptions::new()
                .write(true)
                .open(writer_path)
                .unwrap();
            file.write_all(&[b' '; 64]).unwrap();
        });
        let limits = Limits {
            max_file_bytes: 64,
            ..Limits::default()
        };
        let text = read_limited(&fifo, &limits).unwrap();
        writer.join().unwrap();
        assert_eq!(text.len(), 64);
    }

    #[test]
    fn an_inputs_file_that_is_not_an_object_says_what_was_expected() {
        let graph = temp("array_inputs.json", GRAPH);
        let inputs = temp("array.inputs.json", "[1,2]");
        let err = run(&graph, Some(&inputs)).unwrap_err();
        assert_eq!(err.code(), "E102");
        assert!(
            err.to_string()
                .contains("expected an object mapping input names to"),
            "{err}"
        );
    }
}
