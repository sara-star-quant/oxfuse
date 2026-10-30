//! Shared by the fuzz targets: the limits they run under, the comparison
//! that checks the backends and the passes against each other, and a
//! generator of valid graph files.

pub mod gpu;
pub mod onnx;

use arbitrary::{Result as ArbResult, Unstructured};
use oxfuse_core::passes::default_pipeline;
use oxfuse_core::{Graph, Limits};
use oxfuse_runtime::{Backend, CpuBackend, ReferenceBackend, ReferenceEvaluator, Tensor};
use std::sync::Arc;

/// Small limits keep each run fast and bound memory.
pub fn small() -> Limits {
    Limits::default()
        .with_max_file_bytes(1 << 20)
        .with_max_values(256)
        .with_max_tensor_elements(1 << 13)
        .with_max_total_elements(1 << 17)
        .with_max_rank(8)
}

/// Values that exercise signs, zeros, magnitudes, infinity and NaN. Inputs are
/// bound in Rust, so non-finite values are allowed here even though a JSON
/// file cannot hold them.
/// The bits of `v`, with every NaN mapped to one pattern. IEEE 754 leaves the
/// sign and payload of a NaN made from two NaN operands, or from an invalid
/// operation, to the hardware (x86 and ARM differ) and to the order the
/// compiler gives the operands, so two correct kernels may disagree on them.
pub fn comparable_bits(v: f32) -> u32 {
    if v.is_nan() { f32::NAN.to_bits() } else { v.to_bits() }
}

pub(crate) const PALETTE: [f32; 11] = [
    0.0,
    -0.0,
    1.0,
    -1.5,
    1e30,
    1e-30,
    3.25,
    -7.0,
    f32::INFINITY,
    f32::NEG_INFINITY,
    f32::NAN,
];

/// Finite constants for generated JSON.
const FINITE: [&str; 8] = ["0.0", "-0.0", "1.0", "-1.5", "1e30", "1e-30", "3.25", "-7.0"];

/// Outputs as shapes and bit patterns, so NaN and signed zero compare exactly.
type Bits = Vec<(Vec<usize>, Vec<u32>)>;

fn run(backend: &dyn Backend, graph: &Graph, inputs: &[Tensor]) -> Result<Bits, &'static str> {
    let outputs = backend
        .compile(graph)
        .and_then(|executable| executable.run(inputs))
        .map_err(|error| error.code())?;
    Ok(outputs
        .iter()
        .map(|t| {
            (
                t.shape().to_vec(),
                t.data().iter().map(|&v| comparable_bits(v)).collect(),
            )
        })
        .collect())
}

/// Why `text` does not load under the fuzz limits, if it does not.
pub fn load_error(text: &str) -> Option<String> {
    oxfuse_cli::load_graph(text, &small()).err().map(|e| e.to_string())
}

/// Loads `text` and, if it is a valid graph, checks that the fast backend
/// matches the reference, and that the optimized graph matches the original.
/// Returns whether a graph was actually compared.
pub fn compare(text: &str, seed: usize) -> bool {
    let Ok(graph) = oxfuse_cli::load_graph(text, &small()) else {
        return false;
    };
    compare_graph(&graph, seed)
}

/// The checks of `compare` on a graph that is already loaded.
pub fn compare_graph(graph: &Graph, seed: usize) -> bool {
    let limits = small();
    graph
        .verify_limits(&limits)
        .expect("a loaded graph stays inside the limits");
    let inputs: Vec<Tensor> = graph
        .inputs()
        .iter()
        .enumerate()
        .map(|(k, &id)| {
            let shape = graph.value(id).ty.shape.clone();
            let count: usize = shape.iter().product();
            let values = (0..count)
                .map(|i| PALETTE[seed.wrapping_add(k * 3 + i * 5) % PALETTE.len()])
                .collect();
            Tensor::new(shape, values).expect("shape and data lengths agree")
        })
        .collect();

    let reference = run(&ReferenceBackend, graph, &inputs);
    let fast = run(&CpuBackend, graph, &inputs);
    assert_eq!(reference, fast, "fast backend differs from the reference");

    // A pass error can legitimately replace a run-time error, so it ends the
    // comparison instead of failing it.
    let Ok(optimized) = default_pipeline(limits, Arc::new(ReferenceEvaluator)).run(graph) else {
        return true;
    };
    let optimized = optimized.graph;
    let reference_optimized = run(&ReferenceBackend, &optimized, &inputs);
    let fast_optimized = run(&CpuBackend, &optimized, &inputs);
    assert_eq!(
        reference_optimized, fast_optimized,
        "fast backend differs from the reference on the optimized graph"
    );
    assert_eq!(
        reference.is_ok(),
        reference_optimized.is_ok(),
        "optimization changed whether the run succeeds"
    );
    if let (Ok(a), Ok(b)) = (&reference, &reference_optimized) {
        assert_eq!(a, b, "optimization changed a result");
    }
    true
}

/// A valid graph file: one rank-2 input and a chain of ops over it, with
/// sizes that reach the fast matmul's register block. Valid by construction,
/// so nearly every execution reaches `compare`.
pub fn generate(u: &mut Unstructured) -> ArbResult<String> {
    let (mut r, mut c) = (u.int_in_range(1..=24usize)?, u.int_in_range(1..=24usize)?);
    let input = format!(r#"{{"name":"x","dtype":"f32","shape":[{r},{c}]}}"#);
    let mut constants: Vec<String> = Vec::new();
    let mut nodes: Vec<String> = Vec::new();
    let mut last = String::from("x");
    let mut names = vec![last.clone()];
    for i in 0..u.int_in_range(1..=8usize)? {
        let name = format!("n{i}");
        let node = |op: &str, inputs: &[&str], attrs: &str| {
            let list: Vec<String> = inputs.iter().map(|s| format!("\"{s}\"")).collect();
            format!(
                r#"{{"name":"{name}","op":"{op}","inputs":[{}]{attrs}}}"#,
                list.join(",")
            )
        };
        let mut constant = |shape: &[usize], u: &mut Unstructured| -> ArbResult<String> {
            let count: usize = shape.iter().product();
            let mut values = Vec::new();
            for _ in 0..count {
                values.push(*u.choose(&FINITE)?);
            }
            let cname = format!("k{}", constants.len());
            let dims: Vec<String> = shape.iter().map(usize::to_string).collect();
            constants.push(format!(
                r#"{{"name":"{cname}","dtype":"f32","shape":[{}],"values":[{}]}}"#,
                dims.join(","),
                values.join(",")
            ));
            Ok(cname)
        };
        let n = match u.int_in_range(0..=7u8)? {
            0 => node("Relu", &[&last], ""),
            1 => node("Sigmoid", &[&last], ""),
            2 | 3 => {
                let op = if u.arbitrary()? { "Add" } else { "Mul" };
                let other = if u.arbitrary()? {
                    constant(&[], u)?
                } else {
                    constant(&[r, c], u)?
                };
                node(op, &[&last, &other], "")
            }
            4 => {
                if u.arbitrary()? {
                    let n2 = u.int_in_range(1..=24usize.min(4096 / c).min(4096 / r))?;
                    let w = constant(&[c, n2], u)?;
                    c = n2;
                    node("MatMul", &[&last, &w], "")
                } else {
                    // The running value on the right, so infinity and NaN
                    // from the input meet the constant's zeros.
                    let m2 = u.int_in_range(1..=24usize.min(4096 / r).min(4096 / c))?;
                    let w = constant(&[m2, r], u)?;
                    r = m2;
                    node("MatMul", &[&w, &last], "")
                }
            }
            5 => {
                std::mem::swap(&mut r, &mut c);
                node("Transpose", &[&last], r#","attrs":{"perm":[1,0]}"#)
            }
            6 => {
                let total = r * c;
                let (nr, nc) = if u.arbitrary()? { (total, 1) } else { (1, total) };
                (r, c) = (nr, nc);
                node("Reshape", &[&last], &format!(r#","attrs":{{"shape":[{r},{c}]}}"#))
            }
            _ => {
                // Reuse an earlier value, so CSE and DCE have something to do.
                let earlier = u.choose(&names)?.clone();
                if earlier == last {
                    node("Add", &[&last, &last], "")
                } else {
                    // Shapes may differ, so fall back to a safe unary op.
                    node("Relu", &[&last], "")
                }
            }
        };
        nodes.push(n);
        last = name;
        names.push(last.clone());
    }
    Ok(format!(
        r#"{{"format_version":1,"inputs":[{input}],"constants":[{}],"nodes":[{}],"outputs":["{last}"]}}"#,
        constants.join(","),
        nodes.join(",")
    ))
}
