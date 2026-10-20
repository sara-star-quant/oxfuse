//! Op-level differential against a GPU adapter. A whole-graph bit comparison
//! cannot work on a device (a fused multiply-add changes `MatMul` results, and
//! the difference compounds downstream), so each node is checked on its own:
//! the reference computes every value, then each op runs on the device with the
//! reference's actual inputs and is held to its documented contract.

use oxfuse_core::passes::ConstEvaluator;
use oxfuse_core::{GraphBuilder, Limits, Op, Source, TensorType};
use oxfuse_runtime::{Backend, ReferenceEvaluator, Tensor};
use oxfuse_wgpu::{WgpuBackend, gpu_matmul_bound, sigmoid_matches, sigmoid_ulp_bound_for};
use std::collections::HashMap;
use std::sync::LazyLock;

static GPU: LazyLock<Option<WgpuBackend>> = LazyLock::new(|| WgpuBackend::new().ok());

/// The Sigmoid ulp bound to enforce on this machine's adapter.
fn sigmoid_bound() -> u32 {
    GPU.as_ref()
        .map_or(u32::MAX, |g| sigmoid_ulp_bound_for(&g.context().info().name))
}

/// Whether this machine has an adapter. Without one the target does nothing.
pub fn available() -> bool {
    GPU.is_some()
}

/// The finite tier: finite, and zero or a normal number (the device may flush subnormals).
fn in_tier(x: f32) -> bool {
    x.is_finite() && (x == 0.0 || x.is_normal())
}

/// Whether the device result `got` honours the contract against the reference
/// `want` for one op. Data movement and Relu must match bit for bit on every
/// input; Add and Mul on the finite tier; MatMul within the documented bound of
/// the f64 oracle (plus one smallest-normal of flush slack per term); Sigmoid
/// within the measured ulp bound on finite inputs.
pub fn op_agrees(
    op: &Op,
    inputs: &[(&[usize], &[f32])],
    got: &[f32],
    want: &[f32],
) -> Result<(), String> {
    if got.len() != want.len() {
        return Err(format!(
            "length {} but the reference has {}",
            got.len(),
            want.len()
        ));
    }
    let bit_mismatch = |only_where: &dyn Fn(usize) -> bool| {
        (0..got.len())
            .find(|&i| only_where(i) && got[i].to_bits() != want[i].to_bits())
            .map(|i| format!("element {i}: device {:e}, reference {:e}", got[i], want[i]))
    };
    let mismatch = match op {
        Op::Relu | Op::Reshape { .. } | Op::Transpose { .. } => bit_mismatch(&|_| true),
        Op::Add | Op::Mul => {
            let all_in_tier = inputs.iter().all(|(_, d)| d.iter().all(|&x| in_tier(x)));
            if all_in_tier {
                bit_mismatch(&|i| in_tier(want[i]))
            } else {
                None
            }
        }
        Op::Sigmoid => (0..got.len()).find_map(|i| {
            let x = inputs[0].1[i];
            (x.is_finite() && !sigmoid_matches(got[i], want[i], sigmoid_bound())).then(|| {
                format!(
                    "sigmoid({x:e}): device {:e}, reference {:e}",
                    got[i], want[i]
                )
            })
        }),
        Op::MatMul => {
            let ((a_shape, a), (b_shape, b)) = (inputs[0], inputs[1]);
            let (m, k, n) = (a_shape[0], a_shape[1], b_shape[1]);
            if !inputs.iter().all(|(_, d)| d.iter().all(|&x| in_tier(x))) {
                None
            } else {
                (0..m * n).find_map(|idx| {
                    let (i, j) = (idx / n, idx % n);
                    let row: Vec<f32> = (0..k).map(|x| a[i * k + x]).collect();
                    let col: Vec<f32> = (0..k).map(|x| b[x * n + j]).collect();
                    let exact: f64 = row.iter().zip(&col).map(|(x, y)| f64::from(*x) * f64::from(*y)).sum();
                    if exact.abs() > 1e37 {
                        return None; // near overflow: not promised
                    }
                    let tolerance = gpu_matmul_bound(&row, &col);
                    let err = (f64::from(got[idx]) - exact).abs();
                    (err.is_nan() || err > tolerance).then(|| {
                        format!("matmul element {idx}: device {:e}, exact {exact:e}, tolerance {tolerance:e}", got[idx])
                    })
                })
            }
        }
    };
    mismatch.map_or(Ok(()), Err)
}

fn run_on_device(op: &Op, inputs: &[(&TensorType, &[f32])]) -> Vec<f32> {
    let gpu = GPU.as_ref().expect("checked by available()");
    let mut b = GraphBuilder::new(Limits::default());
    let ids: Vec<_> = inputs
        .iter()
        .enumerate()
        .map(|(n, (ty, _))| {
            b.input(&format!("i{n}"), ty.shape.clone())
                .expect("valid input")
        })
        .collect();
    let out = b
        .node("o", op.clone(), &ids)
        .expect("the node was valid in the loaded graph");
    let graph = b.finish(&[out]).expect("a one-op graph is valid");
    let tensors: Vec<Tensor> = inputs
        .iter()
        .map(|(ty, data)| {
            Tensor::new(ty.shape.clone(), data.to_vec()).expect("shape and data agree")
        })
        .collect();
    let outputs = gpu
        .compile(&graph)
        .and_then(|executable| executable.run(&tensors))
        .unwrap_or_else(|e| panic!("the device failed on a valid {op:?}: {e}"));
    outputs[0].data().to_vec()
}

/// Loads `text` and, if it is a valid graph, checks every node on the device
/// against the reference values. Returns whether a graph was compared.
pub fn gpu_compare(text: &str, seed: usize) -> bool {
    let limits = super::small();
    let Ok(graph) = oxfuse_cli::load_graph(text, &limits) else {
        return false;
    };
    let mut input_data: HashMap<usize, Vec<f32>> = HashMap::new();
    for (k, &id) in graph.inputs().iter().enumerate() {
        let count: usize = graph.value(id).ty.shape.iter().product();
        let values = (0..count)
            .map(|i| super::PALETTE[seed.wrapping_add(k * 3 + i * 5) % super::PALETTE.len()])
            .collect();
        input_data.insert(id.index(), values);
    }
    let mut computed: Vec<Vec<f32>> = Vec::with_capacity(graph.values().len());
    for (index, value) in graph.values().iter().enumerate() {
        let data = match &value.source {
            Source::Input => input_data.remove(&index).expect("every input has data"),
            Source::Constant(data) => data.to_vec(),
            Source::Node { op, inputs } => {
                let ins: Vec<(&TensorType, &[f32])> = inputs
                    .iter()
                    .map(|id| (&graph.value(*id).ty, computed[id.index()].as_slice()))
                    .collect();
                let want = ReferenceEvaluator
                    .eval(op, &ins)
                    .expect("the reference evaluates a verified node");
                let got = run_on_device(op, &ins);
                let shapes: Vec<(&[usize], &[f32])> = ins
                    .iter()
                    .map(|(ty, d)| (ty.shape.as_slice(), *d))
                    .collect();
                if let Err(reason) = op_agrees(op, &shapes, &got, &want) {
                    panic!("{op:?} on the device: {reason}");
                }
                want
            }
        };
        computed.push(data);
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    fn agrees(op: Op, inputs: &[(&[usize], &[f32])], got: &[f32], want: &[f32]) -> bool {
        op_agrees(&op, inputs, got, want).is_ok()
    }

    #[test]
    fn relu_must_match_bit_for_bit_even_for_nan() {
        let x: &[f32] = &[f32::NAN, -1.0, 2.0];
        let want = [f32::NAN, 0.0, 2.0];
        assert!(agrees(Op::Relu, &[(&[3], x)], &want, &want));
        assert!(!agrees(Op::Relu, &[(&[3], x)], &[0.0, 0.0, 2.0], &want));
        assert!(!agrees(
            Op::Relu,
            &[(&[3], x)],
            &[f32::NAN, -0.0, 2.0],
            &want
        ));
    }

    #[test]
    fn add_is_exact_on_the_finite_tier_and_unchecked_outside_it() {
        let a: &[f32] = &[1.0, 2.0];
        let b: &[f32] = &[3.0, 4.0];
        let ins: [(&[usize], &[f32]); 2] = [(&[2], a), (&[2], b)];
        assert!(agrees(Op::Add, &ins, &[4.0, 6.0], &[4.0, 6.0]));
        assert!(!agrees(Op::Add, &ins, &[4.0, 6.5], &[4.0, 6.0]));
        // A subnormal input leaves the tier: a flushed result is not a failure.
        let sub: &[f32] = &[f32::from_bits(1), 2.0];
        let ins: [(&[usize], &[f32]); 2] = [(&[2], sub), (&[2], b)];
        assert!(agrees(
            Op::Add,
            &ins,
            &[0.0, 6.0],
            &[f32::from_bits(1), 6.0]
        ));
        // NaN input: unchecked.
        let nan: &[f32] = &[f32::NAN, 2.0];
        let ins: [(&[usize], &[f32]); 2] = [(&[2], nan), (&[2], b)];
        assert!(agrees(Op::Add, &ins, &[1.0, 6.0], &[f32::NAN, 6.0]));
    }

    #[test]
    fn a_subnormal_reference_result_may_flush() {
        let a: &[f32] = &[1e-30];
        let b: &[f32] = &[1e-10];
        let ins: [(&[usize], &[f32]); 2] = [(&[1], a), (&[1], b)];
        let want = [a[0] * b[0]];
        assert!(
            want[0] != 0.0 && !want[0].is_normal(),
            "the case must be a subnormal result"
        );
        assert!(agrees(Op::Mul, &ins, &[0.0], &want));
    }

    #[test]
    fn matmul_is_held_to_the_oracle_bound() {
        let a: &[f32] = &[1.0, 2.0, 3.0, 4.0];
        let b: &[f32] = &[5.0, 6.0, 7.0, 8.0];
        let ins: [(&[usize], &[f32]); 2] = [(&[2, 2], a), (&[2, 2], b)];
        let exact = [19.0, 22.0, 43.0, 50.0];
        assert!(agrees(Op::MatMul, &ins, &exact, &exact));
        let off_by_an_ulp = [19.0f32.next_up(), 22.0, 43.0, 50.0];
        assert!(agrees(Op::MatMul, &ins, &off_by_an_ulp, &exact));
        assert!(!agrees(Op::MatMul, &ins, &[19.5, 22.0, 43.0, 50.0], &exact));
        assert!(!agrees(
            Op::MatMul,
            &ins,
            &[f32::NAN, 22.0, 43.0, 50.0],
            &exact
        ));
    }

    #[test]
    fn sigmoid_is_held_to_the_measured_ulp_bound() {
        let x: &[f32] = &[0.0, 1.0];
        let want = [0.5, 0.731_058_6];
        assert!(agrees(Op::Sigmoid, &[(&[2], x)], &want, &want));
        assert!(!agrees(Op::Sigmoid, &[(&[2], x)], &[0.5, 0.74], &want));
        let inf: &[f32] = &[f32::INFINITY, 1.0];
        assert!(agrees(
            Op::Sigmoid,
            &[(&[2], inf)],
            &[f32::NAN, 0.731_058_6],
            &[1.0, 0.731_058_6]
        ));
    }

    #[test]
    fn data_movement_ops_must_match_exactly() {
        let x: &[f32] = &[1.0, f32::NAN, -0.0, 4.0];
        assert!(agrees(
            Op::Reshape { shape: vec![4] },
            &[(&[2, 2], x)],
            x,
            x
        ));
        let swapped = [1.0, -0.0, f32::NAN, 4.0];
        assert!(agrees(
            Op::Transpose { perm: vec![1, 0] },
            &[(&[2, 2], x)],
            &swapped,
            &swapped
        ));
        assert!(!agrees(
            Op::Transpose { perm: vec![1, 0] },
            &[(&[2, 2], x)],
            x,
            &swapped
        ));
    }

    #[test]
    fn a_small_graph_runs_end_to_end_when_an_adapter_exists() {
        if !available() {
            return;
        }
        let text = r#"{"format_version":1,"inputs":[{"name":"x","dtype":"f32","shape":[2,3]}],
            "constants":[{"name":"w","dtype":"f32","shape":[3,2],"values":[1.0,0.0,0.0,1.0,1.0,1.0]}],
            "nodes":[{"name":"p","op":"MatMul","inputs":["x","w"]},
                     {"name":"r","op":"Relu","inputs":["p"]},
                     {"name":"s","op":"Sigmoid","inputs":["r"]},
                     {"name":"t","op":"Transpose","inputs":["s"],"attrs":{"perm":[1,0]}}],
            "outputs":["t"]}"#;
        assert!(gpu_compare(text, 3));
    }
}
