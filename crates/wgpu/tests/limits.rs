//! The backend under the smallest guaranteed device limits, with tensors too
//! large for the device, past the dispatch fold, and under repetition and threads.

mod common;

use common::{available, bits, data, reference, run_op, tensor};
use oxfuse_core::{Error, GraphBuilder, Limits, Op};
use oxfuse_runtime::{Backend, Executable, Tensor};
use oxfuse_wgpu::{Context, DeviceCaps, KernelConfig, LimitsProfile, WgpuBackend, matmul_bound};
use std::sync::Arc;

fn downlevel() -> Option<WgpuBackend> {
    Context::new(LimitsProfile::Downlevel)
        .ok()
        .map(|ctx| WgpuBackend::with(Arc::new(ctx), KernelConfig::DEFAULT))
}

#[test]
fn downlevel_limits_run_every_op() {
    if !available() {
        return;
    }
    let backend = downlevel().expect("an adapter exists, so a downlevel device does too");
    assert_eq!(
        backend
            .context()
            .limits()
            .max_compute_invocations_per_workgroup,
        256
    );
    let a = tensor(&[33, 17], 1, true);
    let b = tensor(&[17, 33], 2, true);
    let s = tensor(&[], 3, true);
    let cases: Vec<(Op, Vec<Tensor>)> = vec![
        (Op::Relu, vec![tensor(&[100], 4, false)]),
        (Op::Add, vec![a.clone(), a.clone()]),
        (Op::Mul, vec![a.clone(), s.clone()]),
        (Op::Transpose { perm: vec![1, 0] }, vec![a.clone()]),
        (
            Op::Transpose {
                perm: vec![2, 0, 1],
            },
            vec![tensor(&[3, 4, 5], 5, false)],
        ),
        (
            Op::Reshape {
                shape: vec![17, 33],
            },
            vec![a.clone()],
        ),
    ];
    for (op, inputs) in cases {
        assert_eq!(
            bits(&run_op(&backend, op.clone(), &inputs)),
            bits(&reference(op.clone(), &inputs)),
            "{op:?}"
        );
    }
    let got = run_op(&backend, Op::MatMul, &[a.clone(), b.clone()]);
    for i in 0..33 {
        let row: Vec<f32> = (0..17).map(|x| a.data()[i * 17 + x]).collect();
        for j in 0..33 {
            let col: Vec<f32> = (0..17).map(|x| b.data()[x * 33 + j]).collect();
            let exact: f64 = row
                .iter()
                .zip(&col)
                .map(|(x, y)| f64::from(*x) * f64::from(*y))
                .sum();
            assert!((f64::from(got[i * 33 + j]) - exact).abs() <= matmul_bound(&row, &col));
        }
    }
}

#[test]
fn oversize_tensor_is_a_compile_error() {
    if !available() {
        return;
    }
    let backend = downlevel().unwrap();
    let cap = (DeviceCaps::DOWNLEVEL.max_binding_bytes / 4) as usize;
    let mut b = GraphBuilder::new(Limits::default());
    let x = b.input("x", vec![cap + 1]).unwrap();
    let y = b.node("y", Op::Relu, &[x]).unwrap();
    let graph = b.finish(&[y]).unwrap();
    let message = match backend.compile(&graph) {
        Err(Error::Backend(m)) => m,
        Err(other) => panic!("wrong error: {other}"),
        Ok(_) => panic!("a tensor over the device limit must not compile"),
    };
    assert!(message.contains(&cap.to_string()), "{message}");
    assert!(message.contains("storage buffer"), "{message}");
}

#[test]
fn a_tensor_exactly_at_the_cap_compiles() {
    if !available() {
        return;
    }
    let backend = downlevel().unwrap();
    let cap = (DeviceCaps::DOWNLEVEL.max_binding_bytes / 4) as usize;
    let mut b = GraphBuilder::new(Limits::default());
    let x = b.input("x", vec![cap]).unwrap();
    let y = b.node("y", Op::Relu, &[x]).unwrap();
    let graph = b.finish(&[y]).unwrap();
    assert!(backend.compile(&graph).is_ok());
}

#[test]
fn huge_elementwise_crosses_the_dispatch_fold() {
    if !available() {
        return;
    }
    // One element more than a single row of 65535 workgroups of 256 threads covers.
    let n = 65535 * 256 + 1;
    let values: Vec<f32> = (0..n)
        .map(|i| if i % 3 == 0 { -(i as f32) } else { i as f32 })
        .collect();
    let t = Tensor::new(vec![n], values.clone()).unwrap();
    for backend in [common::gpu(), &downlevel().unwrap()] {
        let out = run_op(backend, Op::Relu, std::slice::from_ref(&t));
        assert_eq!(out.len(), n);
        for i in (0..n)
            .step_by(97)
            .chain([0, 1, n - 2, n - 1, 65535 * 256 - 1, 65535 * 256])
        {
            let want = if values[i] > 0.0 { values[i] } else { 0.0 };
            assert_eq!(out[i].to_bits(), want.to_bits(), "element {i}");
        }
    }
}

fn matmul_relu_graph() -> oxfuse_core::Graph {
    let mut b = GraphBuilder::new(Limits::default());
    let x = b.input("x", vec![40, 24]).unwrap();
    let w = b
        .constant("w", vec![24, 40], data(24 * 40, 9, true))
        .unwrap();
    let p = b.node("p", Op::MatMul, &[x, w]).unwrap();
    let r = b.node("r", Op::Relu, &[p]).unwrap();
    let s = b.node("s", Op::Sigmoid, &[r]).unwrap();
    b.finish(&[s]).unwrap()
}

#[test]
fn repeat_runs_agree() {
    if !available() {
        return;
    }
    let exe = common::gpu().compile(&matmul_relu_graph()).unwrap();
    let input = tensor(&[40, 24], 11, true);
    let first = bits(exe.run(std::slice::from_ref(&input)).unwrap()[0].data());
    for run in 1..50 {
        let again = bits(exe.run(std::slice::from_ref(&input)).unwrap()[0].data());
        assert_eq!(again, first, "run {run} differs");
    }
}

#[test]
fn concurrent_runs_agree() {
    if !available() {
        return;
    }
    let exe = Arc::new(common::gpu().compile_device(&matmul_relu_graph()).unwrap());
    let input = tensor(&[40, 24], 12, true);
    let expected = bits(exe.run(std::slice::from_ref(&input)).unwrap()[0].data());
    let handles: Vec<_> = (0..4)
        .map(|t| {
            let (exe, input, expected) = (Arc::clone(&exe), input.clone(), expected.clone());
            std::thread::spawn(move || {
                for run in 0..20 {
                    let out = exe.run(std::slice::from_ref(&input)).unwrap();
                    assert_eq!(bits(out[0].data()), expected, "thread {t} run {run}");
                }
            })
        })
        .collect();
    for h in handles {
        h.join().unwrap();
    }
}
