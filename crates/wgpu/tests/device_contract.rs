//! The shared backend contract, run on a real adapter, plus the device-only API.
//! Without an adapter the tests pass vacuously, unless OXFUSE_REQUIRE_ADAPTER is
//! set (CI), in which case a missing adapter fails them.

#[path = "../../runtime/tests/contract/mod.rs"]
mod contract;

mod common;

use common::{GPU, available};
use oxfuse_core::{Error, GraphBuilder, Limits, Op};
use oxfuse_runtime::{Executable, Tensor};

contract::contract_tests!(wgpu, *GPU.as_ref().unwrap(), available());

fn relu_executable(shape: Vec<usize>) -> oxfuse_wgpu::WgpuExecutable {
    let mut b = GraphBuilder::new(Limits::default());
    let x = b.input("x", shape).unwrap();
    let y = b.node("y", Op::Relu, &[x]).unwrap();
    let graph = b.finish(&[y]).unwrap();
    GPU.as_ref().unwrap().compile_device(&graph).unwrap()
}

#[test]
fn upload_download_roundtrip_is_exact() {
    if !available() {
        return;
    }
    let exe = relu_executable(vec![2, 3]);
    let values = vec![
        f32::NAN,
        -0.0,
        f32::from_bits(1),
        f32::INFINITY,
        f32::MIN_POSITIVE,
        -3.5,
    ];
    let tensor = Tensor::new(vec![2, 3], values.clone()).unwrap();
    let device = exe.upload(&tensor).unwrap();
    assert_eq!(device.shape(), &[2, 3]);
    let back = exe.download(&device).unwrap();
    let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    assert_eq!(bits(back.data()), bits(&values));
}

#[test]
fn results_stay_valid_after_the_next_run() {
    if !available() {
        return;
    }
    let exe = relu_executable(vec![3]);
    let first = exe
        .upload(&Tensor::new(vec![3], vec![-1.0, 2.0, -3.0]).unwrap())
        .unwrap();
    let second = exe
        .upload(&Tensor::new(vec![3], vec![4.0, -5.0, 6.0]).unwrap())
        .unwrap();
    let out_first = exe.run_device(&[first]).unwrap();
    let out_second = exe.run_device(&[second]).unwrap();
    assert_eq!(
        exe.download(&out_first[0]).unwrap().data(),
        &[0.0, 2.0, 0.0]
    );
    assert_eq!(
        exe.download(&out_second[0]).unwrap().data(),
        &[4.0, 0.0, 6.0]
    );
}

#[test]
fn run_device_checks_its_inputs_like_run_does() {
    if !available() {
        return;
    }
    let exe = relu_executable(vec![2]);
    assert_eq!(
        exe.run_device(&[]).unwrap_err(),
        Error::InputCount {
            expected: 1,
            found: 0
        }
    );
    let wrong = exe
        .upload(&Tensor::new(vec![3], vec![1.0, 2.0, 3.0]).unwrap())
        .unwrap();
    assert_eq!(
        exe.run_device(&[wrong]).unwrap_err(),
        Error::InputShape {
            name: "x".into(),
            expected: vec![2],
            found: vec![3]
        }
    );
}

#[test]
fn threads_with_their_own_executables_on_one_device_get_their_own_results() {
    if !available() {
        return;
    }
    let handles: Vec<_> = (0..8)
        .map(|t| {
            std::thread::spawn(move || {
                // One executable per thread, all on the one shared device.
                let exe = relu_executable(vec![64]);
                for run in 0..40 {
                    let sign = if (t + run) % 2 == 0 { 1.0 } else { -1.0 };
                    let values: Vec<f32> = (0..64).map(|i| sign * (i as f32 + t as f32)).collect();
                    let want: Vec<f32> = values.iter().map(|v| v.max(0.0)).collect();
                    let out = exe
                        .run(&[Tensor::new(vec![64], values).unwrap()])
                        .unwrap_or_else(|e| panic!("thread {t} run {run}: {e}"));
                    assert_eq!(out[0].data(), want.as_slice(), "thread {t} run {run}");
                }
            })
        })
        .collect();
    for handle in handles {
        handle.join().unwrap();
    }
}

#[test]
fn a_kernel_used_twice_in_one_graph_is_compiled_once_and_works_both_times() {
    if !available() {
        return;
    }
    let mut b = GraphBuilder::new(Limits::default());
    let x = b.input("x", vec![4]).unwrap();
    let r1 = b.node("r1", Op::Relu, &[x]).unwrap();
    let r2 = b.node("r2", Op::Relu, &[r1]).unwrap();
    let a1 = b.node("a1", Op::Add, &[r2, x]).unwrap();
    let a2 = b.node("a2", Op::Add, &[a1, a1]).unwrap();
    let graph = b.finish(&[a2]).unwrap();
    let exe = GPU.as_ref().unwrap().compile_device(&graph).unwrap();
    let out = exe
        .run(&[Tensor::new(vec![4], vec![-1.0, 2.0, -3.0, 4.0]).unwrap()])
        .unwrap();
    // relu(relu(x)) + x, doubled.
    assert_eq!(out[0].data(), &[-2.0, 8.0, -6.0, 16.0]);
}

#[test]
fn transposing_a_scalar_is_a_copy() {
    if !available() {
        return;
    }
    let mut b = GraphBuilder::new(Limits::default());
    let x = b.input("x", vec![]).unwrap();
    let t = b.node("t", Op::Transpose { perm: vec![] }, &[x]).unwrap();
    let graph = b.finish(&[t]).unwrap();
    let out = GPU
        .as_ref()
        .unwrap()
        .compile_device(&graph)
        .unwrap()
        .run(&[Tensor::new(vec![], vec![-2.5]).unwrap()])
        .unwrap();
    assert_eq!(out[0].shape(), &[] as &[usize]);
    assert_eq!(out[0].data(), &[-2.5]);
}
