//! Shared helpers for the device tests. Without an adapter the tests pass
//! vacuously, unless OXFUSE_REQUIRE_ADAPTER is set (CI).
#![allow(dead_code)]

use oxfuse_core::{GraphBuilder, Limits, Op};
use oxfuse_runtime::{Backend, ReferenceBackend, Tensor};
use oxfuse_wgpu::WgpuBackend;
use std::path::PathBuf;
use std::sync::LazyLock;

pub static GPU: LazyLock<Option<WgpuBackend>> = LazyLock::new(|| match WgpuBackend::new() {
    Ok(backend) => {
        eprintln!("adapter: {}", backend.context().info().name);
        Some(backend)
    }
    Err(e) => {
        eprintln!("no adapter, tests skipped: {e}");
        None
    }
});

pub fn available() -> bool {
    let ok = GPU.is_some();
    assert!(
        ok || std::env::var_os("OXFUSE_REQUIRE_ADAPTER").is_none(),
        "OXFUSE_REQUIRE_ADAPTER is set but no adapter exists"
    );
    ok
}

pub fn gpu() -> &'static WgpuBackend {
    GPU.as_ref().expect("checked by available()")
}

pub fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

/// Deterministic values spanning many magnitudes, with both zeros mixed in.
/// `finite_normal` keeps NaN, infinity and subnormals out (the finite tier);
/// otherwise they are injected too.
pub fn data(n: usize, seed: u64, finite_normal: bool) -> Vec<f32> {
    let mut state = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (0..n)
        .map(|i| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let unit = ((state >> 40) as f32) / (1u64 << 24) as f32 - 0.5;
            match (finite_normal, i % 11) {
                (false, 3) => f32::NAN,
                (false, 5) => f32::INFINITY,
                (false, 7) => f32::NEG_INFINITY,
                (false, 9) => f32::from_bits(1 + (state >> 60) as u32),
                (_, 1) => -0.0,
                (_, 2) => 0.0,
                _ => unit * 10f32.powi((state >> 33) as i32 % 7 - 3),
            }
        })
        .collect()
}

pub fn tensor(shape: &[usize], seed: u64, finite_normal: bool) -> Tensor {
    let n = shape.iter().product();
    Tensor::new(shape.to_vec(), data(n, seed, finite_normal)).unwrap()
}

pub fn run_op(backend: &dyn Backend, op: Op, inputs: &[Tensor]) -> Vec<f32> {
    let mut b = GraphBuilder::new(Limits::default());
    let ids: Vec<_> = inputs
        .iter()
        .enumerate()
        .map(|(i, t)| b.input(&format!("in{i}"), t.shape().to_vec()).unwrap())
        .collect();
    let out = b.node("out", op, &ids).unwrap();
    let graph = b.finish(&[out]).unwrap();
    backend.compile(&graph).unwrap().run(inputs).unwrap()[0]
        .data()
        .to_vec()
}

pub fn reference(op: Op, inputs: &[Tensor]) -> Vec<f32> {
    run_op(&ReferenceBackend, op, inputs)
}

pub fn device(op: Op, inputs: &[Tensor]) -> Vec<f32> {
    run_op(gpu(), op, inputs)
}

/// Where measurements for the claims ledger are written.
pub fn measurements_dir() -> PathBuf {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("gpu-measurements");
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// The Sigmoid ulp bound to enforce on the adapter under test.
pub fn sigmoid_bound() -> u32 {
    oxfuse_wgpu::sigmoid_ulp_bound_for(&gpu().context().info().name)
}
