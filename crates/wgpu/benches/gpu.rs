//! The GPU backend against the fast CPU backend on the same work, in four ways:
//!
//! - `gpu-cold`: compile (pipelines, buffers) plus the first run, one fresh
//!   executable per sample. A driver may cache compiled shaders across samples,
//!   so this is a lower bound on a true first run in a new process.
//! - `gpu-resident`: inputs already on the device; times `run_device` until the
//!   device is idle, including the copy of the outputs into fresh buffers.
//! - `gpu-with-transfers`: host tensors in and out, what `run` costs a caller.
//! - `cpu` (one thread) and `cpu-threads`: the fast CPU backend, in the same run.
//!
//! Only the machine that ran it can be described by the numbers.

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use oxfuse_core::{Graph, GraphBuilder, Limits, Op};
use oxfuse_runtime::{Backend, CpuBackend, Executable, Tensor};
use oxfuse_wgpu::{DeviceTensor, WgpuBackend};
use std::hint::black_box;
use std::time::{Duration, Instant};

fn data(len: usize, seed: u32) -> Vec<f32> {
    let mut state = seed.wrapping_mul(2_654_435_761).wrapping_add(1);
    (0..len)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (state >> 8) as f32 / (1u32 << 23) as f32 - 1.0
        })
        .collect()
}

fn tensor(shape: &[usize], seed: u32) -> Tensor {
    Tensor::new(shape.to_vec(), data(shape.iter().product(), seed)).unwrap()
}

fn graph(op: Op, shapes: &[&[usize]]) -> Graph {
    let mut b = GraphBuilder::new(Limits::default());
    let ids: Vec<_> = shapes
        .iter()
        .enumerate()
        .map(|(i, s)| b.input(&format!("in{i}"), s.to_vec()).unwrap())
        .collect();
    let out = b.node("out", op, &ids).unwrap();
    b.finish(&[out]).unwrap()
}

fn wait(gpu: &WgpuBackend) {
    gpu.context()
        .device()
        .poll(wgpu_wait())
        .expect("the device finishes");
}

fn wgpu_wait() -> oxfuse_wgpu::PollType {
    oxfuse_wgpu::PollType::wait_indefinitely()
}

fn compare(
    c: &mut Criterion,
    group_name: &str,
    size: usize,
    graph: &Graph,
    inputs: &[Tensor],
    flops: u64,
) {
    let Ok(gpu) = WgpuBackend::new() else {
        eprintln!("no adapter: GPU benchmarks skipped");
        return;
    };
    let mut group = c.benchmark_group(group_name);
    group.throughput(Throughput::Elements(flops));
    group
        .sample_size(20)
        .measurement_time(Duration::from_secs(3));

    group.bench_with_input(BenchmarkId::new("gpu-cold", size), &size, |b, _| {
        b.iter(|| {
            let exe = gpu.compile_device(graph).unwrap();
            let device_inputs: Vec<DeviceTensor> =
                inputs.iter().map(|t| exe.upload(t).unwrap()).collect();
            black_box(exe.run_device(&device_inputs).unwrap());
            wait(&gpu);
        })
    });

    let exe = gpu.compile_device(graph).unwrap();
    let device_inputs: Vec<DeviceTensor> = inputs.iter().map(|t| exe.upload(t).unwrap()).collect();
    group.bench_with_input(BenchmarkId::new("gpu-resident", size), &size, |b, _| {
        b.iter(|| {
            black_box(exe.run_device(&device_inputs).unwrap());
            wait(&gpu);
        })
    });
    group.bench_with_input(
        BenchmarkId::new("gpu-with-transfers", size),
        &size,
        |b, _| b.iter(|| black_box(exe.run(inputs).unwrap())),
    );

    let cpu = CpuBackend.compile(graph).unwrap();
    let one_thread = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap();
    group.bench_with_input(BenchmarkId::new("cpu", size), &size, |b, _| {
        // Compiled and timed inside the one-thread pool, so rayon runs on it.
        b.iter_custom(|iters| {
            one_thread.install(|| {
                let cpu = CpuBackend.compile(graph).unwrap();
                let start = Instant::now();
                for _ in 0..iters {
                    black_box(cpu.run(inputs).unwrap());
                }
                start.elapsed()
            })
        })
    });
    group.bench_with_input(BenchmarkId::new("cpu-threads", size), &size, |b, _| {
        b.iter(|| black_box(cpu.run(inputs).unwrap()))
    });
    group.finish();
}

fn record_adapter() {
    let Ok(gpu) = WgpuBackend::new() else { return };
    let info = gpu.context().info();
    let line = format!("{} ({:?}), wgpu {}\n", info.name, info.backend, "30.0.1");
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/criterion");
    let _ = std::fs::create_dir_all(&dir);
    let _ = std::fs::write(dir.join("gpu-adapter.txt"), line);
}

fn benches(c: &mut Criterion) {
    record_adapter();
    for n in [128usize, 256, 512, 1024, 2048] {
        let g = graph(Op::MatMul, &[&[n, n], &[n, n]]);
        let inputs = [tensor(&[n, n], 1), tensor(&[n, n], 2)];
        compare(c, "gpu-matmul", n, &g, &inputs, 2 * (n as u64).pow(3));
    }
    for n in [1usize << 20, 1 << 24] {
        let g = graph(Op::Add, &[&[n], &[n]]);
        let inputs = [tensor(&[n], 3), tensor(&[n], 4)];
        compare(c, "gpu-add", n, &g, &inputs, n as u64);
    }
}

criterion_group!(gpu, benches);
criterion_main!(gpu);
