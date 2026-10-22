//! Compare the reference backend, the fast backend and ndarray's `dot` (safe
//! code over `matrixmultiply`) on the same work. Backends are compiled once;
//! only `run` is timed, so the numbers include the executor's buffer copies,
//! which is what a user pays.

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use ndarray::Array2;
use oxfuse_core::passes::default_pipeline;
use oxfuse_core::{Graph, GraphBuilder, Limits, Op};
use oxfuse_runtime::{Backend, CpuBackend, ReferenceBackend, ReferenceEvaluator, Tensor};
use std::hint::black_box;
use std::sync::Arc;

const CPU: &str = if cfg!(feature = "parallel") {
    "cpu-threads"
} else {
    "cpu"
};

/// Deterministic values in [-1, 1).
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

fn binary_graph(op: Op, shape: &[usize]) -> Graph {
    let mut b = GraphBuilder::new(Limits::default());
    let x = b.input("x", shape.to_vec()).unwrap();
    let y = b.input("y", shape.to_vec()).unwrap();
    let z = b.node("z", op, &[x, y]).unwrap();
    b.finish(&[z]).unwrap()
}

fn unary_graph(op: Op, shape: &[usize]) -> Graph {
    let mut b = GraphBuilder::new(Limits::default());
    let x = b.input("x", shape.to_vec()).unwrap();
    let z = b.node("z", op, &[x]).unwrap();
    b.finish(&[z]).unwrap()
}

fn time_backends(c: &mut Criterion, name: &str, size: usize, graph: &Graph, inputs: &[Tensor]) {
    let mut group = c.benchmark_group(name);
    let reference = ReferenceBackend.compile(graph).unwrap();
    group.bench_function(format!("reference/{size}"), |b| {
        b.iter(|| black_box(reference.run(black_box(inputs)).unwrap()))
    });
    let cpu = CpuBackend.compile(graph).unwrap();
    group.bench_function(format!("{CPU}/{size}"), |b| {
        b.iter(|| black_box(cpu.run(black_box(inputs)).unwrap()))
    });
    group.finish();
}

fn matmul(c: &mut Criterion) {
    for n in [64usize, 128, 256, 512] {
        let graph = binary_graph(Op::MatMul, &[n, n]);
        let inputs = [tensor(&[n, n], 1), tensor(&[n, n], 2)];
        let mut group = c.benchmark_group("matmul");
        group.throughput(Throughput::Elements(2 * (n * n * n) as u64));
        let a = Array2::from_shape_vec((n, n), data(n * n, 1)).unwrap();
        let b = Array2::from_shape_vec((n, n), data(n * n, 2)).unwrap();
        group.bench_function(format!("ndarray/{n}"), |bench| {
            bench.iter(|| black_box(black_box(&a).dot(black_box(&b))))
        });
        let reference = ReferenceBackend.compile(&graph).unwrap();
        group.bench_function(format!("reference/{n}"), |bench| {
            bench.iter(|| black_box(reference.run(black_box(&inputs)).unwrap()))
        });
        let cpu = CpuBackend.compile(&graph).unwrap();
        group.bench_function(format!("{CPU}/{n}"), |bench| {
            bench.iter(|| black_box(cpu.run(black_box(&inputs)).unwrap()))
        });
        group.finish();
    }
}

/// A batch of `m` rows against a fixed weight: the shape of inference with a
/// small batch. `matmul-skinny` has a 1024 x 1024 weight (a hidden layer),
/// `matmul-head` a 784 x 10 one (a classifier head, 10 is not a multiple of
/// the vector width). The size in the id is the number of rows.
fn skinny(c: &mut Criterion) {
    for (name, k, n) in [
        ("matmul-skinny", 1024usize, 1024usize),
        ("matmul-head", 784, 10),
    ] {
        for m in [1usize, 2, 3, 5, 8, 32] {
            let graph = binary_graph_shapes(Op::MatMul, &[m, k], &[k, n]);
            let inputs = [tensor(&[m, k], 1), tensor(&[k, n], 2)];
            let mut group = c.benchmark_group(name);
            group.throughput(Throughput::Elements(2 * (m * k * n) as u64));
            let a = Array2::from_shape_vec((m, k), data(m * k, 1)).unwrap();
            let b = Array2::from_shape_vec((k, n), data(k * n, 2)).unwrap();
            group.bench_function(format!("ndarray/{m}"), |bench| {
                bench.iter(|| black_box(black_box(&a).dot(black_box(&b))))
            });
            let cpu = CpuBackend.compile(&graph).unwrap();
            group.bench_function(format!("{CPU}/{m}"), |bench| {
                bench.iter(|| black_box(cpu.run(black_box(&inputs)).unwrap()))
            });
            group.finish();
        }
    }
}

fn binary_graph_shapes(op: Op, a: &[usize], b: &[usize]) -> Graph {
    let mut g = GraphBuilder::new(Limits::default());
    let x = g.input("x", a.to_vec()).unwrap();
    let y = g.input("y", b.to_vec()).unwrap();
    let z = g.node("z", op, &[x, y]).unwrap();
    g.finish(&[z]).unwrap()
}

fn elementwise(c: &mut Criterion) {
    let n = 1 << 20;
    let inputs = [tensor(&[n], 3), tensor(&[n], 4)];
    time_backends(c, "add", n, &binary_graph(Op::Add, &[n]), &inputs);
    time_backends(c, "relu", n, &unary_graph(Op::Relu, &[n]), &inputs[..1]);
    time_backends(
        c,
        "sigmoid",
        n,
        &unary_graph(Op::Sigmoid, &[n]),
        &inputs[..1],
    );
}

fn transpose(c: &mut Criterion) {
    let n = 1024;
    let graph = unary_graph(Op::Transpose { perm: vec![1, 0] }, &[n, n]);
    time_backends(c, "transpose", n, &graph, &[tensor(&[n, n], 5)]);
}

/// A graph shaped like an exporter's output: the weight is stored transposed
/// and scaled by a constant, and a dead debug branch hangs off the input.
fn exporter_style_graph(n: usize) -> Graph {
    let mut b = GraphBuilder::new(Limits::default());
    let x = b.input("x", vec![64, n]).unwrap();
    let w = b.constant("w", vec![n, n], data(n * n, 6)).unwrap();
    let scale = b.constant("scale", vec![], vec![0.5]).unwrap();
    let wt = b
        .node("wt", Op::Transpose { perm: vec![1, 0] }, &[w])
        .unwrap();
    let ws = b.node("ws", Op::Mul, &[wt, scale]).unwrap();
    let y = b.node("y", Op::MatMul, &[x, ws]).unwrap();
    let out = b.node("out", Op::Relu, &[y]).unwrap();
    b.node("debug", Op::Sigmoid, &[x]).unwrap();
    b.finish(&[out]).unwrap()
}

fn pipeline(c: &mut Criterion) {
    let n = 256;
    let graph = exporter_style_graph(n);
    let optimized = default_pipeline(Limits::default(), Arc::new(ReferenceEvaluator))
        .run(&graph)
        .unwrap()
        .graph;
    let inputs = [tensor(&[64, n], 7)];
    let mut group = c.benchmark_group("pipeline");
    let plain = CpuBackend.compile(&graph).unwrap();
    group.bench_function(format!("{CPU}-unoptimized/{n}"), |b| {
        b.iter(|| black_box(plain.run(black_box(&inputs)).unwrap()))
    });
    let fast = CpuBackend.compile(&optimized).unwrap();
    group.bench_function(format!("{CPU}-optimized/{n}"), |b| {
        b.iter(|| black_box(fast.run(black_box(&inputs)).unwrap()))
    });
    group.finish();
}

criterion_group!(benches, matmul, skinny, elementwise, transpose, pipeline);
criterion_main!(benches);
