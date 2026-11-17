//! Throwaway experiment, never merged: how fast is an exact int8 matmul (i32
//! accumulators) next to the repo's f32 chain on the same machine, and is its
//! result identical when the sum is split by k or by rows over threads? Sizes
//! are run-time values, as in the repo.
use fearless_simd::{Level, Simd, dispatch};
use fearless_simd_macros::simd;
use oxfuse_core::{GraphBuilder, Limits, Op};
use oxfuse_runtime::{Backend, CpuBackend, Tensor};
use std::hint::black_box;
use std::time::{Duration, Instant};

/// `c[i][..] += sum over p in p0..p1 of a[i][p] * b[p][..]`, one row of A at a time.
#[simd]
fn rows1<S: Simd>(_s: S, a: &[i8], b: &[i8], (m, k, n): (usize, usize, usize), (p0, p1): (usize, usize), c: &mut [i32]) {
    for i in 0..m {
        let row = &mut c[i * n..(i + 1) * n];
        for p in p0..p1 {
            let x = i32::from(a[i * k + p]);
            for (o, &w) in row.iter_mut().zip(&b[p * n..(p + 1) * n]) {
                *o += x * i32::from(w);
            }
        }
    }
}

/// Four rows of A share each loaded row of B.
#[simd]
fn rows4<S: Simd>(s: S, a: &[i8], b: &[i8], (m, k, n): (usize, usize, usize), (p0, p1): (usize, usize), c: &mut [i32]) {
    let full = m / 4 * 4;
    let mut i = 0;
    while i < full {
        let (c0, rest) = c[i * n..(i + 4) * n].split_at_mut(n);
        let (c1, rest) = rest.split_at_mut(n);
        let (c2, c3) = rest.split_at_mut(n);
        for p in p0..p1 {
            let (x0, x1, x2, x3) = (
                i32::from(a[i * k + p]),
                i32::from(a[(i + 1) * k + p]),
                i32::from(a[(i + 2) * k + p]),
                i32::from(a[(i + 3) * k + p]),
            );
            let brow = &b[p * n..(p + 1) * n];
            for j in 0..n {
                let w = i32::from(brow[j]);
                c0[j] += x0 * w;
                c1[j] += x1 * w;
                c2[j] += x2 * w;
                c3[j] += x3 * w;
            }
        }
        i += 4;
    }
    if full < m {
        let tail_a = &a[full * k..];
        rows1(s, tail_a, b, (m - full, k, n), (p0, p1), &mut c[full * n..]);
    }
}

type Kernel = fn(Level, &[i8], &[i8], (usize, usize, usize), (usize, usize), &mut [i32]);

fn run1(l: Level, a: &[i8], b: &[i8], d: (usize, usize, usize), r: (usize, usize), c: &mut [i32]) {
    dispatch!(l, s => rows1(s, a, b, d, r, c))
}
fn run4(l: Level, a: &[i8], b: &[i8], d: (usize, usize, usize), r: (usize, usize), c: &mut [i32]) {
    dispatch!(l, s => rows4(s, a, b, d, r, c))
}

fn bytes(len: usize, seed: u32) -> Vec<i8> {
    let mut s = seed.wrapping_mul(2654435761).wrapping_add(1);
    (0..len)
        .map(|_| {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            (s >> 24) as u8 as i8
        })
        .collect()
}

fn floats(len: usize, seed: u32) -> Vec<f32> {
    bytes(len, seed).into_iter().map(|v| f32::from(v) / 64.0).collect()
}

fn reference(a: &[i8], b: &[i8], (m, k, n): (usize, usize, usize)) -> Vec<i32> {
    let mut c = vec![0i32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0i64;
            for p in 0..k {
                acc += i64::from(a[i * k + p]) * i64::from(b[p * n + j]);
            }
            c[i * n + j] = acc as i32;
        }
    }
    c
}

fn median_us(mut run: impl FnMut()) -> f64 {
    let t = Instant::now();
    let mut reps = 0u64;
    while t.elapsed() < Duration::from_millis(60) {
        run();
        reps += 1;
    }
    let iters = ((0.04 / (t.elapsed().as_secs_f64() / reps as f64)).ceil() as u64).max(1);
    let mut v = Vec::new();
    for _ in 0..7 {
        let t = Instant::now();
        for _ in 0..iters {
            run();
        }
        v.push(t.elapsed().as_secs_f64() / iters as f64 * 1e6);
    }
    v.sort_by(f64::total_cmp);
    v[3]
}

fn f32_us(m: usize, k: usize, n: usize) -> f64 {
    let mut g = GraphBuilder::new(Limits::default());
    let x = g.input("x", vec![m, k]).unwrap();
    let w = g.input("w", vec![k, n]).unwrap();
    let y = g.node("y", Op::MatMul, &[x, w]).unwrap();
    let graph = g.finish(&[y]).unwrap();
    let exe = CpuBackend.compile(&graph).unwrap();
    let inputs = [
        Tensor::new(vec![m, k], floats(m * k, 1)).unwrap(),
        Tensor::new(vec![k, n], floats(k * n, 2)).unwrap(),
    ];
    median_us(|| {
        black_box(exe.run(black_box(&inputs)).unwrap());
    })
}

fn main() {
    let level = Level::new();
    println!("iprobe level={level:?}");
    let shapes = [(1usize, 784usize, 256usize), (8, 784, 256), (32, 784, 256), (32, 256, 10), (128, 128, 128), (512, 512, 512)];
    for (m, k, n) in shapes {
        let (m, k, n) = (black_box(m), black_box(k), black_box(n));
        let a = bytes(m * k, 1);
        let b = bytes(k * n, 2);
        let want = reference(&a, &b, (m, k, n));
        let kernels: [(&str, Kernel); 2] = [("rows1", run1), ("rows4", run4)];
        let mut line = format!("iprobe m={m} k={k} n={n} f32_us={:.1}", f32_us(m, k, n));
        for (name, f) in kernels {
            let mut c = vec![0i32; m * n];
            f(level, &a, &b, (m, k, n), (0, k), &mut c);
            assert_eq!(c, want, "{name} differs from the reference at {m}x{k}x{n}");
            // Split the sum by k: two halves into the same accumulators.
            let mut c2 = vec![0i32; m * n];
            let mid = black_box(k / 2);
            f(level, &a, &b, (m, k, n), (0, mid), &mut c2);
            f(level, &a, &b, (m, k, n), (mid, k), &mut c2);
            let split_k_equal = c2 == want;
            // Split by rows over two threads.
            let mut c3 = vec![0i32; m * n];
            let half = m / 2;
            let (top, bottom) = c3.split_at_mut(half * n);
            std::thread::scope(|s| {
                s.spawn(|| f(level, &a[..half * k], &b, (half, k, n), (0, k), top));
                s.spawn(|| f(level, &a[half * k..], &b, (m - half, k, n), (0, k), bottom));
            });
            let rows_equal = c3 == want;
            let mut c4 = vec![0i32; m * n];
            let us = median_us(|| {
                c4.fill(0);
                f(level, black_box(&a), black_box(&b), (m, k, n), (0, k), black_box(&mut c4));
            });
            line.push_str(&format!(" {name}_us={us:.1} {name}_splitk_exact={split_k_equal} {name}_rows_exact={rows_equal}"));
        }
        println!("{line}");
    }
}
