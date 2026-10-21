//! Runs `relu(x @ w)` on the GPU and on the reference backend, checks every
//! element against the documented bound, then runs it again with the tensors
//! kept on the device. Run with `cargo run --release -p oxfuse-wgpu --example matmul`.

use oxfuse_core::{GraphBuilder, Limits, Op};
use oxfuse_runtime::{Backend, Executable, ReferenceBackend, Tensor};
use oxfuse_wgpu::{WgpuBackend, gpu_matmul_bound};

const N: usize = 256;

/// Small deterministic values, so the example needs no random-number crate.
fn values(seed: u32) -> Vec<f32> {
    (0..N * N)
        .map(|i| {
            let h = (i as u32).wrapping_mul(2_654_435_761).wrapping_add(seed);
            ((h >> 8) % 2001) as f32 / 1000.0 - 1.0
        })
        .collect()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut b = GraphBuilder::new(Limits::default());
    let x = b.input("x", vec![N, N])?;
    let w = b.input("w", vec![N, N])?;
    let m = b.node("m", Op::MatMul, &[x, w])?;
    let y = b.node("y", Op::Relu, &[m])?;
    let graph = b.finish(&[y])?;

    let (xs, ws) = (values(1), values(2));
    let inputs = [
        Tensor::new(vec![N, N], xs.clone())?,
        Tensor::new(vec![N, N], ws.clone())?,
    ];

    let gpu = WgpuBackend::new()?;
    println!("adapter: {}", gpu.context().info().name);
    let exe = gpu.compile_device(&graph)?;

    // Host in, host out: the simple path.
    let got = exe.run(&inputs)?;
    let want = ReferenceBackend.compile(&graph)?.run(&inputs)?;

    // The contract bounds MatMul against the float64 result, not against another f32 run.
    let mut worst = 0.0f64;
    for r in 0..N {
        for c in 0..N {
            let row = &xs[r * N..(r + 1) * N];
            let col: Vec<f32> = (0..N).map(|k| ws[k * N + c]).collect();
            let exact: f64 = row
                .iter()
                .zip(&col)
                .map(|(a, b)| f64::from(*a) * f64::from(*b))
                .sum();
            let exact = exact.max(0.0);
            let err = (f64::from(got[0].data()[r * N + c]) - exact).abs();
            let bound = gpu_matmul_bound(row, &col);
            assert!(err <= bound, "[{r},{c}] error {err} over bound {bound}");
            worst = worst.max(err / bound.max(f64::MIN_POSITIVE));
        }
    }
    println!(
        "all {} elements inside the bound (worst uses {:.1}% of it)",
        N * N,
        worst * 100.0
    );
    let diff = got[0]
        .data()
        .iter()
        .zip(want[0].data())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    println!("largest difference to the reference backend: {diff:e}");

    // Device resident: upload once, run many times, download once.
    let on_device = [exe.upload(&inputs[0])?, exe.upload(&inputs[1])?];
    let mut out = exe.run_device(&on_device)?;
    for _ in 0..9 {
        out = exe.run_device(&on_device)?;
    }
    let back = exe.download(&out[0])?;
    assert_eq!(
        back.data(),
        got[0].data(),
        "the same kernels give the same bits"
    );
    println!("device-resident run matches the host path bit for bit");
    Ok(())
}
