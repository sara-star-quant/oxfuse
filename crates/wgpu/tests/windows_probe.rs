//! Diagnostic, not a gate: runs one matmul stage by stage for one kernel
//! configuration and prints before each stage, so a native crash shows which stage
//! died. Run by the CI diagnostics job with OXFUSE_PROBE=<tile>,<reg_block>.

mod common;

use oxfuse_core::{GraphBuilder, Limits, Op};
use oxfuse_runtime::Tensor;
use oxfuse_wgpu::{KernelConfig, WgpuBackend};
use std::io::Write;

fn say(stage: &str) {
    eprintln!("probe: {stage}");
    let _ = std::io::stderr().flush();
}

#[test]
#[ignore = "diagnostic: run with OXFUSE_PROBE=<tile>,<reg_block> --ignored"]
fn probe_one_matmul_configuration() {
    let Ok(spec) = std::env::var("OXFUSE_PROBE") else {
        return;
    };
    let mut parts = spec.split(',').map(|p| p.parse::<u32>().unwrap());
    let (tile, reg_block) = (parts.next().unwrap(), parts.next().unwrap());
    say(&format!("config tile {tile} reg_block {reg_block}"));
    if !common::available() {
        say("no adapter");
        return;
    }
    let ctx = common::gpu().context().clone();
    say(&format!(
        "adapter {} {:?}",
        ctx.info().name,
        ctx.info().backend
    ));
    let cfg = KernelConfig {
        tile,
        reg_block,
        transpose_tile: tile,
        ..KernelConfig::DEFAULT
    };
    let backend = WgpuBackend::with(ctx, cfg);
    let mut b = GraphBuilder::new(Limits::default());
    let x = b.input("x", vec![2, 2]).unwrap();
    let w = b.input("w", vec![2, 2]).unwrap();
    let p = b.node("p", Op::MatMul, &[x, w]).unwrap();
    let graph = b.finish(&[p]).unwrap();
    say("compile (shader module and pipeline)");
    let exe = backend.compile_device(&graph).unwrap();
    say("upload");
    let a = exe
        .upload(&Tensor::new(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap())
        .unwrap();
    let c = exe
        .upload(&Tensor::new(vec![2, 2], vec![5.0, 6.0, 7.0, 8.0]).unwrap())
        .unwrap();
    say("run_device (dispatch)");
    let out = exe.run_device(&[a, c]).unwrap();
    say("download");
    let got = exe.download(&out[0]).unwrap();
    say(&format!("result {:?}", got.data()));
    assert_eq!(got.data(), &[19.0, 22.0, 43.0, 50.0]);
    say("done");
}
