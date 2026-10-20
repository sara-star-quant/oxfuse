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

const RAW_HEAD: &str = "@group(0) @binding(0) var<storage, read_write> dst: array<f32>;\n";

/// Small shaders that each add one construct the kernels use.
fn raw_shader(name: &str) -> String {
    let body = match name {
        // plain store
        "F" => {
            "@compute @workgroup_size(4)\nfn main(@builtin(local_invocation_id) l: vec3<u32>) {\n  dst[l.x] = f32(l.x);\n}"
        }
        // workgroup memory, no barrier
        "E" => {
            "var<workgroup> t: array<f32, 4>;\n@compute @workgroup_size(4)\nfn main(@builtin(local_invocation_id) l: vec3<u32>) {\n  t[l.x] = f32(l.x) + 1.0;\n  dst[l.x] = t[l.x];\n}"
        }
        // workgroup memory and a barrier
        "A" => {
            "var<workgroup> t: array<f32, 4>;\n@compute @workgroup_size(4)\nfn main(@builtin(local_invocation_id) l: vec3<u32>) {\n  t[l.x] = f32(l.x) + 1.0;\n  workgroupBarrier();\n  dst[l.x] = t[(l.x + 1u) % 4u];\n}"
        }
        // A plus an early return before the barrier
        "B" => {
            "var<workgroup> t: array<f32, 4>;\n@compute @workgroup_size(4)\nfn main(@builtin(local_invocation_id) l: vec3<u32>, @builtin(workgroup_id) w: vec3<u32>) {\n  if (w.x >= 5u) { return; }\n  t[l.x] = f32(l.x) + 1.0;\n  workgroupBarrier();\n  dst[l.x] = t[(l.x + 1u) % 4u];\n}"
        }
        // barriers inside a loop
        "C" => {
            "var<workgroup> t: array<f32, 4>;\n@compute @workgroup_size(4)\nfn main(@builtin(local_invocation_id) l: vec3<u32>) {\n  var k0 = 0u;\n  var acc = 0.0;\n  loop {\n    if (k0 >= 2u) { break; }\n    t[l.x] = f32(k0 + l.x);\n    workgroupBarrier();\n    acc = acc + t[(l.x + 1u) % 4u];\n    workgroupBarrier();\n    k0 += 1u;\n  }\n  dst[l.x] = acc;\n}"
        }
        // C plus an early return before the loop
        "D" => {
            "var<workgroup> t: array<f32, 4>;\n@compute @workgroup_size(4)\nfn main(@builtin(local_invocation_id) l: vec3<u32>, @builtin(workgroup_id) w: vec3<u32>) {\n  if (w.x >= 5u) { return; }\n  var k0 = 0u;\n  var acc = 0.0;\n  loop {\n    if (k0 >= 2u) { break; }\n    t[l.x] = f32(k0 + l.x);\n    workgroupBarrier();\n    acc = acc + t[(l.x + 1u) % 4u];\n    workgroupBarrier();\n    k0 += 1u;\n  }\n  dst[l.x] = acc;\n}"
        }
        other => panic!("unknown raw shader {other}"),
    };
    format!("{RAW_HEAD}{body}\n")
}

#[test]
#[ignore = "diagnostic: run with OXFUSE_RAW=<F|E|A|B|C|D> --ignored"]
fn probe_one_raw_shader() {
    let Ok(name) = std::env::var("OXFUSE_RAW") else {
        return;
    };
    say(&format!("raw shader {name}"));
    if !common::available() {
        say("no adapter");
        return;
    }
    let ctx = common::gpu().context().clone();
    let device = ctx.device();
    say("create module and pipeline");
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(raw_shader(&name).into()),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: None,
        module: &module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });
    let dst = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 16,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 16,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: dst.as_entire_binding(),
        }],
    });
    say("dispatch");
    let mut encoder = device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }
    encoder.copy_buffer_to_buffer(&dst, 0, &staging, 0, 16);
    ctx.queue().submit([encoder.finish()]);
    say("wait for the device");
    staging
        .slice(..)
        .map_async(wgpu::MapMode::Read, |r| r.unwrap());
    ctx.wait_idle().unwrap();
    say("done");
}
