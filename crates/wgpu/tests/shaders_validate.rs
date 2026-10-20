//! Every generated shader must parse and validate offline, for every kernel
//! configuration we can generate, with no optional device capability.

use naga::valid::{Capabilities, ValidationFlags, Validator};
use oxfuse_wgpu::shaders::{self, Binary, Unary};
use oxfuse_wgpu::{DeviceCaps, KernelConfig};

fn check(label: &str, source: &str) {
    let module = naga::front::wgsl::parse_str(source).unwrap_or_else(|e| {
        panic!(
            "{label} does not parse: {}\n{source}",
            e.emit_to_string(source)
        )
    });
    Validator::new(ValidationFlags::all(), Capabilities::empty())
        .validate(&module)
        .unwrap_or_else(|e| panic!("{label} does not validate: {e:?}\n{source}"));
}

fn configs() -> Vec<KernelConfig> {
    let mut all = Vec::new();
    for tile in [1, 2, 4, 8, 16] {
        for wg in [32, 64, 128, 256] {
            for reg_block in [1, 2, 4] {
                all.push(KernelConfig {
                    elementwise_wg: wg,
                    tile,
                    reg_block,
                    transpose_tile: tile,
                });
            }
        }
    }
    all
}

#[test]
fn the_checker_can_fail() {
    let module = naga::front::wgsl::parse_str("fn f() -> u32 { return 1.0; }");
    let rejected = match module {
        Err(_) => true,
        Ok(m) => Validator::new(ValidationFlags::all(), Capabilities::empty())
            .validate(&m)
            .is_err(),
    };
    assert!(rejected, "a type error must not pass the offline check");
}

#[test]
fn every_shader_validates_for_every_config() {
    let mut count = 0;
    for cfg in configs() {
        for op in [Unary::Relu, Unary::Sigmoid] {
            check(&format!("{op:?} {cfg:?}"), &shaders::unary(op, &cfg));
            count += 1;
        }
        for op in [Binary::Add, Binary::Mul] {
            for (lhs, rhs) in [(false, false), (true, false), (false, true)] {
                check(
                    &format!("{op:?} {lhs} {rhs} {cfg:?}"),
                    &shaders::binary(op, lhs, rhs, &cfg),
                );
                count += 1;
            }
        }
        check(&format!("transpose2d {cfg:?}"), &shaders::transpose2d(&cfg));
        check(&format!("matmul {cfg:?}"), &shaders::matmul(&cfg));
        for rank in 1..=8 {
            check(
                &format!("permute {rank} {cfg:?}"),
                &shaders::permute(rank, &cfg),
            );
            count += 1;
        }
        count += 2;
    }
    assert!(count >= 60 * 17, "only {count} shaders were checked");
}

#[test]
fn shaders_stay_within_the_storage_buffer_budget() {
    let cfg = KernelConfig::DEFAULT;
    assert_eq!(cfg.validate(&DeviceCaps::DOWNLEVEL), Ok(()));
    let all = [
        shaders::unary(Unary::Relu, &cfg),
        shaders::binary(Binary::Add, false, false, &cfg),
        shaders::transpose2d(&cfg),
        shaders::permute(4, &cfg),
        shaders::matmul(&cfg),
    ];
    for source in all {
        let storage = source.matches("var<storage").count() as u32;
        assert!(
            storage <= oxfuse_wgpu::KERNEL_STORAGE_BUFFERS,
            "{storage} storage buffers in:\n{source}"
        );
    }
}

/// The DX12 software adapter (Microsoft Basic Render Driver) crashes the process
/// with STATUS_ACCESS_VIOLATION when a kernel returns early before a barrier, even
/// if every thread of the workgroup returns together. Found by running small raw
/// shaders on a Windows CI runner: a barrier alone, or inside a loop, was fine; an
/// early return before it was not. Kernels with barriers must reach them
/// unconditionally and guard their memory accesses instead.
#[test]
fn kernels_with_barriers_never_return_early() {
    for cfg in configs() {
        for (name, source) in [
            ("transpose2d", shaders::transpose2d(&cfg)),
            ("matmul", shaders::matmul(&cfg)),
        ] {
            assert!(
                source.contains("workgroupBarrier()"),
                "{name} has no barrier"
            );
            assert!(
                !source.contains("return;"),
                "{name} returns early in a kernel with barriers:\n{source}"
            );
        }
    }
}
