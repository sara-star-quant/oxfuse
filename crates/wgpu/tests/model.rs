//! The kernel model walks the same grids and index math as the shaders. It must
//! agree bit for bit with the reference backend, write every element exactly
//! once and never touch memory out of range.

use oxfuse_core::{GraphBuilder, Limits, Op};
use oxfuse_runtime::{Backend, ReferenceBackend, Tensor};
use oxfuse_wgpu::shaders::{Binary, Unary};
use oxfuse_wgpu::{DeviceCaps, KernelConfig, model};

const CAPS: DeviceCaps = DeviceCaps::DOWNLEVEL;

fn cfg_reg(wg: u32, tile: u32, reg: u32) -> KernelConfig {
    KernelConfig {
        elementwise_wg: wg,
        tile,
        reg_block: reg,
        transpose_tile: tile,
    }
}

fn cfg(wg: u32, tile: u32) -> KernelConfig {
    cfg_reg(wg, tile, 1)
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

/// Deterministic values, including -0 and a wide magnitude range; `special` also
/// injects NaN, infinities and subnormals.
fn data(n: usize, seed: u64, special: bool) -> Vec<f32> {
    let mut state = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (0..n)
        .map(|i| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let unit = ((state >> 40) as f32) / (1u64 << 24) as f32 - 0.5;
            match (special, i % 11) {
                (true, 3) => f32::NAN,
                (true, 5) => f32::INFINITY,
                (true, 7) => f32::NEG_INFINITY,
                (true, 9) => f32::from_bits(1 + (state >> 60) as u32),
                (_, 1) => -0.0,
                (_, 2) => 0.0,
                _ => unit * 10f32.powi((state >> 33) as i32 % 7 - 3),
            }
        })
        .collect()
}

fn reference(op: Op, inputs: Vec<Tensor>) -> Vec<f32> {
    let mut b = GraphBuilder::new(Limits::default());
    let ids: Vec<_> = inputs
        .iter()
        .enumerate()
        .map(|(i, t)| b.input(&format!("in{i}"), t.shape().to_vec()).unwrap())
        .collect();
    let out = b.node("out", op, &ids).unwrap();
    let graph = b.finish(&[out]).unwrap();
    let result = ReferenceBackend
        .compile(&graph)
        .unwrap()
        .run(&inputs)
        .unwrap();
    result[0].data().to_vec()
}

fn tensor(shape: &[usize], seed: u64, special: bool) -> Tensor {
    let n = shape.iter().product();
    Tensor::new(shape.to_vec(), data(n, seed, special)).unwrap()
}

#[test]
fn unary_ops_match_the_reference_for_every_size_and_workgroup() {
    for wg in [1, 2, 4, 32] {
        for n in 1..=70usize {
            let t = tensor(&[n], n as u64, true);
            for (op, kind) in [(Op::Relu, Unary::Relu), (Op::Sigmoid, Unary::Sigmoid)] {
                let got = model::unary(kind, t.data(), &cfg(wg, 4), &CAPS);
                assert_eq!(
                    bits(&got),
                    bits(&reference(op, vec![t.clone()])),
                    "{kind:?} wg {wg} n {n}"
                );
            }
        }
    }
}

#[test]
fn binary_ops_match_the_reference_including_scalar_operands() {
    for wg in [1, 4, 32] {
        for n in [1usize, 2, 31, 32, 33, 70] {
            let a = tensor(&[n], 1, true);
            let b = tensor(&[n], 2, true);
            let s = tensor(&[], 3, false);
            for (op, kind) in [(Op::Add, Binary::Add), (Op::Mul, Binary::Mul)] {
                let c = cfg(wg, 4);
                let full = model::binary(kind, a.data(), b.data(), false, false, &c, &CAPS);
                assert_eq!(
                    bits(&full),
                    bits(&reference(op.clone(), vec![a.clone(), b.clone()]))
                );
                let rhs = model::binary(kind, a.data(), s.data(), false, true, &c, &CAPS);
                assert_eq!(
                    bits(&rhs),
                    bits(&reference(op.clone(), vec![a.clone(), s.clone()]))
                );
                let lhs = model::binary(kind, s.data(), b.data(), true, false, &c, &CAPS);
                assert_eq!(bits(&lhs), bits(&reference(op, vec![s.clone(), b.clone()])));
            }
        }
    }
}

#[test]
fn transpose2d_matches_the_reference_on_every_small_shape() {
    for tile in [1, 2, 4] {
        for rows in 1..=9usize {
            for cols in 1..=9usize {
                let t = tensor(&[rows, cols], (rows * 10 + cols) as u64, true);
                let got = model::transpose2d(t.data(), rows, cols, &cfg(4, tile), &CAPS);
                let want = reference(Op::Transpose { perm: vec![1, 0] }, vec![t]);
                assert_eq!(bits(&got), bits(&want), "tile {tile} {rows}x{cols}");
            }
        }
    }
}

#[test]
fn matmul_matches_the_reference_on_every_small_shape() {
    for tile in [1, 2, 4] {
        for m in 1..=9usize {
            for k in 1..=9usize {
                for n in 1..=9usize {
                    let a = tensor(&[m, k], (m * 100 + k * 10 + n) as u64, false);
                    let b = tensor(&[k, n], (n * 100 + k * 10 + m) as u64, false);
                    let got = model::matmul(a.data(), b.data(), m, k, n, &cfg(4, tile), &CAPS);
                    let want = reference(Op::MatMul, vec![a, b]);
                    assert_eq!(bits(&got), bits(&want), "tile {tile} {m}x{k}x{n}");
                }
            }
        }
    }
}

#[test]
fn matmul_matches_the_reference_around_larger_tile_multiples() {
    for tile in [3usize, 5, 7, 16] {
        let edges = [1, tile - 1, tile, tile + 1, 2 * tile + 1];
        for &m in &edges {
            for &k in &edges {
                for &n in &edges {
                    let a = tensor(&[m, k], 7, false);
                    let b = tensor(&[k, n], 8, false);
                    let got =
                        model::matmul(a.data(), b.data(), m, k, n, &cfg(4, tile as u32), &CAPS);
                    assert_eq!(
                        bits(&got),
                        bits(&reference(Op::MatMul, vec![a, b])),
                        "tile {tile} {m}x{k}x{n}"
                    );
                }
            }
        }
    }
}

#[test]
fn matmul_with_register_blocks_matches_the_reference_on_every_small_shape() {
    for (tile, reg) in [(1, 2), (2, 2), (2, 3), (3, 2), (4, 2)] {
        for m in 1..=9usize {
            for k in 1..=9usize {
                for n in 1..=9usize {
                    let a = tensor(&[m, k], (m * 100 + k * 10 + n) as u64, false);
                    let b = tensor(&[k, n], (n * 100 + k * 10 + m) as u64, false);
                    let got =
                        model::matmul(a.data(), b.data(), m, k, n, &cfg_reg(4, tile, reg), &CAPS);
                    let want = reference(Op::MatMul, vec![a, b]);
                    assert_eq!(bits(&got), bits(&want), "tile {tile} reg {reg} {m}x{k}x{n}");
                }
            }
        }
    }
}

#[test]
fn matmul_matches_the_reference_around_block_multiples() {
    for (tile, reg) in [(5usize, 1usize), (4, 4), (16, 4)] {
        let block = tile * reg;
        let edges = [
            1,
            tile - 1,
            tile,
            tile + 1,
            block - 1,
            block,
            block + 1,
            2 * block + 1,
        ];
        for &m in &edges {
            for &k in &edges {
                for &n in &edges {
                    let a = tensor(&[m, k], 7, false);
                    let b = tensor(&[k, n], 8, false);
                    let c = cfg_reg(4, tile as u32, reg as u32);
                    let got = model::matmul(a.data(), b.data(), m, k, n, &c, &CAPS);
                    assert_eq!(
                        bits(&got),
                        bits(&reference(Op::MatMul, vec![a, b])),
                        "tile {tile} reg {reg} {m}x{k}x{n}"
                    );
                }
            }
        }
    }
}

fn permutations(n: usize) -> Vec<Vec<usize>> {
    if n == 1 {
        return vec![vec![0]];
    }
    let mut all = Vec::new();
    for tail in permutations(n - 1) {
        for at in 0..n {
            let mut p = tail.clone();
            p.insert(at, n - 1);
            all.push(p);
        }
    }
    all
}

#[test]
fn permute_matches_the_reference_for_every_permutation_up_to_rank_four() {
    for rank in 1..=4usize {
        let shape: Vec<usize> = (0..rank).map(|d| 2 + d % 3).collect();
        let t = tensor(&shape, rank as u64, true);
        for perm in permutations(rank) {
            for wg in [1, 8] {
                let got = model::permute(t.data(), &shape, &perm, &cfg(wg, 4), &CAPS);
                let want = reference(Op::Transpose { perm: perm.clone() }, vec![t.clone()]);
                assert_eq!(bits(&got), bits(&want), "rank {rank} perm {perm:?} wg {wg}");
            }
        }
    }
    // Rank eight reaches the second vec4 of dims and strides.
    let shape = [2, 1, 2, 1, 2, 1, 2, 3];
    let t = tensor(&shape, 9, false);
    let perm = vec![7, 6, 5, 4, 3, 2, 1, 0];
    let got = model::permute(t.data(), &shape, &perm, &cfg(4, 4), &CAPS);
    assert_eq!(
        bits(&got),
        bits(&reference(Op::Transpose { perm }, vec![t]))
    );
}

#[test]
fn the_two_dimensional_fold_covers_everything_with_a_tiny_dispatch_limit() {
    let tiny = DeviceCaps {
        max_workgroups_per_dim: 3,
        ..DeviceCaps::DOWNLEVEL
    };
    let c = cfg_reg(4, 2, 2);
    for n in [1usize, 4, 5, 13, 36] {
        let t = tensor(&[n], 4, true);
        let got = model::unary(Unary::Relu, t.data(), &c, &tiny);
        assert_eq!(bits(&got), bits(&reference(Op::Relu, vec![t])), "n {n}");
    }
    let a = tensor(&[5, 4], 1, false);
    let b = tensor(&[4, 5], 2, false);
    let got = model::matmul(a.data(), b.data(), 5, 4, 5, &c, &tiny);
    assert_eq!(bits(&got), bits(&reference(Op::MatMul, vec![a, b])));
    let t = tensor(&[5, 6], 3, false);
    let got = model::transpose2d(t.data(), 5, 6, &c, &tiny);
    assert_eq!(
        bits(&got),
        bits(&reference(Op::Transpose { perm: vec![1, 0] }, vec![t]))
    );
}

#[test]
#[should_panic(expected = "cannot be dispatched")]
fn a_tensor_too_large_for_the_dispatch_limit_is_refused_not_truncated() {
    let tiny = DeviceCaps {
        max_workgroups_per_dim: 2,
        ..DeviceCaps::DOWNLEVEL
    };
    let t = tensor(&[100], 1, false);
    model::unary(Unary::Relu, t.data(), &cfg(4, 2), &tiny);
}
