//! `Add` and `Mul` broadcast like NumPy. For every pair of shapes up to rank 3
//! with dimensions 1, 2 and 3 (in both orders, on inputs and on a computed
//! operand whose buffer the fast backend may reuse), the reference backend
//! gives what a direct loop gives and the fast backend gives the same bits.

use oxfuse_core::{GraphBuilder, Limits, Op};
use oxfuse_runtime::{Backend, CpuBackend, ReferenceBackend, Tensor};

fn small_shapes() -> Vec<Vec<usize>> {
    let mut shapes = vec![vec![]];
    for rank in 1..=3u32 {
        for code in 0..3usize.pow(rank) {
            shapes.push((0..rank).map(|d| 1 + (code / 3usize.pow(d)) % 3).collect());
        }
    }
    shapes
}

fn values(len: usize, seed: u64) -> Vec<f32> {
    let mut state = seed;
    (0..len)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 40) as f32 / (1u64 << 24) as f32) * 4.0 - 2.0
        })
        .collect()
}

fn bits(t: &Tensor) -> Vec<u32> {
    t.data().iter().map(|v| v.to_bits()).collect()
}

/// NumPy's result by splitting every output index into per-dimension indices.
fn direct(op: &Op, a: &Tensor, b: &Tensor, out: &[usize]) -> Vec<f32> {
    let read = |t: &Tensor, coords: &[usize]| {
        let shape = t.shape();
        let mut index = 0;
        for (k, &size) in shape.iter().enumerate() {
            let d = out.len() - shape.len() + k;
            index = index * size + if size == 1 { 0 } else { coords[d] };
        }
        t.data()[index]
    };
    (0..out.iter().product::<usize>())
        .map(|mut i| {
            let mut coords = vec![0; out.len()];
            for d in (0..out.len()).rev() {
                coords[d] = i % out[d];
                i /= out[d];
            }
            let (x, y) = (read(a, &coords), read(b, &coords));
            if matches!(op, Op::Add) { x + y } else { x * y }
        })
        .collect()
}

fn broadcast(a: &[usize], b: &[usize]) -> Option<Vec<usize>> {
    oxfuse_core::shape::numpy_broadcast(a, b)
}

#[test]
fn both_backends_broadcast_every_small_pair_like_a_direct_loop() {
    let shapes = small_shapes();
    let mut checked = 0;
    for sa in &shapes {
        for sb in &shapes {
            let Some(out) = broadcast(sa, sb) else {
                continue;
            };
            for op in [Op::Add, Op::Mul] {
                let mut g = GraphBuilder::new(Limits::default());
                let x = g.input("x", sa.clone()).unwrap();
                let y = g.input("y", sb.clone()).unwrap();
                let z = g.node("z", op.clone(), &[x, y]).unwrap();
                let graph = g.finish(&[z]).unwrap();
                let (xa, yb) = (
                    Tensor::new(sa.clone(), values(sa.iter().product(), 1)).unwrap(),
                    Tensor::new(sb.clone(), values(sb.iter().product(), 2)).unwrap(),
                );
                let inputs = [xa.clone(), yb.clone()];
                let want = direct(&op, &xa, &yb, &out);
                let reference = ReferenceBackend
                    .compile(&graph)
                    .unwrap()
                    .run(&inputs)
                    .unwrap();
                let fast = CpuBackend.compile(&graph).unwrap().run(&inputs).unwrap();
                assert_eq!(reference[0].shape(), out.as_slice(), "{sa:?} {sb:?}");
                let want_bits: Vec<u32> = want.iter().map(|v| v.to_bits()).collect();
                assert_eq!(bits(&reference[0]), want_bits, "{op:?} {sa:?} {sb:?}");
                assert_eq!(bits(&fast[0]), want_bits, "fast {op:?} {sa:?} {sb:?}");
                checked += 1;
            }
        }
    }
    // 40 x 40 ordered pairs, 940 of which broadcast (NumPy's count), twice (Add, Mul).
    assert_eq!(checked, 2 * 940);
}

#[test]
fn a_computed_operand_that_the_fast_backend_may_overwrite_still_broadcasts() {
    let shapes = small_shapes();
    for sa in &shapes {
        for sb in &shapes {
            let Some(out) = broadcast(sa, sb) else {
                continue;
            };
            for swap in [false, true] {
                let mut g = GraphBuilder::new(Limits::default());
                let x = g.input("x", sa.clone()).unwrap();
                let y = g.input("y", sb.clone()).unwrap();
                // r is computed, read once, and as large as x: a reuse candidate.
                let r = g.node("r", Op::Relu, &[x]).unwrap();
                let operands = if swap { [y, r] } else { [r, y] };
                let z = g.node("z", Op::Add, &operands).unwrap();
                let graph = g.finish(&[z]).unwrap();
                let (xa, yb) = (
                    Tensor::new(sa.clone(), values(sa.iter().product(), 3)).unwrap(),
                    Tensor::new(sb.clone(), values(sb.iter().product(), 4)).unwrap(),
                );
                let relu = Tensor::new(
                    sa.clone(),
                    xa.data()
                        .iter()
                        .map(|&v| if v > 0.0 { v } else { 0.0 })
                        .collect(),
                )
                .unwrap();
                let want: Vec<u32> = if swap {
                    direct(&Op::Add, &yb, &relu, &out)
                } else {
                    direct(&Op::Add, &relu, &yb, &out)
                }
                .iter()
                .map(|v| v.to_bits())
                .collect();
                let inputs = [xa, yb];
                let fast = CpuBackend.compile(&graph).unwrap().run(&inputs).unwrap();
                assert_eq!(bits(&fast[0]), want, "swap={swap} {sa:?} {sb:?}");
            }
        }
    }
}

#[test]
fn a_shared_operand_and_a_constant_broadcast_too() {
    // x + x, and a constant bias row added to a matrix: the common bias-add shape.
    let mut g = GraphBuilder::new(Limits::default());
    let x = g.input("x", vec![2, 3]).unwrap();
    let bias = g.constant("bias", vec![3], vec![10.0, 20.0, 30.0]).unwrap();
    let col = g.constant("col", vec![2, 1], vec![100.0, 200.0]).unwrap();
    let twice = g.node("twice", Op::Add, &[x, x]).unwrap();
    let biased = g.node("biased", Op::Add, &[twice, bias]).unwrap();
    let scaled = g.node("scaled", Op::Mul, &[biased, col]).unwrap();
    let graph = g.finish(&[scaled]).unwrap();
    let input = Tensor::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let want = [1200.0, 2400.0, 3600.0, 3600.0, 6000.0, 8400.0];
    for run in [
        ReferenceBackend
            .compile(&graph)
            .unwrap()
            .run(std::slice::from_ref(&input)),
        CpuBackend
            .compile(&graph)
            .unwrap()
            .run(std::slice::from_ref(&input)),
    ] {
        assert_eq!(run.unwrap()[0].data(), &want);
    }
}
