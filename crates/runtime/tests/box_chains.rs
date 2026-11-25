//! Box analysis over random chains of operators, boxes of extreme magnitude and results that are
//! NaN: every value the reference backend returns for a point of the box lies in the enclosure,
//! and an element whose result is NaN has an enclosure unbounded on both sides (nothing finite
//! may promise anything about a NaN).

use oxfuse_core::{Graph, GraphBuilder, Limits, Op, ValueId};
use oxfuse_runtime::{Backend, ReferenceBackend, Tensor, analyze_box_with};

fn splitmix(s: &mut u64) -> u64 {
    *s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *s;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn unit(s: &mut u64) -> f64 {
    (splitmix(s) >> 11) as f64 / (1u64 << 53) as f64
}

fn pick(s: &mut u64, n: usize) -> usize {
    (splitmix(s) % n as u64) as usize
}

/// A magnitude from a list that includes zero, subnormals, the smallest normal, 1e-30, 1e30 and
/// values near the largest float32.
fn magnitude(s: &mut u64) -> f32 {
    match pick(s, 12) {
        0 => 0.0,
        1 => f32::from_bits(1 + (splitmix(s) % 100) as u32),
        2 => f32::MIN_POSITIVE * (1.0 + unit(s) as f32),
        3 => 1e-30,
        4 => 1e30,
        5 => f32::MAX * (0.5 + 0.5 * unit(s) as f32),
        6 => 1.0,
        7 => 10f32.powf((unit(s) * 76.0 - 38.0) as f32),
        8 => 10f32.powf((unit(s) * 6.0 - 3.0) as f32),
        9 => 2f32.powi(pick(s, 20) as i32 - 10),
        _ => (unit(s) * 4.0) as f32,
    }
}

fn signed(s: &mut u64) -> f32 {
    let m = magnitude(s);
    if splitmix(s) & 1 == 0 { m } else { -m }
}

/// One element of a box: a point, symmetric, ordered pair, one-sided at zero, narrow, or ordinary.
fn box_element(s: &mut u64) -> (f32, f32) {
    match pick(s, 7) {
        0 => {
            let v = signed(s);
            (v, v)
        }
        1 => {
            let m = magnitude(s);
            (-m, m)
        }
        2 => {
            let (a, b) = (signed(s), signed(s));
            (a.min(b), a.max(b))
        }
        3 => (0.0, magnitude(s)),
        4 => (-magnitude(s), 0.0),
        5 => {
            let v = signed(s);
            let r = v.abs() * 0.01 * unit(s) as f32;
            (v - r, v + r)
        }
        _ => {
            let v = unit(s) as f32 * 8.0 - 4.0;
            let r = unit(s) as f32;
            (v - r, v + r)
        }
    }
}

fn tensor(shape: &[usize], data: Vec<f32>) -> Tensor {
    Tensor::new(shape.to_vec(), data).unwrap()
}

fn broadcast(a: &[usize], b: &[usize]) -> Vec<usize> {
    let n = a.len().max(b.len());
    (0..n)
        .map(|i| {
            let x = if i + a.len() >= n {
                a[i + a.len() - n]
            } else {
                1
            };
            let y = if i + b.len() >= n {
                b[i + b.len() - n]
            } else {
                1
            };
            x.max(y)
        })
        .collect()
}

/// A graph of 4 to 8 random nodes over a 3x3 and a 3-vector input (with constants mixed in), and
/// the lower and upper tensors of a random box.
fn chain(seed: u64) -> (Graph, Vec<Tensor>, Vec<Tensor>) {
    let mut s = seed.wrapping_mul(0xA24B_AED4_963E_E407) ^ 0x1234;
    let mut g = GraphBuilder::new(Limits::default());
    let x = g.input("x", vec![3, 3]).unwrap();
    let y = g.input("y", vec![3]).unwrap();
    let mut values: Vec<(ValueId, Vec<usize>)> = vec![(x, vec![3, 3]), (y, vec![3])];
    let (mut xl, mut xh, mut yl, mut yh) = (vec![], vec![], vec![], vec![]);
    let narrow = pick(&mut s, 3) == 0;
    for _ in 0..9 {
        let (a, b) = if narrow {
            let v = (unit(&mut s) * 2.0 - 1.0) as f32;
            (v - 0.05, v + 0.05)
        } else {
            box_element(&mut s)
        };
        xl.push(a);
        xh.push(b);
    }
    for _ in 0..3 {
        let (a, b) = box_element(&mut s);
        yl.push(a);
        yh.push(b);
    }
    let wanted = 4 + pick(&mut s, 5);
    let mut outputs = Vec::new();
    let mut constants = 0;
    for attempt in 0..200 {
        if outputs.len() >= wanted {
            break;
        }
        let a = values[pick(&mut s, values.len())].clone();
        let b = if pick(&mut s, 3) == 0 {
            constants += 1;
            let shape = [vec![3, 3], vec![3], vec![]][pick(&mut s, 3)].clone();
            let count: usize = shape.iter().product::<usize>().max(1);
            let mild = pick(&mut s, 2) == 0;
            let data: Vec<f32> = (0..count)
                .map(|_| {
                    if mild {
                        (unit(&mut s) * 4.0 - 2.0) as f32
                    } else {
                        signed(&mut s)
                    }
                })
                .collect();
            (
                g.constant(&format!("c{constants}"), shape.clone(), data)
                    .unwrap(),
                shape,
            )
        } else {
            values[pick(&mut s, values.len())].clone()
        };
        let (op, inputs, shape): (Op, Vec<ValueId>, Vec<usize>) = match pick(&mut s, 14) {
            0 => (Op::Relu, vec![a.0], a.1.clone()),
            1 => (Op::Sigmoid, vec![a.0], a.1.clone()),
            2 => (Op::Neg, vec![a.0], a.1.clone()),
            3 | 4 => (Op::Add, vec![a.0, b.0], broadcast(&a.1, &b.1)),
            5 => (Op::Sub, vec![a.0, b.0], broadcast(&a.1, &b.1)),
            6 | 7 => (Op::Mul, vec![a.0, b.0], broadcast(&a.1, &b.1)),
            8 => (Op::Div, vec![a.0, b.0], broadcast(&a.1, &b.1)),
            9 => (Op::Min, vec![a.0, b.0], broadcast(&a.1, &b.1)),
            10 => (Op::Max, vec![a.0, b.0], broadcast(&a.1, &b.1)),
            11 | 12 => {
                if a.1.len() == 2 && b.1.len() == 2 && a.1[1] == b.1[0] {
                    (Op::MatMul, vec![a.0, b.0], vec![a.1[0], b.1[1]])
                } else {
                    continue;
                }
            }
            _ => {
                if a.1.len() == 2 {
                    (
                        Op::Transpose { perm: vec![1, 0] },
                        vec![a.0],
                        vec![a.1[1], a.1[0]],
                    )
                } else {
                    let n: usize = a.1.iter().product();
                    (Op::Reshape { shape: vec![1, n] }, vec![a.0], vec![1, n])
                }
            }
        };
        if let Ok(id) = g.node(&format!("n{attempt}"), op, &inputs) {
            values.push((id, shape));
            outputs.push(id);
        }
    }
    let graph = g.finish(&outputs).unwrap();
    (
        graph,
        vec![tensor(&[3, 3], xl), tensor(&[3], yl)],
        vec![tensor(&[3, 3], xh), tensor(&[3], yh)],
    )
}

/// A point of the box: a corner, the middle, zero (either sign), or somewhere inside.
fn point(lo: &[Tensor], hi: &[Tensor], s: &mut u64, mode: usize) -> Vec<Tensor> {
    lo.iter()
        .zip(hi)
        .map(|(l, h)| {
            let data = l
                .data()
                .iter()
                .zip(h.data())
                .map(|(&a, &b)| match mode {
                    0 => a,
                    1 => b,
                    2 => {
                        if splitmix(s) & 1 == 0 {
                            a
                        } else {
                            b
                        }
                    }
                    3 => (a / 2.0 + b / 2.0).clamp(a, b),
                    4 => {
                        if a <= 0.0 && b >= 0.0 {
                            0.0
                        } else {
                            a
                        }
                    }
                    5 => {
                        if a <= 0.0 && b >= 0.0 {
                            if splitmix(s) & 1 == 0 { -0.0 } else { 0.0 }
                        } else {
                            b
                        }
                    }
                    _ => (a + ((f64::from(b) - f64::from(a)) * unit(s)) as f32).clamp(a, b),
                })
                .collect();
            tensor(l.shape(), data)
        })
        .collect()
}

#[test]
fn every_reference_value_of_a_random_chain_is_enclosed_and_a_nan_is_unbounded() {
    let (mut checked, mut nans) = (0usize, 0usize);
    for seed in 0..400 {
        let (graph, lo, hi) = chain(seed);
        let exe = ReferenceBackend.compile(&graph).unwrap();
        for max_symbols in [256, 2] {
            let enclosures = analyze_box_with(&graph, &lo, &hi, max_symbols).unwrap();
            let mut s = seed ^ 0xDEAD;
            for n in 0..24 {
                let inputs = point(&lo, &hi, &mut s, n % 7);
                let Ok(got) = exe.run(&inputs) else { continue };
                for (o, e) in enclosures.iter().enumerate() {
                    for i in 0..e.lo.len() {
                        let v = f64::from(got[o].data()[i]);
                        if v.is_nan() {
                            nans += 1;
                            assert!(
                                e.lo[i] == f64::NEG_INFINITY && e.hi[i] == f64::INFINITY,
                                "seed {seed} budget {max_symbols} output {o} element {i}: a NaN result has the enclosure [{:e}, {:e}]",
                                e.lo[i],
                                e.hi[i]
                            );
                        } else {
                            checked += 1;
                            assert!(
                                e.lo[i] <= v && v <= e.hi[i],
                                "seed {seed} budget {max_symbols} output {o} element {i}: reference {v:e} outside [{:e}, {:e}]",
                                e.lo[i],
                                e.hi[i]
                            );
                        }
                    }
                }
            }
        }
    }
    assert!(checked > 100_000, "only {checked} values were checked");
    assert!(
        nans > 100,
        "only {nans} NaN results were met: the generator lost its edge cases"
    );
}

/// `q = x / y` is NaN at x = 0, y = 0, and what is computed from it may be NaN too.
#[test]
fn what_is_computed_from_a_possible_nan_is_unbounded_on_both_sides() {
    let mut g = GraphBuilder::new(Limits::default());
    let x = g.input("x", vec![1]).unwrap();
    let y = g.input("y", vec![1]).unwrap();
    let q = g.node("q", Op::Div, &[x, y]).unwrap();
    let outputs = [
        g.node("sigmoid", Op::Sigmoid, &[q]).unwrap(),
        g.node("relu", Op::Relu, &[q]).unwrap(),
        g.node("min", Op::Min, &[q, x]).unwrap(),
        g.node("max", Op::Max, &[x, q]).unwrap(),
        g.node("neg", Op::Neg, &[q]).unwrap(),
        g.node("plus", Op::Add, &[q, x]).unwrap(),
    ];
    let graph = g.finish(&outputs).unwrap();
    let (lo, hi) = (
        vec![tensor(&[1], vec![0.0]), tensor(&[1], vec![-1.0])],
        vec![tensor(&[1], vec![1.0]), tensor(&[1], vec![1.0])],
    );
    let got = ReferenceBackend
        .compile(&graph)
        .unwrap()
        .run(&[tensor(&[1], vec![0.0]), tensor(&[1], vec![0.0])])
        .unwrap();
    for (o, e) in analyze_box_with(&graph, &lo, &hi, 256)
        .unwrap()
        .iter()
        .enumerate()
    {
        assert!(
            e.lo[0] == f64::NEG_INFINITY && e.hi[0] == f64::INFINITY,
            "{}: [{:e}, {:e}] while the reference gives {}",
            e.name,
            e.lo[0],
            e.hi[0],
            got[o].data()[0]
        );
    }
}
