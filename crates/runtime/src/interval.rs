//! Box analysis with plain intervals: for every input element in `[lo, hi]`, an
//! enclosure `[L, U]` of each output element that contains the exact real-number
//! result and the float32 result of the reference backend.
//!
//! Each operator maps the enclosures of its operands to an enclosure of its result
//! (interval arithmetic on the real operation), then widens each end by the rounding
//! of the float32 operation: round to nearest is monotone and within half an ulp, so
//! `fl(v)` for `v` in `[a, b]` lies in `[a - half_ulp(a), b + half_ulp(b)]`. The
//! interval arithmetic itself runs in float64 and rounds every endpoint outward
//! (`outward`), so its own rounding can only widen the result.
//!
//! This is sound and simple but loses correlation: a value used twice is treated as
//! two independent ones, so the enclosure grows with depth. It is the fallback and
//! the cross-check for tighter forms.
//!
//! A NaN endpoint or a box with `lo > hi` is an error. An infinite endpoint is
//! allowed and makes what depends on it unbounded.

use crate::analyze::{
    SIGMOID_ABSOLUTE, SIGMOID_RELATIVE, SIGMOID_TINY_BELOW, SIGMOID_TINY_BOUND, half_ulp,
    operand_maps,
};
use crate::{Plan, Tensor, kernels};
use oxfuse_core::{Error, Graph, Op};

/// The result of [`analyze_box`] for one output: `lower[i] <= x <= upper[i]` for the
/// exact result and the reference float32 result of element `i`, for every input in
/// the box. An infinite end means unbounded on that side.
#[derive(Debug, Clone, PartialEq)]
pub struct Enclosure {
    pub name: String,
    pub shape: Vec<usize>,
    pub lo: Vec<f64>,
    pub hi: Vec<f64>,
}

/// A closed interval of float64, ends possibly infinite, never NaN.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Interval {
    pub(crate) lo: f64,
    pub(crate) hi: f64,
}

const WHOLE: Interval = Interval {
    lo: f64::NEG_INFINITY,
    hi: f64::INFINITY,
};

/// The largest finite float32, as a float64.
const F32_MAX: f64 = f32::MAX as f64;

/// Relative slack for a float64 evaluation of the sigmoid (an `exp` within a few ulps, a
/// sum and a quotient).
const SIGMOID_F64_SLACK: f64 = 16.0 * f64::EPSILON;

fn up64(x: f64) -> f64 {
    if x.is_infinite() {
        x
    } else {
        x + (x.abs() * 2.0 * f64::EPSILON + f64::MIN_POSITIVE)
    }
}

fn down64(x: f64) -> f64 {
    if x.is_infinite() {
        x
    } else {
        x - (x.abs() * 2.0 * f64::EPSILON + f64::MIN_POSITIVE)
    }
}

impl Interval {
    pub(crate) fn point(x: f64) -> Interval {
        Interval { lo: x, hi: x }
    }

    /// Two float64 endpoints just computed, rounded outward. NaN means the operation had
    /// no answer (zero times infinity): unbounded.
    fn outward(lo: f64, hi: f64) -> Interval {
        if lo.is_nan() || hi.is_nan() {
            return WHOLE;
        }
        Interval {
            lo: down64(lo),
            hi: up64(hi),
        }
    }
}

/// The float32 rounding of an operation whose real result lies in `span`.
fn round32(span: Interval) -> Interval {
    if span.lo.is_nan() || span.hi.is_nan() {
        return WHOLE;
    }
    // Rounding a real beyond the float32 range gives the largest finite value or an
    // infinity: at least `-F32_MAX` from below, at most `F32_MAX` from above.
    let lo = if span.lo < -F32_MAX {
        f64::NEG_INFINITY
    } else if span.lo > F32_MAX {
        F32_MAX
    } else {
        down64(span.lo - half_ulp(span.lo as f32))
    };
    let hi = if span.hi > F32_MAX {
        f64::INFINITY
    } else if span.hi < -F32_MAX {
        -F32_MAX
    } else {
        up64(span.hi + half_ulp(span.hi as f32))
    };
    Interval { lo, hi }
}

fn add(a: Interval, b: Interval) -> Interval {
    round32(Interval::outward(a.lo + b.lo, a.hi + b.hi))
}

fn sub(a: Interval, b: Interval) -> Interval {
    round32(Interval::outward(a.lo - b.hi, a.hi - b.lo))
}

/// The product of two intervals, before any float32 rounding.
fn product(a: Interval, b: Interval) -> Interval {
    let corners = [a.lo * b.lo, a.lo * b.hi, a.hi * b.lo, a.hi * b.hi];
    if corners.iter().any(|c| c.is_nan()) {
        return WHOLE;
    }
    let lo = corners.iter().copied().fold(f64::INFINITY, f64::min);
    let hi = corners.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    Interval::outward(lo, hi)
}

fn mul(a: Interval, b: Interval) -> Interval {
    round32(product(a, b))
}

/// Unbounded when the divisor's interval reaches zero.
fn div(a: Interval, b: Interval) -> Interval {
    if b.lo <= 0.0 && b.hi >= 0.0 {
        return WHOLE;
    }
    let corners = [a.lo / b.lo, a.lo / b.hi, a.hi / b.lo, a.hi / b.hi];
    if corners.iter().any(|c| c.is_nan()) {
        return WHOLE;
    }
    let lo = corners.iter().copied().fold(f64::INFINITY, f64::min);
    let hi = corners.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    round32(Interval::outward(lo, hi))
}

fn sigmoid(x: Interval) -> Interval {
    let logistic = |v: f64| 1.0 / (1.0 + (-v).exp());
    let (sl, sh) = (logistic(x.lo), logistic(x.hi));
    let lo = sl * (1.0 - SIGMOID_F64_SLACK) - (SIGMOID_RELATIVE * sl + SIGMOID_ABSOLUTE);
    let hi = sh * (1.0 + SIGMOID_F64_SLACK) + (SIGMOID_RELATIVE * sh + SIGMOID_ABSOLUTE);
    // Below this input the relative model of the own `exp` stops holding; both results
    // are smaller than 2^-125.
    let tiny = if x.lo < f64::from(SIGMOID_TINY_BELOW) {
        SIGMOID_TINY_BOUND
    } else {
        0.0
    };
    Interval::outward(lo - tiny, hi + tiny)
}

fn matmul(a: &[Interval], b: &[Interval], shapes: &[&[usize]]) -> Vec<Interval> {
    let (m, k, n) = (shapes[0][0], shapes[0][1], shapes[1][1]);
    let mut out = Vec::with_capacity(m * n);
    for i in 0..m {
        for j in 0..n {
            // The kernel's chain: one fused multiply-add per step in ascending `k`, from +0.0,
            // so each step is the exact product plus the running sum, rounded once.
            let mut acc = Interval::point(0.0);
            for p in 0..k {
                let term = product(a[i * k + p], b[p * n + j]);
                acc = round32(Interval::outward(acc.lo + term.lo, acc.hi + term.hi));
            }
            out.push(acc);
        }
    }
    out
}

fn step(op: &Op, v: &[&[Interval]], shapes: &[&[usize]], out_len: usize) -> Vec<Interval> {
    let binary = |f: fn(Interval, Interval) -> Interval| {
        let (ia, ib) = operand_maps(shapes, out_len);
        (0..out_len).map(|i| f(v[0][ia[i]], v[1][ib[i]])).collect()
    };
    match op {
        Op::Relu => v[0]
            .iter()
            .map(|x| Interval {
                lo: x.lo.max(0.0),
                hi: x.hi.max(0.0),
            })
            .collect(),
        Op::Sigmoid => v[0].iter().map(|&x| sigmoid(x)).collect(),
        Op::Neg => v[0]
            .iter()
            .map(|x| Interval {
                lo: -x.hi,
                hi: -x.lo,
            })
            .collect(),
        Op::Reshape { .. } => v[0].to_vec(),
        Op::Transpose { perm } => kernels::transpose(v[0], shapes[0], perm),
        Op::Add => binary(add),
        Op::Sub => binary(sub),
        Op::Mul => binary(mul),
        Op::Div => binary(div),
        // The result is one of the operands: nothing rounds.
        Op::Min => binary(|a, b| Interval {
            lo: a.lo.min(b.lo),
            hi: a.hi.min(b.hi),
        }),
        Op::Max => binary(|a, b| Interval {
            lo: a.lo.max(b.lo),
            hi: a.hi.max(b.hi),
        }),
        Op::MatMul => matmul(v[0], v[1], shapes),
    }
}

fn check_box(name: &str, shape: &[usize], lo: &Tensor, hi: &Tensor) -> Result<(), Error> {
    for t in [lo, hi] {
        if t.shape() != shape {
            return Err(Error::InputShape {
                name: name.to_owned(),
                expected: shape.to_vec(),
                found: t.shape().to_vec(),
            });
        }
    }
    for (i, (&l, &h)) in lo.data().iter().zip(hi.data()).enumerate() {
        let reason = if l.is_nan() || h.is_nan() {
            Some(format!("element {i} has a NaN endpoint"))
        } else if l > h {
            Some(format!(
                "element {i} has its lower end {l} above its upper end {h}"
            ))
        } else {
            None
        };
        if let Some(reason) = reason {
            return Err(Error::InvalidBox {
                name: name.to_owned(),
                reason,
            });
        }
    }
    Ok(())
}

/// Run `graph` over the input box `[lo, hi]` and enclose every output element.
pub fn analyze_box(graph: &Graph, lo: &[Tensor], hi: &[Tensor]) -> Result<Vec<Enclosure>, Error> {
    let plan = Plan::build(graph);
    for found in [lo.len(), hi.len()] {
        if found != plan.inputs.len() {
            return Err(Error::InputCount {
                expected: plan.inputs.len(),
                found,
            });
        }
    }
    let mut slots: Vec<Vec<Interval>> = vec![Vec::new(); plan.slot_shapes.len()];
    for ((binding, l), h) in plan.inputs.iter().zip(lo).zip(hi) {
        check_box(&binding.name, &binding.shape, l, h)?;
        slots[binding.slot] = l
            .data()
            .iter()
            .zip(h.data())
            .map(|(&l, &h)| Interval {
                lo: f64::from(l),
                hi: f64::from(h),
            })
            .collect();
    }
    for (slot, data) in &plan.constants {
        slots[*slot] = data
            .iter()
            .map(|&x| Interval::point(f64::from(x)))
            .collect();
    }
    for s in &plan.steps {
        let shapes: Vec<&[usize]> = s
            .inputs
            .iter()
            .map(|&i| plan.slot_shapes[i].as_slice())
            .collect();
        let operands: Vec<&[Interval]> = s.inputs.iter().map(|&i| slots[i].as_slice()).collect();
        let len = plan.slot_shapes[s.output].iter().product::<usize>();
        let out = step(&s.op, &operands, &shapes, len);
        slots[s.output] = out;
    }
    Ok(plan
        .outputs
        .iter()
        .zip(graph.output_names())
        .map(|(&slot, name)| Enclosure {
            name: name.clone(),
            shape: plan.slot_shapes[slot].clone(),
            lo: slots[slot].iter().map(|span| span.lo).collect(),
            hi: slots[slot].iter().map(|span| span.hi).collect(),
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analyze::analyze;
    use crate::analyze::tests::{exact, random_case};
    use crate::{Backend, ReferenceBackend};
    use oxfuse_core::{GraphBuilder, Limits};

    fn t(shape: &[usize], data: &[f32]) -> Tensor {
        Tensor::new(shape.to_vec(), data.to_vec()).unwrap()
    }

    /// A graph of one op over two inputs of the given shapes.
    fn binary(op: Op, a: &[usize], b: &[usize]) -> Graph {
        let mut g = GraphBuilder::new(Limits::default());
        let x = g.input("x", a.to_vec()).unwrap();
        let y = g.input("y", b.to_vec()).unwrap();
        let z = g.node("z", op, &[x, y]).unwrap();
        g.finish(&[z]).unwrap()
    }

    fn boxed(graph: &Graph, lo: &[Tensor], hi: &[Tensor]) -> Enclosure {
        analyze_box(graph, lo, hi).unwrap().remove(0)
    }

    /// Both ends within `slack` of the exact endpoints (the float64 and float32 rounding
    /// widen them a little, never narrow them).
    fn assert_ends(e: &Enclosure, i: usize, lo: f64, hi: f64) {
        let slack = 1e-5 * (1.0 + lo.abs().max(hi.abs()));
        assert!(
            e.lo[i] <= lo && lo - e.lo[i] <= slack,
            "lo {} vs {lo}",
            e.lo[i]
        );
        assert!(
            e.hi[i] >= hi && e.hi[i] - hi <= slack,
            "hi {} vs {hi}",
            e.hi[i]
        );
    }

    #[test]
    fn add_and_sub_read_the_right_ends_of_each_operand() {
        let (lo, hi) = (
            [t(&[2], &[1.0, 5.0]), t(&[2], &[10.0, -3.0])],
            [t(&[2], &[2.0, 6.0]), t(&[2], &[20.0, -1.0])],
        );
        let add = boxed(&binary(Op::Add, &[2], &[2]), &lo, &hi);
        assert_ends(&add, 0, 11.0, 22.0);
        assert_ends(&add, 1, 2.0, 5.0);
        let sub = boxed(&binary(Op::Sub, &[2], &[2]), &lo, &hi);
        assert_ends(&sub, 0, -19.0, -8.0);
        assert_ends(&sub, 1, 6.0, 9.0);
    }

    #[test]
    fn mul_takes_the_extreme_of_four_corners() {
        let lo = [t(&[1], &[-2.0]), t(&[1], &[-5.0])];
        let hi = [t(&[1], &[3.0]), t(&[1], &[4.0])];
        // Corners 10, -8, -15, 12.
        assert_ends(
            &boxed(&binary(Op::Mul, &[1], &[1]), &lo, &hi),
            0,
            -15.0,
            12.0,
        );
    }

    #[test]
    fn div_by_a_positive_interval_and_through_zero() {
        let g = binary(Op::Div, &[1], &[1]);
        let away = boxed(
            &g,
            &[t(&[1], &[1.0]), t(&[1], &[2.0])],
            &[t(&[1], &[2.0]), t(&[1], &[4.0])],
        );
        assert_ends(&away, 0, 0.25, 1.0);
        let negative = boxed(
            &g,
            &[t(&[1], &[1.0]), t(&[1], &[-4.0])],
            &[t(&[1], &[2.0]), t(&[1], &[-2.0])],
        );
        assert_ends(&negative, 0, -1.0, -0.25);
        for (l, h) in [(-1.0, 1.0), (0.0, 2.0), (-2.0, 0.0), (0.0, 0.0)] {
            let e = boxed(
                &g,
                &[t(&[1], &[1.0]), t(&[1], &[l])],
                &[t(&[1], &[2.0]), t(&[1], &[h])],
            );
            assert_eq!(
                (e.lo[0], e.hi[0]),
                (f64::NEG_INFINITY, f64::INFINITY),
                "divisor [{l}, {h}]"
            );
        }
    }

    #[test]
    fn min_max_neg_relu_have_no_rounding() {
        let lo = [t(&[1], &[1.0]), t(&[1], &[0.5])];
        let hi = [t(&[1], &[4.0]), t(&[1], &[2.0])];
        let min = boxed(&binary(Op::Min, &[1], &[1]), &lo, &hi);
        assert_eq!((min.lo[0], min.hi[0]), (0.5, 2.0));
        let max = boxed(&binary(Op::Max, &[1], &[1]), &lo, &hi);
        assert_eq!((max.lo[0], max.hi[0]), (1.0, 4.0));
        let mut g = GraphBuilder::new(Limits::default());
        let x = g.input("x", vec![2]).unwrap();
        let n = g.node("n", Op::Neg, &[x]).unwrap();
        let r = g.node("r", Op::Relu, &[x]).unwrap();
        let graph = g.finish(&[n, r]).unwrap();
        let out = analyze_box(&graph, &[t(&[2], &[-3.0, 1.0])], &[t(&[2], &[-1.0, 2.0])]).unwrap();
        assert_eq!((out[0].lo[0], out[0].hi[0]), (1.0, 3.0));
        assert_eq!((out[0].lo[1], out[0].hi[1]), (-2.0, -1.0));
        assert_eq!((out[1].lo[0], out[1].hi[0]), (0.0, 0.0));
        assert_eq!((out[1].lo[1], out[1].hi[1]), (1.0, 2.0));
    }

    #[test]
    fn sigmoid_of_a_box_is_monotone_and_covers_the_tiny_range() {
        let mut g = GraphBuilder::new(Limits::default());
        let x = g.input("x", vec![3]).unwrap();
        let s = g.node("s", Op::Sigmoid, &[x]).unwrap();
        let graph = g.finish(&[s]).unwrap();
        let e = boxed(
            &graph,
            &[t(&[3], &[-1.0, 0.0, -100.0])],
            &[t(&[3], &[1.0, 0.0, -90.0])],
        );
        let logistic = |v: f64| 1.0 / (1.0 + (-v).exp());
        assert!(e.lo[0] <= logistic(-1.0) && e.hi[0] >= logistic(1.0));
        assert!(e.hi[0] - e.lo[0] < 0.6);
        assert!(e.lo[1] <= 0.5 && e.hi[1] >= 0.5 && e.hi[1] - e.lo[1] < 1e-5);
        // The reference value for an input below -87 is below 2^-125 and inside.
        let run = ReferenceBackend
            .compile(&graph)
            .unwrap()
            .run(&[t(&[3], &[0.0, 0.0, -95.0])])
            .unwrap();
        let v = f64::from(run[0].data()[2]);
        assert!(e.lo[2] <= v && v <= e.hi[2]);
    }

    #[test]
    fn matmul_with_a_constant_matrix_sums_the_terms_exactly() {
        let mut g = GraphBuilder::new(Limits::default());
        let x = g.input("x", vec![1, 2]).unwrap();
        let w = g.constant("w", vec![2, 1], vec![1.0, -1.0]).unwrap();
        let y = g.node("y", Op::MatMul, &[x, w]).unwrap();
        let graph = g.finish(&[y]).unwrap();
        // a0 - a1 with a0 in [1, 2] and a1 in [3, 4].
        let e = boxed(
            &graph,
            &[t(&[1, 2], &[1.0, 3.0])],
            &[t(&[1, 2], &[2.0, 4.0])],
        );
        assert_ends(&e, 0, -3.0, -1.0);
    }

    #[test]
    fn matmul_reads_rows_and_columns_from_their_own_places() {
        let mut g = GraphBuilder::new(Limits::default());
        let x = g.input("x", vec![2, 3]).unwrap();
        let w = g
            .constant("w", vec![3, 2], vec![1.0, 10.0, 100.0, 1000.0, 3.0, 5.0])
            .unwrap();
        let y = g.node("y", Op::MatMul, &[x, w]).unwrap();
        let graph = g.finish(&[y]).unwrap();
        let point = t(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let e = boxed(
            &graph,
            std::slice::from_ref(&point),
            std::slice::from_ref(&point),
        );
        // Row 0: [1*1 + 2*100 + 3*3, 1*10 + 2*1000 + 3*5]; row 1 likewise.
        let want = [210.0, 2025.0, 4.0 + 500.0 + 18.0, 40.0 + 5000.0 + 30.0];
        for (i, w) in want.iter().enumerate() {
            assert_ends(&e, i, *w, *w);
        }
    }

    #[test]
    fn transpose_and_reshape_move_each_enclosure_with_its_element() {
        let mut g = GraphBuilder::new(Limits::default());
        let x = g.input("x", vec![2, 3]).unwrap();
        let tr = g
            .node("t", Op::Transpose { perm: vec![1, 0] }, &[x])
            .unwrap();
        let re = g.node("r", Op::Reshape { shape: vec![6] }, &[x]).unwrap();
        let graph = g.finish(&[tr, re]).unwrap();
        let lo = t(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let hi = t(&[2, 3], &[1.5, 2.5, 3.5, 4.5, 5.5, 6.5]);
        let out = analyze_box(&graph, &[lo], &[hi]).unwrap();
        assert_eq!(out[0].lo, [1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
        assert_eq!(out[0].hi, [1.5, 4.5, 2.5, 5.5, 3.5, 6.5]);
        assert_eq!(out[1].lo, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn broadcast_operands_use_their_own_elements() {
        let g = binary(Op::Add, &[2, 2], &[2]);
        let lo = [t(&[2, 2], &[0.0, 0.0, 0.0, 0.0]), t(&[2], &[10.0, 20.0])];
        let hi = [t(&[2, 2], &[1.0, 1.0, 1.0, 1.0]), t(&[2], &[11.0, 21.0])];
        let e = boxed(&g, &lo, &hi);
        assert_ends(&e, 0, 10.0, 12.0);
        assert_ends(&e, 1, 20.0, 22.0);
        assert_ends(&e, 2, 10.0, 12.0);
        assert_ends(&e, 3, 20.0, 22.0);
    }

    #[test]
    fn an_infinite_endpoint_makes_what_depends_on_it_unbounded() {
        let g = binary(Op::Add, &[1], &[1]);
        let e = boxed(
            &g,
            &[t(&[1], &[0.0]), t(&[1], &[1.0])],
            &[t(&[1], &[f32::INFINITY]), t(&[1], &[1.0])],
        );
        assert!(e.hi[0].is_infinite() && e.hi[0] > 0.0);
        assert!(e.lo[0].is_finite() && e.lo[0] <= 1.0 && e.lo[0] > 0.99);
        // Zero times an unbounded value has no answer: unbounded on both sides.
        let m = boxed(
            &binary(Op::Mul, &[1], &[1]),
            &[t(&[1], &[0.0]), t(&[1], &[0.0])],
            &[t(&[1], &[f32::INFINITY]), t(&[1], &[0.0])],
        );
        assert_eq!((m.lo[0], m.hi[0]), (f64::NEG_INFINITY, f64::INFINITY));
    }

    #[test]
    fn a_result_beyond_float32_is_unbounded_on_that_side() {
        let g = binary(Op::Mul, &[1], &[1]);
        let e = boxed(
            &g,
            &[t(&[1], &[1.0e30]), t(&[1], &[1.0e30])],
            &[t(&[1], &[2.0e30]), t(&[1], &[2.0e30])],
        );
        // Rounding 1e60 gives the largest finite float32 or infinity.
        assert_eq!(e.hi[0], f64::INFINITY);
        assert_eq!(e.lo[0], f64::from(f32::MAX));
        let below = boxed(
            &g,
            &[t(&[1], &[-2.0e30]), t(&[1], &[1.0e30])],
            &[t(&[1], &[-1.0e30]), t(&[1], &[2.0e30])],
        );
        assert_eq!(below.lo[0], f64::NEG_INFINITY);
        assert_eq!(below.hi[0], -f64::from(f32::MAX));
    }

    #[test]
    fn invalid_boxes_are_errors() {
        let g = binary(Op::Add, &[2], &[2]);
        let ok = [t(&[2], &[0.0, 0.0]), t(&[2], &[0.0, 0.0])];
        let bad = |lo: [Tensor; 2], hi: [Tensor; 2]| analyze_box(&g, &lo, &hi).unwrap_err();
        let inverted = bad(
            [t(&[2], &[0.0, 2.0]), t(&[2], &[0.0, 0.0])],
            [t(&[2], &[1.0, 1.0]), t(&[2], &[0.0, 0.0])],
        );
        assert!(
            matches!(&inverted, Error::InvalidBox { name, reason } if name == "x" && reason.contains("element 1")),
            "{inverted:?}"
        );
        let nan_hi = bad(
            ok.clone(),
            [t(&[2], &[0.0, 0.0]), t(&[2], &[0.0, f32::NAN])],
        );
        assert!(
            matches!(&nan_hi, Error::InvalidBox { name, .. } if name == "y"),
            "{nan_hi:?}"
        );
        let nan_lo = bad(
            [t(&[2], &[f32::NAN, 0.0]), t(&[2], &[0.0, 0.0])],
            ok.clone(),
        );
        assert!(matches!(nan_lo, Error::InvalidBox { .. }));
        let wrong_shape = bad(
            ok.clone(),
            [t(&[2], &[0.0, 0.0]), t(&[3], &[0.0, 0.0, 0.0])],
        );
        assert!(matches!(&wrong_shape, Error::InputShape { name, .. } if name == "y"));
        let wrong_lo = bad([t(&[3], &[0.0; 3]), t(&[2], &[0.0, 0.0])], ok.clone());
        assert!(matches!(&wrong_lo, Error::InputShape { name, .. } if name == "x"));
        assert!(matches!(
            analyze_box(&g, &ok[..1], &ok).unwrap_err(),
            Error::InputCount {
                expected: 2,
                found: 1
            }
        ));
        assert!(matches!(
            analyze_box(&g, &ok, &ok[..1]).unwrap_err(),
            Error::InputCount {
                expected: 2,
                found: 1
            }
        ));
        assert!(
            analyze_box(&g, &ok, &ok).is_ok(),
            "equal ends are a valid point box"
        );
    }

    /// A box around `x` with a different radius for every element.
    fn box_around(x: &Tensor, fraction: f32) -> (Tensor, Tensor) {
        let radius =
            |i: usize, v: f32| v.abs() * fraction * (1 + i % 3) as f32 + 0.01 * (1 + i % 2) as f32;
        let lo = x
            .data()
            .iter()
            .enumerate()
            .map(|(i, &v)| v - radius(i, v))
            .collect();
        let hi = x
            .data()
            .iter()
            .enumerate()
            .map(|(i, &v)| v + radius(i, v))
            .collect();
        (
            Tensor::new(x.shape().to_vec(), lo).unwrap(),
            Tensor::new(x.shape().to_vec(), hi).unwrap(),
        )
    }

    fn splitmix(state: &mut u64) -> u64 {
        *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = *state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// 64 points of the box: both corners, random corners and random interior points.
    fn samples(lo: &Tensor, hi: &Tensor, seed: u64) -> Vec<Tensor> {
        let mut state = seed;
        let (l, h) = (lo.data(), hi.data());
        let mut points = vec![lo.clone(), hi.clone()];
        for n in 0..62 {
            let data = l
                .iter()
                .zip(h)
                .map(|(&a, &b)| {
                    let r = splitmix(&mut state);
                    let u = (r >> 40) as f32 / (1u64 << 24) as f32;
                    let v = if n % 2 == 0 {
                        if r & 1 == 0 { a } else { b }
                    } else {
                        a + (b - a) * u
                    };
                    v.clamp(a, b)
                })
                .collect();
            points.push(Tensor::new(lo.shape().to_vec(), data).unwrap());
        }
        points
    }

    #[test]
    fn sampled_points_stay_inside_the_enclosure_for_300_random_graphs() {
        for seed in 0..300 {
            let (graph, inputs) = random_case(seed);
            let (lo, hi) = box_around(&inputs[0], 0.05);
            let enclosure = boxed(&graph, std::slice::from_ref(&lo), std::slice::from_ref(&hi));
            let exe = ReferenceBackend.compile(&graph).unwrap();
            for point in samples(&lo, &hi, seed) {
                let got = exe.run(std::slice::from_ref(&point)).unwrap().remove(0);
                let want = exact(&graph, std::slice::from_ref(&point)).remove(0);
                for (i, (&v, &x)) in got.data().iter().zip(&want).enumerate() {
                    let v = f64::from(v);
                    let (l, u) = (enclosure.lo[i], enclosure.hi[i]);
                    assert!(
                        l <= v && v <= u,
                        "seed {seed} element {i}: reference {v} outside [{l}, {u}]"
                    );
                    assert!(
                        l <= x && x <= u,
                        "seed {seed} element {i}: exact {x} outside [{l}, {u}]"
                    );
                }
            }
        }
    }

    #[test]
    fn a_point_box_encloses_the_exact_result_and_is_not_much_wider_than_the_point_bound() {
        let mut widest_ratio = 0.0f64;
        for seed in 0..300 {
            let (graph, inputs) = random_case(seed);
            let e = boxed(&graph, &inputs, &inputs);
            let point = analyze(&graph, &inputs).unwrap().remove(0);
            let want = exact(&graph, &inputs).remove(0);
            for (i, &x) in want.iter().enumerate() {
                let (l, u) = (e.lo[i], e.hi[i]);
                assert!(
                    l <= x && x <= u,
                    "seed {seed} element {i}: {x} outside [{l}, {u}]"
                );
                let v = f64::from(point.values[i]);
                assert!(
                    l <= v && v <= u,
                    "seed {seed} element {i}: value {v} outside [{l}, {u}]"
                );
                if point.bounds[i] > 0.0 && point.bounds[i].is_finite() {
                    widest_ratio = widest_ratio.max((u - l) / (2.0 * point.bounds[i]));
                }
            }
        }
        // The first-order point bound is the tighter statement; a point box may be somewhat
        // wider where the interval form rounds at the larger endpoint.
        assert!(widest_ratio <= 3.0, "widest ratio {widest_ratio}");
    }

    #[test]
    fn a_wider_box_never_gives_a_narrower_enclosure() {
        for seed in 0..100 {
            let (graph, inputs) = random_case(seed);
            let (small_lo, small_hi) = box_around(&inputs[0], 0.01);
            let (big_lo, big_hi) = box_around(&inputs[0], 0.1);
            let small = boxed(
                &graph,
                std::slice::from_ref(&small_lo),
                std::slice::from_ref(&small_hi),
            );
            let big = boxed(
                &graph,
                std::slice::from_ref(&big_lo),
                std::slice::from_ref(&big_hi),
            );
            for i in 0..small.lo.len() {
                assert!(
                    big.lo[i] <= small.lo[i] && small.hi[i] <= big.hi[i],
                    "seed {seed} element {i}"
                );
            }
        }
    }

    #[test]
    fn up64_and_down64_always_move_outward() {
        for x in [
            0.0,
            1.0,
            -1.0,
            1e-300,
            -1e-300,
            3.7e200,
            -3.7e200,
            f64::MIN_POSITIVE,
        ] {
            assert!(up64(x) > x, "{x}");
            assert!(down64(x) < x, "{x}");
        }
        assert_eq!(up64(f64::INFINITY), f64::INFINITY);
        assert_eq!(down64(f64::NEG_INFINITY), f64::NEG_INFINITY);
    }

    #[test]
    fn round32_widens_by_the_ulp_at_each_end() {
        let r = round32(Interval { lo: 1.0, hi: 2.0 });
        assert!(r.lo < 1.0 && 1.0 - r.lo < 1e-6);
        assert!(r.hi > 2.0 && r.hi - 2.0 < 2e-6);
        // The upper end sits in a binade twice as coarse as the lower end.
        assert!(r.hi - 2.0 > 1.0 - r.lo);
        let over = round32(Interval {
            lo: 0.0,
            hi: 4.0e38,
        });
        assert_eq!(over.hi, f64::INFINITY);
        assert!(over.lo.is_finite());
        let under = round32(Interval {
            lo: -4.0e38,
            hi: 0.0,
        });
        assert_eq!(under.lo, f64::NEG_INFINITY);
        let beyond = round32(Interval {
            lo: 5.0e38,
            hi: 6.0e38,
        });
        assert_eq!((beyond.lo, beyond.hi), (f64::from(f32::MAX), f64::INFINITY));
        assert_eq!(round32(WHOLE), WHOLE);
    }

    #[test]
    fn a_nan_in_either_end_means_unbounded() {
        for (lo, hi) in [(f64::NAN, 1.0), (1.0, f64::NAN), (f64::NAN, f64::NAN)] {
            assert_eq!(Interval::outward(lo, hi), WHOLE, "outward({lo}, {hi})");
            assert_eq!(round32(Interval { lo, hi }), WHOLE, "round32({lo}, {hi})");
        }
    }

    /// The edges of the float32 range: rounding a real exactly at the largest finite value
    /// stays finite on both sides, and one just past it does not.
    #[test]
    fn round32_at_the_edges_of_the_float32_range() {
        let max = F32_MAX;
        let at_top = round32(Interval { lo: max, hi: max });
        assert!(
            at_top.lo < max && at_top.lo > max * (1.0 - 1e-6),
            "{at_top:?}"
        );
        assert!(at_top.hi > max && at_top.hi.is_finite(), "{at_top:?}");
        let at_bottom = round32(Interval { lo: -max, hi: -max });
        assert!(
            at_bottom.lo < -max && at_bottom.lo.is_finite(),
            "{at_bottom:?}"
        );
        assert!(
            at_bottom.hi > -max && at_bottom.hi < -max * (1.0 - 1e-6),
            "{at_bottom:?}"
        );
        let past_top = round32(Interval {
            lo: max * 1.0001,
            hi: max * 1.0001,
        });
        assert_eq!((past_top.lo, past_top.hi), (max, f64::INFINITY));
        let past_bottom = round32(Interval {
            lo: -max * 1.0001,
            hi: -max * 1.0001,
        });
        assert_eq!((past_bottom.lo, past_bottom.hi), (f64::NEG_INFINITY, -max));
    }

    fn sigmoid_of_point(x: f32) -> Interval {
        sigmoid(Interval::point(f64::from(x)))
    }

    fn logistic(v: f64) -> f64 {
        1.0 / (1.0 + (-v).exp())
    }

    /// Where the value is large the relative term decides the width, where it is tiny the
    /// absolute subnormal spacing does: each is pinned from both sides.
    #[test]
    fn sigmoid_widens_by_the_relative_and_the_absolute_terms() {
        let s = logistic(0.0);
        let e = sigmoid_of_point(0.0);
        assert!(
            e.lo <= s * (1.0 - SIGMOID_RELATIVE) && e.lo >= s * (1.0 - SIGMOID_RELATIVE) - 1e-12,
            "{e:?}"
        );
        assert!(
            e.hi >= s * (1.0 + SIGMOID_RELATIVE) && e.hi <= s * (1.0 + SIGMOID_RELATIVE) + 1e-12,
            "{e:?}"
        );
        // At -80 the value is about 1.8e-35, so the absolute term (2^-149) is the part of the
        // width that a sign or operator slip would remove.
        let s = logistic(-80.0);
        let e = sigmoid_of_point(-80.0);
        assert!(
            e.lo <= s * (1.0 - SIGMOID_RELATIVE) - SIGMOID_ABSOLUTE,
            "{e:?}"
        );
        assert!(
            e.hi >= s * (1.0 + SIGMOID_RELATIVE) + SIGMOID_ABSOLUTE,
            "{e:?}"
        );
    }

    #[test]
    fn sigmoid_adds_the_tiny_range_only_below_the_threshold() {
        // Exactly at the threshold the relative model still holds: no extra width.
        let s = logistic(-87.0);
        let at = sigmoid_of_point(-87.0);
        assert!(
            at.hi <= s * (1.0 + SIGMOID_RELATIVE) * (1.0 + 1e-9) + SIGMOID_ABSOLUTE * 2.0,
            "{at:?}"
        );
        // Just below it the enclosure carries the 2^-125 range.
        let below = sigmoid_of_point(-87.5);
        assert!(
            below.hi >= logistic(-87.5) + SIGMOID_TINY_BOUND,
            "{below:?}"
        );
    }
}
