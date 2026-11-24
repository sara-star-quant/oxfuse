//! Affine forms for box analysis: every element is `c + sum_t g_t * eps_t +- e` with one
//! shared noise symbol `eps_t` in `[-1, 1]` per tracked input element and an independent error
//! radius `e`. Symbols shared between elements keep their correlations, so `x - x` cancels and a
//! value used twice is not counted twice: the enclosure grows far more slowly with depth than
//! plain intervals do (see `docs/analysis.md` for the measured widths).
//!
//! The rules, all sound for the exact real result and for the float32 reference result:
//!
//! - `Add`, `Sub`, `Neg`, `Reshape`, `Transpose`, and `Mul` or `Div` by a constant, `MatMul` with a
//!   constant operand: exact on `c` and `g` (float64 arithmetic, its rounding bounded into `e`);
//!   the float32 rounding of each result, and of each fused step of a `MatMul` chain, adds half an
//!   ulp at the largest magnitude the value can have to `e`.
//! - `Relu`: exact when the sign is decided, else the minimal-area relaxation
//!   `lambda x - lambda l / 2 +- (-lambda l / 2)` with `lambda = u / (u - l)`.
//! - Everything else (`Min`, `Max`, `Sigmoid`, a product or quotient of two varying values, a
//!   `MatMul` of two varying matrices): the plain interval rule on the operands' enclosures,
//!   re-expressed as a form with no symbols (correlation is lost there, never soundness).
//!
//! The result of every output element is intersected with the plain-interval enclosure, so it is
//! never wider than that. Memory is `elements x symbols`: inputs beyond the symbol budget (the ones
//! with the narrowest boxes) are treated as independent errors.

use crate::analyze::{half_ulp, operand_maps};
use crate::interval::{F32_MAX, Interval, WHOLE, down64, step, up64};
use crate::{Plan, Tensor, kernels};
use oxfuse_core::Op;
use std::collections::HashMap;

const EPS: f64 = f64::EPSILON;

/// Most `elements x symbols` entries the forms of one analysis may hold in total.
const MAX_ENTRIES: usize = 1 << 26;

/// An upper bound for a float64 sum of `terms` nonnegative numbers.
fn up_sum(sum: f64, terms: usize) -> f64 {
    up64(sum * (1.0 + (terms as f64 + 2.0) * EPS))
}

/// Half an ulp of the float32 rounding of a real whose magnitude is at most `m`.
fn round_term(m: f64) -> f64 {
    if m > F32_MAX {
        f64::INFINITY
    } else {
        half_ulp(m as f32)
    }
}

/// The part of two enclosures both hold; the plain one if they somehow do not meet.
fn meet(a: Interval, b: Interval) -> Interval {
    let (lo, hi) = (a.lo.max(b.lo), a.hi.min(b.hi));
    if lo <= hi { Interval { lo, hi } } else { b }
}

#[derive(Clone)]
struct Aff {
    /// Noise symbols per element.
    s: usize,
    c: Vec<f64>,
    /// `len x s`, row-major.
    g: Vec<f64>,
    e: Vec<f64>,
}

impl Aff {
    fn zeros(len: usize, s: usize) -> Aff {
        Aff {
            s,
            c: vec![0.0; len],
            g: vec![0.0; len * s],
            e: vec![0.0; len],
        }
    }

    fn len(&self) -> usize {
        self.c.len()
    }

    fn row(&self, i: usize) -> &[f64] {
        &self.g[i * self.s..(i + 1) * self.s]
    }

    /// How far element `i` can be from its center.
    fn dev(&self, i: usize) -> f64 {
        let sum: f64 = self.row(i).iter().map(|x| x.abs()).sum();
        up64(up_sum(sum, self.s) + self.e[i])
    }

    /// An upper bound for `|value|` of element `i`.
    fn mag(&self, i: usize) -> f64 {
        up64(self.c[i].abs() + self.dev(i))
    }

    fn span(&self, i: usize) -> Interval {
        let d = self.dev(i);
        if !d.is_finite() || !self.c[i].is_finite() {
            return WHOLE;
        }
        Interval {
            lo: down64(self.c[i] - d),
            hi: up64(self.c[i] + d),
        }
    }

    fn unbound(&mut self, i: usize) {
        let s = self.s;
        self.c[i] = 0.0;
        self.g[i * s..(i + 1) * s].fill(0.0);
        self.e[i] = f64::INFINITY;
    }

    /// Anything that is not a finite number (or an infinite radius) becomes unbounded.
    fn clean(&mut self, i: usize) {
        if !self.c[i].is_finite()
            || self.e[i].is_nan()
            || self.row(i).iter().any(|x| !x.is_finite())
        {
            self.unbound(i);
        }
    }

    /// The form of a plain enclosure: its midpoint, its radius as the independent error.
    fn from_spans(spans: &[Interval], s: usize) -> Aff {
        let mut out = Aff::zeros(spans.len(), s);
        for (i, span) in spans.iter().enumerate() {
            // An infinite end gives an infinite or NaN center or radius: `clean` makes it unbounded.
            out.c[i] = (span.lo + span.hi) * 0.5;
            let slack = 4.0 * EPS * (span.lo.abs() + span.hi.abs()) + f64::MIN_POSITIVE;
            out.e[i] = up64((span.hi - span.lo) * 0.5 + slack);
            out.clean(i);
        }
        out
    }
}

/// Close one result element: add the float32 rounding of the operation (when it rounds) and the
/// float64 rounding of the `k` operations that produced it, which is at most
/// `2 eps (k + s + 8)` of the sum of the magnitudes of the terms that went in.
fn finish(out: &mut Aff, i: usize, terms: f64, k: usize, rounding: bool) {
    let gamma = 2.0 * EPS * (k as f64 + out.s as f64 + 8.0);
    let mut e = out.e[i];
    if rounding {
        e = up64(e + round_term(out.mag(i)));
    }
    e = up64(e + gamma * terms);
    out.e[i] = e;
    out.clean(i);
}

fn add_sub(a: &Aff, b: &Aff, ia: &[usize], ib: &[usize], sign: f64) -> Aff {
    let s = a.s;
    let mut out = Aff::zeros(ia.len(), s);
    for (i, (&x, &y)) in ia.iter().zip(ib).enumerate() {
        out.c[i] = a.c[x] + sign * b.c[y];
        for t in 0..s {
            out.g[i * s + t] = a.g[x * s + t] + sign * b.g[y * s + t];
        }
        out.e[i] = up64(a.e[x] + b.e[y]);
        let terms = up64(a.mag(x) + b.mag(y));
        finish(&mut out, i, terms, 2, true);
    }
    out
}

fn neg(a: &Aff) -> Aff {
    let mut out = a.clone();
    out.c.iter_mut().for_each(|x| *x = -*x);
    out.g.iter_mut().for_each(|x| *x = -*x);
    out
}

fn transpose(a: &Aff, shape: &[usize], perm: &[usize]) -> Aff {
    let order: Vec<usize> = kernels::transpose(&(0..a.len()).collect::<Vec<_>>(), shape, perm);
    let s = a.s;
    let mut out = Aff::zeros(a.len(), s);
    for (i, &from) in order.iter().enumerate() {
        out.c[i] = a.c[from];
        out.e[i] = a.e[from];
        out.g[i * s..(i + 1) * s].copy_from_slice(a.row(from));
    }
    out
}

fn relu(a: &Aff) -> Aff {
    let s = a.s;
    let mut out = Aff::zeros(a.len(), s);
    for i in 0..a.len() {
        let span = a.span(i);
        // An infinite end below zero makes lambda or the offset NaN: `finish` cleans it to unbounded.
        if span.lo >= 0.0 {
            // Never negative: relu is the identity.
            out.c[i] = a.c[i];
            out.e[i] = a.e[i];
            out.g[i * s..(i + 1) * s].copy_from_slice(a.row(i));
        } else {
            // relu(x) lies in lambda x + [0, -lambda l] for x in [l, u], with lambda = u / (u - l)
            // and u floored at zero (a never-positive value gets lambda = 0: the zero form).
            let u = span.hi.max(0.0);
            let lambda = u / (u - span.lo);
            let half = up64(-lambda * span.lo * 0.5);
            out.c[i] = lambda * a.c[i] + half;
            for t in 0..s {
                out.g[i * s + t] = lambda * a.g[i * s + t];
            }
            out.e[i] = up64(lambda * a.e[i] + half);
            let terms = up64(lambda * a.mag(i) + half);
            finish(&mut out, i, terms, 3, false);
        }
    }
    out
}

/// `k[i] * a[ia[i]]`, rounded once, for a constant `k` (a product, or a quotient by a constant).
fn scale(a: &Aff, ia: &[usize], k: &[f64]) -> Aff {
    let s = a.s;
    let mut out = Aff::zeros(ia.len(), s);
    for (i, (&x, &k)) in ia.iter().zip(k).enumerate() {
        out.c[i] = k * a.c[x];
        for t in 0..s {
            out.g[i * s + t] = k * a.g[x * s + t];
        }
        out.e[i] = up64(k.abs() * a.e[x]);
        let terms = up64(k.abs() * a.mag(x));
        finish(&mut out, i, terms, 2, true);
    }
    out
}

/// `a (m x k) times b (k x n)` where one operand is a constant (`b_const` says which): each
/// output is the exact linear combination, plus one float32 rounding per fused step of the
/// kernel's ascending-`k` chain at the largest magnitude the running sum can have.
fn matmul(a: &Aff, b: &Aff, b_const: bool, (m, k, n): (usize, usize, usize)) -> Aff {
    let s = a.s;
    let form = if b_const { a } else { b };
    let mags: Vec<f64> = (0..form.len()).map(|i| form.mag(i)).collect();
    let mut out = Aff::zeros(m * n, s);
    let mut acc = vec![0.0f64; s];
    for i in 0..m {
        for j in 0..n {
            acc.fill(0.0);
            let (mut c, mut e, mut terms, mut rounding) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
            for p in 0..k {
                let (w, f) = if b_const {
                    (b.c[p * n + j], i * k + p)
                } else {
                    (a.c[i * k + p], p * n + j)
                };
                c += w * form.c[f];
                for (x, y) in acc.iter_mut().zip(form.row(f)) {
                    *x += w * y;
                }
                e += w.abs() * form.e[f];
                terms += w.abs() * mags[f];
                rounding = up64(rounding + round_term(up64(terms + rounding)));
            }
            let idx = i * n + j;
            out.c[idx] = c;
            out.g[idx * s..(idx + 1) * s].copy_from_slice(&acc);
            out.e[idx] = up64(e + rounding);
            finish(&mut out, idx, up64(terms), k + 2, false);
        }
    }
    out
}

/// How many noise symbols to track: at most `max_symbols`, at most what keeps all the forms
/// within `MAX_ENTRIES`, and no column for a symbol nobody uses (`wanted` elements have a box).
fn symbol_budget(max_symbols: usize, total_elements: usize, wanted: usize) -> usize {
    max_symbols
        .min(MAX_ENTRIES / total_elements.max(1))
        .min(wanted)
}

/// The input elements that get a noise symbol: a positive finite radius, widest first (ties
/// by slot and position), up to the budget. Returns the symbol count and each element's column.
fn assign_symbols(
    boxes: &[(usize, &Tensor, &Tensor)],
    max_symbols: usize,
    total_elements: usize,
) -> (usize, HashMap<(usize, usize), usize>) {
    let mut radii: Vec<(f64, usize, usize)> = Vec::new();
    for &(slot, l, h) in boxes {
        for (j, (&a, &b)) in l.data().iter().zip(h.data()).enumerate() {
            let width = f64::from(b) - f64::from(a);
            if width > 0.0 && width.is_finite() {
                radii.push((width, slot, j));
            }
        }
    }
    radii.sort_by(|x, y| y.0.total_cmp(&x.0).then(x.1.cmp(&y.1)).then(x.2.cmp(&y.2)));
    let s = symbol_budget(max_symbols, total_elements, radii.len());
    let symbol = radii
        .iter()
        .take(s)
        .enumerate()
        .map(|(t, &(_, slot, j))| ((slot, j), t))
        .collect();
    (s, symbol)
}

/// The form of an input box: center, its radius on the element's own symbol (or as an
/// independent error when it has none), and the float64 rounding of both. An infinite end is
/// unbounded.
fn input_form(
    slot: usize,
    l: &Tensor,
    h: &Tensor,
    s: usize,
    symbol: &HashMap<(usize, usize), usize>,
) -> Aff {
    let mut form = Aff::zeros(l.data().len(), s);
    for (j, (&a, &b)) in l.data().iter().zip(h.data()).enumerate() {
        let (a, b) = (f64::from(a), f64::from(b));
        if !a.is_finite() || !b.is_finite() {
            form.unbound(j);
            continue;
        }
        form.c[j] = (a + b) * 0.5;
        let radius = (b - a) * 0.5;
        let slack = 4.0 * EPS * (a.abs() + b.abs()) + f64::MIN_POSITIVE;
        match symbol.get(&(slot, j)) {
            Some(&t) => {
                form.g[j * s + t] = radius;
                form.e[j] = up64(slack);
            }
            None => form.e[j] = up64(radius + slack),
        }
    }
    form
}

/// The enclosure of every output slot: affine forms intersected with the plain enclosures.
pub(crate) fn refine(
    plan: &Plan,
    lo: &[Tensor],
    hi: &[Tensor],
    plain: &[Vec<Interval>],
    max_symbols: usize,
) -> Vec<Vec<Interval>> {
    let slots = plan.slot_shapes.len();
    let total: usize = plan
        .slot_shapes
        .iter()
        .map(|shape| shape.iter().product::<usize>())
        .sum();
    let boxes: Vec<(usize, &Tensor, &Tensor)> = plan
        .inputs
        .iter()
        .zip(lo)
        .zip(hi)
        .map(|((binding, l), h)| (binding.slot, l, h))
        .collect();
    let (s, symbol) = assign_symbols(&boxes, max_symbols, total);

    let mut forms: Vec<Option<Aff>> = vec![None; slots];
    let mut is_const = vec![false; slots];
    for &(slot, l, h) in &boxes {
        forms[slot] = Some(input_form(slot, l, h, s, &symbol));
    }
    for (slot, data) in &plan.constants {
        let mut form = Aff::zeros(data.len(), s);
        for (c, &x) in form.c.iter_mut().zip(data.iter()) {
            *c = f64::from(x);
        }
        forms[*slot] = Some(form);
        is_const[*slot] = true;
    }

    let enclosure = |forms: &[Option<Aff>], slot: usize| -> Vec<Interval> {
        match &forms[slot] {
            Some(form) => (0..form.len())
                .map(|i| meet(form.span(i), plain[slot][i]))
                .collect(),
            None => plain[slot].clone(),
        }
    };

    for st in &plan.steps {
        let shapes: Vec<&[usize]> = st
            .inputs
            .iter()
            .map(|&i| plan.slot_shapes[i].as_slice())
            .collect();
        let len = plan.slot_shapes[st.output].iter().product::<usize>();
        let operands: Option<Vec<&Aff>> = st.inputs.iter().map(|&i| forms[i].as_ref()).collect();
        let fallback = |forms: &[Option<Aff>]| -> Aff {
            let spans: Vec<Vec<Interval>> =
                st.inputs.iter().map(|&i| enclosure(forms, i)).collect();
            let refs: Vec<&[Interval]> = spans.iter().map(Vec::as_slice).collect();
            Aff::from_spans(&step(&st.op, &refs, &shapes, len), s)
        };
        let out = match (&st.op, operands) {
            (Op::Add | Op::Sub, Some(v)) => {
                let (ia, ib) = operand_maps(&shapes, len);
                let sign = if matches!(st.op, Op::Sub) { -1.0 } else { 1.0 };
                add_sub(v[0], v[1], &ia, &ib, sign)
            }
            (Op::Neg, Some(v)) => neg(v[0]),
            (Op::Relu, Some(v)) => relu(v[0]),
            (Op::Reshape { .. }, Some(v)) => v[0].clone(),
            (Op::Transpose { perm }, Some(v)) => transpose(v[0], shapes[0], perm),
            (Op::Mul, Some(v)) if is_const[st.inputs[1]] || is_const[st.inputs[0]] => {
                let (ia, ib) = operand_maps(&shapes, len);
                if is_const[st.inputs[1]] {
                    let k: Vec<f64> = ib.iter().map(|&j| v[1].c[j]).collect();
                    scale(v[0], &ia, &k)
                } else {
                    let k: Vec<f64> = ia.iter().map(|&j| v[0].c[j]).collect();
                    scale(v[1], &ib, &k)
                }
            }
            (Op::Div, Some(v)) if is_const[st.inputs[1]] && v[1].c.iter().all(|&d| d != 0.0) => {
                let (ia, ib) = operand_maps(&shapes, len);
                let k: Vec<f64> = ib.iter().map(|&j| 1.0 / v[1].c[j]).collect();
                scale(v[0], &ia, &k)
            }
            (Op::MatMul, Some(v)) if is_const[st.inputs[0]] || is_const[st.inputs[1]] => {
                let dims = (shapes[0][0], shapes[0][1], shapes[1][1]);
                matmul(v[0], v[1], is_const[st.inputs[1]], dims)
            }
            _ => fallback(&forms),
        };
        forms[st.output] = Some(out);
    }

    plan.outputs
        .iter()
        .map(|&slot| enclosure(&forms, slot))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::interval::{DEFAULT_MAX_SYMBOLS, analyze_box, analyze_box_with, plain_slots};
    use crate::{Backend, ReferenceBackend};
    use oxfuse_core::{Graph, GraphBuilder, Limits};

    /// The center, symbols and error of one element.
    type Term<'a> = (f64, &'a [f64], f64);

    fn t(shape: &[usize], data: &[f32]) -> Tensor {
        Tensor::new(shape.to_vec(), data.to_vec()).unwrap()
    }

    fn unit(state: &mut u64) -> f64 {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (*state >> 40) as f64 / (1u64 << 24) as f64
    }

    /// `depth` layers of matmul, bias and relu (none after the last), `width` wide.
    fn mlp(depth: usize, width: usize, seed: u64) -> Graph {
        let mut state = seed;
        let mut g = GraphBuilder::new(Limits::default());
        let mut cur = g.input("x", vec![1, width]).unwrap();
        for l in 0..depth {
            let w: Vec<f32> = (0..width * width)
                .map(|_| ((unit(&mut state) * 2.0 - 1.0) * 1.7 / (width as f64).sqrt()) as f32)
                .collect();
            let b: Vec<f32> = (0..width)
                .map(|_| ((unit(&mut state) * 2.0 - 1.0) * 0.1) as f32)
                .collect();
            let wc = g.constant(&format!("w{l}"), vec![width, width], w).unwrap();
            let bc = g.constant(&format!("b{l}"), vec![width], b).unwrap();
            let mm = g.node(&format!("m{l}"), Op::MatMul, &[cur, wc]).unwrap();
            let ad = g.node(&format!("a{l}"), Op::Add, &[mm, bc]).unwrap();
            cur = if l + 1 < depth {
                g.node(&format!("r{l}"), Op::Relu, &[ad]).unwrap()
            } else {
                ad
            };
        }
        g.finish(&[cur]).unwrap()
    }

    /// A box of relative half-width `fraction` around a random point of `[-1, 1]^width`.
    fn around(width: usize, fraction: f64, seed: u64) -> (Tensor, Tensor) {
        let mut state = seed ^ 0x9E37_79B9;
        let x: Vec<f32> = (0..width)
            .map(|_| (unit(&mut state) * 2.0 - 1.0) as f32)
            .collect();
        let r = |v: f32| (f64::from(v.abs()) * fraction) as f32;
        (
            t(
                &[1, width],
                &x.iter().map(|&v| v - r(v)).collect::<Vec<_>>(),
            ),
            t(
                &[1, width],
                &x.iter().map(|&v| v + r(v)).collect::<Vec<_>>(),
            ),
        )
    }

    /// Per output element, the range the reference backend reaches over the box corners and
    /// `points` random points.
    fn sampled_range(graph: &Graph, lo: &Tensor, hi: &Tensor, points: usize) -> Vec<(f64, f64)> {
        let exe = ReferenceBackend.compile(graph).unwrap();
        let mut state = 7u64;
        let (l, h) = (lo.data(), hi.data());
        let mut range: Vec<(f64, f64)> = Vec::new();
        for n in 0..points + 2 {
            let data: Vec<f32> = l
                .iter()
                .zip(h)
                .map(|(&a, &b)| match n {
                    0 => a,
                    1 => b,
                    _ => (a + (b - a) * unit(&mut state) as f32).clamp(a, b),
                })
                .collect();
            let out = exe.run(&[t(lo.shape(), &data)]).unwrap().remove(0);
            if range.is_empty() {
                range = out
                    .data()
                    .iter()
                    .map(|&v| (f64::from(v), f64::from(v)))
                    .collect();
            }
            for (r, &v) in range.iter_mut().zip(out.data()) {
                r.0 = r.0.min(f64::from(v));
                r.1 = r.1.max(f64::from(v));
            }
        }
        range
    }

    fn median_ratio(enclosure: &[(f64, f64)], range: &[(f64, f64)]) -> f64 {
        let mut ratios: Vec<f64> = enclosure
            .iter()
            .zip(range)
            .filter(|(_, r)| r.1 > r.0)
            .map(|(e, r)| (e.1 - e.0) / (r.1 - r.0))
            .collect();
        ratios.sort_by(f64::total_cmp);
        ratios[ratios.len() / 2]
    }

    fn pairs(lo: &[f64], hi: &[f64]) -> Vec<(f64, f64)> {
        lo.iter().copied().zip(hi.iter().copied()).collect()
    }

    fn plain_output(graph: &Graph, lo: &Tensor, hi: &Tensor) -> Vec<(f64, f64)> {
        let plan = Plan::build(graph);
        let slots = plain_slots(&plan, std::slice::from_ref(lo), std::slice::from_ref(hi)).unwrap();
        slots[plan.outputs[0]]
            .iter()
            .map(|s| (s.lo, s.hi))
            .collect()
    }

    /// The design gate: at +-1 percent the median width over the sampled range stays within 3x
    /// for depth up to 6 where plain intervals blow up. The sampled range (400 random points and
    /// both corners) is narrower than the true range, so these ratios overstate the width.
    #[test]
    fn the_width_over_the_sampled_range_by_depth() {
        for fraction in [0.01, 0.1] {
            for depth in 1..=6 {
                let width = 16;
                let (mut affine, mut plain) = (Vec::new(), Vec::new());
                for seed in 0..5 {
                    let g = mlp(depth, width, seed);
                    let (lo, hi) = around(width, fraction, seed);
                    let range = sampled_range(&g, &lo, &hi, 400);
                    let e = analyze_box(&g, std::slice::from_ref(&lo), std::slice::from_ref(&hi))
                        .unwrap()
                        .remove(0);
                    affine.push(median_ratio(&pairs(&e.lo, &e.hi), &range));
                    plain.push(median_ratio(&plain_output(&g, &lo, &hi), &range));
                }
                affine.sort_by(f64::total_cmp);
                plain.sort_by(f64::total_cmp);
                let (affine, plain) = (affine[2], plain[2]);
                eprintln!(
                    "RATIO box +-{fraction} depth {depth}: affine {affine:.2} plain {plain:.2}"
                );
                if fraction == 0.01 {
                    assert!(affine <= 3.0, "depth {depth}: affine {affine}");
                }
                if depth >= 2 {
                    assert!(
                        affine < plain,
                        "box +-{fraction} depth {depth}: {affine} vs {plain}"
                    );
                }
                if fraction == 0.01 && depth >= 4 {
                    assert!(
                        plain > 10.0 * affine,
                        "depth {depth}: plain {plain} affine {affine}"
                    );
                }
            }
        }
    }

    #[test]
    fn x_minus_x_keeps_only_the_rounding() {
        let mut g = GraphBuilder::new(Limits::default());
        let x = g.input("x", vec![3]).unwrap();
        let d = g.node("d", Op::Sub, &[x, x]).unwrap();
        let graph = g.finish(&[d]).unwrap();
        let e = analyze_box(
            &graph,
            &[t(&[3], &[1.0, -5.0, 100.0])],
            &[t(&[3], &[2.0, 3.0, 120.0])],
        )
        .unwrap()
        .remove(0);
        for i in 0..3 {
            assert!(e.lo[i] <= 0.0 && e.hi[i] >= 0.0);
            assert!(
                e.hi[i] - e.lo[i] < 1e-9,
                "element {i}: [{}, {}]",
                e.lo[i],
                e.hi[i]
            );
        }
    }

    #[test]
    fn a_value_used_twice_through_constants_is_not_counted_twice() {
        // 3 x - 2.5 x = 0.5 x: plain intervals give [-2, 3.5] for x in [1, 2].
        let mut g = GraphBuilder::new(Limits::default());
        let x = g.input("x", vec![1]).unwrap();
        let a = g.constant("a", vec![], vec![3.0]).unwrap();
        let b = g.constant("b", vec![], vec![2.5]).unwrap();
        let p = g.node("p", Op::Mul, &[x, a]).unwrap();
        let q = g.node("q", Op::Mul, &[b, x]).unwrap();
        let d = g.node("d", Op::Sub, &[p, q]).unwrap();
        let graph = g.finish(&[d]).unwrap();
        let (lo, hi) = (t(&[1], &[1.0]), t(&[1], &[2.0]));
        let e = analyze_box(&graph, std::slice::from_ref(&lo), std::slice::from_ref(&hi))
            .unwrap()
            .remove(0);
        assert!(e.lo[0] <= 0.5 && e.lo[0] > 0.499, "{}", e.lo[0]);
        assert!(e.hi[0] >= 1.0 && e.hi[0] < 1.001, "{}", e.hi[0]);
        let plain = plain_output(&graph, &lo, &hi);
        assert!(plain[0].0 < -1.9 && plain[0].1 > 3.4, "{plain:?}");
    }

    #[test]
    fn a_division_by_a_constant_is_exact_in_the_form() {
        let mut g = GraphBuilder::new(Limits::default());
        let x = g.input("x", vec![1]).unwrap();
        let c = g.constant("c", vec![], vec![4.0]).unwrap();
        let q = g.node("q", Op::Div, &[x, c]).unwrap();
        let d = g.node("d", Op::Sub, &[x, q]).unwrap();
        let graph = g.finish(&[d]).unwrap();
        // x - x / 4 = 0.75 x.
        let e = analyze_box(&graph, &[t(&[1], &[4.0])], &[t(&[1], &[8.0])])
            .unwrap()
            .remove(0);
        assert!(e.lo[0] <= 3.0 && e.lo[0] > 2.999, "{}", e.lo[0]);
        assert!(e.hi[0] >= 6.0 && e.hi[0] < 6.001, "{}", e.hi[0]);
    }

    #[test]
    fn relu_is_exact_when_the_sign_is_decided_and_relaxed_when_not() {
        let mut g = GraphBuilder::new(Limits::default());
        let x = g.input("x", vec![3]).unwrap();
        let r = g.node("r", Op::Relu, &[x]).unwrap();
        let d = g.node("d", Op::Sub, &[r, x]).unwrap();
        let graph = g.finish(&[r, d]).unwrap();
        let (lo, hi) = (t(&[3], &[1.0, -2.0, -1.0]), t(&[3], &[2.0, -1.0, 2.0]));
        let out =
            analyze_box(&graph, std::slice::from_ref(&lo), std::slice::from_ref(&hi)).unwrap();
        // x in [1, 2]: relu x = x, so relu x - x is zero. x in [-2, -1]: relu x = 0.
        assert!(out[1].hi[0] - out[1].lo[0] < 1e-9);
        assert!(out[0].lo[1] <= 0.0 && out[0].hi[1] < 1e-9 && out[0].lo[1] > -1e-9);
        // x in [-1, 2]: sound and no wider than the hull [0, 2] plus the relaxation band.
        assert!(out[0].lo[2] <= 0.0 && out[0].hi[2] >= 2.0 && out[0].hi[2] < 2.01);
        let exe = ReferenceBackend.compile(&graph).unwrap();
        for v in [-1.0f32, -0.3, 0.0, 0.7, 2.0] {
            let got = exe.run(&[t(&[3], &[1.5, -1.5, v])]).unwrap();
            let r = f64::from(got[0].data()[2]);
            let diff = f64::from(got[1].data()[2]);
            assert!(out[0].lo[2] <= r && r <= out[0].hi[2], "relu({v}) = {r}");
            assert!(
                out[1].lo[2] <= diff && diff <= out[1].hi[2],
                "relu({v}) - {v} = {diff}"
            );
        }
    }

    #[test]
    fn the_result_is_never_wider_than_plain_intervals_and_stays_sound() {
        for seed in 0..300 {
            let (graph, inputs) = crate::analyze::tests::random_case(seed);
            let (lo, hi) = {
                let r = |i: usize, v: f32| v.abs() * 0.05 * (1 + i % 3) as f32 + 0.01;
                let d = inputs[0].data();
                (
                    Tensor::new(
                        inputs[0].shape().to_vec(),
                        d.iter().enumerate().map(|(i, &v)| v - r(i, v)).collect(),
                    )
                    .unwrap(),
                    Tensor::new(
                        inputs[0].shape().to_vec(),
                        d.iter().enumerate().map(|(i, &v)| v + r(i, v)).collect(),
                    )
                    .unwrap(),
                )
            };
            let e = analyze_box(&graph, std::slice::from_ref(&lo), std::slice::from_ref(&hi))
                .unwrap()
                .remove(0);
            let plain = plain_output(&graph, &lo, &hi);
            for (i, p) in plain.iter().enumerate() {
                assert!(e.lo[i] >= p.0 && e.hi[i] <= p.1, "seed {seed} element {i}");
            }
        }
    }

    #[test]
    fn any_symbol_budget_stays_sound_and_inside_the_plain_enclosure() {
        let width = 16;
        for depth in [2, 4] {
            for seed in 0..4 {
                let g = mlp(depth, width, seed);
                let (lo, hi) = around(width, 0.05, seed);
                let range = sampled_range(&g, &lo, &hi, 300);
                let plain = plain_output(&g, &lo, &hi);
                for budget in [0, 1, 2, 5, 16, DEFAULT_MAX_SYMBOLS] {
                    let e = analyze_box_with(
                        &g,
                        std::slice::from_ref(&lo),
                        std::slice::from_ref(&hi),
                        budget,
                    )
                    .unwrap()
                    .remove(0);
                    for (i, r) in range.iter().enumerate() {
                        assert!(
                            e.lo[i] <= r.0 && r.1 <= e.hi[i],
                            "depth {depth} seed {seed} budget {budget} element {i}"
                        );
                        assert!(e.lo[i] >= plain[i].0 && e.hi[i] <= plain[i].1);
                    }
                }
            }
        }
    }

    #[test]
    fn more_symbols_never_lose_the_correlation_a_few_keep() {
        let width = 16;
        let g = mlp(4, width, 3);
        let (lo, hi) = around(width, 0.01, 3);
        let widths = |budget: usize| {
            let e = analyze_box_with(
                &g,
                std::slice::from_ref(&lo),
                std::slice::from_ref(&hi),
                budget,
            )
            .unwrap()
            .remove(0);
            e.lo.iter().zip(&e.hi).map(|(l, h)| h - l).sum::<f64>()
        };
        assert!(
            widths(width) < widths(2),
            "{} vs {}",
            widths(width),
            widths(2)
        );
        assert!(widths(2) <= widths(0) * (1.0 + 1e-9));
    }

    #[test]
    fn a_wide_input_runs_within_the_symbol_budget() {
        // 5000 input elements share 256 symbols; the rest are independent errors.
        let (m, k, n) = (2, 2500, 4);
        let mut g = GraphBuilder::new(Limits::default());
        let x = g.input("x", vec![m, k]).unwrap();
        let w: Vec<f32> = (0..k * n).map(|i| ((i % 7) as f32 - 3.0) * 0.01).collect();
        let wc = g.constant("w", vec![k, n], w).unwrap();
        let y = g.node("y", Op::MatMul, &[x, wc]).unwrap();
        let graph = g.finish(&[y]).unwrap();
        let mid: Vec<f32> = (0..m * k).map(|i| ((i % 5) as f32) * 0.1).collect();
        let lo = t(&[m, k], &mid.iter().map(|v| v - 0.01).collect::<Vec<_>>());
        let hi = t(&[m, k], &mid.iter().map(|v| v + 0.01).collect::<Vec<_>>());
        let e = analyze_box(&graph, std::slice::from_ref(&lo), std::slice::from_ref(&hi))
            .unwrap()
            .remove(0);
        let exe = ReferenceBackend.compile(&graph).unwrap();
        for point in [&lo, &hi, &t(&[m, k], &mid)] {
            let out = exe.run(std::slice::from_ref(point)).unwrap().remove(0);
            for (i, &v) in out.data().iter().enumerate() {
                assert!(
                    e.lo[i] <= f64::from(v) && f64::from(v) <= e.hi[i],
                    "element {i}"
                );
            }
        }
    }

    #[test]
    fn an_infinite_endpoint_stays_unbounded_through_the_forms() {
        let mut g = GraphBuilder::new(Limits::default());
        let x = g.input("x", vec![2]).unwrap();
        let c = g.constant("c", vec![], vec![2.0]).unwrap();
        let m = g.node("m", Op::Mul, &[x, c]).unwrap();
        let graph = g.finish(&[m]).unwrap();
        let e = analyze_box(
            &graph,
            &[t(&[2], &[0.0, 1.0])],
            &[t(&[2], &[f32::INFINITY, 2.0])],
        )
        .unwrap()
        .remove(0);
        assert!(e.hi[0].is_infinite() && e.lo[0] <= 0.0 && e.lo[0] > -1e-9);
        assert!(e.hi[1].is_finite() && e.hi[1] >= 4.0);
    }

    // ---- direct tests of each rule, with the expected value computed independently ----

    fn form(c: &[f64], g: &[f64], e: &[f64]) -> Aff {
        let s = g.len() / c.len();
        Aff {
            s,
            c: c.to_vec(),
            g: g.to_vec(),
            e: e.to_vec(),
        }
    }

    /// `got` is at least `expected` (a few ulps allowed below) and not more than a hair above.
    fn assert_near_above(got: f64, expected: f64, what: &str) {
        assert!(
            got >= expected * (1.0 - 1e-15) && got <= expected * (1.0 + 1e-12) + 1e-300,
            "{what}: {got:e} against {expected:e}"
        );
    }

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() <= 1e-12 * (1.0 + b.abs())
    }

    /// A plain-arithmetic magnitude of element `i`: `|c| + sum |g| + e`.
    fn plain_mag(f: &Aff, i: usize) -> f64 {
        f.c[i].abs() + f.row(i).iter().map(|x| x.abs()).sum::<f64>() + f.e[i]
    }

    fn gamma(k: usize, s: usize) -> f64 {
        2.0 * EPS * (k as f64 + s as f64 + 8.0)
    }

    #[test]
    fn up_sum_widens_by_about_the_count_of_terms() {
        let widened = up_sum(1.0, 10);
        assert!(
            (1.0 + 12.0 * EPS..=1.0 + 16.0 * EPS).contains(&widened),
            "{widened}"
        );
        assert!(up_sum(0.0, 5) >= 0.0);
        assert!(up_sum(2.0, 3) > 2.0 + 4.0 * EPS);
    }

    #[test]
    fn round_term_is_finite_up_to_the_float32_maximum_and_infinite_past_it() {
        assert_eq!(round_term(F32_MAX), half_ulp(f32::MAX));
        assert!(round_term(F32_MAX * 1.0001).is_infinite());
        assert_eq!(round_term(1.0), half_ulp(1.0));
        assert_eq!(round_term(0.0), half_ulp(0.0));
    }

    #[test]
    fn span_is_unbounded_for_an_infinite_radius_or_center() {
        let finite = form(&[2.0], &[0.5], &[0.25]);
        let s = finite.span(0);
        assert!(
            s.lo <= 1.25 && s.lo > 1.249 && s.hi >= 2.75 && s.hi < 2.751,
            "{s:?}"
        );
        assert_eq!(form(&[2.0], &[0.5], &[f64::INFINITY]).span(0), WHOLE);
        assert_eq!(form(&[f64::INFINITY], &[0.5], &[0.25]).span(0), WHOLE);
        assert_eq!(form(&[f64::NAN], &[0.5], &[0.25]).span(0), WHOLE);
    }

    #[test]
    fn unbound_clears_only_its_own_row() {
        let mut f = form(&[1.0, 2.0, 3.0], &[1.0; 6], &[0.5; 3]);
        f.unbound(1);
        assert_eq!(f.c, [1.0, 0.0, 3.0]);
        assert_eq!(f.g, [1.0, 1.0, 0.0, 0.0, 1.0, 1.0]);
        assert_eq!(f.e[0], 0.5);
        assert!(f.e[1].is_infinite());
        assert_eq!(f.e[2], 0.5);
    }

    #[test]
    fn clean_unbounds_anything_that_is_not_a_finite_number_and_nothing_else() {
        let cases = [
            (f64::NAN, 1.0, 0.5, true),
            (f64::INFINITY, 1.0, 0.5, true),
            (1.0, f64::NAN, 0.5, true),
            (1.0, f64::INFINITY, 0.5, true),
            (1.0, 1.0, f64::NAN, true),
            (1.0, 1.0, 0.5, false),
            (1.0, 1.0, f64::INFINITY, false),
        ];
        for (c, g, e, unbound) in cases {
            let mut f = form(&[c], &[g, 0.25], &[e]);
            f.clean(0);
            let cleaned = f.c[0] == 0.0 && f.g == [0.0, 0.0] && f.e[0].is_infinite();
            assert_eq!(cleaned, unbound, "c {c} g {g} e {e}");
            if !unbound {
                assert_eq!((f.c[0], f.g[0], f.e[0]), (c, g, e));
            }
        }
    }

    #[test]
    fn a_form_of_a_span_contains_the_span_and_pays_only_the_rounding() {
        let spans = [
            Interval { lo: 1.0, hi: 3.0 },
            Interval { lo: -4.0, hi: -2.0 },
            Interval {
                lo: 1.0e10,
                hi: 2.0e10,
            },
            Interval { lo: 5.0, hi: 5.0 },
        ];
        let f = Aff::from_spans(&spans, 2);
        assert_eq!(f.s, 2);
        for (i, sp) in spans.iter().enumerate() {
            let radius = (sp.hi - sp.lo) * 0.5;
            assert_eq!(f.c[i], (sp.lo + sp.hi) * 0.5);
            assert_eq!(f.row(i), [0.0, 0.0]);
            let slack = 4.0 * EPS * (sp.lo.abs() + sp.hi.abs()) + f64::MIN_POSITIVE;
            assert_near_above(f.e[i], radius + slack, "radius plus slack");
            let back = f.span(i);
            assert!(back.lo <= sp.lo && back.hi >= sp.hi, "{back:?} vs {sp:?}");
        }
        let zero = Aff::from_spans(&[Interval { lo: 0.0, hi: 0.0 }], 1);
        assert!(zero.e[0] >= f64::MIN_POSITIVE, "{:e}", zero.e[0]);
        let open = Aff::from_spans(
            &[
                Interval {
                    lo: f64::NEG_INFINITY,
                    hi: 1.0,
                },
                Interval {
                    lo: 0.0,
                    hi: f64::INFINITY,
                },
            ],
            1,
        );
        assert!(open.e.iter().all(|e| e.is_infinite()) && open.c == [0.0, 0.0]);
    }

    #[test]
    fn finish_adds_the_float32_rounding_and_the_float64_slack() {
        let mut f = form(&[1.0], &[0.0; 3], &[0.0]);
        finish(&mut f, 0, 10.0, 5, false);
        assert_near_above(f.e[0], gamma(5, 3) * 10.0, "slack only");
        let mut f = form(&[1.0], &[0.0; 3], &[0.0]);
        finish(&mut f, 0, 10.0, 5, true);
        assert_near_above(
            f.e[0],
            half_ulp(1.0) + gamma(5, 3) * 10.0,
            "rounding and slack",
        );
        let mut f = form(&[1.0], &[0.0; 3], &[0.125]);
        finish(&mut f, 0, 0.0, 5, false);
        assert_near_above(
            f.e[0],
            0.125,
            "no terms, no slack beyond the outward rounding",
        );
        let mut f = form(&[1.0], &[0.0; 3], &[0.0]);
        finish(&mut f, 0, f64::NAN, 5, false);
        assert!(f.e[0].is_infinite() && f.c[0] == 0.0);
    }

    #[test]
    fn add_and_sub_combine_centers_symbols_and_errors_per_element() {
        let a = form(&[3.0, 10.0], &[0.5, -0.25, 1.0, 2.0], &[0.125, 0.0]);
        let b = form(&[-1.0, 4.0], &[0.25, 0.75, -0.5, 0.5], &[0.0625, 0.25]);
        for (sign, op_name) in [(1.0, "add"), (-1.0, "sub")] {
            let out = add_sub(&a, &b, &[0, 1], &[0, 1], sign);
            for i in 0..2 {
                assert!(close(out.c[i], a.c[i] + sign * b.c[i]), "{op_name} c[{i}]");
                for t in 0..2 {
                    assert!(
                        close(out.g[i * 2 + t], a.g[i * 2 + t] + sign * b.g[i * 2 + t]),
                        "{op_name} g"
                    );
                }
                let m = out.c[i].abs()
                    + out.row(i).iter().map(|x| x.abs()).sum::<f64>()
                    + a.e[i]
                    + b.e[i];
                let expected = a.e[i]
                    + b.e[i]
                    + half_ulp(m as f32)
                    + gamma(2, 2) * (plain_mag(&a, i) + plain_mag(&b, i));
                assert_near_above(out.e[i], expected, &format!("{op_name} e[{i}]"));
            }
        }
        // Broadcasting: one operand element feeds two outputs, in either operand.
        let out = add_sub(&a, &b, &[0, 0], &[1, 0], 1.0);
        assert!(close(out.c[0], 3.0 + 4.0) && close(out.c[1], 3.0 - 1.0));
        let out = add_sub(&a, &b, &[1, 0], &[0, 0], 1.0);
        assert!(close(out.c[0], 10.0 - 1.0) && close(out.c[1], 3.0 - 1.0));
    }

    #[test]
    fn neg_flips_centers_and_symbols_and_keeps_the_error() {
        let a = form(&[3.0, -10.0], &[0.5, -0.25, 1.0, 2.0], &[0.125, 0.5]);
        let n = neg(&a);
        assert_eq!(n.c, [-3.0, 10.0]);
        assert_eq!(n.g, [-0.5, 0.25, -1.0, -2.0]);
        assert_eq!(n.e, a.e);
    }

    #[test]
    fn transpose_moves_centers_symbols_and_errors_together() {
        let a = form(
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            &(0..12).map(f64::from).collect::<Vec<_>>(),
            &[0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        );
        let t = transpose(&a, &[2, 3], &[1, 0]);
        assert_eq!(t.c, [1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
        assert_eq!(t.e, [0.1, 0.4, 0.2, 0.5, 0.3, 0.6]);
        assert_eq!(t.row(1), a.row(3));
        assert_eq!(t.row(2), a.row(1));
        assert_eq!(t.row(5), a.row(5));
    }

    #[test]
    fn relu_follows_the_relaxation_formula_when_the_sign_is_undecided() {
        // Two elements: c 1, radius 3 (range [-2.x, 4.x]) and c -0.5, radius 2.
        let a = form(&[1.0, -0.5], &[2.0, 1.0, 1.5, 0.5], &[0.5, 0.25]);
        let out = relu(&a);
        for i in 0..2 {
            let m = plain_mag(&a, i);
            let (l, u) = (a.c[i] - (m - a.c[i].abs()), a.c[i] + (m - a.c[i].abs()));
            let lambda = u / (u - l);
            let half = -lambda * l * 0.5;
            assert!(close(out.c[i], lambda * a.c[i] + half), "c[{i}]");
            for t in 0..2 {
                assert!(
                    close(out.g[i * 2 + t], lambda * a.g[i * 2 + t]),
                    "g[{i}][{t}]"
                );
            }
            // The arithmetic of the rule, replayed step by step from the concretization.
            let sp = a.span(i);
            let lam = sp.hi.max(0.0) / (sp.hi - sp.lo);
            let offset = up64(-lam * sp.lo * 0.5);
            let terms = up64(lam * a.mag(i) + offset);
            let before = up64(lam * a.e[i] + offset);
            let expected = up64(before + gamma(3, 2) * terms);
            assert_eq!(out.e[i], expected, "e[{i}]");
        }
        // Never negative: unchanged. Never positive: zero. Infinite end: unbounded.
        let keep = relu(&form(&[5.0], &[1.0, 1.0], &[0.5]));
        assert_eq!(
            (keep.c[0], keep.g.clone(), keep.e[0]),
            (5.0, vec![1.0, 1.0], 0.5)
        );
        let zero = relu(&form(&[-5.0], &[1.0, 1.0], &[0.5]));
        assert!(zero.c[0].abs() < 1e-300 && zero.g == [0.0, 0.0] && zero.e[0] < 1e-300);
        let open = relu(&form(&[1.0, 1.0], &[1.0, 1.0], &[f64::INFINITY, 0.5]));
        assert!(open.e[0].is_infinite());
        assert!(open.e[1].is_finite());
    }

    #[test]
    fn scale_multiplies_centers_symbols_and_errors_by_the_constant() {
        let a = form(&[3.0, -2.0], &[0.5, -0.25, 1.0, 2.0], &[0.125, 0.25]);
        let out = scale(&a, &[0, 1, 1], &[-2.0, 0.5, 4.0]);
        let ks = [-2.0, 0.5, 4.0];
        let from = [0usize, 1, 1];
        for i in 0..3 {
            let (k, x) = (ks[i], from[i]);
            assert!(close(out.c[i], k * a.c[x]));
            for t in 0..2 {
                assert!(close(out.g[i * 2 + t], k * a.g[x * 2 + t]));
            }
            let m = (k * a.c[x]).abs()
                + (0..2).map(|t| (k * a.g[x * 2 + t]).abs()).sum::<f64>()
                + k.abs() * a.e[x];
            let expected =
                k.abs() * a.e[x] + half_ulp(m as f32) + gamma(2, 2) * k.abs() * plain_mag(&a, x);
            assert_near_above(out.e[i], expected, &format!("e[{i}]"));
        }
    }

    /// The `e` of a matmul element, replaying the arithmetic of the kernel step by step: the sum of
    /// `|w| e`, one float32 rounding per fused step at the largest magnitude the running sum can
    /// have, then the float64 slack of the `k + 2` operations.
    fn exact_matmul_e(weights: &[f64], forms: &[Term<'_>], s: usize) -> f64 {
        let (mut e, mut terms, mut rounding) = (0.0, 0.0, 0.0);
        for (w, (c, g, fe)) in weights.iter().zip(forms) {
            let mag = up64(c.abs() + up_sum(g.iter().map(|x| x.abs()).sum::<f64>(), s) + fe);
            e += w.abs() * fe;
            terms += w.abs() * mag;
            rounding = up64(rounding + half_ulp(up64(terms + rounding) as f32));
        }
        let pre = up64(e + rounding);
        up64(pre + gamma(weights.len() + 2, s) * up64(terms))
    }

    /// Both operands of the matmul checked element by element: center and symbols are the exact
    /// linear combination, the error is replayed exactly.
    fn check_matmul(a: &Aff, b: &Aff, b_const: bool, (m, k, n): (usize, usize, usize)) {
        let out = matmul(a, b, b_const, (m, k, n));
        for i in 0..m {
            for j in 0..n {
                let (ws, fs): (Vec<f64>, Vec<Term<'_>>) = (0..k)
                    .map(|p| {
                        let (w, f, form) = if b_const {
                            (b.c[p * n + j], i * k + p, a)
                        } else {
                            (a.c[i * k + p], p * n + j, b)
                        };
                        (w, (form.c[f], form.row(f), form.e[f]))
                    })
                    .unzip();
                let idx = i * n + j;
                let c: f64 = ws.iter().zip(&fs).map(|(w, f)| w * f.0).sum();
                assert!(close(out.c[idx], c), "c[{idx}]");
                for t in 0..a.s {
                    let g: f64 = ws.iter().zip(&fs).map(|(w, f)| w * f.1[t]).sum();
                    assert!(close(out.g[idx * a.s + t], g), "g[{idx}][{t}]");
                }
                assert_eq!(out.e[idx], exact_matmul_e(&ws, &fs, a.s), "e[{idx}]");
            }
        }
    }

    fn varying_2x3() -> Aff {
        form(
            &[1.0, -2.0, 3.0, 0.5, 1.5, -1.0],
            &[
                0.5, 0.1, -0.25, 0.2, 1.0, 0.3, 0.1, 0.1, 0.4, -0.4, 0.2, 0.6,
            ],
            &[0.1, 0.0, 0.2, 0.05, 0.3, 0.0],
        )
    }

    #[test]
    fn matmul_with_a_constant_right_operand_is_the_linear_combination() {
        // a is 2 x 3 (varying), b is 3 x 2 (constant).
        let b = form(&[2.0, -1.0, 0.5, 4.0, -3.0, 1.5], &[0.0; 6 * 2], &[0.0; 6]);
        check_matmul(&varying_2x3(), &b, true, (2, 3, 2));
    }

    #[test]
    fn matmul_with_a_constant_left_operand_is_the_linear_combination() {
        // a is 2 x 3 (constant), b is 3 x 2 (varying; the same numbers laid out as 3 x 2).
        let a = form(&[2.0, -1.0, 0.5, 1.5, 3.0, -2.0], &[0.0; 6 * 2], &[0.0; 6]);
        check_matmul(&a, &varying_2x3(), false, (2, 3, 2));
    }

    /// The running sum 4 - 1.3e-7 sits just under the point where a float32 rounds up to 4 (the
    /// midpoint 4 - 1.2e-7). Half an ulp at 1 (6e-8) has already been paid, which lifts the sum
    /// over the midpoint: the next rounding is the coarser one of the binade [4, 8), not that of [2, 4).
    #[test]
    fn matmul_rounding_uses_the_binade_the_running_sum_plus_its_rounding_can_reach() {
        let a = form(&[1.0, 3.0 - 1.3e-7], &[0.0; 2], &[0.0; 2]);
        let b = form(&[1.0, 1.0], &[0.0; 2], &[0.0; 2]);
        check_matmul(&a, &b, true, (1, 2, 1));
        let out = matmul(&a, &b, true, (1, 2, 1));
        assert!(out.e[0] >= half_ulp(1.0) + half_ulp(4.0), "{:e}", out.e[0]);
    }

    #[test]
    fn the_symbol_budget_is_the_smallest_of_the_three_limits() {
        assert_eq!(symbol_budget(256, 1000, 40), 40);
        assert_eq!(symbol_budget(10, 1000, 40), 10);
        assert_eq!(symbol_budget(256, MAX_ENTRIES / 7, 1000), 7);
        assert_eq!(symbol_budget(256, 0, 5), 5);
        assert_eq!(symbol_budget(0, 10, 5), 0);
        // Huge element counts leave room for no symbol at all.
        assert_eq!(symbol_budget(256, MAX_ENTRIES + 1, 5), 0);
    }

    #[test]
    fn symbols_go_to_the_widest_finite_boxes_first() {
        let lo = t(&[5], &[0.0, 0.0, 1.0, 0.0, f32::NEG_INFINITY]);
        let hi = t(&[5], &[2.0, 0.0, 5.0, 4.0, 1.0]);
        // Radii 1, 0 (a point), 2, 2, infinite: the point and the infinite one get none.
        let (s, map) = assign_symbols(&[(7, &lo, &hi)], 256, 10);
        assert_eq!(s, 3);
        assert_eq!(map.len(), 3);
        assert_eq!(map[&(7, 2)], 0, "ties go to the lower position");
        assert_eq!(map[&(7, 3)], 1);
        assert_eq!(map[&(7, 0)], 2);
        let (s, map) = assign_symbols(&[(7, &lo, &hi)], 2, 10);
        assert_eq!((s, map.len()), (2, 2));
        assert!(!map.contains_key(&(7, 0)));
        // Two inputs: ties by slot first.
        let a = t(&[1], &[0.0]);
        let b = t(&[1], &[2.0]);
        let (_, map) = assign_symbols(&[(9, &a, &b), (3, &a, &b)], 256, 10);
        assert_eq!((map[&(3, 0)], map[&(9, 0)]), (0, 1));
    }

    #[test]
    fn an_input_form_has_the_center_the_radius_on_its_symbol_and_the_rounding() {
        let lo = t(&[4], &[1.0, 1.0e10, -3.0, f32::NEG_INFINITY]);
        let hi = t(&[4], &[3.0, 2.0e10, 5.0, 1.0]);
        let mut symbol = HashMap::new();
        symbol.insert((0usize, 0usize), 1usize);
        symbol.insert((0, 2), 0);
        let f = input_form(0, &lo, &hi, 2, &symbol);
        // Element 0: symbol 1 carries the radius.
        assert_eq!(f.c[0], 2.0);
        assert_eq!(f.row(0), [0.0, 1.0]);
        assert_near_above(f.e[0], 4.0 * EPS * 4.0 + f64::MIN_POSITIVE, "slack only");
        // Element 1 has no symbol: its radius is an independent error.
        assert_eq!(f.c[1], 1.5e10);
        assert_eq!(f.row(1), [0.0, 0.0]);
        assert_near_above(
            f.e[1],
            5.0e9 + 4.0 * EPS * 3.0e10 + f64::MIN_POSITIVE,
            "radius and slack",
        );
        // Element 2: symbol 0.
        assert_eq!((f.c[2], f.row(2)[0]), (1.0, 4.0));
        // An infinite end is unbounded.
        assert!(f.e[3].is_infinite() && f.c[3] == 0.0);
        // A point input pays the minimum padding, with or without a symbol.
        let point = t(&[1], &[0.0]);
        for map in [HashMap::new(), HashMap::from([((0usize, 0usize), 0usize)])] {
            let p = input_form(0, &point, &point, 1, &map);
            assert!(p.e[0] >= f64::MIN_POSITIVE, "{:e}", p.e[0]);
        }
    }

    /// A graph where the output is `x - op(x)` for an operator that only moves or flips values:
    /// the correlation survives only if the form rules (not the interval fallback) handle `op`.
    fn tight_difference(
        wrap: impl Fn(&mut GraphBuilder, oxfuse_core::ValueId) -> (oxfuse_core::ValueId, bool),
    ) {
        let mut g = GraphBuilder::new(Limits::default());
        let x = g.input("x", vec![2, 3]).unwrap();
        let (y, subtract) = wrap(&mut g, x);
        let d = if subtract {
            g.node("d", Op::Sub, &[x, y]).unwrap()
        } else {
            g.node("d", Op::Add, &[x, y]).unwrap()
        };
        let graph = g.finish(&[d]).unwrap();
        let lo = t(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let hi = t(&[2, 3], &[2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);
        let e = analyze_box(&graph, &[lo], &[hi]).unwrap().remove(0);
        for i in 0..6 {
            assert!(e.lo[i] <= 0.0 && e.hi[i] >= 0.0);
            assert!(
                e.hi[i] - e.lo[i] < 1e-6,
                "element {i}: [{}, {}]",
                e.lo[i],
                e.hi[i]
            );
        }
    }

    #[test]
    fn neg_reshape_and_transpose_keep_the_correlation() {
        tight_difference(|g, x| (g.node("n", Op::Neg, &[x]).unwrap(), false));
        tight_difference(|g, x| {
            let r = g
                .node("r", Op::Reshape { shape: vec![3, 2] }, &[x])
                .unwrap();
            let back = g
                .node("b", Op::Reshape { shape: vec![2, 3] }, &[r])
                .unwrap();
            (back, true)
        });
        tight_difference(|g, x| {
            let t1 = g
                .node("t1", Op::Transpose { perm: vec![1, 0] }, &[x])
                .unwrap();
            let t2 = g
                .node("t2", Op::Transpose { perm: vec![1, 0] }, &[t1])
                .unwrap();
            (t2, true)
        });
    }

    #[test]
    fn a_matmul_of_two_varying_matrices_is_sound() {
        let (m, k, n) = (2, 3, 2);
        let mut g = GraphBuilder::new(Limits::default());
        let x = g.input("x", vec![m, k]).unwrap();
        let y = g.input("y", vec![k, n]).unwrap();
        let z = g.node("z", Op::MatMul, &[x, y]).unwrap();
        let graph = g.finish(&[z]).unwrap();
        let xs = [1.0f32, -2.0, 3.0, 0.5, 1.5, -1.0];
        let ys = [2.0f32, -1.0, 0.5, 4.0, -3.0, 1.5];
        let widen = |v: &[f32], r: f32| -> (Vec<f32>, Vec<f32>) {
            (
                v.iter().map(|a| a - r).collect(),
                v.iter().map(|a| a + r).collect(),
            )
        };
        let (xl, xh) = widen(&xs, 0.3);
        let (yl, yh) = widen(&ys, 0.2);
        let lo = [t(&[m, k], &xl), t(&[k, n], &yl)];
        let hi = [t(&[m, k], &xh), t(&[k, n], &yh)];
        let e = analyze_box(&graph, &lo, &hi).unwrap().remove(0);
        let exe = ReferenceBackend.compile(&graph).unwrap();
        for corner in 0..64u32 {
            let pick = |l: &[f32], h: &[f32], off: u32| -> Vec<f32> {
                (0..l.len())
                    .map(|i| {
                        if corner >> ((i as u32 + off) % 6) & 1 == 0 {
                            l[i]
                        } else {
                            h[i]
                        }
                    })
                    .collect()
            };
            let out = exe
                .run(&[
                    t(&[m, k], &pick(&xl, &xh, 0)),
                    t(&[k, n], &pick(&yl, &yh, 3)),
                ])
                .unwrap()
                .remove(0);
            for (i, &v) in out.data().iter().enumerate() {
                assert!(
                    e.lo[i] <= f64::from(v) && f64::from(v) <= e.hi[i],
                    "corner {corner} element {i}"
                );
            }
        }
    }
}
