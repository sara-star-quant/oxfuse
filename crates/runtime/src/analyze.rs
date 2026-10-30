//! A certified error bound for a whole graph, for the inputs you give.
//!
//! Every element of an output carries `bound` such that the exact real-number
//! result of the same graph on the same float32 inputs lies within `value ± bound`.
//! `value` is what the reference backend returns, bit for bit.
//!
//! The method is the textbook first-order rounding analysis (Higham, "Accuracy
//! and Stability of Numerical Algorithms") with the IEEE 754 round-to-nearest
//! rule: a rounded result is within half an ulp of the exact one. Because the
//! order and the rounding of every operator are defined (see `docs/semantics.md`),
//! the bound is computed by replaying the same steps. It is rigorous for these
//! inputs, and pessimistic: it adds up worst cases.
//!
//! The bound itself is computed in float64 and rounded up after every operation
//! (`up`), so rounding in the analysis can only make it larger.
//!
//! Elements that are not finite get an infinite bound.

use crate::{Plan, Tensor, kernels};
use oxfuse_core::{Error, Graph, Op};

/// The result of [`analyze`] for one output.
#[derive(Debug, Clone, PartialEq)]
pub struct Certificate {
    pub name: String,
    pub shape: Vec<usize>,
    /// What the reference backend returns.
    pub values: Vec<f32>,
    /// `|exact - value| <= bounds[i]`, or infinity when the element is not finite.
    pub bounds: Vec<f64>,
}

/// 2^n as a float64, exactly (`n` within the normal float64 range).
const fn pow2(n: i32) -> f64 {
    f64::from_bits(((1023 + n) as u64) << 52)
}

/// Relative error of the own `sigmoid`: `exp` is within 1 ulp (2^-23 relative)
/// of the correctly rounded result for every input (docs/claims.md), the
/// addition and the division round once each (2^-24), and the second-order
/// terms fit in the factor.
pub(crate) const SIGMOID_RELATIVE: f64 = pow2(-22) * (1.0 + pow2(-20));

/// Added to a sigmoid bound: the spacing of subnormals, for results that
/// underflow.
const SIGMOID_ABSOLUTE: f64 = pow2(-149);

/// Below this input the sigmoid is smaller than 2^-125, where `exp` of the
/// negated input overflows and the relative model stops holding.
const SIGMOID_TINY_BELOW: f32 = -87.0;
const SIGMOID_TINY_BOUND: f64 = pow2(-125);

/// Round a float64 result up by one unit of its relative error.
fn up(x: f64) -> f64 {
    x * (1.0 + f64::EPSILON)
}

fn add(x: f64, y: f64) -> f64 {
    up(x + y)
}

fn mul(x: f64, y: f64) -> f64 {
    up(x * y)
}

/// Half an ulp of `v`: the most that rounding to nearest can move a result of
/// this size. Below the normal range the spacing is fixed at 2^-149.
pub(crate) fn half_ulp(v: f32) -> f64 {
    if !v.is_finite() {
        return f64::INFINITY;
    }
    // The exponent field is 0 for subnormals, which then count as 2^-126.
    let exponent = ((((v.abs().to_bits() >> 23) & 0xff) as i32) - 127).max(-126);
    f64::from_bits(((1023 + exponent - 24) as u64) << 52)
}

/// Run `graph` on `inputs` and certify every output element.
pub fn analyze(graph: &Graph, inputs: &[Tensor]) -> Result<Vec<Certificate>, Error> {
    let plan = Plan::build(graph);
    if inputs.len() != plan.inputs.len() {
        return Err(Error::InputCount {
            expected: plan.inputs.len(),
            found: inputs.len(),
        });
    }
    let slots = plan.slot_shapes.len();
    let mut values: Vec<Vec<f32>> = vec![Vec::new(); slots];
    let mut bounds: Vec<Vec<f64>> = vec![Vec::new(); slots];
    for (binding, tensor) in plan.inputs.iter().zip(inputs) {
        if tensor.shape() != binding.shape.as_slice() {
            return Err(Error::InputShape {
                name: binding.name.clone(),
                expected: binding.shape.clone(),
                found: tensor.shape().to_vec(),
            });
        }
        values[binding.slot] = tensor.data().to_vec();
        bounds[binding.slot] = vec![0.0; tensor.data().len()];
    }
    for (slot, data) in &plan.constants {
        values[*slot] = data.to_vec();
        bounds[*slot] = vec![0.0; data.len()];
    }
    for step in &plan.steps {
        let in_values: Vec<&[f32]> = step.inputs.iter().map(|&s| values[s].as_slice()).collect();
        let in_bounds: Vec<&[f64]> = step.inputs.iter().map(|&s| bounds[s].as_slice()).collect();
        let shapes: Vec<&[usize]> = step
            .inputs
            .iter()
            .map(|&s| plan.slot_shapes[s].as_slice())
            .collect();
        let out = kernels::apply(&step.op, &in_values, &shapes);
        let mut err = propagate(&step.op, &in_values, &in_bounds, &shapes, &out);
        for (e, v) in err.iter_mut().zip(&out) {
            if !v.is_finite() {
                *e = f64::INFINITY;
            }
        }
        values[step.output] = out;
        bounds[step.output] = err;
    }
    Ok(plan
        .outputs
        .iter()
        .zip(graph.output_names())
        .map(|(&slot, name)| Certificate {
            name: name.clone(),
            shape: plan.slot_shapes[slot].clone(),
            values: values[slot].clone(),
            bounds: bounds[slot].clone(),
        })
        .collect())
}

/// Index of the operand element that pairs with output element `i`, for an op
/// whose operands have equal shapes or one scalar.
fn operand_index(shape: &[usize], i: usize) -> usize {
    if shape.is_empty() { 0 } else { i }
}

fn propagate(op: &Op, v: &[&[f32]], e: &[&[f64]], shapes: &[&[usize]], out: &[f32]) -> Vec<f64> {
    match op {
        Op::Relu => e[0].to_vec(),
        Op::Reshape { .. } => e[0].to_vec(),
        Op::Transpose { perm } => kernels::transpose(e[0], shapes[0], perm),
        Op::Add => (0..out.len())
            .map(|i| {
                let (a, b) = (operand_index(shapes[0], i), operand_index(shapes[1], i));
                add(add(e[0][a], e[1][b]), half_ulp(out[i]))
            })
            .collect(),
        Op::Mul => (0..out.len())
            .map(|i| {
                let (a, b) = (operand_index(shapes[0], i), operand_index(shapes[1], i));
                let (x, y) = (f64::from(v[0][a]), f64::from(v[1][b]));
                let cross = add(mul(x.abs(), e[1][b]), mul(y.abs(), e[0][a]));
                add(add(cross, mul(e[0][a], e[1][b])), half_ulp(out[i]))
            })
            .collect(),
        Op::Sigmoid => out
            .iter()
            .zip(v[0].iter().zip(e[0]))
            .map(|(&s, (&x, &ex))| {
                let mut bound = add(mul(ex, 0.25), mul(SIGMOID_RELATIVE, f64::from(s.abs())));
                bound = add(bound, SIGMOID_ABSOLUTE);
                if x < SIGMOID_TINY_BELOW {
                    bound = add(bound, SIGMOID_TINY_BOUND);
                }
                bound
            })
            .collect(),
        Op::MatMul => matmul_bounds(v, e, shapes),
    }
}

/// Replays the fixed order of the reference kernel: for each output element,
/// one fused multiply-add per step in ascending `k`, from `+0.0`.
fn matmul_bounds(v: &[&[f32]], e: &[&[f64]], shapes: &[&[usize]]) -> Vec<f64> {
    let (m, k, n) = (shapes[0][0], shapes[0][1], shapes[1][1]);
    let mut out = vec![0.0f64; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f32;
            let mut err = 0.0f64;
            for p in 0..k {
                let (a, b) = (v[0][i * k + p], v[1][p * n + j]);
                let (ea, eb) = (e[0][i * k + p], e[1][p * n + j]);
                acc = a.mul_add(b, acc);
                let (x, y) = (f64::from(a), f64::from(b));
                let step = add(add(mul(x.abs(), eb), mul(y.abs(), ea)), mul(ea, eb));
                err = add(add(err, step), half_ulp(acc));
            }
            out[i * n + j] = err;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Backend, ReferenceBackend};
    use oxfuse_core::{GraphBuilder, Limits};

    #[test]
    fn half_ulp_is_half_the_spacing_of_the_binade() {
        assert_eq!(half_ulp(1.0), pow2(-24));
        assert_eq!(half_ulp(1.5), pow2(-24));
        assert_eq!(half_ulp(-1.5), pow2(-24));
        assert_eq!(half_ulp(2.0), pow2(-23));
        assert_eq!(half_ulp(0.5), pow2(-25));
        assert_eq!(half_ulp(0.75), pow2(-25));
        assert_eq!(half_ulp(f32::MAX), pow2(103));
    }

    #[test]
    fn half_ulp_below_the_normal_range_is_the_fixed_subnormal_spacing() {
        assert_eq!(half_ulp(0.0), pow2(-150));
        assert_eq!(half_ulp(-0.0), pow2(-150));
        assert_eq!(half_ulp(f32::MIN_POSITIVE / 2.0), pow2(-150));
        assert_eq!(half_ulp(f32::MIN_POSITIVE), pow2(-150));
        assert_eq!(half_ulp(f32::MIN_POSITIVE * 2.0), pow2(-149));
    }

    #[test]
    fn half_ulp_of_a_non_finite_value_is_infinite() {
        assert_eq!(half_ulp(f32::INFINITY), f64::INFINITY);
        assert_eq!(half_ulp(f32::NAN), f64::INFINITY);
    }

    #[test]
    fn up_only_ever_rounds_up() {
        assert!(up(1.0) > 1.0);
        assert!(up(0.0) == 0.0);
        assert!(up(1.0) <= 1.0 + 4.0 * f64::EPSILON);
        assert!(add(1.0, 2.0) >= 3.0 && mul(3.0, 4.0) >= 12.0);
    }

    /// A graph of one binary op over a fixed pair of inputs.
    fn one(op: Op, a: Vec<usize>, b: Vec<usize>, av: &[f32], bv: &[f32]) -> Certificate {
        let mut g = GraphBuilder::new(Limits::default());
        let x = g.input("x", a.clone()).unwrap();
        let y = g.input("y", b.clone()).unwrap();
        let z = g.node("z", op, &[x, y]).unwrap();
        let graph = g.finish(&[z]).unwrap();
        let inputs = [
            Tensor::new(a, av.to_vec()).unwrap(),
            Tensor::new(b, bv.to_vec()).unwrap(),
        ];
        analyze(&graph, &inputs).unwrap().remove(0)
    }

    #[test]
    fn an_addition_is_bounded_by_half_an_ulp_of_the_sum() {
        let c = one(Op::Add, vec![1], vec![1], &[1.0], &[1e-8]);
        assert_eq!(c.values, [1.0]);
        // The exact sum is 1 + 1e-8, so the rounding error is 1e-8 < half an ulp.
        assert!(c.bounds[0] >= pow2(-24));
        assert!(c.bounds[0] <= pow2(-24) * (1.0 + 1e-14));
        assert!(1e-8 <= c.bounds[0]);
    }

    #[test]
    fn a_product_adds_the_rounding_of_the_result_to_nothing_when_inputs_are_exact() {
        let c = one(Op::Mul, vec![1], vec![1], &[3.0], &[0.1]);
        let exact = 3.0f64 * f64::from(0.1f32);
        assert!((exact - f64::from(c.values[0])).abs() <= c.bounds[0]);
        assert!(c.bounds[0] <= half_ulp(c.values[0]) * (1.0 + 1e-14));
    }

    #[test]
    fn a_scalar_operand_pairs_with_every_element() {
        let c = one(Op::Add, vec![], vec![3], &[0.5], &[1.0, 2.0, 4.0]);
        assert_eq!(c.values, [1.5, 2.5, 4.5]);
        // The bounds differ by element: each uses the half ulp of its own result.
        assert!(c.bounds[0] < c.bounds[1] && c.bounds[1] < c.bounds[2]);
        let c = one(Op::Mul, vec![3], vec![], &[1.0, 2.0, 4.0], &[0.5]);
        assert_eq!(c.values, [0.5, 1.0, 2.0]);
    }

    #[test]
    fn non_finite_values_get_an_infinite_bound() {
        let c = one(Op::Mul, vec![2], vec![2], &[f32::MAX, 1.0], &[2.0, 1.0]);
        assert!(c.values[0].is_infinite());
        assert_eq!(c.bounds[0], f64::INFINITY);
        assert!(c.bounds[1].is_finite());
        let c = one(Op::Add, vec![1], vec![1], &[f32::NAN], &[1.0]);
        assert_eq!(c.bounds[0], f64::INFINITY);
    }

    #[test]
    fn a_wrong_input_count_or_shape_is_an_error() {
        let mut g = GraphBuilder::new(Limits::default());
        let x = g.input("x", vec![2]).unwrap();
        let y = g.node("y", Op::Relu, &[x]).unwrap();
        let graph = g.finish(&[y]).unwrap();
        assert!(matches!(
            analyze(&graph, &[]),
            Err(Error::InputCount { .. })
        ));
        let wrong = Tensor::new(vec![3], vec![0.0; 3]).unwrap();
        assert!(matches!(
            analyze(&graph, &[wrong]),
            Err(Error::InputShape { .. })
        ));
    }

    /// Deterministic values in [-2, 2) times a scale that spans decades.
    fn values(len: usize, seed: u64, scale: f64) -> Vec<f32> {
        let mut state = seed;
        (0..len)
            .map(|_| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let unit = (state >> 40) as f64 / (1u64 << 24) as f64;
                ((unit * 4.0 - 2.0) * scale) as f32
            })
            .collect()
    }

    /// The exact real result, to within float64's own error (about 1e-16
    /// relative), which is far below any bound the analysis returns.
    fn exact(graph: &Graph, inputs: &[Tensor]) -> Vec<Vec<f64>> {
        let plan = Plan::build(graph);
        let mut slots: Vec<Vec<f64>> = vec![Vec::new(); plan.slot_shapes.len()];
        for (b, t) in plan.inputs.iter().zip(inputs) {
            slots[b.slot] = t.data().iter().map(|&x| f64::from(x)).collect();
        }
        for (slot, data) in &plan.constants {
            slots[*slot] = data.iter().map(|&x| f64::from(x)).collect();
        }
        for step in &plan.steps {
            let shapes: Vec<&[usize]> = step
                .inputs
                .iter()
                .map(|&s| plan.slot_shapes[s].as_slice())
                .collect();
            let a = &slots[step.inputs[0]];
            let out: Vec<f64> = match &step.op {
                Op::Relu => a.iter().map(|&x| x.max(0.0)).collect(),
                Op::Sigmoid => a.iter().map(|&x| 1.0 / (1.0 + (-x).exp())).collect(),
                Op::Reshape { .. } => a.clone(),
                Op::Transpose { perm } => kernels::transpose(a, shapes[0], perm),
                Op::Add | Op::Mul => {
                    let b = &slots[step.inputs[1]];
                    let n = plan.slot_shapes[step.output].iter().product::<usize>();
                    (0..n)
                        .map(|i| {
                            let (x, y) = (
                                a[operand_index(shapes[0], i)],
                                b[operand_index(shapes[1], i)],
                            );
                            if matches!(step.op, Op::Add) {
                                x + y
                            } else {
                                x * y
                            }
                        })
                        .collect()
                }
                Op::MatMul => {
                    let b = &slots[step.inputs[1]];
                    let (m, k, n) = (shapes[0][0], shapes[0][1], shapes[1][1]);
                    let mut out = vec![0.0; m * n];
                    for i in 0..m {
                        for j in 0..n {
                            out[i * n + j] = (0..k).map(|p| a[i * k + p] * b[p * n + j]).sum();
                        }
                    }
                    out
                }
            };
            slots[step.output] = out;
        }
        plan.outputs.iter().map(|&s| slots[s].clone()).collect()
    }

    /// A random small graph over every operator, with inputs of mixed magnitude.
    fn random_case(seed: u64) -> (Graph, Vec<Tensor>) {
        let pick = |salt: u64, n: usize| {
            (seed.wrapping_mul(2654435761).wrapping_add(salt * 97) >> 7) as usize % n
        };
        let (m, k, n) = (1 + pick(1, 5), 1 + pick(2, 40), 1 + pick(3, 6));
        let scale = [1e-3, 1.0, 1e3][pick(4, 3)];
        let mut g = GraphBuilder::new(Limits::default());
        let x = g.input("x", vec![m, k]).unwrap();
        let w = g
            .constant("w", vec![k, n], values(k * n, seed + 1, scale))
            .unwrap();
        let mut cur = g.node("h", Op::MatMul, &[x, w]).unwrap();
        if pick(5, 2) == 0 {
            cur = g.node("s", Op::Sigmoid, &[cur]).unwrap();
        }
        if pick(6, 2) == 0 {
            let c = g.constant("c", vec![], vec![0.37]).unwrap();
            cur = g.node("a", Op::Add, &[cur, c]).unwrap();
        }
        if pick(7, 2) == 0 {
            let c = g.constant("c2", vec![], vec![-1.7]).unwrap();
            cur = g.node("p", Op::Mul, &[c, cur]).unwrap();
        }
        if pick(8, 2) == 0 {
            cur = g.node("r", Op::Relu, &[cur]).unwrap();
        }
        if pick(9, 2) == 0 {
            cur = g
                .node("t", Op::Transpose { perm: vec![1, 0] }, &[cur])
                .unwrap();
        }
        let graph = g.finish(&[cur]).unwrap();
        let input = Tensor::new(vec![m, k], values(m * k, seed + 2, scale)).unwrap();
        (graph, vec![input])
    }

    #[test]
    fn the_exact_result_is_inside_the_bound_for_300_random_graphs() {
        let mut widest = 0.0f64;
        for seed in 0..300 {
            let (graph, inputs) = random_case(seed);
            let cert = analyze(&graph, &inputs).unwrap().remove(0);
            let want = exact(&graph, &inputs).remove(0);
            for (i, ((&v, &b), &x)) in cert.values.iter().zip(&cert.bounds).zip(&want).enumerate() {
                let off = (x - f64::from(v)).abs();
                assert!(
                    off <= b * (1.0 + 1e-12) + 1e-300,
                    "seed {seed} element {i}: |{x} - {v}| = {off} > bound {b}"
                );
                if b > 0.0 && b.is_finite() {
                    widest = widest.max(b / off.max(f64::MIN_POSITIVE));
                }
            }
        }
        assert!(widest > 0.0);
    }

    #[test]
    fn the_values_are_the_reference_backends_bits() {
        for seed in 0..100 {
            let (graph, inputs) = random_case(seed);
            let cert = analyze(&graph, &inputs).unwrap().remove(0);
            let run = ReferenceBackend
                .compile(&graph)
                .unwrap()
                .run(&inputs)
                .unwrap();
            let bits = |d: &[f32]| d.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
            assert_eq!(bits(&cert.values), bits(run[0].data()), "seed {seed}");
        }
    }

    #[test]
    fn a_long_sum_is_bounded_but_the_bound_grows_with_k() {
        let bound_for = |k: usize| {
            let mut g = GraphBuilder::new(Limits::default());
            let x = g.input("x", vec![1, k]).unwrap();
            let w = g.constant("w", vec![k, 1], values(k, 9, 1.0)).unwrap();
            let y = g.node("y", Op::MatMul, &[x, w]).unwrap();
            let graph = g.finish(&[y]).unwrap();
            let input = Tensor::new(vec![1, k], values(k, 10, 1.0)).unwrap();
            let cert = analyze(&graph, std::slice::from_ref(&input))
                .unwrap()
                .remove(0);
            let want = exact(&graph, std::slice::from_ref(&input)).remove(0)[0];
            assert!((want - f64::from(cert.values[0])).abs() <= cert.bounds[0]);
            cert.bounds[0]
        };
        assert!(bound_for(1000) > bound_for(10) * 20.0);
    }

    /// Both operands of a `MatMul` carry a different bound in every element (each
    /// came from an `Add` of a different size), so a bound read from the wrong
    /// place changes the result. Compared with a plain replay of the formula.
    #[test]
    fn matmul_reads_each_operand_bound_from_its_own_element() {
        let (m, k, n) = (2, 3, 4);
        let mut g = GraphBuilder::new(Limits::default());
        let x = g.input("x", vec![m, k]).unwrap();
        let cx = g.constant("cx", vec![], vec![0.3]).unwrap();
        let a = g.node("a", Op::Add, &[x, cx]).unwrap();
        let w = g.input("w", vec![k, n]).unwrap();
        let cw = g.constant("cw", vec![], vec![0.7]).unwrap();
        let b = g.node("b", Op::Add, &[w, cw]).unwrap();
        let c = g.node("c", Op::MatMul, &[a, b]).unwrap();
        let graph = g.finish(&[a, b, c]).unwrap();
        let xs = Tensor::new(vec![m, k], vec![1.0, 100.0, 0.01, 7.0, 0.5, 3000.0]).unwrap();
        let ws = Tensor::new(
            vec![k, n],
            (0..k * n).map(|i| 10f32.powi(i as i32 % 5 - 2)).collect(),
        )
        .unwrap();
        let certs = analyze(&graph, &[xs, ws]).unwrap();
        let (av, ea) = (&certs[0].values, &certs[0].bounds);
        let (bv, eb) = (&certs[1].values, &certs[1].bounds);
        assert!(ea.windows(2).any(|p| p[0] != p[1]) && eb.windows(2).any(|p| p[0] != p[1]));
        for i in 0..m {
            for j in 0..n {
                let (mut acc, mut want) = (0.0f32, 0.0f64);
                for p in 0..k {
                    let (x, y) = (av[i * k + p], bv[p * n + j]);
                    let (ex, ey) = (ea[i * k + p], eb[p * n + j]);
                    acc = x.mul_add(y, acc);
                    want +=
                        f64::from(x.abs()) * ey + f64::from(y.abs()) * ex + ex * ey + half_ulp(acc);
                }
                let got = certs[2].bounds[i * n + j];
                assert!(got >= want, "[{i}][{j}] {got} < {want}");
                assert!(got <= want * (1.0 + 1e-12), "[{i}][{j}] {got} vs {want}");
            }
        }
    }

    #[test]
    fn transpose_and_reshape_move_each_bound_with_its_element() {
        let mut g = GraphBuilder::new(Limits::default());
        let x = g.input("x", vec![2, 3]).unwrap();
        let w = g.constant("w", vec![3, 3], values(9, 3, 1000.0)).unwrap();
        let m = g.node("m", Op::MatMul, &[x, w]).unwrap();
        let t = g
            .node("t", Op::Transpose { perm: vec![1, 0] }, &[m])
            .unwrap();
        let graph = g.finish(&[m, t]).unwrap();
        let input = Tensor::new(vec![2, 3], values(6, 4, 1000.0)).unwrap();
        let certs = analyze(&graph, &[input]).unwrap();
        for i in 0..2 {
            for j in 0..3 {
                assert_eq!(certs[0].bounds[i * 3 + j], certs[1].bounds[j * 2 + i]);
            }
        }
    }

    #[test]
    fn relu_keeps_the_bound_of_its_input() {
        let mut g = GraphBuilder::new(Limits::default());
        let x = g.input("x", vec![4]).unwrap();
        let c = g.constant("c", vec![], vec![0.1]).unwrap();
        let a = g.node("a", Op::Add, &[x, c]).unwrap();
        let r = g.node("r", Op::Relu, &[a]).unwrap();
        let graph = g.finish(&[a, r]).unwrap();
        let input = Tensor::new(vec![4], vec![-3.0, -0.05, 0.05, 3.0]).unwrap();
        let certs = analyze(&graph, &[input]).unwrap();
        assert_eq!(certs[0].bounds, certs[1].bounds);
    }

    /// The claimed relative error of the own `sigmoid` holds on a dense sample of
    /// bit patterns (the exhaustive proof of its `exp` is in docs/claims.md).
    #[test]
    fn the_own_sigmoid_stays_within_the_relative_bound_on_a_dense_sample() {
        let mut worst = 0.0f64;
        let mut bits = 0u32;
        while bits < 0x7f80_0000 {
            for sign in [0u32, 0x8000_0000] {
                let x = f32::from_bits(bits | sign);
                let got = f64::from(crate::math::sigmoid(x));
                let exact = 1.0 / (1.0 + (-f64::from(x)).exp());
                if exact >= f64::from(f32::MIN_POSITIVE) {
                    worst = worst.max(((got - exact) / exact).abs());
                }
            }
            bits += 4099;
        }
        assert!(worst <= SIGMOID_RELATIVE, "worst relative error {worst}");
        assert!(
            worst > SIGMOID_RELATIVE / 64.0,
            "the bound is not absurdly loose: {worst}"
        );
    }

    #[test]
    fn the_sigmoid_constants_are_pinned() {
        assert_eq!(SIGMOID_RELATIVE, pow2(-22) * (1.0 + pow2(-20)));
        assert_eq!(SIGMOID_ABSOLUTE, pow2(-149));
        assert_eq!(SIGMOID_TINY_BOUND, pow2(-125));
        assert_eq!(SIGMOID_TINY_BELOW, -87.0);
    }

    /// The extra absolute term starts strictly below -87 and not at it.
    #[test]
    fn the_tiny_result_term_starts_just_below_minus_87() {
        let mut g = GraphBuilder::new(Limits::default());
        let x = g.input("x", vec![2]).unwrap();
        let y = g.node("y", Op::Sigmoid, &[x]).unwrap();
        let graph = g.finish(&[y]).unwrap();
        let below = f32::from_bits((-87.0f32).to_bits() + 1);
        assert!(below < -87.0);
        let input = Tensor::new(vec![2], vec![-87.0, below]).unwrap();
        let cert = analyze(&graph, &[input]).unwrap().remove(0);
        assert!(cert.bounds[0] < 1e-40, "{}", cert.bounds[0]);
        assert!(cert.bounds[1] > 1e-38, "{}", cert.bounds[1]);
    }

    #[test]
    fn a_very_negative_input_is_still_inside_the_bound() {
        let mut g = GraphBuilder::new(Limits::default());
        let x = g.input("x", vec![3]).unwrap();
        let y = g.node("y", Op::Sigmoid, &[x]).unwrap();
        let graph = g.finish(&[y]).unwrap();
        let input = Tensor::new(vec![3], vec![-100.0, -88.0, -87.5]).unwrap();
        let cert = analyze(&graph, std::slice::from_ref(&input))
            .unwrap()
            .remove(0);
        let want = exact(&graph, std::slice::from_ref(&input)).remove(0);
        for (i, ((&w, &v), &b)) in want.iter().zip(&cert.values).zip(&cert.bounds).enumerate() {
            assert!((w - f64::from(v)).abs() <= b, "x index {i}");
        }
    }
}
