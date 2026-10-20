//! What the adapter does with the awkward values. Relu is exact for every input
//! (it works on bits). Add, Mul and Sigmoid are asserted only in the finite tier;
//! everything else is measured and written down, never promised.

mod common;

use common::{available, bits, device, measurements_dir, reference};
use oxfuse_core::Op;
use oxfuse_runtime::Tensor;
use std::fmt::Write;

fn class(x: f32) -> &'static str {
    let sign = if x.is_sign_negative() { '-' } else { '+' };
    match (x.is_nan(), x.is_infinite(), x == 0.0, x.is_normal(), sign) {
        (true, ..) => "nan",
        (_, true, .., '+') => "+inf",
        (_, true, ..) => "-inf",
        (_, _, true, _, '+') => "+0",
        (_, _, true, ..) => "-0",
        (_, _, _, false, '+') => "+subnormal",
        (_, _, _, false, _) => "-subnormal",
        (.., '+') => "+normal",
        _ => "-normal",
    }
}

fn table() -> Vec<f32> {
    vec![
        f32::NAN,
        -f32::NAN,
        f32::INFINITY,
        f32::NEG_INFINITY,
        0.0,
        -0.0,
        1.0,
        -1.0,
        f32::MIN_POSITIVE,
        f32::from_bits(1),
        -f32::from_bits(1),
        f32::MAX,
        -f32::MAX,
        88.0,
        -88.0,
        100.0,
        -100.0,
    ]
}

fn flat(values: &[f32]) -> Tensor {
    Tensor::new(vec![values.len()], values.to_vec()).unwrap()
}

#[test]
fn relu_is_bit_exact_for_every_special_value() {
    if !available() {
        return;
    }
    let t = flat(&table());
    assert_eq!(
        bits(&device(Op::Relu, std::slice::from_ref(&t))),
        bits(&reference(Op::Relu, &[t]))
    );
}

#[test]
fn sigmoid_of_large_finite_inputs_is_finite_and_in_range() {
    if !available() {
        return;
    }
    let xs = [
        -1e30f32, -1000.0, -100.0, -90.0, -88.0, -20.0, 0.0, 20.0, 88.0, 90.0, 100.0, 1000.0, 1e30,
    ];
    let got = device(Op::Sigmoid, &[flat(&xs)]);
    for (x, y) in xs.iter().zip(&got) {
        assert!(
            y.is_finite() && (0.0..=1.0).contains(y),
            "sigmoid({x}) = {y}"
        );
    }
    assert_eq!(got[6], 0.5);
}

#[test]
fn add_and_mul_are_exact_in_the_finite_tier() {
    if !available() {
        return;
    }
    let values = table();
    let (mut a, mut b) = (Vec::new(), Vec::new());
    for x in &values {
        for y in &values {
            a.push(*x);
            b.push(*y);
        }
    }
    let n = a.len();
    for op in [Op::Add, Op::Mul] {
        let inputs = [flat(&a), flat(&b)];
        let got = device(op.clone(), &inputs);
        let want = reference(op.clone(), &inputs);
        for i in 0..n {
            let in_tier = |v: f32| v.is_finite() && (v == 0.0 || v.is_normal());
            if in_tier(a[i]) && in_tier(b[i]) && (want[i] == 0.0 || want[i].is_normal()) {
                assert_eq!(
                    got[i].to_bits(),
                    want[i].to_bits(),
                    "{op:?}({:e}, {:e}) = {:e}, want {:e}",
                    a[i],
                    b[i],
                    got[i],
                    want[i]
                );
            }
        }
    }
}

/// Measured, never asserted: written next to the other measurements for the ledger.
#[test]
fn special_values_report() {
    if !available() {
        return;
    }
    let values = table();
    let mut report = String::new();
    writeln!(report, "adapter: {}", common::gpu().context().info().name).unwrap();
    writeln!(report, "\nunary: input -> device (reference)").unwrap();
    for op in [Op::Relu, Op::Sigmoid] {
        let t = flat(&values);
        let got = device(op.clone(), std::slice::from_ref(&t));
        let want = reference(op.clone(), &[t]);
        for (i, x) in values.iter().enumerate() {
            let note = if got[i].to_bits() == want[i].to_bits() {
                "same"
            } else {
                "differs"
            };
            writeln!(
                report,
                "{op:?} {} {x:e} -> {} {:e} ({} {:e}) {note}",
                class(*x),
                class(got[i]),
                got[i],
                class(want[i]),
                want[i]
            )
            .unwrap();
        }
    }
    writeln!(report, "\nbinary: a, b -> device (reference)").unwrap();
    let (mut a, mut b) = (Vec::new(), Vec::new());
    for x in &values {
        for y in &values {
            a.push(*x);
            b.push(*y);
        }
    }
    for op in [Op::Add, Op::Mul] {
        let inputs = [flat(&a), flat(&b)];
        let got = device(op.clone(), &inputs);
        let want = reference(op.clone(), &inputs);
        let differing = (0..a.len())
            .filter(|&i| class(got[i]) != class(want[i]))
            .count();
        writeln!(
            report,
            "{op:?}: {differing} of {} pairs land in a different class than the reference",
            a.len()
        )
        .unwrap();
        for i in 0..a.len() {
            if class(got[i]) != class(want[i]) {
                writeln!(
                    report,
                    "  {} {:e}, {} {:e} -> {} (reference {})",
                    class(a[i]),
                    a[i],
                    class(b[i]),
                    b[i],
                    class(got[i]),
                    class(want[i])
                )
                .unwrap();
            }
        }
    }
    std::fs::write(measurements_dir().join("special-values.txt"), &report).unwrap();
    eprintln!("{report}");
}
