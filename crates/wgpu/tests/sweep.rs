//! Whole-range checks of Relu and Sigmoid. The stratified sweep (every exponent,
//! sampled mantissas, both signs) runs everywhere; the exhaustive sweep over all
//! 2^32 bit patterns is `#[ignore]` (`just gpu-exhaustive`) and sets the Sigmoid bound.

mod common;

use common::{available, bits, device, measurements_dir, reference};
use oxfuse_core::Op;
use oxfuse_runtime::Tensor;
use oxfuse_wgpu::{sigmoid_matches, ulp_distance};

fn flat(values: Vec<f32>) -> Tensor {
    Tensor::new(vec![values.len()], values).unwrap()
}

/// Every exponent and sign, 64 mantissas each: zero, one, all-ones, single bits and a spread.
fn stratified() -> Vec<f32> {
    let mut mantissas = vec![0u32, 1, 2, 3, 0x7f_ffff, 0x7f_fffe, 0x40_0000, 0x3f_ffff];
    for bit in 0..23 {
        mantissas.push(1 << bit);
    }
    let mut state = 0x9e37_79b9_7f4a_7c15u64;
    while mantissas.len() < 64 {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        mantissas.push((state >> 41) as u32 & 0x7f_ffff);
    }
    let mut values = Vec::new();
    for sign in [0u32, 1] {
        for exponent in 0..=255u32 {
            for &m in &mantissas {
                values.push(f32::from_bits(sign << 31 | exponent << 23 | m));
            }
        }
    }
    values
}

#[test]
fn sweep_stratified() {
    if !available() {
        return;
    }
    let values = stratified();
    let t = flat(values.clone());
    assert_eq!(
        bits(&device(Op::Relu, std::slice::from_ref(&t))),
        bits(&reference(Op::Relu, std::slice::from_ref(&t))),
        "Relu must be bit-exact on every pattern"
    );
    let got = device(Op::Sigmoid, std::slice::from_ref(&t));
    let want = reference(Op::Sigmoid, &[t]);
    let mut worst = 0u32;
    for i in 0..values.len() {
        if !values[i].is_finite() {
            continue;
        }
        assert!(
            sigmoid_matches(got[i], want[i], common::sigmoid_bound()),
            "sigmoid({:e}) = {:e}, reference {:e}",
            values[i],
            got[i],
            want[i]
        );
        if want[i].is_normal() {
            worst = worst.max(ulp_distance(got[i], want[i]).unwrap());
        }
    }
    eprintln!("stratified sigmoid max ulp: {worst}");
}

#[test]
#[ignore = "all 2^32 patterns, minutes; run with `just gpu-exhaustive`"]
fn sweep_exhaustive() {
    if !available() {
        return;
    }
    const CHUNK: u64 = 1 << 24;
    let mut worst = (0u32, 0.0f32, 0.0f32, 0.0f32);
    let mut non_finite_inputs = 0u64;
    for chunk in 0..(1u64 << 32) / CHUNK {
        let values: Vec<f32> = (0..CHUNK)
            .map(|i| f32::from_bits((chunk * CHUNK + i) as u32))
            .collect();
        let t = flat(values.clone());
        let relu = device(Op::Relu, std::slice::from_ref(&t));
        for (i, x) in values.iter().enumerate() {
            let want = if x.is_nan() || *x > 0.0 { *x } else { 0.0 };
            assert_eq!(
                relu[i].to_bits(),
                want.to_bits(),
                "relu({:#010x})",
                x.to_bits()
            );
        }
        let got = device(Op::Sigmoid, std::slice::from_ref(&t));
        let wanted = reference(Op::Sigmoid, &[t]);
        for (i, x) in values.iter().enumerate() {
            if !x.is_finite() {
                non_finite_inputs += 1;
                continue;
            }
            let want = wanted[i];
            assert!(
                sigmoid_matches(got[i], want, common::sigmoid_bound()),
                "sigmoid({:#010x} = {x:e}) = {:e}, reference {want:e}",
                x.to_bits(),
                got[i]
            );
            if want.is_normal() {
                let d = ulp_distance(got[i], want).unwrap();
                if d > worst.0 {
                    worst = (d, *x, got[i], want);
                }
            }
        }
    }
    let line = format!(
        "sigmoid max ulp distance over all finite inputs: {} at x={:e} (device {:e}, reference {:e}); non-finite inputs skipped: {non_finite_inputs}\n",
        worst.0, worst.1, worst.2, worst.3
    );
    std::fs::write(measurements_dir().join("sigmoid-ulp.txt"), &line).unwrap();
    eprintln!("{line}");
}
