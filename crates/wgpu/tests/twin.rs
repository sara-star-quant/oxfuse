//! Model, device and reference on the same inputs. The model is built from the
//! same kernel configuration as the shaders; where an adapter exists these tests
//! tie it to the real shader text, which a model alone cannot prove.

mod common;

use common::{available, bits, device, reference, tensor};
use oxfuse_core::Op;
use oxfuse_wgpu::shaders::{Binary, Unary};
use oxfuse_wgpu::{DeviceCaps, KernelConfig, matmul_bound, model};
use proptest::prelude::*;

const CFG: KernelConfig = KernelConfig::DEFAULT;
const CAPS: DeviceCaps = DeviceCaps::DOWNLEVEL;

proptest! {
    #![proptest_config(ProptestConfig::with_cases(48))]

    #[test]
    fn relu_agrees_everywhere_including_non_finite(n in 1usize..600, seed in 0u64..1000) {
        if !available() {
            return Ok(());
        }
        let t = tensor(&[n], seed, false);
        let m = model::unary(Unary::Relu, t.data(), &CFG, &CAPS);
        let d = device(Op::Relu, std::slice::from_ref(&t));
        prop_assert_eq!(bits(&m), bits(&d));
        prop_assert_eq!(bits(&d), bits(&reference(Op::Relu, &[t])));
    }

    #[test]
    fn add_and_mul_agree_on_finite_normal_values(n in 1usize..600, seed in 0u64..1000) {
        if !available() {
            return Ok(());
        }
        let a = tensor(&[n], seed, true);
        let b = tensor(&[n], seed + 1, true);
        for (op, kind) in [(Op::Add, Binary::Add), (Op::Mul, Binary::Mul)] {
            let m = model::binary(kind, a.data(), b.data(), false, false, &CFG, &CAPS);
            let d = device(op.clone(), &[a.clone(), b.clone()]);
            // Finite tier: the product or sum can still land in the subnormal
            // range, where the device flushes. Compare only where the result is normal.
            for i in 0..n {
                if m[i].is_normal() || m[i] == 0.0 {
                    prop_assert_eq!(m[i].to_bits() & 0x7fff_ffff == 0, d[i].to_bits() & 0x7fff_ffff == 0);
                    if m[i] != 0.0 {
                        prop_assert_eq!(m[i].to_bits(), d[i].to_bits(), "{:?} at {}", kind, i);
                    }
                }
            }
        }
    }

    #[test]
    fn transpose_and_permute_agree(rows in 1usize..40, cols in 1usize..40, seed in 0u64..1000) {
        if !available() {
            return Ok(());
        }
        let t = tensor(&[rows, cols], seed, false);
        let m = model::transpose2d(t.data(), rows, cols, &CFG, &CAPS);
        let d = device(Op::Transpose { perm: vec![1, 0] }, std::slice::from_ref(&t));
        prop_assert_eq!(bits(&m), bits(&d));
        let m = model::permute(t.data(), &[rows, cols], &[0, 1], &CFG, &CAPS);
        prop_assert_eq!(bits(&m), bits(t.data()));
    }

    #[test]
    fn matmul_model_equals_reference_and_device_stays_in_the_bound(
        m in 1usize..140, k in 1usize..140, n in 1usize..140, seed in 0u64..1000,
    ) {
        if !available() {
            return Ok(());
        }
        let a = tensor(&[m, k], seed, true);
        let b = tensor(&[k, n], seed + 1, true);
        let modelled = model::matmul(a.data(), b.data(), m, k, n, &CFG, &CAPS);
        let want = reference(Op::MatMul, &[a.clone(), b.clone()]);
        prop_assert_eq!(bits(&modelled), bits(&want));
        let got = device(Op::MatMul, &[a.clone(), b.clone()]);
        for i in 0..m {
            let row: Vec<f32> = (0..k).map(|x| a.data()[i * k + x]).collect();
            for j in 0..n {
                let col: Vec<f32> = (0..k).map(|x| b.data()[x * n + j]).collect();
                let exact: f64 = row.iter().zip(&col).map(|(x, y)| f64::from(*x) * f64::from(*y)).sum();
                let err = (f64::from(got[i * n + j]) - exact).abs();
                prop_assert!(err <= matmul_bound(&row, &col));
            }
        }
    }
}
