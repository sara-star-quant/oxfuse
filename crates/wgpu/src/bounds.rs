//! How far a device result may be from the reference, written before any
//! device test uses it. A test never loosens these to pass; a bound set from a
//! measurement says so next to its definition.

/// Position of `x` on a line that is monotonic in value, with both zeros at 0.
fn ordered(x: f32) -> i64 {
    let magnitude = i64::from(x.to_bits() & 0x7fff_ffff);
    if x.is_sign_negative() {
        -magnitude
    } else {
        magnitude
    }
}

/// Number of representable `f32` values between `a` and `b`; `None` if either is NaN.
pub fn ulp_distance(a: f32, b: f32) -> Option<u32> {
    if a.is_nan() || b.is_nan() {
        return None;
    }
    Some((ordered(a) - ordered(b)).unsigned_abs() as u32)
}

/// The documented `MatMul` bound against a float64 result: `K * eps * sum |a_i * b_i|`
/// with `eps = f32::EPSILON`, the same bound the runtime tests use (docs/semantics.md).
pub fn matmul_bound(a_row: &[f32], b_col: &[f32]) -> f64 {
    assert_eq!(
        a_row.len(),
        b_col.len(),
        "operands must have the same length"
    );
    let magnitude: f64 = a_row
        .iter()
        .zip(b_col)
        .map(|(a, b)| (f64::from(*a) * f64::from(*b)).abs())
        .sum();
    a_row.len() as f64 * f64::from(f32::EPSILON) * magnitude
}

/// The `MatMul` bound for a device that flushes subnormals to zero: the plain
/// bound plus one smallest-normal number per product and per partial sum (2k for
/// k terms), which is the most a flushed term can lose. The CPU backends keep the plain bound.
pub fn gpu_matmul_bound(a_row: &[f32], b_col: &[f32]) -> f64 {
    matmul_bound(a_row, b_col) + 2.0 * a_row.len() as f64 * f64::from(f32::MIN_POSITIVE)
}

/// Adapters whose `MatMul` was measured bit-identical to the reference on zero
/// and normal inputs. The reference fuses each multiply-add, and these devices
/// do the same, in the same ascending `k` order from one accumulator. Another
/// adapter may or may not fuse: `oxfuse gpu-selftest` reports its distance and
/// the plain bound of `gpu_matmul_bound` is all that is promised there.
const MATMUL_EXACT_ADAPTERS: [&str; 2] = ["Apple M1 Pro", "Tesla T4"];

/// Whether `MatMul` on `adapter` is held to bit equality with the reference.
pub fn matmul_exact_on(adapter: &str) -> bool {
    MATMUL_EXACT_ADAPTERS.contains(&adapter)
}

/// Largest ulp distance from the reference that Sigmoid may have on normal
/// results: 65, measured over all finite `f32` inputs by `sweep_exhaustive` on an
/// Apple M1 Pro (Metal) on 2026-10-20, worst at x = -86.64 where the device `exp`
/// nears overflow. This is a figure for that adapter, not a guarantee for
/// others: the sweeps enforce it where they run, and `oxfuse gpu-selftest`
/// reports the figure for any other adapter.
pub const SIGMOID_ULP_BOUND: u32 = 65;

/// The adapter `SIGMOID_ULP_BOUND` was measured on.
const RECORDED_ADAPTER: &str = "Apple M1 Pro";

/// The ulp bound a test may enforce on `adapter`. The 65 ulp figure is a
/// measurement of the recorded adapter, not a promise about others, so any other
/// adapter is only held to the flush rule of `sigmoid_matches` and its measured
/// figure is reported (`gpu-selftest`, the sweeps) instead of asserted.
pub fn sigmoid_ulp_bound_for(adapter: &str) -> u32 {
    if adapter == RECORDED_ADAPTER {
        SIGMOID_ULP_BOUND
    } else {
        u32::MAX
    }
}

/// Whether a device Sigmoid result is acceptable against the reference. Both must
/// be finite. Below the smallest normal number the device may flush to zero, so
/// only an absolute difference of that size is required there; everywhere else
/// the ulp distance must be within `max_ulp`.
pub fn sigmoid_matches(got: f32, want: f32, max_ulp: u32) -> bool {
    if !got.is_finite() || !want.is_finite() {
        return false;
    }
    if want.abs() < f32::MIN_POSITIVE {
        return (got - want).abs() <= f32::MIN_POSITIVE;
    }
    ulp_distance(got, want).is_some_and(|d| d <= max_ulp)
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn adjacent_floats_are_one_ulp_apart() {
        assert_eq!(ulp_distance(1.0, 1.0f32.next_up()), Some(1));
        assert_eq!(ulp_distance(1.0, 1.0), Some(0));
        assert_eq!(ulp_distance(-1.0, (-1.0f32).next_up()), Some(1));
    }

    #[test]
    fn the_two_zeros_are_equal_and_the_ulp_scale_crosses_zero() {
        assert_eq!(ulp_distance(0.0, -0.0), Some(0));
        let tiny = f32::from_bits(1);
        assert_eq!(ulp_distance(tiny, -tiny), Some(2));
        assert_eq!(ulp_distance(0.0, tiny), Some(1));
    }

    #[test]
    fn nan_has_no_distance() {
        assert_eq!(ulp_distance(f32::NAN, 1.0), None);
        assert_eq!(ulp_distance(1.0, f32::NAN), None);
    }

    #[test]
    fn infinity_is_one_ulp_past_the_largest_finite() {
        assert_eq!(ulp_distance(f32::MAX, f32::INFINITY), Some(1));
        assert_eq!(
            ulp_distance(f32::NEG_INFINITY, f32::INFINITY),
            Some(4_278_190_080)
        );
    }

    proptest! {
        #[test]
        fn distance_is_symmetric_and_adds_up(a in -1e30f32..1e30, b in -1e30f32..1e30, c in -1e30f32..1e30) {
            let mut v = [a, b, c];
            v.sort_by(f32::total_cmp);
            let [lo, mid, hi] = v;
            prop_assert_eq!(ulp_distance(lo, hi), ulp_distance(hi, lo));
            prop_assert_eq!(
                ulp_distance(lo, hi).unwrap(),
                ulp_distance(lo, mid).unwrap() + ulp_distance(mid, hi).unwrap()
            );
        }
    }

    #[test]
    fn matmul_bound_is_k_eps_times_the_sum_of_absolute_products() {
        let a = [1.0f32, -2.0, 3.0];
        let b = [4.0f32, 5.0, -6.0];
        let expected = 3.0 * f64::from(f32::EPSILON) * (4.0 + 10.0 + 18.0);
        assert!((matmul_bound(&a, &b) - expected).abs() < 1e-18);
    }

    #[test]
    fn matmul_bound_of_nothing_is_zero() {
        assert_eq!(matmul_bound(&[], &[]), 0.0);
    }

    #[test]
    #[should_panic(expected = "same length")]
    fn matmul_bound_rejects_mismatched_lengths() {
        matmul_bound(&[1.0], &[1.0, 2.0]);
    }

    #[test]
    fn sigmoid_accepts_results_within_the_ulp_bound() {
        let want = 0.3f32;
        assert!(sigmoid_matches(want, want, 0));
        assert!(sigmoid_matches(want.next_up(), want, 1));
        assert!(sigmoid_matches(want.next_down(), want, 1));
        assert!(!sigmoid_matches(want.next_up().next_up(), want, 1));
    }

    #[test]
    fn sigmoid_allows_a_flush_to_zero_only_below_the_smallest_normal() {
        let subnormal = f32::MIN_POSITIVE / 4.0;
        assert!(
            sigmoid_matches(0.0, subnormal, 1),
            "device flushes a subnormal result"
        );
        assert!(sigmoid_matches(subnormal, 0.0, 1));
        assert!(
            !sigmoid_matches(1e-30, subnormal, 1000),
            "too far from a subnormal"
        );
        assert!(
            !sigmoid_matches(0.0, 1e-30, 1000),
            "a normal result must not flush"
        );
    }

    #[test]
    fn sigmoid_rejects_nan_and_infinity() {
        assert!(!sigmoid_matches(f32::NAN, 0.5, 1000));
        assert!(!sigmoid_matches(0.5, f32::NAN, 1000));
        assert!(!sigmoid_matches(f32::INFINITY, 1.0, 1000));
    }

    #[test]
    fn matmul_bit_equality_is_enforced_only_on_adapters_that_were_measured() {
        assert!(matmul_exact_on("Apple M1 Pro"));
        assert!(matmul_exact_on("Tesla T4"));
        assert!(!matmul_exact_on("llvmpipe (LLVM 19.1.1, 256 bits)"));
        assert!(!matmul_exact_on(""));
    }

    #[test]
    fn the_sigmoid_bound_is_enforced_only_on_the_adapter_it_was_measured_on() {
        assert_eq!(sigmoid_ulp_bound_for("Apple M1 Pro"), SIGMOID_ULP_BOUND);
        assert_eq!(
            sigmoid_ulp_bound_for("llvmpipe (LLVM 19.1.1, 256 bits)"),
            u32::MAX
        );
        assert_eq!(sigmoid_ulp_bound_for(""), u32::MAX);
    }

    #[test]
    fn an_infinite_result_is_never_acceptable_even_with_no_ulp_limit() {
        assert!(!sigmoid_matches(0.5, f32::INFINITY, u32::MAX));
        assert!(!sigmoid_matches(f32::INFINITY, 1.0, u32::MAX));
        assert!(!sigmoid_matches(f32::NAN, f32::NAN, u32::MAX));
    }

    #[test]
    fn the_flush_rule_applies_strictly_below_the_smallest_normal() {
        // At exactly the smallest normal number the ulp rule applies, not the flush rule.
        assert!(!sigmoid_matches(
            f32::MIN_POSITIVE * 1.5,
            f32::MIN_POSITIVE,
            1
        ));
        // Below it, an absolute difference of up to one smallest-normal is fine.
        assert!(sigmoid_matches(
            f32::MIN_POSITIVE,
            f32::MIN_POSITIVE * 0.5,
            1
        ));
        // A reference of exactly the smallest normal number is held to the ulp rule.
        assert!(!sigmoid_matches(
            f32::MIN_POSITIVE * 0.5,
            f32::MIN_POSITIVE,
            1
        ));
        assert!(!sigmoid_matches(
            f32::MIN_POSITIVE * 2.0,
            f32::MIN_POSITIVE * 0.5,
            1
        ));
    }

    #[test]
    fn the_gpu_matmul_bound_adds_two_smallest_normals_per_term() {
        // With a zero row the plain bound is exactly zero, so the extra term shows on its own.
        let min = f64::from(f32::MIN_POSITIVE);
        assert_eq!(gpu_matmul_bound(&[0.0; 3], &[4.0, 5.0, -6.0]), 6.0 * min);
        assert_eq!(gpu_matmul_bound(&[0.0], &[1.0]), 2.0 * min);
        // And it never replaces the plain bound.
        let a = [1.0f32, -2.0, 3.0];
        let b = [4.0f32, 5.0, -6.0];
        assert_eq!(gpu_matmul_bound(&a, &b), matmul_bound(&a, &b) + 6.0 * min);
    }

    #[test]
    fn products_that_flush_to_zero_stay_inside_the_gpu_bound_but_not_the_plain_one() {
        // The reviewer's case: normal inputs and a normal true result (2^-126), but
        // each product is 2^-127, a subnormal, which the device flushes to zero.
        let a = [2f32.powi(-63); 2];
        let b = [2f32.powi(-64); 2];
        let exact = 2.0 * f64::from(a[0]) * f64::from(b[0]);
        assert_eq!(exact, f64::from(f32::MIN_POSITIVE));
        let flushed = 0.0f64;
        let error = (flushed - exact).abs();
        assert!(
            error > matmul_bound(&a, &b),
            "the plain bound cannot cover a flush"
        );
        assert!(error <= gpu_matmul_bound(&a, &b));
    }
}
