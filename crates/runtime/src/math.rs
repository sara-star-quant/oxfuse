//! `exp` and `sigmoid` with the same bits on every platform.
//!
//! The standard library's `exp` comes from the platform's math library, so x86
//! and ARM, or two versions of one libc, can differ in the last bit. This
//! module uses only IEEE multiplications, additions and fused multiply-adds,
//! integer operations and a clamp, all of which are exact or rounded the same
//! way everywhere, so the result depends on nothing but the input.
//!
//! The algorithm is the specification. It is within 1 ulp of the correctly
//! rounded result, and equal to it for 99.7 percent of the inputs in
//! `[-104, 104]` (measured over every one of them against CORE-MATH, see
//! `scripts/exp-oracle`). The output bits of every input are pinned by hash in
//! the tests, so changing a constant here is a visible, deliberate change.
//!
//! This file imports nothing from the crate, so `scripts/exp-oracle` can
//! include it as it is.

pub(crate) const LOG2E: f32 = std::f32::consts::LOG2_E;
/// `ln 2` split into a leading `f32` and the rest, so `n * LN2_HI` loses
/// nothing in the fused step that subtracts it.
pub(crate) const LN2_HI: f32 = std::f32::consts::LN_2;
pub(crate) const LN2_LO: f32 = -1.904_654_3e-9;
/// `1.5 * 2^23`: adding it rounds to the nearest integer in the low mantissa
/// bits, and subtracting it gives that integer back.
pub(crate) const MAGIC: f32 = 12_582_912.0;
/// Inputs beyond these give infinity and zero already; clamping keeps the
/// power-of-two scaling below in range.
pub(crate) const X_MAX: f32 = 89.0;
pub(crate) const X_MIN: f32 = -104.0;
/// `q(r) = (e^r - 1 - r) / r^2` on `[-ln2/2, ln2/2]`, degree 5, least squares
/// fit in `f64` and rounded to `f32`. Lowest order first.
pub(crate) const Q: [f32; 6] = [
    0.5,
    0.166_666_67,
    0.041_666_463,
    0.008_333_298,
    0.001_393_368,
    0.000_198_993_18,
];

/// `2^n` for `n` in `[-126, 127]`.
#[inline(always)]
fn pow2(n: i32) -> f32 {
    f32::from_bits(((n + 127) as u32) << 23)
}

/// `e^x`. NaN gives NaN, `+inf` gives `+inf`, `-inf` gives 0, and a result
/// below the smallest subnormal rounds to 0.
#[inline(always)]
pub(crate) fn exp(x: f32) -> f32 {
    // `clamp` passes NaN through.
    let x = x.clamp(X_MIN, X_MAX);
    let n_f = x.mul_add(LOG2E, MAGIC) - MAGIC;
    let n = n_f as i32;
    let r = (-n_f).mul_add(LN2_HI, x);
    let r = (-n_f).mul_add(LN2_LO, r);
    let q = Q[5]
        .mul_add(r, Q[4])
        .mul_add(r, Q[3])
        .mul_add(r, Q[2])
        .mul_add(r, Q[1])
        .mul_add(r, Q[0]);
    let tail = (r * r) * q;
    // 1 + r kept as a sum and its rounding error, so the single rounding that
    // matters is the last addition.
    let hi = 1.0 + r;
    let lo = r - (hi - 1.0);
    let e = hi + (lo + tail);
    // Two scalings, so a result in the subnormal range is rounded once.
    let half = n >> 1;
    e * pow2(half) * pow2(n - half)
}

/// `1 / (1 + e^-x)`, the definition of the `Sigmoid` op: the same bits on every
/// platform. Large negative `x` makes `exp` overflow to infinity and the
/// result 0, without NaN.
#[inline(always)]
pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + exp(-x))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Distance in representable values between two finite `f32`.
    fn ulps(a: f32, b: f32) -> u32 {
        let key = |x: f32| {
            let bits = x.to_bits() as i64;
            if bits & 0x8000_0000 != 0 {
                -(bits & 0x7fff_ffff)
            } else {
                bits
            }
        };
        (key(a) - key(b)).unsigned_abs() as u32
    }

    /// The correctly rounded result, up to the rare double rounding of `f64::exp`.
    fn nearly_exact(x: f32) -> f32 {
        f64::from(x).exp() as f32
    }

    #[test]
    fn exp_of_the_special_values() {
        assert_eq!(exp(0.0), 1.0);
        assert_eq!(exp(-0.0), 1.0);
        assert_eq!(exp(f32::INFINITY), f32::INFINITY);
        assert_eq!(exp(f32::NEG_INFINITY).to_bits(), 0);
        assert!(exp(f32::NAN).is_nan());
        assert!(exp(-f32::NAN).is_nan());
    }

    #[test]
    fn exp_of_one_is_e() {
        assert_eq!(exp(1.0), std::f32::consts::E);
        assert_eq!(exp(-1.0), 1.0 / std::f32::consts::E);
    }

    #[test]
    fn exp_overflows_exactly_where_the_result_leaves_f32() {
        // A result rounds to infinity once it reaches f32::MAX plus half an ulp.
        let threshold = f64::from(f32::MAX) * (1.0 + 2f64.powi(-24));
        let edge = 88.722_84_f32.to_bits();
        for bits in edge - 40..edge + 40 {
            let x = f32::from_bits(bits);
            let finite = f64::from(x).exp() < threshold;
            assert_eq!(exp(x).is_finite(), finite, "x={x}");
            if finite {
                assert!(ulps(exp(x), nearly_exact(x)) <= 1, "x={x}");
            }
        }
        assert_eq!(exp(100.0), f32::INFINITY);
        assert_eq!(exp(1.0e30), f32::INFINITY);
    }

    #[test]
    fn exp_underflows_through_the_subnormals_to_zero() {
        let smallest = f32::from_bits(1);
        // ln(2^-149) = -103.2789, ln(2^-150) = -103.9720.
        assert_eq!(exp(-103.0), nearly_exact(-103.0));
        assert_eq!(exp(-103.27), smallest);
        assert_eq!(exp(-103.99).to_bits(), 0);
        assert_eq!(exp(-104.0).to_bits(), 0);
        assert_eq!(exp(-1.0e30).to_bits(), 0);
        for k in 1..=40 {
            let x = -87.0 - 0.4 * k as f32;
            assert!(ulps(exp(x), nearly_exact(x)) <= 1, "x={x}");
        }
    }

    #[test]
    fn exp_is_within_one_ulp_of_the_exact_result() {
        let mut state = 7u64;
        for _ in 0..200_000 {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let x = f32::from_bits((state >> 32) as u32);
            if !x.is_finite() || !(-103.0..=88.0).contains(&x) {
                continue;
            }
            assert!(ulps(exp(x), nearly_exact(x)) <= 1, "x={x}");
        }
    }

    #[test]
    fn exp_is_exact_for_tiny_arguments() {
        // e^x = 1 + x to within half an ulp for |x| below 2^-25.
        for e in 26..60 {
            let x = 2f32.powi(-e);
            assert_eq!(exp(x), 1.0, "x=2^-{e}");
            assert_eq!(exp(-x), 1.0, "x=-2^-{e}");
        }
    }

    #[test]
    fn sigmoid_of_the_special_values_and_the_edges() {
        assert_eq!(sigmoid(0.0), 0.5);
        assert_eq!(sigmoid(f32::INFINITY), 1.0);
        assert_eq!(sigmoid(f32::NEG_INFINITY), 0.0);
        assert!(sigmoid(f32::NAN).is_nan());
        assert_eq!(sigmoid(20.0), 1.0);
        // exp(-x) overflows to infinity here, so the result is 0 although
        // the true value is subnormal (docs/semantics.md).
        assert_eq!(sigmoid(-88.8).to_bits(), 0);
        assert_eq!(sigmoid(-100.0).to_bits(), 0);
        assert_eq!(sigmoid(-1.0e30).to_bits(), 0);
    }

    #[test]
    fn sigmoid_matches_the_f64_formula_within_two_ulp() {
        for i in -2000..=2000 {
            let x = i as f32 * 0.01;
            let want = (1.0 / (1.0 + (-f64::from(x)).exp())) as f32;
            assert!(ulps(sigmoid(x), want) <= 2, "x={x}");
        }
    }
}
