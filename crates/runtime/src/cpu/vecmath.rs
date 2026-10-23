//! `exp` and `sigmoid` on explicit SIMD vectors, giving exactly the bits of
//! the scalar versions in `crate::math` at every SIMD level. The steps below
//! are the scalar steps one for one; `mul_add_precise` is the fused
//! multiply-add with a single rounding on every level.

use crate::math::{LN2_HI, LN2_LO, LOG2E, MAGIC, Q, X_MAX, X_MIN};
use fearless_simd::{Level, Simd, dispatch, f32x16, i32x16, prelude::*};
use fearless_simd_macros::simd;

const LANES: usize = 16;

/// `2^n` for `n` in `[-126, 127]`.
#[inline(always)]
fn pow2<S: Simd>(simd: S, n: i32x16<S>) -> f32x16<S> {
    let biased = n + i32x16::splat(simd, 127);
    simd.shl_i32x16(biased, 23).bitcast::<f32x16<S>>()
}

#[inline(always)]
fn splat<S: Simd>(simd: S, v: f32) -> f32x16<S> {
    f32x16::splat(simd, v)
}

#[inline(always)]
pub(super) fn exp_vec<S: Simd>(simd: S, x: f32x16<S>) -> f32x16<S> {
    // Comparisons and selects, so that NaN passes through the clamp.
    let (x_max, x_min) = (splat(simd, X_MAX), splat(simd, X_MIN));
    let x = simd.select_f32x16(simd.simd_gt_f32x16(x, x_max), x_max, x);
    let x = simd.select_f32x16(simd.simd_lt_f32x16(x, x_min), x_min, x);
    let magic = splat(simd, MAGIC);
    let n_f = x.mul_add_precise(splat(simd, LOG2E), magic) - magic;
    let n = simd.cvt_i32_precise_f32x16(n_f);
    let neg_n = simd.neg_f32x16(n_f);
    let r = neg_n.mul_add_precise(splat(simd, LN2_HI), x);
    let r = neg_n.mul_add_precise(splat(simd, LN2_LO), r);
    let q = splat(simd, Q[5])
        .mul_add_precise(r, splat(simd, Q[4]))
        .mul_add_precise(r, splat(simd, Q[3]))
        .mul_add_precise(r, splat(simd, Q[2]))
        .mul_add_precise(r, splat(simd, Q[1]))
        .mul_add_precise(r, splat(simd, Q[0]));
    let tail = (r * r) * q;
    let one = splat(simd, 1.0);
    let hi = one + r;
    let lo = r - (hi - one);
    let e = hi + (lo + tail);
    let half = simd.shr_i32x16(n, 1);
    e * pow2(simd, half) * pow2(simd, n - half)
}

#[inline(always)]
fn sigmoid_vec<S: Simd>(simd: S, x: f32x16<S>) -> f32x16<S> {
    let one = splat(simd, 1.0);
    one / (one + exp_vec(simd, simd.neg_f32x16(x)))
}

/// Vectors of 16 from `src` into `dst`, and the last partial vector through
/// a zero padded copy. Written out once per function and not as one helper
/// taking the vector function as a parameter: on x86 that helper was not
/// compiled with the AVX2 features and ran about ten times slower.
#[simd]
fn sigmoid_into_simd<S: Simd>(simd: S, src: &[f32], dst: &mut [f32]) {
    let (chunks, rest) = src.as_chunks::<LANES>();
    let (out, d) = dst.as_chunks_mut::<LANES>();
    for (s, d) in chunks.iter().zip(out) {
        sigmoid_vec(simd, f32x16::from_slice(simd, s)).store_slice(d);
    }
    if !rest.is_empty() {
        let mut padded = [0.0f32; LANES];
        padded[..rest.len()].copy_from_slice(rest);
        let mut result = [0.0f32; LANES];
        sigmoid_vec(simd, f32x16::from_slice(simd, &padded)).store_slice(&mut result);
        d.copy_from_slice(&result[..rest.len()]);
    }
}

#[simd]
fn sigmoid_in_place_simd<S: Simd>(simd: S, buf: &mut [f32]) {
    let (chunks, rest) = buf.as_chunks_mut::<LANES>();
    for c in chunks {
        sigmoid_vec(simd, f32x16::from_slice(simd, c)).store_slice(c);
    }
    if !rest.is_empty() {
        let mut padded = [0.0f32; LANES];
        padded[..rest.len()].copy_from_slice(rest);
        let mut result = [0.0f32; LANES];
        sigmoid_vec(simd, f32x16::from_slice(simd, &padded)).store_slice(&mut result);
        rest.copy_from_slice(&result[..rest.len()]);
    }
}

#[cfg(test)]
#[simd]
fn exp_into_simd<S: Simd>(simd: S, src: &[f32], dst: &mut [f32]) {
    let (chunks, rest) = src.as_chunks::<LANES>();
    let (out, d) = dst.as_chunks_mut::<LANES>();
    for (s, d) in chunks.iter().zip(out) {
        exp_vec(simd, f32x16::from_slice(simd, s)).store_slice(d);
    }
    if !rest.is_empty() {
        let mut padded = [0.0f32; LANES];
        padded[..rest.len()].copy_from_slice(rest);
        let mut result = [0.0f32; LANES];
        exp_vec(simd, f32x16::from_slice(simd, &padded)).store_slice(&mut result);
        d.copy_from_slice(&result[..rest.len()]);
    }
}

/// `dst[i] = sigmoid(src[i])`, on the SIMD level `level`.
pub(super) fn sigmoid_into(level: Level, src: &[f32], dst: &mut [f32]) {
    dispatch!(level, simd => sigmoid_into_simd(simd, src, dst));
}

pub(super) fn sigmoid_in_place(level: Level, buf: &mut [f32]) {
    dispatch!(level, simd => sigmoid_in_place_simd(simd, buf));
}

#[cfg(test)]
pub(super) fn exp_into(level: Level, src: &[f32], dst: &mut [f32]) {
    dispatch!(level, simd => exp_into_simd(simd, src, dst));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math;

    fn same_value(a: f32, b: f32) -> bool {
        (a.is_nan() && b.is_nan()) || a.to_bits() == b.to_bits()
    }

    /// Bit patterns of every kind: random, infinities, NaN, zeros, subnormals,
    /// the clamp edges and values around them.
    fn inputs(len: usize, seed: u64) -> Vec<f32> {
        let special = [
            f32::NAN,
            f32::INFINITY,
            f32::NEG_INFINITY,
            0.0,
            -0.0,
            f32::from_bits(1),
            -f32::from_bits(1),
            f32::MIN_POSITIVE,
            88.722_84,
            -103.97,
            89.0,
            -104.0,
            1.0e30,
            -1.0e30,
        ];
        let mut state = seed;
        (0..len)
            .map(|i| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                match i % 4 {
                    0 => special[(state >> 33) as usize % special.len()],
                    1 => f32::from_bits((state >> 32) as u32),
                    _ => ((state >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 220.0,
                }
            })
            .collect()
    }

    fn levels() -> [Level; 2] {
        [Level::baseline(), Level::new()]
    }

    #[test]
    fn vector_exp_equals_scalar_exp_for_every_length_and_level() {
        for level in levels() {
            for len in 0..=40 {
                let src = inputs(len, 100 + len as u64);
                let mut dst = vec![0.0f32; len];
                exp_into(level, &src, &mut dst);
                for (i, (&x, &y)) in src.iter().zip(&dst).enumerate() {
                    assert!(
                        same_value(y, math::exp(x)),
                        "{level:?} len={len} i={i} x={x:e} vector={y:e} scalar={:e}",
                        math::exp(x)
                    );
                }
            }
        }
    }

    #[test]
    fn vector_sigmoid_equals_scalar_sigmoid_for_every_length_and_level() {
        for level in levels() {
            for len in 0..=40 {
                let src = inputs(len, 200 + len as u64);
                let mut dst = vec![0.0f32; len];
                sigmoid_into(level, &src, &mut dst);
                let mut in_place = src.clone();
                sigmoid_in_place(level, &mut in_place);
                for (i, &x) in src.iter().enumerate() {
                    let want = math::sigmoid(x);
                    assert!(same_value(dst[i], want), "{level:?} len={len} x={x:e}");
                    assert!(same_value(in_place[i], want), "in place {level:?} x={x:e}");
                }
            }
        }
    }

    #[test]
    fn a_nan_in_one_lane_does_not_touch_its_neighbors() {
        let mut src = vec![0.5f32; 33];
        src[7] = f32::NAN;
        src[16] = f32::INFINITY;
        let mut dst = vec![0.0f32; 33];
        sigmoid_into(Level::new(), &src, &mut dst);
        for (i, &y) in dst.iter().enumerate() {
            assert!(same_value(y, math::sigmoid(src[i])), "lane {i}");
        }
    }

    fn splitmix(mut z: u64) -> u64 {
        z = z.wrapping_add(0x9e37_79b9_7f4a_7c15);
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// Order independent hash of (input bits, output bits) pairs, all NaNs
    /// counted as one value.
    fn pair_hash(bits: u32, out: f32) -> u64 {
        let out = if out.is_nan() {
            0x7fc0_0000
        } else {
            out.to_bits()
        };
        splitmix((u64::from(bits) << 32) | u64::from(out))
    }

    /// Hash of `f` over every input whose bit pattern is `start`, `start +
    /// step`, ... below 2^32, split over all cores.
    fn sweep(level: Level, start: u32, step: usize, f: fn(Level, &[f32], &mut [f32])) -> u64 {
        const BLOCK: usize = 1 << 16;
        let blocks = (1u64 << 32) / (BLOCK * step) as u64 + 1;
        let next = std::sync::atomic::AtomicU64::new(0);
        let threads = std::thread::available_parallelism().map_or(4, |n| n.get());
        let total = std::sync::atomic::AtomicU64::new(0);
        std::thread::scope(|scope| {
            for _ in 0..threads {
                scope.spawn(|| {
                    let mut src = vec![0.0f32; BLOCK];
                    let mut dst = vec![0.0f32; BLOCK];
                    let mut sum = 0u64;
                    loop {
                        let b = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        if b >= blocks {
                            break;
                        }
                        let first = u64::from(start) + b * (BLOCK * step) as u64;
                        let mut used = 0;
                        for slot in src.iter_mut() {
                            let bits = first + (used * step) as u64;
                            if bits >= 1 << 32 {
                                break;
                            }
                            *slot = f32::from_bits(bits as u32);
                            used += 1;
                        }
                        f(level, &src[..used], &mut dst[..used]);
                        for (j, &out) in dst[..used].iter().enumerate() {
                            let bits = (first + (j * step) as u64) as u32;
                            sum = sum.wrapping_add(pair_hash(bits, out));
                        }
                    }
                    total.fetch_add(sum, std::sync::atomic::Ordering::Relaxed);
                });
            }
        });
        total.load(std::sync::atomic::Ordering::Relaxed)
    }

    // Hashes of every (input, output) pair, so a changed constant or step in
    // `math.rs` shows up here. `scripts/exp-oracle` computes the same exp hash
    // from the scalar code and compares the results with correctly rounded ones.
    const EXP_STRIDE: u64 = 0xfe53_1bc3_686d_7cb2;
    const SIGMOID_STRIDE: u64 = 0xabb8_69ef_ae1c_d141;
    const EXP_FULL: u64 = 0x3600_d0e8_170e_cc70;
    const SIGMOID_FULL: u64 = 0xc205_7816_8f66_7b31;

    #[test]
    fn exp_and_sigmoid_match_the_pinned_hashes_on_every_4099th_input() {
        assert_eq!(sweep(Level::new(), 0, 4099, exp_into), EXP_STRIDE);
        assert_eq!(sweep(Level::new(), 0, 4099, sigmoid_into), SIGMOID_STRIDE);
    }

    /// The baseline level is SSE2 on x86, which has no fused multiply-add: the
    /// vector code then uses `fearless_simd`'s software one. It must give the
    /// same pinned bits as the best level, or a dependency update could change
    /// results on machines without FMA and no other test would notice.
    #[test]
    fn the_baseline_level_matches_the_pinned_hashes_too() {
        assert_eq!(sweep(Level::baseline(), 0, 4099, exp_into), EXP_STRIDE);
        assert_eq!(
            sweep(Level::baseline(), 0, 4099, sigmoid_into),
            SIGMOID_STRIDE
        );
    }

    #[test]
    #[ignore = "all 2^32 inputs: cargo test --release -p oxfuse-runtime --lib -- --ignored"]
    fn exp_and_sigmoid_match_the_pinned_hashes_on_every_input() {
        assert_eq!(sweep(Level::new(), 0, 1, exp_into), EXP_FULL);
        assert_eq!(sweep(Level::new(), 0, 1, sigmoid_into), SIGMOID_FULL);
    }
}
