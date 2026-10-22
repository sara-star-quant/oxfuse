//! Compares `oxfuse_runtime::math::exp` with CORE-MATH, which rounds correctly,
//! over every `f32` bit pattern, and prints the hash of each side.
//!
//!     cargo run --release --manifest-path scripts/exp-oracle/Cargo.toml
//!
//! The hash of "ours" is the number pinned in
//! `crates/runtime/src/cpu/vecmath.rs`; the hash of "correctly rounded" is
//! printed for the record. Inputs above 104 in magnitude are included.
#![allow(dead_code)]

#[path = "../../../crates/runtime/src/math.rs"]
mod math;

use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

fn splitmix(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9e37_79b9_7f4a_7c15);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

fn pair_hash(bits: u32, out: f32) -> u64 {
    let out = if out.is_nan() { 0x7fc0_0000 } else { out.to_bits() };
    splitmix((u64::from(bits) << 32) | u64::from(out))
}

fn ulps(a: f32, b: f32) -> u32 {
    let key = |x: f32| {
        let bits = i64::from(x.to_bits());
        if bits & 0x8000_0000 != 0 { -(bits & 0x7fff_ffff) } else { bits }
    };
    (key(a) - key(b)).unsigned_abs() as u32
}

fn main() {
    let step: u64 = std::env::args().nth(1).map_or(1, |s| s.parse().unwrap());
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get()) as u64;
    let (ours, exact, differ, worst, total) = (
        AtomicU64::new(0),
        AtomicU64::new(0),
        AtomicU64::new(0),
        AtomicU64::new(0),
        AtomicU64::new(0),
    );
    std::thread::scope(|scope| {
        for t in 0..threads {
            let (ours, exact, differ, worst, total) = (&ours, &exact, &differ, &worst, &total);
            scope.spawn(move || {
                let (mut o, mut e, mut d, mut w, mut c) = (0u64, 0u64, 0u64, 0u32, 0u64);
                let mut bits = t * step;
                while bits < 1 << 32 {
                    let x = f32::from_bits(bits as u32);
                    let (mine, right) = (math::exp(x), core_math::expf(x));
                    o = o.wrapping_add(pair_hash(bits as u32, mine));
                    e = e.wrapping_add(pair_hash(bits as u32, right));
                    c += 1;
                    if !(mine.is_nan() && right.is_nan()) && mine.to_bits() != right.to_bits() {
                        d += 1;
                        w = w.max(ulps(mine, right));
                    }
                    bits += threads * step;
                }
                ours.fetch_add(o, Relaxed);
                exact.fetch_add(e, Relaxed);
                differ.fetch_add(d, Relaxed);
                worst.fetch_max(u64::from(w), Relaxed);
                total.fetch_add(c, Relaxed);
            });
        }
    });
    println!("step {step}: {} inputs", total.load(Relaxed));
    println!("ours               {:#018x}", ours.load(Relaxed));
    println!("correctly rounded  {:#018x}", exact.load(Relaxed));
    println!(
        "{} inputs differ ({:.4}%), worst {} ulp",
        differ.load(Relaxed),
        100.0 * differ.load(Relaxed) as f64 / total.load(Relaxed) as f64,
        worst.load(Relaxed)
    );
}
