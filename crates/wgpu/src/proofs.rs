//! Bounded proofs (Kani) that the index arithmetic the shaders use stays in
//! range. They cover every value up to the stated bounds, not every integer:
//! tensors up to `MAX_TENSOR_ELEMENTS` (2^30 elements), the default kernel
//! configuration (256 threads per elementwise workgroup, 16 by 16 tiles) and the
//! 65535 dispatch limit. Other configurations are covered by the model tests,
//! not by proofs. The formulas are copied from the shader text,
//! and `the_proved_formulas_are_the_shaders_formulas` fails if a shader changes
//! one without this file. The proofs are about the arithmetic; the model tests
//! and the device tests are about the kernels.

/// Index expressions the proofs below restate, as they appear in the WGSL for
/// the default configuration (16 by 16 threads, 4 by 4 outputs each, 64 square block).
#[cfg(test)]
const MATMUL_FORMULAS: [&str; 12] = [
    "let gr = row0 + idx / 16u;",
    "let gk = k0 + idx % 16u;",
    "if (gr < p.m && gk < p.k)",
    "a[gr * p.k + gk]",
    "let gk2 = k0 + idx / 64u;",
    "let gc = col0 + idx % 64u;",
    "if (gk2 < p.k && gc < p.n)",
    "b[gk2 * p.n + gc]",
    "if (rr < p.m && cc < p.n)",
    "dst[rr * p.n + cc]",
    "let rr = row0 + lid.y * 4u + 0u;",
    "let kmax = min(16u, p.k - k0);",
];
#[cfg(test)]
const TRANSPOSE_FORMULAS: [&str; 4] = [
    "if (r < p.rows && c < p.cols)",
    "if (out_row < p.cols && out_col < p.rows)",
    "src[r * p.cols + c]",
    "dst[out_row * p.rows + out_col]",
];
#[cfg(test)]
const ELEMENTWISE_FORMULA: &str = "let i = (wid.y * p.x_count + wid.x) * 256u + lid.x;";

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::KernelConfig;
    use crate::shaders;

    #[test]
    fn the_proved_formulas_are_the_shaders_formulas() {
        let cfg = KernelConfig::DEFAULT;
        let matmul = shaders::matmul(&cfg);
        for f in MATMUL_FORMULAS {
            assert!(matmul.contains(f), "matmul shader lost `{f}`");
        }
        let transpose = shaders::transpose2d(&cfg);
        for f in TRANSPOSE_FORMULAS {
            assert!(transpose.contains(f), "transpose shader lost `{f}`");
        }
        let relu = shaders::unary(shaders::Unary::Relu, &cfg);
        assert!(
            relu.contains(ELEMENTWISE_FORMULA),
            "elementwise shader lost `{ELEMENTWISE_FORMULA}`"
        );
    }
}

#[cfg(kani)]
mod harnesses {
    use crate::config::{MAX_TENSOR_ELEMENTS, element_index, grid_fold};

    /// Largest matrix side the index lemmas below range over. The lemma is the
    /// identity (r * cols + c) < rows * cols for r < rows, c < cols, whose proof
    /// is nonlinear and only practical at this width. Overflow of the shader's
    /// `u32` arithmetic does not need a wider proof: an in-range index is below
    /// the element count, and `check_tensor` caps that at 2^30 < 2^32.
    const MAX_DIM: u32 = 1 << 8;

    /// `u32` arithmetic exactly as a shader does it: `None` on overflow.
    fn mul_add(a: u64, b: u64, c: u64) -> Option<u32> {
        let a = u32::try_from(a).ok()?;
        let b = u32::try_from(b).ok()?;
        let c = u32::try_from(c).ok()?;
        a.checked_mul(b)?.checked_add(c)
    }

    /// The fold with the production parameters (256 threads, 65535 workgroups per
    /// dimension) and a symbolic element count up to the tensor cap.
    #[kani::proof]
    #[kani::solver(kissat)]
    fn fold_covers_the_tensor_within_the_dispatch_limit() {
        let elements: u64 = kani::any();
        kani::assume((1..=MAX_TENSOR_ELEMENTS).contains(&elements));
        let (x, y) = grid_fold(elements, 256, 65535).expect("2^30 elements always fit");
        assert!(x >= 1 && y >= 1 && x <= 65535 && y <= 65535);
        assert!(u64::from(x) * u64::from(y) * 256 >= elements);
        // The padding is less than one row of workgroups.
        assert!(u64::from(x) * u64::from(y) * 256 - elements < u64::from(x) * 256);
    }

    /// Every thread the fold launches computes its linear index in `u32`
    /// without overflow, and the `u32` value equals the exact one.
    #[kani::proof]
    #[kani::solver(kissat)]
    fn the_largest_element_index_fits_u32() {
        let elements: u64 = kani::any();
        kani::assume((1..=MAX_TENSOR_ELEMENTS).contains(&elements));
        let (x, y) = grid_fold(elements, 256, 65535).expect("2^30 elements always fit");
        let (gx, gy, local): (u32, u32, u32) = (kani::any(), kani::any(), kani::any());
        kani::assume(gx < x && gy < y && local < 256);
        // The shader computes (gy * x + gx) * wg + local in u32.
        let linear = mul_add(u64::from(gy), u64::from(x), u64::from(gx));
        let shader = linear.and_then(|l| mul_add(u64::from(l), 256, u64::from(local)));
        assert_eq!(
            shader.map(u64::from),
            Some(element_index(gx, gy, x, 256, local))
        );
    }

    /// The default configuration: 16 by 16 threads, 4 by 4 outputs per thread,
    /// so a workgroup covers a 64 by 64 block of the result.
    const T: u32 = 16;
    const R: u32 = 4;
    const BLOCK: u32 = T * R;

    /// Each of the matmul accesses is its own lemma over two dimensions.
    /// `tile_row` and `tile_col` range over every block of the grid. (That a folded
    /// group id maps to exactly the blocks of the grid is the fold's own property,
    /// tested exhaustively on small shapes and at the dispatch limits; division
    /// by a symbolic divisor is what the solver cannot do in useful time.)
    ///
    /// `a[gr * p.k + gk]`, guarded by `gr < m && gk < k`, where `idx` is one of
    /// the `BLOCK * T` cells of the A tile: `gr = row0 + idx / T`, `gk = k0 + idx % T`.
    #[kani::proof]
    #[kani::solver(kissat)]
    fn matmul_a_load_stays_in_range() {
        let (m, k): (u32, u32) = (kani::any(), kani::any());
        kani::assume((1..=MAX_DIM).contains(&m) && (1..=MAX_DIM).contains(&k));
        let (tile_row, idx, k0): (u32, u32, u32) = (kani::any(), kani::any(), kani::any());
        kani::assume(tile_row < m.div_ceil(BLOCK) && idx < BLOCK * T && k0 < k && k0 % T == 0);
        let gr = u64::from(tile_row) * u64::from(BLOCK) + u64::from(idx / T);
        let gk = u64::from(k0) + u64::from(idx % T);
        if gr < u64::from(m) && gk < u64::from(k) {
            let index = mul_add(gr, u64::from(k), gk).expect("the index fits u32");
            assert!(u64::from(index) < u64::from(m) * u64::from(k));
        }
        // The k loop's `p.k - k0` never underflows because k0 < k.
        assert!(k - k0 >= 1);
    }

    /// `b[gk2 * p.n + gc]`, guarded by `gk2 < k && gc < n`: `gk2 = k0 + idx / BLOCK`,
    /// `gc = col0 + idx % BLOCK`.
    #[kani::proof]
    #[kani::solver(kissat)]
    fn matmul_b_load_stays_in_range() {
        let (k, n): (u32, u32) = (kani::any(), kani::any());
        kani::assume((1..=MAX_DIM).contains(&k) && (1..=MAX_DIM).contains(&n));
        let (tile_col, idx, k0): (u32, u32, u32) = (kani::any(), kani::any(), kani::any());
        kani::assume(tile_col < n.div_ceil(BLOCK) && idx < BLOCK * T && k0 < k && k0 % T == 0);
        let gk2 = u64::from(k0) + u64::from(idx / BLOCK);
        let gc = u64::from(tile_col) * u64::from(BLOCK) + u64::from(idx % BLOCK);
        if gk2 < u64::from(k) && gc < u64::from(n) {
            let index = mul_add(gk2, u64::from(n), gc).expect("the index fits u32");
            assert!(u64::from(index) < u64::from(k) * u64::from(n));
        }
    }

    /// `dst[rr * p.n + cc]`, guarded by `rr < m && cc < n`, with
    /// `rr = row0 + ly * R + i` and `cc = col0 + lx * R + j`.
    #[kani::proof]
    #[kani::solver(kissat)]
    fn matmul_store_stays_in_range() {
        let (m, n): (u32, u32) = (kani::any(), kani::any());
        kani::assume((1..=MAX_DIM).contains(&m) && (1..=MAX_DIM).contains(&n));
        let (tile_row, tile_col, ly, lx, i, j): (u32, u32, u32, u32, u32, u32) = (
            kani::any(),
            kani::any(),
            kani::any(),
            kani::any(),
            kani::any(),
            kani::any(),
        );
        kani::assume(tile_row < m.div_ceil(BLOCK) && tile_col < n.div_ceil(BLOCK));
        kani::assume(ly < T && lx < T && i < R && j < R);
        let rr = u64::from(tile_row) * u64::from(BLOCK) + u64::from(ly * R + i);
        let cc = u64::from(tile_col) * u64::from(BLOCK) + u64::from(lx * R + j);
        if rr < u64::from(m) && cc < u64::from(n) {
            let index = mul_add(rr, u64::from(n), cc).expect("the index fits u32");
            assert!(u64::from(index) < u64::from(m) * u64::from(n));
        }
    }

    /// The workgroup-memory reads of the inner loop stay inside the two tiles of
    /// `BLOCK * T` cells: `a_tile[(ly * R + i) * T + kk]`, `b_tile[kk * BLOCK + lx * R + j]`.
    #[kani::proof]
    fn matmul_tile_reads_stay_in_range() {
        let (ly, lx, i, j, kk): (u32, u32, u32, u32, u32) = (
            kani::any(),
            kani::any(),
            kani::any(),
            kani::any(),
            kani::any(),
        );
        kani::assume(ly < T && lx < T && i < R && j < R && kk < T);
        assert!((ly * R + i) * T + kk < BLOCK * T);
        assert!(kk * BLOCK + lx * R + j < BLOCK * T);
        // The cooperative load fills each tile exactly once: R cells per thread.
        let (tid, load): (u32, u32) = (kani::any(), kani::any());
        kani::assume(tid < T * T && load < R);
        assert!(tid + load * T * T < BLOCK * T);
    }

    #[kani::proof]
    #[kani::solver(kissat)]
    fn transpose_loads_and_stores_stay_in_range() {
        let (rows, cols): (u32, u32) = (kani::any(), kani::any());
        kani::assume((1..=MAX_DIM).contains(&rows) && (1..=MAX_DIM).contains(&cols));
        let (tile_row, tile_col): (u32, u32) = (kani::any(), kani::any());
        kani::assume(tile_row < rows.div_ceil(T) && tile_col < cols.div_ceil(T));
        let (ly, lx): (u32, u32) = (kani::any(), kani::any());
        kani::assume(ly < T && lx < T);
        let r = u64::from(tile_row) * u64::from(T) + u64::from(ly);
        let c = u64::from(tile_col) * u64::from(T) + u64::from(lx);
        if r < u64::from(rows) && c < u64::from(cols) {
            let index = mul_add(r, u64::from(cols), c).expect("the load index fits u32");
            assert!(u64::from(index) < u64::from(rows) * u64::from(cols));
        }
        let out_row = u64::from(tile_col) * u64::from(T) + u64::from(ly);
        let out_col = u64::from(tile_row) * u64::from(T) + u64::from(lx);
        if out_row < u64::from(cols) && out_col < u64::from(rows) {
            let index =
                mul_add(out_row, u64::from(rows), out_col).expect("the store index fits u32");
            assert!(u64::from(index) < u64::from(rows) * u64::from(cols));
        }
    }
}
