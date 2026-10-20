//! A CPU model of each kernel grid. It walks the same workgroups and threads
//! with the same index arithmetic and guards as the shaders in `shaders.rs`
//! (barriers become phase boundaries), but a device clamps a bad access and
//! returns plausible numbers, while the model panics on any out-of-range read or
//! write, any element written twice and any element never written. That is what
//! lets tests find indexing bugs without a device. It does not prove the WGSL
//! text matches; the twin tests compare it with a real adapter for that.

use crate::config::{
    DeviceCaps, KernelConfig, element_index, grid_fold, permute_layout, permute_offset, tile_grid,
};
use crate::shaders::{Binary, Unary};

/// Output buffer that insists on exactly one write per element.
struct Out {
    bits: Vec<u32>,
    written: Vec<bool>,
}

impl Out {
    fn new(len: usize) -> Out {
        Out {
            bits: vec![0; len],
            written: vec![false; len],
        }
    }

    fn write(&mut self, index: u64, value: u32) {
        let i = usize::try_from(index).unwrap();
        assert!(
            i < self.bits.len(),
            "out-of-range write at {index} of {}",
            self.bits.len()
        );
        assert!(!self.written[i], "element {index} written twice");
        self.bits[i] = value;
        self.written[i] = true;
    }

    fn finish(self) -> Vec<f32> {
        if let Some(missing) = self.written.iter().position(|w| !w) {
            panic!("element {missing} was never written");
        }
        self.bits.into_iter().map(f32::from_bits).collect()
    }
}

fn read<T: Copy>(slice: &[T], index: u64) -> T {
    usize::try_from(index)
        .ok()
        .and_then(|i| slice.get(i))
        .copied()
        .unwrap_or_else(|| panic!("out-of-range read at {index} of {}", slice.len()))
}

/// Workgroup counts for `n` elementwise threads, or a panic if the device could
/// not dispatch it (the real backend rejects such a tensor at compile).
fn elementwise_grid(n: u64, cfg: &KernelConfig, caps: &DeviceCaps) -> (u32, u32) {
    grid_fold(n, cfg.elementwise_wg, caps.max_workgroups_per_dim)
        .unwrap_or_else(|| panic!("{n} elements cannot be dispatched"))
}

/// Runs `body(i)` for every thread whose element index is below `n`, in grid order.
fn for_each_thread(n: u64, cfg: &KernelConfig, caps: &DeviceCaps, mut body: impl FnMut(u64)) {
    let wg = cfg.elementwise_wg;
    let (x_count, y_count) = elementwise_grid(n, cfg, caps);
    for gid_y in 0..y_count {
        for gid_x in 0..x_count {
            for local in 0..wg {
                let i = element_index(gid_x, gid_y, x_count, wg, local);
                if i >= n {
                    continue;
                }
                body(i);
            }
        }
    }
}

pub fn unary(op: Unary, src: &[f32], cfg: &KernelConfig, caps: &DeviceCaps) -> Vec<f32> {
    let n = src.len() as u64;
    let mut out = Out::new(src.len());
    for_each_thread(n, cfg, caps, |i| {
        let x = read(src, i);
        let value = match op {
            Unary::Relu => {
                let bits = x.to_bits();
                let negative = bits & 0x8000_0000 != 0;
                let is_nan = bits & 0x7fff_ffff > 0x7f80_0000;
                if negative && !is_nan { 0 } else { bits }
            }
            Unary::Sigmoid => (1.0 / (1.0 + (-x).exp())).to_bits(),
        };
        out.write(i, value);
    });
    out.finish()
}

pub fn binary(
    op: Binary,
    a: &[f32],
    b: &[f32],
    scalar_lhs: bool,
    scalar_rhs: bool,
    cfg: &KernelConfig,
    caps: &DeviceCaps,
) -> Vec<f32> {
    let len = match (scalar_lhs, scalar_rhs) {
        (true, true) => 1,
        (true, false) => b.len(),
        _ => a.len(),
    };
    let mut out = Out::new(len);
    for_each_thread(len as u64, cfg, caps, |i| {
        let x = read(a, if scalar_lhs { 0 } else { i });
        let y = read(b, if scalar_rhs { 0 } else { i });
        let value = match op {
            Binary::Add => x + y,
            Binary::Mul => x * y,
        };
        out.write(i, value.to_bits());
    });
    out.finish()
}

pub fn permute(
    src: &[f32],
    in_shape: &[usize],
    perm: &[usize],
    cfg: &KernelConfig,
    caps: &DeviceCaps,
) -> Vec<f32> {
    let (dims, strides) = permute_layout(in_shape, perm);
    let mut out = Out::new(src.len());
    for_each_thread(src.len() as u64, cfg, caps, |i| {
        let off = permute_offset(i, &dims, &strides);
        out.write(i, read(src, off).to_bits());
    });
    out.finish()
}

pub fn transpose2d(
    src: &[f32],
    rows: usize,
    cols: usize,
    cfg: &KernelConfig,
    caps: &DeviceCaps,
) -> Vec<f32> {
    let t = u64::from(cfg.transpose_tile);
    let stride = t + 1;
    let grid = tile_grid(
        rows as u64,
        cols as u64,
        cfg.transpose_tile,
        caps.max_workgroups_per_dim,
    )
    .unwrap_or_else(|| panic!("{rows}x{cols} cannot be dispatched"));
    let mut out = Out::new(src.len());
    for gid_y in 0..grid.y_count {
        for gid_x in 0..grid.x_count {
            let g = u64::from(gid_y) * u64::from(grid.x_count) + u64::from(gid_x);
            if g >= grid.total_groups() {
                continue;
            }
            let (tile_row, tile_col) = (g / u64::from(grid.tiles_x), g % u64::from(grid.tiles_x));
            // Workgroup memory starts undefined; reading a cell nobody wrote panics.
            let mut tile: Vec<Option<u32>> = vec![None; (t * stride) as usize];
            for ly in 0..t {
                for lx in 0..t {
                    let (r, c) = (tile_row * t + ly, tile_col * t + lx);
                    if r < rows as u64 && c < cols as u64 {
                        tile[(ly * stride + lx) as usize] =
                            Some(read(src, r * cols as u64 + c).to_bits());
                    }
                }
            }
            // barrier
            for ly in 0..t {
                for lx in 0..t {
                    let (out_row, out_col) = (tile_col * t + ly, tile_row * t + lx);
                    if out_row < cols as u64 && out_col < rows as u64 {
                        let cell = tile[(lx * stride + ly) as usize]
                            .expect("read a tile cell that was never loaded");
                        out.write(out_row * rows as u64 + out_col, cell);
                    }
                }
            }
        }
    }
    out.finish()
}

pub fn matmul(
    a: &[f32],
    b: &[f32],
    m: usize,
    k: usize,
    n: usize,
    cfg: &KernelConfig,
    caps: &DeviceCaps,
) -> Vec<f32> {
    let t = u64::from(cfg.tile);
    let r = u64::from(cfg.reg_block);
    let block = t * r;
    let (m, k, n) = (m as u64, k as u64, n as u64);
    let grid = tile_grid(m, n, cfg.matmul_block(), caps.max_workgroups_per_dim)
        .unwrap_or_else(|| panic!("{m}x{n} cannot be dispatched"));
    let mut out = Out::new((m * n) as usize);
    for gid_y in 0..grid.y_count {
        for gid_x in 0..grid.x_count {
            let g = u64::from(gid_y) * u64::from(grid.x_count) + u64::from(gid_x);
            if g >= grid.total_groups() {
                continue;
            }
            let (row0, col0) = (
                (g / u64::from(grid.tiles_x)) * block,
                (g % u64::from(grid.tiles_x)) * block,
            );
            // One accumulator per output a thread owns, indexed [thread][i][j].
            let mut acc = vec![0.0f32; (t * t * r * r) as usize];
            let mut k0 = 0;
            while k0 < k {
                // Workgroup memory starts undefined; reading a cell nobody loaded panics.
                let cells = (block * t) as usize;
                let mut a_tile: Vec<Option<f32>> = vec![None; cells];
                let mut b_tile: Vec<Option<f32>> = vec![None; cells];
                for tid in 0..t * t {
                    for load in 0..r {
                        let idx = tid + load * t * t;
                        let (gr, gk) = (row0 + idx / t, k0 + idx % t);
                        let av = if gr < m && gk < k {
                            read(a, gr * k + gk)
                        } else {
                            0.0
                        };
                        a_tile[idx as usize] = Some(av);
                        let (gk2, gc) = (k0 + idx / block, col0 + idx % block);
                        let bv = if gk2 < k && gc < n {
                            read(b, gk2 * n + gc)
                        } else {
                            0.0
                        };
                        b_tile[idx as usize] = Some(bv);
                    }
                }
                // barrier
                let kmax = t.min(k - k0);
                for ly in 0..t {
                    for lx in 0..t {
                        for kk in 0..kmax {
                            for i in 0..r {
                                let av = a_tile[((ly * r + i) * t + kk) as usize]
                                    .expect("an a tile cell was never loaded");
                                for j in 0..r {
                                    let bv = b_tile[(kk * block + lx * r + j) as usize]
                                        .expect("a b tile cell was never loaded");
                                    acc[(((ly * t + lx) * r + i) * r + j) as usize] += av * bv;
                                }
                            }
                        }
                    }
                }
                // barrier
                k0 += t;
            }
            for ly in 0..t {
                for lx in 0..t {
                    for i in 0..r {
                        for j in 0..r {
                            let (row, col) = (row0 + ly * r + i, col0 + lx * r + j);
                            if row < m && col < n {
                                let value = acc[(((ly * t + lx) * r + i) * r + j) as usize];
                                out.write(row * n + col, value.to_bits());
                            }
                        }
                    }
                }
            }
        }
    }
    out.finish()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[should_panic(expected = "written twice")]
    fn the_checker_rejects_a_double_write() {
        let mut out = Out::new(2);
        out.write(0, 1);
        out.write(0, 2);
    }

    #[test]
    #[should_panic(expected = "never written")]
    fn the_checker_rejects_a_missing_write() {
        let mut out = Out::new(2);
        out.write(0, 1);
        out.finish();
    }

    #[test]
    #[should_panic(expected = "out-of-range write")]
    fn the_checker_rejects_a_write_past_the_end() {
        Out::new(2).write(2, 1);
    }

    #[test]
    #[should_panic(expected = "out-of-range read")]
    fn the_checker_rejects_a_read_past_the_end() {
        read(&[1.0f32, 2.0], 2);
    }
}
