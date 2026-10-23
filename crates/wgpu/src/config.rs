//! Kernel parameters and dispatch arithmetic as plain integer functions with no
//! wgpu types, so tests and proofs can run them without a device.

/// Storage buffers bound by the widest kernel (two inputs and one output).
/// Downlevel devices guarantee four.
pub const KERNEL_STORAGE_BUFFERS: u32 = 3;

/// Largest tensor, in elements, whose linear index fits the shaders' 32-bit
/// arithmetic with room for the padding the grid fold adds.
pub const MAX_TENSOR_ELEMENTS: u64 = 1 << 30;

/// The device limits the kernels are checked against, as plain numbers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeviceCaps {
    pub max_workgroups_per_dim: u32,
    pub max_invocations: u32,
    pub max_wg_storage_bytes: u32,
    pub max_binding_bytes: u64,
}

impl DeviceCaps {
    /// What every supported device guarantees (`wgpu::Limits::downlevel_defaults`,
    /// pinned by a test).
    pub const DOWNLEVEL: DeviceCaps = DeviceCaps {
        max_workgroups_per_dim: 65535,
        max_invocations: 256,
        max_wg_storage_bytes: 16352,
        max_binding_bytes: 128 << 20,
    };
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KernelConfig {
    /// Threads per workgroup of the elementwise kernels.
    pub elementwise_wg: u32,
    /// Threads per side of a `MatMul` workgroup, which is `tile * tile` threads.
    pub tile: u32,
    /// Outputs each `MatMul` thread computes per side, so one workgroup covers a
    /// `tile * reg_block` square of the result.
    pub reg_block: u32,
    /// Side of the square `Transpose` tile; its row is padded by one element.
    pub transpose_tile: u32,
}

impl KernelConfig {
    pub const DEFAULT: KernelConfig = KernelConfig {
        elementwise_wg: 256,
        tile: 16,
        reg_block: 4,
        transpose_tile: 16,
    };

    /// Row stride of the transpose tile in workgroup memory: one more than the
    /// tile, so a column read does not hit one memory bank for every row.
    pub fn transpose_stride(&self) -> u32 {
        self.transpose_tile + 1
    }

    /// Side of the square of the result one `MatMul` workgroup covers.
    pub fn matmul_block(&self) -> u32 {
        self.tile * self.reg_block
    }

    /// Whether a device with `caps` can run every kernel built from this config.
    pub fn validate(&self, caps: &DeviceCaps) -> Result<(), String> {
        for (name, value) in [
            ("elementwise_wg", self.elementwise_wg),
            ("tile", self.tile),
            ("reg_block", self.reg_block),
            ("transpose_tile", self.transpose_tile),
        ] {
            if value == 0 {
                return Err(format!("{name} is zero"));
            }
        }
        let invocations = |name: &str, needed: u64| {
            if needed > u64::from(caps.max_invocations) {
                Err(format!(
                    "{name} needs {needed} invocations per workgroup, the device allows {}",
                    caps.max_invocations
                ))
            } else {
                Ok(())
            }
        };
        let memory = |name: &str, floats: u64| {
            let bytes = floats * 4;
            if bytes > u64::from(caps.max_wg_storage_bytes) {
                Err(format!(
                    "{name} needs {bytes} bytes of workgroup memory, the device allows {}",
                    caps.max_wg_storage_bytes
                ))
            } else {
                Ok(())
            }
        };
        let tile = u64::from(self.tile);
        let transpose = u64::from(self.transpose_tile);
        invocations("elementwise_wg", u64::from(self.elementwise_wg))?;
        invocations("tile", tile * tile)?;
        invocations("transpose_tile", transpose * transpose)?;
        let block = u64::from(self.matmul_block());
        // One tile of A (block rows by tile columns) and one of B (tile rows by block columns).
        memory("the matmul tiles", 2 * block * tile)?;
        memory("the transpose tile", transpose * (transpose + 1))
    }
}

/// Workgroup counts `(x, y)` for `elements` threads of `wg` each, within
/// `max_per_dim` per dimension, using the fewest rows. `None` for an empty tensor
/// or one too large for the device.
pub fn grid_fold(elements: u64, wg: u32, max_per_dim: u32) -> Option<(u32, u32)> {
    if elements == 0 || wg == 0 || max_per_dim == 0 {
        return None;
    }
    let groups = elements.div_ceil(u64::from(wg));
    let max = u64::from(max_per_dim);
    if groups <= max {
        return Some((groups as u32, 1));
    }
    let y = groups.div_ceil(max);
    if y > max {
        return None;
    }
    let x = groups.div_ceil(y);
    Some((x as u32, y as u32))
}

/// The linear element a thread handles, exactly as the shaders compute it.
pub fn element_index(gid_x: u32, gid_y: u32, x_count: u32, wg: u32, local: u32) -> u64 {
    (u64::from(gid_y) * u64::from(x_count) + u64::from(gid_x)) * u64::from(wg) + u64::from(local)
}

/// Whether a tensor of `elements` can live in one storage buffer on this device.
pub fn check_tensor(elements: u64, caps: &DeviceCaps) -> Result<(), String> {
    let cap = caps.max_binding_bytes / 4;
    if elements > cap {
        return Err(format!(
            "a tensor of {elements} elements exceeds the storage buffer limit of {cap} elements ({} bytes)",
            caps.max_binding_bytes
        ));
    }
    if elements > MAX_TENSOR_ELEMENTS {
        return Err(format!(
            "a tensor of {elements} elements exceeds the 32-bit index limit of {MAX_TENSOR_ELEMENTS} elements"
        ));
    }
    Ok(())
}

/// Tile grid for a `[rows, cols]` matrix covered by `tile`-sized squares, folded
/// into at most `max_per_dim` workgroups per dispatch dimension.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TileGrid {
    pub tiles_x: u32,
    pub tiles_y: u32,
    pub x_count: u32,
    pub y_count: u32,
}

impl TileGrid {
    pub fn total_groups(&self) -> u64 {
        u64::from(self.tiles_x) * u64::from(self.tiles_y)
    }
}

pub fn tile_grid(rows: u64, cols: u64, tile: u32, max_per_dim: u32) -> Option<TileGrid> {
    // Zero rows or columns give zero tiles, which `grid_fold` rejects below.
    if tile == 0 {
        return None;
    }
    let tiles_x = u32::try_from(cols.div_ceil(u64::from(tile))).ok()?;
    let tiles_y = u32::try_from(rows.div_ceil(u64::from(tile))).ok()?;
    let (x_count, y_count) = grid_fold(u64::from(tiles_x) * u64::from(tiles_y), 1, max_per_dim)?;
    Some(TileGrid {
        tiles_x,
        tiles_y,
        x_count,
        y_count,
    })
}

/// Output dimensions and the input stride each output dimension walks, for a
/// row-major input permuted by `perm` (`out dim i = in dim perm[i]`).
pub fn permute_layout(in_shape: &[usize], perm: &[usize]) -> (Vec<u32>, Vec<u32>) {
    let mut dims = vec![0; perm.len()];
    let mut strides = vec![0; perm.len()];
    permute_layout_into(in_shape, perm, &mut dims, &mut strides);
    (dims, strides)
}

/// `permute_layout` without allocation, for ranks up to 8. `dims` and `strides`
/// must have `perm.len()` entries. Kept apart so the proofs can run it.
pub fn permute_layout_into(
    in_shape: &[usize],
    perm: &[usize],
    dims: &mut [u32],
    strides: &mut [u32],
) {
    let rank = in_shape.len();
    assert_eq!(rank, perm.len(), "perm must name every dimension");
    assert!(rank <= 8 && dims.len() == rank && strides.len() == rank);
    let mut in_strides = [1u32; 8];
    for d in (0..rank.saturating_sub(1)).rev() {
        in_strides[d] = in_strides[d + 1] * in_shape[d + 1] as u32;
    }
    for (k, &p) in perm.iter().enumerate() {
        dims[k] = in_shape[p] as u32;
        strides[k] = in_strides[p];
    }
}

/// The input element the permute kernel reads for output element `i`: the walk
/// the shader does, from the last output dimension to the first.
pub fn permute_offset(i: u64, dims: &[u32], strides: &[u32]) -> u64 {
    let mut rem = i;
    let mut off = 0u64;
    for d in (0..dims.len()).rev() {
        let dim = u64::from(dims[d]);
        off += (rem % dim) * u64::from(strides[d]);
        rem /= dim;
    }
    off
}

#[cfg(test)]
mod tests {
    use super::*;

    const PER_DIM: u32 = 65535;

    fn assert_fold_covers(elements: u64, wg: u32, per_dim: u32) {
        let (x, y) = grid_fold(elements, wg, per_dim)
            .unwrap_or_else(|| panic!("{elements} elements at wg {wg} must fit"));
        assert!(
            x >= 1 && y >= 1 && x <= per_dim && y <= per_dim,
            "{x} x {y}"
        );
        let capacity = u64::from(x) * u64::from(y) * u64::from(wg);
        assert!(capacity >= elements, "{x} x {y} x {wg} < {elements}");
        let needed_groups = elements.div_ceil(u64::from(wg));
        let spare_groups = u64::from(x) * u64::from(y) - needed_groups;
        assert!(spare_groups < u64::from(x), "wastes a whole row: {x} x {y}");
    }

    #[test]
    fn fold_boundaries() {
        for wg in [32u32, 256] {
            let w = u64::from(wg);
            let full = u64::from(PER_DIM) * w;
            for elements in [
                1,
                w - 1,
                w,
                w + 1,
                full - 1,
                full,
                full + 1,
                2 * full,
                u64::from(PER_DIM) * u64::from(PER_DIM) * w,
            ] {
                assert_fold_covers(elements, wg, PER_DIM);
            }
        }
    }

    #[test]
    fn small_counts_use_one_row() {
        assert_eq!(grid_fold(1, 256, PER_DIM), Some((1, 1)));
        assert_eq!(grid_fold(257, 256, PER_DIM), Some((2, 1)));
    }

    #[test]
    fn the_fold_has_a_ceiling_and_rejects_empty_tensors() {
        let ceiling = u64::from(PER_DIM) * u64::from(PER_DIM) * 256;
        assert_eq!(grid_fold(ceiling, 256, PER_DIM), Some((PER_DIM, PER_DIM)));
        assert_eq!(grid_fold(ceiling + 1, 256, PER_DIM), None);
        assert_eq!(grid_fold(0, 256, PER_DIM), None);
    }

    #[test]
    fn a_tiny_dimension_limit_forces_two_dimensions() {
        assert_eq!(grid_fold(7 * 4, 4, 3), Some((3, 3)));
        assert_fold_covers(7 * 4, 4, 3);
        assert_fold_covers(9 * 4, 4, 3);
        assert_eq!(grid_fold(10 * 4, 4, 3), None);
    }

    #[test]
    fn element_index_is_row_major_over_groups() {
        assert_eq!(element_index(0, 0, 7, 256, 0), 0);
        assert_eq!(element_index(3, 0, 7, 256, 5), 3 * 256 + 5);
        assert_eq!(element_index(2, 4, 7, 256, 1), (4 * 7 + 2) * 256 + 1);
        // No u32 overflow on the way: the last group of the largest grid.
        let last = element_index(PER_DIM - 1, PER_DIM - 1, PER_DIM, 256, 255);
        assert_eq!(last, u64::from(PER_DIM) * u64::from(PER_DIM) * 256 - 1);
    }

    #[test]
    fn the_default_config_fits_the_downlevel_limits() {
        assert_eq!(
            KernelConfig::DEFAULT.validate(&DeviceCaps::DOWNLEVEL),
            Ok(())
        );
    }

    #[test]
    fn downlevel_caps_match_wgpu() {
        let limits = wgpu::Limits::downlevel_defaults();
        let caps = DeviceCaps::DOWNLEVEL;
        assert_eq!(
            caps.max_workgroups_per_dim,
            limits.max_compute_workgroups_per_dimension
        );
        assert_eq!(
            caps.max_invocations,
            limits.max_compute_invocations_per_workgroup
        );
        assert_eq!(
            caps.max_wg_storage_bytes,
            limits.max_compute_workgroup_storage_size
        );
        assert_eq!(
            caps.max_binding_bytes,
            limits.max_storage_buffer_binding_size
        );
        assert!(limits.max_storage_buffers_per_shader_stage >= KERNEL_STORAGE_BUFFERS);
    }

    #[test]
    fn validate_rejects_configs_a_device_cannot_run() {
        let caps = DeviceCaps::DOWNLEVEL;
        let bad = |config: KernelConfig| config.validate(&caps).unwrap_err();
        assert!(
            bad(KernelConfig {
                tile: 17,
                ..KernelConfig::DEFAULT
            })
            .contains("invocations")
        );
        assert!(
            bad(KernelConfig {
                elementwise_wg: 512,
                ..KernelConfig::DEFAULT
            })
            .contains("invocations")
        );
        assert!(
            bad(KernelConfig {
                transpose_tile: 0,
                ..KernelConfig::DEFAULT
            })
            .contains("zero")
        );
        assert!(
            bad(KernelConfig {
                reg_block: 0,
                ..KernelConfig::DEFAULT
            })
            .contains("zero")
        );
        // 16 x 16 threads each holding 8 x 8 outputs need 2 * 128 * 16 floats of workgroup memory.
        assert!(
            bad(KernelConfig {
                reg_block: 8,
                ..KernelConfig::DEFAULT
            })
            .contains("workgroup memory")
        );
        let tight = DeviceCaps {
            max_wg_storage_bytes: 1024,
            ..caps
        };
        assert!(
            KernelConfig::DEFAULT
                .validate(&tight)
                .unwrap_err()
                .contains("workgroup memory")
        );
    }

    #[test]
    fn validate_is_exact_at_the_limits() {
        let transpose = KernelConfig {
            elementwise_wg: 1,
            tile: 1,
            reg_block: 1,
            transpose_tile: 16,
        };
        // A 16 by 16 transpose tile: 256 invocations and 16 * 17 floats = 1088 bytes.
        let caps = DeviceCaps {
            max_invocations: 256,
            max_wg_storage_bytes: 1088,
            ..DeviceCaps::DOWNLEVEL
        };
        assert_eq!(transpose.validate(&caps), Ok(()));
        let fewer = DeviceCaps {
            max_invocations: 255,
            ..caps
        };
        assert!(
            transpose
                .validate(&fewer)
                .unwrap_err()
                .contains("transpose_tile")
        );
        let smaller = DeviceCaps {
            max_wg_storage_bytes: 1087,
            ..caps
        };
        assert!(
            transpose
                .validate(&smaller)
                .unwrap_err()
                .contains("transpose tile")
        );
        // The default matmul tiles: 2 * 64 * 16 floats = 8192 bytes.
        let caps = DeviceCaps {
            max_wg_storage_bytes: 8192,
            ..DeviceCaps::DOWNLEVEL
        };
        assert_eq!(KernelConfig::DEFAULT.validate(&caps), Ok(()));
        let smaller = DeviceCaps {
            max_wg_storage_bytes: 8191,
            ..caps
        };
        assert!(
            KernelConfig::DEFAULT
                .validate(&smaller)
                .unwrap_err()
                .contains("matmul tiles")
        );
    }

    #[test]
    fn a_zero_in_any_position_means_no_grid() {
        assert_eq!(grid_fold(0, 256, PER_DIM), None);
        assert_eq!(grid_fold(10, 0, PER_DIM), None);
        assert_eq!(grid_fold(10, 256, 0), None);
        assert_eq!(tile_grid(0, 4, 16, PER_DIM), None);
        assert_eq!(tile_grid(4, 0, 16, PER_DIM), None);
        assert_eq!(tile_grid(4, 4, 0, PER_DIM), None);
        assert_eq!(tile_grid(4, 4, 16, 0), None);
    }

    #[test]
    fn the_matmul_block_is_the_tile_times_the_register_block() {
        assert_eq!(KernelConfig::DEFAULT.matmul_block(), 64);
        assert_eq!(KernelConfig::DEFAULT.transpose_stride(), 17);
        let one = KernelConfig {
            reg_block: 1,
            ..KernelConfig::DEFAULT
        };
        assert_eq!(one.matmul_block(), 16);
    }

    #[test]
    fn check_tensor_names_the_limit() {
        let caps = DeviceCaps::DOWNLEVEL;
        let cap = caps.max_binding_bytes / 4;
        assert_eq!(check_tensor(cap, &caps), Ok(()));
        let message = check_tensor(cap + 1, &caps).unwrap_err();
        assert!(message.contains(&cap.to_string()), "{message}");
        assert!(message.contains("storage buffer"), "{message}");
    }

    #[test]
    fn check_tensor_keeps_indices_inside_u32() {
        let roomy = DeviceCaps {
            max_binding_bytes: u64::MAX,
            ..DeviceCaps::DOWNLEVEL
        };
        assert_eq!(check_tensor(MAX_TENSOR_ELEMENTS, &roomy), Ok(()));
        assert!(
            check_tensor(MAX_TENSOR_ELEMENTS + 1, &roomy)
                .unwrap_err()
                .contains("32-bit")
        );
    }

    #[test]
    fn tile_grid_covers_every_tile_exactly_once() {
        let g = tile_grid(33, 17, 16, PER_DIM).unwrap();
        assert_eq!((g.tiles_x, g.tiles_y), (2, 3));
        assert_eq!((g.x_count, g.y_count), (6, 1));
        let g = tile_grid(3, 3, 1, 3).unwrap();
        assert_eq!((g.tiles_x, g.tiles_y, g.x_count, g.y_count), (3, 3, 3, 3));
        let g = tile_grid(2, 3, 1, 3).unwrap();
        assert_eq!((g.total_groups(), g.x_count, g.y_count), (6, 3, 2));
    }

    #[test]
    fn tile_grid_rejects_what_the_device_cannot_dispatch() {
        assert_eq!(tile_grid(0, 4, 16, PER_DIM), None);
        assert_eq!(tile_grid(100, 100, 1, 3), None);
    }

    #[test]
    fn permute_layout_walks_the_input_strides_in_output_order() {
        let (dims, strides) = permute_layout(&[2, 3, 4], &[2, 0, 1]);
        assert_eq!(dims, vec![4, 2, 3]);
        assert_eq!(strides, vec![1, 12, 4]);
        let (dims, strides) = permute_layout(&[5], &[0]);
        assert_eq!((dims, strides), (vec![5], vec![1]));
    }

    #[test]
    fn permute_offset_walks_the_output_in_row_major_order() {
        let (dims, strides) = permute_layout(&[2, 3, 4], &[2, 0, 1]);
        assert_eq!(permute_offset(0, &dims, &strides), 0);
        assert_eq!(permute_offset(1, &dims, &strides), 4);
        assert_eq!(permute_offset(3, &dims, &strides), 12);
    }

    #[test]
    fn permute_offset_visits_every_input_element_exactly_once() {
        for (shape, perm) in [
            (vec![2usize, 3, 4], vec![2usize, 0, 1]),
            (vec![3, 1, 2], vec![1, 2, 0]),
            (vec![5], vec![0]),
            (vec![2, 2, 2, 2], vec![3, 2, 1, 0]),
        ] {
            let (dims, strides) = permute_layout(&shape, &perm);
            let n: u64 = shape.iter().map(|&d| d as u64).product();
            let mut seen = vec![false; n as usize];
            for i in 0..n {
                let off = permute_offset(i, &dims, &strides);
                assert!(off < n, "{shape:?} {perm:?}: offset {off} for {i}");
                assert!(
                    !seen[off as usize],
                    "{shape:?} {perm:?}: {off} visited twice"
                );
                seen[off as usize] = true;
            }
        }
    }
}
