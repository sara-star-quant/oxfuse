//! Fixed WGSL templates. The only inputs are the kernel configuration and a few
//! enums, so no user text ever reaches a shader.
//!
//! Conventions every kernel follows:
//! - `@group(0)`: storage inputs first, then the output, then a uniform `Params`
//!   as the last binding. At most three storage buffers (downlevel devices allow four).
//! - Linear indices are computed as `config::element_index` does; tile kernels
//!   take a linear group id `wid.y * x_count + wid.x`. A group past `total_groups` is
//!   inactive: it still reaches every barrier but guards its loads and stores with
//!   `in_grid`. Kernels with barriers never return early (the DX12 software adapter
//!   crashes on an early return before a barrier).
//! - Kernels that move or classify bits (Relu, Transpose, Permute) read and write
//!   `u32`, so no float rule (flushing, NaN assumptions) can touch them.
//! - `MatMul` sums each output in ascending `k` order from one accumulator.

use crate::config::KernelConfig;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Unary {
    Relu,
    Sigmoid,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Binary {
    Add,
    Mul,
}

const ENTRY_1D: &str =
    "@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>";

fn linear_index(wg: u32) -> String {
    format!("let i = (wid.y * p.x_count + wid.x) * {wg}u + lid.x;\n  if (i >= p.n) {{ return; }}")
}

pub fn unary(op: Unary, cfg: &KernelConfig) -> String {
    let wg = cfg.elementwise_wg;
    let index = linear_index(wg);
    let (ty, body) = match op {
        // Relu on the bit pattern: a set sign bit on a non-NaN becomes +0. This
        // keeps NaN, infinity and subnormals exactly as the reference does.
        Unary::Relu => (
            "u32",
            "let bits = src[i];
  let negative = (bits & 0x80000000u) != 0u;
  let is_nan = (bits & 0x7fffffffu) > 0x7f800000u;
  dst[i] = select(bits, 0u, negative && !is_nan);",
        ),
        Unary::Sigmoid => ("f32", "dst[i] = 1.0 / (1.0 + exp(-src[i]));"),
    };
    format!(
        "struct Params {{ n: u32, x_count: u32 }}
@group(0) @binding(0) var<storage, read> src: array<{ty}>;
@group(0) @binding(1) var<storage, read_write> dst: array<{ty}>;
@group(0) @binding(2) var<uniform> p: Params;

@compute @workgroup_size({wg})
fn main({ENTRY_1D}) {{
  {index}
  {body}
}}
"
    )
}

/// `scalar_lhs` / `scalar_rhs`: that operand holds one element (rank 0) and is
/// broadcast.
pub fn binary(op: Binary, scalar_lhs: bool, scalar_rhs: bool, cfg: &KernelConfig) -> String {
    let wg = cfg.elementwise_wg;
    let index = linear_index(wg);
    let a = if scalar_lhs { "a[0]" } else { "a[i]" };
    let b = if scalar_rhs { "b[0]" } else { "b[i]" };
    let symbol = match op {
        Binary::Add => "+",
        Binary::Mul => "*",
    };
    format!(
        "struct Params {{ n: u32, x_count: u32 }}
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;
@group(0) @binding(3) var<uniform> p: Params;

@compute @workgroup_size({wg})
fn main({ENTRY_1D}) {{
  {index}
  dst[i] = {a} {symbol} {b};
}}
"
    )
}

/// `[rows, cols] -> [cols, rows]` through a workgroup tile whose rows are padded by one.
pub fn transpose2d(cfg: &KernelConfig) -> String {
    let t = cfg.transpose_tile;
    let stride = cfg.transpose_stride();
    let cells = t * stride;
    format!(
        "struct Params {{ rows: u32, cols: u32, tiles_x: u32, x_count: u32, total_groups: u32 }}
@group(0) @binding(0) var<storage, read> src: array<u32>;
@group(0) @binding(1) var<storage, read_write> dst: array<u32>;
@group(0) @binding(2) var<uniform> p: Params;
var<workgroup> tile: array<u32, {cells}>;

@compute @workgroup_size({t}, {t})
fn main({ENTRY_1D}) {{
  let g = wid.y * p.x_count + wid.x;
  let in_grid = g < p.total_groups;
  let tile_row = g / p.tiles_x;
  let tile_col = g % p.tiles_x;
  let r = tile_row * {t}u + lid.y;
  let c = tile_col * {t}u + lid.x;
  if (in_grid && r < p.rows && c < p.cols) {{
    tile[lid.y * {stride}u + lid.x] = src[r * p.cols + c];
  }}
  workgroupBarrier();
  let out_row = tile_col * {t}u + lid.y;
  let out_col = tile_row * {t}u + lid.x;
  if (in_grid && out_row < p.cols && out_col < p.rows) {{
    dst[out_row * p.rows + out_col] = tile[lid.x * {stride}u + lid.y];
  }}
}}
"
    )
}

/// Strided gather for any permutation of a tensor of rank 1 to 8. `dims` holds
/// the output shape and `strides` the input stride that each output dimension
/// walks, both padded to eight entries, packed as two `vec4<u32>`.
pub fn permute(rank: usize, cfg: &KernelConfig) -> String {
    assert!((1..=8).contains(&rank), "rank {rank} is outside 1..=8");
    let wg = cfg.elementwise_wg;
    let index = linear_index(wg);
    let mut walk = String::new();
    for d in (0..rank).rev() {
        let (lane, slot) = (d / 4, d % 4);
        walk.push_str(&format!(
            "  {{
    let dim = p.dims[{lane}][{slot}];
    off += (rem % dim) * p.strides[{lane}][{slot}];
    rem = rem / dim;
  }}
"
        ));
    }
    format!(
        "struct Params {{ n: u32, x_count: u32, dims: array<vec4<u32>, 2>, strides: array<vec4<u32>, 2> }}
@group(0) @binding(0) var<storage, read> src: array<u32>;
@group(0) @binding(1) var<storage, read_write> dst: array<u32>;
@group(0) @binding(2) var<uniform> p: Params;

@compute @workgroup_size({wg})
fn main({ENTRY_1D}) {{
  {index}
  var rem = i;
  var off = 0u;
{walk}  dst[i] = src[off];
}}
"
    )
}

/// `[m, k] x [k, n]`. A workgroup of `tile * tile` threads covers a square of
/// `block = tile * reg_block` rows and columns of the result; each thread holds a
/// `reg_block` by `reg_block` sub-square of it in scalar variables (written out
/// one by one: an array indexed in a loop is spilled to memory and runs about
/// three times slower). Per k-tile the workgroup loads a `block` by `tile` piece
/// of A and a `tile` by `block` piece of B into workgroup memory, `reg_block`
/// cells per thread each. Every output element still sums its products in
/// ascending `k` from one accumulator.
pub fn matmul(cfg: &KernelConfig) -> String {
    let t = cfg.tile;
    let r = cfg.reg_block;
    let b = cfg.matmul_block();
    let cells = b * t;
    let per_thread = r;
    let step = t * t;
    let mut acc = String::new();
    let mut inner = String::new();
    let mut store = String::new();
    for i in 0..r {
        for j in 0..r {
            acc.push_str(&format!("  var c{i}_{j}: f32 = 0.0;\n"));
        }
    }
    for i in 0..r {
        inner.push_str(&format!(
            "      let av{i} = a_tile[(lid.y * {r}u + {i}u) * {t}u + kk];\n"
        ));
    }
    for j in 0..r {
        inner.push_str(&format!(
            "      let bv{j} = b_tile[kk * {b}u + lid.x * {r}u + {j}u];\n"
        ));
    }
    for i in 0..r {
        for j in 0..r {
            inner.push_str(&format!("      c{i}_{j} = c{i}_{j} + av{i} * bv{j};\n"));
            store.push_str(&format!(
                "  {{
    let rr = row0 + lid.y * {r}u + {i}u;
    let cc = col0 + lid.x * {r}u + {j}u;
    if (in_grid && rr < p.m && cc < p.n) {{ dst[rr * p.n + cc] = c{i}_{j}; }}
  }}
"
            ));
        }
    }
    format!(
        "struct Params {{ m: u32, k: u32, n: u32, tiles_x: u32, x_count: u32, total_groups: u32 }}
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;
@group(0) @binding(3) var<uniform> p: Params;
var<workgroup> a_tile: array<f32, {cells}>;
var<workgroup> b_tile: array<f32, {cells}>;

@compute @workgroup_size({t}, {t})
fn main({ENTRY_1D}) {{
  let g = wid.y * p.x_count + wid.x;
  let in_grid = g < p.total_groups;
  let row0 = (g / p.tiles_x) * {b}u;
  let col0 = (g % p.tiles_x) * {b}u;
  let tid = lid.y * {t}u + lid.x;
{acc}  var k0 = 0u;
  loop {{
    if (k0 >= p.k) {{ break; }}
    for (var load = 0u; load < {per_thread}u; load++) {{
      let idx = tid + load * {step}u;
      let gr = row0 + idx / {t}u;
      let gk = k0 + idx % {t}u;
      var v: f32 = 0.0;
      if (in_grid && gr < p.m && gk < p.k) {{ v = a[gr * p.k + gk]; }}
      a_tile[idx] = v;
      let gk2 = k0 + idx / {b}u;
      let gc = col0 + idx % {b}u;
      var w: f32 = 0.0;
      if (in_grid && gk2 < p.k && gc < p.n) {{ w = b[gk2 * p.n + gc]; }}
      b_tile[idx] = w;
    }}
    workgroupBarrier();
    let kmax = min({t}u, p.k - k0);
    for (var kk = 0u; kk < kmax; kk++) {{
{inner}    }}
    workgroupBarrier();
    k0 += {t}u;
  }}
{store}}}
"
    )
}
