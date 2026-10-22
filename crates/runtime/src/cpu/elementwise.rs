//! Elementwise kernels of the fast CPU backend: Sigmoid (split across threads
//! under the `parallel` feature) and the in-place forms the executor uses.
//!
//! They live apart from the matmul on purpose: with both in one module the
//! single-threaded 512 matmul ran 9 percent slower (5.9 ms to 6.4 ms, spread
//! 3 percent) after the in-place kernels were added, and moving them out
//! restored it (docs/benchmarks.md).

use super::vecmath;
use fearless_simd::Level;
use oxfuse_core::Op;

/// Elements below which splitting Sigmoid across threads costs more than it
/// saves. Only Sigmoid is split: it is bound by `exp`, while Relu, Add and Mul
/// are bound by memory and got slower when split (see docs/benchmarks.md).
#[cfg(feature = "parallel")]
const SIGMOID_PARALLEL_MIN: usize = 1 << 16;
#[cfg(feature = "parallel")]
const SIGMOID_CHUNK: usize = 1 << 13;

/// Each element is computed alone, so the bits do not depend on the split.
pub(super) fn sigmoid(input: &[f32]) -> Vec<f32> {
    let level = Level::new();
    let mut out = vec![0.0f32; input.len()];
    #[cfg(feature = "parallel")]
    if input.len() >= SIGMOID_PARALLEL_MIN {
        use rayon::prelude::*;
        out.par_chunks_mut(SIGMOID_CHUNK)
            .zip(input.par_chunks(SIGMOID_CHUNK))
            .for_each(|(out, input)| vecmath::sigmoid_into(level, input, out));
        return out;
    }
    vecmath::sigmoid_into(level, input, &mut out);
    out
}

fn sigmoid_in_place(buf: &mut [f32]) {
    let level = Level::new();
    #[cfg(feature = "parallel")]
    if buf.len() >= SIGMOID_PARALLEL_MIN {
        use rayon::prelude::*;
        buf.par_chunks_mut(SIGMOID_CHUNK)
            .for_each(|chunk| vecmath::sigmoid_in_place(level, chunk));
        return;
    }
    vecmath::sigmoid_in_place(level, buf);
}

/// Computes an elementwise `op` into `buf`, which holds one operand and is
/// overwritten with the result. `other` is the second operand of Add and Mul
/// (one element for a scalar); `buf_first` says which operand `buf` held, so
/// the operation sees the operands in the same order as the reference.
pub(super) fn apply_in_place(op: &Op, buf: &mut [f32], other: Option<&[f32]>, buf_first: bool) {
    let zip = |buf: &mut [f32], other: &[f32], f: fn(f32, f32) -> f32| {
        let at = |i: usize| other[if other.len() == 1 { 0 } else { i }];
        for (i, v) in buf.iter_mut().enumerate() {
            *v = if buf_first {
                f(*v, at(i))
            } else {
                f(at(i), *v)
            };
        }
    };
    match op {
        Op::Relu => buf
            .iter_mut()
            .for_each(|x| *x = crate::kernels::relu_one(*x)),
        Op::Sigmoid => sigmoid_in_place(buf),
        Op::Add => zip(buf, other.expect("Add has two operands"), |x, y| x + y),
        Op::Mul => zip(buf, other.expect("Mul has two operands"), |x, y| x * y),
        Op::Reshape { .. } | Op::MatMul | Op::Transpose { .. } => {
            unreachable!("not an in-place op")
        }
    }
}
