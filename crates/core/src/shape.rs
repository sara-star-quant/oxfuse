//! Shape rules: one function per concern, all arithmetic checked.

use crate::{Error, Limits, Op, TensorType};

/// Product of the dimensions, or `None` on overflow. Rank 0 gives 1.
pub fn checked_num_elements(shape: &[usize]) -> Option<usize> {
    shape.iter().try_fold(1usize, |acc, &d| acc.checked_mul(d))
}

/// Validates a declared shape against the limits and returns its element count.
pub fn check_shape(name: &str, shape: &[usize], limits: &Limits) -> Result<usize, Error> {
    if shape.len() > limits.max_rank {
        return Err(Error::TensorLimit {
            name: name.to_owned(),
            what: "tensor rank",
            found: shape.len(),
            limit: limits.max_rank,
        });
    }
    let invalid = |reason: &'static str| Error::InvalidShape {
        name: name.to_owned(),
        shape: shape.to_vec(),
        reason,
    };
    if shape.contains(&0) {
        return Err(invalid("dimensions must be at least 1"));
    }
    let count = checked_num_elements(shape).ok_or_else(|| invalid("element count overflows"))?;
    if count > limits.max_tensor_elements {
        return Err(Error::TensorLimit {
            name: name.to_owned(),
            what: "tensor elements",
            found: count,
            limit: limits.max_tensor_elements,
        });
    }
    Ok(count)
}

fn mismatch(node: &str, reason: String) -> Error {
    Error::ShapeMismatch {
        node: node.to_owned(),
        reason,
    }
}

/// The general NumPy broadcasting rule: align trailing dimensions; a
/// dimension of 1 or a missing dimension stretches to the other operand's.
pub fn numpy_broadcast(a: &[usize], b: &[usize]) -> Option<Vec<usize>> {
    let rank = a.len().max(b.len());
    let dim = |s: &[usize], i: usize| {
        if i + s.len() >= rank {
            s[i + s.len() - rank]
        } else {
            1
        }
    };
    (0..rank)
        .map(|i| match (dim(a, i), dim(b, i)) {
            (x, y) if x == y => Some(x),
            (1, y) => Some(y),
            (x, 1) => Some(x),
            _ => None,
        })
        .collect()
}

/// The subset of `numpy_broadcast` that v1 accepts: equal shapes, or one
/// operand that is a scalar. Widening this is a change to this one function.
pub fn elementwise_shape(
    node: &str,
    op: &str,
    a: &[usize],
    b: &[usize],
) -> Result<Vec<usize>, Error> {
    if a == b || b.is_empty() {
        return Ok(a.to_vec());
    }
    if a.is_empty() {
        return Ok(b.to_vec());
    }
    let hint = if numpy_broadcast(a, b).is_some() {
        " (NumPy broadcasting is not supported yet)"
    } else {
        ""
    };
    Err(mismatch(
        node,
        format!(
            "{op}: operand shapes {a:?} and {b:?} must be equal or one operand must be a scalar{hint}"
        ),
    ))
}

/// `[M, K] x [K, N] -> [M, N]`. The rule generalizes to `[..., M, K] x
/// [..., K, N]` with matching batch dimensions; v1 accepts rank 2 only.
fn matmul_shape(node: &str, a: &[usize], b: &[usize]) -> Result<Vec<usize>, Error> {
    if a.len() != 2 || b.len() != 2 {
        return Err(mismatch(
            node,
            format!(
                "MatMul: operands must be rank 2, got ranks {} and {}",
                a.len(),
                b.len()
            ),
        ));
    }
    if a[1] != b[0] {
        return Err(mismatch(
            node,
            format!("MatMul: inner dimensions differ: {a:?} x {b:?}"),
        ));
    }
    Ok(vec![a[0], b[1]])
}

fn reshape_shape(node: &str, input: &[usize], target: &[usize]) -> Result<Vec<usize>, Error> {
    match (checked_num_elements(input), checked_num_elements(target)) {
        (Some(have), Some(want)) if have == want => Ok(target.to_vec()),
        (Some(have), Some(want)) => Err(mismatch(
            node,
            format!(
                "Reshape: cannot reshape {input:?} ({have} elements) to {target:?} ({want} elements)"
            ),
        )),
        _ => Err(mismatch(
            node,
            format!("Reshape: target shape {target:?} element count overflows"),
        )),
    }
}

fn transpose_shape(node: &str, input: &[usize], perm: &[usize]) -> Result<Vec<usize>, Error> {
    let rank = input.len();
    if perm.len() != rank {
        return Err(mismatch(
            node,
            format!("Transpose: permutation {perm:?} must have {rank} entries"),
        ));
    }
    let mut seen = vec![false; rank];
    for &p in perm {
        if p >= rank || std::mem::replace(&mut seen[p], true) {
            return Err(mismatch(
                node,
                format!("Transpose: {perm:?} is not a permutation of 0..{rank}"),
            ));
        }
    }
    Ok(perm.iter().map(|&p| input[p]).collect())
}

/// Infers the output type of `op` applied to `inputs`, checking arity first.
pub fn infer(node: &str, op: &Op, inputs: &[&TensorType]) -> Result<TensorType, Error> {
    if inputs.len() != op.arity() {
        return Err(Error::Arity {
            node: node.to_owned(),
            op: op.name(),
            expected: op.arity(),
            found: inputs.len(),
        });
    }
    match op {
        Op::Relu | Op::Sigmoid => Ok(inputs[0].clone()),
        Op::Transpose { perm } => Ok(TensorType::f32(transpose_shape(
            node,
            &inputs[0].shape,
            perm,
        )?)),
        Op::Reshape { shape } => Ok(TensorType::f32(reshape_shape(
            node,
            &inputs[0].shape,
            shape,
        )?)),
        Op::MatMul => Ok(TensorType::f32(matmul_shape(
            node,
            &inputs[0].shape,
            &inputs[1].shape,
        )?)),
        Op::Add | Op::Mul => Ok(TensorType::f32(elementwise_shape(
            node,
            op.name(),
            &inputs[0].shape,
            &inputs[1].shape,
        )?)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rank_zero_has_one_element() {
        assert_eq!(checked_num_elements(&[]), Some(1));
    }

    #[test]
    fn product_overflow_is_none() {
        assert_eq!(checked_num_elements(&[usize::MAX, 2]), None);
    }

    #[test]
    fn check_shape_accepts_valid_shape() {
        assert_eq!(check_shape("x", &[2, 3], &Limits::default()), Ok(6));
        assert_eq!(check_shape("s", &[], &Limits::default()), Ok(1));
    }

    #[test]
    fn check_shape_rejects_zero_dimension() {
        let err = check_shape("x", &[2, 0], &Limits::default()).unwrap_err();
        assert!(matches!(err, Error::InvalidShape { reason, .. } if reason.contains("at least 1")));
    }

    #[test]
    fn check_shape_rejects_overflow_without_panicking() {
        let err = check_shape("x", &[usize::MAX, 2], &Limits::default()).unwrap_err();
        assert!(matches!(err, Error::InvalidShape { reason, .. } if reason.contains("overflows")));
    }

    #[test]
    fn check_shape_rejects_too_many_elements() {
        let err = check_shape("x", &[20_000, 20_000], &Limits::default()).unwrap_err();
        assert!(matches!(
            err,
            Error::TensorLimit {
                what: "tensor elements",
                ..
            }
        ));
    }

    #[test]
    fn check_shape_rejects_high_rank() {
        let err = check_shape("x", &[1; 9], &Limits::default()).unwrap_err();
        assert!(matches!(
            err,
            Error::TensorLimit {
                what: "tensor rank",
                ..
            }
        ));
    }

    #[test]
    fn relu_preserves_type() {
        let t = TensorType::f32(vec![2, 3]);
        assert_eq!(infer("r", &Op::Relu, &[&t]), Ok(t.clone()));
    }

    #[test]
    fn relu_checks_arity() {
        let err = infer("r", &Op::Relu, &[]).unwrap_err();
        assert!(matches!(
            err,
            Error::Arity {
                expected: 1,
                found: 0,
                ..
            }
        ));
    }

    #[test]
    fn tensor_limit_errors_name_the_tensor() {
        let err = check_shape("big", &[20_000, 20_000], &Limits::default()).unwrap_err();
        assert!(matches!(err, Error::TensorLimit { name, .. } if name == "big"));
        let err = check_shape("deep", &[1; 9], &Limits::default()).unwrap_err();
        assert!(matches!(err, Error::TensorLimit { name, .. } if name == "deep"));
    }

    #[test]
    fn numpy_broadcast_follows_the_general_rule() {
        assert_eq!(numpy_broadcast(&[2, 3], &[2, 3]), Some(vec![2, 3]));
        assert_eq!(numpy_broadcast(&[], &[2, 3]), Some(vec![2, 3]));
        assert_eq!(numpy_broadcast(&[3], &[2, 3]), Some(vec![2, 3]));
        assert_eq!(numpy_broadcast(&[2, 1], &[1, 3]), Some(vec![2, 3]));
        assert_eq!(numpy_broadcast(&[2, 3], &[3, 2]), None);
        assert_eq!(numpy_broadcast(&[], &[]), Some(vec![]));
    }

    #[test]
    fn elementwise_accepts_equal_shapes_and_scalars_only() {
        assert_eq!(
            elementwise_shape("n", "Add", &[2, 3], &[2, 3]),
            Ok(vec![2, 3])
        );
        assert_eq!(elementwise_shape("n", "Add", &[], &[2, 3]), Ok(vec![2, 3]));
        assert_eq!(elementwise_shape("n", "Add", &[2, 3], &[]), Ok(vec![2, 3]));
        assert_eq!(elementwise_shape("n", "Mul", &[], &[]), Ok(vec![]));
    }

    #[test]
    fn elementwise_rejects_incompatible_shapes() {
        let err = elementwise_shape("y", "Add", &[2, 3], &[3, 2]).unwrap_err();
        assert_eq!(
            err.to_string(),
            "node `y`: Add: operand shapes [2, 3] and [3, 2] must be equal or one operand must be a scalar"
        );
    }

    #[test]
    fn elementwise_names_the_missing_broadcast_feature() {
        let err = elementwise_shape("y", "Mul", &[2, 3], &[3]).unwrap_err();
        assert_eq!(
            err.to_string(),
            "node `y`: Mul: operand shapes [2, 3] and [3] must be equal or one operand must be a scalar (NumPy broadcasting is not supported yet)"
        );
    }

    #[test]
    fn binary_ops_check_arity() {
        let t = TensorType::f32(vec![2]);
        let err = infer("y", &Op::Add, &[&t]).unwrap_err();
        assert!(matches!(
            err,
            Error::Arity {
                expected: 2,
                found: 1,
                ..
            }
        ));
    }

    #[test]
    fn sigmoid_preserves_type() {
        let t = TensorType::f32(vec![4]);
        assert_eq!(infer("s", &Op::Sigmoid, &[&t]), Ok(t.clone()));
    }

    #[test]
    fn matmul_shape_is_m_by_n() {
        let a = TensorType::f32(vec![2, 3]);
        let b = TensorType::f32(vec![3, 4]);
        assert_eq!(
            infer("y", &Op::MatMul, &[&a, &b]),
            Ok(TensorType::f32(vec![2, 4]))
        );
    }

    #[test]
    fn matmul_rejects_inner_dimension_mismatch() {
        let a = TensorType::f32(vec![2, 3]);
        let b = TensorType::f32(vec![4, 5]);
        let err = infer("y", &Op::MatMul, &[&a, &b]).unwrap_err();
        assert_eq!(
            err.to_string(),
            "node `y`: MatMul: inner dimensions differ: [2, 3] x [4, 5]"
        );
    }

    #[test]
    fn matmul_rejects_non_rank_two_operands() {
        let a = TensorType::f32(vec![2, 3, 4]);
        let b = TensorType::f32(vec![4, 5]);
        let err = infer("y", &Op::MatMul, &[&a, &b]).unwrap_err();
        assert_eq!(
            err.to_string(),
            "node `y`: MatMul: operands must be rank 2, got ranks 3 and 2"
        );
    }

    fn reshape(to: Vec<usize>) -> Op {
        Op::Reshape { shape: to }
    }

    #[test]
    fn reshape_keeps_element_count() {
        let t = TensorType::f32(vec![2, 3]);
        assert_eq!(
            infer("r", &reshape(vec![3, 2]), &[&t]),
            Ok(TensorType::f32(vec![3, 2]))
        );
        assert_eq!(
            infer("r", &reshape(vec![6]), &[&t]),
            Ok(TensorType::f32(vec![6]))
        );
    }

    #[test]
    fn reshape_handles_scalars_and_identity() {
        let scalar = TensorType::f32(vec![]);
        assert_eq!(
            infer("r", &reshape(vec![1]), &[&scalar]),
            Ok(TensorType::f32(vec![1]))
        );
        let one = TensorType::f32(vec![1]);
        assert_eq!(
            infer("r", &reshape(vec![]), &[&one]),
            Ok(TensorType::f32(vec![]))
        );
        let t = TensorType::f32(vec![2, 3]);
        assert_eq!(infer("r", &reshape(vec![2, 3]), &[&t]), Ok(t.clone()));
    }

    #[test]
    fn reshape_rejects_element_count_change() {
        let t = TensorType::f32(vec![2, 3]);
        let err = infer("y", &reshape(vec![4]), &[&t]).unwrap_err();
        assert_eq!(
            err.to_string(),
            "node `y`: Reshape: cannot reshape [2, 3] (6 elements) to [4] (4 elements)"
        );
    }

    #[test]
    fn reshape_target_overflow_is_rejected() {
        let t = TensorType::f32(vec![2, 3]);
        let err = infer("y", &reshape(vec![usize::MAX, 2]), &[&t]).unwrap_err();
        assert!(matches!(err, Error::ShapeMismatch { reason, .. } if reason.contains("overflows")));
    }

    fn transpose(perm: Vec<usize>) -> Op {
        Op::Transpose { perm }
    }

    #[test]
    fn transpose_permutes_dimensions() {
        let t = TensorType::f32(vec![2, 3, 4]);
        assert_eq!(
            infer("t", &transpose(vec![2, 0, 1]), &[&t]),
            Ok(TensorType::f32(vec![4, 2, 3]))
        );
    }

    #[test]
    fn transpose_handles_low_ranks_and_size_one_dims() {
        let scalar = TensorType::f32(vec![]);
        assert_eq!(
            infer("t", &transpose(vec![]), &[&scalar]),
            Ok(scalar.clone())
        );
        let vector = TensorType::f32(vec![5]);
        assert_eq!(
            infer("t", &transpose(vec![0]), &[&vector]),
            Ok(vector.clone())
        );
        let ones = TensorType::f32(vec![1, 3, 1]);
        assert_eq!(
            infer("t", &transpose(vec![1, 2, 0]), &[&ones]),
            Ok(TensorType::f32(vec![3, 1, 1]))
        );
    }

    #[test]
    fn invalid_permutations_are_rejected() {
        let t = TensorType::f32(vec![2, 3]);
        let duplicate = infer("y", &transpose(vec![0, 0]), &[&t]).unwrap_err();
        assert_eq!(
            duplicate.to_string(),
            "node `y`: Transpose: [0, 0] is not a permutation of 0..2"
        );
        let out_of_range = infer("y", &transpose(vec![0, 2]), &[&t]).unwrap_err();
        assert_eq!(
            out_of_range.to_string(),
            "node `y`: Transpose: [0, 2] is not a permutation of 0..2"
        );
        let wrong_length = infer("y", &transpose(vec![0]), &[&t]).unwrap_err();
        assert_eq!(
            wrong_length.to_string(),
            "node `y`: Transpose: permutation [0] must have 2 entries"
        );
    }

    #[test]
    fn check_shape_accepts_exactly_the_rank_and_element_limits() {
        let limits = Limits::default();
        assert_eq!(check_shape("x", &[1; 8], &limits), Ok(1));
        assert_eq!(
            check_shape("x", &[limits.max_tensor_elements], &limits),
            Ok(limits.max_tensor_elements)
        );
        assert!(check_shape("x", &[limits.max_tensor_elements + 1], &limits).is_err());
    }
}
