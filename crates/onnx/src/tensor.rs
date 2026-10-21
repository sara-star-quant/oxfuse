//! `TensorProto` to dense data, with every size checked before any copy.

use crate::{
    OnnxError,
    proto::{TensorProto, data_type},
};
use oxfuse_core::Limits;

/// A decoded tensor: float32 data, or int64 data (shape operands only).
#[derive(Debug, Clone, PartialEq)]
pub enum Dense {
    F32 { dims: Vec<usize>, data: Vec<f32> },
    I64 { dims: Vec<usize>, data: Vec<i64> },
}

pub fn read_tensor(t: &TensorProto, limits: &Limits) -> Result<Dense, OnnxError> {
    let bad = |reason: String| OnnxError::Tensor {
        name: t.name.clone(),
        reason,
    };
    if !t.external_data.is_empty() || t.data_location != 0 {
        return Err(bad("external data is not supported".into()));
    }
    if t.segment.is_some() {
        return Err(bad("segments are not supported".into()));
    }
    let kind = match t.data_type {
        data_type::FLOAT => Kind::F32,
        data_type::INT64 => Kind::I64,
        other => {
            return Err(OnnxError::Dtype {
                what: t.name.clone(),
                dtype: dtype_name(other),
            });
        }
    };
    let (dims, count) = read_dims(t, limits).map_err(bad)?;
    match kind {
        Kind::F32 => {
            if !(t.int32_data.is_empty()
                && t.int64_data.is_empty()
                && t.double_data.is_empty()
                && t.uint64_data.is_empty()
                && t.string_data.is_empty())
            {
                return Err(bad("float data is stored in another typed list".into()));
            }
            let data = pick(&t.raw_data, &t.float_data, count, 4, f32_from_le).map_err(bad)?;
            Ok(Dense::F32 { dims, data })
        }
        Kind::I64 => {
            if !(t.int32_data.is_empty()
                && t.float_data.is_empty()
                && t.double_data.is_empty()
                && t.uint64_data.is_empty()
                && t.string_data.is_empty())
            {
                return Err(bad("int64 data is stored in another typed list".into()));
            }
            let data = pick(&t.raw_data, &t.int64_data, count, 8, i64_from_le).map_err(bad)?;
            Ok(Dense::I64 { dims, data })
        }
    }
}

enum Kind {
    F32,
    I64,
}

fn f32_from_le(b: &[u8]) -> f32 {
    f32::from_le_bytes(b.try_into().expect("chunks_exact(4)"))
}

fn i64_from_le(b: &[u8]) -> i64 {
    i64::from_le_bytes(b.try_into().expect("chunks_exact(8)"))
}

/// Validated dims and their element count. Every step is checked against
/// the tensor cap, so no product can wrap and nothing is allocated for an
/// oversize shape.
fn read_dims(t: &TensorProto, limits: &Limits) -> Result<(Vec<usize>, usize), String> {
    let mut dims = Vec::with_capacity(t.dims.len().min(limits.max_rank + 1));
    let mut count: u64 = 1;
    for &d in &t.dims {
        let d = u64::try_from(d).map_err(|_| format!("negative dimension {d}"))?;
        count = count
            .checked_mul(d)
            .filter(|c| *c <= limits.max_tensor_elements as u64)
            .ok_or_else(|| {
                format!(
                    "shape {:?} exceeds the limit of {} elements",
                    t.dims, limits.max_tensor_elements
                )
            })?;
        dims.push(usize::try_from(d).map_err(|_| "dimension does not fit usize".to_owned())?);
    }
    Ok((dims, count as usize))
}

/// Take the data from `raw` (little endian, `width` bytes per element) or
/// from the typed list, never both, and check its length against `count`.
fn pick<T: Copy, S: Copy + Into<T>>(
    raw: &[u8],
    typed: &[S],
    count: usize,
    width: usize,
    from_le: fn(&[u8]) -> T,
) -> Result<Vec<T>, String> {
    match (raw.is_empty(), typed.is_empty()) {
        (false, false) => Err("both raw_data and a typed list are set".into()),
        (false, true) => {
            if count.checked_mul(width) != Some(raw.len()) {
                Err(format!(
                    "raw_data has {} bytes, expected {count} elements at {width} bytes per element",
                    raw.len()
                ))
            } else {
                Ok(raw.chunks_exact(width).map(from_le).collect())
            }
        }
        (true, false) => {
            if typed.len() == count {
                Ok(typed.iter().map(|v| (*v).into()).collect())
            } else {
                Err(format!(
                    "typed list has {} values, shape needs {count}",
                    typed.len()
                ))
            }
        }
        (true, true) => {
            if count == 0 {
                Ok(Vec::new())
            } else {
                Err(format!("no data for {count} elements"))
            }
        }
    }
}

pub(crate) fn dtype_name(code: i32) -> String {
    match code {
        0 => "UNDEFINED",
        1 => "FLOAT",
        2 => "UINT8",
        3 => "INT8",
        4 => "UINT16",
        5 => "INT16",
        6 => "INT32",
        7 => "INT64",
        8 => "STRING",
        9 => "BOOL",
        10 => "FLOAT16",
        11 => "DOUBLE",
        12 => "UINT32",
        13 => "UINT64",
        14 => "COMPLEX64",
        15 => "COMPLEX128",
        16 => "BFLOAT16",
        n => return format!("type {n}"),
    }
    .to_owned()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proto::{Segment, StringStringEntryProto, data_type};

    fn limits() -> Limits {
        Limits::default()
    }

    fn base(dims: &[i64], data_type: i32) -> TensorProto {
        TensorProto {
            name: "w".into(),
            dims: dims.to_vec(),
            data_type,
            ..TensorProto::default()
        }
    }

    fn raw_f32(values: &[f32]) -> Vec<u8> {
        values.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    fn reason(r: Result<Dense, OnnxError>) -> String {
        match r {
            Err(OnnxError::Tensor { name, reason }) => {
                assert_eq!(name, "w");
                reason
            }
            other => panic!("expected a tensor error, got {other:?}"),
        }
    }

    #[test]
    fn raw_data_and_float_data_are_equivalent() {
        let mut a = base(&[2, 2], data_type::FLOAT);
        a.raw_data = raw_f32(&[1.0, -2.5, 0.0, 7.0]);
        let mut b = base(&[2, 2], data_type::FLOAT);
        b.float_data = vec![1.0, -2.5, 0.0, 7.0];
        let want = Dense::F32 {
            dims: vec![2, 2],
            data: vec![1.0, -2.5, 0.0, 7.0],
        };
        assert_eq!(read_tensor(&a, &limits()).unwrap(), want);
        assert_eq!(read_tensor(&b, &limits()).unwrap(), want);
    }

    #[test]
    fn raw_data_is_little_endian() {
        let mut t = base(&[1], data_type::FLOAT);
        t.raw_data = vec![0x00, 0x00, 0x80, 0x3f];
        let r = read_tensor(&t, &limits()).unwrap();
        assert_eq!(
            r,
            Dense::F32 {
                dims: vec![1],
                data: vec![1.0]
            }
        );
    }

    #[test]
    fn scalar_has_rank_zero() {
        let mut t = base(&[], data_type::FLOAT);
        t.float_data = vec![3.0];
        assert_eq!(
            read_tensor(&t, &limits()).unwrap(),
            Dense::F32 {
                dims: vec![],
                data: vec![3.0]
            }
        );
    }

    #[test]
    fn both_present_is_an_error() {
        let mut t = base(&[1], data_type::FLOAT);
        t.raw_data = raw_f32(&[1.0]);
        t.float_data = vec![1.0];
        assert!(reason(read_tensor(&t, &limits())).contains("both"));
    }

    #[test]
    fn neither_present_is_an_error() {
        let t = base(&[2], data_type::FLOAT);
        assert!(reason(read_tensor(&t, &limits())).contains("no data"));
    }

    #[test]
    fn empty_tensor_needs_no_data() {
        let t = base(&[0, 3], data_type::FLOAT);
        assert_eq!(
            read_tensor(&t, &limits()).unwrap(),
            Dense::F32 {
                dims: vec![0, 3],
                data: vec![]
            }
        );
    }

    #[test]
    fn data_length_mismatch() {
        let mut raw = base(&[2], data_type::FLOAT);
        raw.raw_data = raw_f32(&[1.0]);
        assert!(reason(read_tensor(&raw, &limits())).contains("4 bytes per element"));
        let mut typed = base(&[2], data_type::FLOAT);
        typed.float_data = vec![1.0, 2.0, 3.0];
        assert!(reason(read_tensor(&typed, &limits())).contains("needs 2"));
        let mut odd = base(&[1], data_type::FLOAT);
        odd.raw_data = vec![0, 0, 0];
        assert!(reason(read_tensor(&odd, &limits())).contains("4 bytes per element"));
    }

    #[test]
    fn data_for_an_empty_tensor_is_a_mismatch() {
        let mut t = base(&[0], data_type::FLOAT);
        t.float_data = vec![1.0];
        reason(read_tensor(&t, &limits()));
    }

    #[test]
    fn negative_dim_is_an_error() {
        let t = base(&[2, -1], data_type::FLOAT);
        assert!(reason(read_tensor(&t, &limits())).contains("negative"));
    }

    #[test]
    fn dims_product_overflow_is_an_error() {
        let t = base(&[1 << 40, 1 << 40, 1 << 40], data_type::FLOAT);
        reason(read_tensor(&t, &limits()));
        let t = base(&[i64::MAX, i64::MAX], data_type::FLOAT);
        reason(read_tensor(&t, &limits()));
    }

    #[test]
    fn dims_over_the_limit_are_rejected_before_any_copy() {
        let small = Limits {
            max_tensor_elements: 16,
            ..Limits::default()
        };
        let mut t = base(&[17], data_type::FLOAT);
        t.float_data = vec![0.0; 17];
        assert!(reason(read_tensor(&t, &small)).contains("limit of 16"));
        let t = base(&[4, 4], data_type::FLOAT);
        // At the limit the size check passes (then data is missing).
        assert!(reason(read_tensor(&t, &small)).contains("no data"));
    }

    #[test]
    fn non_float_dtype_names_the_type() {
        match read_tensor(&base(&[1], 11), &limits()) {
            Err(OnnxError::Dtype { what, dtype }) => {
                assert_eq!(what, "w");
                assert_eq!(dtype, "DOUBLE");
            }
            other => panic!("{other:?}"),
        }
        match read_tensor(&base(&[1], 0), &limits()) {
            Err(OnnxError::Dtype { dtype, .. }) => assert_eq!(dtype, "UNDEFINED"),
            other => panic!("{other:?}"),
        }
        match read_tensor(&base(&[1], 99), &limits()) {
            Err(OnnxError::Dtype { dtype, .. }) => assert_eq!(dtype, "type 99"),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn float_tensor_with_data_in_another_list_is_an_error() {
        let mut t = base(&[1], data_type::FLOAT);
        t.double_data = vec![1.0];
        assert!(reason(read_tensor(&t, &limits())).contains("another"));
        let mut t = base(&[1], data_type::FLOAT);
        t.int32_data = vec![1];
        reason(read_tensor(&t, &limits()));
    }

    #[test]
    fn int64_tensor_is_read_as_i64() {
        let mut a = base(&[2], data_type::INT64);
        a.raw_data = [-1i64, 5].iter().flat_map(|v| v.to_le_bytes()).collect();
        let mut b = base(&[2], data_type::INT64);
        b.int64_data = vec![-1, 5];
        let want = Dense::I64 {
            dims: vec![2],
            data: vec![-1, 5],
        };
        assert_eq!(read_tensor(&a, &limits()).unwrap(), want);
        assert_eq!(read_tensor(&b, &limits()).unwrap(), want);
    }

    #[test]
    fn int64_length_mismatch_and_foreign_lists() {
        let mut t = base(&[2], data_type::INT64);
        t.int64_data = vec![1];
        reason(read_tensor(&t, &limits()));
        let mut t = base(&[1], data_type::INT64);
        t.raw_data = vec![0; 4];
        reason(read_tensor(&t, &limits()));
        let mut t = base(&[1], data_type::INT64);
        t.float_data = vec![1.0];
        reason(read_tensor(&t, &limits()));
    }

    #[test]
    fn int64_tensor_with_data_in_any_other_list_is_an_error() {
        let others: [fn(&mut TensorProto); 5] = [
            |t| t.int32_data = vec![1],
            |t| t.float_data = vec![1.0],
            |t| t.double_data = vec![1.0],
            |t| t.uint64_data = vec![1],
            |t| t.string_data = vec![vec![1]],
        ];
        for (i, set) in others.iter().enumerate() {
            let mut t = base(&[1], data_type::INT64);
            t.int64_data = vec![1];
            set(&mut t);
            assert!(
                reason(read_tensor(&t, &limits())).contains("another"),
                "list {i}"
            );
        }
    }

    #[test]
    fn float_tensor_with_data_in_any_other_list_is_an_error() {
        let others: [fn(&mut TensorProto); 5] = [
            |t| t.int32_data = vec![1],
            |t| t.int64_data = vec![1],
            |t| t.double_data = vec![1.0],
            |t| t.uint64_data = vec![1],
            |t| t.string_data = vec![vec![1]],
        ];
        for (i, set) in others.iter().enumerate() {
            let mut t = base(&[1], data_type::FLOAT);
            t.float_data = vec![1.0];
            set(&mut t);
            assert!(
                reason(read_tensor(&t, &limits())).contains("another"),
                "list {i}"
            );
        }
    }

    #[test]
    fn external_data_is_rejected() {
        let mut t = base(&[1], data_type::FLOAT);
        t.external_data = vec![StringStringEntryProto {
            key: "location".into(),
            value: "w.bin".into(),
        }];
        assert!(reason(read_tensor(&t, &limits())).contains("external"));
        let mut t = base(&[1], data_type::FLOAT);
        t.data_location = 1;
        assert!(reason(read_tensor(&t, &limits())).contains("external"));
    }

    #[test]
    fn segment_is_rejected() {
        let mut t = base(&[1], data_type::FLOAT);
        t.float_data = vec![1.0];
        t.segment = Some(Segment { begin: 0, end: 1 });
        assert!(reason(read_tensor(&t, &limits())).contains("segment"));
    }
}
