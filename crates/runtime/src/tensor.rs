//! Owned, contiguous, row-major `f32` tensors.

use oxfuse_core::Error;
use oxfuse_core::shape::checked_num_elements;

#[derive(Debug, Clone, PartialEq)]
pub struct Tensor {
    shape: Vec<usize>,
    data: Vec<f32>,
}

impl Tensor {
    pub fn new(shape: Vec<usize>, data: Vec<f32>) -> Result<Self, Error> {
        Self::named("tensor", shape, data)
    }

    /// Like `new`, but errors name the tensor, for diagnostics on user data.
    pub fn named(name: &str, shape: Vec<usize>, data: Vec<f32>) -> Result<Self, Error> {
        let expected = checked_num_elements(&shape).ok_or_else(|| Error::InvalidShape {
            name: name.to_owned(),
            shape: shape.clone(),
            reason: "element count overflows",
        })?;
        if data.len() != expected {
            return Err(Error::DataLength {
                name: name.to_owned(),
                shape,
                expected,
                found: data.len(),
            });
        }
        Ok(Self { shape, data })
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn data(&self) -> &[f32] {
        &self.data
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_accepts_matching_length() {
        let t = Tensor::new(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
        assert_eq!(t.data(), &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn scalar_has_one_value() {
        assert!(Tensor::new(vec![], vec![5.0]).is_ok());
        assert!(Tensor::new(vec![], vec![]).is_err());
    }

    #[test]
    fn named_error_carries_the_name() {
        let err = Tensor::named("w", vec![2], vec![1.0]).unwrap_err();
        assert_eq!(
            err,
            Error::DataLength {
                name: "w".into(),
                shape: vec![2],
                expected: 2,
                found: 1
            }
        );
    }

    #[test]
    fn overflowing_shape_is_an_error_not_a_panic() {
        let err = Tensor::new(vec![usize::MAX, 2], vec![]).unwrap_err();
        assert!(matches!(err, Error::InvalidShape { .. }));
    }
}
