//! Element types, tensor types and value identifiers.

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DType {
    F32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorType {
    pub dtype: DType,
    pub shape: Vec<usize>,
}

impl TensorType {
    pub fn f32(shape: Vec<usize>) -> Self {
        Self {
            dtype: DType::F32,
            shape,
        }
    }
}

/// Index of a value in a graph. Values are stored in definition order, so a
/// node's inputs always have smaller ids than the node itself.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ValueId(pub u32);

impl ValueId {
    pub fn index(self) -> usize {
        self.0 as usize
    }
}
