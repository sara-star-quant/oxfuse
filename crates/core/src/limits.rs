//! Resource limits applied to untrusted input.

#[derive(Debug, Clone)]
pub struct Limits {
    pub max_file_bytes: usize,
    /// Cap on graph values: inputs, constants and nodes together.
    pub max_values: usize,
    pub max_tensor_elements: usize,
    /// Cap on the elements of all values together, so many legal tensors
    /// cannot add up to an allocation the per-tensor cap would have refused.
    pub max_total_elements: usize,
    pub max_rank: usize,
}

impl Default for Limits {
    fn default() -> Self {
        Self {
            max_file_bytes: 64 * 1024 * 1024,
            max_values: 100_000,
            max_tensor_elements: 1 << 28,
            max_total_elements: 1 << 29,
            max_rank: 8,
        }
    }
}
