//! Resource limits applied to untrusted input.

/// Limits on a graph. The struct can grow, so build one from
/// `Limits::default()` and the `with_*` methods (or assign a field of a value
/// you own); a struct literal is not available outside this crate.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct Limits {
    pub max_file_bytes: usize,
    /// Cap on graph values: inputs, constants and nodes together.
    pub max_values: usize,
    pub max_tensor_elements: usize,
    /// Cap on the elements of all values together, so many legal tensors
    /// cannot add up to an allocation the per-tensor cap would have refused.
    pub max_total_elements: usize,
    pub max_rank: usize,
    /// Cap on `Graph::estimated_macs`, checked when the graph is finished,
    /// before any pass or backend runs it. No cap by default.
    pub max_macs: usize,
}

impl Default for Limits {
    fn default() -> Self {
        Self {
            max_file_bytes: 64 * 1024 * 1024,
            max_values: 100_000,
            max_tensor_elements: 1 << 28,
            max_total_elements: 1 << 29,
            max_rank: 8,
            max_macs: usize::MAX,
        }
    }
}

impl Limits {
    pub fn with_max_file_bytes(mut self, value: usize) -> Self {
        self.max_file_bytes = value;
        self
    }

    pub fn with_max_values(mut self, value: usize) -> Self {
        self.max_values = value;
        self
    }

    pub fn with_max_tensor_elements(mut self, value: usize) -> Self {
        self.max_tensor_elements = value;
        self
    }

    pub fn with_max_total_elements(mut self, value: usize) -> Self {
        self.max_total_elements = value;
        self
    }

    pub fn with_max_rank(mut self, value: usize) -> Self {
        self.max_rank = value;
        self
    }

    pub fn with_max_macs(mut self, value: usize) -> Self {
        self.max_macs = value;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_defaults_are_pinned() {
        let l = Limits::default();
        assert_eq!(l.max_file_bytes, 64 * 1024 * 1024);
        assert_eq!(l.max_values, 100_000);
        assert_eq!(l.max_tensor_elements, 1 << 28);
        assert_eq!(l.max_total_elements, 1 << 29);
        assert_eq!(l.max_rank, 8);
        assert_eq!(l.max_macs, usize::MAX);
    }

    #[test]
    fn each_with_method_sets_its_own_field_and_no_other() {
        let base = Limits::default();
        let l = base.clone().with_max_file_bytes(1);
        assert_eq!((l.max_file_bytes, l.max_values), (1, base.max_values));
        let l = base.clone().with_max_values(2);
        assert_eq!((l.max_values, l.max_file_bytes), (2, base.max_file_bytes));
        let l = base.clone().with_max_tensor_elements(3);
        assert_eq!(
            (l.max_tensor_elements, l.max_total_elements),
            (3, base.max_total_elements)
        );
        let l = base.clone().with_max_total_elements(4);
        assert_eq!(
            (l.max_total_elements, l.max_tensor_elements),
            (4, base.max_tensor_elements)
        );
        let l = base.clone().with_max_rank(5);
        assert_eq!((l.max_rank, l.max_macs), (5, usize::MAX));
        let l = base.with_max_macs(6);
        assert_eq!((l.max_macs, l.max_rank), (6, 8));
    }
}
