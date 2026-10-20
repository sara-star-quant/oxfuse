//! The operation set. Adding an op means adding a variant here, then letting
//! the compiler list every exhaustive `match` that needs a new arm.

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Op {
    Relu,
    Sigmoid,
    Add,
    Mul,
    MatMul,
    Reshape { shape: Vec<usize> },
    Transpose { perm: Vec<usize> },
}

impl Op {
    /// Every op name the loader accepts, in declaration order. The unit test
    /// below fails when a variant is added without updating this list.
    pub const ALL_NAMES: [&'static str; 7] = [
        "Relu",
        "Sigmoid",
        "Add",
        "Mul",
        "MatMul",
        "Reshape",
        "Transpose",
    ];

    pub fn name(&self) -> &'static str {
        match self {
            Op::Relu => "Relu",
            Op::Sigmoid => "Sigmoid",
            Op::Add => "Add",
            Op::Mul => "Mul",
            Op::MatMul => "MatMul",
            Op::Reshape { .. } => "Reshape",
            Op::Transpose { .. } => "Transpose",
        }
    }

    pub fn arity(&self) -> usize {
        match self {
            Op::Relu | Op::Sigmoid | Op::Reshape { .. } | Op::Transpose { .. } => 1,
            Op::Add | Op::Mul | Op::MatMul => 2,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn one_of_each() -> Vec<Op> {
        vec![
            Op::Relu,
            Op::Sigmoid,
            Op::Add,
            Op::Mul,
            Op::MatMul,
            Op::Reshape { shape: vec![1] },
            Op::Transpose { perm: vec![0] },
        ]
    }

    /// No wildcard arm: adding a variant breaks the build here, which points
    /// at `one_of_each` and `ALL_NAMES` above.
    fn assert_listed(op: &Op) {
        match op {
            Op::Relu
            | Op::Sigmoid
            | Op::Add
            | Op::Mul
            | Op::MatMul
            | Op::Reshape { .. }
            | Op::Transpose { .. } => {}
        }
    }

    #[test]
    fn all_names_lists_every_variant_once() {
        let ops = one_of_each();
        ops.iter().for_each(assert_listed);
        let names: Vec<&str> = ops.iter().map(Op::name).collect();
        assert_eq!(names, Op::ALL_NAMES);
    }
}
