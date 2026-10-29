//! Everything that can go wrong while importing a model, with stable codes.

use oxfuse_core::Error;

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum OnnxError {
    #[error("cannot decode the model: {reason}")]
    Decode { reason: String },
    #[error("model is {found} bytes, maximum is {limit}")]
    TooLarge { found: usize, limit: usize },
    #[error("model nests messages too deeply")]
    TooDeep,
    #[error("model has more than {limit} messages")]
    TooManyMessages { limit: usize },
    #[error("unsupported opset {found}: re-export with opset 13 to 21")]
    Opset { found: String },
    #[error("node `{node}`: unsupported operator domain `{domain}`")]
    Domain { node: String, domain: String },
    #[error("node `{node}`: unsupported op `{op}`")]
    UnsupportedOp { node: String, op: String },
    #[error("node `{node}`: {op} does not support attribute `{attr}`")]
    UnsupportedAttr {
        node: String,
        op: String,
        attr: String,
    },
    #[error("`{what}`: unsupported element type {dtype}; only float32 is supported")]
    Dtype { what: String, dtype: String },
    #[error("tensor `{name}`: {reason}")]
    Tensor { name: String, reason: String },
    #[error("invalid model structure: {reason}")]
    Structure { reason: String },
    #[error("invalid value name: {reason}")]
    Name { reason: String },
    #[error("node `{node}`: {reason}")]
    Shape { node: String, reason: String },
    #[error("dimension `{name}` is not fixed; pass --dim {name}=VALUE")]
    DimUnbound { name: String },
    #[error("node `{node}`: {what} is {found}, maximum is {limit}")]
    Limit {
        node: String,
        what: &'static str,
        found: usize,
        limit: usize,
    },
    #[error(transparent)]
    Core(#[from] Error),
}

impl OnnxError {
    pub fn code(&self) -> &'static str {
        match self {
            OnnxError::Decode { .. } => "E300",
            OnnxError::TooLarge { .. } => "E301",
            OnnxError::TooDeep => "E302",
            OnnxError::TooManyMessages { .. } => "E314",
            OnnxError::Opset { .. } => "E303",
            OnnxError::Domain { .. } => "E304",
            OnnxError::UnsupportedOp { .. } => "E305",
            OnnxError::UnsupportedAttr { .. } => "E306",
            OnnxError::Dtype { .. } => "E307",
            OnnxError::Tensor { .. } => "E308",
            OnnxError::Structure { .. } => "E309",
            OnnxError::Name { .. } => "E310",
            OnnxError::Shape { .. } => "E311",
            OnnxError::DimUnbound { .. } => "E312",
            OnnxError::Limit { .. } => "E313",
            OnnxError::Core(e) => e.code(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn one_of_each() -> Vec<OnnxError> {
        let s = || "x".to_owned();
        vec![
            OnnxError::Decode { reason: s() },
            OnnxError::TooLarge { found: 2, limit: 1 },
            OnnxError::TooDeep,
            OnnxError::Opset { found: s() },
            OnnxError::Domain {
                node: s(),
                domain: s(),
            },
            OnnxError::UnsupportedOp { node: s(), op: s() },
            OnnxError::UnsupportedAttr {
                node: s(),
                op: s(),
                attr: s(),
            },
            OnnxError::Dtype {
                what: s(),
                dtype: s(),
            },
            OnnxError::Tensor {
                name: s(),
                reason: s(),
            },
            OnnxError::Structure { reason: s() },
            OnnxError::Name { reason: s() },
            OnnxError::Shape {
                node: s(),
                reason: s(),
            },
            OnnxError::DimUnbound { name: s() },
            OnnxError::Limit {
                node: s(),
                what: "w",
                found: 2,
                limit: 1,
            },
            OnnxError::TooManyMessages { limit: 1 },
        ]
    }

    #[test]
    fn codes_are_unique_and_stable() {
        let codes: Vec<_> = one_of_each().iter().map(OnnxError::code).collect();
        let expected: Vec<String> = (300..315).map(|n| format!("E{n}")).collect();
        assert_eq!(codes, expected);
    }

    #[test]
    fn core_errors_keep_their_code() {
        let e = OnnxError::from(Error::NoOutputs);
        assert_eq!(e.code(), Error::NoOutputs.code());
    }

    #[test]
    fn messages_are_ascii() {
        for e in one_of_each() {
            assert!(e.to_string().is_ascii(), "{e}");
        }
    }
}
