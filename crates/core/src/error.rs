//! One error type for the whole compiler core. Every variant has a stable
//! code so tests and users can match on it without parsing messages.

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    #[error("unsupported format_version {found} (supported: {supported})")]
    UnsupportedVersion { found: u32, supported: u32 },
    #[error("duplicate name `{name}`")]
    DuplicateName { name: String },
    #[error("node `{node}` references unknown value `{name}`")]
    UnknownReference { node: String, name: String },
    #[error("output `{name}` does not exist")]
    UnknownOutput { name: String },
    #[error("graph contains a cycle through node `{node}`")]
    Cycle { node: String },
    #[error("node `{node}`: {op} takes {expected} input(s), got {found}")]
    Arity {
        node: String,
        op: &'static str,
        expected: usize,
        found: usize,
    },
    #[error("node `{node}`: unsupported op `{op}`")]
    UnsupportedOp { node: String, op: String },
    #[error("`{name}`: shape {shape:?} needs {expected} value(s), got {found}")]
    DataLength {
        name: String,
        shape: Vec<usize>,
        expected: usize,
        found: usize,
    },
    #[error("`{name}`: invalid shape {shape:?}: {reason}")]
    InvalidShape {
        name: String,
        shape: Vec<usize>,
        reason: &'static str,
    },
    #[error("limit exceeded: {what} is {found}, maximum is {limit}")]
    LimitExceeded {
        what: &'static str,
        found: usize,
        limit: usize,
    },
    #[error("`{name}`: {what} is {found}, maximum is {limit}")]
    TensorLimit {
        name: String,
        what: &'static str,
        found: usize,
        limit: usize,
    },
    #[error("graph declares no outputs")]
    NoOutputs,
    #[error("`{name}` contains a non-finite value (NaN or infinity), which JSON cannot represent")]
    NonFinite { name: String },
    #[error("expected {expected} input tensor(s), got {found}")]
    InputCount { expected: usize, found: usize },
    #[error("input `{name}`: expected shape {expected:?}, got {found:?}")]
    InputShape {
        name: String,
        expected: Vec<usize>,
        found: Vec<usize>,
    },
    #[error("node `{node}`: {reason}")]
    ShapeMismatch { node: String, reason: String },
    #[error("node `{node}`: {op} requires attribute `{attr}`")]
    MissingAttribute {
        node: String,
        op: &'static str,
        attr: &'static str,
    },
    #[error("node `{node}`: {op} does not take attribute `{attr}`")]
    UnexpectedAttribute {
        node: String,
        op: &'static str,
        attr: &'static str,
    },
    #[error("node `{node}`: {op} attribute `{attr}`: {reason}")]
    InvalidAttribute {
        node: String,
        op: &'static str,
        attr: &'static str,
        reason: String,
    },
    #[error("invalid graph: {0}")]
    Verify(String),
    #[error("backend error: {0}")]
    Backend(String),
}

impl Error {
    pub fn code(&self) -> &'static str {
        match self {
            Error::UnsupportedVersion { .. } => "E001",
            Error::DuplicateName { .. } => "E002",
            Error::UnknownReference { .. } => "E003",
            Error::UnknownOutput { .. } => "E004",
            Error::Cycle { .. } => "E005",
            Error::Arity { .. } => "E006",
            Error::UnsupportedOp { .. } => "E007",
            Error::DataLength { .. } => "E008",
            Error::InvalidShape { .. } => "E009",
            Error::LimitExceeded { .. } => "E010",
            Error::NoOutputs => "E011",
            Error::NonFinite { .. } => "E012",
            Error::InputCount { .. } => "E013",
            Error::InputShape { .. } => "E014",
            Error::Verify(_) => "E015",
            Error::TensorLimit { .. } => "E016",
            Error::ShapeMismatch { .. } => "E017",
            Error::MissingAttribute { .. } => "E018",
            Error::UnexpectedAttribute { .. } => "E019",
            Error::InvalidAttribute { .. } => "E020",
            Error::Backend(_) => "E021",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    fn one_of_each() -> Vec<Error> {
        vec![
            Error::UnsupportedVersion {
                found: 2,
                supported: 1,
            },
            Error::DuplicateName { name: "a".into() },
            Error::UnknownReference {
                node: "n".into(),
                name: "a".into(),
            },
            Error::UnknownOutput { name: "a".into() },
            Error::Cycle { node: "a".into() },
            Error::Arity {
                node: "n".into(),
                op: "Relu",
                expected: 1,
                found: 2,
            },
            Error::UnsupportedOp {
                node: "n".into(),
                op: "Gelu".into(),
            },
            Error::DataLength {
                name: "c".into(),
                shape: vec![2],
                expected: 2,
                found: 1,
            },
            Error::InvalidShape {
                name: "x".into(),
                shape: vec![0],
                reason: "dimensions must be at least 1",
            },
            Error::LimitExceeded {
                what: "values",
                found: 3,
                limit: 2,
            },
            Error::NoOutputs,
            Error::NonFinite { name: "c".into() },
            Error::InputCount {
                expected: 1,
                found: 0,
            },
            Error::InputShape {
                name: "x".into(),
                expected: vec![2],
                found: vec![3],
            },
            Error::TensorLimit {
                name: "x".into(),
                what: "tensor rank",
                found: 9,
                limit: 8,
            },
            Error::ShapeMismatch {
                node: "n".into(),
                reason: "r".into(),
            },
            Error::MissingAttribute {
                node: "n".into(),
                op: "Reshape",
                attr: "shape",
            },
            Error::UnexpectedAttribute {
                node: "n".into(),
                op: "Relu",
                attr: "shape",
            },
            Error::InvalidAttribute {
                node: "n".into(),
                op: "Reshape",
                attr: "shape",
                reason: "r".into(),
            },
            Error::Verify("bad".into()),
            Error::Backend("no adapter".into()),
        ]
    }

    #[test]
    fn codes_are_unique_and_stable_format() {
        let errors = one_of_each();
        let codes: HashSet<&str> = errors.iter().map(Error::code).collect();
        assert_eq!(codes.len(), errors.len());
        assert!(codes.iter().all(|c| c.len() == 4 && c.starts_with('E')));
    }

    #[test]
    fn backend_errors_carry_their_message_and_a_stable_code() {
        let e = Error::Backend("no adapter found".into());
        assert_eq!(e.to_string(), "backend error: no adapter found");
        assert_eq!(e.code(), "E021");
    }

    #[test]
    fn messages_name_the_offender() {
        assert_eq!(
            Error::Cycle { node: "a".into() }.to_string(),
            "graph contains a cycle through node `a`"
        );
        assert_eq!(
            Error::UnknownReference {
                node: "n".into(),
                name: "q".into()
            }
            .to_string(),
            "node `n` references unknown value `q`"
        );
    }
}
