//! Everything that can go wrong in the CLI, with stable codes.

use oxfuse_core::Error;

#[derive(Debug, thiserror::Error)]
pub enum CliError {
    #[error(transparent)]
    Core(#[from] Error),
    /// A core error plus the place in the graph file it comes from.
    #[error("{source} (at {path})")]
    Located { path: String, source: Box<Error> },
    #[error("invalid graph JSON: {0}")]
    GraphJson(#[from] serde_json::Error),
    #[error("cannot read `{path}`: {source}")]
    Io {
        path: String,
        source: std::io::Error,
    },
    #[error("invalid inputs file `{path}`: {source}")]
    InputsJson {
        path: String,
        source: serde_json::Error,
    },
    #[error("graph declares {count} input(s); pass --inputs FILE")]
    MissingInputs { count: usize },
    #[error("inputs file has no entry for graph input `{name}`")]
    MissingInput { name: String },
    #[error("inputs file has entry `{name}` that is not a graph input")]
    UnknownInput { name: String },
    #[error("unknown pass `{name}` (known: {known})")]
    UnknownPass { name: String, known: String },
    #[cfg(feature = "onnx")]
    #[error(transparent)]
    Onnx(#[from] oxfuse_onnx::OnnxError),
    #[error("`{path}` is an ONNX model; this build cannot read it (rebuild with --features onnx)")]
    OnnxDisabled { path: String },
    #[error("--dim `{name}` is given more than once")]
    DuplicateDim { name: String },
    #[error("outputs differ: {verdict}")]
    CompareDiffers { verdict: &'static str },
    #[error("cannot compare: {reason}")]
    CompareOutputs { reason: String },
    #[error("run record `{path}`: {reason}")]
    CompareManifest { path: String, reason: String },
    #[error("invalid outputs file `{path}`: {reason}")]
    CompareFile { path: String, reason: String },
    #[error("graph needs about {estimated} multiply-adds, over --max-macs {limit}")]
    TooMuchWork { estimated: usize, limit: usize },
}

impl CliError {
    pub fn code(&self) -> &'static str {
        match self {
            CliError::Core(e) => e.code(),
            CliError::Located { source, .. } => source.code(),
            CliError::GraphJson(_) => "E100",
            CliError::Io { .. } => "E101",
            CliError::InputsJson { .. } => "E102",
            CliError::MissingInputs { .. } => "E103",
            CliError::MissingInput { .. } => "E104",
            CliError::UnknownInput { .. } => "E105",
            CliError::UnknownPass { .. } => "E106",
            #[cfg(feature = "onnx")]
            CliError::Onnx(e) => e.code(),
            CliError::OnnxDisabled { .. } => "E107",
            CliError::DuplicateDim { .. } => "E108",
            CliError::TooMuchWork { .. } => "E109",
            CliError::CompareDiffers { .. } => "E110",
            CliError::CompareOutputs { .. } => "E111",
            CliError::CompareManifest { .. } => "E112",
            CliError::CompareFile { .. } => "E113",
        }
    }
}
