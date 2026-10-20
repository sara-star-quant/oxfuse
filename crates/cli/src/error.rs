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
        }
    }
}
