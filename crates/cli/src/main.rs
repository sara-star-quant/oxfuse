use clap::{Parser, Subcommand};
use std::io::{self, Write};
use std::path::PathBuf;
use std::process::ExitCode;

#[derive(Parser)]
#[command(name = "oxfuse", version, about = "Tensor graph compiler")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Validate a graph file and print its size
    Check {
        file: PathBuf,
        /// Value for a named dimension of an ONNX model, as NAME=VALUE; repeatable
        #[arg(long = "dim", value_parser = parse_dim)]
        dim: Vec<(String, usize)>,
    },
    /// Execute a graph file and print its outputs as JSON
    Run {
        file: PathBuf,
        /// JSON file mapping each graph input name to {"shape": [...], "values": [...]}
        #[arg(long)]
        inputs: Option<PathBuf>,
        /// Skip every optimization pass
        #[arg(long)]
        no_opt: bool,
        /// Disable one pass by name (fold, layout, cse, dce); repeatable
        #[arg(long = "disable-pass")]
        disable_pass: Vec<String>,
        /// Print the IR before and after each pass to stderr
        #[arg(long)]
        dump_ir: bool,
        /// Which backend executes the graph (reference and cpu print the same bytes)
        #[arg(long, value_enum, default_value_t = BackendArg::Cpu)]
        backend: BackendArg,
        /// Value for a named dimension of an ONNX model, as NAME=VALUE; repeatable
        #[arg(long = "dim", value_parser = parse_dim)]
        dim: Vec<(String, usize)>,
        /// Refuse a graph that needs more than this many multiply-adds
        #[arg(long)]
        max_macs: Option<usize>,
    },
    /// Measure the GPU adapter against the reference and print a report
    #[cfg(feature = "wgpu")]
    GpuSelftest {
        /// Create the device with the smallest limits every device guarantees
        #[arg(long, value_enum)]
        limits: Option<LimitsArg>,
    },
    /// Print the IR before and after each optimization pass
    Inspect {
        file: PathBuf,
        #[arg(long)]
        no_opt: bool,
        #[arg(long = "disable-pass")]
        disable_pass: Vec<String>,
        /// Value for a named dimension of an ONNX model, as NAME=VALUE; repeatable
        #[arg(long = "dim", value_parser = parse_dim)]
        dim: Vec<(String, usize)>,
    },
}

fn parse_dim(arg: &str) -> Result<(String, usize), String> {
    let bad =
        || format!("invalid --dim `{arg}`: expected NAME=VALUE with a non-negative integer VALUE");
    let (name, value) = arg.split_once('=').ok_or_else(bad)?;
    if name.is_empty() {
        return Err(bad());
    }
    Ok((name.to_owned(), value.parse().map_err(|_| bad())?))
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    let result = match &cli.command {
        Command::Check { file, dim } => oxfuse_cli::dims_from_args(dim)
            .and_then(|dims| oxfuse_cli::check_with(file, &dims))
            .map(|text| (text, String::new()))
            .map_err(|error| (error, String::new())),
        Command::Run {
            file,
            inputs,
            no_opt,
            disable_pass,
            dump_ir,
            backend,
            dim,
            max_macs,
        } => oxfuse_cli::dims_from_args(dim)
            .map_err(|error| (error, String::new()))
            .and_then(|dims| {
                let options = oxfuse_cli::RunOptions {
                    backend: (*backend).into(),
                    optimize: !no_opt,
                    disabled: disable_pass.clone(),
                    dump_ir: *dump_ir,
                    dims,
                    max_macs: *max_macs,
                };
                oxfuse_cli::run_with(file, inputs.as_deref(), &options)
                    .map(|out| (out.stdout, out.stderr))
                    .map_err(|failure| (failure.error, failure.stderr))
            }),
        #[cfg(feature = "wgpu")]
        Command::GpuSelftest { limits } => oxfuse_cli::gpu_selftest(limits.is_some())
            .map(|text| (text, String::new()))
            .map_err(|error| (error, String::new())),
        Command::Inspect {
            file,
            no_opt,
            disable_pass,
            dim,
        } => oxfuse_cli::dims_from_args(dim)
            .and_then(|dims| {
                let options = oxfuse_cli::RunOptions {
                    backend: oxfuse_cli::BackendKind::default(),
                    optimize: !no_opt,
                    disabled: disable_pass.clone(),
                    dump_ir: false,
                    dims,
                    max_macs: None,
                };
                oxfuse_cli::inspect(file, &options)
            })
            .map(|text| (text, String::new()))
            .map_err(|error| (error, String::new())),
    };
    match result {
        Ok((stdout, stderr)) => exit_after_writing(emit(&stdout, &stderr)),
        Err((error, stderr)) => {
            let mut err = io::stderr();
            let _ = err.write_all(stderr.as_bytes());
            let _ = writeln!(err, "error[{}]: {error}", error.code());
            ExitCode::FAILURE
        }
    }
}

/// Never `print!`/`eprint!`: they panic on a closed pipe.
fn emit(stdout: &str, stderr: &str) -> io::Result<()> {
    io::stderr().write_all(stderr.as_bytes())?;
    let mut out = io::stdout().lock();
    out.write_all(stdout.as_bytes())?;
    out.flush()
}

/// The reader going away (`... | head`) is normal shell use, not an error.
/// Any other failure to write the output is.
fn exit_after_writing(written: io::Result<()>) -> ExitCode {
    match written {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) if e.kind() == io::ErrorKind::BrokenPipe => ExitCode::SUCCESS,
        Err(e) => {
            let _ = writeln!(io::stderr(), "error: cannot write output: {e}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_complete_write_succeeds() {
        assert_eq!(exit_after_writing(Ok(())), ExitCode::SUCCESS);
    }

    #[test]
    fn a_closed_pipe_is_quiet_and_successful() {
        let closed = io::Error::from(io::ErrorKind::BrokenPipe);
        assert_eq!(exit_after_writing(Err(closed)), ExitCode::SUCCESS);
    }

    #[test]
    fn each_backend_argument_selects_its_own_backend() {
        use oxfuse_cli::BackendKind;
        // Reference and cpu print the same bytes, so only a direct check can tell them apart.
        assert_eq!(
            BackendKind::from(BackendArg::Reference),
            BackendKind::Reference
        );
        assert_eq!(BackendKind::from(BackendArg::Cpu), BackendKind::Cpu);
        #[cfg(feature = "wgpu")]
        assert_eq!(BackendKind::from(BackendArg::Wgpu), BackendKind::Wgpu);
    }

    #[test]
    fn any_other_write_error_fails() {
        for kind in [io::ErrorKind::PermissionDenied, io::ErrorKind::WriteZero] {
            assert_eq!(
                exit_after_writing(Err(io::Error::from(kind))),
                ExitCode::FAILURE,
                "{kind:?}"
            );
        }
    }
}

#[derive(Clone, Copy, clap::ValueEnum)]
enum BackendArg {
    Reference,
    Cpu,
    #[cfg(feature = "wgpu")]
    Wgpu,
}

#[cfg(feature = "wgpu")]
#[derive(Clone, Copy, clap::ValueEnum)]
enum LimitsArg {
    Downlevel,
}

impl From<BackendArg> for oxfuse_cli::BackendKind {
    fn from(arg: BackendArg) -> Self {
        match arg {
            BackendArg::Reference => Self::Reference,
            BackendArg::Cpu => Self::Cpu,
            #[cfg(feature = "wgpu")]
            BackendArg::Wgpu => Self::Wgpu,
        }
    }
}
