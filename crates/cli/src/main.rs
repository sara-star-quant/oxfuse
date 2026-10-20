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
    Check { file: PathBuf },
    /// Execute a graph file and print its outputs as JSON
    Run {
        file: PathBuf,
        /// JSON file mapping each graph input name to {"shape": [...], "values": [...]}
        #[arg(long)]
        inputs: Option<PathBuf>,
        /// Skip every optimization pass
        #[arg(long)]
        no_opt: bool,
        /// Disable one pass by name (fold, cse, dce); repeatable
        #[arg(long = "disable-pass")]
        disable_pass: Vec<String>,
        /// Print the IR before and after each pass to stderr
        #[arg(long)]
        dump_ir: bool,
        /// Which backend executes the graph (both print the same bytes)
        #[arg(long, value_enum, default_value_t = BackendArg::Cpu)]
        backend: BackendArg,
    },
    /// Print the IR before and after each optimization pass
    Inspect {
        file: PathBuf,
        #[arg(long)]
        no_opt: bool,
        #[arg(long = "disable-pass")]
        disable_pass: Vec<String>,
    },
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    let result = match &cli.command {
        Command::Check { file } => oxfuse_cli::check(file)
            .map(|text| (text, String::new()))
            .map_err(|error| (error, String::new())),
        Command::Run {
            file,
            inputs,
            no_opt,
            disable_pass,
            dump_ir,
            backend,
        } => {
            let options = oxfuse_cli::RunOptions {
                backend: (*backend).into(),
                optimize: !no_opt,
                disabled: disable_pass.clone(),
                dump_ir: *dump_ir,
            };
            oxfuse_cli::run_with(file, inputs.as_deref(), &options)
                .map(|out| (out.stdout, out.stderr))
                .map_err(|failure| (failure.error, failure.stderr))
        }
        Command::Inspect {
            file,
            no_opt,
            disable_pass,
        } => {
            let options = oxfuse_cli::RunOptions {
                backend: oxfuse_cli::BackendKind::default(),
                optimize: !no_opt,
                disabled: disable_pass.clone(),
                dump_ir: false,
            };
            oxfuse_cli::inspect(file, &options)
                .map(|text| (text, String::new()))
                .map_err(|error| (error, String::new()))
        }
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
}

impl From<BackendArg> for oxfuse_cli::BackendKind {
    fn from(arg: BackendArg) -> Self {
        match arg {
            BackendArg::Reference => Self::Reference,
            BackendArg::Cpu => Self::Cpu,
        }
    }
}
