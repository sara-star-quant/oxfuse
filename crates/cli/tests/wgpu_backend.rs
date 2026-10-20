//! The wgpu backend is an optional feature: absent from the default build, and
//! when built in, selectable with `--backend wgpu` and measurable with
//! `gpu-selftest`. Without an adapter these tests accept the documented error.

use std::path::PathBuf;
use std::process::{Command, Output};

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn oxfuse(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_oxfuse"))
        .args(args)
        .current_dir(root())
        .output()
        .unwrap()
}

fn text(bytes: &[u8]) -> String {
    String::from_utf8_lossy(bytes).into_owned()
}

#[cfg(not(feature = "wgpu"))]
#[test]
fn the_default_build_has_no_wgpu_backend() {
    let out = oxfuse(&["run", "examples/relu.json", "--backend", "wgpu"]);
    assert!(!out.status.success());
    assert!(
        text(&out.stderr).contains("invalid value"),
        "{}",
        text(&out.stderr)
    );
    assert!(!text(&oxfuse(&["--help"]).stdout).contains("gpu-selftest"));
}

#[cfg(feature = "wgpu")]
mod enabled {
    use super::*;

    const RUN: [&str; 6] = [
        "run",
        "examples/tiny_mlp.json",
        "--inputs",
        "examples/tiny_mlp.inputs.json",
        "--backend",
        "wgpu",
    ];

    fn no_adapter(out: &Output) -> bool {
        !out.status.success() && text(&out.stderr).contains("no adapter")
    }

    #[test]
    fn the_wgpu_flag_is_listed_in_help() {
        assert!(text(&oxfuse(&["run", "--help"]).stdout).contains("wgpu"));
        assert!(text(&oxfuse(&["--help"]).stdout).contains("gpu-selftest"));
    }

    #[test]
    fn running_on_wgpu_prints_the_cpu_answer_for_the_shipped_example() {
        let out = oxfuse(&RUN);
        if no_adapter(&out) {
            return;
        }
        assert!(out.status.success(), "{}", text(&out.stderr));
        let cpu = oxfuse(&[
            "run",
            "examples/tiny_mlp.json",
            "--inputs",
            "examples/tiny_mlp.inputs.json",
        ]);
        assert_eq!(text(&out.stdout), text(&cpu.stdout));
    }

    #[test]
    fn gpu_selftest_prints_the_report() {
        let out = oxfuse(&["gpu-selftest"]);
        if no_adapter(&out) {
            return;
        }
        assert!(out.status.success(), "{}", text(&out.stderr));
        let report = text(&out.stdout);
        assert!(
            report.contains("adapter:") && report.contains("sigmoid max ulp from reference:"),
            "{report}"
        );
        let downlevel = oxfuse(&["gpu-selftest", "--limits", "downlevel"]);
        assert!(downlevel.status.success(), "{}", text(&downlevel.stderr));
    }
}
