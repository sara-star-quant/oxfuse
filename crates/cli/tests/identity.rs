//! The project name is part of the interface: the binary, its help text and
//! the crates must all agree on it.

use std::fs;
use std::path::PathBuf;
use std::process::Command;

fn oxfuse(arg: &str) -> String {
    let output = Command::new(env!("CARGO_BIN_EXE_oxfuse"))
        .arg(arg)
        .output()
        .unwrap();
    assert!(output.status.success());
    String::from_utf8(output.stdout).unwrap()
}

#[test]
fn the_binary_reports_its_name_and_version() {
    assert_eq!(
        oxfuse("--version"),
        format!("oxfuse {}\n", env!("CARGO_PKG_VERSION"))
    );
}

#[test]
fn help_names_the_tool_and_its_commands() {
    let help = oxfuse("--help");
    assert!(help.contains("Usage: oxfuse"), "{help}");
    for command in ["check", "run", "inspect"] {
        assert!(
            help.contains(command),
            "help does not mention `{command}`: {help}"
        );
    }
}

#[test]
fn every_crate_is_named_oxfuse() {
    let crates = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..");
    let mut seen = 0;
    for entry in fs::read_dir(crates).unwrap() {
        let manifest = entry.unwrap().path().join("Cargo.toml");
        if !manifest.exists() {
            continue;
        }
        let text = fs::read_to_string(&manifest).unwrap();
        assert!(
            text.contains("name = \"oxfuse-") || text.contains("name = \"oxfuse\""),
            "{} is not named oxfuse or oxfuse-*",
            manifest.display()
        );
        seen += 1;
    }
    assert_eq!(seen, 4, "expected the core, runtime, wgpu and cli crates");
}
