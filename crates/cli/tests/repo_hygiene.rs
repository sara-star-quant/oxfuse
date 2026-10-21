//! What is published must be what is meant to be published: no assistant or
//! tooling working files, and exactly one licence, stated the same way
//! everywhere.

use std::fs;
use std::path::PathBuf;
use std::process::Command;

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// Tracked files, or `None` outside a git checkout (a source tarball).
fn tracked() -> Option<Vec<String>> {
    let output = Command::new("git")
        .arg("ls-files")
        .current_dir(root())
        .output()
        .ok()?;
    output.status.success().then(|| {
        String::from_utf8(output.stdout)
            .unwrap()
            .lines()
            .map(str::to_owned)
            .collect()
    })
}

#[test]
fn no_working_files_are_tracked() {
    let Some(files) = tracked() else { return };
    let leaked: Vec<&String> = files
        .iter()
        .filter(|f| {
            [".claude/", ".superpowers/", ".remember/"]
                .iter()
                .any(|p| f.starts_with(p))
        })
        .collect();
    assert!(leaked.is_empty(), "working files are tracked: {leaked:?}");
}

#[test]
fn there_is_exactly_one_licence_and_it_is_apache_2_0() {
    let root = root();
    let licence = fs::read_to_string(root.join("LICENSE")).expect("a LICENSE file");
    assert!(
        licence.contains("Apache License"),
        "LICENSE is not the Apache text"
    );
    for stale in ["LICENSE-MIT", "LICENSE-APACHE"] {
        assert!(!root.join(stale).exists(), "{stale} should not exist");
    }
    let manifest = fs::read_to_string(root.join("Cargo.toml")).unwrap();
    assert!(
        manifest.contains("license = \"Apache-2.0\""),
        "Cargo.toml licence differs"
    );
    for doc in ["README.md", "CONTRIBUTING.md"] {
        let text = fs::read_to_string(root.join(doc)).unwrap();
        assert!(!text.contains("MIT"), "{doc} still mentions MIT");
    }
    let readme = fs::read_to_string(root.join("README.md")).unwrap();
    assert!(
        readme.contains("Apache-2.0"),
        "README does not state the licence"
    );
}

const BRANDS: [&str; 12] = [
    "nvidia",
    "cuda",
    "tenstorrent",
    "rocm",
    "tpu",
    "amd",
    "intel",
    "qualcomm",
    "adreno",
    "mali",
    "radeon",
    "geforce",
];

/// Public docs must not name an accelerator vendor, so none reads as a support
/// claim. The exceptions are the two records of measurements: `docs/adapters.md`,
/// whose first line says "measured, not supported" and which may not use the words
/// that would turn a measurement into a promise, and the generated
/// `docs/benchmarks.md`, which states the machine it ran on. The README, the
/// `oxfuse-wgpu` README and its manifest may name vendors only on a line that
/// says they are untested.
#[test]
fn hardware_claims_match_what_exists() {
    let root = root();
    let readme = fs::read_to_string(root.join("README.md")).unwrap();
    assert!(
        readme.contains("CPU by default"),
        "README must state that Oxfuse is CPU by default"
    );
    assert!(
        readme.contains("docs/adapters.md"),
        "README must point at the measured adapters"
    );
    let adapters_path = root.join("docs/adapters.md");
    let benchmarks_path = root.join("docs/benchmarks.md");
    // These may name vendors, but only on a line that says they are untested.
    for file in [
        "README.md",
        "crates/wgpu/README.md",
        "crates/wgpu/Cargo.toml",
    ] {
        let text = fs::read_to_string(root.join(file)).unwrap();
        for line in text.lines() {
            let names_brand = line
                .to_lowercase()
                .split(|c: char| !c.is_alphanumeric())
                .any(|w| BRANDS.contains(&w));
            assert!(
                !names_brand || line.contains("untested"),
                "{file} names a vendor on a line that does not say `untested`: {line}"
            );
        }
    }
    let mut public = vec![fs::read_to_string(root.join("ROADMAP.md")).unwrap()];
    let mut dirs = vec![root.join("docs")];
    while let Some(dir) = dirs.pop() {
        for entry in fs::read_dir(dir).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() {
                dirs.push(path);
            } else if path != adapters_path && path != benchmarks_path {
                public.push(fs::read_to_string(path).unwrap());
            }
        }
    }
    for text in public {
        for word in text.split(|c: char| !c.is_alphanumeric()) {
            let word = word.to_lowercase();
            assert!(
                !BRANDS.contains(&word.as_str()),
                "public docs name `{word}` outside docs/adapters.md"
            );
        }
    }
    let adapters = fs::read_to_string(&adapters_path).expect("docs/adapters.md");
    let first = adapters.lines().next().unwrap_or_default().to_lowercase();
    assert!(
        first.contains("measured, not supported"),
        "docs/adapters.md must be headed `measured, not supported`"
    );
    let lower = adapters.to_lowercase();
    for promise in [
        "supports",
        "supported on",
        "supported by",
        "works on",
        "compatible with",
    ] {
        assert!(
            !lower.contains(promise),
            "docs/adapters.md uses `{promise}`: it records measurements, not support"
        );
    }
}
