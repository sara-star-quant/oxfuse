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

/// Build output and large binaries do not belong in the repository. A tool
/// outside the workspace (`scripts/exp-oracle`) once had its `target/`
/// directory committed: 174 files, 20 MiB, compiled third-party objects.
#[test]
fn no_build_output_or_large_files_are_tracked() {
    let Some(files) = tracked() else { return };
    let root = root();
    let in_target: Vec<&String> = files
        .iter()
        .filter(|f| f.split('/').any(|part| part == "target"))
        .collect();
    assert!(
        in_target.is_empty(),
        "build output is tracked: {in_target:?}"
    );
    let binary: Vec<&String> = files
        .iter()
        .filter(|f| {
            [
                ".o", ".a", ".rlib", ".rmeta", ".so", ".dylib", ".dll", ".exe",
            ]
            .iter()
            .any(|ext| f.ends_with(ext))
        })
        .collect();
    assert!(binary.is_empty(), "compiled files are tracked: {binary:?}");
    const LIMIT: u64 = 1 << 20;
    let large: Vec<String> = files
        .iter()
        .filter_map(|f| {
            let size = fs::metadata(root.join(f)).ok()?.len();
            (size > LIMIT).then(|| format!("{f} ({size} bytes)"))
        })
        .collect();
    assert!(large.is_empty(), "files over 1 MiB are tracked: {large:?}");
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

/// The copyright owner is filled in, and every crate ships the same LICENSE and
/// NOTICE as the root, so a published crate carries the notice (Apache-2.0,
/// section 4(d)).
#[test]
fn every_crate_carries_the_root_licence_and_notice_with_the_owner() {
    let root = root();
    let owner = "Copyright 2026 SARA STAR QUANT LLC";
    let licence = fs::read_to_string(root.join("LICENSE")).unwrap();
    let notice = fs::read_to_string(root.join("NOTICE")).expect("a NOTICE file");
    assert!(licence.contains(owner), "LICENSE lacks the owner line");
    assert!(notice.contains(owner), "NOTICE lacks the owner line");
    assert!(
        !licence.contains("[yyyy]") && !licence.contains("[name of copyright owner]"),
        "LICENSE still has the template line"
    );
    for entry in fs::read_dir(root.join("crates")).unwrap() {
        let dir = entry.unwrap().path();
        if !dir.join("Cargo.toml").exists() {
            continue;
        }
        for (name, want) in [("LICENSE", &licence), ("NOTICE", &notice)] {
            let got = fs::read_to_string(dir.join(name))
                .unwrap_or_else(|_| panic!("{} has no {name}", dir.display()));
            assert_eq!(&got, want, "{} differs from the root {name}", dir.display());
        }
    }
    let readme = fs::read_to_string(root.join("README.md")).unwrap();
    assert!(readme.contains(owner), "README footer lacks the owner");
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
/// says they are untested or measured.
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
    // These may name vendors, but only on a line that says they are untested or measured.
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
                !names_brand || line.contains("untested") || line.contains("measured"),
                "{file} names a vendor on a line that says neither `untested` nor `measured`: {line}"
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
