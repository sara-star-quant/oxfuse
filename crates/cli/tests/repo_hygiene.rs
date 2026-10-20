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

/// The README must say what hardware is supported, and nothing public may
/// claim an accelerator or name a vendor's hardware as a target. When a real
/// accelerator backend lands, update this test, the README, the About text,
/// the roadmap and the claims ledger in the same change.
#[test]
fn hardware_claims_match_what_exists() {
    let root = root();
    let readme = fs::read_to_string(root.join("README.md")).unwrap();
    assert!(
        readme.contains("CPU only"),
        "README must state that Oxfuse is CPU only"
    );
    let mut public = vec![readme, fs::read_to_string(root.join("ROADMAP.md")).unwrap()];
    let mut dirs = vec![root.join("docs")];
    while let Some(dir) = dirs.pop() {
        for entry in fs::read_dir(dir).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() {
                dirs.push(path);
            } else {
                public.push(fs::read_to_string(path).unwrap());
            }
        }
    }
    let brands = ["nvidia", "cuda", "tenstorrent", "rocm", "tpu"];
    for text in public {
        for word in text.split(|c: char| !c.is_alphanumeric()) {
            let word = word.to_lowercase();
            assert!(
                !brands.contains(&word.as_str()),
                "public docs name `{word}`: no vendor support exists"
            );
        }
    }
}
