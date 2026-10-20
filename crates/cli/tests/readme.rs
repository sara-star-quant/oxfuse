//! What a visitor reads first must be true: the README's example is the real
//! example with the real output, and the links in the public docs resolve.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn read(relative: &str) -> String {
    fs::read_to_string(root().join(relative)).unwrap()
}

#[test]
fn the_readme_graph_is_the_tiny_mlp_example() {
    let readme = read("README.md");
    let tour = readme
        .split("## A first run")
        .nth(1)
        .expect("README has a first-run section");
    let block = tour
        .split("```json\n")
        .nth(1)
        .and_then(|rest| rest.split("```").next())
        .expect("the first-run section has a json block");
    let shown: serde_json::Value = serde_json::from_str(block).unwrap();
    let real: serde_json::Value = serde_json::from_str(&read("examples/tiny_mlp.json")).unwrap();
    assert_eq!(shown, real);
}

#[test]
fn the_readme_output_is_what_the_binary_prints() {
    let readme = read("README.md");
    let mut lines = readme.lines();
    let command = "    $ oxfuse run examples/tiny_mlp.json --inputs examples/tiny_mlp.inputs.json";
    assert!(
        lines.any(|line| line == command),
        "README lost its run example"
    );
    let shown = lines
        .next()
        .expect("an output line follows the command")
        .trim();
    let examples = root().join("examples");
    let output = Command::new(env!("CARGO_BIN_EXE_oxfuse"))
        .arg("run")
        .arg(examples.join("tiny_mlp.json"))
        .arg("--inputs")
        .arg(examples.join("tiny_mlp.inputs.json"))
        .output()
        .unwrap();
    assert_eq!(shown, String::from_utf8(output.stdout).unwrap().trim());
}

fn relative_links(markdown: &str) -> Vec<String> {
    let mut links = Vec::new();
    let mut rest = markdown;
    while let Some(start) = rest.find("](") {
        rest = &rest[start + 2..];
        let Some(end) = rest.find(')') else { break };
        let target = &rest[..end];
        let path = target.split('#').next().unwrap_or("");
        if !path.is_empty() && !path.contains("://") && !path.starts_with("mailto:") {
            links.push(path.to_owned());
        }
        rest = &rest[end..];
    }
    links
}

#[test]
fn relative_links_in_the_public_docs_resolve() {
    let mut docs: Vec<(PathBuf, String)> = Vec::new();
    for name in ["README.md", "CONTRIBUTING.md", "ROADMAP.md", "SECURITY.md"] {
        docs.push((PathBuf::from(name), read(name)));
    }
    for entry in fs::read_dir(root().join("docs")).unwrap() {
        let path = entry.unwrap().path();
        if path.extension().is_some_and(|e| e == "md") {
            let text = fs::read_to_string(&path).unwrap();
            docs.push((path.strip_prefix(root()).unwrap().to_path_buf(), text));
        }
    }
    assert!(
        docs.len() >= 6,
        "expected the root docs and the docs/ pages"
    );
    for (doc, text) in docs {
        let base: &Path = doc.parent().unwrap();
        for link in relative_links(&text) {
            let target = root().join(base).join(&link);
            assert!(
                target.exists(),
                "{} links to `{link}`, which does not exist",
                doc.display()
            );
        }
    }
}
