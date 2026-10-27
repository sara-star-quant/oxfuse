//! `docs/comparison.md` quotes other projects, so every row carries its source
//! and the date it was read, and no cell ranks one project against another.

use std::fs;
use std::path::PathBuf;

fn page() -> String {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    fs::read_to_string(root.join("docs/comparison.md")).expect("docs/comparison.md")
}

fn rows(page: &str) -> Vec<Vec<String>> {
    page.lines()
        .filter(|l| l.starts_with("| ") && !l.starts_with("| Project") && !l.starts_with("|---"))
        .map(|l| {
            l.trim_matches('|')
                .split('|')
                .map(|c| c.trim().to_owned())
                .collect()
        })
        .collect()
}

fn is_iso_date(s: &str) -> bool {
    let b = s.as_bytes();
    b.len() == 10
        && b.iter().enumerate().all(|(i, c)| match i {
            4 | 7 => *c == b'-',
            _ => c.is_ascii_digit(),
        })
}

#[test]
fn every_row_has_a_versioned_source_and_a_read_date() {
    let rows = rows(&page());
    assert!(rows.len() >= 30, "the table lost rows: {}", rows.len());
    for cells in rows {
        assert_eq!(cells.len(), 5, "{cells:?}");
        let url = cells[3].trim_start_matches('<').trim_end_matches('>');
        assert!(url.starts_with("https://"), "no source URL: {cells:?}");
        assert!(is_iso_date(&cells[4]), "no ISO read date: {cells:?}");
        assert!(!cells[2].is_empty(), "{cells:?}");
    }
}

#[test]
fn no_cell_ranks_projects() {
    for cells in rows(&page()) {
        let text = cells[2].to_lowercase();
        for word in [
            "faster",
            "slower",
            "fastest",
            "best",
            "better",
            "worse",
            "outperform",
            "times",
        ] {
            assert!(
                !text
                    .split(|c: char| !c.is_alphanumeric())
                    .any(|w| w == word),
                "`{word}` in a comparison cell: {cells:?}"
            );
        }
    }
}

#[test]
fn the_page_says_how_to_correct_a_cell_and_what_it_is_not() {
    let page = page();
    assert!(page.contains("Open an issue"));
    assert!(page.contains("not a benchmark and not a ranking"));
}

#[test]
fn every_project_is_cited_for_every_topic() {
    let rows = rows(&page());
    for topic in [
        "Scope",
        "Licence",
        "Devices",
        "Training",
        "Reproducibility",
        "ONNX operators",
        "Latest release",
    ] {
        let n = rows.iter().filter(|c| c[1] == topic).count();
        assert!(n >= 6, "topic `{topic}` has {n} rows");
    }
}

#[test]
fn the_index_says_who_checks_the_links() {
    let readme =
        fs::read_to_string(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../README.md"))
            .unwrap();
    assert!(readme.contains("docs/comparison.md"));
}
