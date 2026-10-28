//! `oxfuse compare`: two outputs files, per-output bit differences, one verdict.
//!
//! The verdict is read from run records, not guessed: a difference between two
//! runs of the same Oxfuse version on the bit-exact CPU backends breaks the
//! contract; a difference involving the GPU backend is allowed by it; anything
//! else (a foreign runtime, no record, other versions) is unresolved.

use crate::format::{NamedTensorFile, OutputsFile};
use crate::manifest;
use crate::{CliError, RunFailure};
use serde_json::Value;
use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

/// Distance in representable `f32` values; `+0.0` and `-0.0` are 0 apart and
/// two NaNs count as equal. A NaN against a number has no distance.
pub(crate) fn ulp_distance(a: f32, b: f32) -> Option<u64> {
    if a.is_nan() || b.is_nan() {
        return (a.is_nan() && b.is_nan()).then_some(0);
    }
    // Sign-magnitude to a line through zero, so the key grows with the value.
    let key = |x: f32| -> i64 {
        let bits = i64::from(x.to_bits() & 0x7fff_ffff);
        if x.is_sign_negative() { -bits } else { bits }
    };
    Some(key(a).abs_diff(key(b)))
}

fn file_error(path: &Path, reason: impl ToString) -> CliError {
    CliError::CompareFile {
        path: path.display().to_string(),
        reason: reason.to_string(),
    }
}

fn load_outputs(path: &Path) -> Result<BTreeMap<String, NamedTensorFile>, CliError> {
    let text = fs::read_to_string(path).map_err(|source| CliError::Io {
        path: path.display().to_string(),
        source,
    })?;
    let file: OutputsFile = serde_json::from_str(&text).map_err(|e| file_error(path, e))?;
    let mut by_name = BTreeMap::new();
    for tensor in file.outputs {
        let count = tensor
            .shape
            .iter()
            .try_fold(1usize, |n, &d| n.checked_mul(d));
        if count != Some(tensor.values.len()) {
            return Err(file_error(
                path,
                format!(
                    "output `{}` has {} values for shape {:?}",
                    tensor.name,
                    tensor.values.len(),
                    tensor.shape
                ),
            ));
        }
        if by_name.contains_key(&tensor.name) {
            return Err(CliError::CompareOutputs {
                reason: format!("`{}` appears twice in {}", tensor.name, path.display()),
            });
        }
        by_name.insert(tensor.name.clone(), tensor);
    }
    Ok(by_name)
}

/// What a run record says about the run, checked against the file it describes.
struct Record {
    backend: String,
    version: String,
}

fn load_record(
    path: &Path,
    outputs: &BTreeMap<String, NamedTensorFile>,
) -> Result<Record, CliError> {
    let bad = |reason: String| CliError::CompareManifest {
        path: path.display().to_string(),
        reason,
    };
    let text = fs::read_to_string(path).map_err(|source| CliError::Io {
        path: path.display().to_string(),
        source,
    })?;
    let doc: Value = serde_json::from_str(&text).map_err(|e| bad(e.to_string()))?;
    let field = |key: &str| {
        doc[key]
            .as_str()
            .map(str::to_owned)
            .ok_or_else(|| bad(format!("no string `{key}`")))
    };
    let listed = doc["outputs"]
        .as_array()
        .ok_or_else(|| bad("no `outputs` list".to_owned()))?;
    let mut hashes = BTreeMap::new();
    for entry in listed {
        if let (Some(name), Some(sha)) = (entry["name"].as_str(), entry["sha256"].as_str()) {
            hashes.insert(name, sha);
        }
    }
    let same = hashes.len() == outputs.len()
        && outputs.iter().all(|(name, t)| {
            hashes.get(name.as_str()).copied() == Some(manifest::digest_values(&t.values).as_str())
        });
    if !same {
        return Err(bad(
            "it does not describe the outputs file it was given with".to_owned(),
        ));
    }
    Ok(Record {
        backend: field("backend")?,
        version: field("oxfuse_version")?,
    })
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Verdict {
    Identical,
    PermittedVariation,
    ContractViolation,
    Unresolved,
}

impl Verdict {
    fn text(self) -> &'static str {
        match self {
            Verdict::Identical => "identical",
            Verdict::PermittedVariation => "permitted variation",
            Verdict::ContractViolation => "contract violation",
            Verdict::Unresolved => "unresolved",
        }
    }
}

/// The verdict for two runs that differ.
fn classify(a: Option<&Record>, b: Option<&Record>) -> Verdict {
    let (Some(a), Some(b)) = (a, b) else {
        return Verdict::Unresolved;
    };
    if a.version != b.version {
        return Verdict::Unresolved;
    }
    let exact = |r: &Record| matches!(r.backend.as_str(), "reference" | "cpu");
    if exact(a) && exact(b) {
        Verdict::ContractViolation
    } else if a.backend == "wgpu" || b.backend == "wgpu" {
        Verdict::PermittedVariation
    } else {
        Verdict::Unresolved
    }
}

/// One output's line, and whether it differs.
fn compare_output(a: &NamedTensorFile, b: &NamedTensorFile) -> Result<(String, bool), CliError> {
    if a.shape != b.shape {
        return Err(CliError::CompareOutputs {
            reason: format!(
                "output `{}` has shape {:?} in A and {:?} in B",
                a.name, a.shape, b.shape
            ),
        });
    }
    let mut count = 0usize;
    let mut max_ulp = 0u64;
    let mut first = None;
    for (i, (x, y)) in a.values.iter().zip(&b.values).enumerate() {
        if x.to_bits() != y.to_bits() {
            count += 1;
            max_ulp = max_ulp.max(ulp_distance(*x, *y).unwrap_or(u64::MAX));
            first.get_or_insert(i);
        }
    }
    let line = match first {
        None => format!("{}: identical", a.name),
        Some(i) => format!(
            "{}: {count} of {} values differ, max {max_ulp} ulp, first at index {i}",
            a.name,
            a.values.len()
        ),
    };
    Ok((line, first.is_some()))
}

// A failure carries its error and the report to show with it, once per run.
#[allow(clippy::result_large_err)]
pub fn compare(
    a_path: &Path,
    b_path: &Path,
    manifest_a: Option<&Path>,
    manifest_b: Option<&Path>,
) -> Result<String, RunFailure> {
    let run = || -> Result<(String, Verdict), CliError> {
        let a = load_outputs(a_path)?;
        let b = load_outputs(b_path)?;
        for name in a.keys() {
            if !b.contains_key(name) {
                return Err(CliError::CompareOutputs {
                    reason: format!("output `{name}` is only in A"),
                });
            }
        }
        if let Some(name) = b.keys().find(|n| !a.contains_key(*n)) {
            return Err(CliError::CompareOutputs {
                reason: format!("output `{name}` is only in B"),
            });
        }
        let record_a = manifest_a.map(|p| load_record(p, &a)).transpose()?;
        let record_b = manifest_b.map(|p| load_record(p, &b)).transpose()?;
        let mut report = String::new();
        let mut differs = false;
        for (name, left) in &a {
            let (line, d) = compare_output(left, &b[name])?;
            differs |= d;
            report.push_str(&line);
            report.push('\n');
        }
        let verdict = if differs {
            classify(record_a.as_ref(), record_b.as_ref())
        } else {
            Verdict::Identical
        };
        report.push_str(&format!("verdict: {}\n", verdict.text()));
        Ok((report, verdict))
    };
    match run() {
        Ok((report, Verdict::Identical | Verdict::PermittedVariation)) => Ok(report),
        Ok((report, verdict)) => Err(RunFailure::with_stdout(
            CliError::CompareDiffers {
                verdict: verdict.text(),
            },
            report,
        )),
        Err(error) => Err(RunFailure::from(error)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ulp_distance_boundaries() {
        let tiny = f32::from_bits(1);
        assert_eq!(ulp_distance(0.0, -0.0), Some(0));
        assert_eq!(ulp_distance(tiny, -tiny), Some(2));
        assert_eq!(ulp_distance(0.0, tiny), Some(1));
        assert_eq!(ulp_distance(f32::MAX, f32::INFINITY), Some(1));
        assert_eq!(ulp_distance(f32::INFINITY, f32::INFINITY), Some(0));
        assert_eq!(
            ulp_distance(1.0, f32::from_bits(1.0f32.to_bits() + 1)),
            Some(1)
        );
        assert_eq!(
            ulp_distance(-1.0, 1.0),
            Some(2 * u64::from(1.0f32.to_bits()))
        );
        assert_eq!(
            ulp_distance(f32::MIN, f32::MAX),
            Some(2 * u64::from(f32::MAX.to_bits()))
        );
        assert_eq!(ulp_distance(f32::NAN, f32::NAN), Some(0));
        assert_eq!(ulp_distance(f32::NAN, 1.0), None);
        assert_eq!(ulp_distance(1.0, f32::NAN), None);
    }

    fn record(backend: &str, version: &str) -> Record {
        Record {
            backend: backend.to_owned(),
            version: version.to_owned(),
        }
    }

    #[test]
    fn classification_table() {
        let c = |a: &str, b: &str| classify(Some(&record(a, "1")), Some(&record(b, "1")));
        assert_eq!(c("reference", "cpu"), Verdict::ContractViolation);
        assert_eq!(c("cpu", "cpu"), Verdict::ContractViolation);
        assert_eq!(c("cpu", "wgpu"), Verdict::PermittedVariation);
        assert_eq!(c("wgpu", "reference"), Verdict::PermittedVariation);
        assert_eq!(c("wgpu", "wgpu"), Verdict::PermittedVariation);
        assert_eq!(c("cpu", "other"), Verdict::Unresolved);
        assert_eq!(
            classify(Some(&record("cpu", "1")), Some(&record("cpu", "2"))),
            Verdict::Unresolved
        );
        assert_eq!(
            classify(Some(&record("cpu", "1")), None),
            Verdict::Unresolved
        );
        assert_eq!(classify(None, None), Verdict::Unresolved);
    }
}
