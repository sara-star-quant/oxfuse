//! `oxfuse conform`: another runtime's outputs against the enclosure of the graph.
//!
//! The enclosure (see `oxfuse_runtime::analyze_box`) contains the exact real result and the
//! reference float32 result of every element for every input in the box. Each element of the
//! outputs file is then **inside** it, **outside** it, or **unresolved**. Outside means "not
//! explained by the reference rounding model": a runtime that sums in another order has its
//! own rounding error and can leave the enclosure without being wrong in any other sense.
//! There is no tolerance anywhere.

use crate::commands::{Dims, boxes_of, load_and_bind};
use crate::compare::load_outputs;
use crate::{CliError, RunFailure};
use std::path::Path;

/// What one element of the outputs file is against its enclosure.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Judgement {
    Inside,
    Outside,
    Unresolved,
}

/// `lo` and `hi` may be infinite (unbounded). A value outside a finite end is outside, whatever
/// the other end is. A value that is not a finite number, or one inside an enclosure that is
/// unbounded on a side, is unresolved: neither is ever a pass.
pub(crate) fn judge(value: f32, lo: f64, hi: f64) -> Judgement {
    if !value.is_finite() {
        return Judgement::Unresolved;
    }
    let v = f64::from(value);
    if v < lo || v > hi {
        Judgement::Outside
    } else if lo == f64::NEG_INFINITY || hi == f64::INFINITY {
        Judgement::Unresolved
    } else {
        Judgement::Inside
    }
}

/// The verdict of a whole run: any outside element decides it, then any unresolved one.
fn overall(outside: usize, unresolved: usize) -> &'static str {
    if outside > 0 {
        "outside"
    } else if unresolved > 0 {
        "unresolved"
    } else {
        "inside"
    }
}

fn reason(text: String) -> CliError {
    CliError::ConformOutputs { reason: text }
}

/// Checks `outputs_path` (the JSON `run` prints) against the graph over the inputs (points or
/// boxes). The report goes to stdout; anything but an overall `inside` is a failure that
/// carries the report.
// A failure carries its error and the report to show with it, once per run.
#[allow(clippy::result_large_err)]
pub fn conform(
    outputs_path: &Path,
    graph_path: &Path,
    inputs_path: Option<&Path>,
    dims: &Dims,
) -> Result<String, RunFailure> {
    let run = || -> Result<(String, &'static str), CliError> {
        let given = load_outputs(outputs_path)?;
        let (graph, bound) = load_and_bind(graph_path, inputs_path, dims)?;
        let (los, his) = boxes_of(bound);
        let enclosures = oxfuse_runtime::analyze_box(&graph, &los, &his)?;
        if let Some(extra) = given
            .keys()
            .find(|name| !enclosures.iter().any(|e| &e.name == *name))
        {
            return Err(reason(format!(
                "output `{extra}` is not an output of the graph"
            )));
        }
        let mut report = String::new();
        let (mut outside, mut unresolved) = (0usize, 0usize);
        for e in &enclosures {
            let got = given.get(&e.name).ok_or_else(|| {
                reason(format!(
                    "output `{}` is missing from the outputs file",
                    e.name
                ))
            })?;
            if got.shape != e.shape {
                return Err(reason(format!(
                    "output `{}` has shape {:?} in the outputs file and {:?} in the graph",
                    e.name, got.shape, e.shape
                )));
            }
            let (mut inside_n, mut outside_n, mut unresolved_n) = (0usize, 0usize, 0usize);
            let mut first = None;
            for (i, &value) in got.values.iter().enumerate() {
                match judge(value, e.lo[i], e.hi[i]) {
                    Judgement::Inside => inside_n += 1,
                    Judgement::Outside => {
                        outside_n += 1;
                        first.get_or_insert(i);
                    }
                    Judgement::Unresolved => unresolved_n += 1,
                }
            }
            report.push_str(&format!(
                "{}: {inside_n} of {} inside, {outside_n} outside, {unresolved_n} unresolved",
                e.name,
                got.values.len()
            ));
            if let Some(i) = first {
                report.push_str(&format!(
                    "; first outside at index {i}: {} is not in [{}, {}]",
                    got.values[i], e.lo[i], e.hi[i]
                ));
            }
            report.push('\n');
            outside += outside_n;
            unresolved += unresolved_n;
        }
        let verdict = overall(outside, unresolved);
        report.push_str(&format!("verdict: {verdict}\n"));
        Ok((report, verdict))
    };
    match run() {
        Ok((report, "inside")) => Ok(report),
        Ok((report, verdict)) => Err(RunFailure::with_stdout(
            CliError::ConformDiffers { verdict },
            report,
        )),
        Err(error) => Err(RunFailure::from(error)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_value_is_inside_exactly_when_it_is_between_finite_ends_inclusive() {
        assert_eq!(judge(1.0, 1.0, 2.0), Judgement::Inside);
        assert_eq!(judge(2.0, 1.0, 2.0), Judgement::Inside);
        assert_eq!(judge(1.5, 1.0, 2.0), Judgement::Inside);
        assert_eq!(judge(0.5, 1.0, 2.0), Judgement::Outside);
        assert_eq!(judge(2.5, 1.0, 2.0), Judgement::Outside);
        // One float32 step beyond each end.
        assert_eq!(
            judge(f32::from_bits(1.0f32.to_bits() - 1), 1.0, 2.0),
            Judgement::Outside
        );
        assert_eq!(
            judge(f32::from_bits(2.0f32.to_bits() + 1), 1.0, 2.0),
            Judgement::Outside
        );
        assert_eq!(judge(-1.5, -2.0, -1.0), Judgement::Inside);
        assert_eq!(judge(0.0, 0.0, 0.0), Judgement::Inside);
        assert_eq!(judge(-0.0, 0.0, 0.0), Judgement::Inside);
    }

    #[test]
    fn an_end_is_compared_in_float64_not_rounded_to_float32() {
        // 1 + 1e-9 is not a float32: a value of 1.0 is below the lower end by less than one step.
        assert_eq!(judge(1.0, 1.0 + 1e-9, 2.0), Judgement::Outside);
        assert_eq!(judge(2.0, 1.0, 2.0 - 1e-9), Judgement::Outside);
    }

    #[test]
    fn an_unbounded_side_is_never_a_pass_but_a_value_past_the_finite_end_is_outside() {
        let inf = f64::INFINITY;
        assert_eq!(judge(1.5, f64::NEG_INFINITY, inf), Judgement::Unresolved);
        assert_eq!(judge(1.5, 1.0, inf), Judgement::Unresolved);
        assert_eq!(judge(1.5, f64::NEG_INFINITY, 2.0), Judgement::Unresolved);
        assert_eq!(judge(0.5, 1.0, inf), Judgement::Outside);
        assert_eq!(judge(2.5, f64::NEG_INFINITY, 2.0), Judgement::Outside);
    }

    #[test]
    fn a_value_that_is_not_finite_is_unresolved_whatever_the_enclosure() {
        for v in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert_eq!(judge(v, -1.0, 1.0), Judgement::Unresolved);
            assert_eq!(
                judge(v, f64::NEG_INFINITY, f64::INFINITY),
                Judgement::Unresolved
            );
        }
    }

    #[test]
    fn the_overall_verdict_is_outside_before_unresolved_before_inside() {
        assert_eq!(overall(0, 0), "inside");
        assert_eq!(overall(0, 3), "unresolved");
        assert_eq!(overall(2, 0), "outside");
        assert_eq!(overall(2, 3), "outside");
        assert_eq!(overall(1, 0), "outside");
    }
}
