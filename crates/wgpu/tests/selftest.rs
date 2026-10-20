//! The self-test report: what anyone with another device runs to measure it.

mod common;

use common::{available, gpu};
use oxfuse_wgpu::{Context, LimitsProfile, run_selftest};
use std::sync::Arc;

#[test]
fn the_report_names_the_adapter_and_measures_every_tier() {
    if !available() {
        return;
    }
    let report = run_selftest(Arc::clone(gpu().context())).unwrap();
    assert_eq!(report.adapter, gpu().context().info().name);
    assert!(report.finite_tier_exact, "{report}");
    assert!(report.relu_exact_on_all_sampled_patterns, "{report}");
    assert!(report.matmul_within_bound, "{report}");
    assert!(
        report.sigmoid_max_ulp <= common::sigmoid_bound(),
        "{report}"
    );
    assert!(
        report
            .limits
            .iter()
            .any(|(name, _)| *name == "max_compute_workgroups_per_dimension")
    );
    let text = report.to_string();
    for key in [
        "adapter:",
        "api:",
        "finite tier exact:",
        "relu exact on sampled patterns:",
        "matmul within bound:",
        "matmul max ulp from reference:",
        "sigmoid max ulp from reference:",
        "subnormal add flushed:",
        "nan in add gives:",
        "max_storage_buffer_binding_size:",
    ] {
        assert!(text.contains(key), "report lacks `{key}`:\n{text}");
    }
    assert!(text.contains(&report.adapter));
}

#[test]
fn the_report_also_runs_under_the_downlevel_limits() {
    if !available() {
        return;
    }
    let ctx = Arc::new(Context::new(LimitsProfile::Downlevel).unwrap());
    let report = run_selftest(ctx).unwrap();
    assert!(
        report.finite_tier_exact && report.matmul_within_bound,
        "{report}"
    );
}
