//! Graph-to-graph passes and the manager that runs them. A pass never
//! mutates its input: it returns a new graph, so before and after are both
//! intact for dumps and equivalence tests, and no stale ids can exist.

mod cse;
mod dce;
mod fold;
mod rebuild;

pub use cse::CommonSubexpressionElimination;
pub use dce::DeadCodeElimination;
pub use fold::{ConstEvaluator, ConstantFolding};

use crate::{Error, Graph, Limits, TensorType};
use std::sync::Arc;

/// What a pass returns: the new graph and how many rewrites it made.
#[derive(Debug)]
pub struct PassOutput {
    pub graph: Graph,
    pub changed: usize,
}

pub trait Pass {
    /// Short stable name, used by reports and `--disable-pass`.
    fn name(&self) -> &'static str;
    fn run(&self, graph: &Graph) -> Result<PassOutput, Error>;
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PassReport {
    pub pass: &'static str,
    pub values_before: usize,
    pub values_after: usize,
    pub changed: usize,
}

#[derive(Debug)]
pub struct Optimized {
    pub graph: Graph,
    pub reports: Vec<PassReport>,
    /// `(label, ir text)` per stage; empty unless dumps were requested.
    pub stages: Vec<(String, String)>,
}

impl Optimized {
    pub fn dump_text(&self) -> String {
        self.stages
            .iter()
            .map(|(label, ir)| format!("== {label} ==\n{ir}\n"))
            .collect()
    }
}

pub struct PassManager {
    passes: Vec<Box<dyn Pass>>,
    limits: Limits,
    dump: bool,
}

impl PassManager {
    pub fn new(limits: Limits) -> Self {
        Self {
            passes: Vec::new(),
            limits,
            dump: false,
        }
    }

    #[allow(clippy::should_implement_trait)]
    pub fn add(mut self, pass: impl Pass + 'static) -> Self {
        self.passes.push(Box::new(pass));
        self
    }

    /// Removes every pass with this name, for testing a pipeline without it.
    pub fn without(mut self, name: &str) -> Self {
        self.passes.retain(|pass| pass.name() != name);
        self
    }

    pub fn with_dumps(mut self, dump: bool) -> Self {
        self.dump = dump;
        self
    }

    pub fn pass_names(&self) -> Vec<&'static str> {
        self.passes.iter().map(|pass| pass.name()).collect()
    }

    /// Runs the passes once, in order. The graph is verified and checked
    /// against the limits before the first pass and after every pass.
    pub fn run(&self, graph: &Graph) -> Result<Optimized, Error> {
        check(graph, &self.limits, "input graph is invalid")?;
        let expected = interface(graph);
        let mut current = graph.clone();
        let mut reports = Vec::new();
        let mut stages = Vec::new();
        if self.dump {
            stages.push(("input".to_owned(), current.to_string()));
        }
        for pass in &self.passes {
            let values_before = current.values().len();
            let output = pass.run(&current)?;
            check(
                &output.graph,
                &self.limits,
                &format!("pass `{}` produced an invalid graph", pass.name()),
            )?;
            if interface(&output.graph) != expected {
                return Err(Error::Verify(format!(
                    "pass `{}` changed the graph interface",
                    pass.name()
                )));
            }
            let report = PassReport {
                pass: pass.name(),
                values_before,
                values_after: output.graph.values().len(),
                changed: output.changed,
            };
            if self.dump {
                stages.push((
                    format!(
                        "after {} (changed {}, values {} -> {})",
                        report.pass, report.changed, report.values_before, report.values_after
                    ),
                    output.graph.to_string(),
                ));
            }
            reports.push(report);
            current = output.graph;
        }
        Ok(Optimized {
            graph: current,
            reports,
            stages,
        })
    }
}

/// What users see of a graph: inputs (name, type, order), output names and
/// output types. No pass may change it.
type Interface = (Vec<(String, TensorType)>, Vec<String>, Vec<TensorType>);

fn interface(graph: &Graph) -> Interface {
    (
        graph
            .inputs()
            .iter()
            .map(|&id| (graph.label(id), graph.value(id).ty.clone()))
            .collect(),
        graph.output_names().to_vec(),
        graph
            .outputs()
            .iter()
            .map(|&id| graph.value(id).ty.clone())
            .collect(),
    )
}

fn check(graph: &Graph, limits: &Limits, context: &str) -> Result<(), Error> {
    graph
        .verify()
        .and_then(|()| graph.verify_limits(limits))
        .map_err(|e| Error::Verify(format!("{context}: {e}")))
}

/// The standard pipeline: fold constants, merge duplicates, drop what the
/// outputs no longer need. The order is deliberate: folding creates
/// constants that leave their inputs dead, and dead-code elimination runs
/// last to collect them.
pub fn default_pipeline(limits: Limits, evaluator: Arc<dyn ConstEvaluator>) -> PassManager {
    PassManager::new(limits.clone())
        .add(ConstantFolding::new(evaluator, limits.clone()))
        .add(CommonSubexpressionElimination::new(limits.clone()))
        .add(DeadCodeElimination::new(limits))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{GraphBuilder, Op};

    fn sample() -> Graph {
        let mut b = GraphBuilder::new(Limits::default());
        let x = b.input("x", vec![2]).unwrap();
        let y = b.node("y", Op::Relu, &[x]).unwrap();
        b.finish(&[y]).unwrap()
    }

    struct Identity;
    impl Pass for Identity {
        fn name(&self) -> &'static str {
            "identity"
        }
        fn run(&self, graph: &Graph) -> Result<PassOutput, Error> {
            Ok(PassOutput {
                graph: graph.clone(),
                changed: 0,
            })
        }
    }

    /// Returns a graph with no outputs, which the verifier must reject.
    struct Breaker;
    impl Pass for Breaker {
        fn name(&self) -> &'static str {
            "breaker"
        }
        fn run(&self, graph: &Graph) -> Result<PassOutput, Error> {
            let mut broken = graph.clone();
            broken.outputs.clear();
            broken.output_names.clear();
            Ok(PassOutput {
                graph: broken,
                changed: 1,
            })
        }
    }

    #[test]
    fn passes_run_in_order_and_are_reported() {
        let manager = PassManager::new(Limits::default())
            .add(Identity)
            .add(Identity);
        let optimized = manager.run(&sample()).unwrap();
        assert_eq!(optimized.graph, sample());
        assert_eq!(
            optimized.reports,
            vec![
                PassReport {
                    pass: "identity",
                    values_before: 2,
                    values_after: 2,
                    changed: 0
                },
                PassReport {
                    pass: "identity",
                    values_before: 2,
                    values_after: 2,
                    changed: 0
                },
            ]
        );
    }

    #[test]
    fn without_removes_every_pass_with_that_name() {
        let manager = PassManager::new(Limits::default())
            .add(Identity)
            .add(Breaker)
            .without("breaker");
        assert_eq!(manager.pass_names(), ["identity"]);
        assert!(manager.run(&sample()).is_ok());
    }

    #[test]
    fn a_pass_that_breaks_the_graph_is_named() {
        let manager = PassManager::new(Limits::default())
            .add(Identity)
            .add(Breaker);
        let err = manager.run(&sample()).unwrap_err();
        assert!(
            matches!(&err, Error::Verify(m) if m.contains("pass `breaker` produced an invalid graph")),
            "{err}"
        );
    }

    /// Renames the first output, which would change what users see.
    struct Renamer;
    impl Pass for Renamer {
        fn name(&self) -> &'static str {
            "renamer"
        }
        fn run(&self, graph: &Graph) -> Result<PassOutput, Error> {
            let mut renamed = graph.clone();
            renamed.output_names[0] = "renamed".to_owned();
            Ok(PassOutput {
                graph: renamed,
                changed: 1,
            })
        }
    }

    #[test]
    fn a_pass_that_changes_the_interface_is_rejected() {
        let manager = PassManager::new(Limits::default()).add(Renamer);
        let err = manager.run(&sample()).unwrap_err();
        assert!(
            matches!(&err, Error::Verify(m) if m.contains("pass `renamer` changed the graph interface")),
            "{err}"
        );
    }

    #[test]
    fn invalid_input_is_rejected_before_any_pass_runs() {
        let mut broken = sample();
        broken.outputs.clear();
        broken.output_names.clear();
        let manager = PassManager::new(Limits::default()).add(Identity);
        let err = manager.run(&broken).unwrap_err();
        assert!(
            matches!(&err, Error::Verify(m) if m.contains("input graph is invalid")),
            "{err}"
        );
    }

    #[test]
    fn an_input_graph_over_the_limits_is_rejected() {
        let tight = Limits {
            max_values: 1,
            ..Limits::default()
        };
        let manager = PassManager::new(tight).add(Identity);
        let err = manager.run(&sample()).unwrap_err();
        assert!(
            matches!(&err, Error::Verify(m) if m.contains("input graph is invalid")),
            "{err}"
        );
    }

    #[test]
    fn dumps_are_recorded_only_when_asked() {
        let quiet = PassManager::new(Limits::default()).add(Identity);
        assert!(quiet.run(&sample()).unwrap().stages.is_empty());
        let loud = PassManager::new(Limits::default())
            .add(Identity)
            .with_dumps(true);
        let optimized = loud.run(&sample()).unwrap();
        let labels: Vec<&str> = optimized.stages.iter().map(|(l, _)| l.as_str()).collect();
        assert_eq!(
            labels,
            ["input", "after identity (changed 0, values 2 -> 2)"]
        );
        let text = optimized.dump_text();
        assert!(text.starts_with("== input ==\ngraph {\n"), "{text}");
        assert!(
            text.contains("== after identity (changed 0, values 2 -> 2) ==\n"),
            "{text}"
        );
        assert!(text.ends_with("}\n"), "{text}");
    }

    #[test]
    fn the_default_pipeline_runs_fold_cse_dce_in_that_order() {
        struct Never;
        impl ConstEvaluator for Never {
            fn eval(
                &self,
                _: &crate::Op,
                _: &[(&crate::TensorType, &[f32])],
            ) -> Result<Vec<f32>, Error> {
                Err(Error::Verify("unused".into()))
            }
        }
        let manager = default_pipeline(Limits::default(), std::sync::Arc::new(Never));
        assert_eq!(manager.pass_names(), ["fold", "cse", "dce"]);
        assert_eq!(manager.without("cse").pass_names(), ["fold", "dce"]);
    }
}
