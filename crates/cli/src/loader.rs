//! Turns a graph file into a verified `Graph`: checks limits and names,
//! orders nodes topologically (the file's node order is not trusted),
//! detects cycles, and drives the builder.

use crate::CliError;
use crate::format::{FORMAT_VERSION, GraphFile, NodeDecl};
use oxfuse_core::{Error, Graph, GraphBuilder, Limits, Op, ValueId};
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap, HashSet};

pub fn load_graph(json: &str, limits: &Limits) -> Result<Graph, CliError> {
    if json.len() > limits.max_file_bytes {
        return Err(Error::LimitExceeded {
            what: "file bytes",
            found: json.len(),
            limit: limits.max_file_bytes,
        }
        .into());
    }
    let file: GraphFile = serde_json::from_str(json)?;
    build_graph(&file, limits).map_err(|source| match locate(&file, &source) {
        Some(path) => CliError::Located {
            path,
            source: Box::new(source),
        },
        None => CliError::Core(source),
    })
}

/// Where in the file a core error comes from, as a JSON path. Core errors
/// carry names, not positions, so this maps a name back to its declaration.
/// Errors about the graph as a whole have no single place and return `None`.
fn locate(file: &GraphFile, error: &Error) -> Option<String> {
    let node_index = |name: &str| file.nodes.iter().position(|n| n.name == name);
    let constant_index = |name: &str| file.constants.iter().position(|c| c.name == name);
    let node_path = |name: &str| node_index(name).map(|i| format!("nodes[{i}]"));
    match error {
        Error::UnsupportedVersion { .. } => Some("format_version".into()),
        Error::DuplicateName { name } => {
            let declared = file
                .inputs
                .iter()
                .enumerate()
                .map(|(i, d)| ("inputs", i, &d.name))
                .chain(
                    file.constants
                        .iter()
                        .enumerate()
                        .map(|(i, d)| ("constants", i, &d.name)),
                )
                .chain(
                    file.nodes
                        .iter()
                        .enumerate()
                        .map(|(i, d)| ("nodes", i, &d.name)),
                );
            let (section, index, _) = declared.filter(|(_, _, n)| *n == name).nth(1)?;
            Some(format!("{section}[{index}].name"))
        }
        Error::UnknownReference { node, name } => {
            let i = node_index(node)?;
            let j = file.nodes[i]
                .inputs
                .iter()
                .position(|input| input == name)?;
            Some(format!("nodes[{i}].inputs[{j}]"))
        }
        Error::UnknownOutput { name } => {
            let k = file.outputs.iter().position(|output| output == name)?;
            Some(format!("outputs[{k}]"))
        }
        Error::NoOutputs => Some("outputs".into()),
        Error::Cycle { node } | Error::ShapeMismatch { node, .. } => node_path(node),
        Error::Arity { node, .. } => node_path(node).map(|p| format!("{p}.inputs")),
        Error::UnsupportedOp { node, .. } => node_path(node).map(|p| format!("{p}.op")),
        Error::MissingAttribute { node, .. } => node_path(node).map(|p| format!("{p}.attrs")),
        Error::UnexpectedAttribute { node, attr, .. }
        | Error::InvalidAttribute { node, attr, .. } => {
            node_path(node).map(|p| format!("{p}.attrs.{attr}"))
        }
        Error::DataLength { name, .. } | Error::NonFinite { name } => {
            constant_index(name).map(|i| format!("constants[{i}].values"))
        }
        Error::InvalidShape { name, .. } | Error::TensorLimit { name, .. } => {
            if let Some(i) = file.inputs.iter().position(|d| d.name == *name) {
                Some(format!("inputs[{i}].shape"))
            } else if let Some(i) = constant_index(name) {
                Some(format!("constants[{i}].shape"))
            } else {
                node_path(name)
            }
        }
        Error::LimitExceeded { .. }
        | Error::InputCount { .. }
        | Error::InputShape { .. }
        | Error::Verify(_)
        | Error::Backend(_) => None,
    }
}

fn build_graph(file: &GraphFile, limits: &Limits) -> Result<Graph, Error> {
    if file.format_version != FORMAT_VERSION {
        return Err(Error::UnsupportedVersion {
            found: file.format_version,
            supported: FORMAT_VERSION,
        });
    }
    let declared = file.inputs.len() + file.constants.len() + file.nodes.len();
    if declared > limits.max_values {
        return Err(Error::LimitExceeded {
            what: "values",
            found: declared,
            limit: limits.max_values,
        });
    }

    let mut names = HashSet::new();
    let declared_names = file
        .inputs
        .iter()
        .map(|d| &d.name)
        .chain(file.constants.iter().map(|d| &d.name))
        .chain(file.nodes.iter().map(|d| &d.name));
    for name in declared_names {
        if !names.insert(name.as_str()) {
            return Err(Error::DuplicateName { name: name.clone() });
        }
    }

    let mut builder = GraphBuilder::new(limits.clone());
    for input in &file.inputs {
        builder.input(&input.name, input.shape.clone())?;
    }
    for constant in &file.constants {
        if constant.values.iter().any(|v| !v.is_finite()) {
            return Err(Error::NonFinite {
                name: constant.name.clone(),
            });
        }
        builder.constant(
            &constant.name,
            constant.shape.clone(),
            constant.values.clone(),
        )?;
    }

    let ops = file
        .nodes
        .iter()
        .map(parse_op)
        .collect::<Result<Vec<Op>, Error>>()?;
    for index in topological_order(&file.nodes, &builder)? {
        let node = &file.nodes[index];
        let inputs: Vec<ValueId> = node
            .inputs
            .iter()
            .map(|name| {
                builder
                    .lookup(name)
                    .expect("topological order defines every input first")
            })
            .collect();
        builder.node(&node.name, ops[index].clone(), &inputs)?;
    }

    let mut outputs = Vec::with_capacity(file.outputs.len());
    for name in &file.outputs {
        let id = builder
            .lookup(name)
            .ok_or_else(|| Error::UnknownOutput { name: name.clone() })?;
        outputs.push(id);
    }
    builder.finish(&outputs)
}

fn parse_op(node: &NodeDecl) -> Result<Op, Error> {
    let attrs = &node.attrs;
    let op = match node.op.as_str() {
        "Relu" => Op::Relu,
        "Sigmoid" => Op::Sigmoid,
        "Add" => Op::Add,
        "Mul" => Op::Mul,
        "MatMul" => Op::MatMul,
        "Reshape" => Op::Reshape {
            shape: require(node, "Reshape", "shape", &attrs.shape)?,
        },
        "Transpose" => Op::Transpose {
            perm: require(node, "Transpose", "perm", &attrs.perm)?,
        },
        other => {
            return Err(Error::UnsupportedOp {
                node: node.name.clone(),
                op: other.to_owned(),
            });
        }
    };
    if attrs.shape.is_some() && !matches!(op, Op::Reshape { .. }) {
        return Err(unexpected(node, &op, "shape"));
    }
    if attrs.perm.is_some() && !matches!(op, Op::Transpose { .. }) {
        return Err(unexpected(node, &op, "perm"));
    }
    Ok(op)
}

fn require(
    node: &NodeDecl,
    op: &'static str,
    attr: &'static str,
    value: &Option<Vec<i64>>,
) -> Result<Vec<usize>, Error> {
    let values = value.as_ref().ok_or_else(|| Error::MissingAttribute {
        node: node.name.clone(),
        op,
        attr,
    })?;
    values
        .iter()
        .map(|&v| {
            usize::try_from(v).map_err(|_| {
                let hint = if v == -1 && attr == "shape" {
                    " (the -1 wildcard is not supported)"
                } else {
                    ""
                };
                Error::InvalidAttribute {
                    node: node.name.clone(),
                    op,
                    attr,
                    reason: format!("entry {v} is not a non-negative integer{hint}"),
                }
            })
        })
        .collect()
}

fn unexpected(node: &NodeDecl, op: &Op, attr: &'static str) -> Error {
    Error::UnexpectedAttribute {
        node: node.name.clone(),
        op: op.name(),
        attr,
    }
}

/// Kahn's algorithm. Ties break on file order so the result is deterministic.
/// `builder` already holds the graph inputs and constants, so a name that is
/// neither a node nor in the builder is an unknown reference.
fn topological_order(nodes: &[NodeDecl], builder: &GraphBuilder) -> Result<Vec<usize>, Error> {
    let index: HashMap<&str, usize> = nodes
        .iter()
        .enumerate()
        .map(|(i, n)| (n.name.as_str(), i))
        .collect();
    let mut indegree = vec![0usize; nodes.len()];
    let mut dependents: Vec<Vec<usize>> = vec![Vec::new(); nodes.len()];
    for (i, node) in nodes.iter().enumerate() {
        for input in &node.inputs {
            if let Some(&dep) = index.get(input.as_str()) {
                indegree[i] += 1;
                dependents[dep].push(i);
            } else if builder.lookup(input).is_none() {
                return Err(Error::UnknownReference {
                    node: node.name.clone(),
                    name: input.clone(),
                });
            }
        }
    }

    let mut ready: BinaryHeap<Reverse<usize>> = indegree
        .iter()
        .enumerate()
        .filter(|&(_, &d)| d == 0)
        .map(|(i, _)| Reverse(i))
        .collect();
    let mut order = Vec::with_capacity(nodes.len());
    while let Some(Reverse(i)) = ready.pop() {
        order.push(i);
        for &d in &dependents[i] {
            indegree[d] -= 1;
            if indegree[d] == 0 {
                ready.push(Reverse(d));
            }
        }
    }
    if order.len() == nodes.len() {
        return Ok(order);
    }
    let culprit = find_cycle_node(nodes, &index, &indegree);
    Err(Error::Cycle {
        node: nodes[culprit].name.clone(),
    })
}

/// Nodes left with a positive in-degree are on a cycle or downstream of one.
/// Every such node has an unprocessed dependency, so following those edges
/// must revisit a node, and the revisited node is on a cycle.
fn find_cycle_node(nodes: &[NodeDecl], index: &HashMap<&str, usize>, indegree: &[usize]) -> usize {
    let mut current = indegree
        .iter()
        .position(|&d| d > 0)
        .expect("a cycle leaves at least one unprocessed node");
    let mut seen = HashSet::new();
    while seen.insert(current) {
        current = nodes[current]
            .inputs
            .iter()
            .filter_map(|name| index.get(name.as_str()).copied())
            .find(|&dep| indegree[dep] > 0)
            .expect("an unprocessed node has an unprocessed dependency");
    }
    current
}

#[cfg(test)]
mod tests {
    use super::*;

    fn load(json: &str) -> Result<Graph, CliError> {
        load_graph(json, &Limits::default())
    }

    fn core_err(json: &str) -> Error {
        match load(json) {
            Err(CliError::Core(e)) => e,
            Err(CliError::Located { source, .. }) => *source,
            other => panic!("expected a core error, got {other:?}"),
        }
    }

    /// The path and error of a graph the loader rejects with a location.
    fn located(json: &str) -> (String, Error) {
        match load(json) {
            Err(CliError::Located { path, source }) => (path, *source),
            other => panic!("expected a located error, got {other:?}"),
        }
    }

    fn path_of(json: &str) -> String {
        located(json).0
    }

    fn graph(nodes: &str, outputs: &str) -> String {
        format!(
            r#"{{"format_version":1,
                "inputs":[{{"name":"x","dtype":"f32","shape":[2]}}],
                "nodes":[{nodes}],
                "outputs":[{outputs}]}}"#
        )
    }

    fn relu(name: &str, input: &str) -> String {
        format!(r#"{{"name":"{name}","op":"Relu","inputs":["{input}"]}}"#)
    }

    #[test]
    fn loads_a_minimal_graph() {
        let g = load(&graph(&relu("y", "x"), r#""y""#)).unwrap();
        assert_eq!(g.values().len(), 2);
        assert_eq!(g.inputs().len(), 1);
        assert_eq!(g.outputs().len(), 1);
    }

    #[test]
    fn nodes_may_appear_in_any_order() {
        let nodes = format!("{},{}", relu("b", "a"), relu("a", "x"));
        let g = load(&graph(&nodes, r#""b""#)).unwrap();
        let labels: Vec<String> = (0..g.values().len())
            .map(|i| g.label(oxfuse_core::ValueId(i as u32)))
            .collect();
        assert_eq!(labels, ["x", "a", "b"]);
    }

    #[test]
    fn constants_are_loaded() {
        let json = r#"{"format_version":1,
            "constants":[{"name":"c","dtype":"f32","shape":[2],"values":[1.5,-2]}],
            "nodes":[{"name":"r","op":"Relu","inputs":["c"]}],
            "outputs":["r"]}"#;
        let g = load(json).unwrap();
        assert!(g.inputs().is_empty());
        assert_eq!(g.values().len(), 2);
    }

    #[test]
    fn unsupported_version_is_rejected() {
        let json = graph(&relu("y", "x"), r#""y""#)
            .replace("\"format_version\":1", "\"format_version\":2");
        assert_eq!(
            core_err(&json),
            Error::UnsupportedVersion {
                found: 2,
                supported: 1
            }
        );
    }

    #[test]
    fn duplicate_name_across_kinds_is_rejected() {
        let err = core_err(&graph(&relu("x", "x"), r#""x""#));
        assert_eq!(err, Error::DuplicateName { name: "x".into() });
    }

    #[test]
    fn unknown_reference_is_rejected() {
        let err = core_err(&graph(&relu("y", "ghost"), r#""y""#));
        assert_eq!(
            err,
            Error::UnknownReference {
                node: "y".into(),
                name: "ghost".into()
            }
        );
    }

    #[test]
    fn unknown_output_is_rejected() {
        let err = core_err(&graph(&relu("y", "x"), r#""nope""#));
        assert_eq!(
            err,
            Error::UnknownOutput {
                name: "nope".into()
            }
        );
    }

    #[test]
    fn empty_outputs_are_rejected() {
        assert_eq!(core_err(&graph(&relu("y", "x"), "")), Error::NoOutputs);
    }

    #[test]
    fn unsupported_op_is_rejected() {
        let node = r#"{"name":"y","op":"Gelu","inputs":["x"]}"#;
        assert_eq!(
            core_err(&graph(node, r#""y""#)),
            Error::UnsupportedOp {
                node: "y".into(),
                op: "Gelu".into()
            }
        );
    }

    #[test]
    fn wrong_arity_is_rejected() {
        let node = r#"{"name":"y","op":"Relu","inputs":["x","x"]}"#;
        assert!(matches!(
            core_err(&graph(node, r#""y""#)),
            Error::Arity {
                expected: 1,
                found: 2,
                ..
            }
        ));
    }

    #[test]
    fn constant_length_mismatch_is_rejected() {
        let json = r#"{"format_version":1,
            "constants":[{"name":"c","dtype":"f32","shape":[3],"values":[1,2]}],
            "nodes":[],"outputs":["c"]}"#;
        assert!(matches!(
            core_err(json),
            Error::DataLength {
                expected: 3,
                found: 2,
                ..
            }
        ));
    }

    #[test]
    fn overflowing_constant_is_rejected() {
        let json = r#"{"format_version":1,
            "constants":[{"name":"c","dtype":"f32","shape":[1],"values":[1e39]}],
            "nodes":[],"outputs":["c"]}"#;
        assert!(matches!(located(json).1, Error::NonFinite { .. }));
    }

    #[test]
    fn huge_declared_shape_is_rejected() {
        let json = r#"{"format_version":1,
            "inputs":[{"name":"x","dtype":"f32","shape":[4294967296,4294967296]}],
            "nodes":[],"outputs":["x"]}"#;
        assert!(matches!(
            core_err(json),
            Error::InvalidShape { .. } | Error::TensorLimit { .. }
        ));
    }

    #[test]
    fn self_loop_is_a_cycle() {
        let err = core_err(&graph(&relu("a", "a"), r#""a""#));
        assert_eq!(err, Error::Cycle { node: "a".into() });
    }

    #[test]
    fn cycle_blames_a_node_on_the_cycle() {
        // c is only downstream of the a <-> b cycle and is listed first.
        let nodes = format!("{},{},{}", relu("c", "a"), relu("a", "b"), relu("b", "a"));
        match core_err(&graph(&nodes, r#""c""#)) {
            Error::Cycle { node } => assert!(node == "a" || node == "b", "blamed {node}"),
            other => panic!("expected a cycle, got {other:?}"),
        }
    }

    #[test]
    fn unknown_json_field_is_rejected() {
        let json = graph(&relu("y", "x"), r#""y""#).replace("\"nodes\"", "\"extra\":1,\"nodes\"");
        assert!(matches!(load(&json), Err(CliError::GraphJson(_))));
    }

    #[test]
    fn unsupported_dtype_is_rejected() {
        let json = graph(&relu("y", "x"), r#""y""#).replace("\"f32\"", "\"i64\"");
        assert!(matches!(load(&json), Err(CliError::GraphJson(_))));
    }

    #[test]
    fn oversized_file_is_rejected() {
        let limits = Limits {
            max_file_bytes: 10,
            ..Limits::default()
        };
        let err = load_graph(&graph(&relu("y", "x"), r#""y""#), &limits).unwrap_err();
        assert!(matches!(
            err,
            CliError::Core(Error::LimitExceeded {
                what: "file bytes",
                ..
            })
        ));
    }

    #[test]
    fn too_many_nodes_are_rejected() {
        let limits = Limits {
            max_values: 1,
            ..Limits::default()
        };
        let nodes = format!("{},{}", relu("a", "x"), relu("b", "a"));
        let err = load_graph(&graph(&nodes, r#""b""#), &limits).unwrap_err();
        assert!(matches!(err, CliError::Core(Error::LimitExceeded { .. })));
    }

    #[test]
    fn value_limit_counts_inputs_constants_and_nodes_together() {
        let json = r#"{"format_version":1,
            "inputs":[{"name":"x","dtype":"f32","shape":[1]}],
            "constants":[{"name":"c","dtype":"f32","shape":[1],"values":[1]}],
            "nodes":[{"name":"y","op":"Relu","inputs":["x"]}],
            "outputs":["y"]}"#;
        let limits = Limits {
            max_values: 2,
            ..Limits::default()
        };
        let err = load_graph(json, &limits).unwrap_err();
        assert!(matches!(
            err,
            CliError::Core(Error::LimitExceeded {
                what: "values",
                found: 3,
                limit: 2
            })
        ));
        let at_limit = Limits {
            max_values: 3,
            ..Limits::default()
        };
        assert!(load_graph(json, &at_limit).is_ok());
    }

    #[test]
    fn binary_ops_load() {
        let json = r#"{"format_version":1,
            "inputs":[{"name":"a","dtype":"f32","shape":[2]},{"name":"b","dtype":"f32","shape":[2]}],
            "nodes":[{"name":"s","op":"Add","inputs":["a","b"]},{"name":"p","op":"Mul","inputs":["s","b"]}],
            "outputs":["p"]}"#;
        assert_eq!(load(json).unwrap().values().len(), 4);
    }

    #[test]
    fn binary_op_with_one_input_is_an_arity_error() {
        let node = r#"{"name":"y","op":"Add","inputs":["x"]}"#;
        assert!(matches!(
            core_err(&graph(node, r#""y""#)),
            Error::Arity {
                expected: 2,
                found: 1,
                ..
            }
        ));
    }

    #[test]
    fn sigmoid_loads() {
        let node = r#"{"name":"y","op":"Sigmoid","inputs":["x"]}"#;
        assert!(load(&graph(node, r#""y""#)).is_ok());
    }

    #[test]
    fn matmul_loads_and_checks_shapes() {
        let json = r#"{"format_version":1,
            "inputs":[{"name":"a","dtype":"f32","shape":[2,3]},{"name":"b","dtype":"f32","shape":[4,5]}],
            "nodes":[{"name":"y","op":"MatMul","inputs":["a","b"]}],
            "outputs":["y"]}"#;
        assert!(matches!(core_err(json), Error::ShapeMismatch { .. }));
    }

    fn with_attrs(op: &str, attrs: &str) -> String {
        graph(
            &format!(r#"{{"name":"y","op":"{op}","inputs":["x"],"attrs":{attrs}}}"#),
            r#""y""#,
        )
    }

    #[test]
    fn reshape_loads_with_its_shape_attribute() {
        let g = load(&with_attrs("Reshape", r#"{"shape":[1,2]}"#)).unwrap();
        assert_eq!(g.value(g.outputs()[0]).ty.shape, vec![1, 2]);
    }

    #[test]
    fn reshape_missing_shape_is_rejected() {
        let node = r#"{"name":"y","op":"Reshape","inputs":["x"]}"#;
        assert_eq!(
            core_err(&graph(node, r#""y""#)),
            Error::MissingAttribute {
                node: "y".into(),
                op: "Reshape",
                attr: "shape"
            }
        );
    }

    #[test]
    fn relu_with_attrs_is_rejected() {
        assert_eq!(
            core_err(&with_attrs("Relu", r#"{"shape":[2]}"#)),
            Error::UnexpectedAttribute {
                node: "y".into(),
                op: "Relu",
                attr: "shape"
            }
        );
    }

    #[test]
    fn reshape_with_a_perm_attribute_is_rejected() {
        assert_eq!(
            core_err(&with_attrs("Reshape", r#"{"shape":[2],"perm":[0]}"#)),
            Error::UnexpectedAttribute {
                node: "y".into(),
                op: "Reshape",
                attr: "perm"
            }
        );
    }

    #[test]
    fn unknown_attribute_key_is_a_parse_error() {
        assert!(matches!(
            load(&with_attrs("Reshape", r#"{"shape":[2],"axes":[0]}"#)),
            Err(CliError::GraphJson(_))
        ));
    }

    #[test]
    fn negative_dimension_names_the_wildcard() {
        let err = core_err(&with_attrs("Reshape", r#"{"shape":[-1]}"#));
        assert_eq!(
            err.to_string(),
            "node `y`: Reshape attribute `shape`: entry -1 is not a non-negative integer (the -1 wildcard is not supported)"
        );
        assert_eq!(err.code(), "E020");
    }

    #[test]
    fn reshape_target_overflow_is_rejected() {
        let err = core_err(&with_attrs(
            "Reshape",
            r#"{"shape":[4294967296,4294967296]}"#,
        ));
        assert!(matches!(err, Error::ShapeMismatch { reason, .. } if reason.contains("overflows")));
    }

    #[test]
    fn transpose_loads_with_its_perm_attribute() {
        let g = load(&with_attrs("Transpose", r#"{"perm":[0]}"#)).unwrap();
        assert_eq!(g.value(g.outputs()[0]).ty.shape, vec![2]);
    }

    #[test]
    fn transpose_missing_perm_is_rejected() {
        let node = r#"{"name":"y","op":"Transpose","inputs":["x"]}"#;
        assert_eq!(
            core_err(&graph(node, r#""y""#)),
            Error::MissingAttribute {
                node: "y".into(),
                op: "Transpose",
                attr: "perm"
            }
        );
    }

    #[test]
    fn transpose_with_a_shape_attribute_is_rejected() {
        assert_eq!(
            core_err(&with_attrs("Transpose", r#"{"perm":[0],"shape":[2]}"#)),
            Error::UnexpectedAttribute {
                node: "y".into(),
                op: "Transpose",
                attr: "shape"
            }
        );
    }

    #[test]
    fn negative_permutation_entry_is_an_invalid_attribute() {
        let err = core_err(&with_attrs("Transpose", r#"{"perm":[-2]}"#));
        assert_eq!(
            err.to_string(),
            "node `y`: Transpose attribute `perm`: entry -2 is not a non-negative integer"
        );
    }

    #[test]
    fn a_file_exactly_at_the_size_limit_loads_and_one_byte_less_does_not() {
        let json = graph(&relu("y", "x"), r#""y""#);
        let exactly = Limits {
            max_file_bytes: json.len(),
            ..Limits::default()
        };
        assert!(load_graph(&json, &exactly).is_ok());
        let one_less = Limits {
            max_file_bytes: json.len() - 1,
            ..Limits::default()
        };
        assert!(matches!(
            load_graph(&json, &one_less),
            Err(CliError::Core(Error::LimitExceeded { what: "file bytes", found, limit }))
                if found == json.len() && limit == json.len() - 1
        ));
    }

    #[test]
    fn the_value_limit_is_checked_before_names_are_compared() {
        // One input, one constant, one node: 3 values, and a duplicate name.
        // The limit error must win; it is what makes the check cheap.
        let json = r#"{"format_version":1,
            "inputs":[{"name":"x","dtype":"f32","shape":[1]}],
            "constants":[{"name":"x","dtype":"f32","shape":[1],"values":[1]}],
            "nodes":[{"name":"y","op":"Relu","inputs":["x"]}],
            "outputs":["y"]}"#;
        let limits = Limits {
            max_values: 2,
            ..Limits::default()
        };
        assert!(matches!(
            load_graph(json, &limits),
            Err(CliError::Core(Error::LimitExceeded {
                what: "values",
                found: 3,
                limit: 2
            }))
        ));
    }

    #[test]
    fn the_wildcard_hint_appears_only_for_minus_one_in_a_reshape_shape() {
        let hint = "(the -1 wildcard is not supported)";
        let reshape_one = core_err(&with_attrs("Reshape", r#"{"shape":[-1]}"#)).to_string();
        assert!(reshape_one.ends_with(hint), "{reshape_one}");
        let reshape_two = core_err(&with_attrs("Reshape", r#"{"shape":[-2]}"#)).to_string();
        assert!(!reshape_two.contains(hint), "{reshape_two}");
        let perm_one = core_err(&with_attrs("Transpose", r#"{"perm":[-1]}"#)).to_string();
        assert_eq!(
            perm_one,
            "node `y`: Transpose attribute `perm`: entry -1 is not a non-negative integer"
        );
    }

    #[test]
    fn cycle_blame_skips_nodes_that_were_ordered_fine() {
        // `ok` comes first in the file and is not stuck; the cycle is a <-> b.
        let nodes = format!("{},{},{}", relu("ok", "x"), relu("a", "b"), relu("b", "a"));
        match core_err(&graph(&nodes, r#""a""#)) {
            Error::Cycle { node } => assert!(node == "a" || node == "b", "blamed {node}"),
            other => panic!("expected a cycle, got {other:?}"),
        }
    }

    #[test]
    fn cycle_blame_follows_the_stuck_dependency_not_the_first_one() {
        // a depends on `ok` (fine) first and on b (stuck) second.
        let add = r#"{"name":"a","op":"Add","inputs":["ok","b"]}"#;
        let nodes = format!("{},{},{}", relu("ok", "x"), add, relu("b", "a"));
        match core_err(&graph(&nodes, r#""a""#)) {
            Error::Cycle { node } => assert!(node == "a" || node == "b", "blamed {node}"),
            other => panic!("expected a cycle, got {other:?}"),
        }
    }

    #[test]
    fn unsupported_version_path_is_the_version_field() {
        let json = graph(&relu("y", "x"), r#""y""#)
            .replace("\"format_version\":1", "\"format_version\":2");
        assert_eq!(path_of(&json), "format_version");
    }

    #[test]
    fn duplicate_name_points_at_the_second_declaration() {
        let json = r#"{"format_version":1,
            "inputs":[{"name":"x","dtype":"f32","shape":[2]}],
            "constants":[{"name":"x","dtype":"f32","shape":[2],"values":[1.0,2.0]}],
            "nodes":[],"outputs":["x"]}"#;
        assert_eq!(
            located(json),
            (
                "constants[0].name".to_string(),
                Error::DuplicateName { name: "x".into() }
            )
        );
    }

    #[test]
    fn duplicate_node_name_points_at_the_later_node() {
        let nodes = format!("{},{}", relu("y", "x"), relu("y", "x"));
        assert_eq!(path_of(&graph(&nodes, r#""y""#)), "nodes[1].name");
    }

    #[test]
    fn repeated_reference_points_at_the_unknown_one() {
        let add = r#"{"name":"y","op":"Add","inputs":["x","nope"]}"#;
        assert_eq!(path_of(&graph(add, r#""y""#)), "nodes[0].inputs[1]");
    }

    #[test]
    fn unknown_reference_uses_the_node_index_not_the_order_it_was_sorted_into() {
        let nodes = format!("{},{}", relu("b", "a"), relu("a", "missing"));
        assert_eq!(path_of(&graph(&nodes, r#""b""#)), "nodes[1].inputs[0]");
    }

    #[test]
    fn unknown_output_path_names_the_output_index() {
        let json = graph(&relu("y", "x"), r#""y","nope""#);
        assert_eq!(path_of(&json), "outputs[1]");
    }

    #[test]
    fn no_outputs_path_is_the_outputs_field() {
        assert_eq!(path_of(&graph(&relu("y", "x"), "")), "outputs");
    }

    #[test]
    fn cycle_path_matches_the_blamed_node() {
        let nodes = format!("{},{},{}", relu("ok", "x"), relu("a", "b"), relu("b", "a"));
        let (path, error) = located(&graph(&nodes, r#""a""#));
        let Error::Cycle { node } = error else {
            panic!("expected a cycle")
        };
        let expected = if node == "a" { "nodes[1]" } else { "nodes[2]" };
        assert_eq!(path, expected);
    }

    #[test]
    fn arity_path_is_the_inputs_list() {
        let node = r#"{"name":"y","op":"Add","inputs":["x"]}"#;
        assert_eq!(path_of(&graph(node, r#""y""#)), "nodes[0].inputs");
    }

    #[test]
    fn unsupported_op_path_is_the_op_field() {
        let node = r#"{"name":"y","op":"Softmax","inputs":["x"]}"#;
        assert_eq!(path_of(&graph(node, r#""y""#)), "nodes[0].op");
    }

    #[test]
    fn attribute_errors_point_into_attrs() {
        let missing = r#"{"name":"y","op":"Reshape","inputs":["x"]}"#;
        assert_eq!(path_of(&graph(missing, r#""y""#)), "nodes[0].attrs");
        assert_eq!(
            path_of(&with_attrs("Relu", r#"{"shape":[2]}"#)),
            "nodes[0].attrs.shape"
        );
        assert_eq!(
            path_of(&with_attrs("Transpose", r#"{"perm":[-1]}"#)),
            "nodes[0].attrs.perm"
        );
    }

    #[test]
    fn shape_mismatch_path_is_the_node() {
        let nodes = format!(
            "{},{}",
            relu("a", "x"),
            r#"{"name":"m","op":"MatMul","inputs":["a","a"]}"#
        );
        assert_eq!(path_of(&graph(&nodes, r#""m""#)), "nodes[1]");
    }

    #[test]
    fn constant_errors_point_at_the_constant() {
        let short = r#"{"format_version":1,
            "constants":[{"name":"c","dtype":"f32","shape":[3],"values":[1.0]}],
            "nodes":[{"name":"r","op":"Relu","inputs":["c"]}],"outputs":["r"]}"#;
        assert_eq!(path_of(short), "constants[0].values");
        let huge = short.replace("[1.0]", "[1e39]").replace("[3]", "[1]");
        assert_eq!(path_of(&huge), "constants[0].values");
    }

    #[test]
    fn shape_errors_point_at_the_shape_of_the_input_or_constant() {
        let json = graph(&relu("y", "x"), r#""y""#).replace("[2]", "[0]");
        assert_eq!(path_of(&json), "inputs[0].shape");
    }

    #[test]
    fn whole_graph_limits_have_no_path() {
        let limits = Limits {
            max_values: 1,
            ..Limits::default()
        };
        let json = graph(&relu("y", "x"), r#""y""#);
        assert!(matches!(
            load_graph(&json, &limits),
            Err(CliError::Core(Error::LimitExceeded { .. }))
        ));
    }

    #[test]
    fn a_located_error_keeps_its_core_code_and_shows_the_path() {
        let err = load(&graph(&relu("y", "nope"), r#""y""#)).unwrap_err();
        assert_eq!(err.code(), "E003");
        assert!(
            err.to_string().ends_with("(at nodes[0].inputs[0])"),
            "{err}"
        );
    }

    #[test]
    fn duplicate_name_path_counts_within_its_own_section() {
        let json = r#"{"format_version":1,
            "inputs":[{"name":"a","dtype":"f32","shape":[2]},{"name":"b","dtype":"f32","shape":[2]}],
            "constants":[{"name":"c","dtype":"f32","shape":[2],"values":[1.0,2.0]}],
            "nodes":[{"name":"n","op":"Relu","inputs":["a"]},{"name":"b","op":"Relu","inputs":["a"]}],
            "outputs":["n"]}"#;
        assert_eq!(path_of(json), "nodes[1].name");
    }
}
