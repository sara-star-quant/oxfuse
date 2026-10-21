//! Model-level checks: opsets, names, ordering, dimensions.

mod common;

use common::*;
use oxfuse_onnx::{ImportOptions, OnnxError, model::check_model, proto::*};

fn opts() -> ImportOptions {
    ImportOptions::default()
}

fn relu_graph() -> GraphProto {
    graph(
        vec![input("x", &[2])],
        vec![],
        vec![node("Relu", &["x"], &["y"])],
        &["y"],
    )
}

fn err(m: ModelProto) -> OnnxError {
    check_model(m, &opts()).unwrap_err()
}

fn structure_reason(e: OnnxError) -> String {
    match e {
        OnnxError::Structure { reason } => reason,
        other => panic!("expected a structure error, got {other:?}"),
    }
}

fn name_reason(e: OnnxError) -> String {
    match e {
        OnnxError::Name { reason } => reason,
        other => panic!("expected a name error, got {other:?}"),
    }
}

#[test]
fn missing_graph() {
    let m = ModelProto {
        graph: None,
        ..model(13, relu_graph())
    };
    assert!(structure_reason(err(m)).contains("no graph"));
}

#[test]
fn missing_opset() {
    let m = ModelProto {
        opset_import: vec![],
        ..model(13, relu_graph())
    };
    assert_eq!(
        err(m),
        OnnxError::Opset {
            found: "missing".into()
        }
    );
}

#[test]
fn opset_12_rejected() {
    assert_eq!(
        err(model(12, relu_graph())),
        OnnxError::Opset { found: "12".into() }
    );
}

#[test]
fn opset_22_rejected() {
    assert_eq!(
        err(model(22, relu_graph())),
        OnnxError::Opset { found: "22".into() }
    );
}

#[test]
fn opset_13_and_21_accepted() {
    check_model(model(13, relu_graph()), &opts()).unwrap();
    check_model(model(21, relu_graph()), &opts()).unwrap();
}

#[test]
fn opset_ai_onnx_domain_counts_as_default() {
    let mut m = model(13, relu_graph());
    m.opset_import[0].domain = "ai.onnx".into();
    check_model(m, &opts()).unwrap();
}

#[test]
fn conflicting_default_opsets_rejected() {
    let mut m = model(13, relu_graph());
    m.opset_import.push(OperatorSetIdProto {
        domain: String::new(),
        version: Some(14),
    });
    structure_reason(err(m));
}

#[test]
fn opset_entry_without_version_rejected() {
    let mut m = model(13, relu_graph());
    m.opset_import[0].version = None;
    assert!(matches!(err(m), OnnxError::Opset { .. }));
}

#[test]
fn non_default_domain_node_rejected() {
    let mut g = relu_graph();
    g.node[0].domain = "com.microsoft".into();
    match err(model(13, g)) {
        OnnxError::Domain { node, domain } => {
            assert!(node.contains("Relu"));
            assert_eq!(domain, "com.microsoft");
        }
        other => panic!("{other:?}"),
    }
}

#[test]
fn initializer_also_listed_as_input_is_a_constant() {
    let g = graph(
        vec![input("x", &[2]), input("w", &[2])],
        vec![init("w", &[2], &[1.0, 2.0])],
        vec![node("Add", &["x", "w"], &["y"])],
        &["y"],
    );
    let m = check_model(model(13, g), &opts()).unwrap();
    assert_eq!(m.inputs, vec![("x".to_owned(), vec![2])]);
    assert_eq!(m.initializers.len(), 1);
    assert_eq!(m.initializers[0].0, "w");
}

#[test]
fn unused_initializer_is_kept_in_the_model() {
    let mut g = relu_graph();
    g.initializer.push(init("spare", &[1], &[0.0]));
    let m = check_model(model(13, g), &opts()).unwrap();
    assert_eq!(m.initializers.len(), 1);
}

#[test]
fn non_float_initializer_rejected_even_if_unused() {
    let mut g = relu_graph();
    let mut t = init("spare", &[1], &[]);
    t.data_type = 11;
    t.double_data = vec![1.0];
    g.initializer.push(t);
    assert!(matches!(err(model(13, g)), OnnxError::Dtype { .. }));
}

#[test]
fn sparse_initializer_rejected() {
    let mut g = relu_graph();
    g.sparse_initializer.push(SparseTensorProto {});
    assert!(structure_reason(err(model(13, g))).contains("sparse"));
}

#[test]
fn graph_output_that_is_graph_input() {
    let g = graph(vec![input("x", &[2])], vec![], vec![], &["x"]);
    let m = check_model(model(13, g), &opts()).unwrap();
    assert_eq!(m.outputs, vec!["x".to_owned()]);
}

#[test]
fn graph_output_listed_twice() {
    let mut g = relu_graph();
    g.output.push(g.output[0].clone());
    let m = check_model(model(13, g), &opts()).unwrap();
    assert_eq!(m.outputs, vec!["y".to_owned(), "y".to_owned()]);
}

#[test]
fn unknown_graph_output_rejected() {
    let mut g = relu_graph();
    g.output[0].name = "nope".into();
    assert!(structure_reason(err(model(13, g))).contains("nope"));
}

#[test]
fn empty_graph_output_name_rejected() {
    let mut g = relu_graph();
    g.output[0].name = String::new();
    structure_reason(err(model(13, g)));
}

#[test]
fn no_graph_outputs_rejected() {
    let mut g = relu_graph();
    g.output.clear();
    structure_reason(err(model(13, g)));
}

#[test]
fn non_float_graph_output_type_rejected() {
    let mut g = relu_graph();
    g.output[0].r#type = Some(TypeProto {
        tensor_type: Some(TypeTensor {
            elem_type: 7,
            shape: None,
        }),
    });
    assert!(matches!(err(model(13, g)), OnnxError::Dtype { .. }));
}

#[test]
fn duplicate_value_names() {
    let g = graph(
        vec![input("x", &[2])],
        vec![],
        vec![node("Relu", &["x"], &["y"]), node("Relu", &["x"], &["y"])],
        &["y"],
    );
    let r = name_reason(err(model(13, g)));
    assert!(
        r.contains("`y`") && r.contains("node 0") && r.contains("node 1"),
        "{r}"
    );
}

#[test]
fn duplicate_input_and_input_vs_node_names() {
    let g = graph(
        vec![input("x", &[2]), input("x", &[2])],
        vec![],
        vec![],
        &["x"],
    );
    name_reason(err(model(13, g)));
    let g = graph(
        vec![input("x", &[2])],
        vec![],
        vec![node("Relu", &["x"], &["x"])],
        &["x"],
    );
    name_reason(err(model(13, g)));
}

#[test]
fn duplicate_initializer_names() {
    let g = graph(
        vec![input("x", &[1])],
        vec![init("w", &[1], &[1.0]), init("w", &[1], &[2.0])],
        vec![node("Relu", &["x"], &["y"])],
        &["y"],
    );
    name_reason(err(model(13, g)));
}

#[test]
fn empty_value_name_gets_a_generated_name() {
    let g = graph(
        vec![input("x", &[2])],
        vec![],
        vec![node("Relu", &["x"], &[""])],
        &["v1"],
    );
    let m = check_model(model(13, g), &opts()).unwrap();
    assert_eq!(m.nodes[0].outputs, vec!["v1".to_owned()]);
}

#[test]
fn generated_name_collision_names_both_positions() {
    // Ordinals: a = 0, v2 = 1, the nameless output = 2, which would be `v2`.
    let g = graph(
        vec![input("a", &[2]), input("v2", &[2])],
        vec![],
        vec![node("Relu", &["a"], &[""])],
        &["v2"],
    );
    let r = name_reason(err(model(13, g)));
    assert!(
        r.contains("`v2`") && r.contains("input 1") && r.contains("node 0"),
        "{r}"
    );
}

#[test]
fn nodes_not_topological_rejected() {
    let g = graph(
        vec![input("x", &[2])],
        vec![],
        vec![node("Relu", &["b"], &["c"]), node("Relu", &["x"], &["b"])],
        &["c"],
    );
    assert!(structure_reason(err(model(13, g))).contains("topological"));
}

#[test]
fn input_never_defined_rejected() {
    let g = graph(
        vec![input("x", &[2])],
        vec![],
        vec![node("Add", &["x", "zz"], &["y"])],
        &["y"],
    );
    assert!(structure_reason(err(model(13, g))).contains("unknown value `zz`"));
}

#[test]
fn absent_optional_input_is_not_an_unknown_value() {
    let g = graph(
        vec![input("x", &[2])],
        vec![],
        vec![node("Dropout", &["x", ""], &["y"])],
        &["y"],
    );
    let m = check_model(model(13, g), &opts()).unwrap();
    assert_eq!(m.nodes[0].inputs, vec!["x".to_owned(), String::new()]);
}

fn named(dim: &'static str) -> GraphProto {
    graph(
        vec![value_info("x", &[D::P(dim), D::V(3)])],
        vec![],
        vec![node("Relu", &["x"], &["y"])],
        &["y"],
    )
}

fn with_dims(pairs: &[(&str, usize)]) -> ImportOptions {
    let mut o = ImportOptions::default();
    for (k, v) in pairs {
        o.dims.insert((*k).into(), *v);
    }
    o
}

#[test]
fn named_dim_without_dim_flag() {
    assert_eq!(
        err(model(13, named("N"))),
        OnnxError::DimUnbound { name: "N".into() }
    );
}

#[test]
fn named_dim_with_dim_flag() {
    let m = check_model(model(13, named("N")), &with_dims(&[("N", 4)])).unwrap();
    assert_eq!(m.inputs, vec![("x".to_owned(), vec![4, 3])]);
}

#[test]
fn named_dim_set_to_zero_rejected() {
    let e = check_model(model(13, named("N")), &with_dims(&[("N", 0)])).unwrap_err();
    assert!(structure_reason(e).contains("at least 1"));
}

#[test]
fn two_inputs_share_a_named_dim() {
    let g = graph(
        vec![
            value_info("a", &[D::P("N"), D::V(2)]),
            value_info("b", &[D::P("N")]),
        ],
        vec![],
        vec![node("Relu", &["a"], &["y"])],
        &["y"],
    );
    let m = check_model(model(13, g), &with_dims(&[("N", 5)])).unwrap();
    assert_eq!(m.inputs[0].1, vec![5, 2]);
    assert_eq!(m.inputs[1].1, vec![5]);
}

#[test]
fn unused_dim_flag_is_ignored() {
    check_model(model(13, relu_graph()), &with_dims(&[("N", 4)])).unwrap();
}

#[test]
fn unnamed_unknown_dim_rejected() {
    let g = graph(
        vec![value_info("x", &[D::Unknown])],
        vec![],
        vec![node("Relu", &["x"], &["y"])],
        &["y"],
    );
    assert!(structure_reason(err(model(13, g))).contains("unknown"));
    let g = graph(
        vec![value_info("x", &[D::P("")])],
        vec![],
        vec![node("Relu", &["x"], &["y"])],
        &["y"],
    );
    structure_reason(err(model(13, g)));
}

#[test]
fn negative_and_zero_dim_values_rejected() {
    for bad in [-1, 0] {
        let g = graph(
            vec![input("x", &[2, bad])],
            vec![],
            vec![node("Relu", &["x"], &["y"])],
            &["y"],
        );
        structure_reason(err(model(13, g)));
    }
}

#[test]
fn input_without_a_shape_or_type_rejected() {
    let mut g = relu_graph();
    g.input[0]
        .r#type
        .as_mut()
        .unwrap()
        .tensor_type
        .as_mut()
        .unwrap()
        .shape = None;
    structure_reason(err(model(13, g)));
    let mut g = relu_graph();
    g.input[0].r#type = None;
    structure_reason(err(model(13, g)));
}

#[test]
fn non_float_input_rejected_naming_the_input() {
    let mut g = relu_graph();
    g.input[0]
        .r#type
        .as_mut()
        .unwrap()
        .tensor_type
        .as_mut()
        .unwrap()
        .elem_type = 7;
    match err(model(13, g)) {
        OnnxError::Dtype { what, dtype } => {
            assert_eq!(what, "x");
            assert_eq!(dtype, "INT64");
        }
        other => panic!("{other:?}"),
    }
}

#[test]
fn empty_secondary_output_is_absent_not_generated() {
    let g = graph(
        vec![input("x", &[2])],
        vec![],
        vec![node("Dropout", &["x"], &["y", ""])],
        &["y"],
    );
    let m = check_model(model(13, g), &opts()).unwrap();
    assert_eq!(m.nodes[0].outputs, vec!["y".to_owned(), String::new()]);
}

#[test]
fn node_with_no_outputs_rejected() {
    let g = graph(
        vec![input("x", &[2])],
        vec![],
        vec![node("Dropout", &["x"], &[])],
        &["x"],
    );
    assert!(structure_reason(err(model(13, g))).contains("no outputs"));
}

#[test]
fn int64_initializer_listed_as_input_is_a_constant() {
    // Exporters that keep initializers as inputs list every weight, int64
    // shape operands included. The initializer decides; the declared input
    // type and shape are not consulted.
    let mut s = input("s", &[1]);
    s.r#type
        .as_mut()
        .unwrap()
        .tensor_type
        .as_mut()
        .unwrap()
        .elem_type = 7;
    let g = graph(
        vec![input("x", &[2, 3]), s],
        vec![init_i64("s", &[6])],
        vec![node("Reshape", &["x", "s"], &["y"])],
        &["y"],
    );
    let m = check_model(model(13, g), &opts()).unwrap();
    assert_eq!(m.inputs, vec![("x".to_owned(), vec![2, 3])]);
}

#[test]
fn shadowed_input_with_an_unbound_named_dimension_is_fine() {
    let g = graph(
        vec![value_info("x", &[D::V(2)]), value_info("w", &[D::P("N")])],
        vec![init("w", &[2], &[1.0, 2.0])],
        vec![node("Add", &["x", "w"], &["y"])],
        &["y"],
    );
    check_model(model(13, g), &opts()).unwrap();
}
