//! Lowering of the exact ops: Relu, Sigmoid, MatMul, Add, Mul, Identity,
//! Dropout and Constant.

mod common;

use common::*;
use oxfuse_core::Error;
use oxfuse_onnx::{ImportOptions, OnnxError, import, proto::*};

fn single(
    op: &str,
    inputs: Vec<ValueInfoProto>,
    inits: Vec<TensorProto>,
    args: &[&str],
) -> ModelProto {
    model(
        13,
        graph(inputs, inits, vec![node(op, args, &["y"])], &["y"]),
    )
}

#[test]
fn relu_sigmoid_map_directly() {
    for (op, want) in [("Relu", "Relu"), ("Sigmoid", "Sigmoid")] {
        let g = load_ok(&single(op, vec![input("x", &[2, 3])], vec![], &["x"]));
        assert_eq!(ops(&g), vec![want]);
        assert_eq!(shape_of_output(&g, 0), vec![2, 3]);
        assert_eq!(g.output_names(), &["y".to_owned()]);
    }
    let g = load_ok(&single("Relu", vec![input("x", &[3])], vec![], &["x"]));
    assert_eq!(
        run(&g, &[(&[3], &[-1.0, 0.0, 2.0])]),
        vec![vec![0.0, 0.0, 2.0]]
    );
}

#[test]
fn wrong_arity_names_the_node() {
    let m = single(
        "Relu",
        vec![input("x", &[2]), input("z", &[2])],
        vec![],
        &["x", "z"],
    );
    match load(&m) {
        Err(OnnxError::Shape { node, reason }) => {
            assert!(node.contains("Relu"));
            assert!(reason.contains("1 input"), "{reason}");
        }
        other => panic!("{other:?}"),
    }
}

#[test]
fn matmul_rank2() {
    let m = single(
        "MatMul",
        vec![input("a", &[2, 3]), input("b", &[3, 2])],
        vec![],
        &["a", "b"],
    );
    let g = load_ok(&m);
    assert_eq!(ops(&g), vec!["MatMul"]);
    assert_eq!(shape_of_output(&g, 0), vec![2, 2]);
    let out = run(
        &g,
        &[
            (&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            (&[3, 2], &[1.0, 0.0, 0.0, 1.0, 1.0, 1.0]),
        ],
    );
    assert_eq!(out, vec![vec![4.0, 5.0, 10.0, 11.0]]);
}

#[test]
fn matmul_rank3_and_rank1_rejected() {
    for (a, b) in [(vec![2, 2, 2], vec![2, 2]), (vec![3], vec![3, 2])] {
        let m = single(
            "MatMul",
            vec![input("a", &a), input("b", &b)],
            vec![],
            &["a", "b"],
        );
        match load(&m) {
            Err(OnnxError::Shape { reason, .. }) => assert!(reason.contains("rank 2"), "{reason}"),
            other => panic!("{other:?}"),
        }
    }
}

#[test]
fn matmul_inner_dimension_mismatch_is_a_core_error() {
    let m = single(
        "MatMul",
        vec![input("a", &[2, 3]), input("b", &[2, 2])],
        vec![],
        &["a", "b"],
    );
    assert!(matches!(
        load(&m),
        Err(OnnxError::Core(Error::ShapeMismatch { .. }))
    ));
}

#[test]
fn add_equal_shape() {
    let m = single(
        "Add",
        vec![input("a", &[2]), input("b", &[2])],
        vec![],
        &["a", "b"],
    );
    let g = load_ok(&m);
    assert_eq!(ops(&g), vec!["Add"]);
    assert_eq!(
        run(&g, &[(&[2], &[1.0, 2.0]), (&[2], &[10.0, 20.0])]),
        vec![vec![11.0, 22.0]]
    );
}

#[test]
fn add_rank0_either_side() {
    for args in [["x", "s"], ["s", "x"]] {
        let m = single(
            "Add",
            vec![input("x", &[2, 2])],
            vec![init("s", &[], &[5.0])],
            &args,
        );
        let g = load_ok(&m);
        assert_eq!(ops(&g), vec!["Add"], "no expansion for a scalar");
        assert_eq!(
            g.values().len(),
            3,
            "x, the scalar, the sum: nothing expanded"
        );
        assert_eq!(shape_of_output(&g, 0), vec![2, 2]);
        assert_eq!(
            run(&g, &[(&[2, 2], &[1.0, 2.0, 3.0, 4.0])]),
            vec![vec![6.0, 7.0, 8.0, 9.0]]
        );
    }
}

#[test]
fn add_const_n_vs_mn_either_side() {
    for args in [["x", "bias"], ["bias", "x"]] {
        let m = single(
            "Add",
            vec![input("x", &[2, 3])],
            vec![init("bias", &[3], &[10.0, 20.0, 30.0])],
            &args,
        );
        let g = load_ok(&m);
        assert_eq!(shape_of_output(&g, 0), vec![2, 3]);
        assert_eq!(
            run(&g, &[(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])]),
            vec![vec![11.0, 22.0, 33.0, 14.0, 25.0, 36.0]]
        );
    }
}

#[test]
fn add_const_1n_vs_mn() {
    let m = single(
        "Add",
        vec![input("x", &[2, 3])],
        vec![init("bias", &[1, 3], &[10.0, 20.0, 30.0])],
        &["x", "bias"],
    );
    let g = load_ok(&m);
    assert_eq!(
        run(&g, &[(&[2, 3], &[0.0; 6])]),
        vec![vec![10.0, 20.0, 30.0, 10.0, 20.0, 30.0]]
    );
}

#[test]
fn add_const_m1_vs_mn() {
    let m = single(
        "Add",
        vec![input("x", &[2, 3])],
        vec![init("col", &[2, 1], &[100.0, 200.0])],
        &["x", "col"],
    );
    let g = load_ok(&m);
    assert_eq!(
        run(&g, &[(&[2, 3], &[0.0; 6])]),
        vec![vec![100.0, 100.0, 100.0, 200.0, 200.0, 200.0]]
    );
}

#[test]
fn add_const_rank1_of_size_one() {
    let m = single(
        "Add",
        vec![input("x", &[2, 2])],
        vec![init("one", &[1], &[7.0])],
        &["x", "one"],
    );
    let g = load_ok(&m);
    assert_eq!(run(&g, &[(&[2, 2], &[0.0; 4])]), vec![vec![7.0; 4]]);
}

#[test]
fn add_non_constant_broadcast_rejected() {
    let m = single(
        "Add",
        vec![input("a", &[2, 3]), input("b", &[3])],
        vec![],
        &["a", "b"],
    );
    match load(&m) {
        Err(OnnxError::Shape { reason, .. }) => {
            assert!(reason.contains("not a constant"), "{reason}")
        }
        other => panic!("{other:?}"),
    }
}

#[test]
fn add_incompatible_shapes_rejected() {
    let m = single(
        "Add",
        vec![input("a", &[2, 3])],
        vec![init("b", &[2], &[1.0, 2.0])],
        &["a", "b"],
    );
    match load(&m) {
        Err(OnnxError::Shape { reason, .. }) => assert!(reason.contains("cannot be broadcast")),
        other => panic!("{other:?}"),
    }
}

#[test]
fn add_both_constants_with_broadcast() {
    // [2,1] + [1,3] -> [2,3]; both are expanded, the output is a graph output.
    let g = model(
        13,
        graph(
            vec![input("x", &[2, 3])],
            vec![
                init("c", &[2, 1], &[1.0, 2.0]),
                init("r", &[1, 3], &[10.0, 20.0, 30.0]),
            ],
            vec![
                node("Add", &["c", "r"], &["s"]),
                node("Add", &["x", "s"], &["y"]),
            ],
            &["y"],
        ),
    );
    let g = load_ok(&g);
    assert_eq!(
        run(&g, &[(&[2, 3], &[0.0; 6])]),
        vec![vec![11.0, 21.0, 31.0, 12.0, 22.0, 32.0]]
    );
}

#[test]
fn mul_expands_like_add() {
    let m = single(
        "Mul",
        vec![input("x", &[2, 2])],
        vec![init("s", &[2], &[2.0, 3.0])],
        &["s", "x"],
    );
    let g = load_ok(&m);
    assert_eq!(ops(&g), vec!["Mul"]);
    assert_eq!(
        run(&g, &[(&[2, 2], &[1.0, 1.0, 5.0, 5.0])]),
        vec![vec![2.0, 3.0, 10.0, 15.0]]
    );
}

#[test]
fn expansion_over_the_total_limit_is_rejected_before_allocating() {
    let m = single(
        "Add",
        vec![input("x", &[4, 4])],
        vec![init("b", &[4], &[0.0; 4])],
        &["x", "b"],
    );
    // Input 16 + constant 4 = 20 elements exist; expanding adds 16 more.
    let tight = ImportOptions {
        limits: oxfuse_core::Limits {
            max_total_elements: 20,
            ..oxfuse_core::Limits::default()
        },
        ..ImportOptions::default()
    };
    match import(&bytes(&m), &tight) {
        Err(OnnxError::Limit {
            node,
            what,
            found,
            limit,
        }) => {
            assert!(node.contains("Add"));
            assert_eq!(what, "total tensor elements");
            assert_eq!((found, limit), (36, 20));
        }
        other => panic!("{other:?}"),
    }
}

#[test]
fn expansion_over_the_per_tensor_limit() {
    let opts = ImportOptions {
        limits: oxfuse_core::Limits {
            max_tensor_elements: 16,
            max_total_elements: 1000,
            ..oxfuse_core::Limits::default()
        },
        ..ImportOptions::default()
    };
    // Expanding to exactly 16 elements fits the cap.
    let fits = single(
        "Add",
        vec![input("x", &[4, 4])],
        vec![init("b", &[4], &[0.0; 4])],
        &["x", "b"],
    );
    import(&bytes(&fits), &opts).unwrap();
    // Two constants broadcast to [3, 4, 4] = 48 elements.
    let over = model(
        13,
        graph(
            vec![input("x", &[2])],
            vec![
                init("c", &[3, 1, 1], &[0.0; 3]),
                init("r", &[4, 4], &[0.0; 16]),
            ],
            vec![node("Add", &["c", "r"], &["y"])],
            &["y"],
        ),
    );
    match import(&bytes(&over), &opts) {
        Err(OnnxError::Limit {
            what, found, limit, ..
        }) => {
            assert_eq!(what, "expanded constant elements");
            assert_eq!((found, limit), (48, 16));
        }
        other => panic!("{other:?}"),
    }
}

#[test]
fn identity_in_the_middle() {
    let m = model(
        13,
        graph(
            vec![input("x", &[2])],
            vec![],
            vec![
                node("Identity", &["x"], &["a"]),
                node("Relu", &["a"], &["y"]),
            ],
            &["y"],
        ),
    );
    let g = load_ok(&m);
    assert_eq!(ops(&g), vec!["Relu"]);
}

#[test]
fn identity_at_the_output_keeps_the_onnx_name() {
    let m = model(
        13,
        graph(
            vec![input("x", &[2])],
            vec![],
            vec![
                node("Relu", &["x"], &["a"]),
                node("Identity", &["a"], &["out"]),
            ],
            &["out", "a"],
        ),
    );
    let g = load_ok(&m);
    assert_eq!(g.output_names(), &["out".to_owned(), "a".to_owned()]);
    assert_eq!(g.outputs()[0], g.outputs()[1]);
}

#[test]
fn output_that_is_an_input_through_identity_only() {
    let m = model(
        13,
        graph(
            vec![input("x", &[2])],
            vec![],
            vec![node("Identity", &["x"], &["y"])],
            &["y", "x"],
        ),
    );
    let g = load_ok(&m);
    assert!(ops(&g).is_empty());
    assert_eq!(
        run(&g, &[(&[2], &[1.0, 2.0])]),
        vec![vec![1.0, 2.0], vec![1.0, 2.0]]
    );
}

#[test]
fn dropout_inference_is_an_alias() {
    let m = single("Dropout", vec![input("x", &[2])], vec![], &["x"]);
    let g = load_ok(&m);
    assert!(ops(&g).is_empty());
    assert_eq!(run(&g, &[(&[2], &[3.0, 4.0])]), vec![vec![3.0, 4.0]]);
}

#[test]
fn dropout_ratio_input_and_seed_are_accepted() {
    let n = with_int(node("Dropout", &["x", "r"], &["y"]), "seed", 3);
    let m = model(
        13,
        graph(
            vec![input("x", &[2])],
            vec![init("r", &[], &[0.5])],
            vec![n],
            &["y"],
        ),
    );
    load_ok(&m);
}

#[test]
fn dropout_with_mask_output_rejected() {
    let m = model(
        13,
        graph(
            vec![input("x", &[2])],
            vec![],
            vec![node("Dropout", &["x"], &["y", "mask"])],
            &["y"],
        ),
    );
    match load(&m) {
        Err(OnnxError::Shape { reason, .. }) => assert!(reason.contains("mask")),
        other => panic!("{other:?}"),
    }
}

#[test]
fn dropout_training_mode_input_rejected() {
    let m = model(
        13,
        graph(
            vec![input("x", &[2]), input("tm", &[1])],
            vec![],
            vec![node("Dropout", &["x", "", "tm"], &["y"])],
            &["y"],
        ),
    );
    match load(&m) {
        Err(OnnxError::Shape { reason, .. }) => assert!(reason.contains("training_mode")),
        other => panic!("{other:?}"),
    }
}

fn constant_node(attr: AttributeProto) -> NodeProto {
    NodeProto {
        attribute: vec![attr],
        ..node("Constant", &[], &["k"])
    }
}

fn constant_model(n: NodeProto, outputs: &[&str]) -> ModelProto {
    model(
        13,
        graph(
            vec![input("x", &[2])],
            vec![],
            vec![n, node("Add", &["x", "k"], &["y"])],
            outputs,
        ),
    )
}

#[test]
fn constant_node_tensor_value() {
    let attr = AttributeProto {
        name: "value".into(),
        t: Some(init("", &[2], &[10.0, 20.0])),
        r#type: attribute_type::TENSOR,
        ..AttributeProto::default()
    };
    let g = load_ok(&constant_model(constant_node(attr), &["y"]));
    assert_eq!(run(&g, &[(&[2], &[1.0, 2.0])]), vec![vec![11.0, 22.0]]);
}

#[test]
fn constant_scalar_and_vector_forms() {
    let attr = AttributeProto {
        name: "value_float".into(),
        f: Some(2.5),
        r#type: attribute_type::FLOAT,
        ..AttributeProto::default()
    };
    let g = load_ok(&constant_model(constant_node(attr), &["y"]));
    assert_eq!(run(&g, &[(&[2], &[1.0, 2.0])]), vec![vec![3.5, 4.5]]);
    let attr = AttributeProto {
        name: "value_floats".into(),
        floats: vec![1.0, 2.0],
        r#type: attribute_type::FLOATS,
        ..AttributeProto::default()
    };
    let g = load_ok(&constant_model(constant_node(attr), &["y"]));
    assert_eq!(run(&g, &[(&[2], &[1.0, 2.0])]), vec![vec![2.0, 4.0]]);
}

#[test]
fn constant_as_graph_output() {
    let attr = AttributeProto {
        name: "value_float".into(),
        f: Some(1.0),
        r#type: attribute_type::FLOAT,
        ..AttributeProto::default()
    };
    let g = load_ok(&constant_model(constant_node(attr), &["k", "y"]));
    assert_eq!(g.output_names()[0], "k");
    assert_eq!(
        run(&g, &[(&[2], &[1.0, 2.0])]),
        vec![vec![1.0], vec![2.0, 3.0]]
    );
}

#[test]
fn constant_with_no_or_two_values_rejected() {
    let n = node("Constant", &[], &["k"]);
    assert!(matches!(
        load(&constant_model(n, &["y"])),
        Err(OnnxError::Shape { .. })
    ));
    let mut n = constant_node(AttributeProto {
        name: "value_float".into(),
        f: Some(1.0),
        r#type: attribute_type::FLOAT,
        ..AttributeProto::default()
    });
    n.attribute.push(AttributeProto {
        name: "value_floats".into(),
        floats: vec![1.0, 2.0],
        r#type: attribute_type::FLOATS,
        ..AttributeProto::default()
    });
    assert!(matches!(
        load(&constant_model(n, &["y"])),
        Err(OnnxError::Shape { .. })
    ));
}

#[test]
fn constant_string_value_rejected_naming_the_attribute() {
    let attr = AttributeProto {
        name: "value_string".into(),
        s: Some(b"hi".to_vec()),
        r#type: attribute_type::STRING,
        ..AttributeProto::default()
    };
    match load(&constant_model(constant_node(attr), &["y"])) {
        Err(OnnxError::UnsupportedAttr { op, attr, .. }) => {
            assert_eq!(op, "Constant");
            assert_eq!(attr, "value_string");
        }
        // The node fails on "no supported value attribute" first when the
        // unknown attribute is the only one; either way it names the node.
        Err(OnnxError::Shape { node, .. }) => assert!(node.contains("Constant")),
        other => panic!("{other:?}"),
    }
}

#[test]
fn unknown_attribute_rejected_naming_node() {
    let n = with_float(node("Relu", &["x"], &["y"]), "alpha", 0.1);
    let m = model(13, graph(vec![input("x", &[2])], vec![], vec![n], &["y"]));
    match load(&m) {
        Err(OnnxError::UnsupportedAttr { node, op, attr }) => {
            assert!(node.contains("Relu"));
            assert_eq!(op, "Relu");
            assert_eq!(attr, "alpha");
        }
        other => panic!("{other:?}"),
    }
}

#[test]
fn int64_initializer_used_as_data_rejected() {
    let m = single(
        "Add",
        vec![input("x", &[2])],
        vec![init_i64("k", &[1, 2])],
        &["x", "k"],
    );
    match load(&m) {
        Err(OnnxError::Dtype { what, .. }) => assert_eq!(what, "k"),
        other => panic!("{other:?}"),
    }
}

#[test]
fn int64_initializer_as_graph_output_rejected() {
    let m = model(
        13,
        graph(
            vec![input("x", &[2])],
            vec![init_i64("k", &[1, 2])],
            vec![],
            &["k"],
        ),
    );
    assert!(matches!(load(&m), Err(OnnxError::Dtype { .. })));
}

#[test]
fn unsupported_op_names_node_and_op() {
    let mut n = node("Softmax", &["x"], &["y"]);
    n.name = "head".into();
    let m = model(13, graph(vec![input("x", &[2])], vec![], vec![n], &["y"]));
    assert_eq!(
        load(&m).unwrap_err(),
        OnnxError::UnsupportedOp {
            node: "head".into(),
            op: "Softmax".into()
        }
    );
}

#[test]
fn unused_initializer_stays_out_of_the_graph() {
    let mut m = single("Relu", vec![input("x", &[2])], vec![], &["x"]);
    m.graph
        .as_mut()
        .unwrap()
        .initializer
        .push(init("spare", &[2], &[1.0, 2.0]));
    let g = load_ok(&m);
    assert_eq!(g.values().len(), 2);
}

#[test]
fn initializer_used_twice_is_one_constant() {
    let m = model(
        13,
        graph(
            vec![input("x", &[2])],
            vec![init("w", &[2], &[1.0, 2.0])],
            vec![
                node("Add", &["x", "w"], &["a"]),
                node("Mul", &["a", "w"], &["y"]),
            ],
            &["y"],
        ),
    );
    let g = load_ok(&m);
    assert_eq!(g.values().len(), 4, "x, w, a, y");
}

#[test]
fn unused_graph_input_is_kept() {
    let m = model(
        13,
        graph(
            vec![input("x", &[2]), input("unused", &[2])],
            vec![],
            vec![node("Relu", &["x"], &["y"])],
            &["y"],
        ),
    );
    assert_eq!(load_ok(&m).inputs().len(), 2);
}

#[test]
fn nan_and_infinity_initializers_are_accepted() {
    let m = single(
        "Add",
        vec![input("x", &[2])],
        vec![init("w", &[2], &[f32::NAN, f32::INFINITY])],
        &["x", "w"],
    );
    load_ok(&m);
}

#[test]
fn runtime_scalar_operand_needs_no_expansion() {
    let m = single(
        "Add",
        vec![input("x", &[2, 2]), input("s", &[])],
        vec![],
        &["x", "s"],
    );
    let g = load_ok(&m);
    assert_eq!(g.values().len(), 3);
    assert_eq!(
        run(&g, &[(&[2, 2], &[1.0, 2.0, 3.0, 4.0]), (&[], &[10.0])]),
        vec![vec![11.0, 12.0, 13.0, 14.0]]
    );
}

#[test]
fn constant_expansion_keeps_every_axis_stride() {
    // A rank 2 constant stretched along a new leading axis: both of its
    // axes must keep their own stride.
    let m = single(
        "Add",
        vec![input("x", &[2, 2, 3])],
        vec![init("c", &[2, 3], &[1.0, 2.0, 3.0, 10.0, 20.0, 30.0])],
        &["x", "c"],
    );
    let g = load_ok(&m);
    assert_eq!(
        run(&g, &[(&[2, 2, 3], &[0.0; 12])]),
        vec![vec![
            1.0, 2.0, 3.0, 10.0, 20.0, 30.0, 1.0, 2.0, 3.0, 10.0, 20.0, 30.0
        ]]
    );
}

fn shape_reason(r: Result<oxfuse_core::Graph, OnnxError>) -> String {
    match r {
        Err(OnnxError::Shape { reason, .. }) => reason,
        other => panic!("expected a shape error, got {:?}", other.map(|_| ())),
    }
}

fn dropout_model(inputs: &[&str], outputs: &[&str]) -> ModelProto {
    model(
        13,
        graph(
            vec![input("x", &[2]), input("r", &[]), input("tm", &[1])],
            vec![],
            vec![node("Dropout", inputs, outputs)],
            &[outputs[0]],
        ),
    )
}

#[test]
fn dropout_input_count_is_checked() {
    for inputs in [&[][..], &["x", "r", "", "x"], &["", "r"]] {
        let r = shape_reason(load(&dropout_model(inputs, &["y"])));
        assert!(r.contains("1 to 3 inputs"), "{inputs:?}: {r}");
    }
}

#[test]
fn dropout_extra_outputs_are_rejected_and_an_absent_mask_is_fine() {
    let r = shape_reason(load(&dropout_model(&["x"], &["y", "mask", "extra"])));
    assert!(r.contains("mask"), "{r}");
    // An empty second output name is ONNX's way of leaving the mask out.
    let g = load_ok(&dropout_model(&["x"], &["y", ""]));
    assert_eq!(
        run(&g, &[(&[2], &[1.0, 2.0]), (&[], &[0.5]), (&[1], &[0.0])]),
        vec![vec![1.0, 2.0]]
    );
}

#[test]
fn constant_node_shape_is_checked() {
    let attr = || AttributeProto {
        name: "value_float".into(),
        f: Some(1.0),
        r#type: attribute_type::FLOAT,
        ..AttributeProto::default()
    };
    let with_input = NodeProto {
        attribute: vec![attr()],
        ..node("Constant", &["x"], &["k"])
    };
    shape_reason(load(&constant_model(with_input, &["y"])));
    let two_outputs = NodeProto {
        attribute: vec![attr()],
        ..node("Constant", &[], &["k", "k2"])
    };
    shape_reason(load(&constant_model(two_outputs, &["y"])));
}

#[test]
fn constant_value_count_is_named_in_the_error() {
    let none = node("Constant", &[], &["k"]);
    assert!(shape_reason(load(&constant_model(none, &["y"]))).contains("no supported"));
    let mut two = constant_node(AttributeProto {
        name: "value_float".into(),
        f: Some(1.0),
        r#type: attribute_type::FLOAT,
        ..AttributeProto::default()
    });
    two.attribute.push(AttributeProto {
        name: "value_floats".into(),
        floats: vec![1.0],
        r#type: attribute_type::FLOATS,
        ..AttributeProto::default()
    });
    assert!(shape_reason(load(&constant_model(two, &["y"]))).contains("more than one"));
}

#[test]
fn attribute_of_the_wrong_type_is_rejected_and_an_untyped_one_is_read() {
    // `alpha` must be a float; an int attribute of that name is a mistake.
    let mut n = node("Gemm", &["a", "b"], &["y"]);
    n = with_int(n, "alpha", 2);
    let m = |n| {
        model(
            13,
            graph(
                vec![input("a", &[2, 2]), input("b", &[2, 2])],
                vec![],
                vec![n],
                &["y"],
            ),
        )
    };
    assert!(shape_reason(load(&m(n))).contains("has type"));
    // Older exporters leave the type field unset; the value field decides.
    let mut n = node("Gemm", &["a", "b"], &["y"]);
    n.attribute.push(AttributeProto {
        name: "alpha".into(),
        f: Some(2.0),
        r#type: 0,
        ..AttributeProto::default()
    });
    let g = load_ok(&m(n));
    assert!(ops(&g).contains(&"Mul"), "alpha 2.0 must be applied");
}

#[test]
fn attribute_without_a_value_of_its_type_is_rejected() {
    let m = |attr| {
        let mut n = node("Gemm", &["a", "b"], &["y"]);
        n.attribute.push(attr);
        model(
            13,
            graph(
                vec![input("a", &[2, 2]), input("b", &[2, 2])],
                vec![],
                vec![n],
                &["y"],
            ),
        )
    };
    // Typed float, but the float field is missing.
    let typed_empty = AttributeProto {
        name: "alpha".into(),
        r#type: attribute_type::FLOAT,
        ..AttributeProto::default()
    };
    assert!(shape_reason(load(&m(typed_empty))).contains("no value"));
    // Untyped, and the value sits in the wrong field: not read as alpha = 1.
    let untyped_wrong = AttributeProto {
        name: "alpha".into(),
        i: Some(5),
        ..AttributeProto::default()
    };
    assert!(shape_reason(load(&m(untyped_wrong))).contains("no value"));
}

fn gemm_with_attr(attr: AttributeProto) -> ModelProto {
    let mut n = node("Gemm", &["a", "b"], &["y"]);
    n.attribute.push(attr);
    model(
        13,
        graph(
            vec![input("a", &[2, 2]), input("b", &[2, 2])],
            vec![],
            vec![n],
            &["y"],
        ),
    )
}

#[test]
fn typed_int_attribute_without_its_value_is_rejected() {
    let attr = AttributeProto {
        name: "transA".into(),
        r#type: attribute_type::INT,
        ..AttributeProto::default()
    };
    assert!(shape_reason(load(&gemm_with_attr(attr))).contains("no value"));
}

#[test]
fn typed_tensor_attribute_without_its_value_is_rejected() {
    let attr = AttributeProto {
        name: "value".into(),
        r#type: attribute_type::TENSOR,
        ..AttributeProto::default()
    };
    assert!(shape_reason(load(&constant_model(constant_node(attr), &["y"]))).contains("no value"));
}

#[test]
fn list_attributes_follow_their_type_or_their_content() {
    let floats = |type_, values: Vec<f32>| {
        constant_model(
            constant_node(AttributeProto {
                name: "value_floats".into(),
                floats: values,
                r#type: type_,
                ..AttributeProto::default()
            }),
            &["y"],
        )
    };
    // Untyped but filled: read as given.
    let g = load_ok(&floats(0, vec![1.0, 2.0]));
    assert_eq!(run(&g, &[(&[2], &[0.0, 0.0])]), vec![vec![1.0, 2.0]]);
    // Untyped and empty: no value.
    assert!(shape_reason(load(&floats(0, vec![]))).contains("no value"));
    // Typed and empty is a real empty list; the IR then refuses a zero-size constant.
    assert!(matches!(
        load(&floats(attribute_type::FLOATS, vec![])),
        Err(OnnxError::Core(Error::InvalidShape { .. }))
    ));
}
