//! Reshape, Flatten and Transpose lowering.

mod common;

use common::*;
use oxfuse_core::Error;
use oxfuse_onnx::{OnnxError, proto::*};

fn iota(n: usize) -> Vec<f32> {
    (0..n).map(|i| i as f32).collect()
}

fn reshape_model(in_shape: &[i64], shape: &[i64], allowzero: Option<i64>) -> ModelProto {
    let mut n = node("Reshape", &["x", "s"], &["y"]);
    if let Some(a) = allowzero {
        n = with_int(n, "allowzero", a);
    }
    model(
        13,
        graph(
            vec![input("x", in_shape)],
            vec![init_i64("s", shape)],
            vec![n],
            &["y"],
        ),
    )
}

fn reshaped(in_shape: &[i64], shape: &[i64]) -> Vec<usize> {
    shape_of_output(&load_ok(&reshape_model(in_shape, shape, None)), 0)
}

fn shape_reason(r: Result<oxfuse_core::Graph, OnnxError>) -> String {
    match r {
        Err(OnnxError::Shape { reason, .. }) => reason,
        other => panic!("expected a shape error, got {:?}", other.map(|_| ())),
    }
}

#[test]
fn reshape_plain_keeps_element_order() {
    let g = load_ok(&reshape_model(&[2, 3], &[3, 2], None));
    assert_eq!(ops(&g), vec!["Reshape"]);
    let d = iota(6);
    assert_eq!(run(&g, &[(&[2, 3], &d)]), vec![d]);
    assert_eq!(shape_of_output(&g, 0), vec![3, 2]);
}

#[test]
fn reshape_zero_dim_copies_the_input_dimension() {
    assert_eq!(reshaped(&[2, 3, 4], &[0, 12]), vec![2, 12]);
    assert_eq!(reshaped(&[2, 3, 4], &[0, 0, 4]), vec![2, 3, 4]);
}

#[test]
fn reshape_minus_one_is_inferred() {
    assert_eq!(reshaped(&[2, 3, 4], &[-1, 4]), vec![6, 4]);
    assert_eq!(reshaped(&[2, 3, 4], &[2, -1]), vec![2, 12]);
    assert_eq!(reshaped(&[2, 3], &[-1]), vec![6]);
}

#[test]
fn reshape_zero_and_minus_one_together() {
    assert_eq!(reshaped(&[2, 3, 4], &[0, -1]), vec![2, 12]);
}

#[test]
fn reshape_two_minus_one_rejected() {
    let r = shape_reason(load(&reshape_model(&[2, 3], &[-1, -1], None)));
    assert!(r.contains("-1"), "{r}");
}

#[test]
fn reshape_dimension_below_minus_one_rejected() {
    let r = shape_reason(load(&reshape_model(&[2, 3], &[-2, 3], None)));
    assert!(r.contains("entry 0 is -2"), "{r}");
}

#[test]
fn reshape_minus_one_that_does_not_divide_rejected() {
    shape_reason(load(&reshape_model(&[2, 3], &[4, -1], None)));
}

#[test]
fn reshape_minus_one_next_to_a_zero_extent_rejected() {
    let r = shape_reason(load(&reshape_model(&[2, 3], &[0, -1, 0], Some(1))));
    assert!(r.contains("cannot combine"), "{r}");
}

#[test]
fn reshape_zero_beyond_the_input_rank_rejected() {
    let r = shape_reason(load(&reshape_model(&[2, 3], &[1, 1, 0], None)));
    assert!(r.contains("rank"), "{r}");
}

#[test]
fn reshape_allowzero_makes_zero_a_literal_extent() {
    // A literal zero extent can never hold the input's elements, and the IR
    // has no zero-size tensors, so this is an error where allowzero=0 copies.
    let r = shape_reason(load(&reshape_model(&[2, 3], &[0, 3], Some(1))));
    assert!(r.contains("zero"), "{r}");
    assert_eq!(
        shape_of_output(&load_ok(&reshape_model(&[2, 3], &[0, 3], Some(0))), 0),
        vec![2, 3]
    );
}

#[test]
fn reshape_allowzero_one_without_zeros_is_plain() {
    let g = load_ok(&reshape_model(&[2, 3], &[-1, 2], Some(1)));
    assert_eq!(shape_of_output(&g, 0), vec![3, 2]);
}

#[test]
fn reshape_allowzero_other_than_0_or_1_rejected() {
    shape_reason(load(&reshape_model(&[2, 3], &[6], Some(2))));
}

#[test]
fn reshape_element_count_mismatch() {
    assert!(matches!(
        load(&reshape_model(&[2, 3], &[4, 2], None)),
        Err(OnnxError::Core(Error::ShapeMismatch { .. }))
    ));
}

#[test]
fn reshape_shape_not_an_initializer_rejected() {
    let m = model(
        13,
        graph(
            vec![input("x", &[2, 3]), input("s", &[2])],
            vec![],
            vec![node("Reshape", &["x", "s"], &["y"])],
            &["y"],
        ),
    );
    let r = shape_reason(load(&m));
    assert!(r.contains("initializer"), "{r}");
}

#[test]
fn reshape_shape_not_int64_rejected() {
    let m = model(
        13,
        graph(
            vec![input("x", &[2, 3])],
            vec![init("s", &[2], &[3.0, 2.0])],
            vec![node("Reshape", &["x", "s"], &["y"])],
            &["y"],
        ),
    );
    let r = shape_reason(load(&m));
    assert!(r.contains("int64"), "{r}");
}

#[test]
fn reshape_shape_must_be_rank_one() {
    let mut t = init_i64("s", &[3, 2]);
    t.dims = vec![1, 2];
    let m = model(
        13,
        graph(
            vec![input("x", &[2, 3])],
            vec![t],
            vec![node("Reshape", &["x", "s"], &["y"])],
            &["y"],
        ),
    );
    shape_reason(load(&m));
}

#[test]
fn reshape_shape_from_a_constant_node_and_identity() {
    let c = NodeProto {
        attribute: vec![AttributeProto {
            name: "value_ints".into(),
            ints: vec![3, 2],
            r#type: attribute_type::INTS,
            ..AttributeProto::default()
        }],
        ..node("Constant", &[], &["k"])
    };
    let m = model(
        13,
        graph(
            vec![input("x", &[2, 3])],
            vec![],
            vec![
                c,
                node("Identity", &["k"], &["s"]),
                node("Reshape", &["x", "s"], &["y"]),
            ],
            &["y"],
        ),
    );
    assert_eq!(shape_of_output(&load_ok(&m), 0), vec![3, 2]);
}

#[test]
fn reshape_unknown_attribute_rejected() {
    let n = with_int(node("Reshape", &["x", "s"], &["y"]), "axis", 1);
    let m = model(
        13,
        graph(
            vec![input("x", &[2, 3])],
            vec![init_i64("s", &[6])],
            vec![n],
            &["y"],
        ),
    );
    assert!(matches!(load(&m), Err(OnnxError::UnsupportedAttr { .. })));
}

fn flatten(in_shape: &[i64], axis: Option<i64>) -> Result<oxfuse_core::Graph, OnnxError> {
    let mut n = node("Flatten", &["x"], &["y"]);
    if let Some(a) = axis {
        n = with_int(n, "axis", a);
    }
    load(&model(
        13,
        graph(vec![input("x", in_shape)], vec![], vec![n], &["y"]),
    ))
}

fn flattened(in_shape: &[i64], axis: Option<i64>) -> Vec<usize> {
    shape_of_output(&flatten(in_shape, axis).unwrap(), 0)
}

#[test]
fn flatten_axis_values() {
    assert_eq!(
        flattened(&[2, 3, 4], None),
        vec![2, 12],
        "default axis is 1"
    );
    assert_eq!(flattened(&[2, 3, 4], Some(0)), vec![1, 24]);
    assert_eq!(flattened(&[2, 3, 4], Some(1)), vec![2, 12]);
    assert_eq!(flattened(&[2, 3, 4], Some(2)), vec![6, 4]);
    assert_eq!(flattened(&[2, 3, 4], Some(3)), vec![24, 1], "axis == rank");
}

#[test]
fn flatten_negative_axis() {
    assert_eq!(flattened(&[2, 3, 4], Some(-1)), vec![6, 4]);
    assert_eq!(flattened(&[2, 3, 4], Some(-3)), vec![1, 24]);
}

#[test]
fn flatten_axis_out_of_range() {
    for axis in [4, -4, 100] {
        let r = shape_reason(flatten(&[2, 3, 4], Some(axis)));
        assert!(r.contains("axis"), "{r}");
    }
}

#[test]
fn flatten_keeps_element_order() {
    let g = flatten(&[2, 3], Some(1)).unwrap();
    let d = iota(6);
    assert_eq!(run(&g, &[(&[2, 3], &d)]), vec![d]);
}

fn transpose(in_shape: &[i64], perm: Option<&[i64]>) -> Result<oxfuse_core::Graph, OnnxError> {
    let mut n = node("Transpose", &["x"], &["y"]);
    if let Some(p) = perm {
        n = with_ints(n, "perm", p);
    }
    load(&model(
        13,
        graph(vec![input("x", in_shape)], vec![], vec![n], &["y"]),
    ))
}

#[test]
fn transpose_default_perm_reverses() {
    let g = transpose(&[2, 3, 4], None).unwrap();
    assert_eq!(shape_of_output(&g, 0), vec![4, 3, 2]);
    let g = transpose(&[2, 3], None).unwrap();
    let d = iota(6);
    assert_eq!(
        run(&g, &[(&[2, 3], &d)]),
        vec![vec![0.0, 3.0, 1.0, 4.0, 2.0, 5.0]]
    );
}

#[test]
fn transpose_explicit_perm() {
    let g = transpose(&[2, 3, 4], Some(&[1, 0, 2])).unwrap();
    assert_eq!(shape_of_output(&g, 0), vec![3, 2, 4]);
    let g = transpose(&[2, 3, 4], Some(&[0, 1, 2])).unwrap();
    assert_eq!(shape_of_output(&g, 0), vec![2, 3, 4]);
}

#[test]
fn transpose_invalid_perm_rejected() {
    for perm in [&[0, 0, 1][..], &[0, 1, 3], &[-1, 0, 1], &[1, 0]] {
        let r = shape_reason(transpose(&[2, 3, 4], Some(perm)));
        assert!(r.contains("perm"), "{perm:?}: {r}");
    }
}

#[test]
fn transpose_rank1_is_fine() {
    let g = transpose(&[5], None).unwrap();
    assert_eq!(shape_of_output(&g, 0), vec![5]);
}

fn transpose_with(type_: i32, perm: &[i64]) -> Result<oxfuse_core::Graph, OnnxError> {
    let mut n = node("Transpose", &["x"], &["y"]);
    n.attribute.push(AttributeProto {
        name: "perm".into(),
        ints: perm.to_vec(),
        r#type: type_,
        ..AttributeProto::default()
    });
    load(&model(
        13,
        graph(vec![input("x", &[2, 3])], vec![], vec![n], &["y"]),
    ))
}

#[test]
fn perm_attribute_follows_its_type_or_its_content() {
    // Typed, empty: the default (reversed) permutation.
    let g = transpose_with(attribute_type::INTS, &[]).unwrap();
    assert_eq!(shape_of_output(&g, 0), vec![3, 2]);
    // Typed and filled.
    let g = transpose_with(attribute_type::INTS, &[0, 1]).unwrap();
    assert_eq!(shape_of_output(&g, 0), vec![2, 3]);
    // Untyped but filled: read as given.
    let g = transpose_with(0, &[0, 1]).unwrap();
    assert_eq!(shape_of_output(&g, 0), vec![2, 3]);
    // Untyped and empty: no value at all.
    assert!(shape_reason(transpose_with(0, &[])).contains("no value"));
}
