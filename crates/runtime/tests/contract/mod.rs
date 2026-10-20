//! The contract every backend must meet. Each check takes the backend as a
//! trait object; `contract_tests!` instantiates all of them per backend. The
//! third argument disables the instances when a backend is not available on
//! this machine (a GPU backend on a machine with no adapter).
#![allow(dead_code, unused_imports)]

use oxfuse_core::{Error, Graph, GraphBuilder, Limits, Op};
use oxfuse_runtime::{Backend, Tensor};

fn run(backend: &dyn Backend, graph: &Graph, inputs: &[Tensor]) -> Result<Vec<Tensor>, Error> {
    backend.compile(graph)?.run(inputs)
}

fn bits(tensor: &Tensor) -> Vec<u32> {
    tensor.data().iter().map(|v| v.to_bits()).collect()
}

fn relu_graph(shape: Vec<usize>) -> Graph {
    let mut b = GraphBuilder::new(Limits::default());
    let x = b.input("x", shape).unwrap();
    let y = b.node("y", Op::Relu, &[x]).unwrap();
    b.finish(&[y]).unwrap()
}

pub fn relu_clamps_negatives(backend: &dyn Backend) {
    let input = Tensor::new(vec![2, 3], vec![-1.5, 0.0, 2.5, -3.0, 3.0, -0.5]).unwrap();
    let out = run(backend, &relu_graph(vec![2, 3]), &[input]).unwrap();
    assert_eq!(out[0].shape(), &[2, 3]);
    assert_eq!(out[0].data(), &[0.0, 0.0, 2.5, 0.0, 3.0, 0.0]);
}

pub fn relu_of_negative_zero_is_positive_zero(backend: &dyn Backend) {
    let input = Tensor::new(vec![1], vec![-0.0]).unwrap();
    let out = run(backend, &relu_graph(vec![1]), &[input]).unwrap();
    assert_eq!(out[0].data()[0].to_bits(), 0.0f32.to_bits());
}

pub fn nan_propagates_and_infinities_behave(backend: &dyn Backend) {
    let input = Tensor::new(vec![3], vec![f32::NAN, f32::NEG_INFINITY, f32::INFINITY]).unwrap();
    let out = run(backend, &relu_graph(vec![3]), &[input]).unwrap();
    assert!(out[0].data()[0].is_nan());
    assert_eq!(out[0].data()[1], 0.0);
    assert_eq!(out[0].data()[2], f32::INFINITY);
}

pub fn scalar_input_works(backend: &dyn Backend) {
    let input = Tensor::new(vec![], vec![-4.0]).unwrap();
    let out = run(backend, &relu_graph(vec![]), &[input]).unwrap();
    assert_eq!(out[0].shape(), &[] as &[usize]);
    assert_eq!(out[0].data(), &[0.0]);
}

pub fn constant_only_graph_runs_with_no_inputs(backend: &dyn Backend) {
    let mut b = GraphBuilder::new(Limits::default());
    let c = b.constant("c", vec![2], vec![-1.0, 2.0]).unwrap();
    let r = b.node("r", Op::Relu, &[c]).unwrap();
    let g = b.finish(&[r]).unwrap();
    assert_eq!(run(backend, &g, &[]).unwrap()[0].data(), &[0.0, 2.0]);
}

pub fn output_can_be_an_input(backend: &dyn Backend) {
    let mut b = GraphBuilder::new(Limits::default());
    let x = b.input("x", vec![2]).unwrap();
    let g = b.finish(&[x]).unwrap();
    let input = Tensor::new(vec![2], vec![-1.0, 1.0]).unwrap();
    let out = run(backend, &g, std::slice::from_ref(&input)).unwrap();
    assert_eq!(out[0], input);
}

pub fn same_value_can_be_output_twice(backend: &dyn Backend) {
    let mut b = GraphBuilder::new(Limits::default());
    let x = b.input("x", vec![1]).unwrap();
    let y = b.node("y", Op::Relu, &[x]).unwrap();
    let g = b.finish(&[y, y]).unwrap();
    let out = run(backend, &g, &[Tensor::new(vec![1], vec![-2.0]).unwrap()]).unwrap();
    assert_eq!(out.len(), 2);
    assert_eq!(out[0], out[1]);
}

pub fn a_constant_output_is_returned_unchanged_on_every_run(backend: &dyn Backend) {
    let mut b = GraphBuilder::new(Limits::default());
    let c = b.constant("c", vec![2], vec![-1.0, 2.0]).unwrap();
    let g = b.finish(&[c, c]).unwrap();
    let exe = backend.compile(&g).unwrap();
    for _ in 0..3 {
        let out = exe.run(&[]).unwrap();
        assert_eq!(out[0].data(), &[-1.0, 2.0]);
        assert_eq!(out[1].data(), &[-1.0, 2.0]);
    }
}

pub fn inputs_are_not_modified_by_a_run(backend: &dyn Backend) {
    let input = Tensor::new(vec![3], vec![-1.0, 0.5, -0.0]).unwrap();
    let before = bits(&input);
    run(backend, &relu_graph(vec![3]), std::slice::from_ref(&input)).unwrap();
    assert_eq!(bits(&input), before);
}

pub fn wrong_input_count_is_rejected(backend: &dyn Backend) {
    let err = run(backend, &relu_graph(vec![1]), &[]).unwrap_err();
    assert_eq!(
        err,
        Error::InputCount {
            expected: 1,
            found: 0
        }
    );
}

pub fn wrong_input_shape_is_rejected(backend: &dyn Backend) {
    let input = Tensor::new(vec![3], vec![1.0, 2.0, 3.0]).unwrap();
    let err = run(backend, &relu_graph(vec![2]), &[input]).unwrap_err();
    assert_eq!(
        err,
        Error::InputShape {
            name: "x".into(),
            expected: vec![2],
            found: vec![3]
        }
    );
}

pub fn an_executable_is_reusable(backend: &dyn Backend) {
    let exe = backend.compile(&relu_graph(vec![1])).unwrap();
    let a = exe
        .run(&[Tensor::new(vec![1], vec![-1.0]).unwrap()])
        .unwrap();
    let b = exe
        .run(&[Tensor::new(vec![1], vec![5.0]).unwrap()])
        .unwrap();
    assert_eq!(a[0].data(), &[0.0]);
    assert_eq!(b[0].data(), &[5.0]);
}

pub fn matmul_reshape_transpose_by_hand(backend: &dyn Backend) {
    let mut b = GraphBuilder::new(Limits::default());
    let x = b.input("x", vec![2, 2]).unwrap();
    let w = b
        .constant("w", vec![2, 2], vec![5.0, 6.0, 7.0, 8.0])
        .unwrap();
    let p = b.node("p", Op::MatMul, &[x, w]).unwrap();
    let t = b
        .node("t", Op::Transpose { perm: vec![1, 0] }, &[p])
        .unwrap();
    let r = b.node("r", Op::Reshape { shape: vec![4] }, &[t]).unwrap();
    let g = b.finish(&[p, t, r]).unwrap();
    let input = Tensor::new(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let out = run(backend, &g, &[input]).unwrap();
    assert_eq!(out[0].data(), &[19.0, 22.0, 43.0, 50.0]);
    assert_eq!(out[1].data(), &[19.0, 43.0, 22.0, 50.0]);
    assert_eq!(out[2].shape(), &[4]);
    assert_eq!(out[2].data(), &[19.0, 43.0, 22.0, 50.0]);
}

pub fn an_output_that_is_an_input_and_a_node_input_stays_intact(backend: &dyn Backend) {
    let mut b = GraphBuilder::new(Limits::default());
    let x = b.input("x", vec![3]).unwrap();
    let r = b.node("r", Op::Relu, &[x]).unwrap();
    let g = b.finish(&[x, r, x, r]).unwrap();
    let input = Tensor::new(vec![3], vec![-1.0, 2.0, -3.0]).unwrap();
    let out = run(backend, &g, std::slice::from_ref(&input)).unwrap();
    assert_eq!(out[0], input);
    assert_eq!(out[2], input);
    assert_eq!(out[1].data(), &[0.0, 2.0, 0.0]);
    assert_eq!(out[1], out[3]);
}

pub fn a_constant_feeding_a_node_is_not_consumed(backend: &dyn Backend) {
    let mut b = GraphBuilder::new(Limits::default());
    let c = b.constant("c", vec![2], vec![-1.0, 2.0]).unwrap();
    let r = b.node("r", Op::Relu, &[c]).unwrap();
    let g = b.finish(&[c, r]).unwrap();
    let exe = backend.compile(&g).unwrap();
    for _ in 0..3 {
        let out = exe.run(&[]).unwrap();
        assert_eq!(out[0].data(), &[-1.0, 2.0]);
        assert_eq!(out[1].data(), &[0.0, 2.0]);
    }
}

macro_rules! contract_tests {
    ($module:ident, $backend:expr) => {
        $crate::contract::contract_tests!($module, $backend, true);
    };
    ($module:ident, $backend:expr, $enabled:expr) => {
        mod $module {
            #[allow(unused_imports)]
            use super::*;
            macro_rules! check {
                ($name:ident) => {
                    #[test]
                    fn $name() {
                        if !$enabled {
                            return;
                        }
                        $crate::contract::$name(&$backend)
                    }
                };
            }
            check!(relu_clamps_negatives);
            check!(relu_of_negative_zero_is_positive_zero);
            check!(nan_propagates_and_infinities_behave);
            check!(scalar_input_works);
            check!(constant_only_graph_runs_with_no_inputs);
            check!(output_can_be_an_input);
            check!(same_value_can_be_output_twice);
            check!(a_constant_output_is_returned_unchanged_on_every_run);
            check!(inputs_are_not_modified_by_a_run);
            check!(wrong_input_count_is_rejected);
            check!(wrong_input_shape_is_rejected);
            check!(an_executable_is_reusable);
            check!(matmul_reshape_transpose_by_hand);
            check!(an_output_that_is_an_input_and_a_node_input_stays_intact);
            check!(a_constant_feeding_a_node_is_not_consumed);
        }
    };
}

pub(crate) use contract_tests;
