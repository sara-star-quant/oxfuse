//! Random valid-graph generator shared by the equivalence tests. A random
//! program is interpreted step by step; a step the shape rules reject is
//! skipped, so every generated graph is valid by construction.
#![allow(dead_code)]

use oxfuse_core::shape::infer;
use oxfuse_core::{Graph, GraphBuilder, Limits, Op, TensorType, ValueId};
use oxfuse_runtime::Tensor;
use proptest::prelude::*;

#[derive(Debug, Clone)]
pub struct Step {
    kind: u8,
    a: u16,
    b: u16,
    seed: u8,
}

#[derive(Debug, Clone)]
pub struct Case {
    steps: Vec<Step>,
    data: Vec<f32>,
    constants: Vec<f32>,
}

pub fn special() -> impl Strategy<Value = f32> {
    prop::sample::select(vec![
        0.0f32,
        -0.0,
        1.0,
        -1.0,
        0.5,
        2.0,
        -3.25,
        f32::INFINITY,
        f32::NAN,
        3.0e38,
    ])
}

pub fn case() -> impl Strategy<Value = Case> {
    (
        prop::collection::vec(
            (0u8..7, any::<u16>(), any::<u16>(), any::<u8>()).prop_map(|(kind, a, b, seed)| Step {
                kind,
                a,
                b,
                seed,
            }),
            1..14,
        ),
        prop::collection::vec(special(), 32),
        prop::collection::vec(special(), 32),
    )
        .prop_map(|(steps, data, constants)| Case {
            steps,
            data,
            constants,
        })
}

pub fn take(source: &[f32], start: usize, count: usize) -> Vec<f32> {
    (0..count)
        .map(|i| source[(start + i) % source.len()])
        .collect()
}

pub fn count(shape: &[usize]) -> usize {
    shape.iter().product()
}

pub fn build(case: &Case) -> (Graph, Vec<Tensor>) {
    let mut b = GraphBuilder::new(Limits::default());
    let mut ids: Vec<ValueId> = Vec::new();
    let mut types: Vec<TensorType> = Vec::new();
    let mut inputs = Vec::new();
    let mut last_step: Option<(Op, Vec<usize>)> = None;
    let mut offset = 0;
    for (name, shape) in [("x0", vec![2, 3]), ("x1", vec![3, 2]), ("x2", vec![])] {
        ids.push(b.input(name, shape.clone()).unwrap());
        types.push(TensorType::f32(shape.clone()));
        inputs.push(Tensor::new(shape.clone(), take(&case.data, offset, count(&shape))).unwrap());
        offset += count(&shape);
    }
    for (name, shape) in [("c0", vec![2, 3]), ("c1", vec![])] {
        let data = take(&case.constants, offset, count(&shape));
        offset += count(&shape);
        ids.push(b.constant(name, shape.clone(), data).unwrap());
        types.push(TensorType::f32(shape));
    }
    for (k, step) in case.steps.iter().enumerate() {
        let ia = step.a as usize % ids.len();
        let ib = step.b as usize % ids.len();
        let fresh: (Op, Vec<usize>) = match step.kind % 7 {
            0 => (Op::Relu, vec![ia]),
            1 => (Op::Sigmoid, vec![ia]),
            2 => (Op::Add, vec![ia, ib]),
            3 => (Op::Mul, vec![ia, ib]),
            4 => (Op::MatMul, vec![ia, ib]),
            5 => {
                let n = count(&types[ia].shape);
                let target = match step.seed % 3 {
                    0 => vec![n],
                    1 => vec![1, n],
                    _ => vec![n, 1],
                };
                (Op::Reshape { shape: target }, vec![ia])
            }
            _ => {
                let rank = types[ia].shape.len();
                let perm = (0..rank)
                    .map(|i| (i + step.seed as usize) % rank.max(1))
                    .collect();
                (Op::Transpose { perm }, vec![ia])
            }
        };
        // One step in three repeats the previous node, which forces the
        // duplicates common-subexpression elimination exists to merge.
        let (op, operands) = match (&last_step, step.seed % 3) {
            (Some(previous), 0) => previous.clone(),
            _ => fresh,
        };
        let name = format!("n{k}");
        let operand_types: Vec<&TensorType> = operands.iter().map(|&i| &types[i]).collect();
        let Ok(ty) = infer(&name, &op, &operand_types) else {
            continue;
        };
        let operand_ids: Vec<ValueId> = operands.iter().map(|&i| ids[i]).collect();
        let applied = (op.clone(), operands.clone());
        if let Ok(id) = b.node(&name, op, &operand_ids) {
            ids.push(id);
            types.push(ty);
            last_step = Some(applied);
        }
    }
    // Several differently named outputs, one value sometimes listed twice:
    // the interface property then checks that names survive merging.
    let last = *ids.last().unwrap();
    let pick = ids[case.steps[0].a as usize % ids.len()];
    let mut outputs = vec![("out0".to_owned(), last), ("out1".to_owned(), pick)];
    if ids.len() >= 2 {
        outputs.push(("out2".to_owned(), ids[ids.len() - 2]));
    }
    if case.steps[0].seed.is_multiple_of(2) {
        outputs.push(("again".to_owned(), last));
    }
    (b.finish_named(outputs).unwrap(), inputs)
}

/// The bits of `v`, with every NaN mapped to one pattern: IEEE 754 leaves the
/// sign and payload of a NaN produced from two NaN operands or from an invalid
/// operation to the hardware (x86 and ARM differ) and to the operand order the
/// compiler picks, so two correct kernels may disagree on them.
pub fn comparable_bits(v: f32) -> u32 {
    if v.is_nan() {
        f32::NAN.to_bits()
    } else {
        v.to_bits()
    }
}

pub fn bits(tensors: &[Tensor]) -> Vec<(Vec<usize>, Vec<u32>)> {
    tensors
        .iter()
        .map(|t| {
            (
                t.shape().to_vec(),
                t.data().iter().map(|&v| comparable_bits(v)).collect(),
            )
        })
        .collect()
}
