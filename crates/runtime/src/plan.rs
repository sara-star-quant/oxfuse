//! A verified graph flattened into steps over numbered buffer slots, so the
//! executor does no name or hash lookups at run time.

use oxfuse_core::{Graph, Op, Source, ValueId};
use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct Step {
    pub op: Op,
    pub inputs: Vec<usize>,
    pub output: usize,
}

#[derive(Debug, Clone)]
pub struct InputBinding {
    pub name: String,
    pub slot: usize,
    pub shape: Vec<usize>,
}

#[derive(Debug, Clone)]
pub struct Plan {
    pub slot_shapes: Vec<Vec<usize>>,
    pub constants: Vec<(usize, Arc<[f32]>)>,
    pub inputs: Vec<InputBinding>,
    pub steps: Vec<Step>,
    pub outputs: Vec<usize>,
}

impl Plan {
    /// Slot `i` holds value `i`. Values are in definition order, so `steps`
    /// is already topologically ordered.
    pub fn build(graph: &Graph) -> Plan {
        let mut plan = Plan {
            slot_shapes: Vec::with_capacity(graph.values().len()),
            constants: Vec::new(),
            inputs: Vec::new(),
            steps: Vec::new(),
            outputs: graph.outputs().iter().map(|id| id.index()).collect(),
        };
        for (slot, value) in graph.values().iter().enumerate() {
            plan.slot_shapes.push(value.ty.shape.clone());
            match &value.source {
                Source::Input => plan.inputs.push(InputBinding {
                    name: graph.label(ValueId(slot as u32)),
                    slot,
                    shape: value.ty.shape.clone(),
                }),
                Source::Constant(data) => plan.constants.push((slot, Arc::clone(data))),
                Source::Node { op, inputs } => plan.steps.push(Step {
                    op: op.clone(),
                    inputs: inputs.iter().map(|id| id.index()).collect(),
                    output: slot,
                }),
            }
        }
        plan
    }
}
