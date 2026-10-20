//! The fast CPU backend. It runs the same `Plan` as the reference backend and
//! returns the same bits; it exists to be faster. It has its own executor so
//! the reference executor stays a plain, obviously correct oracle.

mod kernels;

use crate::{Backend, Executable, Plan, Tensor};
use oxfuse_core::{Error, Graph};

pub struct CpuBackend;

impl Backend for CpuBackend {
    fn compile(&self, graph: &Graph) -> Result<Box<dyn Executable>, Error> {
        Ok(Box::new(CpuExecutable {
            plan: Plan::build(graph),
        }))
    }
}

struct CpuExecutable {
    plan: Plan,
}

/// A buffer for one value during a run. Inputs and constants are borrowed,
/// computed values are owned. Borrowed slots are never written.
enum Slot<'a> {
    Empty,
    Borrowed(&'a [f32]),
    Owned(Vec<f32>),
}

impl Slot<'_> {
    fn data(&self) -> &[f32] {
        match self {
            Slot::Empty => &[],
            Slot::Borrowed(data) => data,
            Slot::Owned(data) => data,
        }
    }
}

impl Executable for CpuExecutable {
    fn run(&self, inputs: &[Tensor]) -> Result<Vec<Tensor>, Error> {
        let plan = &self.plan;
        if inputs.len() != plan.inputs.len() {
            return Err(Error::InputCount {
                expected: plan.inputs.len(),
                found: inputs.len(),
            });
        }
        let mut slots: Vec<Slot> = (0..plan.slot_shapes.len()).map(|_| Slot::Empty).collect();
        for (binding, tensor) in plan.inputs.iter().zip(inputs) {
            if tensor.shape() != binding.shape.as_slice() {
                return Err(Error::InputShape {
                    name: binding.name.clone(),
                    expected: binding.shape.clone(),
                    found: tensor.shape().to_vec(),
                });
            }
            slots[binding.slot] = Slot::Borrowed(tensor.data());
        }
        for (slot, data) in &plan.constants {
            slots[*slot] = Slot::Borrowed(data);
        }
        for step in &plan.steps {
            let out = {
                let inputs: Vec<&[f32]> = step.inputs.iter().map(|&s| slots[s].data()).collect();
                let shapes: Vec<&[usize]> = step
                    .inputs
                    .iter()
                    .map(|&s| plan.slot_shapes[s].as_slice())
                    .collect();
                kernels::apply(&step.op, &inputs, &shapes)
            };
            slots[step.output] = Slot::Owned(out);
        }
        // A computed buffer listed once as an output moves out; anything else
        // (borrowed, or listed more than once) is copied.
        let mut uses = vec![0usize; slots.len()];
        for &slot in &plan.outputs {
            uses[slot] += 1;
        }
        plan.outputs
            .iter()
            .map(|&slot| {
                let data = match &mut slots[slot] {
                    Slot::Owned(buffer) if uses[slot] == 1 => std::mem::take(buffer),
                    other => other.data().to_vec(),
                };
                Tensor::new(plan.slot_shapes[slot].clone(), data)
            })
            .collect()
    }
}
