//! The fast CPU backend. It runs the same `Plan` as the reference backend and
//! returns the same bits; it exists to be faster. It has its own executor so
//! the reference executor stays a plain, obviously correct oracle.

mod elementwise;
mod kernels;
mod matmul;

use crate::plan::Step;
use crate::{Backend, Executable, Plan, Tensor};
use oxfuse_core::{Error, Graph, Op};

pub struct CpuBackend;

impl Backend for CpuBackend {
    fn compile(&self, graph: &Graph) -> Result<Box<dyn Executable>, Error> {
        let plan = Plan::build(graph);
        // Uses of each slot: once per step that reads it, once per listing as
        // an output. Outputs are never released, because they are not consumed.
        let mut uses = vec![0usize; plan.slot_shapes.len()];
        for &slot in plan
            .steps
            .iter()
            .flat_map(|s| &s.inputs)
            .chain(&plan.outputs)
        {
            uses[slot] += 1;
        }
        Ok(Box::new(CpuExecutable { plan, uses }))
    }
}

struct CpuExecutable {
    plan: Plan,
    uses: Vec<usize>,
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

/// An operand buffer this step may overwrite with its result: computed, read
/// by no later step and not an output (its remaining use count is this step's
/// one read), and the same size as the result. Returns the taken buffer, the
/// slot of the other operand if any, and whether the buffer was the first
/// operand. A slot read twice by one step never qualifies.
fn take_reusable(
    step: &Step,
    plan: &Plan,
    remaining: &[usize],
    slots: &mut [Slot],
) -> Option<(Vec<f32>, Option<usize>, bool)> {
    let candidates: &[usize] = match step.op {
        Op::Relu | Op::Sigmoid | Op::Reshape { .. } => &[0],
        Op::Add | Op::Mul => &[0, 1],
        Op::MatMul | Op::Transpose { .. } => return None,
    };
    let size = plan.slot_shapes[step.output].iter().product::<usize>();
    for &at in candidates {
        let slot = step.inputs[at];
        let reusable = remaining[slot] == 1
            && matches!(slots[slot], Slot::Owned(_))
            && plan.slot_shapes[slot].iter().product::<usize>() == size;
        if reusable {
            let Slot::Owned(buf) = std::mem::replace(&mut slots[slot], Slot::Empty) else {
                unreachable!("checked above")
            };
            return Some((buf, step.inputs.get(1 - at).copied(), at == 0));
        }
    }
    None
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
        let mut remaining = self.uses.clone();
        for step in &plan.steps {
            let out = match take_reusable(step, plan, &remaining, &mut slots) {
                Some((mut buf, other, buf_first)) => {
                    if !matches!(step.op, Op::Reshape { .. }) {
                        let other = other.map(|s| slots[s].data());
                        elementwise::apply_in_place(&step.op, &mut buf, other, buf_first);
                    }
                    buf
                }
                None => {
                    let inputs: Vec<&[f32]> =
                        step.inputs.iter().map(|&s| slots[s].data()).collect();
                    let shapes: Vec<&[usize]> = step
                        .inputs
                        .iter()
                        .map(|&s| plan.slot_shapes[s].as_slice())
                        .collect();
                    kernels::apply(&step.op, &inputs, &shapes)
                }
            };
            // Release operands this step consumed last, so peak memory is the
            // live set and not every intermediate.
            for &s in &step.inputs {
                remaining[s] -= 1;
                if remaining[s] == 0 {
                    slots[s] = Slot::Empty;
                }
            }
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
