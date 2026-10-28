//! The fast CPU backend. It runs the same `Plan` as the reference backend and
//! returns the same bits; it exists to be faster. It has its own executor so
//! the reference executor stays a plain, obviously correct oracle.

mod elementwise;
mod kernels;
mod matmul;
mod vecmath;

use crate::plan::Step;
use crate::{Backend, Executable, Plan, Tensor};
use oxfuse_core::{Error, Graph, Op};

pub struct CpuBackend;

/// The SIMD level the fast backend picks on this machine, for run records.
pub fn simd_level() -> String {
    let debug = format!("{:?}", fearless_simd::Level::new());
    // `Neon(Neon { .. })`: the variant name is the part worth recording.
    debug.split('(').next().unwrap_or_default().to_owned()
}

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

// Output buffers copied in a run on this thread. The bits are the same
// whether an output is moved or copied, so a test counts the copies instead.
#[cfg(test)]
thread_local! {
    static OUTPUT_COPIES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    // Steps that wrote their result into an operand's buffer.
    static REUSED: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
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
/// one read), and the same size as the result. Returns the taken buffer and the
/// slot of the other operand if any. A slot read twice by one step never
/// qualifies.
fn take_reusable(
    step: &Step,
    plan: &Plan,
    remaining: &[usize],
    slots: &mut [Slot],
) -> Option<(Vec<f32>, Option<usize>)> {
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
            #[cfg(test)]
            REUSED.with(|c| c.set(c.get() + 1));
            return Some((buf, step.inputs.get(1 - at).copied()));
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
                Some((mut buf, other)) => {
                    if !matches!(step.op, Op::Reshape { .. }) {
                        let other = other.map(|s| slots[s].data());
                        elementwise::apply_in_place(&step.op, &mut buf, other);
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
                    other => {
                        #[cfg(test)]
                        OUTPUT_COPIES.with(|c| c.set(c.get() + 1));
                        other.data().to_vec()
                    }
                };
                Tensor::new(plan.slot_shapes[slot].clone(), data)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxfuse_core::{GraphBuilder, Limits, ValueId};

    #[test]
    fn the_simd_level_is_a_plain_name() {
        let name = simd_level();
        let known = [
            "Fallback",
            "Neon",
            "WasmSimd128",
            "Sse2",
            "Sse4_2",
            "Avx512",
            "Avx2",
        ];
        assert!(known.contains(&name.as_str()), "{name}");
    }

    /// Runs the graph on `[1, -2, 3]` and returns the output buffers copied
    /// and the steps that reused an operand's buffer.
    fn counts_when_running(
        build: impl FnOnce(&mut GraphBuilder) -> Vec<ValueId>,
    ) -> (usize, usize) {
        let mut b = GraphBuilder::new(Limits::default());
        let outputs = build(&mut b);
        let graph = b.finish(&outputs).unwrap();
        let exe = CpuBackend.compile(&graph).unwrap();
        let input = Tensor::new(vec![3], vec![1.0, -2.0, 3.0]).unwrap();
        OUTPUT_COPIES.with(|c| c.set(0));
        REUSED.with(|c| c.set(0));
        exe.run(&[input]).unwrap();
        (OUTPUT_COPIES.with(Cell::get), REUSED.with(Cell::get))
    }

    fn copies_when_running(build: impl FnOnce(&mut GraphBuilder) -> Vec<ValueId>) -> usize {
        counts_when_running(build).0
    }

    fn reuses_when_running(build: impl FnOnce(&mut GraphBuilder) -> Vec<ValueId>) -> usize {
        counts_when_running(build).1
    }

    use std::cell::Cell;

    #[test]
    fn a_computed_output_listed_once_moves_out_without_a_copy() {
        let copies = copies_when_running(|b| {
            let x = b.input("x", vec![3]).unwrap();
            vec![b.node("y", Op::Relu, &[x]).unwrap()]
        });
        assert_eq!(copies, 0);
    }

    #[test]
    fn a_computed_output_listed_twice_is_copied_each_time() {
        let copies = copies_when_running(|b| {
            let x = b.input("x", vec![3]).unwrap();
            let y = b.node("y", Op::Relu, &[x]).unwrap();
            vec![y, y]
        });
        assert_eq!(copies, 2);
    }

    #[test]
    fn a_borrowed_input_listed_as_an_output_is_copied() {
        let copies = copies_when_running(|b| vec![b.input("x", vec![3]).unwrap()]);
        assert_eq!(copies, 1);
    }

    #[test]
    fn an_output_that_is_also_read_later_is_still_moved_once_at_the_end() {
        let copies = copies_when_running(|b| {
            let x = b.input("x", vec![3]).unwrap();
            let y = b.node("y", Op::Relu, &[x]).unwrap();
            let z = b.node("z", Op::Add, &[y, y]).unwrap();
            vec![y, z]
        });
        assert_eq!(copies, 0);
    }

    #[test]
    fn a_chain_reuses_each_intermediate_buffer_but_not_the_borrowed_input() {
        let reused = reuses_when_running(|b| {
            let x = b.input("x", vec![3]).unwrap();
            let a = b.node("a", Op::Relu, &[x]).unwrap();
            let c = b.node("c", Op::Relu, &[a]).unwrap();
            vec![b.node("d", Op::Relu, &[c]).unwrap()]
        });
        assert_eq!(reused, 2);
    }

    #[test]
    fn a_value_that_is_also_an_output_is_never_overwritten() {
        let reused = reuses_when_running(|b| {
            let x = b.input("x", vec![3]).unwrap();
            let y = b.node("y", Op::Relu, &[x]).unwrap();
            let z = b.node("z", Op::Relu, &[y]).unwrap();
            vec![y, z]
        });
        assert_eq!(reused, 0);
    }

    #[test]
    fn a_value_read_twice_by_one_step_is_never_overwritten() {
        let reused = reuses_when_running(|b| {
            let x = b.input("x", vec![3]).unwrap();
            let y = b.node("y", Op::Relu, &[x]).unwrap();
            vec![b.node("z", Op::Add, &[y, y]).unwrap()]
        });
        assert_eq!(reused, 0);
    }

    #[test]
    fn the_last_reader_of_a_shared_value_reuses_its_buffer() {
        // y feeds a and then c: a must copy, c may overwrite y.
        let reused = reuses_when_running(|b| {
            let x = b.input("x", vec![3]).unwrap();
            let y = b.node("y", Op::Relu, &[x]).unwrap();
            let a = b.node("a", Op::Relu, &[y]).unwrap();
            let c = b.node("c", Op::Relu, &[y]).unwrap();
            vec![a, c]
        });
        assert_eq!(reused, 1);
    }

    #[test]
    fn a_scalar_operand_is_not_taken_as_the_buffer_of_a_larger_result() {
        // r is a computed rank-0 value of size 1; the sum has size 3.
        let reused = reuses_when_running(|b| {
            let x = b.input("x", vec![3]).unwrap();
            let y = b.node("y", Op::Relu, &[x]).unwrap();
            let s = b.constant("s", vec![], vec![0.5]).unwrap();
            let r = b.node("r", Op::Relu, &[s]).unwrap();
            vec![b.node("z", Op::Add, &[r, y]).unwrap()]
        });
        assert_eq!(reused, 1);
    }
}
