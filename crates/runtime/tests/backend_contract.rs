//! Every backend meets the same contract: see `contract/mod.rs`.

mod contract;

use oxfuse_runtime::{CpuBackend, ReferenceBackend};

contract::contract_tests!(reference, ReferenceBackend);
contract::contract_tests!(cpu, CpuBackend);

/// The default `run_many` is `run` once per element, in order.
#[test]
fn run_many_equals_run_per_element_on_every_cpu_backend() {
    use oxfuse_core::{GraphBuilder, Limits, Op};
    use oxfuse_runtime::{Backend, CpuBackend, ReferenceBackend, Tensor};
    let mut b = GraphBuilder::new(Limits::default());
    let x = b.input("x", vec![3]).unwrap();
    let y = b.node("y", Op::Relu, &[x]).unwrap();
    let graph = b.finish(&[y]).unwrap();
    let batches: Vec<Vec<Tensor>> = [[-1.0, 2.0, 3.0], [4.0, -5.0, 0.0], [0.5, 0.25, -9.0]]
        .iter()
        .map(|d| vec![Tensor::new(vec![3], d.to_vec()).unwrap()])
        .collect();
    let backends: [Box<dyn Backend>; 2] = [Box::new(ReferenceBackend), Box::new(CpuBackend)];
    for backend in backends {
        let exe = backend.compile(&graph).unwrap();
        let many = exe.run_many(&batches).unwrap();
        assert_eq!(many.len(), 3);
        for (batch, got) in batches.iter().zip(&many) {
            assert_eq!(got[0].data(), exe.run(batch).unwrap()[0].data());
        }
        assert!(exe.run_many(&[]).unwrap().is_empty());
        // One bad element fails the whole call.
        let wrong = vec![Tensor::new(vec![2], vec![1.0, 2.0]).unwrap()];
        assert!(exe.run_many(&[batches[0].clone(), wrong]).is_err());
    }
}
