//! Every backend meets the same contract: see `contract/mod.rs`.

mod contract;

use oxfuse_runtime::{CpuBackend, ReferenceBackend};

contract::contract_tests!(reference, ReferenceBackend);
contract::contract_tests!(cpu, CpuBackend);
