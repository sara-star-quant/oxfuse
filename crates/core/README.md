# oxfuse-core

Part of [oxfuse](https://github.com/sara-star-quant/oxfuse), a small tensor graph compiler in Rust. Start with the [repository README](https://github.com/sara-star-quant/oxfuse#readme) if you landed here from a search. What it is for: exact, repeatable float32 results and a reference for checking other runtimes; not for large models, training or production inference ([use cases](https://github.com/sara-star-quant/oxfuse/blob/main/docs/use-cases.md)).

The typed tensor-graph IR (intermediate representation) behind [oxfuse](https://github.com/sara-star-quant/oxfuse): JSON graph parsing with input limits, shape inference, a verifier, and the exact optimization passes (constant folding, common-subexpression elimination, dead-code elimination).

Experimental. See the repository for the semantics, the claims ledger and the benchmarks.

Licensed under Apache-2.0.
