# oxfuse-core

Part of [oxfuse](https://github.com/sara-star-quant/oxfuse), a small tensor graph compiler in Rust. Start with the [repository README](https://github.com/sara-star-quant/oxfuse#readme) if you landed here from a search.

The typed tensor-graph IR (intermediate representation) behind [oxfuse](https://github.com/sara-star-quant/oxfuse): JSON graph parsing with input limits, shape inference, a verifier, and the exact optimization passes (constant folding, common-subexpression elimination, dead-code elimination).

**For AI work:** describe a small float32 model as a graph, check it (shapes and types are verified), and optimize it without changing a single output bit. The graph is plain data you can dump, diff and test. Not for large models, training or production inference ([use cases](https://github.com/sara-star-quant/oxfuse/blob/main/docs/use-cases.md)).

```rust
use oxfuse_core::{GraphBuilder, Limits, Op};

fn main() -> Result<(), oxfuse_core::Error> {
    let mut b = GraphBuilder::new(Limits::default());
    let x = b.input("x", vec![2, 3])?;
    let w = b.constant("w", vec![3, 2], vec![0.5, -1.0, 1.5, 2.0, -0.5, 1.0])?;
    let m = b.node("m", Op::MatMul, &[x, w])?;
    let y = b.node("y", Op::Relu, &[m])?;
    let graph = b.finish(&[y])?; // runs the verifier
    println!("{} output(s)", graph.outputs().len());
    Ok(())
}
```

Add it with `cargo add oxfuse-core`; `oxfuse-runtime` runs the graph. Experimental. See the repository for the semantics, the claims ledger and the benchmarks.

Licensed under Apache-2.0.
