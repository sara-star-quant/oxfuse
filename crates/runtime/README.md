# oxfuse-runtime

Part of [oxfuse](https://github.com/sara-star-quant/oxfuse), a small tensor graph compiler in Rust. Start with the [repository README](https://github.com/sara-star-quant/oxfuse#readme) if you landed here from a search.

Tensors, execution plans and the CPU backends for [oxfuse](https://github.com/sara-star-quant/oxfuse) graphs: a reference backend, and a fast backend that returns the same bits on the platforms it is tested on. The fast matmul and `Sigmoid` run on SIMD vectors picked at run time (AVX2 or AVX-512 on x86, NEON on ARM) through `fearless_simd`, which sets the minimum Rust to 1.89. The optional `parallel` feature threads the fast matmul without changing results (tested on 64-bit platforms).

**For AI work:** run a graph on a reference backend and on a fast backend that agree bit for bit, and use the pair as a numerical baseline or an oracle in tests. Not for large models, training or production inference ([use cases](https://github.com/sara-star-quant/oxfuse/blob/main/docs/use-cases.md)).

```rust
// cargo add oxfuse-core oxfuse-runtime
use oxfuse_core::{GraphBuilder, Limits, Op};
use oxfuse_runtime::{Backend, CpuBackend, ReferenceBackend, Tensor};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut b = GraphBuilder::new(Limits::default());
    let x = b.input("x", vec![2, 3])?;
    let w = b.constant("w", vec![3, 2], vec![0.5, -1.0, 1.5, 2.0, -0.5, 1.0])?;
    let m = b.node("m", Op::MatMul, &[x, w])?;
    let y = b.node("y", Op::Relu, &[m])?;
    let graph = b.finish(&[y])?;

    let input = Tensor::new(vec![2, 3], vec![1.0, -2.0, 3.0, 4.0, 5.0, -6.0])?;
    let reference = ReferenceBackend.compile(&graph)?.run(std::slice::from_ref(&input))?;
    let fast = CpuBackend.compile(&graph)?.run(std::slice::from_ref(&input))?;
    let bits = |t: &Tensor| t.data().iter().map(|v| v.to_bits()).collect::<Vec<_>>();
    assert_eq!(bits(&reference[0]), bits(&fast[0])); // same bits
    println!("{:?}", fast[0].data()); // [0.0, 0.0, 12.5, 0.0]
    Ok(())
}
```

In the repository, `examples/optimize_and_run.rs` also runs the optimization passes (`cargo run --release -p oxfuse-runtime --example optimize_and_run`).

Experimental. The optional GPU backend is the `oxfuse-wgpu` crate. Measurements are in the repository's `docs/benchmarks.md`.

Licensed under Apache-2.0.
