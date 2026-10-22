# Mutation testing

`cargo-mutants` 27.1.0 breaks the code in small ways (a `<` becomes `<=`, a function returns a constant) and reruns the tests. A mutant that no test notices is a gap in the tests, or a change that cannot alter any result. Every survivor below is the second kind, with the reason. Nothing is left unexplained.

## Result

Run on an Apple M1 Pro on 2026-10-21, except the `exec.rs` and `context.rs` rows, rerun on 2026-10-22 after the host path was reworked. Rows without a tree note ran on crates tree 026fd4c; rows with one ran on that tree (`git rev-parse --short HEAD:crates` names the tree).

| Scope | Mutants | Caught | Timed out (counts as caught) | Unviable (do not compile) | Missed |
|---|---|---|---|---|---|
| `oxfuse-core`, `oxfuse-runtime`, `oxfuse` (CLI, built with the GPU feature) | 594 | 538 | 4 | 49 | 3 |
| `oxfuse-wgpu` `config.rs` | 100 | 98 | 0 | 1 | 1 |
| `oxfuse-wgpu` `bounds.rs` | 45 | 45 | 0 | 0 | 0 |
| `oxfuse-wgpu` `model.rs` (tree 2e3e5d7) | 245 | 240 | 0 | 2 | 3 |
| `oxfuse-wgpu` `exec.rs` (tree 6204ce2) | 100 | 74 | 0 | 24 | 2 |
| `oxfuse-wgpu` `context.rs` (tree 6204ce2) | 33 | 23 | 0 | 9 | 1 |
| `oxfuse-wgpu` `selftest.rs` (tree 2e3e5d7) | 91 | 88 | 0 | 3 | 0 |
| `oxfuse-onnx` (tree 521c719) | 273 | 254 | 0 | 18 | 1 |
| Total | 1481 | 1360 | 4 | 106 | 11 |

Of the 1375 mutants that compile, 1364 were stopped (99.2%) and the other 11 are explained below. The first self-test run missed 40 of 77, which led to extracting its logic into small functions with their own tests; that is why it has its own tree id.

Reproduce with `just mutants` (the first two commands below) and `just mutants-gpu crates/wgpu/src/exec.rs` (one GPU host file, needs an adapter). They run:

    cargo mutants -p oxfuse-core -p oxfuse-runtime -p oxfuse --features oxfuse/wgpu,oxfuse/onnx --test-workspace true -j 2 --timeout 120
    cargo mutants -p oxfuse-onnx --test-package oxfuse-onnx -j 2 --timeout 120
    cargo mutants -p oxfuse-wgpu -f crates/wgpu/src/exec.rs --test-package oxfuse-wgpu -j 2 --timeout 180

The second command is per file; the GPU tests need an adapter, otherwise most of them do nothing. Use `-j 2` and `CARGO_INCREMENTAL=0`, because every worker builds its own copy and four workers fill a laptop disk.

## In CI

`.github/workflows/mutants.yml` reports and never gates. A pull request runs only the mutants that overlap its diff (`--in-diff`), in one job for the CPU-side crates and one for `oxfuse-wgpu` on a software Vulkan adapter. The pull request job builds with all features, so code behind `onnx`, `wgpu` and `parallel` is tested. A weekly run covers every mutant of the CPU-side crates, one job per package. The missed list is the artifact `mutants.out`. A missed mutant in a changed region needs a test or an entry below. The WGSL text and a second hardware adapter stay outside it, see the end of this page.

## Survivors

### CPU backends (3)

- `cpu/mod.rs`: `uses[slot] += 1` replaced by `*=`, and the guard `uses[slot] == 1` replaced by `false`. They only decide whether an output buffer is moved or copied, so the bits are the same. The opposite mutation (always move) is caught by `same_value_can_be_output_twice`.
- `cpu/kernels.rs`, `matmul_rows`: the leftover-column guard `j < j1` replaced by `j <= j1`. When `j == j1` the leftover loop runs over empty slices.

### ONNX importer (1)

- `lower.rs`, `resolve_reshape`: the guard `v > 0` replaced by `v >= 0`. The `0` entry is matched by the arm above it, so the guard never sees zero. The first run of this crate missed 19; they led to tests for the absent-output rule, operand counts, error reasons, expansion strides and attribute types, and to a fix: a Dropout with an empty second output name (the mask left out) was rejected. A review of the finished crate then found a panic on a node without outputs, a `beta = 0` Gemm that let infinity in `C` through, and int64 initializers listed as inputs being rejected; each got a test that failed first, and the attribute value check they led to got its own tests.

### GPU host code (7)

- `exec.rs` (2): the guard that sends a rank-two `[1, 0]` transpose to the tiled kernel replaced by `false`, or `==` by `!=`. The generic permute kernel gives the same bits, so only the choice of kernel changes. The identity permutation and rank-zero transposes have their own tests.
- `context.rs` (1): deleting `power_preference` from the adapter request. It only chooses between several adapters, which a one-adapter machine cannot show.
- `config.rs` (1): in `tile_grid`, `rows == 0 || cols == 0` with `||` replaced by `&&`. A zero row count still produces zero tiles, which `grid_fold` rejects. Each zero argument has its own test.
- `model.rs` (3): `k - k0` replaced by `k + k0` in `kmax`. The extra iterations read zero-filled tile cells and add `+0.0`. `t + 1` replaced by `t * 1` in the transpose tile stride: the padding avoids bank conflicts on a device, which the model does not have. The `(true, true)` arm of `binary` deleted: the fallback gives the same length.

## What mutation testing does not cover

- The Kani proof harnesses. They are compiled only by Kani (`cfg(kani)`), so no test run can see a change to them; `.cargo/mutants.toml` excludes them and Kani checks them (`just kani`).

- The WGSL kernel text. `cargo-mutants` mutates Rust only. The kernels are checked by bugs planted by hand, by the device tests, the exhaustive Relu and Sigmoid sweeps, a CPU model of every kernel grid, and bounded model checking of the index arithmetic (`just kani`).
- Other adapters. The GPU runs above used one adapter; [adapters.md](adapters.md) records the self-test on the others.

## Rule

A new survivor gets a test, or an entry here with the reason it cannot change a result. A mutation score alone says little; the list is the record.
