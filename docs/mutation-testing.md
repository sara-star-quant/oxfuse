# Mutation testing

`cargo-mutants` 27.1.0 breaks the code in small ways (a `<` becomes `<=`, a function returns a constant) and reruns the tests. A mutant that no test notices is a gap in the tests, or a change that cannot alter any result. Every survivor below is the second kind, with the reason. Nothing is left unexplained.

## Result

Run on an Apple M1 Pro on 2026-10-21. Rows marked "crates tree 026fd4c" ran on that tree, `selftest.rs` ran on tree da4eea4 after its tests were added (`git rev-parse --short HEAD:crates` names the tree).

| Scope | Mutants | Caught | Timed out (counts as caught) | Unviable (do not compile) | Missed |
|---|---|---|---|---|---|
| `oxfuse-core`, `oxfuse-runtime`, `oxfuse` (CLI, built with the GPU feature) | 594 | 538 | 4 | 49 | 3 |
| `oxfuse-wgpu` `config.rs` | 100 | 98 | 0 | 1 | 1 |
| `oxfuse-wgpu` `bounds.rs` | 45 | 45 | 0 | 0 | 0 |
| `oxfuse-wgpu` `model.rs` | 250 | 243 | 1 | 2 | 4 |
| `oxfuse-wgpu` `exec.rs` | 81 | 58 | 0 | 15 | 8 |
| `oxfuse-wgpu` `context.rs` | 12 | 7 | 0 | 4 | 1 |
| `oxfuse-wgpu` `selftest.rs` (tree da4eea4) | 89 | 84 | 0 | 3 | 2 |
| Total | 1171 | 1073 | 5 | 74 | 19 |

Of the 1097 mutants that compile, 1078 were stopped (98.3%) and the other 19 are explained below. The first self-test run missed 40 of 77, which led to extracting its logic into small functions with their own tests; that is why it has its own tree id.

Reproduce:

    cargo mutants -p oxfuse-core -p oxfuse-runtime -p oxfuse --features oxfuse/wgpu --test-workspace true -j 2 --timeout 120
    cargo mutants -p oxfuse-wgpu -f crates/wgpu/src/exec.rs --test-package oxfuse-wgpu -j 2 --timeout 180

The second command is per file; the GPU tests need an adapter, otherwise most of them do nothing. Use `-j 2` and `CARGO_INCREMENTAL=0`, because every worker builds its own copy and four workers fill a laptop disk.

## Survivors

### CPU backends (3)

- `cpu/mod.rs`: `uses[slot] += 1` replaced by `*=`, and the guard `uses[slot] == 1` replaced by `false`. They only decide whether an output buffer is moved or copied, so the bits are the same. The opposite mutation (always move) is caught by `same_value_can_be_output_twice`.
- `cpu/kernels.rs`, `matmul_rows`: the leftover-column guard `j < j1` replaced by `j <= j1`. When `j == j1` the leftover loop runs over empty slices.

### GPU host code (16)

- `exec.rs` (6): `|` replaced by `^` on `wgpu::BufferUsages` flags, in `build`, `upload`, `download` and `run_device`. The flags are disjoint bits, so the result is the same.
- `exec.rs` (2): the guard that sends a rank-two `[1, 0]` transpose to the tiled kernel replaced by `false`, or `==` by `!=`. The generic permute kernel gives the same bits, so only the choice of kernel changes. The identity permutation and rank-zero transposes have their own tests.
- `context.rs` (1): deleting `power_preference` from the adapter request. It only chooses between several adapters, which a one-adapter machine cannot show.
- `config.rs` (1): in `tile_grid`, `rows == 0 || cols == 0` with `||` replaced by `&&`. A zero row count still produces zero tiles, which `grid_fold` rejects. Each zero argument has its own test.
- `model.rs` (4): the matmul tile loop bound `k0 < k` replaced by `<=`, and `k - k0` replaced by `k + k0` in `kmax`. The extra iterations read zero-filled tile cells and add `+0.0`. `t + 1` replaced by `t * 1` in the transpose tile stride: the padding avoids bank conflicts on a device, which the model does not have. The `(true, true)` arm of `binary` deleted: the fallback gives the same length.
- `selftest.rs` (2): `|` replaced by `^` when `pattern_sweep` builds a bit pattern from sign, exponent and mantissa. The fields do not overlap.

One more mutant, `k0 += t` replaced by `*=` in the model's matmul loop, never ends and times out. A timeout counts as caught.

## What mutation testing does not cover

- The WGSL kernel text. `cargo-mutants` mutates Rust only. The kernels are checked by bugs planted by hand, by the device tests, the exhaustive Relu and Sigmoid sweeps, a CPU model of every kernel grid, and bounded model checking of the index arithmetic (`just kani`).
- Other adapters. The GPU runs above used one adapter; [adapters.md](adapters.md) records the self-test on the others.

## Rule

A new survivor gets a test, or an entry here with the reason it cannot change a result. A mutation score alone says little; the list is the record.
