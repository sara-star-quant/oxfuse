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

## Shards

A shard is one `cargo mutants` run over one package, with the tests that cover it. Mutation testing is split this way because `cargo-mutants` builds the baseline only for the packages whose mutants it runs, so a feature of another package (`oxfuse/wgpu`) cannot be named in a run that selects `oxfuse-core`, and because one run over everything exceeds a two hour CI job. Each shard is a matrix entry of the weekly job in `.github/workflows/mutants.yml` and a recipe in the `justfile`; the local command is the CI command.

| Shard | Recipe | Mutates | Notes |
|---|---|---|---|
| core | `just mutants-core` | `crates/core` | |
| runtime | `just mutants-runtime` | `crates/runtime`, including the SIMD kernels and `math.rs`; built with the `parallel` feature so the threaded code is compiled | 491 mutants at about 90 s each on a hosted runner: CI splits it into 6 sub-shards (`--shard k/6`) that run in parallel |
| cli | `just mutants-cli` | `crates/cli`, built with `wgpu` and `onnx` | runs with an adapter (software Vulkan in CI) |
| onnx | `just mutants-onnx` | `crates/onnx` | tested by its own package only |
| wgpu | `just mutants-wgpu` | `crates/wgpu` | 634 mutants, about 100 minutes on an M1 Pro; needs a real adapter, so it runs locally and, for the changed lines only, in the pull request job on software Vulkan |

A package that does not fit one two hour CI job is split further with `cargo-mutants`'s `--shard k/n` (k counts from 0, so `--shard 0/6` to `--shard 5/6`), which deals the mutants of a run out to `n` jobs; the sub-shards run in parallel, so the wall time is that of the largest one. Choose `n` so a sub-shard has at most about 100 mutants (about 75 minutes on a hosted runner). A sub-shard checks only for new misses; the check for entries that are now caught runs on unsharded packages, because one sub-shard cannot see the mutants of another.

`just mutants` runs core, runtime, cli and onnx in a row. Narrow any shard with extra arguments or a name filter, for example `just mutants-wgpu -f crates/wgpu/src/exec.rs` or `cargo mutants -p oxfuse-runtime -F 'matmul.rs'` (the filter matches `path:line:col: description`).

Every shard uses `-j 2 --timeout-multiplier 5 --minimum-test-timeout 20`: two workers (each builds its own copy of the tree, so more fill a laptop disk), and a hung mutant is cut at five times the baseline test time instead of a fixed two minutes. A mutant that only times out counts as caught.

Do not switch branches or edit files while a shard runs. `cargo-mutants` copies the tree per worker as it goes and fails with "does not exist" or "File exists" if the tree changes underneath it. Run it from a detached worktree (`git worktree add --detach <dir> <branch>`) when you need to keep working.

## In CI

`.github/workflows/mutants.yml` runs three kinds of job, none of which blocks a merge:

- **Pull request, CPU side:** only the mutants that overlap the diff (`--in-diff`), all features built, so code behind `onnx`, `wgpu` and `parallel` is tested.
- **Pull request, GPU:** the same for `oxfuse-wgpu` on a software Vulkan adapter.
- **Weekly (Sunday 03:00 UTC) and on demand:** every shard above except wgpu, one job each. Start one with `gh workflow run mutants --ref <branch>`.

A job turns red when a mutant is missed and is not listed in [known-survivors.txt](known-survivors.txt), when a listed one is now caught (a full run only: the list may only shrink), or when the baseline fails. `scripts/mutants_check.py` makes that comparison, with line numbers stripped from the names; `just mutants-check crates/core/` runs it on the last local result. The missed list of each job is the artifact `mutants.out`.

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
