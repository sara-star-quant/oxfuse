# Mutation testing

`cargo-mutants` 27.1.0 breaks the code in small ways (a `<` becomes `<=`, a function returns a constant) and reruns the tests. A mutant that no test notices is a gap in the tests or redundant code. The rule here is to fix the cause, with a test or simpler code, and to list a survivor only when no input and no platform can tell the two programs apart. Today the list is empty.

## Result

CPU-side crates, full run of every shard in CI on 2026-10-23 (run 39985230656, `ubuntu-latest`, commit of PR 42):

| Shard | Mutants | Caught | Timed out (counts as caught) | Unviable (do not compile) | Missed |
|---|---|---|---|---|---|
| core | 201 | 170 | 0 | 31 | 0 |
| runtime (6 sub-shards) | 480 | 454 | 0 | 26 | 0 |
| cli | 113 | 93 | 6 | 14 | 0 |
| onnx | 268 | 250 | 0 | 18 | 0 |
| **Total** | **1062** | **967** | **6** | **89** | **0** |

All 973 mutants that compile were stopped. The six runtime sub-shards add up to the 480 mutants `cargo mutants --list` reports for the package with the `parallel` feature; check that sum whenever the shard count changes (shards are numbered from 0, and a skipped shard shows only as a smaller total).

GPU crate, run locally on an Apple M1 Pro (Metal) because it needs a real adapter. It has 624 mutants since the Kani-only file `proofs.rs` is excluded (`.cargo/mutants.toml`). It ran in three passes on 2026-10-23, each on the code of that moment:

| Pass | Mutants | Caught | Timed out | Unviable | Missed |
|---|---|---|---|---|---|
| 1. whole crate, stopped at 587 of 634 | 587 | 540 | 1 | 37 | 9 |
| 2. the sites fixed after pass 1 plus the 47 not reached | 212 | 205 | 0 | 6 | 1 |
| 3. `binary` and `transpose2d` after the last fix | 65 | 65 | 0 | 0 | 0 |

The 9 misses of pass 1 were six speed-only or redundant sites (`tile_grid`, the adapter request, the transpose kernel guard twice, the `(true, true)` arm of `binary`, the transpose tile stride) and three constants inside the Kani-only module. The six were fixed with a pure function, a stored choice a test can read, or deleted code, and the constants were excluded with their file. Pass 2 found one more in `binary` (an `&&` that equal-length operands made equivalent); the length is now the longer operand and pass 3 confirms it. The mutation job for pull requests runs the changed GPU lines on software Vulkan; the whole crate is a local run.

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

- **Pull request, CPU side:** only the mutants that overlap the diff (`--in-diff`), all features built, so code behind `onnx`, `wgpu` and `parallel` is tested. The job runs as four shards on four runners, and a new push to the pull request cancels the running jobs.
- **Pull request, GPU:** the same for `oxfuse-wgpu` on a software Vulkan adapter.
- **Weekly (Sunday 03:00 UTC) and on demand:** every shard above except wgpu, one job each. Start one with `gh workflow run mutants --ref <branch>`.

A job turns red when a mutant is missed and is not listed in [known-survivors.txt](known-survivors.txt), when a listed one is now caught (a full run only: the list may only shrink), or when the baseline fails. `scripts/mutants_check.py` makes that comparison, with line numbers stripped from the names; `just mutants-check crates/core/` runs it on the last local result. The missed list of each job is the artifact `mutants.out`.

## Survivors

None. `known-survivors.txt` is empty, and the mutation job turns red on any miss.

The decisions that used to survive are now visible to tests: whether a Sigmoid or a transpose is split across threads is a pure function with exact boundary tests; whether an executor step reuses a buffer, or copies an output, is counted by a test-only counter; which transpose kernel runs is stored where a test reads it; the order of the operands of an in-place `Add` or `Mul` was deleted, since only a NaN payload depends on it and the contract counts all NaNs as one value.

## What mutation testing does not cover

- The Kani proof harnesses. They are compiled only by Kani (`cfg(kani)`), so no test run can see a change to them; `.cargo/mutants.toml` excludes them and Kani checks them (`just kani`).

- The WGSL kernel text. `cargo-mutants` mutates Rust only. The kernels are checked by bugs planted by hand, by the device tests, the exhaustive Relu and Sigmoid sweeps, a CPU model of every kernel grid, and bounded model checking of the index arithmetic (`just kani`).
- Other adapters. The GPU runs above used one adapter; [adapters.md](adapters.md) records the self-test on the others.

## Rule

A surviving mutant gets a test or simpler code. It is listed in `known-survivors.txt` only when no input and no platform can tell the two programs apart, with the reason in this file. A mutation score alone says little; the evidence is the empty list, the shards that reproduce it and the tests that name what they pin.
