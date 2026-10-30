# Contributing

Oxfuse is a small, deliberately scoped project. See the [roadmap](ROADMAP.md) for what is in and out of scope, and open an issue before proposing a larger change.

## Set up

    rustup show          # installs the toolchain pinned in rust-toolchain.toml
    just ci              # format, lint, tests, docs (install just: https://just.systems)
    just onnx-check      # ONNX fixtures vs onnxruntime and the onnx reference evaluator (needs .venv-onnx, see the recipe in the justfile)
    just oracle-check    # cross-checks examples and the corpus against NumPy (pip install -r scripts/requirements.txt)
    just mutants         # optional, about an hour (core, runtime, cli, onnx shards): lists breakages no test notices (cargo install cargo-mutants)

## How changes are made

1. Write the failing test first and watch it fail for the right reason.
2. Make it pass with the smallest change, then run `just ci`.
3. Keep one logical change per commit. Commit titles are lowercase imperative with no `type:` prefix ("fix the cycle blame for ordered nodes"). Use plain ASCII in code, comments, docs and commit messages.
4. No performance claim without a reproducible measurement, and no claim in the README that is not backed by a test or a benchmark.

## Adding things

- A new op: follow [adding-an-op.md](docs/adding-an-op.md). The test suite fails if an op has no example or no README row.
- A new pass: follow "Adding a pass" in [optimization-passes.md](docs/optimization-passes.md). Passes must be exact.

## Mutation testing

`just mutants` runs the tests against small code breakages, one shard per package (`just mutants-core`, `-runtime`, `-cli`, `-onnx`, `-wgpu`; see "Shards" in [docs/mutation-testing.md](docs/mutation-testing.md)). A surviving mutant gets a test or simpler code. It is listed in `docs/known-survivors.txt` only when no input can tell the two programs apart.

## Known test limit

The fast backend returns the reference bits by design, so no output-level test can tell whether a fast kernel is wired in or the reference kernel runs instead. The unit tests in `crates/runtime/src/cpu/kernels.rs` call the fast kernels directly, and only the benchmarks show that `apply` dispatches to them. Check `just bench` after changing the dispatch.

## Reporting an adapter

The GPU backend is measured, not supported, on every device. If you have one that is not in [docs/adapters.md](docs/adapters.md), run `cargo run --release -p oxfuse --features wgpu -- gpu-selftest` (and again with `--limits downlevel`) and open an issue with the output, your operating system and your driver version. A failing line is a result and is recorded too.

The GPU tests do nothing without an adapter, and CI sets `OXFUSE_REQUIRE_ADAPTER=1` on its software Vulkan job so a missing adapter fails there. Software adapters get a 1 by 1 matmul register block instead of 4 by 4 (see `default_config` in `crates/wgpu/src/exec.rs`), because the Windows software adapter took more than 30 seconds to run its first 4 by 4 dispatch (a 16 by 16 tile with 1 by 1 blocks runs a 128 by 128 matmul in about a second there). The same adapter crashes the process on an early `return` before a `workgroupBarrier()`, which is why the barrier kernels never return early.

## Licence

By contributing you agree that your work is licensed under Apache-2.0, as the rest of the project is.

## Fuzzing

`fuzz/` is a separate crate for cargo-fuzz (outside the workspace, nightly only; the pinned nightly is `nightly-2026-05-22`, set in the `justfile`). Install the tool once with `cargo +stable install cargo-fuzz --version 0.13.2 --locked`.

| Target | What it checks |
|---|---|
| `loader` | arbitrary text never panics the loader, and a graph that loads stays inside its limits |
| `differential` | arbitrary text that happens to load gives the same bits on the reference backend, the fast backend, and with and without the optimization passes |
| `generated` | the same comparison on graphs generated to be valid (sizes that reach the fast matmul's register block, inputs that include infinity and NaN), so nearly every execution compares something |
| `onnx` | arbitrary bytes into the ONNX importer never panic or hang, and a model that imports gives the same bits on both backends (a model carrying a non-canonical NaN is imported but not compared, because which NaN payload survives is unspecified) |
| `onnx_generated` | models built from `prost` structs, valid by construction: the importer must accept every one and the backends must agree on it |

Raw bytes rarely form a valid graph: in a 60 second `differential` run about 0.2% of executions loaded one. Quote `generated` for evidence about the backends, and use `differential` for the loader's behavior on near-valid text.

`just fuzz loader 60`, `just fuzz differential 60`, `just fuzz generated 60`, `just fuzz onnx 60` and `just fuzz onnx_generated 60` run one target for that many seconds, seeded from the shipped graphs. A weekly workflow (`fuzz.yml`) runs all five for five minutes. A crash lands in `fuzz/artifacts/`: minimise it, add it as a graph under `tests/invalid/` or a regression test, fix the cause with a test that fails first, then rerun. Pass errors end the differential comparison on purpose, because a pass can turn a run-time error into a fold-time one.

The claims ledger records how long each target has actually run. A short run proves little; the number is stated next to the claim.

## Minimum Rust version

`rust-version` is 1.89, the floor of `fearless_simd` 1.1 that the SIMD matmul kernel uses. Before that it was 1.88, measured with `cargo msrv find --min 1.85.0 -- cargo check --workspace --locked` (cargo-msrv 0.19.3) on 2026-10-20. `just msrv` and the `msrv` CI job build every target (benches included) and run the whole test suite on it, with and without the `parallel` feature. The dev dependencies (proptest 1.12 and globset need 1.88) set the floor together with the libraries. Formatting and clippy run on the pinned 1.99.1 only, because lint sets differ by version.

The floor is now set by `fearless_simd` 1.1, which declares 1.89, and `just msrv` builds and tests everything on 1.89. The number only goes up, and each stage re-measures it. A raise is a deliberate change: say why in the commit message.

## Sign-off (Developer Certificate of Origin)

By contributing you certify the [Developer Certificate of Origin](https://developercertificate.org): the work is yours, or you have the right to submit it under Apache-2.0. Say so on each commit with `git commit -s`, which adds a `Signed-off-by: Your Name <your@email>` line using the address of the commit's author. CI fails a pull request with an unsigned commit from anyone not listed in `.github/dco-exempt.txt` (the maintainer and Dependabot are). Do not copy code from another project; if a pull request must carry some, say where from and under which licence, and it will be checked against [docs/provenance.md](docs/provenance.md) and [docs/licensing.md](docs/licensing.md).

## Releasing

All five crates (`oxfuse-core`, `oxfuse-runtime`, `oxfuse-wgpu`, `oxfuse-onnx`, `oxfuse`) move together on one version. **The version lives in one place:** `[workspace.package] version` in the root `Cargo.toml`. Everything that follows it is moved or checked by `scripts/release.py`:

| Command | What it does |
|---|---|
| `just release-status` | prints the version and every place that repeats it (path dependencies, `Cargo.lock`, the comparison row) |
| `just release-check` | fails when a path dependency or `Cargo.lock` drifted, the changelog lacks `## Unreleased`, or a tool or action is pinned to two different versions across the workflows. CI runs it on every change |
| `just release-bump X.Y.Z` | sets the version and the path dependencies, rolls `## Unreleased` into `## X.Y.Z - today`, runs `cargo update --workspace` |
| `just release-post` | after publishing: sets the "Latest release" row of `docs/comparison.md` to the current version and date |

On a `release-*` branch CI also runs `release.py check --release` (the changelog must carry the version) and `cargo semver-checks --workspace` against the last published version. Run `cargo semver-checks --workspace` yourself early: a new variant on a public enum or a new field on a public struct needs a minor bump while the crates are 0.x.

A release, in order:

1. **Content PRs first**, each with its own changelog bullet under `## Unreleased`. Then one docs sweep PR (benchmark report, mutation totals from the last full run, crate READMEs: they cannot be edited on crates.io after publishing).
2. **Scans.** On a `release-*` branch CI runs a ScanCode licence scan (`licence-scan`): it fails on any licence or copyright holder outside the policy in `scripts/licence_scan_check.py`. Before the release PR, run `just licence-snippets` once on your machine (SCANOSS fingerprints of our source against open-source code; its free API rate-limits hosted runners): a new match fails until it is read and added to `ASSESSED` with the reason.
3. **The release PR.** Branch `release-X.Y.Z`, run `just release-bump X.Y.Z`, `just release-check --release`, commit as `release X.Y.Z`. Right after opening it, run the `mutants` and `fuzz` workflows on the branch (`gh workflow run mutants.yml --ref release-X.Y.Z`).
4. **After the merge, on the merged head:** CI green on all platforms, the gate (`just ci && just test-parallel && just oracle-check && cargo deny check`), `cargo publish --workspace --dry-run --locked` (no `--allow-dirty`), a `release-binaries` rehearsal with `upload` off, and the evidence archive (gate output, sweep, workspace tests, mutation outcomes, `Cargo.lock`, `rustc -vV`, machine notes, benchmark report, `SHA256SUMS`; `$HOME` scrubbed).
5. **Publish** with a token scoped to `publish-new` and `publish-update` for these names: `cargo publish --workspace --locked`. It publishes in dependency order and ends with `oxfuse`. Crates.io is immutable: a mistake means `cargo yank` and a fix in the next version.
6. **Tag and release:** `git tag -a vX.Y.Z -m vX.Y.Z <commit>`, `git push origin vX.Y.Z`, `gh release create vX.Y.Z` with the evidence archive and the golden corpus (`just golden`). Then run `release-binaries` with `ref` and `tag` set to the tag and `upload` on.
7. **Check what shipped:** `cargo install oxfuse --version X.Y.Z` in a clean directory, run the README session, compare `--backend reference` and `--backend cpu` with `cmp`, open docs.rs for all five crates, `cargo owner --list` for each, and download one binary to check `SHA256SUMS` and `gh attestation verify`.
8. **Afterwards:** `just release-post`, open a small PR with the changed row.

New crates join the publish list before `oxfuse`, and the `sara-star-quant` team becomes an owner of each right after its first publish.
