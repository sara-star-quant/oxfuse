# Contributing

Oxfuse is a small, deliberately scoped project. See the [roadmap](ROADMAP.md) for what is in and out of scope, and open an issue before proposing a larger change.

## Set up

    rustup show          # installs the toolchain pinned in rust-toolchain.toml
    just ci              # format, lint, tests, docs (install just: https://just.systems)
    just onnx-check      # ONNX fixtures vs onnxruntime and the onnx reference evaluator (needs .venv-onnx, see the recipe in the justfile)
    just oracle-check    # cross-checks examples and the corpus against NumPy (pip install -r scripts/requirements.txt)
    just mutants         # optional, about 16 minutes: lists breakages no test notices (cargo install cargo-mutants)

## How changes are made

1. Write the failing test first and watch it fail for the right reason.
2. Make it pass with the smallest change, then run `just ci`.
3. Keep one logical change per commit. Commit titles are lowercase imperative with no `type:` prefix ("fix the cycle blame for ordered nodes"). Use plain ASCII in code, comments, docs and commit messages.
4. No performance claim without a reproducible measurement, and no claim in the README that is not backed by a test or a benchmark.

## Adding things

- A new op: follow [adding-an-op.md](docs/adding-an-op.md). The test suite fails if an op has no example or no README row.
- A new pass: follow "Adding a pass" in [optimization-passes.md](docs/optimization-passes.md). Passes must be exact.

## Mutation testing

`just mutants` runs the tests against small code breakages. The results, how to rerun them and the explained survivors are in [docs/mutation-testing.md](docs/mutation-testing.md). A new survivor gets a test or an entry there.

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

`rust-version` is 1.88, measured with `cargo msrv find --min 1.85.0 -- cargo check --workspace --locked` (cargo-msrv 0.19.3) on 2026-10-20. `just msrv` and the `msrv` CI job build every target (benches included) and run the whole test suite on it, with and without the `parallel` feature. The dev dependencies (proptest 1.12 and globset need 1.88) set the floor together with the libraries. Formatting and clippy run on the pinned 1.99.1 only, because lint sets differ by version.

Re-measured at the end of the hardening stage with all targets (`cargo check --workspace --all-targets`): still 1.88. The number only goes up, and each stage re-measures it. A raise is a deliberate change: say why in the commit message.

## Releasing

Version 0.1.0 of `oxfuse-core`, `oxfuse-runtime` and `oxfuse` was published first. Version 0.2.0 added `oxfuse-wgpu` and `Error::Backend`, which breaks exhaustive matches on `Error`; 0.3.0 added `oxfuse-onnx`. All five crates move together. A release, in order:

1. Run the gate: `just ci && just test-parallel && just test-wgpu && just oracle-check && just msrv`, and `cargo publish --workspace --dry-run --locked`.
2. Publish with a token scoped to `publish-new` and `publish-update` for these names (or with crates.io trusted publishing from GitHub Actions): `cargo publish --workspace --locked`. It publishes `oxfuse-core`, `oxfuse-runtime`, `oxfuse-wgpu`, `oxfuse-onnx`, then `oxfuse` (the command-line tool).
3. Tag the commit and create a GitHub release.
4. In a clean directory with no path dependencies, run `cargo install oxfuse --version <new>` and a small consumer crate against the registry versions, and check docs.rs built every crate.
5. Add the `sara-star-quant` team as an owner of all five crates.
