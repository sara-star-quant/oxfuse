# Contributing

Oxfuse is a small, deliberately scoped project. See the [roadmap](ROADMAP.md) for what is in and out of scope, and open an issue before proposing a larger change.

## Set up

    rustup show          # installs the toolchain pinned in rust-toolchain.toml
    just ci              # format, lint, tests, docs (install just: https://just.systems)
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

## Known equivalent mutants

`cargo mutants` reports these in `crates/runtime/src/cpu/mod.rs` as missed. Neither can change a result: they only decide whether an output buffer is moved or copied, so the bits are the same.

- `uses[slot] += 1` replaced by `*=`: every count stays 0, so every output is copied.
- the guard `uses[slot] == 1` replaced by `false`: every output is copied.

The opposite mutation (always move) is caught by `same_value_can_be_output_twice`.

In `matmul_rows` (`crates/runtime/src/cpu/kernels.rs`), replacing the leftover-column guard `j < j1` with `j <= j1` is also missed. When `j == j1` the leftover loop runs over empty slices, so nothing changes. A related mutant, `+=` replaced by `*=` in that leftover loop, times out instead of failing: the result is wrong and the run is killed, which counts as caught.

### GPU crate

Mutation testing of the host-side code in `crates/wgpu/src/{config,bounds,model,exec}.rs` (about 450 mutants, run per file with the test binaries that exercise it, on the recorded adapter) leaves only mutants that cannot change an output. The WGSL text is not covered by `cargo mutants`; the kernels are checked by planting bugs in them by hand (see the ledger) and by the device tests.

- `exec.rs`: `|` replaced by `^` on `wgpu::BufferUsages` flags. The flags are disjoint bits, so the result is the same.
- `exec.rs`: the guard that sends a rank-two `[1, 0]` transpose to the tiled kernel replaced by `false`, or `len() == 2` replaced by `!=`. The generic permute kernel gives the same bits, so only the choice of kernel changes. The identity permutation `[0, 1]` and rank-zero transposes have their own tests.
- `context.rs`: deleting `power_preference` from the adapter request. It only chooses between several adapters, which a one-adapter machine cannot show.
- `config.rs`: in `tile_grid`, `rows == 0 || ...` replaced by `&&`. A zero row count still produces zero tiles, which `grid_fold` rejects.
- `model.rs`: the matmul tile loop bound `k0 < k` replaced by `<=`, or `k - k0` replaced by `k + k0` in `kmax`. The extra iterations read zero-filled tile cells and add `+0.0`, which cannot change a sum that starts at `+0.0`. The same holds for the shader's bound.
- `model.rs`: `t + 1` replaced by `t * 1` in the transpose tile stride. The padding exists to avoid bank conflicts on a device, which the model does not have.
- `model.rs`: the `(true, true)` arm of `binary` deleted. The fallback gives the same length.
- `model.rs`: `k0 += t` replaced by `*=` times out: the loop never ends, which counts as caught.

## Known test limit

The fast backend returns the reference bits by design, so no output-level test can tell whether a fast kernel is wired in or the reference kernel runs instead. The unit tests in `crates/runtime/src/cpu/kernels.rs` call the fast kernels directly, and only the benchmarks show that `apply` dispatches to them. Check `just bench` after changing the dispatch.

## Reporting an adapter

The GPU backend is measured, not supported, on every device. If you have one that is not in [docs/adapters.md](docs/adapters.md), run `cargo run --release -p oxfuse --features wgpu -- gpu-selftest` (and again with `--limits downlevel`) and open an issue with the output, your operating system and your driver version. A failing line is a result and is recorded too.

The GPU tests do nothing without an adapter, and CI sets `OXFUSE_REQUIRE_ADAPTER=1` on its software Vulkan job so a missing adapter fails there. They are skipped on Windows (set `OXFUSE_WINDOWS_GPU_TESTS=1` to run them): on the CI runner the DX12 software adapter passed every test and then crashed the test process at exit, and the cause is not diagnosed.

## Licence

By contributing you agree that your work is licensed under Apache-2.0, as the rest of the project is.

## Fuzzing

`fuzz/` is a separate crate for cargo-fuzz (outside the workspace, nightly only; the pinned nightly is `nightly-2026-05-22`, set in the `justfile`). Install the tool once with `cargo +stable install cargo-fuzz --version 0.13.2 --locked`.

| Target | What it checks |
|---|---|
| `loader` | arbitrary text never panics the loader, and a graph that loads stays inside its limits |
| `differential` | arbitrary text that happens to load gives the same bits on the reference backend, the fast backend, and with and without the optimization passes |
| `generated` | the same comparison on graphs generated to be valid (sizes that reach the fast matmul's register block, inputs that include infinity and NaN), so nearly every execution compares something |

Raw bytes rarely form a valid graph: in a 60 second `differential` run about 0.2% of executions loaded one. Quote `generated` for evidence about the backends, and use `differential` for the loader's behavior on near-valid text.

`just fuzz loader 60`, `just fuzz differential 60` and `just fuzz generated 60` run one target for that many seconds, seeded from the shipped graphs. A weekly workflow (`fuzz.yml`) runs all three for five minutes. A crash lands in `fuzz/artifacts/`: minimise it, add it as a graph under `tests/invalid/` or a regression test, fix the cause with a test that fails first, then rerun. Pass errors end the differential comparison on purpose, because a pass can turn a run-time error into a fold-time one.

The claims ledger records how long each target has actually run. A short run proves little; the number is stated next to the claim.

## Minimum Rust version

`rust-version` is 1.88, measured with `cargo msrv find --min 1.85.0 -- cargo check --workspace --locked` (cargo-msrv 0.19.3) on 2026-10-20. `just msrv` and the `msrv` CI job build every target (benches included) and run the whole test suite on it, with and without the `parallel` feature. The dev dependencies (proptest 1.12 and globset need 1.88) set the floor together with the libraries. Formatting and clippy run on the pinned 1.99.1 only, because lint sets differ by version.

Re-measured at the end of the hardening stage with all targets (`cargo check --workspace --all-targets`): still 1.88. The number only goes up, and each stage re-measures it. A raise is a deliberate change: say why in the commit message.

## Releasing

Not done yet: no crate is published. The manifests are ready, and `cargo publish --workspace --dry-run --locked` passes. A release, in order:

1. The repository is public, so the crate pages link somewhere real.
2. Publish from GitHub Actions with crates.io trusted publishing, or with a token scoped to `publish-new` and `publish-update` for these names.
3. `cargo publish --workspace --locked` publishes `oxfuse-core`, `oxfuse-runtime`, then `oxfuse` (the command-line tool).
4. Add the `sara-star-quant` team as an owner of all three crates.
