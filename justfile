set windows-shell := ["powershell.exe", "-NoLogo", "-Command"]

export RUSTDOCFLAGS := "-D warnings"

default: ci

fmt-check:
    cargo fmt --all -- --check

lint:
    cargo clippy --workspace --all-targets --locked -- -D warnings

test:
    cargo test --workspace --locked

# Build everything and run every test on the oldest supported compiler.
msrv:
    cargo +1.89 check --workspace --all-targets --locked
    cargo +1.89 test --workspace --locked
    cargo +1.89 test -p oxfuse-runtime -p oxfuse --features parallel --locked

doc:
    cargo doc --workspace --no-deps --locked

deny:
    cargo deny check

ci: fmt-check lint test doc

# Cross-check hand-computed examples and the committed corpus against the NumPy oracle.
oracle-check:
    python3 -I scripts/gen_expected.py examples --check
    python3 -I scripts/gen_expected.py corpus --check

# Check the committed ONNX fixtures against onnxruntime and the onnx reference
# evaluator. Needs the virtualenv: `uv venv --python 3.14 .venv-onnx` (onnxruntime
# has no wheel for newer Pythons), then `uv pip install --python .venv-onnx/bin/python -r scripts/requirements-onnx.txt`.
onnx-check:
    .venv-onnx/bin/python -I scripts/test_gen_onnx_fixtures.py
    .venv-onnx/bin/python -I scripts/gen_onnx_fixtures.py --check

# Regenerate the seeded random-graph corpus (needs numpy, see scripts/requirements.txt).
corpus:
    python3 -I scripts/gen_expected.py corpus

# Mutation testing: break the code in small ways and list the breakages no test notices.
# Slow (about 16 minutes for the first line); needs `cargo install cargo-mutants`.
# Results land in mutants.out/. Two workers, because every worker builds its own copy.
mutants:
    cargo mutants -p oxfuse-core -p oxfuse-runtime -p oxfuse --features oxfuse/wgpu,oxfuse/onnx --test-workspace true -j 2 --timeout 120
    cargo mutants -p oxfuse-onnx --test-package oxfuse-onnx -j 2 --timeout 120

# Mutants of one GPU host file, for example `just mutants-gpu crates/wgpu/src/exec.rs`. Needs an adapter.
mutants-gpu file:
    cargo mutants -p oxfuse-wgpu -f {{file}} --test-package oxfuse-wgpu -j 2 --timeout 180

# Compare a benchmark filter between a base commit and the working tree, interleaved over several rounds,
# so a result is judged against the run-to-run spread. Example:
# just bench-ab 'sigmoid/cpu' --package oxfuse-runtime --features parallel
bench-ab filter *args:
    python3 -I scripts/bench_ab.py {{args}} '{{filter}}'

# Full benchmark run. Slow, noisy on a busy machine; close other apps first.
bench:
    cargo bench -p oxfuse-runtime --locked

# Rewrite docs/benchmarks.md from the last benchmark run.
bench-report:
    python3 -I scripts/bench_report.py --write

# Compile the benchmarks without running them (what CI does).
bench-check:
    cargo bench -p oxfuse-runtime -p oxfuse-wgpu --locked --no-run

# The CLI built with the GPU backend: lint and tests (needs an adapter, or the tests do nothing).
test-wgpu:
    cargo clippy -p oxfuse -p oxfuse-wgpu --all-targets --features oxfuse/wgpu --locked -- -D warnings
    cargo test -p oxfuse -p oxfuse-wgpu --features oxfuse/wgpu --locked

# Tests with the threaded matmul compiled in.
test-parallel:
    cargo clippy -p oxfuse-runtime -p oxfuse --all-targets --features parallel --locked -- -D warnings
    cargo test -p oxfuse-runtime -p oxfuse --features parallel --locked

# Fuzzing needs the nightly toolchain and `cargo +stable install cargo-fuzz --version 0.13.2 --locked`.
fuzz_toolchain := "nightly-2026-05-22"

# Copy every shipped graph into the fuzz corpus.
fuzz-seed:
    mkdir -p fuzz/corpus/loader fuzz/corpus/differential fuzz/corpus/onnx
    cp crates/onnx/tests/fixtures/*.onnx fuzz/corpus/onnx/
    for dir in examples tests/corpus tests/invalid tests/passes; do \
      for f in $dir/*.json; do \
        case "$f" in *.inputs.json|*.expected.json) continue;; esac; \
        cp "$f" "fuzz/corpus/loader/$(echo $f | tr / _)"; \
        cp "$f" "fuzz/corpus/differential/$(echo $f | tr / _)"; \
      done; \
    done

# `just fuzz loader 60` runs one target for that many seconds. The target triple is the compiler's host, because a
# cargo-fuzz installed as a static musl binary (as in CI) would otherwise build for musl, where sanitizers fail.
fuzz target seconds="60": fuzz-seed
    cd fuzz && cargo +{{fuzz_toolchain}} fuzz run --target "$(rustc +{{fuzz_toolchain}} -vV | sed -n 's/host: //p')" {{ if target =~ '^gpu' { '--sanitizer none' } else { '' } }} {{target}} -- -max_total_time={{seconds}}

# All 2^32 f32 patterns through Relu and Sigmoid on the GPU (minutes); writes the measured Sigmoid error.
gpu-exhaustive:
    cargo test -p oxfuse-wgpu --release --locked --test sweep sweep_exhaustive -- --ignored --nocapture

# Bounded proofs of the GPU index arithmetic and of the permutation composition in the layout pass. Needs `cargo +stable install kani-verifier --version 0.68.0 --locked` and `cargo kani setup`.
kani:
    cargo kani -p oxfuse-core --lib
    cargo kani -p oxfuse-wgpu --lib

# GPU benchmarks against the fast CPU backend. Slow, needs an adapter; writes target/criterion for bench-report.
bench-gpu:
    cargo bench -p oxfuse-wgpu --locked
