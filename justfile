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
    cargo +1.88 check --workspace --all-targets --locked
    cargo +1.88 test --workspace --locked
    cargo +1.88 test -p oxfuse-runtime -p oxfuse --features parallel --locked

doc:
    cargo doc --workspace --no-deps --locked

deny:
    cargo deny check

ci: fmt-check lint test doc

# Cross-check hand-computed examples and the committed corpus against the NumPy oracle.
oracle-check:
    python3 -I scripts/gen_expected.py examples --check
    python3 -I scripts/gen_expected.py corpus --check

# Regenerate the seeded random-graph corpus (needs numpy, see scripts/requirements.txt).
corpus:
    python3 -I scripts/gen_expected.py corpus

# Mutation testing: break the code in small ways and list the breakages no test notices.
# Slow (about 16 minutes); needs `cargo install cargo-mutants`. Results land in mutants.out/.
mutants:
    cargo mutants --test-workspace true -j 4 --timeout 120

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
    mkdir -p fuzz/corpus/loader fuzz/corpus/differential
    for dir in examples tests/corpus tests/invalid tests/passes; do \
      for f in $dir/*.json; do \
        case "$f" in *.inputs.json|*.expected.json) continue;; esac; \
        cp "$f" "fuzz/corpus/loader/$(echo $f | tr / _)"; \
        cp "$f" "fuzz/corpus/differential/$(echo $f | tr / _)"; \
      done; \
    done

# `just fuzz loader 60` runs one target for that many seconds.
fuzz target seconds="60": fuzz-seed
    cd fuzz && cargo +{{fuzz_toolchain}} fuzz run {{ if target =~ '^gpu' { '--sanitizer none' } else { '' } }} {{target}} -- -max_total_time={{seconds}}

# All 2^32 f32 patterns through Relu and Sigmoid on the GPU (minutes); writes the measured Sigmoid error.
gpu-exhaustive:
    cargo test -p oxfuse-wgpu --release --locked --test sweep sweep_exhaustive -- --ignored --nocapture

# Bounded proofs of the GPU index arithmetic. Needs `cargo +stable install kani-verifier --version 0.68.0 --locked` and `cargo kani setup`.
kani:
    cargo kani -p oxfuse-wgpu --lib

# GPU benchmarks against the fast CPU backend. Slow, needs an adapter; writes target/criterion for bench-report.
bench-gpu:
    cargo bench -p oxfuse-wgpu --locked
