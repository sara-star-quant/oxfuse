# Adapters: measured, not supported

This page records what `oxfuse gpu-selftest` measured on devices someone ran it on. A row is a measurement of one device on one date, not a statement that the device is supported, and not a promise about any other device or driver version. Behavior outside the finite tier (NaN, infinity, subnormals) differs between devices and is recorded here, never guaranteed. The contract is in [semantics.md](semantics.md).

## Measurements

| Adapter | API | Date | Finite tier exact | Relu exact | MatMul within bound | MatMul max ulp from reference | Sigmoid max ulp from reference | Subnormal add flushed | NaN in add gives |
|---|---|---|---|---|---|---|---|---|---|
| Apple M1 Pro | Metal | 2026-10-20 | yes | yes | yes | 1720 | 43 | yes | nan |
| Apple Paravirtual device (CI macOS runner) | Metal | 2026-10-20 | yes | yes | yes | 1720 | 43 | yes | nan |
| llvmpipe (LLVM 20.1.2, 256 bits; CI software Vulkan) | Vulkan | 2026-10-20 | yes | yes | yes | 0 | 43 | no | nan |

The Sigmoid column is the self-test's sample of every exponent and both signs; only the first row also has the full sweep over all 2^32 inputs. On llvmpipe MatMul matched the reference bit for bit and subnormals were not flushed: it does not fuse multiply-add and follows IEEE for these cases, unlike the two Metal devices.

Notes on the first row, from the full runs in the test suite: Relu was bit-exact on all 2^32 bit patterns. Sigmoid was within 65 ulp of the reference on every finite input, worst at x = -86.64 (the self-test samples fewer inputs and reports a lower figure). `MatMul` always fuses multiply-add on this compiler, so it differs from the reference by up to a few thousand ulp on cancelling sums while staying inside the documented bound. Add and Mul flush subnormal values to zero, and `infinity * subnormal` gives NaN.

## How to add a row

Build with the feature and run the report:

    cargo run --release -p oxfuse --features wgpu -- gpu-selftest
    cargo run --release -p oxfuse --features wgpu -- gpu-selftest --limits downlevel

Open an issue with the output and the operating system and driver version. A report is added as a row if every "yes" in the first columns is a measurement rather than a claim. A "no" is a result and is added too.
