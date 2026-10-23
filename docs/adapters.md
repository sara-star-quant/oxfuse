# Adapters: measured, not supported

This page records what `oxfuse gpu-selftest` measured on devices someone ran it on. A row is a measurement of one device on one date, not a statement that the device is supported, and not a promise about any other device or driver version. Behavior outside the finite tier (NaN, infinity, subnormals) differs between devices and is recorded here, never guaranteed. The contract is in [semantics.md](semantics.md).

## Measurements

| Adapter | API | Date | Finite tier exact | Relu exact | MatMul within bound | MatMul max ulp from reference | Sigmoid max ulp from reference | Subnormal add flushed | NaN in add gives |
|---|---|---|---|---|---|---|---|---|---|
| Apple M1 Pro | Metal | 2026-10-22 | yes | yes | yes | 0 | 43 | yes | nan |
| Apple Paravirtual device (CI macOS runner) | Metal | 2026-10-23 | yes | yes | yes | 0 | 43 | yes | nan |
| llvmpipe (LLVM 20.1.2, 256 bits; CI software Vulkan) | Vulkan | 2026-10-23 | yes | yes | yes | 1720 | 43 | no | nan |
| Tesla T4 (driver 580, free notebook) | Vulkan | 2026-10-22 | yes | yes | yes | 0 | 42 | yes | nan |

The Sigmoid column is the self-test's sample of every exponent and both signs; only the first row also has the full sweep over all 2^32 inputs. On llvmpipe MatMul does not fuse multiply-add and subnormals are not flushed: it follows IEEE for these cases, so it is 1720 ulp from the reference (which fuses) on the sampled cancelling sums, the same figure the Metal devices had before the reference fused. The two Metal devices and the T4 fuse like the reference and match it bit for bit.

Notes on the first row, from the full runs in the test suite: Relu was bit-exact on all 2^32 bit patterns. Sigmoid was within 65 ulp of the reference on every finite input, worst at x = -86.64 (the self-test samples fewer inputs and reports a lower figure). `MatMul` fuses multiply-add here, like the reference does now, so it matches the reference bit for bit (it was up to 1720 ulp off when the reference did not fuse). Add and Mul flush subnormal values to zero, and `infinity * subnormal` gives NaN.

The Tesla T4 row is the first real NVIDIA card. Like the Metal rows, its MatMul fuses multiply-add and flushes subnormals. It now matches the reference bit for bit on every case in the test suite (0 ulp); before matmul fused, it was 1720 ulp off on the cancelling sums. The Sigmoid figure is the self-test sample, not the full sweep.

## How to add a row

Build with the feature and run the report:

    cargo run --release -p oxfuse --features wgpu -- gpu-selftest
    cargo run --release -p oxfuse --features wgpu -- gpu-selftest --limits downlevel

Open an issue with the output and the operating system and driver version. A report is added as a row if every "yes" in the first columns is a measurement rather than a claim. A "no" is a result and is added too.

## Trying a card you do not have

A free notebook with an NVIDIA GPU (Kaggle or Colab; the account needs to be verified for GPU and internet access) works for a quick run. These are the steps that produced the Tesla T4 row, on Ubuntu 24.04 as root:

    apt-get update && apt-get install -y vulkan-tools libvulkan1 build-essential pkg-config curl
    nvidia-smi --query-gpu=driver_version --format=csv,noheader     # first number is the major version, for example 580
    apt-get install -y libnvidia-gl-580                              # use your major version
    vulkaninfo --summary | grep deviceName                           # should list the GPU, not only llvmpipe
    curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal && . $HOME/.cargo/env
    cargo install oxfuse --features wgpu
    oxfuse gpu-selftest

The notebook image has the CUDA driver but not the Vulkan one, which is why the `libnvidia-gl` package is needed. Without it `vulkaninfo` finds no driver and the run falls back to a software adapter or fails. The build takes several minutes on a notebook CPU, so build in a session without a GPU and run the self-test in a short GPU session if the GPU hours are limited.

