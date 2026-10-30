# Licensing

Oxfuse is licensed under the Apache License, Version 2.0 (`LICENSE`), copyright 2026 SARA STAR QUANT LLC. Every published crate carries `LICENSE` and `NOTICE`, and `Cargo.toml` says `license = "Apache-2.0"`. This page records what we checked and why. It is not legal advice.

## What we ship and under what terms

| Part | Terms | How it is kept in line |
|---|---|---|
| Our code (five crates, WGSL kernels, scripts, tests, docs) | Apache-2.0 | written here; no line of another project's source is copied ([provenance.md](provenance.md)) |
| Dependencies that ship in the crates | MIT, Apache-2.0, Zlib, ISC, Unicode-3.0 (several offer a choice such as "MIT OR Apache-2.0" or "Unlicense OR MIT"; we take the permissive one) | `cargo deny check` allows only those five licences, blocks unknown registries and git sources, and runs on every pull request |
| Release binaries | our `LICENSE` and `NOTICE`, plus `THIRD_PARTY_LICENSES.txt` with the full text of every licence of the code built in (`cargo-about`, same allow list) | built by the `release-binaries` workflow, checked by `scripts/package_release.py` |
| The ONNX file format | the ONNX project is Apache-2.0 (`onnx.proto` carries `SPDX-License-Identifier: Apache-2.0`). We wrote the Rust structs by hand from its field numbers and names; `NOTICE` says so | `crates/onnx/src/proto.rs` header |
| Test and benchmark tools (`proptest`, `criterion`, `insta`, `ndarray`, `matrixmultiply`, `core-math`) | MIT or Apache-2.0 | dev dependencies only; none ships in a crate |
| NumPy, onnxruntime and the `onnx` Python package | BSD-3-Clause, MIT and Apache-2.0 | run to produce or check expected files; none is copied or redistributed |
| Other projects quoted in [comparison.md](comparison.md) | their own terms | short factual quotes with the link, the version and the date read; no code |
| GitHub Actions used by the workflows | their own terms | not redistributed |

No dependency of the workspace is GPL, LGPL, AGPL, MPL or source-available (the separate `fuzz` and `scripts/exp-oracle` crates are tools and ship nowhere). One transitive crate (`r-efi`) lists LGPL-2.1-or-later as a third option next to MIT and Apache-2.0; we use it under MIT or Apache-2.0, which `cargo deny` confirms.

## Contributions

Under Apache-2.0 section 5, a contribution submitted for inclusion is licensed under the same terms unless it says otherwise ([CONTRIBUTING.md](../CONTRIBUTING.md)). There is no separate contributor agreement yet.

## Options we looked at

Read on 2026-10-30 from the sources named; the table compares terms that matter for a Rust library and tool.

| Licence | Patent grant | Copyleft | Fit |
|---|---|---|---|
| Apache-2.0 ([text](https://www.apache.org/licenses/LICENSE-2.0.txt), [summary](https://choosealicense.com/licenses/apache-2.0/)) | explicit, and it ends for anyone who sues over the Work (section 3) | none; keep the licence and `NOTICE`, state changes | what we use. Compatible with GPLv3 but, per the FSF, not with GPLv2-only ([ASF](https://www.apache.org/licenses/GPL-compatibility.html)) |
| MIT ([summary](https://choosealicense.com/licenses/mit/)) | none stated | none | shortest, no `NOTICE`, but no patent grant |
| MIT OR Apache-2.0 | from the Apache half | none | what the Rust project uses, and what its API guidelines recommend for "maximum compatibility with the Rust ecosystem" ([C-PERMISSIVE](https://rust-lang.github.io/api-guidelines/necessities.html)). Peers: tract, rten, candle and Burn are dual; ONNX Runtime is MIT; ONNX is Apache-2.0 ([comparison.md](comparison.md)) |
| MPL-2.0 ([summary](https://choosealicense.com/licenses/mpl-2.0/)) | explicit | per file | keeps changes to our files open, but adds friction for users who embed the code |
| GPL, LGPL, AGPL | varies | strong | would stop most commercial and Apache or MIT projects from using the crates |
| Source-available or field-restricted licences | varies | n/a | do not meet the Open Source Definition (criterion 6: no restriction by field of endeavor, [OSI](https://opensource.org/osd)); not considered |

## Decision

Apache-2.0 stays. The patent grant matters for a numerics library that companies may embed, the copyright holder is a company, and all dependencies fit. The one real cost is the GPLv2-only incompatibility, which affects few projects.

Revisit before the first outside contribution is merged. Adding "OR MIT" is easy while the company holds all the copyright, and needs every contributor's agreement afterwards. If outside contributions start, add a sign-off rule (DCO) or a contributor agreement at the same time.
