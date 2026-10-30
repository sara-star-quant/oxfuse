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

Under Apache-2.0 section 5, a contribution submitted for inclusion is licensed under the same terms unless it says otherwise. On top of that, every commit from outside the maintainer list needs a `Signed-off-by` line, the [Developer Certificate of Origin](https://developercertificate.org) ([CONTRIBUTING.md](../CONTRIBUTING.md)). CI enforces it with `scripts/dco_check.py`.

## Scans

Two scanners, because they find different things.

- **ScanCode** (licences and copyright holders, per file) runs in CI on every `release-*` branch (`licence-scan`) and fails on a licence or a copyright holder outside the policy in `scripts/licence_scan_check.py`. The first run looked at 482 files: our files carry Apache-2.0 and our holder, no other holder appears, and other licences are named only in seven files that list or discuss them (the allow lists, this page, the provenance ledger, the comparison page, and tests and scripts that read licence text).
- **SCANOSS** (snippets of other open-source code, matched by fingerprint) is a local release step, `just licence-snippets`, because its free API rate-limits hosted runners (we got HTTP 503 on the first CI try). It sends fingerprints, not source. On 2026-10-30 it fingerprinted 131 source files and matched 4; all four were read and judged: a derive line and a newtype index in `types.rs`, the prost structs of `proto.rs`, the Numerical Recipes generator constants in the benchmark data, and a `urllib` request in the link checker. A new match fails the step until it is read and added to the list with its reason.

The `proto.rs` match deserves a note. SCANOSS matched 75 percent of it to a crate (`kenosis-core` 0.3.1, Apache-2.0) that is no longer on crates.io, so we could not read it. The structs follow the field numbers and names of `onnx.proto`, which fixes most of the text, and the match is Apache-2.0, the same licence as ours. We recorded it and credited ONNX in `NOTICE`.

## Risks beyond the licence

None of these is a reason to stop. Each says what we do and what is open.

| Risk | What we found | What we do, and what is open |
|---|---|---|
| Code written with AI assistance | Copyright needs human authorship. The US Copyright Office's January 2025 report says prompts alone do not make a user the author of AI output, while human selection, arrangement and modification can be protected ([overview at copyright.gov/ai](https://www.copyright.gov/ai/); we read law-firm summaries, so read the report itself). The terms of the AI tool decide who owns the output as between you and the vendor (for Anthropic's commercial terms, the customer, [terms](https://www.anthropic.com/legal/commercial-terms); check the plan in use) | A person directs, reviews and tests every change, and the design decisions are recorded in the repository history. Parts that are purely machine-written may not be protectable, which would only mean anyone may reuse them. Accidental copying is what the snippet scan is for. Open: no lawyer has reviewed this |
| The name | A web search found no trademark or company called Oxfuse. It did find a small D-language package named `oxfuse` (a binding to libfuse3 for building Linux filesystems, on [Codeberg](https://codeberg.org/os-18/oxfuse) and the [D package index](https://codemirror.dlang.org/packages/oxfuse)) and the older macOS `osxfuse` | Different field, no registration found. Open: the USPTO, EUIPO and WIPO Global Brand databases were not searched (web search does not cover them) and are worth a check before any filing or heavier promotion |
| Who contributed what | No outside contributions yet | The sign-off rule above, so the chain starts clean |
| Liability | Apache-2.0 sections 7 and 8 disclaim warranty and liability | The README and [use-cases.md](use-cases.md) say experimental and not for production inference |
| EU Cyber Resilience Act | The Act applies from 11 December 2027, with reporting duties from 11 September 2026 ([Commission](https://digital-strategy.ec.europa.eu/en/policies/cyber-resilience-act)). Free and open-source software supplied outside a commercial activity is outside it. A legal person that supports such a project on a sustained basis for commercial use can be an "open-source software steward", with duties to document a cybersecurity policy and cooperate with authorities ([OpenSSF summary](https://policy.openssf.org/CRA/stewards-one-pager.html)). A company that sells a product containing the code is a manufacturer | `SECURITY.md` is a first vulnerability-handling policy. Open: whether SARA STAR QUANT LLC counts as a steward or manufacturer depends on its commercial activity; ask counsel before charging for support, hosting or binaries |
| Patents | Nothing found; none searched | The techniques are published and old; Apache-2.0 gives users a patent licence from us. Open: no freedom-to-operate search |

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
