# 0010: Licence: MIT OR Apache-2.0, replaced by Apache-2.0 only

Date: 2026-10-20. Status: superseded.

## Context

Rust projects commonly offer the dual licence `MIT OR Apache-2.0`, and the design spec chose it. Nothing had been published when the choice was reconsidered on 2026-10-19.

## Decision

The decision recorded here is the original one: license the project as `MIT OR Apache-2.0`.

It is superseded. The project is licensed under Apache-2.0 only, with one `LICENSE` file. The reasons: Apache-2.0 matches the convention of the organisation that hosts the repository, and one licence is clearer to read than a choice between two. `repo_hygiene.rs` fails if a second licence file or a different licence string appears.

## Consequences

- Apache-2.0 is the licence in force. The dual-licence choice is kept here instead of deleted, so the history of the decision stays readable.
- Nothing was published under the dual licence, so no one holds code under it.
- Changing the licence is possible while the project has only its own code, and gets harder once others contribute.
