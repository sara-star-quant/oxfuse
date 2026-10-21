# 0008: published format versions keep loading

Date: 2026-10-20. Status: accepted.

## Context

The JSON file is the interface users depend on. A format that changes silently makes every saved graph a risk.

## Decision

`format_version` is required and must be 1 today. Once a version is published it does not change. A change that would break existing files gets a new number, and later releases read both. Unknown fields are errors, so a typo is never ignored.

## Consequences

- Files written today keep loading. New fields arrive as a new version, not as a quiet extension.
- Strict parsing means a field added in a later version fails loudly in an older reader instead of being dropped.
- Version 1 has been public since release 0.1.0, so it is fixed.
