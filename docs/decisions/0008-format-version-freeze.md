# 0008: format version 1 is frozen once published

Date: 2026-10-20. Status: accepted.

## Context

The JSON file is the interface users depend on. A format that changes silently makes every saved graph a risk.

## Decision

`format_version` is required and must be 1. Once the format is published, a breaking change bumps the version and the loader keeps reading the old one. Unknown fields are errors, so a typo is never ignored.

## Consequences

- Files written today keep loading. New fields arrive as a new version, not as a quiet extension.
- Strict parsing means a field added in a later version fails loudly in an older reader instead of being dropped.
- Nothing is published yet. The freeze takes effect at the first public release.
