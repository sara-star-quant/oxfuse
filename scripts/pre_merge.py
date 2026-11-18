"""Reads every check of a pull request before it is merged.

  python3 -I scripts/pre_merge.py PR [--accept NAME ...]

The branch ruleset requires one check, `required`. Mutation testing, Kani, the
semver check and the others run on the pull request but cannot block a merge (they
are slow or path-filtered), so a red one is easy to miss. This exits 1 on any check
that failed, was cancelled or is still running, unless it is named with --accept.
`required` itself can never be accepted away. Standard library plus the `gh` CLI.
"""
import json
import subprocess
import sys

GATE = "required"


def problems(checks, accepted=()):
    """Lines that stop a merge: (check, why)."""
    found = []
    seen_gate = False
    for c in checks:
        name, bucket = c.get("name", "?"), c.get("bucket", "?")
        workflow = c.get("workflow", "")
        label = f"{workflow}/{name}" if workflow else name
        if name == GATE:
            seen_gate = True
            if bucket != "pass":
                found.append((label, f"the gate is {bucket}"))
            continue
        if bucket in ("pass", "skipping"):
            continue
        if name in accepted or label in accepted:
            continue
        found.append((label, bucket))
    if not seen_gate:
        found.append((GATE, "the gate has not reported"))
    return sorted(found)


def main(argv):
    args = argv[1:]
    if not args or not args[0].isdigit():
        print(__doc__)
        return 2
    pr, accepted = args[0], []
    rest = args[1:]
    while rest:
        if rest[0] != "--accept" or len(rest) < 2:
            print(__doc__)
            return 2
        accepted.append(rest[1])
        rest = rest[2:]
    # gh exits 8 while checks are pending; the JSON is still on stdout.
    out = subprocess.run(
        ["gh", "pr", "checks", pr, "--json", "name,workflow,bucket"],
        capture_output=True,
        text=True,
    )
    try:
        checks = json.loads(out.stdout)
    except json.JSONDecodeError:
        print(f"could not read the checks of PR {pr}: {out.stderr.strip()}")
        return 2
    for c in sorted(checks, key=lambda c: (c.get("workflow", ""), c.get("name", ""))):
        print(f"{c.get('bucket', '?'):9} {c.get('workflow', '')}/{c.get('name', '')}")
    found = problems(checks, accepted)
    for label, why in found:
        print(f"BLOCKED {label}: {why}")
    print(f"{'ok  ' if not found else 'FAIL'} PR {pr}, {len(checks)} checks, {len(found)} blocking, {len(accepted)} accepted by name")
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
