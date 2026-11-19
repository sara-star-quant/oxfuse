"""Reads every check of a pull request before it is merged.

  python3 -I scripts/pre_merge.py PR [--wait] [--sha SHA] [--accept NAME ...]

The branch ruleset requires one check, `required`. Mutation testing, Kani, the
semver check and the others run on the pull request but cannot block a merge (they
are slow or path-filtered), so a red one is easy to miss. This exits 1 on any check
that failed, was cancelled or is still running, unless it is named with --accept.
`required` itself can never be accepted away.

--wait    poll until no check is pending (at most 40 minutes), then judge.
--sha     fail unless the pull request head is exactly this commit, so a merge never
          lands a commit nobody checked.
A change under a performance-sensitive path (the CPU kernels, the math, the plan, the
GPU shaders) also blocks until `--accept perf-ab` says a same-hardware A/B result is
linked in the pull request: hosted runners are too noisy to gate on speed.

Standard library plus the `gh` CLI.
"""
import json
import subprocess
import sys
import time

GATE = "required"
PERF = "perf-ab"
PERF_PATHS = (
    "crates/runtime/src/cpu/",
    "crates/runtime/src/kernels.rs",
    "crates/runtime/src/math.rs",
    "crates/runtime/src/plan.rs",
    "crates/runtime/benches/",
    "crates/wgpu/src/shaders.rs",
    "crates/wgpu/src/shaders/",
)
WAIT_SECONDS = 40 * 60
POLL_SECONDS = 15


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


def perf_files(files):
    """The changed files that sit under a performance-sensitive path."""
    return sorted(f for f in files if f.startswith(PERF_PATHS))


def wait_for(fetch, sleep=time.sleep, now=time.monotonic, limit=WAIT_SECONDS, poll=POLL_SECONDS):
    """Fetch the checks until none is pending or the limit passes; returns the last fetch."""
    start = now()
    while True:
        checks = fetch()
        pending = any(c.get("bucket") == "pending" for c in checks)
        if not pending or now() - start >= limit:
            return checks
        sleep(poll)


def gh_json(*args):
    # gh exits 8 while checks are pending; the JSON is still on stdout.
    out = subprocess.run(["gh", *args], capture_output=True, text=True)
    try:
        return json.loads(out.stdout)
    except json.JSONDecodeError:
        raise SystemExit(f"gh {' '.join(args)} gave no JSON: {out.stderr.strip()}")


def fetch_checks(pr):
    return gh_json("pr", "checks", pr, "--json", "name,workflow,bucket")


def fetch_head(pr):
    return gh_json("pr", "view", pr, "--json", "headRefOid")["headRefOid"]


def fetch_files(pr):
    return [f["path"] for f in gh_json("pr", "view", pr, "--json", "files")["files"]]


def parse(argv):
    """(pr, wait, sha, accepted) or None for a bad command line."""
    args = list(argv)
    if not args or not args[0].isdigit():
        return None
    pr, wait, sha, accepted = args.pop(0), False, None, []
    while args:
        flag = args.pop(0)
        if flag == "--wait":
            wait = True
        elif flag in ("--accept", "--sha") and args:
            value = args.pop(0)
            if flag == "--sha":
                sha = value
            else:
                accepted.append(value)
        else:
            return None
    return pr, wait, sha, accepted


def main(argv):
    parsed = parse(argv[1:])
    if parsed is None:
        print(__doc__)
        return 2
    pr, wait, sha, accepted = parsed
    found = []
    if sha is not None:
        head = fetch_head(pr)
        if head != sha:
            found.append(("head", f"the pull request is at {head[:12]}, not {sha[:12]}"))
    checks = wait_for(lambda: fetch_checks(pr)) if wait else fetch_checks(pr)
    for c in sorted(checks, key=lambda c: (c.get("workflow", ""), c.get("name", ""))):
        print(f"{c.get('bucket', '?'):9} {c.get('workflow', '')}/{c.get('name', '')}")
    found += problems(checks, accepted)
    touched = perf_files(fetch_files(pr))
    if touched and PERF not in accepted:
        found.append((PERF, f"touches {len(touched)} performance-sensitive file(s), first {touched[0]}; link a same-hardware A/B result, then pass --accept {PERF}"))
    for label, why in found:
        print(f"BLOCKED {label}: {why}")
    print(f"{'ok  ' if not found else 'FAIL'} PR {pr}, {len(checks)} checks, {len(found)} blocking, {len(accepted)} accepted by name")
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
