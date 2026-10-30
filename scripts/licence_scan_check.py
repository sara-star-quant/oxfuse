"""Checks the reports of a licence scan and fails on what needs a human look.

  python3 -I scripts/licence_scan_check.py scancode scancode.json
  python3 -I scripts/licence_scan_check.py snippets scanoss.json

`scancode`: ScanCode's per-file licence and copyright findings against the policy
below. Our files carry Apache-2.0 and our own holder; a few files only talk about
other licences (allow lists, the licensing page); anything else fails.
`snippets`: SCANOSS matches (fingerprints of our files against open-source code).
Matches already read and judged are listed in ASSESSED; a new one fails until someone
reads it and adds a line with the reason.

Standard library only.
"""
import json
import sys

OWN_HOLDER = "sara star quant"

# Files that mention other licences (they quote or list them); ScanCode reports every mention.
MENTIONS = {
    "about.toml": "the licence allow list",
    "deny.toml": "the licence allow list",
    "crates/cli/tests/repo_hygiene.rs": "tests that read licence files",
    "docs/comparison.md": "what other projects say about their licences",
    "docs/licensing.md": "the licence review",
    "docs/provenance.md": "the licences of sources and dependencies",
    "scripts/dco_check.py": "the Developer Certificate of Origin",
}

OWN_LICENCES = {"Apache-2.0"}

# SCANOSS matches that were read and judged. Key: file, value: why it is fine.
ASSESSED = {
    "crates/core/src/types.rs": "a derive line and a newtype index; generic Rust",
    "crates/onnx/src/proto.rs": "prost structs dictated by the field numbers and names of onnx.proto (Apache-2.0); credited in NOTICE",
    "crates/runtime/benches/mlp_graph/mod.rs": "the Numerical Recipes linear congruential constants, widely published",
    "scripts/check_links.py": "a urllib request with a User-Agent; generic standard library use",
}


def relative(path):
    """ScanCode reports paths under the scanned directory's name."""
    return path.split("/", 1)[1] if "/" in path else path


def scancode_findings(report):
    findings = []
    for f in report.get("files", []):
        if f.get("type") != "file":
            continue
        path = relative(f["path"])
        expression = f.get("detected_license_expression_spdx") or ""
        if expression and expression not in OWN_LICENCES and path not in MENTIONS:
            findings.append((path, f"licence {expression}"))
        for h in f.get("holders", []):
            holder = h.get("holder", "").strip().rstrip(".")
            if holder and not holder.lower().startswith(OWN_HOLDER):
                findings.append((path, f"copyright holder {holder}"))
    return sorted(set(findings))


def snippet_findings(report):
    """Files with a match that nobody has judged yet: (path, share, component, licences)."""
    rows = []
    for path, matches in report.items():
        for m in matches:
            if m.get("id") in (None, "none") or path in ASSESSED:
                continue
            rows.append((path, m.get("matched", "?"), m.get("component", "?"), [x.get("name") for x in m.get("licenses", [])]))
    return sorted(rows)


def main(argv):
    if len(argv) != 3 or argv[1] not in ("scancode", "snippets"):
        print(__doc__)
        return 2
    report = json.load(open(argv[2]))
    if argv[1] == "scancode":
        files = [f for f in report.get("files", []) if f.get("type") == "file"]
        findings = scancode_findings(report)
        print(f"ScanCode looked at {len(files)} files.")
        for path, what in findings:
            print(f"FAIL {path}: {what}")
        print(f"{'ok  ' if not findings else 'FAIL'} licence scan, {len(findings)} finding(s)")
        return 1 if findings else 0
    rows = snippet_findings(report)
    for path, share, component, licences in rows:
        print(f"FAIL {path}: {share} like {component} ({', '.join(x for x in licences if x)}); read it, then add it to ASSESSED")
    print(f"{'ok  ' if not rows else 'FAIL'} snippet scan, {len(rows)} new match(es), {len(ASSESSED)} judged")
    return 1 if rows else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
