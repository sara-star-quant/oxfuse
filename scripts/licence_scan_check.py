"""Reads the reports of a licence scan and says what needs a human look.

  python3 -I scripts/licence_scan_check.py scancode.json [scanoss.json]

ScanCode (licences and copyright holders, per file) is checked against a policy;
SCANOSS (snippets of other open-source code, matched by fingerprint) is listed with
the share of each file that matched. Standard library only.
"""
import json
import sys
from collections import defaultdict

OWN_HOLDERS = ("sara star quant llc",)


def scancode_findings(report, policy):
    """(path, what) pairs the policy does not allow."""
    findings = []
    for f in report.get("files", []):
        if f.get("type") != "file":
            continue
        path = f["path"]
        expression = f.get("detected_license_expression_spdx") or ""
        if expression and not policy_allows(path, expression, policy):
            findings.append((path, f"licence {expression}"))
        for h in f.get("holders", []):
            holder = h.get("holder", "").strip()
            if holder and holder.lower() not in OWN_HOLDERS and not policy_allows_holder(path, policy):
                findings.append((path, f"copyright holder {holder}"))
    return sorted(set(findings))


def policy_allows(path, expression, policy):
    if path in policy["mentions"] or any(path.startswith(p) for p in policy["mention_prefixes"]):
        return True
    return expression in policy["own_licences"]


def policy_allows_holder(path, policy):
    return path in policy["mentions"] or any(path.startswith(p) for p in policy["mention_prefixes"])


POLICY = {
    # Our own files carry Apache-2.0 (LICENSE and the per-crate copies).
    "own_licences": {"Apache-2.0"},
    # Files that discuss or quote other licences; ScanCode reports every mention.
    "mentions": set(),
    "mention_prefixes": (),
}


def snippet_matches(report):
    rows = []
    for path, matches in report.items():
        for m in matches:
            if m.get("id") not in (None, "none"):
                rows.append((path, m.get("matched", "?"), m.get("component", "?"), [l.get("name") for l in m.get("licenses", [])]))
    return sorted(rows)


def main(argv):
    if len(argv) not in (2, 3):
        print(__doc__)
        return 2
    scancode = json.load(open(argv[1]))
    findings = scancode_findings(scancode, POLICY)
    files = [f for f in scancode.get("files", []) if f.get("type") == "file"]
    by_expr = defaultdict(list)
    for f in files:
        by_expr[f.get("detected_license_expression_spdx") or "(none)"].append(f["path"])
    print("## Licence scan\n")
    print(f"ScanCode looked at {len(files)} files.\n")
    print("| Licence expression | Files |\n|---|---|")
    for expr, paths in sorted(by_expr.items(), key=lambda kv: -len(kv[1])):
        print(f"| {expr} | {len(paths)} |")
    print(f"\n{len(findings)} finding(s) outside the policy.\n")
    for path, what in findings:
        print(f"- `{path}`: {what}")
    if len(argv) == 3:
        rows = snippet_matches(json.load(open(argv[2])))
        print(f"\n## Snippet scan\n\n{len(rows)} file(s) with a match (read each: schema-dictated or trivial matches are expected).\n")
        for path, share, component, licences in rows:
            print(f"- `{path}`: {share} like {component} ({', '.join(x for x in licences if x)})")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
